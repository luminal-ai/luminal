use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
use luminal_cuda_lite::ops::cublaslt::exec::LtComputation;

#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

fn q(value: GraphTensor) -> GraphTensor {
    value.cast(DType::TF32).cast(DType::F32)
}

fn saturate(model: &Graph, table: &str) -> EGraph {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers)
        .replace(WHITELIST.trim_end(), table);
    let bound = luminal_cuda_lite::CudaBindings::leaves(&model.logical)
        .bind(&model.logical)
        .unwrap();
    let mut graph = luminal::egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(
            None,
            &format!(
                "{program}\n{}\n{}",
                bound.prefix,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            ),
        )
        .unwrap();
    graph.serialize(Default::default()).egraph
}

fn calls(graph: &EGraph) -> (Vec<ApprovedCublasLt>, Vec<CublasLtAccumulateInPlace>) {
    let index = SerializedIndex::new(graph);
    let mut separate = Vec::new();
    let mut in_place = Vec::new();
    for (id, node) in &graph.nodes {
        if node.subsumed {
            continue;
        }
        let site = ExtractionSite {
            egraph: graph,
            node_id: id,
            node,
            index: &index,
        };
        match node.op.as_str() {
            CONSTRUCTOR => separate.push(
                ApprovedCublasLtMatcher
                    .extract(&site)
                    .as_any()
                    .downcast_ref::<ApprovedCublasLt>()
                    .unwrap()
                    .clone(),
            ),
            IN_PLACE => in_place.push(
                CublasLtAccumulateInPlaceMatcher
                    .extract(&site)
                    .as_any()
                    .downcast_ref::<CublasLtAccumulateInPlace>()
                    .unwrap()
                    .clone(),
            ),
            _ => (),
        }
    }
    (separate, in_place)
}

fn model(batch: bool, round_a: bool, round_b: bool) -> Graph {
    let mut graph = Graph::new();
    let mut a = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F32);
    let mut b = graph.tensor(if batch { vec![2, 4, 5] } else { vec![4, 5] }, DType::F32);
    if round_a {
        a = q(a);
    }
    if round_b {
        b = q(b);
    }
    let c = graph.tensor(if batch { vec![2, 3, 5] } else { vec![3, 5] }, DType::F32);
    let _out = a.matmul(b) + c;
    graph
}

#[test]
fn both_authored_round_trips_are_required_and_storage_permissions_stay_mode_specific() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for batch in [false, true] {
        for round_a in [false, true] {
            for round_b in [false, true] {
                let graph = saturate(&model(batch, round_a, round_b), WHITELIST);
                let (separate, in_place) = calls(&graph);
                let want = round_a && round_b;
                assert_eq!(
                    separate
                        .iter()
                        .any(|op| op.computation == LtComputation::TensorFloat32),
                    want
                );
                assert_eq!(
                    in_place
                        .iter()
                        .any(|op| op.computation == LtComputation::TensorFloat32),
                    want
                );
                let registry =
                    ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
                let view = EGraphView::new(&graph, &registry);
                for op in separate {
                    for (matrix, semantic) in [(&op.a, &op.logical_a), (&op.b, &op.logical_b)] {
                        assert_eq!(matrix.dtype, luminal::dtype::PlanDtype::F32);
                        let tensor = view.class(&matrix.tensor);
                        let literal = tensor.nodes_named("LayoutTensorLit").next().unwrap();
                        let source = literal.child(0).unwrap();
                        assert_eq!(
                            source.id() == semantic,
                            op.computation == LtComputation::Float32,
                            "ordinary F32 consumes materialized operands; TF32 may read pre-cast F32 storage"
                        );
                    }
                    assert_eq!(op.resolve(&Default::default()).computation, op.computation);
                }
            }
        }
    }
}

#[test]
fn tf32_admission_requires_the_exact_whitelist_row_for_both_executable_forms() {
    for batch in [false, true] {
        let model = model(batch, true, true);
        let (all, _) = calls(&saturate(&model, WHITELIST));
        let target = all
            .iter()
            .find(|op| op.computation == LtComputation::TensorFloat32 && op.c.is_some())
            .unwrap();
        let key = whitelist::key(target);
        let strict = whitelist::approved_keys()
            .into_iter()
            .filter(|key| !key.contains("CublasLtTensorFloat32Computation"))
            .map(|key| format!("(cublaslt-approved-configuration {key})\n"))
            .collect::<String>();
        for table in [String::new(), strict] {
            let (separate, in_place) = calls(&saturate(&model, &table));
            assert!(
                separate
                    .iter()
                    .all(|op| op.computation == LtComputation::Float32)
            );
            assert!(
                in_place
                    .iter()
                    .all(|op| op.computation == LtComputation::Float32)
            );
        }
        let one = format!("(cublaslt-approved-configuration {key})");
        let (separate, in_place) = calls(&saturate(&model, &one));
        assert!(!separate.is_empty() && !in_place.is_empty());
        assert!(separate.iter().all(|op| whitelist::key(op) == key));
        assert!(in_place.iter().all(|op| whitelist::in_place_key(op) == key));
    }
}

#[test]
fn rounding_after_arithmetic_never_authorizes_tf32_input_computation() {
    for batch in [false, true] {
        let mut graph = Graph::new();
        let a = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F32);
        let b = graph.tensor(if batch { vec![2, 4, 5] } else { vec![4, 5] }, DType::F32);
        let _out = q(a.matmul(b));
        assert!(
            calls(&saturate(&graph, WHITELIST))
                .0
                .iter()
                .all(|op| op.computation == LtComputation::Float32)
        );
    }
    let mut graph = Graph::new();
    let a = q(graph.tensor([3, 4], DType::F32));
    let b = q(graph.tensor([4, 5], DType::F32));
    let products = a.expand_dim(1, 5) * b.permute(&[1, 0]).expand_dim(0, 3);
    let _out = q(products).sum(&[2]);
    assert!(
        calls(&saturate(&graph, WHITELIST)).0.is_empty(),
        "rounding each product is a separate boundary"
    );
}

#[test]
fn tf32_preserves_earlier_narrowing_and_composes_with_final_casts_and_views() {
    for dtype in [DType::Bf16, DType::F16] {
        let mut graph = Graph::new();
        let a = graph
            .tensor([4, 3], DType::F32)
            .cast(dtype)
            .cast(DType::F32);
        let b = graph.tensor([5, 4], DType::F32);
        let _out = q(a)
            .permute(&[1, 0])
            .matmul(q(b).permute(&[1, 0]))
            .cast(dtype);
        let serialized = saturate(&graph, WHITELIST);
        let (calls, _) = calls(&serialized);
        assert!(
            calls
                .iter()
                .any(|op| op.computation == LtComputation::TensorFloat32)
        );
        use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
        let registry = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
        let view = EGraphView::new(&serialized, &registry);
        for op in calls
            .iter()
            .filter(|op| op.computation == LtComputation::TensorFloat32)
        {
            let has_narrowing = [&op.a, &op.b].iter().any(|matrix| {
                let tensor = view.class(&matrix.tensor);
                let literal = tensor.nodes_named("LayoutTensorLit").next().unwrap();
                let mut current = literal.child(0).unwrap();
                for _ in 0..6 {
                    let Some(cast) = current.nodes_named("LogicalCast").next() else {
                        break;
                    };
                    if cast
                        .child(1)
                        .unwrap()
                        .nodes_named(&format!("{dtype:?}"))
                        .next()
                        .is_some()
                    {
                        return true;
                    }
                    current = cast.child(0).unwrap();
                }
                false
            });
            assert!(
                has_narrowing,
                "a TF32 source must retain the earlier narrow -> F32 value"
            );
        }
    }
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer, RegisteredOp};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;

    fn reference(graph: &Graph, out: GraphTensor, inputs: &[(GraphTensor, Vec<f32>)]) -> Vec<f32> {
        let mut runtime =
            ReferenceRuntime::load_with(graph, ReferenceBindings::dense(&graph.logical, &[out.id]))
                .unwrap();
        let data: FxHashMap<_, _> = inputs
            .iter()
            .map(|(x, data)| (x.id, TypedBuffer::F32(data.clone())))
            .collect();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &luminal_reference::harness_search_options(),
            )
            .unwrap();
        for (id, data) in data {
            runtime.set_data(id, data);
        }
        runtime.execute().unwrap();
        runtime.get_f32(out.id).unwrap().clone()
    }

    fn execute(
        graph: &Graph,
        out: GraphTensor,
        bindings: CudaBindings,
        inputs: &[(GraphTensor, Vec<f32>)],
        registry: Vec<RegisteredOp>,
    ) -> (CudaRuntime, Vec<f32>) {
        let kept_inputs: Vec<_> = inputs
            .iter()
            .filter(|(x, _)| bindings.outputs().iter().any(|out| out.value == x.id))
            .cloned()
            .collect();
        let mut runtime = CudaRuntime::load_with(graph, bindings, registry).unwrap();
        let data: FxHashMap<_, _> = inputs
            .iter()
            .map(|(x, data)| (x.id, HostBuffer::from(data.clone())))
            .collect();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for (id, data) in data {
            runtime.upload(&mut arena, id, data).unwrap();
        }
        runtime.execute(arena.arena(), &mut staging).unwrap();
        for (input, expected) in kept_inputs {
            let (data, binding) = runtime.download(&arena, input.id).unwrap();
            assert_eq!(
                luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout)
                    .unwrap(),
                expected,
                "a kept input must survive accumulation"
            );
        }
        let (data, binding) = runtime.download(&arena, out.id).unwrap();
        let raw = match data.dtype {
            luminal::dtype::PlanDtype::F32 => data.as_f32().unwrap(),
            dtype => data
                .bytes
                .chunks_exact(2)
                .map(|bytes| {
                    let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
                    match dtype {
                        luminal::dtype::PlanDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
                        luminal::dtype::PlanDtype::F16 => half::f16::from_bits(bits).to_f32(),
                        _ => panic!("unexpected output {dtype:?}"),
                    }
                })
                .collect(),
        };
        let dense = luminal_cuda_lite::layouts::dense_f32(&raw, &binding.layout).unwrap();
        (runtime, dense)
    }

    fn assert_tf32_plan(runtime: &CudaRuntime, in_place: bool) {
        let mut found = false;
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                if let Some(call) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    assert!(!in_place);
                    assert_eq!(call.computation, LtComputation::TensorFloat32);
                    found = true;
                }
                if let Some(call) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    assert!(in_place);
                    assert_eq!(call.computation, LtComputation::TensorFloat32);
                    found = true;
                }
            }
        }
        assert!(found, "test must elect the authored TF32 contract");
    }

    fn native_tf32() -> bool {
        cudarc::driver::CudaContext::new(0)
            .unwrap()
            .compute_capability()
            .unwrap()
            .0
            >= 8
    }

    fn values(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 17 + seed) % 29) as f32 - 14.0) / 8.0 + 5.0 / 32768.0)
            .collect()
    }

    #[test]
    fn tf32_calls_use_device_input_precision_even_for_small_products() {
        if !native_tf32() {
            return;
        }
        for size in [1usize, 3, 16] {
            let mut graph = Graph::new();
            let a = graph.tensor([size, size], DType::F32);
            let b = graph.tensor([size, size], DType::F32);
            let out = q(a).matmul(q(b));
            // Use values that need TF32 rounding, away from halfway cases.
            // Full-F32 dispatch would skip the authored precision boundary.
            let av = (0..size * size)
                .map(|i| {
                    if i % 2 == 0 {
                        1.0 + 3.0 / 8192.0
                    } else {
                        -1.0 - 3.0 / 8192.0
                    }
                })
                .collect();
            let inputs = [(a, av), (b, vec![1.0 + 3.0 / 8192.0; size * size])];
            let expected = reference(&graph, out, &inputs);
            let registry = whitelist::tf32_registry()
                .into_iter()
                .filter(|row| row.label() != "CublasLtAccumulateInPlace")
                .collect();
            let (runtime, actual) = execute(
                &graph,
                out,
                CudaBindings::leaves(&graph.logical),
                &inputs,
                registry,
            );
            assert_tf32_plan(&runtime, false);
            assert_eq!(actual, expected, "size={size}");
        }
    }

    #[test]
    fn materialized_device_rounding_and_decomposed_execution_preserve_tf32_precision() {
        let mut bits = vec![
            0, 0x80000000, 1, 0x80000001, 0x7f800000, 0xff800000, 0x7fffffff, 0xff800001,
            0x7f7fffff,
        ];
        for exponent in [0u32, 1, 126, 127, 128, 254] {
            for fraction in [0u32, 1, 0x2000, 0x4000, 0x7fc000] {
                for tail in [0u32, 0xfff, 0x1000, 0x1001, 0x1fff] {
                    for sign in [0u32, 0x80000000] {
                        bits.push(sign | (exponent << 23) | fraction | tail);
                    }
                }
            }
        }
        let mut graph = Graph::new();
        let a = graph.tensor([bits.len()], DType::F32);
        let out = q(a);
        let inputs = [(a, bits.iter().copied().map(f32::from_bits).collect())];
        let expected = reference(&graph, out, &inputs);
        let (runtime, actual) = execute(
            &graph,
            out,
            CudaBindings::leaves(&graph.logical),
            &inputs,
            luminal_cuda_lite::cuda_registry_without_cublaslt()
                .into_iter()
                .filter(|row| row.label() != "CastGeneric")
                .collect(),
        );
        assert!(runtime.plan().unwrap().dag.node_weights().any(
            |node| matches!(node, BufferNode::Compute {op, ..} if op.label() == "Tf32RoundTrip")
        ));
        for ((got, nearest_even), source) in actual.iter().zip(expected).zip(bits) {
            if nearest_even.is_nan() {
                assert!(got.is_nan());
                continue;
            }
            let nearest_away = if source & 0x7f800000 == 0x7f800000 {
                source
            } else {
                source.wrapping_add(0x1000) & 0xffffe000
            };
            assert!(
                got.to_bits() == nearest_even.to_bits() || got.to_bits() == nearest_away,
                "TF32 conversion must round to a nearest representable value: input={source:08x} got={:08x}",
                got.to_bits()
            );
        }

        // A fully decomposed GEMM must preserve the very same authored casts.
        let mut graph = Graph::new();
        let a = graph.tensor([3, 4], DType::F32);
        let b = graph.tensor([4, 5], DType::F32);
        let out = q(a).matmul(q(b));
        let inputs = [(a, values(12, 1)), (b, values(20, 2))];
        let expected = reference(&graph, out, &inputs);
        let (_, actual) = execute(
            &graph,
            out,
            CudaBindings::leaves(&graph.logical),
            &inputs,
            luminal_cuda_lite::cuda_registry_without_cublaslt(),
        );
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() <= 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn tf32_composes_with_views_epilogues_and_final_rounding() {
        if !native_tf32() {
            return;
        }
        for activation in [0, 1, 2] {
            for final_type in [DType::F32, DType::Bf16, DType::F16, DType::TF32] {
                let mut graph = Graph::new();
                let a = graph.tensor([4, 3], DType::F32);
                let b = graph.tensor([5, 4], DType::F32);
                let c = graph.tensor([3, 5], DType::F32);
                let bias = graph.tensor([5], DType::F32);
                let value =
                    q(a).permute(&[1, 0]).matmul(q(b).permute(&[1, 0])) + c + bias.expand_dim(0, 3);
                let value = match activation {
                    0 => value,
                    1 => value.relu(),
                    _ => value.gelu_fast_tanh_approximation(),
                };
                let out = if final_type == DType::TF32 {
                    q(value)
                } else {
                    value.cast(final_type)
                };
                let inputs = [
                    (a, values(12, 1)),
                    (b, values(20, 2)),
                    (c, values(15, 3)),
                    (bias, values(5, 4)),
                ];
                // The reference owns the F32 calculation. Use the established
                // half codecs for narrow outputs that reference does not store.
                let expected = reference(&graph, value, &inputs);
                let registry = whitelist::tf32_registry()
                    .into_iter()
                    .filter(|row| {
                        !matches!(
                            row.label(),
                            "AddFunctionalGeneric"
                                | "SelectFunctionalGeneric"
                                | "MulFunctionalGeneric"
                                | "CublasLtAccumulateInPlace"
                        )
                    })
                    .collect();
                let (runtime, actual) = execute(
                    &graph,
                    out,
                    CudaBindings::dense(&graph.logical, &[out.id]),
                    &inputs,
                    registry,
                );
                assert_tf32_plan(&runtime, false);

                for (got, unrounded) in actual.iter().zip(expected) {
                    let want = match final_type {
                        DType::Bf16 => half::bf16::from_f32(unrounded).to_f32(),
                        DType::F16 => half::f16::from_f32(unrounded).to_f32(),
                        DType::TF32 => luminal_reference::Tf32::from_f32(unrounded).to_f32(),
                        _ => unrounded,
                    };
                    let tolerance = if activation == 2 { 5e-3 } else { 1e-5 };
                    // TF32 tie choice belongs to the device. For the final
                    // round trip, either nearest neighbor at a tie is valid.
                    let alternate = if final_type == DType::TF32 && unrounded.is_finite() {
                        f32::from_bits(unrounded.to_bits().wrapping_add(0x1000) & 0xffffe000)
                    } else {
                        want
                    };
                    assert!(
                        (got - want).abs().min((got - alternate).abs())
                            <= tolerance * (1.0 + want.abs()),
                        "activation={activation} final={final_type:?}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn batched_tf32_accumulation_survives_c_reuse_and_copy_repair() {
        if !native_tf32() {
            return;
        }
        use luminal::layout_ir::{Access, FreedBy};
        for in_place in [false, true] {
            for (access, keep_c) in [
                (Access::ReadWrite, false),
                (Access::ReadOnly, false),
                (Access::ReadWrite, true),
            ] {
                let mut graph = Graph::new();
                let a = graph.tensor([2, 3, 4], DType::F32);
                let b = graph.tensor([2, 4, 5], DType::F32);
                let c = graph.tensor([2, 3, 5], DType::F32);
                let out = q(a).matmul(q(b)) + c;
                let inputs = [(a, values(24, 1)), (b, values(40, 2)), (c, values(30, 3))];
                let expected = reference(&graph, out, &inputs);
                let mut bindings = CudaBindings::new();
                bindings.input(a.id);
                bindings.input(b.id);
                let home = bindings.input(c.id);
                bindings.output(out.id);
                bindings.declare(home, access, FreedBy::Caller);
                if keep_c {
                    bindings.output(c.id);
                }
                let registry = whitelist::tf32_registry()
                    .into_iter()
                    .filter(|row| {
                        row.label() != "AddFunctionalGeneric"
                            && row.label()
                                != if in_place {
                                    "CublasLtCall"
                                } else {
                                    "CublasLtAccumulateInPlace"
                                }
                    })
                    .collect();
                let (runtime, actual) = execute(&graph, out, bindings, &inputs, registry);
                assert_tf32_plan(&runtime, in_place);
                for (got, want) in actual.iter().zip(expected) {
                    assert!(
                        (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "{got} vs {want}"
                    );
                }
            }
        }
    }
}
