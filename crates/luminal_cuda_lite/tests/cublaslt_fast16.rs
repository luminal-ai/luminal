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

fn q(value: GraphTensor, dtype: DType) -> GraphTensor {
    value.cast(dtype).cast(DType::F32)
}

fn mode(dtype: DType) -> LtComputation {
    match dtype {
        DType::F16 => LtComputation::Float32FastFloat16,
        DType::Bf16 => LtComputation::Float32FastBfloat16,
        _ => unreachable!(),
    }
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

fn model(batch: bool, a_rounding: Option<DType>, b_rounding: Option<DType>) -> Graph {
    let mut graph = Graph::new();
    let mut a = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F32);
    let mut b = graph.tensor(if batch { vec![2, 4, 5] } else { vec![4, 5] }, DType::F32);
    if let Some(dtype) = a_rounding {
        a = q(a, dtype);
    }
    if let Some(dtype) = b_rounding {
        b = q(b, dtype);
    }
    let c = graph.tensor(if batch { vec![2, 3, 5] } else { vec![3, 5] }, DType::F32);
    let _out = a.matmul(b) + c;
    graph
}

#[test]
fn both_operands_require_the_same_authored_input_rounding_contract() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for batch in [false, true] {
        for a_dtype in [None, Some(DType::F16), Some(DType::Bf16)] {
            for b_dtype in [None, Some(DType::F16), Some(DType::Bf16)] {
                let serialized = saturate(&model(batch, a_dtype, b_dtype), WHITELIST);
                let (separate, tied) = calls(&serialized);
                for dtype in [DType::F16, DType::Bf16] {
                    let want = a_dtype == Some(dtype) && b_dtype == Some(dtype);
                    assert_eq!(
                        separate.iter().any(|op| op.computation == mode(dtype)),
                        want
                    );
                    assert_eq!(tied.iter().any(|op| op.computation == mode(dtype)), want);
                }
                let registry =
                    ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
                let view = EGraphView::new(&serialized, &registry);
                for op in separate {
                    for (matrix, semantic) in [(&op.a, &op.logical_a), (&op.b, &op.logical_b)] {
                        let physical = view.class(&matrix.tensor);
                        let physical = physical.nodes_named("LayoutTensorLit").next().unwrap();
                        let physical = physical.child(0).unwrap();
                        if op.computation == LtComputation::Float32 {
                            if matrix.dtype == luminal::dtype::PlanDtype::F32 {
                                assert_eq!(
                                    physical.id(),
                                    semantic,
                                    "ordinary FP32 must consume the authored operand"
                                );
                            }
                        } else {
                            let logical = view.class(semantic);
                            let outer = logical.nodes_named("LogicalCast").next().unwrap();
                            let inner = outer.child(0).unwrap();
                            let inner = inner.nodes_named("LogicalCast").next().unwrap();
                            let original = inner.child(0).unwrap();
                            assert_eq!(matrix.dtype, luminal::dtype::PlanDtype::F32);
                            assert_eq!(physical.id(), original.id());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fast16_calls_need_the_exact_complete_whitelist_row() {
    for dtype in [DType::F16, DType::Bf16] {
        for batch in [false, true] {
            let model = model(batch, Some(dtype), Some(dtype));
            let (all, _) = calls(&saturate(&model, WHITELIST));
            let op = all
                .iter()
                .find(|op| op.computation == mode(dtype) && op.c.is_some())
                .unwrap();
            let key = whitelist::key(op);
            let table = format!("(cublaslt-approved-configuration {key})");
            let (separate, tied) = calls(&saturate(&model, &table));
            assert!(!separate.is_empty() && !tied.is_empty());
            assert!(separate.iter().all(|op| whitelist::key(op) == key));
            assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key));
            for rejected in [
                String::new(),
                table.replace(
                    mode(dtype).egglog_constructor(),
                    mode(if dtype == DType::F16 {
                        DType::Bf16
                    } else {
                        DType::F16
                    })
                    .egglog_constructor(),
                ),
            ] {
                let (separate, tied) = calls(&saturate(&model, &rejected));
                assert!(separate.is_empty() && tied.is_empty());
            }
        }
    }
}

#[test]
fn casts_after_arithmetic_cannot_authorize_input_conversion() {
    for dtype in [DType::F16, DType::Bf16] {
        for batch in [false, true] {
            let mut graph = Graph::new();
            let a = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F32);
            let b = graph.tensor(if batch { vec![2, 4, 5] } else { vec![4, 5] }, DType::F32);
            let _out = q(a.matmul(b), dtype);
            assert!(
                calls(&saturate(&graph, WHITELIST))
                    .0
                    .iter()
                    .all(|op| op.computation == LtComputation::Float32)
            );
        }
        let mut graph = Graph::new();
        let a = q(graph.tensor([3, 4], DType::F32), dtype);
        let b = q(graph.tensor([4, 5], DType::F32), dtype);
        let products = a.expand_dim(1, 5) * b.permute(&[1, 0]).expand_dim(0, 3);
        let _out = q(products, dtype).sum(&[2]);
        assert!(calls(&saturate(&graph, WHITELIST)).0.is_empty());
    }
}

#[test]
fn earlier_rounding_is_preserved_through_views_and_final_casts() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for dtype in [DType::F16, DType::Bf16] {
        let earlier = if dtype == DType::F16 {
            DType::Bf16
        } else {
            DType::F16
        };
        let mut graph = Graph::new();
        let a = graph
            .tensor([4, 3], DType::F32)
            .cast(earlier)
            .cast(DType::F32);
        let b = graph.tensor([5, 4], DType::F32);
        let _out = q(a, dtype)
            .permute(&[1, 0])
            .matmul(q(b, dtype).permute(&[1, 0]))
            .cast(dtype);
        let serialized = saturate(&graph, WHITELIST);
        let (all, _) = calls(&serialized);
        let matching: Vec<_> = all
            .iter()
            .filter(|op| op.computation == mode(dtype))
            .collect();
        assert!(!matching.is_empty());
        let registry = ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
        let view = EGraphView::new(&serialized, &registry);
        for op in matching {
            assert!([&op.a, &op.b].iter().any(|matrix| {
                let physical = view.class(&matrix.tensor);
                let physical = physical.nodes_named("LayoutTensorLit").next().unwrap();
                let mut current = physical.child(0).unwrap();
                for _ in 0..6 {
                    let Some(cast) = current.nodes_named("LogicalCast").next() else {
                        break;
                    };
                    if cast
                        .child(1)
                        .unwrap()
                        .nodes_named(&format!("{earlier:?}"))
                        .next()
                        .is_some()
                    {
                        return true;
                    }
                    current = cast.child(0).unwrap();
                }
                false
            }));
        }
    }
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime};
    use support::TestTransfers;
    use support::input_rounding::{execute, reference};

    fn registry(dtype: DType, in_place: bool) -> Vec<luminal_cuda_lite::RegisteredOp> {
        whitelist::fast16_registry(dtype == DType::Bf16)
            .into_iter()
            .filter(|row| {
                !matches!(
                    row.label(),
                    "AddFunctionalGeneric" | "MulFunctionalGeneric" | "SelectFunctionalGeneric"
                ) && row.label()
                    != if in_place {
                        "CublasLtCall"
                    } else {
                        "CublasLtAccumulateInPlace"
                    }
            })
            .collect()
    }
    fn check_plan(runtime: &CudaRuntime, dtype: DType, in_place: bool) {
        let mut found = false;
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    assert!(!in_place);
                    assert_eq!(op.computation, mode(dtype));
                    found = true;
                    assert_eq!(
                        (op.a.dtype, op.b.dtype),
                        (
                            luminal::dtype::PlanDtype::F32,
                            luminal::dtype::PlanDtype::F32
                        )
                    );
                }
                if let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    assert!(in_place);
                    assert_eq!(op.computation, mode(dtype));
                    found = true;
                }
                assert!(
                    !op.label().contains("Cast"),
                    "input casts must be consumed by the selected call"
                );
            }
        }
        assert!(found);
    }
    fn supported(dtype: DType) -> bool {
        dtype == DType::F16
            || cudarc::driver::CudaContext::new(0)
                .unwrap()
                .compute_capability()
                .unwrap()
                .0
                >= 8
    }
    fn close(actual: &[f32], expected: &[f32]) {
        for (&got, &want) in actual.iter().zip(expected) {
            assert!(
                (got.is_nan() && want.is_nan())
                    || got == want
                    || (want.is_finite() && (got - want).abs() <= 2e-5 * (1. + want.abs())),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn small_products_honor_ties_overflow_and_subnormals() {
        for dtype in [DType::F16, DType::Bf16] {
            if !supported(dtype) {
                continue;
            }
            for size in [1, 3, 16] {
                let mut graph = Graph::new();
                let a = graph.tensor([size, size], DType::F32);
                let b = graph.tensor([size, size], DType::F32);
                let out = q(a, dtype).matmul(q(b, dtype));
                let tie = if dtype == DType::F16 {
                    2f32.powi(-11)
                } else {
                    2f32.powi(-8)
                };
                let inputs = [
                    (a, vec![1. + tie; size * size]),
                    (b, vec![1. + tie; size * size]),
                ];
                let expected = reference(&graph, out, &inputs);
                let (runtime, actual) = execute(
                    &graph,
                    out,
                    CudaBindings::dense(&graph.logical, &[out.id]),
                    &inputs,
                    registry(dtype, false),
                );
                check_plan(&runtime, dtype, false);
                assert_eq!(actual, expected, "size={size}, dtype={dtype:?}");
            }
            let mut graph = Graph::new();
            let values = vec![
                1. + 2f32.powi(-11),
                1. + 3. * 2f32.powi(-11),
                1. + 2f32.powi(-8),
                -1. - 2f32.powi(-8),
                65504.,
                65520.,
                1e5,
                2f32.powi(-24),
                2f32.powi(-25),
                2f32.powi(-126),
                2f32.powi(-133),
                2f32.powi(-134),
                f32::MAX,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
            ];
            let a = graph.tensor([values.len(), 1], DType::F32);
            let b = graph.tensor([1, 1], DType::F32);
            let out = q(a, dtype).matmul(q(b, dtype));
            let inputs = [(a, values), (b, vec![1.])];
            let expected = reference(&graph, out, &inputs);
            let (runtime, actual) = execute(
                &graph,
                out,
                CudaBindings::dense(&graph.logical, &[out.id]),
                &inputs,
                registry(dtype, false),
            );
            check_plan(&runtime, dtype, false);
            for (got, want) in actual.iter().zip(expected) {
                assert!(
                    (want.is_nan() && got.is_nan()) || *got == want,
                    "{dtype:?}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn views_bias_activations_and_final_casts_match_reference() {
        for dtype in [DType::F16, DType::Bf16] {
            if !supported(dtype) {
                continue;
            }
            for activation in 0..3 {
                let mut graph = Graph::new();
                let a = graph.tensor([4, 3], DType::F32);
                let b = graph.tensor([5, 4], DType::F32);
                let c = graph.tensor([3, 5], DType::F32);
                let bias = graph.tensor([5], DType::F32);
                let value = q(a, dtype)
                    .permute(&[1, 0])
                    .matmul(q(b, dtype).permute(&[1, 0]))
                    + c
                    + bias.expand_dim(0, 3);
                let value = match activation {
                    0 => value,
                    1 => value.relu(),
                    _ => value.gelu_fast_tanh_approximation(),
                };
                let out = value.cast(dtype);
                let data = |n, seed| {
                    (0..n)
                        .map(|i| ((i * 7 + seed) % 19) as f32 / 8. - 1. + 3. / 8192.)
                        .collect()
                };
                let inputs = [
                    (a, data(12, 0)),
                    (b, data(20, 1)),
                    (c, data(15, 2)),
                    (bias, data(5, 3)),
                ];
                let expected = reference(&graph, value, &inputs);
                let (runtime, actual) = execute(
                    &graph,
                    out,
                    CudaBindings::dense(&graph.logical, &[out.id]),
                    &inputs,
                    registry(dtype, false),
                );
                // The authored final cast is intentionally outside the F32 call.
                assert!(runtime.plan().unwrap().dag.node_weights().any(|node|matches!(node,BufferNode::Compute{op,..} if op.as_any().downcast_ref::<ApprovedCublasLt>().is_some_and(|op|op.computation==mode(dtype)&&op.bias.is_some()))));
                let rounded = |value| {
                    if dtype == DType::F16 {
                        half::f16::from_f32(value).to_f32()
                    } else {
                        half::bf16::from_f32(value).to_f32()
                    }
                };
                for (&got, want) in actual.iter().zip(expected) {
                    // Apply the authored final cast to the permitted F32
                    // interval, including the existing GELU approximation.
                    let tolerance = if activation == 2 { 5e-4 } else { 2e-5 } * (1.0 + want.abs());
                    assert!(
                        got >= rounded(want - tolerance) && got <= rounded(want + tolerance),
                        "activation={activation}, dtype={dtype:?}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn preconversion_a_storage_can_also_be_the_preserved_c_input() {
        for dtype in [DType::F16, DType::Bf16] {
            if !supported(dtype) {
                continue;
            }
            let mut graph = Graph::new();
            let a = graph.tensor([3, 3], DType::F32);
            let b = graph.tensor([3, 3], DType::F32);
            let out = q(a, dtype).matmul(q(b, dtype)) + a;
            let values = |seed| {
                (0..9)
                    .map(|i| ((i * 7 + seed) % 17) as f32 / 8. - 1. + 3. / 8192.)
                    .collect()
            };
            let inputs = [(a, values(0)), (b, values(1))];
            let expected = reference(&graph, out, &inputs);
            let mut bindings = CudaBindings::new();
            let home = bindings.input(a.id);
            bindings.input(b.id);
            bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
            bindings.output(out.id);
            bindings.output(a.id);
            let (runtime, actual) = execute(&graph, out, bindings, &inputs, registry(dtype, true));
            check_plan(&runtime, dtype, true);
            close(&actual, &expected);
        }
    }

    #[test]
    fn batch_broadcasts_and_device_coefficients_preserve_c_and_replay() {
        for dtype in [DType::F16, DType::Bf16] {
            if !supported(dtype) {
                continue;
            }
            for in_place in [false, true] {
                for sharing in [(false, false), (true, false), (false, true), (true, true)] {
                    let mut graph = Graph::new();
                    let a = graph.tensor(
                        if sharing.0 { vec![3, 4] } else { vec![2, 3, 4] },
                        DType::F32,
                    );
                    let b = graph.tensor(
                        if sharing.1 { vec![4, 5] } else { vec![2, 4, 5] },
                        DType::F32,
                    );
                    let c = graph.tensor([2, 3, 5], DType::F32);
                    let alpha = graph.tensor(Vec::<usize>::new(), DType::F32);
                    let beta = graph.tensor(Vec::<usize>::new(), DType::F32);
                    let mut aa = q(a, dtype);
                    let mut bb = q(b, dtype);
                    if sharing.0 {
                        aa = aa.expand_dim(0, 2);
                    }
                    if sharing.1 {
                        bb = bb.expand_dim(0, 2);
                    }
                    let p = aa.matmul(bb);
                    let out = p * alpha.expand_rhs(p.dims()) + c * beta.expand_rhs(c.dims());
                    let data = |n, seed| {
                        (0..n)
                            .map(|i| ((i * 7 + seed) % 17) as f32 / 8. - 1. + 3. / 8192.)
                            .collect()
                    };
                    let inputs = [
                        (a, data(if sharing.0 { 12 } else { 24 }, 0)),
                        (b, data(if sharing.1 { 20 } else { 40 }, 1)),
                        (c, data(30, 2)),
                        (alpha, vec![1.25]),
                        (beta, vec![-0.5]),
                    ];
                    let expected = reference(&graph, out, &inputs);
                    let mut bindings = CudaBindings::new();
                    bindings.input(a.id);
                    bindings.input(b.id);
                    let home = bindings.input(c.id);
                    bindings.input(alpha.id);
                    bindings.input(beta.id);
                    bindings.output(out.id);
                    bindings.output(c.id);
                    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
                    let (mut runtime, actual) =
                        execute(&graph, out, bindings, &inputs, registry(dtype, in_place));
                    check_plan(&runtime, dtype, in_place);
                    close(&actual, &expected);
                    let mut replay = inputs.clone();
                    replay[3].1 = vec![-0.75];
                    replay[4].1 = vec![2.0];
                    let expected = reference(&graph, out, &replay);
                    let mut arena = support::allocate_arena(&runtime).unwrap();
                    let mut staging = runtime.allocate_staging().unwrap();
                    for (tensor, values) in replay {
                        runtime
                            .upload(
                                &mut arena,
                                tensor.id,
                                luminal_cuda_lite::HostBuffer::from(values),
                            )
                            .unwrap();
                    }
                    runtime.execute(arena.arena(), &mut staging).unwrap();
                    let (actual, binding) = runtime.download(&arena, out.id).unwrap();
                    let actual = luminal_cuda_lite::layouts::dense_f32(
                        &actual.as_f32().unwrap(),
                        &binding.layout,
                    )
                    .unwrap();
                    close(&actual, &expected);
                    assert_eq!(
                        runtime.download(&arena, c.id).unwrap().0.as_f32().unwrap(),
                        inputs[2].1
                    );
                }
            }
        }
    }
}
