use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST,
};
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

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

#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Fixture {
    graph: Graph,
    inputs: Vec<GraphTensor>,
    out: GraphTensor,
    arithmetic: GraphTensor,
    dtype: DType,
    batch: bool,
    activation: usize,
    with_c: bool,
    data: Option<Vec<Vec<f32>>>,
    padded: bool,
    fallback: bool,
}
fn fixture(
    dtype: DType,
    batch: bool,
    narrow_output: bool,
    with_bias: bool,
    activation: usize,
) -> Fixture {
    fixture_with_c(dtype, batch, narrow_output, with_bias, activation, true)
}

fn fixture_with_c(
    dtype: DType,
    batch: bool,
    narrow_output: bool,
    with_bias: bool,
    activation: usize,
    with_c: bool,
) -> Fixture {
    let mut graph = Graph::new();
    let a = graph.tensor(if batch { vec![2, 3, 8] } else { vec![3, 8] }, dtype);
    let b = graph.tensor(if batch { vec![2, 8, 4] } else { vec![8, 4] }, dtype);
    let c = with_c.then(|| graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, dtype));
    let mut inputs = vec![a, b];
    let mut out = a.cast(DType::F32).matmul(b.cast(DType::F32));
    if let Some(c) = c {
        inputs.push(c);
        out += c.cast(DType::F32);
    }
    if with_bias {
        let bias = graph.tensor([4], dtype);
        inputs.push(bias);
        out += bias.cast(DType::F32).expand_dim(0, 3);
    }
    out = match activation {
        0 => out,
        1 => out.relu(),
        _ => out.gelu_fast_tanh_approximation(),
    };
    let arithmetic = out;
    if narrow_output {
        out = out.cast(dtype);
    }
    Fixture {
        graph,
        inputs,
        out,
        arithmetic,
        dtype,
        batch,
        activation,
        with_c,
        data: None,
        padded: false,
        fallback: false,
    }
}

fn earlier_narrowing_fixture(dtype: DType, with_c: bool) -> Fixture {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 8], DType::F32);
    let b = graph.tensor([8, 4], DType::F32);
    let addend = graph.tensor(if with_c { vec![3, 4] } else { vec![4] }, DType::F32);
    let wide = |x: GraphTensor| x.cast(dtype).cast(DType::F32);
    let arithmetic = wide(a).matmul(wide(b))
        + if with_c {
            wide(addend)
        } else {
            wide(addend).expand_dim(0, 3)
        };
    let out = arithmetic.cast(dtype);
    Fixture {
        graph,
        inputs: vec![a, b, addend],
        out,
        arithmetic,
        dtype,
        batch: false,
        activation: 0,
        with_c,
        data: None,
        padded: false,
        fallback: false,
    }
}

#[test]
fn direct_storage_readings_retain_earlier_source_narrowing() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for dtype in [DType::F16, DType::Bf16] {
        for with_c in [false, true] {
            let f = earlier_narrowing_fixture(dtype, with_c);
            let graph = saturate(&f.graph, WHITELIST);
            let (ops, in_place) = calls(&graph);
            let registry =
                ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
            let view = EGraphView::new(&graph, &registry);
            let narrow: Vec<_> = ops
                .iter()
                .filter(|op| op.d.dtype != PlanDtype::F32)
                .collect();
            assert!(!narrow.is_empty());
            for op in narrow {
                for matrix in [&op.a, &op.b].into_iter().chain(op.c.iter()) {
                    let class = view.class(&matrix.tensor);
                    let literal = class.nodes_named("LayoutTensorLit").next().unwrap();
                    assert!(
                        literal
                            .child(0)
                            .unwrap()
                            .nodes_named("LogicalCast")
                            .any(|cast| cast
                                .child(1)
                                .unwrap()
                                .nodes_named(&format!("{dtype:?}"))
                                .next()
                                .is_some()),
                        "earlier narrowing must remain in the selected storage value"
                    );
                }
            }
            assert!(
                in_place
                    .iter()
                    .all(|op| op.resolve(&Default::default()).c.unwrap().descriptor.ld > 0),
                "broadcast C is never writable"
            );
        }
    }
}

#[test]
fn narrow_storage_is_admitted_only_with_authored_final_rounding_and_exact_widening() {
    for (dtype, physical) in [(DType::Bf16, PlanDtype::Bf16), (DType::F16, PlanDtype::F16)] {
        for batch in [false, true] {
            for with_c in [false, true] {
                if batch && !with_c {
                    continue;
                }
                let with_bias = !batch && !with_c;
                for narrow_output in [false, true] {
                    let f = fixture_with_c(
                        dtype,
                        batch,
                        narrow_output,
                        with_bias,
                        if batch { 0 } else { 1 },
                        with_c,
                    );
                    let (separate, in_place) = calls(&saturate(&f.graph, WHITELIST));
                    let narrow: Vec<_> = separate
                        .iter()
                        .filter(|op| {
                            op.d.dtype == physical
                                && (!with_bias
                                    || op
                                        .resolve(&Default::default())
                                        .c
                                        .is_some_and(|c| c.descriptor.ld == 0))
                        })
                        .collect();
                    assert_eq!(!narrow.is_empty(), narrow_output);
                    if with_c {
                        assert_eq!(
                            in_place.iter().any(|op| op.c_and_d.dtype == physical),
                            narrow_output
                        );
                    }
                    for op in narrow {
                        assert_eq!(op.a.dtype, physical);
                        assert_eq!(op.b.dtype, physical);
                        assert_eq!(op.c.as_ref().map(|c| c.dtype), Some(physical));
                        assert_eq!(op.bias.as_ref().map(|b| b.dtype), None);
                        assert_eq!(
                            op.resolve(&Default::default()).bias.map(|b| b.data_type),
                            None
                        );
                    }
                    for op in separate.iter().filter(|op| op.d.dtype == PlanDtype::F32) {
                        assert!(op.c.as_ref().is_none_or(|c| c.dtype == PlanDtype::F32));
                        assert!(op.bias.as_ref().is_none_or(|b| b.dtype == PlanDtype::F32));
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_calls_require_the_complete_whitelist_row_for_both_forms() {
    for dtype in [DType::Bf16, DType::F16] {
        let f = fixture(dtype, false, true, false, 1);
        let (all, _) = calls(&saturate(&f.graph, WHITELIST));
        let op = all
            .iter()
            .find(|op| op.d.dtype != PlanDtype::F32 && op.c.is_some() && op.bias.is_none())
            .unwrap();
        let key = whitelist::key(op);
        let (empty, empty_in_place) = calls(&saturate(&f.graph, ""));
        assert!(empty.is_empty() && empty_in_place.is_empty());
        let table = format!("(cublaslt-approved-configuration {key})");
        let (separate, in_place) = calls(&saturate(&f.graph, &table));
        assert!(!separate.is_empty() && !in_place.is_empty());
        assert!(separate.iter().all(|op| whitelist::key(op) == key));
        assert!(in_place.iter().all(|op| whitelist::in_place_key(op) == key));
    }
}

#[test]
fn rounding_between_arithmetic_steps_cannot_be_swallowed_by_a_complete_call() {
    for dtype in [DType::Bf16, DType::F16] {
        for after_c in [false, true] {
            let mut graph = Graph::new();
            let a = graph.tensor([3, 8], dtype).cast(DType::F32);
            let b = graph.tensor([8, 4], dtype).cast(DType::F32);
            let c = graph.tensor([3, 4], dtype).cast(DType::F32);
            let bias = graph.tensor([4], dtype).cast(DType::F32).expand_dim(0, 3);
            let p = a.matmul(b);
            let round = |x: GraphTensor| x.cast(dtype).cast(DType::F32);
            let value = if after_c {
                round(p + c) + bias
            } else {
                round(p) + c + bias
            };
            let _out = value.cast(dtype);
            let (separate, _) = calls(&saturate(&graph, WHITELIST));
            assert!(
                separate
                    .iter()
                    .all(|op| op.c.is_none() || op.bias.is_none())
            );
        }
    }
}

#[test]
fn epilogues_that_can_round_early_have_no_narrow_whitelist_row() {
    for dtype in [DType::Bf16, DType::F16] {
        for with_c in [false, true] {
            for activation in [0, 1, 2] {
                let f = fixture_with_c(dtype, false, true, true, activation, with_c);
                let (ops, _) = calls(&saturate(&f.graph, WHITELIST));
                assert!(
                    ops.iter()
                        .filter(|op| op.d.dtype != PlanDtype::F32)
                        .all(|op| op.bias.is_none())
                );
                if with_c || activation == 2 {
                    assert!(
                        ops.iter()
                            .any(|op| op.d.dtype == PlanDtype::F32 && op.bias.is_some())
                    );
                }
            }
        }
    }
    assert!(
        whitelist::approved_keys()
            .iter()
            .filter(|key| {
                ["Bf16", "F16"].iter().any(|dtype| {
                    key.contains(&format!(") ({dtype}) (CublasLtNoBiasDataType)"))
                        || key.contains(&format!(") ({dtype}) (CublasLtBiasDataTypeOf"))
                })
            })
            .all(|key| key.contains("CublasLtNoBiasDataType")
                && !key.contains("GaussianErrorLinear"))
    );
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;

    fn buffers(dtype: DType, values: Vec<f32>) -> (HostBuffer, TypedBuffer) {
        let values = values.into_iter();
        match dtype {
            DType::F32 => (
                HostBuffer::from(values.clone().collect::<Vec<_>>()),
                TypedBuffer::F32(values.collect()),
            ),
            DType::F16 => {
                let values: Vec<_> = values.map(half::f16::from_f32).collect();
                (
                    HostBuffer::new(
                        PlanDtype::F16,
                        values
                            .iter()
                            .flat_map(|x| x.to_bits().to_ne_bytes())
                            .collect(),
                    )
                    .unwrap(),
                    TypedBuffer::F16(values),
                )
            }
            DType::Bf16 => {
                let values: Vec<_> = values.map(half::bf16::from_f32).collect();
                (
                    HostBuffer::new(
                        PlanDtype::Bf16,
                        values
                            .iter()
                            .flat_map(|x| x.to_bits().to_ne_bytes())
                            .collect(),
                    )
                    .unwrap(),
                    TypedBuffer::Bf16(values),
                )
            }
            _ => unreachable!(),
        }
    }
    fn decode(data: &HostBuffer) -> Vec<f32> {
        data.bytes
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
                match data.dtype {
                    PlanDtype::F16 => half::f16::from_bits(bits).to_f32(),
                    PlanDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
                    _ => panic!("narrow output required"),
                }
            })
            .collect()
    }
    fn quantize(dtype: DType, value: f32) -> f32 {
        match dtype {
            DType::F16 => half::f16::from_f32(value).to_f32(),
            DType::Bf16 => half::bf16::from_f32(value).to_f32(),
            _ => unreachable!(),
        }
    }

    fn execute(f: Fixture, in_place: bool, access: luminal::layout_ir::Access, keep_c: bool) {
        use luminal::layout_ir::FreedBy;
        let (dtype, batch, activation, with_c) = (f.dtype, f.batch, f.activation, f.with_c);
        let mut gpu_data = FxHashMap::default();
        let mut ref_data = FxHashMap::default();
        for (index, input) in f.inputs.iter().enumerate() {
            let len = match index {
                0 => {
                    if batch {
                        48
                    } else {
                        24
                    }
                }
                1 => {
                    if batch {
                        64
                    } else {
                        32
                    }
                }
                2 if with_c => {
                    if batch {
                        24
                    } else {
                        12
                    }
                }
                _ => 4,
            };
            let values = f.data.as_ref().map_or_else(
                || {
                    (0..len)
                        .map(|i| ((i * 17 + index + 1) % 29) as f32 / 8.0 - 1.5 + 5.0 / 32768.0)
                        .collect()
                },
                |data| data[index].clone(),
            );
            let (gpu, reference) = buffers(input.dtype, values);
            gpu_data.insert(input.id, gpu);
            ref_data.insert(input.id, reference);
        }
        let mut reference = ReferenceRuntime::load_with(
            &f.graph,
            ReferenceBindings::dense(&f.graph.logical, &[f.out.id, f.arithmetic.id]),
        )
        .unwrap();
        reference
            .search(
                &Default::default(),
                &Default::default(),
                &ref_data,
                &luminal_reference::harness_search_options(),
            )
            .unwrap();
        for (id, data) in ref_data {
            reference.set_data(id, data);
        }
        reference.execute().unwrap();
        let expected = reference.get_f32(f.arithmetic.id).unwrap().clone();
        let original_c = with_c.then(|| gpu_data[&f.inputs[2].id].clone());
        let mut bindings = CudaBindings::new();
        for (index, input) in f.inputs.iter().enumerate() {
            if index == 2 && with_c && f.padded {
                let strides = if batch { vec![32, 8, 1] } else { vec![8, 1] };
                bindings.input_with(
                    input.id,
                    luminal_cuda_lite::bindings::BoundaryLayout::strided_literal(strides),
                );
                let data = gpu_data.get_mut(&input.id).unwrap();
                let count = if batch { 2 } else { 1 };
                let span = (count - 1) * 32 + 2 * 8 + 4;
                let mut packed = vec![0u8; span * 2];
                for b in 0..count {
                    for row in 0..3 {
                        for col in 0..4 {
                            let source = (b * 12 + row * 4 + col) * 2;
                            let target = (b * 32 + row * 8 + col) * 2;
                            packed[target..target + 2]
                                .copy_from_slice(&data.bytes[source..source + 2]);
                        }
                    }
                }
                data.bytes = packed;
            } else {
                bindings.input(input.id);
            }
        }
        if with_c {
            let c_home = bindings.inputs()[2].buffer;
            bindings.declare(c_home, access, FreedBy::Caller);
        }
        if f.padded {
            let strides = if batch {
                if in_place {
                    vec![32, 8, 1]
                } else {
                    vec![24, 6, 1]
                }
            } else if in_place {
                vec![8, 1]
            } else {
                vec![6, 1]
            };
            bindings.output_with(
                f.out.id,
                luminal_cuda_lite::bindings::BoundaryLayout::strided_literal(strides),
            );
        } else {
            bindings.output(f.out.id);
        }
        if keep_c {
            bindings.output(f.inputs[2].id);
        }
        // Exact widening and the final cast must be absorbed by the complete
        // call. Earlier narrowing remains executable; views and tie repair
        // remain available through production code.
        let earlier_narrowing = f.inputs.iter().any(|input| input.dtype == DType::F32);
        let narrow_registry = whitelist::narrow_registry()
            .into_iter()
            .filter(|row| {
                !matches!(
                    row.label(),
                    "AddFunctionalGeneric" | "MulFunctionalGeneric" | "SelectFunctionalGeneric"
                ) && (earlier_narrowing || row.label() != "CastGeneric")
                    && (with_c
                        || !matches!(row.label(), "IndexMapApplyMaterialize" | "CopyGeneric"))
                    && row.label()
                        != if in_place {
                            "CublasLtCall"
                        } else {
                            "CublasLtAccumulateInPlace"
                        }
            })
            .collect();
        let registry = if f.fallback {
            luminal_cuda_lite::cuda_registry_filtered(|row| row.label() != "ReduceSumGeneric")
        } else {
            narrow_registry
        };
        let mut runtime = CudaRuntime::load_with(&f.graph, bindings, registry).unwrap();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &gpu_data,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        let mut found = false;
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                if let Some(call) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    if !f.fallback {
                        if !with_c && !earlier_narrowing {
                            assert!(call.bias.is_none());
                            assert_eq!(
                                call.resolve(&Default::default()).c.unwrap().descriptor.ld,
                                0,
                                "a stored narrow bias must use packed broadcast C: {call:?}"
                            );
                        }
                        assert!(!in_place);
                        assert_ne!(call.d.dtype, PlanDtype::F32);
                    }
                    found = true;
                }
                if let Some(call) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    if !f.fallback {
                        assert!(in_place);
                        assert_ne!(call.c_and_d.dtype, PlanDtype::F32);
                    }
                    found = true;
                }
                if !f.fallback && !earlier_narrowing {
                    assert_ne!(op.label(), "CastGeneric");
                }
            }
        }
        assert!(found);
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for (id, data) in gpu_data {
            runtime.upload(&mut arena, id, data).unwrap();
        }
        runtime.execute(arena.arena(), &mut staging).unwrap();
        let (output, binding) = runtime.download(&arena, f.out.id).unwrap();
        let actual =
            luminal_cuda_lite::layouts::dense_f32(&decode(&output), &binding.layout).unwrap();
        for (got, want) in actual.iter().zip(expected) {
            // Allow the existing FP32 reassociation/GELU approximation policy,
            // then apply output rounding; do not grant a blanket narrow ULP.
            let eps = if activation == 2 { 5e-4 } else { 1e-5 } * (1.0 + want.abs());
            assert!(
                *got >= quantize(dtype, want - eps) && *got <= quantize(dtype, want + eps),
                "{dtype:?} activation={activation}: {got} vs {want}"
            );
        }
        if keep_c {
            let (data, binding) = runtime.download(&arena, f.inputs[2].id).unwrap();
            let kept =
                luminal_cuda_lite::layouts::dense_f32(&decode(&data), &binding.layout).unwrap();
            assert_eq!(kept, decode(&original_c.unwrap()));
        }
    }

    #[test]
    fn supported_narrow_calls_absorb_all_casts_and_keep_fp32_arithmetic() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for with_c in [false, true] {
                for activation in [0, 1] {
                    execute(
                        fixture_with_c(dtype, false, true, !with_c, activation, with_c),
                        false,
                        Access::ReadOnly,
                        false,
                    );
                }
            }
        }
    }

    #[test]
    fn earlier_narrowing_executes_before_the_vendor_reads_narrow_c_or_bias() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for with_c in [false, true] {
                execute(
                    earlier_narrowing_fixture(dtype, with_c),
                    false,
                    Access::ReadOnly,
                    false,
                );
            }
        }
    }

    #[test]
    fn changing_batch_counts_replays_narrow_storage_calls() {
        for dtype in [DType::F16, DType::Bf16] {
            let mut graph = Graph::new();
            let a = graph.tensor(vec![IntExpr::from('b'), 3.into(), 8.into()], dtype);
            let b = graph.tensor(vec![IntExpr::from('b'), 8.into(), 4.into()], dtype);
            let c = graph.tensor(vec![IntExpr::from('b'), 3.into(), 4.into()], dtype);
            let out =
                (a.cast(DType::F32).matmul(b.cast(DType::F32)) + c.cast(DType::F32)).cast(dtype);
            let data = FxHashMap::from_iter([
                (a.id, buffers(dtype, vec![2.; 48]).0),
                (b.id, buffers(dtype, vec![0.25; 64]).0),
                (c.id, buffers(dtype, vec![1.; 24]).0),
            ]);
            let registry = whitelist::narrow_registry()
                .into_iter()
                .filter(|row| !matches!(row.label(), "CastGeneric" | "AddFunctionalGeneric"))
                .collect();
            let mut runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
            runtime
                .search(
                    &SymbolBounds::from_ranges([('b'.into(), (1, 3))]).unwrap(),
                    &[('b'.into(), 2)].into_iter().collect(),
                    &data,
                    &luminal_cuda_lite::harness_search_options(),
                )
                .unwrap();
            let mut arena = support::allocate_arena(&runtime).unwrap();
            let mut staging = runtime.allocate_staging().unwrap();
            for count in [1, 3, 2, 1] {
                runtime.set_dim('b', count);
                for (id, value, len) in [
                    (a.id, count as f32, count * 24),
                    (b.id, 0.25, count * 32),
                    (c.id, 1.0, count * 12),
                ] {
                    runtime
                        .upload(&mut arena, id, buffers(dtype, vec![value; len]).0)
                        .unwrap();
                }
                runtime.execute(arena.arena(), &mut staging).unwrap();
                let (data, binding) = runtime.download(&arena, out.id).unwrap();
                assert_eq!(
                    luminal_cuda_lite::layouts::dense_f32(&decode(&data), &binding.layout).unwrap(),
                    vec![count as f32 * 2.0 + 1.0; count * 12]
                );
            }
        }
    }

    #[test]
    fn narrow_c_and_d_support_independent_pitches_and_matching_in_place_pitches() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for batch in [false, true] {
                for in_place in [false, true] {
                    let mut f = fixture(dtype, batch, true, false, 0);
                    f.padded = true;
                    execute(f, in_place, Access::ReadOnly, true);
                }
            }
        }
    }

    fn cancellation_fixture(dtype: DType, with_c: bool, activation: usize) -> Fixture {
        let mut f = fixture_with_c(dtype, false, true, true, activation, with_c);
        let a = if dtype == DType::F16 {
            65504.0
        } else {
            half::bf16::from_bits(if with_c { 0x7f7f } else { 0x7f7e }).to_f32()
        };
        let b = if with_c {
            1.0
        } else if dtype == DType::F16 {
            1.0 + 1.0 / 1024.0
        } else {
            1.0 + 1.0 / 128.0
        };
        let c = if dtype == DType::F16 {
            32.0
        } else {
            3.0 * 2.0f32.powi(118)
        };
        let mut data = vec![
            (0..24).map(|i| if i % 8 == 0 { a } else { 0.0 }).collect(),
            (0..32).map(|i| if i < 4 { b } else { 0.0 }).collect(),
        ];
        if with_c {
            data.push(vec![c; 12]);
        }
        data.push(vec![-a; 4]);
        f.data = Some(data);
        f
    }

    #[test]
    fn packed_bias_via_broadcast_c_preserves_cancellation_before_final_rounding() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for activation in [0, 1] {
                execute(
                    cancellation_fixture(dtype, false, activation),
                    false,
                    Access::ReadOnly,
                    false,
                );
            }
        }
    }

    #[test]
    fn unsupported_narrow_epilogues_keep_finite_fp32_cancellation_via_fallback() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for with_c in [false, true] {
                for activation in [0, 1, 2] {
                    if !with_c && activation != 2 {
                        continue;
                    }
                    let mut f = cancellation_fixture(dtype, with_c, activation);
                    f.fallback = true;
                    execute(f, false, Access::ReadOnly, false);
                }
            }
        }
    }

    #[test]
    fn narrow_gelu_fallback_preserves_the_unrounded_activation_input() {
        use luminal::layout_ir::Access;
        let mut f = fixture_with_c(DType::Bf16, false, true, false, 2, false);
        f.data = Some(vec![
            (0..24)
                .map(|i| if i % 8 == 0 { 0.50390625 } else { 0.0 })
                .collect(),
            (0..32)
                .map(|i| if i < 4 { 0.734375 } else { 0.0 })
                .collect(),
        ]);
        f.fallback = true;
        execute(f, false, Access::ReadOnly, false);
    }

    #[test]
    fn narrow_outputs_keep_large_partial_sums_in_fp32() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            let mut graph = Graph::new();
            let a = graph.tensor([1, 8192], dtype);
            let b = graph.tensor([8192, 1], dtype);
            let c = graph.tensor([1, 1], dtype);
            let arithmetic = a.cast(DType::F32).matmul(b.cast(DType::F32)) + c.cast(DType::F32);
            let out = arithmetic.cast(dtype);
            let mut bv = vec![64.0; 8192];
            bv[0] = 64.5;
            bv[4096..].fill(-64.0);
            let f = Fixture {
                graph,
                inputs: vec![a, b, c],
                out,
                arithmetic,
                dtype,
                batch: false,
                activation: 0,
                with_c: true,
                data: Some(vec![vec![64.0; 8192], bv, vec![1.0]]),
                padded: false,
                fallback: false,
            };
            execute(f, false, Access::ReadOnly, false);
        }
    }

    #[test]
    fn narrow_c_reuse_and_repair_work_for_matrices_and_strided_batches() {
        use luminal::layout_ir::Access;
        for dtype in [DType::F16, DType::Bf16] {
            for batch in [false, true] {
                for (access, keep) in [
                    (Access::ReadWrite, false),
                    (Access::ReadOnly, false),
                    (Access::ReadWrite, true),
                ] {
                    execute(fixture(dtype, batch, true, false, 0), true, access, keep);
                }
            }
        }
    }
}
