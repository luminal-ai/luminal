//! Scalar location/type is an explicit part of complete cuBLASLt admission.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::egraph_serialize::EGraph;
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::accumulate_in_place::{
    CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
};
use luminal_cuda_lite::ops::cublaslt::approved::{
    ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, Scalars, WHITELIST,
};
#[cfg(feature = "device")]
use luminal_cuda_lite::ops::cublaslt::exec::LtScalars;
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

#[derive(Clone, Copy, Debug)]
enum Source {
    Host(f32),
    Device,
    Computed,
}
#[cfg_attr(not(feature = "device"), allow(dead_code))]
struct Fixture {
    graph: Graph,
    out: GraphTensor,
    arithmetic: GraphTensor,
    inputs: Vec<(GraphTensor, Vec<f32>)>,
    scalars: [Option<GraphTensor>; 2],
    c: Option<GraphTensor>,
    dtype: DType,
    activation: usize,
    host_values: Option<(f32, f32)>,
}
fn fixture(
    batch: bool,
    dtype: DType,
    sources: [Source; 2],
    with_c: bool,
    activation: usize,
    with_bias: bool,
) -> Fixture {
    fixture_with_operand_order(batch, dtype, sources, with_c, activation, with_bias, false)
}

fn fixture_with_operand_order(
    batch: bool,
    dtype: DType,
    sources: [Source; 2],
    with_c: bool,
    activation: usize,
    with_bias: bool,
    scalar_and_c_first: bool,
) -> Fixture {
    let mut graph = Graph::new();
    let mut inputs = Vec::new();
    let mut input = |graph: &mut Graph, shape: Vec<usize>| {
        let tensor = graph.tensor(shape.clone(), DType::F32);
        let seed = inputs.len();
        inputs.push((
            tensor,
            (0..shape.iter().product())
                .map(|i| ((i * 7 + seed) % 17) as f32 / 8.0 - 1.0)
                .collect(),
        ));
        tensor
    };
    let a = input(&mut graph, if batch { vec![2, 3, 8] } else { vec![3, 8] });
    let b = input(&mut graph, if batch { vec![2, 8, 4] } else { vec![8, 4] });
    let c = with_c.then(|| input(&mut graph, if batch { vec![2, 3, 4] } else { vec![3, 4] }));
    assert!(!batch || !with_bias);
    let bias = with_bias.then(|| input(&mut graph, vec![4]));
    let mut scalars = [None, None];
    let values = std::array::from_fn::<_, 2, _>(|i| match sources[i] {
        Source::Host(value) => graph.constant_f32(value),
        source => {
            let tensor = input(&mut graph, vec![]);
            scalars[i] = Some(tensor);
            if matches!(source, Source::Computed) {
                tensor.exp2()
            } else {
                tensor
            }
        }
    });
    for (i, scalar) in scalars.iter().enumerate() {
        if let Some(scalar) = scalar {
            inputs
                .iter_mut()
                .find(|(tensor, _)| tensor.id == scalar.id)
                .unwrap()
                .1 = vec![if i == 0 { 1.25 } else { -0.5 }];
        }
    }
    let wide = |x: GraphTensor| {
        if dtype == DType::F32 {
            x
        } else {
            x.cast(dtype).cast(DType::F32)
        }
    };
    let mut arithmetic = wide(a).matmul(wide(b));
    let alpha = values[0].expand_rhs(arithmetic.dims());
    arithmetic = if scalar_and_c_first {
        alpha * arithmetic
    } else {
        arithmetic * alpha
    };
    if let Some(c) = c {
        let c = if dtype == DType::TF32 { c } else { wide(c) };
        let beta = values[1].expand_rhs(c.dims());
        arithmetic = if scalar_and_c_first {
            beta * c + arithmetic
        } else {
            arithmetic + c * beta
        };
    }
    if let Some(bias) = bias {
        arithmetic += bias.expand_dim(0, 3);
    }
    arithmetic = match activation {
        0 => arithmetic,
        1 => arithmetic.relu(),
        _ => arithmetic.gelu_fast_tanh_approximation(),
    };
    let out = if matches!(dtype, DType::F16 | DType::Bf16) {
        arithmetic.cast(dtype)
    } else {
        arithmetic
    };
    Fixture {
        graph,
        out,
        arithmetic,
        inputs,
        scalars,
        c,
        dtype,
        activation,
        host_values: match sources {
            [Source::Host(alpha), Source::Host(beta)] => {
                Some((alpha, if with_c { beta } else { 0.0 }))
            }
            _ => None,
        },
    }
}
fn saturate(graph: &Graph, table: &str) -> EGraph {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers)
        .replace(WHITELIST.trim_end(), table);
    let bound = luminal_cuda_lite::CudaBindings::leaves(&graph.logical)
        .bind(&graph.logical)
        .unwrap();
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "{program}\n{}\n{}",
                bound.prefix,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            ),
        )
        .unwrap();
    egraph.serialize(Default::default()).egraph
}
fn calls(graph: &EGraph) -> (Vec<ApprovedCublasLt>, Vec<CublasLtAccumulateInPlace>) {
    let index = SerializedIndex::new(graph);
    let mut separate = Vec::new();
    let mut in_place = Vec::new();
    for (node_id, node) in &graph.nodes {
        if node.subsumed {
            continue;
        }
        let site = ExtractionSite {
            egraph: graph,
            node_id,
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

#[test]
fn host_values_and_device_pointers_compose_with_all_existing_compute_and_output_modes() {
    for batch in [false, true] {
        for dtype in [DType::F32, DType::F16, DType::Bf16, DType::TF32] {
            let f = fixture(
                batch,
                dtype,
                [Source::Host(1.25), Source::Host(-0.5)],
                true,
                0,
                false,
            );
            let (ops, tied) = calls(&saturate(&f.graph, WHITELIST));
            assert!(
                ops.iter()
                    .any(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
            );
            assert!(
                tied.iter()
                    .any(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
            );
            assert!(
                ops.iter()
                    .any(|op| matches!(op.scalars, Scalars::Device { .. }))
            );
            for op in tied {
                let resolved = op.resolve(&Default::default());
                assert_eq!(resolved.c.unwrap().operand, op.destination_operand());
                assert_eq!(
                    op.destination_operand(),
                    resolved.form.lit_arity() + resolved.scalars.input_count() - 1
                );
            }
        }
    }
}

#[test]
fn authored_scalar_and_c_operand_orders_do_not_require_global_float_commutation() {
    for batch in [false, true] {
        for dtype in [DType::F32, DType::F16, DType::Bf16, DType::TF32] {
            let f = fixture_with_operand_order(
                batch,
                dtype,
                [Source::Host(1.25), Source::Host(-0.5)],
                true,
                0,
                false,
                true,
            );
            let (ops, tied) = calls(&saturate(&f.graph, WHITELIST));
            assert!(ops.iter().any(
                |op| op.c.is_some() && op.scalars.resolve().host_values() == Some((1.25, -0.5))
            ));
            assert!(
                tied.iter()
                    .any(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
            );
        }
    }
}

#[test]
fn graph_scalars_are_rank_zero_fp32_reads_and_cannot_be_reinterpreted_as_host_values() {
    for sources in [
        [Source::Device, Source::Device],
        [Source::Device, Source::Host(-0.5)],
        [Source::Host(1.25), Source::Device],
        [Source::Computed, Source::Device],
    ] {
        let f = fixture(false, DType::F32, sources, true, 0, false);
        let graph = saturate(&f.graph, WHITELIST);
        let (ops, _) = calls(&graph);
        let index = SerializedIndex::new(&graph);
        assert!(
            ops.iter()
                .any(|op| matches!(op.scalars, Scalars::Device { .. }))
        );
        assert!(
            !ops.iter()
                .any(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
        );
        for op in ops {
            if let Scalars::Device { alpha, beta } = op.scalars {
                for scalar in [alpha, beta] {
                    let (node_id, node) = graph
                        .nodes
                        .iter()
                        .find(|(_, node)| node.eclass == scalar && node.op == "LayoutTensorLit")
                        .unwrap();
                    let site = ExtractionSite {
                        egraph: &graph,
                        node_id,
                        node,
                        index: &index,
                    };
                    assert_eq!(
                        site.logical_dtype(&site.child_class(0)),
                        Some(PlanDtype::F32)
                    );
                }
            }
        }
    }
}

#[test]
fn scalar_pointer_modes_require_their_exact_whitelist_row() {
    let f = fixture(
        false,
        DType::F32,
        [Source::Host(1.25), Source::Host(-0.5)],
        true,
        0,
        false,
    );
    let (ops, _) = calls(&saturate(&f.graph, WHITELIST));
    let host = ops
        .iter()
        .find(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
        .unwrap();
    let key = whitelist::key(host);
    let (separate, tied) = calls(&saturate(
        &f.graph,
        &format!("(cublaslt-approved-configuration {key})"),
    ));
    assert!(
        separate
            .iter()
            .any(|op| op.scalars.resolve().host_values() == Some((1.25, -0.5)))
    );
    assert!(!tied.is_empty());
    assert!(
        separate
            .iter()
            .all(|op| matches!(op.scalars, Scalars::Host { .. }))
    );
    let (separate, tied) = calls(&saturate(&f.graph, ""));
    assert!(separate.is_empty() && tied.is_empty());
    for dtype in ["F16", "Bf16", "F64"] {
        let invalid = key.replacen(
            "(CublasLtFloat32Computation) (F32)",
            &format!("(CublasLtFloat32Computation) ({dtype})"),
            1,
        );
        let (separate, tied) = calls(&saturate(
            &f.graph,
            &format!("(cublaslt-approved-configuration {invalid})"),
        ));
        assert!(
            separate.is_empty() && tied.is_empty(),
            "a whitelist row cannot authorize the wrong scalar type"
        );
    }
}

#[test]
fn scaling_cannot_move_across_an_authored_rounding_boundary() {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 8], DType::F32);
    let b = graph.tensor([8, 4], DType::F32);
    let _out = a.matmul(b).cast(DType::F16).cast(DType::F32) * 1.25;
    let (ops, tied) = calls(&saturate(&graph, WHITELIST));
    assert!(ops.iter().all(|op| {
        op.scalars
            .resolve()
            .host_values()
            .is_none_or(|(alpha, _)| alpha == 1.0)
    }));
    assert!(tied.is_empty());
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime};
    use support::TestTransfers;

    // These runs profile and capture independent runtimes on the primary CUDA
    // context. A context synchronization must not overlap another run's capture.
    static EXECUTION: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn decode(buffer: &HostBuffer) -> Vec<f32> {
        match buffer.dtype {
            PlanDtype::F32 => buffer
                .bytes
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
                .collect(),
            dtype => buffer
                .bytes
                .chunks_exact(2)
                .map(|x| {
                    let bits = u16::from_ne_bytes(x.try_into().unwrap());
                    match dtype {
                        PlanDtype::F16 => half::f16::from_bits(bits).to_f32(),
                        PlanDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
                        _ => panic!("unsupported output {dtype:?}"),
                    }
                })
                .collect(),
        }
    }
    fn run(f: Fixture, device: bool, in_place: bool, keep_c: bool) {
        let _execution = EXECUTION.lock().unwrap();
        let mut data: FxHashMap<_, _> = f
            .inputs
            .iter()
            .map(|(tensor, values)| (tensor.id, HostBuffer::from(values.clone())))
            .collect();
        let reference_data: FxHashMap<_, _> = f
            .inputs
            .iter()
            .map(|(tensor, values)| (tensor.id, values.clone().into()))
            .collect();
        let mut reference = ReferenceRuntime::load_with(
            &f.graph,
            ReferenceBindings::dense(&f.graph.logical, &[f.arithmetic.id]),
        )
        .unwrap();
        reference
            .search(
                &Default::default(),
                &Default::default(),
                &reference_data,
                &luminal_reference::harness_search_options(),
            )
            .unwrap();
        for (id, values) in reference_data {
            reference.set_data(id, values);
        }
        let mut bindings = CudaBindings::new();
        for (tensor, _) in &f.inputs {
            bindings.input(tensor.id);
        }
        if let Some(c) = f.c {
            let index = f
                .inputs
                .iter()
                .position(|(input, _)| input.id == c.id)
                .unwrap();
            bindings.declare(
                bindings.inputs()[index].buffer,
                Access::ReadWrite,
                FreedBy::Caller,
            );
            if keep_c {
                bindings.output(c.id);
            }
        }
        bindings.output(f.out.id);
        let narrow = matches!(f.dtype, DType::F16 | DType::Bf16);
        let registry = whitelist::scalar_registry(device, narrow)
            .into_iter()
            .filter(|row| {
                !matches!(
                    row.label(),
                    "AddFunctionalGeneric"
                        | "MulFunctionalGeneric"
                        | "SelectFunctionalGeneric"
                        | "ReduceSumGeneric"
                ) && row.label()
                    != if in_place {
                        "CublasLtCall"
                    } else {
                        "CublasLtAccumulateInPlace"
                    }
            })
            .collect();
        let mut runtime = CudaRuntime::load_with(&f.graph, bindings, registry).unwrap();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        let mut calls = 0;
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                let resolved = if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    assert!(!in_place);
                    Some(op.resolve(&Default::default()))
                } else if let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    assert!(in_place);
                    Some(op.resolve(&Default::default()))
                } else {
                    None
                };
                if let Some(call) = resolved {
                    assert_eq!(matches!(call.scalars, LtScalars::Device { .. }), device);
                    assert_eq!(call.scale_dtype, PlanDtype::F32);
                    assert_eq!(call.d_dtype != PlanDtype::F32, narrow);
                    if !device {
                        assert_eq!(call.scalars.host_values(), f.host_values);
                    }
                    calls += 1;
                }
            }
        }
        assert_eq!(
            calls, 1,
            "the complete scaled expression must be one cuBLASLt call"
        );
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for (alpha, beta) in if device {
            vec![(1.25, -0.5), (-0.75, 2.0), (0.0, 1.0), (2.0, 0.0)]
        } else {
            vec![f.host_values.unwrap()]
        } {
            for (scalar, value) in f.scalars.iter().zip([alpha, beta]) {
                if let Some(scalar) = scalar {
                    data.insert(scalar.id, HostBuffer::from(vec![value]));
                    reference.set_data(scalar.id, vec![value]);
                }
            }
            reference.execute().unwrap();
            let expected = reference.get_f32(f.arithmetic.id).unwrap();
            for (id, buffer) in &data {
                runtime.upload(&mut arena, *id, buffer.clone()).unwrap();
            }
            runtime.execute(arena.arena(), &mut staging).unwrap();
            let (output, binding) = runtime.download(&arena, f.out.id).unwrap();
            let actual =
                luminal_cuda_lite::layouts::dense_f32(&decode(&output), &binding.layout).unwrap();
            let round = |x| match f.dtype {
                DType::F16 => half::f16::from_f32(x).to_f32(),
                DType::Bf16 => half::bf16::from_f32(x).to_f32(),
                _ => x,
            };
            for (got, want) in actual.iter().zip(expected) {
                let eps = (if f.activation == 2 { 5e-4 } else { 1e-5 }) * (1.0 + want.abs());
                assert!(
                    *got >= round(want - eps) && *got <= round(want + eps),
                    "{got} vs {want}, dtype={:?}, alpha={alpha}, beta={beta}",
                    f.dtype
                );
            }
            if keep_c {
                let c = f.c.unwrap();
                let (kept, binding) = runtime.download(&arena, c.id).unwrap();
                assert_eq!(
                    luminal_cuda_lite::layouts::dense_f32(&decode(&kept), &binding.layout).unwrap(),
                    f.inputs
                        .iter()
                        .find(|(input, _)| input.id == c.id)
                        .unwrap()
                        .1
                );
            }
        }
    }

    #[test]
    fn authored_scalar_and_c_first_execute_against_reference() {
        for batch in [false, true] {
            for alias in [false, true] {
                run(
                    fixture_with_operand_order(
                        batch,
                        DType::F32,
                        [Source::Device, Source::Host(-0.5)],
                        true,
                        0,
                        false,
                        true,
                    ),
                    true,
                    alias,
                    alias,
                );
            }
        }
    }

    #[test]
    fn host_and_gpu_scalars_execute_all_current_precisions_and_strided_batches() {
        for batch in [false, true] {
            for dtype in [DType::F32, DType::F16, DType::Bf16, DType::TF32] {
                for device in [false, true] {
                    for in_place in [false, true] {
                        let sources = if device {
                            [Source::Device, Source::Device]
                        } else {
                            [Source::Host(1.25), Source::Host(-0.5)]
                        };
                        run(
                            fixture(batch, dtype, sources, true, 0, false),
                            device,
                            in_place,
                            in_place,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn mixed_host_literals_and_gpu_scalars_compose_with_epilogues() {
        for sources in [
            [Source::Device, Source::Host(-0.5)],
            [Source::Host(1.25), Source::Device],
        ] {
            for activation in [0, 1, 2] {
                run(
                    fixture(false, DType::F32, sources, true, activation, false),
                    true,
                    false,
                    false,
                );
            }
        }
    }

    #[test]
    fn scaled_products_without_c_keep_beta_zero_in_both_pointer_modes() {
        for device in [false, true] {
            let source = if device {
                Source::Device
            } else {
                Source::Host(1.25)
            };
            run(
                fixture(
                    false,
                    DType::F32,
                    [source, Source::Host(0.0)],
                    false,
                    0,
                    false,
                ),
                device,
                false,
                false,
            );
        }
    }

    #[test]
    fn scalars_computed_on_the_gpu_and_bias_are_live_dependencies_on_replay() {
        for activation in [0, 1, 2] {
            run(
                fixture(
                    false,
                    DType::F32,
                    [Source::Computed, Source::Device],
                    true,
                    activation,
                    true,
                ),
                true,
                true,
                true,
            );
        }
    }

    #[test]
    fn host_scalars_support_zero_negative_and_nonunit_coefficients() {
        for (alpha, beta) in [(0.0, 1.0), (1.0, 0.0), (-0.75, 2.0), (1.25, -0.5)] {
            run(
                fixture(
                    false,
                    DType::F32,
                    [Source::Host(alpha), Source::Host(beta)],
                    true,
                    0,
                    false,
                ),
                false,
                true,
                true,
            );
        }
    }
}
