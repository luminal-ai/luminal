//! Native FP16 arithmetic has its own storage, accumulation and scalar contract.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::{
    accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
    },
    approved::{ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST},
    exec::{LtActivation, LtComputation, LtOrder, LtScalars},
};
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

fn fixture(batch: bool, sharing: (bool, bool), device: bool, transposed: (bool, bool)) -> Graph {
    let mut graph = Graph::new();
    let shape = |shared: bool, rows, cols| {
        if batch && !shared {
            vec![2, rows, cols]
        } else {
            vec![rows, cols]
        }
    };
    let mut a = graph.tensor(
        shape(
            sharing.0,
            if transposed.0 { 8 } else { 3 },
            if transposed.0 { 3 } else { 8 },
        ),
        DType::F16,
    );
    let mut b = graph.tensor(
        shape(
            sharing.1,
            if transposed.1 { 4 } else { 8 },
            if transposed.1 { 8 } else { 4 },
        ),
        DType::F16,
    );
    if transposed.0 {
        a = a.permute(if batch && !sharing.0 {
            &[0, 2, 1][..]
        } else {
            &[1, 0][..]
        });
    }
    if transposed.1 {
        b = b.permute(if batch && !sharing.1 {
            &[0, 2, 1][..]
        } else {
            &[1, 0][..]
        });
    }
    if batch && sharing.0 {
        a = a.expand_dim(0, 2);
    }
    if batch && sharing.1 {
        b = b.expand_dim(0, 2);
    }
    let c = graph.tensor(shape(false, 3, 4), DType::F16);
    let (alpha, beta) = if device {
        (
            graph.tensor([] as [usize; 0], DType::F16),
            graph.tensor([] as [usize; 0], DType::F16),
        )
    } else {
        (
            graph.constant_f32(1.5).cast(DType::F16),
            graph.constant_f32(-0.5).cast(DType::F16),
        )
    };
    let p = a.matmul(b);
    let _scaled = p * alpha.expand_rhs(p.dims());
    let _acc = _scaled + c * beta.expand_rhs(c.dims());
    graph
}

fn epilogue_fixture(device: bool, bias: bool, activation: usize) -> Graph {
    let mut graph = Graph::new();
    // Build a separate expression here so every activation uses authored F16
    // constants and the same transposed-view storage proofs as the base GEMM.
    let a = graph.tensor([8, 3], DType::F16).permute(&[1, 0]);
    let b = graph.tensor([4, 8], DType::F16).permute(&[1, 0]);
    let c = graph.tensor([3, 4], DType::F16);
    let scalar = |graph: &mut Graph, value| {
        if device {
            graph.tensor([] as [usize; 0], DType::F16)
        } else {
            graph.constant_f32(value).cast(DType::F16)
        }
    };
    let alpha = scalar(&mut graph, 1.5);
    let beta = scalar(&mut graph, -0.5);
    let product = a.matmul(b);
    let mut out = product * alpha.expand_rhs(product.dims()) + c * beta.expand_rhs(c.dims());
    if bias {
        out += graph.tensor([4], DType::F16).expand_dim(0, 3);
    }
    let _out = match activation {
        0 => out,
        1 => out.relu(),
        2 => out.gelu_fast_tanh_approximation(),
        _ => out.gelu(),
    };
    graph
}

fn calls(graph: &Graph, table: &str) -> (Vec<ApprovedCublasLt>, Vec<CublasLtAccumulateInPlace>) {
    let matchers: Vec<_> = luminal_cuda_lite::cuda_registry()
        .into_iter()
        .map(|row| row.matcher)
        .collect();
    let program = luminal::egglog_snippet::assembled_program_for(&matchers)
        .replace(WHITELIST.trim_end(), table);
    let bindings = luminal_cuda_lite::CudaBindings::leaves(&graph.logical)
        .bind(&graph.logical)
        .unwrap();
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(
            None,
            &format!(
                "{program}\n{}\n{}",
                bindings.prefix,
                luminal_cuda_lite::CudaBindings::SCHEDULE
            ),
        )
        .unwrap();
    let graph = egraph.serialize_compact_filtered(|_| true);
    let index = SerializedIndex::new(&graph);
    let mut separate = Vec::new();
    let mut tied = Vec::new();
    for (node_id, node) in &graph.nodes {
        if node.subsumed {
            continue;
        }
        let site = ExtractionSite {
            egraph: &graph,
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
            IN_PLACE => tied.push(
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
    (separate, tied)
}

#[test]
fn native_half_products_and_residuals_use_half_storage_and_typed_identities() {
    for batch in [false, true] {
        let mut graph = Graph::new();
        let a = graph.tensor(if batch { vec![2, 3, 8] } else { vec![3, 8] }, DType::F16);
        let b = graph.tensor(if batch { vec![2, 8, 4] } else { vec![8, 4] }, DType::F16);
        let c = graph.tensor(if batch { vec![2, 3, 4] } else { vec![3, 4] }, DType::F16);
        let p = a.matmul(b);
        let _out = p + c;
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(
            separate
                .iter()
                .any(|op| op.c.is_none() && op.scalars.resolve().host_values() == Some((1.0, 0.0)))
        );
        assert!(
            tied.iter()
                .any(|op| op.scalars.resolve().host_values() == Some((1.0, 1.0)))
        );
        for op in &separate {
            assert_eq!(op.computation, LtComputation::Float16);
            assert_eq!(
                (op.a.dtype, op.b.dtype, op.d.dtype, op.scale),
                (
                    PlanDtype::F16,
                    PlanDtype::F16,
                    PlanDtype::F16,
                    PlanDtype::F16
                )
            );
            assert!(matches!(
                op.scalars.resolve(),
                LtScalars::HostF16 { .. } | LtScalars::Device { .. }
            ));
        }
    }
}

#[test]
fn general_half_coefficients_compose_with_views_and_shared_batches() {
    for batch in [false, true] {
        for sharing in [(false, false), (true, false), (false, true), (true, true)] {
            for device in [false, true] {
                let graph = fixture(batch, sharing, device, (true, true));
                let (separate, tied) = calls(&graph, WHITELIST);
                let selected = |call: &luminal_cuda_lite::ops::cublaslt::exec::LtCall| {
                    if device {
                        matches!(call.scalars, LtScalars::Device { .. })
                    } else {
                        call.scalars.host_values() == Some((1.5, -0.5))
                    }
                };
                assert!(
                    separate
                        .iter()
                        .any(|op| op.c.is_some() && selected(&op.resolve(&Default::default())))
                );
                assert!(
                    tied.iter()
                        .any(|op| selected(&op.resolve(&Default::default())))
                );
                if batch {
                    assert!(tied.iter().any(|op| {
                        let c = op.resolve(&Default::default());
                        selected(&c)
                            && (c.a.batch_stride_in_elements == 0) == sharing.0
                            && (c.b.batch_stride_in_elements == 0) == sharing.1
                    }));
                }
            }
        }
    }
}

#[test]
fn half_host_literals_round_once_and_preserve_bits() {
    for (alpha, beta) in [(1.0007, -0.5), (65520.0, -0.5), (2.0f32.powi(-24), -0.5)] {
        let mut graph = Graph::new();
        let a = graph.tensor([1, 1], DType::F16);
        let b = graph.tensor([1, 1], DType::F16);
        let c = graph.tensor([1, 1], DType::F16);
        let alpha_tensor = graph
            .constant_f32(alpha)
            .cast(DType::F16)
            .expand_rhs([1, 1]);
        let beta_tensor = graph.constant_f32(beta).cast(DType::F16).expand_rhs([1, 1]);
        let _out = a.matmul(b) * alpha_tensor + c * beta_tensor;
        let (separate, _) = calls(&graph, WHITELIST);
        assert!(separate.iter().any(|op| op.c.is_some()
            && op.scalars.resolve()
                == LtScalars::HostF16 {
                    alpha_bits: half::f16::from_f32(alpha).to_bits(),
                    beta_bits: half::f16::from_f32(beta).to_bits()
                }));
    }
}

#[test]
fn half_admission_requires_its_complete_row_and_scalar_dtype() {
    let graph = fixture(false, (false, false), false, (false, false));
    let (separate, _) = calls(&graph, WHITELIST);
    let key = whitelist::key(
        separate
            .iter()
            .find(|op| op.c.is_some() && matches!(op.scalars.resolve(), LtScalars::HostF16 { .. }))
            .unwrap(),
    );
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&graph, &table);
    assert!(!separate.is_empty() && !tied.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    for table in [
        String::new(),
        table.replace(
            "(CublasLtFloat16Computation) (F16)",
            "(CublasLtFloat16Computation) (F32)",
        ),
    ] {
        let (separate, tied) = calls(&graph, &table);
        assert!(separate.is_empty() && tied.is_empty());
    }
}

#[test]
fn half_storage_and_final_rounding_do_not_authorize_half_computation() {
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let mut graph = Graph::new();
        let a = graph.tensor([3, 8], dtype).cast(DType::F32);
        let b = graph.tensor([8, 4], dtype).cast(DType::F32);
        let _out = a.matmul(b).cast(DType::F16);
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(
            separate
                .iter()
                .all(|op| op.computation != LtComputation::Float16)
        );
        assert!(
            tied.iter()
                .all(|op| op.computation != LtComputation::Float16)
        );
    }
}

#[test]
fn half_dispatch_validation_requires_half_scalar_storage() {
    let (calls, _) = calls(
        &fixture(false, (false, false), false, (false, false)),
        WHITELIST,
    );
    let mut call = calls
        .iter()
        .find(|op| op.c.is_none())
        .unwrap()
        .resolve(&Default::default());
    let elems = vec![1024; call.form.lit_arity() + call.scalars.input_count()];
    call.validate_against(&elems, 1024).unwrap();
    call.scale_dtype = PlanDtype::F32;
    assert!(call.validate_against(&elems, 1024).is_err());
    call.scale_dtype = PlanDtype::F16;
    call.scalars = LtScalars::host(1.0, 0.0);
    assert!(call.validate_against(&elems, 1024).is_err());
}

#[cfg(feature = "device")]
#[test]
fn every_exposed_shared_half_batch_configuration_executes() {
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    for sharing in [(false, false), (true, false), (false, true), (true, true)] {
        let mut witnesses = std::collections::BTreeMap::new();
        for transposed in [(false, false), (true, false), (false, true), (true, true)] {
            for device in [false, true] {
                let (separate, tied) =
                    calls(&fixture(true, sharing, device, transposed), WHITELIST);
                for op in separate {
                    let call = op.resolve(&Default::default());
                    if (call.a.batch_stride_in_elements == 0) == sharing.0
                        && (call.b.batch_stride_in_elements == 0) == sharing.1
                    {
                        witnesses
                            .entry((whitelist::key(&op), false))
                            .or_insert(call);
                    }
                }
                for op in tied {
                    let call = op.resolve(&Default::default());
                    if (call.a.batch_stride_in_elements == 0) == sharing.0
                        && (call.b.batch_stride_in_elements == 0) == sharing.1
                    {
                        witnesses
                            .entry((whitelist::in_place_key(&op), true))
                            .or_insert(call);
                    }
                }
            }
        }
        // Dense rank-three output offers row-major matrices. The ordinary
        // board separately validates both destination orientations.
        assert_eq!(witnesses.len(), 96);
        for ((_, alias), call) in witnesses {
            support::fp16::execute_call(&stream, &call, alias);
        }
    }
}

#[cfg(feature = "device")]
#[test]
fn compiled_half_calls_match_reference_with_live_coefficients_and_retained_c() {
    for batched in [false, true] {
        for alias in [false, true] {
            for source in 0..4 {
                run_compiled_half_case(batched, alias, source, false, 0);
            }
        }
    }
}

#[cfg(feature = "device")]
#[test]
fn compiled_half_bias_matches_reference_with_live_coefficients_and_retained_c() {
    for (bias, activation) in [(true, 0)] {
        for alias in [false, true] {
            // Literal, live GPU, and GPU-produced half coefficients.
            for source in 0..3 {
                run_compiled_half_case(false, alias, source, bias, activation);
            }
        }
    }
}

#[test]
fn half_bias_epilogues_keep_typed_bias_scalars_and_column_destinations() {
    for device in [false, true] {
        for bias in [false, true] {
            for activation in 0..4 {
                let graph = epilogue_fixture(device, bias, activation);
                let (separate, tied) = calls(&graph, WHITELIST);
                let matching = |call: &luminal_cuda_lite::ops::cublaslt::exec::LtCall| {
                    call.c.is_some()
                        && call.bias.is_some() == bias
                        && call.activation == LtActivation::Identity
                        && matches!(call.scalars, LtScalars::Device { .. }) == device
                };
                let separate: Vec<_> = separate
                    .iter()
                    .map(|op| op.resolve(&Default::default()))
                    .filter(matching)
                    .collect();
                let tied: Vec<_> = tied
                    .iter()
                    .map(|op| op.resolve(&Default::default()))
                    .filter(matching)
                    .collect();
                assert!(
                    !separate.is_empty() && !tied.is_empty(),
                    "bias={bias}, activation={activation}, device={device}"
                );
                for call in separate.into_iter().chain(tied) {
                    assert_eq!(call.computation, LtComputation::Float16);
                    assert_eq!(call.scale_dtype, PlanDtype::F16);
                    assert_eq!(matches!(call.scalars, LtScalars::Device { .. }), device);
                    if bias {
                        assert_eq!(call.d.order, LtOrder::Col);
                    }
                    if let Some(bias) = call.bias {
                        assert_eq!(bias.data_type, PlanDtype::F16);
                    }
                }
            }
        }
    }
}

#[test]
fn half_bias_requires_the_complete_row_and_exact_storage() {
    let graph = epilogue_fixture(false, true, 0);
    let (separate, _) = calls(&graph, WHITELIST);
    let op = separate
        .iter()
        .find(|op| op.c.is_some() && op.bias.is_some() && op.activation == LtActivation::Identity)
        .unwrap();
    let key = whitelist::key(op);
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&graph, &table);
    assert!(!separate.is_empty() && !tied.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key));
    for table in [
        String::new(),
        table.replace(
            "(CublasLtBiasDataTypeOf (F16))",
            "(CublasLtBiasDataTypeOf (Bf16))",
        ),
    ] {
        let (separate, tied) = calls(&graph, &table);
        assert!(separate.is_empty() && tied.is_empty());
    }
}

#[test]
fn half_rounding_between_gemm_and_epilogues_stays_an_arithmetic_boundary() {
    for activation in 0..4 {
        let mut graph = Graph::new();
        let a = graph.tensor([3, 8], DType::F16);
        let b = graph.tensor([8, 4], DType::F16);
        let bias = graph.tensor([4], DType::F16);
        let rounded = a.matmul(b).cast(DType::Bf16).cast(DType::F16);
        let biased = rounded + bias.expand_dim(0, 3);
        let _out = match activation {
            0 => biased,
            1 => biased.relu(),
            2 => biased.gelu_fast_tanh_approximation(),
            _ => biased.gelu(),
        };
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(!separate.is_empty());
        assert!(tied.is_empty());
        assert!(
            separate
                .iter()
                .all(|op| op.bias.is_none() && op.activation == LtActivation::Identity)
        );
    }
}

#[cfg(feature = "device")]
fn run_compiled_half_case(
    batched: bool,
    alias: bool,
    source: usize,
    bias: bool,
    activation: usize,
) {
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;
    let payload = |values: &[f32]| {
        HostBuffer::new(
            PlanDtype::F16,
            values
                .iter()
                .flat_map(|x| half::f16::from_f32(*x).to_bits().to_ne_bytes())
                .collect(),
        )
        .unwrap()
    };
    let device = source != 0 && source != 4;
    let mut graph = Graph::new();
    let a = graph.tensor(if batched { vec![2, 2, 4] } else { vec![2, 4] }, DType::F16);
    let b = graph.tensor([4, 3], DType::F16);
    let c = graph.tensor(if batched { vec![2, 2, 3] } else { vec![2, 3] }, DType::F16);
    let mut inputs = vec![a, b, c];
    let mut scalar_inputs = [None, None];
    let scalar_type = if source == 2 { DType::F32 } else { DType::F16 };
    let mut scalar = |index: usize, value: f32| {
        if source == 0 || source == 4 || source == 3 && index == 0 {
            graph
                .constant_f32(if source == 4 { 1.0 } else { value })
                .cast(DType::F16)
        } else {
            let tensor = graph.tensor([] as [usize; 0], scalar_type);
            inputs.push(tensor);
            scalar_inputs[index] = Some(tensor);
            tensor.cast(DType::F16)
        }
    };
    let alpha = scalar(0, 1.5);
    let beta = scalar(1, -0.5);
    let b_view = if batched { b.expand_dim(0, 2) } else { b };
    let product = a.matmul(b_view);
    let mut out = product * alpha.expand_rhs(product.dims()) + c * beta.expand_rhs(c.dims());
    let bias_input = bias.then(|| graph.tensor([3], DType::F16));
    if let Some(bias) = bias_input {
        inputs.push(bias);
        out += bias.expand_dim(0, 2);
    }
    out = match activation {
        0 => out,
        1 => out.relu(),
        2 => out.gelu_fast_tanh_approximation(),
        _ => out.gelu(),
    };
    let expected_activation = match activation {
        0 => LtActivation::Identity,
        1 => LtActivation::RectifiedLinear,
        _ => LtActivation::GaussianErrorLinear,
    };
    let mut a_values: Vec<_> = (0..if batched { 16 } else { 8 })
        .map(|i| (i % 5) as f32 / 4.0 - 0.5)
        .collect();
    let mut b_values: Vec<_> = (0..12).map(|i| (i % 3) as f32 / 4.0 - 0.25).collect();
    let mut c_values = vec![0.25; if batched { 12 } else { 6 }];
    if source == 4 {
        assert!(!batched);
        a_values.fill(0.0);
        b_values.fill(0.0);
        c_values = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0_f32.powi(-24),
            -2.0_f32.powi(-24),
            -0.0,
        ];
    }
    let mut bindings = CudaBindings::new();
    for input in &inputs {
        bindings.input(input.id);
    }
    bindings.declare(
        bindings.inputs()[2].buffer,
        Access::ReadWrite,
        FreedBy::Caller,
    );
    bindings.output(c.id);
    bindings.output(out.id);
    let registry = whitelist::fp16_epilogue_registry(device, bias, expected_activation)
        .into_iter()
        .filter(|row| {
            row.label() != "ReduceSumGeneric"
                && !matches!(row.label(), "AddFunctionalGeneric" | "MulFunctionalGeneric")
                && row.label()
                    != if alias {
                        "CublasLtCall"
                    } else {
                        "CublasLtAccumulateInPlace"
                    }
        })
        .collect();
    let mut runtime = CudaRuntime::load_with(&graph, bindings, registry).unwrap();
    let mut reference =
        ReferenceRuntime::load_with(&graph, ReferenceBindings::dense(&graph.logical, &[out.id]))
            .unwrap();
    let mut searched = false;
    let mut execution = None;
    for (alpha, beta) in [(1.5, -0.5), (-0.75, 2.0)] {
        if searched && !device {
            break;
        }
        let mut data: FxHashMap<_, _> = [
            (a.id, payload(&a_values)),
            (b.id, payload(&b_values)),
            (c.id, payload(&c_values)),
        ]
        .into_iter()
        .collect();
        let mut ref_data: FxHashMap<_, _> =
            [(a.id, &a_values), (b.id, &b_values), (c.id, &c_values)]
                .into_iter()
                .map(|(id, v)| {
                    (
                        id,
                        TypedBuffer::F16(v.iter().copied().map(half::f16::from_f32).collect()),
                    )
                })
                .collect();
        if let Some(bias) = bias_input {
            data.insert(bias.id, payload(&[-0.25, 0.125, 0.5]));
            ref_data.insert(
                bias.id,
                TypedBuffer::F16([-0.25, 0.125, 0.5].map(half::f16::from_f32).to_vec()),
            );
        }
        for (index, value) in [alpha, beta].into_iter().enumerate() {
            if let Some(tensor) = scalar_inputs[index] {
                data.insert(
                    tensor.id,
                    if scalar_type == DType::F32 {
                        HostBuffer::from(vec![value])
                    } else {
                        payload(&[value])
                    },
                );
                ref_data.insert(
                    tensor.id,
                    if scalar_type == DType::F32 {
                        TypedBuffer::F32(vec![value])
                    } else {
                        TypedBuffer::F16(vec![half::f16::from_f32(value)])
                    },
                );
            }
        }
        if !searched {
            // This case validates the direct shared-batch descriptor. Without
            // an explicit constraint, device timing may legitimately select
            // the equivalent tiny materialized broadcast instead, whose B
            // descriptor has a dense (nonzero) batch stride.
            let mut options = luminal_cuda_lite::harness_search_options();
            options.max_intermediate_bytes = Some(0);
            runtime
                .search(&Default::default(), &Default::default(), &data, &options)
                .unwrap();
            reference
                .search(
                    &Default::default(),
                    &Default::default(),
                    &ref_data,
                    &luminal_reference::harness_search_options(),
                )
                .unwrap();
            let calls: Vec<_> = runtime
                .plan()
                .unwrap()
                .dag
                .node_weights()
                .filter_map(|node| {
                    let BufferNode::Compute { op, .. } = node else {
                        return None;
                    };
                    if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                        Some(op.resolve(&Default::default()))
                    } else {
                        op.as_any()
                            .downcast_ref::<CublasLtAccumulateInPlace>()
                            .map(|op| op.resolve(&Default::default()))
                    }
                })
                .collect();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].computation, LtComputation::Float16);
            assert_eq!(calls[0].activation, expected_activation);
            assert_eq!(calls[0].bias.is_some(), bias);
            if bias || expected_activation != LtActivation::Identity {
                assert_eq!(calls[0].d.order, LtOrder::Col);
            }
            assert_eq!(matches!(calls[0].scalars, LtScalars::Device { .. }), device);
            if batched {
                assert_eq!(calls[0].b.batch_stride_in_elements, 0);
            }
            execution = Some((
                support::allocate_arena(&runtime).unwrap(),
                runtime.allocate_staging().unwrap(),
            ));
            searched = true;
        }
        for (id, values) in ref_data {
            reference.set_data(id, values);
        }
        reference.execute().unwrap();
        let (arena, staging) = execution.as_mut().unwrap();
        for (id, values) in data {
            runtime.upload(arena, id, values).unwrap();
        }
        runtime.execute(arena.arena(), staging).unwrap();
        let (actual, binding) = runtime.download(arena, out.id).unwrap();
        let decode = |data: &HostBuffer| {
            data.bytes
                .chunks_exact(2)
                .map(|v| half::f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32())
                .collect::<Vec<_>>()
        };
        let actual =
            luminal_cuda_lite::layouts::dense_f32(&decode(&actual), &binding.layout).unwrap();
        let expected: Vec<_> = reference
            .get_f16(out.id)
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        for (actual, expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                if expected_activation == LtActivation::RectifiedLinear {
                    // Approved ReLU uses the vendor kernel's NaN policy;
                    // both propagation and suppression to zero are accepted.
                    assert!(actual.is_nan() || *actual == 0.0);
                } else {
                    assert!(actual.is_nan());
                }
            } else if expected.is_infinite() {
                assert_eq!(*actual, expected);
            } else if expected_activation == LtActivation::GaussianErrorLinear {
                assert!(
                    (actual - expected).abs() <= 0.003 + 0.002 * expected.abs(),
                    "bias={bias}, activation={activation}, alias={alias}, source={source}: {actual} vs {expected}"
                );
            } else {
                assert_eq!(
                    *actual, expected,
                    "bias={bias}, activation={activation}, alias={alias}, source={source}"
                );
            }
        }
        let (kept, binding) = runtime.download(arena, c.id).unwrap();
        assert_eq!(
            luminal_cuda_lite::layouts::dense_f32(&decode(&kept), &binding.layout)
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            c_values
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn half_gelu_requires_authored_coefficients_and_a_valid_sign_spelling() {
    for legacy_erf in [false, true] {
        let mut graph = Graph::new();
        let a = graph.tensor([3, 8], DType::F16);
        let b = graph.tensor([8, 4], DType::F16);
        let x = a.matmul(b);
        let _out = if legacy_erf {
            let u = x * std::f32::consts::FRAC_1_SQRT_2;
            let t = (1.0 + 0.327_591_1 * u.abs()).reciprocal();
            let poly = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72)
                * t
                + 0.254_829_6)
                * t;
            // This epsilon becomes half zero. At x=0 the sign is 0/0,
            // whereas native GELU returns zero, so this graph cannot fuse.
            let sign = u * (u.abs() + 1.0e-10).reciprocal();
            0.5 * x * (1.0 + sign * (1.0 - poly * (-(u * u)).exp()))
        } else {
            // Similar shape is insufficient: a different cubic coefficient
            // cannot borrow the frontend GELU's vendor approximation contract.
            let scaled = 1.595_769_2 * x * (1.0 + 0.05 * x * x);
            x * scaled.sigmoid()
        };
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(!separate.is_empty());
        assert!(
            separate
                .iter()
                .all(|op| op.activation != LtActivation::GaussianErrorLinear)
        );
        assert!(
            tied.iter()
                .all(|op| op.activation != LtActivation::GaussianErrorLinear)
        );
    }
}

#[cfg(feature = "device")]
#[test]
#[ignore = "vendor GELU is capability-only; frontend GELU remains generic"]
fn vendor_half_gelu_capability_classifies_nan_and_infinity() {
    use cudarc::driver::DevicePtr;
    use luminal_cuda_lite::ops::cublaslt::device_call::{self, DeviceRange};
    let mut graph = Graph::new();
    let a = graph.tensor([2, 4], DType::F16);
    let b = graph.tensor([4, 3], DType::F16);
    let c = graph.tensor([2, 3], DType::F16);
    let _out = (a.matmul(b) + c).gelu_fast_tanh_approximation();
    let (separate, _) = calls(&graph, WHITELIST);
    let call = separate
        .iter()
        .find(|op| {
            op.c.as_ref().is_some_and(|c| {
                c.resolve(&Default::default()) == op.d.resolve(&Default::default())
            }) && op.activation == LtActivation::GaussianErrorLinear
                && op.scalars.resolve().host_values() == Some((1.0, 1.0))
        })
        .unwrap()
        .resolve(&Default::default());
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let c_values = [0x7e01, 0x7c00, 0xfc00, 0x0001, 0x8001, 0x8000];
    for alias in [false, true] {
        let raw = |values: &[u16]| {
            values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>()
        };
        // The vendor's transposed expression frame may exchange A and B.
        // Size physical reads from their selected descriptors.
        let zeros = |d: &luminal_cuda_lite::ops::cublaslt::exec::LtDesc| {
            let major = if d.order == LtOrder::Row {
                d.rows
            } else {
                d.cols
            };
            raw(&vec![0; (major * d.ld) as usize])
        };
        let mut payloads = vec![zeros(&call.a), zeros(&call.b)];
        payloads.resize(call.form.lit_arity(), Vec::new());
        payloads[call.c.unwrap().operand] = raw(&c_values);
        let buffers: Vec<_> = payloads
            .iter()
            .map(|v| stream.clone_htod(v).unwrap())
            .collect();
        let range = |v: &cudarc::driver::CudaSlice<u8>| DeviceRange {
            ptr: v.device_ptr(&stream).0,
            bytes: v.len(),
        };
        let inputs: Vec<_> = buffers.iter().map(range).collect();
        let d = stream.clone_htod(&raw(&[0; 6])).unwrap();
        let output = if alias {
            &buffers[call.c.unwrap().operand]
        } else {
            &d
        };
        device_call::dispatch(&call, &inputs, range(output), &stream).unwrap();
        let actual = stream.clone_dtoh(output).unwrap();
        for (bytes, expected) in
            actual
                .chunks_exact(2)
                .zip([f32::NAN, f32::INFINITY, f32::NAN, 0.0, 0.0, 0.0])
        {
            let got = half::f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f32();
            assert!(
                if expected.is_nan() {
                    got.is_nan()
                } else {
                    got == expected
                },
                "alias={alias}: {got} vs {expected}"
            );
        }
        if !alias {
            assert_eq!(
                stream
                    .clone_dtoh(&buffers[call.c.unwrap().operand])
                    .unwrap(),
                raw(&c_values)
            );
        }
    }
}

#[cfg(feature = "device")]
#[test]
#[ignore = "vendor ReLU is capability-only; frontend ReLU remains generic"]
fn vendor_half_relu_capability_exposes_its_nan_behavior() {
    for bias in [false, true] {
        for alias in [false, true] {
            run_compiled_half_case(false, alias, 4, bias, 1);
        }
    }
}
