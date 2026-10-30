//! Native FP16 arithmetic has its own storage, accumulation and scalar contract.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::{
    accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
    },
    approved::{ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST},
    exec::{LtComputation, LtScalars},
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
    let graph = egraph.serialize(Default::default()).egraph;
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
    for batched in [false, true] {
        for alias in [false, true] {
            // Host half literals, GPU half scalars, GPU-produced half scalars,
            // and a host literal paired with a GPU scalar.
            for source in 0..4 {
                let device = source != 0;
                let mut graph = Graph::new();
                let a = graph.tensor(if batched { vec![2, 2, 4] } else { vec![2, 4] }, DType::F16);
                let b = graph.tensor([4, 3], DType::F16);
                let c = graph.tensor(if batched { vec![2, 2, 3] } else { vec![2, 3] }, DType::F16);
                let mut inputs = vec![a, b, c];
                let mut scalar_inputs = [None, None];
                let scalar_type = if source == 2 { DType::F32 } else { DType::F16 };
                let mut scalar = |index: usize, value: f32| {
                    if source == 0 || source == 3 && index == 0 {
                        graph.constant_f32(value).cast(DType::F16)
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
                let out =
                    product * alpha.expand_rhs(product.dims()) + c * beta.expand_rhs(c.dims());
                let a_values: Vec<_> = (0..if batched { 16 } else { 8 })
                    .map(|i| (i % 5) as f32 / 4.0 - 0.5)
                    .collect();
                let b_values: Vec<_> = (0..12).map(|i| (i % 3) as f32 / 4.0 - 0.25).collect();
                let c_values = vec![0.25; if batched { 12 } else { 6 }];
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
                let registry = whitelist::fp16_registry(device)
                    .into_iter()
                    .filter(|row| {
                        !matches!(
                            row.label(),
                            "AddFunctionalGeneric" | "MulFunctionalGeneric" | "ReduceSumGeneric"
                        ) && row.label()
                            != if alias {
                                "CublasLtCall"
                            } else {
                                "CublasLtAccumulateInPlace"
                            }
                    })
                    .collect();
                let mut runtime = CudaRuntime::load_with(&graph, bindings, registry).unwrap();
                let mut reference = ReferenceRuntime::load_with(
                    &graph,
                    ReferenceBindings::dense(&graph.logical, &[out.id]),
                )
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
                                    TypedBuffer::F16(
                                        v.iter().copied().map(half::f16::from_f32).collect(),
                                    ),
                                )
                            })
                            .collect();
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
                        runtime
                            .search(
                                &Default::default(),
                                &Default::default(),
                                &data,
                                &luminal_cuda_lite::harness_search_options(),
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
                            .map(|v| {
                                half::f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap()))
                                    .to_f32()
                            })
                            .collect::<Vec<_>>()
                    };
                    let actual =
                        luminal_cuda_lite::layouts::dense_f32(&decode(&actual), &binding.layout)
                            .unwrap();
                    let expected: Vec<_> = reference
                        .get_f16(out.id)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_f32())
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "batch={batched}, alias={alias}, source={source}"
                    );
                    let (kept, binding) = runtime.download(arena, c.id).unwrap();
                    assert_eq!(
                        luminal_cuda_lite::layouts::dense_f32(&decode(&kept), &binding.layout)
                            .unwrap(),
                        c_values
                    );
                }
            }
        }
    }
}
