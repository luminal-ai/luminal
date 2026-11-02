//! Explicit I8 -> F32 widening may read signed byte matrices directly.
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{ExtractionSite, OpMatcher, SerializedIndex};
use luminal::prelude::*;
use luminal_cuda_lite::ops::cublaslt::{
    accumulate_in_place::{
        CONSTRUCTOR as IN_PLACE, CublasLtAccumulateInPlace, CublasLtAccumulateInPlaceMatcher,
    },
    approved::{ApprovedCublasLt, ApprovedCublasLtMatcher, CONSTRUCTOR, WHITELIST},
    exec::{LtActivation, LtComputation, LtScalars},
};
#[cfg(feature = "device")]
mod support;
#[path = "support/whitelist.rs"]
mod whitelist;

fn fixture(batch: bool, sharing: (bool, bool), device: bool, view_before_cast: bool) -> Graph {
    let mut graph = Graph::new();
    let shape = |shared: bool, rows, cols| {
        if batch && !shared {
            vec![2, rows, cols]
        } else {
            vec![rows, cols]
        }
    };
    let a = graph.tensor(shape(sharing.0, 7, 3), DType::I8);
    let b = graph.tensor(shape(sharing.1, 5, 7), DType::I8);
    let transpose = |t: GraphTensor, shared: bool| {
        t.permute(if batch && !shared {
            &[0, 2, 1][..]
        } else {
            &[1, 0][..]
        })
    };
    let mut a = if view_before_cast {
        transpose(a, sharing.0).cast(DType::F32)
    } else {
        transpose(a.cast(DType::F32), sharing.0)
    };
    let mut b = if view_before_cast {
        transpose(b, sharing.1).cast(DType::F32)
    } else {
        transpose(b.cast(DType::F32), sharing.1)
    };
    if batch && sharing.0 {
        a = a.expand_dim(0, 2);
    }
    if batch && sharing.1 {
        b = b.expand_dim(0, 2);
    }
    let c = graph.tensor(shape(false, 3, 5), DType::F32);
    let (alpha, beta) = if device {
        (
            graph.tensor(Vec::<usize>::new(), DType::F32),
            graph.tensor(Vec::<usize>::new(), DType::F32),
        )
    } else {
        (graph.constant_f32(1.5), graph.constant_f32(-0.5))
    };
    let p = a.matmul(b);
    let _plain = p;
    let _scaled = p * alpha.expand_rhs(p.dims());
    let _out = _scaled + c * beta.expand_rhs(c.dims());
    graph
}
fn is_int8(op: &ApprovedCublasLt) -> bool {
    op.a.dtype == PlanDtype::I8 || op.b.dtype == PlanDtype::I8
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
fn explicit_widenings_admit_int8_storage_views_and_independent_shared_batches() {
    for (batch, sharing) in [
        (false, (false, false)),
        (true, (false, false)),
        (true, (true, false)),
        (true, (false, true)),
        (true, (true, true)),
    ] {
        for device in [false, true] {
            for before in [false, true] {
                let (separate, tied) = calls(&fixture(batch, sharing, device, before), WHITELIST);
                let selected = |call: &luminal_cuda_lite::ops::cublaslt::exec::LtCall| {
                    matches!(call.scalars, LtScalars::Device { .. }) == device
                        && (device || call.scalars.host_values() == Some((1.5, -0.5)))
                };
                assert!(separate.iter().any(|op| is_int8(op)
                    && op.c.is_some()
                    && selected(&op.resolve(&Default::default()))));
                assert!(
                    tied.iter().any(|op| op.a.dtype == PlanDtype::I8
                        && selected(&op.resolve(&Default::default())))
                );
                // Materialized broadcasts are valid too; require a direct
                // shared-storage alternative without rejecting those choices.
                if batch {
                    assert!(separate.iter().filter(|op| is_int8(op)).any(|op| {
                        let call = op.resolve(&Default::default());
                        selected(&call)
                            && (
                                call.a.batch_stride_in_elements == 0,
                                call.b.batch_stride_in_elements == 0,
                            ) == sharing
                    }));
                }
                for op in separate.iter().filter(|op| is_int8(op)) {
                    assert_eq!(
                        (op.computation, op.a.dtype, op.b.dtype, op.d.dtype, op.scale),
                        (
                            LtComputation::Float32,
                            PlanDtype::I8,
                            PlanDtype::I8,
                            PlanDtype::F32,
                            PlanDtype::F32
                        )
                    );
                    assert!(op.c.as_ref().is_none_or(|c| c.dtype == PlanDtype::F32));
                    assert!(op.bias.is_none());
                    assert_eq!(op.activation, LtActivation::Identity);
                }
            }
        }
    }
}

#[test]
fn exact_complete_int8_rows_gate_both_destination_forms() {
    let graph = fixture(false, (false, false), false, false);
    let (separate, _) = calls(&graph, WHITELIST);
    let key = whitelist::key(
        separate
            .iter()
            .find(|op| is_int8(op) && op.c.is_some())
            .unwrap(),
    );
    let table = format!("(cublaslt-approved-configuration {key})");
    let (separate, tied) = calls(&graph, &table);
    assert!(!separate.is_empty() && !tied.is_empty());
    assert!(separate.iter().all(|op| whitelist::key(op) == key));
    assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key));
    for table in [
        String::new(),
        table.replace(
            "(CublasLtFloat32Computation) (F32)",
            "(CublasLtFloat32Computation) (F16)",
        ),
    ] {
        let (separate, tied) = calls(&graph, &table);
        assert!(separate.is_empty() && tied.is_empty());
    }
}

#[test]
fn integer_arithmetic_unsigned_and_mixed_storage_cannot_select_the_int8_contract() {
    for (left, right, compute) in [
        (DType::I8, DType::I8, DType::I8),
        (DType::I8, DType::I8, DType::Int),
        (DType::U8, DType::U8, DType::F32),
        (DType::I8, DType::F32, DType::F32),
        (DType::F32, DType::I8, DType::F32),
    ] {
        let mut graph = Graph::new();
        let a = graph.tensor([3, 7], left).cast(compute);
        let b = graph.tensor([7, 5], right).cast(compute);
        let _out = a.matmul(b).cast(DType::F32);
        let (separate, tied) = calls(&graph, WHITELIST);
        assert!(!separate.iter().any(is_int8));
        assert!(
            tied.iter()
                .all(|op| op.a.dtype != PlanDtype::I8 && op.b.dtype != PlanDtype::I8)
        );
    }
}

#[test]
fn input_scaling_and_offsets_stay_before_the_gemm() {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 7], DType::I8).cast(DType::F32);
    let b = graph.tensor([7, 5], DType::I8).cast(DType::F32);
    let _scaled = (a * 0.5).matmul(b);
    let _offset = (a + 1.0).matmul(b);
    let (separate, _) = calls(&graph, WHITELIST);
    assert!(!separate.iter().any(is_int8));
}

#[test]
fn int8_c_bias_and_narrow_outputs_do_not_gain_native_storage_permissions() {
    let mut graph = Graph::new();
    let a = graph.tensor([3, 7], DType::I8).cast(DType::F32);
    let b = graph.tensor([7, 5], DType::I8).cast(DType::F32);
    let c = graph.tensor([3, 5], DType::I8).cast(DType::F32);
    let bias = graph
        .tensor([5], DType::I8)
        .cast(DType::F32)
        .expand_dim(0, 3);
    let p = a.matmul(b);
    let out = p + c;
    let _biased = out + bias;
    let _relu = out.relu();
    let _rounded = out.cast(DType::F16);
    let (separate, _) = calls(&graph, WHITELIST);
    assert!(separate.iter().any(|op| is_int8(op) && op.c.is_some()));
    for op in separate.iter().filter(|op| is_int8(op)) {
        assert!(op.bias.is_none());
        assert_eq!(op.activation, LtActivation::Identity);
        assert_eq!(op.d.dtype, PlanDtype::F32);
        assert!(op.c.as_ref().is_none_or(|c| c.dtype == PlanDtype::F32));
    }
}

#[test]
fn earlier_integer_narrowing_remains_in_the_operand_storage() {
    let mut graph = Graph::new();
    let source = graph.tensor([3, 7], DType::I16);
    let a = source.cast(DType::I8).cast(DType::F32);
    let b = graph.tensor([7, 5], DType::I8).cast(DType::F32);
    let _out = a.matmul(b);
    let (separate, _) = calls(&graph, WHITELIST);
    assert!(separate.iter().any(is_int8));
    for op in separate.iter().filter(|op| is_int8(op)) {
        assert_eq!((op.a.dtype, op.b.dtype), (PlanDtype::I8, PlanDtype::I8));
    }
}

#[cfg(feature = "device")]
#[test]
fn compiled_int8_calls_match_reference_preserve_narrowing_and_read_live_scalars() {
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;
    for batched in [false, true] {
        for alias in [false, true] {
            // Host/GPU scalars, GPU-produced alpha plus earlier integer
            // narrowing, and a host literal paired with a GPU scalar.
            for source in 0..4 {
                let device = source != 0;
                let mut graph = Graph::new();
                let earlier_narrowing = source == 2;
                let a = graph.tensor(
                    if batched { vec![2, 2, 4] } else { vec![2, 4] },
                    if earlier_narrowing {
                        DType::I16
                    } else {
                        DType::I8
                    },
                );
                let b = graph.tensor([4, 3], DType::I8);
                let c = graph.tensor(if batched { vec![2, 2, 3] } else { vec![2, 3] }, DType::F32);
                let mut inputs = vec![a, b, c];
                let mut scalar_inputs = [None, None];
                let mut scalar = |index: usize, value: f32| {
                    if source == 0 || source == 3 && index == 0 {
                        graph.constant_f32(value)
                    } else {
                        let tensor = graph.tensor(Vec::<usize>::new(), DType::F32);
                        inputs.push(tensor);
                        scalar_inputs[index] = Some(tensor);
                        if source == 2 && index == 0 {
                            tensor.exp2()
                        } else {
                            tensor
                        }
                    }
                };
                let alpha = scalar(0, 1.5);
                let beta = scalar(1, -0.5);
                let b_view = if batched {
                    b.cast(DType::F32).expand_dim(0, 2)
                } else {
                    b.cast(DType::F32)
                };
                let product = a.cast(DType::I8).cast(DType::F32).matmul(b_view);
                let out =
                    product * alpha.expand_rhs(product.dims()) + c * beta.expand_rhs(c.dims());
                let a_values: Vec<i16> = (0..if batched { 16 } else { 8 })
                    .map(|i| {
                        if earlier_narrowing {
                            [-129, 128, 255, 256, 511, -256, -257][i % 7]
                        } else {
                            [-128, -127, -3, 0, 1, 17, 126, 127][i % 8]
                        }
                    })
                    .collect();
                let b_values: Vec<i8> = (0..12).map(|i| [-128, -1, 0, 1, 63, 127][i % 6]).collect();
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
                let registry = whitelist::int8_registry(device)
                    .into_iter()
                    .filter(|row| {
                        !matches!(
                            row.label(),
                            "AddFunctionalGeneric"
                                | "MulFunctionalGeneric"
                                | "ReduceSumGeneric"
                                | "IndexMapApplyMaterializeGeneric"
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
                for (alpha, beta) in if source == 2 {
                    [(1.0, -0.5), (2.0, 2.0)]
                } else {
                    [(1.5, -0.5), (-0.75, 2.0)]
                } {
                    if searched && !device {
                        break;
                    }
                    let a_payload = HostBuffer::new(
                        if earlier_narrowing {
                            PlanDtype::I16
                        } else {
                            PlanDtype::I8
                        },
                        if earlier_narrowing {
                            a_values.iter().flat_map(|v| v.to_ne_bytes()).collect()
                        } else {
                            a_values.iter().map(|v| *v as i8 as u8).collect()
                        },
                    )
                    .unwrap();
                    let b_payload =
                        HostBuffer::new(PlanDtype::I8, b_values.iter().map(|v| *v as u8).collect())
                            .unwrap();
                    let mut data: FxHashMap<_, _> = [
                        (a.id, a_payload),
                        (b.id, b_payload),
                        (c.id, HostBuffer::from(c_values.clone())),
                    ]
                    .into_iter()
                    .collect();
                    let a_ref = if earlier_narrowing {
                        TypedBuffer::I16(a_values.clone())
                    } else {
                        TypedBuffer::I8(a_values.iter().map(|v| *v as i8).collect())
                    };
                    let mut ref_data: FxHashMap<_, _> = [
                        (a.id, a_ref),
                        (b.id, TypedBuffer::I8(b_values.clone())),
                        (c.id, TypedBuffer::F32(c_values.clone())),
                    ]
                    .into_iter()
                    .collect();
                    for (index, value) in [alpha, beta].into_iter().enumerate() {
                        if let Some(tensor) = scalar_inputs[index] {
                            data.insert(tensor.id, HostBuffer::from(vec![value]));
                            ref_data.insert(tensor.id, TypedBuffer::F32(vec![value]));
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
                        assert_eq!(
                            (
                                calls[0].a_dtype,
                                calls[0].b_dtype,
                                calls[0].d_dtype,
                                calls[0].scale_dtype
                            ),
                            (PlanDtype::I8, PlanDtype::I8, PlanDtype::F32, PlanDtype::F32)
                        );
                        let cast_count=runtime.plan().unwrap().dag.node_weights().filter(|node| matches!(node,BufferNode::Compute {op,..} if op.label()=="CastGeneric")).count();
                        assert_eq!(
                            cast_count,
                            usize::from(earlier_narrowing),
                            "input widening must be absorbed, earlier narrowing must execute"
                        );
                        assert_eq!(calls[0].computation, LtComputation::Float32);
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
                    let actual = luminal_cuda_lite::layouts::dense_f32(
                        &actual.as_f32().unwrap(),
                        &binding.layout,
                    )
                    .unwrap();
                    let expected = reference.get_f32(out.id).unwrap();
                    assert_eq!(
                        &actual, expected,
                        "batch={batched}, alias={alias}, source={source}"
                    );
                    let (kept, binding) = runtime.download(arena, c.id).unwrap();
                    assert_eq!(
                        luminal_cuda_lite::layouts::dense_f32(
                            &kept.as_f32().unwrap(),
                            &binding.layout
                        )
                        .unwrap(),
                        c_values
                    );
                }
            }
        }
    }
}

#[cfg(feature = "device")]
#[test]
fn long_int8_contraction_accumulates_in_fp32_past_the_int32_range() {
    use cudarc::driver::{CudaContext, DevicePtr};
    use luminal_cuda_lite::ops::cublaslt::device_call::{DeviceRange, dispatch};
    let k = 140_000;
    let mut graph = Graph::new();
    let a = graph.tensor([1, k], DType::I8);
    let b = graph.tensor([k, 1], DType::I8);
    let _out = a.cast(DType::F32).matmul(b.cast(DType::F32));
    let (separate, _) = calls(&graph, WHITELIST);
    let call = separate
        .iter()
        .find(|op| is_int8(op) && !op.trans_a && !op.trans_b)
        .unwrap()
        .resolve(&Default::default());
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let a = stream.clone_htod(&vec![i8::MIN; k]).unwrap();
    let b = stream.clone_htod(&vec![i8::MIN; k]).unwrap();
    let out = stream.clone_htod(&[0.0f32]).unwrap();
    let inputs = [
        DeviceRange {
            ptr: a.device_ptr(&stream).0,
            bytes: k,
        },
        DeviceRange {
            ptr: b.device_ptr(&stream).0,
            bytes: k,
        },
    ];
    dispatch(
        &call,
        &inputs,
        DeviceRange {
            ptr: out.device_ptr(&stream).0,
            bytes: 4,
        },
        &stream,
    )
    .unwrap();
    // Every product is 2^14, and all partial sums are exact in FP32.
    // The result exceeds i32::MAX, distinguishing integer accumulation.
    let expected = k as f32 * 16384.0;
    assert!(f64::from(expected) > f64::from(i32::MAX));
    assert_eq!(stream.clone_dtoh(&out).unwrap(), vec![expected]);
}

#[test]
fn symbolic_int8_batch_descriptors_keep_byte_storage_and_positive_geometry() {
    let mut graph = Graph::new();
    let a = graph.tensor(['g', 'm', 'k'], DType::I8).cast(DType::F32);
    let b = graph.tensor(['g', 'k', 'n'], DType::I8).cast(DType::F32);
    let _out = a.matmul(b);
    let runtime = luminal_cuda_lite::CudaRuntime::load(&graph).unwrap();
    let bounds = SymbolBounds::from_ranges([
        ('g'.into(), (1, 3)),
        ('m'.into(), (2, 4)),
        ('k'.into(), (1, 9)),
        ('n'.into(), (1, 5)),
    ])
    .unwrap();
    let egraph = runtime.saturated_egraph(&bounds).unwrap();
    let index = SerializedIndex::new(&egraph);
    let dims = [
        ('g'.into(), 2),
        ('m'.into(), 3),
        ('k'.into(), 7),
        ('n'.into(), 5),
    ]
    .into_iter()
    .collect();
    let mut found = false;
    for (node_id, node) in &egraph.nodes {
        if node.subsumed || node.op != CONSTRUCTOR {
            continue;
        }
        let op = ApprovedCublasLtMatcher.extract(&ExtractionSite {
            egraph: &egraph,
            node_id,
            node,
            index: &index,
        });
        let op = op.as_any().downcast_ref::<ApprovedCublasLt>().unwrap();
        if !is_int8(op) {
            continue;
        }
        found = true;
        let call = op.resolve(&dims);
        assert_eq!((call.m, call.n, call.k), (3, 5, 7));
        assert_eq!(
            (call.a.batch_count, call.b.batch_count, call.d.batch_count),
            (2, 2, 2)
        );
        assert_eq!(
            (call.a_dtype, call.b_dtype, call.d_dtype),
            (PlanDtype::I8, PlanDtype::I8, PlanDtype::F32)
        );
    }
    assert!(found);
}
