//! Explicit batch broadcasts reuse A/B storage without weakening write proofs.
use super::*;

#[cfg(feature = "device")]
#[test]
fn native_dispatches_every_shared_a_b_and_both_configuration() {
    dispatch_batch_configurations(true);
}

#[test]
fn explicit_broadcasts_admit_either_or_both_shared_operands() {
    use luminal::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    for (shared_a, shared_b) in [(true, false), (false, true), (true, true)] {
        for singleton_source in [false, true] {
            let mut model = Graph::new();
            let a = if shared_a {
                if singleton_source {
                    model.tensor([1, 3, 4], DType::F32).expand([2, 3, 4])
                } else {
                    model.tensor([3, 4], DType::F32).expand_dim(0, 2)
                }
            } else {
                model.tensor([2, 3, 4], DType::F32)
            };
            let b = if shared_b {
                if singleton_source {
                    model.tensor([1, 4, 5], DType::F32).expand([2, 4, 5])
                } else {
                    model.tensor([4, 5], DType::F32).expand_dim(0, 2)
                }
            } else {
                model.tensor([2, 4, 5], DType::F32)
            };
            let _out = a.matmul(b) + model.tensor([2, 3, 5], DType::F32);
            let egraph = saturate(&model, WHITELIST);
            let (ops, tied) = calls(&egraph);
            assert!(
                ops.iter().any(|op| {
                    let call = op.resolve(&Default::default());
                    (call.a.batch_stride_in_elements == 0) == shared_a
                        && (call.b.batch_stride_in_elements == 0) == shared_b
                }),
                "A shared={shared_a}, B shared={shared_b}"
            );
            let registry =
                ConstructorRegistry::new(luminal::egglog_snippet::core_decoders()).unwrap();
            let view = EGraphView::new(&egraph, &registry);
            for op in ops {
                for (matrix, transpose, rows, cols) in [
                    (&op.a, op.trans_a, 3, 4),
                    (&op.b, op.trans_b, 4, 5),
                    (&op.d, false, 3, 5),
                ]
                .into_iter()
                .chain(op.c.iter().map(|c| (c, false, 3, 5)))
                {
                    let tensor = view.class(&matrix.tensor);
                    let literal = tensor.nodes_named("LayoutTensorLit").next().unwrap();
                    let layout = luminal::layouts::DecodedLayout::from_class(
                        &literal.child(1).unwrap(),
                        Some(matrix.dtype),
                    )
                    .unwrap();
                    let d = matrix.resolve(&Default::default());
                    assert_eq!(d.batch_count, 2);
                    for batch in 0..2 {
                        for row in 0..rows {
                            for col in 0..cols {
                                let (r, c) = if transpose { (col, row) } else { (row, col) };
                                let expected = batch * d.batch_stride_in_elements as usize
                                    + match d.order {
                                        LtOrder::Row => r * d.ld as usize + c,
                                        LtOrder::Col => c * d.ld as usize + r,
                                    };
                                assert_eq!(
                                    layout.element_index(&[batch, row, col]).unwrap(),
                                    expected
                                );
                            }
                        }
                    }
                }
                assert!(op.resolve(&Default::default()).d.batch_stride_in_elements > 0);
            }
            assert!(!tied.is_empty());
            for op in tied {
                let call = op.resolve(&Default::default());
                assert_eq!(call.c.unwrap().descriptor, call.d);
                assert!(call.d.batch_stride_in_elements > 0);
            }
        }
    }
}

#[test]
fn shared_operands_still_require_a_complete_whitelist_row() {
    let model = graph_with_sharing(DType::F32, true, false, true, false, false, (true, true));
    let (ops, tied) = calls(&saturate(&model, ""));
    assert!(ops.is_empty() && tied.is_empty());
    let (ops, _) = calls(&saturate(&model, WHITELIST));
    let op = ops
        .iter()
        .filter(|op| op.c.is_some())
        .find(|op| {
            let call = op.resolve(&Default::default());
            call.a.batch_stride_in_elements == 0 && call.b.batch_stride_in_elements == 0
        })
        .unwrap();
    let key = whitelist::key(op);
    let table = format!("(cublaslt-approved-configuration {key})");
    let (ops, tied) = calls(&saturate(&model, &table));
    assert!(!ops.is_empty() && !tied.is_empty());
    assert!(ops.iter().all(|op| whitelist::key(op) == key));
    assert!(tied.iter().all(|op| whitelist::in_place_key(op) == key));
}

#[cfg(feature = "device")]
mod execution {
    use super::*;
    use luminal::bufferize::BufferNode;
    use luminal::layout_ir::{Access, FreedBy};
    use luminal_cuda_lite::ops::cublaslt::exec::{LtComputation, LtScalars};
    use luminal_cuda_lite::{CudaBindings, HostBuffer};
    use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
    use support::TestTransfers;

    fn payload(dtype: DType, data: &[f32]) -> (HostBuffer, TypedBuffer) {
        match dtype {
            DType::F16 => {
                let values: Vec<_> = data.iter().map(|x| half::f16::from_f32(*x)).collect();
                let raw = values
                    .iter()
                    .flat_map(|x| x.to_bits().to_ne_bytes())
                    .collect();
                (
                    HostBuffer::new(PlanDtype::F16, raw).unwrap(),
                    TypedBuffer::F16(values),
                )
            }
            DType::Bf16 => {
                let values: Vec<_> = data.iter().map(|x| half::bf16::from_f32(*x)).collect();
                let raw = values
                    .iter()
                    .flat_map(|x| x.to_bits().to_ne_bytes())
                    .collect();
                (
                    HostBuffer::new(PlanDtype::Bf16, raw).unwrap(),
                    TypedBuffer::Bf16(values),
                )
            }
            DType::F32 => (
                HostBuffer::from(data.to_vec()),
                TypedBuffer::F32(data.to_vec()),
            ),
            other => panic!("unexpected input storage {other:?}"),
        }
    }

    fn run(dtype: DType, sharing: (bool, bool), device: bool, alias: bool, dynamic: bool) {
        let _execution = DEVICE_EXECUTION.lock().unwrap();
        if matches!(dtype, DType::Bf16 | DType::TF32)
            && cudarc::driver::CudaContext::new(0)
                .unwrap()
                .compute_capability()
                .unwrap()
                .0
                < 8
        {
            return;
        }
        let (shared_a, shared_b) = sharing;
        let mut graph = Graph::new();
        let count = if dynamic {
            IntExpr::from('b')
        } else {
            2.into()
        };
        let narrow = matches!(dtype, DType::F16 | DType::Bf16);
        let storage = if narrow { dtype } else { DType::F32 };
        let shape = |shared, matrix: Vec<IntExpr>| {
            if shared {
                matrix
            } else {
                [vec![count], matrix].concat()
            }
        };
        let a_source = graph.tensor(shape(shared_a, vec![4.into(), 3.into()]), storage);
        let b_source = graph.tensor(shape(shared_b, vec![4.into(), 5.into()]), storage);
        let c_source = graph.tensor(vec![count, 3.into(), 5.into()], storage);
        let wide = |x: GraphTensor| {
            if dtype == DType::TF32 {
                x.cast(DType::TF32).cast(DType::F32)
            } else {
                x.cast(DType::F32)
            }
        };
        let mut a = wide(a_source).permute(if shared_a {
            &[1, 0][..]
        } else {
            &[0, 2, 1][..]
        });
        let mut b = wide(b_source);
        if shared_a {
            a = a.expand_dim(0, count);
        }
        if shared_b {
            b = b.expand_dim(0, count);
        }
        let alpha = device.then(|| graph.tensor(Vec::<usize>::new(), DType::F32));
        let beta = device.then(|| graph.tensor(Vec::<usize>::new(), DType::F32));
        let c = if dtype == DType::TF32 {
            c_source
        } else {
            wide(c_source)
        };
        let product = a.matmul(b);
        let arithmetic = product
            * alpha
                .unwrap_or_else(|| graph.constant_f32(1.25))
                .expand_rhs(product.dims())
            + c * beta
                .unwrap_or_else(|| graph.constant_f32(-0.5))
                .expand_rhs(c.dims());
        let out = if narrow {
            arithmetic.cast(dtype)
        } else {
            arithmetic
        };
        let bounds = if dynamic {
            SymbolBounds::from_ranges([('b'.into(), (1, 3))]).unwrap()
        } else {
            Default::default()
        };
        let dims = |n| {
            if dynamic {
                [('b'.into(), n)].into_iter().collect()
            } else {
                Default::default()
            }
        };
        let inputs = |n: usize| {
            let mut data: FxHashMap<_, Vec<f32>> = [
                (a_source.id, 12 * if shared_a { 1 } else { n }, 0),
                (b_source.id, 20 * if shared_b { 1 } else { n }, 1),
                (c_source.id, 15 * n, 2),
            ]
            .into_iter()
            .map(|(id, len, seed)| {
                (
                    id,
                    (0..len)
                        .map(|i| ((i * 7 + seed) % 17) as f32 / 8.0 - 1.0)
                        .collect(),
                )
            })
            .collect();
            if let Some(alpha) = alpha {
                data.insert(alpha.id, vec![if n == 3 { -0.75 } else { 1.25 }]);
            }
            if let Some(beta) = beta {
                data.insert(beta.id, vec![if n == 3 { 2.0 } else { -0.5 }]);
            }
            data
        };
        let initial = inputs(2);
        // Stage the actual matrix storage type in both runtimes. The fused
        // call must consume its authored widening and output rounding directly.
        let input_dtype = |id| {
            if [a_source.id, b_source.id, c_source.id].contains(&id) {
                storage
            } else {
                DType::F32
            }
        };
        let reference_inputs = initial
            .iter()
            .map(|(id, data)| (*id, payload(input_dtype(*id), data).1))
            .collect();
        let mut reference = ReferenceRuntime::load_with(
            &graph,
            ReferenceBindings::dense(&graph.logical, &[arithmetic.id]),
        )
        .unwrap();
        reference
            .search(
                &bounds,
                &dims(2),
                &reference_inputs,
                &luminal_reference::harness_search_options(),
            )
            .unwrap();
        let mut bindings = CudaBindings::new();
        for id in initial.keys() {
            bindings.input(*id);
        }
        if alias {
            let buffer = bindings
                .inputs()
                .iter()
                .find(|input| input.value == c_source.id)
                .unwrap()
                .buffer;
            bindings.declare(buffer, Access::ReadWrite, FreedBy::Caller);
            bindings.output(c_source.id);
        }
        bindings.output(out.id);
        // Force the fused call and pure views so a copied batch cannot conceal
        // a missing shared-storage proof. The production whitelist stays intact
        // apart from selecting scalar-pointer mode and authored narrow output.
        let registry = whitelist::scalar_registry(device, narrow)
            .into_iter()
            .filter(|row| {
                row.label() == "IndexMapApplyViewGeneric"
                    || row.label()
                        == if alias {
                            "CublasLtAccumulateInPlace"
                        } else {
                            "CublasLtCall"
                        }
            })
            .collect();
        let mut runtime = CudaRuntime::load_with(&graph, bindings, registry).unwrap();
        let cuda_inputs = initial
            .iter()
            .map(|(id, data)| (*id, payload(input_dtype(*id), data).0))
            .collect();
        runtime
            .search(
                &bounds,
                &dims(2),
                &cuda_inputs,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap();
        let mut selected = Vec::new();
        for node in runtime.plan().unwrap().dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                if let Some(op) = op.as_any().downcast_ref::<ApprovedCublasLt>() {
                    assert!(!alias);
                    selected.push(op.resolve(&dims(2)));
                } else if let Some(op) = op.as_any().downcast_ref::<CublasLtAccumulateInPlace>() {
                    assert!(alias);
                    selected.push(op.resolve(&dims(2)));
                }
                assert_ne!(op.label(), "IndexMapApplyMaterialize");
                assert_ne!(op.label(), "CastGeneric");
            }
        }
        assert_eq!(selected.len(), 1);
        let call = &selected[0];
        assert_eq!(
            call.a.batch_stride_in_elements == 0,
            shared_a,
            "{dtype:?} {sharing:?} alias={alias}: {call:?}"
        );
        assert_eq!(
            call.b.batch_stride_in_elements == 0,
            shared_b,
            "{dtype:?} {sharing:?} alias={alias}: {call:?}"
        );
        assert_eq!(matches!(call.scalars, LtScalars::Device { .. }), device);
        assert_eq!(
            call.computation == LtComputation::TensorFloat32,
            dtype == DType::TF32
        );
        assert!(call.d.batch_stride_in_elements > 0);
        for (id, elements) in [
            (a_source.id, 12 * if shared_a { 1 } else { 2 }),
            (b_source.id, 20 * if shared_b { 1 } else { 2 }),
        ] {
            assert_eq!(
                runtime.input_arena_range(id).unwrap().bytes,
                elements * if narrow { 2 } else { 4 }
            );
        }
        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        for n in if dynamic { vec![1, 3, 2, 1] } else { vec![2] } {
            if dynamic {
                runtime.set_dim('b', n);
                reference.set_dim('b', n);
            }
            let data = inputs(n);
            for (id, values) in &data {
                let (cuda, reference_data) = payload(input_dtype(*id), values);
                reference.set_data(*id, reference_data);
                runtime.upload(&mut arena, *id, cuda).unwrap();
            }
            reference.execute().unwrap();
            let expected = reference.get_f32(arithmetic.id).unwrap();
            runtime.execute(arena.arena(), &mut staging).unwrap();
            let (raw, binding) = runtime.download(&arena, out.id).unwrap();
            let physical: Vec<f32> = if narrow {
                raw.bytes
                    .chunks_exact(2)
                    .map(|bytes| {
                        let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
                        if dtype == DType::F16 {
                            half::f16::from_bits(bits).to_f32()
                        } else {
                            half::bf16::from_bits(bits).to_f32()
                        }
                    })
                    .collect()
            } else {
                raw.as_f32().unwrap()
            };
            let actual = luminal_cuda_lite::layouts::dense_f32(&physical, &binding.layout).unwrap();
            assert_eq!(actual.len(), expected.len());
            let round = |x| match dtype {
                DType::F16 => half::f16::from_f32(x).to_f32(),
                DType::Bf16 => half::bf16::from_f32(x).to_f32(),
                _ => x,
            };
            for (got, want) in actual.into_iter().zip(expected) {
                let eps = 1e-5 * (1.0 + want.abs());
                assert!(
                    got >= round(want - eps) && got <= round(want + eps),
                    "{got} vs {want}, dtype={dtype:?}, sharing={sharing:?}, alias={alias}, count={n}"
                );
            }
            if alias {
                let (kept, _) = runtime.download(&arena, c_source.id).unwrap();
                assert_eq!(kept, payload(storage, &data[&c_source.id]).0);
            }
        }
    }

    #[test]
    fn broadcasts_execute_all_precisions_and_destinations_against_reference() {
        for dtype in [DType::F32, DType::F16, DType::Bf16, DType::TF32] {
            for sharing in [(true, false), (false, true), (true, true)] {
                for alias in [false, true] {
                    run(dtype, sharing, sharing.0 != alias, alias, false);
                }
            }
        }
    }

    #[test]
    fn symbolic_broadcasts_replay_singleton_and_larger_batches_with_device_scalars() {
        for sharing in [(true, false), (false, true), (true, true)] {
            for alias in [false, true] {
                run(DType::F32, sharing, true, alias, true);
            }
        }
    }
}

#[test]
fn padded_shared_inputs_keep_one_matrix_and_symbolic_batch_bounds() {
    use luminal_cuda_lite::bindings::BoundaryLayout;
    for shared_a in [false, true] {
        for column_major in [false, true] {
            let mut model = Graph::new();
            let count = IntExpr::from('b');
            let (rows, cols) = if shared_a { (3, 4) } else { (4, 5) };
            let shared = model.tensor([rows, cols], DType::F32);
            let shared_view = shared.expand_dim(0, count);
            let other = model.tensor(
                vec![
                    count,
                    if shared_a { 4 } else { 3 }.into(),
                    if shared_a { 5 } else { 4 }.into(),
                ],
                DType::F32,
            );
            let out = if shared_a {
                shared_view.matmul(other)
            } else {
                other.matmul(shared_view)
            };
            let mut bindings = luminal_cuda_lite::CudaBindings::new();
            bindings.input_with(
                shared.id,
                BoundaryLayout::strided_literal(if column_major { [1, 8] } else { [8, 1] }),
            );
            bindings.input(other.id);
            bindings.output_with(out.id, BoundaryLayout::strided_literal([32, 8, 1]));
            let runtime =
                CudaRuntime::load_with(&model, bindings, luminal_cuda_lite::cuda_registry())
                    .unwrap();
            for (low, high, admitted) in [
                (0, 3, false),
                (1, 3, true),
                (1, i32::MAX as usize + 1, false),
            ] {
                let bounds = SymbolBounds::from_ranges([('b'.into(), (low, high))]).unwrap();
                let (ops, _) = calls(&runtime.saturated_egraph(&bounds).unwrap());
                let witness = ops.iter().find(|op| {
                    let dims = [('b'.into(), 3)].into_iter().collect();
                    let call = op.resolve(&dims);
                    let d = if shared_a { call.a } else { call.b };
                    d.batch_stride_in_elements == 0
                        && d.ld == 8
                        && call.d.ld == 8
                        && call.d.batch_stride_in_elements == 32
                });
                assert_eq!(witness.is_some(), admitted, "bounds {low}..{high}");
                if let Some(op) = witness {
                    for count in [1, 3, 2] {
                        let dims = [('b'.into(), count)].into_iter().collect();
                        let call = op.resolve(&dims);
                        let shared = if shared_a { call.a } else { call.b };
                        assert_eq!(shared.batch_count, count as i64);
                        let capacity = if column_major {
                            (cols - 1) * 8 + rows
                        } else {
                            (rows - 1) * 8 + cols
                        };
                        validate_ld_bounds("shared", &shared, capacity).unwrap();
                        assert!(validate_ld_bounds("shared", &shared, capacity - 1).is_err());
                    }
                }
            }
        }
    }
}
