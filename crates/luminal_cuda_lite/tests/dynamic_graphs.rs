//! Numerical and resource-lifetime regressions for graph-only dynamic execution.
#![cfg(feature = "device")]
mod support;
use luminal::{dtype::DType, graph::Graph, shape::IntExpr};
use luminal_cuda_lite::{CudaRuntime, harness_search_options};
#[cfg(feature = "device")]
use support::TestTransfers;

#[test]
fn shape_changes_reuse_one_arena_and_replay_cached_graphs() {
    let mut g = Graph::new();
    let x = g.tensor(vec![IntExpr::from('a'), 2.into()], DType::F32);
    let y = g.tensor(vec![IntExpr::from('a'), 2.into()], DType::F32);
    let out = x * y + x;
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap(),
        &[('a'.into(), 3)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    let max = rt.arena_bytes().unwrap();
    let mut base = None;
    for (iteration, n) in [2, 4, 3, 9, 5, 7, 2, 9].into_iter().enumerate() {
        rt.set_dim('a', n);
        let xdata: Vec<f32> = (0..n * 2).map(|i| (i + iteration) as f32).collect();
        let ydata: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.25 - 1.).collect();
        let expected: Vec<_> = xdata.iter().zip(&ydata).map(|(x, y)| x * y + x).collect();
        rt.upload(&mut arena_rt, x.id, xdata).unwrap();
        rt.upload(&mut arena_rt, y.id, ydata).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), expected);
        let stats = rt.graph_stats().unwrap();
        assert_eq!(stats.arena_bytes, max);
        assert_eq!(*base.get_or_insert(stats.arena_base), stats.arena_base);
        assert_eq!(stats.arena_generation - before.arena_generation, 1);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.instantiations - before.instantiations, 1);
    assert_eq!(stats.launches - before.launches, 8);
    rt.set_dim('a', 10);
    assert!(rt.execute(arena_rt.arena(), &mut staging_arena_rt).is_err());
}

#[test]
fn metadata_only_dimension_change_patches_no_nodes() {
    let mut g = Graph::new();
    let a = IntExpr::from('a');
    let out = g.iota(5, |c| c[0] + a);
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (1, 19))]).unwrap(),
        &[('a'.into(), 10)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    for n in [1, 19, 7, 1] {
        rt.set_dim('a', n);
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(
            rt.read_i32(&arena_rt, out.id).unwrap(),
            (n..n + 5).map(|v| v as i32).collect::<Vec<_>>()
        );
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.instantiations - before.instantiations, 1);
    assert_eq!(stats.node_updates, 0);
    assert_eq!(stats.kernel_compilations, 1);
}

#[test]
fn dynamic_transpose_and_reduction_use_live_strides() {
    let mut g = Graph::new();
    let x = g.tensor(vec![IntExpr::from(3), 'a'.into()], DType::F32);
    let out = (x.permute(vec![1, 0]) + 1.).sum(0);
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 11))]).unwrap(),
        &[('a'.into(), 6)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    for n in [2, 11, 5, 2] {
        let data: Vec<_> = (0..3 * n).map(|i| i as f32 / 2.).collect();
        let expected: Vec<_> = (0..3)
            .map(|row| {
                data[row * n..(row + 1) * n]
                    .iter()
                    .map(|x| x + 1.)
                    .sum::<f32>()
            })
            .collect();
        rt.set_dim('a', n);
        rt.upload(&mut arena_rt, x.id, data).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), expected);
    }
    assert_eq!(
        rt.graph_stats().unwrap().instantiations - before.instantiations,
        1
    );
}

#[test]
fn cublas_geometry_changes_rerecord_the_child_every_execution() {
    let mut g = Graph::new();
    let a = g.tensor(vec!['m', 'k'], DType::F32);
    let b = g.tensor(vec!['k', 'n'], DType::F32);
    let out = a.matmul(b);
    let mut rt = CudaRuntime::load_with_registry(
        &g,
        luminal_cuda_lite::cuda_registry_filtered(|row| !matches!(row.label(), "ReduceSumGeneric")),
    )
    .unwrap();

    let mut options = harness_search_options();
    options.generations = 4;
    options.generation_size = 12;
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([
            ('m'.into(), (2, 12)),
            ('k'.into(), (2, 12)),
            ('n'.into(), (2, 12)),
        ])
        .unwrap(),
        &[('m'.into(), 7), ('k'.into(), 7), ('n'.into(), 7)]
            .into_iter()
            .collect(),
        &Default::default(),
        &options,
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    assert!(rt.plan().unwrap().dag.node_weights().any(|node|matches!(node,luminal::bufferize::BufferNode::Compute{op,..} if luminal_cuda_lite::as_host_op(op.as_ref()).is_some())));
    for (m, n, k) in [(2, 3, 4), (12, 9, 7), (5, 2, 11), (2, 3, 4)] {
        let av: Vec<_> = (0..m * k).map(|i| (i % 9) as f32 / 4. - 1.).collect();
        let bv: Vec<_> = (0..k * n).map(|i| (i % 7) as f32 / 3. - 0.5).collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|i| {
                (0..k)
                    .map(|j| av[(i / n) * k + j] * bv[j * n + i % n])
                    .sum()
            })
            .collect();
        for (s, v) in [('m', m), ('n', n), ('k', k)] {
            rt.set_dim(s, v);
        }
        rt.upload(&mut arena_rt, a.id, av).unwrap();
        rt.upload(&mut arena_rt, b.id, bv).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        let actual = rt.read_f32(&arena_rt, out.id).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
    let stats = rt.graph_stats().unwrap();
    // Every execution here changes the dims, and every such execution
    // re-records the library call: one recording per launch.
    assert_eq!(
        stats.host_captures - before.host_captures,
        stats.launches - before.launches
    );
    assert_eq!(stats.arena_generation - before.arena_generation, 1);
}

#[test]
fn profiling_and_serving_share_dynamic_graph_execution() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = x + 2.;
    let mut rt = CudaRuntime::load(&g).unwrap();
    let data = [(x.id, vec![1f32, 2., 3.].into())].into_iter().collect();
    let mut options = harness_search_options();

    options.trials = 2;
    let outcome = rt
        .search(
            &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap(),
            &[('a'.into(), 3)].into_iter().collect(),
            &data,
            &options,
        )
        .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    assert!(outcome.plans_profiled > 0);
    let before = rt.graph_stats().unwrap();
    assert!(before.launches > before.instantiations);
    rt.set_dim('a', 9);
    rt.upload(&mut arena_rt, x.id, vec![7f32; 9]).unwrap();
    rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
    assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![9f32; 9]);
}

#[test]
fn zero_extents_disable_copies_and_restore_them() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = x + 3.;
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (0, 9))]).unwrap(),
        &[('a'.into(), 4)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    for n in [0, 9, 0, 2] {
        rt.set_dim('a', n);
        rt.upload(&mut arena_rt, x.id, vec![2f32; n]).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![5f32; n]);
    }
    assert_eq!(
        rt.graph_stats().unwrap().instantiations - before.instantiations,
        1
    );
}

#[test]
fn gather_scatter_update_symbolic_coordinates() {
    let mut g = Graph::new();
    let data = g.tensor(vec![IntExpr::from('a'), 3.into()], DType::F32);
    let rows = g.tensor('a', DType::Int);
    let cols = g.iota(vec![IntExpr::from('a'), 3.into()], |c| c[1]);
    let coords = [rows.expand_dim(1, 3), cols];
    let gathered = data.gather(&coords);
    let scattered = (data * 0.).scatter(&coords, data + 1.);
    let mut rt = CudaRuntime::load(&g).unwrap();
    let mut opts = harness_search_options();
    opts.generations = 4;
    opts.generation_size = 8;
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap(),
        &[('a'.into(), 5)].into_iter().collect(),
        &Default::default(),
        &opts,
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    for n in [2, 9, 5, 2] {
        let values: Vec<_> = (0..n * 3).map(|i| i as f32 / 4.).collect();
        let indices: Vec<_> = (0..n).map(|i| ((i + 1) % n) as i32).collect();
        let mut expected_gather = vec![0f32; n * 3];
        let mut expected_scatter = vec![0f32; n * 3];
        for i in 0..n {
            for j in 0..3 {
                expected_gather[i * 3 + j] = values[indices[i] as usize * 3 + j];
                expected_scatter[indices[i] as usize * 3 + j] = values[i * 3 + j] + 1.;
            }
        }
        rt.set_dim('a', n);
        rt.upload(&mut arena_rt, data.id, values).unwrap();
        rt.upload(&mut arena_rt, rows.id, indices).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(
            rt.read_f32(&arena_rt, gathered.id).unwrap(),
            expected_gather
        );
        assert_eq!(
            rt.read_f32(&arena_rt, scattered.id).unwrap(),
            expected_scatter
        );
    }
    assert_eq!(
        rt.graph_stats().unwrap().instantiations - before.instantiations,
        1
    );
}

#[test]
fn installing_larger_plan_invalidates_graphs_before_arena_growth() {
    use luminal_cuda_lite::device::CudaDevice;
    fn plan(n: usize) -> luminal_cuda_lite::CudaPlan {
        let mut g = Graph::new();
        let _ = g.iota(n, |c| c[0]);
        let mut rt = CudaRuntime::load(&g).unwrap();
        rt.search(
            &Default::default(),
            &Default::default(),
            &Default::default(),
            &harness_search_options(),
        )
        .unwrap();
        rt.plan().unwrap().clone()
    }
    let mut device = CudaDevice::new(0).unwrap().executable();
    let first_plan = plan(4);
    device
        .install((first_plan.clone(), Default::default()))
        .unwrap();
    let mut first_arena =
        support::Allocation::new(device.stream().clone(), device.slab_bytes()).unwrap();
    let mut staging_first_arena = device
        .allocate_staging(device.stats().staging_bytes)
        .unwrap();
    device
        .execute(
            first_arena.arena(),
            &mut staging_first_arena,
            &Default::default(),
        )
        .unwrap();
    let retained =
        support::download_plan(&device, &first_plan, &first_arena, &Default::default()).unwrap();
    let before = device.stats();
    let second_plan = plan(4097);
    device
        .install((second_plan.clone(), Default::default()))
        .unwrap();
    let mut second_arena =
        support::Allocation::new(device.stream().clone(), device.slab_bytes()).unwrap();
    let mut staging_second_arena = device
        .allocate_staging(device.stats().staging_bytes)
        .unwrap();
    device
        .execute(
            second_arena.arena(),
            &mut staging_second_arena,
            &Default::default(),
        )
        .unwrap();
    let result =
        support::download_plan(&device, &second_plan, &second_arena, &Default::default()).unwrap();
    assert_eq!(device.stats().arena_generation, before.arena_generation + 1);
    assert!(device.stats().arena_bytes > before.arena_bytes);
    assert_eq!(
        result[&0].0.as_i32().unwrap(),
        (0..4097).collect::<Vec<_>>()
    );
    assert_eq!(retained[&0].0.as_i32().unwrap(), vec![0, 1, 2, 3]);
    device.uninstall();
    assert_eq!(device.slab_bytes(), 0);
    assert!(!device.is_installed());
}

#[test]
fn ceil_division_in_dynamic_iota_is_evaluated_on_device() {
    // The recorder currently rejects ceil_div in iota bodies; exercise the
    // supported IntExpr IR directly at the buffer-plan/codegen boundary.
    use luminal::index_expr::IotaExpr;
    use luminal_cuda_lite::ops::iota::IotaDps;
    let mut g = Graph::new();
    let a = IntExpr::from('a');
    let _ = g.iota(7, |c| c[0] + a);
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (1, 12))]).unwrap(),
        &[('a'.into(), 6)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut plan = rt.plan().unwrap().clone();
    let mut found = false;
    for node in plan.dag.node_weights_mut() {
        if let luminal::bufferize::BufferNode::Compute { op, .. } = node
            && let Some(iota) = op.as_any().downcast_ref::<IotaDps>()
        {
            *op = Box::new(IotaDps {
                expr: Some(IotaExpr::CeilDiv(
                    Box::new(iota.expr.clone().unwrap()),
                    Box::new(IotaExpr::Lit(3)),
                )),
            });
            found = true;
        }
    }
    assert!(found);
    let mut device = luminal_cuda_lite::device::CudaDevice::new(0)
        .unwrap()
        .executable();
    device
        .install((plan.clone(), [('a'.into(), (1, 12))].into_iter().collect()))
        .unwrap();
    let mut arena = support::Allocation::new(device.stream().clone(), device.slab_bytes()).unwrap();
    let mut staging_arena = device
        .allocate_staging(device.stats().staging_bytes)
        .unwrap();
    for n in [1usize, 12, 2] {
        device
            .execute(
                arena.arena(),
                &mut staging_arena,
                &[('a'.into(), n)].into_iter().collect(),
            )
            .unwrap();
        let outputs = support::download_plan(
            &device,
            &plan,
            &arena,
            &[('a'.into(), n)].into_iter().collect(),
        )
        .unwrap();
        assert_eq!(
            outputs[&0].0.as_i32().unwrap(),
            (n..n + 7).map(|i| i.div_ceil(3) as i32).collect::<Vec<_>>()
        );
    }
    assert_eq!(device.stats().instantiations, 1);
}

#[test]
fn all_bounded_dimensions_stay_symbolic() {
    let mut g = Graph::new();
    let x = g.tensor(vec!['a', 'b'], DType::F32);
    let out = x + 1.;
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.set_dim('b', 3); // This current value must not narrow the searched interval.
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('b'.into(), (2, 8)), ('a'.into(), (2, 9))])
            .unwrap(),
        &[('b'.into(), 3), ('a'.into(), 5)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    for (a, b) in [(2, 3), (4, 8), (5, 2), (9, 7)] {
        rt.set_dim('a', a);
        rt.set_dim('b', b);
        rt.upload(&mut arena_rt, x.id, vec![2f32; a * b]).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![3f32; a * b]);
    }
    assert_eq!(
        rt.graph_stats().unwrap().instantiations - before.instantiations,
        1
    );
}

#[test]
fn dynamic_cublas_bias_epilogue_rebinds_geometry() {
    let mut g = Graph::new();
    let a = g.tensor(vec![IntExpr::from('m'), 4.into()], DType::F32);
    let b = g.tensor(vec![IntExpr::from(4), 'n'.into()], DType::F32);
    let bias = g.tensor('n', DType::F32);
    let out = (a.matmul(b) + bias.expand_dim(0, 'm')).relu();
    let mut rt = CudaRuntime::load_with_registry(
        &g,
        luminal_cuda_lite::cuda_registry_filtered(|row| {
            // Exercise the bias epilogue regardless of profiling noise. The
            // Accumulate form can also add a materialized broadcast bias, but
            // that valid alternative does not exercise the bias binding.
            !matches!(
                row.label(),
                "ReduceSumGeneric" | "AddFunctionalGeneric" | "CublasLtAccumulate"
            )
        }),
    )
    .unwrap();

    let mut options = harness_search_options();
    options.generations = 12;
    options.generation_size = 16;
    options.mutations = 4;
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('m'.into(), (2, 9)), ('n'.into(), (2, 9))])
            .unwrap(),
        &[('m'.into(), 5), ('n'.into(), 5)].into_iter().collect(),
        &Default::default(),
        &options,
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let mut staging_arena_rt = rt.allocate_staging().expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    assert!(rt.plan().unwrap().dag.node_weights().any(|node|matches!(node,luminal::bufferize::BufferNode::Compute{op,..} if luminal_cuda_lite::as_host_op(op.as_ref()).is_some()&&op.label().contains("Bias"))));
    for (m, n) in [(2, 3), (9, 7), (4, 9), (2, 3)] {
        let av: Vec<_> = (0..m * 4).map(|i| (i % 5) as f32 - 2.).collect();
        let bv: Vec<_> = (0..4 * n).map(|i| (i % 7) as f32 * 0.25 - 0.5).collect();
        let biasv: Vec<_> = (0..n).map(|i| i as f32 - 3.).collect();
        let expected: Vec<f32> = (0..m * n)
            .map(|i| {
                ((0..4)
                    .map(|j| av[i / n * 4 + j] * bv[j * n + i % n])
                    .sum::<f32>()
                    + biasv[i % n])
                    .max(0.)
            })
            .collect();
        rt.set_dim('m', m);
        rt.set_dim('n', n);
        rt.upload(&mut arena_rt, a.id, av).unwrap();
        rt.upload(&mut arena_rt, b.id, bv).unwrap();
        rt.upload(&mut arena_rt, bias.id, biasv).unwrap();
        rt.execute(arena_rt.arena(), &mut staging_arena_rt).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), expected);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(
        stats.host_captures - before.host_captures,
        stats.launches - before.launches
    );
}
