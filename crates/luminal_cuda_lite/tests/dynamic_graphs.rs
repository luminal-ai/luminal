//! Numerical and resource-lifetime regressions for graph-only dynamic execution.
#![cfg(feature = "device")]
use luminal::{
    dtype::DType,
    graph::{DimBucket, Graph},
    shape::IntExpr,
};
use luminal_cuda_lite::{CudaRuntime, harness_search_options};

#[test]
fn bucket_switches_overlay_one_arena_and_replay_cached_graphs() {
    let mut g = Graph::new();
    let x = g.tensor(('a', 2), DType::F32);
    let y = g.tensor(('a', 2), DType::F32);
    let out = (x * y + x).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    let max = rt
        .bucket_plans()
        .iter()
        .map(|p| p.slab_bytes)
        .max()
        .unwrap();
    let mut base = None;
    for (iteration, n) in [2, 4, 3, 9, 5, 7, 2, 9].into_iter().enumerate() {
        rt.set_dim('a', n);
        let xdata: Vec<f32> = (0..n * 2).map(|i| (i + iteration) as f32).collect();
        let ydata: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.25 - 1.).collect();
        let expected: Vec<_> = xdata.iter().zip(&ydata).map(|(x, y)| x * y + x).collect();
        rt.set_data(x.id, xdata);
        rt.set_data(y.id, ydata);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), expected);
        let stats = rt.graph_stats().unwrap();
        assert_eq!(stats.arena_bytes, max);
        assert_eq!(*base.get_or_insert(stats.arena_base), stats.arena_base);
        assert_eq!(stats.arena_generation, 1);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.instantiations, 2);
    assert_eq!(stats.launches, 8);
    rt.set_dim('a', 10);
    assert!(rt.execute().is_err());
}

#[test]
fn metadata_only_dimension_change_patches_no_nodes() {
    let mut g = Graph::new();
    let a = IntExpr::from('a');
    let out = g.iota(5, |c| c[0] + a).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('a', 1, 19).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for n in [1, 19, 7, 1] {
        rt.set_dim('a', n);
        rt.execute().unwrap();
        assert_eq!(
            rt.get_i32(out.id).unwrap(),
            (n..n + 5).map(|v| v as i32).collect::<Vec<_>>()
        );
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.instantiations, 1);
    assert_eq!(stats.node_updates, 0);
    assert_eq!(stats.kernel_compilations, 1);
}

#[test]
fn dynamic_transpose_and_reduction_use_live_strides() {
    let mut g = Graph::new();
    let x = g.tensor((3, 'a'), DType::F32);
    let out = (x.permute((1, 0)) + 1.).sum(0).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('a', 2, 11).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
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
        rt.set_data(x.id, data);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), expected);
    }
    assert_eq!(rt.graph_stats().unwrap().instantiations, 1);
}

#[test]
fn cublas_geometry_changes_refresh_child_and_reuse_capture() {
    let mut g = Graph::new();
    let a = g.tensor(('m', 'k'), DType::F32);
    let b = g.tensor(('k', 'n'), DType::F32);
    let out = a.matmul(b).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    for s in ['m', 'k', 'n'] {
        rt.bind_dyn_range(s, 2, 12).unwrap();
    }
    let mut options = harness_search_options();
    options.generations = 4;
    options.generation_size = 12;
    rt.search(&Default::default(), &options).unwrap();
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
        rt.set_data(a.id, av);
        rt.set_data(b.id, bv);
        rt.execute().unwrap();
        let actual = rt.get_f32(out.id).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.host_captures, 3);
    assert_eq!(stats.host_cache_hits, 1);
    assert_eq!(stats.arena_generation, 1);
}

#[test]
fn profiling_and_serving_share_dynamic_graph_execution() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = (x + 2.).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .unwrap();
    let data = [(x.id, vec![1f32, 2., 3.].into())].into_iter().collect();
    let mut options = harness_search_options();
    options.profile_on_device = true;
    options.trials = 2;
    let outcome = rt.search(&data, &options).unwrap();
    assert!(outcome.plans_profiled > 0);
    let before = rt.graph_stats().unwrap();
    assert!(before.launches > before.instantiations);
    rt.set_dim('a', 9);
    rt.set_data(x.id, vec![7f32; 9]);
    rt.execute().unwrap();
    assert_eq!(rt.get_f32(out.id).unwrap(), vec![9f32; 9]);
}

#[test]
fn zero_extents_disable_copies_and_restore_them() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = (x + 3.).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('a', 0, 9).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for n in [0, 9, 0, 2] {
        rt.set_dim('a', n);
        rt.set_data(x.id, vec![2f32; n]);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![5f32; n]);
    }
    assert_eq!(rt.graph_stats().unwrap().instantiations, 1);
}

#[test]
fn gather_scatter_update_symbolic_coordinates() {
    let mut g = Graph::new();
    let data = g.tensor(('a', 3), DType::F32);
    let rows = g.tensor('a', DType::Int);
    let cols = g.iota(('a', 3), |c| c[1]);
    let coords = [rows.expand_dim(1, 3), cols];
    let gathered = data.gather(&coords).output();
    let scattered = (data * 0.).scatter(&coords, data + 1.).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('a', 2, 9).unwrap();
    let mut opts = harness_search_options();
    opts.generations = 4;
    opts.generation_size = 8;
    rt.search(&Default::default(), &opts).unwrap();
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
        rt.set_data(data.id, values);
        rt.set_data(rows.id, indices);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(gathered.id).unwrap(), expected_gather);
        assert_eq!(rt.get_f32(scattered.id).unwrap(), expected_scatter);
    }
    assert_eq!(rt.graph_stats().unwrap().instantiations, 1);
}

#[test]
fn installing_larger_plan_invalidates_graphs_before_arena_growth() {
    use luminal_cuda_lite::device::CudaDevice;
    fn plan(n: usize) -> luminal_cuda_lite::CudaPlan {
        let mut g = Graph::new();
        g.iota(n, |c| c[0]).output();
        let mut rt = CudaRuntime::load(&g).unwrap();
        rt.search(&Default::default(), &harness_search_options())
            .unwrap();
        rt.plan().unwrap().clone()
    }
    let mut device = CudaDevice::new(0).unwrap();
    device.install(vec![(plan(4), Default::default())]).unwrap();
    let retained = device
        .execute(0, &Default::default(), &Default::default())
        .unwrap();
    let before = device.stats();
    device
        .install(vec![(plan(4097), Default::default())])
        .unwrap();
    let result = device
        .execute(0, &Default::default(), &Default::default())
        .unwrap();
    assert_eq!(device.stats().arena_generation, before.arena_generation + 1);
    assert!(device.stats().arena_bytes > before.arena_bytes);
    assert_eq!(
        result[&0].0.as_i32().unwrap(),
        (0..4097).collect::<Vec<_>>()
    );
    assert_eq!(retained[&0].0.as_i32().unwrap(), vec![0, 1, 2, 3]);
    device.release_slab();
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
    g.iota(7, |c| c[0] + a).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('a', 1, 12).unwrap();
    rt.search(&Default::default(), &harness_search_options())
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
    let mut device = luminal_cuda_lite::device::CudaDevice::new(0).unwrap();
    device
        .install(vec![(plan, [('a'.into(), (1, 12))].into_iter().collect())])
        .unwrap();
    for n in [1usize, 12, 2] {
        let outputs = device
            .execute(
                0,
                &Default::default(),
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
fn bucketed_and_range_bound_dimensions_both_stay_symbolic() {
    let mut g = Graph::new();
    let x = g.tensor(('a', 'b'), DType::F32);
    let out = (x + 1.).output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.bind_dyn_range('b', 2, 8).unwrap();
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .unwrap();
    rt.set_dim('b', 3); // This current value must not narrow the searched interval.
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for (a, b) in [(2, 3), (4, 8), (5, 2), (9, 7)] {
        rt.set_dim('a', a);
        rt.set_dim('b', b);
        rt.set_data(x.id, vec![2f32; a * b]);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![3f32; a * b]);
    }
    assert_eq!(rt.graph_stats().unwrap().instantiations, 2);
}

#[test]
fn dynamic_cublas_bias_epilogue_rebinds_geometry() {
    let mut g = Graph::new();
    let a = g.tensor(('m', 4), DType::F32);
    let b = g.tensor((4, 'n'), DType::F32);
    let bias = g.tensor('n', DType::F32);
    let out = (a.matmul(b) + bias.expand_dim(0, 'm')).relu().output();
    let mut rt = CudaRuntime::load(&g).unwrap();
    for s in ['m', 'n'] {
        rt.bind_dyn_range(s, 2, 9).unwrap();
    }
    let mut options = harness_search_options();
    options.generations = 12;
    options.generation_size = 16;
    options.mutations = 4;
    rt.search(&Default::default(), &options).unwrap();
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
        rt.set_data(a.id, av);
        rt.set_data(b.id, bv);
        rt.set_data(bias.id, biasv);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), expected);
    }
    assert_eq!(rt.graph_stats().unwrap().host_captures, 3);
}
