#![cfg(target_os = "macos")]
use luminal::{
    dtype::DType,
    graph::{DimBucket, Graph},
    shape::IntExpr,
};
use luminal_metal::{MetalRuntime, harness_search_options};

#[test]
fn bucket_switches_overlay_one_arena() {
    let mut g = Graph::new();
    let x = g.tensor(('a', 2), DType::F32);
    let y = g.tensor(('a', 2), DType::F32);
    let out = (x * y + x).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
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
        assert_eq!(stats.arena_generation, 1);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.launches, 8);
    rt.set_dim('a', 10);
    assert!(rt.execute().is_err());
}

#[test]
fn metadata_only_dimension_changes_reuse_compiled_kernel() {
    let mut g = Graph::new();
    let a = IntExpr::from('a');
    let out = g.iota(5, |c| c[0] + a).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
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
    assert_eq!(stats.kernel_compilations, 1);
}

#[test]
fn dynamic_transpose_and_reduction_use_live_strides() {
    let mut g = Graph::new();
    let x = g.tensor((3, 'a'), DType::F32);
    let out = (x.permute((1, 0)) + 1.).sum(0).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
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
}

#[test]
fn profiling_and_serving_share_dynamic_graph_execution() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = (x + 2.).output();
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.bind_dim_buckets('a', vec![DimBucket::new(2, 4), DimBucket::new(5, 9)])
        .unwrap();
    let data = [(x.id, vec![1f32, 2., 3.].into())].into_iter().collect();
    let mut options = harness_search_options();
    options.profile_on_device = true;
    options.trials = 2;
    let outcome = rt.search(&data, &options).unwrap();
    assert!(outcome.plans_profiled > 0);
    let before = rt.graph_stats().unwrap();
    assert!(before.launches > 0);
    rt.set_dim('a', 9);
    rt.set_data(x.id, vec![7f32; 9]);
    rt.execute().unwrap();
    assert_eq!(rt.get_f32(out.id).unwrap(), vec![9f32; 9]);
}
