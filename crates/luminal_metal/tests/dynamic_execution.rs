#![cfg(target_os = "macos")]
mod support;
use luminal::{dtype::DType, graph::Graph, shape::IntExpr};
use luminal_metal::{MetalRuntime, harness_search_options};
#[cfg(target_os = "macos")]
use support::TestTransfers;

#[test]
fn shape_changes_reuse_one_arena() {
    let mut g = Graph::new();
    let x = g.tensor(('a', 2), DType::F32);
    let y = g.tensor(('a', 2), DType::F32);
    let out = x * y + x;
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 9))]).unwrap(),
        &[('a'.into(), 3)].into_iter().collect(),
        &[(x.id, vec![1f32; 6].into()), (y.id, vec![2f32; 6].into())]
            .into_iter()
            .collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    let before = rt.graph_stats().unwrap();
    let max = rt.arena_bytes().unwrap();
    for (iteration, n) in [2, 4, 3, 9, 5, 7, 2, 9].into_iter().enumerate() {
        rt.set_dim('a', n);
        let xdata: Vec<f32> = (0..n * 2).map(|i| (i + iteration) as f32).collect();
        let ydata: Vec<f32> = (0..n * 2).map(|i| i as f32 * 0.25 - 1.).collect();
        let expected: Vec<_> = xdata.iter().zip(&ydata).map(|(x, y)| x * y + x).collect();
        rt.upload(&mut arena_rt, x.id, xdata).unwrap();
        rt.upload(&mut arena_rt, y.id, ydata).unwrap();
        rt.execute(arena_rt.buffer()).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), expected);
        let stats = rt.graph_stats().unwrap();
        assert_eq!(stats.arena_bytes, max);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.launches - before.launches, 8);
    rt.set_dim('a', 10);
    assert!(rt.execute(arena_rt.buffer()).is_err());
}

#[test]
fn metadata_only_dimension_changes_reuse_compiled_kernel() {
    let mut g = Graph::new();
    let a = IntExpr::from('a');
    let out = g.iota(5, |c| c[0] + a);
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (1, 19))]).unwrap(),
        &[('a'.into(), 10)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();
    let arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    for n in [1, 19, 7, 1] {
        rt.set_dim('a', n);
        rt.execute(arena_rt.buffer()).unwrap();
        assert_eq!(
            rt.read_i32(&arena_rt, out.id).unwrap(),
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
    let out = (x.permute((1, 0)) + 1.).sum(0);
    let mut rt = MetalRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (2, 11))]).unwrap(),
        &[('a'.into(), 6)].into_iter().collect(),
        &[(x.id, vec![1f32; 18].into())].into_iter().collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
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
        rt.execute(arena_rt.buffer()).unwrap();
        assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), expected);
    }
}

#[test]
fn profiling_and_serving_share_dynamic_graph_execution() {
    let mut g = Graph::new();
    let x = g.tensor('a', DType::F32);
    let out = x + 2.;
    let mut rt = MetalRuntime::load(&g).unwrap();
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
    assert!(outcome.plans_profiled > 0);
    let before = rt.graph_stats().unwrap();
    assert!(before.launches > 0);
    rt.set_dim('a', 9);
    rt.upload(&mut arena_rt, x.id, vec![7f32; 9]).unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![9f32; 9]);
}
