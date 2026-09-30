#![cfg(target_os = "macos")]
mod support;
use luminal::{bufferize::BufferNode, dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options, metal_registry_filtered};
#[cfg(target_os = "macos")]
use support::TestTransfers;

#[test]
fn fused_dot_handles_broadcast_views_and_dynamic_contraction() {
    let mut g = Graph::new();
    let a = g.tensor((3, 'k'), DType::F32);
    let b = g.tensor((2, 'k'), DType::F32);
    let out = a.matmul(b.t());
    // Require the fused operation to prove that its own generated shader runs.
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "ReduceSumGeneric"),
    )
    .unwrap();
    rt.search(
        &luminal::shape::DimensionBounds::from_ranges([('k'.into(), (0, 7))]).unwrap(),
        &[('k'.into(), 3)].into_iter().collect(),
        &[(a.id, vec![1f32; 9].into()), (b.id, vec![1f32; 6].into())]
            .into_iter()
            .collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    assert!(rt.plan().unwrap().dag.node_weights().any(
        |node| matches!(node, BufferNode::Compute{op,..} if op.label() == "MulReduceSumGeneric")
    ));
    for k in [7, 1, 0, 4] {
        let av: Vec<f32> = (0..3 * k).map(|i| (i as f32 - 5.) * 0.25).collect();
        let bv: Vec<f32> = (0..2 * k).map(|i| (i as f32 - 3.) * 0.5).collect();
        let want: Vec<f32> = (0..6)
            .map(|i| (0..k).fold(0., |acc, j| acc + av[i / 2 * k + j] * bv[i % 2 * k + j]))
            .collect();
        rt.set_dim('k', k);
        rt.upload(&mut arena_rt, a.id, av).unwrap();
        rt.upload(&mut arena_rt, b.id, bv).unwrap();
        rt.execute(arena_rt.buffer()).unwrap();
        let (data, binding) = rt.download(&arena_rt, out.id).unwrap();
        let actual =
            luminal_metal::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap();
        assert_eq!(actual, want);
    }
}

#[test]
fn fused_dot_does_not_contract_multiply_and_add() {
    let mut g = Graph::new();
    let a = g.tensor(2, DType::F32);
    let b = g.tensor(2, DType::F32);
    let out = (a * b).sum(0);
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "ReduceSumGeneric"),
    )
    .unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &[(a.id, vec![1f32; 2].into()), (b.id, vec![1f32; 2].into())]
            .into_iter()
            .collect(),
        &harness_search_options(),
    )
    .unwrap();
    let mut arena_rt = support::allocate_arena(&rt).expect("allocate execution arena");
    rt.upload(&mut arena_rt, a.id, vec![-1f32, 1. + f32::EPSILON])
        .unwrap();
    rt.upload(&mut arena_rt, b.id, vec![1f32, 1. - f32::EPSILON])
        .unwrap();
    rt.execute(arena_rt.buffer()).unwrap();
    assert_eq!(rt.read_f32(&arena_rt, out.id).unwrap(), vec![0f32]);
}

#[test]
fn default_search_measures_a_matmul_chain() {
    use luminal_metal::CompileOptions;
    let mut graph = Graph::new();
    let input = graph.tensor((4, 32), DType::F32);
    let weight = graph.tensor((32, 32), DType::F32);
    let mut value = input;
    for _ in 0..6 {
        value = value.matmul(weight);
    }
    let output = value;
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    let outcome = runtime
        .search(
            &Default::default(),
            &Default::default(),
            &[
                (input.id, vec![1f32; 128].into()),
                (weight.id, vec![1f32; 1024].into()),
            ]
            .into_iter()
            .collect(),
            &CompileOptions {
                generations: 1,
                generation_size: 1,
                search_log: false,
                ..Default::default()
            },
        )
        .unwrap();
    let mut arena_runtime = support::allocate_arena(&runtime).expect("allocate execution arena");
    assert!(outcome.plans_profiled > 0);
    assert!(outcome.best_nanos > 0);
    let values: Vec<_> = (0..128).map(|i| i as f32 / 16.).collect();
    let identity: Vec<_> = (0..1024)
        .map(|i| if i / 32 == i % 32 { 1. } else { 0. })
        .collect();
    runtime
        .upload(&mut arena_runtime, input.id, values.clone())
        .unwrap();
    runtime
        .upload(&mut arena_runtime, weight.id, identity)
        .unwrap();
    runtime.execute(arena_runtime.buffer()).unwrap();
    assert_eq!(runtime.read_f32(&arena_runtime, output.id).unwrap(), values);
}

#[test]
fn deep_fork_join_search_executes_without_cost_overflow() {
    use luminal_metal::CompileOptions;
    let mut graph = Graph::new();
    let input = graph.tensor((4, 32), DType::F32);
    let weight = graph.tensor((32, 32), DType::F32);
    let mut value = input;
    // Repeated joins have exponentially many paths but linear graph depth.
    // Extraction must remain finite without a recursive byte estimate.
    for _ in 0..32 {
        let projected = value.matmul(weight);
        let twice = projected + projected;
        value = twice + twice;
    }
    let output = value;
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &[
                (input.id, vec![1f32; 128].into()),
                (weight.id, vec![1f32; 1024].into()),
            ]
            .into_iter()
            .collect(),
            &CompileOptions {
                generations: 1,
                generation_size: 1,
                search_log: false,
                ..Default::default()
            },
        )
        .unwrap();
    let mut arena_runtime = support::allocate_arena(&runtime).expect("allocate execution arena");
    let values: Vec<_> = (0..128).map(|i| i as f32 / 16.).collect();
    let identity: Vec<_> = (0..1024)
        .map(|i| if i / 32 == i % 32 { 1. } else { 0. })
        .collect();
    runtime
        .upload(&mut arena_runtime, input.id, values.clone())
        .unwrap();
    runtime
        .upload(&mut arena_runtime, weight.id, identity)
        .unwrap();
    runtime.execute(arena_runtime.buffer()).unwrap();
    assert_eq!(
        runtime.read_f32(&arena_runtime, output.id).unwrap(),
        values.iter().map(|x| x * 4f32.powi(32)).collect::<Vec<_>>()
    );
}
