#![cfg(target_os = "macos")]
use luminal::{bufferize::BufferNode, dtype::DType, graph::Graph};
use luminal_metal::{MetalRuntime, harness_search_options, metal_registry_filtered};

#[test]
fn fused_dot_handles_broadcast_views_and_dynamic_contraction() {
    let mut g = Graph::new();
    let a = g.tensor((3, 'k'), DType::F32);
    let b = g.tensor((2, 'k'), DType::F32);
    let out = a.matmul(b.t()).output();
    // Require the fused operation to prove that its own generated shader runs.
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "ReduceSumGeneric"),
    )
    .unwrap();
    rt.bind_dyn_range('k', 0, 7).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
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
        rt.set_data(a.id, av);
        rt.set_data(b.id, bv);
        rt.execute().unwrap();
        let (data, binding) = rt.fetch(out.id).unwrap();
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
    let out = (a * b).sum(0).output();
    let mut rt = MetalRuntime::load_with_registry(
        &g,
        metal_registry_filtered(|row| row.label() != "ReduceSumGeneric"),
    )
    .unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    rt.set_data(a.id, vec![-1f32, 1. + f32::EPSILON]);
    rt.set_data(b.id, vec![1f32, 1. - f32::EPSILON]);
    rt.execute().unwrap();
    assert_eq!(rt.get_f32(out.id).unwrap(), vec![0f32]);
}

#[test]
fn default_search_seeds_a_compact_matmul_chain() {
    use luminal_metal::CompileOptions;
    let mut graph = Graph::new();
    let input = graph.tensor((4, 32), DType::F32);
    let weight = graph.tensor((32, 32), DType::F32);
    let mut value = input;
    for _ in 0..6 {
        value = value.matmul(weight);
    }
    let output = value.output();
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    let outcome = runtime
        .search(
            &Default::default(),
            &CompileOptions {
                generations: 1,
                generation_size: 1,
                search_log: false,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        outcome.best_heuristic_cost,
        1 + 6 * (2 * 4 * 32 * 32 * 4 + 4 * 32 * 4)
    );
    let reductions: Vec<_> = runtime
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } if op.label().contains("ReduceSum") => Some(op.label()),
            _ => None,
        })
        .collect();
    assert_eq!(reductions, vec!["MulReduceSumGeneric"; 6]);
    let values: Vec<_> = (0..128).map(|i| i as f32 / 16.).collect();
    let identity: Vec<_> = (0..1024)
        .map(|i| if i / 32 == i % 32 { 1. } else { 0. })
        .collect();
    runtime.set_data(input.id, values.clone());
    runtime.set_data(weight.id, identity);
    runtime.execute().unwrap();
    assert_eq!(runtime.get_f32(output.id).unwrap(), values);
}

#[test]
fn deep_fork_join_seed_does_not_materialize_broadcast_operands() {
    use luminal_metal::{CompileOptions, symbolic};
    let mut graph = Graph::new();
    let input = graph.tensor((4, 32), DType::F32);
    let weight = graph.tensor((32, 32), DType::F32);
    let mut value = input;
    // Recursive subtree sums overflow a u64 on this compact DAG, making
    // otherwise distinct producer costs tie and electing broadcast copies.
    for _ in 0..32 {
        let projected = value.matmul(weight);
        let twice = projected + projected;
        value = twice + twice;
    }
    let output = value.output();
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &CompileOptions {
                generations: 1,
                generation_size: 1,
                search_log: false,
                ..Default::default()
            },
        )
        .unwrap();
    for buffer in runtime.plan().unwrap().buffers.values() {
        assert!(
            symbolic::capacity_bytes(&buffer.layout, &Default::default()).unwrap() <= 4096,
            "expanded broadcast allocated: {:?}",
            buffer.layout.shape()
        );
    }
    let values: Vec<_> = (0..128).map(|i| i as f32 / 16.).collect();
    let identity: Vec<_> = (0..1024)
        .map(|i| if i / 32 == i % 32 { 1. } else { 0. })
        .collect();
    runtime.set_data(input.id, values.clone());
    runtime.set_data(weight.id, identity);
    runtime.execute().unwrap();
    assert_eq!(
        runtime.get_f32(output.id).unwrap(),
        values.iter().map(|x| x * 4f32.powi(32)).collect::<Vec<_>>()
    );
}
