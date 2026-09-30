#![cfg(feature = "device")]
mod support;
use luminal::prelude::*;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "device")]
use support::TestTransfers;

static SEEN_DOMAINS: AtomicUsize = AtomicUsize::new(0);
fn inspect_domain(
    graph: &mut egraph_serialize::EGraph,
    context: &luminal_cuda_lite::egraph_postpass::PostPassContext<'_>,
) -> anyhow::Result<()> {
    assert!(!graph.nodes.is_empty());
    assert_eq!(context.arena_budget_bytes, 40 * 1024 * 1024);
    assert_eq!(context.bounds[&'q'.into()], (1, 128));
    SEEN_DOMAINS.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

#[test]
fn oversized_products_are_pruned_before_search_and_cublas_matches_reference() {
    let mut graph = Graph::new();
    let a = graph.tensor(('q', 256), DType::F32);
    let b = graph.tensor((256, 512), DType::F32);
    let output = a.matmul(b);
    let mut runtime = CudaRuntime::load(&graph).unwrap();
    let weights: Vec<_> = (0..256 * 512)
        .map(|i| ((i % 7) as f32 - 3.) / 16.)
        .collect();
    let data: FxHashMap<_, HostBuffer> = [
        (a.id, vec![1.; 128 * 256].into()),
        (b.id, weights.clone().into()),
    ]
    .into_iter()
    .collect();
    SEEN_DOMAINS.store(0, Ordering::SeqCst);
    let options = CompileOptions {
        generations: 1,
        generation_size: 1,
        trials: 1,
        search_log: false,
        device_budget_bytes: Some(40 * 1024 * 1024),
        serialized_graph_passes: vec![inspect_domain],
        ..Default::default()
    };
    let outcome = runtime
        .search(
            &luminal::shape::DimensionBounds::from_ranges([('q'.into(), (1, 128))]).unwrap(),
            &[('q'.into(), 128)].into_iter().collect(),
            &data,
            &options,
        )
        .unwrap();
    let mut arena_runtime = support::allocate_arena(&runtime).expect("allocate execution arena");
    let mut staging_arena_runtime = runtime
        .allocate_staging()
        .expect("allocate execution arena");
    assert_eq!(SEEN_DOMAINS.load(Ordering::SeqCst), 1);
    assert!(outcome.memory_pruning.oversized_tensors > 0);
    assert!(runtime.arena_bytes().unwrap() <= options.device_budget_bytes.unwrap());
    assert!(
        runtime
            .plan()
            .unwrap()
            .dag
            .node_weights()
            .any(|node| matches!(node,
        luminal::bufferize::BufferNode::Compute { op, .. } if op.label() == "CublasLt"))
    );
    for q in [128, 1, 7] {
        let values: Vec<_> = (0..q * 256).map(|i| ((i % 11) as f32 - 5.) / 8.).collect();
        let expected: Vec<f32> = (0..q * 512)
            .map(|i| {
                (0..256)
                    .map(|k| values[i / 512 * 256 + k] * weights[k * 512 + i % 512])
                    .sum()
            })
            .collect();
        runtime.set_dim('q', q);
        runtime.upload(&mut arena_runtime, a.id, values).unwrap();
        runtime
            .upload(&mut arena_runtime, b.id, weights.clone())
            .unwrap();
        runtime
            .execute(arena_runtime.arena(), &mut staging_arena_runtime)
            .unwrap();
        assert_eq!(
            runtime.read_f32(&arena_runtime, output.id).unwrap(),
            expected
        );
    }
}

#[test]
fn impossible_boundary_is_rejected_before_any_device_launch() {
    let mut graph = Graph::new();
    let input = graph.tensor(1024, DType::F32);
    let _output = input + 1.;
    let mut runtime = CudaRuntime::load(&graph).unwrap();
    let error = runtime
        .search(
            &Default::default(),
            &Default::default(),
            &Default::default(),
            &CompileOptions {
                device_budget_bytes: Some(1024),
                search_log: false,
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("1024-byte arena budget"));
    assert!(format!("{error:#}").contains("required Buffer"));
    assert_eq!(runtime.graph_stats().unwrap().launches, 0);
}
