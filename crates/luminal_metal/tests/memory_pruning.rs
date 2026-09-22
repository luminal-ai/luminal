#![cfg(target_os = "macos")]
use luminal::{graph::DimBucket, prelude::*};
use luminal_metal::{CompileOptions, HostBuffer, MetalRuntime};
use std::sync::atomic::{AtomicUsize, Ordering};

static SEEN_BUCKETS: AtomicUsize = AtomicUsize::new(0);
fn inspect_bucket(
    graph: &mut egraph_serialize::EGraph,
    context: &luminal_metal::egraph_postpass::PostPassContext<'_>,
) -> anyhow::Result<()> {
    assert!(!graph.nodes.is_empty());
    assert_eq!(context.arena_budget_bytes, 80 * 1024 * 1024);
    SEEN_BUCKETS.fetch_or(
        if context.bounds[&'q'.into()] == (1, 1) {
            1
        } else {
            2
        },
        Ordering::SeqCst,
    );
    Ok(())
}

#[test]
fn oversized_products_are_pruned_before_search_and_fused_dot_matches_reference() {
    let mut graph = Graph::new();
    let a = graph.tensor(('q', 256), DType::F32);
    let b = graph.tensor((256, 512), DType::F32);
    let output = a.matmul(b);
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .bind_dim_buckets(
            'q',
            vec![
                DimBucket::new(1, 1),
                DimBucket::new(2, 128).representative(128),
            ],
        )
        .unwrap();
    let weights: Vec<_> = (0..256 * 512)
        .map(|i| ((i % 7) as f32 - 3.) / 16.)
        .collect();
    let data: FxHashMap<_, HostBuffer> =
        [(a.id, vec![1.; 256].into()), (b.id, weights.clone().into())]
            .into_iter()
            .collect();
    SEEN_BUCKETS.store(0, Ordering::SeqCst);
    let options = CompileOptions {
        generations: 1,
        generation_size: 1,
        trials: 1,
        search_log: false,
        device_budget_bytes: Some(80 * 1024 * 1024),
        // The 512 KiB weight and its zero-copy views exceed this cap.
        max_intermediate_bytes: Some(256 * 1024),
        serialized_graph_passes: vec![inspect_bucket],
        ..Default::default()
    };
    let profiles = [1usize, 128]
        .into_iter()
        .map(|q| {
            (
                [('q'.into(), q)].into_iter().collect(),
                [(a.id, vec![1f32; q * 256].into())].into_iter().collect(),
            )
        })
        .collect::<Vec<_>>();
    runtime
        .search_with_profile_inputs(&data, &profiles, &options)
        .unwrap();
    assert_eq!(SEEN_BUCKETS.load(Ordering::SeqCst), 3);
    assert!(
        runtime.bucket_plans()[0]
            .outcome
            .memory_pruning
            .oversized_tensors
            > 0
    );
    let prefill = &runtime.bucket_plans()[1];
    assert!(prefill.outcome.memory_pruning.oversized_tensors > 0);
    assert!(prefill.slab_bytes <= options.device_budget_bytes.unwrap());
    assert!(
        prefill.plan.dag.node_weights().any(|node| matches!(node,
        luminal::bufferize::BufferNode::Compute { op, .. } if op.label() == "MulReduceSumGeneric"))
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
        runtime.set_data(a.id, values);
        runtime.set_data(b.id, weights.clone());
        runtime.execute().unwrap();
        assert_eq!(runtime.get_f32(output.id).unwrap(), expected);
    }
}

#[test]
fn impossible_boundary_is_rejected_before_any_device_launch() {
    let mut graph = Graph::new();
    let input = graph.tensor(1024, DType::F32);
    let _output = input + 1.;
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    let error = runtime
        .search(
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
