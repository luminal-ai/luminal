//! The example applications' DEVICE-INDEPENDENT half, pinned as a test
//! (Train-2 examples landing): the conv example's exact graph + seeded
//! pairs from the mini convolution fixture search through
//! `CudaRuntime` on this backend's allow list with ZERO refusals, and
//! the resulting plan exposes kernel/buffer/output statistics. This pins the
//! CUDA search path without making the full-size applications into smoke
//! tests.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;
use mini_conv::MiniConvNet;

/// The examples' shared seeding discipline (examples/support/mod.rs,
/// verbatim from the mini measure harnesses).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

#[test]
fn conv_example_graph_searches_with_zero_refusals() {
    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let x = cx.tensor((1, 1, 5, 5), DType::F32);
    let out = model.forward(x).output();
    let pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(25, 1).into()),
        (model.conv1.weight.id, weights(18, 2).into()),
        (model.conv2.weight.id, weights(54, 3).into()),
        (model.head.weight.id, weights(6, 4).into()),
    ];
    let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.iter().cloned().collect();

    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let outcome = rt
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("cuda search");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    let b = &outcome.refusal_breakdown;
    assert_eq!(
        (
            b.extract_refusals,
            b.plan_build_refusals,
            b.execute_refusals
        ),
        (0, 0, 0),
        "examples expect zero refusals: {}",
        b.summary()
    );

    // The stats surface the examples read: plan present, with kernels,
    // buffers, and the output slot for `out`.
    let plan = rt.plan().expect("plan loaded after search");
    let mut kernels = 0usize;
    let mut outputs = 0usize;
    for idx in plan.dag.node_indices() {
        match &plan.dag[idx] {
            BufferNode::Compute { .. } => kernels += 1,
            BufferNode::BufferOutput { slots } => outputs += slots.len(),
            _ => {}
        }
    }
    assert!(kernels > 0, "conv plan has no compute kernels");
    assert_eq!(outputs, 1, "conv example binds exactly one output");
    assert!(!plan.buffers.is_empty(), "plan has no buffers");
    let _ = out;
}
