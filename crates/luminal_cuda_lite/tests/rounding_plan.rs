//! Plan-level coverage for the rounding logical ops and the explicit
//! truncating float -> int conversion on CUDA Lite: they must saturate,
//! elect, and expose a kernel interface (device execution is exercised by
//! the reference and Metal suites; this host has no CUDA device).

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{CudaRuntime, as_kernel_op, harness_search_options};

#[test]
fn rounding_and_trunc_cast_plan_with_kernel_interfaces() {
    let mut cx = luminal::graph::Graph::new();
    let x = cx.tensor(6usize, DType::F32);
    let _floor = x.floor().output();
    let _ceil = x.ceil().output();
    let _trunc = x.trunc().output();
    let _round = x.round().output();
    let _ints = x.trunc_cast(DType::Int).output();

    let mut rt = CudaRuntime::load(&cx).expect("load");
    let data: FxHashMap<_, _> = [(x.id, vec![1.9f32, 1.5, 0.5, -0.5, -1.5, -1.9].into())]
        .into_iter()
        .collect();
    rt.search(&data, &harness_search_options())
        .expect("search under the CUDA allow list");

    let plan = rt.plan().expect("plan loaded");
    let labels: Vec<String> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => Some(op.label().to_string()),
            _ => None,
        })
        .collect();
    for expected in [
        "FloorFunctionalGeneric",
        "CeilFunctionalGeneric",
        "TruncFunctionalGeneric",
        "RoundFunctionalGeneric",
        "TruncCastGeneric",
    ] {
        assert!(
            labels.iter().any(|label| label == expected),
            "elected plan is missing {expected}: {labels:?}"
        );
    }
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            let label = op.label();
            if matches!(label, "BufferAlloc" | "BufferFree") {
                continue;
            }
            assert!(
                as_kernel_op(op.as_ref()).is_some(),
                "elected op {label} has no CUDA kernel interface"
            );
        }
    }
}
