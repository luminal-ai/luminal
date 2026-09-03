//! THE MARKER 2-CYCLE, at the HARNESS budget (sampler ruling 2026-09-02).
//!
//! The cuBLASLt double-transpose collapse (`(union ?w ?x)` in
//! `cublaslt_marker_canonicalize.egg`) makes two DIFFERENT logical
//! values re-describe each other: class X gains
//! `LogicalIndexMapApply(V, transpose)` while class V holds
//! `LogicalIndexMapApply(X, transpose)`. The 2026-08-07 sampler grouped
//! re-description candidates by SHARED LOGICAL VALUE, so it saw both
//! spellings as "progressing", sampled both, and the extractor refused
//! the genome as a choice cycle — at the 2x4 harness budget every
//! genome could be eaten that way and the search died with "no
//! candidate genome produced an executable plan".
//!
//! Components of the candidate graph do not care whose logical value a
//! class instantiates, so the 2-cycle is a component and the forest
//! sampler cannot elect both arms. ACCEPTANCE: the search completes at
//! the harness budget, and its choice-cycle refusal count is ZERO.
//! Election at 2x4 is REPORTED, not asserted (the 12x16 pin in
//! `cublaslt_election.rs` owns that claim).

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

#[test]
fn canonical_2d_matmul_searches_green_at_the_harness_budget() {
    let mut cx = Graph::new();
    let a = cx.tensor((4usize, 8usize), DType::F32);
    let b = cx.tensor((8usize, 3usize), DType::F32);
    let _out = a.matmul(b).output();

    let mut rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
    let data: FxHashMap<NodeIndex, TypedBuffer> =
        [(a.id, weights(32, 1).into()), (b.id, weights(24, 2).into())]
            .into_iter()
            .collect();

    let options = luminal::test_support::harness_search_options();
    let outcome = rt
        .search(&data, &options)
        .unwrap_or_else(|e| panic!("the marker 2-cycle must not exhaust the 2x4 budget: {e:#}"));

    let breakdown = &outcome.refusal_breakdown;
    let plan = rt.plan().expect("plan");
    let mut elected = 0usize;
    let mut computes = 0usize;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            let label = op.label();
            if label == "BufferAlloc" || label == "BufferFree" {
                continue;
            }
            computes += 1;
            if label.starts_with("CublasLt") {
                elected += 1;
            }
        }
    }
    println!(
        "SCC-SAMPLER 2x4 matmul_2d(4x8 . 8x3): cublaslt_elected={elected} computes={computes} \
         plans={} refusals=[{}]",
        outcome.plans_profiled,
        breakdown.summary()
    );
    for exemplar in &breakdown.exemplars {
        println!("SCC-SAMPLER 2x4 exemplar: {exemplar}");
    }
    assert_eq!(
        breakdown.with_choice_cycles,
        0,
        "the forest sampler cannot elect both arms of the collapse's 2-cycle: {}",
        breakdown.summary()
    );
    assert!(outcome.plans_profiled > 0, "a plan must be profiled");
}
