//! ROUND-10 FOLLOW-UP PROBE (Austin 2026-08-26): does the sibling-view copy
//! occur when other operations FOLLOW the matmul, or only at a directly
//! bound output?
//!
//! Two fixtures, both bufferized and printed:
//!   1. matmul -> elementwise (add) -> output: the matmul's result is
//!      consumed mid-graph, not directly bound.
//!   2. matmul -> matmul -> output (the chained case): the interior view
//!      feeds the second call's operand.
//!
//! Observational: prints the DPS ops and the buffer plan; asserts only
//! that bufferize succeeds and counts BufferCopy occurrences honestly.
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;

fn bufferize_and_report(name: &str, text: &str, prefer: &[&str]) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, prefer);
    let dps = luminal::dps::dps_rewrite(&graph);
    for node in dps.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            println!("[{name}] DPS {}: ins={ins:?}", op.op.label());
        }
    }
    let plan = luminal::test_support::bufferize_mock(&dps).expect("bufferize");
    let summary = plan.summary();
    let copies = summary.matches("BufferCopy").count();
    let allocs = summary.matches("BufferAlloc").count();
    println!("[{name}] plan:\n{summary}");
    println!("[{name}] BufferAlloc count = {allocs}, BufferCopy count = {copies}");
}

#[test]
fn matmul_then_elementwise_bufferize() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        // Mul, not Add: Add is the C-fold decorator's own pattern, which
        // would fuse into the call instead of standing downstream.
        let y = x.matmul(w) * c;
        let _ = y.output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    bufferize_and_report(
        "mm+add",
        &text,
        &[
            "LayoutTensorOpCublasLt",
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
}

#[test]
fn chained_matmuls_bufferize() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w1 = cx.tensor((8usize, 3usize), DType::F32);
        let w2 = cx.tensor((3usize, 5usize), DType::F32);
        // The original boundary-flowing spelling (restored under
        // escape-and-disclose, ruling 2026-08-27: a view-produced bound
        // output escapes, so no dodge is needed). The probe's subject is
        // the INTERIOR sibling view feeding the second call.
        let _ = x.matmul(w1).matmul(w2).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    bufferize_and_report(
        "mm.mm",
        &text,
        &[
            "LayoutTensorOpCublasLt",
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
}
