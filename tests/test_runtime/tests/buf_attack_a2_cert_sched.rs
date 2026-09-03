//! A2 ATTACK PROBES (certificates + scheduling lens, P1 review 2026-08-26).
//! Observational, against the CURRENT planner (no P1 code exists):
//!   1. single transpose view to bound output — the alloc+copy+free baseline
//!      P1 claims to eliminate;
//!   2. transpose-roundtrip (two stacked views composing to identity) — does
//!      the e-graph/extraction hand the planner ONE composed view, TWO
//!      stacked views, or zero (welded back to the parent class)? This
//!      decides whether P1's per-hop extent oracle ever sees multi-hop
//!      chains in practice;
//!   3. fan-out: the matmul value reaches a bound output through a view AND
//!      feeds a mid-graph elementwise consumer — pins the copy/anti-edge
//!      shape whose donated counterpart the admission argument must order;
//!   4. the same matmul value bound to TWO output slots (one direct, one
//!      through a view) — pins today's seed application (direct slot seeds,
//!      view slot delivery-copies) and the seen_poisons dedup P1 rides.
//!
//! ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27, supersedes the 5b typed
//! refusal AND the alloc+copy+free baseline these probes originally
//! measured): a folded view bound to an output now ESCAPES — the kernel's
//! alloc is handed to the caller (FreedBy::Caller, no free), zero boundary
//! copies, and the slot's binding discloses the elected layout. P1's
//! claimed elimination arrived as escape semantics.
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;

const GENOME: &[&str] = &[
    "LayoutTensorOpCublasLt",
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

fn report(name: &str, text: &str) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, GENOME);
    let dps = luminal::dps::dps_rewrite(&graph);
    let mut views = 0usize;
    for node in dps.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            let outs: Vec<String> = op
                .outputs
                .iter()
                .map(|o| format!("{}:dims={:?}", o.eclass, o.dims))
                .collect();
            if op.op.label().contains("IndexMapApplyView") {
                views += 1;
            }
            println!("[{name}] DPS {}: ins={ins:?} outs={outs:?}", op.op.label());
        }
    }
    // ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27): each fixture's
    // folded transpose VIEW output ESCAPES. Pin the improvement over the
    // recorded alloc+copy+free baseline: ZERO copies, no free of the
    // escaping backing buffer, the view slot backed by minted storage
    // handed to the caller, and the layout disclosed on the binding.
    let plan = luminal::test_support::bufferize_mock(&dps).expect("the view output escapes");
    let table = luminal::test_support::mock_layout_table(&dps);
    let summary = plan.summary();
    println!("[{name}] plan:\n{summary}");
    use luminal::bufferize::{BufferId, BufferNode};
    assert!(
        !plan
            .dag
            .node_indices()
            .any(|i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. })),
        "[{name}] zero copies under escape:\n{summary}"
    );
    let slot = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
            _ => None,
        })
        .expect("slot 0 (the view output)");
    assert!(
        matches!(slot.buffer, BufferId::Allocated(_)),
        "[{name}] the view slot is backed by the kernel's escaping alloc:\n{summary}"
    );
    assert_eq!(
        plan.buffers[&slot.buffer].freed_by,
        luminal::layout_ir::FreedBy::Caller,
        "[{name}] the backing buffer escapes to the caller:\n{summary}"
    );
    assert!(
        !plan.dag.node_indices().any(|i| matches!(
            &plan.dag[i],
            BufferNode::Compute { op, reads, .. }
                if op.label() == "BufferFree" && reads.contains(&slot.buffer)
        )),
        "[{name}] no free for the escaping buffer:\n{summary}"
    );
    // OPTION B: the layout is TOTAL on bindings; what is pinned is that
    // the binding carries the SLOT VALUE's own decoded layout, not the
    // backing buffer's resident layout.
    assert_eq!(
        Some(&slot.layout),
        table.get(&slot.value),
        "[{name}] the binding discloses the slot value's own elected layout:\n{summary}"
    );
    println!("[{name}] view-nodes(dps)={views}");
}

/// Baseline P1 target: matmul -> transpose view -> bound output.
#[test]
fn a2_single_view_to_bound() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w).transpose(0, 1).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.t", &text);
}

/// Two stacked transposes composing to identity: does the planner ever SEE
/// a multi-hop view chain, or does composition normalize it away?
#[test]
fn a2_double_view_roundtrip() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w).transpose(0, 1).transpose(0, 1).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.t.t", &text);
}

/// Fan-out: the matmul value goes to a bound output through a view AND to a
/// mid-graph elementwise consumer whose result is a second output.
#[test]
fn a2_view_fanout() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let y = x.matmul(w);
        let _ = y.transpose(0, 1).output();
        let _ = (y * c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.fanout", &text);
}

/// Same matmul value bound to two slots: direct + through a transpose view.
/// FINDING (2026-08-26): under the shared-handle-id recorder both outputs
/// minted ONE `natout` stem with conflicting shapes and egglog rejected
/// the shadowing before the planner ran — pinned then as should_panic
/// ("the advocate's case (c) is cold via this frontend spelling").
/// RE-FOUND (2026-09-01, PR #423): `GraphTensor.id` is the canonical SSA
/// identity, so the transpose view and the direct binding are DIFFERENT
/// nodes with distinct stems — the configuration is now reachable and
/// well-formed. The pin flips: it must build cleanly and plan two
/// distinct output slots (the advocate's case (c) is live; this board
/// now covers it for real).
#[test]
fn a2_two_slots_same_value() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let y = x.matmul(w);
        let _ = y.transpose(0, 1).output();
        let _ = y.output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    report("mm.2slots", &text);
}
