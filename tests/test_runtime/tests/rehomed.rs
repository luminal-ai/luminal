//! The rehomed fused-op estate (ruling 2026-08-13): the reference runtime
//! implements only plain spellings, so AddMulFused and its test estate —
//! golden-plan pin, plan-fingerprint identity, the genome multi-output
//! invariants — were deleted from the main tree at aff22598 and recreated
//! here on the TestRuntime's vocabulary and fixture runner. Test bodies
//! are adapted from the pre-deletion `src/test_support.rs`.

use std::fs;

use luminal::bufferize::BufferId;
use luminal::layout_ir::ExtractedNode;

/// The restored multi-output fixture script (deleted from
/// `src/egglog_core/test_scripts/` at aff22598).
const ADD_MUL_FUSED: &str = include_str!("../fixtures/add_mul_fused.egg");

/// GENOME WALK, consistent-fused: both boundary classes choose the SAME
/// fused enode; instance dedup (keyed by concrete enode) yields ONE
/// kernel claiming both output slots into the caller's buffers.
#[test]
fn genome_fused_choice_dedups_one_instance_claiming_both_slots() {
    let (graph, _) = test_runtime::extract_fixture_with_genome(
        ADD_MUL_FUSED,
        &["LayoutTensorOpAddMulFusedGeneric"],
    );
    let computes = graph
        .dag
        .node_weights()
        .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
        .count();
    assert_eq!(computes, 1, "instance dedup across both claimed slots");
    let plan = luminal::test_support::bufferize_mock(&luminal::dps::dps_rewrite(&graph))
        .expect("bufferizes");
    let allocs = plan
        .buffers
        .keys()
        .filter(|id| matches!(id, BufferId::Allocated(_)))
        .count();
    assert_eq!(
        allocs,
        0,
        "both slots land in caller buffers:\n{}",
        plan.summary()
    );
}

/// GENOME WALK, the MIXED plan (the paper-walk's scenario 3, now
/// executable): the add value chooses the fused kernel, the mul value
/// chooses the standalone Mul. The fused instance's unclaimed mul slot
/// writes a fresh WASTE destination — allocated, written, freed unread —
/// instead of double-writing the caller's mul buffer (which the
/// two-writers tripwire would reject). Wasteful, correct, priced by
/// profiling: the single-mutation bridge between the pair and fused
/// plans.
#[test]
fn genome_mixed_choice_mints_a_waste_destination() {
    let (graph, _) = test_runtime::extract_fixture_with_genome(
        ADD_MUL_FUSED,
        &[
            "LayoutTensorOpMulFunctionalGeneric",
            "LayoutTensorOpAddMulFusedGeneric",
        ],
    );
    let computes = graph
        .dag
        .node_weights()
        .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
        .count();
    assert_eq!(computes, 2, "fused kernel + standalone mul");
    let plan = luminal::test_support::bufferize_mock(&luminal::dps::dps_rewrite(&graph))
        .expect("bufferizes");
    let summary = plan.summary();
    assert!(summary.contains("AddMulFusedGeneric"), "{summary}");
    assert!(summary.contains("MulFunctionalGeneric"), "{summary}");
    let allocs = plan
        .buffers
        .keys()
        .filter(|id| matches!(id, BufferId::Allocated(_)))
        .count();
    assert_eq!(
        allocs, 1,
        "the unclaimed fused slot writes scratch:\n{summary}"
    );
}

/// Many genomes, one plan: the fingerprint identifies the PLAN, so the
/// search can skip re-profiling genomes that build what it already
/// timed (plan-hash dedup ruling, 2026-07-27).
#[test]
fn genome_plan_fingerprints_identify_plans() {
    let (_, fused_a) = test_runtime::extract_fixture_with_genome(
        ADD_MUL_FUSED,
        &["LayoutTensorOpAddMulFusedGeneric"],
    );
    let (_, fused_b) = test_runtime::extract_fixture_with_genome(
        ADD_MUL_FUSED,
        &["LayoutTensorOpAddMulFusedGeneric"],
    );
    let (_, mixed) = test_runtime::extract_fixture_with_genome(
        ADD_MUL_FUSED,
        &[
            "LayoutTensorOpMulFunctionalGeneric",
            "LayoutTensorOpAddMulFusedGeneric",
        ],
    );
    assert_eq!(fused_a, fused_b, "same genome, same plan, same fingerprint");
    assert_ne!(fused_a, mixed, "different plans, different fingerprints");
}

/// RANK 3: the golden is an ENFORCED pin, not documentation. Only the
/// add_mul_fused arm is rehomed — the other stems the old
/// `golden_plans_are_pinned` walked still live in the main tree and stay
/// covered by the main-tree suite. The golden was RE-BLESSED at rehoming
/// (2026-08-13): the registry differs from the pre-deletion tree (no
/// MatMulFused, view re-registered), so the pinned bytes are the freshly
/// observed plan under THIS runtime's vocabulary. Review verdict: the
/// fresh plan came out byte-identical to the pre-deletion pin
/// (`aff22598~1:output/add_mul_fused.bufferized.txt`) — the deterministic
/// extractor still prefers the plain Add + Mul pair over the fused
/// kernel on this fixture.
#[test]
fn golden_plans_are_pinned() {
    let golden_path = "golden/add_mul_fused.bufferized.txt";
    let graph = test_runtime::extract_fixture(ADD_MUL_FUSED);
    let plan = luminal::test_support::bufferize_mock(&luminal::dps::dps_rewrite(&graph))
        .expect("add_mul_fused");
    let golden = fs::read_to_string(golden_path)
        .unwrap_or_else(|_| panic!("{golden_path} missing — run regenerate_golden_plans by name"));
    assert_eq!(
        plan.summary(),
        golden,
        "plan for add_mul_fused diverged from golden"
    );
}

/// Regenerate golden/add_mul_fused.bufferized.txt — run explicitly by name:
/// `cargo test --release -p test_runtime regenerate_golden_plans -- --ignored`
/// (invocation-by-name IS the programmatic opt-in; regenerated diffs are
/// REVIEWED, never rubber-stamped).
#[test]
#[ignore = "golden regenerator — run explicitly by name"]
fn regenerate_golden_plans() {
    let graph = test_runtime::extract_fixture(ADD_MUL_FUSED);
    let plan = luminal::test_support::bufferize_mock(&luminal::dps::dps_rewrite(&graph))
        .expect("add_mul_fused");
    fs::write("golden/add_mul_fused.bufferized.txt", plan.summary()).expect("golden writes");
}
