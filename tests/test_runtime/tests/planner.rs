//! THE BUFFERIZER + DPS ESTATE that rode on `basic_program.egg`.
//!
//! `basic_program.egg` declares boundary layouts the reference runtime
//! cannot accept — it is a VIEW program, and the reference is now
//! canonical-layout-only, so its own corpus gate rejects the script by
//! construction. That is correct: the script's job was never to prove the
//! reference computes the right numbers. Every test here is about core's
//! machinery — `dps_rewrite`'s idempotency, the must-allocate boundary
//! coercion, poison folding — using a runtime merely as a vehicle.
//!
//! That makes them TestRuntime work by the same rule as the view and
//! mutation estates: the reference runtime's suite is for being simple
//! and correct (its 28 differential tests against candle), and everything
//! that exercises `Bufferizable` / `ToDps` / the planner lives here.

use luminal::test_support::bufferize_mock;

/// Idempotency: DPS forms answer to_dps() = None, so a second rewrite is a
/// no-op (same node and edge counts).
#[test]
fn dps_rewrite_is_idempotent() {
    let graph = test_runtime::extract_fixture_by_name("basic_program.egg");
    let once = luminal::dps::dps_rewrite(&graph);
    let twice = luminal::dps::dps_rewrite(&once);
    assert_eq!(once.dag.node_count(), twice.dag.node_count());
    assert_eq!(once.dag.edge_count(), twice.dag.edge_count());
}

/// MUST-ALLOCATE OUTPUTS — a pinned gap, not a feature. basic_program's
/// output buffers (7, 8, 10) appear only in BufferOutputLit, never in BufferInputLit;
/// under existence-is-BufferInputLit-membership that storage does NOT exist at
/// launch, so the plan itself is responsible for allocating and returning
/// it. Today the planner cannot express that obligation: every boundary
/// buffer is interned caller-owned, silently coercing "allocate and
/// return" into "the caller passes it in pre-allocated". This test pins
/// the coercion; when the Alloc-node / plan-manifest stage lands, these
/// buffers must become system-owned allocations that escape, and this
/// test must be flipped to assert exactly that.
#[test]
fn must_alloc_outputs_are_coerced_to_caller_provided_storage() {
    use std::collections::HashSet;

    use luminal::bufferize::{BufferId, BufferNode, Owner};

    let graph = test_runtime::extract_fixture_preferring(
        "basic_program.egg",
        &["LayoutTensorOpIndexMapApplyViewGeneric"],
    );
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    // Storage that exists at launch = the buffers backing BufferInput
    // nodes (the extractor admits those from BufferInputLit membership).
    let launch_existing: HashSet<BufferId> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferInput { slots } => Some(slots.iter()),
            _ => None,
        })
        .flatten()
        .map(|slot| slot.buffer.clone())
        .collect();

    // Every output destination that is not launch-existing storage is a
    // must-allocate output.
    let must_alloc: Vec<BufferId> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots.iter()),
            _ => None,
        })
        .flatten()
        .filter(|slot| !launch_existing.contains(&slot.buffer))
        .map(|slot| slot.buffer.clone())
        .collect();
    // TWO, not three — and the missing one is the point of this runtime.
    //
    // On the REFERENCE vocabulary this fixture has three must-allocate
    // outputs (z, w, left_view): that registry has no view op, so
    // `left_view` can only be reached by materializing a copy into fresh
    // storage. THIS runtime declares `IndexMapApplyView`, extraction
    // elects it, and under escape-and-disclose a view bound to an output
    // slot rides its parent's buffer instead of paying a boundary copy —
    // so `left_view` resolves to the caller's `left` input buffer and was
    // never storage the plan had to demand. z and w remain.
    //
    // The count is therefore a property of the VOCABULARY, not of the
    // fixture, which is precisely why this witness lives on the runtime
    // that owns the view op rather than beside the engine.
    assert_eq!(
        must_alloc.len(),
        2,
        "basic_program has two must-allocate outputs (z, w); left_view \
         escapes into its parent's buffer:\n{}",
        plan.summary()
    );

    for buffer in &must_alloc {
        let info = &plan.buffers[buffer];
        // THE COERCION: must-allocate storage is still pinned
        // caller-owned — the plan demands a handle the caller was never
        // supposed to provide, instead of allocating and returning it.
        assert!(
            matches!(buffer, BufferId::Boundary(_)),
            "must-allocate output still interned as a pinned boundary buffer"
        );
        assert!(
            matches!(info.owner, Owner::Caller),
            "must-allocate output still recorded caller-owned: {}",
            info.label
        );
    }
}

/// End to end through the pipeline: every DPS destination is admitted in
/// place into its poison's fresh storage — reads[dest] == writes[tied] for
/// every compute node, and Poison producers are folded (no compute node).
#[test]
fn dps_destinations_admitted_and_poisons_folded() {
    use luminal::bufferize::BufferNode;
    let graph = test_runtime::extract_fixture_by_name("basic_program.egg");
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");
    let mut computes = 0;
    for idx in plan.dag.node_indices() {
        if let BufferNode::Compute {
            op, reads, writes, ..
        } = &plan.dag[idx]
        {
            assert_ne!(op.label(), "Poison", "poison producers must be folded");
            if reads.is_empty() || writes.is_empty() {
                continue; // storage nodes (alloc/free): no DPS shape
            }
            computes += 1;
            // Trailing destination operands read the buffer they write.
            let dests = writes.len();
            let data = reads.len() - dests;
            for (j, write) in writes.iter().enumerate() {
                assert_eq!(&reads[data + j], write, "{}", plan.summary());
            }
        }
    }
    assert!(computes > 0);
}
