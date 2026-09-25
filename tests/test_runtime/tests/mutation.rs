//! THE MUTATING FAMILY's estate, rehomed from core's `src/test_support.rs`.
//!
//! Every op here declares `Must(0 -> 0)` with operand 0 READ — a tie that
//! writes storage it also reads. That is the declaration shape driving the
//! bufferizer's read-after-write conflict engine and its relocation
//! repair, and (for the alias-safe add) the tree's only `Sharing::May`
//! permit.
//!
//! Two of these tests assert purely FUNCTIONAL behaviour and named no
//! mutating op at all. They still had to move: their fixture
//! (`boundary_in_place_mutation.egg`) carries a `(check ...)` naming
//! `LayoutTensorOpSqrtMutatingGeneric`, and a constructor no loaded
//! matcher declares is an egglog PARSE failure — it takes the whole
//! script down before a single rule runs. The fixture moved, so they
//! moved with it.
//!
//! Every fixture read here is this crate's own, under `fixtures/`.
//! Several are forked copies of scripts the reference corpus also
//! carries — the two runtimes share no files.

use luminal::prelude::petgraph;
use test_runtime::test_support::bufferize_mock;

/// THE MUTATING TIER, end to end: extraction restricted to
/// SqrtMutatingGeneric proves the one-buffer kernel through analysis and
/// planning — the tie is admitted (the op's own read is the same-use
/// exemption), the result rides x's caller buffer, and the boundary
/// passes through. Zero copies, zero allocations, no permits involved.
#[test]
fn mutating_sqrt_bufferizes_zero_copy_in_place() {
    use luminal::bufferize::{BufferId, BufferNode};

    let graph = test_runtime::extract_fixture_with_ops(
        "boundary_in_place_mutation.egg",
        &["LayoutTensorOpSqrtMutatingGeneric"],
    );
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    assert!(
        plan.buffers
            .keys()
            .all(|id| matches!(id, BufferId::Boundary(_))),
        "zero allocations:\n{}",
        plan.summary()
    );
    let mut computes = 0;
    for idx in plan.dag.node_indices() {
        match &plan.dag[idx] {
            BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
            BufferNode::Compute {
                op, reads, writes, ..
            } => {
                computes += 1;
                assert_eq!(op.label(), "SqrtMutatingGeneric");
                assert_eq!(reads.len(), 1, "one buffer in:\n{}", plan.summary());
                assert_eq!(reads[0], writes[0], "mutated in place:\n{}", plan.summary());
                assert!(matches!(&writes[0], BufferId::Boundary(_)));
            }
            _ => {}
        }
    }
    assert_eq!(computes, 1);
}

/// THE MAY-SHARE PERMIT, end to end: z = x + x back into x's buffer,
/// extraction restricted to the input-alias-safe mutating add. The rhs
/// read aliases the mutated storage; the permit (whose all-layouts-equal
/// precondition the egglog match discharged) excuses it, and the whole
/// accumulation is zero-copy in the caller's buffer.
#[test]
fn alias_safe_add_accumulates_x_plus_x_in_place() {
    use luminal::bufferize::{BufferId, BufferNode};

    let graph = test_runtime::extract_fixture_with_ops(
        "boundary_alias_safe_add.egg",
        &["LayoutTensorOpAddMutatingInputAliasSafeGeneric"],
    );
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    assert!(
        plan.buffers
            .keys()
            .all(|id| matches!(id, BufferId::Boundary(_))),
        "zero allocations:\n{}",
        plan.summary()
    );
    let mut computes = 0;
    for idx in plan.dag.node_indices() {
        match &plan.dag[idx] {
            BufferNode::BufferCopy { .. } => panic!("zero copies:\n{}", plan.summary()),
            BufferNode::Compute {
                op, reads, writes, ..
            } => {
                computes += 1;
                assert_eq!(op.label(), "AddMutatingInputAliasSafeGeneric");
                assert_eq!(reads.len(), 2, "{}", plan.summary());
                assert_eq!(reads[0], writes[0], "lhs mutated in place");
                assert_eq!(
                    reads[1], writes[0],
                    "rhs reads the SAME storage (permitted)"
                );
            }
            _ => {}
        }
    }
    assert_eq!(computes, 1);
}

/// THE CONTRAST: the same x + x program through the PLAIN mutating add,
/// which declares no permit — absence is restrict semantics. The rhs
/// aliasing the mutated storage rejects the tie; the generic repair
/// relocates (copy-in to a fresh buffer, mutate there) and the boundary
/// copies back. Sound, two copies, one allocation — the price of the
/// missing permit.
#[test]
fn plain_mutating_add_on_x_plus_x_degrades_to_copies() {
    use luminal::bufferize::{BufferId, BufferNode};

    let graph = test_runtime::extract_fixture_with_ops(
        "boundary_alias_safe_add.egg",
        &["LayoutTensorOpAddMutatingGeneric"],
    );
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    let allocs = plan
        .buffers
        .keys()
        .filter(|id| matches!(id, BufferId::Allocated(_)))
        .count();
    assert_eq!(allocs, 1, "one relocation:\n{}", plan.summary());
    let copies = plan
        .dag
        .node_indices()
        .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
        .count();
    assert_eq!(copies, 2, "copy-in + boundary copy:\n{}", plan.summary());
}

/// The allow-list is a hard filter: a program not implementable within
/// it fails extraction loudly, never silently substitutes.
#[test]
fn op_filter_excluding_every_implementation_fails_loudly() {
    let err = test_runtime::try_extract_fixture_with_ops(
        "boundary_in_place_mutation.egg",
        &["LayoutTensorOpExpMutatingGeneric"], // no Exp in this program
    )
    .expect_err("no implementation for sqrt is allowed");
    assert!(err.to_string().contains("failed to extract"), "{err}");
}

/// Boundary in-place mutation through the FUNCTIONAL op, after the
/// engine stopped checking layouts: the same-op read of x against the
/// seeded destination has no unconditional permit, so the seed is
/// rejected and the plan degrades soundly — fresh allocation, boundary
/// copy back into the caller's buffer. No Anti edge is needed: the copy
/// is dataflow-ordered after the buffer's only reader (its source IS the
/// sqrt's output). (The zero-copy lowering for this program is the
/// MutatingGeneric op, which extraction does not yet prefer — the
/// recorded extraction-preference decision.)
#[test]
fn boundary_mutation_via_functional_degrades_to_copy() {
    use luminal::bufferize::{BufferId, BufferNode, EdgeKind};
    use petgraph::visit::EdgeRef;

    let graph = test_runtime::extract_fixture_by_name("boundary_in_place_mutation.egg");
    let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

    let allocs = plan
        .buffers
        .keys()
        .filter(|id| matches!(id, BufferId::Allocated(_)))
        .count();
    assert_eq!(allocs, 1, "one relocated destination:\n{}", plan.summary());
    let copies = plan
        .dag
        .node_indices()
        .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
        .count();
    assert_eq!(copies, 1, "one boundary copy:\n{}", plan.summary());
    // No WAR anti is needed (the copy is dataflow-ordered after the
    // read); the one anti is lifetime — the boundary copy's src-read
    // before the fresh buffer's free.
    let anti: Vec<_> = plan
        .dag
        .edge_references()
        .filter(|edge| edge.weight().kind == EdgeKind::Anti)
        .collect();
    assert_eq!(anti.len(), 1, "one lifetime anti only:\n{}", plan.summary());
    assert!(
        matches!(&plan.dag[anti[0].target()], BufferNode::Compute { writes, .. }
            if writes.is_empty()),
        "the sole anti targets the free:\n{}",
        plan.summary()
    );
}

/// The DPS rewrite: every capable op gains one poison destination per
/// result (trailing operands), produced by synthesized Poison nodes whose
/// values carry the tied result's LAYOUT (the equivalence gate keys on it).
#[test]
fn dps_rewrite_appends_tied_poison_destinations() {
    use luminal::layout_ir::ExtractedNode;
    let graph = test_runtime::extract_fixture_by_name("boundary_in_place_mutation.egg");
    let rewritten = luminal::dps::dps_rewrite(&graph);

    // The DPS form keeps the base op's label (label policy: IR names are
    // never edited), so DPS-ness is witnessed semantically: among
    // extractable ops only DPS forms answer to_dps() = None.
    let (op, result_layout) = rewritten
        .dag
        .node_weights()
        .find_map(|node| match node {
            ExtractedNode::LayoutOp(op)
                if op.op.label() == "SqrtFunctionalGeneric" && op.op.to_dps().is_none() =>
            {
                Some((op.clone(), op.outputs[0].layout.eclass.clone()))
            }
            _ => None,
        })
        .expect("Sqrt was rewritten to its DPS form");
    assert_eq!(op.inputs.len(), 2, "input + one destination");
    assert!(op.inputs[1].port.starts_with("dest"));
    // The poison producer exists and its value carries the result's layout.
    let poison = rewritten
        .dag
        .node_weights()
        .find_map(|node| match node {
            ExtractedNode::LayoutOp(p) if p.op.label() == "Poison" => Some(p.clone()),
            _ => None,
        })
        .expect("Poison producer synthesized");
    assert_eq!(poison.outputs[0].eclass, op.inputs[1].value);
    assert_eq!(poison.outputs[0].layout.eclass, result_layout);
}

/// The in-place scatter path (R8 dual — the user's in-place ruling):
/// forced to the MUTATING implementation, the KV-cache update writes
/// straight into the caller's cache buffer — no result allocation, no
/// BufferCopy, and the operand slots read init/src/coord0/coord1.
/// (Full extraction currently prefers the functional form on a cost
/// tie and REPAIRS the in-place demand with a copy — the golden pins
/// that honest outcome; extraction preference is a deferred lever.)
#[test]
fn scatter_mutating_updates_the_cache_in_place() {
    use luminal::layout_ir::ExtractedNode;
    let graph = test_runtime::extract_fixture_with_ops(
        "boundary_scatter.egg",
        &[
            "LayoutTensorOpScatterMutatingGeneric",
            "LayoutTensorOpIotaGeneric",
        ],
    );
    let scatter = graph
        .dag
        .node_weights()
        .find_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label() == "ScatterMutatingGeneric" => {
                Some(op.clone())
            }
            _ => None,
        })
        .expect("mutating scatter extracted");
    assert_eq!(
        scatter.inputs.len(),
        4,
        "init + src + one coord per init axis"
    );
    assert_eq!(scatter.inputs[0].port, "init");
    assert_eq!(scatter.inputs[1].port, "src");
    assert_eq!(scatter.inputs[2].port, "coord0");
    assert_eq!(scatter.inputs[3].port, "coord1");

    let plan =
        bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("in-place scatter bufferizes");
    let summary = plan.summary();
    assert!(summary.contains("ScatterMutatingGeneric"), "{summary}");
    assert!(
        !summary.contains("BufferCopy"),
        "in-place must be zero-copy: {summary}"
    );
}

/// THE ONLY `Sharing::May` DECLARER in the tree, and the reason the
/// permit machinery in `bufferize` is reachable at all. Split off from
/// `luminal_reference::ops`'s registry-contract test when the op moved:
/// that crate keeps the negative claim (nothing IT registers declares a
/// permit), and this is the positive one. The permit is DIRECTIONAL —
/// rhs may share the mutated storage; the reverse is not a permit.
#[test]
fn permit_is_declared_and_directional() {
    use luminal::layout_ir::permits_sharing;
    use test_runtime::ops::mutating::AddMutatingInputAliasSafe;

    assert!(permits_sharing(&AddMutatingInputAliasSafe, 1, 0));
    assert!(!permits_sharing(&AddMutatingInputAliasSafe, 0, 1));
}
