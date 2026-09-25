//! P1 ATTACKER A1 (aliasing + caller-contract lens) — Austin 2026-08-26 review.
//!
//! P1 proposes seeding an op's DPS destination into a BOUND output buffer
//! reached through zero-movement views. These probes run against the CURRENT
//! planner (P1 unimplemented) and pin, with executed evidence, the exact
//! machinery P1 inherits for the four A1 attacks:
//!   (1) bound output buffer cohabits an op input (same BufferLit e-class):
//!       the seed is refused absent a trusted May permit, and ADMITTED with
//!       one — the permit is the ONLY gate on an in-kernel read+write of one
//!       bound buffer (the P-E exposure, live today without any view).
//!   (2) one value to two bound outputs, with a view in the mix: the direct
//!       slot seeds, the view slot is served by a bound->bound copy — today,
//!       with zero Allocated buffers.
//!   (3) an unordered reader of the cohabiting buffer: the materialize path
//!       TOLERATES it via Anti(reader -> copy); P1's admission would refuse
//!       and must degrade to exactly this certified plan.
//!   (4) a mutating consumer reaching a bound value THROUGH a view: vetoed
//!       (END_OF_PROGRAM read never happens-before) and repaired.
//!
//! Hand-authored graphs via test_runtime::test_support, per the assignment rule:
//! these shapes are defined by the Bufferizable interface (Must ties, May
//! permits, poison roots), which have no egglog surface.
use luminal::bufferize::{BufferId, BufferNode, EdgeKind};
use luminal::layout_ir::Access;
use luminal::prelude::petgraph;
use petgraph::visit::EdgeRef;
use test_runtime::test_support::{EmptyOp, MockOp, MockView, TestGraph};

fn copies(
    plan: &luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
) -> Vec<(BufferId, BufferId)> {
    plan.dag
        .node_weights()
        .filter_map(|n| match n {
            BufferNode::BufferCopy { src, dst, .. } => Some((src.clone(), dst.clone())),
            _ => None,
        })
        .collect()
}

fn war_antis(
    plan: &luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
) -> usize {
    plan.dag
        .edge_references()
        .filter(|e| e.weight().kind == EdgeKind::Anti)
        .filter(|e| {
            // exclude lifetime antis (targets that are frees: write-nothing computes)
            !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                if writes.is_empty())
        })
        .count()
}

/// ATTACK 1, arm A (refusal): the bound output buffer B cohabits an op input
/// x (same e-class). The op reads x through operand 0 and its DPS dest is
/// operand 1. The seed pins the poison into B, the pre-union makes x a
/// cohabitant, and veto (2) hits the same-op read with NO May permit ->
/// seed refused, plan degrades to scratch + delivery copy. This is the exact
/// refusal P1's view-hop seeds inherit unchanged.
#[test]
fn a1_cohabiting_input_refuses_seed_without_permit() {
    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: false,
            }),
            &[&x, &p],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "B");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    assert!(
        matches!(plan.value_buffer[&r], BufferId::Allocated(_)),
        "seed into a cohabited buffer must be refused without a permit:\n{}",
        plan.summary()
    );
    let cps = copies(&plan);
    assert_eq!(cps.len(), 1, "one delivery copy:\n{}", plan.summary());
    assert_eq!(
        cps[0].1, plan.value_buffer[&x],
        "copy lands in the shared bound buffer"
    );
}

/// ATTACK 1, arm B (the P-E exposure, executed): same program, op now grants
/// the unconditional trusted May permit (cuBLASLt's C-may-alias-D shape).
/// The seed is ADMITTED: the compute reads x from B and writes B in the same
/// kernel, zero copies. The planner performs no layout/extent checking here —
/// the permit alone licenses the plan. Under P1 the dest arrives through a
/// view (parent-layout descriptors vs slot-layout binding), so the permit's
/// egglog-side discharge no longer describes the actual storage relation:
/// this measured behavior is why P-E (quarantine permit-bearing chain roots)
/// is a REQUIRED precondition, not optional hygiene.
#[test]
fn a1_cohabiting_input_admitted_with_trusted_permit_writes_bound_in_kernel() {
    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: true,
            }),
            &[&x, &p],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "B");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    assert!(
        matches!(plan.value_buffer[&r], BufferId::Boundary(_)),
        "the permit admits the seed:\n{}",
        plan.summary()
    );
    assert_eq!(copies(&plan), vec![], "zero copies");
    // The one compute both reads and writes the SAME bound buffer.
    let rw = plan.dag.node_weights().any(|n| {
        matches!(n, BufferNode::Compute { reads, writes, .. }
        if !writes.is_empty() && !reads.is_empty() && reads[0] == writes[0]
            && matches!(writes[0], BufferId::Boundary(_)))
    });
    assert!(
        rw,
        "in-kernel read+write of the bound buffer:\n{}",
        plan.summary()
    );
}

/// ATTACK 1, arm C (laundering blocked): the cohabiting input is ReadOnly.
/// Even WITH the permit, the writability veto kills the seed and the
/// delivery copy into the ReadOnly buffer is a hard error — the program is
/// rejected loudly. P4.9 (input declarations intern first) holds under seeds.
#[test]
fn a1_readonly_cohabitant_rejects_program_even_with_permit() {
    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadOnly, "rm");
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: true,
            }),
            &[&x, &p],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "B");
    let err = test_runtime::test_support::bufferize_mock(&g.build()).unwrap_err();
    println!("rejected: {err:#}");
    assert!(err.to_string().contains("read-only buffer"), "{err}");
}

/// ATTACK 2, arm A: value y bound to slot 0 (buffer D) AND its view v bound
/// to slot 1 (buffer E). The direct slot's chain seeds (no view hop in it)
/// and y computes straight into D; ESCAPE-AND-DISCLOSE (ruling 2026-08-27,
/// supersedes the 5b refusal and the older bound->bound copy) then returns
/// the view slot ZERO-COPY: v's residence IS caller storage D, so slot E is
/// backed by D too — the declared E buffer goes unused — and the binding
/// discloses the view layout. Zero copies, zero Allocated buffers.
#[test]
fn a1_value_plus_view_to_two_outputs_direct_slot_first() {
    let mut g = TestGraph::new();
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    g.output(&y, "D");
    g.output(&v, "E");
    let graph = g.build();
    let table = test_runtime::test_support::mock_layout_table(&graph);
    let plan = test_runtime::test_support::bufferize_mock(&graph).expect("bufferize");
    println!("{}", plan.summary());

    assert!(
        matches!(plan.value_buffer[&y], BufferId::Boundary(_)),
        "y seeded into D"
    );
    assert!(
        plan.buffers
            .keys()
            .all(|b| matches!(b, BufferId::Boundary(_))),
        "zero Allocated buffers:\n{}",
        plan.summary()
    );
    assert_eq!(
        copies(&plan).len(),
        0,
        "zero copies — both slots ride D:\n{}",
        plan.summary()
    );
    let slots: Vec<_> = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots.clone()),
            _ => None,
        })
        .expect("the output node");
    assert_eq!(
        slots[0].buffer, plan.value_buffer[&y],
        "the direct slot is D"
    );
    assert_eq!(
        slots[1].buffer, plan.value_buffer[&y],
        "the view slot is backed by D too"
    );
    // OPTION B: the layout is TOTAL on every binding — "disclosed" is no
    // longer a presence question. The distinction is WHICH layout: the
    // dense slot carries y's own, the view slot carries the view's, and
    // the two are different functions over the SAME buffer.
    assert_eq!(&slots[0].layout, &table[&y], "dense slot: y's own layout");
    assert_eq!(
        &slots[1].layout, &table[&v],
        "view slot: the view's own layout"
    );
    assert_ne!(
        slots[0].layout, slots[1].layout,
        "one buffer, two deliveries"
    );
}

/// ATTACK 2, arm B: same program, slot order flipped (view slot is slot 0).
/// The view slot's seed walk stops at the view, the direct slot (E) still
/// seeds — y computes into E — and the view slot returns ZERO-COPY backed
/// by E (its residence). Slot order does not reopen the
/// one-poison-two-proposals door (seen_poisons dedup), and it changes only
/// WHICH caller buffer carries the bytes — never the zero-copy outcome.
#[test]
fn a1_value_plus_view_to_two_outputs_view_slot_first() {
    let mut g = TestGraph::new();
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    g.output(&v, "D");
    g.output(&y, "E");
    let graph = g.build();
    let table = test_runtime::test_support::mock_layout_table(&graph);
    let plan = test_runtime::test_support::bufferize_mock(&graph).expect("bufferize");
    println!("{}", plan.summary());

    assert!(
        matches!(plan.value_buffer[&y], BufferId::Boundary(_)),
        "y seeded into E"
    );
    assert!(
        plan.buffers
            .keys()
            .all(|b| matches!(b, BufferId::Boundary(_))),
        "zero Allocated buffers:\n{}",
        plan.summary()
    );
    assert_eq!(copies(&plan).len(), 0, "zero copies:\n{}", plan.summary());
    let slots: Vec<_> = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots.clone()),
            _ => None,
        })
        .expect("the output node");
    assert_eq!(
        slots[0].buffer, plan.value_buffer[&y],
        "the view slot rides y's residence"
    );
    assert_eq!(
        &slots[0].layout, &table[&v],
        "…carrying the VIEW's own layout"
    );
    assert_eq!(
        slots[1].buffer, plan.value_buffer[&y],
        "the direct slot is the residence"
    );
}

/// ATTACK 3: an unordered reader of the cohabiting bound buffer. The chain's
/// value reaches slot D through a VIEW (so today no seed is even proposed).
/// Under escape-and-disclose the chain's minted residence escapes and backs
/// slot D directly — D's declared buffer is never written, so the hazard
/// this probe attacked (a delivery overwriting D while the unordered reader
/// still needs x's bytes) is GONE, not tolerated: zero copies into D, zero
/// WAR antis.
#[test]
fn a1_unordered_reader_of_cohabited_buffer_hazard_gone_under_escape() {
    let mut g = TestGraph::new();
    let x = g.input("x", "D", Access::ReadWrite, "rm");
    // unordered reader of x, output elsewhere: keeps D's old bytes live
    let s = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&x],
            &[("s", "rm")],
        )
        .remove(0);
    // independent chain -> view -> bound slot D
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    g.output(&v, "D");
    g.output(&s, "E");
    // ESCAPE-AND-DISCLOSE (ruling 2026-08-27): the view slot no longer
    // writes into D at all — the chain's minted residence ESCAPES and
    // backs slot D directly, so the hazard this probe attacked (a
    // delivery overwriting D while the unordered reader still needs x's
    // bytes) is GONE, not tolerated: x's buffer is never written, the
    // WAR edge the old pin measured has nothing to order, and only s's
    // dense delivery into E copies.
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());
    assert!(matches!(plan.value_buffer[&y], BufferId::Allocated(_)));
    let slot = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
            _ => None,
        })
        .expect("slot 0");
    assert_eq!(
        slot.buffer, plan.value_buffer[&y],
        "slot D's declared buffer is unused"
    );
    assert_eq!(
        plan.buffers[&slot.buffer].freed_by,
        luminal::layout_ir::FreedBy::Caller,
        "the chain residence escapes:\n{}",
        plan.summary()
    );
    let into_x: Vec<_> = copies(&plan)
        .into_iter()
        .filter(|(_, dst)| *dst == plan.value_buffer[&x])
        .collect();
    assert!(
        into_x.is_empty(),
        "nothing writes the input buffer anymore:\n{}",
        plan.summary()
    );
    assert_eq!(
        copies(&plan).len(),
        1,
        "one dense delivery (s -> E) remains:\n{}",
        plan.summary()
    );
    assert_eq!(
        war_antis(&plan),
        0,
        "no hazard left to WAR-order:\n{}",
        plan.summary()
    );
}

/// ATTACK 4: a mutating consumer reaching a bound-output value THROUGH a
/// view. y is bound to slot D (its END_OF_PROGRAM read never happens-before
/// anything); the accumulator takes v = view(y) in place. Views decide first
/// (v~y union), so the accumulator's write provably aliases the bound value
/// and is vetoed; the repair copies the bytes out and the consumer reads the
/// copy. The shared buffer keeps exactly one writing compute.
///
/// RULING 2026-08-27 (repair destinations are fresh single-writer buffers):
/// the repair copy of a FOLDED operand targets a FRESHLY minted
/// parent-shaped buffer — never the tied result's buffer, whose writer
/// votes result-shaped geometry the base-storage copy would contradict —
/// and the consumer's operand re-roots onto it through its unchanged fold.
#[test]
fn a1_mutating_consumer_through_view_of_bound_value_vetoed_and_repaired() {
    let mut g = TestGraph::new();
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    // accumulator: reads v, wants to write v's storage in place
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&y, "D");
    g.output(&r, "E");
    let graph = g.build();
    let table = test_runtime::test_support::mock_layout_table(&graph);
    let plan = test_runtime::test_support::bufferize_mock(&graph).expect("bufferize");
    println!("{}", plan.summary());

    // y seeded into D; the accumulator must NOT write D.
    let d = plan.value_buffer[&y].clone();
    assert!(
        matches!(d, BufferId::Boundary(_)),
        "y seeds into D:\n{}",
        plan.summary()
    );
    assert_ne!(
        plan.value_buffer[&r], d,
        "accumulator relocated off the bound buffer"
    );
    let writers_of_d = plan
        .dag
        .node_weights()
        .filter(|n| matches!(n, BufferNode::Compute { writes, .. } if writes.contains(&d)))
        .count();
    assert_eq!(
        writers_of_d,
        1,
        "exactly one writer of the bound buffer:\n{}",
        plan.summary()
    );
    // The repair copied D's bytes into a FRESH single-writer buffer the
    // consumer re-roots onto — never the accumulator's own result buffer.
    let repair = copies(&plan)
        .into_iter()
        .find(|(src, _)| *src == d)
        .expect("repair copy reads the bound buffer");
    assert!(matches!(repair.1, BufferId::Allocated(_)));
    assert_ne!(
        repair.1,
        plan.value_buffer[&r],
        "fresh repair destination, not the tied result's buffer:\n{}",
        plan.summary()
    );
    let consumer_operand = plan
        .dag
        .node_weights()
        .find_map(|n| match n {
            BufferNode::Compute { operand_info, .. }
                if operand_info.first().is_some_and(|info| info.value == v) =>
            {
                Some(operand_info[0].clone())
            }
            _ => None,
        })
        .expect("the accumulator survives lowering");
    assert_eq!(
        consumer_operand.buffer, repair.1,
        "the consumer reads the re-rooted view"
    );
    // …through the VIEW's own layout, unchanged by the re-root (the
    // layout addresses the residence's bytes, and the copy is
    // byte-identical).
    assert_eq!(
        &consumer_operand.layout, &table[&v],
        "the view's own layout, unchanged"
    );
}
