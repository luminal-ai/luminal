//! ATTACKER B1 — lifetime + memory-safety attacks on P2 (view aliasing).
//! Austin 2026-08-26 adversarial review.
//!
//! Six attack lines against the claim that folded-view aliasing is safe:
//!  (1) parent freed while the alias lives (free placement vs unordered
//!      view readers; the poison/no-final-resident panic door);
//!  (2) double-free via alias records (incl. Program-freed DONATED
//!      boundary storage read only through a view);
//!  (3) alias of a PINNED input elected as a DPS in-place destination
//!      (ReadOnly veto depth; pass-through obligations; the ReadWrite
//!      donation admission that really does write the caller's buffer);
//!  (4) WAR: the anti-edge that materialization used to make unnecessary
//!      (view reader of an input buffer vs the delivery copy that
//!      overwrites it);
//!  (5) in-place mutators vs readers-through-views ordered AFTER the
//!      mutation (stale-read hole probe);
//!  (6) executor binding is covered by code-reading (see report): no
//!      probe here can execute MockOps (no reference kernel).
use luminal::bufferize::{BufferId, BufferNode, EdgeKind};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::petgraph;
use petgraph::algo::has_path_connecting;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use test_runtime::test_support::{EmptyOp, MockOp, MockView, TestGraph};

fn computes(
    plan: &luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
    label: &str,
) -> Vec<(NodeIndex, Vec<BufferId>, Vec<BufferId>)> {
    plan.dag
        .node_indices()
        .filter_map(|i| match &plan.dag[i] {
            BufferNode::Compute {
                op, reads, writes, ..
            } if op.label() == label => Some((i, reads.clone(), writes.clone())),
            _ => None,
        })
        .collect()
}

fn copies(
    plan: &luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
) -> Vec<(NodeIndex, BufferId, BufferId)> {
    plan.dag
        .node_indices()
        .filter_map(|i| match &plan.dag[i] {
            BufferNode::BufferCopy { src, dst, .. } => Some((i, src.clone(), dst.clone())),
            _ => None,
        })
        .collect()
}

fn input_buffer(
    plan: &luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
) -> BufferId {
    plan.dag
        .node_weights()
        .find_map(|n| match n {
            BufferNode::BufferInput { slots } => Some(slots[0].buffer.clone()),
            _ => None,
        })
        .expect("input node")
}

/// ATTACK 1a — free placed after the "last toucher" while ANOTHER reader
/// of the shared buffer (through the folded view) is UNORDERED against it.
/// If free placement keyed on the parent VALUE's last direct reader, the
/// view reader could execute after the free. The defense on trial:
/// touchers are buffer-keyed and include readers-through-views
/// (buffer_tensor_ir.rs:1224-1245), and unordered touchers get Anti edges
/// into the free (buffer_tensor_ir.rs:1391-1415).
#[test]
fn attack1_unordered_direct_and_view_readers_both_precede_free() {
    let mut g = TestGraph::new();
    let x = g.input("x", "x", Access::ReadOnly, "rm");
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    // writer W: produces y in a System allocation (in-place into p's poison)
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: false,
            }),
            &[&x, &p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    // two UNORDERED readers: one direct, one through the alias
    let r1 = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&y],
            &[("r1", "rm")],
        )
        .remove(0);
    let r2 = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v],
            &[("r2", "rm")],
        )
        .remove(0);
    g.output(&r1, "o1");
    g.output(&r2, "o2");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    // Identify the parent buffer: written by the MockOp whose operand is the
    // Boundary input (the writer W).
    let mocks = computes(&plan, "MockOp");
    let parent = mocks
        .iter()
        .find(|(_, r, w)| {
            matches!(r[0], BufferId::Boundary(_)) && matches!(w[0], BufferId::Allocated(_))
        })
        .expect("writer W")
        .2[0]
        .clone();

    // Exactly one alloc, one free of the parent (no alias-driven double free).
    let allocs: Vec<_> = computes(&plan, "BufferAlloc")
        .into_iter()
        .filter(|(_, _, w)| w[0] == parent)
        .collect();
    let frees: Vec<_> = computes(&plan, "BufferFree")
        .into_iter()
        .filter(|(_, r, _)| r[0] == parent)
        .collect();
    assert_eq!(allocs.len(), 1, "one alloc of the parent");
    assert_eq!(frees.len(), 1, "one free of the parent");
    let free = frees[0].0;

    // BOTH readers of the parent buffer (direct AND through the folded view)
    // must be path-ordered before the free. The view reader reads the SAME
    // BufferId (fold), so if either path is missing, the plan use-after-frees.
    let readers: Vec<_> = mocks
        .iter()
        .filter(|(_, r, w)| r[0] == parent && w[0] != parent)
        .collect();
    assert_eq!(
        readers.len(),
        2,
        "direct reader + view reader both read the parent buffer"
    );
    for (idx, _, _) in &readers {
        assert!(
            has_path_connecting(&plan.dag, *idx, free, None),
            "a reader of the shared buffer is not ordered before its free:\n{}",
            plan.summary()
        );
    }
    // At least one of them needed a synthesized Anti edge (they are unordered
    // by dataflow, so only one can precede the free by Data edges alone).
    let anti_into_free = plan
        .dag
        .edges_directed(free, petgraph::Direction::Incoming)
        .filter(|e| e.weight().kind == EdgeKind::Anti)
        .count();
    assert!(
        anti_into_free >= 1,
        "the unordered toucher needs an Anti edge into the free:\n{}",
        plan.summary()
    );
}

/// ATTACK 1b — SECOND ENTRANCE to the poison door the P2 advocate found:
/// no reader op at all, just a VIEW OF UNDEFINED STORAGE bound straight to
/// an output slot. The slot-binds-poison check (bufferize.rs:1236-1244) is
/// value-keyed and the view is a different value, so validation passes;
/// the delivery copy then touches a buffer nothing ever wrote. Expected
/// today: panic at buffer_tensor_ir.rs:1353 ("a freed buffer has a final
/// resident"), NOT a loud Result. Documents that the door is wider than
/// the reader-op shape.
#[test]
fn attack1b_view_of_poison_bound_to_output_slot() {
    let graph = {
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
        let v = g.op(Box::new(MockView), &[&e], &[("v", "view")]).remove(0);
        g.output(&v, "out");
        g.build()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        test_runtime::test_support::bufferize_mock(&graph)
    }));
    match result {
        Ok(Ok(plan)) => panic!(
            "LAUNDERED: undefined bytes delivered to a bound output through a view:\n{}",
            plan.summary()
        ),
        Ok(Err(e)) => println!("REJECTED loudly (door closed): {e:#}"),
        Err(_) => println!(
            "PANIC confirmed: slot-poison check is value-keyed; the view \
             launders the undefined value into a delivery copy and optimize \
             dies on the missing final resident"
        ),
    }
}

/// ATTACK 2 — double free / free placement on DONATED (FreedBy::Program)
/// caller storage whose value is read ONLY through a view. If the alias
/// were invisible to the toucher scan, the free would land right after the
/// Input node and the view reader would use-after-free; if the alias
/// minted its own record, the buffer would free twice.
#[test]
fn attack2_program_freed_input_read_only_through_view() {
    let mut g = TestGraph::new();
    let x = g.input_binding(
        "x",
        "xb",
        Some(Access::ReadWrite),
        Some(FreedBy::Program),
        "rm",
    );
    let v = g.op(Box::new(MockView), &[&x], &[("v", "view")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    let xb = input_buffer(&plan);
    assert!(matches!(xb, BufferId::Boundary(_)));
    let frees: Vec<_> = computes(&plan, "BufferFree")
        .into_iter()
        .filter(|(_, r, _)| r[0] == xb)
        .collect();
    assert_eq!(
        frees.len(),
        1,
        "exactly one free of the donated buffer:\n{}",
        plan.summary()
    );
    // The reader-through-the-view precedes the free.
    let reader = computes(&plan, "MockOp")
        .into_iter()
        .find(|(_, r, _)| r[0] == xb)
        .expect("reader reads the donated buffer through the folded view");
    assert!(
        has_path_connecting(&plan.dag, reader.0, frees[0].0, None),
        "the view reader must precede the donated buffer's free:\n{}",
        plan.summary()
    );
}

/// ATTACK 3a — writability veto DEPTH: an in-place accumulator offered a
/// VIEW OF A VIEW of read-only pinned storage. If the veto only checked
/// one alias hop, the writer would scribble on caller weights.
#[test]
fn attack3a_view_of_view_of_readonly_vetoes_writer() {
    let mut g = TestGraph::new();
    let x = g.input("x", "x", Access::ReadOnly, "rm");
    let v1 = g
        .op(Box::new(MockView), &[&x], &[("v1", "view1")])
        .remove(0);
    let v2 = g
        .op(Box::new(MockView), &[&v1], &[("v2", "view2")])
        .remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&v2],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    let xb = input_buffer(&plan);
    for i in plan.dag.node_indices() {
        match &plan.dag[i] {
            BufferNode::Compute { op, writes, .. } => assert!(
                !writes.contains(&xb),
                "read-only caller storage written by {} through a two-view chain",
                op.label()
            ),
            BufferNode::BufferCopy { dst, .. } => {
                assert_ne!(dst, &xb, "copy into read-only caller storage")
            }
            _ => {}
        }
    }
    // Repair: the accumulator got a copy OUT of the caller's buffer.
    let cps = copies(&plan);
    assert!(
        cps.iter()
            .any(|(_, src, dst)| *src == xb && matches!(dst, BufferId::Allocated(_))),
        "repair copies the parent bytes out of the read-only buffer:\n{}",
        plan.summary()
    );
}

/// ATTACK 3b — the PASS-THROUGH obligation: the caller's ReadWrite input is
/// promised back (output slot pass-through), and an accumulator elects a
/// VIEW of it in place. The end-of-program read of x aliases the view, so
/// the write must be rejected — otherwise the caller gets the accumulator's
/// bytes where the pass-through promised x's.
#[test]
fn attack3b_passthrough_obligation_blocks_inplace_writer_via_view() {
    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let v = g.op(Box::new(MockView), &[&x], &[("v", "view")]).remove(0);
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
    g.output(&x, "B"); // pass-through: B's final contents must be x
    g.output(&r, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    let xb = input_buffer(&plan);
    for i in plan.dag.node_indices() {
        match &plan.dag[i] {
            BufferNode::Compute { op, writes, .. } => assert!(
                !writes.contains(&xb) || op.label() == "BufferFree",
                "pass-through buffer written by {}",
                op.label()
            ),
            BufferNode::BufferCopy { dst, .. } => {
                assert_ne!(dst, &xb, "copy into pass-through buffer")
            }
            _ => {}
        }
    }
    // The accumulator relocated: repair copy B -> fresh alloc.
    let cps = copies(&plan);
    assert!(
        cps.iter()
            .any(|(_, src, dst)| *src == xb && matches!(dst, BufferId::Allocated(_))),
        "repair copy expected:\n{}",
        plan.summary()
    );
}

/// ATTACK 3c — THE ADMITTED CASE, pinned down: with NO other reader and NO
/// pass-through obligation, an accumulator on a view of a ReadWrite caller
/// input IS admitted in place — the plan really does overwrite the
/// caller's input buffer through the alias. This is contract-sanctioned
/// (Access::ReadWrite = exclusive read/write, layout_ir), but it is the
/// exact behavior attack 3 warned about: P2's alias makes a caller input
/// electable as a DPS destination. The probe pins the precondition:
/// admission REQUIRES ReadWrite + no surviving reads, nothing less.
#[test]
fn attack3c_readwrite_input_is_admitted_as_inplace_dest_through_view() {
    let mut g = TestGraph::new();
    let x = g.input("x", "xb", Access::ReadWrite, "rm");
    let v = g.op(Box::new(MockView), &[&x], &[("v", "view")]).remove(0);
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
    g.output(&r, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    let xb = input_buffer(&plan);
    let acc = computes(&plan, "MockOp").pop().expect("accumulator");
    // Admitted: reads and writes the CALLER's buffer in place.
    assert_eq!(
        acc.1[0], xb,
        "accumulator reads the caller buffer through the view"
    );
    assert_eq!(
        acc.2[0], xb,
        "accumulator WRITES the caller's ReadWrite input buffer in place"
    );
    // The output is served by a delivery copy out of the caller's buffer.
    let cps = copies(&plan);
    assert_eq!(cps.len(), 1);
    assert_eq!(cps[0].1, xb);
    assert!(matches!(cps[0].2, BufferId::Boundary(_)));
}

/// ATTACK 4 — the WAR edge materialization used to make unnecessary: a
/// consumer reads the caller input buffer THROUGH a folded view while a
/// delivery copy overwrites that same buffer (an output slot binds a
/// computed value into it). Materialized, the view would be a separate
/// buffer and no hazard would exist; folded, the reader and the copy
/// share a BufferId and are dataflow-unordered. The residency rule must
/// install Anti(reader -> copy).
#[test]
fn attack4_view_reader_anti_ordered_before_delivery_copy_overwrite() {
    let mut g = TestGraph::new();
    let x = g.input("x", "B", Access::ReadWrite, "rm");
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&x],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&x], &[("v", "view")]).remove(0);
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&y, "B"); // delivery copy: alloc(y) -> B, overwriting the input buffer
    g.output(&r, "C");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    let xb = input_buffer(&plan);
    let (copy_b, _, _) = copies(&plan)
        .into_iter()
        .find(|(_, _, dst)| *dst == xb)
        .expect("delivery copy overwrites the input buffer");

    // The view reader: identified as the MockOp whose result feeds slot C's
    // delivery copy (its write buffer is the src of the copy into C).
    let copy_c = copies(&plan)
        .into_iter()
        .find(|(_, _, dst)| *dst != xb && matches!(dst, BufferId::Boundary(_)))
        .expect("delivery copy into C");
    let reader = computes(&plan, "MockOp")
        .into_iter()
        .find(|(_, _, w)| w[0] == copy_c.1)
        .expect("view reader feeds slot C");
    assert_eq!(
        reader.1[0], xb,
        "the reader reads the input buffer through the folded view"
    );

    // THE EDGE: Anti(reader -> copy-into-B). Without it, an eager executor
    // may overwrite x mid-read — the exact WAR that a materialized view
    // could never have.
    let anti: Vec<_> = plan
        .dag
        .edges_directed(copy_b, petgraph::Direction::Incoming)
        .filter(|e| e.weight().kind == EdgeKind::Anti)
        .collect();
    assert_eq!(
        anti.len(),
        1,
        "exactly one WAR anti into the overwriting copy:\n{}",
        plan.summary()
    );
    assert_eq!(
        anti[0].source(),
        reader.0,
        "the anti's source is the reader-through-the-view"
    );
}

/// ATTACK 5 — stale-read probe: a reader through the view that is
/// dataflow-ordered AFTER an in-place mutator of the parent. If admission
/// treated "ordered" as sufficient (rather than ordered-BEFORE), the
/// reader would observe the mutated bytes through the alias while naming
/// the old value. The mutator must be rejected.
#[test]
fn attack5_view_reader_after_mutator_still_blocks_inplace() {
    let mut g = TestGraph::new();
    let x = g.input("x", "x", Access::ReadOnly, "rm");
    let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")]).remove(0);
    let y = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: false,
            }),
            &[&x, &p],
            &[("y", "rm")],
        )
        .remove(0);
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    // in-place mutator of y (accumulator)
    let m = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&y],
            &[("m", "rm")],
        )
        .remove(0);
    // reader of the view, ordered AFTER the mutator (consumes m too)
    let r = g
        .op(
            Box::new(MockOp {
                reads: vec![true, true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v, &m],
            &[("r", "rm")],
        )
        .remove(0);
    g.output(&r, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    // Identify the parent buffer (written by W, whose operand 0 is Boundary).
    let mocks = computes(&plan, "MockOp");
    let parent = mocks
        .iter()
        .find(|(_, r, w)| {
            matches!(r[0], BufferId::Boundary(_)) && matches!(w[0], BufferId::Allocated(_))
        })
        .expect("writer W")
        .2[0]
        .clone();

    // The mutator was REJECTED: exactly one Allocated->Allocated repair copy
    // (parent -> the mutator's fresh buffer), and the parent buffer keeps a
    // single writing compute.
    let repairs: Vec<_> = copies(&plan)
        .into_iter()
        .filter(|(_, src, dst)| *src == parent && matches!(dst, BufferId::Allocated(_)))
        .collect();
    assert_eq!(
        repairs.len(),
        1,
        "the mutator relocated via a repair copy:\n{}",
        plan.summary()
    );
    let parent_writers = mocks.iter().filter(|(_, _, w)| w[0] == parent).count();
    assert_eq!(
        parent_writers,
        1,
        "the shared buffer keeps exactly one writer:\n{}",
        plan.summary()
    );
    // And the late reader still reads the parent buffer through the fold.
    // (Identified as the MockOp whose SECOND operand is the mutator's
    // relocated buffer — the plan's `reads` vector lists every operand
    // buffer, so the writer W also has two entries and cannot be used.)
    let mutator_buffer = repairs[0].2.clone();
    let late_reader = mocks
        .iter()
        .find(|(_, r, _)| r.len() == 2 && r[1] == mutator_buffer)
        .expect("the view reader consumes the relocated mutator result");
    assert_eq!(
        late_reader.1[0], parent,
        "reader reads the OLD value's bytes in the parent buffer"
    );
}
