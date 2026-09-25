//! P2 (VIEW ALIASING) ADVERSARIAL PROBES — Austin 2026-08-26 review.
//!
//! P2 proposes: lower IndexMapApplyViewGeneric to an alias record (view's
//! (value,buffer) = parent's buffer) instead of materialize, with lifetime
//! extension of the parent. These probes test what the CURRENT bufferizer
//! already does for admitted views, and attack the exact seams P2 would
//! widen: lifetime extension through view readers, double-free, views of
//! pinned/read-only storage, WAR against writers of the shared buffer,
//! view-of-view chains, and poison laundering through a non-reading alias.
//!
//! Hand-authored graphs via test_runtime::test_support (MockOp/MockView/EmptyOp),
//! per the assignment rule: these shapes are defined by the Bufferizable
//! interface, not reachable through egg scripts.
use luminal::bufferize::{BufferId, BufferNode};
use luminal::layout_ir::Access;
use luminal::prelude::petgraph;
use petgraph::algo::has_path_connecting;
use petgraph::graph::NodeIndex;
use test_runtime::test_support::{EmptyOp, MockOp, MockView, TestGraph};

fn compute_nodes<'a>(
    plan: &'a luminal::bufferize::BufferIrGraph<test_runtime::test_support::MockLayout>,
    label: &str,
) -> Vec<(NodeIndex, &'a Vec<BufferId>, &'a Vec<BufferId>)> {
    plan.dag
        .node_indices()
        .filter_map(|i| match &plan.dag[i] {
            BufferNode::Compute {
                op, reads, writes, ..
            } if op.label() == label => Some((i, reads, writes)),
            _ => None,
        })
        .collect()
}

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

/// (a)+(b): a consumer reading THROUGH a folded view must keep the parent
/// buffer alive — the free lands after the reader — and the parent gets
/// exactly one alloc and one free (no double-free from the alias).
#[test]
fn a_view_reader_extends_parent_lifetime_single_free() {
    let mut g = TestGraph::new();
    let x = g.input("x", "x", Access::ReadOnly, "rm");
    // writer: dest-tied into EmptyOp storage (System alloc after bufferize)
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
    // the view (alias) of y
    let v = g.op(Box::new(MockView), &[&y], &[("v", "view")]).remove(0);
    // consumer reads v, writes its own dest (seeded to the bound output)
    let q = g.op(Box::new(EmptyOp), &[], &[("q", "rm")]).remove(0);
    let z = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: false,
            }),
            &[&v, &q],
            &[("z", "rm")],
        )
        .remove(0);
    g.output(&z, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    // The view folded: zero copies anywhere (the consumer's dest was seeded
    // into the bound output, so not even a delivery copy).
    assert_eq!(copies(&plan), vec![], "no copies expected");
    // No view node survives.
    assert!(
        compute_nodes(&plan, "MockView").is_empty(),
        "view must fold"
    );

    // Identify the parent buffer: the one the writer MockOp writes and the
    // consumer MockOp reads (through the folded view).
    let mocks = compute_nodes(&plan, "MockOp");
    assert_eq!(mocks.len(), 2);
    let (writer, consumer) = {
        let w = mocks
            .iter()
            .find(|(_, _, w)| matches!(w[0], BufferId::Allocated(_)))
            .expect("writer into System buffer");
        let c = mocks
            .iter()
            .find(|(_, _, w)| matches!(w[0], BufferId::Boundary(_)))
            .expect("consumer into bound output (seeded)");
        (*w, *c)
    };
    let parent = writer.2[0].clone();
    assert_eq!(
        consumer.1[0], parent,
        "consumer reads the parent buffer through the alias"
    );

    // Exactly one alloc and one free of the parent buffer.
    let allocs = compute_nodes(&plan, "BufferAlloc");
    let frees = compute_nodes(&plan, "BufferFree");
    let parent_allocs: Vec<_> = allocs.iter().filter(|(_, _, w)| w[0] == parent).collect();
    let parent_frees: Vec<_> = frees.iter().filter(|(_, r, _)| r[0] == parent).collect();
    assert_eq!(parent_allocs.len(), 1, "one alloc for the parent");
    assert_eq!(
        parent_frees.len(),
        1,
        "one free for the parent (no alias double-free)"
    );

    // Lifetime extension: the free is path-ordered AFTER the view reader.
    let free = parent_frees[0].0;
    assert!(
        has_path_connecting(&plan.dag, consumer.0, free, None),
        "the parent's free must come after the reader-through-the-view"
    );
}

/// (e): view-of-view chains fold transitively — the consumer reads the
/// grandparent's buffer directly, no copies, no surviving view nodes.
#[test]
fn e_view_of_view_chain_folds_to_grandparent() {
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
    let v1 = g
        .op(Box::new(MockView), &[&y], &[("v1", "view1")])
        .remove(0);
    let v2 = g
        .op(Box::new(MockView), &[&v1], &[("v2", "view2")])
        .remove(0);
    let q = g.op(Box::new(EmptyOp), &[], &[("q", "rm")]).remove(0);
    let z = g
        .op(
            Box::new(MockOp {
                reads: vec![true, false],
                in_place_operand: Some(1),
                not_conflicting: false,
            }),
            &[&v2, &q],
            &[("z", "rm")],
        )
        .remove(0);
    g.output(&z, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());
    assert_eq!(copies(&plan), vec![], "no copies through a two-view chain");
    assert!(compute_nodes(&plan, "MockView").is_empty());
    let mocks = compute_nodes(&plan, "MockOp");
    let writer = mocks
        .iter()
        .find(|(_, _, w)| matches!(w[0], BufferId::Allocated(_)))
        .unwrap();
    let consumer = mocks
        .iter()
        .find(|(_, _, w)| matches!(w[0], BufferId::Boundary(_)))
        .unwrap();
    assert_eq!(
        consumer.1[0], writer.2[0],
        "consumer reads the grandparent buffer"
    );
}

/// (c): an in-place accumulator offered a VIEW OF READ-ONLY pinned storage
/// must be rejected (writability veto propagates through the alias set) and
/// repaired with a copy out of the caller's buffer — the pinned bytes are
/// never written.
#[test]
fn c_view_of_readonly_input_vetoes_inplace_writer() {
    let mut g = TestGraph::new();
    let x = g.input("x", "x", Access::ReadOnly, "rm");
    let v = g.op(Box::new(MockView), &[&x], &[("v", "view")]).remove(0);
    // accumulator: reads operand 0 and writes result 0 into the same storage
    let z = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&v],
            &[("z", "rm")],
        )
        .remove(0);
    g.output(&z, "out");
    let plan = test_runtime::test_support::bufferize_mock(&g.build()).expect("bufferize");
    println!("{}", plan.summary());

    // x's boundary buffer is never written by any compute or copy.
    let x_buf = plan
        .dag
        .node_weights()
        .find_map(|n| match n {
            BufferNode::BufferInput { slots } => Some(slots[0].buffer.clone()),
            _ => None,
        })
        .expect("input node");
    for i in plan.dag.node_indices() {
        match &plan.dag[i] {
            BufferNode::Compute { op, writes, .. } => {
                assert!(
                    !writes.contains(&x_buf) || op.label() == "BufferFree",
                    "read-only pinned buffer written by {}",
                    op.label()
                );
            }
            BufferNode::BufferCopy { dst, .. } => {
                assert_ne!(dst, &x_buf, "copy into read-only buffer")
            }
            _ => {}
        }
    }
    // The repair copied the parent's bytes out of x's buffer into fresh
    // System storage (plus the ordinary boundary delivery copy to `out`).
    let cps = copies(&plan);
    assert_eq!(cps.len(), 2, "repair copy + boundary delivery copy");
    let repair = cps
        .iter()
        .find(|(src, _)| *src == x_buf)
        .expect("repair copy reads the read-only parent");
    assert!(matches!(repair.1, BufferId::Allocated(_)));
}

/// (d): a writer into the shared buffer, UNORDERED against a reader through
/// the view, must not be admitted in place — the repair keeps the reader's
/// bytes intact (copy-before-overwrite), and the plan still certifies.
#[test]
fn d_unordered_view_reader_blocks_inplace_writer() {
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
    // reader through the view: plain consumer, fresh result storage
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
    // in-place accumulator DIRECTLY on y: wants to overwrite the shared buffer,
    // unordered against the view reader (no dataflow path either way).
    let w2 = g
        .op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&y],
            &[("w2", "rm")],
        )
        .remove(0);
    g.output(&r, "out_r");
    g.output(&w2, "out_w");
    let plan = test_runtime::test_support::bufferize_mock(&g.build())
        .expect("bufferize must still certify");
    println!("{}", plan.summary());

    // The accumulator's in-place bid was rejected: its overwrite happens in a
    // COPY of the parent buffer. Exactly one Allocated->Allocated repair copy
    // exists (the two Allocated->Boundary copies are ordinary deliveries).
    let cps = copies(&plan);
    let repairs: Vec<_> = cps
        .iter()
        .filter(|(src, dst)| {
            matches!(src, BufferId::Allocated(_)) && matches!(dst, BufferId::Allocated(_))
        })
        .collect();
    assert_eq!(
        repairs.len(),
        1,
        "one repair copy for the rejected accumulator"
    );
    // No compute writes the parent buffer twice: the parent has exactly one
    // writing compute (the original writer).
    let parent = repairs[0].0.clone();
    let writers: Vec<_> = plan
        .dag
        .node_weights()
        .filter(|n| {
            matches!(n, BufferNode::Compute { op, writes, .. }
            if op.label() == "MockOp" && writes.contains(&parent))
        })
        .collect();
    assert_eq!(writers.len(), 1, "the shared buffer keeps a single writer");
}

/// (c)+P1 boundary: a view result bound directly to an output slot.
/// ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27, supersedes both the
/// P1-era delivery-copy pin and the 5b typed refusal): P2 (aliasing)
/// folds the view AND the boundary costs nothing — the chain's minted
/// residence ESCAPES (FreedBy::Caller, no free), the slot is backed by
/// it, and the binding discloses the view layout. Zero copies.
#[test]
fn c_view_bound_to_output_escapes_the_chain_residence() {
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
    g.output(&v, "out");
    let graph = g.build();
    let table = test_runtime::test_support::mock_layout_table(&graph);
    let plan = test_runtime::test_support::bufferize_mock(&graph).expect("the view output escapes");
    println!("{}", plan.summary());
    let cps = copies(&plan);
    assert_eq!(
        cps.len(),
        0,
        "zero copies — the residence is handed over:\n{}",
        plan.summary()
    );
    let slot = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
            _ => None,
        })
        .expect("one output slot");
    assert_eq!(
        slot.buffer, plan.value_buffer[&y],
        "backed by the chain's residence"
    );
    assert!(matches!(slot.buffer, BufferId::Allocated(_)));
    assert_eq!(
        plan.buffers[&slot.buffer].freed_by,
        luminal::layout_ir::FreedBy::Caller,
        "…which ESCAPES:\n{}",
        plan.summary()
    );
    // OPTION B: the binding carries the VIEW's own elected layout — the
    // delivery description an externally loaded plan would otherwise
    // have no way to obtain. (Presence is total now; identity is the
    // fact worth pinning.)
    assert_eq!(
        &slot.layout, &table[&v],
        "the view's own layout rides the binding"
    );
    assert_ne!(
        table[&v], table[&y],
        "…a different function from the residence's"
    );
}

/// (g): POISON THROUGH A VIEW. A view's operand is not READ (the poison door
/// is effect-keyed, bufferize.rs:1167-1169), and undefinedness does not
/// propagate to the view's result — so a consumer reading undefined bytes
/// THROUGH a view passes input validation. MEASURED OUTCOME (2026-08-26):
/// the plan is not silently certified — `optimize` PANICS at
/// buffer_tensor_ir.rs:1353 (`expect("a freed buffer has a final resident")`)
/// because the shared buffer is freed (its view reader is a toucher) yet
/// nothing ever wrote it. A panic, not a Result error: the validation door
/// that should reject this program loudly is missed by the effect-keyed
/// check. Today unreachable from the frontend (dps_rewrite mints poisons
/// post-extraction); P2's admission of views into general extraction moves
/// this door closer to reachable.
#[test]
fn g_view_of_poison_read_panics_not_bails() {
    let graph = {
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")]).remove(0);
        let v = g.op(Box::new(MockView), &[&e], &[("v", "view")]).remove(0);
        let q = g.op(Box::new(EmptyOp), &[], &[("q", "rm")]).remove(0);
        let z = g
            .op(
                Box::new(MockOp {
                    reads: vec![true, false],
                    in_place_operand: Some(1),
                    not_conflicting: false,
                }),
                &[&v, &q],
                &[("z", "rm")],
            )
            .remove(0);
        g.output(&z, "out");
        g.build()
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        test_runtime::test_support::bufferize_mock(&graph)
    }));
    match result {
        Ok(Ok(plan)) => {
            panic!(
                "LAUNDERED: read-of-undefined-through-view certified.\n{}",
                plan.summary()
            );
        }
        Ok(Err(e)) => {
            println!("REJECTED loudly (door closed): {e:#}");
        }
        Err(_) => {
            println!(
                "PANIC confirmed: the effect-keyed poison door admits the \
                 view-mediated undefined read; optimize dies on the missing \
                 final resident (buffer_tensor_ir.rs:1353)"
            );
        }
    }
}
