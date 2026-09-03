//! FOLDED-OPERAND LAYOUT PIN (was `phase3_composed_access_probe`).
//!
//! The Phase-3 machinery this file pinned — `ComposedAccess`, per-slot hop
//! chains, `parent_dims` — is DELETED (corrected contract, 2026-08-31).
//! The reason is stated on `SlotDescriptor`: recording per-slot view-fold
//! chains was the PLANNER composing layout knowledge, and the e-graph
//! already mints every view value's composed layout at view creation. The
//! runtime's decoded `L` for that value IS the read path.
//!
//! So the plan-level fact worth pinning changed shape. It is no longer
//! "the fold's map survived onto the descriptor"; it is:
//!
//!   * the consumer's operand descriptor names the VIEW value while
//!     residing in the PARENT's buffer (the fold's redirect), and
//!   * it carries the VIEW's own elected layout — a different function
//!     from the parent's, over the same bytes.
//!
//! That second point is the whole content of the old hop chain, now held
//! in one opaque value the planner transports and never interprets.
//!
//! ASSERTION DISCIPLINE. These plans are built with `MockLayout`, which
//! transports the layout e-class IDENTITY and nothing evaluable — the
//! test runtime has no layout decoder of its own. Element-level
//! evaluation of REAL decoded layouts is pinned where real layouts
//! exist: `luminal_cuda_lite/tests/view_admission.rs` (searched plans,
//! flat-index evaluation against hand-computed maps) and
//! `test_runtime::test_equality` (readback through a returned
//! (buffer, layout) pair). Here we pin identity and distinctness.

use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::test_support::MockLayout;
use std::collections::HashMap;

type FoldedSlot = (String, usize, MockLayout);

/// Bufferize a frontend program with views preferred; return the plan,
/// the decoded (mock) layout table, and every operand slot READING
/// THROUGH A FOLD.
///
/// THE DISCRIMINATOR IS THE LAYOUT, not the assignment. The plan does not
/// label folds, and it should not: a folded view and a DPS in-place
/// cohabitant are both just "this value lives in this buffer", and the
/// `backs` row names whichever value the buffer was minted for (a poison
/// destination, on every DPS chain). What separates them is the FUNCTION:
/// a cohabitant carries the same `L` the buffer was allocated for (the
/// poison clones its tied result's layout), a view carries a different
/// one over the same bytes. Layout equality is a runtime/test-side
/// operation on a concrete `L` — core never compares layouts.
fn folded_slots(
    text: &str,
    prefer: &[&str],
) -> (
    BufferIrGraph<MockLayout>,
    HashMap<luminal::prelude::egraph_serialize::ClassId, MockLayout>,
    Vec<FoldedSlot>,
) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, prefer);
    let dps = luminal::dps::dps_rewrite(&graph);
    let table = luminal::test_support::mock_layout_table(&dps);
    let plan = luminal::test_support::bufferize_mock(&dps).expect("bufferize");
    let mut found = Vec::new();
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = node
        {
            for (slot, info) in operand_info.iter().enumerate() {
                // The slot carries its OWN value's layout, always.
                assert_eq!(
                    Some(&info.layout),
                    table.get(&info.value),
                    "{} operand {slot}: the descriptor carries the slot value's layout",
                    op.label()
                );
                if info.layout != plan.buffers[&info.buffer].layout {
                    found.push((op.label().to_string(), slot, info.layout.clone()));
                }
            }
            for info in result_info {
                assert_eq!(
                    info.layout,
                    plan.buffers[&info.buffer].layout,
                    "{}: a compute RESULT is written in its buffer's own layout — \
                     never produced through a fold",
                    op.label()
                );
            }
        }
    }
    (plan, table, found)
}

/// TRANSPOSE VIEW: `x.permute((1,0))` feeding a mul. The folded view's
/// consumer must read the VIEW value out of the PARENT's buffer, carrying
/// the view's own layout.
#[test]
fn transpose_view_consumer_carries_the_views_own_layout() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 3usize), DType::F32);
        let c = cx.tensor((3usize, 2usize), DType::F32);
        let _ = (x.permute((1, 0)) * c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, table, found) = folded_slots(&text, &["LayoutTensorOpIndexMapApplyViewGeneric"]);
    assert!(
        !found.is_empty(),
        "no operand reads through a fold — was the view elected?\n{}",
        plan.summary()
    );
    let (label, slot, layout) = &found[0];
    println!("folded read on {label} operand {slot}");
    // The residence's own layout is a DIFFERENT function: that difference
    // is exactly what the old hop chain encoded, now held in one opaque
    // value the planner transports and never interprets.
    let residence_layout = plan
        .dag
        .node_weights()
        .find_map(|n| match n {
            BufferNode::Compute {
                op, operand_info, ..
            } if op.label() == label && operand_info.len() > *slot => {
                Some(plan.buffers[&operand_info[*slot].buffer].layout.clone())
            }
            _ => None,
        })
        .expect("the folded slot's buffer has a record");
    assert_ne!(
        layout, &residence_layout,
        "the view's layout must differ from the one its buffer was allocated \
         for — the buffer id alone cannot describe the read"
    );
    assert!(
        table.values().any(|l| l == layout),
        "…and it is a decoded table row, transported verbatim"
    );
}

/// 2-HOP CHAIN: slice rows 1..3 of a (4,6), then transpose. The e-graph
/// composes at view creation, so however many view ops the genome elects,
/// the consumer ends up with ONE layout describing the whole composite —
/// there is no chain to count. (What the composite computes is pinned on
/// real layouts in `view_admission`.)
#[test]
fn sliced_transpose_chain_arrives_as_one_layout() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 6usize), DType::F32);
        let c = cx.tensor((6usize, 2usize), DType::F32);
        let _ = (x.slice((1..3, ..)).permute((1, 0)) * c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, table, found) = folded_slots(&text, &["LayoutTensorOpIndexMapApplyViewGeneric"]);
    assert!(
        !found.is_empty(),
        "no operand reads through the folded chain\n{}",
        plan.summary()
    );
    let (label, slot, layout) = &found[0];
    println!("folded read on {label} operand {slot} — one layout, no chain");
    assert!(
        table.values().any(|l| l == layout),
        "the carried layout is a decoded table row, transported verbatim"
    );
}

/// THE r10 CHAINED-MATMUL FIXTURE (CublasLt + views): the interior view
/// feeding the second call reads through a fold, and its descriptor
/// carries the view's own layout.
#[test]
fn r10_chained_matmuls_read_through_folded_layouts() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w1 = cx.tensor((8usize, 3usize), DType::F32);
        let w2 = cx.tensor((3usize, 5usize), DType::F32);
        let _ = x.matmul(w1).matmul(w2).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (plan, table, found) = folded_slots(
        &text,
        &[
            "LayoutTensorOpCublasLt",
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
    println!(
        "r10 chain: {} folded slot(s): {:?}",
        found.len(),
        found
            .iter()
            .map(|(l, s, _)| (l.clone(), *s))
            .collect::<Vec<_>>()
    );
    assert!(
        !found.is_empty(),
        "the interior view folded but no descriptor reads through it\n{}",
        plan.summary()
    );
    for (label, slot, layout) in &found {
        assert!(
            table.values().any(|l| l == layout),
            "{label} operand {slot}: the carried layout must be a decoded row, \
             transported verbatim — never synthesized by the planner"
        );
    }
}
