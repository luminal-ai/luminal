//! MARKER-design fixtures. Fixtures are RECORDED from the live frontend
//! wherever the frontend can spell them; every genome-pinned test checks
//! the parsed LtMatmulSpec field by field and cross-checks the Lit-arity
//! contract (a constant of the constructor name).

use std::time::Instant;

use luminal::buffer_tensor_ir::OpSlotNames;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use test_runtime::cublaslt_marker::{CuEpilogue, CublasLt};

const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    // ROUND 10 (view admission in ELECTION): the sibling site's result is
    // routed to the recorder's boundary value by a transpose VIEW; prefer
    // the view op over materialize/copy wherever both produce a class.
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

fn timed_serialize(text: &str) -> (luminal::prelude::egraph_serialize::EGraph, f64) {
    let start = Instant::now();
    let egraph = test_runtime::serialize_fixture(text);
    (egraph, start.elapsed().as_secs_f64())
}

/// All CublasLt compute nodes of a genome-pinned extraction, each with its
/// Lit-order input value classes.
fn pinned_cublaslt_ops(text: &str) -> Vec<(CublasLt, Vec<String>)> {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, PIN);
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                let concrete = (*op.op)
                    .as_any()
                    .downcast_ref::<CublasLt>()
                    .expect("CublasLt instance downcasts")
                    .clone();
                let inputs = op
                    .inputs
                    .iter()
                    .map(|input| input.value.to_string())
                    .collect();
                Some((concrete, inputs))
            }
            _ => None,
        })
        .collect()
}

fn single_pinned(text: &str) -> (CublasLt, Vec<String>) {
    let mut ops = pinned_cublaslt_ops(text);
    assert_eq!(ops.len(), 1, "exactly one CublasLt op in the pinned plan");
    ops.remove(0)
}

// ---------------------------------------------------------------------------
// Fixture 1 — plain 2D canonical form A[m,k],B[k,n]: x[2,4] @ w[4,3], live recorder.
// ---------------------------------------------------------------------------

fn record_plain_2d() -> String {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((4usize, 3usize), DType::F32);
    let _out = x.matmul(w).output();
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

#[test]
fn fixture1_plain_2d_amk_bkn_spec_field_by_field() {
    let text = record_plain_2d();
    let (egraph, secs) = timed_serialize(&text);
    println!("MEASURE fixture1: {} nodes, {secs:.2}s", egraph.nodes.len());

    let (op, inputs) = single_pinned(&text);
    let spec = op.spec.as_ref().expect("spec parses");

    // Swapped-COL (R9): descA = w[4,3] COL view = 3x4 ld 3 (call A);
    // descB = x[2,4] COL view = 4x2 ld 4; D = out[2,3] COL view = 3x2 ld 3.
    // Call frame: m = logical n = 3, n = logical m = 2, k = 4.
    assert_eq!(spec.m, 3, "call m = logical n");
    assert_eq!(spec.n, 2, "call n = logical m");
    assert_eq!(spec.k, 4);
    assert!(!spec.trans_a, "canonical A[m,k],B[k,n] form: trans_a = N");
    assert!(!spec.trans_b, "canonical A[m,k],B[k,n] form: trans_b = N");
    assert_eq!(spec.lda, 3, "A = w COL view, ld = w storage cols");
    assert_eq!(spec.ldb, 4, "B = x COL view, ld = x storage cols");
    assert_eq!(spec.ldc, 3);
    assert_eq!(spec.ldd, 3);
    assert!(spec.order_col);
    assert!(!spec.has_c);
    assert!(!spec.has_bias);
    assert_eq!(spec.epilogue, CuEpilogue::Default);
    assert!(spec.c_tensor.is_none());
    assert!(spec.bias_tensor.is_none());

    // Lit-arity cross-check: base op reads exactly [a, b].
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
    assert_eq!(inputs.len(), 2);
    assert_eq!(op.operand_name(0), "a");
    assert_eq!(op.operand_name(1), "b");

    // VIEW ADMISSION (round 10): the elected op lives on the sandwich
    // sibling, whose D layout tensor is a transpose VIEW with no
    // BufferTensor of its own; the spec's view walk must ground all three
    // descriptors in real buffers — D through the view to the recorder's
    // boundary buffer.
    assert!(spec.desc_a_buffer.is_some(), "A grounded in w's buffer");
    assert!(spec.desc_b_buffer.is_some(), "B grounded in x's buffer");
    assert!(
        spec.d_buffer.is_some(),
        "D grounded THROUGH the transpose view in the caller's out buffer"
    );
    assert_ne!(
        spec.desc_a_buffer, spec.desc_b_buffer,
        "distinct operand buffers"
    );
    assert_ne!(spec.d_buffer, spec.desc_a_buffer);
    assert_ne!(spec.d_buffer, spec.desc_b_buffer);
}

// ---------------------------------------------------------------------------
// Fixture 2 — the folded-permute spelling A[m,k], B[n,k] on the LIVE
// recorder: x.matmul(w.permute((1,0))),
// Round-1 algebra minted ZERO here; the marker must mint one, with
// trans_a = T.
// ---------------------------------------------------------------------------

fn record_amk_bnk() -> String {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((3usize, 4usize), DType::F32); // stored [n, k]
    let _out = x.matmul(w.permute((1usize, 0usize))).output();
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

#[test]
fn fixture2_amk_bnk_live_recorder_mints() {
    let text = record_amk_bnk();
    let (egraph, secs) = timed_serialize(&text);
    println!("MEASURE fixture2: {} nodes, {secs:.2}s", egraph.nodes.len());

    let (op, inputs) = single_pinned(&text);
    let spec = op.spec.as_ref().expect("spec parses");

    // w' stored [3,4] = [n,k]; COL view = 4x3 ld 4; trans_a = T makes
    // op(A') = 3x4 = call-m x k. x[2,4] COL view = 4x2 ld 4, trans_b = N.
    // D = out[2,3] COL view = 3x2 ld 3.
    assert_eq!(spec.m, 3);
    assert_eq!(spec.n, 2);
    assert_eq!(spec.k, 4);
    assert!(
        spec.trans_a,
        "folded-permute A[m,k],B[n,k] form: trans_a = T"
    );
    assert!(
        !spec.trans_b,
        "folded-permute A[m,k],B[n,k] form: trans_b = N"
    );
    assert_eq!(spec.lda, 4, "A = w' COL view, ld = w' storage cols = k");
    assert_eq!(spec.ldb, 4);
    assert_eq!(spec.ldd, 3);
    assert_eq!(spec.epilogue, CuEpilogue::Default);
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
}

/// Square-weight folded-permute form (k = n = 4, b stored [n,k]): logical
/// SHAPES alone cannot distinguish
/// [k,n] from [n,k] here — only the map entry permutation can. Exactly one
/// A reading (the map disambiguates); a shape-derived design would have to
/// guess.
///
/// ROUND 9 (2026-08-25): the assertions below are UNCHANGED and still
/// hold, but the mechanism that satisfies them moved. Round 5 read the
/// operation off the map's ENTRY ORDER. Round 9 reads it off the operand
/// layout PRE-COMPOSED WITH that map: the composed broadcast chain of a
/// right-major [n,k] w has its unit stride on the k axis, so the reading is
/// T. The map is still what makes the square case decidable — it is the
/// composition key — but it is bound opaquely and never destructured, so a
/// column-major w would flip the conclusion instead of silently inverting
/// it. See src/egg/cublaslt_marker_desc.egg.
#[test]
fn fixture2b_square_amk_bnk_single_reading() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _out = x.matmul(w.permute((1usize, 0usize))).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let serialized = test_runtime::serialize_fixture(&text);
    let a_readings = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
        .count();
    // ROUND-11 RE-PIN (was 2, R10; was 1, R9): canonicalization + the
    // canonical-form sandwich give TWO sites (canonicalized original +
    // sibling), and each site's a operand now carries TWO readable layout
    // tensors, so 4 A readings total:
    //   site1 a = x:        (x right-major, T) and (x column-form, N) —
    //     the column-form layout is x re-derived THROUGH the collapse
    //     union x = (x^T)^T as a view of x^T's fresh right-major
    //     materialization frame (the preamble mints a fresh right-major
    //     layout tensor for every value, including the transpose views);
    //   site2 a = w (the collapse folds (w^T)^T back into w):
    //     (w right-major, T) and (w column-form, N), same mechanism.
    // Each reading is tied to ITS OWN layout tensor by the descriptor
    // term (bounded, per-candidate-consistent multiplicity — election
    // picks; the N readings imply a materialize-first plan the strict
    // election never prefers). The square stays disambiguated: no
    // cross-site leak, each site's map is its own composition key.
    assert_eq!(
        a_readings, 4,
        "square A[m,k],B[n,k]: two readings per site (own layout + collapse-derived column form)"
    );
    let (op, _) = single_pinned(&text);
    let spec = op.spec.as_ref().expect("spec parses");
    // The elected op is the SIBLING site's; its A operand is the stored w
    // read through the re-indexed map, whose composed layout has the unit
    // stride on k — still T, still read from the layout, never the map.
    assert!(
        spec.trans_a,
        "square folded-permute form reads T from the composed layout"
    );
    assert_eq!(spec.mnk_lits(), (4, 2, 4));
}

// ---------------------------------------------------------------------------
// Fixture 3 — TWO same-shape matmuls (the Q/K case): two DISTINCT ops.
// The R1 regression: round-1 witness ABORTED here.
// ---------------------------------------------------------------------------

fn record_two_same_shape_matmuls() -> String {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let wq = cx.tensor((4usize, 3usize), DType::F32);
    let wk = cx.tensor((4usize, 3usize), DType::F32);
    let _q = x.matmul(wq).output();
    let _k = x.matmul(wk).output();
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

#[test]
fn fixture3_two_same_shape_matmuls_two_distinct_ops() {
    let text = record_two_same_shape_matmuls();
    let (egraph, secs) = timed_serialize(&text);
    let sites = egraph
        .nodes
        .values()
        .filter(|node| node.op == "CublasLtLogicalMatmulSite")
        .count();
    println!(
        "MEASURE fixture3: {} nodes, {secs:.2}s, {sites} CublasLtLogicalMatmulSite enodes",
        egraph.nodes.len()
    );
    // ROUND-10 RE-PIN (was 2): every matmul now carries TWO sites — the
    // recorder's and the transpose-sandwich sibling the rewrite mints.
    assert_eq!(
        sites, 4,
        "two matmuls, two marker sites each (original + sibling)"
    );

    let ops = pinned_cublaslt_ops(&text);
    assert_eq!(ops.len(), 2, "two DISTINCT cublaslt ops in one plan");
    let (op0, in0) = &ops[0];
    let (op1, in1) = &ops[1];
    let s0 = op0.spec.as_ref().expect("spec 0 parses");
    let s1 = op1.spec.as_ref().expect("spec 1 parses");
    // Same geometry...
    assert_eq!(s0.mnk_lits(), (3, 2, 4));
    assert_eq!(s1.mnk_lits(), (3, 2, 4));
    // ...but distinct identities. ROUND-10 RE-PIN: the elected ops live on
    // the SIBLING sites, whose a operand is the recorder matmul's weight
    // and whose b operand is the shared x — the shared/distinct pattern
    // swaps namespaces accordingly (the swap now lives in the logical
    // rewrite, not the descriptor wiring).
    assert_ne!(
        s0.logical_a, s1.logical_a,
        "distinct weights (sibling a role)"
    );
    assert_eq!(
        s0.logical_b, s1.logical_b,
        "both matmuls share x (sibling b role)"
    );
    assert_ne!(s0.logical_out, s1.logical_out, "distinct outputs");
    assert_ne!(in0, in1, "distinct Lit input lists");
}
