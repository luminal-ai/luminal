//! Austin's hypothesis regressions (the site-keyed-function hazard) plus
//! the round-6 Ruling-1 regression.
//!
//! Round-2 recorded behavior of these exact fixtures (site-keyed :no-merge
//! functions carrying layout/spelling data):
//!   two-layout + left-major arm:
//!     "Panic: Illegal merge attempted for function cu-desc-a-of"
//!   dual-spelling:
//!     "Panic: Illegal merge attempted for function cu-trans-a-of"
//! With descriptor TERMS the same programs are legal multiplicity.

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

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

fn count_op(egraph: &luminal::prelude::egraph_serialize::EGraph, op: &str) -> usize {
    egraph.nodes.values().filter(|n| n.op == op).count()
}

fn count_cublaslt(egraph: &luminal::prelude::egraph_serialize::EGraph) -> usize {
    egraph
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count()
}

/// (role, site) pairs whose readings include BOTH an N and a T
/// orientation — over different layout tensors of the operand, since a
/// layout has one unit axis. The multiplicity the descriptor TERMS exist
/// to hold — a site-keyed function had to pick one and panicked.
fn sites_read_both_ways(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
) -> Vec<(&'static str, luminal::prelude::egraph_serialize::ClassId)> {
    use std::collections::BTreeMap;
    let mut ops: BTreeMap<
        (&'static str, luminal::prelude::egraph_serialize::ClassId),
        Vec<String>,
    > = BTreeMap::new();
    for (role, ctor) in [
        ("A", "CublasLtOperandADescriptor"),
        ("B", "CublasLtOperandBDescriptor"),
    ] {
        for node in egraph.nodes.values().filter(|n| n.op == ctor) {
            let (Some(site), Some(op_class)) = (
                node.children
                    .first()
                    .and_then(|id| egraph.nodes.get(id))
                    .map(|c| c.eclass.clone()),
                node.children
                    .get(2)
                    .and_then(|id| egraph.nodes.get(id))
                    .map(|c| c.eclass.clone()),
            ) else {
                continue;
            };
            let Some(op) = egraph
                .nodes
                .values()
                .find(|m| m.eclass == op_class && m.op.starts_with("CublasLtOperation"))
                .map(|m| m.op.clone())
            else {
                continue;
            };
            ops.entry((role, site)).or_default().push(op);
        }
    }
    ops.into_iter()
        .filter(|(_, seen)| {
            seen.iter().any(|o| o == "CublasLtOperationN")
                && seen.iter().any(|o| o == "CublasLtOperationT")
        })
        .map(|(key, _)| key)
        .collect()
}

fn pinned_cublaslt(text: &str) -> Vec<test_runtime::cublaslt_marker::CublasLt> {
    use luminal::layout_ir::ExtractedNode;
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, PIN);
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => (*op.op)
                .as_any()
                .downcast_ref::<test_runtime::cublaslt_marker::CublasLt>()
                .cloned(),
            _ => None,
        })
        .collect()
}

/// 2D A[m,k],B[k,n] matmul skeleton, x[2,4] @ w[4,3], with w carrying BOTH a
/// right-major-contiguous AND a left-major-contiguous LayoutTensorLit —
/// two valid layout readings of one logical operand.
fn two_layout_fixture() -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout_rm (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let w_layout_lm (LeftMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt_rm (LayoutTensorLit w_logical w_layout_rm))
(let w_lt_lm (LayoutTensorLit w_logical w_layout_lm))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_rm_buffer_id (BufferLit 11))
(set (buffer-access-of w_rm_buffer_id) (ReadOnly))
(set (buffer-freed-by w_rm_buffer_id) (CallerFrees))
(let w_lm_buffer_id (BufferLit 12))
(set (buffer-access-of w_lm_buffer_id) (ReadOnly))
(set (buffer-freed-by w_lm_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 13))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_rm_buffer_tensor (BufferTensorLit w_lt_rm w_rm_buffer_id))
(let w_lm_buffer_tensor (BufferTensorLit w_lt_lm w_lm_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output
  (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    )
}

/// Dual-spelling: the SAME square matmul x[2,4] @ w[4,4] spelled both A[m,k],B[k,n]
/// and A[m,k],B[n,k], the two logical outs unioned.
fn dual_spelling_fixture() -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_kn_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let w_nk_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 1)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_kn_applied (LogicalIndexMapApply w_logical w_kn_map prod_shape))
(let w_nk_applied (LogicalIndexMapApply w_logical w_nk_map prod_shape))
(let out_amk_bkn (LogicalReduceSum (LogicalMul x_applied w_kn_applied) 0))
(let out_amk_bnk (LogicalReduceSum (LogicalMul x_applied w_nk_applied) 0))
(union out_amk_bkn out_amk_bnk)
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_amk_bkn out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output
  (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    )
}

/// Two coexisting layouts of one operand => two A readings, two op
/// candidates, election picks one, plan sound.
#[test]
fn hypothesis_two_layout_multiplicity() {
    let fx = two_layout_fixture();
    let s = test_runtime::serialize_fixture(&fx); // round 2 (+LM arm): PANIC
    let a_readings = count_op(&s, "CublasLtOperandADescriptor");
    let op_enodes = count_cublaslt(&s);
    println!(
        "two-layout: {} nodes, {a_readings} A reading(s), {op_enodes} op enode(s)",
        s.nodes.len()
    );
    // ROUND-11 RE-PIN: every site operand carries TWO readable layout
    // tensors (storage frame + collapse-derived column-form frame — the
    // r8d probe pins the mechanism), so readings and candidate products
    // scale accordingly; each reading stays tied to its own layout
    // tensor (per-candidate soundness swept below / in the extractor).
    // The previously "extra" seeded LM layout hash-conses with the
    // collapse-derived column-form frame, so the census is the uniform
    // two-frames-per-operand one.
    assert_eq!(a_readings, 4, "two frames per site's a operand");
    assert_eq!(op_enodes, 8, "the 2 A x 2 B frame cross product per site");

    let ops = pinned_cublaslt(&fx);
    assert_eq!(ops.len(), 1, "election picks exactly one reading");
    let spec = ops[0].spec.as_ref().expect("spec parses");
    println!("  elected: trans_a={} lda={}", spec.trans_a, spec.lda);
    // w is [4,3]: the RM reading is (N, lda=3), the LM reading is (T, lda=4).
    assert!(
        (!spec.trans_a && spec.lda == 3) || (spec.trans_a && spec.lda == 4),
        "elected reading is one of the two coherent ones"
    );
    assert_eq!(spec.mnk_lits(), (3, 2, 4));
}

/// Two coexisting spellings (A[m,k],B[k,n] and A[m,k],B[n,k] of one square matmul) => two A
/// readings over the SAME layout tensor, two op spellings in one class.
#[test]
fn hypothesis_dual_spelling_multiplicity() {
    let fx = dual_spelling_fixture();
    let s = test_runtime::serialize_fixture(&fx); // round 2: PANIC (cu-trans-a-of)
    println!(
        "dual-spelling: {} nodes, {} site(s), {} A reading(s), {} op enode(s)",
        s.nodes.len(),
        count_op(&s, "CublasLtLogicalMatmulSite"),
        count_op(&s, "CublasLtOperandADescriptor"),
        count_cublaslt(&s)
    );
    // THE FACT: a site carries BOTH an N and a T reading of an operand,
    // and both survive as descriptor terms. A site-keyed function holding
    // the orientation had to merge the two and panicked here (round 2).
    // How many sites, readings and op spellings the estate mints around
    // that is not this test's subject.
    let both_ways = sites_read_both_ways(&s);
    println!("  read both ways: {both_ways:?}");
    assert!(
        !both_ways.is_empty(),
        "some site must carry both an N and a T reading of an operand"
    );

    let ops = pinned_cublaslt(&fx);
    assert_eq!(ops.len(), 1, "election picks exactly one reading");
    let spec = ops[0].spec.as_ref().expect("spec parses");
    println!(
        "  elected: trans_a={} (both sound under the seeded union)",
        spec.trans_a
    );
    assert_eq!(spec.mnk_lits(), (4, 2, 4));
    assert!(!spec.trans_b);
}

// ---------------------------------------------------------------------------
// The split-sources / bucketing proof fixture. x[2,s] @ w[s,3], s in [2,8]:
// x is padded to a static row pitch of 8 by the CREATOR REWRITE (driven by
// the estate-shaped bucket request), which asserts both creator facts.
// ---------------------------------------------------------------------------

fn static_bucket_fixture() -> String {
    format!(
        r#"(let s_var (IntVar "s"))
(set (lower-bound-of s_var) (bigint 2))
(set (upper-bound-of s_var) (bigint 8))
(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons s_var (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons s_var (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout_contig (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt_contig (LayoutTensorLit x_logical x_layout_contig))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_static (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_static_lt (LayoutTensorLit x_logical x_static))
(set (injectivity-of x_static_lt) (Injective))
(strided-lists x_static a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))
(let xc_buffer_id (BufferLit 10))
(set (buffer-access-of xc_buffer_id) (ReadOnly))
(set (buffer-freed-by xc_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 12))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 13))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let xc_buffer_tensor (BufferTensorLit x_lt_contig xc_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    )
}

#[test]
fn hypothesis_static_bucket_layout_multiplicity() {
    let fx = static_bucket_fixture();
    let s = test_runtime::serialize_fixture(&fx);
    let a_readings = count_op(&s, "CublasLtOperandADescriptor");
    let b_readings = count_op(&s, "CublasLtOperandBDescriptor");
    let d_readings = count_op(&s, "CublasLtOutputDDescriptor");
    let op_enodes = count_cublaslt(&s);
    println!(
        "static-bucket: {} nodes, readings a={a_readings} b={b_readings} d={d_readings}, {op_enodes} op enode(s)",
        s.nodes.len()
    );
    // ROUND-10 RE-PIN (was a=1 b=2 d=1 ops=2). The pitched x layout now
    // AUTHORS its strided-lists row (the creator has the lists in hand —
    // the sanctioned write site), so it climbs the ladder and the NATIVE
    // arms read it; the round-9 map-spelling fallback arm is DELETED.
    //   a=3: sibling w (N) + outer x contiguous (T) + outer x pitched (T).
    //   b=3: sibling x contiguous (N, ldb symbolic) + sibling x pitched
    //        (N, ldb=8 — RULING 1 multiplicity, election decides) + outer
    //        w (T).
    //   d=2: sibling composed view + the outer blanket-transposed route.
    //   ops=4: sibling A(1) x B(2) x D(1) = 2, outer A(2) x B(1) x D(1) = 2.
    // ROUND-11 RE-PIN: every site operand carries TWO readable layout
    // tensors (storage frame + collapse-derived column-form frame — the
    // r8d probe pins the mechanism), so readings and candidate products
    // scale accordingly; each reading stays tied to its own layout
    // tensor (per-candidate soundness swept below / in the extractor).
    // a=5: outer x {contiguous T, pitched T, column-form N} + sibling
    //      view(w) {composed N, materialized-frame T}.
    // b=5: sibling view(x) {composed-from-contiguous N (symbolic ldb),
    //      composed-from-pitched N (ldb=8), materialized-frame T} + outer
    //      w {storage T, column-form N}.
    // d=2: the two transpose-tie views, as in round 10.
    assert_eq!(
        a_readings, 5,
        "outer x three ways + sibling view(w) two frames"
    );
    assert_eq!(
        b_readings, 5,
        "sibling view(x) three ways + outer w two frames"
    );
    assert_eq!(d_readings, 2);
    assert_eq!(op_enodes, 12, "the per-site products: 3x2x1 + 2x3x1");

    let ops = pinned_cublaslt(&fx);
    assert_eq!(ops.len(), 1);
    let spec = ops[0].spec.as_ref().expect("spec parses WITH symbolic k");
    println!(
        "  elected: m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    use test_runtime::cublaslt_marker::CuDim;
    assert_eq!(spec.m, 3);
    assert_eq!(spec.n, 2);
    assert!(
        matches!(spec.k, CuDim::Symbolic(_)),
        "k rides symbolic; the executor binds it from the dyn map"
    );
    assert!(
        spec.ldb == 8 || matches!(spec.ldb, CuDim::Symbolic(_)),
        "elected B ld is the bucket pitch or the symbolic contiguous extent, got {}",
        spec.ldb
    );
    assert_eq!(spec.lda, 3);
    assert_eq!(spec.ldd, 3);
    assert!(!spec.trans_a && !spec.trans_b);
}

// ---------------------------------------------------------------------------
// Round-6 RULING 1 regression: a fully-symbolic-k CONTIGUOUS matmul with
// NO bucketed layout anywhere. Rounds 3-5 refused this (the literal ld
// gate); the refusal pinned conservatism, not soundness.
// ---------------------------------------------------------------------------
#[test]
fn ruling1_symbolic_k_contiguous_mints() {
    let fx = format!(
        r#"(let s_var (IntVar "s"))
(set (lower-bound-of s_var) (bigint 2))
(set (upper-bound-of s_var) (bigint 8))
(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons s_var (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons s_var (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let ops = count_cublaslt(&s);
    println!("ruling1 symbolic-k contiguous: {ops} op enode(s)");
    // ROUND-11 RE-PIN (was 2): the 2 A x 2 B frame cross product on each
    // of the two sites.
    assert_eq!(
        ops, 8,
        "the pure-contiguous symbolic-k matmul mints (frame products)"
    );
    let elected = pinned_cublaslt(&fx);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec.as_ref().expect("spec parses — never None");
    use test_runtime::cublaslt_marker::CuDim;
    println!(
        "  m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(spec.m, 3);
    assert_eq!(spec.n, 2);
    assert!(matches!(spec.k, CuDim::Symbolic(_)));
    assert_eq!(spec.lda, 3, "w storage cols literal");
    assert!(
        matches!(spec.ldb, CuDim::Symbolic(_)),
        "ldb = Symbolic(k), bound at call time"
    );
    assert_eq!(spec.ldd, 3);
    assert!(!spec.trans_a && !spec.trans_b);
}
