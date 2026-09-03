//! ADVERSARIAL fixtures for the MARKER design.
//!
//! Fixture 4: the round-1 mirror FATAL — sliced sources (storage k=7,
//! product k=3). The marker's logical-shape unification must REFUSE the
//! site; the decomposed route must survive.
//!
//! Fixture 7: degenerate extents (m=1, the extent-1 pointing weld, and
//! the all-ones corner) — must not panic; readings stay bounded and
//! every elected spec is sound.

use luminal::dtype::DType;
use luminal::layout_ir::ExtractedNode;
use test_runtime::cublaslt_marker::CublasLt;

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

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

/// Parameterized handwritten 2D matmul skeleton (ported from the round-1
/// adversarial estate). Geometry args are raw IntExpr text so storage
/// extents can deliberately mismatch the product coordinates.
fn matmul_2d(
    m: &str,
    n: &str,
    k_prod: &str,
    a_rows: &str,
    a_cols: &str,
    b_rows: &str,
    b_cols: &str,
) -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons {a_rows} (IntExprCons {a_cols} (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons {b_rows} (IntExprCons {b_cols} (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprCons {k_prod} (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprNil)))))
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
(let output
  (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
"#
    )
}

fn cublaslt_ops(graph: &luminal::layout_ir::ExtractedGraph) -> Vec<CublasLt> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                (*op.op).as_any().downcast_ref::<CublasLt>().cloned()
            }
            _ => None,
        })
        .collect()
}

// ===========================================================================
// Fixture 4 — sliced-source counterexample. product (m n k) = (2, 4, 3),
// A stored [2,7] (col slice), B stored [7,4] (row slice). Round-1 mirror
// silently minted k=7. The marker must refuse: storage k=7 does not unify
// with product k=3.
// ===========================================================================
#[test]
fn fixture4_sliced_source_refused() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 2)",
            "(IntLit 4)",
            "(IntLit 3)",
            "(IntLit 2)",
            "(IntLit 7)",
            "(IntLit 7)",
            "(IntLit 4)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    let ops = serialized
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count();
    println!("ADVERSARIAL sliced-source: {sites} sites, {ops} cublaslt op enodes");
    assert_eq!(sites, 0, "logical-shape unification refuses the marker");
    assert_eq!(ops, 0, "no site, no op");

    // The decomposed route survives: extraction still reaches the boundary
    // even with the genome asking for cublaslt everywhere.
    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert!(ops.is_empty(), "no cublaslt candidate exists to elect");
    let computes = graph
        .dag
        .node_weights()
        .filter(|node| matches!(node, ExtractedNode::LayoutOp(_)))
        .count();
    println!("  decomposed plan has {computes} compute nodes");
    assert!(computes > 0, "decomposed route reaches the boundary");
}

// ===========================================================================
// Fixture 7 — degenerate extents must not panic; readings stay bounded.
// ===========================================================================

/// m=1 (live-recorder geometry x[1,4] @ w[4,3]).
#[test]
fn fixture7_m1_degenerate_no_panic() {
    use luminal::graph::Graph;
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((1usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _out = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    // serialize_fixture panics if saturation panics.
    let serialized = test_runtime::serialize_fixture(&text);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    println!(
        "fixture7 m=1: {} nodes, {sites} site(s)",
        serialized.nodes.len()
    );
    // ROUND-10 RE-PIN (was 1): original + transpose-sandwich sibling.
    assert_eq!(
        sites, 2,
        "the degenerate matmul carries original + sibling sites"
    );

    let (graph, _) = test_runtime::extract_fixture_with_genome(&text, PIN);
    let ops = cublaslt_ops(&graph);
    assert_eq!(ops.len(), 1);
    let spec = ops[0].spec.as_ref().expect("spec parses");
    println!(
        "  m={} n={} k={} trans_a={} trans_b={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.trans_a, spec.trans_b, spec.lda, spec.ldb, spec.ldd
    );
    // ROUND-10 RE-PIN (was (3,1,4), the round-9 swapped frame): at m=1 a
    // right-major [1,n] out IS column-major 1 x n (ld=1), so the ORIGINAL
    // site presents a legal unswapped call directly, claims the boundary
    // tensor itself (no view hop), and wins the election over the sibling.
    // Both frames are sound spellings of the same GEMV; the elected one is
    // the direct (1, n, k) call.
    assert_eq!(spec.mnk_lits(), (1, 3, 4));
    assert!(!spec.trans_a, "direct call: A = x[1,4] COL 1x4 ld 1, op N");
    assert!(spec.trans_b, "direct call: B = w[4,3] COL 3x4 ld 3, op T");
    assert_eq!(spec.lda, 1);
    assert_eq!(spec.ldb, 3);
    assert_eq!(spec.ldd, 1);
}

/// The extent-1 pointing-weld geometry: m=1, k=1, n=3 (A [1,1], B [1,3]).
/// Round 1 aborted here; round 2 forced a canonical single reading to
/// protect its :no-merge functions. Descriptor TERMS make the structural
/// same-extent weld legal multiplicity: both readings of the same bytes
/// mint, the election picks either, and the parser's cross-checks hold.
#[test]
fn fixture7_extent1_weld_single_canonical_reading() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 1)",
            "(IntLit 3)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 3)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    let a_readings = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
        .count();
    println!("fixture7 weld: {sites} site(s), {a_readings} A reading(s)");
    // ROUND-10 RE-PIN (was 1): original + transpose-sandwich sibling(s).
    // The weld can congruence several sibling spellings into the same
    // term; what matters is boundedness and that every elected spec is
    // sound (checked below).
    assert!(
        (2..=4).contains(&sites),
        "bounded sites despite the coordinate weld: {sites}"
    );
    assert!(a_readings >= 1, "the welded bytes are read at least once");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert_eq!(ops.len(), 1, "election picks one reading");
    let spec = ops[0]
        .spec
        .as_ref()
        .expect("spec parses and cross-validates");
    println!(
        "  m={} n={} k={} trans_a={} trans_b={}",
        spec.m, spec.n, spec.k, spec.trans_a, spec.trans_b
    );
    // ROUND-10 RE-PIN (was (3,1,1)): same direct-call election as the m=1
    // fixture above — at m=1 the original site is COL-presenting and wins.
    assert_eq!(spec.mnk_lits(), (1, 3, 1));
}

/// All-ones corner m=n=k=1: LogicalMul commutativity can present the role
/// swap and coordinate welds make every map pattern congruent — observe
/// what the marker does. Duplicate SITES are acceptable (each is a sound
/// 1x1 call); a panic or a wrong descriptor is not.
#[test]
fn fixture7_all_ones_corner_observed() {
    let fx = format!(
        "{}{SCHEDULE}\n",
        matmul_2d(
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
            "(IntLit 1)",
        )
    );
    let serialized = test_runtime::serialize_fixture(&fx);
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();
    println!("fixture7 all-ones: {sites} site(s)");
    assert!(sites >= 1, "the 1x1x1 matmul still marks");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let ops = cublaslt_ops(&graph);
    assert_eq!(ops.len(), 1, "one op elected at the boundary");
    let spec = ops[0].spec.as_ref().expect("spec parses");
    assert_eq!(spec.mnk_lits(), (1, 1, 1));
    assert_eq!(spec.lda, 1);
    assert_eq!(spec.ldb, 1);
    assert_eq!(spec.ldd, 1);
}
