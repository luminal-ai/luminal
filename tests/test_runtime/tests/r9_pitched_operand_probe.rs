//! ROUND-9: what the layout-native reading arms can do that the round-5
//! (map spelling x layout constructor) arms could not.
//!
//! THE FIXTURE. `w_parent` is a [3,8] right-major weight. `w` is a COLUMN
//! SLICE of it — the first 4 columns — expressed as an index-map view, not
//! a copy. Its composed layout is therefore ROW-PITCHED: shape [3,4],
//! chain [8*c1, c0], pitch 8 > 4 columns. Then x[2,3] @ w[3,4] -> [2,4].
//!
//! Round 5 had FOUR A arms, and every one of them required the operand's
//! layout to be `RightMajorContiguousElementLayoutLit` or
//! `LeftMajorContiguousElementLayoutLit`. A pitched b matched none of
//! them: the whole matmul was silently refused. (Grep-checkable against
//! the archived round-5 desc.egg: `StridedElementLayoutLit` appears twice
//! in the whole marker estate, on the *a* side and the output, never on
//! the A side.)
//!
//! Round 9 has no layout-constructor premise at all. The operation is the
//! composed broadcast chain's unit-stride axis, so the pitched operand is
//! read by the SAME arm as the contiguous one — no new arm, no new case.
//!
//! WHAT IS STILL BLOCKED, and it is one fact: `injectivity-of`. The
//! preamble proves it for the two contiguous constructors
//! (egglog_preamble.egg:4217-4238) and nothing else — the STRIDED
//! no-overlap fold was deleted on 2026-08-05 for being order-sensitive
//! (:4240-4250: "a transposed contiguous layout is a bijection it could
//! not see"). So a view-born pitched layout has no injectivity fact, and
//! the round-9 arms fail closed on it. `pitched_view_refuses_without_an
//! _injectivity_fact` pins that; `pitched_view_reads_n_with_the_creator
//! _certificate` pins that the fact is the ONLY thing missing.

use luminal::layout_ir::ExtractedNode;
use luminal::prelude::egraph_serialize::EGraph;
use test_runtime::cublaslt_marker::CublasLt;

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

/// x[2,3] @ w[3,4], where w is a column-slice VIEW of a [3,8] parent.
/// `certificate` supplies the creator's injectivity assertion (or not).
fn pitched_view_fixture(certificate: bool) -> String {
    let cert = if certificate {
        "(set (injectivity-of w_lt) (Injective))"
    } else {
        ""
    };
    format!(
        r#"(let x_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let parent_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 8) (IntExprNil)))))
(let w_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") x_shape (F32)))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "w_parent") parent_shape (F32)))
; w = the first four COLUMNS of the [3,8] parent, as a view.
(let slice_map
  (IndexMapLit
    (IntExprCons (CoordVar w_shape 1)
      (IntExprCons (CoordVar w_shape 0) (IntExprNil)))
    parent_shape))
(let w_logical (LogicalIndexMapApply parent_logical slice_map w_shape))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    x_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    w_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit x_shape (bits-of (F32))))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
; The view's own layout, named by its CHAIN: row pitch 8, unit inner
; stride. Hash-conses onto the class the composition walk derives.
(let w_layout (StridedElementLayoutLit w_shape
  (IntAffineExprCons (IntMul (CoordVar w_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar w_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
{cert}
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
(let parent_buffer_tensor (BufferTensorLit parent_lt w_buffer_id))
; the view ALIASES the parent's buffer — no copy.
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    )
}

/// A readings whose layout tensor's class carries a Strided spelling but NO
/// contiguous one — i.e. readings over the PITCHED view itself, not over
/// the materialising copy the preamble also offers.
fn a_readings_over_the_pitched_view(s: &EGraph) -> Vec<&'static str> {
    let class_has = |class: &luminal::prelude::egraph_serialize::ClassId, op: &str| {
        s.nodes.values().any(|n| &n.eclass == class && n.op == op)
    };
    let mut out = Vec::new();
    for n in s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
    {
        let Some(lt) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        let lt_class = lt.eclass.clone();
        let Some(layout) = s.nodes.values().find_map(|m| {
            (m.eclass == lt_class && m.op == "LayoutTensorLit")
                .then(|| {
                    m.children
                        .get(1)
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| c.eclass.clone())
                })
                .flatten()
        }) else {
            continue;
        };
        if !class_has(&layout, "StridedElementLayoutLit")
            || class_has(&layout, "RightMajorContiguousElementLayoutLit")
            || class_has(&layout, "LeftMajorContiguousElementLayoutLit")
        {
            continue;
        }
        let op = match n
            .children
            .get(2)
            .and_then(|id| s.nodes.get(id))
            .map(|c| c.eclass.clone())
        {
            Some(c) if class_has(&c, "CublasLtOperationT") => "T",
            Some(c) if class_has(&c, "CublasLtOperationN") => "N",
            _ => "?",
        };
        out.push(op);
    }
    out
}

#[test]
fn pitched_view_refuses_without_an_injectivity_fact() {
    let s = test_runtime::serialize_fixture(&pitched_view_fixture(false));
    let readings = a_readings_over_the_pitched_view(&s);
    println!("pitched view, NO certificate: A readings over the view = {readings:?}");
    assert!(
        readings.is_empty(),
        "fail-closed: no injectivity fact, no reading (the ld >= rows clamp is unproven)"
    );
}

#[test]
fn pitched_view_reads_n_with_the_creator_certificate() {
    let s = test_runtime::serialize_fixture(&pitched_view_fixture(true));
    let readings = a_readings_over_the_pitched_view(&s);
    println!("pitched view, certificate asserted: A readings over the view = {readings:?}");
    assert_eq!(
        readings,
        vec!["N"],
        "the column-slice view is read by the SAME arm as a contiguous b: its \
         composed broadcast chain is [0, c1, 8*c0], unit stride on n, so op = N"
    );

    // TWO candidates coexist: the zero-copy pitched view, and the
    // materialising contiguous copy the preamble always offers. MEASURED:
    // with this geometry the extractor ELECTS THE COPY (lda = 4, not 8) --
    // 12 elements is cheaper than nothing under a bytes-moved cost model
    // that does not yet charge for the copy's own kernel. So this probe
    // pins the READING, not the election; whether a pitched operand is
    // ever preferred is a cost-model question, not a marker question.
    let ops = s
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count();
    println!("  op candidates: {ops} (the view, and the materialising copy)");
    assert!(ops >= 1, "the pitched reading assembles into a candidate");

    let (graph, _) = test_runtime::extract_fixture_with_genome(
        &pitched_view_fixture(true),
        &["LayoutTensorOpCublasLt"],
    );
    let specs: Vec<_> = graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => (*op.op)
                .as_any()
                .downcast_ref::<CublasLt>()
                .and_then(|c| c.spec.clone()),
            _ => None,
        })
        .collect();
    for spec in &specs {
        println!(
            "  spec: m={} n={} k={} trans_a={} lda={} ldb={} ldd={}",
            spec.m, spec.n, spec.k, spec.trans_a, spec.lda, spec.ldb, spec.ldd
        );
    }
}
