//! ROUND-9 FOLLOW-UP PROBE — is the chain-born gap actually two gaps?
//!
//! gap_probe.rs shows a layout born as a bare affine CHAIN never climbs to
//! a BitOffsetExpressionLayoutLit spelling, so it cannot be a composition
//! parent. But the source-seam doctrine (egglog_preamble.egg:2981-2987)
//! names "fixture authoring" as a sanctioned strided-lists WRITE SITE — and
//! a CREATOR-minted pitched layout's creator has the (dims, strides) lists
//! in hand at mint time. This probe authors the SAME pitched geometry via a
//! `strided-lists` row instead of a bare chain and asks whether the full
//! ladder appears.
//!
//! If YES: the creator-pitched population is unblocked by a fixture/creator
//! convention change (no preamble edit); only genuinely list-less layouts
//! (chain-walk-composed views used as parents) need the chain-walk seed fix.
const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

#[test]
fn strided_lists_authored_pitch_climbs_the_ladder() {
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_static (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_static_lt (LayoutTensorLit x_logical x_static))
(set (injectivity-of x_static_lt) (Injective))
(strided-lists x_static a_shape
  (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let mut pitched_class_spellings: Option<std::collections::BTreeSet<String>> = None;
    for n in s
        .nodes
        .values()
        .filter(|n| n.op == "StridedElementLayoutLit")
    {
        let ops: std::collections::BTreeSet<String> = s
            .nodes
            .values()
            .filter(|m| m.eclass == n.eclass)
            .map(|m| m.op.clone())
            .collect();
        println!("class {:?} spellings {:?}", n.eclass, ops);
        // The pitched class is the one that is NOT right-major contiguous.
        if !ops.contains("RightMajorContiguousElementLayoutLit") {
            pitched_class_spellings = Some(ops);
        }
    }
    let ops = pitched_class_spellings.expect("pitched StridedElementLayoutLit class must exist");
    assert!(
        ops.contains("BitOffsetExpressionLayoutLit"),
        "strided-lists-authored pitched layout did not climb to BitOffset: {ops:?}"
    );
    assert!(
        ops.contains("ElementOffsetExpressionLayoutLit"),
        "missing ElementOffset rung: {ops:?}"
    );
}
