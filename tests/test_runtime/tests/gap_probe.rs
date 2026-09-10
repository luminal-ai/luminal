//! ROUND-9 GAP PROBE — the exact mechanism that stops the layout-native
//! reading from covering PITCHED operands.
//!
//! A hand-declared (creator-minted, bucket-padded) StridedElementLayoutLit
//! never acquires a BitOffsetExpressionLayoutLit spelling. The upward
//! ladder Strided -> ElementOffset -> BitOffset is seeded by the
//! `strided-lists` provenance relation (egglog_preamble.egg:2988), which is
//! written ONLY at the two contiguous fold construction sites (:3032
//! right-major, :3121 left-major) — i.e. only for layouts born from a
//! stride LIST. A layout born as a bare affine CHAIN gets no record.
//!
//! Both index-map/layout composition entry points require that spelling
//! (:3865, :3882 and the native chain-walk seed :4076, :4090), so a pitched
//! operand's broadcast has NO composed layout, and the layout-native arms
//! have nothing to read. That is why one map-spelling B arm survives.
const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

#[test]
fn raw_strided_never_climbs_to_bit_offset() {
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_static (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_static_lt (LayoutTensorLit x_logical x_static))
(set (injectivity-of x_static_lt) (Injective))
(let x_contig (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let x_contig_lt (LayoutTensorLit x_logical x_contig))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    for target in ["StridedElementLayoutLit"] {
        let mut seen: Vec<luminal::prelude::egraph_serialize::ClassId> = Vec::new();
        for n in s.nodes.values().filter(|n| n.op == target) {
            if seen.contains(&n.eclass) {
                continue;
            }
            seen.push(n.eclass.clone());
            let ops: std::collections::BTreeSet<String> = s
                .nodes
                .values()
                .filter(|m| m.eclass == n.eclass)
                .map(|m| m.op.clone())
                .collect();
            println!("class {:?} spellings {:?}", n.eclass, ops);
        }
    }
}
