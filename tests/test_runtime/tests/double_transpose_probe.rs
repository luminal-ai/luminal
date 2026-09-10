//! DOUBLE-TRANSPOSE COLLAPSE PROBE (round-10 termination prerequisite,
//! Austin 2026-08-25: "we'll have to make sure that this does cons /
//! terminate. I am relying on that").
//!
//! The transpose-rewrite design terminates iff apply(apply(x, T), T)
//! rejoins x's class, so the rewrite's re-fire on its own output
//! hash-conses shut. HISTORY: before the round-11 marker estate landed in
//! this crate, no preamble rule unioned a nested apply back onto its
//! grandparent, and this probe pinned the ABSENCE of the rejoin as the
//! RED baseline the collapse rule must flip. FLIPPED (by design, marker
//! landing 2026-08-26): `test_runtime::matchers()` now registers the
//! CublasLt marker family, whose snippet
//! `src/egg/cublaslt_marker_canonicalize.egg` carries THE
//! DOUBLE-TRANSPOSE COLLAPSE rule ("the termination anchor (round 11)"
//! — union-only: apply(apply(x, T2), T1) with mutually inverse rank-2
//! transpose maps unions with x). This probe is the PERMANENT PIN of
//! that anchor: if the collapse rule is ever weakened or dropped, the
//! transpose-sandwich rewrite mints views-of-views without bound and
//! saturation never closes (see tests/r11_collapse_revert_probe.rs).
const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

#[test]
fn double_transpose_rejoins_via_the_collapse_rule() {
    let fx = format!(
        r#"(let x_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let t_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 2) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") x_shape (F32)))
(let x_lt (LayoutTensorLit x (RightMajorContiguousElementLayoutLit x_shape (bits-of (F32)))))
(let t1 (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar t_shape 0)
    (IntExprCons (CoordVar t_shape 1) (IntExprNil))) x_shape)
  t_shape))
(let t2 (LogicalIndexMapApply t1
  (IndexMapLit (IntExprCons (CoordVar x_shape 0)
    (IntExprCons (CoordVar x_shape 1) (IntExprNil))) t_shape)
  x_shape))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    // Anchor the logical classes structurally (never by spelling): x is the
    // input; a candidate t1 is any apply whose child is x's class; the
    // rejoin holds iff some apply over a candidate t1's class itself lives
    // in x's class. Existential over candidates because the marker rules
    // may mint additional applies over these classes (sandwich siblings) —
    // a first-match anchor would be hash-order nondeterministic.
    let x_class = s
        .nodes
        .values()
        .find(|n| n.op == "LogicalTensorInputLit")
        .expect("input exists")
        .eclass
        .clone();
    let applies: Vec<_> = s
        .nodes
        .values()
        .filter(|n| n.op == "LogicalIndexMapApply")
        .collect();
    assert!(
        applies.len() >= 2,
        "both applies must survive to the egraph"
    );
    let child_class = |n: &&luminal::prelude::egraph_serialize::Node| {
        n.children
            .first()
            .and_then(|c| s.nodes.get(c))
            .map(|c| c.eclass.clone())
    };
    let t1_classes: Vec<_> = applies
        .iter()
        .filter(|n| child_class(n).map(|c| c == x_class).unwrap_or(false))
        .map(|n| n.eclass.clone())
        .collect();
    assert!(!t1_classes.is_empty(), "t1 anchored at x");
    let rejoined = applies.iter().any(|n| {
        n.eclass == x_class
            && child_class(n)
                .map(|c| t1_classes.contains(&c))
                .unwrap_or(false)
    });
    println!(
        "double-transpose rejoin: t1 candidate classes {t1_classes:?} vs x class {x_class:?} -> rejoined = {rejoined}"
    );
    // PINNED FACT (flipped 2026-08-26): the double-transpose collapse rule
    // in src/egg/cublaslt_marker_canonicalize.egg unions apply(apply(x,T),T)
    // with x — round 10/11's termination tower is anchored on exactly this.
    assert!(
        rejoined,
        "the logical double-transpose collapse no longer fires — the \
         transpose-sandwich rewrite's termination anchor is GONE (see the \
         union-only collapse rule in src/egg/cublaslt_marker_canonicalize.egg \
         and the divergence demonstration in r11_collapse_revert_probe.rs)"
    );
}
