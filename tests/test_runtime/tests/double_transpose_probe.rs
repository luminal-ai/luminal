//! Core transpose involution must keep complete-expression proofs finite.
const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)) (saturate (saturate (run main_ruleset) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

#[test]
fn double_transpose_rejoins_via_core_involution() {
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
    assert!(
        rejoined,
        "core transpose involution must rejoin the original value"
    );
}
