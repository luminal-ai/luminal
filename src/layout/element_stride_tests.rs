use super::*;

#[test]
fn addresses_and_spans_observe_singletons_padding_and_empty_domains() {
    for (shape, strides, last, span) in [
        (
            vec![1, 4],
            vec![
                ElementStride::Any,
                ElementStride::Exact(IntExprTerm::Lit(1)),
            ],
            vec![0, 3],
            4,
        ),
        (
            vec![3, 4],
            vec![
                ElementStride::Exact(IntExprTerm::Lit(8)),
                ElementStride::Exact(IntExprTerm::Lit(1)),
            ],
            vec![2, 3],
            20,
        ),
        (
            vec![0, 4],
            vec![
                ElementStride::Exact(IntExprTerm::Lit(8)),
                ElementStride::Exact(IntExprTerm::Lit(1)),
            ],
            vec![],
            0,
        ),
    ] {
        let layout = ElementStridedLayout {
            shape: ShapeTerm(shape.into_iter().map(IntExprTerm::Lit).collect()),
            strides,
            width: BitWidthTerm(32),
        };
        layout.validate().unwrap();
        assert_eq!(layout.span().eval_literal(), Some(span));
        if !last.is_empty() {
            assert_eq!(layout.element_index(&last).unwrap(), span as usize - 1);
        }
    }
    let invalid = ElementStridedLayout {
        shape: ShapeTerm(vec![IntExprTerm::Var("m".into())]),
        strides: vec![ElementStride::Any],
        width: BitWidthTerm(32),
    };
    assert!(invalid.validate().is_err());
}

const SATURATE: &str = "(run-schedule (saturate (run main_ruleset) (run prop)))";

fn graph() -> EGraph {
    let mut graph = crate::egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
        .unwrap();
    graph
}

fn graph_without_reverse_bridge() -> EGraph {
    // Prove the new rules work without feeding authored ElementStride layouts
    // into the old affine/offset machinery first. Runtime assemblies retain it.
    let bridge = r#"(rule ((= ?l (ElementStridedLayoutLit ?shape ?ss ?w)) (= ?shape (ShapeLit ?dims))
       (element-stride-expressions-from-list ?dims ?ss ?plain))
 ((strided-lists ?l ?shape ?dims ?plain ?w)) :ruleset main_ruleset)"#;
    let program = crate::egglog_snippet::assembled_program_for(&[]);
    assert_eq!(program.matches(bridge).count(), 1);
    let mut graph = crate::egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(None, &program.replace(bridge, ""))
        .unwrap();
    graph
}

fn run(graph: &mut EGraph, script: &str) {
    graph
        .parse_and_run_program(None, script)
        .unwrap_or_else(|e| panic!("{e}\n{script}"));
}

fn list(items: &[String], cons: &str, nil: &str) -> String {
    items.iter().rev().fold(format!("({nil})"), |tail, item| {
        format!("({cons} {item} {tail})")
    })
}

fn layout(name: &str, dims: &[&str], strides: &[&str]) -> String {
    let dims = list(
        &dims.iter().map(|v| v.to_string()).collect::<Vec<_>>(),
        "IntExprCons",
        "IntExprNil",
    );
    let strides = list(
        &strides
            .iter()
            .map(|v| format!("(ElementStrideExact {v})"))
            .collect::<Vec<_>>(),
        "ElementStrideCons",
        "ElementStrideNil",
    );
    format!(
        r#"
(let {name}_shape (ShapeLit {dims}))
(let {name}_logical (LogicalTensorInputLit (LogicalIdLit "{name}") {name}_shape (F32)))
(let {name}_layout (ElementStridedLayoutLit {name}_shape {strides} (BitWidthLit 32)))
(let {name}_lt (LayoutTensorLit {name}_logical {name}_layout))
"#
    )
}

#[test]
fn singleton_normalization_is_contextual_and_contiguity_uses_lists() {
    let mut g = graph_without_reverse_bridge();
    run(
        &mut g,
        &(layout(
            "one",
            &["(IntLit 1)", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + &layout(
            "live",
            &["(IntLit 3)", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + &layout(
            "broadcast",
            &["(IntLit 3)", "(IntLit 4)"],
            &["(IntLit 0)", "(IntLit 1)"],
        ) + SATURATE),
    );
    run(
        &mut g,
        r#"
(check (element-stride-reading one_lt 1 (ElementStrideAny)))
(check (= one_layout (RightMajorContiguousElementLayoutLit one_shape (BitWidthLit 32))))
(check (element-stride-reading live_lt 1 (ElementStrideExact (IntLit 8))))
(check (element-stride-reading broadcast_lt 1 (ElementStrideExact (IntLit 0))))
(fail (check (element-stride-reading live_lt 1 (ElementStrideAny))))
(fail (check (element-stride-reading broadcast_lt 1 (ElementStrideAny))))
(fail (check (= live_layout (RightMajorContiguousElementLayoutLit live_shape (BitWidthLit 32)))))
(fail (check (= (ElementStrideExact (IntLit 8)) (ElementStrideAny))))
(fail (check (= (IntLit 8) (IntLit 4))))
"#,
    );
}

#[test]
fn dynamic_extent_crossing_one_keeps_the_authored_exact_witness() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(let m (IntVar "m"))
(set (lower-bound-of m) (bigint 1))
(set (upper-bound-of m) (bigint 8))
"#,
    );
    run(
        &mut g,
        &(layout(
            "dynamic",
            &["m", "(IntLit 8)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + SATURATE),
    );
    run(
        &mut g,
        r#"
(check (element-stride-reading dynamic_lt 1 (ElementStrideExact (IntLit 8))))
(fail (check (element-stride-reading dynamic_lt 1 (ElementStrideAny))))
"#,
    );
}

#[test]
fn read_comparator_refires_after_late_equality_in_naive_rules() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(relation element-stride-test-request (ElementStride ElementStride))
(relation element-stride-test-hit (ElementStride ElementStride))
(rule ((element-stride-test-request ?a ?b)
       (= (element-stride-compatible ?a ?b) true))
 ((element-stride-test-hit ?a ?b)) :ruleset backend :naive)
(let a (ElementStrideExact (IntVar "late")))
(let b (ElementStrideExact (IntLit 8)))
(element-stride-test-request a b)
(element-stride-test-request (ElementStrideAny) b)
(run-schedule (saturate (run backend)))
(fail (check (element-stride-test-hit a b)))
(check (element-stride-test-hit (ElementStrideAny) b))
(union (IntVar "late") (IntLit 8))
(run-schedule (saturate (run backend)))
(check (element-stride-test-hit a b))
(fail (check (= b (ElementStrideAny))))
"#,
    );
}

#[test]
fn merge_prefers_any_without_corrupting_shared_exact_values() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(set (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 8))) (ElementStrideExact (IntLit 8)))
(set (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 8))) (ElementStrideAny))
(set (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 4))) (ElementStrideAny))
(set (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 4))) (ElementStrideExact (IntLit 4)))
(check (= (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 8))) (ElementStrideAny)))
(check (= (element-stride-normalized-axis (IntLit 1) (ElementStrideExact (IntLit 4))) (ElementStrideAny)))
(fail (check (= (ElementStrideExact (IntLit 8)) (ElementStrideAny))))
(fail (check (= (ElementStrideExact (IntLit 8)) (ElementStrideExact (IntLit 4)))))
"#,
    );
}

#[test]
fn layout_axis_readings_preserve_exact_proofs_through_late_equality() {
    let mut g = graph();
    run(
        &mut g,
        &(layout(
            "live",
            &["(IntLit 3)", "(IntLit 4)"],
            &["(IntLit 4)", "(IntLit 1)"],
        ) + &layout(
            "equivalent",
            &["(IntLit 3)", "(IntLit 4)"],
            &["(IntAdd (IntLit 2) (IntLit 2))", "(IntLit 1)"],
        ) + "(union live_layout equivalent_layout)"
            + SATURATE),
    );
    run(
        &mut g,
        r#"
(check (= (ElementStrideExact (IntLit 4)) (ElementStrideExact (IntAdd (IntLit 2) (IntLit 2)))))
(check (element-stride-readings-of live_layout 1 (ElementStrideExact (IntLit 4))))
(check (element-stride-reading live_lt 1 (ElementStrideExact (IntLit 4))))
(fail (check (element-stride-readings-of live_layout 1 (ElementStrideAny))))
(fail (check (= (ElementStrideExact (IntLit 4)) (ElementStrideAny))))
"#,
    );
}

#[test]
fn empty_layout_equality_preserves_different_exact_coefficients() {
    let mut g = graph();
    run(
        &mut g,
        &(layout("first", &["(IntLit 0)", "(IntLit 4)"], &["(IntLit 8)", "(IntLit 1)"])
        + &layout("second", &["(IntLit 0)", "(IntLit 4)"], &["(IntLit 12)", "(IntLit 1)"])
        // Both layouts have the same empty domain and therefore agree at every
        // valid coordinate. Their coefficients need not be globally equal.
        + "(union first_layout second_layout)" + SATURATE),
    );
    run(
        &mut g,
        r#"
(check (element-stride-readings-of first_layout 1 (ElementStrideExact (IntLit 8))))
(check (element-stride-readings-of first_layout 1 (ElementStrideExact (IntLit 12))))
(check (element-stride-reading first_lt 1 (ElementStrideExact (IntLit 8))))
(check (element-stride-reading first_lt 1 (ElementStrideExact (IntLit 12))))
(fail (check (element-stride-readings-of first_layout 1 (ElementStrideAny))))
(fail (check (element-stride-readings-of first_layout 0 (ElementStrideAny))))
(fail (check (= (IntLit 8) (IntLit 12))))
"#,
    );
}

#[test]
fn late_singleton_proof_adds_freedom_without_merging_exact_values() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(let extent (IntVar "late-singleton"))
(set (lower-bound-of extent) (bigint 1))
(set (upper-bound-of extent) (bigint 8))
"#,
    );
    run(
        &mut g,
        &(layout(
            "late",
            &["extent", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + SATURATE),
    );
    run(
        &mut g,
        r#"
(check (element-stride-readings-of late_layout 1 (ElementStrideExact (IntLit 8))))
(fail (check (element-stride-readings-of late_layout 1 (ElementStrideAny))))
(union extent (IntLit 1))
"#,
    );
    run(&mut g, SATURATE);
    run(
        &mut g,
        r#"
(check (element-stride-readings-of late_layout 1 (ElementStrideAny)))
(check (element-stride-reading late_lt 1 (ElementStrideAny)))
(check (= late_layout (RightMajorContiguousElementLayoutLit late_shape (BitWidthLit 32))))
(fail (check (= (ElementStrideExact (IntLit 8)) (ElementStrideAny))))
(fail (check (= (IntLit 8) (IntLit 4))))
"#,
    );
}

#[test]
fn view_created_by_a_rule_gets_strides_during_the_same_saturation() {
    let mut g = graph_without_reverse_bridge();
    run(
        &mut g,
        &layout(
            "pitched",
            &["(IntLit 3)", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ),
    );
    run(
        &mut g,
        r#"
(let transposed_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let transpose_map (IndexMapLit (IntExprCons (CoordVar transposed_shape 0)
    (IntExprCons (CoordVar transposed_shape 1) (IntExprNil))) pitched_shape))
(relation element-stride-test-trigger (LogicalTensor))
(rule ((element-stride-test-trigger ?v))
 ((LogicalIndexMapApply ?v transpose_map transposed_shape)) :ruleset main_ruleset)
(element-stride-test-trigger pitched_logical)
"#,
    );
    run(&mut g, SATURATE);
    run(
        &mut g,
        r#"
(check (element-stride-reading
 (LayoutTensorLit (LogicalIndexMapApply pitched_logical transpose_map transposed_shape)
  (ElementStridedLayoutLit transposed_shape
   (ElementStrideCons (ElementStrideExact (IntLit 1))
    (ElementStrideCons (ElementStrideExact (IntLit 8)) (ElementStrideNil))) (BitWidthLit 32)))
 0 (ElementStrideExact (IntLit 8))))
"#,
    );
}

#[test]
fn broadcasting_a_singleton_creates_zero_stride_on_the_live_output_axis() {
    let mut g = graph();
    run(
        &mut g,
        &layout(
            "one",
            &["(IntLit 1)", "(IntLit 4)"],
            &["(IntLit 19)", "(IntLit 1)"],
        ),
    );
    run(
        &mut g,
        r#"
(let broad_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let broad_map (IndexMapLit (IntExprCons (IntLit 0)
    (IntExprCons (CoordVar broad_shape 0) (IntExprNil))) one_shape))
(let broad_logical (LogicalIndexMapApply one_logical broad_map broad_shape))
"#,
    );
    run(&mut g, SATURATE);
    run(
        &mut g,
        r#"
(check (element-stride-reading (LayoutTensorLit broad_logical
 (ElementStridedLayoutLit broad_shape
  (ElementStrideCons (ElementStrideExact (IntLit 0))
   (ElementStrideCons (ElementStrideExact (IntLit 1)) (ElementStrideNil))) (BitWidthLit 32)))
 1 (ElementStrideExact (IntLit 0))))
(fail (check (element-stride-reading (LayoutTensorLit broad_logical ?layout) 1 (ElementStrideAny))))
"#,
    );
}

#[test]
fn literal_expression_recovery_preserves_origin_and_singleton_freedom() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let e (IntAdd (IntMul (CoordVar shape 1) (IntLit 8)) (CoordVar shape 0)))
(let zero_origin (ElementOffsetExpressionLayoutLit e shape (BitWidthLit 32)))
(let shifted (ElementOffsetExpressionLayoutLit (IntAdd e (IntLit 2)) shape (BitWidthLit 32)))
(let nonlinear (ElementOffsetExpressionLayoutLit (IntMul (CoordVar shape 1) (CoordVar shape 0)) shape (BitWidthLit 32)))
(let singleton_shape (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let dead (CoordVar singleton_shape 1))
(let singleton (ElementOffsetExpressionLayoutLit
 (IntAdd (IntMul dead (IntLit 999)) (CoordVar singleton_shape 0)) singleton_shape (BitWidthLit 32)))
; Multiplication by a collapsed coordinate must not invent a nonzero origin.
(let zero (ElementOffsetExpressionLayoutLit (IntMul dead (IntLit 32)) singleton_shape (BitWidthLit 32)))
(element-stride-discovery-demand zero_origin)
(element-stride-discovery-demand shifted)
(element-stride-discovery-demand nonlinear)
(element-stride-discovery-demand singleton)
(element-stride-discovery-demand zero)
(run-schedule (saturate (run element-stride-discovery) (run main_ruleset) (run prop)))
(check (element-stride-discovered zero_origin))
(check (= zero_origin (ElementStridedLayoutLit shape
 (ElementStrideCons (ElementStrideExact (IntLit 8))
  (ElementStrideCons (ElementStrideExact (IntLit 1)) (ElementStrideNil))) (BitWidthLit 32))))
(check (element-stride-discovered singleton))
(check (= singleton (RightMajorContiguousElementLayoutLit singleton_shape (BitWidthLit 32))))
(check (element-stride-discovered zero))
(fail (check (element-stride-discovered shifted)))
(fail (check (element-stride-discovered nonlinear)))
(fail (check (= (IntLit 0) (IntLit 32))))
(fail (check (= (ElementStrideAny) (ElementStrideExact (IntLit 999)))))
"#,
    );
    let before = g
        .serialize(egglog::SerializeConfig::default())
        .egraph
        .nodes
        .len();
    run(
        &mut g,
        "(run-schedule (repeat 40 (run element-stride-discovery) (run main_ruleset) (run prop)))",
    );
    let after = g
        .serialize(egglog::SerializeConfig::default())
        .egraph
        .nodes
        .len();
    assert_eq!(
        before, after,
        "a claimed fixpoint grew on another 40 rounds"
    );
    println!(
        "literal recovery fixpoint: {before} serialized nodes, unchanged after 40 extra rounds"
    );
}

#[test]
fn exact_conflicts_are_not_proof_of_singleton_freedom() {
    let mut g = graph();
    run(
        &mut g,
        r#"
; This deliberately models the proposed UNGUARDED conflict->Any merge.
; Distinct expression IDs do not imply distinct arithmetic values.
(function element-stride-test-unsafe-join (i64) ElementStride :merge (ElementStrideAny))
(let s4 (ElementStrideExact (IntLit 4)))
(let s22 (ElementStrideExact (IntAdd (IntLit 2) (IntLit 2))))
(fail (check (= s4 s22)))
(set (element-stride-test-unsafe-join 0) s4)
(set (element-stride-test-unsafe-join 0) s22)
(check (= (element-stride-test-unsafe-join 0) (ElementStrideAny)))
(run-schedule (saturate (run main_ruleset) (run prop)))
(check (= s4 s22))
; Learning the equality later cannot undo the invented freedom.
(check (= (element-stride-test-unsafe-join 0) (ElementStrideAny)))
(fail (check (= s4 (ElementStrideAny))))
"#,
    );
}

#[test]
fn cons_list_certificates_preserve_signed_intermediates_and_publication_limits() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let row (CoordVar shape 1))
(let column (CoordVar shape 0))
; (-4*row - column)*-2 = 8*row + 2*column.
(let expression (IntMul (IntLit -2)
 (IntAdd (IntMul row (IntLit -4)) (IntMul column (IntLit -1)))))
(let recovered (ElementOffsetExpressionLayoutLit expression shape (BitWidthLit 32)))
(let negative (ElementOffsetExpressionLayoutLit (IntMul row (IntLit -1)) shape (BitWidthLit 32)))
(element-stride-discovery-demand recovered)
(element-stride-discovery-demand negative)
(run-schedule (saturate (run element-stride-discovery) (run main_ruleset) (run prop)))
(check (element-stride-linear-certificate expression shape (ElementStrideLiteralCoefficientsCons (bigint 8) (ElementStrideLiteralCoefficientsCons (bigint 2) (ElementStrideLiteralCoefficientsNil))) (bigint 0)))
(check (= recovered (ElementStridedLayoutLit shape
 (ElementStrideCons (ElementStrideExact (IntLit 8))
  (ElementStrideCons (ElementStrideExact (IntLit 2)) (ElementStrideNil))) (BitWidthLit 32))))
(fail (check (element-stride-discovered negative)))
"#,
    );
    use crate::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    let (sort, value) = g
        .eval_expr(&egglog::ast::Expr::Var(
            egglog::ast::Span::Panic,
            "recovered".into(),
        ))
        .unwrap();
    let recovered_class = g.value_to_class_id(&sort, value);
    let serialized = g.serialize(egglog::SerializeConfig::default()).egraph;
    let registry = ConstructorRegistry::new(crate::egglog_snippet::core_decoders()).unwrap();
    let view = EGraphView::new(&serialized, &registry);
    let mut checked = 0;
    let spellings = view.class(&recovered_class).spellings::<Layout>();
    for layout in spellings.all::<ElementStridedLayout>() {
        for row in 0..3 {
            for column in 0..4 {
                assert_eq!(
                    layout.element_index(&[row, column]).unwrap(),
                    8 * row + 2 * column
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 12);
    // Exercise BigInt publication without asking the i64 IntExpr ring to fold
    // an unrepresentable literal product. The certificate arithmetic is wider.
    let mut large = graph();
    run(
        &mut large,
        r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(CoordVar shape 0)
(run-schedule (saturate (run main_ruleset) (run prop)))
(let expression (IntMul (IntLit 2) (IntMul (IntLit 9223372036854775807) (CoordVar shape 0))))
(let enormous (ElementOffsetExpressionLayoutLit expression shape (BitWidthLit 32)))
(element-stride-discovery-demand enormous)
(run-schedule (saturate (run element-stride-discovery)))
(check (element-stride-linear-certificate expression shape
 (ElementStrideLiteralCoefficientsCons (from-string "18446744073709551614") (ElementStrideLiteralCoefficientsNil)) (bigint 0)))
(fail (check (element-stride-discovered enormous)))
"#,
    );
}

#[test]
fn diagonal_composition_adds_coefficients_and_dynamic_transpose_keeps_exact() {
    let mut g = graph_without_reverse_bridge();
    run(
        &mut g,
        &(layout(
            "square",
            &["(IntLit 3)", "(IntLit 3)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + &layout(
            "dynamic",
            &["(IntVar \"m\")", "(IntLit 8)"],
            &["(IntLit 8)", "(IntLit 1)"],
        )),
    );
    run(
        &mut g,
        r#"
(set (lower-bound-of (IntVar "m")) (bigint 1))
(set (upper-bound-of (IntVar "m")) (bigint 8))
(let dshape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let diagonal (LogicalIndexMapApply square_logical
 (IndexMapLit (IntExprCons (CoordVar dshape 0) (IntExprCons (CoordVar dshape 0) (IntExprNil))) square_shape) dshape))
(let tshape (ShapeLit (IntExprCons (IntLit 8) (IntExprCons (IntVar "m") (IntExprNil)))))
(let transpose (LogicalIndexMapApply dynamic_logical
 (IndexMapLit (IntExprCons (CoordVar tshape 0) (IntExprCons (CoordVar tshape 1) (IntExprNil))) dynamic_shape) tshape))
"#,
    );
    run(&mut g, SATURATE);
    run(
        &mut g,
        r#"
(check (element-stride-reading (LayoutTensorLit diagonal
 (ElementStridedLayoutLit dshape (ElementStrideCons (ElementStrideExact (IntLit 9)) (ElementStrideNil)) (BitWidthLit 32)))
 0 (ElementStrideExact (IntLit 9))))
(check (element-stride-reading (LayoutTensorLit transpose
 (ElementStridedLayoutLit tshape (ElementStrideCons (ElementStrideExact (IntLit 1))
  (ElementStrideCons (ElementStrideExact (IntLit 8)) (ElementStrideNil))) (BitWidthLit 32)))
 0 (ElementStrideExact (IntLit 8))))
(fail (check (element-stride-reading (LayoutTensorLit transpose ?l) 0 (ElementStrideAny))))
"#,
    );
}

#[test]
fn recovered_layout_decoders_agree_on_every_coordinate_in_a_small_corpus() {
    use crate::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    use crate::layouts::ElementOffsetExpressionLayout;
    let mut g = graph();
    let mut source = String::new();
    let mut cases = Vec::new();
    for (m, n) in [(1, 1), (1, 3), (3, 1), (2, 3), (3, 2)] {
        for (row, col) in [(0, 0), (0, 1), (1, 0), (8, 1), (1, 5), (2, 3)] {
            cases.push((vec![m, n], vec![row, col]));
        }
    }
    cases.extend([
        (vec![], vec![]),
        (vec![4], vec![3]),
        (vec![2, 3, 4], vec![20, 5, 1]),
        (vec![1, 2, 1, 3], vec![80, 10, 4, 2]),
    ]);
    for (index, (dimensions, coefficients)) in cases.iter().enumerate() {
        let dimensions = list(
            &dimensions
                .iter()
                .map(|n| format!("(IntLit {n})"))
                .collect::<Vec<_>>(),
            "IntExprCons",
            "IntExprNil",
        );
        let expression = coefficients.iter().enumerate().fold(
            "(IntLit 0)".to_string(),
            |sum, (position, coefficient)| {
                let axis = coefficients.len() - position - 1;
                format!(
                    "(IntAdd {sum} (IntMul (CoordVar shape_{index} {axis}) (IntLit {coefficient})))"
                )
            },
        );
        source += &format!(
            r#"
(let shape_{index} (ShapeLit {dimensions}))
(let layout_{index} (ElementOffsetExpressionLayoutLit {expression} shape_{index} (BitWidthLit 32)))
(element-stride-discovery-demand layout_{index})
"#
        );
    }
    source +=
        "(run-schedule (saturate (run element-stride-discovery) (run main_ruleset) (run prop)))";
    run(&mut g, &source);
    for i in 0..cases.len() {
        run(
            &mut g,
            &format!("(check (element-stride-discovered layout_{i}))"),
        );
    }
    let serialized = g.serialize(egglog::SerializeConfig::default()).egraph;
    let registry = ConstructorRegistry::new(crate::egglog_snippet::core_decoders()).unwrap();
    let view = EGraphView::new(&serialized, &registry);
    let classes: std::collections::HashSet<_> = serialized
        .nodes
        .values()
        .filter(|n| n.op == "ElementOffsetExpressionLayoutLit")
        .map(|n| n.eclass.clone())
        .collect();
    let mut addresses = 0;
    for class in classes {
        let spellings = view.class(&class).spellings::<Layout>();
        let Some(expr) = spellings.first::<ElementOffsetExpressionLayout>() else {
            continue;
        };
        let strides = spellings.all::<ElementStridedLayout>();
        assert!(!strides.is_empty());
        let dimensions: Vec<_> = expr
            .shape
            .0
            .iter()
            .map(|extent| usize::try_from(extent.eval_literal().unwrap()).unwrap())
            .collect();
        for mut linear in 0..dimensions.iter().product() {
            let mut coordinates = vec![0; dimensions.len()];
            for (axis, extent) in dimensions.iter().enumerate().rev() {
                coordinates[axis] = linear % extent;
                linear /= extent;
            }
            for stride in &strides {
                assert_eq!(
                    stride.element_index(&coordinates).unwrap(),
                    expr.element_index(&coordinates).unwrap()
                );
                addresses += 1;
            }
        }
    }
    assert!(addresses > 100);
    let before = serialized.nodes.len();
    run(
        &mut g,
        "(run-schedule (repeat 40 (run element-stride-discovery) (run main_ruleset) (run prop)))",
    );
    assert_eq!(
        before,
        g.serialize(egglog::SerializeConfig::default())
            .egraph
            .nodes
            .len()
    );
    println!(
        "{} expression sources: {addresses} decoded addresses agree; {before} nodes plateau",
        cases.len()
    );
}

#[test]
fn malformed_authored_strides_do_not_enter_backend_readings() {
    let mut g = graph();
    run(
        &mut g,
        &layout(
            "bad",
            &["(IntLit 2)", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ),
    );
    run(
        &mut g,
        r#"
(let bad_any (LayoutTensorLit bad_logical (ElementStridedLayoutLit bad_shape
 (ElementStrideCons (ElementStrideAny) (ElementStrideCons (ElementStrideExact (IntLit 1)) (ElementStrideNil))) (BitWidthLit 32))))
(let bad_rank (LayoutTensorLit bad_logical (ElementStridedLayoutLit bad_shape
 (ElementStrideCons (ElementStrideExact (IntLit 8)) (ElementStrideNil)) (BitWidthLit 32))))
"#,
    );
    run(&mut g, SATURATE);
    run(
        &mut g,
        r#"
(fail (check (element-stride-reading bad_any ?axis ?stride)))
(fail (check (element-stride-reading bad_rank ?axis ?stride)))
"#,
    );
}

#[test]
fn empty_authored_layouts_have_zero_span_through_the_shared_decoder() {
    use crate::egglog_utils::eclass::{ConstructorRegistry, EGraphView};
    use crate::layouts::DecodedLayout;
    let mut g = graph();
    run(
        &mut g,
        &(layout(
            "empty_outer",
            &["(IntLit 0)", "(IntLit 4)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + &layout(
            "empty_inner",
            &["(IntLit 3)", "(IntLit 0)"],
            &["(IntLit 8)", "(IntLit 1)"],
        ) + SATURATE),
    );
    let serialized = g.serialize(egglog::SerializeConfig::default()).egraph;
    let registry = ConstructorRegistry::new(crate::egglog_snippet::core_decoders()).unwrap();
    let view = EGraphView::new(&serialized, &registry);
    let mut checked = 0;
    for node in serialized
        .nodes
        .values()
        .filter(|n| n.op == "ElementStridedLayoutLit")
    {
        let decoded = DecodedLayout::from_class(&view.class(&node.eclass), None).unwrap();
        assert_eq!(decoded.span_with(&Default::default()).unwrap(), 0);
        let spellings = view.class(&node.eclass).spellings::<Layout>();
        let affine = spellings
            .require::<crate::layouts::StridedElementLayout>("empty compatibility bridge")
            .unwrap();
        assert_eq!(affine.span().eval_literal(), Some(0));
        checked += 1;
    }
    assert!(checked >= 2);
}

#[test]
fn read_comparator_requires_naive_rules_and_returns_boolean_for_all_readings() {
    let mut g = graph();
    run(
        &mut g,
        r#"
(ElementStrideExact (IntLit 8))
(check (= (element-stride-compatible (ElementStrideAny) (ElementStrideExact (IntLit 8))) true))
(check (= (element-stride-compatible (ElementStrideExact (IntLit 8)) (ElementStrideAny)) true))
(check (= (element-stride-compatible (ElementStrideExact (IntLit 8)) (ElementStrideExact (IntLit 8))) true))
(check (= (element-stride-compatible (ElementStrideExact (IntLit 1)) (ElementStrideExact (IntLit 8))) false))
(relation element-stride-test-comparison-request (ElementStride ElementStride))
(relation element-stride-test-comparison-hit (ElementStride ElementStride))
"#,
    );
    let rule = r#"
(rule ((element-stride-test-comparison-request ?left ?right)
       (= (element-stride-compatible ?left ?right) true))
 ((element-stride-test-comparison-hit ?left ?right)) :ruleset backend)
"#;
    let error = g
        .parse_and_run_program(None, rule)
        .expect_err("a database-reading comparator requires a naive rule body");
    // The pinned typechecker hides primitives that lack the required context.
    assert!(
        error
            .to_string()
            .contains("Unbound function element-stride-compatible"),
        "{error}"
    );
    run(
        &mut g,
        &rule.replace(":ruleset backend)", ":ruleset backend :naive)"),
    );
}
