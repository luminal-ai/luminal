//! Termination and semantic regressions for integer offset normalization.

use luminal::{egglog_snippet, prelude::egglog};

fn graph(seed: &str) -> egglog::EGraph {
    let mut graph = egglog_snippet::new_egraph();
    graph
        .parse_and_run_program(None, &egglog_snippet::assembled_program_for(&[]))
        .unwrap();
    graph.parse_and_run_program(None, seed).unwrap();
    graph
}

// Bound each round and the number of rounds, so the unfixed regression fails
// before the associativity query itself can allocate gigabytes of matches.
fn saturate_main(graph: &mut egglog::EGraph) -> bool {
    saturate_main_with_limit(graph, 8_000)
}

fn saturate_main_with_limit(graph: &mut egglog::EGraph, max_rows: usize) -> bool {
    for _ in 0..256 {
        let report = graph
            .parse_and_run_program(None, "(run-schedule (run main_ruleset) (run prop))")
            .unwrap();
        assert!(
            graph.num_tuples() < max_rows,
            "integer normalization grew past {max_rows} rows"
        );
        let updated = report.iter().any(
            |output| matches!(output, egglog::CommandOutput::RunSchedule(report) if report.updated),
        );
        if !updated {
            return true;
        }
    }
    false
}

fn assert_coordinates_have_no_live_additive_aliases(graph: &egglog::EGraph) {
    let snapshot = graph.serialize(egglog::SerializeConfig::default());
    let coordinate_classes: std::collections::HashSet<_> = snapshot
        .egraph
        .nodes
        .values()
        .filter(|node| node.op == "CoordVar")
        .map(|node| &node.eclass)
        .collect();
    assert!(
        snapshot.egraph.nodes.values().all(|node| {
            node.op != "IntAdd" || node.subsumed || !coordinate_classes.contains(&node.eclass)
        }),
        "a cancelled sum survived as a live alias of a coordinate"
    );
}

#[test]
fn shifted_coordinate_cancellation_reaches_a_fixpoint() {
    // On 0ae288ca, round 6 reassociates to q + (1 + -1), round 7
    // folds the literal sum, and round 8's additive identity first unions
    // restored with q. The original restored = (q + 1) + -1 stays live
    // in q's class, closing the inverse cycle and feeding reassociation.
    let mut graph = graph(
        "(let shape (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let x (CoordVar shape 0))
         (let shifted (IntAdd x (IntLit 1)))
         (let restored (IntAdd shifted (IntLit -1)))",
    );
    assert!(
        saturate_main(&mut graph),
        "offset cancellation did not converge"
    );
    graph
        .parse_and_run_program(None, "(check (= restored x))")
        .unwrap();
    assert_coordinates_have_no_live_additive_aliases(&graph);
}

#[test]
fn substitution_of_a_shifted_slice_reaches_a_fixpoint() {
    let mut graph = graph(
        "(let source (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let output (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
         (let p (CoordVar source 0))
         (let q (CoordVar output 0))
         (let parent (IntAdd p (IntLit -1)))
         (let map (IndexMapLit
             (IntExprCons (IntAdd q (IntLit 1)) (IntExprNil)) source))
         (int-subst-demand parent map)",
    );
    // The input is benign. The walk introduces (q + 1) + (-1).
    assert!(saturate_main(&mut graph));
    graph
        .parse_and_run_program(None, "(run-schedule (run subst-walk))")
        .unwrap();
    assert!(
        saturate_main(&mut graph),
        "post-substitution cancellation did not converge"
    );
    graph
        .parse_and_run_program(None, "(check (= (int-subst-of parent map) q))")
        .unwrap();
    assert_coordinates_have_no_live_additive_aliases(&graph);
    // A later walk generation must preserve the retirement fence.
    graph
        .parse_and_run_program(None, "(run-schedule (run subst-walk))")
        .unwrap();
    assert!(saturate_main(&mut graph));
    assert_coordinates_have_no_live_additive_aliases(&graph);
}

#[test]
fn offsets_in_both_operand_orders_combine_across_a_symbolic_sum() {
    let mut graph = graph(
        "(let shape (ShapeLit (IntExprCons (IntLit 8)
                                (IntExprCons (IntLit 8) (IntExprNil)))))
         (let x (CoordVar shape 0))
         (let y (CoordVar shape 1))
         (let a (IntAdd (IntAdd x (IntLit 3)) (IntAdd y (IntLit -5))))
         (let b (IntAdd (IntAdd (IntLit -5) y) (IntAdd (IntLit 3) x)))
         (let expected (IntAdd (IntAdd x y) (IntLit -2)))",
    );
    assert!(saturate_main(&mut graph));
    graph
        .parse_and_run_program(None, "(check (= a expected)) (check (= b expected))")
        .unwrap();
}

#[test]
fn zero_inner_offset_does_not_retire_the_canonical_shift() {
    let mut graph = graph(
        "(let shape (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let x (CoordVar shape 0))
         (let result (IntAdd (IntAdd x (IntLit 0)) (IntLit 3)))",
    );
    assert!(saturate_main(&mut graph));
    graph
        .parse_and_run_program(None, "(check (= result (IntAdd x (IntLit 3))))")
        .unwrap();
    let snapshot = graph.serialize(egglog::SerializeConfig::default());
    assert!(
        snapshot
            .egraph
            .nodes
            .values()
            .any(|node| { node.op == "IntAdd" && !node.subsumed })
    );
}

#[test]
fn normalization_preserves_truncated_division_at_negative_inputs() {
    let mut graph = graph(
        "(let x (IntVar \"x\"))
         (set (lower-bound-of x) (bigint -12))
         (set (upper-bound-of x) (bigint 12))
         (let quotient (IntTruncDiv x (IntLit 3)))
         (let result (IntAdd (IntLit -2) (IntAdd quotient (IntLit 5))))",
    );
    assert!(saturate_main(&mut graph));
    for x in -12_i64..=12 {
        let mut pinned = graph.clone();
        pinned
            .parse_and_run_program(None, &format!("(union x (IntLit {x}))"))
            .unwrap();
        assert!(saturate_main(&mut pinned));
        let reference = x / 3 + 5 - 2;
        pinned
            .parse_and_run_program(None, &format!("(check (= result (IntLit {reference})))"))
            .unwrap();
    }
}

#[test]
fn combined_offset_outside_i64_is_not_wrapped() {
    let mut graph = graph(
        "(let shape (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
         (let x (CoordVar shape 0))
         (let result (IntAdd (IntAdd x (IntLit 9223372036854775807)) (IntLit 1)))",
    );
    assert!(saturate_main(&mut graph));
    graph
        .parse_and_run_program(
            None,
            "(fail (check (= result (IntAdd x (IntLit -9223372036854775808)))))",
        )
        .unwrap();
}

#[test]
fn mixed_radix_digits_recompose_for_signed_dividends() {
    let mut graph = graph(
        r#"(let n (IntVar "n"))
        (set (lower-bound-of n) (bigint -10000))
        (set (upper-bound-of n) (bigint 10000))
        (let a (IntVar "a"))
        (set (lower-bound-of a) (bigint 1))
        (set (upper-bound-of a) (bigint 16))
        (let b (IntVar "b"))
        (set (lower-bound-of b) (bigint 1))
        (set (upper-bound-of b) (bigint 16))
        (let low (IntTruncRem n a))
        (let middle (IntTruncRem (IntTruncDiv n a) b))
        (let combined (IntAdd (IntMul middle a) low))
        (let combined_rev (IntAdd low (IntMul a middle)))
        (let expected (IntTruncRem n (IntMul a b)))
        (let restored (IntAdd (IntMul (IntTruncDiv n (IntMul a b)) (IntMul a b)) combined))"#,
    );
    assert!(saturate_main(&mut graph));
    graph.parse_and_run_program(None,
        "(check (= combined expected)) (check (= combined_rev expected)) (check (= restored n))"
    ).unwrap();
    // Independent arithmetic check of the identity including both signs,
    // unit radices, and exact/non-exact divisions.
    for n in -257_i64..=257 {
        for a in 1..=16 {
            for b in 1..=16 {
                assert_eq!(a * ((n / a) % b) + n % a, n % (a * b));
            }
        }
    }
}

#[test]
fn flatten_unflatten_substitution_reaches_a_fixpoint() {
    // Captured from ViT's convolution -> flatten -> transpose -> concat
    // prefix. Preserve the entire rule/graph interaction as a CPU regression.
    let mut graph = egglog_snippet::new_egraph();
    let source = include_str!("../../repro/subst_layout_blowup.egg");
    // The fixture contains only graph/boundary declarations. Rules always
    // come from the current core, with no frozen or duplicate rule copies.
    graph
        .parse_and_run_program(None, &egglog_snippet::assembled_program_for(&[]))
        .unwrap();
    graph.parse_and_run_program(None, source).unwrap();
    graph
        .parse_and_run_program(None, "(run-schedule (saturate (run prop)))")
        .unwrap();
    // Complete ElementStride certificates retain extra layout-proof rows.
    // This fixture converges at about 36,400 rows with those proofs present;
    // retain a bound over every table, including proof metadata.
    let max_rows = 40_000;
    for _ in 0..64 {
        assert!(saturate_main_with_limit(&mut graph, max_rows));
        let report = graph
            .parse_and_run_program(None, "(run-schedule (run subst-walk))")
            .unwrap();
        assert!(graph.num_tuples() < max_rows);
        if !report.iter().any(
            |output| matches!(output, egglog::CommandOutput::RunSchedule(report) if report.updated),
        ) {
            return;
        }
    }
    panic!("flatten/unflatten substitution did not converge after 64 generations");
}
