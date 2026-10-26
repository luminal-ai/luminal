//! Complete-expression transposition must reach a finite fixed point.
use luminal::prelude::*;
#[test]
fn complete_expression_transposition_reaches_a_fixed_point() {
    let mut g = Graph::new();
    let a = g.tensor([2, 4], DType::F32);
    let b = g.tensor([4, 3], DType::F32);
    let _out = a.matmul(b);
    let text = test_runtime::bind_leaves(&g);
    let preamble = luminal::egglog_snippet::assembled_program_for(&test_runtime::matchers());
    let counts: Vec<_> = [60, 80, 100]
        .into_iter()
        .map(|n| {
            let schedule = format!(
                "(run-schedule (repeat {n} (seq (run main_ruleset) (run backend) (run prop))))"
            );
            let script = format!(
                "{preamble}\n{}",
                text.replace(
                    test_runtime::TestRuntimeBindings::SCHEDULE.trim_end(),
                    &schedule
                )
            );
            let mut e = luminal::egglog_snippet::new_egraph();
            e.parse_and_run_program(None, &script).unwrap();
            e.serialize(Default::default()).egraph.nodes.len()
        })
        .collect();
    assert!(
        counts.windows(2).all(|p| p[0] == p[1]),
        "proof graph failed to saturate: {counts:?}"
    );
}
