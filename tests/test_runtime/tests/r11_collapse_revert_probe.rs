//! ROUND-11 REVERT PROBE (permanent): the transpose sandwich needs a
//! termination anchor, and the probe pins that one is in force.
//!
//! The canonical-form sandwich mints a sibling whose operands are rank-2
//! transpose VIEWS, and the sibling is itself canonical, so the sandwich
//! fires on it too. Unless views-of-views union back into the original
//! tensor, each generation's views are NEW values and the main ruleset
//! never saturates (measured 2026-08-26 with no anchor: 3729 / 4616 /
//! 5386 / 6156 nodes at K = 40/60/80/100, ~38 nodes per iteration).
//!
//! Two anchors exist. The cuBLASLt estate's double-transpose collapse
//! rule (cublaslt_marker_canonicalize.egg) unions an apply-of-apply
//! transpose with its base directly. Core's `LogicalHelperMatrixTranspose`
//! recognizer names every rank-2 transpose view, and its involution rule
//! unions a transpose of a transpose with the base, so core anchors the
//! sandwich on its own. The probe runs a BOUNDED schedule ((run-schedule
//! (repeat K (seq (run) (run backend) (run prop))))) at increasing K WITH
//! and WITHOUT the collapse rule
//! (removed by exact string surgery on the assembled program) and
//! requires both to reach the same flat node count inside the range: the
//! collapse rule is redundant, and if this ever grows again the anchor
//! has been lost.

use luminal::dtype::DType;
use luminal::graph::Graph;

/// The assembled program for the canonical fixture, with the recorder's
/// saturating schedule replaced by a bounded one, and (optionally) the
/// collapse rule excised.
fn bounded_program(iters: usize, with_collapse: bool) -> String {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _out = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let preamble = luminal::egglog_snippet::assembled_program_for(&test_runtime::matchers());
    let mut program = format!("{preamble}\n\n{text}");

    // Replace the recorder's saturating schedule with a bounded run of
    // the MAIN ruleset only (the divergence lives entirely in the main
    // ruleset: sandwich + collapse are unscheduled rules). Propagation
    // is pre-saturated and stepped alongside, as every schedule does.
    let sat = test_runtime::TestRuntimeBindings::SCHEDULE.trim_end();
    assert!(
        program.contains(sat),
        "recorder schedule line not found — probe surgery is stale"
    );
    program = program.replace(
        sat,
        &format!("(run-schedule (saturate (run prop)) (repeat {iters} (seq (run) (run backend) (run prop))))"),
    );

    if !with_collapse {
        // Excise the collapse rule: from its marker header to the next
        // file-end marker. The rule is the LAST item in
        // cublaslt_marker_canonicalize.egg, so slicing from its header
        // comment to that file's trailing separator removes exactly it.
        let start_marker = "; THE DOUBLE-TRANSPOSE COLLAPSE";
        let start = program
            .find(start_marker)
            .expect("collapse rule marker present in assembled program");
        // The rule's closing: the last `(union ?w ?x)` action block ends
        // with a line `)` followed by the separator. Find the separator
        // AFTER the marker.
        let tail = &program[start..];
        let end_rel = tail
            .find("(union ?w ?x)")
            .and_then(|p| tail[p..].find("\n)\n").map(|q| p + q + 3))
            .expect("collapse rule body found");
        // Also strip the header comment block back to its opening
        // separator line so no dangling comment remains (comments are
        // inert; precision here is cosmetic).
        program.replace_range(start..start + end_rel, "");
    }
    program
}

fn node_count(program: &str) -> usize {
    use luminal::prelude::egglog::SerializeConfig;
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, program)
        .unwrap_or_else(|err| panic!("egglog failed: {err}"));
    egraph
        .serialize(SerializeConfig::default())
        .egraph
        .nodes
        .len()
}

#[test]
fn r11_sandwich_terminates_with_and_without_the_collapse_rule() {
    let counts = |with_collapse: bool| -> Vec<usize> {
        [60usize, 80, 100]
            .into_iter()
            .map(|iters| {
                let n = node_count(&bounded_program(iters, with_collapse));
                println!(
                    "collapse {}, run {iters}: {n} nodes",
                    if with_collapse { "PRESENT" } else { "REMOVED" }
                );
                n
            })
            .collect()
    };
    let without = counts(false);
    let with = counts(true);
    assert!(
        without.windows(2).all(|p| p[0] == p[1]),
        "without the collapse rule the main ruleset must still saturate inside \
         the probe range — core's transpose involution is the anchor (got {without:?})"
    );
    assert!(
        with.windows(2).all(|p| p[0] == p[1]),
        "with the collapse rule the main ruleset must saturate inside the \
         probe range (got {with:?})"
    );
    assert_eq!(
        without[0], with[0],
        "the two anchors must close at the same fixed point"
    );
}
