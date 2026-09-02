//! ROUND-11 REVERT PROBE (permanent): the double-transpose collapse rule
//! is the sandwich's termination anchor.
//!
//! The canonical-form sandwich mints a sibling whose operands are rank-2
//! transpose VIEWS, and the sibling is itself canonical, so the sandwich
//! fires on it too. WITH the collapse rule, generation-3 operands
//! (views-of-views) union back into generation 1 and every rebuilt chain
//! hash-conses into the original — saturation closes. WITHOUT it, each
//! generation's views are NEW values and the main ruleset never
//! saturates.
//!
//! The default schedule ((saturate ...)) would simply HANG without the
//! collapse rule, so this probe runs a BOUNDED schedule ((run-schedule
//! (repeat K (run)))) at increasing K and prints node counts:
//!   * collapse REMOVED:  counts grow strictly with every added
//!     iteration block — unbounded growth (measured 2026-08-26: 3729 /
//!     4616 / 5386 / 6156 at K = 40/60/80/100, ~38 nodes per iteration
//!     with no plateau);
//!   * collapse PRESENT:  the count reaches the main ruleset's fixed
//!     point and stays flat (measured: 3022 at K = 60, 80, and 100).
//!
//! The rule text is removed by exact string surgery on the assembled
//! program (a marker-comment slice), so what runs without the rule is
//! byte-identical everywhere else.

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
    // ruleset: sandwich + collapse are unscheduled rules).
    let sat = test_runtime::TestRuntimeBindings::SCHEDULE.trim_end();
    assert!(
        program.contains(sat),
        "recorder schedule line not found — probe surgery is stale"
    );
    program = program.replace(sat, &format!("(run-schedule (repeat {iters} (run)))"));

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
    use egglog::SerializeConfig;
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
fn r11_collapse_removed_diverges_and_present_saturates() {
    // WITHOUT the collapse rule: strictly growing node counts — every
    // added iteration mints a fresh generation of views-of-views.
    let mut without = Vec::new();
    for iters in [40usize, 60, 80, 100] {
        let n = node_count(&bounded_program(iters, false));
        println!("collapse REMOVED, run {iters}: {n} nodes");
        without.push(n);
    }
    for pair in without.windows(2) {
        assert!(
            pair[1] > pair[0],
            "without the collapse rule the count must grow every added \
             iteration block (got {without:?}) — if this ever plateaus, the \
             divergence closed some other way and the collapse rule may be \
             re-litigated"
        );
    }

    // WITH the collapse rule: the same K range is DEEP saturation — the
    // count reaches its fixed point and stays flat.
    let mut with = Vec::new();
    for iters in [60usize, 80, 100] {
        let n = node_count(&bounded_program(iters, true));
        println!("collapse PRESENT, run {iters}: {n} nodes");
        with.push(n);
    }
    assert!(
        with.windows(2).all(|p| p[0] == p[1]),
        "with the collapse rule the main ruleset must saturate inside the \
         probe range (got {with:?})"
    );
}
