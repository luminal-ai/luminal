//! THE VIEW-ARITY TRIPWIRE, pinned from both sides (ruling 2026-09-01).
//!
//! The tripwire asserts "an index map's entry count is the rank of its
//! source-shape tag". It is a property of the `IndexMapLit`, and it is
//! checked on the literal. It used to be a `:no-merge` cell keyed on the
//! apply's OUTPUT e-class, which silently also asserted "every apply
//! spelling in one class has a same-rank parent" — a route fact stored
//! in a value slot. That fired on a sound equivalence the moment any
//! rule unioned two different-parent-rank spellings (the cuBLASLt
//! marker's double-transpose collapse was the first). These tests keep
//! both halves honest:
//!
//!  * the SOUND union must saturate — `x2 = x3[0,:,:]` (3 entries over a
//!    rank-3 parent) and `y = identity(x2)` (2 entries over a rank-2
//!    parent) are the same tensor by definition, with ONE output shape;
//!  * the GENUINE arity error must still fail loudly, naming the literal.
//!
//! Every run assembles the program itself so the egglog error is
//! captured as text rather than panicking through `serialize_fixture`.

use luminal::layout_ir::OpMatcher;

const FULL: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";
const NO_INVARIANTS: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata))";

/// This runtime's vocabulary WITHOUT the cuBLASLt marker estate — the
/// tripwire's behaviour must not depend on any backend's rules.
fn matchers_without_marker() -> Vec<Box<dyn OpMatcher>> {
    let mut m = test_runtime::ops::functional::functional_matchers();
    m.push(Box::new(test_runtime::IndexMapApplyViewMatcher));
    m.push(Box::new(test_runtime::AddMulFusedMatcher));
    m.extend(test_runtime::ops::mutating::mutating_matchers());
    m
}

fn run(matchers: &[Box<dyn OpMatcher>], script: &str) -> Result<(), String> {
    let preamble = luminal::egglog_snippet::assembled_program_for(matchers);
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &format!("{preamble}\n\n{script}"))
        .map(|_| ())
        .map_err(|e| format!("{e}"))
}

/// x3 : [2,3,4]. x2 = x3[0,:,:] — 3-entry map over the RANK-3 parent, out [3,4].
/// y = x2 through the identity map — 2-entry map over the RANK-2 parent, out [3,4].
/// Entries are parent-outermost-first; `(CoordVar shape k)` is output coordinate
/// k-from-the-end (axis 0 = innermost).
const TWO_SPELLINGS: &str = r#"
(let s3  (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let s34 (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x3 (LogicalTensorInputLit (LogicalIdLit "x3") s3 (F32)))
(let select_map (IndexMapLit (IntExprCons (IntLit 0) (IntExprCons (CoordVar s34 1) (IntExprCons (CoordVar s34 0) (IntExprNil)))) s3))
(let x2 (LogicalIndexMapApply x3 select_map s34))
(let identity_map (IndexMapLit (IntExprCons (CoordVar s34 1) (IntExprCons (CoordVar s34 0) (IntExprNil))) s34))
(let y (LogicalIndexMapApply x2 identity_map s34))
(let x3_lt (LayoutTensorLit x3 (RightMajorContiguousElementLayoutLit s3 (bits-of (F32)))))
(let x3_buf (BufferLit 1))
(set (buffer-access-of x3_buf) (ReadOnly))
(set (buffer-freed-by x3_buf) (CallerFrees))
(let input_boundary (BufferInputLit (BufferTensorCons (BufferTensorLit x3_lt x3_buf) (BufferTensorNil))))
(let y_lt (LayoutTensorLit y (RightMajorContiguousElementLayoutLit s34 (bits-of (F32)))))
(let y_buf (BufferLit 2))
(set (buffer-access-of y_buf) (ReadWrite))
(set (buffer-freed-by y_buf) (CallerFrees))
(let output (BufferOutputLit (BufferTensorCons (BufferTensorLit y_lt y_buf) (BufferTensorNil))))
"#;

/// The equivalence is sound and rank is a property of the VALUE: after
/// the union both spellings have output shape [3,4], rank 2, one class.
#[test]
fn two_spellings_of_one_value_share_one_shape_and_one_rank() {
    let script = format!(
        "{TWO_SPELLINGS}\n(union y x2)\n{NO_INVARIANTS}\n\
         (check (= (shape-of x2) s34))\n(check (= (shape-of y) s34))\n\
         (check (= (rank-of (shape-of x2)) 2))\n(check (= x2 y))\n"
    );
    let r = run(&matchers_without_marker(), &script);
    assert!(
        r.is_ok(),
        "the union of two spellings of one value is sound: {r:?}"
    );
}

/// THE MISFIRE THAT WAS: a sound union of two apply spellings whose
/// parents have different rank must saturate under the full schedule.
#[test]
fn sound_union_of_different_parent_rank_spellings_saturates() {
    let script = format!("{TWO_SPELLINGS}\n(union y x2)\n{FULL}\n");
    let r = run(&matchers_without_marker(), &script);
    assert!(
        r.is_ok(),
        "a route fact (parent rank) must never be asserted per value class: {r:?}"
    );
}

/// The collision as it actually arose: the cuBLASLt marker's
/// double-transpose collapse unions a view-of-view back into a tensor that
/// is itself a view from a rank-3 parent. With the marker vocabulary
/// loaded, this must saturate too.
#[test]
fn marker_double_transpose_collapse_over_a_rank_changing_view_saturates() {
    let script = r#"
(let s3  (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let s34 (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let s43 (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x3 (LogicalTensorInputLit (LogicalIdLit "x3") s3 (F32)))
(let select_map (IndexMapLit (IntExprCons (IntLit 0) (IntExprCons (CoordVar s34 1) (IntExprCons (CoordVar s34 0) (IntExprNil)))) s3))
(let x2 (LogicalIndexMapApply x3 select_map s34))
(let tmap1 (IndexMapLit (IntExprCons (CoordVar s43 0) (IntExprCons (CoordVar s43 1) (IntExprNil))) s34))
(let t (LogicalIndexMapApply x2 tmap1 s43))
(let tmap2 (IndexMapLit (IntExprCons (CoordVar s34 0) (IntExprCons (CoordVar s34 1) (IntExprNil))) s43))
(let tt (LogicalIndexMapApply t tmap2 s34))
(let x3_lt (LayoutTensorLit x3 (RightMajorContiguousElementLayoutLit s3 (bits-of (F32)))))
(let x3_buf (BufferLit 1))
(set (buffer-access-of x3_buf) (ReadOnly))
(set (buffer-freed-by x3_buf) (CallerFrees))
(let input_boundary (BufferInputLit (BufferTensorCons (BufferTensorLit x3_lt x3_buf) (BufferTensorNil))))
(let tt_lt (LayoutTensorLit tt (RightMajorContiguousElementLayoutLit s34 (bits-of (F32)))))
(let tt_buf (BufferLit 2))
(set (buffer-access-of tt_buf) (ReadWrite))
(set (buffer-freed-by tt_buf) (CallerFrees))
(let output (BufferOutputLit (BufferTensorCons (BufferTensorLit tt_lt tt_buf) (BufferTensorNil))))
"#
    .to_string()
        + FULL
        + "\n(check (= tt x2))\n";
    // test_runtime::matchers() carries the marker estate.
    let r = run(&test_runtime::matchers(), &script);
    assert!(
        r.is_ok(),
        "the collapse's union is sound and must be admitted: {r:?}"
    );
}

/// THE ERROR THE TRIPWIRE EXISTS FOR: a 2-entry map tagged with (and
/// applied to) a rank-3 parent is malformed and must fail loudly,
/// naming the literal.
#[test]
fn genuine_arity_error_is_still_caught_loudly() {
    let script = format!(
        r#"
(let s3  (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let s34 (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x3 (LogicalTensorInputLit (LogicalIdLit "x3") s3 (F32)))
(let bad_map (IndexMapLit (IntExprCons (CoordVar s34 1) (IntExprCons (CoordVar s34 0) (IntExprNil))) s3))
(let bad (LogicalIndexMapApply x3 bad_map s34))
(let x3_lt (LayoutTensorLit x3 (RightMajorContiguousElementLayoutLit s3 (bits-of (F32)))))
(let x3_buf (BufferLit 1))
(set (buffer-access-of x3_buf) (ReadOnly))
(set (buffer-freed-by x3_buf) (CallerFrees))
(let input_boundary (BufferInputLit (BufferTensorCons (BufferTensorLit x3_lt x3_buf) (BufferTensorNil))))
(let bad_lt (LayoutTensorLit bad (RightMajorContiguousElementLayoutLit s34 (bits-of (F32)))))
(let bad_buf (BufferLit 2))
(set (buffer-access-of bad_buf) (ReadWrite))
(set (buffer-freed-by bad_buf) (CallerFrees))
(let output (BufferOutputLit (BufferTensorCons (BufferTensorLit bad_lt bad_buf) (BufferTensorNil))))
{FULL}
"#
    );
    let r = run(&matchers_without_marker(), &script);
    let msg = r.expect_err("a genuine arity error must be caught");
    assert!(
        msg.contains("IndexMapLit entry count disagrees with the rank of its source-shape tag"),
        "the failure must name the literal and the invariant: {msg}"
    );
}
