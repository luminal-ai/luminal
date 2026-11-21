use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dtype-changing materializer: the target dtype rides in the term.
#[derive(Debug, Clone, Copy)]
pub struct LogicalCast;

impl LogicalOp for LogicalCast {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalCast"
    }

    fn display_name(&self) -> &'static str {
        "cast"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let dtype = ctx
            .child_short(node, 1, 2, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalCast({input}, dtype={dtype})")
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite_view.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("forward_layout.egg"),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    #[test]
    fn views_preserve_cast_chains_without_crossing_arithmetic() {
        let script = r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 5) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let y (LogicalTensorInputLit (LogicalIdLit "y") shape (F32)))
(let narrow (LogicalCast x (Bf16)))
(let wide (LogicalCast narrow (F32)))
(let transposed (LogicalHelperMatrixTranspose wide))
(let tx (LogicalHelperMatrixTranspose x))
(let rounded_product (LogicalCast (LogicalMul x y) (Bf16)))
(let narrow_product (LogicalMul narrow (LogicalCast y (Bf16))))
(let narrow_sum (LogicalCast (LogicalReduceSum narrow 0) (F32)))
(let wide_sum (LogicalReduceSum wide 0))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= transposed (LogicalCast (LogicalCast tx (Bf16)) (F32))))
(check (= (dtype-of transposed) (F32)))
(fail (check (= transposed tx)))
(fail (check (= rounded_product narrow_product)))
(fail (check (= narrow_sum wide_sum)))
"#;
        for dtype in ["Bf16", "TF32"] {
            let script = script.replace("(Bf16)", &format!("({dtype})"));
            new_egraph()
                .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
                .unwrap();
        }
    }

    const SCHEDULE: &str = "(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

    fn run(script: &str) {
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap_or_else(|error| panic!("cast script: {error}"));
    }

    #[test]
    fn a_cast_to_the_operand_dtype_is_the_operand() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let same (LogicalCast x (F32)))
(let other (LogicalCast x (Bf16)))
{SCHEDULE}
(check (= same x))
(fail (check (= other x)))
"#
        ));
    }

    /// Int -> Int64 -> Int is the source wherever the source is proven to
    /// lie in the Int range, and nowhere else: an exact add of two
    /// unbounded Int inputs keeps its honest interval, which a union with
    /// the checked narrowing cast would replace by the Int range.
    #[test]
    fn int32_widening_round_trip_collapses_only_onto_a_proven_source() {
        let inputs = r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape (Int)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape (Int)))
"#;
        // A stored Int input, read through a view.
        run(&format!(
            r#"
(let stored (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let turned (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 2) (IntExprNil)))))
(let ids (LogicalTensorInputLit (LogicalIdLit "ids") stored (Int)))
(let map (IndexMapLit (IntExprCons (CoordVar turned 1) (IntExprCons (CoordVar turned 0) (IntExprNil))) stored))
(let view (LogicalIndexMapApply ids map turned))
(let trip (LogicalCast (LogicalCast view (Int64)) (Int)))
(let wide (LogicalTensorInputLit (LogicalIdLit "wide") stored (Int64)))
(let lossy (LogicalCast (LogicalCast wide (Int)) (Int64)))
{SCHEDULE}
(check (= trip view))
(fail (check (= lossy wide)))
"#
        ));
        // An exact add nothing bounds: no union, and its interval stays honest.
        run(&format!(
            r#"{inputs}
(let sum (LogicalAdd a b))
(let trip (LogicalCast (LogicalCast sum (Int64)) (Int)))
{SCHEDULE}
(fail (check (= trip sum)))
(check (= (value-upper-bound-of sum) (bigint 4294967294)))
"#
        ));
        // The same add with operands proven small.
        run(&format!(
            r#"{inputs}
(set (value-lower-bound-of a) (bigint -100)) (set (value-upper-bound-of a) (bigint 100))
(set (value-lower-bound-of b) (bigint -100)) (set (value-upper-bound-of b) (bigint 100))
(let sum (LogicalAdd a b))
(let trip (LogicalCast (LogicalCast sum (Int64)) (Int)))
{SCHEDULE}
(check (= trip sum))
"#
        ));
    }

    /// An Int64 -> Int cast keeps its source bounds only when every source
    /// value fits in Int. A wider range can wrap on the device (2^32 stores as
    /// 0), so the result is just the Int range.
    #[test]
    fn int64_to_int_keeps_source_bounds_only_when_they_fit() {
        const INT_MIN: i64 = -2_147_483_648;
        const INT_MAX: i64 = 2_147_483_647;
        // (source lower, source upper, result lower, result upper)
        let cases: [(i64, i64, i64, i64); 8] = [
            // The whole range fits.
            (1, 100, 1, 100),
            (1, INT_MAX, 1, INT_MAX),
            (INT_MIN, INT_MAX, INT_MIN, INT_MAX),
            // One value past either end.
            (1, INT_MAX + 1, INT_MIN, INT_MAX),
            (INT_MIN - 1, 100, INT_MIN, INT_MAX),
            // Up to 2^32, and exactly 2^32.
            (1, 4_294_967_296, INT_MIN, INT_MAX),
            (4_294_967_296, 4_294_967_296, INT_MIN, INT_MAX),
            // Entirely outside the Int range.
            (4_294_967_297, 4_294_967_300, INT_MIN, INT_MAX),
        ];
        for (lo, hi, want_lo, want_hi) in cases {
            run(&format!(
                r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let wide (LogicalTensorInputLit (LogicalIdLit "wide") shape (Int64)))
(set (value-lower-bound-of wide) (bigint {lo}))
(set (value-upper-bound-of wide) (bigint {hi}))
(let narrow (LogicalCast wide (Int)))
{SCHEDULE}
(check (= (value-lower-bound-of narrow) (bigint {want_lo})))
(check (= (value-upper-bound-of narrow) (bigint {want_hi})))
"#
            ));
        }
    }

    /// A cast of a view is also the same view of the cast parent when the
    /// view keeps the parent's element count; the commuted term is named
    /// only inside the check, so nothing but the rule can have built it.
    #[test]
    fn a_cast_commutes_with_a_view_of_the_same_element_count() {
        run(&format!(
            r#"
(let inner (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let outer (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 2) (IntExprNil)))))
(let weight (LogicalTensorInputLit (LogicalIdLit "weight") inner (Bf16)))
(let map (IndexMapLit (IntExprCons (CoordVar outer 1) (IntExprCons (CoordVar outer 0) (IntExprNil))) inner))
(let cast (LogicalCast (LogicalIndexMapApply weight map outer) (F32)))
{SCHEDULE}
(check (= cast (LogicalIndexMapApply (LogicalCast weight (F32)) map outer)))
"#
        ));
        // A view that reads two of four elements: the parent is not cast whole.
        run(&format!(
            r#"
(let inner (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let outer (ShapeLit (IntExprCons (IntLit 2) (IntExprNil))))
(let weight (LogicalTensorInputLit (LogicalIdLit "weight") inner (Bf16)))
(let map (IndexMapLit (IntExprCons (CoordVar outer 0) (IntExprNil)) inner))
(let cast (LogicalCast (LogicalIndexMapApply weight map outer) (F32)))
{SCHEDULE}
(fail (check (= cast (LogicalIndexMapApply (LogicalCast weight (F32)) map outer))))
"#
        ));
    }
}
