use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Elementwise comparison; the result is Bool.
#[derive(Debug, Clone, Copy)]
pub struct LogicalLessThan;

impl LogicalOp for LogicalLessThan {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalLessThan"
    }

    fn display_name(&self) -> &'static str {
        "less_than"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("lhs", 0), ("rhs", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalLessThan({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1)
        )
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
                category: SpliceCategory::Dtype,
                text: include_str!("dtype_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("value_bounds.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("forward_layout.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("fixpoint.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("fixpoint_2.egg"),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    const SCHEDULE: &str = "(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

    fn run(script: &str) {
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap_or_else(|error| panic!("less_than script: {error}"));
    }

    /// `position < 0` over positions 0..16 is false everywhere, and
    /// `position < 16` is true everywhere. Both stay decided through the
    /// indicator cast, so arithmetic on the indicator (the `-(x < 0)`
    /// correction of a floor division) is bounded exactly.
    #[test]
    fn an_integer_comparison_with_disjoint_intervals_is_decided() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 16) (IntExprNil))))
(let position (LogicalIota (CoordVar shape 0) shape))
(let zero (LogicalIota (IntLit 0) shape))
(let extent (LogicalIota (IntLit 16) shape))
(let never (LogicalLessThan position zero))
(let always (LogicalLessThan position extent))
(let never_int (LogicalCast never (Int)))
(let never_wide (LogicalCast never (Int64)))
(let always_int (LogicalCast always (Int)))
(let always_wide (LogicalCast always (Int64)))
{SCHEDULE}
(check (= (value-lower-bound-of never) (bigint 0)))
(check (= (value-upper-bound-of never) (bigint 0)))
(check (= (value-lower-bound-of always) (bigint 1)))
(check (= (value-upper-bound-of always) (bigint 1)))
(check (= (value-upper-bound-of never_int) (bigint 0)))
(check (= (value-upper-bound-of never_wide) (bigint 0)))
(check (= (value-lower-bound-of always_int) (bigint 1)))
(check (= (value-lower-bound-of always_wide) (bigint 1)))
"#
        ));
    }

    /// Intervals that overlap, or only touch from below, decide nothing:
    /// the comparison is the plain indicator [0, 1].
    #[test]
    fn an_integer_comparison_with_overlapping_intervals_is_the_indicator() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 16) (IntExprNil))))
(let position (LogicalIota (CoordVar shape 0) shape))
(let two (LogicalIota (IntLit 2) shape))
(let last (LogicalIota (IntLit 15) shape))
(let mixed (LogicalLessThan position two))
(let touching (LogicalLessThan position last))
(let mixed_int (LogicalCast mixed (Int)))
{SCHEDULE}
(check (= (value-lower-bound-of mixed) (bigint 0)))
(check (= (value-upper-bound-of mixed) (bigint 1)))
(check (= (value-lower-bound-of touching) (bigint 0)))
(check (= (value-upper-bound-of touching) (bigint 1)))
(check (= (value-lower-bound-of mixed_int) (bigint 0)))
(check (= (value-upper-bound-of mixed_int) (bigint 1)))
"#
        ));
    }

    /// A float interval says nothing about NaN, which compares false
    /// against everything, so a float comparison gets no bounds at all,
    /// even between operands whose intervals are disjoint.
    #[test]
    fn a_float_comparison_is_not_bounded() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 16) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape (F32)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape (F32)))
(set (value-lower-bound-of a) (bigint 0)) (set (value-upper-bound-of a) (bigint 3))
(set (value-lower-bound-of b) (bigint 5)) (set (value-upper-bound-of b) (bigint 9))
(let less (LogicalLessThan a b))
{SCHEDULE}
(check (= (dtype-of less) (Bool)))
(check (= ?operand_lower (value-lower-bound-of a)))
(fail (check (= ?lower (value-lower-bound-of less))))
(fail (check (= ?upper (value-upper-bound-of less))))
"#
        ));
    }
}
