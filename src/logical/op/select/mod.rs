use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Ternary selection: `LogicalSelect(cond, if_true, if_false)` picks
/// elementwise from the two value branches by the boolean condition. The
/// output takes the branches' shape and dtype.
#[derive(Debug, Clone, Copy)]
pub struct LogicalSelect;

impl LogicalOp for LogicalSelect {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalSelect"
    }

    fn display_name(&self) -> &'static str {
        "select"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("condition", 0), ("if_true", 1), ("if_false", 2)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalSelect({}, {}, {})",
            ctx.child_expr(node, 0),
            ctx.child_expr(node, 1),
            ctx.child_expr(node, 2)
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
                category: SpliceCategory::Rewrites,
                text: include_str!("value_bounds.egg"),
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

    const SCHEDULE: &str = "(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

    /// A select whose condition the value bounds decide is that branch: a
    /// comparison of a zero iota against a one iota is true everywhere, the
    /// reverse false everywhere, and a comparison against a position iota is
    /// neither.
    #[test]
    fn a_select_on_a_decided_condition_is_that_branch() {
        new_egraph()
            .parse_and_run_program(
                None,
                &format!(
                    r#"{}
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape (F32)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape (F32)))
(let zero (LogicalIota (IntLit 0) shape))
(let one (LogicalIota (IntLit 1) shape))
(let position (LogicalIota (CoordVar shape 0) shape))
(let decided_true (LogicalSelect (LogicalLessThan zero one) a b))
(let decided_false (LogicalSelect (LogicalLessThan one zero) a b))
(let undecided (LogicalSelect (LogicalLessThan position one) a b))
{SCHEDULE}
(check (= decided_true a))
(check (= decided_false b))
(fail (check (= undecided a)))
(fail (check (= undecided b)))
"#,
                    assembled_program_for(&[])
                ),
            )
            .unwrap();
    }

    /// A view of a select is decided over what the view reads: a window
    /// into a position iota compared against a constant is true over one
    /// window and false over another, and each window of the select is the
    /// window of that branch, while the whole select stays a select.
    #[test]
    fn a_view_of_a_select_decided_over_the_view_is_the_view_of_the_branch() {
        new_egraph()
            .parse_and_run_program(
                None,
                &format!(
                    r#"{}
(let shape (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
(let window (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape (F32)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape (F32)))
(let position (LogicalIota (CoordVar shape 0) shape))
(let four (LogicalIota (IntLit 4) shape))
(let select (LogicalSelect (LogicalLessThan position four) a b))
(let low (IndexMapLit (IntExprCons (CoordVar window 0) (IntExprNil)) shape))
(let high (IndexMapLit (IntExprCons (IntAdd (CoordVar window 0) (IntLit 5)) (IntExprNil)) shape))
(let across (IndexMapLit (IntExprCons (IntAdd (CoordVar window 0) (IntLit 3)) (IntExprNil)) shape))
(let low_view (LogicalIndexMapApply select low window))
(let high_view (LogicalIndexMapApply select high window))
(let across_view (LogicalIndexMapApply select across window))
{SCHEDULE}
(check (= low_view (LogicalIndexMapApply a low window)))
(check (= high_view (LogicalIndexMapApply b high window)))
(fail (check (= across_view (LogicalIndexMapApply a across window))))
(fail (check (= across_view (LogicalIndexMapApply b across window))))
(fail (check (= select a)))
"#,
                    assembled_program_for(&[])
                ),
            )
            .unwrap();
    }
}
