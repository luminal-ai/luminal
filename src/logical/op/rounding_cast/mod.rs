use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The explicit rounding integer -> float conversion: round to nearest,
/// ties to even. Kept separate from [`crate::logical_op::LogicalCast`],
/// which is lossless-only by policy.
#[derive(Debug, Clone, Copy)]
pub struct LogicalRoundingCast;

impl LogicalOp for LogicalRoundingCast {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalRoundingCast"
    }

    fn display_name(&self) -> &'static str {
        "rounding_cast"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let dtype = ctx
            .child_short(node, 1, 2, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalRoundingCast({input}, dtype={dtype})")
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
                category: SpliceCategory::Shape,
                text: include_str!("shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("forward_layout.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite.egg"),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    /// A mean's count divisor: the rounding conversion of a literal F32 (or F64) represents
    /// exactly is the lossless cast; past 2^24 the F32 rounding stays its own op.
    #[test]
    fn an_exact_integer_literal_rounds_to_its_lossless_cast() {
        let mut graph = new_egraph();
        graph.parse_and_run_program(None, &assembled_program_for(&[])).unwrap();
        graph
            .parse_and_run_program(
                None,
                r#"
(let count (LogicalIota (IntLit 128) (ShapeLit (IntExprNil))))
(let big (LogicalIota (IntLit 16777217) (ShapeLit (IntExprNil))))
(let r32 (LogicalRoundingCast count (F32)))
(let r64 (LogicalRoundingCast big (F64)))
(let rbig (LogicalRoundingCast big (F32)))
(run-schedule (saturate (run main_ruleset) (run prop)))
(check (= r32 (LogicalCast count (F32))))
(check (= r64 (LogicalCast big (F64))))
(fail (check (= rbig (LogicalCast big (F32)))))
"#,
            )
            .unwrap();
    }
}
