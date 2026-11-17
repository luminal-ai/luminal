use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Elementwise multiplication.
#[derive(Debug, Clone, Copy)]
pub struct LogicalMul;

impl LogicalOp for LogicalMul {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalMul"
    }

    fn display_name(&self) -> &'static str {
        "mul"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("lhs", 0), ("rhs", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalMul({}, {})",
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
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite_3.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite_one.egg"),
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

    /// torch's `x / 1.0` over a bf16 tensor, as opmath records it:
    /// `cast(cast(x, F32) * recip(expand(cast(cast(1.0, Bf16), F32))), Bf16)`
    /// is `x`. Exact, rounding included: the reciprocal of 1 is 1 in every
    /// float dtype; multiplying by 1.0 is the identity in the opmath type
    /// (F32), for every value (signed zeros, infinities, subnormals; a NaN
    /// stays a NaN); widening bf16 to F32 is exact; and the final rounding
    /// to bf16 rounds a value bf16 already represents, which is the identity.
    /// The same holds for an F32 product by a one directly. A product by 2.0
    /// is not folded, nor is a bf16 value rounded through F16.
    #[test]
    fn a_product_by_one_is_its_operand_rounding_included() {
        let mut graph = new_egraph();
        graph.parse_and_run_program(None, &assembled_program_for(&[])).unwrap();
        graph
            .parse_and_run_program(
                None,
                r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 8) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (Bf16)))
(let y (LogicalTensorInputLit (LogicalIdLit "y") shape (F32)))
(let scalar (LogicalCast (LogicalCast (LogicalConstant 1.0) (Bf16)) (F32)))
(let ones (LogicalIndexMapApply scalar (IndexMapLit (IntExprNil) (ShapeLit (IntExprNil))) shape))
(let divided (LogicalCast (LogicalMul (LogicalCast x (F32)) (LogicalRecip ones)) (Bf16)))
(let scaled (LogicalMul y ones))
(let scaled_left (LogicalMul ones y))
(let twos (LogicalIndexMapApply (LogicalConstant 2.0) (IndexMapLit (IntExprNil) (ShapeLit (IntExprNil))) shape))
(let doubled (LogicalMul y twos))
(let through_f16 (LogicalCast (LogicalCast x (F32)) (F16)))
(run-schedule (saturate (run main_ruleset) (run prop)))
(check (= divided x))
(check (= scaled y))
(check (= scaled_left y))
(fail (check (= doubled y)))
(fail (check (= through_f16 x)))
"#,
            )
            .unwrap();
    }
}
