use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Elementwise reciprocal.
#[derive(Debug, Clone, Copy)]
pub struct LogicalRecip;

impl LogicalOp for LogicalRecip {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalRecip"
    }

    fn display_name(&self) -> &'static str {
        "recip"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!("LogicalRecip({})", ctx.child_expr(node, 0))
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
                text: include_str!("rewrite_view.egg"),
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

    /// The reciprocal of a broadcast is the broadcast of the reciprocal, at
    /// the broadcast's shape and the operand's dtype; the reciprocal of a
    /// transpose likewise. Arithmetic is not crossed: a product's
    /// reciprocal stays one.
    #[test]
    fn a_reciprocal_of_a_view_is_the_view_of_the_reciprocal() {
        let script = r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 5) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let y (LogicalTensorInputLit (LogicalIdLit "y") shape (F32)))
(let wide_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprCons (IntLit 5) (IntExprNil))))))
(let recip_of_broadcast (LogicalRecip (LogicalHelperBroadcastAxis x 1 (IntLit 4))))
(let broadcast_of_recip (LogicalHelperBroadcastAxis (LogicalRecip x) 1 (IntLit 4)))
(let recip_of_transpose (LogicalRecip (LogicalHelperMatrixTranspose x)))
(let transpose_of_recip (LogicalHelperMatrixTranspose (LogicalRecip x)))
(let recip_of_product (LogicalRecip (LogicalMul x y)))
(let product_of_recips (LogicalMul (LogicalRecip x) (LogicalRecip y)))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= recip_of_broadcast broadcast_of_recip))
(check (= (dtype-of recip_of_broadcast) (F32)))
(check (= (shape-of recip_of_broadcast) wide_shape))
(check (= recip_of_transpose transpose_of_recip))
(fail (check (= recip_of_product product_of_recips)))
"#;
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
    }
}
