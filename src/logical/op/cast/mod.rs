use egraph_serialize::Node;

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
}
