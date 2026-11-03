use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// A view: apply an index map to a logical tensor (index map and shape are
/// non-tensor children, not ports).
#[derive(Debug, Clone, Copy)]
pub struct LogicalIndexMapApply;

impl LogicalOp for LogicalIndexMapApply {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalIndexMapApply"
    }

    fn display_name(&self) -> &'static str {
        "index_map_apply"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let index_map = ctx
            .child_index_map(node, 1)
            .unwrap_or_else(|| "?".to_string());
        let shape = ctx.child_shape(node, 2).unwrap_or_else(|| "?".to_string());
        format!("LogicalIndexMapApply(input={input}, index_map={index_map}, shape={shape})")
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
            .unwrap();
    }

    /// A view whose map is the box's own coordinates is its operand; the
    /// transpose of a square operand, whose entries are the coordinates in
    /// the other order, is not.
    #[test]
    fn a_view_through_the_identity_map_is_its_operand() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let identity (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar shape 1) (IntExprCons (CoordVar shape 0) (IntExprNil))) shape) shape))
(let transposed (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar shape 0) (IntExprCons (CoordVar shape 1) (IntExprNil))) shape) shape))
{SCHEDULE}
(check (= identity x))
(fail (check (= transposed x)))
"#
        ));
    }
}
