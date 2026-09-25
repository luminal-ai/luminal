use egraph_serialize::Node;

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
