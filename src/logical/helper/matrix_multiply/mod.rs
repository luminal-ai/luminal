use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The rank-2 matrix product (see `constructor.egg`).
#[derive(Debug, Clone, Copy)]
pub struct LogicalHelperMatrixMultiply;

impl LogicalOp for LogicalHelperMatrixMultiply {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalHelperMatrixMultiply"
    }

    fn display_name(&self) -> &'static str {
        "matrix_multiply"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("a", 0), ("b", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalHelperMatrixMultiply({}, {})",
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
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand.egg"),
            },
        ]
    }
}
