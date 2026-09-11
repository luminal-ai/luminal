use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The rank-2 transpose view (see `constructor.egg`).
#[derive(Debug, Clone, Copy)]
pub struct LogicalHelperMatrixTranspose;

impl LogicalOp for LogicalHelperMatrixTranspose {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalHelperMatrixTranspose"
    }

    fn display_name(&self) -> &'static str {
        "matrix_transpose"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!("LogicalHelperMatrixTranspose({})", ctx.child_expr(node, 0))
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
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("involution.egg"),
            },
        ]
    }
}
