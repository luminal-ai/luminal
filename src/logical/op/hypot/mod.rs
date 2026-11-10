use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Elementwise Euclidean norm with authored runtime semantics.
#[derive(Debug, Clone, Copy)]
pub struct LogicalHypot;

impl LogicalOp for LogicalHypot {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalHypot"
    }

    fn display_name(&self) -> &'static str {
        "hypot"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("lhs", 0), ("rhs", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalHypot({}, {})",
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
