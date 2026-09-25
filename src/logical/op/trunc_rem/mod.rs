use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Integer truncated remainder, the TruncDiv companion (sign follows the dividend). Proof-gated like TruncDiv.
#[derive(Debug, Clone, Copy)]
pub struct LogicalTruncRem;

impl LogicalOp for LogicalTruncRem {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalTruncRem"
    }

    fn display_name(&self) -> &'static str {
        "trunc rem"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("numerator", 0), ("denominator", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalTruncRem({}, {})",
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
        ]
    }
}
