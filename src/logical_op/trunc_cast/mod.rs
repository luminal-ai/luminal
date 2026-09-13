use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The explicit lossy float -> integer conversion: truncate toward zero.
/// Kept separate from [`crate::logical_op::LogicalCast`], which is
/// lossless-only by policy.
#[derive(Debug, Clone, Copy)]
pub struct LogicalTruncCast;

impl LogicalOp for LogicalTruncCast {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalTruncCast"
    }

    fn display_name(&self) -> &'static str {
        "trunc_cast"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let dtype = ctx
            .child_short(node, 1, 2, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalTruncCast({input}, dtype={dtype})")
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
