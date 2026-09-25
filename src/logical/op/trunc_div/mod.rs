use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Integer truncated division (toward zero) — a DIFFERENT mathematical function from float Div, mirroring the scalar IntTruncDiv (ruling 2026-08-11). Proof-gated: implemented only where the divisor's value bounds exclude zero and the quotient stays in width.
#[derive(Debug, Clone, Copy)]
pub struct LogicalTruncDiv;

impl LogicalOp for LogicalTruncDiv {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalTruncDiv"
    }

    fn display_name(&self) -> &'static str {
        "trunc div"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("numerator", 0), ("denominator", 1)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalTruncDiv({}, {})",
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
