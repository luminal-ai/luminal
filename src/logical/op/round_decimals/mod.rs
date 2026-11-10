use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Elementwise decimal-place rounding; the decimal count is op metadata.
#[derive(Debug, Clone, Copy)]
pub struct LogicalRoundDecimals;

impl LogicalOp for LogicalRoundDecimals {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalRoundDecimals"
    }

    fn display_name(&self) -> &'static str {
        "round_decimals"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let decimals = ctx
            .child_short(node, 1, 4, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalRoundDecimals({input}, decimals={decimals})")
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
