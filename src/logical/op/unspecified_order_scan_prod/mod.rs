use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Inclusive product scan along one axis; the evaluation order is
/// unspecified, so any bracketing is a valid implementation.
///
/// No value-bounds rule: a prefix product needs a `pow` on BigInt, which the
/// lattice has no primitive for.
#[derive(Debug, Clone, Copy)]
pub struct LogicalUnspecifiedOrderScanProd;

impl LogicalOp for LogicalUnspecifiedOrderScanProd {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalUnspecifiedOrderScanProd"
    }

    fn display_name(&self) -> &'static str {
        "unspecified_order_scan_prod"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let axis = ctx
            .child_short(node, 1, 4, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalUnspecifiedOrderScanProd(input={input}, axis={axis})")
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
