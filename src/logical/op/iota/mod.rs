use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The index-expression generator: `out[c0..ck] = expr(c0..ck)` — a
/// zero-input source whose expr and shape are non-tensor children.
#[derive(Debug, Clone, Copy)]
pub struct LogicalIota;

impl LogicalOp for LogicalIota {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalIota"
    }

    fn display_name(&self) -> &'static str {
        "iota"
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let expr = ctx
            .child_int_expr(node, 0)
            .unwrap_or_else(|| "?".to_string());
        let shape = ctx.child_shape(node, 1).unwrap_or_else(|| "?".to_string());
        format!("LogicalIota(expr={expr}, shape={shape})")
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
