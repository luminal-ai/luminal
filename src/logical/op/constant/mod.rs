use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The scalar constant: one value, a true rank-0 tensor.
#[derive(Debug, Clone, Copy)]
pub struct LogicalConstant;

impl LogicalOp for LogicalConstant {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalConstant"
    }

    fn display_name(&self) -> &'static str {
        "constant"
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let value = ctx
            .child_short(node, 0, 2, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalConstant({value})")
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
