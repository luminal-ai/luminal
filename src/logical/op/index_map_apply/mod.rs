use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// A view: apply an index map to a logical tensor (index map and shape are
/// non-tensor children, not ports).
#[derive(Debug, Clone, Copy)]
pub struct LogicalIndexMapApply;

impl LogicalOp for LogicalIndexMapApply {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalIndexMapApply"
    }

    fn display_name(&self) -> &'static str {
        "index_map_apply"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let index_map = ctx
            .child_index_map(node, 1)
            .unwrap_or_else(|| "?".to_string());
        let shape = ctx.child_shape(node, 2).unwrap_or_else(|| "?".to_string());
        format!("LogicalIndexMapApply(input={input}, index_map={index_map}, shape={shape})")
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
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("fixpoint.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("fixpoint_2.egg"),
            },
        ]
    }
}
