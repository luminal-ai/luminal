use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Sum reduction along one axis (the axis is a non-tensor child, not a port).
#[derive(Debug, Clone, Copy)]
pub struct LogicalReduceSum;

impl LogicalOp for LogicalReduceSum {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalReduceSum"
    }

    fn display_name(&self) -> &'static str {
        "reduce_sum"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let axis = ctx
            .child_short(node, 1, 4, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalReduceSum(input={input}, axis={axis})")
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
                category: SpliceCategory::Seed,
                text: include_str!("seed.egg"),
            },
        ]
    }
}
