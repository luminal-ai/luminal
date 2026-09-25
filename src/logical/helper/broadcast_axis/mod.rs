use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// One new axis inserted at a from-end position (see `constructor.egg`).
/// Instances cover parents of rank 1 and 2 at every insertion position, plus
/// the rank-2 parent read transposed (the fused maps matmul operands record).
#[derive(Debug, Clone, Copy)]
pub struct LogicalHelperBroadcastAxis;

impl LogicalOp for LogicalHelperBroadcastAxis {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalHelperBroadcastAxis"
    }

    fn display_name(&self) -> &'static str {
        "broadcast_axis"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let axis = ctx
            .child_short(node, 1, 1, None)
            .unwrap_or_else(|| "?".to_string());
        let extent = ctx
            .child_int_expr(node, 2)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalHelperBroadcastAxis(input={input}, axis_from_end={axis}, extent={extent})")
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_rank1_pos0.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand_rank1_pos0.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_rank1_pos1.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand_rank1_pos1.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_rank2_pos0.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand_rank2_pos0.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_rank2_pos1.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand_rank2_pos1.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_rank2_pos2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("expand_rank2_pos2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_transposed_rank2_pos0.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_transposed_rank2_pos1.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("recognize_transposed_rank2_pos2.egg"),
            },
        ]
    }
}
