use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Coordinate-form gather: `out[c] = data[coord_0[c], .., coord_{r-1}[c]]`.
/// The coordinate tensors ride behind the list child, so only `data` is a
/// direct port; the coordinates surface through the readable expression.
#[derive(Debug, Clone, Copy)]
pub struct LogicalGather;

impl LogicalOp for LogicalGather {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalGather"
    }

    fn display_name(&self) -> &'static str {
        "gather"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("data", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let data = ctx.child_expr(node, 0);
        let coords = ctx
            .child_short(node, 1, 6, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalGather(data={data}, coords={coords})")
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
            EgglogSnippet {
                category: SpliceCategory::Seed,
                text: include_str!("seed_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Coordinate,
                text: include_str!("unification.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Coordinate,
                text: include_str!("unification_2.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Coordinate,
                text: include_str!("unification_3.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Coordinate,
                text: include_str!("unification_4.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: include_str!("fixpoint.egg"),
            },
        ]
    }
}
