use super::{LogicalOp, LogicalRender};
use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use egraph_serialize::Node;

/// Rank-local input stacked in communicator order. Group and sequence are part
/// of value identity, so separate collective calls cannot be merged.
#[derive(Debug, Clone, Copy)]
pub struct LogicalAllGather;
impl LogicalOp for LogicalAllGather {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalAllGather"
    }
    fn display_name(&self) -> &'static str {
        "AllGather"
    }
    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }
    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!("AllGather({})", ctx.child_expr(node, 0))
    }
    fn snippets(&self) -> Vec<EgglogSnippet> {
        vec![
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: "(rule ((= ?out (LogicalAllGather ?x ?group ?width ?seq)) (= ?shape (shape-of ?out)) (= ?bits (bits-of (dtype-of ?out)))) ((LayoutTensorLit ?out (RightMajorContiguousElementLayoutLit ?shape ?bits))) :ruleset main_ruleset)",
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: "(rule ((= ?out (LogicalAllGather ?x ?group ?width ?seq)) (LayoutTensorLit ?x (RightMajorContiguousElementLayoutLit ?in_shape ?bits)) (= ?shape (shape-of ?out))) ((LayoutTensorLit ?out (RightMajorContiguousElementLayoutLit ?shape ?bits))) :ruleset main_ruleset)",
            },
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: "(constructor LogicalAllGather (LogicalTensor String i64 i64) LogicalTensor)",
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: "(rule ((= ?out (LogicalAllGather ?x ?group ?width ?seq)) (= ?dtype (dtype-of ?x))) ((set (dtype-of ?out) ?dtype)) :ruleset prop)",
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: "(rule ((= ?out (LogicalAllGather ?x ?group ?width ?seq)) (= (shape-of ?x) (ShapeLit ?dims))) ((set (shape-of ?out) (ShapeLit (IntExprCons (IntLit ?width) ?dims)))) :ruleset prop)",
            },
        ]
    }
}
