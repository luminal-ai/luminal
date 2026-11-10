use crate::snapshot::compact::Node;

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
                category: SpliceCategory::Rewrites,
                text: include_str!("compose.egg"),
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

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    #[test]
    fn bound_reshape_permute_chain_has_a_direct_root_view() {
        let script = r#"
(let root-shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 2) (IntExprNil))))))
(let mid-shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprNil))))))
(let out-shape root-shape)
(let root (LogicalTensorInputLit (LogicalIdLit "root") root-shape (F32)))

; reshape [2,3,2] -> [3,2,2]
(let mid-flat (IntAdd (IntMul (CoordVar mid-shape 2) (IntLit 4))
              (IntAdd (IntMul (CoordVar mid-shape 1) (IntLit 2))
                      (CoordVar mid-shape 0))))
(let mid-map (IndexMapLit
  (IntExprCons (IntTruncDiv mid-flat (IntLit 6))
    (IntExprCons (IntTruncRem (IntTruncDiv mid-flat (IntLit 2)) (IntLit 3))
      (IntExprCons (IntTruncRem mid-flat (IntLit 2)) (IntExprNil))))
  root-shape))
(let mid (LogicalIndexMapApply root mid-map mid-shape))

; permute [3,2,2] by [1,0,2] -> [2,3,2]
(let outer-map (IndexMapLit
  (IntExprCons (CoordVar out-shape 1)
    (IntExprCons (CoordVar out-shape 2)
      (IntExprCons (CoordVar out-shape 0) (IntExprNil))))
  mid-shape))
(let outer (LogicalIndexMapApply mid outer-map out-shape))
(let outer-layout (RightMajorContiguousElementLayoutLit out-shape (bits-of (F32))))
(let outer-layout-tensor (LayoutTensorLit outer outer-layout))
(let outer-buffer-tensor (BufferTensorLit outer-layout-tensor (BufferLit 7)))
(let output-boundary
  (BufferOutputLit (BufferTensorCons outer-buffer-tensor (BufferTensorNil))))

(let direct-flat (IntAdd (IntMul (CoordVar out-shape 1) (IntLit 4))
                 (IntAdd (IntMul (CoordVar out-shape 2) (IntLit 2))
                         (CoordVar out-shape 0))))
(let direct-map (IndexMapLit
  (IntExprCons (IntTruncDiv direct-flat (IntLit 6))
    (IntExprCons (IntTruncRem (IntTruncDiv direct-flat (IntLit 2)) (IntLit 3))
      (IntExprCons (IntTruncRem direct-flat (IntLit 2)) (IntExprNil))))
  root-shape))
(let direct (LogicalIndexMapApply root direct-map out-shape))

(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= outer direct))
"#;
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
    }
}
