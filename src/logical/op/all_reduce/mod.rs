use crate::snapshot::compact::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The ordered-group all-reduce: `out = sum over the members of x`, in `x`'s
/// own dtype, with the order of the additions unspecified. That is c10d's
/// `all_reduce` contract (NCCL, gloo and TorchNeuron sum a bf16 tensor in
/// bf16, rounding at each step of their algorithm), so a bf16 operand is not
/// refused as narrow-float arithmetic: its rounding is the op's definition,
/// not a backend's choice. The group is a descriptor over logical ranks and
/// the sequence an effect identity, both non-tensor children; a backend
/// lowers the op to its reduction collective. Nothing rewrites it.
#[derive(Debug, Clone, Copy)]
pub struct LogicalAllReduce;

impl LogicalOp for LogicalAllReduce {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalAllReduce"
    }

    fn display_name(&self) -> &'static str {
        "all_reduce"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "all_reduce({}, group={}, width={}, seq={})",
            ctx.child_expr(node, 0),
            ctx.child_short(node, 1, 2, None).unwrap_or_else(|| "?".to_string()),
            ctx.child_short(node, 2, 2, None).unwrap_or_else(|| "?".to_string()),
            ctx.child_short(node, 3, 2, None).unwrap_or_else(|| "?".to_string()),
        )
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

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    /// The sum keeps the operand's shape, dtype (bf16 included) and a
    /// right-major layout; two sequences over the same operand stay distinct
    /// classes.
    #[test]
    fn an_all_reduce_keeps_its_operand_and_never_unions_across_sequences() {
        let script = r#"
(let xshape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") xshape (Bf16)))
(let x_layout (LayoutTensorLit x (RightMajorContiguousElementLayoutLit xshape (bits-of (Bf16)))))
(let a (LogicalAllReduce x "{\"groups\":[[0,1]],\"world\":2}" 2 0))
(let b (LogicalAllReduce x "{\"groups\":[[0,1]],\"world\":2}" 2 1))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= (shape-of a) xshape))
(check (= (dtype-of a) (Bf16)))
(check (LayoutTensorLit a (RightMajorContiguousElementLayoutLit (shape-of a) (bits-of (Bf16)))))
(fail (check (= a b)))
"#;
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
    }
}
