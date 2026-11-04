use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// The dtype-changing materializer: the target dtype rides in the term.
#[derive(Debug, Clone, Copy)]
pub struct LogicalCast;

impl LogicalOp for LogicalCast {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalCast"
    }

    fn display_name(&self) -> &'static str {
        "cast"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let input = ctx.child_expr(node, 0);
        let dtype = ctx
            .child_short(node, 1, 2, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalCast({input}, dtype={dtype})")
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
                text: include_str!("rewrite.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite_view.egg"),
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

    #[test]
    fn views_preserve_cast_chains_without_crossing_arithmetic() {
        let script = r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 5) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let y (LogicalTensorInputLit (LogicalIdLit "y") shape (F32)))
(let narrow (LogicalCast x (Bf16)))
(let wide (LogicalCast narrow (F32)))
(let transposed (LogicalHelperMatrixTranspose wide))
(let tx (LogicalHelperMatrixTranspose x))
(let rounded_product (LogicalCast (LogicalMul x y) (Bf16)))
(let narrow_product (LogicalMul narrow (LogicalCast y (Bf16))))
(let narrow_sum (LogicalCast (LogicalReduceSum narrow 0) (F32)))
(let wide_sum (LogicalReduceSum wide 0))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= transposed (LogicalCast (LogicalCast tx (Bf16)) (F32))))
(check (= (dtype-of transposed) (F32)))
(fail (check (= transposed tx)))
(fail (check (= rounded_product narrow_product)))
(fail (check (= narrow_sum wide_sum)))
"#;
        for dtype in ["Bf16", "TF32"] {
            let script = script.replace("(Bf16)", &format!("({dtype})"));
            new_egraph()
                .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
                .unwrap();
        }
    }

    /// A constant's round trip through a narrower dtype is the constant
    /// exactly when the narrow dtype holds it: `1.0`, a power of two and a
    /// small integer do; the GELU's `0.044715` does not in either; `257.0`
    /// (nine significant bits) fits f16's eleven, not bf16's eight.
    #[test]
    fn an_exact_constants_round_trip_through_a_narrow_dtype_is_the_constant() {
        let script = r#"
(let one (LogicalConstant 1.0))
(let one_bf16 (LogicalCast (LogicalCast one (Bf16)) (F32)))
(let one_f16 (LogicalCast (LogicalCast one (F16)) (F32)))
(let tiny (LogicalConstant 0.00000095367431640625))
(let tiny_bf16 (LogicalCast (LogicalCast tiny (Bf16)) (F32)))
(let gelu (LogicalConstant 0.044715))
(let gelu_bf16 (LogicalCast (LogicalCast gelu (Bf16)) (F32)))
(let gelu_f16 (LogicalCast (LogicalCast gelu (F16)) (F32)))
(let nine_bits (LogicalConstant 257.0))
(let nine_bits_bf16 (LogicalCast (LogicalCast nine_bits (Bf16)) (F32)))
(let nine_bits_f16 (LogicalCast (LogicalCast nine_bits (F16)) (F32)))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= one_bf16 one))
(check (= one_f16 one))
(check (= tiny_bf16 tiny))
(check (= (dtype-of one_bf16) (F32)))
(fail (check (= gelu_bf16 gelu)))
(fail (check (= gelu_f16 gelu)))
(fail (check (= nine_bits_bf16 nine_bits)))
(check (= nine_bits_f16 nine_bits))
"#;
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
    }

    /// Views and casts commute both ways: the export's "permute, then cast"
    /// and "cast, then permute" of one operand are one tensor, and so are
    /// the cast of a broadcast and the broadcast of the cast; the shapes
    /// and dtypes of the two spellings agree (no function merges), and a
    /// chain of two views over a cast saturates with the cast at every
    /// position.
    #[test]
    fn a_cast_and_a_view_commute_both_ways() {
        let script = r#"
(let shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 5) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (Bf16)))
(let cast_then_transpose (LogicalHelperMatrixTranspose (LogicalCast x (F32))))
(let transpose_then_cast (LogicalCast (LogicalHelperMatrixTranspose x) (F32)))
(let wide_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprCons (IntLit 5) (IntExprNil))))))
(let cast_then_broadcast (LogicalHelperBroadcastAxis (LogicalCast x (F32)) 1 (IntLit 4)))
(let broadcast_then_cast (LogicalCast (LogicalHelperBroadcastAxis x 1 (IntLit 4)) (F32)))
(let twice (LogicalHelperMatrixTranspose (LogicalCast (LogicalHelperMatrixTranspose x) (F32))))
(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))
(check (= cast_then_transpose transpose_then_cast))
(check (= (dtype-of cast_then_transpose) (F32)))
(check (= (shape-of cast_then_transpose) (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil))))))
(check (= cast_then_broadcast broadcast_then_cast))
(check (= (dtype-of cast_then_broadcast) (F32)))
(check (= (shape-of cast_then_broadcast) wide_shape))
(check (= twice (LogicalCast x (F32))))
"#;
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
    }
}
