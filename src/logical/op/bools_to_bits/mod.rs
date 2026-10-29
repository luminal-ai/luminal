use egraph_serialize::Node;

use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::logical_op::{LogicalOp, LogicalRender};

/// Boolean representation operation; see docs/boolean-bits.md.
#[derive(Debug, Clone, Copy)]
pub struct LogicalBoolsToBits;

impl LogicalOp for LogicalBoolsToBits {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalBoolsToBits"
    }

    fn display_name(&self) -> &'static str {
        "bools_to_bits"
    }

    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[("input", 0)]
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        format!(
            "LogicalBoolsToBits({}, {})",
            ctx.child_expr(node, 0),
            ctx.child_short(node, 1, 2, None).unwrap_or_default()
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
    fn check_view(offset: &str, base: &str, expected: bool) {
        let program = format!(
            r#"
(let s (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 16) (IntExprNil)))))
(let input (LogicalTensorInputLit (LogicalIdLit "bits") s (Bool)))
(let il (BitOffsetExpressionLayoutLit {offset} s (BitWidthLit 1)))
(let it (LayoutTensorLit input il))
(let output (LogicalBoolsToBits input (U8)))
(let os (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprNil)))))
(let ol (BitOffsetExpressionLayoutLit {base} os (BitWidthLit 8)))
{schedule}
(check (= (shape-of output) os))
{query}
"#,
            schedule = luminal_reference::ReferenceBindings::SCHEDULE,
            query = if expected {
                "(check (bit-packing-view output il ol))"
            } else {
                "(fail (check (bit-packing-view output il ol)))"
            }
        );
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(
            None,
            &format!(
                "{}{}",
                crate::egglog_snippet::assembled_program_for(&[]),
                program
            ),
        )
        .unwrap();
    }

    #[test]
    fn packed_groups_allow_strided_rows_and_unaligned_origins() {
        check_view(
            "(IntAdd (IntLit 3) (IntAdd (IntMul (CoordVar s 1) (IntLit 32)) (CoordVar s 0)))",
            "(IntAdd (IntLit 3) (IntAdd (IntMul (CoordVar os 1) (IntLit 32)) (IntMul (CoordVar os 0) (IntLit 8))))",
            true,
        );
    }
    #[test]
    fn gaps_between_groups_do_not_require_a_copy() {
        check_view(
            "(IntAdd (IntMul (CoordVar s 1) (IntLit 64)) (IntAdd (IntMul (IntTruncDiv (CoordVar s 0) (IntLit 8)) (IntLit 16)) (IntTruncRem (CoordVar s 0) (IntLit 8))))",
            "(IntAdd (IntMul (CoordVar os 1) (IntLit 64)) (IntMul (CoordVar os 0) (IntLit 16)))",
            true,
        );
    }
    #[test]
    fn gaps_within_groups_cannot_be_reinterpreted() {
        check_view(
            "(IntAdd (IntMul (CoordVar s 1) (IntLit 64)) (IntMul (CoordVar s 0) (IntLit 2)))",
            "(IntAdd (IntMul (CoordVar os 1) (IntLit 64)) (IntMul (CoordVar os 0) (IntLit 16)))",
            false,
        );
    }
    #[test]
    fn broadcast_bits_cannot_be_reinterpreted_as_a_word() {
        check_view("(IntLit 0)", "(IntLit 0)", false);
    }
    #[test]
    fn reversed_bits_cannot_be_reinterpreted_as_a_word() {
        check_view(
            "(IntAdd (IntLit 15) (IntMul (CoordVar s 0) (IntLit -1)))",
            "(IntAdd (IntLit 15) (IntMul (CoordVar os 0) (IntLit -8)))",
            false,
        );
    }

    #[test]
    fn byte_boolean_contracts_and_invalid_input_dtypes() {
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(
            None,
            &format!(
                r#"{}
(let s (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") s (Bool8)))
(let b (LogicalTensorInputLit (LogicalIdLit "b") s (Bool8)))
(let ab (LogicalAnd a b))
(let either (LogicalOr a b))
(let inverse (LogicalNot a))
{}
(check (= (dtype-of ab) (Bool8)))
(check (= (dtype-of either) (Bool8)))
(check (= (dtype-of inverse) (Bool8)))
"#,
                crate::egglog_snippet::assembled_program_for(&[]),
                luminal_reference::ReferenceBindings::SCHEDULE
            ),
        )
        .unwrap();
        let err = eg.parse_and_run_program(None,"(let invalid (LogicalNot (LogicalCast a (Int)))) (run-schedule (saturate (run prop)) (saturate (run main_ruleset)))").unwrap_err();
        assert!(err.to_string().contains("requires Bool or Bool8"));
    }
}

#[cfg(test)]
mod encoding_equality_tests {
    #[test]
    fn floating_operand_permutation_is_not_an_encoding_equality() {
        let mut eg = crate::egglog_snippet::new_egraph();
        eg.parse_and_run_program(
            None,
            &format!(
                r#"{}
(let s (ShapeLit (IntExprCons (IntLit 1) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "nan-a") s (F32)))
(let b (LogicalTensorInputLit (LogicalIdLit "nan-b") s (F32)))
(let ab (LogicalBitsToBools (LogicalAdd a b)))
(let ba (LogicalBitsToBools (LogicalAdd b a)))
(let am (LogicalBitsToBools (LogicalMul a b)))
(let bm (LogicalBitsToBools (LogicalMul b a)))
{}
(fail (check (= ab ba)))
(fail (check (= am bm)))
"#,
                crate::egglog_snippet::assembled_program_for(&[]),
                luminal_reference::ReferenceBindings::SCHEDULE
            ),
        )
        .unwrap();
    }
}

#[cfg(test)]
mod width_tests {
    #[test]
    fn packing_view_proof_uses_encoding_width_including_subbyte_types() {
        for (dtype, width) in [
            ("Bool", 1),
            ("I4", 4),
            ("F6E2M3", 6),
            ("U8", 8),
            ("TF32", 19),
            ("Int", 32),
            ("Int64", 64),
        ] {
            let mut eg = crate::egglog_snippet::new_egraph();
            eg.parse_and_run_program(None,&format!(r#"{}
(let s (ShapeLit (IntExprCons (IntLit {}) (IntExprNil))))
(let input (LogicalTensorInputLit (LogicalIdLit "bits") s (Bool)))
(let il (BitOffsetExpressionLayoutLit (CoordVar s 0) s (BitWidthLit 1)))
(let it (LayoutTensorLit input il))
(let out (LogicalBoolsToBits input ({dtype})))
(let os (ShapeLit (IntExprCons (IntLit 2) (IntExprNil))))
(let ol (BitOffsetExpressionLayoutLit (IntMul (CoordVar os 0) (IntLit {width})) os (BitWidthLit {width})))
{}
(check (= (shape-of out) os))
(check (bit-packing-view out il ol))
"#,crate::egglog_snippet::assembled_program_for(&[]),2*width,luminal_reference::ReferenceBindings::SCHEDULE)).unwrap();
        }
    }
}
