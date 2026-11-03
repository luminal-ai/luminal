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
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite.egg"),
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

    const SCHEDULE: &str = "(run-schedule (saturate (run prop))
  (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

    fn try_run(script: &str) -> Result<(), String> {
        new_egraph()
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .map(|_| ())
            .map_err(|e| format!("{e}"))
    }

    fn run(script: &str) {
        try_run(script).unwrap();
    }

    /// The channels-last round trip's shapes and maps: `[1,C,H,W]` squeezed
    /// to `[C,H,W]`, permuted to `[H,W,C]`, permuted back to `[C,H,W]`,
    /// expanded to `[1,C,H,W]`. Entries run highest parent axis first;
    /// `(CoordVar shape k)` is the view's coordinate k from the end.
    const ROUND_TRIP: &str = r#"
(let nchw (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))))
(let chw (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let hwc (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprCons (IntLit 2) (IntExprNil))))))
(let m_s (IndexMapLit (IntExprCons (IntLit 0) (IntExprCons (CoordVar chw 2) (IntExprCons (CoordVar chw 1) (IntExprCons (CoordVar chw 0) (IntExprNil))))) nchw))
(let m_p1 (IndexMapLit (IntExprCons (CoordVar hwc 0) (IntExprCons (CoordVar hwc 2) (IntExprCons (CoordVar hwc 1) (IntExprNil)))) chw))
(let m_p2 (IndexMapLit (IntExprCons (CoordVar chw 1) (IntExprCons (CoordVar chw 0) (IntExprCons (CoordVar chw 2) (IntExprNil)))) hwc))
(let m_e (IndexMapLit (IntExprCons (CoordVar nchw 2) (IntExprCons (CoordVar nchw 1) (IntExprCons (CoordVar nchw 0) (IntExprNil)))) chw))
"#;

    /// A view of a view is the view through the closed composite of its
    /// maps, and a composite of coordinate-only maps (a squeeze then a
    /// permute) opens without demand to the literal with the composed
    /// entries, tagged with the operand's shape.
    #[test]
    fn a_view_of_a_view_is_the_view_through_the_composite() {
        run(&format!(
            r#"
{ROUND_TRIP}
(let x (LogicalTensorInputLit (LogicalIdLit "x") nchw (F32)))
(let squeezed (LogicalIndexMapApply x m_s chw))
(let permuted (LogicalIndexMapApply squeezed m_p1 hwc))
{SCHEDULE}
(check (= permuted (LogicalIndexMapApply x (IndexMapCompose m_s m_p1) hwc)))
(check (= (index-map-tag (IndexMapCompose m_s m_p1)) nchw))
(check (index-map-coordinate-only (IndexMapCompose m_s m_p1)))
(check (= (IndexMapCompose m_s m_p1)
  (IndexMapLit (IntExprCons (IntLit 0) (IntExprCons (CoordVar hwc 0) (IntExprCons (CoordVar hwc 2) (IntExprCons (CoordVar hwc 1) (IntExprNil))))) nchw)))
"#
        ));
    }

    /// The channels-last round trip (squeeze, permute, permute back,
    /// expand) is its operand: the coordinate-only composites open, the
    /// chain flattens to the identity literal over `[1,C,H,W]` (the batch
    /// entry is the literal 0 on a unit axis), and the identity fold ends
    /// it.
    #[test]
    fn a_coordinate_only_round_trip_is_its_operand() {
        run(&format!(
            r#"
{ROUND_TRIP}
(let x (LogicalTensorInputLit (LogicalIdLit "x") nchw (F32)))
(let s (LogicalIndexMapApply x m_s chw))
(let p1 (LogicalIndexMapApply s m_p1 hwc))
(let p2 (LogicalIndexMapApply p1 m_p2 chw))
(let back (LogicalIndexMapApply p2 m_e nchw))
{SCHEDULE}
(check (= back x))
(fail (check (= p1 x)))
"#
        ));
    }

    /// SATURATION item 1: the round trip over a cast, with the result unioned
    /// with the cast (the channels-last rewrite's union), under the
    /// view-of-cast rule. The cast lifts through the chain; the lifted chain
    /// over the wide operand is the operand, so the cast of it is the cast
    /// already there and the ladder closes. The test returning is the claim.
    #[test]
    fn the_round_trip_over_a_cast_unioned_with_the_cast_saturates() {
        run(&format!(
            r#"
{ROUND_TRIP}
(let wide (LogicalTensorInputLit (LogicalIdLit "wide") nchw (F32)))
(let out (LogicalCast wide (Bf16)))
(let s (LogicalIndexMapApply out m_s chw))
(let p1 (LogicalIndexMapApply s m_p1 hwc))
(let p2 (LogicalIndexMapApply p1 m_p2 chw))
(let back (LogicalIndexMapApply p2 m_e nchw))
(union out back)
{SCHEDULE}
(check (= (LogicalIndexMapApply (LogicalIndexMapApply (LogicalIndexMapApply (LogicalIndexMapApply wide m_s chw) m_p1 hwc) m_p2 chw) m_e nchw) wide))
(check (= out (LogicalCast wide (Bf16))))
"#
        ));
    }

    /// Two flips of the outer axis (`i -> 3 - i`) are arithmetic maps: their
    /// composite is named, tagged, and not coordinate-only, and it opens
    /// (composing to the identity) only under `index-map-open-demand`.
    #[test]
    fn an_arithmetic_composite_opens_only_on_demand() {
        let script = |demand: &str| {
            format!(
                r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let flip (IndexMapLit (IntExprCons (IntAdd (IntLit 3) (IntMul (CoordVar shape 1) (IntLit -1))) (IntExprCons (CoordVar shape 0) (IntExprNil))) shape))
(let once (LogicalIndexMapApply x flip shape))
(let twice (LogicalIndexMapApply once flip shape))
{demand}
{SCHEDULE}
(check (= twice (LogicalIndexMapApply x (IndexMapCompose flip flip) shape)))
(check (= (index-map-tag (IndexMapCompose flip flip)) shape))
(fail (check (= once x)))
"#
            )
        };
        run(&format!(
            "{}\n(check (= twice x))",
            script("(index-map-open-demand (IndexMapCompose flip flip))")
        ));
        // Undemanded, the composite stays closed: not coordinate-only (its
        // entries are arithmetic), so nothing opens it, and the pair stays
        // two views.
        run(&format!(
            "{}\n(fail (check (index-map-coordinate-only (IndexMapCompose flip flip))))\n(fail (check (= twice x)))",
            script("")
        ));
    }

    /// Substitution under a composite is sequential: the parent's offset
    /// substituted under the composite is its image under the inner map
    /// substituted under the outer, so the composite spelling's layout is
    /// the nested spelling's.
    #[test]
    fn substitution_under_a_composite_is_the_nested_substitution() {
        run(&format!(
            r#"
{ROUND_TRIP}
(let x (LogicalTensorInputLit (LogicalIdLit "x") nchw (F32)))
(let offset (IntAdd (IntMul (CoordVar nchw 2) (IntLit 12)) (IntAdd (IntMul (CoordVar nchw 1) (IntLit 4)) (CoordVar nchw 0))))
(let x_lt (LayoutTensorLit x (BitOffsetExpressionLayoutLit (IntMul offset (IntLit 32)) nchw (bits-of (F32)))))
(let squeezed (LogicalIndexMapApply x m_s chw))
(let permuted (LogicalIndexMapApply squeezed m_p1 hwc))
{SCHEDULE}
(check (= (int-subst-of (IntMul offset (IntLit 32)) (IndexMapCompose m_s m_p1))
          (int-subst-of (int-subst-of (IntMul offset (IntLit 32)) m_s) m_p1)))
"#
        ));
    }

    /// A composite reads its operand through its inner map, so its tag (the
    /// inner's) must be the operand's shape: a hand-seeded closed composite
    /// (two flips of a `[4,4]` box, arithmetic, so nothing opens it) over an
    /// operand of another shape is caught by the fixpoint lock.
    #[test]
    fn a_composite_over_the_wrong_operand_is_caught_by_the_lock() {
        let result = try_run(&format!(
            r#"
{ROUND_TRIP}
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let flip (IndexMapLit (IntExprCons (IntAdd (IntLit 3) (IntMul (CoordVar shape 1) (IntLit -1))) (IntExprCons (CoordVar shape 0) (IntExprNil))) shape))
(let x (LogicalTensorInputLit (LogicalIdLit "x") nchw (F32)))
(let bad (LogicalIndexMapApply x (IndexMapCompose flip flip) shape))
{SCHEDULE}
(run-schedule (saturate (run fixpoint-invariants)))
"#
        ));
        let message = result.expect_err("the composite's tag is [4,4], the operand's shape is [1,2,3,4]");
        assert!(
            message.contains("IndexMapCompose tag"),
            "the failure must name the composite lock: {message}"
        );
    }

    /// A view whose map is the box's own coordinates is its operand; the
    /// transpose of a square operand, whose entries are the coordinates in
    /// the other order, is not.
    #[test]
    fn a_view_through_the_identity_map_is_its_operand() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let identity (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar shape 1) (IntExprCons (CoordVar shape 0) (IntExprNil))) shape) shape))
(let transposed (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar shape 0) (IntExprCons (CoordVar shape 1) (IntExprNil))) shape) shape))
{SCHEDULE}
(check (= identity x))
(fail (check (= transposed x)))
"#
        ));
    }
}
