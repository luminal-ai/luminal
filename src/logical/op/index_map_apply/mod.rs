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
                category: SpliceCategory::Rewrites,
                text: include_str!("rewrite.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Rewrites,
                text: include_str!("viewed_bounds.egg"),
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

    /// A view of a view is the view through the composite of its maps: the
    /// literal with the composed entries (a squeeze then a permute), tagged
    /// with the operand's shape.
    #[test]
    fn a_view_of_a_view_is_the_view_through_the_composite() {
        run(&format!(
            r#"
{ROUND_TRIP}
(let x (LogicalTensorInputLit (LogicalIdLit "x") nchw (F32)))
(let squeezed (LogicalIndexMapApply x m_s chw))
(let permuted (LogicalIndexMapApply squeezed m_p1 hwc))
{SCHEDULE}
(check (= permuted (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (IntLit 0) (IntExprCons (CoordVar hwc 0) (IntExprCons (CoordVar hwc 2) (IntExprCons (CoordVar hwc 1) (IntExprNil))))) nchw)
  hwc)))
"#
        ));
    }

    /// The channels-last round trip (squeeze, permute, permute back,
    /// expand) is its operand: each view of a view composes, the chain
    /// flattens to the identity literal over `[1,C,H,W]` (the batch
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

    /// Two flips of the outer axis (`i -> 3 - i`) are arithmetic maps, and
    /// they compose like any other: the composite's entry `3 - (3 - i)` is
    /// `i`, so the pair is the identity and the fold unions it with the
    /// operand. One flip is not.
    #[test]
    fn two_arithmetic_flips_compose_to_the_operand() {
        run(&format!(
            r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") shape (F32)))
(let flip (IndexMapLit (IntExprCons (IntAdd (IntLit 3) (IntMul (CoordVar shape 1) (IntLit -1))) (IntExprCons (CoordVar shape 0) (IntExprNil))) shape))
(let once (LogicalIndexMapApply x flip shape))
(let twice (LogicalIndexMapApply once flip shape))
{SCHEDULE}
(check (= twice x))
(fail (check (= once x)))
"#
        ));
    }

    /// The wrap-around shift by one over `[8]`: `roll(x)[i] = x[(i + 1) % 8]`.
    const ROLL: &str = r#"
(let s1 (ShapeLit (IntExprCons (IntLit 1) (IntExprNil))))
(let s8 (ShapeLit (IntExprCons (IntLit 8) (IntExprNil))))
(let roll (IndexMapLit (IntExprCons (IntTruncRem (IntAdd (CoordVar s8 0) (IntLit 1)) (IntLit 8)) (IntExprNil)) s8))
"#;

    /// Run `script` under the schedule's saturating stratum for at most
    /// `rounds` rounds, then `checks`. A run that has not reached a fixpoint
    /// by then fails here instead of hanging the suite.
    fn run_bounded(script: &str, rounds: usize, checks: &str) {
        let mut graph = new_egraph();
        graph
            .parse_and_run_program(None, &format!("{}\n{script}", assembled_program_for(&[])))
            .unwrap();
        graph
            .parse_and_run_program(None, "(run-schedule (saturate (run prop)))")
            .unwrap();
        let converged = (0..rounds).any(|_| {
            let report = graph
                .parse_and_run_program(
                    None,
                    "(run-schedule (run main_ruleset) (run prop) (run subst-walk))",
                )
                .unwrap();
            !report.iter().any(|output| {
                matches!(output, egglog::CommandOutput::RunSchedule(report) if report.updated)
            })
        });
        assert!(converged, "no fixpoint within {rounds} rounds");
        graph.parse_and_run_program(None, checks).unwrap();
    }

    /// An arithmetic map composed with itself: two rolls are the roll by
    /// two, one literal with the folded entry `(i + 2) % 8`, and eight are
    /// the operand (the entry folds to the coordinate and the identity fold
    /// ends it).
    #[test]
    fn rolls_compose_and_the_full_cycle_is_the_operand() {
        let chain: String = (1..=8)
            .map(|k| format!("(let r{k} (LogicalIndexMapApply r{} roll s8))\n", k - 1))
            .collect();
        run_bounded(
            &format!(
                r#"
{ROLL}
(let r0 (LogicalTensorInputLit (LogicalIdLit "x") s8 (F32)))
{chain}
"#
            ),
            200,
            r#"
(check (= r2 (LogicalIndexMapApply r0
  (IndexMapLit (IntExprCons (IntTruncRem (IntAdd (CoordVar s8 0) (IntLit 2)) (IntLit 8)) (IntExprNil)) s8)
  s8)))
(check (= r8 r0))
(fail (check (= r1 r0)))
(fail (check (= r4 r0)))
"#,
        );
    }

    /// A tensor that is its own view through an arithmetic map. `t`
    /// broadcasts one element to eight, so every roll of it is `t`; the
    /// union below states that, and the class of `t` then holds a view of
    /// itself. Every view of a view composes, so the class collects the
    /// powers of the roll: it saturates because the nested remainders fold
    /// (`core: a nested IntTruncRem folds into the outer one`) and the roll
    /// has eight powers, the eighth the identity. Without the fold each
    /// power is a new map and this never ends.
    #[test]
    fn a_tensor_that_is_its_own_roll_saturates() {
        run_bounded(
            &format!(
                r#"
{ROLL}
(let b (LogicalTensorInputLit (LogicalIdLit "b") s1 (F32)))
(let t (LogicalIndexMapApply b (IndexMapLit (IntExprCons (IntLit 0) (IntExprNil)) s1) s8))
(let v (LogicalIndexMapApply t roll s8))
(union t v)
"#
            ),
            200,
            r#"
(check (= t (LogicalIndexMapApply t
  (IndexMapLit (IntExprCons (IntTruncRem (IntAdd (CoordVar s8 0) (IntLit 3)) (IntLit 8)) (IntExprNil)) s8)
  s8)))
(check (= t (LogicalIndexMapApply t
  (IndexMapLit (IntExprCons (IntTruncRem (IntAdd (CoordVar s8 0) (IntLit 7)) (IntLit 8)) (IntExprNil)) s8)
  s8)))
"#,
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
