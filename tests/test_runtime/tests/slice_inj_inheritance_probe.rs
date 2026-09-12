//! SLICE-INJECTIVITY INHERITANCE PROBES (adversarial review of Austin's
//! 2026-08-25 proposal: "the slice of an injective layout is injective").
//!
//! THE MATH. A slice view is offset_view = offset_parent o f, where f is
//! the index map (out coordinate -> parent coordinate). If f is injective
//! ON THE VIEW'S BOX and lands INSIDE the parent's box, and the parent
//! layout tensor is injective on its own box, the composition is
//! injective. Both side conditions are load-bearing:
//!   * in-range: the parent token says nothing about offsets outside the
//!     parent box (this is why padding — domain EXTENSION — does NOT
//!     inherit: RM [3,4] strides [4,1] is a bijection, but extending the
//!     declared domain to [3,8] with the same strides collides (0,4) with
//!     (1,0) at offset 4).
//!   * f injective: a broadcast entry (constant / step-0) leaves an out
//!     axis unread, so two out coordinates share one parent coordinate.
//!
//! THE RECOGNIZER. Per never-depend-on-spelling, "sliceness" is read off
//! the apply's OWN IndexMapLit entries (a construction-site input list —
//! the same anchored-membership family as the preamble's sigma-fold and
//! chain-walk arms), never off the composed chain's spelling. The
//! candidate rule stamps the composed layout tensor, tied by
//! (= ?composed (int-subst-of ?bit_expr ?map)) exactly as the round-9
//! reading arms tie theirs — never "some layout of the child", so the
//! blanket right-major materialize LT (preamble :4281) is untouched.
//! The (= ?sigma (sigma-out ?map)) premise rides the sigma fold's own
//! in-range guards (interval minbox / structural same-extent), closing
//! the leaf-expression bypass where int-subst-of rows exist without any
//! entry legality check.
//!
//! V1 SCOPE: rank-2, zero-base, axis-aligned selection entries — the
//! column/row-slice family the r9 pitched-operand probe fails closed on
//! today. Nonzero-start slices compose to bit-offset layouts but have no
//! Strided chain spelling (no base-offset field in the chain ontology;
//! arithmetic entries stall the chain walk), so they wait for the queued
//! slice/base-offset ontology.

use luminal::prelude::egraph_serialize::{ClassId, EGraph};

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

/// THE CANDIDATE RULE (probe-local; rank-2 axis-aligned arm). Derived,
/// never asserted: every premise is structural or an anchored membership
/// on the apply's own map entries; the conclusion carries a pointer
/// (the composed layout tensor) and a finite classification (Injective).
const INHERIT_RULE_RANK2: &str = r#"
(rule
  (
    (= ?out_logical (LogicalIndexMapApply ?in_logical ?map ?shape))
    (= ?in_lt (LayoutTensorLit ?in_logical ?in_layout))
    (= ?in_layout (BitOffsetExpressionLayoutLit ?bit_expr ?in_shape ?bit_width))
    (= (injectivity-of ?in_lt) (Injective))
    ; the map is the apply's own; its source tag must BE the parent
    ; layout's declared domain (the coherence the fixpoint tripwire
    ; enforces loudly; here it is a silent fail-closed premise).
    (= ?map (IndexMapLit (IntExprCons ?e1 (IntExprCons ?e0 (IntExprNil))) ?in_shape))
    ; sigma exists only if EVERY entry passed the in-range guards
    ; (interval minbox or structural same-extent) — the domain-
    ; restriction half of the theorem, inherited from the subst road.
    (= ?sigma (sigma-out ?map))
    ; the tie: stamp the COMPOSED layout tensor, nothing else.
    (= ?composed (int-subst-of ?bit_expr ?map))
    (= (shape-of ?out_logical) ?shape)
    ; OUT-RANK PIN (load-bearing, see the rank-3 attack below): the two
    ; entries cover ALL out axes only if the out shape has exactly two.
    (= ?shape (ShapeLit (IntExprCons ?d1 (IntExprCons ?d0 (IntExprNil)))))
    ; injectivity of f: axis-aligned selection entries, distinct axes,
    ; every out axis read. Anchored membership on the map's own entries.
    (= ?e1 (CoordVar ?shape 1))
    (= ?e0 (CoordVar ?shape 0))
  )
  (
    (set (injectivity-of
      (LayoutTensorLit ?out_logical
        (BitOffsetExpressionLayoutLit ?composed ?shape ?bit_width)))
      (Injective))
  )
)
"#;

/// THE REVERT PROBE: the same rule with the out-rank pin REMOVED. A
/// rank-3 view whose entries name only axes 1 and 0 leaves axis 2
/// unread (non-injective), and this variant stamps it anyway — shown
/// red by `rank3_unread_axis_attack`, proving the pin is load-bearing.
const INHERIT_RULE_NO_RANK_PIN: &str = r#"
(rule
  (
    (= ?out_logical (LogicalIndexMapApply ?in_logical ?map ?shape))
    (= ?in_lt (LayoutTensorLit ?in_logical ?in_layout))
    (= ?in_layout (BitOffsetExpressionLayoutLit ?bit_expr ?in_shape ?bit_width))
    (= (injectivity-of ?in_lt) (Injective))
    (= ?map (IndexMapLit (IntExprCons ?e1 (IntExprCons ?e0 (IntExprNil))) ?in_shape))
    (= ?sigma (sigma-out ?map))
    (= ?composed (int-subst-of ?bit_expr ?map))
    (= (shape-of ?out_logical) ?shape)
    (= ?e1 (CoordVar ?shape 1))
    (= ?e0 (CoordVar ?shape 0))
  )
  (
    (set (injectivity-of
      (LayoutTensorLit ?out_logical
        (BitOffsetExpressionLayoutLit ?composed ?shape ?bit_width)))
      (Injective))
  )
)
"#;

/// The r9 pitched-operand fixture, verbatim geometry: x[2,3] @ w[3,4]
/// where w is a zero-base column-slice view of a [3,8] right-major
/// parent. NO creator certificate — `extra` is where the candidate
/// inheritance rule goes (or nothing, for the baseline).
fn matmul_slice_fixture(extra: &str) -> String {
    format!(
        r#"(let x_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let parent_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 8) (IntExprNil)))))
(let w_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") x_shape (F32)))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "w_parent") parent_shape (F32)))
(let slice_map
  (IndexMapLit
    (IntExprCons (CoordVar w_shape 1)
      (IntExprCons (CoordVar w_shape 0) (IntExprNil)))
    parent_shape))
(let w_logical (LogicalIndexMapApply parent_logical slice_map w_shape))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    x_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    w_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit x_shape (bits-of (F32))))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let w_layout (StridedElementLayoutLit w_shape
  (IntAffineExprCons (IntMul (CoordVar w_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar w_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let parent_buffer_tensor (BufferTensorLit parent_lt w_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{extra}
{SCHEDULE}
"#
    )
}

/// ATTACK (c): a broadcast-style map. Parent [3,8] contiguous
/// (injective), view [3,4], entries [CoordVar(v,1), (IntLit 0)] — the
/// parent's inner axis is pinned at 0 and the view's inner axis (extent
/// 4) is read by NOBODY: f(c1,c0) = (c1, 0), non-injective. The
/// composed layout exists (chain [8*c1, 0]) and must NOT be stamped.
fn broadcast_attack_fixture(extra: &str) -> String {
    format!(
        r#"(let parent_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 8) (IntExprNil)))))
(let view_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "p") parent_shape (F32)))
(let bcast_map
  (IndexMapLit
    (IntExprCons (CoordVar view_shape 1)
      (IntExprCons (IntLit 0) (IntExprNil)))
    parent_shape))
(let view_logical (LogicalIndexMapApply parent_logical bcast_map view_shape))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
{extra}
{SCHEDULE}
"#
    )
}

/// ATTACK (rank pin): parent [3,8] contiguous, view [5,3,4] (rank 3),
/// entries [CoordVar(v,1), CoordVar(v,0)] — both entries are honest
/// selection entries and in-range, but out axis 2 (extent 5) is unread:
/// f(c2,c1,c0) = (c1,c0), non-injective. The sound rule's out-rank pin
/// refuses; the pin-removed variant stamps it (the revert probe).
fn rank3_attack_fixture(extra: &str) -> String {
    format!(
        r#"(let parent_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 8) (IntExprNil)))))
(let view_shape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "p") parent_shape (F32)))
(let deep_map
  (IndexMapLit
    (IntExprCons (CoordVar view_shape 1)
      (IntExprCons (CoordVar view_shape 0) (IntExprNil)))
    parent_shape))
(let view_logical (LogicalIndexMapApply parent_logical deep_map view_shape))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
{extra}
{SCHEDULE}
"#
    )
}

/// ATTACK (repeated entry): parent [8,8] contiguous, view [4,4],
/// entries [CoordVar(v,0), CoordVar(v,0)] — one out coordinate feeds
/// BOTH parent axes; out axis 1 is unread: f(c1,c0) = (c0,c0),
/// non-injective. (The composed chain is [0, 9*c0] via the diagonal
/// accumulate arms.) The pinned-distinct-axes premises must refuse.
fn repeated_entry_attack_fixture(extra: &str) -> String {
    format!(
        r#"(let parent_shape (ShapeLit (IntExprCons (IntLit 8) (IntExprCons (IntLit 8) (IntExprNil)))))
(let view_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "p") parent_shape (F32)))
(let diag_map
  (IndexMapLit
    (IntExprCons (CoordVar view_shape 0)
      (IntExprCons (CoordVar view_shape 0) (IntExprNil)))
    parent_shape))
(let view_logical (LogicalIndexMapApply parent_logical diag_map view_shape))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
{extra}
{SCHEDULE}
"#
    )
}

/// SCOPE BOUNDARY: a NONZERO-START slice (columns 2..6 of the [3,8]
/// parent). Mathematically it inherits injectivity (per-axis affine
/// reindex, step 1, start 2 — injective, in-range), but TODAY its
/// composed layout is a bit-offset expression with no Strided chain
/// spelling: the chain ontology has no base-offset field and the chain
/// walk has no arithmetic-entry arm (v1 scope comment). The v1 rule's
/// CoordVar entry premise stalls — conservative refusal, never a
/// miscompile. This probe pins all three facts.
fn nonzero_start_fixture(extra: &str) -> String {
    format!(
        r#"(let parent_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 8) (IntExprNil)))))
(let view_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil)))))
(let parent_logical (LogicalTensorInputLit (LogicalIdLit "p") parent_shape (F32)))
(let start_map
  (IndexMapLit
    (IntExprCons (CoordVar view_shape 1)
      (IntExprCons (IntAdd (CoordVar view_shape 0) (IntLit 2)) (IntExprNil)))
    parent_shape))
(let view_logical (LogicalIndexMapApply parent_logical start_map view_shape))
(let parent_layout (RightMajorContiguousElementLayoutLit parent_shape (bits-of (F32))))
(let parent_lt (LayoutTensorLit parent_logical parent_layout))
{extra}
{SCHEDULE}
"#
    )
}

fn class_has(s: &EGraph, class: &ClassId, op: &str) -> bool {
    s.nodes.values().any(|n| &n.eclass == class && n.op == op)
}

/// Layout tensor classes whose layout carries a Strided spelling and NO
/// contiguous one — the view-born pitched/broadcast layouts, never the
/// blanket right-major mints or the declared contiguous operands.
fn strided_only_lt_classes(s: &EGraph) -> Vec<ClassId> {
    let mut out: Vec<ClassId> = Vec::new();
    for n in s.nodes.values().filter(|n| n.op == "LayoutTensorLit") {
        let Some(layout) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        let layout_class = layout.eclass.clone();
        if class_has(s, &layout_class, "StridedElementLayoutLit")
            && !class_has(s, &layout_class, "RightMajorContiguousElementLayoutLit")
            && !class_has(s, &layout_class, "LeftMajorContiguousElementLayoutLit")
            && !out.contains(&n.eclass)
        {
            out.push(n.eclass.clone());
        }
    }
    out
}

/// The strided-only layout tensor classes that carry an injectivity-of
/// row — i.e. what the candidate rule (or a false assertion) stamped.
fn stamped_strided_only_lt_classes(s: &EGraph) -> Vec<ClassId> {
    let strided_only = strided_only_lt_classes(s);
    let mut out: Vec<ClassId> = Vec::new();
    for n in s.nodes.values().filter(|n| n.op == "injectivity-of") {
        let Some(lt) = n.children.first().and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        if strided_only.contains(&lt.eclass) && !out.contains(&lt.eclass) {
            out.push(lt.eclass.clone());
        }
    }
    out
}

/// ROUND-10 SCOPE FILTER: the transpose-sandwich rewrite's injectivity
/// TRANSPORT (cublaslt_marker_rewrite.egg) legitimately stamps the
/// sandwich's transpose-view layout tensors — those are views of the
/// MATMUL OUT, not of an input. This probe's subject is the SLICE OF AN
/// INPUT, so its counters keep only stamped tensors whose view parent is
/// a LogicalTensorInputLit.
fn stamped_input_view_lt_classes(s: &EGraph) -> Vec<ClassId> {
    stamped_strided_only_lt_classes(s)
        .into_iter()
        .filter(|lt_class| {
            let Some(logical) = s.nodes.values().find_map(|m| {
                (m.eclass == *lt_class && m.op == "LayoutTensorLit")
                    .then(|| {
                        m.children
                            .first()
                            .and_then(|id| s.nodes.get(id))
                            .map(|c| c.eclass.clone())
                    })
                    .flatten()
            }) else {
                return false;
            };
            s.nodes.values().any(|m| {
                m.eclass == logical
                    && m.op == "LogicalIndexMapApply"
                    && m.children
                        .first()
                        .and_then(|id| s.nodes.get(id))
                        .map(|parent| {
                            s.nodes.values().any(|q| {
                                q.eclass == parent.eclass && q.op == "LogicalTensorInputLit"
                            })
                        })
                        .unwrap_or(false)
            })
        })
        .collect()
}

/// r9's detector, verbatim semantics: A readings whose operand layout is
/// Strided-only (the zero-copy pitched view, not the materialising copy).
fn a_readings_over_the_pitched_view(s: &EGraph) -> Vec<&'static str> {
    let mut out = Vec::new();
    for n in s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
    {
        let Some(lt) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        let lt_class = lt.eclass.clone();
        let Some(layout) = s.nodes.values().find_map(|m| {
            (m.eclass == lt_class && m.op == "LayoutTensorLit")
                .then(|| {
                    m.children
                        .get(1)
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| c.eclass.clone())
                })
                .flatten()
        }) else {
            continue;
        };
        if !class_has(s, &layout, "StridedElementLayoutLit")
            || class_has(s, &layout, "RightMajorContiguousElementLayoutLit")
            || class_has(s, &layout, "LeftMajorContiguousElementLayoutLit")
        {
            continue;
        }
        let op = match n
            .children
            .get(2)
            .and_then(|id| s.nodes.get(id))
            .map(|c| c.eclass.clone())
        {
            Some(c) if class_has(s, &c, "CublasLtOperationT") => "T",
            Some(c) if class_has(s, &c, "CublasLtOperationN") => "N",
            _ => "?",
        };
        out.push(op);
    }
    out
}

/// (a) THE BASELINE, reproduced in this probe's own terms: with no
/// creator certificate and no inheritance rule, the zero-base column
/// slice gets NO derived injectivity and the reading fails closed.
/// The setup assertion (the strided-only view LT exists at all) keeps
/// the test non-vacuous: composition really ran.
#[test]
fn baseline_zero_base_slice_has_no_derived_injectivity() {
    let s = test_runtime::serialize_fixture(&matmul_slice_fixture(""));
    let views = strided_only_lt_classes(&s);
    assert!(
        !views.is_empty(),
        "SETUP BROKE: no strided-only view layout tensor exists — composition did not run"
    );
    // ROUND-10 RE-SCOPE: count only INPUT-view stamps (the sandwich
    // rewrite's transport legitimately stamps matmul-out views).
    let stamped = stamped_input_view_lt_classes(&s);
    let readings = a_readings_over_the_pitched_view(&s);
    println!(
        "baseline: strided-only LTs = {}, stamped input views = {}, readings = {readings:?}",
        views.len(),
        stamped.len()
    );
    assert!(
        stamped.is_empty(),
        "no rule and no certificate, yet an input-view LT is stamped injective"
    );
    assert!(
        readings.is_empty(),
        "fail-closed baseline: the pitched view must not be read"
    );
}

/// (b) THE CANDIDATE RULE fires on the r9 geometry with NO creator
/// certificate: the slice's composed layout tensor inherits the parent's
/// injectivity, and the round-9 reading arm admits the operand (op N).
#[test]
fn inheritance_rule_stamps_the_slice_and_the_reading_admits_it() {
    let s = test_runtime::serialize_fixture(&matmul_slice_fixture(INHERIT_RULE_RANK2));
    // ROUND-10 RE-SCOPE: input-view stamps only (see the baseline test).
    let stamped = stamped_input_view_lt_classes(&s);
    let readings = a_readings_over_the_pitched_view(&s);
    println!(
        "with rule: stamped input-view LTs = {}, readings = {readings:?}",
        stamped.len()
    );
    assert_eq!(
        stamped.len(),
        1,
        "exactly the slice's composed layout tensor inherits (the rank-3 broadcasts must not)"
    );
    assert_eq!(
        readings,
        vec!["N"],
        "the derived token is the ONLY thing the r9 arms were missing: reading must now admit N"
    );
}

/// (c) THE BROADCAST ATTACK: a constant (step-0) entry leaves the view's
/// inner axis unread — f is not injective and the rule's entry premises
/// must refuse. Setup assertion: the non-injective composed layout DOES
/// exist (chain [8*c1, 0]); it just must not be stamped.
#[test]
fn broadcast_constant_entry_does_not_inherit() {
    let s = test_runtime::serialize_fixture(&broadcast_attack_fixture(INHERIT_RULE_RANK2));
    let views = strided_only_lt_classes(&s);
    assert!(
        !views.is_empty(),
        "SETUP BROKE: the broadcast view's composed strided layout tensor was never derived"
    );
    let stamped = stamped_strided_only_lt_classes(&s);
    println!(
        "broadcast attack: strided-only LTs = {}, stamped = {}",
        views.len(),
        stamped.len()
    );
    assert!(
        stamped.is_empty(),
        "MISCOMPILE HOLE: a broadcast view (out axis unread, extent 4) inherited injectivity"
    );
}

/// THE RANK-3 ATTACK + REVERT PROBE. Honest in-range selection entries,
/// but a rank-3 out shape whose top axis is unread. The sound rule
/// refuses (the out-rank pin); the pin-removed variant stamps the
/// non-injective view — the guard shown load-bearing both ways.
#[test]
fn rank3_unread_axis_attack() {
    let s = test_runtime::serialize_fixture(&rank3_attack_fixture(INHERIT_RULE_RANK2));
    let views = strided_only_lt_classes(&s);
    assert!(
        !views.is_empty(),
        "SETUP BROKE: rank-3 view composition did not run"
    );
    let stamped = stamped_strided_only_lt_classes(&s);
    println!("rank-3 attack, sound rule: stamped = {}", stamped.len());
    assert!(
        stamped.is_empty(),
        "MISCOMPILE HOLE: rank-3 view with an unread extent-5 axis inherited injectivity"
    );

    // The revert probe: reintroduce the break (drop the rank pin) and
    // show the attack lands — the pin is what stands between the rule
    // and stamping a non-injective layout.
    let s = test_runtime::serialize_fixture(&rank3_attack_fixture(INHERIT_RULE_NO_RANK_PIN));
    let stamped = stamped_strided_only_lt_classes(&s);
    println!("rank-3 attack, PIN REMOVED: stamped = {}", stamped.len());
    assert_eq!(
        stamped.len(),
        1,
        "revert probe: without the out-rank pin the non-injective view IS stamped \
         (if this stops firing, the attack fixture has gone vacuous)"
    );
}

/// THE SCOPE BOUNDARY, pinned: a nonzero-start slice composes at the
/// bit level (a BitOffset-spelled, non-contiguous view layout exists)
/// but gets NO Strided chain spelling (no base-offset field in the
/// ontology; the chain walk stalls on the arithmetic entry) and NO
/// inherited token from the v1 rule (its CoordVar entry premise stalls
/// — conservative, fail-closed). After the slice/base-offset ontology
/// lands, a (coordinate + literal-start) entry arm extends the rule.
#[test]
fn nonzero_start_slice_is_outside_v1_scope_and_fails_closed() {
    let s = test_runtime::serialize_fixture(&nonzero_start_fixture(INHERIT_RULE_RANK2));
    // The composed view layout: on a LayoutTensorLit over an apply,
    // bit-offset-spelled, not contiguous.
    let mut composed_view_lts: Vec<ClassId> = Vec::new();
    for n in s.nodes.values().filter(|n| n.op == "LayoutTensorLit") {
        let (Some(logical), Some(layout)) = (
            n.children.first().and_then(|id| s.nodes.get(id)),
            n.children.get(1).and_then(|id| s.nodes.get(id)),
        ) else {
            continue;
        };
        if class_has(&s, &logical.eclass, "LogicalIndexMapApply")
            && class_has(&s, &layout.eclass, "BitOffsetExpressionLayoutLit")
            && !class_has(&s, &layout.eclass, "RightMajorContiguousElementLayoutLit")
            && !class_has(&s, &layout.eclass, "LeftMajorContiguousElementLayoutLit")
        {
            assert!(
                !class_has(&s, &layout.eclass, "StridedElementLayoutLit"),
                "the ontology moved: a nonzero-start slice now HAS a Strided chain spelling \
                 — revisit the v1 scope (a start arm may be extendable now)"
            );
            if !composed_view_lts.contains(&n.eclass) {
                composed_view_lts.push(n.eclass.clone());
            }
        }
    }
    assert!(
        !composed_view_lts.is_empty(),
        "SETUP BROKE: the nonzero-start slice never composed at the bit level"
    );
    for n in s.nodes.values().filter(|n| n.op == "injectivity-of") {
        if let Some(lt) = n.children.first().and_then(|id| s.nodes.get(id)) {
            assert!(
                !composed_view_lts.contains(&lt.eclass),
                "v1 rule stamped a nonzero-start slice it has no premises for"
            );
        }
    }
    println!(
        "nonzero-start slice: composed bit-level view LTs = {}, no chain spelling, no token (scope boundary pinned)",
        composed_view_lts.len()
    );
}

/// THE REPEATED-ENTRY ATTACK: one out coordinate feeds both parent axes
/// (a diagonal-style map into a rank-2 view), leaving out axis 1 unread.
/// The pinned-distinct-axes premises must refuse.
#[test]
fn repeated_entry_map_does_not_inherit() {
    let s = test_runtime::serialize_fixture(&repeated_entry_attack_fixture(INHERIT_RULE_RANK2));
    let views = strided_only_lt_classes(&s);
    assert!(
        !views.is_empty(),
        "SETUP BROKE: repeated-entry view composition did not run"
    );
    let stamped = stamped_strided_only_lt_classes(&s);
    println!("repeated-entry attack: stamped = {}", stamped.len());
    assert!(
        stamped.is_empty(),
        "MISCOMPILE HOLE: a doubly-read out coordinate (axis 1 unread) inherited injectivity"
    );
}
