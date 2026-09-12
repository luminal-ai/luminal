//! E3 pins: what the injectivity premise and the nonnegativity bound each
//! refuse (the revert-probe anchors for the deleted certificate relation).

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (saturate (saturate (run) (run backend) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

fn skeleton(x_layout_lines: &str) -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
{x_layout_lines}
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
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
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    )
}

/// Operand readings (A and B — the roles are unswapped since round 10,
/// so the pitched x is the A operand and its transpose view a sibling B
/// operand) whose layout class holds a Strided spelling and NO contiguous
/// spelling of EITHER majority. ROUND-11 COUNTER FIX: the original proxy
/// excluded only RightMajorContiguous; the transpose views the round-11
/// rewrites mint give every operand a LEFT-major contiguous frame layout
/// (the transpose of a fresh right-major materialization), whose readings
/// are certified by the preamble's contiguous arms and are NOT what these
/// pins refuse. The refusal target is exactly the layouts with no
/// contiguity certificate: the raw pitched x and every composed view
/// carrying its expression.
fn readings_over_raw_strided(s: &luminal::prelude::egraph_serialize::EGraph) -> usize {
    let class_has = |class: &luminal::prelude::egraph_serialize::ClassId, op: &str| {
        s.nodes.values().any(|n| &n.eclass == class && n.op == op)
    };
    s.nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandBDescriptor" || n.op == "CublasLtOperandADescriptor")
        .filter(|n| {
            let Some(lt) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
                return false;
            };
            let lt_class = lt.eclass.clone();
            let layout = s.nodes.values().find_map(|m| {
                if m.eclass == lt_class && m.op == "LayoutTensorLit" {
                    m.children
                        .get(1)
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| c.eclass.clone())
                } else {
                    None
                }
            });
            let Some(layout) = layout else { return false };
            class_has(&layout, "StridedElementLayoutLit")
                && !class_has(&layout, "RightMajorContiguousElementLayoutLit")
                && !class_has(&layout, "LeftMajorContiguousElementLayoutLit")
        })
        .count()
}

/// A RAW strided x (pitch 8) — structurally perfect, but NO creator and
/// therefore NO injectivity fact: the row-major arm must refuse (the
/// injectivity premise is the deleted certificate's replacement).
#[test]
fn e3_raw_uncertified_strided_refuses() {
    let fx = skeleton(
        r#"(let x_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))"#,
    );
    let s = test_runtime::serialize_fixture(&fx);
    let b = readings_over_raw_strided(&s);
    println!("E3 raw uncertified strided: {b} uncertified-layout reading(s)");
    assert_eq!(b, 0, "no injectivity fact, no reading (fail-closed)");
}

/// The adversarial finding, pinned: a NEGATIVE-pitch strided x with a
/// hand-asserted (TRUE!) injectivity fact — injective, structurally
/// row-pitch-shaped, NOT callable (cuBLASLt rejects negative ld). The
/// nonnegativity premise (entry lower bound >= 0) refuses it; without
/// that premise, injectivity alone would have minted an uncallable
/// reading — the one content the old certificate carried beyond
/// injectivity, now carried by the lattice bound instead.
#[test]
fn e3_negative_pitch_refuses_despite_true_injectivity() {
    let fx = skeleton(
        r#"(let x_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit -16))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(set (injectivity-of x_lt) (Injective))"#,
    );
    let s = test_runtime::serialize_fixture(&fx);
    let b = readings_over_raw_strided(&s);
    println!(
        "E3 negative pitch (true injectivity asserted): {b} uncertified/negative-layout reading(s)"
    );
    assert_eq!(
        b, 0,
        "negative pitch refused via the entry's lattice lower bound"
    );
}
