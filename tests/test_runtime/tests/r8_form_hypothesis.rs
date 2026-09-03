//! ROUND 8 / E1(a)+(c): Austin's hypothesis — "wouldn't there always be a
//! 'strided' constructor which could be used during extraction?"
//!
//! (a) Per layout FAMILY, after saturation: does the layout class of each
//!     LayoutTensorLit contain a StridedElementLayoutLit spelling? A "no"
//!     anywhere is the design answer.
//! (b) THE CRUX: is "find A strided spelling" spelling-INDEPENDENT? A
//!     layout class holding two strided spellings with DIFFERENT pitch
//!     factors would make the uniform walk exactly the hazard the doctrine
//!     forbids. Measured directly here.
//!
//! Observational: these tests report, and assert only what they measure.

use std::collections::{BTreeMap, BTreeSet};

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

fn class_has(s: &EGraph, class: &ClassId, op: &str) -> bool {
    s.nodes.values().any(|n| &n.eclass == class && n.op == op)
}

/// Every layout class reachable as a LayoutTensorLit's layout child, with
/// the set of layout-constructor names present in that class.
fn layout_classes(s: &EGraph) -> BTreeMap<ClassId, BTreeSet<String>> {
    let mut out: BTreeMap<ClassId, BTreeSet<String>> = BTreeMap::new();
    for n in s.nodes.values() {
        if n.op != "LayoutTensorLit" {
            continue;
        }
        let Some(layout) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        let class = layout.eclass.clone();
        let names: BTreeSet<String> = s
            .nodes
            .values()
            .filter(|m| m.eclass == class)
            .map(|m| m.op.clone())
            .filter(|op| op.ends_with("ElementLayoutLit"))
            .collect();
        out.insert(class, names);
    }
    out
}

/// For one layout class: the set of DISTINCT row-entry pitch factor
/// classes discoverable by the extractor's discriminator (the co-factor of
/// an (IntMul coord pitch) spelling whose other child is a CoordVar), and
/// the literal values among them.
fn pitch_factors(s: &EGraph, class: &ClassId) -> (BTreeSet<ClassId>, BTreeSet<i64>) {
    let mut classes = BTreeSet::new();
    let mut lits = BTreeSet::new();
    for layout in s
        .nodes
        .values()
        .filter(|n| n.eclass == *class && n.op == "StridedElementLayoutLit")
    {
        let Some(chain) = layout.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        let chain_class = chain.eclass.clone();
        for cons in s
            .nodes
            .values()
            .filter(|n| n.eclass == chain_class && n.op == "IntAffineExprCons")
        {
            let Some(row_entry) = cons.children.first().and_then(|id| s.nodes.get(id)) else {
                continue;
            };
            let entry_class = row_entry.eclass.clone();
            for mul in s
                .nodes
                .values()
                .filter(|n| n.eclass == entry_class && n.op == "IntMul")
            {
                for child in 0..2usize {
                    let other = 1 - child;
                    let other_is_coord = mul
                        .children
                        .get(other)
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| class_has(s, &c.eclass, "CoordVar"))
                        .unwrap_or(false);
                    if !other_is_coord {
                        continue;
                    }
                    if let Some(f) = mul.children.get(child).and_then(|id| s.nodes.get(id)) {
                        classes.insert(f.eclass.clone());
                        for lit in s
                            .nodes
                            .values()
                            .filter(|n| n.eclass == f.eclass && n.op == "IntLit")
                        {
                            if let Some(v) = lit
                                .children
                                .first()
                                .and_then(|id| s.nodes.get(id))
                                .and_then(|c| c.op.parse::<i64>().ok())
                            {
                                lits.insert(v);
                            }
                        }
                    }
                }
            }
        }
    }
    (classes, lits)
}

fn report(name: &str, s: &EGraph) -> (usize, usize) {
    let classes = layout_classes(s);
    let mut with_strided = 0usize;
    let mut without = 0usize;
    for (class, names) in &classes {
        let strided = names.iter().any(|n| n == "StridedElementLayoutLit");
        if strided {
            with_strided += 1;
        } else {
            without += 1;
            println!("  {name}: layout class {class:?} has NO strided spelling; has {names:?}");
        }
    }
    println!(
        "E1(a) {name}: {} layout classes, {with_strided} WITH a strided spelling, {without} without",
        classes.len()
    );
    (with_strided, without)
}

/// Family 1: right-major contiguous, from the LIVE recorder.
#[test]
fn e1a_right_major_contiguous_live_recorder() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let s = test_runtime::serialize_fixture(&text);
    let (with_strided, without) = report("RM-live", &s);
    assert!(with_strided > 0, "the recorder's contiguous layouts exist");
    assert_eq!(
        without, 0,
        "HYPOTHESIS: every layout class carries a strided spelling"
    );
}

/// Family 2: left-major contiguous, boundary-seeded.
#[test]
fn e1a_left_major_contiguous_seeded() {
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_layout_lm (LeftMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout_lm))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let output (BufferOutputLit (BufferTensorCons x_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let (with_strided, without) = report("LM-seeded", &s);
    assert!(with_strided > 0 || without > 0, "layout classes exist");
    assert_eq!(
        without, 0,
        "HYPOTHESIS: the left-major class also carries a strided spelling"
    );
}

/// Family 3: estate-padded strided (minted directly; round-8c deleted the
/// creator rewrite from the cuBLASLt vocabulary).
#[test]
fn e1a_estate_padded_strided() {
    let fx = format!(
        r#"(let s_var (IntVar "s"))
(set (lower-bound-of s_var) (bigint 2))
(set (upper-bound-of s_var) (bigint 8))
(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_layout_contig (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let x_lt_contig (LayoutTensorLit x_logical x_layout_contig))
(let x_layout_static (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_static (LayoutTensorLit x_logical x_layout_static))
(set (injectivity-of x_lt_static) (Injective))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt_contig x_buffer_id))
(let output (BufferOutputLit (BufferTensorCons x_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let (with_strided, without) = report("estate-padded", &s);
    assert!(with_strided > 0, "the padded layout exists and is strided");
    assert_eq!(without, 0, "HYPOTHESIS: holds for the padded family too");
}

// ===========================================================================
// (c) THE CRUX: is the strided spelling's pitch CANONICAL per class?
// ===========================================================================

/// Across a battery of fixtures (live matmul, LM-seeded, creator-padded,
/// and the rt4-style THREE-pitch tensor), no layout class may hold two
/// strided spellings whose discriminated pitch factors disagree.
#[test]
fn e1c_pitch_is_canonical_per_layout_class() {
    let live = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).relu().output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    // The rt4 shape: ONE logical tensor, THREE layouts (contiguous +
    // pitch 8 + pitch 16). Round-3's rt4 reported three distinct layout
    // CLASSES; confirm rather than assume.
    let three_pitch = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let x_layout_s8 (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_s8 (LayoutTensorLit x_logical x_layout_s8))
(set (injectivity-of x_lt_s8) (Injective))
(let x_layout_s16 (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 16))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_s16 (LayoutTensorLit x_logical x_layout_s16))
(set (injectivity-of x_lt_s16) (Injective))
(let b0 (BufferLit 10))
(set (buffer-access-of b0) (ReadOnly))
(set (buffer-freed-by b0) (CallerFrees))
(let bt0 (BufferTensorLit x_lt b0))
(let b1 (BufferLit 11))
(set (buffer-access-of b1) (ReadOnly))
(set (buffer-freed-by b1) (CallerFrees))
(let bt1 (BufferTensorLit x_lt_s8 b1))
(let b2 (BufferLit 12))
(set (buffer-access-of b2) (ReadOnly))
(set (buffer-freed-by b2) (CallerFrees))
(let bt2 (BufferTensorLit x_lt_s16 b2))
(let output (BufferOutputLit (BufferTensorCons bt0 (BufferTensorCons bt1 (BufferTensorCons bt2 (BufferTensorNil))))))
{SCHEDULE}
"#
    );
    let mut violations = Vec::new();
    for (name, text) in [("live-bias-relu", live), ("three-pitch", three_pitch)] {
        let s = test_runtime::serialize_fixture(&text);
        let classes = layout_classes(&s);
        let mut multi = 0usize;
        for (class, names) in &classes {
            if !names.iter().any(|n| n == "StridedElementLayoutLit") {
                continue;
            }
            let (factor_classes, lits) = pitch_factors(&s, class);
            if factor_classes.len() > 1 || lits.len() > 1 {
                multi += 1;
                violations.push(format!(
                    "{name}: layout class {class:?} has {} distinct pitch factor classes, literals {lits:?}",
                    factor_classes.len()
                ));
            }
        }
        println!(
            "E1(c) {name}: {} layout classes, {} strided, {multi} with AMBIGUOUS pitch",
            classes.len(),
            classes
                .values()
                .filter(|n| n.iter().any(|x| x == "StridedElementLayoutLit"))
                .count()
        );
    }
    for v in &violations {
        println!("  AMBIGUITY {v}");
    }
    assert!(
        violations.is_empty(),
        "the uniform walk needs a canonical pitch per layout class; found {} ambiguous class(es)",
        violations.len()
    );
}

// ===========================================================================
// E1, PINNED: orientation is not in the LAYOUT — it is in the OPERATION.
//
// The measurement below is unchanged and still true: at extent 1 an axis
// coordinate welds to the zero class, so its stride entry also carries a
// bare-coordinate spelling and BOTH chain positions look unit-stride.
// RM[1,4] and LM[1,4] denote the identical element mapping, so no walk of
// the layout can name the orientation.
//
// The round-8 conclusion drawn from that — "therefore the form tag is
// irreducible" — was WRONG, and this test now pins why. Orientation never
// had to come from the layout: it is already in the descriptor's
// OPERATION child, which the rules prove from the index map. Given the
// operation, the column-major view is fixed (A: N->(n,k), T->(k,n);
// B: N->(k,m), T->(m,k); D: (n,m)) and ld is that view's row count unless
// the layout is padded. The degenerate gemv is the proof: the SAME [1,4]
// bytes read as (op=N, ld=4) and (op=T, ld=1) — a matched pair the
// operation selects, not an ambiguity the layout must resolve.
// ===========================================================================
#[test]
fn e1_orientation_lives_in_the_operation_not_the_layout() {
    let seed = |ctor: &str| {
        format!(
            r#"(let a_shape (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let x_layout ({ctor} a_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let b0 (BufferLit 10))
(set (buffer-access-of b0) (ReadOnly))
(set (buffer-freed-by b0) (CallerFrees))
(let bt0 (BufferTensorLit x_lt b0))
(let output (BufferOutputLit (BufferTensorCons bt0 (BufferTensorNil))))
{SCHEDULE}
"#
        )
    };
    // For each orientation, ask: how many chain positions carry a BARE
    // coordinate spelling (i.e. look unit-stride)?
    let mut unit_counts = Vec::new();
    for ctor in [
        "RightMajorContiguousElementLayoutLit",
        "LeftMajorContiguousElementLayoutLit",
    ] {
        let s = test_runtime::serialize_fixture(&seed(ctor));
        let mut counted = None;
        for (class, names) in layout_classes(&s) {
            if !names.iter().any(|n| n == ctor) {
                continue;
            }
            let Some(strided) = s
                .nodes
                .values()
                .find(|n| n.eclass == class && n.op == "StridedElementLayoutLit")
            else {
                continue;
            };
            let Some(chain) = strided.children.get(1).and_then(|id| s.nodes.get(id)) else {
                continue;
            };
            let mut cur = chain.eclass.clone();
            let mut unit = 0usize;
            let mut total = 0usize;
            loop {
                let Some(cons) = s
                    .nodes
                    .values()
                    .find(|n| n.eclass == cur && n.op == "IntAffineExprCons")
                else {
                    break;
                };
                if let Some(entry) = cons.children.first().and_then(|id| s.nodes.get(id)) {
                    total += 1;
                    if class_has(&s, &entry.eclass, "CoordVar") {
                        unit += 1;
                    }
                }
                let Some(tail) = cons.children.get(1).and_then(|id| s.nodes.get(id)) else {
                    break;
                };
                cur = tail.eclass.clone();
                if total > 4 {
                    break;
                }
            }
            counted = Some((unit, total));
            break;
        }
        let (unit, total) = counted.expect("the seeded layout has a strided spelling");
        println!("E1 finding [1,4] {ctor}: {unit}/{total} chain positions look unit-stride");
        unit_counts.push(unit);
    }
    assert!(
        unit_counts.iter().any(|u| *u >= 2),
        "at extent 1 at least one orientation shows MULTIPLE unit-stride positions — \
         which is why the extractor must NOT infer orientation from the layout; it \
         reads the OPERATION child instead (got {unit_counts:?})"
    );
}
