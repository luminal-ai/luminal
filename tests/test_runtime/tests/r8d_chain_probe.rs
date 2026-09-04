//! ROUND 8d: is the matmul CHAIN load-bearing in the reading arms, or can
//! the arms reach the index map through a product shape carried on the
//! site?
//!
//! THE ADVERSARIAL CASE (step 2 of the charter): a SQUARE weight used
//! BOTH ways — one matmul consumes `b` as B[k,n], another consumes the same
//! `b` as B[n,k]. Because k == n, the two maps have the SAME source shape
//! term, and both applies target the SAME product shape term, so they are
//! hash-consed siblings living side by side in the e-graph.
//!
//! This is not a contrived shape: tied/shared weights used as W and W^T
//! are ordinary transformer practice.
//!
//! If a reading arm reaches the map via (b, prod_shape) alone, the A[m,k],B[n,k]
//! site can pick up the B[k,n] sibling's map and mint an operation=N
//! descriptor — which would NOT transpose the weight. The chain premise
//! (the apply must be a multiplicand of the product THIS out reduces) is
//! what excludes it.

use std::collections::BTreeSet;

use luminal::layout_ir::ExtractionSite;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use test_runtime::cublaslt_marker::{CublasLtForm, parse_spec};

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

/// Two matmuls sharing ONE square weight `b` [3,3]:
///   out1 = x[2,3] @ b        (A[m,k],B[k,n] spelling, map (c0 c1))
///   out2 = y[2,3] @ b^T      (A[m,k],B[n,k] spelling, map (c1 c0))
/// Both product shapes are [2,3,3] — the same term.
fn shared_square_weight_both_ways() -> String {
    format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 3) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let y_logical (LogicalTensorInputLit (LogicalIdLit "y") a_shape (F32)))
(let b_logical (LogicalTensorInputLit (LogicalIdLit "b") b_shape (F32)))
(let lhs_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let b_kn_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let b_nk_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 1)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical lhs_map prod_shape))
(let y_applied (LogicalIndexMapApply y_logical lhs_map prod_shape))
(let b_kn_applied (LogicalIndexMapApply b_logical b_kn_map prod_shape))
(let b_nk_applied (LogicalIndexMapApply b_logical b_nk_map prod_shape))
(let out1_logical (LogicalReduceSum (LogicalMul x_applied b_kn_applied) 0))
(let out2_logical (LogicalReduceSum (LogicalMul y_applied b_nk_applied) 0))
(let a_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let b_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical a_layout))
(let y_lt (LayoutTensorLit y_logical a_layout))
(let b_lt (LayoutTensorLit b_logical b_layout))
(let out1_lt (LayoutTensorLit out1_logical out_layout))
(let out2_lt (LayoutTensorLit out2_logical out_layout))
(let bx (BufferLit 10))
(set (buffer-access-of bx) (ReadOnly))
(set (buffer-freed-by bx) (CallerFrees))
(let by (BufferLit 11))
(set (buffer-access-of by) (ReadOnly))
(set (buffer-freed-by by) (CallerFrees))
(let bb (BufferLit 12))
(set (buffer-access-of bb) (ReadOnly))
(set (buffer-freed-by bb) (CallerFrees))
(let bo1 (BufferLit 13))
(set (buffer-access-of bo1) (ReadWrite))
(set (buffer-freed-by bo1) (CallerFrees))
(let bo2 (BufferLit 14))
(set (buffer-access-of bo2) (ReadWrite))
(set (buffer-freed-by bo2) (CallerFrees))
(let btx (BufferTensorLit x_lt bx))
(let bty (BufferTensorLit y_lt by))
(let btb (BufferTensorLit b_lt bb))
(let bto1 (BufferTensorLit out1_lt bo1))
(let bto2 (BufferTensorLit out2_lt bo2))
(let output (BufferOutputLit (BufferTensorCons bto1 (BufferTensorCons bto2 (BufferTensorNil)))))
{SCHEDULE}
"#
    )
}

fn class_of_child(
    s: &EGraph,
    node: &luminal::prelude::egraph_serialize::Node,
    i: usize,
) -> Option<ClassId> {
    node.children
        .get(i)
        .and_then(|id| s.nodes.get(id))
        .map(|c| c.eclass.clone())
}

fn class_has(s: &EGraph, class: &ClassId, op: &str) -> bool {
    s.nodes.values().any(|n| &n.eclass == class && n.op == op)
}

/// (site out class, operation, layout tensor class) for every A reading.
fn a_readings(s: &EGraph) -> Vec<(ClassId, &'static str, ClassId)> {
    let mut out = Vec::new();
    for n in s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOperandADescriptor")
    {
        let Some(site_class) = class_of_child(s, n, 0) else {
            continue;
        };
        let site_out = s
            .nodes
            .values()
            .find(|m| m.eclass == site_class && m.op == "CublasLtLogicalMatmulSite")
            .and_then(|m| class_of_child(s, m, 2));
        let Some(site_out) = site_out else { continue };
        let Some(lt) = class_of_child(s, n, 1) else {
            continue;
        };
        let op = match class_of_child(s, n, 2) {
            Some(c) if class_has(s, &c, "CublasLtOperationT") => "T",
            Some(c) if class_has(s, &c, "CublasLtOperationN") => "N",
            _ => "?",
        };
        out.push((site_out, op, lt));
    }
    out
}

/// THE GATE. With the CURRENT chain-bearing arms, each of the two sites
/// must have exactly ONE A reading, carrying ITS OWN spelling's
/// operation — the B[k,n] site N, the B[n,k] site T. A second reading on either
/// site means a foreign apply was picked up.
#[test]
fn r8d_shared_square_weight_each_site_reads_only_its_own_map() {
    let fx = shared_square_weight_both_ways();
    let s = test_runtime::serialize_fixture(&fx);

    let sites: BTreeSet<ClassId> = s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .map(|n| n.eclass.clone())
        .collect();
    let applies = s
        .nodes
        .values()
        .filter(|n| n.op == "LogicalIndexMapApply")
        .count();
    let readings = a_readings(&s);
    println!(
        "r8d shared square weight: {} sites, {applies} applies, {} A readings",
        sites.len(),
        readings.len()
    );
    for (site_out, op, lt) in &readings {
        println!("  A reading on site-out {site_out:?}: operation {op} (lt {lt:?})");
    }

    // ROUND-10 RE-PIN (was 2): two matmuls, each with its canonicalized
    // site AND its transpose-sandwich sibling.
    assert_eq!(sites.len(), 4, "two matmuls, two marker sites each");

    // ROUND-11 RE-PIN (was: exactly one A reading per site). Every site's
    // a operand now carries TWO readable layout tensors — its own storage
    // layout (read T here: unit stride lands on k for every a operand of
    // this program) and the collapse-derived COLUMN-FORM layout (x =
    // (x^T)^T re-described as a view of x^T's fresh right-major
    // materialization frame; read N). Two readings per site, each tied to
    // ITS OWN layout tensor, is bounded per-candidate-consistent
    // multiplicity. The MISCOMPILE signature this probe guards against —
    // a foreign apply's map donating an operation to the wrong site — now
    // shows up as two readings on the SAME layout tensor with DISAGREEING
    // operations, which the loop below rejects.
    let mut per_site: std::collections::BTreeMap<ClassId, Vec<(&'static str, ClassId)>> =
        Default::default();
    for (site_out, op, lt) in readings {
        per_site.entry(site_out).or_default().push((op, lt));
    }
    assert_eq!(per_site.len(), 4, "all four sites have A readings");
    for (site_out, entries) in &per_site {
        assert_eq!(
            entries.len(),
            2,
            "site-out {site_out:?} has {} A readings {entries:?} — expected exactly the \
             storage-frame and column-form-frame pair",
            entries.len()
        );
        // per-layout-tensor consistency: one operation per layout tensor.
        let mut by_lt: std::collections::BTreeMap<ClassId, std::collections::BTreeSet<&str>> =
            Default::default();
        for (op, lt) in entries {
            by_lt.entry(lt.clone()).or_default().insert(op);
        }
        assert_eq!(
            by_lt.len(),
            2,
            "the two readings ride two DISTINCT layout tensors"
        );
        for (lt, ops) in &by_lt {
            assert_eq!(
                ops.len(),
                1,
                "layout tensor {lt:?} on site-out {site_out:?} read with disagreeing \
                 operations {ops:?} — a FOREIGN apply leaked in (the miscompile signature)"
            );
        }
        // and the pair covers both frames: {N, T}.
        let mut ops: Vec<&str> = entries.iter().map(|(op, _)| *op).collect();
        ops.sort();
        assert_eq!(
            ops,
            vec!["N", "T"],
            "each site reads its storage frame (T) and its column-form frame (N)"
        );
    }
}

/// ROUND 10 — THE SOUNDNESS GATE (re-expressed per Austin's charter):
/// bounded multiplicity of candidates is fine (election picks); a
/// candidate whose operation is inconsistent with its own descriptor's
/// bytes is a MISCOMPILE. Every candidate is parsed via the extractor
/// (whose call-frame derivation, view/storage transposition tripwires and
/// COL ld clamps fire on inconsistency) and its clamps are checked.
#[test]
fn r8d_candidate_count_on_the_shared_weight() {
    let fx = shared_square_weight_both_ways();
    let s = test_runtime::serialize_fixture(&fx);
    let ops = s
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count();
    println!("r8d op candidates on the shared-weight program: {ops}");
    // ROUND-11 RE-PIN (was 4, R10; was 2, R9): every site's a AND b
    // operand now carries two readable layout tensors (storage frame +
    // collapse-derived column-form frame — see the reading probe above),
    // so assembly's cross product mints 2 A x 2 B x 1 D = 4 candidates
    // per site, 4 sites, 16 candidates. Bounded sound multiplicity, every
    // candidate audited below; election picks (the strict level-0 genome
    // never prefers the materialize-first column-form frames).
    assert_eq!(
        ops, 16,
        "four candidates per site — the 2x2 frame cross product"
    );

    // PER-CANDIDATE SOUNDNESS: parse EVERY candidate enode; parse_spec's
    // internal cross-checks panic on any operation/descriptor
    // inconsistency, and the COL clamp is checked here explicitly.
    let mut checked = 0usize;
    for (id, node) in s.nodes.iter() {
        if node.op != "LayoutTensorOpCublasLt" {
            continue;
        }
        let site = ExtractionSite {
            egraph: &s,
            node_id: id,
            node,
        };
        let spec =
            parse_spec(&site, CublasLtForm::Base).expect("every candidate parses (no silent None)");
        let (m, n, k) = spec.mnk_lits();
        // The frame pair of THIS shared-weight program: (2,3,3)/(3,2,3).
        let canonical = if m <= n { (m, n, k) } else { (n, m, k) };
        assert_eq!(canonical, (2, 3, 3), "candidate frame belongs to the call");
        let rows_a = if spec.trans_a { k } else { m };
        let rows_b = if spec.trans_b { n } else { k };
        let lda = spec.lda.literal().expect("literal lda");
        let ldb = spec.ldb.literal().expect("literal ldb");
        let ldd = spec.ldd.literal().expect("literal ldd");
        assert!(lda >= rows_a, "UNCALLABLE lda={lda} < {rows_a}");
        assert!(ldb >= rows_b, "UNCALLABLE ldb={ldb} < {rows_b}");
        assert!(ldd >= m, "UNCALLABLE ldd={ldd} < {m}");
        checked += 1;
        println!(
            "  candidate ({m},{n},{k}) ta={} tb={} lda={lda} ldb={ldb} ldd={ldd}: SOUND",
            spec.trans_a, spec.trans_b
        );
    }
    // ROUND-11 RE-PIN (was 4): see the candidate-count re-pin above.
    assert_eq!(checked, 16, "every candidate audited");
}
