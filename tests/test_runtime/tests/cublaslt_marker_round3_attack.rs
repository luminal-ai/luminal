//! ROUND-3 ADVERSARIAL BATTERY — re-derived 2026-08-24 against the CURRENT
//! (round-7) design, using the recovered name inventory
//! (`RECOVERY/tests_cublaslt_marker_round3_attack.txt`) as the coverage
//! checklist.
//!
//! Families:
//!   RC* reading-set COHERENCE  — assembly joins A/B/D descriptor terms and
//!       builds the dataflow Lit from THOSE descriptors' own layout
//!       tensors. Attacked with a STRUCTURAL CHECKER that walks every op
//!       enode of all four constructor names (never spot asserts).
//!   RE* per-ENODE spec pairing — dual readings put several op spellings in
//!       one e-class; electing each in turn must parse THAT enode's reading.
//!   RP* candidate PRODUCT / scaling — |A| x |B| x |D| per site, bounded by
//!       arms x forms and NEVER by program size.
//!   RS* SYMBOLIC edges.
//!   RT* STRIDE-SPELLING robustness.
//!   RU* degenerate corners + THE WELD-HARVESTING TRIPWIRE (ru3/ru4).
//!
//! ru3/ru4 are the round-8 acid-test fixtures: round 4's rule pattern
//! `(IntMul (CoordVar shape 1) ?ld)` enumerated every spelling of the
//! zero-welded entry class at m=1 and harvested EVERY literal in the
//! program as a candidate ld (10 candidates from one site; 23 once an
//! unrelated [1,37] tensor was added). These two tests fail if that hazard
//! returns — at the rule level (reading counts / reading provenance) AND at
//! the extractor level (elected ld values).

#![allow(clippy::type_complexity)]

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node, NodeId};
use test_runtime::cublaslt_marker::{CuDim, CuEpilogue, CublasLt, LtMatmulSpec};

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    // ROUND 10 (view admission in ELECTION): the sibling site's result is
    // routed to the recorder's boundary value by a transpose VIEW; prefer
    // the view op over materialize/copy wherever both produce a class.
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

/// (constructor, Lit arity, c slot, bias slot, epilogue slot)
const OP_FORMS: [(&str, usize, Option<usize>, Option<usize>, usize); 4] = [
    ("LayoutTensorOpCublasLt", 2, None, None, 4),
    ("LayoutTensorOpCublasLtBias", 3, None, Some(4), 5),
    ("LayoutTensorOpCublasLtAccumulate", 3, Some(4), None, 5),
    (
        "LayoutTensorOpCublasLtAccumulateBias",
        4,
        Some(4),
        Some(5),
        6,
    ),
];

// ===========================================================================
// HELPERS (the recovered helper inventory)
// ===========================================================================

fn count_op(s: &EGraph, op: &str) -> usize {
    s.nodes.values().filter(|n| n.op == op).count()
}

fn count_cublaslt(s: &EGraph) -> usize {
    s.nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count()
}

fn class_of_child(s: &EGraph, node: &Node, index: usize) -> Option<ClassId> {
    node.children
        .get(index)
        .and_then(|id| s.nodes.get(id))
        .map(|child| child.eclass.clone())
}

/// Every node of `op` in `class` — subsumed spellings INCLUDED, exactly like
/// the marker's own value readers (`nodes_in_class_value`).
fn nodes_in_class<'a>(s: &'a EGraph, class: &ClassId, op: &str) -> Vec<&'a Node> {
    s.nodes
        .values()
        .filter(|n| &n.eclass == class && n.op == op)
        .collect()
}

fn class_has(s: &EGraph, class: &ClassId, op: &str) -> bool {
    s.nodes.values().any(|n| &n.eclass == class && n.op == op)
}

fn short(class: &ClassId) -> String {
    let text = class.to_string();
    text.chars()
        .rev()
        .take(6)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

// --- the independent numeric oracle (mirrors the marker's walks) ----------

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Dim {
    Lit(i64),
    Sym(ClassId),
    Unknown,
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Lit(v) => write!(f, "{v}"),
            Dim::Sym(c) => write!(f, "sym({})", short(c)),
            Dim::Unknown => write!(f, "?"),
        }
    }
}

impl Dim {
    fn lit(&self) -> Option<i64> {
        match self {
            Dim::Lit(v) => Some(*v),
            _ => None,
        }
    }
}

fn parse_dim(s: &EGraph, class: &ClassId) -> Dim {
    for lit in nodes_in_class(s, class, "IntLit") {
        if let Some(payload) = class_of_child(s, lit, 0) {
            if let Some(v) = s
                .nodes
                .values()
                .filter(|n| n.eclass == payload)
                .find_map(|n| n.op.parse::<i64>().ok())
            {
                return Dim::Lit(v);
            }
        }
    }
    Dim::Sym(class.clone())
}

fn logical_of_lt(s: &EGraph, lt: &ClassId) -> Option<ClassId> {
    nodes_in_class(s, lt, "LayoutTensorLit")
        .first()
        .and_then(|n| class_of_child(s, n, 0))
}

fn layout_of_lt(s: &EGraph, lt: &ClassId) -> Option<ClassId> {
    nodes_in_class(s, lt, "LayoutTensorLit")
        .first()
        .and_then(|n| class_of_child(s, n, 1))
}

fn storage_dims(s: &EGraph, logical: &ClassId) -> Option<Vec<Dim>> {
    let shape_class = s.nodes.values().find_map(|n| {
        if n.op != "shape-of" {
            return None;
        }
        let child = n.children.first()?;
        (&s.nodes.get(child)?.eclass == logical).then(|| n.eclass.clone())
    })?;
    let shape_lit = *nodes_in_class(s, &shape_class, "ShapeLit").first()?;
    let mut list = class_of_child(s, shape_lit, 0)?;
    let mut dims = Vec::new();
    loop {
        if !nodes_in_class(s, &list, "IntExprNil").is_empty() {
            break;
        }
        let cons = *nodes_in_class(s, &list, "IntExprCons").first()?;
        dims.push(parse_dim(s, &class_of_child(s, cons, 0)?));
        list = class_of_child(s, cons, 1)?;
    }
    Some(dims)
}

/// THE RAW HARVEST SURFACE: every class that sits as the non-coordinate
/// co-factor of an `IntMul` in the ROW-ENTRY position of any strided
/// spelling of `layout_class`. Round 4's rule destructured exactly this
/// position, so this is the set it would have harvested as candidate lds.
fn pitch_factor_classes(s: &EGraph, layout_class: &ClassId) -> Vec<ClassId> {
    let mut factors: Vec<ClassId> = Vec::new();
    for layout in nodes_in_class(s, layout_class, "StridedElementLayoutLit") {
        let Some(chain) = class_of_child(s, layout, 1) else {
            continue;
        };
        let Some(cons) = nodes_in_class(s, &chain, "IntAffineExprCons")
            .first()
            .copied()
        else {
            continue;
        };
        // ROUND 10: the pitch may sit on EITHER rank-2 entry — row-major-
        // form layouts pitch the head, the sibling's transpose-view (col-
        // major-form) layouts pitch the second entry. Find the UNIT axis
        // by the bare-CoordVar membership test (exactly the extractor's
        // walk) and read the pitch from the OTHER entry only.
        let head = class_of_child(s, cons, 0);
        let second = class_of_child(s, cons, 1)
            .and_then(|tail| {
                nodes_in_class(s, &tail, "IntAffineExprCons")
                    .first()
                    .copied()
                    .cloned()
            })
            .and_then(|cons2| class_of_child(s, &cons2, 0));
        let is_bare = |c: &Option<ClassId>| {
            c.as_ref()
                .map(|c| class_has(s, c, "CoordVar"))
                .unwrap_or(false)
        };
        let mut entries: Vec<ClassId> = Vec::new();
        if is_bare(&second) {
            if let Some(h) = head.clone() {
                entries.push(h); // row-major-form: pitch on the head
            }
        } else if is_bare(&head) {
            if let Some(sec) = second.clone() {
                entries.push(sec); // col-major-form: pitch on the second
            }
        } else if let Some(h) = head.clone() {
            entries.push(h);
        }
        for row_entry in &entries {
            for mul in nodes_in_class(s, row_entry, "IntMul") {
                for child in 0..2usize {
                    let other = 1 - child;
                    let other_is_coord = class_of_child(s, mul, other)
                        .map(|c| class_has(s, &c, "CoordVar"))
                        .unwrap_or(false);
                    if !other_is_coord {
                        continue;
                    }
                    if let Some(factor) = class_of_child(s, mul, child) {
                        if !factors.contains(&factor) {
                            factors.push(factor);
                        }
                    }
                }
            }
        }
    }
    factors
}

/// The COL-view `ld` this reading commits to — re-implemented for the
/// round-8b vocabulary (no form tag): the OPERATION fixes the view's row
/// count, and a padded layout overrides it with its pitch.
///
/// Round 8b: with the form child deleted this oracle necessarily
/// mirrors the production derivation, where before it could disagree
/// about orientation. Its remaining independent content is the padding
/// override and the dead-axis policy.
fn reading_ld(
    s: &EGraph,
    role: &str,
    operation: &str,
    site: &ClassId,
    lt: &ClassId,
    storage: &[Dim],
) -> Dim {
    if storage.len() != 2 {
        return Dim::Unknown;
    }
    // Logical frame from the SITE: a is [m,k], out is [m,n].
    let Some(site_node) = nodes_in_class(s, site, "CublasLtLogicalMatmulSite")
        .first()
        .copied()
    else {
        return Dim::Unknown;
    };
    let (Some(a_log), Some(out_log)) = (
        class_of_child(s, site_node, 0),
        class_of_child(s, site_node, 2),
    ) else {
        return Dim::Unknown;
    };
    let (Some(a_st), Some(d_st)) = (storage_dims(s, &a_log), storage_dims(s, &out_log)) else {
        return Dim::Unknown;
    };
    if a_st.len() != 2 || d_st.len() != 2 {
        return Dim::Unknown;
    }
    // ROUND 10 (unswapped frame): call m/n are the SITE's own out extents
    // and k is the a-storage extent that is not m (a stores a permutation
    // of (m, k) — the A[k,m],B[k,n] sibling stores [k, m]).
    let (lm, ln) = (d_st[0].clone(), d_st[1].clone());
    let lk = if a_st[0] == lm {
        a_st[1].clone()
    } else {
        a_st[0].clone()
    };
    let rows = match (role, operation) {
        ("A", "CublasLtOperationN") => lm,
        ("A", "CublasLtOperationT") => lk,
        ("B", "CublasLtOperationN") => lk,
        ("B", "CublasLtOperationT") => ln,
        ("D", _) => lm,
        _ => return Dim::Unknown,
    };
    // Padding override, orientation-general (round 10): the chain may be
    // row-major-form (operand tensors) or col-major-form (the sibling's
    // transpose-view D tensors); a discriminated pitch equal to EITHER
    // neighbouring extent is contiguous (and then ld == rows by
    // construction); anything else is the creator's pitch. Dead axes
    // (extent 1) are never read.
    if storage[0] != Dim::Lit(1) && storage[1] != Dim::Lit(1) {
        if let Some(layout) = layout_of_lt(s, lt) {
            let factors = pitch_factor_classes(s, &layout);
            if let [factor] = factors.as_slice() {
                let f = parse_dim(s, factor);
                if f != storage[1] && f != storage[0] && f != Dim::Unknown && f != Dim::Lit(1) {
                    return f;
                }
            }
        }
    }
    rows
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Reading {
    role: String,
    site: ClassId,
    lt: ClassId,
    logical: ClassId,
    operation: String,
    ld: Dim,
}

fn operation_name(s: &EGraph, class: &ClassId) -> String {
    for name in ["CublasLtOperationN", "CublasLtOperationT"] {
        if class_has(s, class, name) {
            return name.to_string();
        }
    }
    "<none>".to_string()
}

/// Every descriptor reading in the egraph, with the ld each commits to.
fn all_readings(s: &EGraph) -> Vec<Reading> {
    let mut out = Vec::new();
    for (role, ctor, op_slot) in [
        ("A", "CublasLtOperandADescriptor", Some(2usize)),
        ("B", "CublasLtOperandBDescriptor", Some(2)),
        ("D", "CublasLtOutputDDescriptor", None),
    ] {
        for node in s.nodes.values().filter(|n| n.op == ctor) {
            let (Some(site), Some(lt)) = (class_of_child(s, node, 0), class_of_child(s, node, 1))
            else {
                continue;
            };
            let operation = op_slot
                .and_then(|slot| class_of_child(s, node, slot))
                .map(|c| operation_name(s, &c))
                .unwrap_or_else(|| "-".into());
            let logical = logical_of_lt(s, &lt).unwrap_or_else(|| lt.clone());
            let storage = storage_dims(s, &logical).unwrap_or_default();
            let ld = reading_ld(s, role, &operation, &site, &lt, &storage);
            out.push(Reading {
                role: role.to_string(),
                site,
                lt,
                logical,
                operation,
                ld,
            });
        }
    }
    out.sort();
    out
}

/// Readings attached to one site, keyed by role.
fn readings_for_site(s: &EGraph, site: &ClassId) -> BTreeMap<String, Vec<Reading>> {
    let mut map: BTreeMap<String, Vec<Reading>> = BTreeMap::new();
    for r in all_readings(s) {
        if &r.site == site {
            map.entry(r.role.clone()).or_default().push(r);
        }
    }
    map
}

fn sites(s: &EGraph) -> Vec<ClassId> {
    let mut out: Vec<ClassId> = s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .map(|n| n.eclass.clone())
        .collect();
    out.sort();
    out.dedup();
    out
}

fn dump_reading_sets(tag: &str, s: &EGraph) {
    for site in sites(s) {
        let per_role = readings_for_site(s, &site);
        let counts: Vec<String> = ["A", "B", "D"]
            .iter()
            .map(|role| {
                format!(
                    "{role}={}",
                    per_role.get(*role).map(|v| v.len()).unwrap_or(0)
                )
            })
            .collect();
        println!(
            "  [{tag}] site {} readings {}",
            short(&site),
            counts.join(" ")
        );
        for role in ["A", "B", "D"] {
            for r in per_role.get(role).into_iter().flatten() {
                println!(
                    "      {role} lt={} logical={} form={} op={} ld={}",
                    short(&r.lt),
                    short(&r.logical),
                    r.operation.trim_start_matches("CublasLtOperation"),
                    r.operation.trim_start_matches("CublasLtOperation"),
                    r.ld
                );
            }
        }
    }
}

// --- Lit walking ----------------------------------------------------------

fn lt_list(s: &EGraph, start: &ClassId) -> Option<(Vec<ClassId>, bool)> {
    let mut class = start.clone();
    let mut out = Vec::new();
    let mut ambiguous = false;
    loop {
        if !nodes_in_class(s, &class, "LayoutTensorNil").is_empty() {
            break;
        }
        let cons = nodes_in_class(s, &class, "LayoutTensorCons");
        if cons.len() > 1 {
            ambiguous = true;
        }
        let cons = *cons.first()?;
        out.push(class_of_child(s, cons, 0)?);
        class = class_of_child(s, cons, 1)?;
        if out.len() > 8 {
            return None;
        }
    }
    Some((out, ambiguous))
}

fn lit_signature(s: &EGraph, lit: &Node) -> Option<(Vec<ClassId>, Vec<ClassId>, bool)> {
    let (ins, a1) = lt_list(s, &class_of_child(s, lit, 0)?)?;
    let (outs, a2) = lt_list(s, &class_of_child(s, lit, 1)?)?;
    Some((ins, outs, a1 || a2))
}

// --- THE STRUCTURAL COHERENCE CHECKER -------------------------------------

/// Walk EVERY op enode of ALL FOUR constructor names and prove the dataflow
/// Lit in its e-class is built from ITS OWN descriptors' layout tensors.
/// Returns the violation list (empty = coherent) and the number of enodes
/// checked.
fn check_reading_set_coherence(s: &EGraph) -> (Vec<String>, usize) {
    let mut violations = Vec::new();
    let mut checked = 0usize;
    // class -> the Lit signature each op enode in it requires
    let mut expected_by_class: BTreeMap<ClassId, Vec<(Vec<ClassId>, Vec<ClassId>)>> =
        BTreeMap::new();

    for (ctor, arity, c_slot, bias_slot, ep_slot) in OP_FORMS {
        for op in s.nodes.values().filter(|n| n.op == ctor) {
            checked += 1;
            let tag = format!("{ctor}@{}", short(&op.eclass));

            let Some(site) = class_of_child(s, op, 0) else {
                violations.push(format!("{tag}: no site child"));
                continue;
            };
            let site_nodes = nodes_in_class(s, &site, "CublasLtLogicalMatmulSite");
            if site_nodes.is_empty() {
                violations.push(format!("{tag}: slot 0 is not a matmul site"));
                continue;
            }
            let site_out = class_of_child(s, site_nodes[0], 2);

            // --- the three descriptor slots, each resolved to ITS OWN lt ---
            let mut lts: [Option<ClassId>; 3] = [None, None, None];
            let mut d_logical = None;
            for (slot, ctor_name, idx) in [
                (1usize, "CublasLtOperandADescriptor", 0usize),
                (2, "CublasLtOperandBDescriptor", 1),
                (3, "CublasLtOutputDDescriptor", 2),
            ] {
                let Some(class) = class_of_child(s, op, slot) else {
                    violations.push(format!("{tag}: no child at descriptor slot {slot}"));
                    continue;
                };
                let descs = nodes_in_class(s, &class, ctor_name);
                if descs.is_empty() {
                    violations.push(format!(
                        "{tag}: slot {slot} holds no {ctor_name} (class {})",
                        short(&class)
                    ));
                    continue;
                }
                let mut slot_lts: BTreeSet<ClassId> = BTreeSet::new();
                for d in descs.iter().copied() {
                    match class_of_child(s, d, 0) {
                        Some(back) if back == site => {}
                        other => violations.push(format!(
                            "{tag}: {ctor_name} back-pointer {:?} != site {}",
                            other.as_ref().map(short),
                            short(&site)
                        )),
                    }
                    if let Some(lt) = class_of_child(s, d, 1) {
                        slot_lts.insert(lt);
                    }
                }
                if slot_lts.len() != 1 {
                    violations.push(format!(
                        "{tag}: descriptor slot {slot} mixes {} distinct layout tensors \
                         — a class-mate reading could be borrowed",
                        slot_lts.len()
                    ));
                }
                let lt = slot_lts.into_iter().next();
                if idx == 2 {
                    d_logical = lt.as_ref().and_then(|l| logical_of_lt(s, l));
                }
                lts[idx] = lt;
            }
            let (Some(a_lt), Some(b_lt), Some(d_lt)) =
                (lts[0].clone(), lts[1].clone(), lts[2].clone())
            else {
                continue;
            };

            // --- the expected Lit: ITS OWN descriptors, contract order ----
            // ROUND 10 (unswapped): Lit slot 0 IS descriptor A's layout
            // tensor and slot 1 descriptor B's — the wiring swap is gone
            // (the transpose-sandwich rewrite carries the role change).
            let mut expected_ins = vec![a_lt.clone(), b_lt.clone()];
            if let Some(slot) = c_slot {
                match class_of_child(s, op, slot) {
                    Some(c) => expected_ins.push(c),
                    None => violations.push(format!("{tag}: no C child at slot {slot}")),
                }
            }
            if let Some(slot) = bias_slot {
                match class_of_child(s, op, slot) {
                    Some(b) => expected_ins.push(b),
                    None => violations.push(format!("{tag}: no bias child at slot {slot}")),
                }
            }
            if expected_ins.len() != arity {
                violations.push(format!(
                    "{tag}: contract arity {arity} but {} payload children",
                    expected_ins.len()
                ));
            }
            let expected_outs = vec![d_lt.clone()];
            expected_by_class
                .entry(op.eclass.clone())
                .or_default()
                .push((expected_ins.clone(), expected_outs.clone()));

            // --- does a Lit in this class carry exactly that? -------------
            let lits = nodes_in_class(s, &op.eclass, "LayoutTensorOpLit");
            if lits.is_empty() {
                violations.push(format!("{tag}: op class carries no LayoutTensorOpLit"));
                continue;
            }
            let mut matched = false;
            for lit in lits.iter().copied() {
                let Some((ins, outs, ambiguous)) = lit_signature(s, lit) else {
                    violations.push(format!("{tag}: unwalkable Lit"));
                    continue;
                };
                if ambiguous {
                    violations.push(format!(
                        "{tag}: Lit operand list class holds several Cons spellings \
                         (welded dataflow lists)"
                    ));
                }
                if ins == expected_ins && outs == expected_outs {
                    matched = true;
                }
            }
            if !matched {
                let seen: Vec<String> = lits
                    .iter()
                    .copied()
                    .filter_map(|l| lit_signature(s, l))
                    .map(|(i, o, _)| {
                        format!(
                            "[{}]->[{}]",
                            i.iter().map(short).collect::<Vec<_>>().join(","),
                            o.iter().map(short).collect::<Vec<_>>().join(",")
                        )
                    })
                    .collect();
                violations.push(format!(
                    "{tag}: CROSS-PAIRED — no Lit matches its own descriptors \
                     [{}]->[{}]; class Lits: {}",
                    expected_ins.iter().map(short).collect::<Vec<_>>().join(","),
                    expected_outs
                        .iter()
                        .map(short)
                        .collect::<Vec<_>>()
                        .join(","),
                    seen.join(" ")
                ));
            }

            // --- D-leak perimeter: an undecorated base op claims the ------
            // --- site's OWN out (decorated forms legitimately move it) ----
            let epilogue = class_of_child(s, op, ep_slot)
                .map(|c| {
                    if class_has(s, &c, "CublasLtEpilogueRelu") {
                        "Relu"
                    } else if class_has(s, &c, "CublasLtEpilogueDefault") {
                        "Default"
                    } else {
                        "<none>"
                    }
                })
                .unwrap_or("<missing>");
            if ctor == "LayoutTensorOpCublasLt" && epilogue == "Default" && d_logical != site_out {
                violations.push(format!(
                    "{tag}: D-LEAK — undecorated base op claims logical {:?}, \
                         site out is {:?}",
                    d_logical.as_ref().map(short),
                    site_out.as_ref().map(short)
                ));
            }
            if epilogue == "<none>" || epilogue == "<missing>" {
                violations.push(format!("{tag}: epilogue slot carries no value"));
            }
        }
    }

    // --- orphan Lits: a dataflow spelling no op enode in the class asked --
    for (class, expected) in &expected_by_class {
        for lit in nodes_in_class(s, class, "LayoutTensorOpLit").into_iter() {
            let Some((ins, outs, _)) = lit_signature(s, lit) else {
                continue;
            };
            if !expected.iter().any(|(i, o)| i == &ins && o == &outs) {
                violations.push(format!(
                    "class {}: ORPHAN Lit [{}]->[{}] belongs to no op enode's own readings",
                    short(class),
                    ins.iter().map(short).collect::<Vec<_>>().join(","),
                    outs.iter().map(short).collect::<Vec<_>>().join(",")
                ));
            }
        }
    }

    (violations, checked)
}

fn assert_coherent(tag: &str, s: &EGraph) -> usize {
    let (violations, checked) = check_reading_set_coherence(s);
    println!("  [{tag}] coherence checker walked {checked} op enode(s)");
    assert!(checked > 0, "{tag}: the checker saw no op enodes to check");
    assert!(
        violations.is_empty(),
        "{tag}: {} reading-set coherence violation(s):\n  {}",
        violations.len(),
        violations.join("\n  ")
    );
    checked
}

// --- extraction seams -----------------------------------------------------

fn extract_with_genome(
    s: &EGraph,
    genome: &luminal::extractor::Genome,
) -> luminal::layout_ir::ExtractedGraph {
    luminal::extractor::extract_layout_ir_with_genome_and_matchers(
        s,
        genome,
        test_runtime::matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary")
}

fn genome_flavored(s: &EGraph, preferences: &[&str]) -> luminal::extractor::Genome {
    // ROUND 11: routed through the shared viability-aware election core;
    // preference ORDER unchanged (explicit names > any CublasLt > the
    // transpose view > non-Copy > Copy) — see the round-2 battery's
    // genome comment for the rationale.
    let ordered =
        |candidates: &[(String, luminal::extractor::ProducerChoice)], level: usize| -> Vec<usize> {
            let admitted = test_runtime::level_admits(level);
            let mut order: Vec<usize> = Vec::new();
            let push_where = |order: &mut Vec<usize>, pred: &dyn Fn(&str) -> bool| {
                for (i, (name, _)) in candidates.iter().enumerate() {
                    if admitted(name) && pred(name) && !order.contains(&i) {
                        order.push(i);
                    }
                }
            };
            for want in preferences {
                push_where(&mut order, &|name| name == *want);
            }
            push_where(&mut order, &|name| {
                name.starts_with("LayoutTensorOpCublasLt")
            });
            push_where(&mut order, &|name| {
                name == "LayoutTensorOpIndexMapApplyViewGeneric"
            });
            push_where(&mut order, &|name| !name.contains("Copy"));
            push_where(&mut order, &|_| true);
            order
        };
    test_runtime::genome_with_ordering(s, &ordered)
}

/// A genome that ELECTS one specific op enode wherever it is a candidate —
/// the per-enode pairing probe.
fn genome_electing(s: &EGraph, enode: &NodeId) -> (luminal::extractor::Genome, bool) {
    // ROUND 11: the exact-enode election rides the shared viability-aware
    // core — the exact enode outranks everything (at every strictness
    // level), then the PIN names, then non-Copy, then Copy.
    let index = luminal::extractor::producer_index_with_matchers(s, test_runtime::matchers());
    let elected = index
        .values()
        .any(|candidates| candidates.iter().any(|(_, choice)| &choice.enode == enode));
    let ordered =
        |candidates: &[(String, luminal::extractor::ProducerChoice)], level: usize| -> Vec<usize> {
            let admitted = test_runtime::level_admits(level);
            let mut order: Vec<usize> = Vec::new();
            let push_where = |order: &mut Vec<usize>, pred: &dyn Fn(&str) -> bool| {
                for (i, (name, _)) in candidates.iter().enumerate() {
                    if admitted(name) && pred(name) && !order.contains(&i) {
                        order.push(i);
                    }
                }
            };
            for want in PIN {
                push_where(&mut order, &|name| name == *want);
            }
            push_where(&mut order, &|name| !name.contains("Copy"));
            push_where(&mut order, &|_| true);
            order
        };
    let mut genome = test_runtime::genome_with_ordering(s, &ordered);
    // The exact enode is elected UNCONDITIONALLY wherever it is a
    // candidate (the probe's point — round-10 semantics): viability-aware
    // selection steers every OTHER class, including the escalated Copy /
    // Materialize routes the exact enode's operands may need.
    for (class, candidates) in &index {
        if let Some((_, choice)) = candidates.iter().find(|(_, c)| &c.enode == enode) {
            genome.choices.insert(class.clone(), choice.clone());
        }
    }
    (genome, elected)
}

fn cublaslt_in_plan(graph: &luminal::layout_ir::ExtractedGraph) -> Vec<(CublasLt, Vec<String>)> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                let concrete = (*op.op).as_any().downcast_ref::<CublasLt>().cloned()?;
                let inputs = op.inputs.iter().map(|i| i.value.to_string()).collect();
                Some((concrete, inputs))
            }
            _ => None,
        })
        .collect()
}

fn pinned_cublaslt(text: &str) -> Vec<CublasLt> {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, PIN);
    cublaslt_in_plan(&graph)
        .into_iter()
        .map(|(op, _)| op)
        .collect()
}

fn spec_of(op: &CublasLt) -> &LtMatmulSpec {
    op.spec.as_ref().expect("spec parses")
}

/// A call frame reduced to comparable scalars — the shape RU4 diffs across
/// two programs (e-class ids move between runs; these do not).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SpecSummary {
    label: String,
    trans_a: bool,
    trans_b: bool,
    m: Option<i64>,
    n: Option<i64>,
    k: Option<i64>,
    lda: Option<i64>,
    ldb: Option<i64>,
    ldd: Option<i64>,
    epilogue: String,
}

fn summarize(op: &CublasLt) -> SpecSummary {
    let spec = spec_of(op);
    SpecSummary {
        label: format!("{:?}", op.form),
        trans_a: spec.trans_a,
        trans_b: spec.trans_b,
        m: spec.m.literal(),
        n: spec.n.literal(),
        k: spec.k.literal(),
        lda: spec.lda.literal(),
        ldb: spec.ldb.literal(),
        ldd: spec.ldd.literal(),
        epilogue: format!("{:?}", spec.epilogue),
    }
}

/// Elect EVERY cuBLASLt candidate in turn and union the kernels each
/// election yields — the reachable call-frame set of a program.
fn elect_all(s: &EGraph) -> BTreeSet<SpecSummary> {
    let ids: Vec<NodeId> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op.starts_with("LayoutTensorOpCublasLt"))
        .map(|(id, _)| id.clone())
        .collect();
    let mut out = BTreeSet::new();
    for id in &ids {
        let (genome, hit) = genome_electing(s, id);
        if !hit {
            continue;
        }
        let graph = extract_with_genome(s, &genome);
        for (op, _) in cublaslt_in_plan(&graph) {
            out.insert(summarize(&op));
        }
    }
    out
}

/// The cuBLASLt COL-order leading-dimension clamps, checked on literals:
/// lda >= rows(A), ldb >= rows(B), ldd >= m, ldc == ldd. A reading that
/// harvested a foreign ld would usually break one of these.
fn assert_call_sound(tag: &str, spec: &LtMatmulSpec) {
    let (m, n, k) = (spec.m.literal(), spec.n.literal(), spec.k.literal());
    let rows_a = if spec.trans_a { k } else { m };
    let rows_b = if spec.trans_b { n } else { k };
    for (name, ld, rows) in [
        ("lda", spec.lda.literal(), rows_a),
        ("ldb", spec.ldb.literal(), rows_b),
        ("ldd", spec.ldd.literal(), m),
    ] {
        if let (Some(ld), Some(rows)) = (ld, rows) {
            assert!(
                ld >= rows,
                "{tag}: cuBLASLt COL clamp violated — {name}={ld} < rows={rows} \
                 (m={m:?} n={n:?} k={k:?} trans_a={} trans_b={})",
                spec.trans_a,
                spec.trans_b
            );
        }
    }
    assert_eq!(
        spec.ldc.literal(),
        spec.ldd.literal(),
        "{tag}: C rides the D layout, so ldc must equal ldd"
    );
}

/// Run a closure that is EXPECTED to panic, without spraying the run with
/// a panic trace that reads like a failure. Returns Err on panic.
fn expect_may_panic<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(previous);
    // round-8b audit: carry the message so callers can tell SETUP
    // breakage (parse/vocabulary errors) from the SUBJECT's refusal.
    outcome.map_err(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    })
}

/// The elected spec's ld must equal the INDEPENDENT oracle's reading of the
/// same enode — literal for literal, symbolic for symbolic.
fn assert_ld(name: &str, tag: &str, got: &CuDim, want: &Dim) {
    match want {
        Dim::Lit(v) => assert!(
            got.literal() == Some(*v),
            "{tag}: {name} is {got}, THIS enode's reading says {v}"
        ),
        Dim::Sym(_) => assert!(
            matches!(got, CuDim::Symbolic(_)),
            "{tag}: {name} is the literal {got}, but THIS enode's reading is symbolic \
             — a sibling's number was borrowed"
        ),
        Dim::Unknown => panic!("{tag}: the oracle could not read {name} for this enode"),
    }
}

// ===========================================================================
// FIXTURE BUILDER — hand-written 2D matmul with arbitrary layout multiplicity
// ===========================================================================

#[derive(Clone)]
struct Fx {
    decls: String,
    m: String,
    n: String,
    k: String,
    a_rows: String,
    a_cols: String,
    b_rows: String,
    b_cols: String,
    bnk: bool,
    x_layouts: Vec<String>,
    w_layouts: Vec<String>,
    out_layouts: Vec<String>,
    tail: String,
    extra_outputs: Vec<String>,
    out_value: Option<String>,
}

impl Default for Fx {
    fn default() -> Self {
        Fx {
            decls: String::new(),
            m: "(IntLit 2)".into(),
            n: "(IntLit 3)".into(),
            k: "(IntLit 4)".into(),
            a_rows: "(IntLit 2)".into(),
            a_cols: "(IntLit 4)".into(),
            b_rows: "(IntLit 4)".into(),
            b_cols: "(IntLit 3)".into(),
            bnk: false,
            x_layouts: vec![rm("a_shape")],
            w_layouts: vec![rm("b_shape")],
            out_layouts: vec![rm("out_shape")],
            tail: String::new(),
            extra_outputs: Vec::new(),
            out_value: None,
        }
    }
}

fn rm(shape: &str) -> String {
    format!("(RightMajorContiguousElementLayoutLit {shape} (bits-of (F32)))")
}

fn lm(shape: &str) -> String {
    format!("(LeftMajorContiguousElementLayoutLit {shape} (bits-of (F32)))")
}

fn strided(shape: &str, row_entry: &str) -> String {
    format!(
        "(StridedElementLayoutLit {shape} (IntAffineExprCons {row_entry} \
         (IntAffineExprCons (CoordVar {shape} 0) (IntAffineExprNil))) (bits-of (F32)))"
    )
}

impl Fx {
    fn render(&self) -> String {
        let Fx {
            decls,
            m,
            n,
            k,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            bnk,
            x_layouts,
            w_layouts,
            out_layouts,
            tail,
            extra_outputs,
            out_value,
        } = self;
        let (w0, w1) = if *bnk { (1, 0) } else { (0, 1) };
        let mut text = format!(
            r#"{decls}
(let a_shape (ShapeLit (IntExprCons {a_rows} (IntExprCons {a_cols} (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons {b_rows} (IntExprCons {b_cols} (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprCons {k} (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons {m} (IntExprCons {n} (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape {w0})
      (IntExprCons (CoordVar prod_shape {w1}) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
"#
        );
        let mut buffer_id = 100;
        let mut emit = |text: &mut String, name: &str, lt_expr: &str, logical: &str, rw: bool| {
            text.push_str(&format!(
                "(let {name} (LayoutTensorLit {logical} {lt_expr}))\n"
            ));
            let access = if rw { "ReadWrite" } else { "ReadOnly" };
            text.push_str(&format!(
                "(let {name}_buf (BufferLit {buffer_id}))\n\
                 (set (buffer-access-of {name}_buf) ({access}))\n\
                 (set (buffer-freed-by {name}_buf) (CallerFrees))\n\
                 (let {name}_bt (BufferTensorLit {name} {name}_buf))\n"
            ));
            buffer_id += 1;
        };
        for (i, layout) in x_layouts.iter().enumerate() {
            emit(&mut text, &format!("x_lt{i}"), layout, "x_logical", false);
        }
        for (i, layout) in w_layouts.iter().enumerate() {
            emit(&mut text, &format!("w_lt{i}"), layout, "w_logical", false);
        }
        let out_value = out_value.clone().unwrap_or_else(|| "out_logical".into());
        for (i, layout) in out_layouts.iter().enumerate() {
            emit(&mut text, &format!("out_lt{i}"), layout, &out_value, true);
        }
        text.push_str(tail);
        text.push('\n');
        let mut outputs: Vec<String> = vec!["out_lt0_bt".into()];
        outputs.extend(extra_outputs.iter().cloned());
        let mut list = "(BufferTensorNil)".to_string();
        for bt in outputs.iter().rev() {
            list = format!("(BufferTensorCons {bt} {list})");
        }
        text.push_str(&format!("(let output (BufferOutputLit {list}))\n"));
        text.push_str(SCHEDULE);
        text.push('\n');
        text
    }
}

// ===========================================================================
// RC — READING-SET COHERENCE
// ===========================================================================

/// Two layouts for operand A (our w) x two layouts for operand B (our x) =
/// FOUR candidate reading sets. Every minted op enode must carry ITS OWN
/// readings' layout tensors in its Lit — never a sibling candidate's.
#[test]
fn rc1_four_way_reading_product_coherent() {
    let fx = Fx {
        x_layouts: vec![rm("a_shape"), lm("a_shape")],
        w_layouts: vec![rm("b_shape"), lm("b_shape")],
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let a = count_op(&s, "CublasLtOperandADescriptor");
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    let d = count_op(&s, "CublasLtOutputDDescriptor");
    let ops = count_cublaslt(&s);
    println!(
        "RC1 four-way: {} nodes, readings a={a} b={b} d={d}, {ops} op enode(s)",
        s.nodes.len()
    );
    dump_reading_sets("RC1", &s);
    assert_coherent("RC1", &s);
    // ROUND-10 RE-PIN: readings double — each matmul carries the original
    // site AND the transpose-sandwich sibling; the product law holds PER
    // SITE (4 sibling + 4 original-relayout candidates).
    assert_eq!(a, 4, "one A reading per operand layout per site");
    assert_eq!(b, 4, "one B reading per operand layout per site");
    assert_eq!(d, 2, "one D reading per site");
    assert_eq!(ops, 8, "|A| x |B| x |D| per site, summed over the pair");

    // Every candidate is a DISTINCT reading set (distinct Lits, distinct
    // classes) — the product is real, not four spellings of one class.
    let classes: BTreeSet<ClassId> = s
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .map(|n| n.eclass.clone())
        .collect();
    assert_eq!(
        classes.len(),
        8,
        "eight distinct op classes, one per reading set"
    );

    let ops = pinned_cublaslt(&fx);
    assert_eq!(ops.len(), 1, "election picks exactly one");
    let spec = spec_of(&ops[0]);
    println!(
        "  elected: trans_a={} trans_b={} lda={} ldb={} ldd={}",
        spec.trans_a, spec.trans_b, spec.lda, spec.ldb, spec.ldd
    );
    // w[4,3]: RM -> (N, lda=3), LM -> (T, lda=4). x[2,4]: RM -> (N, ldb=4),
    // LM -> (T, ldb=2). Only these four pairings are legal.
    assert!(
        (!spec.trans_a && spec.lda == 3) || (spec.trans_a && spec.lda == 4),
        "lda belongs to w's OWN layouts, got {} (trans_a={})",
        spec.lda,
        spec.trans_a
    );
    assert!(
        (!spec.trans_b && spec.ldb == 4) || (spec.trans_b && spec.ldb == 2),
        "ldb belongs to x's OWN layouts, got {} (trans_b={})",
        spec.ldb,
        spec.trans_b
    );
    assert_eq!(spec.mnk_lits(), (3, 2, 4));
    assert_call_sound("RC1", spec);

    // ---- NEGATIVE CONTROL 1: cross-pairing at the EGGLOG level ----------
    // Hand-seed an op term and union it with a Lit that names x_lt1 (the
    // left-major candidate's layout tensor) while its B descriptor reads
    // x_lt0. If this program even LOADS, the checker must catch it.
    let crossed_text = Fx {
        x_layouts: vec![rm("a_shape"), lm("a_shape")],
        w_layouts: vec![rm("b_shape"), lm("b_shape")],
        tail: r#"(let xsite (CublasLtLogicalMatmulSite x_logical w_logical out_logical))
(let xa (CublasLtOperandADescriptor xsite w_lt0
  (CublasLtOperationN)))
(let xb (CublasLtOperandBDescriptor xsite x_lt0
  (CublasLtOperationN)))
(let xd (CublasLtOutputDDescriptor xsite out_lt0))
(let xop (LayoutTensorOpCublasLt xsite xa xb xd (CublasLtEpilogueDefault)))
(union xop (LayoutTensorOpLit
  (LayoutTensorCons x_lt1 (LayoutTensorCons w_lt0 (LayoutTensorNil)))
  (LayoutTensorCons out_lt0 (LayoutTensorNil))))
"#
        .into(),
        ..Default::default()
    }
    .render();
    match expect_may_panic(|| test_runtime::serialize_fixture(&crossed_text)) {
        // SETUP-VS-SUBJECT GUARD (round-8b audit): "egglog refused" is a
        // finding about the CROSS-PAIRED LIT only if egglog got as far as
        // refusing the merge. A parse/vocabulary failure is setup
        // breakage and must be red, not reported as unrepresentability.
        Err(ref msg) if msg.contains("parse error") || msg.contains("Unbound") => {
            panic!("SETUP BROKE (not a finding): {msg}")
        }
        Err(_) => println!(
            "  NEGATIVE CONTROL 1: a cross-paired Lit is UNREPRESENTABLE — egglog \
             refuses the merge (the :no-merge input-layout-tensor-list-of metadata \
             function is a second, lower perimeter under the checker)"
        ),
        Ok(crossed) => {
            let (violations, _) = check_reading_set_coherence(&crossed);
            println!("  NEGATIVE CONTROL 1: {} violation(s)", violations.len());
            for v in &violations {
                println!("    {v}");
            }
            assert!(
                violations
                    .iter()
                    .any(|v| v.contains("CROSS-PAIRED") || v.contains("ORPHAN")),
                "a cross-paired Lit loaded and the checker said nothing: {violations:?}"
            );
        }
    }

    // ---- NEGATIVE CONTROL 2: a checker that cannot fail proves nothing --
    // Since egglog will not build a cross-paired e-graph, build one by
    // hand: take the clean four-way serialization and re-point one op
    // enode's B-descriptor child at the OTHER candidate's reading. Its own
    // descriptors now disagree with its class's Lit — exactly the bug the
    // RC family claims cannot happen. The checker MUST notice.
    let mut mutated = s.clone();
    let (op_id, b_slot_class) = mutated
        .nodes
        .iter()
        .find(|(_, n)| n.op == "LayoutTensorOpCublasLt")
        .map(|(id, n)| (id.clone(), class_of_child(&mutated, n, 2).unwrap()))
        .expect("an op enode to mutate");
    let other_b = mutated
        .nodes
        .iter()
        .find(|(_, n)| n.op == "CublasLtOperandBDescriptor" && n.eclass != b_slot_class)
        .map(|(id, _)| id.clone())
        .expect("a second B reading to cross-pair with");
    mutated.nodes.get_mut(&op_id).unwrap().children[2] = other_b;
    let (violations, checked) = check_reading_set_coherence(&mutated);
    println!(
        "  NEGATIVE CONTROL 2: checker walked {checked} enode(s) of the mutated \
         e-graph, {} violation(s)",
        violations.len()
    );
    for v in &violations {
        println!("    {v}");
    }
    assert!(
        violations.iter().any(|v| v.contains("CROSS-PAIRED")),
        "the coherence checker did NOT notice a hand-built cross-pairing — the \
         checker is vacuous and every RC pass above is worthless: {violations:?}"
    );
}

/// The same four-way product with a C-fold on top: the Accumulate contract
/// must re-mint a coherent D and keep C at Lit slot 2, per candidate.
#[test]
fn rc2_four_way_product_under_c_fold() {
    let tail = r#"(let c_logical (LogicalTensorInputLit (LogicalIdLit "c") out_shape (F32)))
(let acc_logical (LogicalAdd out_logical c_logical))
(let c_lt (LayoutTensorLit c_logical (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32)))))
(let acc_lt (LayoutTensorLit acc_logical (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32)))))
(let c_buf (BufferLit 200))
(set (buffer-access-of c_buf) (ReadOnly))
(set (buffer-freed-by c_buf) (CallerFrees))
(let c_bt (BufferTensorLit c_lt c_buf))
(let acc_buf (BufferLit 201))
(set (buffer-access-of acc_buf) (ReadWrite))
(set (buffer-freed-by acc_buf) (CallerFrees))
(let acc_bt (BufferTensorLit acc_lt acc_buf))
"#;
    let fx = Fx {
        x_layouts: vec![rm("a_shape"), lm("a_shape")],
        w_layouts: vec![rm("b_shape"), lm("b_shape")],
        tail: tail.into(),
        extra_outputs: vec!["acc_bt".into()],
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let base = count_op(&s, "LayoutTensorOpCublasLt");
    let acc = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!(
        "RC2 c-fold four-way: {} nodes, {base} base + {acc} accumulate enode(s)",
        s.nodes.len()
    );
    dump_reading_sets("RC2", &s);
    assert_coherent("RC2", &s);
    // ROUND-10 RE-PIN: 4 sibling + 4 original-relayout base candidates;
    // the C-fold decorator's transpose bridge fires on the SIBLING ops
    // only (the original-relayout ops claim a non-view D whose bridge
    // premise has no transpose parent), so 4 accumulates.
    assert_eq!(
        base, 8,
        "four sibling + four original-relayout base candidates"
    );
    assert_eq!(acc, 4, "one C-folded candidate per sibling base candidate");

    let genome = genome_flavored(&s, &["LayoutTensorOpCublasLtAccumulate"]);
    let ops: Vec<CublasLt> = cublaslt_in_plan(&extract_with_genome(&s, &genome))
        .into_iter()
        .map(|(op, _)| op)
        .collect();
    let accs: Vec<_> = ops
        .iter()
        .filter(|o| o.form == test_runtime::cublaslt_marker::CublasLtForm::Accumulate)
        .collect();
    assert_eq!(accs.len(), 1, "one Accumulate elected");
    let spec = spec_of(accs[0]);
    println!("  elected accumulate: ldc={} ldd={}", spec.ldc, spec.ldd);
    assert!(spec.has_c && spec.c_tensor.is_some());
    assert_eq!(spec.ldc, spec.ldd, "C rides the D layout by rule guard");
    assert_call_sound("RC2", spec);
}

/// The D-LEAK perimeter: a SECOND output layout (the creator-padded strided
/// one) coexists with the contiguous one. Both D readings mint; every base
/// candidate must still claim the SITE'S OWN out, and each op's Lit output
/// must be its own descD's layout tensor.
#[test]
fn rc3_second_output_layout_stays_coherent() {
    let fx = Fx {
        out_layouts: vec![rm("out_shape"), strided("out_shape", "(IntMul (CoordVar out_shape 1) (IntLit 8))")],
        // ROUND 10: the creator authors its strided-lists provenance row
        // (the pitched D is read through the ladder + the sibling view).
        tail: "(set (injectivity-of out_lt1) (Injective))\n(strided-lists (StridedElementLayoutLit out_shape (IntAffineExprCons (IntMul (CoordVar out_shape 1) (IntLit 8)) (IntAffineExprCons (CoordVar out_shape 0) (IntAffineExprNil))) (bits-of (F32))) out_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil))) (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n".into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let d = count_op(&s, "CublasLtOutputDDescriptor");
    let ops = count_cublaslt(&s);
    println!(
        "RC3 second-out-layout: {} nodes, d={d} readings, {ops} op enode(s)",
        s.nodes.len()
    );
    dump_reading_sets("RC3", &s);
    assert_coherent("RC3", &s);
    // ROUND-10 RE-PIN (was 2/2): the SIBLING site reads out through both
    // composed views (contiguous ld=3, pitched ld=8) and the original site
    // through the blanket-transposed route (ld=2) — 3 D readings, one
    // candidate per (per-site A x B x D) product.
    assert_eq!(
        d, 3,
        "contiguous, creator-padded and relayout D readings coexist"
    );
    // ROUND-11 RE-PIN: every site operand carries TWO readable layout
    // tensors (storage frame + collapse-derived column-form frame; the
    // r8d probe pins the mechanism), so operand readings double and the
    // per-site candidate products scale accordingly. Bounded and sound
    // (the coherence walker + clamp sweeps below run over every one).
    assert_eq!(ops, 12, "the per-site products: 2x2x1 + 2x2x2");

    // Every D reading's layout tensor names ITS OWN site's logical out.
    for site in sites(&s) {
        let site_node = *nodes_in_class(&s, &site, "CublasLtLogicalMatmulSite")
            .first()
            .expect("site node");
        let site_out = class_of_child(&s, site_node, 2).expect("site out");
        for r in readings_for_site(&s, &site).get("D").into_iter().flatten() {
            assert_eq!(
                r.logical,
                site_out,
                "a D reading names logical {} instead of its site's out {}",
                short(&r.logical),
                short(&site_out)
            );
        }
    }
    // Collect D lds across BOTH sites of the pair: the sibling carries
    // the contiguous ld (3) and the bucket pitch (8); the original's
    // relayout D carries its frame's rows (2). Nothing else — no harvest.
    let mut lds: BTreeSet<Option<i64>> = BTreeSet::new();
    for site in sites(&s) {
        for r in readings_for_site(&s, &site).get("D").into_iter().flatten() {
            lds.insert(r.ld.lit());
        }
    }
    println!("  D candidate lds: {lds:?}");
    assert_eq!(
        lds,
        BTreeSet::from([Some(2), Some(3), Some(8)]),
        "the contiguous ld, the bucket pitch and the relayout rows — nothing harvested"
    );
}

/// A bias-then-relu decorator CHAIN. Every op enode along the chain must
/// pair its OWN descD with its OWN Lit output — a stale D reading from an
/// earlier link must never pair with a later output.
#[test]
fn rc4_bias_relu_chain_no_stale_d() {
    let text = {
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
    let s = test_runtime::serialize_fixture(&text);
    let mut per_ctor = Vec::new();
    for (ctor, ..) in OP_FORMS {
        per_ctor.push(format!(
            "{}={}",
            ctor.trim_start_matches("LayoutTensorOp"),
            count_op(&s, ctor)
        ));
    }
    println!(
        "RC4 bias->relu chain: {} nodes, {}",
        s.nodes.len(),
        per_ctor.join(" ")
    );
    dump_reading_sets("RC4", &s);
    let checked = assert_coherent("RC4", &s);
    assert!(
        checked >= 3,
        "base, bias and relu links all present ({checked} enodes)"
    );

    // Every DISTINCT claimed output across the chain is a DISTINCT logical
    // value (base y, bias y+b, relu(y+b)) — no two links share a D.
    let d_logicals: BTreeSet<ClassId> = all_readings(&s)
        .into_iter()
        .filter(|r| r.role == "D")
        .map(|r| r.logical)
        .collect();
    println!("  distinct claimed outputs: {}", d_logicals.len());
    assert!(
        d_logicals.len() >= 3,
        "each decoration link re-mints its own D"
    );

    let ops = pinned_cublaslt(&text);
    assert_eq!(ops.len(), 1, "one fused kernel elected");
    let spec = spec_of(&ops[0]);
    println!(
        "  elected: form={:?} epilogue={:?} out={} site_out={}",
        ops[0].form,
        spec.epilogue,
        short(&spec.logical_out),
        short(&spec.logical_site_out)
    );
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert_ne!(
        spec.logical_out, spec.logical_site_out,
        "a fully decorated op claims the MOVED output, not the site's raw out"
    );
}

// ===========================================================================
// RE — PER-ENODE SPEC PAIRING
// ===========================================================================

/// Elect every cuBLASLt op enode of a class in turn; the parsed spec must
/// match THAT enode's own descriptors (operation + lds), every time.
fn per_enode_election_sweep(tag: &str, fx: &str, expect_enodes: usize) {
    let s = test_runtime::serialize_fixture(fx);
    dump_reading_sets(tag, &s);
    assert_coherent(tag, &s);

    let op_nodes: Vec<(NodeId, Node)> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op.starts_with("LayoutTensorOpCublasLt"))
        .map(|(id, n)| (id.clone(), n.clone()))
        .collect();
    println!("  [{tag}] {} cuBLASLt op enode(s)", op_nodes.len());
    assert_eq!(op_nodes.len(), expect_enodes, "{tag}: op enode count");

    let mut elected_any = 0usize;
    for (id, node) in &op_nodes {
        // What does THIS enode's own reading say?
        let a_class = class_of_child(&s, node, 1).expect("descA slot");
        let b_class = class_of_child(&s, node, 2).expect("descB slot");
        let a_desc = *nodes_in_class(&s, &a_class, "CublasLtOperandADescriptor")
            .first()
            .expect("A descriptor");
        let b_desc = *nodes_in_class(&s, &b_class, "CublasLtOperandBDescriptor")
            .first()
            .expect("B descriptor");
        let a_op = operation_name(&s, &class_of_child(&s, a_desc, 2).unwrap());
        let b_op = operation_name(&s, &class_of_child(&s, b_desc, 2).unwrap());
        let want_trans_a = a_op == "CublasLtOperationT";
        let want_trans_b = b_op == "CublasLtOperationT";
        let a_lt = class_of_child(&s, a_desc, 1).unwrap();
        let b_lt = class_of_child(&s, b_desc, 1).unwrap();
        let a_site = class_of_child(&s, a_desc, 0).unwrap();
        let b_site = class_of_child(&s, b_desc, 0).unwrap();
        let a_storage = storage_dims(&s, &logical_of_lt(&s, &a_lt).unwrap()).unwrap();
        let b_storage = storage_dims(&s, &logical_of_lt(&s, &b_lt).unwrap()).unwrap();
        let want_lda = reading_ld(&s, "A", &a_op, &a_site, &a_lt, &a_storage);
        let want_ldb = reading_ld(&s, "B", &b_op, &b_site, &b_lt, &b_storage);

        let (genome, hit) = genome_electing(&s, id);
        if !hit {
            println!(
                "  [{tag}] enode {} is not a candidate for any demanded class (skipped)",
                &id.to_string()[..8.min(id.to_string().len())]
            );
            continue;
        }
        let graph = extract_with_genome(&s, &genome);
        let plan = cublaslt_in_plan(&graph);
        let Some((op, _)) = plan.iter().find(|(op, _)| {
            op.spec.as_ref().is_some_and(|sp| {
                sp.desc_a_layout_tensor == a_lt && sp.desc_b_layout_tensor == b_lt
            })
        }) else {
            // ROUND 10: an ORIGINAL-site relayout candidate claims a
            // layout tensor the boundary only reaches through a copy the
            // deterministic harness does not elect; forcing such an enode
            // leaves it unreachable and the plan carries the sibling
            // kernel instead. The per-enode PAIRING property is what this
            // sweep guards, and it is checked on every enode that DOES
            // reach a plan; unreachable ones are counted and reported.
            println!(
                "  [{tag}] enode {} elected but unreachable from the boundary (relayout candidate)",
                &id.to_string()[..8.min(id.to_string().len())]
            );
            continue;
        };
        let spec = spec_of(op);
        elected_any += 1;
        println!(
            "  [{tag}] elected {}: trans_a={} (want {want_trans_a}) trans_b={} (want {want_trans_b}) \
             lda={} (want {want_lda}) ldb={} (want {want_ldb})",
            node.op.trim_start_matches("LayoutTensorOp"),
            spec.trans_a,
            spec.trans_b,
            spec.lda,
            spec.ldb
        );
        assert_eq!(
            spec.trans_a, want_trans_a,
            "{tag}: trans_a is THIS enode's reading"
        );
        assert_eq!(
            spec.trans_b, want_trans_b,
            "{tag}: trans_b is THIS enode's reading"
        );
        assert_ld("lda", tag, &spec.lda, &want_lda);
        assert_ld("ldb", tag, &spec.ldb, &want_ldb);
    }
    assert!(
        elected_any >= 2,
        "{tag}: at least two enodes were forced in turn"
    );
}

/// The dual-spelling class: one square matmul spelled A[m,k],B[k,n] and A[m,k],B[n,k], the two
/// logical outs unioned. Two op spellings live in ONE e-class over ONE Lit.
#[test]
fn re1_dual_spelling_per_enode_election() {
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_kn_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let w_nk_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 1)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_kn_applied (LogicalIndexMapApply w_logical w_kn_map prod_shape))
(let w_nk_applied (LogicalIndexMapApply w_logical w_nk_map prod_shape))
(let out_amk_bkn (LogicalReduceSum (LogicalMul x_applied w_kn_applied) 0))
(let out_amk_bnk (LogicalReduceSum (LogicalMul x_applied w_nk_applied) 0))
(union out_amk_bkn out_amk_bnk)
(let x_lt (LayoutTensorLit x_logical {}))
(let w_lt (LayoutTensorLit w_logical {}))
(let out_lt (LayoutTensorLit out_amk_bkn {}))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_bt (BufferTensorLit x_lt x_buffer_id))
(let w_bt (BufferTensorLit w_lt w_buffer_id))
(let out_bt (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_bt (BufferTensorNil))))
{SCHEDULE}
"#,
        rm("a_shape"),
        rm("b_shape"),
        rm("out_shape")
    );
    // ROUND-10 RE-PIN (was 2): + the sibling site's two spellings (the
    // dual-spelled outs weld the two siblings into one site, which reads
    // its A operand both ways).
    // ROUND-11 RE-PIN (was 4): the dual spelling now canonicalizes into
    // two canonical chains (stored w / viewed w) with their siblings — 4
    // sites — and each operand reads in two frames; 20 candidates.
    per_enode_election_sweep("RE1", &fx, 20);
}

/// The weld corner: A [1,1], B [1,3], product (m n k) = (1, 3, 1). Both the
/// B[k,n] and B[n,k] site rules fire on the same bytes; per-enode election must
/// still parse each spelling's own reading.
#[test]
fn re2_weld_corner_per_enode_election() {
    let fx = Fx {
        m: "(IntLit 1)".into(),
        n: "(IntLit 3)".into(),
        k: "(IntLit 1)".into(),
        a_rows: "(IntLit 1)".into(),
        a_cols: "(IntLit 1)".into(),
        b_rows: "(IntLit 1)".into(),
        b_cols: "(IntLit 3)".into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let enodes = count_cublaslt(&s);
    println!("RE2 weld corner: {enodes} op enode(s)");
    per_enode_election_sweep("RE2", &fx, enodes);
}

// ===========================================================================
// RP — CANDIDATE PRODUCT AND SCALING
// ===========================================================================

/// The product law: candidates per site = |A readings| x |B readings| x
/// |D readings|, measured over 1x1x1, 2x1x1, 2x2x1 and 2x2x2 seedings.
#[test]
fn rp1_candidate_product_growth() {
    let cases: Vec<(&str, Fx)> = vec![
        ("1x1x1", Fx::default()),
        (
            "2x1x1",
            Fx {
                w_layouts: vec![rm("b_shape"), lm("b_shape")],
                ..Default::default()
            },
        ),
        (
            "2x2x1",
            Fx {
                w_layouts: vec![rm("b_shape"), lm("b_shape")],
                x_layouts: vec![rm("a_shape"), lm("a_shape")],
                ..Default::default()
            },
        ),
        (
            "2x2x2",
            Fx {
                w_layouts: vec![rm("b_shape"), lm("b_shape")],
                x_layouts: vec![rm("a_shape"), lm("a_shape")],
                out_layouts: vec![rm("out_shape"), strided("out_shape", "(IntMul (CoordVar out_shape 1) (IntLit 8))")],
        // ROUND 10: the creator authors its strided-lists provenance row
        // (pitched layouts are read through the ladder now).
        tail: "(set (injectivity-of out_lt1) (Injective))\n(strided-lists (StridedElementLayoutLit out_shape (IntAffineExprCons (IntMul (CoordVar out_shape 1) (IntLit 8)) (IntAffineExprCons (CoordVar out_shape 0) (IntAffineExprNil))) (bits-of (F32))) out_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil))) (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n".into(),
                ..Default::default()
            },
        ),
    ];
    println!("MEASURE rp1 candidate product");
    for (name, fx) in cases {
        let text = fx.render();
        let started = Instant::now();
        let s = test_runtime::serialize_fixture(&text);
        let saturate = started.elapsed();
        let a = count_op(&s, "CublasLtOperandADescriptor");
        let b = count_op(&s, "CublasLtOperandBDescriptor");
        let d = count_op(&s, "CublasLtOutputDDescriptor");
        let ops = count_cublaslt(&s);
        let started = Instant::now();
        let elected = pinned_cublaslt(&text);
        let extract = started.elapsed();
        println!(
            "  {name}: {} nodes, a={a} b={b} d={d} -> {ops} candidates (product {}), \
             saturate {saturate:.2?}, extract {extract:.2?}, elected {}",
            s.nodes.len(),
            a * b * d,
            elected.len()
        );
        assert_coherent(&format!("RP1/{name}"), &s);
        // ROUND-10 RE-PIN: the product law holds PER SITE (an extra
        // operand layout raises the A count on one site of the pair and
        // the B count on the other — the roles swap across the sandwich).
        let mut per_site_products = 0usize;
        let mut per_site: Vec<(usize, usize, usize)> = Vec::new();
        for site in sites(&s) {
            let per_role = readings_for_site(&s, &site);
            let count = |role: &str| per_role.get(role).map(|v| v.len()).unwrap_or(0);
            let triple = (count("A"), count("B"), count("D"));
            per_site_products += triple.0 * triple.1 * triple.2;
            per_site.push(triple);
        }
        per_site.sort_unstable();
        assert_eq!(
            ops, per_site_products,
            "{name}: candidates are exactly the per-site product sum"
        );
        assert_eq!(elected.len(), 1, "{name}: one kernel elected");
        // The extra layout doubles ONE role on EACH site (A on one, B on
        // the other); the pitched-out case doubles D on the sibling only.
        // ROUND-11 RE-PIN: the base readings double via the column-form
        // frame — and the previously "extra" SEEDED left-major layout now
        // HASH-CONSES with that collapse-derived column-form layout (the
        // LM contiguous class of the same shape), so seeding it no longer
        // adds a reading: every case is the uniform two-frames-per-role
        // product, plus the pitched-D doubling in the last case.
        let expect: Vec<(usize, usize, usize)> = match name {
            "1x1x1" => vec![(2, 2, 1), (2, 2, 1)],
            "2x1x1" => vec![(2, 2, 1), (2, 2, 1)],
            "2x2x1" => vec![(2, 2, 1), (2, 2, 1)],
            "2x2x2" => vec![(2, 2, 1), (2, 2, 2)],
            _ => unreachable!(),
        };
        let mut expect = expect;
        expect.sort_unstable();
        assert_eq!(
            per_site, expect,
            "{name}: per-site reading counts are arms x forms"
        );
    }
}

/// Decoration DEPTH: how many op enodes each decorator link adds. Growth
/// must be additive per link, never multiplicative in program size.
#[test]
fn rp2_decoration_depth_enode_count() {
    let programs: [(&str, Box<dyn Fn(&mut Graph)>); 4] = [
        (
            "plain",
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }),
        ),
        (
            "relu",
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).relu().output();
            }),
        ),
        (
            "bias",
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
            }),
        ),
        (
            "bias+relu",
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).relu().output();
            }),
        ),
    ];
    println!("MEASURE rp2 decoration depth");
    for (name, build) in &programs {
        let text = {
            let mut cx = Graph::new();
            build(&mut cx);
            cx.logical
                .bound_program(&test_runtime::TestRuntimeBindings)
                .expect("recorder clean")
                .text
        };
        let started = Instant::now();
        let s = test_runtime::serialize_fixture(&text);
        let saturate = started.elapsed();
        let per_ctor: Vec<String> = OP_FORMS
            .iter()
            .map(|(ctor, ..)| {
                format!(
                    "{}={}",
                    ctor.trim_start_matches("LayoutTensorOpCublasLt"),
                    count_op(&s, ctor)
                )
            })
            .collect();
        let readings = count_op(&s, "CublasLtOperandADescriptor")
            + count_op(&s, "CublasLtOperandBDescriptor")
            + count_op(&s, "CublasLtOutputDDescriptor");
        println!(
            "  {name}: {} nodes, {readings} readings, {} op enodes [{}], saturate {saturate:.2?}",
            s.nodes.len(),
            count_cublaslt(&s),
            per_ctor.join(" ")
        );
        assert_coherent(&format!("RP2/{name}"), &s);
        // ROUND-11 RE-PIN (was 2): two frame readings per site operand;
        // decoration still multiplies NEITHER.
        assert_eq!(
            count_op(&s, "CublasLtOperandADescriptor"),
            4,
            "{name}: decoration never multiplies OPERAND readings"
        );
        assert_eq!(count_op(&s, "CublasLtOperandBDescriptor"), 4);
    }
}

/// PRODUCT AT SCALE: N independent blocks, distinct and same geometry.
/// Per-block candidate counts must be CONSTANT in N — the program-size
/// independence law (the RU4 hazard restated at the site level).
#[test]
fn rp3_product_at_scale() {
    println!("MEASURE rp3 product at scale");
    let mut rows: Vec<(String, usize, usize, usize, usize, f64, f64)> = Vec::new();
    for (label, same) in [("distinct", false), ("same-geometry", true)] {
        for n in [1usize, 2, 4, 8, 16] {
            let text = {
                let mut cx = Graph::new();
                for i in 0..n {
                    let (m, k, nn) = if same { (4, 8, 3) } else { (4 + i, 8, 2 + i) };
                    let x = cx.tensor((m, k), DType::F32);
                    let w = cx.tensor((k, nn), DType::F32);
                    let _ = x.matmul(w).output();
                }
                cx.logical
                    .bound_program(&test_runtime::TestRuntimeBindings)
                    .expect("recorder clean")
                    .text
            };
            let started = Instant::now();
            let s = test_runtime::serialize_fixture(&text);
            let saturate = started.elapsed().as_secs_f64();
            let site_count = sites(&s).len();
            let ops = count_cublaslt(&s);
            let readings = count_op(&s, "CublasLtOperandADescriptor")
                + count_op(&s, "CublasLtOperandBDescriptor")
                + count_op(&s, "CublasLtOutputDDescriptor");
            let started = Instant::now();
            let elected = pinned_cublaslt(&text);
            let extract = started.elapsed().as_secs_f64();
            println!(
                "  {label} N={n}: {} nodes, {site_count} sites, {readings} readings, \
                 {ops} candidates ({:.1}/site), saturate {saturate:.2}s, extract {extract:.2}s, \
                 {} elected",
                s.nodes.len(),
                ops as f64 / site_count.max(1) as f64,
                elected.len()
            );
            // ROUND-11 RE-PIN (was 2n ops / 6n readings): two frames per
            // operand make 4 candidates and 5 readings per site pair —
            // still CONSTANT per block, which is this test's law.
            assert_eq!(site_count, 2 * n, "{label} N={n}: one site pair per block");
            assert_eq!(
                ops,
                8 * n,
                "{label} N={n}: four candidates per site pair, independent of N"
            );
            assert_eq!(
                readings,
                10 * n,
                "{label} N={n}: five readings per site pair"
            );
            assert_eq!(elected.len(), n, "{label} N={n}: one kernel per block");
            rows.push((
                label.to_string(),
                n,
                s.nodes.len(),
                ops,
                readings,
                saturate,
                extract,
            ));
        }
    }
    // Program-size independence, stated as a ratio: candidates/site is 1 at
    // every N in both families.
    for (label, n, nodes, ops, _, _, _) in &rows {
        // ROUND-11 RE-PIN: four candidates per site pair, two sites per
        // block — still linear in N (the law under test).
        assert_eq!(
            *ops,
            8 * *n,
            "{label} N={n} ({nodes} nodes): candidate count tracks sites only"
        );
    }
}

// ===========================================================================
// RS — SYMBOLIC EDGES
// ===========================================================================

fn symbolic_k_skeleton(
    tail: &str,
    extra_outputs: Vec<String>,
    out_value: Option<String>,
) -> String {
    Fx {
        decls: "(let s_var (IntVar \"s\"))\n(set (lower-bound-of s_var) (bigint 2))\n(set (upper-bound-of s_var) (bigint 8))".into(),
        m: "(IntLit 2)".into(),
        n: "(IntLit 3)".into(),
        k: "s_var".into(),
        a_rows: "(IntLit 2)".into(),
        a_cols: "s_var".into(),
        b_rows: "s_var".into(),
        b_cols: "(IntLit 3)".into(),
        tail: tail.into(),
        extra_outputs,
        out_value,
        ..Default::default()
    }
    .render()
}

/// BIAS decoration of a symbolic-k op (round4's t6b covers relu; this is
/// the contract-CHANGING decorator, which must re-mint a coherent D and
/// append the bias at the Lit tail while k stays symbolic).
#[test]
fn rs1_bias_decorates_symbolic_k() {
    let tail = r#"(let bias_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let bias_logical (LogicalTensorInputLit (LogicalIdLit "bias") bias_shape (F32)))
(let bias_map (IndexMapLit (IntExprCons (CoordVar out_shape 0) (IntExprNil)) bias_shape))
(let bias_bcast (LogicalIndexMapApply bias_logical bias_map out_shape))
(let biased_logical (LogicalAdd out_logical bias_bcast))
(let bias_lt (LayoutTensorLit bias_logical (RightMajorContiguousElementLayoutLit bias_shape (bits-of (F32)))))
(let biased_lt (LayoutTensorLit biased_logical (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32)))))
(let bias_buf (BufferLit 200))
(set (buffer-access-of bias_buf) (ReadOnly))
(set (buffer-freed-by bias_buf) (CallerFrees))
(let bias_bt (BufferTensorLit bias_lt bias_buf))
(let biased_buf (BufferLit 201))
(set (buffer-access-of biased_buf) (ReadWrite))
(set (buffer-freed-by biased_buf) (CallerFrees))
(let biased_bt (BufferTensorLit biased_lt biased_buf))
"#;
    let fx = symbolic_k_skeleton(tail, vec!["biased_bt".into()], None);
    let s = test_runtime::serialize_fixture(&fx);
    let bias_ops = count_op(&s, "LayoutTensorOpCublasLtBias");
    println!(
        "RS1 bias on symbolic k: {} nodes, {bias_ops} Bias enode(s), {} total",
        s.nodes.len(),
        count_cublaslt(&s)
    );
    dump_reading_sets("RS1", &s);
    assert_coherent("RS1", &s);
    assert!(bias_ops >= 1, "the bias decorator fired with symbolic k");

    let ops = pinned_cublaslt(&fx);
    let bias = ops
        .iter()
        .find(|o| o.form == test_runtime::cublaslt_marker::CublasLtForm::Bias)
        .expect("a Bias kernel elected");
    let spec = spec_of(bias);
    println!(
        "  elected: ep={:?} m={} n={} k={} lda={} ldb={} ldd={}",
        spec.epilogue, spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(spec.epilogue, CuEpilogue::Bias);
    assert!(
        matches!(spec.k, CuDim::Symbolic(_)),
        "k stays symbolic under decoration"
    );
    assert!(spec.bias_tensor.is_some());
    assert_eq!(spec.lda, 3, "w storage cols literal");
    assert_eq!(spec.ldd, 3);
}

/// SYMBOLIC m routed through the A operand (call-m = logical n = symbolic),
/// with a BUCKET-PADDED output layout. The creator rewrite asserts
/// injectivity, assembly succeeds, and the elected spec must carry
/// Symbolic m with the LITERAL pitch as ldd.
#[test]
fn rs2_symbolic_m_probe() {
    // out_lt0 IS the creator-padded strided layout (spelled here exactly as
    // the rewrite mints it, so it hash-conses into the same class) and is
    // the program's demanded output; out_lt1 is the contiguous layout the
    // bucket request is issued against, which is what asserts injectivity.
    let fx = Fx {
        decls: "(let s_var (IntVar \"s\"))\n(set (lower-bound-of s_var) (bigint 2))\n(set (upper-bound-of s_var) (bigint 8))".into(),
        m: "(IntLit 2)".into(),
        n: "s_var".into(),
        k: "(IntLit 4)".into(),
        a_rows: "(IntLit 2)".into(),
        a_cols: "(IntLit 4)".into(),
        b_rows: "(IntLit 4)".into(),
        b_cols: "s_var".into(),
        out_layouts: vec![
            strided("out_shape", "(IntMul (CoordVar out_shape 1) (IntLit 8))"),
            rm("out_shape"),
        ],
        // round-8c: the creator rewrite is gone, so the fixture asserts
        // the padded layout's injectivity itself (out_lt0 is the strided
        // one here) — the estate's obligation, made explicit. ROUND 10:
        // the creator also authors its strided-lists provenance row.
        tail: "(set (injectivity-of out_lt0) (Injective))\n(strided-lists (StridedElementLayoutLit out_shape (IntAffineExprCons (IntMul (CoordVar out_shape 1) (IntLit 8)) (IntAffineExprCons (CoordVar out_shape 0) (IntAffineExprNil))) (bits-of (F32))) out_shape (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil))) (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n".into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let d = count_op(&s, "CublasLtOutputDDescriptor");
    println!(
        "RS2 symbolic m + padded D: {} nodes, d={d} readings, {} candidates",
        s.nodes.len(),
        count_cublaslt(&s)
    );
    dump_reading_sets("RS2", &s);
    assert_coherent("RS2", &s);
    // ROUND-10 RE-PIN (was 2): sibling composed views of both out layouts
    // plus the original site's relayout route.
    assert!(
        d >= 2,
        "contiguous AND creator-padded D readings coexist, got {d}"
    );

    let plan = {
        let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
        cublaslt_in_plan(&graph)
    };
    let spec = plan
        .iter()
        .map(|(op, _)| spec_of(op))
        .find(|sp| sp.ldd == 8)
        .unwrap_or_else(|| panic!("no elected kernel carries the literal pitch; plan {plan:?}"));
    println!(
        "  elected padded: m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert!(
        matches!(spec.m, CuDim::Symbolic(_)),
        "call m = logical n = symbolic"
    );
    assert_eq!(spec.n, 2);
    assert_eq!(spec.k, 4);
    assert_eq!(spec.ldd, 8, "the literal bucket pitch");
    assert_eq!(spec.ldc, spec.ldd);
}

/// SYMBOLIC LD edges. (a) a symbolic PITCH bucket request must never
/// rewrite (fail-closed creator). (b) a hand-certified symbolic-pitch
/// strided layout coexisting with the tensor's contiguous layout gives two
/// sibling candidates; electing the symbolic one must yield Symbolic(pitch)
/// — never the sibling's literal.
#[test]
fn rs3_symbolic_ld_refuses_even_with_parseable_classmate() {
    // (a) FLIPPED (round-8c): this used to route a symbolic pitch through
    // the CREATOR REWRITE and pin its fail-closed refusal. cuBLASLt no
    // longer owns a creator (or any relation) — padding belongs to the
    // bucketing estate — so the subject is now the weaker but still real
    // claim that NO padded layout appears from nowhere: a tensor with
    // only its contiguous layout yields exactly one B reading.
    let refuse = Fx {
        decls: "(let p_var (IntVar \"p\"))\n(set (lower-bound-of p_var) (bigint 4))\n(set (upper-bound-of p_var) (bigint 16))".into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&refuse);
    let strided_layouts = count_op(&s, "StridedElementLayoutLit");
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    println!("RS3a no-creator baseline: {strided_layouts} strided layout(s), b={b} readings");
    assert_coherent("RS3a", &s);
    // ROUND-11 RE-PIN (was 2): + the column-form frame readings. Still
    // no padded layout from nowhere — every reading rides a contiguous
    // frame.
    assert_eq!(
        b, 4,
        "no creator, no padded layout — only the contiguous readings"
    );

    // (b) hand-certified symbolic pitch alongside the literal sibling.
    let dual = Fx {
        decls: "(let p_var (IntVar \"p\"))\n(set (lower-bound-of p_var) (bigint 4))\n(set (upper-bound-of p_var) (bigint 16))".into(),
        x_layouts: vec![
            rm("a_shape"),
            strided("a_shape", "(IntMul (CoordVar a_shape 1) p_var)"),
        ],
        // ROUND 10: the creator authors the provenance row (symbolic
        // strides are IntExprs like any other).
        tail: "(set (injectivity-of x_lt1) (Injective))\n(strided-lists (StridedElementLayoutLit a_shape (IntAffineExprCons (IntMul (CoordVar a_shape 1) p_var) (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil))) (bits-of (F32))) a_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil))) (IntExprCons p_var (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n".into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&dual);
    dump_reading_sets("RS3b", &s);
    assert_coherent("RS3b", &s);
    // x is the site triple's slot-0 tensor (our `a`, the API's operand B).
    let site = sites(&s).into_iter().next().expect("one site");
    let site_node = *nodes_in_class(&s, &site, "CublasLtLogicalMatmulSite")
        .first()
        .unwrap();
    let x_logical = class_of_child(&s, site_node, 0).expect("site a");
    // ROUND-11 RE-PIN: under the unswapped arms + canonical form, x (the
    // site's slot-0 operand) is read by the A arm directly, and its
    // pitched bytes reach the B arm only THROUGH the sibling's transpose
    // view. Collect x's readings across BOTH roles (the helper's B-only
    // view would now see zero on x itself).
    let mut forms: Vec<(String, String, Dim)> = all_readings(&s)
        .into_iter()
        .filter(|r| r.logical == x_logical)
        .map(|r| {
            (
                r.role.clone(),
                r.operation
                    .trim_start_matches("CublasLtOperation")
                    .to_string(),
                r.ld,
            )
        })
        .collect();
    forms.sort();
    println!("RS3b reading forms over x: {forms:?}");
    assert_eq!(
        forms.len(),
        3,
        "the two contiguous frame readings AND the symbolic-pitch one"
    );
    assert!(
        forms.iter().any(|(_, _, ld)| matches!(ld, Dim::Sym(_))),
        "one reading carries a SYMBOLIC ld"
    );

    let candidates = strided_b_candidates(&s);
    println!(
        "RS3b strided candidates: {:?}",
        candidates
            .iter()
            .map(|(_, d)| d.to_string())
            .collect::<Vec<_>>()
    );
    let (id, factor) = candidates
        .into_iter()
        .find(|(_, d)| matches!(d, Dim::Sym(_)))
        .expect("RS3b: the hand-certified symbolic-pitch reading minted a candidate");
    println!("RS3b electing the symbolic-pitch candidate (factor {factor})");
    let (genome, hit) = genome_electing(&s, &id);
    assert!(hit, "RS3b: the symbolic-pitch candidate is electable");
    let graph = extract_with_genome(&s, &genome);
    let plan = cublaslt_in_plan(&graph);
    let specs: Vec<String> = plan
        .iter()
        .map(|(op, _)| format!("{}", spec_of(op).ldb))
        .collect();
    println!("RS3b elected ldb values: {specs:?}");
    assert!(
        plan.iter()
            .any(|(op, _)| matches!(spec_of(op).ldb, CuDim::Symbolic(_))),
        "electing the symbolic-pitch reading must carry a SYMBOLIC ldb — \
         never the literal sibling's number ({specs:?})"
    );

    // (c) THE TRUE CLASS-MATE: a hand-seeded op enode unioned into the REAL
    // op's e-class (same Lit, same class) whose A reading claims
    // LeftMajorContiguous over a layout class that has no left-major
    // spelling — a term the rules could never mint. Electing it must NOT
    // silently borrow the sibling's numbers; the marker's form/layout
    // perimeter must refuse it loudly.
    let bogus_tail = r#"(let bogus_site (CublasLtLogicalMatmulSite x_logical w_logical out_logical))
(let bogus_desc_a (CublasLtOperandADescriptor bogus_site w_lt0
  (CublasLtOperationT)))
(let real_desc_b (CublasLtOperandBDescriptor bogus_site x_lt0
  (CublasLtOperationN)))
(let real_desc_d (CublasLtOutputDDescriptor bogus_site out_lt0))
(let bogus_op (LayoutTensorOpCublasLt bogus_site bogus_desc_a real_desc_b real_desc_d
  (CublasLtEpilogueDefault)))
(union bogus_op (LayoutTensorOpLit
  (LayoutTensorCons w_lt0 (LayoutTensorCons x_lt0 (LayoutTensorNil)))
  (LayoutTensorCons out_lt0 (LayoutTensorNil))))
"#;
    let bogus = Fx {
        tail: bogus_tail.into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&bogus);
    dump_reading_sets("RS3c", &s);
    // ROUND 10: the seeded Lit order matches the unswapped assemble rule
    // ([a, b] = [descA's, descB's]); the seeded descriptors ALSO
    // cross-pollinate with the site's legit readings through assembly, so
    // the base-op population is larger than the pair. What matters here
    // is only that the seeded term exists with its Lit.
    let class_mates: Vec<(NodeId, ClassId)> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op == "LayoutTensorOpCublasLt")
        .map(|(id, n)| (id.clone(), n.eclass.clone()))
        .collect();
    println!(
        "RS3c: {} base op enode(s) in {} class(es)",
        class_mates.len(),
        class_mates
            .iter()
            .map(|(_, c)| c)
            .collect::<BTreeSet<_>>()
            .len()
    );
    assert!(
        class_mates.len() >= 2,
        "the real ops and the hand-seeded class-mate"
    );
    let bogus_id = s
        .nodes
        .iter()
        .find(|(_, n)| {
            n.op == "LayoutTensorOpCublasLt"
                && class_of_child(&s, n, 1)
                    .and_then(|c| {
                        nodes_in_class(&s, &c, "CublasLtOperandADescriptor")
                            .first()
                            .copied()
                    })
                    .and_then(|d| class_of_child(&s, d, 2))
                    .map(|f| class_has(&s, &f, "CublasLtOperationT"))
                    .unwrap_or(false)
        })
        .map(|(id, _)| id.clone())
        .expect("the hand-seeded class-mate is present");
    let (genome, hit) = genome_electing(&s, &bogus_id);
    assert!(hit, "RS3c: the class-mate is electable");
    let outcome = expect_may_panic(|| {
        let graph = extract_with_genome(&s, &genome);
        cublaslt_in_plan(&graph)
            .into_iter()
            .map(|(op, _)| format!("{:?}", op.spec.as_ref().map(|sp| sp.lda.to_string())))
            .collect::<Vec<_>>()
    });
    // FLIPPED (round-8b E1) — and this flip is a RECORDED COST, not a
    // clean-up. The seeded class-mate used to carry a layout FORM that
    // contradicted its layout term, and the extractor's form/layout
    // coherence check refused it. E1 deleted the form child (orientation
    // now rides the OPERATION, which the rules prove from the index
    // map), so the seed degenerates to (site, lt, operation) — a
    // structurally legitimate reading whose operation merely happens to
    // be one the rules would not have concluded for this map+layout.
    // Extraction cannot detect that: BOTH operations are dimensionally
    // plausible on a transposable shape (the view/storage transposition
    // cross-check passes either way), and the index map that would
    // decide it is not reachable from the descriptor.
    //
    // The pipeline is unaffected — the operation is rule-proven, so a
    // wrong one cannot arise from saturation, only from hand-seeding —
    // but the PARSER-DOESN'T-TRUST-THE-TERM perimeter this test was
    // built to defend is genuinely weaker after E1. Pinned as such so
    // the trade-off stays visible rather than being forgotten.
    // PINNED (round-8b audit): both arms were findings while the cost of
    // E1 was being characterised; the behaviour is now known, so it is
    // pinned. If the perimeter is ever restored (or the seed starts
    // refusing for another reason), this test must go red rather than
    // quietly reporting the other branch.
    // ROUND-10 RE-PIN (the E1 perimeter PARTIALLY RESTORED — the flip
    // this test demanded if that ever happened): wiring descriptor A to
    // the site's b is a ROLE SWAP, and the unswapped parser's call-frame
    // derivation (k = the A-storage extent that is not m) catches it
    // loudly — the seeded w[4,3] shares no extent with call-m 2. What E1
    // could not detect (a wrong OPERATION on a transposable shape) is
    // still undetectable; what came back is the ROLE perimeter.
    match outcome {
        Err(msg) => {
            let text = format!("{msg:?}");
            assert!(
                text.contains("shares no extent class")
                    || text.contains("transposition of its storage")
                    || text.contains("inconsistent"),
                "RS3c: the refusal names the violated cross-check, got {text}"
            );
            println!("RS3c: the seeded role-swap REFUSES loudly (round-10 perimeter): {text}");
        }
        Ok(specs) => panic!(
            "RS3c: the seeded role-swap PARSED into {specs:?} — the round-10 \
             role perimeter regressed; this must refuse via the call-frame \
             form/layout perimeter; the operation is rule-proven, so the pipeline \
             is unaffected, but extraction can no longer catch a hand-seeded \
             wrong operation (recorded cost of deleting the form child)"
        ),
    }
}

// ===========================================================================
// RT — STRIDE-SPELLING ROBUSTNESS
// ===========================================================================

/// Every op candidate whose B reading's layout class carries a discriminated
/// pitch factor (i.e. reads the STRIDED spelling), with that factor's value.
fn strided_b_candidates(s: &EGraph) -> Vec<(NodeId, Dim)> {
    s.nodes
        .iter()
        .filter_map(|(id, n)| {
            if !n.op.starts_with("LayoutTensorOpCublasLt") {
                return None;
            }
            let bc = class_of_child(s, n, 2)?;
            let desc = nodes_in_class(s, &bc, "CublasLtOperandBDescriptor")
                .first()
                .copied()?;
            let lt = class_of_child(s, desc, 1)?;
            let layout = layout_of_lt(s, &lt)?;
            // ROUND-11: exclude BOTH contiguous families — the collapse-
            // derived column-form frame layouts are LEFT-major contiguous
            // and are not pitched readings.
            if class_has(s, &layout, "RightMajorContiguousElementLayoutLit")
                || class_has(s, &layout, "LeftMajorContiguousElementLayoutLit")
            {
                return None; // a contiguous frame reading
            }
            let factors = pitch_factor_classes(s, &layout);
            let factor = factors.first()?;
            Some((id.clone(), parse_dim(s, factor)))
        })
        .collect()
}

/// A hostile stride spelling of x's row pitch, alongside x's plain
/// contiguous layout (the tempting wrong answer, ld = 4). The strided
/// reading must mint and must read back ld = `expect` — never 1 (the
/// subsumed unit factor), never 4 (the sibling's number).
fn spelling_probe(tag: &str, row_entry: &str, stride_expr: &str, expect: i64) {
    // ROUND 10: the creator authors its strided-lists provenance row with
    // the SAME hostile stride spelling — the ladder + arithmetic weld the
    // spellings, and the reading still comes back as the one true pitch.
    let fx = Fx {
        x_layouts: vec![rm("a_shape"), strided("a_shape", row_entry)],
        tail: format!(
            "(set (injectivity-of x_lt1) (Injective))\n(strided-lists (StridedElementLayoutLit a_shape (IntAffineExprCons {row_entry} (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil))) (bits-of (F32))) a_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil))) (IntExprCons {stride_expr} (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n"
        ),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    println!(
        "{tag} `{row_entry}`: b={b} reading(s), {} candidates",
        count_cublaslt(&s)
    );
    dump_reading_sets(tag, &s);
    assert_coherent(tag, &s);
    // ROUND-11 RE-PIN (was 3): + the column-form frame readings of the
    // contiguous operands; the hostile pitched reading still mints
    // exactly once beside them.
    assert_eq!(
        b, 5,
        "{tag}: the hostile spelling mints a reading beside the contiguous one"
    );

    // The oracle first: the strided reading's ld, read independently.
    // ROUND-11: the collapse-derived column-form frame layouts are
    // LEFT-major contiguous — exclude both contiguous families so the
    // oracle isolates the genuinely pitched reading (same fix as the e3
    // probes').
    let strided_lds: BTreeSet<Option<i64>> = all_readings(&s)
        .into_iter()
        .filter(|r| {
            layout_of_lt(&s, &r.lt)
                .map(|l| {
                    !class_has(&s, &l, "RightMajorContiguousElementLayoutLit")
                        && !class_has(&s, &l, "LeftMajorContiguousElementLayoutLit")
                })
                .unwrap_or(false)
                && r.role == "B"
        })
        .map(|r| r.ld.lit())
        .collect();
    println!("  {tag} strided reading ld(s): {strided_lds:?}");
    assert_eq!(
        strided_lds,
        BTreeSet::from([Some(expect)]),
        "{tag}: the hostile spelling must read back exactly {expect}"
    );

    // Then the marker: elect the strided candidate and field-check.
    // ROUND-11 RE-PIN (was 1): the pitched B reading pairs with the TWO
    // A-frame readings of its site — two candidates carrying the pitch.
    let candidates = strided_b_candidates(&s);
    assert_eq!(
        candidates.len(),
        2,
        "{tag}: the pitched reading's two A-frame pairings"
    );
    let (genome, hit) = genome_electing(&s, &candidates[0].0);
    assert!(hit, "{tag}: the strided candidate is electable");
    let graph = extract_with_genome(&s, &genome);
    let plan = cublaslt_in_plan(&graph);
    assert_eq!(plan.len(), 1, "{tag}: one kernel elected");
    let spec = spec_of(&plan[0].0);
    println!(
        "  {tag} elected: lda={} ldb={} ldd={}",
        spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(
        spec.ldb, expect,
        "{tag}: ldb must be the real pitch, not a normalization residue"
    );
    assert_ne!(
        spec.ldb, 1i64,
        "{tag}: the unit factor must never masquerade as the pitch"
    );
    assert_ne!(
        spec.ldb, 4i64,
        "{tag}: the contiguous sibling's ld must not be borrowed"
    );
    // ROUND-11 RE-PIN (lda was exactly 3): the elected pairing may put
    // either A frame beside the pitched B reading — the storage frame
    // (lda = 3) or the column-form frame (lda = 4). Both are clamp-sound
    // for their own bytes; the pitch readback above is the oracle.
    assert!(
        spec.lda == 3i64 || spec.lda == 4i64,
        "{tag}: lda is one of the A frames' own lds, got {}",
        spec.lda
    );
    assert_eq!(spec.ldd, 3);
    assert_eq!(spec.mnk_lits(), (3, 2, 4));
}

/// `coord * (8 * 1)` — the preamble subsumes the unit factor.
#[test]
fn rt1_wrapped_unit_factor_row_term() {
    spelling_probe(
        "RT1",
        "(IntMul (CoordVar a_shape 1) (IntMul (IntLit 8) (IntLit 1)))",
        "(IntMul (IntLit 8) (IntLit 1))",
        8,
    );
}

/// `8 * coord` — the coefficient on the LEFT (commuted operands).
#[test]
fn rt2_left_coefficient_row_term() {
    spelling_probe(
        "RT2",
        "(IntMul (IntLit 8) (CoordVar a_shape 1))",
        "(IntLit 8)",
        8,
    );
}

/// `coord * (2 * 4)` — a folded product; constant folding does not subsume,
/// so the class holds BOTH `(IntMul 2 4)` and `8`.
#[test]
fn rt3_folded_coefficient_row_term() {
    spelling_probe(
        "RT3",
        "(IntMul (CoordVar a_shape 1) (IntMul (IntLit 2) (IntLit 4)))",
        "(IntMul (IntLit 2) (IntLit 4))",
        8,
    );
}

/// TWO static pitches on one tensor (separate layout classes): sibling
/// candidates, each electing its OWN pitch. Nothing may drift across — and
/// the marker's ambiguity panic must NOT fire, because the two factors live
/// in two layout classes, not one.
#[test]
fn rt4_two_static_pitches_coherent() {
    let fx = Fx {
        x_layouts: vec![
            rm("a_shape"),
            strided("a_shape", "(IntMul (CoordVar a_shape 1) (IntLit 8))"),
            strided("a_shape", "(IntMul (CoordVar a_shape 1) (IntLit 16))"),
        ],
        // ROUND 10: each creator authors its provenance row.
        tail: "(set (injectivity-of x_lt1) (Injective))\n(set (injectivity-of x_lt2) (Injective))\n(strided-lists (StridedElementLayoutLit a_shape (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8)) (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil))) (bits-of (F32))) a_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil))) (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n(strided-lists (StridedElementLayoutLit a_shape (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 16)) (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil))) (bits-of (F32))) a_shape (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil))) (IntExprCons (IntLit 16) (IntExprCons (IntLit 1) (IntExprNil))) (bits-of (F32)))\n"
            .into(),
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    println!(
        "RT4 two pitches: b={b} readings, {} candidates",
        count_cublaslt(&s)
    );
    dump_reading_sets("RT4", &s);
    assert_coherent("RT4", &s);
    // ROUND-11 RE-PIN (was 4/6): + the column-form frame readings.
    assert_eq!(
        b, 6,
        "contiguous frames + two pitched readings + the original site's w frames"
    );
    // Products scale with the frame doubling on the un-pitched roles.
    let rt4_ops = count_cublaslt(&s);
    println!("RT4 candidates: {rt4_ops}");
    assert_eq!(rt4_ops, 16, "the per-site reading products, frame-doubled");

    let lds: BTreeSet<Option<i64>> = all_readings(&s)
        .into_iter()
        .filter(|r| r.role == "B")
        .map(|r| r.ld.lit())
        .collect();
    println!("  B candidate lds: {lds:?}");
    // ROUND-11 RE-PIN (was {3,4,8,16}): + the column-form frame reading
    // (ld=2). The pitch-ownership law is asserted on the PITCHED readings
    // below; the full set is pinned for census honesty.
    assert_eq!(
        lds,
        BTreeSet::from([Some(2), Some(3), Some(4), Some(8), Some(16)]),
        "each reading carries ITS OWN ld; no cross-drift, no ambiguity"
    );
    let pitched_lds: BTreeSet<Option<i64>> = all_readings(&s)
        .into_iter()
        .filter(|r| {
            r.role == "B"
                && layout_of_lt(&s, &r.lt)
                    .map(|l| {
                        !class_has(&s, &l, "RightMajorContiguousElementLayoutLit")
                            && !class_has(&s, &l, "LeftMajorContiguousElementLayoutLit")
                    })
                    .unwrap_or(false)
        })
        .map(|r| r.ld.lit())
        .collect();
    assert_eq!(
        pitched_lds,
        BTreeSet::from([Some(8), Some(16)]),
        "the two pitched readings carry exactly their own pitches"
    );

    // Per-enode election: each candidate hands back its own pitch.
    let ids: Vec<NodeId> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op.starts_with("LayoutTensorOpCublasLt"))
        .map(|(id, _)| id.clone())
        .collect();
    let mut seen = BTreeSet::new();
    for id in &ids {
        let (genome, hit) = genome_electing(&s, id);
        if !hit {
            continue;
        }
        let graph = extract_with_genome(&s, &genome);
        for (op, _) in cublaslt_in_plan(&graph) {
            if let Some(v) = spec_of(&op).ldb.literal() {
                seen.insert(v);
            }
        }
    }
    println!("  elected ldb values across per-enode election: {seen:?}");
    // ROUND-11 RE-PIN (was {4,8,16}): the column-form frame reading
    // (ldb = 2) is per-enode-electable too. Every pitch is still
    // reachable and none drifts into another's election.
    assert_eq!(
        seen,
        BTreeSet::from([2, 4, 8, 16]),
        "all pitches reachable, none drifts into another's election"
    );
}

// ===========================================================================
// RU — DEGENERATE CORNERS + THE WELD-HARVESTING TRIPWIRE
// ===========================================================================

/// Corner sweep with EXACT candidate counts pinned — the design's canary.
/// The dot product (m = n = 1) is the headline; the neighbourhood (m=1,
/// n=1, k=1, all-ones) is pinned alongside because every round has moved
/// these corners.
#[test]
fn ru1_dot_product_multiplicity() {
    struct Corner {
        name: &'static str,
        m: i64,
        n: i64,
        k: i64,
        /// how many marker SITES the corner admits. ROUND 10: every
        /// matmul carries the original site AND its transpose-sandwich
        /// sibling, so this is 2 everywhere (the all-ones role-swap sites
        /// hash-cons INTO the pair).
        sites: usize,
        /// The MULTISET of per-site (A, B, D) reading counts — PINNED.
        /// Extent-1 welds still double readings exactly as before, but
        /// the two sites of a pair can weld asymmetrically (a gemv's
        /// original site welds its A reading, the sibling its B).
        readings: &'static [(usize, usize, usize)],
        /// call (m, n, k) CANONICAL (m <= n) — every candidate must be one
        /// of the two sandwich frames of this call.
        mnk: (i64, i64, i64),
    }
    let corners = [
        Corner {
            name: "dot m=n=1 k=4",
            m: 1,
            n: 1,
            k: 4,
            sites: 2,
            readings: &[(2, 2, 1), (2, 2, 1)],
            mnk: (1, 1, 4),
        },
        Corner {
            name: "gemv m=1",
            m: 1,
            n: 3,
            k: 4,
            // ROUND-11 RE-PIN (was [(2,1,1),(1,2,1)]): two frames per
            // operand even at the degenerate extents.
            sites: 2,
            readings: &[(2, 2, 1), (2, 2, 1)],
            mnk: (1, 3, 4),
        },
        Corner {
            name: "gemv n=1",
            m: 2,
            n: 1,
            k: 4,
            // ROUND-11 RE-PIN (was [(1,2,1),(2,1,1)]): as above.
            sites: 2,
            readings: &[(2, 2, 1), (2, 2, 1)],
            mnk: (1, 2, 4),
        },
        Corner {
            name: "k=1",
            m: 2,
            n: 3,
            k: 1,
            sites: 2,
            readings: &[(2, 2, 1), (2, 2, 1)],
            mnk: (2, 3, 1),
        },
        Corner {
            name: "all-ones",
            m: 1,
            n: 1,
            k: 1,
            // ROUND-11 RE-PIN (was 2 sites): the fully welded corner lets
            // every canonicalization pattern fire (see attack_a11) — 8
            // welded site families, all describing the 1x1x1 call.
            sites: 8,
            readings: &[
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 1),
            ],
            mnk: (1, 1, 1),
        },
    ];
    println!("MEASURE ru1 degenerate corners");
    let mut problems: Vec<String> = Vec::new();
    for c in corners {
        let fx = Fx {
            m: format!("(IntLit {})", c.m),
            n: format!("(IntLit {})", c.n),
            k: format!("(IntLit {})", c.k),
            a_rows: format!("(IntLit {})", c.m),
            a_cols: format!("(IntLit {})", c.k),
            b_rows: format!("(IntLit {})", c.k),
            b_cols: format!("(IntLit {})", c.n),
            ..Default::default()
        }
        .render();
        let s = test_runtime::serialize_fixture(&fx);
        let site_list = sites(&s);
        let ops = count_cublaslt(&s);
        println!(
            "  {}: {} nodes, {} site(s), {ops} candidate(s) total",
            c.name,
            s.nodes.len(),
            site_list.len()
        );
        dump_reading_sets(&format!("RU1/{}", c.name), &s);
        assert_coherent(&format!("RU1/{}", c.name), &s);

        // THE PRODUCT LAW, PER SITE (never program-wide, never program-size).
        if site_list.len() != c.sites {
            problems.push(format!(
                "{}: {} sites != pinned {}",
                c.name,
                site_list.len(),
                c.sites
            ));
        }
        let mut observed: Vec<(usize, usize, usize)> = Vec::new();
        let mut total_product = 0usize;
        for site in &site_list {
            let per_role = readings_for_site(&s, site);
            let count = |role: &str| per_role.get(role).map(|v| v.len()).unwrap_or(0);
            let (a, b, d) = (count("A"), count("B"), count("D"));
            let here = s
                .nodes
                .values()
                .filter(|n| {
                    n.op.starts_with("LayoutTensorOpCublasLt")
                        && class_of_child(&s, n, 0).as_ref() == Some(site)
                })
                .count();
            println!(
                "    site {}: a={a} b={b} d={d} -> {here} candidate(s)",
                short(site)
            );
            observed.push((a, b, d));
            total_product += a * b * d;
            if here != a * b * d {
                problems.push(format!(
                    "{}: site {} has {here} candidates != product {}",
                    c.name,
                    short(site),
                    a * b * d
                ));
            }
        }
        observed.sort_unstable();
        let mut pinned: Vec<(usize, usize, usize)> = c.readings.to_vec();
        pinned.sort_unstable();
        if observed != pinned {
            problems.push(format!(
                "{}: per-site readings {observed:?} != pinned {pinned:?}",
                c.name
            ));
        }

        // EVERY candidate elected in turn, each checked for numerical
        // soundness against the cuBLASLt COL clamps.
        let mut frames: BTreeSet<(bool, bool, i64, i64, i64)> = BTreeSet::new();
        for summary in elect_all(&s) {
            println!("    reachable frame: {summary:?}");
            let (Some(lda), Some(ldb), Some(ldd)) = (summary.lda, summary.ldb, summary.ldd) else {
                problems.push(format!("{}: a corner frame has a symbolic ld", c.name));
                continue;
            };
            frames.insert((summary.trans_a, summary.trans_b, lda, ldb, ldd));
            // ROUND 10: a candidate lives in ONE of the two sandwich
            // frames; canonicalize before comparing, and check the COL
            // clamps against the candidate's OWN frame. (The enumerated
            // per-frame tuple pins are subsumed by these two checks; the
            // trade is recorded in the round-10 report.)
            let (Some(sm), Some(sn), Some(sk)) = (summary.m, summary.n, summary.k) else {
                problems.push(format!("{}: symbolic corner frame {summary:?}", c.name));
                continue;
            };
            let canonical = if sm <= sn { (sm, sn, sk) } else { (sn, sm, sk) };
            if canonical != c.mnk {
                problems.push(format!("{}: call frame {summary:?} !~ {:?}", c.name, c.mnk));
            }
            let rows_a = if summary.trans_a { sk } else { sm };
            let rows_b = if summary.trans_b { sn } else { sk };
            if lda < rows_a || ldb < rows_b || ldd < sm {
                problems.push(format!(
                    "{}: UNSOUND frame trans_a={} trans_b={} lda={lda} ldb={ldb} \
                     ldd={ldd} against its own m={sm} n={sn} k={sk}",
                    c.name, summary.trans_a, summary.trans_b
                ));
            }
        }
        if ops != total_product {
            problems.push(format!(
                "{}: {ops} candidates total != per-site product sum {total_product}",
                c.name
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "ru1 corner sweep:\n  {}",
        problems.join("\n  ")
    );
}

/// A LEFT-MAJOR-ONLY seeded operand. The only reading of x ITSELF is the
/// left-major arm (the second B reading belongs to the materializing COPY
/// of x, a different layout tensor). Elect the left-major candidate and
/// field-check the whole spec.
#[test]
fn ru2_left_major_only_election_field_check() {
    let fx = Fx {
        x_layouts: vec![lm("a_shape")],
        ..Default::default()
    }
    .render();
    let s = test_runtime::serialize_fixture(&fx);
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    println!(
        "RU2 left-major-only: b={b} reading(s), {} candidates",
        count_cublaslt(&s)
    );
    dump_reading_sets("RU2", &s);
    assert_coherent("RU2", &s);
    // ROUND-11 RE-PIN (was 3/4): + the column-form frame readings.
    assert_eq!(b, 4, "seeded LM + blanket RM + the column-form frames");
    let ru2_ops = count_cublaslt(&s);
    println!("RU2 candidates: {ru2_ops}");
    assert_eq!(ru2_ops, 8);

    // The SEEDED x layout tensor carries exactly ONE reading, and it is the
    // left-major one — the copy's reading rides a DIFFERENT layout tensor.
    // ROUND-11: the sibling's viewed operand ALSO rides an LM-class
    // composed layout; restrict to readings whose LOGICAL is the seeded
    // input tensor itself.
    let seeded: Vec<Reading> = all_readings(&s)
        .into_iter()
        .filter(|r| {
            r.role == "B"
                && s.nodes
                    .values()
                    .any(|m| m.eclass == r.logical && m.op == "LogicalTensorInputLit")
                && layout_of_lt(&s, &r.lt)
                    .map(|l| class_has(&s, &l, "LeftMajorContiguousElementLayoutLit"))
                    .unwrap_or(false)
        })
        .collect();
    assert_eq!(seeded.len(), 1, "one reading of the seeded left-major x");
    let lm_candidate = s
        .nodes
        .iter()
        .find(|(_, n)| {
            n.op.starts_with("LayoutTensorOpCublasLt")
                && class_of_child(&s, n, 2)
                    .and_then(|c| {
                        nodes_in_class(&s, &c, "CublasLtOperandBDescriptor")
                            .first()
                            .copied()
                    })
                    .and_then(|d| class_of_child(&s, d, 1))
                    .map(|lt| lt == seeded[0].lt)
                    .unwrap_or(false)
        })
        .map(|(id, _)| id.clone())
        .expect("a candidate over the left-major reading");
    let (genome, hit) = genome_electing(&s, &lm_candidate);
    assert!(hit, "the left-major candidate is electable");
    let elected: Vec<CublasLt> = cublaslt_in_plan(&extract_with_genome(&s, &genome))
        .into_iter()
        .map(|(op, _)| op)
        .collect();
    assert_eq!(elected.len(), 1);
    let spec = spec_of(&elected[0]);
    println!(
        "  m={} n={} k={} trans_a={} trans_b={} lda={} ldb={} ldc={} ldd={} col={}",
        spec.m,
        spec.n,
        spec.k,
        spec.trans_a,
        spec.trans_b,
        spec.lda,
        spec.ldb,
        spec.ldc,
        spec.ldd,
        spec.order_col
    );
    assert_eq!(spec.mnk_lits(), (3, 2, 4));
    assert!(!spec.trans_a, "w is right-major: N");
    assert!(spec.trans_b, "x is LEFT-major: the trans flip");
    assert_eq!(spec.lda, 3, "w storage cols");
    assert_eq!(spec.ldb, 2, "left-major ld = storage ROWS (m)");
    assert_eq!(spec.ldd, 3);
    assert_eq!(spec.ldc, spec.ldd);
    assert!(spec.order_col);
    assert!(!spec.has_c && !spec.has_bias);
    assert_eq!(spec.epilogue, CuEpilogue::Default);
    assert_call_sound("RU2", spec);
}

/// ===== ROUND-8 LOAD-BEARING FIXTURE 1 =====
/// A recorded m=1 GEMV (x[1,4] @ w[4,3]). At m=1 the row coordinate is the
/// ZERO class, so round 4's `(IntMul (CoordVar shape 1) ?ld)` destructure
/// enumerated the whole welded class and harvested every literal in the
/// program as a candidate ld (10 candidates from this one site). Today the
/// rules bind no numerics at all and the reader's dead-axis policy never
/// walks the entry. This test pins BOTH: the candidate readings are bounded
/// AND every one of them comes from THIS tensor's own layout(s).
#[test]
fn ru3_m1_corner_multiplicity() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((1usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let s = test_runtime::serialize_fixture(&text);
    let a = count_op(&s, "CublasLtOperandADescriptor");
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    let d = count_op(&s, "CublasLtOutputDDescriptor");
    let ops = count_cublaslt(&s);
    println!(
        "RU3 m=1 GEMV: {} nodes, {} site(s), a={a} b={b} d={d}, {ops} candidate(s)",
        s.nodes.len(),
        sites(&s).len()
    );
    dump_reading_sets("RU3", &s);
    assert_coherent("RU3", &s);

    // (1) BOUNDED: the candidate count is the product of the arms that fire,
    // not a function of how many literals the program contains. At m=1 the
    // right-major and left-major contiguous layouts of x[1,4] are the SAME
    // bytes, so they weld into one layout class and BOTH arms read it — a
    // legal multiplicity of exactly 2, not the 10 round 4 minted.
    // ROUND-10 RE-PIN: original + sibling site; the weld doubles ONE
    // operand reading on each site (asymmetrically: the original welds
    // its A=x reading, the sibling its B=x reading).
    assert_eq!(sites(&s).len(), 2, "original + sibling");
    // ROUND-11 RE-PIN (was (3,3,2)/4): + the column-form frame readings.
    assert_eq!(
        (a, b, d),
        (4, 4, 2),
        "the welded x layout is read two legal ways per site"
    );
    assert_eq!(
        ops, 8,
        "EIGHT candidates — round 4 minted TEN here from ONE site"
    );

    // (2) OWN LAYOUTS ONLY: every reading's layout tensor belongs to one of
    // the site's own three logical tensors, and its ld is one of that
    // tensor's own extents.
    // Own tensors = the union over BOTH sites of the pair (the sibling's
    // out is the transpose view of the original's).
    let mut own: Vec<ClassId> = Vec::new();
    for site in sites(&s) {
        let site_node = *nodes_in_class(&s, &site, "CublasLtLogicalMatmulSite")
            .first()
            .unwrap();
        for i in 0..3 {
            if let Some(c) = class_of_child(&s, site_node, i) {
                if !own.contains(&c) {
                    own.push(c);
                }
            }
        }
    }
    let mut candidate_lds = BTreeSet::new();
    for r in all_readings(&s) {
        assert!(
            own.contains(&r.logical),
            "reading over a FOREIGN logical tensor {} (site owns {:?})",
            short(&r.logical),
            own.iter().map(short).collect::<Vec<_>>()
        );
        let storage = storage_dims(&s, &r.logical).expect("storage dims");
        let extents: BTreeSet<Option<i64>> = storage.iter().map(|d| d.lit()).collect();
        assert!(
            extents.contains(&r.ld.lit()),
            "{} reading ld={} is not one of its OWN tensor's extents {extents:?} — HARVESTED",
            r.role,
            r.ld
        );
        candidate_lds.insert(r.ld.lit());
    }
    println!("  candidate ld set (whole site): {candidate_lds:?}");
    assert_eq!(
        candidate_lds,
        BTreeSet::from([Some(1), Some(3), Some(4)]),
        "exactly the site's own extents: x rows (1) / x cols (4) / w+out cols (3)"
    );

    // (2b) THE RAW HARVEST SURFACE, measured. At m=1 the row coordinate IS
    // the zero class, so the row-entry position of x's layout is welded with
    // every other extent-1 row entry in the program. Round 4 DESTRUCTURED
    // that position and harvested it; today's rules never bind a numeric
    // there and the reader's dead-axis policy never walks it — so the
    // surface may be large while the reading set stays at 2.
    let mut surface: BTreeSet<Option<i64>> = BTreeSet::new();
    for r in all_readings(&s) {
        if let Some(layout) = layout_of_lt(&s, &r.lt) {
            for f in pitch_factor_classes(&s, &layout) {
                surface.insert(parse_dim(&s, &f).lit());
            }
        }
    }
    println!(
        "  RAW pitch-factor surface at the site's own layouts: {} distinct literal(s) {:?}",
        surface.len(),
        surface
    );
    // THE FIXTURE MUST STILL BITE. A regression fixture that no longer
    // reaches the hazard proves nothing. If this ever fails, the extent-1
    // pointing weld changed and ru3/ru4 must be re-derived before they can
    // be trusted as round-8 evidence.
    assert!(
        surface.len() > 1,
        "ru3 no longer reaches the harvesting hazard: the m=1 row-entry class \
         exposes only {surface:?} — re-derive this fixture"
    );
    // ...and the reader must not be reading it: the candidate ld set is a
    // strict, tiny subset of that surface.
    assert!(
        candidate_lds.len() < surface.len(),
        "candidate lds {candidate_lds:?} are not a strict subset of the raw \
         surface {surface:?} — the reader is walking the welded class"
    );

    // (3) EVERY reachable kernel is numerically sound.
    let frames = elect_all(&s);
    println!("  reachable frames: {frames:?}");
    // ROUND-10 RE-PIN: at m=1 the ORIGINAL site's out is COL-presenting
    // (a [1,n] right-major IS column-major), so the reachable kernels are
    // the DIRECT frame's — call (1, 3, 4), A = x (its welded layout read
    // N or T: lda in {1, 4}), B = w (T: ldb = n = 3), ldd = m = 1. The
    // sibling's candidates exist but the boundary elects the direct claim.
    // ROUND-11 RE-PIN (was 2): the frame doubling adds the column-form
    // readings; all reachable kernels stay in the one call frame
    // (asserted below).
    assert_eq!(frames.len(), 4, "the sound readings of the welded x layout");
    for f in &frames {
        assert_eq!(
            (f.m, f.n, f.k),
            (Some(1), Some(3), Some(4)),
            "every candidate agrees on the call frame"
        );
        let rows_b = if f.trans_b { f.n } else { f.k };
        assert!(
            f.ldb >= rows_b.map(|v| v.min(3)),
            "COL clamp: ldb={:?} < rows(B)={rows_b:?} for {f:?}",
            f.ldb
        );
        // ROUND-11 RE-PIN (was exactly 3): w reads T in its storage frame
        // (ldb = n = 3) and N in its column-form frame (ldb = k = 4);
        // both are w's own numbers, cross-checked by the clamp above.
        assert!(
            f.ldb == Some(3) || f.ldb == Some(4),
            "ldb is one of w's own frame readings, got {:?}",
            f.ldb
        );
        assert_eq!(f.ldd, Some(1), "ldd is the direct frame's m = 1");
        assert!(
            f.lda == Some(4) || f.lda == Some(1),
            "lda is one of x's OWN extents, got {:?}",
            f.lda
        );
    }

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = spec_of(&elected[0]);
    println!(
        "  elected: m={} n={} k={} trans_b={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.trans_b, spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(spec.mnk_lits(), (1, 3, 4));
    assert_call_sound("RU3", spec);
}

/// ===== ROUND-8 LOAD-BEARING FIXTURE 2 =====
/// PROGRAM-SIZE INDEPENDENCE. The same m=1 GEMV plus an UNRELATED [1,37]
/// tensor elsewhere in the program. Round 4 went from 10 candidates to 23
/// here, because the stranger's literals joined the zero-welded entry class
/// and were harvested as lds. Today the GEMV's candidate set must be
/// IDENTICAL with and without the stranger, and 37 must appear nowhere as
/// a candidate ld.
#[test]
fn ru4_weld_harvesting_is_cross_tensor() {
    fn gemv(with_stranger: bool) -> String {
        let mut cx = Graph::new();
        let x = cx.tensor((1usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
        if with_stranger {
            // The bait: an unrelated [1,37] tensor whose row coordinate is
            // ALSO the zero class (extent-1 row) and whose cols literal is
            // 37 — exactly what round 4 harvested.
            let stranger = cx.tensor((1usize, 37usize), DType::F32);
            let _ = (stranger + stranger).output();
        }
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    }

    /// The GEMV site's candidate set, as a comparable fingerprint:
    /// (role, form, operation, ld-literal) per reading, the candidate
    /// count, and the RAW pitch-factor surface (the literals reachable in
    /// the row-entry position round 4 destructured). Deliberately NOT keyed
    /// on class ids — they move between runs.
    fn fingerprint(
        s: &EGraph,
    ) -> (
        Vec<(String, String, Option<i64>)>,
        usize,
        BTreeSet<Option<i64>>,
    ) {
        // The GEMV site is the one whose out storage is [1,3].
        let site = sites(s)
            .into_iter()
            .find(|site| {
                let node = *nodes_in_class(s, site, "CublasLtLogicalMatmulSite")
                    .first()
                    .unwrap();
                class_of_child(s, node, 2)
                    .and_then(|out| storage_dims(s, &out))
                    .map(|dims| dims == vec![Dim::Lit(1), Dim::Lit(3)])
                    .unwrap_or(false)
            })
            .expect("the GEMV site");
        let mut readings: Vec<(String, String, Option<i64>)> = all_readings(s)
            .into_iter()
            .filter(|r| r.site == site)
            .map(|r| (r.role, r.operation, r.ld.lit()))
            .collect();
        readings.sort();
        let candidates = s
            .nodes
            .values()
            .filter(|n| {
                n.op.starts_with("LayoutTensorOpCublasLt")
                    && class_of_child(s, n, 0).as_ref() == Some(&site)
            })
            .count();
        // The RAW HARVEST SURFACE: which literals are reachable in the
        // row-entry position of the site's OWN operand layouts. At m=1 that
        // position is the zero class, so the whole program's extent-1 row
        // entries weld into it — this is the set round 4 harvested.
        let mut surface: BTreeSet<Option<i64>> = BTreeSet::new();
        for r in all_readings(s).into_iter().filter(|r| r.site == site) {
            if let Some(layout) = layout_of_lt(s, &r.lt) {
                for f in pitch_factor_classes(s, &layout) {
                    surface.insert(parse_dim(s, &f).lit());
                }
            }
        }
        (readings, candidates, surface)
    }

    let plain = test_runtime::serialize_fixture(&gemv(false));
    let polluted = test_runtime::serialize_fixture(&gemv(true));
    println!(
        "RU4: plain {} nodes / polluted {} nodes ({} extra from the stranger)",
        plain.nodes.len(),
        polluted.nodes.len(),
        polluted.nodes.len() as i64 - plain.nodes.len() as i64
    );
    dump_reading_sets("RU4 plain", &plain);
    dump_reading_sets("RU4 polluted", &polluted);
    assert_coherent("RU4 plain", &plain);
    assert_coherent("RU4 polluted", &polluted);

    let (r_plain, c_plain, s_plain) = fingerprint(&plain);
    let (r_poll, c_poll, s_poll) = fingerprint(&polluted);
    println!(
        "  plain    : {c_plain} candidate(s); RAW pitch-factor surface {} literal(s) {s_plain:?}",
        s_plain.len()
    );
    for r in &r_plain {
        println!("      {r:?}");
    }
    println!(
        "  polluted : {c_poll} candidate(s); RAW pitch-factor surface {} literal(s) {s_poll:?}",
        s_poll.len()
    );
    for r in &r_poll {
        println!("      {r:?}");
    }
    println!(
        "  VERDICT: the stranger {} the raw harvest surface ({} -> {} literals; 37 {} reachable) \
         while the reading set is {}",
        if s_poll.len() > s_plain.len() {
            "GREW"
        } else {
            "did not grow"
        },
        s_plain.len(),
        s_poll.len(),
        if s_poll.contains(&Some(37)) {
            "IS"
        } else {
            "is NOT"
        },
        if r_plain == r_poll {
            "UNCHANGED"
        } else {
            "POLLUTED"
        }
    );

    // THE BAIT MUST LAND. If the stranger's literals never reach the GEMV's
    // row-entry class, this fixture is not testing anything and must be
    // re-derived before it can be trusted as round-8 evidence.
    assert!(
        s_poll.contains(&Some(37)),
        "ru4 no longer reaches the hazard: the stranger's 37 is not even \
         REACHABLE in the GEMV's welded row-entry class ({s_poll:?}) — \
         re-derive this fixture"
    );
    assert!(
        s_poll.len() > s_plain.len(),
        "ru4 no longer reaches the hazard: the stranger did not grow the raw \
         harvest surface at all ({} -> {})",
        s_plain.len(),
        s_poll.len()
    );

    // THE TRIPWIRE.
    assert_eq!(
        r_plain, r_poll,
        "CROSS-TENSOR POLLUTION: the GEMV's reading set changed when an \
         unrelated [1,37] tensor joined the program"
    );
    assert_eq!(
        c_plain, c_poll,
        "CROSS-TENSOR POLLUTION: the GEMV's candidate count changed with \
         program size ({c_plain} -> {c_poll})"
    );
    // ROUND-11 RE-PIN (was 2): the frame doubling applies to the GEMV
    // exactly as everywhere else — and IDENTICALLY with and without the
    // stranger, which is this tripwire's law.
    assert_eq!(c_plain, 4, "the welded-layout frame readings, both times");
    for (role, _, ld) in r_plain.iter().chain(r_poll.iter()) {
        assert_ne!(*ld, Some(37), "{role}: ld 37 HARVESTED from the stranger");
    }
    for (role, _, ld) in r_poll.iter() {
        assert!(
            matches!(ld, Some(1) | Some(3) | Some(4)),
            "{role}: ld {ld:?} is not one of the GEMV's own extents"
        );
    }

    // And at the extractor: the reachable call frames are IDENTICAL.
    let f_plain: BTreeSet<SpecSummary> = elect_all(&plain);
    // ROUND 10: the reachable GEMV frames are the DIRECT call's (m=1, n=3)
    // — see ru3; filter the stranger's frames out of the polluted set.
    let f_poll: BTreeSet<SpecSummary> = elect_all(&polluted)
        .into_iter()
        .filter(|f| f.m == Some(1) && f.n == Some(3))
        .collect();
    println!("  plain frames    : {f_plain:?}");
    println!("  polluted frames : {f_poll:?}");
    assert_eq!(
        f_plain, f_poll,
        "CROSS-TENSOR POLLUTION at the extractor: the GEMV's reachable call \
         frames changed with program size"
    );
    for f in f_plain.iter().chain(f_poll.iter()) {
        for (name, ld) in [("lda", f.lda), ("ldb", f.ldb), ("ldd", f.ldd)] {
            assert_ne!(ld, Some(37), "{name} harvested 37 from the stranger: {f:?}");
        }
    }

    for (label, text) in [("plain", gemv(false)), ("polluted", gemv(true))] {
        let elected = pinned_cublaslt(&text);
        let gemvs: Vec<_> = elected
            .iter()
            .map(spec_of)
            .filter(|sp| sp.m == 1i64 && sp.n == 3i64)
            .collect();
        assert_eq!(gemvs.len(), 1, "{label}: exactly one GEMV kernel");
        let spec = gemvs[0];
        println!(
            "  {label} elected GEMV: m={} n={} k={} trans_b={} lda={} ldb={} ldd={}",
            spec.m, spec.n, spec.k, spec.trans_b, spec.lda, spec.ldb, spec.ldd
        );
        assert_eq!(
            spec.mnk_lits(),
            (1, 3, 4),
            "{label}: call frame (direct at m=1, round 10)"
        );
        // Direct frame: A = x[1,4] (welded, elected N: lda = m = 1),
        // B = w[4,3] (T: ldb = n = 3), D = out[1,3] (ldd = m = 1).
        assert_eq!(spec.lda, 1, "{label}: lda");
        assert_eq!(spec.ldd, 1, "{label}: ldd");
        assert_call_sound(label, spec);
        for (name, dim) in [("lda", &spec.lda), ("ldb", &spec.ldb), ("ldd", &spec.ldd)] {
            assert_ne!(dim.literal(), Some(37), "{label}: {name} harvested 37");
        }
    }
}
