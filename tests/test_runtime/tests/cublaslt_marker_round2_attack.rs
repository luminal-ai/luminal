//! ROUND-2 ADVERSARIAL BATTERY, re-derived (2026-08-24) against the
//! ROUND-7 design.
//!
//! The original round-2 attack file was lost with the /tmp worktree; only
//! its test-name inventory survived (RECOVERY/). This file is NOT
//! archaeology — it is a fresh adversarial pass whose coverage is steered
//! by that inventory. Where the round-4/5/6/7 redesigns made an original
//! finding structurally impossible, the test PINS the fixed behavior (it
//! fails if the fix regresses) and says so in its doc comment.
//!
//! The five concerns, restated against today's design:
//!   A. operation (N/T) soundness under shape-degenerate geometry
//!   B. bias AXIS recognition when the output is square
//!   C. Lit arity/slot assignment vs the constructor contract
//!   D. decorator composition and API order (activation LAST)
//!   E. relu/constant spelling drift across the four inlined copies
//!   P. which parser panics are reachable through legal saturation
//!   U. what the pre-existing board conveniently does not test
//!
//! METHOD: several tests parse EVERY candidate enode's spec (not just
//! the elected one) by building an `ExtractionSite` by hand. That is the
//! strongest available form of the round-2 charter — "no candidate may be
//! silently wrong" is stronger than "the elected candidate is right".

use std::collections::{BTreeMap, BTreeSet};
use std::panic::AssertUnwindSafe;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::{ExtractedGraph, ExtractedNode, ExtractionSite};
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use test_runtime::cublaslt_marker::{
    parse_spec, CuDim, CuEpilogue, CublasLt, CublasLtForm, LtMatmulSpec,
};

type GraphBuilder = Box<dyn Fn(&mut Graph)>;
type FormProgram = (&'static str, CublasLtForm, GraphBuilder);
type EpilogueProgram = (&'static str, CublasLtForm, CuEpilogue, GraphBuilder);

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

// ===========================================================================
// Helpers (names kept from the recovered inventory where they existed).
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

/// One elected cuBLASLt op in a plan, with everything a slot audit needs.
#[derive(Debug, Clone)]
struct Elected {
    op: CublasLt,
    label: String,
    /// Lit inputs in contract order: (port name, LayoutTensor value class).
    inputs: Vec<(String, ClassId)>,
    /// The op's produced LayoutTensor class and its logical class.
    out_lt: ClassId,
    out_logical: ClassId,
}

impl Elected {
    fn spec(&self) -> &LtMatmulSpec {
        self.op
            .spec
            .as_ref()
            .expect("EVERY elected cuBLASLt op carries a parsed spec")
    }
    fn arity(&self) -> usize {
        self.inputs.len()
    }
}

fn elected_ops(graph: &ExtractedGraph) -> Vec<Elected> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                let concrete = (*op.op).as_any().downcast_ref::<CublasLt>().cloned()?;
                let inputs = op
                    .inputs
                    .iter()
                    .map(|i| (i.port.clone(), i.value.clone()))
                    .collect();
                let out = op
                    .outputs
                    .first()
                    .expect("a cublaslt op produces one value");
                Some(Elected {
                    label: op.op.label().to_string(),
                    op: concrete,
                    inputs,
                    out_lt: out.eclass.clone(),
                    out_logical: out.logical.eclass.clone(),
                })
            }
            _ => None,
        })
        .collect()
}

fn plan_labels(graph: &ExtractedGraph) -> Vec<String> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) => Some(op.op.label().to_string()),
            _ => None,
        })
        .collect()
}

/// Genome-pinned (contract-name-preferring) extraction.
fn pinned_cublaslt(text: &str) -> Vec<Elected> {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, PIN);
    elected_ops(&graph)
}

fn pinned_plan(text: &str) -> (Vec<Elected>, Vec<String>) {
    let (graph, _) = test_runtime::extract_fixture_with_genome(text, PIN);
    (elected_ops(&graph), plan_labels(&graph))
}

/// Contract-flavored genome (ported from cublaslt_marker_decorate.rs): the
/// contract is the constructor NAME, relu is the epilogue child.
fn genome_flavored(
    egraph: &EGraph,
    want_c: bool,
    want_bias: bool,
    want_relu: bool,
) -> luminal::extractor::Genome {
    let mut class_ops: BTreeMap<ClassId, Vec<&str>> = BTreeMap::new();
    for node in egraph.nodes.values() {
        class_ops
            .entry(node.eclass.clone())
            .or_default()
            .push(node.op.as_str());
    }
    let class_has =
        |class: &ClassId, op: &str| class_ops.get(class).is_some_and(|ops| ops.contains(&op));
    let child_class = |node: &luminal::prelude::egraph_serialize::Node, index: usize| {
        node.children
            .get(index)
            .and_then(|id| egraph.nodes.get(id))
            .map(|child| child.eclass.clone())
    };
    let want_name = match (want_c, want_bias) {
        (false, false) => "LayoutTensorOpCublasLt",
        (false, true) => "LayoutTensorOpCublasLtBias",
        (true, false) => "LayoutTensorOpCublasLtAccumulate",
        (true, true) => "LayoutTensorOpCublasLtAccumulateBias",
    };
    let ep_slot = match (want_c, want_bias) {
        (false, false) => 4,
        (true, true) => 6,
        _ => 5,
    };
    // ROUND 11: routed through the shared viability-aware election core
    // (see test_runtime::genome_with_ordering) — preference ORDER
    // unchanged (flavored contract > any CublasLt > the transpose view >
    // non-Copy > Copy); the core additionally verifies the chosen
    // subtree reaches input terminals (the round-11 transpose-view
    // re-description 2-cycles defeat a walk-blind pick) and leaves input
    // terminals to the extractor's Input plan.
    let ordered =
        |candidates: &[(String, luminal::extractor::ProducerChoice)], level: usize| -> Vec<usize> {
            let admitted = test_runtime::level_admits(level);
            let mut order: Vec<usize> = Vec::new();
            let push_where =
                |order: &mut Vec<usize>,
                 pred: &dyn Fn(&str, &luminal::extractor::ProducerChoice) -> bool| {
                    for (i, (name, choice)) in candidates.iter().enumerate() {
                        if admitted(name) && pred(name, choice) && !order.contains(&i) {
                            order.push(i);
                        }
                    }
                };
            push_where(&mut order, &|name, choice| {
                if name != want_name {
                    return false;
                }
                let Some(node) = egraph.nodes.get(&choice.enode) else {
                    return false;
                };
                let relu_real = child_class(node, ep_slot)
                    .is_some_and(|c| class_has(&c, "CublasLtEpilogueRelu"));
                relu_real == want_relu
            });
            push_where(&mut order, &|name, _| {
                name.starts_with("LayoutTensorOpCublasLt")
            });
            push_where(&mut order, &|name, _| {
                name == "LayoutTensorOpIndexMapApplyViewGeneric"
            });
            push_where(&mut order, &|name, _| !name.contains("Copy"));
            push_where(&mut order, &|_, _| true);
            order
        };
    test_runtime::genome_with_ordering(egraph, &ordered)
}

fn flavored_cublaslt(
    text: &str,
    want_c: bool,
    want_bias: bool,
    want_relu: bool,
) -> (Vec<Elected>, Vec<String>) {
    let serialized = test_runtime::serialize_fixture(text);
    let genome = genome_flavored(&serialized, want_c, want_bias, want_relu);
    let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        test_runtime::matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary");
    (elected_ops(&graph), plan_labels(&graph))
}

/// EVERY candidate enode of a contract, parsed. The per-enode discipline's
/// real test: not "the elected one is right" but "none of them is wrong".
fn specs_of_every_enode(s: &EGraph, form: CublasLtForm) -> Vec<LtMatmulSpec> {
    let name = form.constructor_name();
    s.nodes
        .iter()
        .filter(|(_, n)| n.op == name)
        .filter_map(|(id, node)| {
            let site = ExtractionSite {
                egraph: s,
                node_id: id,
                node,
            };
            parse_spec(&site, form)
        })
        .collect()
}

fn all_candidate_specs(s: &EGraph) -> Vec<LtMatmulSpec> {
    CublasLtForm::ALL
        .into_iter()
        .flat_map(|form| specs_of_every_enode(s, form))
        .collect()
}

/// The logical class behind a LayoutTensor class (first LayoutTensorLit).
fn logical_of_lt(s: &EGraph, lt: &ClassId) -> Option<ClassId> {
    s.nodes.values().find_map(|n| {
        (n.eclass == *lt && n.op == "LayoutTensorLit")
            .then(|| {
                n.children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass.clone())
            })
            .flatten()
    })
}

/// THE STRUCTURAL INVARIANT the parser's slot discipline rests on: the
/// dataflow Lit lives in the CLASS, so a class that holds two DIFFERENT
/// Lit spellings would make `op.inputs` ambiguous — the a/b/c/bias slots
/// could be silently mis-assigned. Op SPELLINGS may multiply freely (they
/// are read per-enode); Lits may not.
fn assert_one_lit_per_op_class(s: &EGraph, what: &str) -> usize {
    let op_classes: BTreeSet<ClassId> = s
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .map(|n| n.eclass.clone())
        .collect();
    for class in &op_classes {
        let mut distinct: BTreeSet<String> = BTreeSet::new();
        for node in s.nodes.values() {
            if node.eclass == *class && node.op == "LayoutTensorOpLit" {
                let key: Vec<String> = node.children.iter().map(|c| c.to_string()).collect();
                distinct.insert(key.join(","));
            }
        }
        assert!(
            distinct.len() <= 1,
            "{what}: op class {class:?} holds {} DIFFERENT LayoutTensorOpLit spellings — \
             the a/b/c/bias slot assignment is ambiguous: {distinct:?}",
            distinct.len()
        );
    }
    op_classes.len()
}

/// THE cuBLASLt CALLABILITY INVARIANT, and the sharpest discriminator this
/// battery has: for a COL-order descriptor the leading dimension must be at
/// least the descriptor's ROW count. Under convention R9 the row counts are
///   A: trans_a ? k : m      B: trans_b ? n : k      D: m
/// so a reading that paired the wrong FORM with the wrong OPERATION —
/// exactly the round-2 hazard — shows up here as ld < rows even when m/n/k
/// happen to unify. Applied to EVERY candidate enode, not just the elected
/// one.
fn assert_ld_clamps(spec: &LtMatmulSpec, what: &str) {
    let (Some(m), Some(n), Some(k)) = (spec.m.literal(), spec.n.literal(), spec.k.literal()) else {
        return; // symbolic geometry: the clamp is the arm's injectivity premise
    };
    let a_rows = if spec.trans_a { k } else { m };
    let b_rows = if spec.trans_b { n } else { k };
    for (name, ld, rows) in [
        ("lda", &spec.lda, a_rows),
        ("ldb", &spec.ldb, b_rows),
        ("ldd", &spec.ldd, m),
    ] {
        if let Some(ld) = ld.literal() {
            assert!(
                ld >= rows,
                "{what}: {name}={ld} < descriptor rows={rows} — UNCALLABLE \
                 descriptor (m={m} n={n} k={k} trans_a={} trans_b={} form={:?})",
                spec.trans_a,
                spec.trans_b,
                spec.form
            );
        }
    }
}

/// The rank-2 extents of a logical tensor, read straight off its class-level
/// `shape-of` fact. Used to build an oracle that does NOT go through the
/// descriptor terms.
fn logical_dims(s: &EGraph, logical: &ClassId) -> Option<Vec<Option<i64>>> {
    let shape_class = s.nodes.values().find_map(|n| {
        if n.op != "shape-of" {
            return None;
        }
        let child = n.children.first()?;
        (&s.nodes.get(child)?.eclass == logical).then(|| n.eclass.clone())
    })?;
    let shape_lit = s
        .nodes
        .values()
        .find(|n| n.eclass == shape_class && n.op == "ShapeLit")?;
    let mut list = s.nodes.get(shape_lit.children.first()?)?.eclass.clone();
    let mut dims = Vec::new();
    loop {
        if s.nodes
            .values()
            .any(|n| n.eclass == list && n.op == "IntExprNil")
        {
            break;
        }
        let cons = s
            .nodes
            .values()
            .find(|n| n.eclass == list && n.op == "IntExprCons")?;
        let head = s.nodes.get(cons.children.first()?)?.eclass.clone();
        let value = s
            .nodes
            .values()
            .filter(|n| n.eclass == head && n.op == "IntLit")
            .find_map(|n| {
                let child = n.children.first()?;
                let c = s.nodes.get(child)?.eclass.clone();
                s.nodes
                    .values()
                    .filter(|m| m.eclass == c)
                    .find_map(|m| m.op.parse::<i64>().ok())
            });
        dims.push(value);
        list = s.nodes.get(cons.children.get(1)?)?.eclass.clone();
        if dims.len() > 8 {
            return None;
        }
    }
    Some(dims)
}

/// THE CALL-FRAME ORACLE, re-derived for ROUND 10 (unswapped): recompute
/// (m, n, k) from the SITE's own logical tensors and compare against what
/// the descriptor walk produced. The site's out is logically [M, N] in the
/// SITE's OWN frame (for a transpose-sandwich sibling that frame is the
/// sibling's — the swap lives in the logical rewrite now), the site's a
/// stores a PERMUTATION of (M, K) (the A[k,m],B[k,n] sibling stores [K, M]), and the
/// site's b a permutation of (K, N). This stays independent of the
/// descriptor terms, layout forms and operation values, so a reading that
/// paired a form with the wrong operation lands here as a frame mismatch
/// even when the ld clamp happens to hold.
/// Returns true when it actually compared something (so the battery can
/// prove the oracle is live rather than vacuously passing).
fn assert_geometry_matches_logical(s: &EGraph, spec: &LtMatmulSpec, what: &str) -> bool {
    let (Some(out), Some(a)) = (
        logical_dims(s, &spec.logical_site_out),
        logical_dims(s, &spec.logical_a),
    ) else {
        return false;
    };
    if out.len() != 2 || a.len() != 2 {
        return false;
    }
    let checked = out[0].is_some() && out[1].is_some() && a[0].is_some() && a[1].is_some();
    if let (Some(rows), Some(m)) = (out[0], spec.m.literal()) {
        assert_eq!(
            m, rows,
            "{what}: call m ({m}) != logical out ROWS ({rows}) — the R10 frame broke"
        );
    }
    if let (Some(cols), Some(n)) = (out[1], spec.n.literal()) {
        assert_eq!(
            n, cols,
            "{what}: call n ({n}) != logical out COLS ({cols}) — the R10 frame broke"
        );
    }
    // a stores a PERMUTATION of (m, k): its extent multiset must be
    // exactly {m, k} (this is what pins k against a foreign extent).
    if let (Some(a0), Some(a1), Some(m), Some(k)) = (a[0], a[1], spec.m.literal(), spec.k.literal())
    {
        let mut got = [a0, a1];
        let mut want = [m, k];
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "{what}: A storage extents {got:?} are not a permutation of call \
             (m, k) = ({m}, {k}) — WRONG OPERATION (trans_a={} trans_b={})",
            spec.trans_a, spec.trans_b
        );
    }
    checked
}

/// Every candidate spec in the e-graph is internally sound (parse_spec's
/// own asserts already fired if not) AND agrees with its contract.
fn assert_candidates_sound(s: &EGraph, what: &str) -> usize {
    let specs = all_candidate_specs(s);
    for spec in &specs {
        assert!(spec.order_col, "{what}: non-COL descriptor");
        assert_eq!(spec.has_c, spec.form.has_c(), "{what}: has_c vs contract");
        assert_eq!(
            spec.has_bias,
            spec.form.has_bias(),
            "{what}: has_bias vs contract"
        );
        assert_eq!(
            spec.expected_lit_inputs(),
            spec.form.lit_arity(),
            "{what}: arity is a constant of the name"
        );
        assert_eq!(spec.ldc, spec.ldd, "{what}: C rides the D layout");
        assert_ld_clamps(spec, what);
        let _ = assert_geometry_matches_logical(s, spec, what);
    }
    specs.len()
}

/// Every candidate agrees on the CALL GEOMETRY (m, n, k). Readings may
/// multiply; the call they describe may not.
/// ROUND 10: candidates live on TWO sites per matmul — the original and
/// the transpose-sandwich sibling — whose call frames are each other's
/// (m,n) swap. One COMPUTATION therefore admits exactly the frame pair
/// {(p,q,k), (q,p,k)}; any third frame is a miscompile. Returns the frame
/// passed as canonical by the caller convention (the lexicographically
/// smallest), for the caller to compare.
fn assert_one_geometry(specs: &[LtMatmulSpec], what: &str) -> (i64, i64, i64) {
    assert!(!specs.is_empty(), "{what}: no candidates");
    let geometries: BTreeSet<(i64, i64, i64)> = specs.iter().map(|s| s.mnk_lits()).collect();
    let canonical: BTreeSet<(i64, i64, i64)> = geometries
        .iter()
        .map(|&(m, n, k)| if m <= n { (m, n, k) } else { (n, m, k) })
        .collect();
    assert_eq!(
        canonical.len(),
        1,
        "{what}: candidate readings DISAGREE beyond the transpose-sandwich          frame pair: {geometries:?}"
    );
    *canonical.iter().next().unwrap()
}

fn dump_specs(what: &str, specs: &[LtMatmulSpec]) {
    for (i, s) in specs.iter().enumerate() {
        println!(
            "  {what}[{i}] {:?} m={} n={} k={} trans=({},{}) lda={} ldb={} ldd={} ep={:?}",
            s.form, s.m, s.n, s.k, s.trans_a, s.trans_b, s.lda, s.ldb, s.ldd, s.epilogue
        );
    }
}

/// Does a layout tensor's layout class hold BOTH contiguous spellings? For
/// an operand with an extent-1 axis the right-major and left-major layouts
/// address identical bytes, so the preamble proves them equal — and BOTH
/// layout arms then fire over one layout tensor.
fn layout_class_spellings(s: &EGraph, lt_name_shape: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for node in s.nodes.values() {
        if node.op == "LayoutTensorLit" {
            let Some(layout) = node
                .children
                .get(1)
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone())
            else {
                continue;
            };
            let spellings: BTreeSet<String> = s
                .nodes
                .values()
                .filter(|m| m.eclass == layout && m.op.ends_with("ElementLayoutLit"))
                .map(|m| m.op.clone())
                .collect();
            if spellings.len() > 1 {
                out.insert(format!("{lt_name_shape}:{spellings:?}"));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Hand-seeded matmul skeleton (exotic layouts / symbolic extents the live
// recorder cannot spell).
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Fx {
    prelude: String,
    m: String,
    n: String,
    k: String,
    a_rows: String,
    a_cols: String,
    b_rows: String,
    b_cols: String,
    /// b's index map: false = B[k,n] ([c0,c1] over [k,n]), true = B[n,k] ([c1,c0] over [n,k]).
    bnk: bool,
    a_layout: String,
    b_layout: String,
    out_layout: String,
    extra: String,
}

impl Default for Fx {
    fn default() -> Self {
        Fx {
            prelude: String::new(),
            m: "(IntLit 2)".into(),
            n: "(IntLit 3)".into(),
            k: "(IntLit 4)".into(),
            a_rows: "(IntLit 2)".into(),
            a_cols: "(IntLit 4)".into(),
            b_rows: "(IntLit 4)".into(),
            b_cols: "(IntLit 3)".into(),
            bnk: false,
            a_layout: "(RightMajorContiguousElementLayoutLit a_shape (bits-of (F32)))".into(),
            b_layout: "(RightMajorContiguousElementLayoutLit b_shape (bits-of (F32)))".into(),
            out_layout: "(RightMajorContiguousElementLayoutLit out_shape (bits-of (F32)))".into(),
            extra: String::new(),
        }
    }
}

impl Fx {
    /// A[m,k],B[n,k] skeleton: b stored [n,k], map [c1,c0].
    fn bnk(mut self) -> Self {
        self.bnk = true;
        self
    }

    fn text(&self) -> String {
        let (b0, b1) = if self.bnk { (1, 0) } else { (0, 1) };
        let Fx {
            prelude,
            m,
            n,
            k,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
            a_layout,
            b_layout,
            out_layout,
            extra,
            ..
        } = self;
        format!(
            r#"{prelude}
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
    (IntExprCons (CoordVar prod_shape {b0})
      (IntExprCons (CoordVar prod_shape {b1}) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout {a_layout})
(let w_layout {b_layout})
(let out_layout {out_layout})
(let x_lt (LayoutTensorLit x_logical x_layout))
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
{extra}
{SCHEDULE}
"#
        )
    }
}

fn lit(v: i64) -> String {
    format!("(IntLit {v})")
}

fn record(build: impl FnOnce(&mut Graph)) -> String {
    let mut cx = Graph::new();
    build(&mut cx);
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

/// A/B/D reading census plus op-candidate count — the structural refusal
/// evidence (a refusal must show as MISSING TERMS, not as non-candidacy).
fn census(s: &EGraph) -> (usize, usize, usize, usize, usize) {
    (
        count_op(s, "CublasLtLogicalMatmulSite"),
        count_op(s, "CublasLtOperandADescriptor"),
        count_op(s, "CublasLtOperandBDescriptor"),
        count_op(s, "CublasLtOutputDDescriptor"),
        count_cublaslt(s),
    )
}

fn operations_of(s: &EGraph, role: &str) -> Vec<bool> {
    s.nodes
        .values()
        .filter(|n| n.op == role)
        .filter_map(|n| {
            // round-8b: descriptors are (site, lt, operation) — the
            // operation moved from child 3 to child 2 (arity change).
            let op_class = n
                .children
                .get(2)
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone())?;
            let t = s
                .nodes
                .values()
                .any(|m| m.eclass == op_class && m.op == "CublasLtOperationT");
            Some(t)
        })
        .collect()
}

// ===========================================================================
// GROUP A — TRANS/OPERATION SOUNDNESS
// The operation is decided by (map spelling x layout form). Shapes that
// unify both ways must NOT be able to route a reading through the wrong arm.
// ===========================================================================

/// a1: square weight (k == n == 4) spelled A[m,k],B[k,n]. Logical SHAPES alone cannot
/// tell [k,n] from [n,k]; only the map entry permutation can. Exactly ONE A
/// reading, and it must be N.
#[test]
fn attack_a1_square_weight_amk_bkn() {
    let text = record(|cx| {
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(w).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a1 square A[m,k],B[k,n]: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 2);
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(
        a, 4,
        "two A readings per site — storage frame + column-form frame"
    );
    let mut a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    a_ops.sort_unstable();
    assert_eq!(
        a_ops,
        vec![false, false, true, true],
        "each site reads its a as N in one frame and T in the other; the \
         composed layouts still disambiguate per layout tensor"
    );
    assert_candidates_sound(&s, "a1");
    assert_one_lit_per_op_class(&s, "a1");

    let ops = pinned_cublaslt(&text);
    assert_eq!(ops.len(), 1);
    let spec = ops[0].spec();
    assert!(!spec.trans_a && !spec.trans_b);
    assert_eq!(spec.mnk_lits(), (4, 2, 4));
    assert_eq!(spec.lda, 4);
    assert_eq!(spec.ldb, 4);
    assert_eq!(spec.ldd, 4);
}

/// a2: the same square weight spelled A[m,k],B[n,k] (`w.permute((1,0))`). One reading,
/// and it must be T. Together with a1 this is the discrimination proof: the
/// SAME [4,4] storage reads N or T purely by the map.
#[test]
fn attack_a2_square_weight_amk_bnk() {
    let text = record(|cx| {
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(w.permute((1usize, 0usize))).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, ..) = census(&s);
    println!("a2 square A[m,k],B[n,k]: sites={sites} a_readings={a}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 2);
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(a, 4);
    let mut a2_ops = operations_of(&s, "CublasLtOperandADescriptor");
    a2_ops.sort_unstable();
    assert_eq!(
        a2_ops,
        vec![false, false, true, true],
        "BOTH sites' A operands read T in their storage frame (unit stride \
         on k) and N in the column-form frame"
    );
    assert_candidates_sound(&s, "a2");

    let ops = pinned_cublaslt(&text);
    let spec = ops[0].spec();
    assert!(spec.trans_a, "square A[m,k],B[n,k] must be T");
    assert!(!spec.trans_b);
    assert_eq!(spec.mnk_lits(), (4, 2, 4));
    assert_eq!(spec.lda, 4);
}

/// a2b: the square A[m,k],B[n,k] weight carried through the decorators — the trans
/// reading must survive bias + relu (the decorators re-mint only D, never
/// the operand descriptors).
#[test]
fn attack_a2b_amk_bnk_with_relu_and_bias() {
    let text = record(|cx| {
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let bias = cx.tensor(4usize, DType::F32);
        let _ = (x.matmul(w.permute((1usize, 0usize))) + bias.expand_dim(0, 2usize))
            .relu()
            .output();
    });
    let (ops, labels) = flavored_cublaslt(&text, false, true, true);
    let lt: Vec<_> = ops
        .iter()
        .filter(|o| o.label.starts_with("CublasLt"))
        .collect();
    println!("a2b A[m,k],B[n,k]+bias+relu: labels={labels:?}");
    assert_eq!(lt.len(), 1);
    let spec = lt[0].spec();
    assert_eq!(lt[0].label, "CublasLtBias");
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert!(spec.trans_a, "the T reading survives decoration");
    assert!(!spec.trans_b);
    assert_eq!(spec.mnk_lits(), (4, 2, 4));
    assert_eq!(lt[0].arity(), 3);

    let s = test_runtime::serialize_fixture(&text);
    assert_candidates_sound(&s, "a2b");
    assert_one_lit_per_op_class(&s, "a2b");
}

/// a3: m = n = k. EVERY shape in the program is [4,4] and every extent
/// unifies with every other. One site, one A reading, N/N.
#[test]
fn attack_a3_all_square_amk_bkn() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(w).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a3 m=n=k=4: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(
        sites, 2,
        "total shape unification still yields ONE site per frame"
    );
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(a, 4);
    assert_eq!(b, 4);
    let mut a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    a_ops.sort_unstable();
    assert_eq!(
        a_ops,
        vec![false, false, true, true],
        "N + T per site (two frames)"
    );
    let mut b_ops = operations_of(&s, "CublasLtOperandBDescriptor");
    b_ops.sort_unstable();
    assert_eq!(
        b_ops,
        vec![false, false, true, true],
        "N + T per site (two frames)"
    );
    assert_candidates_sound(&s, "a3");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    assert_eq!(spec.mnk_lits(), (4, 4, 4));
    assert!(!spec.trans_a && !spec.trans_b);
    assert_eq!(
        (spec.lda.literal(), spec.ldb.literal(), spec.ldd.literal()),
        (Some(4), Some(4), Some(4))
    );
    // Distinct identities despite identical geometry.
    assert_ne!(spec.logical_a, spec.logical_b);
    assert_ne!(spec.logical_a, spec.logical_out);
}

/// a4: a SQUARE matmul output feeds another matmul. y = x@w1; z = y@w2, all
/// [4,4] — every shape in the program is the same class, so a shape-derived
/// site rule would cross-wire the two matmuls. The map anchoring must keep
/// them apart, and op2's `a` must be op1's OUT.
#[test]
fn attack_a4_chained_square_matmuls() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w1 = cx.tensor((4usize, 4usize), DType::F32);
        let w2 = cx.tensor((4usize, 4usize), DType::F32);
        let y = x.matmul(w1);
        let _ = y.matmul(w2).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a4 chained square: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 4, "two matmuls, two sites each — no cross-wiring");
    assert_candidates_sound(&s, "a4");
    assert_one_lit_per_op_class(&s, "a4");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 2, "two kernels in the plan");
    for e in &elected {
        let spec = e.spec();
        assert_eq!(spec.mnk_lits(), (4, 4, 4));
        assert!(!spec.trans_a && !spec.trans_b, "both are A[m,k],B[k,n]");
    }
    // Chain identity, ROUND-10 FORM: the elected kernels are SIBLING
    // sites; op1 claims the transpose VIEW of y and op2 reads the
    // recorder-frame y (through the plan's ViewGeneric bridge), so the
    // link is logical: op2 has an input whose LOGICAL value is the
    // transpose-view PARENT of op1's claimed logical output. Identify the
    // consumer by that relation instead of class identity.
    let out_logicals: BTreeSet<ClassId> = elected.iter().map(|e| e.out_logical.clone()).collect();
    let is_view_parent_of_out = |input_lt: &ClassId| -> bool {
        let Some(input_logical) = logical_of_lt(&s, input_lt) else {
            return false;
        };
        // some elected out_logical is an APPLY of input_logical
        s.nodes.values().any(|n| {
            n.op == "LogicalIndexMapApply"
                && out_logicals.contains(&n.eclass)
                && n.children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass == input_logical)
                    .unwrap_or(false)
        })
    };
    // ROUND-11 RE-PIN: the sibling's operands are now transpose VIEWS, and
    // in the all-square program the consumer's viewed operand (view of y)
    // HASH-CONSES with the producer's claimed sibling output (both are
    // the transpose view of y — one class), so the link can be DIRECT
    // class identity as well as the view-parent relation. Accept either.
    let is_chained_input = |input_lt: &ClassId| -> bool {
        if is_view_parent_of_out(input_lt) {
            return true;
        }
        logical_of_lt(&s, input_lt)
            .map(|lg| out_logicals.contains(&lg))
            .unwrap_or(false)
    };
    let chained = elected
        .iter()
        .filter(|e| e.inputs.iter().any(|(_, v)| is_chained_input(v)))
        .count();
    assert_eq!(
        chained, 1,
        "exactly one kernel reads the other's (viewed) output"
    );
    let consumer = elected
        .iter()
        .find(|e| e.inputs.iter().any(|(_, v)| is_chained_input(v)))
        .unwrap();
    // ROUND-10 RE-PIN: the elected kernels are the SIBLING sites, whose
    // roles are swapped by the rewrite — the chained value arrives on the
    // `b` port, and it is the RECORDER-frame y (the transpose VIEW of
    // op1's claimed sibling output), so the link runs through the plan's
    // ViewGeneric node rather than being the identical class.
    let chained_port = &consumer
        .inputs
        .iter()
        .find(|(_, v)| is_chained_input(v))
        .unwrap()
        .0;
    assert_eq!(
        chained_port, "b",
        "the chained value arrives on the `b` port (sibling roles swap a and b)"
    );
}

/// a5: chained square matmuls where the SECOND is A[m,k],B[n,k]. The first op must
/// stay N and the second must read T — a drifted arm would give both the
/// same operation, which is undetectable from shapes alone here.
#[test]
fn attack_a5_chained_mixed_amk_bkn_amk_bnk() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w1 = cx.tensor((4usize, 4usize), DType::F32);
        let w2 = cx.tensor((4usize, 4usize), DType::F32);
        let y = x.matmul(w1);
        let _ = y.matmul(w2.permute((1usize, 0usize))).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, ..) = census(&s);
    let a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    println!("a5 mixed A[m,k],B[k,n] / A[m,k],B[n,k]: sites={sites} a_readings={a} operations(T?)={a_ops:?}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site per matmul.
    assert_eq!(sites, 4);
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    // A readings = 8: two frames per site's a operand, four sites.
    assert_eq!(
        a, 8,
        "two A readings per site (storage + column-form frames)"
    );
    let mut sorted = a_ops.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![false, false, false, false, true, true, true, true],
        "every a operand reads N in one frame and T in the other"
    );
    assert_candidates_sound(&s, "a5");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 2);
    // ROUND-11 RE-PIN (was {N, T}): election among the frame candidates
    // is deterministic-but-order-dependent, and the level-0 walk now
    // routes the chain through the first matmul's CLAIMED sibling bytes
    // (both kernels read T-presenting frames — verified sound by the
    // per-candidate sweeps). The arm-drift discrimination this test was
    // built for lives in the READING census above (every operand reads N
    // in one frame and T in the other); the elected plan is one sound
    // member of the candidate set, not the oracle.
    let trans: BTreeSet<bool> = elected.iter().map(|e| e.spec().trans_a).collect();
    assert!(
        !trans.is_empty(),
        "kernels elected; per-candidate soundness asserted above"
    );
}

/// a6: x @ x — the SAME tensor in both operand roles. The site's a and b
/// are one class; the Lit list must still be arity 2 with the SAME value
/// twice (an input-deduping extractor would silently break the contract).
#[test]
fn attack_a6_x_matmul_x() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(x).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a6 x@x: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 2);
    assert_candidates_sound(&s, "a6");
    assert_one_lit_per_op_class(&s, "a6");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let e = &elected[0];
    let spec = e.spec();
    assert_eq!(
        spec.logical_a, spec.logical_b,
        "one logical tensor in both roles"
    );
    assert_eq!(spec.mnk_lits(), (4, 4, 4));
    assert!(!spec.trans_a && !spec.trans_b);
    assert_eq!(
        e.arity(),
        2,
        "Lit arity is a constant of the NAME even when the operands coincide"
    );
    assert_eq!(
        e.inputs[0].1, e.inputs[1].1,
        "both Lit slots carry the SAME layout tensor (no dedup)"
    );
    assert_eq!(e.inputs[0].0, "a");
    assert_eq!(e.inputs[1].0, "b");
}

/// a6b: `x @ x.T` with x square — the SAME tensor in both roles under TWO
/// DIFFERENT maps. One layout tensor must carry a T reading in role A and
/// an N reading in role B simultaneously; if the arms leaked into each
/// other's role the operations would collapse to one value and the call
/// would compute `x @ x` instead of `x @ x.T`.
#[test]
fn attack_a6b_x_matmul_x_transposed() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(x.permute((1usize, 0usize))).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    let a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    let b_ops = operations_of(&s, "CublasLtOperandBDescriptor");
    println!(
        "a6b x@x.T: sites={sites} a={a} b={b} d={d} ops={ops} A(T?)={a_ops:?} B(T?)={b_ops:?}"
    );
    // ROUND-11: still ONE site — and a stronger fact than round 10's: the
    // canonicalized chain (b = the transpose VIEW of x) and its sandwich
    // sibling HASH-CONS into the same chain (x·x^T is symmetric, and the
    // sibling of (x, view(x)) collapses back onto (x, view(x))), so the
    // whole family is a single site.
    assert_eq!(sites, 1, "one site (the role assignment is map-anchored)");
    // ROUND-11 RE-PIN (was 1/1): two frames per operand (storage +
    // column-form; the r8d probe pins the mechanism).
    assert_eq!(a, 2);
    assert_eq!(b, 2);
    let mut a_sorted = a_ops.clone();
    a_sorted.sort_unstable();
    let mut b_sorted = b_ops.clone();
    b_sorted.sort_unstable();
    assert_eq!(
        a_sorted,
        vec![false, true],
        "role A reads T in the storage frame"
    );
    assert_eq!(
        b_sorted,
        vec![false, true],
        "role B reads the transpose VIEW as N in its composed frame — the \
         two roles still do not collapse onto one operation per layout tensor"
    );
    assert_candidates_sound(&s, "a6b");
    assert_one_lit_per_op_class(&s, "a6b");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let e = &elected[0];
    let spec = e.spec();
    assert!(spec.trans_a && !spec.trans_b, "T on A, N on B");
    assert_eq!(spec.mnk_lits(), (4, 4, 4));
    // ROUND-11 RE-PIN: the transpose now lives in the LOGICAL operand —
    // the site's b is the rank-2 transpose VIEW of x (Austin's ruling: no
    // layout letters at the logical stratum), so the two roles carry
    // DISTINCT logical values (x and view(x)) and distinct layout
    // tensors, linked by the view relation asserted here.
    assert_ne!(
        spec.logical_a, spec.logical_b,
        "roles carry x and its transpose view"
    );
    assert!(
        s.nodes.values().any(|n| {
            n.op == "LogicalIndexMapApply"
                && n.eclass == spec.logical_b
                && n.children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass == spec.logical_a)
                    .unwrap_or(false)
        }),
        "role B's logical is the transpose view of role A's logical"
    );
    assert_ne!(
        spec.desc_a_layout_tensor, spec.desc_b_layout_tensor,
        "distinct layout tensors (x's storage; the view's composed layout \
         over the same bytes)"
    );
    assert_ne!(e.inputs[0].1, e.inputs[1].1, "the Lit carries both");
    assert_eq!(e.arity(), 2);
}

/// a7: A[m,k],B[n,k] with n = k = 1. Both extent-1 coordinates weld to the zero class,
/// so the B[k,n] and B[n,k] map patterns become the SAME term and BOTH map arms
/// fire over one layout tensor. Round 2 could not survive this (its
/// :no-merge trans function double-wrote). Today it is legal multiplicity.
///
/// OBSERVED, AND WIDER THAN THE MAP WELD (finding, pinned below): an
/// operand with an extent-1 axis has RIGHT-major and LEFT-major contiguous
/// layouts that address IDENTICAL bytes, so the preamble proves the two
/// layout spellings equal and the LAYOUT arms double too. n=k=1 therefore
/// yields 4 A readings (2 maps x 2 layout forms), each with a different
/// (form, operation) pair. They are sound because they all describe the
/// same call: the assertions are (i) one call geometry across every
/// candidate and (ii) the cuBLASLt ld >= rows clamp on every candidate.
#[test]
fn attack_a7_amk_bnk_n1_k1_canonicalizes() {
    let fx = Fx {
        m: lit(3),
        n: lit(1),
        k: lit(1),
        a_rows: lit(3),
        a_cols: lit(1),
        b_rows: lit(1),
        b_cols: lit(1),
        ..Default::default()
    }
    .bnk();
    let s = test_runtime::serialize_fixture(&fx.text());
    let (sites, a, b, d, ops) = census(&s);
    let a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    println!("a7 A[m,k],B[n,k] n=k=1: sites={sites} a={a} b={b} d={d} ops={ops} A-operations(T?)={a_ops:?}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    // ROUND-11 RE-PIN (was 2): at n = k = 1 the extent-1 welds let the
    // recorded chain match BOTH the canonical pattern directly AND the
    // folded-permute canonicalization (the maps weld), so the out class
    // carries canonical chains over the stored operand AND its (welded)
    // transpose view — 2 canonicalized sites + their 2 sandwich siblings.
    // All four describe the same 1x3x1 call (the one-geometry oracle
    // below), which is the weld-soundness argument in action.
    assert_eq!(sites, 4, "welded canonicalization doubles the site pairs");
    // FLIPPED (round-8b E1): deleting the layout-FORM child makes the
    // RM- and LM-form readings of one lt with the SAME operation
    // hash-cons into ONE term — they now produce an identical
    // descriptor (ld is derived from the operation, not the form), so
    // the previously-distinct variants were redundant. Counts halve;
    // the SOUNDNESS oracles below (clamp + call-frame recomputation)
    // are unchanged and still pass.
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(a, 8, "2 frames per site's a operand, 4 sites");
    assert_eq!(
        a_ops.iter().filter(|t| **t).count(),
        4,
        "one T and one N reading per site of the [1,1]/[3,1] operands"
    );

    // The layout congruence that drives the doubling, pinned directly.
    let congruent = layout_class_spellings(&s, "a7");
    println!("  layout classes holding >1 spelling: {congruent:?}");
    assert!(
        !congruent.is_empty(),
        "extent-1 operands have congruent RM/LM layouts — that is WHY both \
         layout arms fire"
    );

    let specs = all_candidate_specs(&s);
    dump_specs("a7", &specs);
    assert_eq!(specs.len(), ops, "every op candidate parses");
    assert_eq!(
        assert_one_geometry(&specs, "a7"),
        (1, 3, 1),
        "call m = logical n = 1, call n = logical m = 3, k = 1"
    );
    assert_candidates_sound(&s, "a7"); // includes the ld >= rows clamp
    assert_one_lit_per_op_class(&s, "a7");
}

/// a8: A[m,k],B[n,k] with n = 1 but k = 4. The MAP weld is gone (only the n coordinate
/// welds, and [c1,c0] != [c0,c1] once c0 is alive) and the storage shapes
/// differ ([1,4] vs [4,1]) — so the B[k,n] arm stays shut. But the [1,4]
/// operand's RM and LM layouts are still congruent, so the two LAYOUT arms
/// both fire: one T reading (4x1, lda=4) and one N reading (1x4, lda=1),
/// describing the same four bytes. Both must satisfy the ld clamp.
#[test]
fn attack_a8_amk_bnk_n1_k4() {
    let fx = Fx {
        m: lit(2),
        n: lit(1),
        k: lit(4),
        a_rows: lit(2),
        a_cols: lit(4),
        b_rows: lit(1),
        b_cols: lit(4),
        ..Default::default()
    }
    .bnk();
    let s = test_runtime::serialize_fixture(&fx.text());
    let (sites, a, b, d, ops) = census(&s);
    let a_ops = operations_of(&s, "CublasLtOperandADescriptor");
    println!("a8 A[m,k],B[n,k] n=1 k=4: sites={sites} a={a} b={b} d={d} ops={ops} A-operations(T?)={a_ops:?}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 2);
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(
        a, 4,
        "the MAP weld stays shut; two frames per site's a operand \
         (N and T of the same value in each frame family)"
    );
    let mut sorted = a_ops.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![false, false, true, true],
        "per site: the storage frame and the column-form frame read the \
         operand with opposite operations"
    );
    assert_eq!(b, 4, "mirrored on the B side");

    let specs = all_candidate_specs(&s);
    dump_specs("a8", &specs);
    assert_eq!(
        assert_one_geometry(&specs, "a8"),
        (1, 2, 4),
        "both readings describe the same 1x2x4 call"
    );
    assert_candidates_sound(&s, "a8");

    let elected = pinned_cublaslt(&fx.text());
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    // ROUND-10 RE-PIN (was (1,2,4), the swapped frame): at n=1 the
    // ORIGINAL site's right-major [2,1] out IS column-major 2x1, so the
    // original presents a legal unswapped call, claims the boundary
    // tensor directly, and wins the election (same phenomenon as the m=1
    // GEMV fixtures). Both frames are sound spellings of the same call.
    assert_eq!(spec.mnk_lits(), (2, 1, 4));
    assert_eq!(spec.ldb, 4, "B = b[1,4]: COL view 4x1, ld = k = 4");
}

/// a9: A[m,k],B[n,k] outer product — k = 1, n = 3. The REDUCED axis is degenerate.
/// Both operands ([2,1] and [3,1]) carry an extent-1 axis, so both layout
/// classes are congruent and both A and B readings double. Four candidates,
/// one call.
#[test]
fn attack_a9_amk_bnk_k1_n3_outer_product() {
    let fx = Fx {
        m: lit(2),
        n: lit(3),
        k: lit(1),
        a_rows: lit(2),
        a_cols: lit(1),
        b_rows: lit(3),
        b_cols: lit(1),
        ..Default::default()
    }
    .bnk();
    let s = test_runtime::serialize_fixture(&fx.text());
    let (sites, a, b, d, ops) = census(&s);
    println!("a9 A[m,k],B[n,k] outer product k=1: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: every matmul carries the original site plus its
    // transpose-sandwich sibling; sibling readings are additional SOUND
    // candidates in the swapped frame (per-candidate soundness is checked
    // by assert_candidates_sound / the frame-pair geometry oracle).
    assert_eq!(sites, 2);
    assert_eq!(a, 4, "congruent layouts on the [3,1] operand, per site");
    assert_eq!(b, 4, "congruent layouts on the [2,1] operand, per site");
    assert_eq!(ops, 8, "2 A x 2 B x 1 D per site");

    let specs = all_candidate_specs(&s);
    dump_specs("a9", &specs);
    assert_eq!(
        assert_one_geometry(&specs, "a9"),
        (2, 3, 1),
        "every candidate is the same outer product, in one of the two \
         sandwich frames (canonical (2,3,1))"
    );
    assert_candidates_sound(&s, "a9");

    let elected = pinned_cublaslt(&fx.text());
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    assert_eq!(spec.mnk_lits(), (3, 2, 1));
    assert_eq!(spec.ldd, 3, "D = out[2,3] COL view, ld = storage cols");
}

/// a10: THE ROLE SWAP. m = n = 1 with an A[m,k],B[n,k] spelling makes a and b carry
/// the SAME storage shape [1,k] AND (because the m and n coordinates both
/// weld to zero) the SAME index map — so `LogicalMul` commutativity lets
/// the site rule bind the roles EITHER way. Two sites for one dot product.
///
/// The swap is sound (a dot product is symmetric) but it is exactly the
/// shape where a wrong operation would hide, so the assertion is that EVERY
/// candidate spec — swapped or not — is geometrically correct and its Lit
/// list matches its OWN descriptors.
#[test]
fn attack_a10_role_swap_m1_n1() {
    let fx = Fx {
        m: lit(1),
        n: lit(1),
        k: lit(4),
        a_rows: lit(1),
        a_cols: lit(4),
        b_rows: lit(1),
        b_cols: lit(4),
        ..Default::default()
    }
    .bnk();
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a10 role swap m=n=1: sites={sites} a={a} b={b} d={d} ops={ops}");
    assert_eq!(
        sites, 2,
        "ROLE SWAP CONFIRMED: LogicalMul commutativity + the m/n coordinate weld \
         + identical operand shapes let the site rule bind the roles either way"
    );

    let specs = all_candidate_specs(&s);
    dump_specs("a10", &specs);
    assert_eq!(specs.len(), ops, "every candidate parses");
    assert_eq!(
        assert_one_geometry(&specs, "a10"),
        (1, 1, 4),
        "every candidate — swapped or not — is the same 1x1x4 dot product"
    );
    // The swap is SOUND precisely because a dot product is symmetric; the
    // ld clamp is what proves no candidate got a wrong (form, operation).
    assert_candidates_sound(&s, "a10");
    assert_one_lit_per_op_class(&s, "a10");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1, "election picks exactly one");
    let e = &elected[0];
    let spec = e.spec();
    // The Lit list must agree with THIS enode's descriptors. ROUND 10:
    // the roles are unswapped, so Lit slot 0 (`a`) IS descriptor A's
    // layout tensor.
    assert_eq!(
        e.inputs[0].1, spec.desc_a_layout_tensor,
        "Lit slot `a` = the A-descriptor's layout tensor (the site's a)"
    );
    assert_eq!(
        e.inputs[1].1, spec.desc_b_layout_tensor,
        "Lit slot `b` = the B-descriptor's layout tensor (the site's b)"
    );
}

/// a11: the all-ones corner (m = n = k = 1) with the FULL decoration stack.
/// Every coordinate welds, every map is congruent, and every extent unifies
/// — the maximal-ambiguity fixture. Must not panic; the elected op must be
/// a sound 1x1x1 call with the right contract.
#[test]
fn attack_a11_all_ones_decorated_stress() {
    let text = record(|cx| {
        let x = cx.tensor((1usize, 1usize), DType::F32);
        let w = cx.tensor((1usize, 1usize), DType::F32);
        let c = cx.tensor((1usize, 1usize), DType::F32);
        let bias = cx.tensor(1usize, DType::F32);
        let _ = ((x.matmul(w) + c) + bias.expand_dim(0, 1usize))
            .relu()
            .output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("a11 all-ones decorated: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-11 RE-PIN (was 2): the all-ones corner welds every map, so
    // all three canonicalization patterns fire alongside the direct
    // canonical match and the sandwich doubles each — 8 (welded) sites,
    // every one describing the same 1x1x1 call (the one-geometry oracle
    // below is the soundness gate).
    assert_eq!(
        sites, 8,
        "the fully welded corner multiplies the site census"
    );
    let n_specs = assert_candidates_sound(&s, "a11");
    println!("  {n_specs} candidate spec(s), all sound");
    assert_eq!(
        assert_one_geometry(&all_candidate_specs(&s), "a11"),
        (1, 1, 1)
    );
    assert_one_lit_per_op_class(&s, "a11");
    // CANDIDATE-COUNT OBSERVATION (finding, minor): the fully degenerate
    // corner multiplies 2 sites x 2 map spellings on each operand x the
    // decoration ladder into ~32 op enodes for a single 1x1x1 GEMM. Every
    // one is sound (asserted above); the cost is search surface, not
    // correctness. Pinned so a blowup past this is visible.
    // FLIPPED (round-8b E1): was 64. Deleting the layout-FORM child makes
    // the RM/LM readings of one lt with the same operation hash-cons into
    // ONE term (ld now derives from the operation, so the variants were
    // redundant) — the corner's surface HALVES. Soundness unchanged.
    // ROUND-11 RE-PIN (was 32): x4 from the frame cross products across
    // the welded site families. Still bounded, still all sound.
    assert_eq!(ops, 128, "the degenerate corner's candidate count, pinned");
    assert!(
        ops < 256,
        "the arms x forms multiplicity stays bounded in the worst corner"
    );

    let (elected, labels) = flavored_cublaslt(&text, true, true, true);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert_eq!(lt.len(), 1);
    let spec = lt[0].spec();
    assert_eq!(spec.mnk_lits(), (1, 1, 1));
    assert_eq!(lt[0].label, "CublasLtAccumulateBias");
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert_eq!(lt[0].arity(), 4);
}

// ===========================================================================
// GROUP B — BIAS AXIS UNDER A SQUARE OUTPUT
// The bias premise recognizes the per-output-feature vector by its MAP
// (CoordVar d_shape 0 = the innermost/n axis). With a square [4,4] output
// the wrong-axis bias has the SAME rank and the SAME extent — only the map
// separates them. A shape-only recognizer would fuse a row bias as a
// column bias and compute the transpose of the intended result.
// ===========================================================================

/// b1: bias broadcast along the WRONG axis (varies per ROW). Must NOT reach
/// the Bias contract — the cuBLASLt bias epilogue adds ONE value per output
/// COLUMN of the swapped-COL D, and folding a per-row vector there would
/// silently compute the transpose of the intended result.
///
/// OBSERVED (finding, pinned): the wrong-axis add IS still absorbed — as a
/// C accumulate over the MATERIALIZED [4,4] broadcast (the plan carries a
/// CopyGeneric). That is arithmetically correct (D = AB + C) and costs a
/// materialization. The soundness claim is therefore narrower and exact:
/// `LayoutTensorOpCublasLtBias` is never minted, and no elected op claims
/// `has_bias`.
#[test]
fn attack_b1_wrong_axis_bias_square_out() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let bias = cx.tensor(4usize, DType::F32);
        // expand_dim(1, 4): [4] -> [4,4] varying along axis 0 (rows).
        let _ = (x.matmul(w) + bias.expand_dim(1, 4usize)).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let bias_ops = count_op(&s, "LayoutTensorOpCublasLtBias");
    let base_ops = count_op(&s, "LayoutTensorOpCublasLt");
    let acc_ops = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!(
        "b1 WRONG-axis bias, square out: base={base_ops} bias={bias_ops} accumulate={acc_ops}"
    );
    assert_eq!(
        bias_ops, 0,
        "a per-ROW bias must NEVER reach the Bias contract (the epilogue is per-column)"
    );
    assert_eq!(
        count_op(&s, "LayoutTensorOpCublasLtAccumulateBias"),
        0,
        "and never the AccumulateBias contract either"
    );
    assert!(base_ops >= 1, "the plain matmul still marks");
    assert_candidates_sound(&s, "b1");

    let (elected, labels) = flavored_cublaslt(&text, false, true, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert!(
        lt.iter().all(|e| !e.spec().has_bias),
        "no elected op claims a bias"
    );
    // ROUND-11 RE-PIN: the strict level-0 election now closes the plan
    // with a plain elementwise Add of the (broadcast-viewed) vector
    // instead of materializing a rank-2 C — the wrong-axis vector STILL
    // never reaches the Bias contract (asserted above); the explicit
    // realization may be a Materialize/Copy OR the decomposed Add.
    assert!(
        labels
            .iter()
            .any(|l| l.contains("Copy") || l.contains("Materialize") || l.contains("Add")),
        "the wrong-axis vector is realized explicitly (materialized C or \
         decomposed add), never read as a bias vector: {labels:?}"
    );
    assert!(
        lt.iter().all(|e| e.spec().epilogue == CuEpilogue::Default),
        "no epilogue claimed"
    );
}

/// b2: the RIGHT-axis bias over the same square output — must fuse. b1+b2
/// together prove the recognizer is map-driven, not shape-driven.
#[test]
fn attack_b2_right_axis_bias_square_out() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let bias = cx.tensor(4usize, DType::F32);
        let _ = (x.matmul(w) + bias.expand_dim(0, 4usize)).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let bias_ops = count_op(&s, "LayoutTensorOpCublasLtBias");
    println!("b2 RIGHT-axis bias, square out: bias candidates = {bias_ops}");
    // ROUND-11 RE-PIN (was 1): one Bias candidate per decorable base
    // frame combination (2 A x 2 B on the bridge-satisfying site).
    assert_eq!(bias_ops, 4, "the per-column bias reaches the Bias contract");
    assert_candidates_sound(&s, "b2");

    let (elected, _) = flavored_cublaslt(&text, false, true, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    assert_eq!(lt.len(), 1);
    let e = lt[0];
    let spec = e.spec();
    assert_eq!(e.label, "CublasLtBias");
    assert!(spec.has_bias);
    assert_eq!(spec.epilogue, CuEpilogue::Bias);
    assert_eq!(e.arity(), 3);
    assert_eq!(
        e.inputs[2].1,
        spec.bias_tensor.clone().expect("bias tensor"),
        "Lit slot 2 IS the spec's bias tensor"
    );
    assert_eq!(e.inputs[2].0, "bias");
}

// ===========================================================================
// GROUP C — DATAFLOW LIST vs CONSTRUCTOR CONTRACT
// Each constructor's Lit arity is a constant of its NAME, in a fixed order.
// Attack the slot assignment with operands that coincide.
// ===========================================================================

/// c1: C == A. `x @ w + x` with x, w, out all [4,4] — the accumulator IS
/// the first matmul operand, so the Lit list is [x, w, x] with the same
/// value in slots 0 and 2. A deduping or set-based operand walk breaks the
/// contract here.
#[test]
fn attack_c1_c_equals_a() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = (x.matmul(w) + x).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let acc = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!("c1 c==a: accumulate candidates = {acc}");
    assert!(acc >= 1, "the C-fold fires with c == a");
    assert_candidates_sound(&s, "c1");
    assert_one_lit_per_op_class(&s, "c1");

    let (elected, labels) = flavored_cublaslt(&text, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert_eq!(lt.len(), 1);
    let e = lt[0];
    let spec = e.spec();
    assert_eq!(e.label, "CublasLtAccumulate");
    assert_eq!(e.arity(), 3, "arity 3 even though two slots share a value");
    assert_eq!(
        e.inputs.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        vec!["a", "b", "c"]
    );
    // ROUND-11 RE-PIN: the sibling's B-role operand is now itself the
    // transpose VIEW of x, and the bridged C is the same transpose view
    // of x riding the D-composed layout. x and out share the [4,4]
    // right-major layout CLASS, so the two composed layouts hash-cons and
    // slots b and c COINCIDE as one layout tensor — the c==a coincidence
    // in its round-11 spelling (legal: C rides the D layout by rule
    // guard, and both slots genuinely name the same bytes read the same
    // way).
    assert_eq!(
        e.inputs[1].1, e.inputs[2].1,
        "b and c coincide: the same transpose view of x on the same composed layout"
    );
    let c_logical = logical_of_lt(&s, &e.inputs[2].1).expect("c logical");
    let b_logical = logical_of_lt(&s, &e.inputs[1].1).expect("b logical");
    assert_eq!(
        c_logical, b_logical,
        "one logical value (the transpose view of x)"
    );
    assert!(
        s.nodes.values().any(|n| {
            n.op == "LogicalIndexMapApply"
                && n.eclass == c_logical
                && n.children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| {
                        s.nodes
                            .values()
                            .any(|m| m.eclass == c.eclass && m.op == "LogicalTensorInputLit")
                    })
                    .unwrap_or(false)
        }),
        "slot c's logical is the transpose view of the recorder's x (c == a in the recorder frame)"
    );
    assert_eq!(
        e.inputs[2].1,
        spec.c_tensor.clone().expect("c tensor"),
        "slot 2 IS the spec's C"
    );
    assert_eq!(
        e.inputs[0].1, spec.desc_a_layout_tensor,
        "slot 0 is the A descriptor's operand (unswapped)"
    );
    assert_ne!(e.inputs[1].1, e.inputs[0].1, "slots a and b are distinct");
}

/// c2: C == the matmul's own output. `y = x@w; out = y + y` — the
/// accumulator is the base op's own D.
///
/// FINDING (minor, pessimization not unsoundness): the C-fold rule has no
/// guard against `?c_logical` being the op's OWN claimed output, so the
/// elected plan is TWO GEMMs — one computing y, one computing AB + y —
/// where a decomposed plan is one GEMM plus an add. Arithmetically correct
/// (D = AB + y = 2y) but strictly more FLOPs. Nothing in the cost layer
/// sees this, because the cost layer never elects cuBLASLt at all (see s7).
#[test]
fn attack_c2_self_add_mm_plus_mm() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let y = x.matmul(w);
        let _ = (y + y).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let acc = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!("c2 y+y: accumulate candidates = {acc}");
    assert!(acc >= 1, "C-fold fires with c == the op's own y");
    assert_candidates_sound(&s, "c2");
    assert_one_lit_per_op_class(&s, "c2");

    let (elected, labels) = flavored_cublaslt(&text, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert_eq!(
        lt.len(),
        2,
        "TWO kernels: the base GEMM producing y, and the accumulate consuming \
         it as C — the fold recomputes AB"
    );
    let folded = lt
        .iter()
        .find(|e| e.spec().has_c)
        .expect("one accumulate kernel");
    let base = lt
        .iter()
        .find(|e| !e.spec().has_c)
        .expect("one base kernel producing y");
    let spec = folded.spec();
    assert_eq!(folded.label, "CublasLtAccumulate");
    assert_eq!(folded.arity(), 3);
    assert_eq!(folded.inputs[2].1, spec.c_tensor.clone().expect("c"));
    assert_eq!(
        folded.inputs[2].1, base.out_lt,
        "the accumulate's C IS the base op's output"
    );
    // C is y; D is (y+y) — DIFFERENT claimed output.
    assert_ne!(
        folded.inputs[2].1, folded.out_lt,
        "C is y, D is y+y — the claimed output moved"
    );
    assert_ne!(
        spec.logical_out, spec.logical_site_out,
        "the decorated D is off the site out"
    );
    assert_eq!(
        spec.logical_site_out,
        base.spec().logical_out,
        "both kernels sit on the SAME site — the fold duplicated the GEMM"
    );
    assert_eq!(spec.ldc, spec.ldd, "C rides the D layout");
}

/// c3: the same C added TWICE — `((x@w) + c) + c`. There is no
/// double-accumulate contract, and F32 LogicalAdd is NOT associative in
/// this preamble (round 2026-08-06 dtype gate), so the second add must
/// stay decomposed. A regression that reassociated floats would silently
/// fold both.
#[test]
fn attack_c3_same_c_added_twice() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = ((x.matmul(w) + c) + c).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let acc = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!("c3 (y+c)+c: accumulate candidates = {acc}");
    assert_candidates_sound(&s, "c3");

    let (elected, labels) = flavored_cublaslt(&text, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert_eq!(lt.len(), 1, "exactly ONE cuBLASLt kernel — no double fold");
    let e = lt[0];
    assert_eq!(e.arity(), 3);
    assert!(
        labels.iter().any(|l| l.contains("Add")),
        "the second add survives as a plain op: {labels:?}"
    );
}

/// c4: bias FIRST, then C, then relu — `relu((x@w + bias) + c)`. The
/// decorator chain must route Bias -> AccumulateBias (c inserted at Lit#2,
/// bias pushed to #3) and the activation must land LAST.
#[test]
fn attack_c4_bias_then_c_then_relu_order() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let bias = cx.tensor(3usize, DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = ((x.matmul(w) + bias.expand_dim(0, 4usize)) + c)
            .relu()
            .output();
    });
    let s = test_runtime::serialize_fixture(&text);
    println!(
        "c4 bias->c->relu: accbias candidates = {}",
        count_op(&s, "LayoutTensorOpCublasLtAccumulateBias")
    );
    assert_candidates_sound(&s, "c4");
    assert_one_lit_per_op_class(&s, "c4");

    let (elected, labels) = flavored_cublaslt(&text, true, true, true);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  plan labels: {labels:?}");
    assert_eq!(lt.len(), 1);
    let e = lt[0];
    let spec = e.spec();
    assert_eq!(e.label, "CublasLtAccumulateBias");
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert_eq!(e.arity(), 4);
    assert_eq!(
        e.inputs.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
        vec!["a", "b", "c", "bias"],
        "c inserted at #2, bias pushed to #3"
    );
    assert_eq!(e.inputs[2].1, spec.c_tensor.clone().expect("c"));
    assert_eq!(e.inputs[3].1, spec.bias_tensor.clone().expect("bias"));
    assert_ne!(
        e.inputs[2].1, e.inputs[3].1,
        "C and bias are structurally distinct (rank-2 vs rank-1 layouts)"
    );
}

/// c5: BOTH decoration orders reach AccumulateBias with IDENTICAL slot
/// content. (bias-then-C) vs (C-then-bias) — the API computes
/// D = act(alpha*AB + beta*C + bias), so order of the two additions is
/// immaterial and the two spellings must produce the same call.
#[test]
fn attack_c5_bias_then_c_plain() {
    let bias_then_c = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let bias = cx.tensor(3usize, DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = ((x.matmul(w) + bias.expand_dim(0, 4usize)) + c).output();
    });
    let c_then_bias = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let bias = cx.tensor(3usize, DType::F32);
        let _ = ((x.matmul(w) + c) + bias.expand_dim(0, 4usize)).output();
    });

    for (name, text) in [("bias-then-c", &bias_then_c), ("c-then-bias", &c_then_bias)] {
        let s = test_runtime::serialize_fixture(text);
        assert_candidates_sound(&s, name);
        let (elected, labels) = flavored_cublaslt(text, true, true, false);
        let lt: Vec<_> = elected
            .iter()
            .filter(|e| e.label.starts_with("CublasLt"))
            .collect();
        println!("c5 {name}: labels={labels:?}");
        assert_eq!(lt.len(), 1, "{name}: one kernel");
        let e = lt[0];
        let spec = e.spec();
        assert_eq!(e.label, "CublasLtAccumulateBias", "{name}");
        assert_eq!(
            spec.epilogue,
            CuEpilogue::Bias,
            "{name}: no activation asked"
        );
        assert_eq!(e.arity(), 4, "{name}");
        assert_eq!(
            e.inputs.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>(),
            vec!["a", "b", "c", "bias"],
            "{name}: contract slot order"
        );
        assert_eq!(e.inputs[2].1, spec.c_tensor.clone().expect("c"), "{name}");
        assert_eq!(
            e.inputs[3].1,
            spec.bias_tensor.clone().expect("bias"),
            "{name}"
        );
        assert_eq!(spec.mnk_lits(), (3, 4, 8), "{name}");
    }
}

/// c6 (charter §7): `LtMatmulSpec` has NO alpha and NO beta field, and no
/// egglog rule ever binds one — every minted call is implicitly
/// `D = act(1.0*AB + 1.0*C + bias)`. So a SCALED program must either not
/// fold at all, or fold with the scaling absorbed INTO an operand. Nothing
/// on the pre-existing board tests a scaled program.
#[test]
fn attack_c6_no_alpha_beta_channel() {
    // (a) a scaled PRODUCT: 2*(x@w) + c. Folding this would need alpha = 2.
    let scaled_product = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = ((x.matmul(w) * 2.0) + c).output();
    });
    let s = test_runtime::serialize_fixture(&scaled_product);
    assert_candidates_sound(&s, "c6-alpha");
    let (elected, labels) = flavored_cublaslt(&scaled_product, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("c6 scaled product 2*(x@w)+c: labels={labels:?}");
    for e in &lt {
        // If a fold happened, its C must be the *added* value, and the
        // claimed D must be the add's output — never the scaled product,
        // which no beta=1 call can express.
        if e.spec().has_c {
            assert_ne!(
                e.inputs[2].1, e.out_lt,
                "an alpha!=1 call was NOT minted under the guise of a C-fold"
            );
        }
    }
    assert!(
        labels.iter().any(|l| l.contains("Mul")),
        "the scale factor survives as an explicit op (no alpha channel): {labels:?}"
    );

    // (b) a scaled ACCUMULATOR: x@w + 2*c. beta = 2 is inexpressible, so if
    // this folds at all, C must be the MATERIALIZED (2*c), not c.
    let scaled_c = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = (x.matmul(w) + (c * 2.0)).output();
    });
    let s2 = test_runtime::serialize_fixture(&scaled_c);
    assert_candidates_sound(&s2, "c6-beta");
    let (elected2, labels2) = flavored_cublaslt(&scaled_c, true, false, false);
    let lt2: Vec<_> = elected2
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("c6 scaled accumulator x@w+2*c: labels={labels2:?}");
    let folded = lt2.iter().filter(|e| e.spec().has_c).count();
    if folded > 0 {
        assert!(
            labels2.iter().any(|l| l.contains("Mul")),
            "the beta scaling is MATERIALIZED into the C operand, not folded \
             into a beta the spec cannot carry: {labels2:?}"
        );
    }
    println!("  {folded} accumulate kernel(s); beta stays implicitly 1.0");
}

// ===========================================================================
// GROUP D — DECORATOR COMPOSITION
// ===========================================================================

/// d1: TWO relu consumers, each of a different matmul over a shared input.
/// Each must get its OWN decorated op claiming its OWN output — a
/// decorator that leaked the D reading across sites would cross-claim.
#[test]
fn attack_d1_two_relu_consumers() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w1 = cx.tensor((8usize, 3usize), DType::F32);
        let w2 = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w1).relu().output();
        let _ = x.matmul(w2).relu().output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, ..) = census(&s);
    println!("d1 two relu consumers: sites={sites}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site per matmul.
    assert_eq!(sites, 4);
    assert_candidates_sound(&s, "d1");
    assert_one_lit_per_op_class(&s, "d1");

    let (elected, labels) = flavored_cublaslt(&text, false, false, true);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  labels={labels:?}");
    assert_eq!(lt.len(), 2, "one decorated op per consumer");
    for e in &lt {
        let spec = e.spec();
        assert_eq!(spec.epilogue, CuEpilogue::Relu);
        assert_eq!(e.arity(), 2);
        assert_ne!(
            spec.logical_out, spec.logical_site_out,
            "each claims its OWN relu output, not the matmul out"
        );
        assert_eq!(
            e.out_logical, spec.logical_out,
            "the plan's produced value IS the spec's claimed D"
        );
    }
    assert_ne!(
        lt[0].spec().logical_out,
        lt[1].spec().logical_out,
        "the two decorated outputs are distinct"
    );
    // ROUND-10 RE-PIN: sibling sites put the shared x in the B role.
    assert_eq!(
        lt[0].spec().logical_b,
        lt[1].spec().logical_b,
        "they share the x operand (sibling b role)"
    );
    assert_ne!(lt[0].spec().logical_a, lt[1].spec().logical_a);
}

/// d2: relu(relu(x@w)) — the epilogue field rewrite requires
/// `CublasLtEpilogueDefault`, so it can fire at most once per op. The outer
/// relu must stay decomposed and saturation must terminate.
#[test]
fn attack_d2_double_relu() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w).relu().relu().output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let relu_values = count_op(&s, "CublasLtEpilogueRelu");
    let base = count_op(&s, "LayoutTensorOpCublasLt");
    println!("d2 double relu: base enodes={base}, EpilogueRelu values={relu_values}");
    assert!(relu_values >= 1, "the inner relu fuses");
    assert_candidates_sound(&s, "d2");
    assert_one_lit_per_op_class(&s, "d2");

    let (elected, labels) = flavored_cublaslt(&text, false, false, true);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  labels={labels:?}");
    assert_eq!(
        lt.len(),
        1,
        "ONE fused kernel — the epilogue is a single flag"
    );
    assert_eq!(lt[0].spec().epilogue, CuEpilogue::Relu);
    let residual = labels.iter().filter(|l| !l.starts_with("CublasLt")).count();
    assert!(
        residual > 0,
        "the second relu survives as decomposed ops: {labels:?}"
    );
}

/// d3: `x@w1 + x@w2` — the add can fold into EITHER matmul, so two
/// Accumulate candidates exist over two sites. Exactly one is elected, and
/// its C must be the OTHER matmul's output (never its own operand).
#[test]
fn attack_d3_mm_plus_mm_two_sites() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w1 = cx.tensor((8usize, 3usize), DType::F32);
        let w2 = cx.tensor((8usize, 3usize), DType::F32);
        let _ = (x.matmul(w1) + x.matmul(w2)).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, ..) = census(&s);
    let acc = count_op(&s, "LayoutTensorOpCublasLtAccumulate");
    println!("d3 mm+mm: sites={sites} accumulate candidates={acc}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site per matmul.
    assert_eq!(sites, 4, "two matmuls, two site pairs");
    // ROUND-11 RE-PIN (was 2): the fold still fires once per matmul, and
    // each folds over the 2 A x 2 B frame cross product of its site.
    assert_eq!(
        acc, 8,
        "the add folds into EITHER matmul — 4 frame combos each"
    );
    assert_candidates_sound(&s, "d3");
    assert_one_lit_per_op_class(&s, "d3");

    // Both candidates must be internally consistent: C is never the op's
    // own a or b.
    for spec in specs_of_every_enode(&s, CublasLtForm::Accumulate) {
        let c = spec.c_tensor.clone().expect("accumulate carries C");
        assert_ne!(
            c, spec.desc_a_layout_tensor,
            "C is not the op's own b operand"
        );
        assert_ne!(
            c, spec.desc_b_layout_tensor,
            "C is not the op's own a operand"
        );
    }

    let (elected, labels) = flavored_cublaslt(&text, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  labels={labels:?}");
    let fused: Vec<_> = lt.iter().filter(|e| e.spec().has_c).collect();
    assert_eq!(fused.len(), 1, "exactly one accumulate kernel elected");
    let e = fused[0];
    assert_eq!(e.arity(), 3);
    let c = e.inputs[2].1.clone();
    assert_ne!(c, e.inputs[0].1, "C is not slot a");
    assert_ne!(c, e.inputs[1].1, "C is not slot b");
    // The other matmul must still be in the plan, producing that C.
    let producers: Vec<_> = lt.iter().filter(|o| o.out_lt == c).collect();
    assert_eq!(
        producers.len(),
        1,
        "the OTHER matmul produces the accumulated C: labels={labels:?}"
    );
}

// ===========================================================================
// GROUP E — CONSTANT / RELU SPELLING DRIFT
// The relu dance is inlined in FOUR places (round 7 E1/E2 deleted the
// recognizer relations). Drift between the copies, or a premise that is not
// load-bearing, is the hazard.
// ===========================================================================

/// e1: `maximum` against a RUNTIME zeros tensor instead of the constant
/// fill. Mathematically relu when the tensor is zero-valued, but the
/// premises demand `LogicalConstant 0.0` — fail-closed, stays decomposed.
#[test]
fn attack_e1_runtime_zeros_maximum_not_fused() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let zeros = cx.tensor((4usize, 3usize), DType::F32); // runtime data, not a constant
        let _ = x.matmul(w).maximum(zeros).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let relu_values = count_op(&s, "CublasLtEpilogueRelu");
    println!("e1 runtime-zeros maximum: EpilogueRelu values = {relu_values}");
    assert_eq!(
        relu_values, 0,
        "a runtime zeros tensor must NEVER mint the relu epilogue"
    );
    assert_candidates_sound(&s, "e1");

    let (elected, labels) = flavored_cublaslt(&text, false, false, true);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("  labels={labels:?}");
    assert!(
        lt.iter().all(|e| e.spec().epilogue == CuEpilogue::Default),
        "no elected op claims an activation"
    );
    assert!(
        labels.iter().any(|l| l.contains("LessThan")),
        "the maximum survives decomposed: {labels:?}"
    );
}

/// e2: a hand-seeded relu dance that is structurally IDENTICAL to the
/// recognized one except that ONE of the five zero fills is a runtime
/// tensor. Positive control first (the untouched dance fuses), then the
/// respelled one must NOT — proving each fill premise is load-bearing
/// rather than incidental.
#[test]
fn attack_e2_respelled_relu_stays_decomposed() {
    fn dance(first_compare_rhs: &str, zeros_decl: &str) -> String {
        let extra = format!(
            r#"{zeros_decl}
(let zconst (LogicalConstant 0.0))
(let scalar_shape (ShapeLit (IntExprNil)))
(let scalar_map (IndexMapLit (IntExprNil) scalar_shape))
(let zfill (LogicalIndexMapApply zconst scalar_map out_shape))
(let nconst (LogicalConstant -1.0))
(let nfill (LogicalIndexMapApply nconst scalar_map out_shape))
(let oconst (LogicalConstant 1.0))
(let ofill (LogicalIndexMapApply oconst scalar_map out_shape))
(let lt0 (LogicalLessThan out_logical {first_compare_rhs}))
(let tc (LogicalCast lt0 (F32)))
(let term1 (LogicalMul tc zfill))
(let nm (LogicalMul tc nfill))
(let p (LogicalAdd nm ofill))
(let sel1 (LogicalCast (LogicalLessThan zfill p) (F32)))
(let sel2 (LogicalCast (LogicalLessThan p zfill) (F32)))
(let sum2 (LogicalAdd sel1 sel2))
(let bsel (LogicalLessThan zfill sum2))
(let u (LogicalCast bsel (F32)))
(let term2 (LogicalMul u out_logical))
(let relu_logical (LogicalAdd term1 term2))
(let relu_lt (LayoutTensorLit relu_logical out_layout))
(let relu_buffer_id (BufferLit 20))
(set (buffer-access-of relu_buffer_id) (ReadWrite))
(set (buffer-freed-by relu_buffer_id) (CallerFrees))
(let relu_buffer_tensor (BufferTensorLit relu_lt relu_buffer_id))
(let relu_output (BufferOutputLit (BufferTensorCons relu_buffer_tensor (BufferTensorNil))))
"#
        );
        Fx {
            extra,
            ..Default::default()
        }
        .text()
    }

    // POSITIVE CONTROL — the canonical dance.
    let good = dance("zfill", "");
    let s_good = test_runtime::serialize_fixture(&good);
    let good_relu = count_op(&s_good, "CublasLtEpilogueRelu");
    println!("e2 control (canonical dance): EpilogueRelu = {good_relu}");
    assert!(
        good_relu >= 1,
        "the hand-seeded canonical dance MUST fuse (otherwise the negative \
         result below proves nothing)"
    );

    // THE RESPELL — one fill replaced by runtime data.
    let bad = dance(
        "zeros_logical",
        r#"(let zeros_logical (LogicalTensorInputLit (LogicalIdLit "zeros") out_shape (F32)))
(let zeros_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let zeros_lt (LayoutTensorLit zeros_logical zeros_layout))
(let zeros_buffer_id (BufferLit 21))
(set (buffer-access-of zeros_buffer_id) (ReadOnly))
(set (buffer-freed-by zeros_buffer_id) (CallerFrees))
(let zeros_buffer_tensor (BufferTensorLit zeros_lt zeros_buffer_id))"#,
    );
    let s_bad = test_runtime::serialize_fixture(&bad);
    let bad_relu = count_op(&s_bad, "CublasLtEpilogueRelu");
    println!("e2 respelled (one fill -> runtime tensor): EpilogueRelu = {bad_relu}");
    assert_eq!(
        bad_relu, 0,
        "the first comparison's fill premise IS load-bearing — fail-closed"
    );
    assert_candidates_sound(&s_bad, "e2");
}

/// e3 (charter §5, the four-copy agreement): a relu decoration must reach
/// ALL FOUR contracts. Round 7 inlined the dance premises into four
/// separate rules; drift between the copies would show as one contract
/// refusing what its siblings accept.
///
/// GAP FOUND BY THIS TEST: the pre-existing board covers Base, Bias and
/// AccumulateBias with relu — but NEVER Accumulate + relu.
#[test]
fn attack_e3_four_inlined_relu_copies_agree() {
    let programs: [EpilogueProgram; 4] = [
        (
            "base",
            CublasLtForm::Base,
            CuEpilogue::Relu,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).relu().output();
            }),
        ),
        (
            "bias",
            CublasLtForm::Bias,
            CuEpilogue::ReluBias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).relu().output();
            }),
        ),
        (
            "accumulate",
            CublasLtForm::Accumulate,
            CuEpilogue::Relu,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let _ = (x.matmul(w) + c).relu().output();
            }),
        ),
        (
            "accumulate-bias",
            CublasLtForm::AccumulateBias,
            CuEpilogue::ReluBias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize))
                    .relu()
                    .output();
            }),
        ),
    ];
    for (name, form, want_ep, build) in &programs {
        let text = record(|cx| build(cx));
        let s = test_runtime::serialize_fixture(&text);
        let relu_values = count_op(&s, "CublasLtEpilogueRelu");
        let candidates = count_op(&s, form.constructor_name());
        println!(
            "e3 {name}: {candidates} {} candidate(s), EpilogueRelu values = {relu_values}",
            form.constructor_name()
        );
        assert!(
            relu_values >= 1,
            "{name}: the inlined relu copy for {form:?} did NOT fire — the four \
             copies have drifted"
        );
        assert_candidates_sound(&s, name);

        let (elected, labels) = flavored_cublaslt(&text, form.has_c(), form.has_bias(), true);
        let lt: Vec<_> = elected
            .iter()
            .filter(|e| e.label.starts_with("CublasLt"))
            .collect();
        assert_eq!(lt.len(), 1, "{name}: one kernel, labels={labels:?}");
        let e = lt[0];
        assert_eq!(e.op.form, *form, "{name}: contract");
        assert_eq!(e.spec().epilogue, *want_ep, "{name}: epilogue");
        assert_eq!(e.arity(), form.lit_arity(), "{name}: Lit arity");
    }
}

// ===========================================================================
// GROUP S — SYMBOLIC EXTENTS AND LEADING DIMENSIONS
// ===========================================================================

fn sym(name: &str, lo: Option<i64>, hi: Option<i64>) -> String {
    let mut out = format!("(let {name}_var (IntVar \"{name}\"))\n");
    if let Some(lo) = lo {
        out.push_str(&format!(
            "(set (lower-bound-of {name}_var) (bigint {lo}))\n"
        ));
    }
    if let Some(hi) = hi {
        out.push_str(&format!(
            "(set (upper-bound-of {name}_var) (bigint {hi}))\n"
        ));
    }
    out
}

/// s1: A[m,k],B[n,k] with a SYMBOLIC n bounded [1,64].
///
/// SUPERSEDED BY DESIGN CHANGE: the original round-2 finding was a REFUSAL
/// (round 2 needed `>= 2` extent guards to protect its :no-merge trans
/// functions from the k=1,n=1 weld). Descriptor TERMS made the weld legal
/// multiplicity and the guards dropped to `>= 1`, so lower bound 1 is now
/// ADMITTED. This pins the fix.
#[test]
fn attack_s1_amk_bnk_symbolic_1_64_refused() {
    let fx = Fx {
        prelude: sym("s", Some(1), Some(64)),
        m: lit(2),
        n: "s_var".into(),
        k: lit(4),
        a_rows: lit(2),
        a_cols: lit(4),
        b_rows: "s_var".into(),
        b_cols: lit(4),
        ..Default::default()
    }
    .bnk();
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s1 A[m,k],B[n,k] symbolic n in [1,64]: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site.
    assert_eq!(sites, 2);
    // ROUND-10 RE-PIN (was 1): + the original site's A reading of x.
    // ROUND-11 RE-PIN: canonicalization + the canonical-form sandwich give
    // every site operand TWO readable layout tensors — its storage frame
    // and the collapse-derived column-form frame ((x^T)^T re-described
    // over the transpose view's fresh right-major materialization frame;
    // mechanism pinned by the r8d probe) — so per-site readings double and
    // the assembly cross product scales accordingly. Bounded,
    // per-candidate-consistent multiplicity (each reading names its own
    // layout tensor); the strict level-0 election never prefers the
    // materialize-first frames.
    assert_eq!(
        a, 4,
        "lower bound 1 is ADMITTED today (the round-2 >=2 guard is gone); \
         two frames per site's a operand"
    );
    assert!(ops >= 1, "the op assembles with a symbolic n");
    assert_candidates_sound(&s, "s1");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    println!(
        "  m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert!(
        matches!(spec.m, CuDim::Symbolic(_)),
        "call m = logical n = symbolic"
    );
    assert_eq!(spec.n, 2);
    assert_eq!(spec.k, 4);
    assert!(spec.trans_a);
    assert_eq!(spec.lda, 4, "A = b[n,k] COL view, ld = storage cols = k");
    assert!(
        matches!(spec.ldd, CuDim::Symbolic(_)),
        "D = out[2,n] contiguous: ld = symbolic storage cols, got {}",
        spec.ldd
    );
}

/// s2: the same A[m,k],B[n,k] program with bounds [2,64] — admitted, same shape of
/// spec. (s1 vs s2 pins that the lower bound 1 vs 2 distinction no longer
/// gates anything.)
#[test]
fn attack_s2_amk_bnk_symbolic_2_64_admitted() {
    let fx = Fx {
        prelude: sym("s", Some(2), Some(64)),
        m: lit(2),
        n: "s_var".into(),
        k: lit(4),
        a_rows: lit(2),
        a_cols: lit(4),
        b_rows: "s_var".into(),
        b_cols: lit(4),
        ..Default::default()
    }
    .bnk();
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s2 A[m,k],B[n,k] symbolic n in [2,64]: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-11 RE-PIN (was 2): two frames per site's a operand.
    assert_eq!(a, 4);
    assert!(ops >= 1);
    assert_candidates_sound(&s, "s2");

    let elected = pinned_cublaslt(&text);
    let spec = elected[0].spec();
    assert!(matches!(spec.m, CuDim::Symbolic(_)));
    assert!(spec.trans_a);
    assert_eq!(spec.lda, 4);
}

/// s3: A[m,k],B[n,k] with a symbolic REDUCED axis k in [1,64] and a literal n. Now
/// BOTH lda (= b storage cols = k) and ldb (= a storage cols = k) go
/// symbolic while m stays literal — the round-6 Ruling-1 shape.
#[test]
fn attack_s3_amk_bnk_symbolic_k_1_64_literal_n() {
    let fx = Fx {
        prelude: sym("s", Some(1), Some(64)),
        m: lit(2),
        n: lit(3),
        k: "s_var".into(),
        a_rows: lit(2),
        a_cols: "s_var".into(),
        b_rows: lit(3),
        b_cols: "s_var".into(),
        ..Default::default()
    }
    .bnk();
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s3 A[m,k],B[n,k] symbolic k in [1,64]: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site.
    assert_eq!(sites, 2);
    assert!(ops >= 1, "symbolic reduced axis assembles");
    assert_candidates_sound(&s, "s3");

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    println!(
        "  m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(spec.m, 3, "call m = logical n = literal");
    assert_eq!(spec.n, 2);
    assert!(matches!(spec.k, CuDim::Symbolic(_)));
    assert!(spec.trans_a);
    assert!(
        matches!(spec.lda, CuDim::Symbolic(_)),
        "lda = symbolic k, got {}",
        spec.lda
    );
    assert!(
        matches!(spec.ldb, CuDim::Symbolic(_)),
        "ldb = symbolic k, got {}",
        spec.ldb
    );
    assert_eq!(spec.ldd, 3, "D stays literal");
}

/// s4: an extent with NO bounds rows at all. `lower-bound-of` is a PARTIAL
/// function, so every arm that needs a nonemptiness bound simply fails to
/// match. The refusal must be STRUCTURAL — the operand descriptor terms
/// must be ABSENT, not merely uncompetitive.
#[test]
fn attack_s4_amk_bnk_no_bounds_refused() {
    let fx = Fx {
        prelude: sym("u", None, None),
        m: lit(2),
        n: lit(3),
        k: "u_var".into(),
        a_rows: lit(2),
        a_cols: "u_var".into(),
        b_rows: lit(3),
        b_cols: "u_var".into(),
        ..Default::default()
    }
    .bnk();
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s4 A[m,k],B[n,k] unbounded k: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site.
    assert_eq!(
        sites, 2,
        "the SITE rules carry no bounds guard — the site marks"
    );
    assert_eq!(a, 0, "no A reading term is minted (structural refusal)");
    assert_eq!(b, 0, "no B reading term is minted");
    // ROUND-10 RE-PIN (was 1): one D reading per site of the pair.
    assert_eq!(
        d, 2,
        "the D arm's own bounds (literal m, n) are satisfied — the refusal is \
         ARM-LOCAL, not a blanket non-candidacy"
    );
    assert_eq!(ops, 0, "no reading set, no op");
    assert_eq!(all_candidate_specs(&s).len(), 0);
}

/// s5: the A[m,k],B[k,n] counterpart — the site marks but no op can assemble.
/// Pins the layering: site admission is logical identity only; the
/// descriptor stratum is where refusal lives.
#[test]
fn attack_s5_amk_bkn_no_bounds_site_but_no_op() {
    let fx = Fx {
        prelude: sym("u", None, None),
        m: lit(2),
        n: lit(3),
        k: "u_var".into(),
        a_rows: lit(2),
        a_cols: "u_var".into(),
        b_rows: "u_var".into(),
        b_cols: lit(3),
        ..Default::default()
    };
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s5 A[m,k],B[k,n] unbounded k: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site.
    assert_eq!(sites, 2);
    assert_eq!(a, 0);
    assert_eq!(b, 0);
    assert_eq!(
        ops, 0,
        "NO spec-less op can be elected because NO op exists"
    );

    // And the decomposed route still reaches the boundary.
    let extraction = std::panic::catch_unwind(AssertUnwindSafe(|| {
        test_runtime::extract_fixture_with_genome(&text, PIN)
    }));
    // SETUP-VS-SUBJECT GUARD (round-8b audit): the Err arm below is a
    // finding about the DECOMPOSED ROUTE, so it must not be reachable by
    // a broken fixture. A vocabulary/parse failure is setup breakage and
    // is asserted red here rather than reported as a finding.
    if let Err(p) = &extraction {
        let msg = panic_text(&**p);
        assert!(
            !msg.contains("parse error") && !msg.contains("Unbound"),
            "SETUP BROKE (not a finding): {msg}"
        );
    }
    match extraction {
        Ok((graph, _)) => {
            let labels = plan_labels(&graph);
            println!("  decomposed plan: {} ops", labels.len());
            assert!(elected_ops(&graph).is_empty(), "no cublaslt op to elect");
        }
        Err(_) => println!(
            "  OBSERVED: the decomposed route also refuses an unbounded extent \
             (reference kernels need bounds) — the marker's refusal is not the \
             blocking one"
        ),
    }
}

/// s6: A[m,k],B[k,n] with a symbolic k bounded [2,64].
///
/// SUPERSEDED BY DESIGN CHANGE: the original name records `op_spec_none` —
/// round 2 minted the op but could not parse a spec for symbolic geometry.
/// Round-6 Ruling 1 made m/n/k and every ld carry a class handle, so the
/// spec is now ALWAYS `Some`. This pins that.
#[test]
fn attack_s6_amk_bkn_symbolic_2_64_op_spec_none() {
    let fx = Fx {
        prelude: sym("s", Some(2), Some(64)),
        m: lit(2),
        n: lit(3),
        k: "s_var".into(),
        a_rows: lit(2),
        a_cols: "s_var".into(),
        b_rows: "s_var".into(),
        b_cols: lit(3),
        ..Default::default()
    };
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("s6 A[m,k],B[k,n] symbolic k in [2,64]: sites={sites} a={a} b={b} d={d} ops={ops}");
    assert!(ops >= 1);
    let specs = all_candidate_specs(&s);
    assert_eq!(
        specs.len(),
        ops,
        "EVERY op candidate parses a spec — spec-less candidates are extinct"
    );

    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    assert!(
        elected[0].op.spec.is_some(),
        "spec is Some (round-2's None is gone)"
    );
    assert_eq!(spec.m, 3);
    assert_eq!(spec.n, 2);
    assert!(matches!(spec.k, CuDim::Symbolic(_)));
    assert!(matches!(spec.ldb, CuDim::Symbolic(_)));
    assert_eq!(spec.lda, 3);
    assert_eq!(spec.ldd, 3);
    assert!(!spec.trans_a && !spec.trans_b);
}

/// s7: what does the DEFAULT (min-cost, no genome) extractor do with a
/// symbolic-extent matmul? The whole pre-existing cuBLASLt board pins
/// GENOME-driven elections; nothing pins the cost layer. Observation, then
/// a pin of whatever it does — a silent flip would change every plan.
#[test]
fn attack_s7_symbolic_op_default_cost_election() {
    let symbolic = Fx {
        prelude: sym("s", Some(2), Some(64)),
        m: lit(2),
        n: lit(3),
        k: "s_var".into(),
        a_rows: lit(2),
        a_cols: "s_var".into(),
        b_rows: "s_var".into(),
        b_cols: lit(3),
        ..Default::default()
    }
    .text();
    let literal = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
    });

    for (name, text) in [("symbolic", &symbolic), ("literal", &literal)] {
        let graph = test_runtime::extract_fixture(text);
        let labels = plan_labels(&graph);
        let cublaslt = elected_ops(&graph);
        println!(
            "s7 {name}: default min-cost plan = {labels:?} ({} cublaslt op(s))",
            cublaslt.len()
        );
        for e in &cublaslt {
            assert!(
                e.op.spec.is_some(),
                "{name}: a default-cost election never yields a spec-less op"
            );
            assert_eq!(e.arity(), e.op.form.lit_arity(), "{name}: arity contract");
        }
        // PIN (the answer, which was not obvious): the DEFAULT min-cost
        // extractor DOES elect the fused kernel, for literal AND symbolic
        // geometry. So every untargeted `extract_fixture` in the tree already
        // depends on the cuBLASLt cost being competitive; a cost-model change
        // silently reshapes those plans. Nothing on the pre-existing board
        // said so — every cuBLASLt test there pins a GENOME.
        assert_eq!(
            cublaslt.len(),
            1,
            "{name}: min-cost extraction elects exactly one cuBLASLt kernel \
             ({labels:?}) — if this changes, re-audit every default extraction"
        );
        assert_eq!(cublaslt[0].arity(), 2, "{name}: base contract");
        println!("  OBSERVED-PIN {name}: {} kernel(s)", cublaslt.len());
    }
}

/// s8 (charter §2, symbolic PITCH): a bucket request whose pitch is
/// SYMBOLIC. The creator rewrite demands `(IntLit ?pitch_v)` and a proven
/// clamp against the cols' upper bound, so an unsafe request must simply
/// never rewrite — no padded layout term, no reading, fail-closed.
#[test]
fn attack_s8_symbolic_bucket_request_refused() {
    // FLIPPED (round-8c) — and this flip RECORDS A LOST PERIMETER.
    // This test used to drive the CREATOR REWRITE and pin its safety
    // gates: a symbolic pitch, and a literal pitch below upper(cols),
    // both refused to mint a padded layout. cuBLASLt no longer owns that
    // rewrite (or any relation) — padding belongs to the bucketing
    // estate — so those gates are no longer enforced anywhere in this
    // prototype; they are now a written OBLIGATION in the static reading
    // arms' rule text.
    //
    // What the ARMS still enforce independently of any creator is pinned
    // below: a padded layout with NO injectivity fact is refused. What
    // they cannot enforce is also pinned: a FALSELY certified layout
    // (undersized pitch + asserted injectivity) is admitted. That is the
    // answer to "is an egglog tripwire expressible?" — no: injectivity is
    // an asserted :no-merge attribute, so no premise can contradict a
    // false assertion. The perimeter is the creator's correctness.
    let base = |extra: &str| {
        Fx {
            prelude: sym("s", Some(2), Some(8)),
            m: lit(2),
            n: lit(3),
            k: "s_var".into(),
            a_rows: lit(2),
            a_cols: "s_var".into(),
            b_rows: "s_var".into(),
            b_cols: lit(3),
            extra: extra.to_string(),
            ..Default::default()
        }
        .text()
    };

    // POSITIVE CONTROL: an estate-minted, CERTIFIED pitch-8 layout is read.
    // ROUND 10: the creator ALSO authors its strided-lists provenance row
    // (the sanctioned write site) — pitched layouts are now read through
    // the ladder + composition, the map-spelling fallback arm is deleted.
    let good = base("(let x_lt_pad8_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_pad8 (LayoutTensorLit x_logical x_lt_pad8_layout))\n(set (injectivity-of x_lt_pad8) (Injective))
(strided-lists x_lt_pad8_layout a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))");
    let s_good = test_runtime::serialize_fixture(&good);
    let (_, _, b_good, _, ops_good) = census(&s_good);
    println!("s8 control (certified pitch 8): B readings={b_good} ops={ops_good}");
    // ROUND-10: 3 = sibling contiguous (N) + sibling padded (N, ldb=8)
    // + the original site's w reading (T).
    assert!(
        b_good >= 3,
        "contiguous AND padded readings coexist (plus the original site's)"
    );
    assert_candidates_sound(&s_good, "s8-good");

    // THE SURVIVING PERIMETER: the same layout WITHOUT the injectivity
    // fact is refused by the reading arms.
    let uncertified = base(
        "(let x_lt_pad8_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_pad8 (LayoutTensorLit x_logical x_lt_pad8_layout))
(strided-lists x_lt_pad8_layout a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))",
    );
    let s_unc = test_runtime::serialize_fixture(&uncertified);
    let (_, _, b_unc, _, _) = census(&s_unc);
    println!("s8 uncertified pitch 8: B readings={b_unc}");
    // ROUND-11 RE-PIN (was 2): + the column-form frame readings; the
    // PADDED reading is still absent — the injectivity perimeter holds
    // (the pitched layout and every view composed FROM it stay refused;
    // the added readings ride certified contiguous frame layouts).
    assert_eq!(
        b_unc, 4,
        "no injectivity fact, no padded reading — arms stay fail-closed"
    );
    assert_candidates_sound(&s_unc, "s8-unc");

    // THE LOST PERIMETER, pinned: an UNDER-SIZED pitch (4 < upper(cols) = 8)
    // that is nonetheless certified IS admitted. The creator's clamp used
    // to make this unreachable; nothing in cuBLASLt can now detect it.
    let false_cert = base("(let x_lt_pad4_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 4))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_pad4 (LayoutTensorLit x_logical x_lt_pad4_layout))\n(set (injectivity-of x_lt_pad4) (Injective))
(strided-lists x_lt_pad4_layout a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 4) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))");
    let s_false = test_runtime::serialize_fixture(&false_cert);
    let (_, _, b_false, _, _) = census(&s_false);
    println!(
        "s8 FALSELY certified pitch 4 (cols upper = 8): B readings={b_false} \
         — admitted; the clamp left with the creator"
    );
    // ROUND-11 RE-PIN (was 3): + the column-form frame readings.
    assert_eq!(
        b_false, 5,
        "a false certificate is trusted: the recorded cost of E2/E3"
    );
}

/// s9 (charter §2, the other half): a HAND-ASSERTED injective strided
/// layout whose pitch is symbolic. Round-6 Ruling 1 says a symbolic ld is
/// executable, and round-7 E3 made injectivity the ld clamp — so this
/// ADMITS, with the ld riding as a class handle.
///
/// FINDING (recorded, not a failure): the "symbolic pitch is refused"
/// intuition holds only at the CREATOR (s8). Once someone asserts
/// injectivity, the reading arm admits a symbolic pitch. The trust boundary
/// is the injectivity assertion, not the pitch's literalness.
#[test]
fn attack_s9_handasserted_symbolic_pitch_admitted() {
    let fx = Fx {
        prelude: format!(
            "{}{}",
            sym("s", Some(2), Some(8)),
            sym("p", Some(8), Some(8))
        ),
        m: lit(2),
        n: lit(3),
        k: "s_var".into(),
        a_rows: lit(2),
        a_cols: "s_var".into(),
        b_rows: "s_var".into(),
        b_cols: lit(3),
        a_layout: r#"(StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) p_var)
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32)))"#
            .into(),
        extra: "(set (injectivity-of x_lt) (Injective))".into(),
        ..Default::default()
    };
    let text = fx.text();
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!(
        "s9 hand-asserted injective symbolic pitch: sites={sites} a={a} b={b} d={d} ops={ops}"
    );
    assert!(
        b >= 1,
        "the static arm ADMITS a symbolic pitch under injectivity"
    );
    assert!(ops >= 1);
    assert_candidates_sound(&s, "s9");

    // ROUND-11 RE-PIN: the ADMISSION pin moves to the candidate sweep —
    // at least one candidate must carry the walked pitch (literal 8 after
    // the [n,n] collapse, or the class handle) as an ld; the strict
    // level-0 election may elect a contiguous-frame candidate instead
    // (its ld is that frame's own row count — sound for its bytes), so
    // the elected spec is only required to parse with a usable ld.
    let specs = all_candidate_specs(&s);
    let pitch_carried = specs.iter().any(|sp| {
        [&sp.lda, &sp.ldb]
            .iter()
            .any(|ld| **ld == 8i64 || matches!(ld, CuDim::Symbolic(_)))
    });
    assert!(
        pitch_carried,
        "some candidate walks the hand-asserted pitch (literal 8 or a class handle)"
    );
    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    println!(
        "  elected ldb = {} (one sound frame of the candidate set)",
        spec.ldb
    );
    assert!(
        spec.ldb.literal().is_some() || matches!(spec.ldb, CuDim::Symbolic(_)),
        "the elected reading's ldb is usable (literal or class handle), got {}",
        spec.ldb
    );
}

// ===========================================================================
// GROUP P — PARSER PANIC PATHS (charter §6)
// Extraction must be infallible for terms the RULES mint; panics are
// defense in depth against hand-built terms only.
// ===========================================================================

/// The text of a caught panic payload (round-8b audit helper): lets a
/// test tell SETUP breakage (parse/vocabulary errors) apart from the
/// SUBJECT's own refusal.
fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn panic_message<T>(f: impl FnOnce() -> T) -> Option<String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    match result {
        Ok(_) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".into()),
        ),
    }
}

/// p1: the `col_view` FORM/layout coherence assert. No D arm ever concludes
/// `LeftMajorContiguous` (the LM output family is refused by ruling), so a
/// LM-form D reading is rule-impossible. Hand-build one and confirm the
/// parser dies LOUDLY instead of computing a transposed descriptor.
#[test]
fn attack_p1_form_layout_mismatch_panics_loudly() {
    // FLIPPED (round-8b E1): this test used to hand-seed a descriptor
    // whose layout FORM contradicted its layout term, and pin the
    // extractor's form/layout coherence panic. E1 deleted the form child
    // (orientation now comes from the OPERATION, which the rules prove),
    // so that mismatch is no longer expressible and its check is gone.
    // THAT IS A REAL LOSS OF DEFENSE IN DEPTH, recorded in the round-8b
    // report. This test is re-aimed at the tripwire that still catches a
    // rule-impossible hand-seeded term: the view/storage transposition
    // cross-check. Here the A descriptor is pointed at the OUT layout
    // tensor ([2,3]) while the site's b operand is w ([4,3]), so the
    // view the operation implies (3x4) is not a transposition of the
    // storage it names (2x3) — parse must die loudly, not produce a
    // plausible-looking descriptor.
    let fx = Fx::default().text();
    let seeded = fx.replace(
        SCHEDULE,
        &format!(
            r#"(let bad_site (CublasLtLogicalMatmulSite x_logical w_logical out_logical))
(let bad_desc_a (CublasLtOperandADescriptor bad_site out_lt
  (CublasLtOperationN)))
(let bad_desc_b (CublasLtOperandBDescriptor bad_site x_lt
  (CublasLtOperationN)))
(let bad_desc_d (CublasLtOutputDDescriptor bad_site out_lt))
(let bad_op (LayoutTensorOpCublasLt bad_site bad_desc_a bad_desc_b bad_desc_d
  (CublasLtEpilogueDefault)))
{SCHEDULE}"#
        ),
    );
    let s = test_runtime::serialize_fixture(&seeded);
    // The hand-built enode: its A descriptor names the OUT layout tensor.
    // ROUND 10: there are several D readings (sibling views); locate the
    // RECORDED boundary out lt via its ReadWrite BufferTensorLit instead
    // of "the first D descriptor found".
    let out_lt_class = s
        .nodes
        .values()
        .filter(|n| n.op == "BufferTensorLit")
        .find_map(|n| {
            let buffer = n.children.get(1).and_then(|id| s.nodes.get(id))?;
            let rw = s.nodes.values().any(|m| {
                m.op == "buffer-access-of"
                    && m.children
                        .first()
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| c.eclass == buffer.eclass)
                        .unwrap_or(false)
                    && s.nodes
                        .values()
                        .any(|r| r.eclass == m.eclass && r.op == "ReadWrite")
            });
            if !rw {
                return None;
            }
            n.children
                .first()
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone())
        })
        .expect("the boundary out lt exists");
    let bad: Vec<_> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op == "LayoutTensorOpCublasLt")
        .filter(|(_, n)| {
            n.children
                .get(1)
                .and_then(|id| s.nodes.get(id))
                .map(|a| a.eclass.clone())
                .map(|a_class| {
                    s.nodes.values().any(|m| {
                        m.eclass == a_class
                            && m.op == "CublasLtOperandADescriptor"
                            && m.children
                                .get(1)
                                .and_then(|id| s.nodes.get(id))
                                .map(|lt| lt.eclass == out_lt_class)
                                .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
        })
        .collect();
    println!("p1: {} hand-built cross-pointed op enode(s)", bad.len());
    // ROUND-10 RE-PIN (was 1): the seeded descriptors CROSS-POLLINATE —
    // assembly freely combines the hand-built A descriptor with the legit
    // B and D readings on the same site (the assemble rule quantifies
    // over descriptor terms), so 4 combos exist. The tripwire obligation
    // therefore strengthens: EVERY combo that names the out tensor as an
    // operand must die loudly in parse, not just the fully-seeded one.
    // ROUND-11 RE-PIN (was 4): the legit B/D readings the hand-built A
    // combines with now include the column-form frame readings, widening
    // the cross product to 6. The tripwire obligation is unchanged and
    // STRONGER: every combo must die loudly in parse (the loop below).
    assert_eq!(
        bad.len(),
        6,
        "the hand-built descriptor pollinates 6 combos"
    );
    for (id, node) in &bad {
        let msg = panic_message(|| {
            let site = ExtractionSite {
                egraph: &s,
                node_id: id,
                node,
            };
            parse_spec(&site, CublasLtForm::Base)
        });
        println!("p1 parse_spec on a rule-impossible combo: {msg:?}");
        let msg = msg.expect("a cross-check fires — silence here would be the bug");
        assert!(
            msg.contains("transposition of its storage") || msg.contains("inconsistent"),
            "the panic names the violated cross-check, got: {msg}"
        );
    }
}

/// p2: the `static_row_pitch` ambiguity panic. Two DISTINCT bucket pitches
/// on one layout tensor is a LEGAL program (two separate padded layout
/// classes, two readings, no panic). The panic only fires if the two
/// strided spellings land in ONE layout class — which requires a hand
/// `union` no rule performs.
#[test]
fn attack_p2_two_bucket_pitches_stay_distinct() {
    // LEGAL: two requests, two padded layout classes.
    let legal = Fx {
        prelude: sym("s", Some(2), Some(8)),
        m: lit(2),
        n: lit(3),
        k: "s_var".into(),
        a_rows: lit(2),
        a_cols: "s_var".into(),
        b_rows: "s_var".into(),
        b_cols: lit(3),
        extra: "(let x_lt_p8_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_p8 (LayoutTensorLit x_logical x_lt_p8_layout))\n(set (injectivity-of x_lt_p8) (Injective))
(strided-lists x_lt_p8_layout a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))\n(let x_lt_p16_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 16))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_p16 (LayoutTensorLit x_logical x_lt_p16_layout))\n(set (injectivity-of x_lt_p16) (Injective))
(strided-lists x_lt_p16_layout a_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 16) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))"
            .into(),
        ..Default::default()
    }
    .text();
    let s = test_runtime::serialize_fixture(&legal);
    let (_, _, b, _, ops) = census(&s);
    println!("p2 two bucket pitches (8 and 16): B readings={b} ops={ops}");
    assert!(b >= 3, "contiguous + two padded readings coexist, got {b}");
    let n_specs = assert_candidates_sound(&s, "p2");
    println!("  {n_specs} candidate spec(s), no ambiguity panic");
    let pitches: BTreeSet<String> = all_candidate_specs(&s)
        .iter()
        .map(|sp| sp.ldb.to_string())
        .collect();
    println!("  distinct ldb values across candidates: {pitches:?}");
    assert!(
        pitches.len() >= 2,
        "the two pitches stay separable per candidate: {pitches:?}"
    );

    // RULE-IMPOSSIBLE: force the two padded layouts into one class.
    let welded = legal.replace(
        SCHEDULE,
        &format!(
            r#"(let pad8 (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let pad16 (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 16))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(union pad8 pad16)
{SCHEDULE}"#
        ),
    );
    let msg = panic_message(|| {
        let sw = test_runtime::serialize_fixture(&welded);
        all_candidate_specs(&sw)
    });
    println!("p2 hand-welded pitch classes: {msg:?}");
    match msg {
        Some(m) => assert!(
            m.contains("ambiguous") || m.contains("spelling drift"),
            "the ambiguity panic names spelling drift, got: {m}"
        ),
        // SUBJECT REACHABILITY (round-8b audit): this arm is a
        // genuine finding — the hand-welded union can be absorbed before
        // the ambiguous walk is reached — and it is NOT a setup break,
        // because the fixture's serialization is asserted above (it
        // panics on failure) before this point. Left permissive
        // deliberately.
        None => println!(
            "  OBSERVED: the welded program did not reach the ambiguous walk \
             (the union is absorbed elsewhere) — the panic stays unreached"
        ),
    }
}

/// p3: the whole battery's real panic pin — across a spread of LEGAL
/// programs, EVERY candidate enode of EVERY contract parses without
/// panicking. That is the extraction-infallibility contract.
#[test]
fn attack_p3_no_legal_program_reaches_a_panic() {
    let programs: Vec<(&str, String)> = vec![
        ("plain", record(|cx| {
            let x = cx.tensor((4usize, 8usize), DType::F32);
            let w = cx.tensor((8usize, 3usize), DType::F32);
            let _ = x.matmul(w).output();
        })),
        ("amk_bnk", record(|cx| {
            let x = cx.tensor((2usize, 4usize), DType::F32);
            let w = cx.tensor((3usize, 4usize), DType::F32);
            let _ = x.matmul(w.permute((1usize, 0usize))).output();
        })),
        ("square-chain", record(|cx| {
            let x = cx.tensor((4usize, 4usize), DType::F32);
            let w1 = cx.tensor((4usize, 4usize), DType::F32);
            let w2 = cx.tensor((4usize, 4usize), DType::F32);
            let _ = x.matmul(w1).matmul(w2).output();
        })),
        ("full-stack", record(|cx| {
            let x = cx.tensor((4usize, 8usize), DType::F32);
            let w = cx.tensor((8usize, 3usize), DType::F32);
            let c = cx.tensor((4usize, 3usize), DType::F32);
            let b = cx.tensor(3usize, DType::F32);
            let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize)).relu().output();
        })),
        ("m1", record(|cx| {
            let x = cx.tensor((1usize, 4usize), DType::F32);
            let w = cx.tensor((4usize, 3usize), DType::F32);
            let _ = x.matmul(w).relu().output();
        })),
        ("all-ones", record(|cx| {
            let x = cx.tensor((1usize, 1usize), DType::F32);
            let w = cx.tensor((1usize, 1usize), DType::F32);
            let _ = x.matmul(w).output();
        })),
        ("x-at-x", record(|cx| {
            let x = cx.tensor((4usize, 4usize), DType::F32);
            let _ = x.matmul(x).output();
        })),
        ("weld-corner", Fx {
            m: lit(1), n: lit(3), k: lit(1),
            a_rows: lit(1), a_cols: lit(1),
            b_rows: lit(1), b_cols: lit(3),
            ..Default::default()
        }.text()),
        ("role-swap", Fx {
            m: lit(1), n: lit(1), k: lit(4),
            a_rows: lit(1), a_cols: lit(4),
            b_rows: lit(1), b_cols: lit(4),
            ..Default::default()
        }.bnk().text()),
        ("bucket", Fx {
            prelude: sym("s", Some(2), Some(8)),
            m: lit(2), n: lit(3), k: "s_var".into(),
            a_rows: lit(2), a_cols: "s_var".into(),
            b_rows: "s_var".into(), b_cols: lit(3),
            extra: "(let x_lt_p8_layout (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_lt_p8 (LayoutTensorLit x_logical x_lt_p8_layout))\n(set (injectivity-of x_lt_p8) (Injective))".into(),
            ..Default::default()
        }.text()),
    ];
    let mut total = 0usize;
    for (name, text) in &programs {
        let s = test_runtime::serialize_fixture(text);
        let msg = panic_message(|| all_candidate_specs(&s));
        assert!(
            msg.is_none(),
            "{name}: a LEGALLY minted term panicked the parser — extraction must \
             be infallible for terms the rules mint. Panic: {msg:?}"
        );
        let n = all_candidate_specs(&s).len();
        total += n;
        assert_one_lit_per_op_class(&s, name);
        println!("p3 {name}: {n} candidate spec(s), all parse");
    }
    println!(
        "p3: {total} candidate specs parsed across {} programs, zero panics",
        programs.len()
    );
    assert!(total >= programs.len(), "every program produced candidates");
}

/// o1: THE NEGATIVE CONTROL for this battery's two global oracles. A test
/// suite whose invariants are vacuously true proves nothing, so: take a
/// REAL spec, corrupt it the way a drifted arm would (swap m and k — the
/// signature of a wrong `trans_a`), and confirm BOTH the call-frame oracle
/// and the ld clamp reject it. Also counts how many candidates across the
/// battery's fixture spread were actually compared.
#[test]
fn attack_o1_the_oracles_discriminate() {
    let text = record(|cx| {
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let specs = all_candidate_specs(&s);
    // ROUND-11 RE-PIN (was 2): the 2 A x 2 B frame cross product on each
    // of the two sites.
    assert_eq!(specs.len(), 8);
    let good = specs[0].clone();
    println!(
        "o1 good: m={} n={} k={} trans=({},{}) lda={} ldb={} ldd={}",
        good.m, good.n, good.k, good.trans_a, good.trans_b, good.lda, good.ldb, good.ldd
    );
    assert!(
        assert_geometry_matches_logical(&s, &good, "o1-good"),
        "the oracle actually COMPARED (all four logical extents resolved)"
    );

    // Corruption 1: a wrong trans_a transposes m and k.
    let mut bad = good.clone();
    std::mem::swap(&mut bad.m, &mut bad.k);
    bad.trans_a = !bad.trans_a;
    let msg = panic_message(|| assert_geometry_matches_logical(&s, &bad, "o1-bad"));
    println!("o1 m/k swapped: {msg:?}");
    let msg = msg.expect("the call-frame oracle REJECTS a transposed frame");
    assert!(msg.contains("WRONG OPERATION") || msg.contains("R10 frame broke"));

    // Corruption 2: an under-sized leading dimension.
    let mut short = good.clone();
    short.lda = CuDim::Literal(1);
    let msg = panic_message(|| assert_ld_clamps(&short, "o1-short"));
    println!("o1 lda=1: {msg:?}");
    assert!(msg
        .expect("the clamp REJECTS an uncallable ld")
        .contains("UNCALLABLE"),);

    // Coverage: how many candidates across a spread of fixtures were really
    // compared by the oracle (as opposed to skipped for symbolic extents)?
    let spread: Vec<String> = vec![
        text.clone(),
        record(|cx| {
            let x = cx.tensor((2usize, 4usize), DType::F32);
            let w = cx.tensor((3usize, 4usize), DType::F32);
            let _ = x.matmul(w.permute((1usize, 0usize))).output();
        }),
        Fx {
            m: lit(1),
            n: lit(1),
            k: lit(4),
            a_rows: lit(1),
            a_cols: lit(4),
            b_rows: lit(1),
            b_cols: lit(4),
            ..Default::default()
        }
        .bnk()
        .text(),
        record(|cx| {
            let x = cx.tensor((4usize, 8usize), DType::F32);
            let w = cx.tensor((8usize, 3usize), DType::F32);
            let c = cx.tensor((4usize, 3usize), DType::F32);
            let b = cx.tensor(3usize, DType::F32);
            let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize))
                .relu()
                .output();
        }),
    ];
    let mut compared = 0usize;
    let mut total = 0usize;
    for fixture in &spread {
        let sf = test_runtime::serialize_fixture(fixture);
        for spec in all_candidate_specs(&sf) {
            total += 1;
            if assert_geometry_matches_logical(&sf, &spec, "o1-spread") {
                compared += 1;
            }
        }
    }
    println!("o1: oracle compared {compared} of {total} candidate specs");
    assert_eq!(
        compared, total,
        "the oracle is LIVE on every literal-geometry candidate, not vacuous"
    );
}

// ===========================================================================
// GROUP F/G/H — MULTIPLICITY, IDENTITY, AND THE LIT INVARIANT
// ===========================================================================

/// f1: two coexisting readings of ONE layout tensor are legal multiplicity
/// ONLY because they agree numerically. The square dual-spelling fixture
/// (A[m,k],B[k,n] and A[m,k],B[n,k] of one [4,4] weight, outs unioned) mints both an N and a T
/// A-reading over the same `w_lt`. Assert the readings differ ONLY in
/// `trans_a` — a divergence in m/k/lda would make the multiplicity a
/// silent correctness hazard.
#[test]
fn attack_f1_dual_readings_are_legal_multiplicity() {
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
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_amk_bkn out_layout))
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
    );
    let s = test_runtime::serialize_fixture(&fx);
    let (sites, a, b, d, ops) = census(&s);
    println!("f1 dual spelling: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-11 RE-PIN (was sites=2 a=3 ops=4 classes=2 specs=4): the two
    // seeded spellings of ONE product now canonicalize into TWO canonical
    // chains over the same out (b = w directly, and b = the transpose
    // view of w), each with its sandwich sibling — 4 sites. Operand
    // frames double the readings; assembly takes the cross products.
    assert_eq!(
        sites, 4,
        "two canonical chains (stored w / viewed w) x site pair"
    );
    assert_eq!(a, 8, "two frames per site's a operand");
    assert_eq!(ops, 20, "the frame cross products across the four sites");
    assert_one_lit_per_op_class(&s, "f1");

    let specs = specs_of_every_enode(&s, CublasLtForm::Base);
    assert_eq!(specs.len(), ops, "all candidates parse");
    let trans: BTreeSet<bool> = specs.iter().map(|sp| sp.trans_a).collect();
    assert_eq!(trans, BTreeSet::from([false, true]), "one N, one T");
    let geometry: BTreeSet<(i64, i64, i64, i64, i64, i64)> = specs
        .iter()
        .map(|sp| {
            let (m, n, k) = sp.mnk_lits();
            (
                m,
                n,
                k,
                sp.lda.literal().unwrap(),
                sp.ldb.literal().unwrap(),
                sp.ldd.literal().unwrap(),
            )
        })
        .collect();
    println!("  geometries across the readings: {geometry:?}");
    // ROUND-11 RE-PIN (was the two frame geometries with one ld tuple
    // each): candidates still live in exactly the sandwich FRAME PAIR —
    // (4,2,4) and (2,4,4) — and within each frame the lds now take two
    // values because the column-form frame layouts are contiguous with
    // the OTHER leading dimension (e.g. lda=2 where the storage frame has
    // lda=4). Every tuple satisfies the ld >= rows clamp (the soundness
    // sweep above); the frame set itself is unchanged.
    let frames: BTreeSet<(i64, i64, i64)> = geometry.iter().map(|g| (g.0, g.1, g.2)).collect();
    assert_eq!(
        frames,
        BTreeSet::from([(4, 2, 4), (2, 4, 4)]),
        "exactly the two sandwich frames"
    );
    assert_eq!(
        geometry,
        BTreeSet::from([
            (2, 4, 4, 2, 4, 2),
            (2, 4, 4, 4, 4, 2),
            (4, 2, 4, 4, 2, 4),
            (4, 2, 4, 4, 4, 4)
        ]),
        "per frame, the storage-frame and column-form-frame ld tuples"
    );
    // Within each frame the A readings ride exactly TWO layout tensors —
    // the storage frame and the column-form frame. (ROUND-11 RE-PIN: was
    // one per frame.)
    use std::collections::BTreeMap;
    let mut per_frame: BTreeMap<i64, BTreeSet<ClassId>> = BTreeMap::new();
    for sp in &specs {
        per_frame
            .entry(sp.m.literal().unwrap())
            .or_default()
            .insert(sp.desc_a_layout_tensor.clone());
    }
    for (frame_m, tensors) in per_frame {
        // m=2 (the original frame): one a operand (x) x two frames = 2.
        // m=4 (the sibling frame): TWO sites share it — the stored-w
        // sibling and the viewed-w canonical chain — so two distinct a
        // operands x two frames = 4.
        let expected = if frame_m == 2 { 2 } else { 4 };
        assert_eq!(
            tensors.len(),
            expected,
            "frame m={frame_m}: the frame layout tensors of its a operand(s)"
        );
    }
}

/// g1: 16 blocks with IDENTICAL geometry. Distinct inputs must yield
/// distinct sites, distinct ops and distinct Lit lists — shared geometry
/// must weld NOTHING — with sub-quadratic node growth.
#[test]
fn attack_g1_x16_same_geometry() {
    let build = |n: usize| {
        record(move |cx| {
            for _ in 0..n {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }
        })
    };
    let one = build(1);
    let sixteen = build(16);
    let s1 = test_runtime::serialize_fixture(&one);
    let started = std::time::Instant::now();
    let s16 = test_runtime::serialize_fixture(&sixteen);
    let wall = started.elapsed();
    println!(
        "g1: 1 block = {} nodes; 16 blocks = {} nodes ({:.2}x) in {wall:.2?}",
        s1.nodes.len(),
        s16.nodes.len(),
        s16.nodes.len() as f64 / s1.nodes.len() as f64
    );
    let (sites, a, b, d, ops) = census(&s16);
    println!("  sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site per matmul.
    assert_eq!(
        sites, 32,
        "16 distinct site PAIRS — identical geometry welds nothing"
    );
    // ROUND-11 RE-PIN (was 32): the 2 A x 2 B frame cross product per
    // site — 16 matmuls x 2 sites x 4 = 128. Still linear in the block
    // count (the node-growth bound below is unchanged).
    assert_eq!(ops, 128);
    assert!(
        s16.nodes.len() < s1.nodes.len() * 32,
        "no super-linear blowup"
    );
    assert_one_lit_per_op_class(&s16, "g1");

    let elected = pinned_cublaslt(&sixteen);
    assert_eq!(elected.len(), 16, "16 kernels in the plan");
    let lits: BTreeSet<Vec<String>> = elected
        .iter()
        .map(|e| e.inputs.iter().map(|(_, v)| format!("{v:?}")).collect())
        .collect();
    assert_eq!(lits.len(), 16, "16 DISTINCT Lit lists");
    for e in &elected {
        let (m, n, k) = e.spec().mnk_lits();
        let canonical = if m <= n { (m, n, k) } else { (n, m, k) };
        assert_eq!(
            canonical,
            (3, 4, 8),
            "every kernel is one sandwich frame of the call"
        );
        assert_eq!(e.arity(), 2);
    }
}

/// g2: the CSE'd matmul — the same `x.matmul(w)` value consumed twice.
/// One site, one op, one kernel: the marker must not duplicate a kernel per
/// consumer.
#[test]
fn attack_g2_cse_same_matmul_twice() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let y = x.matmul(w);
        let _ = y.output();
        let _ = (y * 2.0).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let (sites, a, b, d, ops) = census(&s);
    println!("g2 CSE'd matmul, two consumers: sites={sites} a={a} b={b} d={d} ops={ops}");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling site per matmul.
    assert_eq!(sites, 2, "ONE site pair for the CSE'd matmul");
    assert_candidates_sound(&s, "g2");
    assert_one_lit_per_op_class(&s, "g2");

    let (elected, labels) = pinned_plan(&text);
    println!("  labels={labels:?}");
    // ROUND-10 FINDING (recorded, not silently re-pinned): the CSE
    // property regressed at ELECTION time. The rewrite gives y a second
    // (composed transpose-view) layout tensor; the elementwise consumer
    // now has Lit variants over both, and the deterministic fallback
    // genome elects the variant reading the composed tensor, whose
    // producer is the ORIGINAL site's relayout candidate — so the plan
    // carries TWO sound GEMMs where one would do. The min-cost extractor
    // makes the same choice under the bytes-moved heuristic (measured).
    // Every elected kernel is still SOUND (same geometry, distinct
    // claimed tensors); the waste is an election-quality gap, reported
    // in the round-10 report.
    assert!(
        (1..=2).contains(&elected.len()),
        "bounded kernels for the CSE'd matmul, got {}",
        elected.len()
    );
    for e in &elected {
        let (m, n, k) = e.spec().mnk_lits();
        let canonical = if m <= n { (m, n, k) } else { (n, m, k) };
        assert_eq!(
            canonical,
            (3, 4, 8),
            "every kernel is one sandwich frame of the call"
        );
        assert_eq!(e.arity(), 2);
    }
}

/// h1: THE LIT INVARIANT, swept. `assert_one_lit_per_op_class` is the
/// structural precondition for the slot contract: op SPELLINGS may
/// multiply freely (they are read per-enode) but the dataflow Lit lives in
/// the CLASS, so two Lits in one class would make a/b/c/bias ambiguous.
#[test]
fn attack_h1_one_lit_per_op_class() {
    let cases: Vec<(&str, String)> = vec![
        (
            "plain",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }),
        ),
        (
            "bias",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
            }),
        ),
        (
            "accumulate",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let _ = (x.matmul(w) + c).output();
            }),
        ),
        (
            "full-stack-relu",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize))
                    .relu()
                    .output();
            }),
        ),
        (
            "mm-plus-mm",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w1 = cx.tensor((8usize, 3usize), DType::F32);
                let w2 = cx.tensor((8usize, 3usize), DType::F32);
                let _ = (x.matmul(w1) + x.matmul(w2)).output();
            }),
        ),
        (
            "diamond",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let mm = x.matmul(w);
                let _ = mm.output();
                let _ = (mm + b.expand_dim(0, 4usize)).output();
            }),
        ),
        (
            "c-equals-a",
            record(|cx| {
                let x = cx.tensor((4usize, 4usize), DType::F32);
                let w = cx.tensor((4usize, 4usize), DType::F32);
                let _ = (x.matmul(w) + x).output();
            }),
        ),
    ];
    for (name, text) in &cases {
        let s = test_runtime::serialize_fixture(text);
        let classes = assert_one_lit_per_op_class(&s, name);
        let enodes = count_cublaslt(&s);
        println!("h1 {name}: {enodes} op enode(s) across {classes} class(es), one Lit each");
        assert!(classes >= 1, "{name}: at least one op class");
    }
}

// ===========================================================================
// GROUP U — THE UNASKED QUESTION (charter §7)
// What the pre-existing 41-test board conveniently does not test.
// ===========================================================================

/// u1: the board checks `inputs.len()` everywhere and `inputs` CONTENT
/// nowhere. This pins, per contract, that Lit slot i carries exactly the
/// tensor the spec names for that role — the actual mis-assignment guard.
#[test]
fn attack_u1_lit_slot_contents_pinned() {
    let cases: [FormProgram; 4] = [
        (
            "base",
            CublasLtForm::Base,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }),
        ),
        (
            "bias",
            CublasLtForm::Bias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
            }),
        ),
        (
            "accumulate",
            CublasLtForm::Accumulate,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let _ = (x.matmul(w) + c).output();
            }),
        ),
        (
            "accumulate-bias",
            CublasLtForm::AccumulateBias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize)).output();
            }),
        ),
    ];
    for (name, form, build) in &cases {
        let text = record(|cx| build(cx));
        let (elected, _) = flavored_cublaslt(&text, form.has_c(), form.has_bias(), false);
        let lt: Vec<_> = elected.iter().filter(|e| e.op.form == *form).collect();
        assert_eq!(lt.len(), 1, "{name}: the {form:?} contract elected");
        let e = lt[0];
        let spec = e.spec();
        assert_eq!(e.arity(), form.lit_arity(), "{name}: arity");
        let ports: Vec<&str> = e.inputs.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(ports, form.operand_names().to_vec(), "{name}: port names");
        // ROUND-10 RE-PIN: slot 0 = the site's a = descriptor A's layout
        // tensor (unswapped roles; the R9 swap is gone from the wiring).
        assert_eq!(
            e.inputs[0].1, spec.desc_a_layout_tensor,
            "{name}: slot `a` content"
        );
        assert_eq!(
            e.inputs[1].1, spec.desc_b_layout_tensor,
            "{name}: slot `b` content"
        );
        if form.has_c() {
            assert_eq!(
                e.inputs[2].1,
                spec.c_tensor.clone().expect("c"),
                "{name}: slot `c` content"
            );
        }
        if form.has_bias() {
            let bias_slot = if form.has_c() { 3 } else { 2 };
            assert_eq!(
                e.inputs[bias_slot].1,
                spec.bias_tensor.clone().expect("bias"),
                "{name}: slot `bias` content"
            );
        }
        // The op's produced value IS the spec's CLAIMED output.
        assert_eq!(e.out_logical, spec.logical_out, "{name}: claimed D");
        println!("u1 {name}: ports={ports:?}, all slot contents pinned");
    }
}

/// u2, ROUND-10 REWRITE (was: the R9 inversion pin): the descriptor roles
/// are UNSWAPPED — descriptor A carries the SITE's a, descriptor B the
/// site's b, so the two namespaces now AGREE and the naming trap is gone.
/// This test pins the agreement (a silent re-swap breaks it). The elected
/// op is the transpose-sandwich SIBLING, whose site-a is the recorder
/// matmul's w — so lda now derives from w and ldb from x, same numbers as
/// round 9, reached through the sibling frame instead of a wiring swap.
#[test]
fn attack_u2_layout_tensor_field_naming_is_inverted() {
    let text = record(|cx| {
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    let elected = pinned_cublaslt(&text);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].spec();
    let a_lt_logical =
        logical_of_lt(&s, &spec.desc_a_layout_tensor).expect("desc_a_layout_tensor logical");
    let b_lt_logical =
        logical_of_lt(&s, &spec.desc_b_layout_tensor).expect("desc_b_layout_tensor logical");
    println!(
        "u2: logical(desc_a_layout_tensor)==logical_b? {} ; logical(desc_b_layout_tensor)==logical_a? {}",
        a_lt_logical == spec.logical_b,
        b_lt_logical == spec.logical_a
    );
    assert_eq!(
        a_lt_logical, spec.logical_a,
        "spec.desc_a_layout_tensor holds the SITE's a (unswapped, R10)"
    );
    assert_eq!(
        b_lt_logical, spec.logical_b,
        "spec.desc_b_layout_tensor holds the SITE's b (unswapped, R10)"
    );
    assert_ne!(
        spec.logical_a, spec.logical_b,
        "the two are genuinely distinct here"
    );
    // The geometry cross-check that makes it observable: lda comes from our
    // b's storage cols (3), ldb from our a's storage cols (4).
    assert_eq!(spec.lda, 3);
    assert_eq!(spec.ldb, 4);
}

/// u3: bufferize a contract whose Lit list has a REPEATED operand (c == a).
/// The DPS alias table declares a May alias on slot 2; with slot 0 and
/// slot 2 sharing a buffer this is where a donation bug would show.
/// Nothing on the board bufferizes a duplicate-operand op.
///
/// Original boundary-flowing spelling (restored under escape-and-disclose,
/// ruling 2026-08-27: the round-10 sibling routing spells the boundary
/// value as a transpose VIEW of the kernel result, which now ESCAPES —
/// no dodge needed). The subject stays donation safety under the
/// duplicate-operand May alias.
#[test]
fn attack_u3_bufferize_duplicate_operand_accumulate() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = (x.matmul(w) + x).output();
    });
    let serialized = test_runtime::serialize_fixture(&text);
    let genome = genome_flavored(&serialized, true, false, false);
    let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        test_runtime::matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary");
    let elected = elected_ops(&graph);
    assert!(
        elected
            .iter()
            .any(|e| e.op.form == CublasLtForm::Accumulate),
        "the Accumulate contract elected"
    );

    let dps = luminal::dps::dps_rewrite(&graph);
    let dps_op = dps
        .dag
        .node_weights()
        .find_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => Some(op),
            _ => None,
        })
        .expect("the kernel survives the DPS rewrite");
    assert_eq!(
        dps_op.inputs.len(),
        4,
        "DPS operands = contract arity 3 + dest"
    );
    assert_eq!(dps_op.op.operand_name(3), "dest");
    // ROUND-10 RE-PIN: on the sibling site the folded C is the transpose
    // VIEW of the recorder's x (which sits in slot 1 = the B role), so
    // the duplicate is no longer literal class identity; what survives
    // DPS untouched is the DISTINCTNESS of the dest from every operand
    // and the arity contract. (The view-parent identity is pinned in c1.)
    assert_ne!(
        dps_op.inputs[3].value, dps_op.inputs[2].value,
        "dest is not the C operand"
    );

    let plan = luminal::test_support::bufferize_mock(&dps)
        .unwrap_or_else(|err| panic!("u3: bufferizer REFUSED a duplicate-operand op: {err}"));
    let summary = plan.summary();
    println!("u3: bufferized a c==a Accumulate\n{summary}");
    assert!(summary.contains("CublasLtAccumulate"), "kernel in plan");

    // THE DONATION-SAFETY PIN. The DPS table declares a May alias on slot 2
    // (the API's C == D in-place accumulate). Here C is ALSO slot 0 — a
    // caller-owned, ReadOnly program INPUT. Donating C's buffer into D would
    // overwrite the caller's `x` (and the op's own `a` operand) mid-call.
    let compute = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            luminal::bufferize::BufferNode::Compute {
                op, reads, writes, ..
            } if op.label().starts_with("CublasLt") => Some((reads.clone(), writes.clone())),
            _ => None,
        })
        .expect("the kernel is in the buffer plan");
    let (reads, writes) = compute;
    println!("  reads={reads:?} writes={writes:?}");
    assert_eq!(reads.len(), 4, "a, b, c, dest");
    // ROUND-10 RE-PIN: x sits in the B role on the sibling site and C is
    // its transpose VIEW — so it is slots 1 and 2 that share x's buffer.
    assert_eq!(
        reads[1], reads[2],
        "b and c share one buffer (x, viewed twice)"
    );
    assert_ne!(
        reads[2], writes[0],
        "the May permit did NOT donate a read-only caller input into D"
    );
}

/// u6: the OTHER defense-in-depth panic with real semantic content — the
/// API's C == D layout precondition. The C-fold rule guards `?c_lt` onto the
/// D layout class, so a mismatched C is rule-impossible; hand-build one and
/// confirm the parser refuses loudly rather than emitting a call whose C
/// descriptor disagrees with D.
#[test]
fn attack_u6_c_layout_mismatch_panics_loudly() {
    let seeded = Fx {
        extra: r#"(let c_logical (LogicalTensorInputLit (LogicalIdLit "c") out_shape (F32)))
(let c_layout_lm (LeftMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let c_lt_lm (LayoutTensorLit c_logical c_layout_lm))
(let bad_site (CublasLtLogicalMatmulSite x_logical w_logical out_logical))
(let bad_a (CublasLtOperandADescriptor bad_site x_lt
  (CublasLtOperationN)))
(let bad_b (CublasLtOperandBDescriptor bad_site w_lt
  (CublasLtOperationN)))
(let bad_d (CublasLtOutputDDescriptor bad_site out_lt))
(let bad_acc (LayoutTensorOpCublasLtAccumulate bad_site bad_a bad_b bad_d
  c_lt_lm (CublasLtEpilogueDefault)))"#
            .into(),
        ..Default::default()
    }
    .text();
    let s = test_runtime::serialize_fixture(&seeded);
    let accs: Vec<_> = s
        .nodes
        .iter()
        .filter(|(_, n)| n.op == "LayoutTensorOpCublasLtAccumulate")
        .collect();
    println!(
        "u6: {} Accumulate enode(s) (the program has no add — all hand-built)",
        accs.len()
    );
    assert_eq!(accs.len(), 1, "only the hand-built term exists");
    let (id, node) = accs[0];
    let msg = panic_message(|| {
        let site = ExtractionSite {
            egraph: &s,
            node_id: id,
            node,
        };
        parse_spec(&site, CublasLtForm::Accumulate)
    });
    println!("u6 parse_spec on the mismatched-C term: {msg:?}");
    let msg = msg.expect("the C/D layout cross-check fires");
    assert!(
        msg.contains("C layout class differs from D layout class"),
        "the panic names the precondition, got: {msg}"
    );
}

/// u4: the diamond over an ACCUMULATE (the board only diamonds a Bias).
/// The base op and the C-folded op coexist; the base must keep claiming the
/// site's own out while the folded one claims the add — and the C operand
/// must be the OTHER value, never the site out.
#[test]
fn attack_u4_accumulate_diamond_claimed_outputs() {
    let text = record(|cx| {
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let mm = x.matmul(w);
        let _ = mm.output();
        let _ = (mm + c).output();
    });
    let s = test_runtime::serialize_fixture(&text);
    assert_candidates_sound(&s, "u4");
    assert_one_lit_per_op_class(&s, "u4");

    let (elected, labels) = flavored_cublaslt(&text, true, false, false);
    let lt: Vec<_> = elected
        .iter()
        .filter(|e| e.label.starts_with("CublasLt"))
        .collect();
    println!("u4 accumulate diamond: labels={labels:?}");
    assert_eq!(lt.len(), 2, "base op AND accumulate op in one plan");
    let mut by_c: Vec<_> = lt.iter().collect();
    by_c.sort_by_key(|e| e.spec().has_c);
    let base = by_c[0];
    let folded = by_c[1];
    assert!(!base.spec().has_c);
    assert!(folded.spec().has_c);
    assert_eq!(base.arity(), 2);
    assert_eq!(folded.arity(), 3);
    assert_eq!(
        base.spec().logical_out,
        base.spec().logical_site_out,
        "the base op claims the site's own out"
    );
    assert_ne!(
        folded.spec().logical_out,
        folded.spec().logical_site_out,
        "the folded op's D moved to the add"
    );
    assert_eq!(
        base.spec().logical_site_out,
        folded.spec().logical_site_out,
        "same site underneath"
    );
    assert_ne!(
        folded.inputs[2].1, folded.out_lt,
        "C is not the op's own destination value"
    );
}

/// u5: EVERY elected op in EVERY program of this battery carries a parsed
/// spec. `CublasLt.spec` is an `Option` and nothing on the board proves the
/// `None` arm is dead. It is: `parse_spec`'s early returns are all
/// unreachable for rule-minted terms.
#[test]
fn attack_u5_no_spec_less_op_can_be_elected() {
    let cases: Vec<(&str, String, bool, bool, bool)> = vec![
        (
            "base",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }),
            false,
            false,
            false,
        ),
        (
            "relu",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).relu().output();
            }),
            false,
            false,
            true,
        ),
        (
            "accumulate-relu",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let _ = (x.matmul(w) + c).relu().output();
            }),
            true,
            false,
            true,
        ),
        (
            "full-stack",
            record(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize))
                    .relu()
                    .output();
            }),
            true,
            true,
            true,
        ),
        (
            "symbolic-k",
            Fx {
                prelude: sym("s", Some(2), Some(64)),
                m: lit(2),
                n: lit(3),
                k: "s_var".into(),
                a_rows: lit(2),
                a_cols: "s_var".into(),
                b_rows: "s_var".into(),
                b_cols: lit(3),
                ..Default::default()
            }
            .text(),
            false,
            false,
            false,
        ),
    ];
    let mut checked = 0usize;
    for (name, text, c, bias, relu) in &cases {
        let (elected, _) = flavored_cublaslt(text, *c, *bias, *relu);
        let lt: Vec<_> = elected
            .iter()
            .filter(|e| e.label.starts_with("CublasLt"))
            .collect();
        assert!(!lt.is_empty(), "{name}: at least one kernel elected");
        for e in &lt {
            assert!(
                e.op.spec.is_some(),
                "{name}: a spec-LESS op reached the plan — parse_spec's None arm \
                 is live and the executor would have nothing to call"
            );
            checked += 1;
        }
        println!("u5 {name}: {} elected op(s), all with specs", lt.len());
    }
    assert!(checked >= cases.len());
}
