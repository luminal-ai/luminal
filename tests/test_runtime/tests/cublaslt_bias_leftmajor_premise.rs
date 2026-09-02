//! THE LAYOUT HALF of the LeftMajor-D premise proof (ruling 2026-09-01).
//!
//! The bias decorators (`cublaslt_marker_decorate.egg`) mint a bias form
//! only when the sibling D's layout class holds
//! `LeftMajorContiguousElementLayoutLit`. This board asks, on the SAME
//! recorded program (x[4,8] @ w[8,3] + b[3]), what happens when that
//! spelling is present and when it is absent:
//!
//!  1. `contiguous_recorder_output_mints_the_bias_form` — the recorder's
//!     row-major out: the sibling D is left-major, discovery finds it,
//!     the bias form is minted.
//!  2. `padded_output_binding_does_not_remove_the_row_major_route` — the
//!     boundary out2 bound to a PADDED strided layout (ld = n + 1 = 4).
//!     FINDING, recorded rather than wished away: the boundary binding
//!     does NOT gate the mint, because layout tensors flow FORWARD from
//!     the row-major inputs (`src/logical_op/*/forward_layout.egg`: a
//!     ReduceSum out is always minted row-major; Add forwards each
//!     operand's layout to its out), so `y_outer` and `out2` carry a
//!     right-major layout tensor whatever the boundary says, and the
//!     decorator's own `(= ?d_layout (RightMajorContiguousElementLayoutLit
//!     ?d_shape ?d_bits))` premise picks it. The padded binding adds a
//!     second route; it removes none. The test pins the mechanism.
//!  3. `without_left_major_discovery_the_bias_form_is_not_minted` — the
//!     ISOLATING negative: identical program, identical estate, with the
//!     ONE preamble rule that unions a strided chain with the left-major
//!     literal (arm 3a) ablated from the assembled text. The base form
//!     still assembles (its D reading is chain-native and injectivity
//!     rides the transpose transport), the bias form does not. The only
//!     difference between 1 and 3 is whether the LeftMajor spelling
//!     exists in the D class — which is exactly the new premise.

use std::collections::BTreeSet;

use egglog::SerializeConfig;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node};

fn count(s: &EGraph, op: &str) -> usize {
    s.nodes.values().filter(|n| n.op == op).count()
}

fn class_ops(s: &EGraph, class: &ClassId) -> BTreeSet<String> {
    s.nodes
        .values()
        .filter(|n| &n.eclass == class)
        .map(|n| n.op.clone())
        .collect()
}

fn child_class(s: &EGraph, node: &Node, index: usize) -> ClassId {
    s.nodes[&node.children[index]].eclass.clone()
}

fn node_in_class<'a>(s: &'a EGraph, class: &ClassId, op: &str) -> Option<&'a Node> {
    s.nodes.values().find(|n| &n.eclass == class && n.op == op)
}

/// D descriptor at slot 3 -> layout tensor (child 1) -> LayoutTensorLit's
/// layout (child 1).
fn d_layout_class(s: &EGraph, op_node: &Node) -> ClassId {
    let desc_class = child_class(s, op_node, 3);
    let desc = node_in_class(s, &desc_class, "CublasLtOutputDDescriptor").expect("D descriptor");
    let lt_class = child_class(s, desc, 1);
    let lt = node_in_class(s, &lt_class, "LayoutTensorLit").expect("layout tensor");
    child_class(s, lt, 1)
}

fn d_class_ops_for(s: &EGraph, op: &str) -> Vec<BTreeSet<String>> {
    let classes: BTreeSet<ClassId> = s
        .nodes
        .values()
        .filter(|n| n.op == op)
        .map(|n| d_layout_class(s, n))
        .collect();
    classes.iter().map(|c| class_ops(s, c)).collect()
}

/// x[4,8] @ w[8,3] + b[3] broadcast over the batch axis — the recorder's
/// per-feature bias spelling (rank-1 [n] applied through (CoordVar d_shape
/// 0) into [m, n]; `expand_dim(0, 4)` and `Linear`'s `expand_lhs(&[4])`
/// record the same map). Returns the program text and the boundary
/// output's SSA name and slot key.
fn recorded_bias_program() -> (String, String, usize) {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 8usize), DType::F32);
    let w = cx.tensor((8usize, 3usize), DType::F32);
    let b = cx.tensor(3usize, DType::F32);
    let out = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
    let text = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    (text, format!("v{}", out.id.index()), out.id.index())
}

fn report(s: &EGraph, what: &str) -> (usize, usize) {
    let base = count(s, "LayoutTensorOpCublasLt");
    let bias = count(s, "LayoutTensorOpCublasLtBias");
    let acc = count(s, "LayoutTensorOpCublasLtAccumulate");
    let acc_bias = count(s, "LayoutTensorOpCublasLtAccumulateBias");
    println!(
        "LM-PREMISE {what}: base={base} bias={bias} accumulate={acc} accumulate_bias={acc_bias}"
    );
    for ops in d_class_ops_for(s, "LayoutTensorOpCublasLt") {
        println!("LM-PREMISE {what}: base D layout class ops = {ops:?}");
    }
    for ops in d_class_ops_for(s, "LayoutTensorOpCublasLtBias") {
        println!("LM-PREMISE {what}: bias D layout class ops = {ops:?}");
    }
    (base, bias)
}

// ---------------------------------------------------------------------------
// 1. The contiguous case: minted.
// ---------------------------------------------------------------------------

#[test]
fn contiguous_recorder_output_mints_the_bias_form() {
    let (text, _out_name, _key) = recorded_bias_program();
    let s = test_runtime::serialize_fixture(&text);
    let (base, bias) = report(&s, "contiguous");
    assert!(base > 0);
    assert!(
        bias > 0,
        "the row-major recorder out yields a left-major sibling D: minted"
    );
    for ops in d_class_ops_for(&s, "LayoutTensorOpCublasLtBias") {
        assert!(
            ops.contains("LeftMajorContiguousElementLayoutLit"),
            "every bias form's D class holds the left-major spelling: {ops:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The padded boundary binding: the finding.
// ---------------------------------------------------------------------------

/// Rewrite the boundary output's binding from the row-major contiguous
/// spelling to a PADDED strided one (row pitch 4 over 3 columns), with the
/// creator's `strided-lists` row (so the layout climbs the ladder and can
/// be a composition parent) and the creator's injectivity certificate
/// (pitch >= extent: a non-overlapping layout the preamble cannot prove).
fn pad_output_binding(text: &str, key: usize) -> String {
    let stem = format!("natout{key}");
    let prefix = format!("(let {stem}_layout (RightMajorContiguousElementLayoutLit ");
    let suffix = " (bits-of (F32))))";
    let line = text
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("the boundary output binding line for {stem} exists"));
    let shape = &line[prefix.len()..line.len() - suffix.len()];
    let padded = format!(
        "(let {stem}_padded_shape {shape})\n\
         (let {stem}_layout (StridedElementLayoutLit {stem}_padded_shape\n\
         \x20 (IntAffineExprCons (IntMul (CoordVar {stem}_padded_shape 1) (IntLit 4))\n\
         \x20   (IntAffineExprCons (CoordVar {stem}_padded_shape 0) (IntAffineExprNil)))\n\
         \x20 (bits-of (F32))))\n\
         (strided-lists {stem}_layout {stem}_padded_shape\n\
         \x20 (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))\n\
         \x20 (IntExprCons (IntLit 4) (IntExprCons (IntLit 1) (IntExprNil)))\n\
         \x20 (bits-of (F32)))"
    );
    let lt_line = format!("(let {stem}_layout_tensor (LayoutTensorLit ");
    let mut out = String::new();
    let mut replaced = 0;
    for l in text.lines() {
        if l == line {
            out.push_str(&padded);
            out.push('\n');
            replaced += 1;
        } else if l.starts_with(&lt_line) {
            out.push_str(l);
            out.push('\n');
            out.push_str(&format!(
                "(set (injectivity-of {stem}_layout_tensor) (Injective))\n"
            ));
        } else {
            out.push_str(l);
            out.push('\n');
        }
    }
    assert_eq!(replaced, 1, "exactly one boundary layout line rewritten");
    out
}

#[test]
fn padded_output_binding_does_not_remove_the_row_major_route() {
    let (text, out_name, key) = recorded_bias_program();
    let padded = pad_output_binding(&text, key);
    println!("LM-PREMISE padded binding text:\n{}", {
        let stem = format!("natout{key}");
        padded
            .lines()
            .filter(|l| l.contains(&stem))
            .collect::<Vec<_>>()
            .join("\n")
    });
    let s = test_runtime::serialize_fixture(&padded);
    let (base, bias) = report(&s, "padded-boundary");
    assert!(base > 0);

    // THE MECHANISM, read off the e-graph: the boundary value out2 (the
    // recorder's `v{out}`) carries MORE THAN ONE layout tensor — the
    // padded one we bound AND a right-major one the forward-propagation
    // rules minted from the row-major inputs. Find out2's class through
    // the padded layout tensor we authored, then list every layout class
    // any LayoutTensorLit gives it.
    let padded_lt_name_class: Option<ClassId> = {
        // The padded StridedElementLayoutLit we wrote is the ONLY strided
        // [4,3] layout with a literal-4 pitch; its class must NOT carry the
        // right-major spelling (pitch 4 != 3).
        s.nodes
            .values()
            .filter(|n| n.op == "StridedElementLayoutLit")
            .map(|n| n.eclass.clone())
            .find(|c| {
                let ops = class_ops(&s, c);
                !ops.contains("RightMajorContiguousElementLayoutLit")
                    && !ops.contains("LeftMajorContiguousElementLayoutLit")
                    && s.nodes.values().any(|lt| {
                        lt.op == "LayoutTensorLit"
                            && lt.children.len() == 2
                            && s.nodes[&lt.children[1]].eclass == *c
                    })
            })
    };
    let padded_class =
        padded_lt_name_class.expect("the padded layout class exists, distinct from RM/LM");
    println!(
        "LM-PREMISE padded-boundary: padded layout class {padded_class:?} ops = {:?}",
        class_ops(&s, &padded_class)
    );
    // out2's logical class = the logical child of a LayoutTensorLit whose
    // layout is the padded class.
    let out2_class: ClassId = s
        .nodes
        .values()
        .find(|lt| lt.op == "LayoutTensorLit" && s.nodes[&lt.children[1]].eclass == padded_class)
        .map(|lt| s.nodes[&lt.children[0]].eclass.clone())
        .expect("a layout tensor rides the padded layout");
    let out2_layout_classes: BTreeSet<ClassId> = s
        .nodes
        .values()
        .filter(|lt| lt.op == "LayoutTensorLit" && s.nodes[&lt.children[0]].eclass == out2_class)
        .map(|lt| s.nodes[&lt.children[1]].eclass.clone())
        .collect();
    let mut has_rm_route = false;
    for c in &out2_layout_classes {
        let ops = class_ops(&s, c);
        println!("LM-PREMISE padded-boundary: {out_name} layout tensor class {c:?} ops = {ops:?}");
        has_rm_route |= ops.contains("RightMajorContiguousElementLayoutLit");
    }
    assert!(
        has_rm_route,
        "the boundary value still carries a RIGHT-MAJOR layout tensor (forward layout \
         propagation from the row-major inputs) — the route the bias decorator reads"
    );
    assert!(
        bias > 0,
        "and therefore the bias form is STILL minted: a boundary binding cannot remove the \
         forward-propagated right-major route; the premise is gated by discovery, not by the \
         boundary spelling (finding recorded 2026-09-01)"
    );
}

// ---------------------------------------------------------------------------
// 3. The isolating negative: no left-major literal, no bias form.
// ---------------------------------------------------------------------------

/// The ONE preamble rule that unions a strided chain with the left-major
/// literal — arm (3a) of DOWNWARD LAYOUT DISCOVERY
/// (`src/egglog_core/egglog_preamble.egg`, "(3a) contiguous
/// discovery"), quoted verbatim so the ablation cannot drift.
const LEFT_MAJOR_DISCOVERY_RULE: &str = "(rule
  (
    (= ?layout (StridedElementLayoutLit ?shape ?chain ?bits))
    (= ?shape (ShapeLit ?dims))
    (= (left-major-strides-state-of ?dims)
      (LeftMajorStridesFoldStateLit ?rank ?total ?strides))
    (affine-zip-out ?shape ?dims ?strides ?chain)
  )
  (
    (union ?layout (LeftMajorContiguousElementLayoutLit ?shape ?bits))
  )
)";

fn serialize_with_ablated_left_major_discovery(script_text: &str) -> EGraph {
    let preamble = luminal::egglog_snippet::assembled_program_for(&test_runtime::matchers());
    assert_eq!(
        preamble.matches(LEFT_MAJOR_DISCOVERY_RULE).count(),
        1,
        "the (3a) left-major discovery rule appears exactly once in the assembled program"
    );
    let ablated = preamble.replace(
        LEFT_MAJOR_DISCOVERY_RULE,
        "; ABLATED FOR THE LM-PREMISE NEGATIVE: (3a) Strided -> LeftMajorContiguous discovery",
    );
    let program = format!("{ablated}\n\n{script_text}");
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &program)
        .unwrap_or_else(|err| panic!("egglog failed on the ablated program: {err}"));
    egraph.serialize(SerializeConfig::default()).egraph
}

#[test]
fn without_left_major_discovery_the_bias_form_is_not_minted() {
    let (text, _out_name, _key) = recorded_bias_program();
    let s = serialize_with_ablated_left_major_discovery(&text);
    let (base, bias) = report(&s, "ablated-3a");
    assert_eq!(
        count(&s, "LeftMajorContiguousElementLayoutLit"),
        0,
        "with arm 3a ablated nothing mints the left-major literal (the decorator does not)"
    );
    assert!(
        base > 0,
        "the BASE form still assembles: its D reading is chain-native (unit stride on m) and \
         injectivity rides the transpose transport — neither needs the literal"
    );
    assert_eq!(
        bias, 0,
        "NO bias form: the only thing that changed is the LeftMajor spelling's presence in \
         the D class, i.e. the new premise is what gates the mint"
    );
    assert_eq!(count(&s, "LayoutTensorOpCublasLtAccumulateBias"), 0);
}
