//! ROUND-9 EVIDENCE PROBE (the separation): the census the layout-native
//! reading arms are built on.
//!
//! Dumps every LayoutTensorLit and every strided chain the r8d shared-
//! square-weight fixture derives. Two facts it establishes, both quoted in
//! src/egg/cublaslt_marker_desc.egg:
//!
//!  1. The b-side BROADCASTS carry composed layouts even in a hand-written
//!     fixture (no frontend involved). The B[k,n] broadcast's chain is
//!     [0, c1, 3*c0] (unit stride on n) and the B[n,k] broadcast's is
//!     [0, 3*c1, c0] (unit stride on k). That single bit IS the operation.
//!
//!  2. Every one of those broadcasts ALSO carries a blanket right-major
//!     layout over the PRODUCT shape (chain [9*c2, 3*c1, c0]), minted
//!     unconditionally by egglog_preamble.egg:4281-4291. Reading the
//!     operation off THAT layout while pointing the descriptor at the
//!     operand's own buffer is the unsoundness the composition tie exists
//!     to prevent — see the revert-probe in the round-9 report.

use luminal::prelude::egraph_serialize::{ClassId, EGraph};

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

fn fixture() -> String {
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

fn render(s: &EGraph, class: &ClassId, depth: usize) -> String {
    if depth == 0 {
        return format!("<{class:?}>");
    }
    let mut parts: Vec<String> = Vec::new();
    for n in s.nodes.values().filter(|n| &n.eclass == class) {
        let kids: Vec<String> = n
            .children
            .iter()
            .filter_map(|id| s.nodes.get(id))
            .map(|c| render(s, &c.eclass, depth - 1))
            .collect();
        parts.push(if kids.is_empty() {
            n.op.clone()
        } else {
            format!("({} {})", n.op, kids.join(" "))
        });
        if parts.len() > 6 {
            parts.push("...".into());
            break;
        }
    }
    format!("[{}]", parts.join(" | "))
}

#[test]
fn r9_layout_census_for_the_shared_weight_fixture() {
    let s = test_runtime::serialize_fixture(&fixture());
    println!("nodes = {}", s.nodes.len());

    // Every LayoutTensorLit, grouped by its logical child.
    let mut rows: Vec<(String, String)> = Vec::new();
    for n in s.nodes.values().filter(|n| n.op == "LayoutTensorLit") {
        let logical = class_of_child(&s, n, 0).unwrap();
        let layout = class_of_child(&s, n, 1).unwrap();
        let logical_ops: Vec<String> = s
            .nodes
            .values()
            .filter(|m| m.eclass == logical)
            .map(|m| m.op.clone())
            .collect();
        let layout_ops: Vec<String> = s
            .nodes
            .values()
            .filter(|m| m.eclass == layout)
            .map(|m| m.op.clone())
            .collect();
        rows.push((
            format!("{logical:?} {logical_ops:?}"),
            format!("{layout:?} {layout_ops:?}"),
        ));
    }
    rows.sort();
    rows.dedup();
    for (l, ly) in &rows {
        println!("LT: logical {l}\n     layout {ly}");
    }

    // Any left-major anywhere?
    let lm = s
        .nodes
        .values()
        .filter(|n| n.op == "LeftMajorContiguousElementLayoutLit")
        .count();
    println!("LeftMajorContiguousElementLayoutLit nodes = {lm}");

    // Strided chains present, entry-by-entry, ONE-LEVEL rendering only.
    let mut seen: Vec<ClassId> = Vec::new();
    for n in s
        .nodes
        .values()
        .filter(|n| n.op == "StridedElementLayoutLit")
    {
        if seen.contains(&n.eclass) {
            continue;
        }
        seen.push(n.eclass.clone());
        let chain = class_of_child(&s, n, 1).unwrap();
        let mut cur = chain;
        let mut entries = Vec::new();
        loop {
            if s.nodes
                .values()
                .any(|m| m.eclass == cur && m.op == "IntAffineExprNil")
            {
                break;
            }
            let Some(cons) = s
                .nodes
                .values()
                .find(|m| m.eclass == cur && m.op == "IntAffineExprCons")
            else {
                entries.push("<stall>".to_string());
                break;
            };
            let e = class_of_child(&s, cons, 0).unwrap();
            entries.push(render(&s, &e, 2));
            cur = class_of_child(&s, cons, 1).unwrap();
            if entries.len() > 5 {
                break;
            }
        }
        println!(
            "STRIDED layout {:?}\n  chain: {}",
            n.eclass,
            entries.join("\n         ")
        );
    }
}
