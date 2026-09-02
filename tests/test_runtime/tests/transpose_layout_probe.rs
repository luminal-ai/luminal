//! PROBE (read-only analysis): does the existing views/layout machinery
//! fold a transpose VIEW into a LAYOUT, and what exactly does it produce?
//!
//! Run by name with --nocapture.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};

fn ops_in(s: &EGraph, class: &ClassId) -> Vec<String> {
    let mut v: Vec<String> = s
        .nodes
        .values()
        .filter(|n| &n.eclass == class)
        .map(|n| n.op.clone())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Pretty-print an IntExprList / ShapeLit class as a dim vector, best effort.
fn render(s: &EGraph, class: &ClassId, depth: usize) -> String {
    if depth > 24 {
        return "…".into();
    }
    // prefer the SMALLEST spelling in the class (an IntAdd zero-tower and a
    // bare IntMul can be the same class; show the readable one)
    let Some(node) = s
        .nodes
        .values()
        .filter(|n| &n.eclass == class)
        .min_by_key(|n| (n.op == "IntAdd") as usize * 100 + n.children.len())
    else {
        return "?".into();
    };
    let kids: Vec<String> = node
        .children
        .iter()
        .filter_map(|id| s.nodes.get(id))
        .map(|c| render(s, &c.eclass, depth + 1))
        .collect();
    if kids.is_empty() {
        node.op.clone()
    } else {
        format!("({} {})", node.op, kids.join(" "))
    }
}

/// Every LayoutTensorLit whose logical child is `logical_class`, with the
/// full op-name set of its layout class.
fn dump_layouts_for(s: &EGraph, label: &str, logical_class: &ClassId) {
    println!("--- layouts for {label} (class {logical_class}) ---");
    let mut seen: Vec<ClassId> = Vec::new();
    for n in s.nodes.values() {
        if n.op != "LayoutTensorLit" {
            continue;
        }
        let Some(lg) = n.children.first().and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        if &lg.eclass != logical_class {
            continue;
        }
        let Some(ly) = n.children.get(1).and_then(|id| s.nodes.get(id)) else {
            continue;
        };
        if seen.contains(&ly.eclass) {
            continue;
        }
        seen.push(ly.eclass.clone());
        println!("  layout class {}:", ly.eclass);
        for op in ops_in(s, &ly.eclass) {
            println!("     * {op}");
        }
        // render the strided chain if there is one
        let mut shown: Vec<ClassId> = Vec::new();
        for m in s.nodes.values() {
            if m.eclass == ly.eclass && m.op == "StridedElementLayoutLit" {
                if let Some(ch) = m.children.get(1).and_then(|id| s.nodes.get(id)) {
                    if shown.contains(&ch.eclass) {
                        continue;
                    }
                    shown.push(ch.eclass.clone());
                    println!("     chain = {}", render(s, &ch.eclass, 0));
                }
            }
            if m.eclass == ly.eclass
                && (m.op == "LeftMajorContiguousElementLayoutLit"
                    || m.op == "RightMajorContiguousElementLayoutLit")
            {
                if let Some(sh) = m.children.first().and_then(|id| s.nodes.get(id)) {
                    println!("     {} over shape {}", m.op, render(s, &sh.eclass, 0));
                }
            }
        }
        // which buffers does this (logical, layout) pair sit in?
        for bt in s.nodes.values() {
            if bt.op != "BufferTensorLit" {
                continue;
            }
            let Some(lt) = bt.children.first().and_then(|id| s.nodes.get(id)) else {
                continue;
            };
            if lt.eclass != n.eclass {
                continue;
            }
            let buf = bt
                .children
                .get(1)
                .and_then(|id| s.nodes.get(id))
                .map(|c| render(s, &c.eclass, 0))
                .unwrap_or_default();
            println!("     BufferTensorLit -> {buf}");
        }
    }
    if seen.is_empty() {
        println!("  (no LayoutTensorLit found for this logical)");
    }
}

fn logical_apply_classes(s: &EGraph) -> Vec<(ClassId, String)> {
    let mut out: Vec<(ClassId, String)> = Vec::new();
    for n in s.nodes.values() {
        if n.op != "LogicalIndexMapApply" {
            continue;
        }
        if out.iter().any(|(c, _)| c == &n.eclass) {
            continue;
        }
        let shape = n
            .children
            .get(2)
            .and_then(|id| s.nodes.get(id))
            .map(|c| render(s, &c.eclass, 0))
            .unwrap_or_default();
        let map = n
            .children
            .get(1)
            .and_then(|id| s.nodes.get(id))
            .map(|c| render(s, &c.eclass, 0))
            .unwrap_or_default();
        out.push((n.eclass.clone(), format!("shape={shape}\n      map={map}")));
    }
    out
}

/// FIXTURE A: a bare transpose view, kept as its own value.
#[test]
fn probe_a_bare_transpose_view() {
    let mut cx = Graph::new();
    let w = cx.tensor((4usize, 3usize), DType::F32); // stored row-major [4,3]
    let _t = w.permute((1usize, 0usize)).output(); // [3,4] view
    let program = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    println!("=== FIXTURE A native_program (w[4,3].permute((1,0)).output()) ===");
    println!("{program}");

    let s = test_runtime::serialize_fixture(&program);
    println!("=== FIXTURE A: {} nodes ===", s.nodes.len());
    for (class, desc) in logical_apply_classes(&s) {
        println!("LogicalIndexMapApply class {class}: {desc}");
        dump_layouts_for(&s, "the view", &class);
    }
    let count = |op: &str| s.nodes.values().filter(|n| n.op == op).count();
    println!(
        "op census: Right={} Left={} Strided={} ElementOffset={} BitOffset={} Copy-ish={}",
        count("RightMajorContiguousElementLayoutLit"),
        count("LeftMajorContiguousElementLayoutLit"),
        count("StridedElementLayoutLit"),
        count("ElementOffsetExpressionLayoutLit"),
        count("BitOffsetExpressionLayoutLit"),
        s.nodes
            .values()
            .filter(|n| n.op.contains("Copy") || n.op.contains("Materialize"))
            .count()
    );
    for n in s.nodes.values() {
        if n.op == "LayoutTensorOpIndexMapApplyViewGeneric" {
            let out_layout = n
                .children
                .get(3)
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone());
            println!(
                "VIEW OP: out_layout class {:?} -> ops {:?}",
                out_layout,
                out_layout.as_ref().map(|c| ops_in(&s, c))
            );
        }
        if n.op == "LayoutTensorOpIndexMapApplyMaterialize" {
            let out_layout = n
                .children
                .get(3)
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone());
            println!(
                "MATERIALIZE OP: out_layout class {:?} -> ops {:?}",
                out_layout,
                out_layout.as_ref().map(|c| ops_in(&s, c))
            );
        }
    }
    let mut opnames: Vec<String> = s
        .nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOp"))
        .map(|n| n.op.clone())
        .collect();
    opnames.sort();
    opnames.dedup();
    println!("LayoutTensorOp constructors present: {opnames:?}");
}

/// FIXTURE B: the matmul folded-permute case A[m,k],B[n,k] — the permute folds into the apply.
#[test]
fn probe_b_matmul_amk_bnk() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((3usize, 4usize), DType::F32); // stored [n,k]
    let _out = x.matmul(w.permute((1usize, 0usize))).output();
    let program = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    let s = test_runtime::serialize_fixture(&program);
    println!("=== FIXTURE B: {} nodes ===", s.nodes.len());
    for (class, desc) in logical_apply_classes(&s) {
        println!("LogicalIndexMapApply class {class}: {desc}");
        dump_layouts_for(&s, "broadcast view", &class);
    }
    // the STORED w's own layout classes
    for n in s.nodes.values() {
        if n.op == "LogicalTensorInputLit" {
            let id = n
                .children
                .first()
                .and_then(|c| s.nodes.get(c))
                .map(|c| c.op.clone())
                .unwrap_or_default();
            println!("input {id} class {}", n.eclass);
            dump_layouts_for(&s, &format!("input {id}"), &n.eclass);
        }
    }
}
