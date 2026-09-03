//! E2 evidence: (1) what depth the relu dance's constant fills arrive at
//! from the LIVE recorder; (2) whether the preamble's composition rules
//! canonicalize a hand-nested apply-of-constant into one level. Together
//! these justify inlining the ONE-LEVEL fill pattern into the dance
//! premises and deleting the constant-fill relation.

use luminal::dtype::DType;
use luminal::graph::Graph;

#[test]
fn e2_fill_depth_from_live_recorder() {
    for (name, text) in [
        ("relu", {
            let mut cx = Graph::new();
            let x = cx.tensor((4usize, 8usize), DType::F32);
            let w = cx.tensor((8usize, 3usize), DType::F32);
            let _ = x.matmul(w).relu().output();
            cx.logical
                .bound_program(&test_runtime::TestRuntimeBindings)
                .expect("recorder clean")
                .text
        }),
        ("bias+relu", {
            let mut cx = Graph::new();
            let x = cx.tensor((4usize, 8usize), DType::F32);
            let w = cx.tensor((8usize, 3usize), DType::F32);
            let b = cx.tensor(3usize, DType::F32);
            let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).relu().output();
            cx.logical
                .bound_program(&test_runtime::TestRuntimeBindings)
                .expect("recorder clean")
                .text
        }),
    ] {
        let s = test_runtime::serialize_fixture(&text);
        let const_classes: std::collections::BTreeSet<_> = s
            .nodes
            .values()
            .filter(|n| n.op == "LogicalConstant")
            .map(|n| n.eclass.clone())
            .collect();
        let mut depth1 = 0usize;
        let mut apply_of_const = std::collections::BTreeSet::new();
        for n in s.nodes.values() {
            if n.op == "LogicalIndexMapApply" {
                if let Some(src) = n.children.first().and_then(|id| s.nodes.get(id)) {
                    if const_classes.contains(&src.eclass) {
                        depth1 += 1;
                        apply_of_const.insert(n.eclass.clone());
                    }
                }
            }
        }
        let mut depth2 = 0usize;
        for n in s.nodes.values() {
            if n.op == "LogicalIndexMapApply" {
                if let Some(src) = n.children.first().and_then(|id| s.nodes.get(id)) {
                    if apply_of_const.contains(&src.eclass) {
                        depth2 += 1;
                    }
                }
            }
        }
        println!("E2 {name}: depth-1 fill applies = {depth1}, depth-2+ = {depth2}");
        assert!(depth1 >= 1, "{name}: fills exist at depth 1");
        assert_eq!(depth2, 0, "{name}: the recorder never nests fills");
    }
}

#[test]
fn e2_composition_canonicalization_probe() {
    let fx = r#"(let out_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let vec_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let scalar_shape (ShapeLit (IntExprNil)))
(let scalar_map (IndexMapLit (IntExprNil) scalar_shape))
(let vec_map (IndexMapLit (IntExprCons (CoordVar out_shape 0) (IntExprNil)) vec_shape))
(let zc (LogicalConstant 0.0))
(let level1 (LogicalIndexMapApply zc scalar_map vec_shape))
(let level2 (LogicalIndexMapApply level1 vec_map out_shape))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") out_shape (F32)))
(let probe (LogicalAdd x_logical level2))
(let x_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let probe_lt (LayoutTensorLit probe x_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 11))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let out_buffer_tensor (BufferTensorLit probe_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
    let s = test_runtime::serialize_fixture(fx);
    let const_classes: std::collections::BTreeSet<_> = s
        .nodes
        .values()
        .filter(|n| n.op == "LogicalConstant")
        .map(|n| n.eclass.clone())
        .collect();
    let mut nested_class = None;
    for n in s.nodes.values() {
        if n.op != "LogicalIndexMapApply" {
            continue;
        }
        if let Some(src) = n.children.first().and_then(|id| s.nodes.get(id)) {
            if src.op == "LogicalIndexMapApply" {
                if let Some(inner) = src.children.first().and_then(|id| s.nodes.get(id)) {
                    if const_classes.contains(&inner.eclass) {
                        nested_class = Some(n.eclass.clone());
                    }
                }
            }
        }
    }
    let one_level = s
        .nodes
        .values()
        .filter(|n| {
            n.op == "LogicalIndexMapApply"
                && n.children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|src| const_classes.contains(&src.eclass))
                    .unwrap_or(false)
        })
        .count();
    match nested_class {
        Some(_) => println!("E2 composition probe: nested spelling SURVIVES; {one_level} one-level"),
        None => println!(
            "E2 composition probe: NESTED SPELLING GONE (canonicalized away); {one_level} one-level apply-of-const spelling(s) remain"
        ),
    }
    assert!(one_level >= 1, "the canonical one-level form exists");
}
