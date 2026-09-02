//! ROUND-10 scratch debug probe (not a gate).
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};
use test_runtime::cublaslt_marker::CublasLt;

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

#[test]
fn r10_debug_fixture1() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((2usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 3usize), DType::F32);
        let _out = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let s = test_runtime::serialize_fixture(&text);
    println!("nodes: {}", s.nodes.len());
    for n in s.nodes.values() {
        if n.op == "CublasLtLogicalMatmulSite" {
            println!(
                "SITE class {} a={:?} b={:?} out={:?}",
                n.eclass,
                class_of_child(&s, n, 0),
                class_of_child(&s, n, 1),
                class_of_child(&s, n, 2)
            );
        }
    }
    for n in s.nodes.values() {
        if n.op == "CublasLtOperandADescriptor" || n.op == "CublasLtOperandBDescriptor" {
            let op = class_of_child(&s, n, 2)
                .map(|c| ops_in(&s, &c))
                .unwrap_or_default();
            println!(
                "{} class {} site={:?} lt={:?} op={op:?}",
                n.op,
                n.eclass,
                class_of_child(&s, n, 0),
                class_of_child(&s, n, 1)
            );
        }
        if n.op == "CublasLtOutputDDescriptor" {
            println!(
                "{} class {} site={:?} lt={:?}",
                n.op,
                n.eclass,
                class_of_child(&s, n, 0),
                class_of_child(&s, n, 1)
            );
        }
    }
    for n in s.nodes.values() {
        if n.op.starts_with("LayoutTensorOpCublasLt") {
            println!("OP {} class {}", n.op, n.eclass);
        }
    }
    // Which op constructors produce which Lit output classes?
    for n in s.nodes.values() {
        if n.op == "LayoutTensorOpLit" {
            let outs = class_of_child(&s, n, 1);
            let out_head = outs
                .as_ref()
                .and_then(|c| {
                    s.nodes
                        .values()
                        .find(|m| &m.eclass == c && m.op == "LayoutTensorCons")
                })
                .and_then(|m| class_of_child(&s, m, 0));
            let op_spellings = ops_in(&s, &n.eclass);
            println!(
                "OPLIT class {} out_head={:?} spellings={:?}",
                n.eclass, out_head, op_spellings
            );
        }
    }
    // The boundary lt (out, RM): which class is it, and does the
    // transpose-roundtrip view lt weld into it?
    for n in s.nodes.values() {
        if n.op == "BufferOutputLit" {
            println!("BufferOutputLit class {}", n.eclass);
        }
    }
    let (graph, _) = test_runtime::extract_fixture_with_genome(&text, PIN);
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::LayoutOp(op) => {
                println!("PLAN op: {}", op.op.label());
                if let Some(c) = (*op.op).as_any().downcast_ref::<CublasLt>() {
                    if let Some(spec) = &c.spec {
                        println!(
                            "  spec m={} n={} k={} ta={} tb={} lda={} ldb={} ldd={} a_buf={:?} b_buf={:?} d_buf={:?}",
                            spec.m, spec.n, spec.k, spec.trans_a, spec.trans_b,
                            spec.lda, spec.ldb, spec.ldd,
                            spec.desc_a_buffer, spec.desc_b_buffer, spec.d_buffer
                        );
                    } else {
                        println!("  spec: None");
                    }
                }
            }
            other => println!("PLAN other: {other:?}"),
        }
    }
}

#[test]
fn r10_debug_c6() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _ = ((x.matmul(w) * 2.0) + c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    // PIN-preferring genome: does it reach the boundary?
    let r = std::panic::catch_unwind(|| test_runtime::extract_fixture_with_genome(&text, PIN));
    println!("PIN genome: ok={}", r.is_ok());
    // Plain deterministic min-cost extraction:
    let s = test_runtime::serialize_fixture(&text);
    let graph = luminal::extractor::extract_layout_ir_with_matchers(&s, test_runtime::matchers());
    println!(
        "plain extraction: {:?}",
        graph.as_ref().map(|g| g.is_some())
    );
    // Walk the genome choices from the boundary lt: which class dies?
    let index = luminal::extractor::producer_index_with_matchers(&s, test_runtime::matchers());
    let genome = test_runtime::genome_preferring(&s, PIN);
    fn lit_inputs_of_op_class(s: &EGraph, op_class: &ClassId) -> Vec<Vec<ClassId>> {
        let mut lists = Vec::new();
        for n in s.nodes.values() {
            if n.eclass == *op_class && n.op == "LayoutTensorOpLit" {
                // walk the cons spine of child 0
                let mut items = Vec::new();
                let mut cur = n
                    .children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass.clone());
                let mut guard = 0;
                while let Some(list) = cur {
                    guard += 1;
                    if guard > 16 {
                        break;
                    }
                    if s.nodes
                        .values()
                        .any(|m| m.eclass == list && m.op == "LayoutTensorNil")
                    {
                        break;
                    }
                    let Some(cons) = s
                        .nodes
                        .values()
                        .find(|m| m.eclass == list && m.op == "LayoutTensorCons")
                    else {
                        break;
                    };
                    if let Some(h) = cons.children.first().and_then(|id| s.nodes.get(id)) {
                        items.push(h.eclass.clone());
                    }
                    cur = cons
                        .children
                        .get(1)
                        .and_then(|id| s.nodes.get(id))
                        .map(|c| c.eclass.clone());
                }
                lists.push(items);
            }
        }
        lists
    }
    fn walk(
        s: &EGraph,
        index: &std::collections::BTreeMap<
            ClassId,
            Vec<(String, luminal::extractor::ProducerChoice)>,
        >,
        genome: &luminal::extractor::Genome,
        class: &ClassId,
        path: &mut Vec<ClassId>,
        terminals: &std::collections::BTreeSet<ClassId>,
    ) {
        if terminals.contains(class) {
            return;
        }
        if path.contains(class) {
            println!("CYCLE at {class}: path={path:?}");
            return;
        }
        // describe the class
        for n in s.nodes.values() {
            if n.eclass == *class && n.op == "LayoutTensorLit" {
                let lg = n
                    .children
                    .first()
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass.clone());
                let ly = n
                    .children
                    .get(1)
                    .and_then(|id| s.nodes.get(id))
                    .map(|c| c.eclass.clone());
                println!("  lt {class}: logical={lg:?} layout={ly:?}");
                break;
            }
        }
        let Some(cands) = index.get(class) else {
            println!(
                "DEAD END: class {class} has NO viable producers (path tail {:?})",
                path.last()
            );
            return;
        };
        let Some(choice) = genome.choices.get(class) else {
            println!("NO GENOME CHOICE for {class}");
            return;
        };
        let Some((_, chosen)) = cands.iter().find(|(_, c)| c.enode == choice.enode) else {
            println!(
                "CHOICE NOT IN VIABLE CANDIDATES for {class}: {:?}",
                choice.enode
            );
            return;
        };
        let op_class = s
            .nodes
            .get(&chosen.enode)
            .map(|n| n.eclass.clone())
            .unwrap();
        let op_name = s.nodes.get(&chosen.enode).map(|n| n.op.clone()).unwrap();
        println!("  class {class} chosen {op_name}");
        path.push(class.clone());
        for inputs in lit_inputs_of_op_class(s, &op_class) {
            for input in inputs {
                if path.len() < 40 {
                    let _ = &op_name;
                    walk(s, index, genome, &input, path, terminals);
                }
            }
        }
        path.pop();
    }
    let mut terminals: std::collections::BTreeSet<ClassId> = Default::default();
    let mut boundary_lts: Vec<ClassId> = Vec::new();
    for n in s.nodes.values() {
        if n.op == "BufferTensorLit" {
            if let Some(lt) = n.children.first().and_then(|id| s.nodes.get(id)) {
                // input terminals: ReadOnly buffers
                boundary_lts.push(lt.eclass.clone());
            }
        }
    }
    // crude: treat lts of LogicalTensorInputLit values as terminals
    for lt_class in &boundary_lts {
        for n in s.nodes.values() {
            if n.eclass == *lt_class && n.op == "LayoutTensorLit" {
                if let Some(lg) = n.children.first().and_then(|id| s.nodes.get(id)) {
                    if s.nodes
                        .values()
                        .any(|m| m.eclass == lg.eclass && m.op == "LogicalTensorInputLit")
                    {
                        terminals.insert(lt_class.clone());
                    }
                }
            }
        }
    }
    for lt in &boundary_lts {
        if terminals.contains(lt) {
            continue;
        }
        println!("=== walking boundary lt {lt} ===");
        let mut path = Vec::new();
        walk(&s, &index, &genome, lt, &mut path, &terminals);
    }
}

#[test]
fn r10_debug_a4() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w1 = cx.tensor((4usize, 4usize), DType::F32);
        let w2 = cx.tensor((4usize, 4usize), DType::F32);
        let y = x.matmul(w1);
        let _ = y.matmul(w2).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (graph, _) = test_runtime::extract_fixture_with_genome(&text, PIN);
    for node in graph.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            let outs: Vec<String> = op
                .outputs
                .iter()
                .map(|o| format!("{} (logical {})", o.eclass, o.logical.eclass))
                .collect();
            println!("PLAN {}: ins={ins:?} outs={outs:?}", op.op.label());
            if let Some(c) = (*op.op).as_any().downcast_ref::<CublasLt>() {
                if let Some(spec) = &c.spec {
                    println!(
                        "   spec m={} n={} k={} ta={} tb={} logical_a={} logical_b={} site_out={}",
                        spec.m,
                        spec.n,
                        spec.k,
                        spec.trans_a,
                        spec.trans_b,
                        spec.logical_a,
                        spec.logical_b,
                        spec.logical_site_out
                    );
                }
            }
        }
    }
}

#[test]
fn r10_debug_g2_mincost() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let y = x.matmul(w);
        let _ = y.output();
        let _ = (y * 2.0).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let graph = test_runtime::extract_fixture(&text);
    let labels: Vec<String> = graph
        .dag
        .node_weights()
        .filter_map(|n| match n {
            ExtractedNode::LayoutOp(op) => Some(op.op.label().to_string()),
            _ => None,
        })
        .collect();
    println!("g2 min-cost labels: {labels:?}");
}

#[test]
fn r10_debug_p1() {
    // replicate p1's seeded fixture and print the matched op enodes
    let fx_text = {
        // Fx::default() equivalent: hand 2D A[m,k],B[k,n] matmul x[2,4] w[4,3]
        let sched = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";
        let base = format!(
            r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map (IndexMapLit (IntExprCons (CoordVar prod_shape 2) (IntExprCons (CoordVar prod_shape 0) (IntExprNil))) a_shape))
(let w_to_prod_map (IndexMapLit (IntExprCons (CoordVar prod_shape 0) (IntExprCons (CoordVar prod_shape 1) (IntExprNil))) b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
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
(let bad_site (CublasLtLogicalMatmulSite x_logical w_logical out_logical))
(let bad_desc_a (CublasLtOperandADescriptor bad_site out_lt (CublasLtOperationN)))
(let bad_desc_b (CublasLtOperandBDescriptor bad_site x_lt (CublasLtOperationN)))
(let bad_desc_d (CublasLtOutputDDescriptor bad_site out_lt))
(let bad_op (LayoutTensorOpCublasLt bad_site bad_desc_a bad_desc_b bad_desc_d (CublasLtEpilogueDefault)))
{sched}
"#
        );
        base
    };
    let s = test_runtime::serialize_fixture(&fx_text);
    for (id, n) in s.nodes.iter() {
        if n.op == "LayoutTensorOpCublasLt" {
            let kids: Vec<String> = n
                .children
                .iter()
                .filter_map(|c| s.nodes.get(c))
                .map(|c| c.eclass.to_string())
                .collect();
            println!("op enode {id} class {}: kids={kids:?}", n.eclass);
        }
        if n.op == "CublasLtOperandADescriptor" {
            let kids: Vec<String> = n
                .children
                .iter()
                .filter_map(|c| s.nodes.get(c))
                .map(|c| c.eclass.to_string())
                .collect();
            println!("descA enode class {}: kids={kids:?}", n.eclass);
        }
    }
}

#[test]
fn r10_debug_rc3() {
    let sched = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map (IndexMapLit (IntExprCons (CoordVar prod_shape 2) (IntExprCons (CoordVar prod_shape 0) (IntExprNil))) a_shape))
(let w_to_prod_map (IndexMapLit (IntExprCons (CoordVar prod_shape 0) (IntExprCons (CoordVar prod_shape 1) (IntExprNil))) b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_lt (LayoutTensorLit x_logical (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32)))))
(let w_lt (LayoutTensorLit w_logical (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32)))))
(let out_lt0 (LayoutTensorLit out_logical (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32)))))
(let pitched (StridedElementLayoutLit out_shape (IntAffineExprCons (IntMul (CoordVar out_shape 1) (IntLit 8)) (IntAffineExprCons (CoordVar out_shape 0) (IntAffineExprNil))) (bits-of (F32))))
(let out_lt1 (LayoutTensorLit out_logical pitched))
(set (injectivity-of out_lt1) (Injective))
(strided-lists pitched out_shape
  (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))
(let x_buf (BufferLit 10))
(set (buffer-access-of x_buf) (ReadOnly))
(set (buffer-freed-by x_buf) (CallerFrees))
(let w_buf (BufferLit 11))
(set (buffer-access-of w_buf) (ReadOnly))
(set (buffer-freed-by w_buf) (CallerFrees))
(let o_buf (BufferLit 12))
(set (buffer-access-of o_buf) (ReadWrite))
(set (buffer-freed-by o_buf) (CallerFrees))
(let x_bt (BufferTensorLit x_lt x_buf))
(let w_bt (BufferTensorLit w_lt w_buf))
(let o_bt (BufferTensorLit out_lt0 o_buf))
(let output (BufferOutputLit (BufferTensorCons o_bt (BufferTensorNil))))
{sched}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    // find inner_out: the logical that is an apply of out_logical
    let out_class = s
        .nodes
        .values()
        .find(|n| n.op == "LogicalReduceSum")
        .map(|n| n.eclass.clone())
        .unwrap();
    // out class contains the reduce; inner_out = class containing apply(out,...)
    for n in s.nodes.values() {
        if n.op == "LogicalIndexMapApply" {
            let parent = n
                .children
                .first()
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone());
            if parent.as_ref() == Some(&out_class) && n.eclass != out_class {
                println!("inner_out class {} (apply of out)", n.eclass);
                let inner = n.eclass.clone();
                // its layout tensors:
                for m in s.nodes.values() {
                    if m.op == "LayoutTensorLit" {
                        let lg = m
                            .children
                            .first()
                            .and_then(|id| s.nodes.get(id))
                            .map(|c| c.eclass.clone());
                        if lg.as_ref() == Some(&inner) {
                            let ly = m
                                .children
                                .get(1)
                                .and_then(|id| s.nodes.get(id))
                                .map(|c| c.eclass.clone())
                                .unwrap();
                            let ops = ops_in(&s, &ly);
                            // injectivity?
                            let inj = s.nodes.values().any(|f| {
                                f.op == "injectivity-of"
                                    && f.children
                                        .first()
                                        .and_then(|id| s.nodes.get(id))
                                        .map(|c| c.eclass == m.eclass)
                                        .unwrap_or(false)
                            });
                            println!(
                                "  lt class {} layout {} inj-fact={} spellings={:?}",
                                m.eclass, ly, inj, ops
                            );
                        }
                    }
                }
            }
        }
    }
    for n in s.nodes.values() {
        if n.op == "CublasLtOutputDDescriptor" {
            let lt = n
                .children
                .get(1)
                .and_then(|id| s.nodes.get(id))
                .map(|c| c.eclass.clone());
            println!("D reading lt={lt:?}");
        }
    }
    let d = s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtOutputDDescriptor")
        .count();
    println!("D readings: {d}");
}

#[test]
fn r10_debug_bufferize() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _ = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (graph, _) = test_runtime::extract_fixture_with_genome(
        &text,
        &[
            "LayoutTensorOpCublasLt",
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
    let dps = luminal::dps::dps_rewrite(&graph);
    for node in dps.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            println!("DPS {}: ins={ins:?}", op.op.label());
        }
    }
    // ESCAPE-AND-DISCLOSE (ruling 2026-08-27): mm's boundary value is the
    // sibling transpose VIEW of the kernel result. The kernel's scratch
    // alloc ESCAPES — the slot is backed by it (FreedBy::Caller, no free),
    // zero copies — and the binding discloses the weld's layout for the
    // caller to interpret the bytes under.
    let plan = luminal::test_support::bufferize_mock(&dps).expect("the view output escapes");
    println!("{}", plan.summary());
    use luminal::bufferize::{BufferId, BufferNode};
    assert!(
        !plan
            .dag
            .node_indices()
            .any(|i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. })),
        "zero boundary copies — the alloc itself is handed over:\n{}",
        plan.summary()
    );
    let slot = plan
        .dag
        .node_weights()
        .find_map(|node| match node {
            BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
            _ => None,
        })
        .expect("one output slot");
    assert!(
        matches!(slot.buffer, BufferId::Allocated(_)),
        "the slot is backed by the escaping kernel alloc:\n{}",
        plan.summary()
    );
    assert_eq!(
        plan.buffers[&slot.buffer].freed_by,
        luminal::layout_ir::FreedBy::Caller,
        "the backing buffer escapes to the caller"
    );
    assert_eq!(
        Some(&slot.layout),
        luminal::test_support::mock_layout_table(&dps).get(&slot.value),
        "the binding discloses the slot value's own elected (weld) layout"
    );
}
