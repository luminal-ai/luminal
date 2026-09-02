//! Decoration fixtures (5, 6, 8) — field-rewrite legality across the four
//! runtime contracts, all recorded from the live frontend.

use std::time::Instant;

use luminal::buffer_tensor_ir::OpSlotNames;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use test_runtime::cublaslt_marker::{CuEpilogue, CublasLt, CublasLtForm};

/// Contract-flavored genome. Round 4 made the runtime contract the
/// CONSTRUCTOR NAME, so the pick is name-first; relu (a field, not a
/// contract) still needs the epilogue child at the form's slot.
fn genome_flavored(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    want_c: bool,
    want_bias: bool,
    want_relu: bool,
) -> luminal::extractor::Genome {
    use std::collections::HashMap;
    let mut class_ops: HashMap<&luminal::prelude::egraph_serialize::ClassId, Vec<&str>> =
        HashMap::new();
    for node in egraph.nodes.values() {
        class_ops
            .entry(&node.eclass)
            .or_default()
            .push(node.op.as_str());
    }
    let class_has = |class: &luminal::prelude::egraph_serialize::ClassId, op: &str| {
        class_ops.get(class).is_some_and(|ops| ops.contains(&op))
    };
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
    // ROUND 11: routed through the shared VIABILITY-AWARE election core
    // (test_runtime::genome_with_ordering) — the transpose-view value
    // pairs the canonical-form sandwich mints carry mutually-derived
    // layout tensors whose view producers form 2-cycles of pure
    // re-descriptions, and the old walk-blind name pick deterministically
    // elected them ("outputs with no plan"). Preference ORDER is
    // unchanged: flavored contract > any CublasLt > the transpose view >
    // non-Copy > Copy; the core additionally checks the chosen subtree
    // reaches terminals and escalates its Materialize/Copy strictness
    // only when nothing viable exists below.
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

fn collect_ops(graph: &luminal::layout_ir::ExtractedGraph) -> Vec<(CublasLt, Vec<String>, String)> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) => {
                let label = op.op.label().to_string();
                let concrete = (*op.op).as_any().downcast_ref::<CublasLt>().cloned();
                let inputs = op.inputs.iter().map(|i| i.value.to_string()).collect();
                Some((concrete, inputs, label))
            }
            _ => None,
        })
        .filter_map(|(concrete, inputs, label)| {
            if label.starts_with("CublasLt") {
                concrete.map(|c| (c, inputs, label))
            } else {
                Some((
                    CublasLt {
                        form: CublasLtForm::Base,
                        spec: None,
                    },
                    inputs,
                    label,
                ))
            }
        })
        .collect()
}

fn flavored_ops(
    text: &str,
    want_c: bool,
    want_bias: bool,
    want_relu: bool,
) -> Vec<(CublasLt, Vec<String>, String)> {
    let serialized = test_runtime::serialize_fixture(text);
    let genome = genome_flavored(&serialized, want_c, want_bias, want_relu);
    let graph = luminal::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        test_runtime::matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary");
    collect_ops(&graph)
}

fn cublaslt_only(ops: &[(CublasLt, Vec<String>, String)]) -> Vec<&(CublasLt, Vec<String>, String)> {
    ops.iter()
        .filter(|(_, _, label)| label.starts_with("CublasLt"))
        .collect()
}

fn timed(text: &str) -> f64 {
    let start = Instant::now();
    let _ = test_runtime::serialize_fixture(text);
    start.elapsed().as_secs_f64()
}

// ---------------------------------------------------------------------------
// Fixture 5 — the decoration paths.
// ---------------------------------------------------------------------------

/// relu(x @ w): epilogue Relu, base contract, Lit arity 2.
#[test]
fn fixture5_relu() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let _out = x.matmul(w).relu().output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    println!("MEASURE fixture5 relu: {:.2}s", timed(&text));
    let ops = flavored_ops(&text, false, false, true);
    let lt = cublaslt_only(&ops);
    assert_eq!(
        lt.len(),
        1,
        "one cublaslt op (the decorated one) in the plan"
    );
    let (op, inputs, _) = lt[0];
    let spec = op.spec.as_ref().expect("spec parses");
    assert_eq!(spec.epilogue, CuEpilogue::Relu);
    assert!(!spec.has_c);
    assert!(!spec.has_bias);
    assert_eq!(spec.mnk_lits(), (3, 4, 8));
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
}

/// x @ w + c (C-fold): the Accumulate contract, Lit arity 3, ldc = ldd.
#[test]
fn fixture5_c_fold() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _out = (x.matmul(w) + c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    println!("MEASURE fixture5 c-fold: {:.2}s", timed(&text));
    let ops = flavored_ops(&text, true, false, false);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let (op, inputs, label) = lt[0];
    assert_eq!(label, "CublasLtAccumulate");
    let spec = op.spec.as_ref().expect("spec parses");
    assert!(spec.has_c);
    assert!(spec.c_tensor.is_some());
    assert!(!spec.has_bias);
    assert_eq!(spec.epilogue, CuEpilogue::Default);
    assert_eq!(spec.mnk_lits(), (3, 4, 8));
    assert_eq!(spec.ldc, spec.ldd);
    assert_eq!(inputs.len(), 3, "Lit reads [a, b, c]");
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
    assert_eq!(op.operand_name(2), "c");
}

/// C-fold with the REVERSED Add orientation (c + x@w): LogicalAdd
/// commutativity in the assembled program supplies the spelled orientation.
#[test]
fn fixture5_c_fold_reversed_orientation() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _out = (c + x.matmul(w)).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let ops = flavored_ops(&text, true, false, false);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1, "reversed orientation still folds");
    let spec = lt[0].0.spec.as_ref().expect("spec parses");
    assert!(spec.has_c);
    assert_eq!(spec.epilogue, CuEpilogue::Default);
}

/// x @ w + bias (rank-1, expanded over rows): the Bias contract.
#[test]
fn fixture5_bias() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _out = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    println!("MEASURE fixture5 bias: {:.2}s", timed(&text));

    // OBSERVED AMBIGUITY (finding, kept loud): the broadcast also gets a
    // materialized spelling, so the same add offers a C-fold reading too —
    // base + Bias-form + Accumulate-of-materialized-broadcast. The CONTRACT
    // is the constructor name, so each candidate is countable by name.
    let serialized = test_runtime::serialize_fixture(&text);
    let count = |op: &str| serialized.nodes.values().filter(|n| n.op == op).count();
    let base_ops = count("LayoutTensorOpCublasLt");
    let bias_ops = count("LayoutTensorOpCublasLtBias");
    let acc_ops = count("LayoutTensorOpCublasLtAccumulate");
    println!("  candidates: base={base_ops} bias={bias_ops} accumulate={acc_ops}");
    // ROUND-11 RE-PIN (base was 2, R10; bias/accumulate were 1): every
    // site's a AND b operand now carries two readable layout tensors
    // (its storage frame + the collapse-derived column-form frame — the
    // r8d probe pins the mechanism), so each of the two sites assembles
    // the 2 A x 2 B cross product: base = 2 sites x 4 = 8. The
    // decorators fire once per decorable base candidate on the
    // bridge-satisfying site, carrying its (desc_a, desc_b) pair through:
    // bias = 4, accumulate = 4. Bounded per-candidate-consistent
    // multiplicity; the strict level-0 election never prefers the
    // materialize-first column-form frames.
    assert_eq!(base_ops, 8);
    assert_eq!(
        bias_ops, 4,
        "the Bias-contract candidates exist (one per sibling base frame)"
    );
    assert_eq!(
        acc_ops, 4,
        "the Accumulate-of-materialized-broadcast candidates also exist"
    );

    let ops = flavored_ops(&text, false, true, false);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let (op, inputs, _) = lt[0];
    let spec = op.spec.as_ref().expect("spec parses");
    assert!(spec.has_bias);
    assert!(spec.bias_tensor.is_some());
    assert!(!spec.has_c);
    assert_eq!(spec.epilogue, CuEpilogue::Bias);
    assert_eq!(spec.mnk_lits(), (3, 4, 8));
    assert_eq!(inputs.len(), 3, "Lit reads [a, b, bias]");
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
    assert_eq!(op.operand_name(2), "bias");
}

/// The constructor-split payoff: elect the Bias contract BY CONSTRUCTOR
/// NAME ALONE — no slot-flavored picker. Rounds 1-3 could not do this
/// (both readings shared one constructor name).
#[test]
fn fixture5_bias_elected_by_name_alone() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _out = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (graph, _) = test_runtime::extract_fixture_with_genome(
        &text,
        // The CONTRACT is still elected by name alone; the view op is
        // routing infrastructure (round 10: the sibling's result reaches
        // the boundary through a transpose view), not a contract.
        &[
            "LayoutTensorOpCublasLtBias",
            "LayoutTensorOpIndexMapApplyViewGeneric",
        ],
    );
    let ops = collect_ops(&graph);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let (op, inputs, label) = lt[0];
    assert_eq!(label, "CublasLtBias");
    let spec = op.spec.as_ref().expect("spec parses");
    assert!(spec.has_bias && !spec.has_c);
    assert_eq!(spec.epilogue, CuEpilogue::Bias);
    assert_eq!(inputs.len(), 3);
    println!(
        "fixture5 bias BY NAME: elected {label} with arity {}",
        inputs.len()
    );
}

/// relu(x @ w + bias): epilogue ReluBias, Bias contract, Lit arity 3.
#[test]
fn fixture5_bias_relu() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _out = (x.matmul(w) + b.expand_dim(0, 4usize)).relu().output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    println!("MEASURE fixture5 bias+relu: {:.2}s", timed(&text));
    let ops = flavored_ops(&text, false, true, true);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let (op, inputs, _) = lt[0];
    let spec = op.spec.as_ref().expect("spec parses");
    assert!(spec.has_bias);
    assert!(!spec.has_c);
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert_eq!(inputs.len(), 3);
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
}

/// The full legal stack relu((x@w + c) + bias): the AccumulateBias
/// contract, Lit arity 4 in contract order [a, b, c, bias].
#[test]
fn fixture5_full_stack() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _out = ((x.matmul(w) + c) + b.expand_dim(0, 4usize))
            .relu()
            .output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    println!("MEASURE fixture5 full stack: {:.2}s", timed(&text));
    let ops = flavored_ops(&text, true, true, true);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let (op, inputs, label) = lt[0];
    assert_eq!(label, "CublasLtAccumulateBias");
    let spec = op.spec.as_ref().expect("spec parses");
    assert!(spec.has_c);
    assert!(spec.has_bias);
    assert_eq!(spec.epilogue, CuEpilogue::ReluBias);
    assert_eq!(inputs.len(), 4, "Lit reads [a, b, c, bias]");
    assert_eq!(inputs.len(), spec.expected_lit_inputs());
    assert_eq!(op.operand_name(2), "c");
    assert_eq!(op.operand_name(3), "bias");
}

// ---------------------------------------------------------------------------
// Fixture 6 — ordering soundness: relu(x@w) + c and relu(x@w) + bias must
// NOT fold the post-activation add (the epilogue field is no longer
// Default).
// ---------------------------------------------------------------------------

#[test]
fn fixture6_relu_then_add_c_not_folded() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let c = cx.tensor((4usize, 3usize), DType::F32);
        let _out = (x.matmul(w).relu() + c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let ops = flavored_ops(&text, false, false, true);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1, "only the relu op fuses");
    let spec = lt[0].0.spec.as_ref().expect("spec parses");
    assert_eq!(spec.epilogue, CuEpilogue::Relu);
    assert!(!spec.has_c, "the post-activation add must NOT fold into C");
    let decomposed_adds = ops
        .iter()
        .filter(|(_, _, label)| label.contains("Add"))
        .count();
    assert!(
        decomposed_adds >= 1,
        "the outer add survives as a plain op (labels: {:?})",
        ops.iter().map(|(_, _, l)| l.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn fixture6_relu_then_bias_not_folded() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let _out = (x.matmul(w).relu() + b.expand_dim(0, 4usize)).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let ops = flavored_ops(&text, false, false, true);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 1);
    let spec = lt[0].0.spec.as_ref().expect("spec parses");
    assert_eq!(spec.epilogue, CuEpilogue::Relu);
    assert!(!spec.has_bias, "bias-after-activation must NOT fold");
}

// ---------------------------------------------------------------------------
// Fixture 8 — diamond: the matmul out feeds a bias-add AND is itself an
// output. Base op and decorated op coexist; the second consumer is
// unaffected.
// ---------------------------------------------------------------------------

#[test]
fn fixture8_diamond_base_and_decorated_coexist() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let b = cx.tensor(3usize, DType::F32);
        let mm = x.matmul(w);
        let _mm_out = mm.output();
        let _biased = (mm + b.expand_dim(0, 4usize)).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let ops = flavored_ops(&text, false, true, false);
    let lt = cublaslt_only(&ops);
    assert_eq!(lt.len(), 2, "base op AND decorated op in one plan");
    let mut specs: Vec<_> = lt
        .iter()
        .map(|(op, inputs, _)| (op.spec.as_ref().expect("spec parses"), inputs.len()))
        .collect();
    specs.sort_by_key(|(spec, _)| spec.has_bias);
    let (base, base_arity) = specs[0];
    let (decorated, dec_arity) = specs[1];
    assert!(!base.has_bias);
    assert_eq!(base.epilogue, CuEpilogue::Default);
    assert_eq!(base_arity, 2);
    assert!(decorated.has_bias);
    assert_eq!(decorated.epilogue, CuEpilogue::Bias);
    assert_eq!(dec_arity, 3);
    // Same site identity underneath: same a/b and same SITE out.
    assert_eq!(base.logical_a, decorated.logical_a);
    assert_eq!(base.logical_b, decorated.logical_b);
    assert_eq!(base.logical_site_out, decorated.logical_site_out);
    // But the CLAIMED output (the D the executor binds) differs.
    assert_eq!(
        base.logical_out, base.logical_site_out,
        "base claims the matmul out"
    );
    assert_ne!(
        base.logical_out, decorated.logical_out,
        "decorated op's D is the bias-add output, not the matmul out"
    );
    assert_ne!(
        decorated.logical_out, decorated.logical_site_out,
        "decorated D moved off the site out"
    );
}
