//! NULL TENSOR impact study: could the Lit dataflow list ALSO be
//! fixed-arity, using a placeholder LayoutTensor in absent slots?
//! Observational probes — the deliverable is findings, not green.

use luminal::dtype::DType;
use luminal::graph::Graph;

fn base_program() -> String {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((4usize, 3usize), DType::F32);
    let _out = x.matmul(w).output();
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

/// Probe (i): does the EGGLOG side accept a null LayoutTensor term, and
/// does any rule derive facts (injectivity, shape) for it?
///
/// Row ids below track the RECORDED matmul spelling (v0/v1 inputs, the
/// output ReduceSum on v5). The OUTPUT STEM is DERIVED from the program
/// text rather than tracked by hand — it is keyed by the output node's
/// id, which has already shifted twice (fold-2 removal; PR #423's
/// petgraph numbering). Setup, not subject: the refs just have to bind.
#[test]
fn nulltensor_probe_egglog_side() {
    let base = base_program();
    let fx = format!(
        "{base}\n(constructor CublasLtNullTensor (CublasLtLogicalSite) LayoutTensor)\n(let probe_site (CublasLtLogicalMatmulSite v0 v1 v5))\n(let probe_null (CublasLtNullTensor probe_site))\n"
    );
    let s = test_runtime::serialize_fixture(&fx);
    let nulls = s
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtNullTensor")
        .count();
    let null_class = s
        .nodes
        .values()
        .find(|n| n.op == "CublasLtNullTensor")
        .map(|n| n.eclass.clone())
        .expect("null tensor exists");
    let mut fact_rows: Vec<&str> = Vec::new();
    for node in s.nodes.values() {
        if let Some(child) = node.children.first() {
            if let Some(cn) = s.nodes.get(child) {
                if cn.eclass == null_class && node.op != "CublasLtNullTensor" {
                    fact_rows.push(node.op.as_str());
                }
            }
        }
    }
    fact_rows.sort();
    fact_rows.dedup();
    println!("PROBE i: saturation OK, {nulls} null tensor(s); derived rows over it: {fact_rows:?}");
    assert_eq!(nulls, 1, "the null term exists and nothing duplicated it");
    assert!(
        !fact_rows.contains(&"injectivity-of"),
        "no injectivity proof for a null tensor (fail-closed)"
    );
}

/// Probe (ii): a fixed-arity Lit [a, b, null] unioned into the real op
/// class — what do the cost layer / operand walk do with an input that has
/// no producer and no buffer?
/// The recorded output stem (`natout{K}`), read off the program text —
/// K is the output NODE's id and shifts whenever recorder numbering
/// does; hardcoding it is how this probe has broken twice.
fn output_stem(base: &str) -> String {
    let start = base
        .find("natout")
        .expect("bound program has an output stem");
    let digits: String = base[start + 6..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    format!("natout{digits}")
}

#[test]
fn nulltensor_probe_extraction_side() {
    let base = base_program();
    let stem = output_stem(&base);
    let fx = format!(
        r#"{base}
(constructor CublasLtNullTensor (CublasLtLogicalSite) LayoutTensor)
(let probe_site (CublasLtLogicalMatmulSite v0 v1 v5))
(let probe_null (CublasLtNullTensor probe_site))
(let probe_desc_a (CublasLtOperandADescriptor probe_site nat1_layout_tensor
  (CublasLtOperationN)))
(let probe_desc_b (CublasLtOperandBDescriptor probe_site nat0_layout_tensor
  (CublasLtOperationN)))
(let probe_desc_d (CublasLtOutputDDescriptor probe_site {stem}_layout_tensor))
(let probe_op (LayoutTensorOpCublasLt probe_site probe_desc_a probe_desc_b probe_desc_d
  (CublasLtEpilogueDefault)))
(let probe_lit (LayoutTensorOpLit
  (LayoutTensorCons nat0_layout_tensor
    (LayoutTensorCons nat1_layout_tensor
      (LayoutTensorCons probe_null (LayoutTensorNil))))
  (LayoutTensorCons {stem}_layout_tensor (LayoutTensorNil))))
(union probe_lit probe_op)
"#
    );
    // SETUP, NOT SUBJECT (round-8b audit): this fixture must PARSE. It
    // once did not — the seeds still carried deleted form children — and
    // the old `else { println!(); return; }` shape made the test pass
    // green while testing nothing. Setup failure is now a hard failure;
    // only what extraction does with the null tensor is a finding.
    let _s = test_runtime::serialize_fixture(&fx);
    let extraction = std::panic::catch_unwind(|| {
        test_runtime::extract_fixture_with_genome(&fx, &["LayoutTensorOpCublasLt"])
    });
    // SUBJECT: both outcomes are legitimate findings, but the observed
    // behaviour is stable and now PINNED — a silent change of channel
    // (from loud refusal to quiet success) must break this test.
    match extraction {
        Ok((graph, _)) => {
            use luminal::layout_ir::ExtractedNode;
            let ops: Vec<(String, usize)> = graph
                .dag
                .node_weights()
                .filter_map(|node| match node {
                    ExtractedNode::LayoutOp(op) => {
                        Some((op.op.label().to_string(), op.inputs.len()))
                    }
                    _ => None,
                })
                .collect();
            println!("PROBE ii OBSERVED: extraction SUCCEEDS, plan ops: {ops:?}");
            for (label, arity) in &ops {
                if label.starts_with("CublasLt") {
                    assert_eq!(*arity, 2, "a null-input op node entered the plan — hazard");
                }
            }
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            println!("PROBE ii OBSERVED: extraction PANICS (loud refusal): {msg}");
            // ROUND 10: the refusal may now fire UPSTREAM of the cost
            // layer — the genome walk finds no plan for the null-tensor
            // chain and the boundary refuses loudly. Either message is a
            // loud structured refusal; silence would be the hazard.
            assert!(
                msg.contains("has no LayoutTensorLit spelling")
                    || msg.contains("outputs with no plan"),
                "the refusal is loud and structured, got: {msg}"
            );
        }
    }
}
