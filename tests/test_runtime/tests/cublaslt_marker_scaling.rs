//! Scaling probe. N matmul(+relu) blocks in one recorded program:
//! N = 1/2/4/8 with DISTINCT geometry per block, and N = 8 with the SAME
//! geometry in every block (identity stress — the round-1 witness died
//! here). Watch for super-linear node growth.

use std::time::Instant;

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use test_runtime::cublaslt_marker::{CuEpilogue, CublasLt};

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

fn record_blocks(geometries: &[(usize, usize, usize)]) -> String {
    let mut cx = Graph::new();
    for &(m, k, n) in geometries {
        let x = cx.tensor((m, k), DType::F32);
        let w = cx.tensor((k, n), DType::F32);
        let _out = x.matmul(w).relu().output();
    }
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

fn probe(name: &str, geometries: &[(usize, usize, usize)], prev_nodes: Option<usize>) -> usize {
    let text = record_blocks(geometries);
    let started = Instant::now();
    let serialized = test_runtime::serialize_fixture(&text);
    let wall = started.elapsed();
    let nodes = serialized.nodes.len();
    let sites = serialized
        .nodes
        .values()
        .filter(|n| n.op == "CublasLtLogicalMatmulSite")
        .count();

    let (graph, _) = test_runtime::extract_fixture_with_genome(&text, PIN);
    let ops: Vec<_> = graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                (*op.op).as_any().downcast_ref::<CublasLt>().cloned()
            }
            _ => None,
        })
        .collect();
    let relu_ops = ops
        .iter()
        .filter(|op| {
            op.spec
                .as_ref()
                .is_some_and(|s| s.epilogue == CuEpilogue::Relu)
        })
        .count();
    let delta = prev_nodes.map(|p| nodes as i64 - p as i64);
    println!(
        "MEASURE scaling {name}: N={} -> {nodes} nodes (delta {delta:?}), {wall:.2?}, {sites} sites, {} cublaslt kernels ({relu_ops} relu)",
        geometries.len(),
        ops.len()
    );
    // ROUND-10 RE-PIN (was == blocks): each matmul carries the original
    // site plus its transpose-sandwich sibling.
    assert_eq!(
        sites,
        geometries.len() * 2,
        "original + sibling site per block"
    );
    assert_eq!(ops.len(), geometries.len(), "one cublaslt kernel per block");
    assert_eq!(
        relu_ops,
        geometries.len(),
        "every elected kernel carries the relu epilogue"
    );
    nodes
}

#[test]
fn fixture9_distinct_geometry_scaling() {
    let mut prev = None;
    for n in [1usize, 2, 4, 8] {
        let geometries: Vec<_> = (0..n).map(|i| (4 + i, 8, 2 + i)).collect();
        prev = Some(probe("distinct", &geometries, prev));
    }
}

#[test]
fn fixture9_same_geometry_identity_stress() {
    // The witness-killer: 8 blocks, every one x[4,8] @ w[8,3] + relu.
    // Distinct input tensors => distinct sites => distinct ops; shared
    // geometry must weld NOTHING.
    let geometries: Vec<_> = (0..8).map(|_| (4usize, 8usize, 3usize)).collect();
    let nodes = probe("same-geometry", &geometries, None);

    let single = probe("same-geometry-baseline", &geometries[..1], None);
    println!(
        "MEASURE same-geometry: 8 blocks {nodes} nodes vs 1 block {single} nodes ({:.2}x)",
        nodes as f64 / single as f64
    );
    assert!(nodes < single * 16, "no super-linear blowup");
}
