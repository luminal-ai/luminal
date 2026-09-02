//! throwaway diagnostic (deleted before landing)
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::ExtractedNode;
use test_runtime::cublaslt_marker::CublasLt;
const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    "LayoutTensorOpIndexMapApplyViewGeneric",
];
#[test]
fn diag_a5() {
    {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w1 = cx.tensor((4usize, 4usize), DType::F32);
        let w2 = cx.tensor((4usize, 4usize), DType::F32);
        let y = x.matmul(w1);
        let _ = y.matmul(w2.permute((1usize, 0usize))).output();
    };
    let program = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w1 = cx.tensor((4usize, 4usize), DType::F32);
        let w2 = cx.tensor((4usize, 4usize), DType::F32);
        let y = x.matmul(w1);
        let _ = y.matmul(w2.permute((1usize, 0usize))).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (graph, _) = test_runtime::extract_fixture_with_genome(&program, PIN);
    for node in graph.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            let ins: Vec<String> = op
                .inputs
                .iter()
                .map(|i| format!("{}={}", i.port, i.value))
                .collect();
            let outs: Vec<String> = op.outputs.iter().map(|o| format!("{}", o.eclass)).collect();
            println!("PLAN {}: ins={ins:?} outs={outs:?}", op.op.label());
            if let Some(c) = (*op.op).as_any().downcast_ref::<CublasLt>() {
                if let Some(spec) = &c.spec {
                    println!(
                        "  spec m={} n={} k={} ta={} tb={} lda={} ldb={} a_lt={} b_lt={}",
                        spec.m,
                        spec.n,
                        spec.k,
                        spec.trans_a,
                        spec.trans_b,
                        spec.lda,
                        spec.ldb,
                        spec.desc_a_layout_tensor,
                        spec.desc_b_layout_tensor
                    );
                }
            }
        }
    }
}
