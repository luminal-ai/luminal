//! COST PROBE (analysis-only, not a landing): what does an extra transposing
//! VIEW cost in the e-graph, relative to the measured matmul-block baseline?
//! This is the marginal number behind the (A x B) = ((B^T)(A^T))^T estimate.

use luminal::dtype::DType;
use luminal::graph::Graph;

fn nodes(text: &str, name: &str) -> usize {
    let started = std::time::Instant::now();
    let serialized = test_runtime::serialize_fixture(text);
    let wall = started.elapsed();
    let n = serialized.nodes.len();
    let applies = serialized
        .nodes
        .values()
        .filter(|x| x.op == "LogicalIndexMapApply")
        .count();
    let maps = serialized
        .nodes
        .values()
        .filter(|x| x.op == "IndexMapLit")
        .count();
    let sites = serialized
        .nodes
        .values()
        .filter(|x| x.op == "CublasLtLogicalMatmulSite")
        .count();
    println!(
        "MEASURE probe {name}: {n} nodes, {applies} applies, {maps} maps, {sites} sites, {wall:.2?}"
    );
    n
}

#[test]
fn transpose_view_marginal_cost() {
    // P0: the scaling test's own block, A[m,k],B[k,n]-spelled.
    let mut cx = Graph::new();
    let x = cx.tensor((32usize, 48usize), DType::F32);
    let w = cx.tensor((48usize, 64usize), DType::F32);
    let _ = x.matmul(w).relu().output();
    let p0 = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;

    // P1: same block, plus ONE transposing view on the output.
    let mut cx = Graph::new();
    let x = cx.tensor((32usize, 48usize), DType::F32);
    let w = cx.tensor((48usize, 64usize), DType::F32);
    let _ = x.matmul(w).relu().permute((1usize, 0usize)).output();
    let p1 = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;

    // P2: A[m,k],B[n,k]-spelled -- the permute folds into the operand's index map.
    let mut cx = Graph::new();
    let x = cx.tensor((32usize, 48usize), DType::F32);
    let w = cx.tensor((64usize, 48usize), DType::F32);
    let _ = x.matmul(w.permute((1usize, 0usize))).relu().output();
    let p2 = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;

    let n0 = nodes(&p0, "P0 amk_bkn-block");
    let n1 = nodes(&p1, "P1 amk_bkn-block + output transpose view");
    let n2 = nodes(&p2, "P2 amk_bnk-spelled (permute folded into map)");
    println!("MEASURE delta P1-P0 = {}", n1 as i64 - n0 as i64);
    println!("MEASURE delta P2-P0 = {}", n2 as i64 - n0 as i64);
}
