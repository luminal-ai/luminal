//! Probe: the true frontend spelling of x.matmul(w.permute((1,0))) for the
//! folded-permute case A[m,k],B[n,k]. Run by name with --nocapture.

use luminal::dtype::DType;
use luminal::graph::Graph;

#[test]
fn print_amk_bnk_spelling() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((3usize, 4usize), DType::F32); // stored [n, k]
    let _out = x.matmul(w.permute((1usize, 0usize))).output();
    println!("=== native_program for x.matmul(w.permute((1,0))) ===");
    println!(
        "{}",
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    );
}
