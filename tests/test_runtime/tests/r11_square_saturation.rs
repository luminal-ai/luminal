//! ROUND-11 gate (g): square programs saturate with bounded node counts
//! under the full default schedule (the termination anchor at work on the
//! corners where the sandwich is self-adjoint).
use luminal::dtype::DType;
use luminal::graph::Graph;

#[test]
fn r11_squares_saturate_bounded() {
    let a3 = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let w = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(w).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let s3 = test_runtime::serialize_fixture(&a3);
    println!("a3 (x[4,4] @ w[4,4]) saturates: {} nodes", s3.nodes.len());

    let a6b = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 4usize), DType::F32);
        let _ = x.matmul(x.permute((1usize, 0usize))).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let s6 = test_runtime::serialize_fixture(&a6b);
    println!(
        "a6b (x @ x^T, self-sibling) saturates: {} nodes",
        s6.nodes.len()
    );

    // Bounded: same order as the plain rectangular fixture (5,098) —
    // the squares mint no extra generations.
    assert!(
        s3.nodes.len() < 20_000,
        "a3 bounded, got {}",
        s3.nodes.len()
    );
    assert!(
        s6.nodes.len() < 20_000,
        "a6b bounded, got {}",
        s6.nodes.len()
    );
}
