//! Valid PyTorch vector/batch matmul capture must reach a canonical contraction.
use super::test_support::*;
use luminal::prelude::*;

#[test]
fn matmul_accepts_vector_and_broadcast_ranks_without_dtype_casts() {
    let shapes: &[(&[i64], &[i64], &[i64])] = &[
        (&[5], &[5], &[]),
        (&[3, 5], &[5], &[3]),
        (&[5], &[2, 5, 3], &[2, 3]),
        (&[2, 3, 5], &[5], &[2, 3]),
        (&[2, 1, 3, 5], &[4, 5, 2], &[2, 4, 3, 2]),
        (&[2, 1, 2, 1, 3, 5], &[2, 4, 5, 2], &[2, 1, 2, 4, 3, 2]),
    ];
    for dtype in [INT, FLOAT] {
        for &(a, b, out) in shapes {
            let t = translate_one(
                node(
                    "torch.ops.aten.matmul.default",
                    vec![tensor_input("self", "a"), tensor_input("other", "b")],
                    &["out"],
                ),
                &[("a", dtype, a), ("b", dtype, b), ("out", dtype, out)],
                &["a", "b"],
                &["out"],
            );
            assert_eq!(
                t.outputs[0]
                    .shape
                    .iter()
                    .map(|d| d.to_usize().unwrap())
                    .collect::<Vec<_>>(),
                out.iter().map(|&d| d as usize).collect::<Vec<_>>()
            );
            assert_eq!(
                t.outputs[0].dtype,
                if dtype == INT { DType::Int } else { DType::F32 }
            );
            assert!(
                !t.graph
                    .logical
                    .render_all()
                    .unwrap()
                    .contains("LogicalCast")
            );
        }
    }
}
