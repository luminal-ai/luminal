//! Exact integer remainder, including signed divisors and i64 scalar literals.
use super::test_support::*;
use luminal_reference::{harness::run_reference_with_ranges, typed_buffer::TypedBuffer};

#[test]
fn integer_remainders_preserve_bits_and_pytorch_signs() {
    for dtype in [INT, 5] {
        for divisor in [3i64, -3] {
            for op in ["remainder", "fmod"] {
                let t = translate_one(
                    node(
                        &format!("torch.ops.aten.{op}.Tensor"),
                        vec![tensor_input("self", "x"), tensor_input("other", "b")],
                        &["y"],
                    ),
                    &[
                        ("x", dtype, &[2, 3]),
                        ("b", dtype, &[3]),
                        ("y", dtype, &[2, 3]),
                    ],
                    &["x", "b"],
                    &["y"],
                );
                let values = if dtype == INT {
                    vec![i32::MIN as i64, i32::MAX as i64, -6, -5, 0, 5]
                } else {
                    vec![
                        i64::MIN,
                        i64::MAX,
                        -(1i64 << 55) - 1,
                        (1i64 << 55) + 1,
                        0,
                        -6,
                    ]
                };
                let buffer = |data: Vec<i64>| {
                    if dtype == INT {
                        TypedBuffer::I32(data.into_iter().map(|v| v as i32).collect())
                    } else {
                        TypedBuffer::I64(data)
                    }
                };
                let rt = run_reference_with_ranges(
                    &t.graph,
                    &[
                        (t.inputs[0].tensor, buffer(values.clone())),
                        (t.inputs[1].tensor, buffer(vec![divisor; 3])),
                    ],
                    &[(t.inputs[1].tensor, divisor, divisor)],
                );
                let actual = if dtype == INT {
                    rt.get_i32(t.outputs[0].tensor)
                        .unwrap()
                        .iter()
                        .map(|&v| v as i64)
                        .collect::<Vec<_>>()
                } else {
                    rt.get_i64(t.outputs[0].tensor).unwrap().to_vec()
                };
                let expected = values
                    .iter()
                    .map(|&a| {
                        let r = a % divisor;
                        if op == "remainder" && r != 0 && (r < 0) != (divisor < 0) {
                            r + divisor
                        } else {
                            r
                        }
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{op}, dtype {dtype}, divisor {divisor}");
            }
        }
    }
}

#[test]
fn int64_remainder_keeps_large_scalar_divisors_exact() {
    for divisor in [(1i64 << 55) + 1, -(1i64 << 55) - 1] {
        for op in ["remainder", "fmod"] {
            let t = translate_one(
                node(
                    &format!("torch.ops.aten.{op}.Scalar"),
                    vec![tensor_input("self", "x"), scalar_int("other", divisor)],
                    &["y"],
                ),
                &[("x", 5, &[4]), ("y", 5, &[4])],
                &["x"],
                &["y"],
            );
            let values = vec![i64::MIN, i64::MAX, divisor, -divisor];
            let rt = run_reference_with_ranges(
                &t.graph,
                &[(t.inputs[0].tensor, TypedBuffer::I64(values.clone()))],
                &[],
            );
            let expected = values
                .iter()
                .map(|&a| {
                    let r = a % divisor;
                    if op == "remainder" && r != 0 && (r < 0) != (divisor < 0) {
                        r + divisor
                    } else {
                        r
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &expected);
        }
    }
}

#[test]
fn integer_remainder_promotes_tensor_operands_before_arithmetic() {
    let t = translate_one(
        node(
            "torch.ops.aten.remainder.Tensor",
            vec![tensor_input("self", "x"), tensor_input("other", "b")],
            &["y"],
        ),
        &[("x", INT, &[2]), ("b", 5, &[]), ("y", 5, &[2])],
        &["x", "b"],
        &["y"],
    );
    let divisor = (1i64 << 55) + 1;
    let rt = run_reference_with_ranges(
        &t.graph,
        &[
            (t.inputs[0].tensor, TypedBuffer::I32(vec![-5, 5])),
            (t.inputs[1].tensor, TypedBuffer::I64(vec![divisor])),
        ],
        &[(t.inputs[1].tensor, divisor, divisor)],
    );
    assert_eq!(rt.get_i64(t.outputs[0].tensor).unwrap(), &[divisor - 5, 5]);
}
