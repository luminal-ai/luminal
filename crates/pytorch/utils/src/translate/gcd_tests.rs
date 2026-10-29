//! `aten.gcd`: the select-based Euclid unroll must run every division a
//! pair of the operand dtype can need. A flat 32-round cut once returned the
//! unfinished state (gcd(F35, F34) came out as 2), so these pin the round
//! bound itself, the recorded round count, and the worst-case pairs'
//! values on the reference runtime.

use luminal::prelude::*;
use luminal_reference::{harness::run_reference_with_ranges, typed_buffer::TypedBuffer};

use super::Translation;
use super::elementwise_more::{gcd_euclid_rounds, gcd_magnitude_bound};
use super::test_support::*;
use crate::dtype::TorchDType;

/// Divisions until `b == 0`, with the lowering's remainder: truncated, by
/// `|b|`.
fn euclid_divisions(mut a: i64, mut b: i64) -> usize {
    let mut divisions = 0;
    while b != 0 {
        (a, b) = (b, a % b.abs());
        divisions += 1;
    }
    divisions
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// `F(k)`, with `F(1) = F(2) = 1`.
fn fibonacci(k: usize) -> u128 {
    let (mut previous, mut current) = (0u128, 1u128);
    for _ in 1..k {
        (previous, current) = (current, previous + current);
    }
    current
}

fn gcd_graph(dtype: TorchDType, len: i64) -> Translation {
    let code = dtype.code();
    translate_one(
        node(
            "torch.ops.aten.gcd.default",
            vec![tensor_input("self", "a"), tensor_input("other", "b")],
            &["y"],
        ),
        &[
            ("a", code, &[len]),
            ("b", code, &[len]),
            ("y", code, &[len]),
        ],
        &["a", "b"],
        &["y"],
    )
}

fn count(t: &Translation, op: impl Fn(&LogicalOp) -> bool) -> usize {
    let graph = t.graph.logical.petgraph();
    graph.node_indices().filter(|&i| op(&graph[i].op)).count()
}

#[test]
fn round_bound_is_the_exhaustive_worst_case() {
    const LIMIT: i64 = 300;
    // worst[m]: the most divisions any signed pair of largest magnitude m needs.
    let mut worst = vec![0usize; LIMIT as usize + 1];
    for a in -LIMIT..=LIMIT {
        for b in -LIMIT..=LIMIT {
            let magnitude = a.abs().max(b.abs()) as usize;
            worst[magnitude] = worst[magnitude].max(euclid_divisions(a, b));
        }
    }
    for magnitude in 1..=LIMIT as usize {
        worst[magnitude] = worst[magnitude].max(worst[magnitude - 1]);
        assert_eq!(
            gcd_euclid_rounds(magnitude as u64),
            worst[magnitude],
            "every pair with magnitudes <= {magnitude}"
        );
    }
}

#[test]
fn round_bound_per_dtype_is_attained_by_the_largest_fibonacci_pair() {
    // (dtype, rounds, K): F(K) is the largest Fibonacci number in range.
    for (dtype, rounds, k) in [
        (DType::I64, 91, 92),
        (DType::U64, 91, 92),
        (DType::Int, 45, 46),
        (DType::U16, 23, 24),
        (DType::I16, 22, 23),
        (DType::U8, 12, 13),
        (DType::I8, 10, 11),
        (DType::Bool, 1, 2),
    ] {
        let bound = gcd_magnitude_bound(dtype) as u128;
        assert!(
            fibonacci(k) <= bound && fibonacci(k + 1) > bound,
            "{dtype:?}"
        );
        assert_eq!(gcd_euclid_rounds(bound as u64), rounds, "{dtype:?}");
        let (smaller, larger) = (fibonacci(k - 1) as i64, fibonacci(k) as i64);
        assert_eq!(euclid_divisions(smaller, larger), rounds, "{dtype:?}");
        if smaller < larger {
            assert_eq!(euclid_divisions(larger, smaller), rounds - 1, "{dtype:?}");
        }
    }
}

#[test]
fn gcd_records_one_division_per_round_of_the_operand_dtype() {
    for (dtype, rounds) in [
        (TorchDType::Long, 91),
        (TorchDType::Int, 45),
        (TorchDType::Short, 22),
        (TorchDType::Byte, 12),
        (TorchDType::Char, 10),
    ] {
        let t = gcd_graph(dtype, 4);
        assert_eq!(
            count(&t, |op| matches!(op, LogicalOp::TruncRem)),
            rounds,
            "{dtype:?}"
        );
    }
}

/// Run gcd over `pairs` on the reference runtime, with both inputs attested
/// to `range`; the values travel as I32 when `dtype` is Int.
fn run_gcd(dtype: TorchDType, pairs: &[(i64, i64)], range: (i64, i64)) -> Vec<i64> {
    let (a, b): (Vec<i64>, Vec<i64>) = pairs.iter().copied().unzip();
    let t = gcd_graph(dtype, pairs.len() as i64);
    let (lhs, rhs) = (t.inputs[0].tensor, t.inputs[1].tensor);
    let buffer = |values: Vec<i64>| match dtype {
        TorchDType::Int => TypedBuffer::I32(values.into_iter().map(|v| v as i32).collect()),
        _ => TypedBuffer::I64(values),
    };
    let rt = run_reference_with_ranges(
        &t.graph,
        &[(lhs, buffer(a)), (rhs, buffer(b))],
        &[(lhs, range.0, range.1), (rhs, range.0, range.1)],
    );
    match dtype {
        TorchDType::Int => rt
            .get_i32(t.outputs[0].tensor)
            .unwrap()
            .iter()
            .map(|&v| v as i64)
            .collect(),
        _ => rt.get_i64(t.outputs[0].tensor).unwrap().to_vec(),
    }
}

#[test]
fn int64_gcd_finishes_the_worst_case_fibonacci_pairs() {
    let f = |k: usize| fibonacci(k) as i64;
    let pairs = [
        (f(91), f(92)),
        (f(92), f(91)),
        (-f(91), f(92)),
        (f(91), -f(92)),
        (f(35), f(34)),
        (f(34), f(35)),
        (f(92), f(92)),
        (0, 0),
        (0, -f(92)),
        (f(92), 0),
        (-12, 18),
        (18, -24),
    ];
    // The attested range is the whole I64 range but `i64::MIN`, whose
    // magnitude I64 cannot hold (`|rhs|` would overflow).
    let actual = run_gcd(TorchDType::Long, &pairs, (-i64::MAX, i64::MAX));
    let expected = pairs.iter().map(|&(a, b)| gcd(a, b)).collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn int32_gcd_finishes_the_worst_case_fibonacci_pairs() {
    let f = |k: usize| fibonacci(k) as i64;
    let max = i32::MAX as i64;
    let pairs = [
        (f(45), f(46)),
        (f(46), f(45)),
        (-f(45), f(46)),
        (f(35), f(34)),
        (f(46), f(46)),
        (0, 0),
        (0, -f(46)),
        (-12, 18),
        (max, max - 1),
    ];
    let actual = run_gcd(TorchDType::Int, &pairs, (-max, max));
    let expected = pairs.iter().map(|&(a, b)| gcd(a, b)).collect::<Vec<_>>();
    assert_eq!(actual, expected);
}
