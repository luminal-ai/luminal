use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

/// Independent oracle: each output sums `a * b` in increasing `k`, from zero,
/// the order the decomposed `Mul` then `ReduceSum` chain uses.
fn sequential_matmul(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// Deterministic values with mixed signs and magnitudes, so rounding differences
/// would show up in the bit-exact comparison.
fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            (unit - 0.5) * 8.0
        })
        .collect()
}

fn run_matmul(m: usize, k: usize, n: usize, lhs: Vec<f32>, rhs: Vec<f32>) -> Vec<f32> {
    let mut graph = Graph::new();
    let a = graph.tensor(vec![m, k], DType::F32);
    let b = graph.tensor(vec![k, n], DType::F32);
    let c = a.matmul(b);
    let inputs =
        FxHashMap::from_iter([(a.id, TypedBuffer::F32(lhs)), (b.id, TypedBuffer::F32(rhs))]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    for (id, data) in inputs {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    runtime.get_f32(c.id).unwrap().clone()
}

#[test]
fn small_matmul_matches_sequential_order_bit_for_bit() {
    let (m, k, n) = (3, 5, 4);
    let lhs = values(m * k, 1);
    let rhs = values(k * n, 2);
    let expected = sequential_matmul(&lhs, &rhs, m, k, n);
    let actual = run_matmul(m, k, n, lhs, rhs);
    assert_eq!(actual, expected);
}

#[test]
fn whisper_scale_matmul_plans_and_matches_oracle() {
    // [1500, 1152] x [1152, 384]: the decomposed product is 2,654,208,000 bytes,
    // above the 2 GiB intermediate cap, so it failed to plan before this op.
    let (m, k, n) = (1500, 1152, 384);
    let lhs = values(m * k, 3);
    let rhs = values(k * n, 4);
    let expected = sequential_matmul(&lhs, &rhs, m, k, n);
    let actual = run_matmul(m, k, n, lhs, rhs);
    assert_eq!(actual, expected);
}

fn sequential_matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for inner in 0..k {
                acc += lhs[row * k + inner] * rhs[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[test]
fn small_f64_matmul_uses_decomposed_path_and_matches_oracle() {
    let (m, k, n) = (3, 5, 4);
    let lhs: Vec<f64> = values(m * k, 5).into_iter().map(f64::from).collect();
    let rhs: Vec<f64> = values(k * n, 6).into_iter().map(f64::from).collect();
    let expected = sequential_matmul_f64(&lhs, &rhs, m, k, n);
    let mut graph = Graph::new();
    let a = graph.tensor(vec![m, k], DType::F64);
    let b = graph.tensor(vec![k, n], DType::F64);
    let c = a.matmul(b);
    let inputs =
        FxHashMap::from_iter([(a.id, TypedBuffer::F64(lhs)), (b.id, TypedBuffer::F64(rhs))]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    for (id, data) in inputs {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    assert_eq!(runtime.get_f64(c.id).unwrap(), &expected);
}

#[test]
fn whisper_scale_f64_matmul_still_refuses_on_the_intermediate_cap() {
    // F64 is not covered by the fused op. Its decomposed [1500, 1152, 384]
    // product is 5,308,416,000 bytes, above the 2 GiB intermediate cap.
    let (m, k, n) = (1500, 1152, 384);
    let mut graph = Graph::new();
    let a = graph.tensor(vec![m, k], DType::F64);
    let b = graph.tensor(vec![k, n], DType::F64);
    let _c = a.matmul(b);
    let inputs = FxHashMap::from_iter([
        (a.id, TypedBuffer::F64(vec![0.0; m * k])),
        (b.id, TypedBuffer::F64(vec![0.0; k * n])),
    ]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    let error = runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .expect_err("decomposed F64 product above the cap must refuse");
    assert!(
        format!("{error:#}").contains("no candidate genome produced an executable plan"),
        "unexpected refusal: {error:#}"
    );
}

#[test]
fn small_i32_matmul_uses_decomposed_path_and_matches_oracle() {
    let (m, k, n) = (3, 4, 2);
    let lhs: Vec<i32> = (0..(m * k) as i32).map(|v| v % 7 - 3).collect();
    let rhs: Vec<i32> = (0..(k * n) as i32).map(|v| v % 5 - 2).collect();
    let mut expected = vec![0i32; m * n];
    for row in 0..m {
        for col in 0..n {
            for inner in 0..k {
                expected[row * n + col] += lhs[row * k + inner] * rhs[inner * n + col];
            }
        }
    }
    let mut graph = Graph::new();
    let a = graph.tensor(vec![m, k], DType::Int);
    let b = graph.tensor(vec![k, n], DType::Int);
    let c = a.matmul(b);
    let inputs =
        FxHashMap::from_iter([(a.id, TypedBuffer::I32(lhs)), (b.id, TypedBuffer::I32(rhs))]);
    let mut runtime = ReferenceRuntime::load(&graph).unwrap();
    // Int sums are proof-gated: the value bounds must show every partial sum fits.
    // Inputs lie in [-3, 3], so each product is at most 9 in magnitude and the sum
    // of k = 4 products at most 36, well inside the i32 range.
    runtime.bind_value_range(a.id, -3, 3).unwrap();
    runtime.bind_value_range(b.id, -3, 3).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    for (id, data) in inputs {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    assert_eq!(runtime.get_i32(c.id).unwrap(), &expected);
}
