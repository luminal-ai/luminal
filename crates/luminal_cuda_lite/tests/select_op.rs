#![cfg(feature = "device")]
//! End-to-end tests for CUDA-lite's native ternary `Select` op. This is the op
//! that closes the ReLU/GELU epilogue gap: without it, a graph lowering
//! `maximum`/`where`/`erf` dead-ends extraction and the search finds no plan.

use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, HostBuffer, harness_search_options};
use rustc_hash::FxHashMap;

/// Build `select(condition, if_true, if_false)`, search, execute, read back.
fn run(condition: HostBuffer, if_true: HostBuffer, if_false: HostBuffer) -> Vec<f32> {
    let mut cx = Graph::new();
    let condition_t = cx.tensor((4,), DType::Bool);
    let if_true_t = cx.tensor((4,), DType::F32);
    let if_false_t = cx.tensor((4,), DType::F32);
    let out = condition_t.select(if_true_t, if_false_t);

    let data: FxHashMap<_, _> = [
        (condition_t.id, condition.clone()),
        (if_true_t.id, if_true.clone()),
        (if_false_t.id, if_false.clone()),
    ]
    .into_iter()
    .collect();

    let mut runtime = CudaRuntime::load(&cx).expect("load");
    runtime
        .search(&data, &harness_search_options())
        .expect("search finds a Select plan");
    runtime.set_data(condition_t.id, condition).unwrap();
    runtime.set_data(if_true_t.id, if_true).unwrap();
    runtime.set_data(if_false_t.id, if_false).unwrap();
    runtime.execute().expect("execute");
    runtime.get_f32(out.id).expect("f32 output")
}

#[test]
fn select_picks_branches_elementwise() {
    let got = run(
        HostBuffer::bool8(vec![1, 0, 1, 0]).unwrap(),
        vec![1.0f32, 2.0, 3.0, 4.0].into(),
        vec![10.0f32, 20.0, 30.0, 40.0].into(),
    );
    assert_eq!(got, vec![1.0, 20.0, 3.0, 40.0]);
}

/// A true select must not leak an unselected NaN the way `cond*a + (1-cond)*b`
/// does through `NaN * 0`: the selected `if_true` lanes stay finite, while the
/// lanes that select the NaN branch are NaN.
#[test]
fn select_does_not_leak_unselected_nan() {
    let got = run(
        HostBuffer::bool8(vec![1, 0, 1, 0]).unwrap(),
        vec![1.0f32, 2.0, 3.0, 4.0].into(),
        vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN].into(),
    );
    assert_eq!(got[0], 1.0);
    assert_eq!(got[2], 3.0);
    assert!(got[1].is_nan() && got[3].is_nan());
}
