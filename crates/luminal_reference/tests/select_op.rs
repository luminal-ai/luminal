//! End-to-end test for the native ternary `Select` logical op: a boolean
//! condition chooses elementwise between two same-shape/same-dtype value
//! branches. Unlike `cond` (arithmetic masking), this must lower to a real
//! boolean select, so it is also the NaN-safe selection primitive.

use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

#[test]
fn select_picks_branches_elementwise() {
    let mut cx = Graph::new();
    let condition = cx.tensor((4,), DType::Bool);
    let if_true = cx.tensor((4,), DType::F32);
    let if_false = cx.tensor((4,), DType::F32);
    let out = condition.select(if_true, if_false);

    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(condition.id, TypedBuffer::bool8(vec![1, 0, 1, 0]).unwrap());
    data.insert(if_true.id, TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]));
    data.insert(if_false.id, TypedBuffer::F32(vec![10.0, 20.0, 30.0, 40.0]));
    runtime
        .search(&data, &harness_search_options())
        .expect("search");

    runtime.set_data(condition.id, TypedBuffer::bool8(vec![1, 0, 1, 0]).unwrap());
    runtime.set_data(if_true.id, TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]));
    runtime.set_data(if_false.id, TypedBuffer::F32(vec![10.0, 20.0, 30.0, 40.0]));
    runtime.execute().expect("execute");

    let got = runtime.get_f32(out.id).expect("f32 output");
    assert_eq!(got, &vec![1.0, 20.0, 3.0, 40.0]);
}

/// The selection must be a *true* select, not an arithmetic blend: a NaN in
/// the unselected branch must not leak into the result (which is what
/// `cond*a + (1-cond)*b` does through `NaN * 0`).
#[test]
fn select_does_not_leak_unselected_nan() {
    let mut cx = Graph::new();
    let condition = cx.tensor((2,), DType::Bool);
    let if_true = cx.tensor((2,), DType::F32);
    let if_false = cx.tensor((2,), DType::F32);
    let out = condition.select(if_true, if_false);

    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(condition.id, TypedBuffer::bool8(vec![1, 0]).unwrap());
    data.insert(if_true.id, TypedBuffer::F32(vec![1.0, 2.0]));
    data.insert(if_false.id, TypedBuffer::F32(vec![f32::NAN, f32::NAN]));
    runtime
        .search(&data, &harness_search_options())
        .expect("search");

    runtime.set_data(condition.id, TypedBuffer::bool8(vec![1, 0]).unwrap());
    runtime.set_data(if_true.id, TypedBuffer::F32(vec![1.0, 2.0]));
    runtime.set_data(if_false.id, TypedBuffer::F32(vec![f32::NAN, f32::NAN]));
    runtime.execute().expect("execute");

    let got = runtime.get_f32(out.id).expect("f32 output");
    // index 0 picks the finite branch; index 1 picks the NaN branch.
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_nan());
}

/// `abs` now lowers to `(x < 0).select(-x, x)`; it must still plan and run
/// on the reference backend. This is also the regression guard for the
/// e-graph blow-up: the old masked-multiply `abs` chained into the integer
/// AC/distributivity closure and never converged.
#[test]
fn abs_uses_select_and_runs() {
    let mut cx = Graph::new();
    let x = cx.tensor((4,), DType::F32);
    let out = x.abs();

    let values = vec![-1.5f32, 2.0, -3.0, 0.0];
    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(x.id, TypedBuffer::F32(values.clone()));
    runtime
        .search(&data, &harness_search_options())
        .expect("search");
    runtime.set_data(x.id, TypedBuffer::F32(values));
    runtime.execute().expect("execute");

    let got = runtime.get_f32(out.id).expect("f32 output");
    assert_eq!(got, &vec![1.5, 2.0, 3.0, 0.0]);
}

/// Integer select over I64 branches exercises the typed kernel arms and the
/// value-bounds union rule.
#[test]
fn select_i64_branches() {
    let mut cx = Graph::new();
    let condition = cx.tensor((3,), DType::Bool);
    let if_true = cx.tensor((3,), DType::I64);
    let if_false = cx.tensor((3,), DType::I64);
    let out = condition.select(if_true, if_false);

    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(condition.id, TypedBuffer::bool8(vec![1, 0, 1]).unwrap());
    data.insert(if_true.id, TypedBuffer::I64(vec![7, 8, 9]));
    data.insert(if_false.id, TypedBuffer::I64(vec![-1, -2, -3]));
    runtime
        .search(&data, &harness_search_options())
        .expect("search");
    runtime.set_data(condition.id, TypedBuffer::bool8(vec![1, 0, 1]).unwrap());
    runtime.set_data(if_true.id, TypedBuffer::I64(vec![7, 8, 9]));
    runtime.set_data(if_false.id, TypedBuffer::I64(vec![-1, -2, -3]));
    runtime.execute().expect("execute");

    let got = runtime.get_i64(out.id).expect("i64 output");
    assert_eq!(got, &vec![7, -2, 9]);
}
