//! Narrow-width and geometry coverage for the three scan logical ops,
//! `UnspecifiedOrderScanSum`, `UnspecifiedOrderScanProd` and
//! `UnspecifiedOrderScanMax`.
//!
//! F32 is covered by the core tests in `src/frontend/unary.rs` and
//! `src/frontend/reduction.rs`, and I32 by `int_sum_scan_runs_unattested` and
//! its siblings in `crates/luminal_reference/src/runtime.rs`. The nine narrow
//! arms, I8, U8 and I16 across the three ops, had none. These cover one op per
//! width, which is one per `narrow_int_kernel_helpers` expansion, plus the
//! `outer * reduced * inner` indexing that needs a rank-3 middle axis to run
//! with both extents above one, and the `prod` parity rules that record no
//! scan at all.

use luminal::prelude::*;
use luminal::shape::SymbolBounds;
use luminal_reference::harness::{run_reference, run_reference_bound};
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------- integer arms

#[test]
fn i8_prod_scan() {
    let mut cx = Graph::new();
    let a = cx.tensor((3,), DType::I8);
    let out = a.cumprod(0);
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::I8(vec![2, 3, 5]))]);
    assert_eq!(rt.get_i8(out.id).expect("i8 output"), &vec![2, 6, 30]);
}

#[test]
fn u8_sum_scan() {
    let mut cx = Graph::new();
    let a = cx.tensor((3,), DType::U8);
    let out = a.cumsum(0);
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::U8(vec![10, 20, 30]))]);
    assert_eq!(rt.get_u8(out.id).expect("u8 output"), &vec![10, 30, 60]);
}

#[test]
fn i16_max_scan() {
    let mut cx = Graph::new();
    let a = cx.tensor((4,), DType::I16);
    let out = a.cummax(0);
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::I16(vec![5, -3, 9, 9]))]);
    assert_eq!(rt.get_i16(out.id).expect("i16 output"), &vec![5, 5, 9, 9]);
}

/// The scans accumulate with `checked_mul` at every width, so a narrow product
/// that leaves range refuses rather than wrapping. This diverges from the
/// narrow-int carve-out that `narrow_int_kernel_helpers` documents and that
/// `ops/add` and `ops/reduce_sum` implement with `wrapping_add`: there, I8, U8
/// and I16 wrap at their own width and only I32 and I64 are non-wrapping.
/// Pinned as it behaves today; `int_prod_overflow_fails_loudly` pins the same
/// refusal at I32, where it matches the carve-out.
#[test]
fn narrow_prod_scan_refuses_on_overflow() {
    let mut cx = Graph::new();
    let a = cx.tensor((2,), DType::I8);
    let _out = a.cumprod(0);
    let mut rt = ReferenceRuntime::load(&cx).expect("load");
    let data: FxHashMap<_, TypedBuffer> = [(a.id, TypedBuffer::I8(vec![100, 2]))]
        .into_iter()
        .collect();
    // The search profiles by executing, so the refusal surfaces there.
    let error = rt
        .search(
            &SymbolBounds::default(),
            &FxHashMap::default(),
            &data,
            &luminal_reference::harness_search_options(),
        )
        .expect_err("an i8 product leaving range must refuse");
    let message = format!("{error:#}");
    assert!(
        message.contains("i8 scan-prod overflow"),
        "expected the narrow overflow refusal, got: {message}"
    );
}

/// `prod` builds its empty-axis fill in the operand's own dtype, because a
/// float literal cast to an integer tensor is a refused lossy read
/// (the dtype match in `GraphTensor::prod`). That integer branch had no coverage.
#[test]
fn prod_over_an_empty_int_axis_is_one() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 0), DType::Int);
    let out = a.prod(1);
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::I32(vec![]))]);
    assert_eq!(rt.get_i32(out.id).expect("i32 output"), &vec![1, 1]);
}

// ------------------------------------------------------------------- geometry

/// `scan_axis` indexes as outer * reduced * inner. Every other scan test in
/// the repo is rank 1 or rank 2, so the middle axis of a rank-3 operand is the
/// only shape where both outer and inner exceed one.
#[test]
fn scan_runs_per_outer_and_inner_pair_on_a_middle_axis() {
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3, 2), DType::F32);
    let out = a.cumsum(1);
    let input: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::F32(input))]);
    // Each (outer, inner) pair scans down the middle axis independently:
    // column [1,3,5] -> [1,4,9] and [2,4,6] -> [2,6,12], then [7,9,11] and
    // [8,10,12] in the second outer slice.
    assert_eq!(
        rt.get_f32(out.id).expect("f32 output"),
        &vec![
            1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 7.0, 8.0, 16.0, 18.0, 27.0, 30.0
        ]
    );
}

/// Core proptests `cumprod(1)` on a 2-D operand but only `cumsum` on axis 0,
/// so this pins the leading-axis product exactly.
#[test]
fn leading_axis_cumprod_is_exact() {
    let mut cx = Graph::new();
    let a = cx.tensor((3, 2), DType::F32);
    let out = a.cumprod(0);
    let rt = run_reference(
        &cx,
        &[(a.id, TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))],
    );
    assert_eq!(
        rt.get_f32(out.id).expect("f32 output"),
        &vec![1.0, 2.0, 3.0, 8.0, 15.0, 48.0]
    );
}

// -------------------------------------------------------------- prod parities

/// A rank-0 product is the operand itself (`GraphTensor::prod` returns the operand when it has no dims), so
/// the output is a pass-through of an input and names no leaf. That is what
/// `run_reference_bound` is for; binding the result explicitly is the same
/// treatment `pow(x, 1)` gets in `run_pow`.
#[test]
fn prod_of_a_rank0_operand_is_the_value() {
    let mut cx = Graph::new();
    let a = cx.tensor((), DType::F32);
    let out = a.prod(());
    let bindings = ReferenceBindings::dense(&cx.logical, &[out.id]);
    let rt = run_reference_bound(&cx, bindings, &[(a.id, vec![7.0f32].into())], &[]);
    assert_eq!(rt.get_f32(out.id).expect("f32 output"), &vec![7.0]);
}

// ------------------------------------------------------------- exact non-finite

/// The scans are exact, so a non-finite value propagates through the fold
/// rather than being absorbed. `inf * 0` is NaN and then poisons the tail,
/// which the old `exp(sum(log(x)))` spelling of a product could not express.
#[test]
fn cumprod_propagates_infinity_and_nan() {
    let mut cx = Graph::new();
    let a = cx.tensor((4,), DType::F32);
    let out = a.cumprod(0);
    let rt = run_reference(
        &cx,
        &[(a.id, TypedBuffer::F32(vec![1.0, f32::INFINITY, 0.0, 2.0]))],
    );
    let got = rt.get_f32(out.id).expect("f32 output");
    assert_eq!(got[0], 1.0);
    assert!(got[1].is_infinite() && got[1].is_sign_positive());
    assert!(got[2].is_nan(), "inf * 0 is NaN, got {}", got[2]);
    assert!(got[3].is_nan(), "NaN * 2 stays NaN, got {}", got[3]);
}

/// Two different scan ops chained. Core records a scan feeding a scan via
/// `cumsum((0,1))`; this adds mixing two kinds.
#[test]
fn scans_compose() {
    let mut cx = Graph::new();
    let a = cx.tensor((4,), DType::F32);
    let out = a.cumprod(0).cumsum(0);
    let rt = run_reference(&cx, &[(a.id, TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]))]);
    // cumprod -> [1, 2, 6, 24], then cumsum -> [1, 3, 9, 33]
    assert_eq!(
        rt.get_f32(out.id).expect("f32 output"),
        &vec![1.0, 3.0, 9.0, 33.0]
    );
}
