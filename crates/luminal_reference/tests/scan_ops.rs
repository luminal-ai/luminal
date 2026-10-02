//! End-to-end tests for the three scan logical ops, `UnspecifiedOrderScanSum`,
//! `UnspecifiedOrderScanProd` and `UnspecifiedOrderScanMax`, on the reference
//! runtime. #540 noted the new core ops carry no standalone reference
//! coverage; these pin the semantics the frontend documents, with negative
//! inputs on the product path (the case the old `log`/`exp` spelling of `prod`
//! returned NaN for) and the empty and rank-0 parity rules `prod` states.

use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

/// Run a single-input graph on the reference runtime and read F32 output.
fn run_f32(build: impl Fn(&mut Graph) -> (GraphTensor, GraphTensor), input: Vec<f32>) -> Vec<f32> {
    let mut cx = Graph::new();
    let (operand, out) = build(&mut cx);

    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(operand.id, TypedBuffer::F32(input.clone()));
    runtime
        .search(
            &SymbolBounds::default(),
            &FxHashMap::default(),
            &data,
            &harness_search_options(),
        )
        .expect("search");

    runtime.set_data(operand.id, TypedBuffer::F32(input));
    runtime.execute().expect("execute");
    runtime.get_f32(out.id).expect("f32 output").to_vec()
}

#[test]
fn cumsum_is_an_inclusive_scan() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((4,), DType::F32);
            (a, a.cumsum(0))
        },
        vec![1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0]);
}

#[test]
fn cummax_is_an_inclusive_scan() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((5,), DType::F32);
            (a, a.cummax(0))
        },
        vec![1.0, 3.0, 2.0, 5.0, 4.0],
    );
    assert_eq!(got, vec![1.0, 3.0, 3.0, 5.0, 5.0]);
}

#[test]
fn cumprod_is_an_inclusive_scan() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((4,), DType::F32);
            (a, a.cumprod(0))
        },
        vec![1.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(got, vec![1.0, 2.0, 6.0, 24.0]);
}

/// The old `prod` was `exp(sum(log(x)))`, which returned NaN for any negative
/// element. The scan spelling is exact, so signs survive and the parity of the
/// negative count decides the result's sign.
#[test]
fn cumprod_carries_signs_through_negatives() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((4,), DType::F32);
            (a, a.cumprod(0))
        },
        vec![-1.0, 2.0, -3.0, 4.0],
    );
    assert_eq!(got, vec![-1.0, -2.0, 6.0, 24.0]);
}

#[test]
fn prod_reduces_negatives_without_nan() {
    for (input, expected) in [
        (vec![-2.0f32, 3.0], -6.0f32),
        (vec![-2.0, -3.0], 6.0),
        (vec![2.0, 3.0], 6.0),
        (vec![-2.0, 0.0], 0.0),
    ] {
        let cols = input.len();
        let got = run_f32(
            move |cx| {
                let a = cx.tensor((1, cols), DType::F32);
                (a, a.prod(1))
            },
            input.clone(),
        );
        assert_eq!(got.len(), 1, "one row reduces to one value");
        assert!(got[0].is_finite(), "prod({input:?}) = {}", got[0]);
        assert_eq!(got[0], expected, "prod({input:?})");
    }
}

#[test]
fn prod_over_a_statically_empty_axis_is_one() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((1, 0), DType::F32);
            (a, a.prod(1))
        },
        vec![],
    );
    assert_eq!(got, vec![1.0]);
}

#[test]
#[ignore = "blocked by #563: a rank-0 prod returns the operand itself, and a graph whose output is also an input refuses to load"]
fn prod_of_a_rank0_operand_is_the_value() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((), DType::F32);
            (a, a.prod(()))
        },
        vec![7.0],
    );
    assert_eq!(got, vec![7.0]);
}

/// A scan keeps the operand's dims, so scanning the inner axis of a 2-D
/// operand runs one independent scan per row.
#[test]
fn cumsum_scans_each_row_independently() {
    let got = run_f32(
        |cx| {
            let a = cx.tensor((2, 3), DType::F32);
            (a, a.cumsum(1))
        },
        vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    );
    assert_eq!(got, vec![1.0, 3.0, 6.0, 10.0, 30.0, 60.0]);
}
