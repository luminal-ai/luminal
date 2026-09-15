//! End-to-end gcd: translate the select-based Euclidean unroll, declare
//! non-negative input ranges (the proof-gated I64 `trunc_rem` needs a
//! zero-free divisor), then plan and execute on the reference runtime.
//!
//! The divisor is `maximum(|rhs|, 1)`: the structural max bound rule in
//! `logical_op/select` derives its lower bound `1` statically, and the
//! sign-aware `trunc_rem` bounds rule keeps the remainder bounded. No
//! runtime logic is involved.
//!
//! The 32-round unroll is a deep DAG; this test runs on the default test
//! stack on purpose, guarding the extractor's iterative (non-recursive)
//! traversal.

use luminal_pytorch_utils::{InputKind, parse_pt2, translate};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

#[test]
fn gcd_plans_and_runs_with_declared_ranges() {
    let path = format!(
        "{}/../../../target/luminal_fixtures/b7_gcd.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    let translation = translate(&parsed).expect("translate gcd");

    let user_inputs: Vec<_> = translation
        .inputs
        .iter()
        .filter(|input| matches!(input.kind, InputKind::UserInput { .. }))
        .collect();
    assert_eq!(user_inputs.len(), 2, "gcd has two user inputs");

    // Mixed signs: the divisor is `|rhs|`, so signed inputs must work.
    let a: Vec<i64> = vec![
        -12, 18, -24, 7, 9, -14, 25, -5, 8, -15, 21, -6, 11, -13, 17, -19,
    ];
    let b: Vec<i64> = vec![
        18, -24, 35, -5, -6, 21, -30, 10, -12, 9, -14, 3, -22, 26, -34, 38,
    ];

    let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
    // Attest that the caller's data is bounded (the divisor |rhs| then has
    // the max-rule lower bound 1, and the sign-aware remainder bounds keep
    // the loop's values bounded).
    for input in &user_inputs {
        runtime
            .bind_value_range(input.tensor, -1024, 1024)
            .expect("value range binds");
    }

    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(user_inputs[0].tensor, TypedBuffer::I64(a.clone()));
    data.insert(user_inputs[1].tensor, TypedBuffer::I64(b.clone()));
    runtime
        .search(&data, &harness_search_options())
        .expect("gcd plans once ranges are declared");

    runtime.set_data(user_inputs[0].tensor, TypedBuffer::I64(a.clone()));
    runtime.set_data(user_inputs[1].tensor, TypedBuffer::I64(b.clone()));
    runtime.execute().expect("gcd executes");

    let got = runtime
        .get_i64(translation.outputs[0].tensor)
        .expect("i64 output");
    let expected: Vec<i64> = a.iter().zip(&b).map(|(x, y)| gcd(*x, *y)).collect();
    assert_eq!(got, &expected);
}
