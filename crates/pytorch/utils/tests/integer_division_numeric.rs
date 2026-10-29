//! Python's integer `%` and `//` on the reference runtime, with negative
//! operands. The remainder is spelled in exact Int (truncated remainder,
//! moved by one divisor where it disagrees with the divisor's sign) and is
//! bounded by its divisor alone, so a literal divisor needs no attested
//! range at all and a tensor divisor needs only its own. Floor division's
//! quotient is bounded through its dividend, so those programs attest the
//! dividend's range, as core's non-wrapping ruling requires.
//!
//! Fixtures live in `<workspace>/target/luminal_fixtures`; tests skip when
//! absent.
use luminal_pytorch_utils::{InputKind, Translation, parse_pt2, translate};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

const A: [i64; 8] = [7, -7, 13, -13, 0, 5, -1, 22];
const B: [i64; 8] = [2, 2, 4, 4, 3, 5, 3, 7];

fn python_floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn python_rem(a: i64, b: i64) -> i64 {
    a - python_floor_div(a, b) * b
}

fn load(name: &str) -> Option<Translation> {
    let path = format!(
        "{}/../../../target/luminal_fixtures/{name}.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    Some(translate(&parsed).expect("translate"))
}

/// Plan and run with the given inputs, attesting the listed ranges by input
/// position; the inputs and outputs are 32-bit integers when `narrow`.
/// Returns every output.
fn run(
    translation: &Translation,
    inputs: &[&[i64]],
    ranges: &[Option<(i64, i64)>],
    narrow: bool,
) -> Vec<Vec<i64>> {
    let buffer = |values: &[i64]| {
        if narrow {
            TypedBuffer::I32(values.iter().map(|v| *v as i32).collect())
        } else {
            TypedBuffer::I64(values.to_vec())
        }
    };
    let user_inputs: Vec<_> = translation
        .inputs
        .iter()
        .filter(|input| matches!(input.kind, InputKind::UserInput { .. }))
        .collect();
    assert_eq!(user_inputs.len(), inputs.len());
    let mut runtime = ReferenceRuntime::load(&translation.graph).expect("load");
    for (input, range) in user_inputs.iter().zip(ranges) {
        if let Some((lower, upper)) = range {
            runtime
                .bind_value_range(input.tensor, *lower, *upper)
                .expect("value range binds");
        }
    }
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    for (input, values) in user_inputs.iter().zip(inputs) {
        data.insert(input.tensor, buffer(values));
    }
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .expect("plans");
    for (input, values) in user_inputs.iter().zip(inputs) {
        runtime.set_data(input.tensor, buffer(values));
    }
    runtime.execute().expect("executes");
    translation
        .outputs
        .iter()
        .map(|output| {
            if narrow {
                runtime
                    .get_i32(output.tensor)
                    .expect("i32 output")
                    .iter()
                    .map(|v| *v as i64)
                    .collect()
            } else {
                runtime.get_i64(output.tensor).expect("i64 output").clone()
            }
        })
        .collect()
}

#[test]
fn remainder_by_a_literal_needs_no_attested_range() {
    let Some(translation) = load("integer_remainder_scalar") else {
        return;
    };
    let outputs = run(&translation, &[&A], &[None], false);
    assert_eq!(
        outputs[0],
        A.iter().map(|a| python_rem(*a, -3)).collect::<Vec<_>>(),
        "a % -3"
    );
    assert_eq!(
        outputs[1],
        A.iter().map(|a| python_rem(*a, 4)).collect::<Vec<_>>(),
        "a % 4"
    );
}

#[test]
fn remainder_by_a_tensor_needs_only_the_divisors_range() {
    let Some(translation) = load("integer_remainder_tensor") else {
        return;
    };
    let outputs = run(&translation, &[&A, &B], &[None, Some((1, 8))], false);
    let expected: Vec<i64> = A.iter().zip(&B).map(|(a, b)| python_rem(*a, *b)).collect();
    assert_eq!(outputs[0], expected, "a % b");
}

#[test]
fn floor_division_by_a_literal_rounds_toward_negative_infinity() {
    let Some(translation) = load("integer_floor_div_scalar") else {
        return;
    };
    let outputs = run(&translation, &[&A], &[Some((-64, 64))], false);
    assert_eq!(
        outputs[0],
        A.iter()
            .map(|a| python_floor_div(*a, 4))
            .collect::<Vec<_>>(),
        "a // 4"
    );
    assert_eq!(
        outputs[1],
        A.iter()
            .map(|a| python_floor_div(*a, -3))
            .collect::<Vec<_>>(),
        "a // -3"
    );
}

#[test]
fn floor_division_by_a_tensor_rounds_toward_negative_infinity() {
    let Some(translation) = load("integer_floor_div_tensor") else {
        return;
    };
    let outputs = run(
        &translation,
        &[&A, &B],
        &[Some((-64, 64)), Some((1, 8))],
        false,
    );
    let expected: Vec<i64> = A
        .iter()
        .zip(&B)
        .map(|(a, b)| python_floor_div(*a, *b))
        .collect();
    assert_eq!(outputs[0], expected, "a // b");
}

/// The 32-bit program: `%` and `//` by literals on int32 data. Its literals
/// are one iota each, not an I64 limb assembly, so a backend without I64
/// kernels can take it too; `//` still attests the dividend's range.
#[test]
fn int32_division_by_literals() {
    let Some(translation) = load("int32_division") else {
        return;
    };
    let outputs = run(&translation, &[&A], &[Some((-64, 64))], true);
    assert_eq!(
        outputs[0],
        A.iter().map(|a| python_rem(*a, 4)).collect::<Vec<_>>(),
        "a % 4"
    );
    assert_eq!(
        outputs[1],
        A.iter().map(|a| python_rem(*a, -3)).collect::<Vec<_>>(),
        "a % -3"
    );
    assert_eq!(
        outputs[2],
        A.iter()
            .map(|a| python_floor_div(*a, 4))
            .collect::<Vec<_>>(),
        "a // 4"
    );
    assert_eq!(
        outputs[3],
        A.iter()
            .map(|a| python_floor_div(*a, -3))
            .collect::<Vec<_>>(),
        "a // -3"
    );
}
