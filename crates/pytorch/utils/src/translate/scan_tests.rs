//! Facts about the scans torch's `prod`/`cumsum`/`cumprod` translate to:
//! the reduced axis is the one the node names, and the recorded values are
//! the exact left-to-right fold, signs and all.

use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

use super::Translation;
use super::test_support::*;
use crate::pt2_schema::NodeInput;

/// The dims of the value the translator actually produced for the first
/// output (not the declared boundary shape).
fn output_dims(t: &Translation) -> Vec<usize> {
    t.graph
        .logical
        .value_dims(t.outputs[0].tensor)
        .iter()
        .map(|extent| extent.to_usize().expect("a literal extent"))
        .collect()
}

/// Run the translation's one F32 input through the reference runtime and
/// read back the first output.
fn run_f32(t: &Translation, x: &[f32]) -> Vec<f32> {
    let mut runtime = ReferenceRuntime::load(&t.graph).expect("load");
    let dims = t
        .dims
        .iter()
        .map(|(&symbol, &hint)| (symbol, hint))
        .collect();
    let bounds = luminal::shape::SymbolBounds::exact(&dims).unwrap();
    let input = t.inputs[0].tensor;
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(input, TypedBuffer::F32(x.to_vec()));
    runtime
        .search(&bounds, &dims, &data, &harness_search_options())
        .expect("search");
    runtime.set_data(input, TypedBuffer::F32(x.to_vec()));
    runtime.execute().expect("execute");
    runtime
        .get_f32(t.outputs[0].tensor)
        .expect("f32 output")
        .clone()
}

fn assert_exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), e.to_bits(), "index {index}: got {a}, want {e}");
    }
}

/// A one-node program over the F32 input `x` and the F32 output `y`.
fn one_node(
    target: &str,
    args: Vec<NodeInput>,
    in_shape: &[i64],
    out_shape: &[i64],
) -> Translation {
    translate_one(
        node(target, args, &["y"]),
        &[("x", FLOAT, in_shape), ("y", FLOAT, out_shape)],
        &["x"],
        &["y"],
    )
}

#[test]
fn prod_dim_int_reduces_only_the_named_axis() {
    let t = one_node(
        "torch.ops.aten.prod.dim_int",
        vec![
            tensor_input("self", "x"),
            scalar_int("dim", 1),
            scalar_bool("keepdim", false),
            none("dtype"),
        ],
        &[2, 2],
        &[2],
    );
    assert_eq!(output_dims(&t), vec![2], "only axis 1 is reduced");
    assert_exact(&run_f32(&t, &[-2.0, 3.0, -2.0, -3.0]), &[-6.0, 6.0]);

    let kept = one_node(
        "torch.ops.aten.prod.dim_int",
        vec![
            tensor_input("self", "x"),
            scalar_int("dim", 1),
            scalar_bool("keepdim", true),
            none("dtype"),
        ],
        &[2, 2],
        &[2, 1],
    );
    assert_eq!(output_dims(&kept), vec![2, 1], "keepdim keeps the axis");
    assert_exact(&run_f32(&kept, &[-2.0, 3.0, -2.0, -3.0]), &[-6.0, 6.0]);
}

#[test]
fn cumprod_default_is_exact_on_negatives() {
    let t = one_node(
        "torch.ops.aten.cumprod.default",
        vec![
            tensor_input("self", "x"),
            scalar_int("dim", 0),
            none("dtype"),
        ],
        &[4],
        &[4],
    );
    assert_exact(
        &run_f32(&t, &[-1.0, 2.0, -3.0, 4.0]),
        &[-1.0, -2.0, 6.0, 24.0],
    );
}

#[test]
fn cumsum_default_matches_sequential_fold() {
    let input = [0.1f32, 0.2, 0.3, 0.4];
    let mut expected = Vec::with_capacity(input.len());
    let mut acc = 0.0f32;
    for value in input {
        acc += value;
        expected.push(acc);
    }
    let t = one_node(
        "torch.ops.aten.cumsum.default",
        vec![
            tensor_input("self", "x"),
            scalar_int("dim", 0),
            none("dtype"),
        ],
        &[4],
        &[4],
    );
    assert_exact(&run_f32(&t, &input), &expected);
}

#[test]
fn prod_default_full_reduce() {
    let t = one_node(
        "torch.ops.aten.prod.default",
        vec![tensor_input("self", "x"), none("dtype")],
        &[2],
        &[],
    );
    // A full reduction squeezes every axis: the output is a rank-0 scalar.
    assert_eq!(output_dims(&t), Vec::<usize>::new());
    assert_exact(&run_f32(&t, &[-2.0, 3.0]), &[-6.0]);
}

/// `var.correction` with `dim` omitted: PT2 drops omitted defaults, so the
/// integer `correction` sits where `dim` would. Reading it as a dim would
/// reduce axis 0 instead of the whole value.
#[test]
fn var_correction_without_dim_reduces_everything() {
    let t = one_node(
        "torch.ops.aten.var.correction",
        vec![tensor_input("self", "x"), scalar_int("correction", 0)],
        &[3, 4],
        &[],
    );
    let x: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let got = run_f32(&t, &x);
    assert_eq!(got.len(), 1, "a full reduction is one value");
    assert!(
        (got[0] - 143.0 / 12.0).abs() < 1e-5,
        "got {}, want {}",
        got[0],
        143.0 / 12.0
    );
}

/// A rank-0 value has no axis: torch's `prod.dim_int` and `cumsum.default`
/// both return it unchanged rather than folding. The translation records no
/// op at all, so the output value IS the input value.
#[test]
fn rank_zero_prod_and_cumsum_are_the_identity() {
    for (target, args) in [
        (
            "torch.ops.aten.prod.dim_int",
            vec![
                tensor_input("self", "x"),
                scalar_int("dim", 0),
                scalar_bool("keepdim", false),
                none("dtype"),
            ],
        ),
        (
            "torch.ops.aten.cumsum.default",
            vec![
                tensor_input("self", "x"),
                scalar_int("dim", 0),
                none("dtype"),
            ],
        ),
    ] {
        let t = one_node(target, args, &[], &[]);
        assert_eq!(
            output_dims(&t),
            Vec::<usize>::new(),
            "{target} stays rank-0"
        );
        assert_eq!(
            t.outputs[0].tensor, t.inputs[0].tensor,
            "{target} on a rank-0 value records nothing"
        );
    }
}
