//! Boolean indexing consumes the mask axes, rather than right-aligning them.

use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

use super::Translation;
use super::test_support::*;
use crate::pt2_schema::Argument;

fn boolean_put(
    dtype: u32,
    shape: &[i64],
    mask: &[i64],
    axis: usize,
    values: &[i64],
) -> Translation {
    let mut indices = vec![serde_json::json!({"as_none": true}); axis];
    indices.push(serde_json::json!({"as_tensor": {"name": "mask"}}));
    let indices: Argument = serde_json::from_value(serde_json::json!({
        "as_optional_tensors": indices,
    }))
    .unwrap();
    translate_one(
        node(
            "torch.ops.aten.index_put.default",
            vec![
                tensor_input("self", "x"),
                input("indices", indices),
                tensor_input("values", "v"),
                scalar_bool("accumulate", false),
            ],
            &["y"],
        ),
        &[
            ("x", dtype, shape),
            ("mask", BOOL, mask),
            ("v", dtype, values),
            ("y", dtype, shape),
        ],
        &["x", "mask", "v"],
        &["y"],
    )
}

fn run(t: &Translation, inputs: [TypedBuffer; 3]) -> ReferenceRuntime {
    let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
    let dims = t.dims.iter().map(|(&s, &hint)| (s, hint)).collect();
    let bounds = luminal::shape::SymbolBounds::exact(&dims).unwrap();
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(inputs)
        .map(|(input, values)| (input.tensor, values))
        .collect();
    runtime
        .search(&bounds, &dims, &data, &harness_search_options())
        .unwrap();
    for (input, values) in data {
        runtime.set_data(input, values);
    }
    runtime.execute().unwrap();
    runtime
}

#[test]
fn boolean_index_put_masks_prefix_rows_with_exact_selection() {
    let t = boolean_put(FLOAT, &[1, 2, 3], &[1, 2], 0, &[]);
    let runtime = run(
        &t,
        [
            TypedBuffer::F32(vec![f32::NAN, f32::INFINITY, -0.0, 4.0, 5.0, 6.0]),
            TypedBuffer::Bool8(vec![1, 0]),
            TypedBuffer::F32(vec![7.0]),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[7.0, 7.0, 7.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn boolean_index_put_broadcasts_updates_over_selected_rows_not_input_axes() {
    // x[:, mask, :] = v; v's first dimension selects true mask slots,
    // not x's leading basic axis, even though both happen to have size 2.
    let t = boolean_put(FLOAT, &[2, 3, 2], &[3], 1, &[2, 1]);
    let runtime = run(
        &t,
        [
            TypedBuffer::F32((0..12).map(|x| x as f32).collect()),
            TypedBuffer::Bool8(vec![1, 0, 1]),
            TypedBuffer::F32(vec![20.0, 30.0]),
        ],
    );
    assert_eq!(
        runtime.get_f32(t.outputs[0].tensor).unwrap(),
        &[
            20.0, 20.0, 2.0, 3.0, 30.0, 30.0, 20.0, 20.0, 8.0, 9.0, 30.0, 30.0
        ]
    );
}

#[test]
fn boolean_index_put_preserves_int64_bits() {
    let t = boolean_put(5, &[2, 2], &[2], 0, &[2]);
    let runtime = run(
        &t,
        [
            TypedBuffer::I64(vec![
                i64::MIN,
                i64::MAX,
                (1_i64 << 55) + 1,
                -(1_i64 << 55) - 1,
            ]),
            TypedBuffer::Bool8(vec![1, 0]),
            TypedBuffer::I64(vec![i64::MAX - 1, i64::MIN + 1]),
        ],
    );
    assert_eq!(
        runtime.get_i64(t.outputs[0].tensor).unwrap(),
        &[
            i64::MAX - 1,
            i64::MIN + 1,
            (1_i64 << 55) + 1,
            -(1_i64 << 55) - 1
        ]
    );
}

#[test]
fn boolean_index_put_handles_full_empty_and_scalar_masks() {
    for (mask_shape, mask, value_shape, values, expected) in [
        (
            vec![2, 2],
            vec![0, 1, 1, 0],
            vec![2],
            vec![8.0, 9.0],
            vec![1.0, 8.0, 9.0, 4.0],
        ),
        (vec![], vec![1], vec![], vec![8.0], vec![8.0; 4]),
        (vec![], vec![0], vec![], vec![8.0], vec![1.0, 2.0, 3.0, 4.0]),
    ] {
        let t = boolean_put(FLOAT, &[2, 2], &mask_shape, 0, &value_shape);
        let runtime = run(
            &t,
            [
                TypedBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]),
                TypedBuffer::Bool8(mask),
                TypedBuffer::F32(values),
            ],
        );
        assert_eq!(runtime.get_f32(t.outputs[0].tensor).unwrap(), &expected);
    }
    let empty = boolean_put(FLOAT, &[2, 2], &[2], 0, &[0, 2]);
    assert_eq!(empty.outputs[0].tensor, empty.inputs[0].tensor);
}
