use super::test_support::*;

#[test]
fn repeat_left_pads_missing_input_axes() {
    let translated = translate_one(
        node(
            "torch.ops.aten.repeat.default",
            vec![tensor_input("self", "x"), ints("repeats", &[4, 1, 1])],
            &["y"],
        ),
        &[("x", FLOAT, &[2, 3]), ("y", FLOAT, &[4, 2, 3])],
        &["x"],
        &["y"],
    );

    assert_eq!(
        translated.outputs[0]
            .shape
            .iter()
            .map(|dim| dim.to_usize())
            .collect::<Option<Vec<_>>>(),
        Some(vec![4, 2, 3])
    );
}

#[test]
fn repeat_can_left_pad_and_tile_an_existing_axis() {
    let translated = translate_one(
        node(
            "torch.ops.aten.repeat.default",
            vec![tensor_input("self", "x"), ints("repeats", &[2, 3])],
            &["y"],
        ),
        &[("x", FLOAT, &[3]), ("y", FLOAT, &[2, 9])],
        &["x"],
        &["y"],
    );

    assert_eq!(
        translated.outputs[0]
            .shape
            .iter()
            .map(|dim| dim.to_usize())
            .collect::<Option<Vec<_>>>(),
        Some(vec![2, 9])
    );
}
