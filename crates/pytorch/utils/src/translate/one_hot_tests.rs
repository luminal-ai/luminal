use luminal::prelude::*;

use super::test_support::*;

#[test]
fn one_hot_adds_a_class_axis_and_preserves_the_declared_long_output() {
    let translation = translate_one(
        node(
            "torch.ops.aten.one_hot.default",
            vec![tensor_input("self", "input"), scalar_int("num_classes", 5)],
            &["output"],
        ),
        &[("input", LONG, &[2, 3]), ("output", LONG, &[2, 3, 5])],
        &["input"],
        &["output"],
    );

    let output = &translation.outputs[0];
    assert_eq!(output.dtype, DType::I64);
    assert_eq!(
        output
            .shape
            .iter()
            .map(|dim| dim.as_num().expect("static fixture dimension"))
            .collect::<Vec<_>>(),
        vec![2, 3, 5]
    );
    assert_eq!(
        translation.graph.logical.value_dtype(output.tensor),
        DType::I64
    );
}

#[test]
fn one_hot_refuses_non_long_inputs() {
    let parsed = parsed_nodes(
        vec![node(
            "torch.ops.aten.one_hot.default",
            vec![tensor_input("self", "input"), scalar_int("num_classes", 5)],
            &["output"],
        )],
        &[("input", FLOAT, &[3]), ("output", INT, &[3, 5])],
        &["input"],
        &["output"],
    );
    let error = match crate::translate::translate(&parsed) {
        Ok(_) => panic!("one_hot unexpectedly accepted a float input"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("one_hot requires an int64 input"));
}
