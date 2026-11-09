use super::test_support::*;
use crate::pt2_schema::Argument;

#[test]
fn custom_all_gather_preserves_identity_shape_and_dtype() {
    let translated = translate_one(
        node(
            "torch.ops.luminal_collective.all_gather.default",
            vec![
                tensor_input("input", "x"),
                input(
                    "group",
                    Argument::Other(serde_json::json!({"as_string": "workers"})),
                ),
                scalar_int("width", 3),
                scalar_int("sequence", 7),
            ],
            &["y"],
        ),
        &[("x", FLOAT, &[2, 4]), ("y", FLOAT, &[3, 2, 4])],
        &["x"],
        &["y"],
    );
    let output = translated.outputs[0].tensor;
    let text = translated.graph.logical.render(&[output]).unwrap();
    assert!(text.contains("LogicalAllGather"), "{text}");
    assert!(text.contains("\"workers\" 3 7"), "{text}");
}
