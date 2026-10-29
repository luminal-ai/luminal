use super::test_support::*;

#[test]
fn unbind_dispatches_to_logical_slices() {
    for target in [
        "torch.ops.aten.unbind.int",
        "torch.ops.aten.unbind_copy.int",
    ] {
        let t = translate_one(
            node(
                target,
                vec![tensor_input("self", "x"), scalar_int("dim", -1)],
                &["a", "b"],
            ),
            &[
                ("x", FLOAT, &[3, 2]),
                ("a", FLOAT, &[3]),
                ("b", FLOAT, &[3]),
            ],
            &["x"],
            &["a", "b"],
        );
        assert_eq!(t.outputs.len(), 2);
        assert_eq!(t.outputs[0].shape[0].to_usize(), Some(3));
        assert_eq!(t.outputs[1].shape[0].to_usize(), Some(3));
    }
}

#[test]
fn unknown_aten_is_rejected_at_dispatch() {
    let parsed = parsed_nodes(
        vec![node(
            "torch.ops.aten.definitely_unknown.default",
            vec![tensor_input("self", "x")],
            &["y"],
        )],
        &[("x", FLOAT, &[3]), ("y", FLOAT, &[3])],
        &["x"],
        &["y"],
    );
    let error = super::translate(&parsed)
        .err()
        .expect("unknown op must fail");
    assert!(format!("{error:#}").contains("unsupported ATen op"));
}
