//! Optional PT2 arguments are named schema fields; omitted defaults do not
//! occupy positional slots in the serialized input list.
use super::test_support::*;
use crate::pt2_schema::NodeInput;
use luminal_reference::harness::run_reference;

fn assert_clamp(target: &str, bounds: Vec<NodeInput>, expected: &[f32]) {
    let mut args = vec![tensor_input("self", "x")];
    args.extend(bounds);
    let t = translate_one(
        node(target, args, &["y"]),
        &[
            ("x", FLOAT, &[2, 3]),
            ("lo", FLOAT, &[3]),
            ("hi", FLOAT, &[]),
            ("y", FLOAT, &[2, 3]),
        ],
        &["x", "lo", "hi"],
        &["y"],
    );
    let data = [
        &[-2.0f32, -0.5, 2.0, -1.0, 3.0, -4.0][..],
        &[-1.0, 1.0, 0.0],
        &[1.5],
    ];
    let inputs = t
        .inputs
        .iter()
        .zip(data)
        .map(|(input, data)| (input.tensor, data.to_vec().into()))
        .collect::<Vec<_>>();
    let rt = run_reference(&t.graph, &inputs);
    assert_eq!(rt.get_f32(t.outputs[0].tensor).unwrap(), expected);
}

#[test]
fn tensor_clamp_accepts_omitted_none_and_reordered_bounds() {
    let min = || tensor_input("min", "lo");
    let max = || tensor_input("max", "hi");
    for (args, expected) in [
        (vec![min()], vec![-1., 1., 2., -1., 3., 0.]),
        (vec![max()], vec![-2., -0.5, 1.5, -1., 1.5, -4.]),
        (vec![min(), none("max")], vec![-1., 1., 2., -1., 3., 0.]),
        (
            vec![none("min"), max()],
            vec![-2., -0.5, 1.5, -1., 1.5, -4.],
        ),
        (vec![min(), max()], vec![-1., 1., 1.5, -1., 1.5, 0.]),
        (vec![max(), min()], vec![-1., 1., 1.5, -1., 1.5, 0.]),
    ] {
        assert_clamp("torch.ops.aten.clamp.Tensor", args, &expected);
    }
}

#[test]
fn scalar_clamp_accepts_omitted_none_and_reordered_bounds() {
    let min = || scalar_float("min", -1.0);
    let max = || scalar_float("max", 1.5);
    for (args, expected) in [
        (vec![min()], vec![-1., -0.5, 2., -1., 3., -1.]),
        (vec![max()], vec![-2., -0.5, 1.5, -1., 1.5, -4.]),
        (vec![min(), none("max")], vec![-1., -0.5, 2., -1., 3., -1.]),
        (
            vec![none("min"), max()],
            vec![-2., -0.5, 1.5, -1., 1.5, -4.],
        ),
        (vec![min(), max()], vec![-1., -0.5, 1.5, -1., 1.5, -1.]),
        (vec![max(), min()], vec![-1., -0.5, 1.5, -1., 1.5, -1.]),
    ] {
        assert_clamp("torch.ops.aten.clamp.default", args, &expected);
    }
}

#[test]
fn tensor_clamp_min_accepts_scalar_input_and_bound() {
    let t = translate_one(
        node(
            "torch.ops.aten.clamp.Tensor",
            vec![tensor_input("self", "x"), tensor_input("min", "lo")],
            &["y"],
        ),
        &[("x", FLOAT, &[]), ("lo", FLOAT, &[]), ("y", FLOAT, &[])],
        &["x", "lo"],
        &["y"],
    );
    let rt = run_reference(
        &t.graph,
        &[
            (t.inputs[0].tensor, vec![-2.0f32].into()),
            (t.inputs[1].tensor, vec![1.5f32].into()),
        ],
    );
    assert_eq!(rt.get_f32(t.outputs[0].tensor).unwrap(), &[1.5]);
}
