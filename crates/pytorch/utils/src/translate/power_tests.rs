use super::Translation;
use super::test_support::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn run(t: &Translation, values: Vec<TypedBuffer>) -> Vec<f32> {
    let data: FxHashMap<_, _> = t
        .inputs
        .iter()
        .zip(&values)
        .map(|(input, value)| (input.tensor, value.clone()))
        .collect();
    let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    for (input, value) in t.inputs.iter().zip(values) {
        runtime.set_data(input.tensor, value);
    }
    runtime.execute().unwrap();
    runtime.get_f32(t.outputs[0].tensor).unwrap().clone()
}

#[test]
fn zero_tensor_exponents_return_one_for_all_real_bases() {
    let t = translate_one(
        node(
            "torch.ops.aten.pow.Tensor_Tensor",
            vec![tensor_input("self", "a"), tensor_input("exponent", "b")],
            &["y"],
        ),
        &[
            ("a", FLOAT, &[1, 7]),
            ("b", FLOAT, &[2, 1]),
            ("y", FLOAT, &[2, 7]),
        ],
        &["a", "b"],
        &["y"],
    );
    let bases = vec![
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
        -2.0,
    ];
    assert_eq!(
        run(
            &t,
            vec![TypedBuffer::F32(bases), TypedBuffer::F32(vec![0.0, -0.0])]
        ),
        vec![1.0; 14]
    );
}

#[test]
fn scalar_bases_also_respect_zero_tensor_exponents() {
    for base in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -2.0] {
        let t = translate_one(
            node(
                "torch.ops.aten.pow.Scalar",
                vec![scalar_float("self", base), tensor_input("exponent", "b")],
                &["y"],
            ),
            &[("b", FLOAT, &[2]), ("y", FLOAT, &[2])],
            &["b"],
            &["y"],
        );
        assert_eq!(
            run(&t, vec![TypedBuffer::F32(vec![0.0, -0.0])]),
            vec![1.0; 2],
            "base {base}"
        );
    }
}
