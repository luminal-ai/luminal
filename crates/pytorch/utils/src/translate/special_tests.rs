use super::test_support::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

#[test]
fn bessel_j1_preserves_oscillations_and_odd_symmetry() {
    let values = vec![
        -20.0, -10.0, -8.0, -6.0, -5.0, -4.0, -1.0, -0.0, 0.0, 1.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0,
    ];
    // CPU PyTorch values span both sides of multiple positive roots.
    let expected = [
        -0.066_833_29_f32,
        -0.043472752,
        -0.23463635,
        0.27668375,
        0.3275791,
        0.06604333,
        -0.44005063,
        -0.0,
        0.0,
        0.44005063,
        -0.06604333,
        -0.3275791,
        -0.27668375,
        0.23463635,
        0.043472752,
        0.066_833_29,
    ];
    let t = translate_one(
        node(
            "torch.ops.aten.special_bessel_j1.default",
            vec![tensor_input("self", "a")],
            &["y"],
        ),
        &[("a", FLOAT, &[16]), ("y", FLOAT, &[16])],
        &["a"],
        &["y"],
    );
    let input = t.inputs[0].tensor;
    let value = TypedBuffer::F32(values);
    let data: FxHashMap<_, _> = [(input, value.clone())].into_iter().collect();
    let mut runtime = ReferenceRuntime::load(&t.graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    runtime.set_data(input, value);
    runtime.execute().unwrap();
    let actual = runtime.get_f32(t.outputs[0].tensor).unwrap();
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-5,
            "index {index}: {actual} != {expected}"
        );
        assert_eq!(
            actual.is_sign_negative(),
            expected.is_sign_negative(),
            "index {index}"
        );
    }
}
