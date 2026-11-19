use std::f32::consts::{FRAC_PI_2, PI};

use luminal::prelude::*;
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer, harness::run_reference};

#[test]
fn sine_and_cosine_preserve_wide_f32_angles_through_artifact_round_trip() {
    let values = [
        -30000.0, -3500.0, -700.0, -FRAC_PI_2, 0.0, PI, 300.0, 3500.0, 30000.0,
    ];
    let mut graph = Graph::new();
    let x = graph.tensor([values.len()], DType::F32);
    let sine = x.sin();
    let cosine = x.cos();
    let runtime = run_reference(&graph, &[(x.id, TypedBuffer::F32(values.to_vec()))]);
    assert_eq!(runtime.get_f32(sine.id).unwrap(), &values.map(f32::sin));
    assert_eq!(runtime.get_f32(cosine.id).unwrap(), &values.map(f32::cos));

    let bindings = ReferenceBindings::leaves(&graph.logical);
    let (output_ids, output_buffers): (Vec<_>, Vec<_>) = bindings
        .outputs()
        .iter()
        .map(|bound| (bound.value, bound.buffer))
        .unzip();
    let artifact = runtime
        .serialize_compiled(&[x.id], &output_buffers)
        .unwrap();
    let mut restored = ReferenceRuntime::default();
    restored
        .deserialize_compiled(&artifact, &[x.id], &output_ids, 1 << 20)
        .unwrap();
    let changed = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0];
    restored.set_data(x.id, TypedBuffer::F32(changed.to_vec()));
    restored.execute().unwrap();
    assert_eq!(restored.get_f32(sine.id).unwrap(), &changed.map(f32::sin));
    assert_eq!(restored.get_f32(cosine.id).unwrap(), &changed.map(f32::cos));
}

#[test]
fn sine_and_cosine_preserve_large_f64_angles_and_exceptional_values() {
    let values = [
        -1e300,
        -3500.0,
        -0.0,
        0.0,
        3500.0,
        1e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut graph = Graph::new();
    let x = graph.tensor([values.len()], DType::F64);
    let outputs = [(x.sin(), f64::sin as fn(f64) -> f64), (x.cos(), f64::cos)];
    let runtime = run_reference(&graph, &[(x.id, TypedBuffer::F64(values.to_vec()))]);
    for (output, expected) in outputs {
        let actual = runtime.get_f64(output.id).unwrap();
        assert_eq!(actual.len(), values.len());
        for (&actual, input) in actual.iter().zip(values) {
            if input.is_finite() {
                assert_eq!(actual.to_bits(), expected(input).to_bits());
            } else {
                assert!(actual.is_nan());
            }
        }
    }
}

#[test]
fn sine_and_cosine_evaluate_stored_narrow_angles_and_round_the_result() {
    let values = [FRAC_PI_2, 10.0, -300.0, 3500.0, 30000.0, f32::INFINITY];
    let f16_values = values.map(f16::from_f32);
    let bf16_values = values.map(bf16::from_f32);
    let mut graph = Graph::new();
    let f16_input = graph.tensor([values.len()], DType::F16);
    let bf16_input = graph.tensor([values.len()], DType::Bf16);
    let outputs = [
        (
            f16_input.sin(),
            bf16_input.sin(),
            f32::sin as fn(f32) -> f32,
        ),
        (f16_input.cos(), bf16_input.cos(), f32::cos),
    ];
    let runtime = run_reference(
        &graph,
        &[
            (f16_input.id, TypedBuffer::F16(f16_values.to_vec())),
            (bf16_input.id, TypedBuffer::Bf16(bf16_values.to_vec())),
        ],
    );

    let finite = values.len() - 1;
    for (f16_output, bf16_output, expected) in outputs {
        let f16_actual = runtime.get_f16(f16_output.id).unwrap();
        let bf16_actual = runtime.get_bf16(bf16_output.id).unwrap();
        assert_eq!(f16_actual.len(), values.len());
        assert_eq!(bf16_actual.len(), values.len());
        assert_eq!(
            f16_actual[..finite],
            f16_values.map(|x| f16::from_f32(expected(x.to_f32())))[..finite]
        );
        assert_eq!(
            bf16_actual[..finite],
            bf16_values.map(|x| bf16::from_f32(expected(x.to_f32())))[..finite]
        );
        assert!(f16_actual[finite].is_nan());
        assert!(bf16_actual[finite].is_nan());
    }
}
