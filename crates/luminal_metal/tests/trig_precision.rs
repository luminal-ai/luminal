#![cfg(target_os = "macos")]

mod support;

use half::f16;
use luminal::{
    dtype::{DType, PlanDtype},
    graph::Graph,
    prelude::GraphTensor,
};
use luminal_metal::{HostBuffer, MetalRuntime, harness_search_options};
use support::TestTransfers;

#[test]
fn cosine_reduces_the_original_angle() {
    check_original_angles(GraphTensor::cos, f32::cos);
}

#[test]
fn sine_reduces_the_original_angle() {
    check_original_angles(GraphTensor::sin, f32::sin);
}

fn check_original_angles(op: fn(GraphTensor) -> GraphTensor, cpu: fn(f32) -> f32) {
    let angles = [
        -30000.0_f32,
        -3500.0,
        -700.0,
        -std::f32::consts::FRAC_PI_2,
        -0.0,
        0.0,
        std::f32::consts::PI,
        // Avoid an F16 rounding midpoint for sine.
        301.0,
        3500.0,
        30000.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let half_angles = angles.map(f16::from_f32);
    let float_angles = [angles.as_slice(), &[-f32::MAX, f32::MAX]].concat();

    let mut graph = Graph::new();
    let x = graph.tensor(vec![float_angles.len()], DType::F32);
    let half_x = graph.tensor(vec![half_angles.len()], DType::F16);
    let y = op(x);
    let half_y = op(half_x);
    let inputs = [
        (x.id, HostBuffer::from(float_angles.clone())),
        (
            half_x.id,
            HostBuffer::new(
                PlanDtype::F16,
                half_angles.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            )
            .unwrap(),
        ),
    ]
    .into_iter()
    .collect();
    let mut runtime = MetalRuntime::load(&graph).unwrap();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &inputs,
            &harness_search_options(),
        )
        .unwrap();
    let mut arena = support::allocate_arena(&runtime).unwrap();
    for (id, data) in inputs {
        runtime.upload(&mut arena, id, data).unwrap();
    }
    runtime.execute(arena.buffer()).unwrap();

    let actual = runtime.read_f32(&arena, y.id).unwrap();
    assert_eq!(actual.len(), float_angles.len());
    for (input, actual) in float_angles.into_iter().zip(actual) {
        if input.is_finite() {
            let expected = cpu(input);
            assert!(
                (actual - expected).abs() <= 1e-5,
                "input {input} expected {expected}, got {actual}"
            );
            if input == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits(), "input {input}");
            }
        } else {
            assert!(actual.is_nan(), "input {input} expected NaN, got {actual}");
        }
    }

    let (actual, _) = runtime.download(&arena, half_y.id).unwrap();
    assert_eq!(actual.dtype, PlanDtype::F16);
    assert_eq!(actual.len(), half_angles.len());
    let (actual, remainder) = actual.bytes.as_chunks::<2>();
    assert!(remainder.is_empty());
    for (input, &actual) in half_angles.into_iter().zip(actual) {
        let actual = f16::from_ne_bytes(actual);
        if input.is_finite() {
            let expected = f16::from_f32(cpu(input.to_f32()));
            assert_eq!(actual.to_bits(), expected.to_bits(), "input {input}");
        } else {
            assert!(actual.is_nan(), "input {input} expected NaN, got {actual}");
        }
    }
}
