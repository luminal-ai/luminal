use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn run(cx: &Graph, inputs: Vec<(petgraph::prelude::NodeIndex, TypedBuffer)>) -> ReferenceRuntime {
    let mut runtime = ReferenceRuntime::load(cx).unwrap();
    let data: FxHashMap<_, _> = inputs.into_iter().collect();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    for (id, value) in data {
        runtime.set_data(id, value);
    }
    runtime.execute().unwrap();
    runtime
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.to_bits(), b.to_bits(), "{actual:?} != {expected:?}");
    }
}

#[test]
fn padding_preserves_nonfinite_values_and_signed_zero() {
    let mut cx = Graph::new();
    let x = cx.tensor(vec![4], DType::F32);
    let out = x.pad(vec![(1, 2)], -0.0);
    let nan = f32::from_bits(0x7fc00123);
    let runtime = run(
        &cx,
        vec![(
            x.id,
            TypedBuffer::F32(vec![f32::INFINITY, nan, -0.0, f32::NEG_INFINITY]),
        )],
    );
    assert_bits(
        runtime.get_f32(out.id).unwrap(),
        &[
            -0.0,
            f32::INFINITY,
            nan,
            -0.0,
            f32::NEG_INFINITY,
            -0.0,
            -0.0,
        ],
    );
}

#[test]
fn concatenation_and_stack_copy_nonfinite_values_and_signed_zero() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![2, 2], DType::F32);
    let b = cx.tensor(vec![2, 2], DType::F32);
    let concat = a.concat_along(b, 1);
    let stack = cx.stack(&[a, b], 2);
    let nan = f32::from_bits(0x7fc00123);
    let runtime = run(
        &cx,
        vec![
            (a.id, TypedBuffer::F32(vec![f32::INFINITY, -0.0, nan, 3.])),
            (
                b.id,
                TypedBuffer::F32(vec![1., f32::NEG_INFINITY, -0.0, 4.]),
            ),
        ],
    );
    assert_bits(
        runtime.get_f32(concat.id).unwrap(),
        &[
            f32::INFINITY,
            -0.0,
            1.,
            f32::NEG_INFINITY,
            nan,
            3.,
            -0.0,
            4.,
        ],
    );
    assert_bits(
        runtime.get_f32(stack.id).unwrap(),
        &[
            f32::INFINITY,
            1.,
            -0.0,
            f32::NEG_INFINITY,
            nan,
            -0.0,
            3.,
            4.,
        ],
    );
}

#[test]
fn concatenation_and_padding_preserve_full_width_integers() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![2], DType::I64);
    let b = cx.tensor(vec![2], DType::I64);
    let concat = a.concat_along(b, 0);
    let fill = cx.constant_i64((1_i64 << 55) + 3);
    let padded = a.pad_with(vec![(1, 1)], fill);
    let av = vec![i64::MIN, (1_i64 << 55) + 1];
    let bv = vec![i64::MAX, -(1_i64 << 55) - 1];
    let runtime = run(
        &cx,
        vec![
            (a.id, TypedBuffer::I64(av.clone())),
            (b.id, TypedBuffer::I64(bv.clone())),
        ],
    );
    assert_eq!(
        runtime.get_i64(concat.id).unwrap(),
        &[av[0], av[1], bv[0], bv[1]]
    );
    assert_eq!(
        runtime.get_i64(padded.id).unwrap(),
        &[(1_i64 << 55) + 3, av[0], av[1], (1_i64 << 55) + 3]
    );
}

#[test]
fn empty_input_padding_uses_only_the_fill() {
    let mut cx = Graph::new();
    let x = cx.tensor(vec![0], DType::F32);
    let fill = cx.constant_f32(-0.0);
    let out = x.pad_with(vec![(1, 2)], fill);
    let runtime = run(&cx, vec![]);
    assert_bits(runtime.get_f32(out.id).unwrap(), &[-0.0; 3]);
}
