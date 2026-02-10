mod misc;
mod mxfp4;
mod nvfp4;
mod ops;

use cudarc::driver::CudaContext;
use luminal::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use crate::runtime::CudaRuntime;
use candle_core::{Device, Tensor};

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}

pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

pub fn get_cuda_stream() -> Option<Arc<cudarc::driver::CudaStream>> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.bind_to_thread().ok()?;
    Some(ctx.default_stream())
}

pub fn identity(v: Vec<f32>) -> Vec<f32> {
    v
}

pub fn make_positive(v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| x.abs() + 0.1).collect()
}

pub fn shift_from_zero(v: Vec<f32>) -> Vec<f32> {
    v.into_iter()
        .map(|x| if x.abs() < 0.1 { 0.5 } else { x })
        .collect()
}

/// Base unary test function with input transform
pub fn test_unary_transform(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
    transform: impl Fn(Vec<f32>) -> Vec<f32>,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let shape: Vec<usize> = shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let n_elements: usize = shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(shape.clone());
    let b = func(a).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = transform(random_vec(n_elements));
    rt.set_data(a, input_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(b);

    // Reference using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(input_data, shape, &device).unwrap();
    let ref_b = ref_func(ref_a).flatten_all().unwrap();

    assert_close(&result, &ref_b.to_vec1::<f32>().unwrap());
}

/// Test a unary operation on CUDA against candle reference
pub fn test_unary(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
) {
    test_unary_transform(shape, func, ref_func, identity);
}

/// Test a unary operation with positive input data (for sqrt, log)
pub fn test_unary_positive(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
) {
    test_unary_transform(shape, func, ref_func, make_positive);
}

/// Test a unary operation with non-zero input data (for recip)
pub fn test_unary_nonzero(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
) {
    test_unary_transform(shape, func, ref_func, shift_from_zero);
}

/// Base binary test function with input transforms
pub fn test_binary_transforms(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
    a_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    b_transform: impl Fn(Vec<f32>) -> Vec<f32>,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let a_shape: Vec<usize> = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let a_elements: usize = a_shape.iter().product();
    let b_elements: usize = b_shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let a_data = a_transform(random_vec(a_elements));
    let b_data = b_transform(random_vec(b_elements));
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);

    // Reference using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(a_data, a_shape, &device).unwrap();
    let ref_b = Tensor::from_vec(b_data, b_shape, &device).unwrap();
    let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();

    assert_close(&result, &ref_c.to_vec1::<f32>().unwrap());
}

/// Test a binary operation on CUDA against candle reference
pub fn test_binary(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
) {
    test_binary_transforms(a_shape, b_shape, func, ref_func, identity, identity);
}

/// Test mod operation with element-wise reference using Rust's % operator
pub fn test_mod(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let a_shape: Vec<usize> = a_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b_shape
        .to_shape()
        .into_iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let a_elements: usize = a_shape.iter().product();
    let b_elements: usize = b_shape.iter().product();

    let mut cx = Graph::default();
    let a = cx.tensor(a_shape.clone());
    let b = cx.tensor(b_shape.clone());
    let c = func(a, b).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let a_data = random_vec(a_elements);
    let b_data = shift_from_zero(random_vec(b_elements));
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(c);

    // Reference: Rust's % operator matches CUDA's fmodf (IEEE 754 remainder)
    let expected: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| x % y)
        .collect();

    assert_close(&result, &expected);
}
