use candle_core::{Device, Tensor};
use cudarc::driver::CudaContext;
use luminal::prelude::*;
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use crate::cuda_bandwidth_gbps;
use crate::runtime::CudaRuntime;

fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}

fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
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

fn get_cuda_stream() -> Option<Arc<cudarc::driver::CudaStream>> {
    let ctx = CudaContext::new(0).ok()?;
    ctx.bind_to_thread().ok()?;
    Some(ctx.default_stream())
}

/// Test a unary operation on CUDA against candle reference
pub fn test_unary(
    shape: impl ToShape,
    func: impl Fn(GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor) -> Tensor,
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

    let input_data = random_vec(n_elements);
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

/// Test a binary operation on CUDA against candle reference
pub fn test_binary(
    a_shape: impl ToShape,
    b_shape: impl ToShape,
    func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ref_func: impl Fn(Tensor, Tensor) -> Tensor,
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
    let b_data = random_vec(b_elements);
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_add(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
    }

    #[test]
    fn test_mul(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a * b, |a, b| (&a * &b).unwrap());
    }

    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
    }

    #[test]
    fn test_matmul(m in 1usize..128, n in 1usize..128, k in 1usize..128) {
        // a_shape: (m, k), b_shape: (n, k) - b gets transposed to (k, n) with k-contiguous strides
        test_binary(
            (m, k),
            (n, k),
            |a, b| a.matmul(b.t()),
            |a, b| a.matmul(&b.t().unwrap()).unwrap(),
        );
    }
}

/// Test that measures bandwidth utilization for a large element-wise add kernel.
/// This demonstrates that KernelAdd can achieve reasonable bandwidth with large tensors.
#[test]
pub fn kernel_add_bandwidth_test() {
    // 64M elements = 256MB per tensor, 768MB total memory traffic (2 reads + 1 write)
    let size = 64 * 1024 * 1024;

    let mut cx = Graph::default();
    let a = cx.tensor(size);
    let b = cx.tensor(size);
    let output = (a + b).output();

    // Generate test data
    let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 * 0.001).collect();
    let data_b: Vec<f32> = (0..size)
        .map(|i| ((i + 500) % 1000) as f32 * 0.001)
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(a, data_a.clone());
    rt.set_data(b, data_b.clone());
    rt = cx.search(rt, 5);

    // Warm up
    rt.execute(&cx.dyn_map);

    // Run and measure
    rt.execute(&cx.dyn_map);

    // Print stats
    println!("\n=== Large KernelAdd Bandwidth Test ===");
    println!(
        "Tensor size: {} elements ({} MB per tensor)",
        size,
        size * 4 / 1024 / 1024
    );
    println!(
        "Total memory traffic: {} MB (2 reads + 1 write)",
        size * 4 * 3 / 1024 / 1024
    );
    rt.print_execution_stats();

    // Verify correctness (spot check)
    let result = rt.get_f32(output);
    for i in [0, size / 2, size - 1] {
        let expected = data_a[i] + data_b[i];
        let got = result[i];
        assert!(
            (got - expected).abs() < 1e-5,
            "Mismatch at {}: expected {}, got {}",
            i,
            expected,
            got
        );
    }

    // Check bandwidth is reasonable (at least 50% of peak for large kernels)
    if let Some(peak_bw) = cuda_bandwidth_gbps(&ctx) {
        for stat in &rt.last_kernel_stats {
            let total_bytes = stat.bytes_loaded + stat.bytes_stored;
            if stat.name == "Add" && total_bytes > 0 {
                let utilization = stat.bandwidth_gbps / peak_bw as f64 * 100.0;
                println!(
                    "\nAdd kernel achieved {:.1} GB/s ({:.1}% of {:.0} GB/s peak)",
                    stat.bandwidth_gbps, utilization, peak_bw
                );
                println!(
                    "  Loaded: {} bytes, Stored: {} bytes",
                    stat.bytes_loaded, stat.bytes_stored
                );
                // Large adds should achieve decent bandwidth
                assert!(
                    utilization > 50.0,
                    "Bandwidth utilization too low: {:.1}%",
                    utilization
                );
            }
        }
    }
}

#[test]
pub fn cuda_argsort_test() {
    let rows = 10; // shmem tet
    let cols = 5000; // no shmem test
    let total = rows * cols;

    let mut cx = Graph::default();
    let input = cx.tensor((rows, cols));
    let sorted_dim0 = input.argsort(0, true).output(); // descend
    let sorted_dim1 = input.argsort(1, false).output(); // ascend

    // random and unique data
    let data: Vec<f32> = (0..total).map(|i| ((i * 73 + 17) % total) as f32).collect();

    let sorted_cols: Vec<Vec<i32>> = (0..cols)
        .map(|col| {
            let mut indices: Vec<i32> = (0..rows as i32).collect();
            indices.sort_by(|&a, &b| {
                let va = data[(a as usize) * cols + col];
                let vb = data[(b as usize) * cols + col];
                vb.partial_cmp(&va).unwrap()
            });
            indices
        })
        .collect();

    let expected_dim0: Vec<i32> = (0..rows)
        .flat_map(|row| {
            (0..cols)
                .map(|col| sorted_cols[col][row])
                .collect::<Vec<_>>()
        })
        .collect();

    let expected_dim1: Vec<i32> = (0..rows)
        .flat_map(|row| {
            let mut indices: Vec<i32> = (0..cols as i32).collect();
            indices.sort_by(|&a, &b| {
                let va = data[row * cols + (a as usize)];
                let vb = data[row * cols + (b as usize)];
                va.partial_cmp(&vb).unwrap()
            });
            indices
        })
        .collect();

    let ctx = CudaContext::new(0).unwrap();
    ctx.bind_to_thread().unwrap();
    let stream = ctx.default_stream();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(input, data);
    rt = cx.search(rt, 10);
    rt.execute(&cx.dyn_map);

    let out_dim0 = rt.get_i32(sorted_dim0.id).clone();
    let out_dim1 = rt.get_i32(sorted_dim1.id).clone();

    assert_eq!(out_dim0.len(), expected_dim0.len(), "dim0 length mismatch");
    assert_eq!(out_dim1.len(), expected_dim1.len(), "dim1 length mismatch");

    for i in 0..out_dim0.len() {
        assert_eq!(
            out_dim0[i], expected_dim0[i],
            "dim0 mismatch at {i}: got {}, expected {}",
            out_dim0[i], expected_dim0[i]
        );
    }

    for i in 0..out_dim1.len() {
        assert_eq!(
            out_dim1[i], expected_dim1[i],
            "dim1 mismatch at {i}: got {}, expected {}",
            out_dim1[i], expected_dim1[i]
        );
    }
}
