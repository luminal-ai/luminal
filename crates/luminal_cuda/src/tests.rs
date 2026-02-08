use candle_core::{Device, Tensor};
use cudarc::driver::CudaContext;
use luminal::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, random_initial_choice, validate_choice_set,
};
use luminal::prelude::*;
use proptest::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::sync::Arc;

use crate::cuda_bandwidth_gbps;
use crate::runtime::CudaRuntime;
use tracing::{Level, enabled};

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

    // Unary ops tests
    #[test]
    fn test_exp2(x in 1usize..100, y in 1usize..5) {
        // exp2(x) = 2^x, verified by computing 2^x using exp(x * ln(2))
        test_unary(x, |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
        test_unary((y, x), |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
    }

    #[test]
    fn test_log2(x in 1usize..100, y in 1usize..5) {
        // log2(x) = ln(x) / ln(2)
        test_unary_positive(x, |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
        test_unary_positive((y, x), |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
    }

    #[test]
    fn test_sin(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a.sin(), |a| a.sin().unwrap());
        test_unary((y, x), |a| a.sin(), |a| a.sin().unwrap());
    }

    #[test]
    fn test_recip(x in 1usize..100, y in 1usize..5) {
        test_unary_nonzero(x, |a| a.reciprocal(), |a| a.recip().unwrap());
        test_unary_nonzero((y, x), |a| a.reciprocal(), |a| a.recip().unwrap());
    }

    #[test]
    fn test_sqrt(x in 1usize..100, y in 1usize..5) {
        test_unary_positive(x, |a| a.sqrt(), |a| a.sqrt().unwrap());
        test_unary_positive((y, x), |a| a.sqrt(), |a| a.sqrt().unwrap());
    }

    // Binary ops tests
    #[test]
    fn test_mod_op(size in 1usize..100, rows in 1usize..5) {
        test_mod(size, size, |a, b| a % b);
        test_mod((rows, size), (rows, size), |a, b| a % b);
    }

    #[test]
    fn test_less_than(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        test_binary((y, x), (y, x), |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
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
    if enabled!(Level::INFO) {
        rt.print_execution_stats();
    }

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
    let rows = 5; // shmem tet
    let cols = 500; // no shmem test
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

/// Fuzz test that generates many random genomes and verifies they all produce correct results.
/// This tests the genetic algorithm search by validating each genome individually.
#[test]
fn fuzz_test_cuda_genomes() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping test");
        return;
    };

    // Build a graph with operations that have rewrite alternatives
    let mut cx = Graph::default();
    let a = cx.tensor((4, 8));
    let b = cx.tensor((8, 4));
    let c = cx.tensor((4, 4));

    // Matmul + add + relu creates opportunities for rewrites
    let d = a.matmul(b);
    let e = (d + c).relu();
    let out = e.output();

    cx.build_search_space::<CudaRuntime>();
    let egraph = cx.egraph().unwrap();
    let ops = cx.egglog_ops().unwrap();

    // Count mutable eclasses
    let mutable_eclasses: usize = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .count();
    println!(
        "CUDA search space: {} total eclasses, {} mutable",
        egraph.eclasses.len(),
        mutable_eclasses
    );

    // Generate test data
    let a_data = random_vec(32);
    let b_data = random_vec(32);
    let c_data = random_vec(16);

    // Compute reference result using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(a_data.clone(), (4, 8), &device).unwrap();
    let ref_b = Tensor::from_vec(b_data.clone(), (8, 4), &device).unwrap();
    let ref_c = Tensor::from_vec(c_data.clone(), (4, 4), &device).unwrap();
    let ref_d = ref_a.matmul(&ref_b).unwrap();
    let ref_e = (&ref_d + &ref_c).unwrap().relu().unwrap();
    let expected: Vec<f32> = ref_e.flatten_all().unwrap().to_vec1().unwrap();

    let mut rng = rand::rng();
    let mut prev_selected: FxHashSet<u64> = FxHashSet::default();

    // Test initial genome
    let initial = random_initial_choice(egraph, &mut rng);
    prev_selected.insert(hash_choice_set(&initial));

    if let Err(e) = validate_choice_set(egraph, &initial, ops) {
        panic!("Initial genome invalid: {}", e);
    }

    // Extract and execute initial genome
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();
    let llir_graph = egglog_to_llir(
        egraph,
        initial.clone(),
        ops,
        &cx.custom_ops,
        &mut list_cache,
        &mut expr_cache,
        None,
    );

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.load_llir(&llir_graph);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt.set_data(c, c_data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);
    assert_close(&result, &expected);
    println!("Initial genome: correct");

    // If no mutable eclasses, only one valid graph exists
    if mutable_eclasses == 0 {
        println!("No mutable eclasses, only one valid graph - test passed");
        return;
    }

    // Generate and test many genomes
    let mut base = initial;
    let mut tested = 0;
    let target = 50;

    for _generation in 0..100 {
        let offspring = extract_generation(egraph, &base, 10, 2, &mut prev_selected, &mut rng);

        if offspring.is_empty() {
            println!("Search space exhausted");
            break;
        }

        for genome in offspring {
            // Validate
            if let Err(e) = validate_choice_set(egraph, &genome, ops) {
                panic!("Invalid genome: {}", e);
            }

            // Extract and execute
            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();
            let llir_graph = egglog_to_llir(
                egraph,
                genome.clone(),
                ops,
                &cx.custom_ops,
                &mut list_cache,
                &mut expr_cache,
                None,
            );

            // Create fresh runtime for this genome
            let mut rt = CudaRuntime::initialize(stream.clone());
            rt.load_llir(&llir_graph);
            rt.set_data(a, a_data.clone());
            rt.set_data(b, b_data.clone());
            rt.set_data(c, c_data.clone());
            rt.execute(&cx.dyn_map);
            let result = rt.get_f32(out);

            // Verify correctness
            assert_close(&result, &expected);

            tested += 1;
            base = genome;

            if tested >= target {
                break;
            }
        }

        if tested >= target {
            break;
        }
    }

    println!(
        "Fuzz test: verified {} genomes produce correct results",
        tested
    );
    assert!(tested > 0, "No genomes were tested");
}
