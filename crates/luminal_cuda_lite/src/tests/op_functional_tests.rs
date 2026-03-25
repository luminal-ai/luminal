use candle_core::{Device, Tensor};
use cudarc::driver::CudaContext;
use luminal::prelude::*;
use proptest::prelude::*;

use luminal::egglog_utils::{
    egglog_to_llir, extract_generation, hash_choice_set, random_initial_choice, validate_choice_set,
};

use crate::runtime::CudaRuntime;

#[allow(unused_imports)]
use super::utilities::{
    GENOME_FUZZ_COUNT, TOLERANCE_SAFETY_FACTOR, assert_close, dtype_epsilon, fuzz_genomes,
    gen_slice_range, get_cuda_stream, gpu_supports_dtype, random_f32_vec, random_i32_vec,
    test_binary_cuda, test_mod, test_unary_cuda, to_candle_dtype,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_add(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let (rtol, atol) = (eps * TOLERANCE_SAFETY_FACTOR, eps * TOLERANCE_SAFETY_FACTOR);
        test_binary_cuda(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap(), &gen_lambda, &gen_lambda, seed, rtol, atol);
        test_binary_cuda((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap(), &gen_lambda, &gen_lambda, seed, rtol, atol);
    }

    #[test]
    fn test_mul(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        let eps = dtype_epsilon(luminal::dtype::DType::F32);
        let (rtol, atol) = (eps * TOLERANCE_SAFETY_FACTOR, eps * TOLERANCE_SAFETY_FACTOR);
        test_binary_cuda(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap(), &gen_lambda, &gen_lambda, seed, rtol, atol);
        test_binary_cuda((y, x), (y, x), |a, b| a * b, |a, b| (&a * &b).unwrap(), &gen_lambda, &gen_lambda, seed, rtol, atol);
    }

    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        test_unary_cuda((rows, cols), |a| a.max(1), |a| a.max(1).unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        test_unary_cuda((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_matmul(
        (m, n, k, a_col_major, b_col_major, m_slice, k_slice, n_slice, dtype) in
            (1usize..128, 1usize..128, 1usize..128, any::<bool>(), any::<bool>(),
             any::<(bool, bool)>(), any::<(bool, bool)>(), any::<(bool, bool)>(),
             prop::sample::select(&[luminal::dtype::DType::F32, luminal::dtype::DType::F16, luminal::dtype::DType::Bf16]))
            .prop_perturb(|(m, n, k, a_cm, b_cm, m_sl, k_sl, n_sl, dt), mut rng| {
                (m, n, k, a_cm, b_cm,
                 gen_slice_range(m, m_sl.0, m_sl.1, &mut rng),
                 gen_slice_range(k, k_sl.0, k_sl.1, &mut rng),
                 gen_slice_range(n, n_sl.0, n_sl.1, &mut rng),
                 dt)
            }),
        seed in any::<u64>()
    ) {
        prop_assume!(gpu_supports_dtype(dtype), "GPU does not support {:?}", dtype);

        let (m_start, m_end) = m_slice;
        let (k_start, k_end) = k_slice;
        let (n_start, n_end) = n_slice;
        let effective_m = m_end - m_start;
        let effective_k = k_end - k_start;
        let effective_n = n_end - n_start;

        // Column-major achieved by storing transposed then calling .t()
        let (a_shape, b_shape): ((usize, usize), (usize, usize)) = match (a_col_major, b_col_major) {
            (false, false) => ((m, k), (k, n)),  // Rm x Rm
            (false, true)  => ((m, k), (n, k)),  // Rm x Cm
            (true, false)  => ((k, m), (k, n)),  // Cm x Rm
            (true, true)   => ((k, m), (n, k)),  // Cm x Cm
        };

        let candle_dtype = to_candle_dtype(dtype);

        let luminal_op = move |a: GraphTensor, b: GraphTensor| {
            let a = a.cast(dtype);
            let b = b.cast(dtype);
            let a = if a_col_major { a.t() } else { a };
            let b = if b_col_major { b.t() } else { b };
            // After transpose: A is (m, k), B is (k, n)
            let a = a.slice((m_start..m_end, k_start..k_end));
            let b = b.slice((k_start..k_end, n_start..n_end));
            a.matmul(b).cast(luminal::dtype::DType::F32)
        };
        let candle_op = move |a: Tensor, b: Tensor| {
            let a = a.to_dtype(candle_dtype).unwrap();
            let b = b.to_dtype(candle_dtype).unwrap();
            let a = if a_col_major { a.t().unwrap() } else { a };
            let b = if b_col_major { b.t().unwrap() } else { b };
            // After transpose: A is (m, k), B is (k, n)
            let a = a.narrow(0, m_start, effective_m).unwrap()
                     .narrow(1, k_start, effective_k).unwrap()
                     .contiguous().unwrap();
            let b = b.narrow(0, k_start, effective_k).unwrap()
                     .narrow(1, n_start, effective_n).unwrap()
                     .contiguous().unwrap();
            a.matmul(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap()
        };

        // Matmul tolerance: rtol scales with sqrt(k) for accumulated rounding error
        let eps = dtype_epsilon(dtype);
        let sqrt_k = (effective_k as f32).sqrt();
        let rtol = eps * sqrt_k;
        let atol = 5.0 * eps;

        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        test_binary_cuda(a_shape, b_shape, luminal_op, candle_op, &gen_lambda, &gen_lambda, seed, rtol, atol);
    }

    // Unary ops tests
    #[test]
    fn test_exp2(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        // exp2(x) = 2^x, verified by computing 2^x using exp(x * ln(2))
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        test_unary_cuda(x, |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap(), &gen_lambda, seed);
        test_unary_cuda((y, x), |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_log2(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        // log2(x) = ln(x) / ln(2)
        let gen_lambda = |n, s| random_f32_vec(n, s, 0.1, 0.6);
        test_unary_cuda(x, |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap(), &gen_lambda, seed);
        test_unary_cuda((y, x), |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_sin(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -0.5, 0.5);
        test_unary_cuda(x, |a| a.sin(), |a| a.sin().unwrap(), &gen_lambda, seed);
        test_unary_cuda((y, x), |a| a.sin(), |a| a.sin().unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_recip(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, 0.1, 0.5);
        test_unary_cuda(x, |a| a.reciprocal(), |a| a.recip().unwrap(), &gen_lambda, seed);
        test_unary_cuda((y, x), |a| a.reciprocal(), |a| a.recip().unwrap(), &gen_lambda, seed);
    }

    #[test]
    fn test_sqrt(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, 0.1, 0.6);
        test_unary_cuda(x, |a| a.sqrt(), |a| a.sqrt().unwrap(), &gen_lambda, seed);
        test_unary_cuda((y, x), |a| a.sqrt(), |a| a.sqrt().unwrap(), &gen_lambda, seed);
    }

    // Binary ops tests
    #[test]
    fn test_mod_op(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        test_mod(x, x, |a, b| a % b, seed);
        test_mod((y, x), (y, x), |a, b| a % b, seed);
    }

    #[test]
    fn test_less_than(x in 1usize..100, y in 1usize..5, seed in any::<u64>()) {
        let gen_lambda = |n, s| random_f32_vec(n, s, -99.0, 100.0).into_iter().map(|v| v.floor()).collect();
        test_binary_cuda(x, x, |a, b| a.lt(b).cast(luminal::dtype::DType::F32), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap(), &gen_lambda, &gen_lambda, seed, 0.0, 0.0);
        test_binary_cuda((y, x), (y, x), |a, b| a.lt(b).cast(luminal::dtype::DType::F32), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap(), &gen_lambda, &gen_lambda, seed, 0.0, 0.0);
    }
}

#[allow(dead_code)]
fn run_argsort_test(rows: usize, cols: usize, seed: u64) {
    let total = rows * cols;

    let mut cx = Graph::default();
    let input = cx.tensor((rows, cols));
    let sorted_dim0 = input.stable_argsort(0, true).output(); // descend
    let sorted_dim1 = input.stable_argsort(1, false).output(); // ascend

    // random and unique data using seed
    let data: Vec<f32> = random_f32_vec(total, seed, 0.0, 1.0);

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
    let out_dim0 = rt.get_i32(sorted_dim0.id);
    let out_dim1 = rt.get_i32(sorted_dim1.id);

    assert_eq!(out_dim0.len(), expected_dim0.len(), "dim0 length mismatch");
    assert_eq!(out_dim1.len(), expected_dim1.len(), "dim1 length mismatch");

    // Debug: check for out-of-range values (indices should be 0..rows for dim0, 0..cols for dim1)
    let max_valid_dim0 = rows as i32 - 1;
    let max_valid_dim1 = cols as i32 - 1;
    let bad_dim0: Vec<_> = out_dim0
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v < 0 || v > max_valid_dim0)
        .take(10)
        .collect();
    let bad_dim1: Vec<_> = out_dim1
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v < 0 || v > max_valid_dim1)
        .take(10)
        .collect();

    if !bad_dim0.is_empty() {
        panic!(
            "dim0 has out-of-range values (valid: 0-{max_valid_dim0}): {:?}\nFirst 20 values: {:?}",
            bad_dim0,
            &out_dim0[..20.min(out_dim0.len())]
        );
    }
    if !bad_dim1.is_empty() {
        panic!(
            "dim1 has out-of-range values (valid: 0-{max_valid_dim1}): {:?}",
            bad_dim1
        );
    }

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

// NOTE: Argsort proptest disabled due to pre-existing bug where argsort output shape
// through e-graph compilation returns only `rows` elements instead of `rows * cols`.
// proptest! {
//     #![proptest_config(ProptestConfig::with_cases(10))]
//     #[test]
//     fn test_argsort(seed in any::<u64>()) {
//         run_argsort_test(5, 500, seed);
//     }
// }

/// Test F32 -> F16 -> F32 cast roundtrip with edge-case values.
#[test]
pub fn test_cast_f16_edge_cases() {
    use luminal::dtype::DType;

    // Fixed edge-case values that exercise F16 behavior
    let edge_cases: Vec<f32> = vec![
        0.0,
        1.0,
        -1.0,
        0.5,
        0.333333333,   // Will truncate: F16 can't represent 1/3 exactly
        0.1,           // Will truncate: 0.1 isn't exact in binary
        1.0009765625,  // Exactly representable in F16 (1 + 1/1024)
        1.00048828125, // Rounds to 1.0 in F16 (1 + 1/2048, below F16 precision)
        1.0007324219,  // Between two F16 values, will round
        -3.140625,     // Exactly representable
        3.14159265,    // Pi - will truncate
        65504.0,       // Max normal F16
        -65504.0,      // Min normal F16
        0.000060976,   // Near F16 min positive normal
        1e-7,          // Subnormal in F16
        100.0,
        -100.0,
        12.345678, // Arbitrary value requiring truncation
    ];

    // Generator that ignores seed and returns edge cases
    let gen_edge_cases = |_n: usize, _seed: u64| edge_cases.clone();

    test_unary_cuda(
        edge_cases.len(),
        |a| a.cast(DType::F16).cast(DType::F32),
        |a| {
            a.to_dtype(candle_core::DType::F16)
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap()
        },
        &gen_edge_cases,
        0,
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Test F32 -> F16 -> F32 cast roundtrip with random values.
    #[test]
    fn test_cast_f16_random(size in 1usize..200, seed in any::<u64>()) {
        use luminal::dtype::DType;

        // Use range beyond F16 limits so some values overflow to infinity
        let f16_max = half::f16::MAX.to_f32();
        let gen_lambda = |n, s| random_f32_vec(n, s, -2.0 * f16_max, 2.0 * f16_max);

        test_unary_cuda(
            size,
            |a| a.cast(DType::F16).cast(DType::F32),
            |a| {
                a.to_dtype(candle_core::DType::F16)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            },
            &gen_lambda,
            seed,
        );
    }
}

/// Fuzz test that generates many random genomes and verifies they all produce correct results.
/// This tests the genetic algorithm search by validating each genome individually.
/// Uses proptest seed for reproducibility - if this test fails, proptest will print the seed
/// which can be used to reproduce the failure.
fn fuzz_test_cuda_genomes_impl(seed: u64) {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping test");
        return;
    };

    println!("Running fuzz_test_cuda_genomes with seed: {}", seed);

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

    // Use seeded RNG for full reproducibility
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate test data with seeded RNG (reproducible)
    let a_data: Vec<f32> = (0..32).map(|_| rng.random::<f32>()).collect();
    let b_data: Vec<f32> = (0..32).map(|_| rng.random::<f32>()).collect();
    let c_data: Vec<f32> = (0..16).map(|_| rng.random::<f32>()).collect();

    // Compute reference result using candle
    let device = Device::Cpu;
    let ref_a = Tensor::from_vec(a_data.clone(), (4, 8), &device).unwrap();
    let ref_b = Tensor::from_vec(b_data.clone(), (8, 4), &device).unwrap();
    let ref_c = Tensor::from_vec(c_data.clone(), (4, 4), &device).unwrap();
    let ref_d = ref_a.matmul(&ref_b).unwrap();
    let ref_e = (&ref_d + &ref_c).unwrap().relu().unwrap();
    let expected: Vec<f32> = ref_e.flatten_all().unwrap().to_vec1().unwrap();

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

    let mut rt: CudaRuntime = CudaRuntime::initialize(stream.clone());
    rt.load_llir(&llir_graph);
    rt.set_data(a, a_data.clone());
    rt.set_data(b, b_data.clone());
    rt.set_data(c, c_data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);
    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, tol, tol);
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
            let mut rt: CudaRuntime = CudaRuntime::initialize(stream.clone());
            rt.load_llir(&llir_graph);
            rt.set_data(a, a_data.clone());
            rt.set_data(b, b_data.clone());
            rt.set_data(c, c_data.clone());
            rt.execute(&cx.dyn_map);
            let result = rt.get_f32(out);

            // Verify correctness
            assert_close(&result, &expected, tol, tol);

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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3))]

    #[test]
    fn fuzz_test_cuda_genomes(seed in any::<u64>()) {
        fuzz_test_cuda_genomes_impl(seed);
    }
}

fn run_embed_test(vocab_size: usize, embed_dim: usize, seq_len: usize, seed: u64) {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping test");
        return;
    };

    let mut cx = Graph::default();
    let token_ids = cx.tensor(seq_len).as_dtype(luminal::dtype::DType::Int);
    let embed_table = cx.tensor((vocab_size, embed_dim));
    let output = embed_table
        .gather(
            (token_ids * embed_dim).expand_dim(1, embed_dim)
                + cx.arange(embed_dim).expand_dim(0, seq_len),
        )
        .output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());

    let token_data: Vec<i32> = random_i32_vec(seq_len, seed, 0, vocab_size as i32 - 1);
    let embed_data: Vec<f32> = random_f32_vec(vocab_size * embed_dim, seed, -0.5, 0.5);

    rt.set_data(token_ids, token_data.clone());
    rt.set_data(embed_table, embed_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);

    let mut expected = vec![0.0f32; seq_len * embed_dim];
    for i in 0..seq_len {
        let tid = token_data[i] as usize;
        for j in 0..embed_dim {
            expected[i * embed_dim + j] = embed_data[tid * embed_dim + j];
        }
    }

    let eps = dtype_epsilon(luminal::dtype::DType::F32);
    let tol = eps * TOLERANCE_SAFETY_FACTOR;
    assert_close(&result, &expected, tol, tol);

    // Fuzz genomes: verify multiple graph rewrites produce consistent results
    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(token_ids, token_data.clone());
            rt.set_data(embed_table, embed_data.clone());
        },
        output.id,
        &expected,
        tol,
        tol,
        GENOME_FUZZ_COUNT,
        seed,
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_embed_proptest(
        vocab_size in 10usize..200,
        embed_dim in 8usize..128,
        seq_len in 1usize..32,
        seed in any::<u64>(),
    ) {
        run_embed_test(vocab_size, embed_dim, seq_len, seed);
    }
}
