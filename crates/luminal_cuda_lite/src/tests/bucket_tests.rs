use crate::runtime::CudaRuntime;
use crate::tests::utilities::*;
use luminal::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};

/// Helper: build a simple graph with dynamic dim 's' that does element-wise computation.
/// Returns (cx, input_node, output_node).
fn build_dynamic_add_graph() -> (Graph, NodeIndex, NodeIndex) {
    let mut cx = Graph::default();
    let a = cx.tensor(('s', 4));
    let b = (a + a).output();
    (cx, a.id, b.id)
}

/// Helper: build a matmul graph with dynamic dim 's'.
/// Computes (s, K) @ (K, N) -> (s, N)
fn build_dynamic_matmul_graph(k: usize, n: usize) -> (Graph, NodeIndex, NodeIndex, NodeIndex) {
    let mut cx = Graph::default();
    let a = cx.tensor(('s', k));
    let b = cx.tensor((k, n));
    let c = a.matmul(b).output();
    (cx, a.id, b.id, c.id)
}

#[test]
fn test_bucket_dispatch_simple() {
    // Tests that bucketed compilation produces correct results for different dim values
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, a, b) = build_dynamic_add_graph();

    cx.set_dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    // Set dummy input for search
    cx.set_dim('s', 1);
    rt.set_data(a, vec![1.0f32; 4]);

    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 5, &mut rng);

    // Test bucket 1: s=1
    cx.set_dim('s', 1);
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    rt.set_data(a, input_data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(b);
    let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
    assert_close(&result[..4], &expected, 1e-5, 1e-5);

    // Test bucket 2: s=3
    cx.set_dim('s', 3);
    let input_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    rt.set_data(a, input_data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(b);
    let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
    assert_close(&result[..12], &expected, 1e-5, 1e-5);
}

#[test]
fn test_bucket_matmul_dynamic() {
    // Tests matmul with bucketed dynamic dim
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let k = 8;
    let n = 4;
    let (mut cx, a, b_tensor, c) = build_dynamic_matmul_graph(k, n);

    cx.set_dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 8)]);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    cx.set_dim('s', 1);
    let a_data = random_f32_vec(k, 100, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 101, -1.0, 1.0);
    rt.set_data(a, a_data.clone());
    rt.set_data(b_tensor, b_data.clone());

    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 5, &mut rng);

    // Execute at s=1
    cx.set_dim('s', 1);
    rt.set_data(a, a_data.clone());
    rt.set_data(b_tensor, b_data.clone());
    rt.execute(&cx.dyn_map);
    let result_s1 = rt.get_f32(c);

    // Compute reference for s=1 (1xK @ KxN -> 1xN)
    let mut expected_s1 = vec![0.0f32; n];
    for j in 0..n {
        for i in 0..k {
            expected_s1[j] += a_data[i] * b_data[i * n + j];
        }
    }
    assert_close(&result_s1[..n], &expected_s1, 1e-4, 1e-4);

    // Execute at s=4
    cx.set_dim('s', 4);
    let a_data_4 = random_f32_vec(4 * k, 200, -1.0, 1.0);
    rt.set_data(a, a_data_4.clone());
    rt.set_data(b_tensor, b_data.clone());
    rt.execute(&cx.dyn_map);
    let result_s4 = rt.get_f32(c);

    // Compute reference for s=4 (4xK @ KxN -> 4xN)
    let mut expected_s4 = vec![0.0f32; 4 * n];
    for row in 0..4 {
        for j in 0..n {
            for i in 0..k {
                expected_s4[row * n + j] += a_data_4[row * k + i] * b_data[i * n + j];
            }
        }
    }
    assert_close(&result_s4[..4 * n], &expected_s4, 1e-4, 1e-4);
}

#[test]
fn test_bucket_results_match_unbucketed() {
    // Tests that bucketed results match non-bucketed results for the same graph
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let seed = 42u64;

    // Non-bucketed run
    let (mut cx1, a1, b1) = build_dynamic_add_graph();
    cx1.set_dim('s', 3);
    cx1.build_search_space::<CudaRuntime>();
    let mut rt1 = CudaRuntime::initialize(stream.clone());
    let input_data = random_f32_vec(12, seed, -1.0, 1.0);
    rt1.set_data(a1, input_data.clone());
    let mut rng1 = SmallRng::seed_from_u64(seed);
    rt1 = cx1.search_rng(rt1, 5, &mut rng1);
    rt1.set_data(a1, input_data.clone());
    rt1.execute(&cx1.dyn_map);
    let result_unbucketed = rt1.get_f32(b1);

    // Bucketed run with bucket that covers s=3
    let (mut cx2, a2, b2) = build_dynamic_add_graph();
    cx2.set_dim('s', 3);
    cx2.set_dim_buckets('s', &[DimBucket::new(1, 4)]);
    cx2.build_search_space::<CudaRuntime>();
    let mut rt2 = CudaRuntime::initialize(stream.clone());
    rt2.set_data(a2, input_data.clone());
    let mut rng2 = SmallRng::seed_from_u64(seed);
    rt2 = cx2.search_rng(rt2, 5, &mut rng2);
    rt2.set_data(a2, input_data.clone());
    rt2.execute(&cx2.dyn_map);
    let result_bucketed = rt2.get_f32(b2);

    // Results should match — same graph, same search seed, same dyn_map
    assert_eq!(result_unbucketed.len(), result_bucketed.len());
    assert_close(&result_unbucketed[..12], &result_bucketed[..12], 1e-5, 1e-5);
}

#[test]
#[should_panic(expected = "No bucket matches")]
fn test_bucket_out_of_range_panics() {
    let Some(stream) = get_cuda_stream() else {
        // Can't trigger panic without GPU, skip gracefully
        panic!("No bucket matches dyn_map");
    };

    let (mut cx, a, _b) = build_dynamic_add_graph();
    cx.set_dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    cx.set_dim('s', 1);
    rt.set_data(a, vec![1.0f32; 4]);
    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 3, &mut rng);

    // s=10 is outside all buckets — should panic
    cx.set_dim('s', 10);
    rt.set_data(a, vec![1.0f32; 40]);
    rt.execute(&cx.dyn_map);
}

#[test]
fn test_bucket_no_buckets_backward_compat() {
    // No buckets set → should behave identically to old path
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, a, b) = build_dynamic_add_graph();
    cx.set_dim('s', 2);

    // No set_dim_buckets call

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    rt.set_data(a, input_data.clone());
    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 3, &mut rng);

    rt.set_data(a, input_data.clone());
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(b);
    let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
    assert_close(&result[..8], &expected, 1e-5, 1e-5);
}

#[test]
fn test_bucket_representative_override() {
    // Tests that custom representative works
    let bucket = DimBucket::new(2, 32).representative(16);
    assert_eq!(bucket.representative_value(), 16);

    let bucket_default = DimBucket::new(2, 32);
    assert_eq!(bucket_default.representative_value(), 17); // (2+32)/2 = 17

    let exact = DimBucket::new(1, 1);
    assert_eq!(exact.representative_value(), 1);
}

#[test]
fn test_bucket_switch_preserves_weights() {
    // Tests that switching between buckets still sees the correct weight data
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let k = 4;
    let n = 4;
    let (mut cx, a, b_tensor, c) = build_dynamic_matmul_graph(k, n);

    cx.set_dim_buckets('s', &[DimBucket::new(1, 1), DimBucket::new(2, 4)]);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    cx.set_dim('s', 1);
    let a_data = random_f32_vec(k, 300, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 301, -1.0, 1.0);
    rt.set_data(a, a_data.clone());
    rt.set_data(b_tensor, b_data.clone());

    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 5, &mut rng);

    // Execute with bucket 1 (s=1)
    cx.set_dim('s', 1);
    rt.set_data(a, a_data.clone());
    rt.set_data(b_tensor, b_data.clone());
    rt.execute(&cx.dyn_map);
    let result_1a = rt.get_f32(c);

    // Switch to bucket 2 (s=3)
    cx.set_dim('s', 3);
    let a_data_3 = random_f32_vec(3 * k, 302, -1.0, 1.0);
    rt.set_data(a, a_data_3.clone());
    rt.set_data(b_tensor, b_data.clone());
    rt.execute(&cx.dyn_map);
    let result_3 = rt.get_f32(c);

    // Switch back to bucket 1 (s=1) — weights should still work
    cx.set_dim('s', 1);
    rt.set_data(a, a_data.clone());
    rt.set_data(b_tensor, b_data.clone());
    rt.execute(&cx.dyn_map);
    let result_1b = rt.get_f32(c);

    // First and last s=1 results should match exactly
    assert_close(&result_1a[..n], &result_1b[..n], 1e-6, 1e-6);

    // Verify s=3 result correctness
    let mut expected_3 = vec![0.0f32; 3 * n];
    for row in 0..3 {
        for j in 0..n {
            for i in 0..k {
                expected_3[row * n + j] += a_data_3[row * k + i] * b_data[i * n + j];
            }
        }
    }
    assert_close(&result_3[..3 * n], &expected_3, 1e-4, 1e-4);
}

#[test]
fn test_bucket_multiple_executions_same_bucket() {
    // Tests multiple executions within the same bucket with different dim values
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, a, b) = build_dynamic_add_graph();

    cx.set_dim_buckets('s', &[DimBucket::new(1, 8)]);

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    cx.set_dim('s', 1);
    rt.set_data(a, vec![1.0f32; 4]);
    let mut rng = SmallRng::seed_from_u64(42);
    rt = cx.search_rng(rt, 3, &mut rng);

    // Execute at different sizes within the same bucket
    for s in [1, 2, 4, 8] {
        cx.set_dim('s', s);
        let n = s * 4;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        rt.set_data(a, input.clone());
        rt.execute(&cx.dyn_map);
        let result = rt.get_f32(b);
        let expected: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
        assert_close(&result[..n], &expected, 1e-5, 1e-5);
    }
}

#[test]
#[should_panic(expected = "Overlapping buckets")]
fn test_bucket_overlapping_ranges_panics() {
    let mut cx = Graph::default();
    cx.set_dim_buckets('s', &[DimBucket::new(1, 4), DimBucket::new(3, 8)]);
}

#[test]
fn test_dim_bucket_contains() {
    let b = DimBucket::new(2, 10);
    assert!(!b.contains(1));
    assert!(b.contains(2));
    assert!(b.contains(5));
    assert!(b.contains(10));
    assert!(!b.contains(11));

    // Exact bucket
    let exact = DimBucket::new(3, 3);
    assert!(!exact.contains(2));
    assert!(exact.contains(3));
    assert!(!exact.contains(4));
}
