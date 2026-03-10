use crate::{
    kernel::{lower_expression_for_metal, MetalEmbed, MetalGather},
    runtime::MetalRuntime,
};
use luminal::prelude::*;
use proptest::prelude::*;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_err = diff / e.abs().max(1.0);
        assert!(
            rel_err < tolerance,
            "Mismatch at index {}: got {}, expected {}, rel_err={}",
            i,
            a,
            e,
            rel_err
        );
    }
}

/// dynamic symbols in kernel expressions should route through dyn buffer.
#[test]
fn dynamic_const_codegen_uses_dyn_buffer() {
    let expr = (Expression::from('a') * 2 + Expression::from('z')).simplify();
    let code = lower_expression_for_metal(&expr, "idx");

    assert!(
        !code.contains("*const_"),
        "dynamic symbols should be lowered via dyn buffer, got: {code}"
    );
    assert!(
        code.contains("dyn["),
        "expected generated kernel expression to reference dyn buffer, got: {code}"
    );
}

/// dynamic-dimension reduction should compile and execute on Metal.
#[test]
fn dynamic_dim_sum_reduce_runs() {
    let mut cx = Graph::default();
    cx.set_dim('a', 3);
    let input = cx.tensor(('a', 2));
    let output = input.sum(0).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[9.0, 12.0], 0.001);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Test basic addition: input + input = 2 * input
    #[test]
    fn metal_add_test(len in 1usize..32, values in proptest::collection::vec(-5.0f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input + input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * 2.0).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test basic multiplication: input * input = input^2
    #[test]
    fn metal_mul_test(len in 1usize..32, values in proptest::collection::vec(0.1f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input * input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * v).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test exp2: 2^x
    #[test]
    fn metal_exp2_test(len in 1usize..32, values in proptest::collection::vec(-3.0f32..3.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = input.exp2().output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| 2.0f32.powf(*v)).collect();
        assert_close(&out, &expected, 0.001);
    }
}

/// Simple deterministic test for add
#[test]
fn metal_simple_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a + b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
}

/// Simple deterministic test for mul
#[test]
fn metal_simple_mul() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a * b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
}

/// Simple deterministic test for exp2
#[test]
fn metal_simple_exp2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.exp2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[0.0, 1.0, 2.0, 3.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 4.0, 8.0], 0.001);
}

#[test]
fn metal_simple_log2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.log2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 2.0, 3.0], 0.001);
}

#[test]
fn metal_simple_sin() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sin().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(
        input,
        &[
            0.0,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            3.0 * std::f32::consts::FRAC_PI_2,
        ],
    );
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 0.0, -1.0], 0.01);
}

#[test]
fn metal_simple_sqrt() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sqrt().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 4.0, 9.0, 16.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.001);
}

#[test]
fn metal_simple_recip() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.reciprocal().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 0.5, 0.25, 0.2], 0.001);
}

#[test]
fn metal_simple_mod() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a % b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[7.0, 10.0, 15.0, 8.5]);
    rt.set_data(b, &[3.0, 4.0, 6.0, 2.5]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 1.0], 0.001);
}

#[test]
fn metal_simple_less_than() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = a.lt(b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 5.0, 3.0, 4.0]);
    rt.set_data(b, &[2.0, 3.0, 3.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    // 1 < 2 = true (1.0), 5 < 3 = false (0.0), 3 < 3 = false (0.0), 4 < 5 = true (1.0)
    assert_eq!(out, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn metal_simple_sum_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // sum over axis 1
    let output = input.sum(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,2,3,4], [5,6,7,8]] -> [10, 26]
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[10.0, 26.0], 0.001);
}

#[test]
fn metal_simple_max_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // max over axis 1
    let output = input.max(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,4,2,3], [8,5,7,6]] -> [4, 8]
    rt.set_data(input, &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[4.0, 8.0], 0.001);
}

#[test]
fn test_scatter_basic() {
    let mut cx = Graph::default();
    let src = cx.tensor(3);
    let indexes = cx.tensor(3).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[10.0, 20.0, 30.0]);
    rt.set_data_i32(indexes, &[1i32, 3, 4]);
    rt.set_data(dest, &[0.0, 0.0, 0.0, 0.0, 0.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[0.0, 10.0, 0.0, 20.0, 30.0], 0.001);
}

#[test]
fn test_scatter_into_nonzero_dest() {
    let mut cx = Graph::default();
    let src = cx.tensor(1);
    let indexes = cx.tensor(1).as_dtype(DType::Int);
    let dest = cx.tensor(5);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[99.0]);
    rt.set_data_i32(indexes, &[2i32]);
    rt.set_data(dest, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[1.0, 2.0, 99.0, 4.0, 5.0], 0.001);
}

#[test]
fn test_scatter_all_positions() {
    let mut cx = Graph::default();
    let src = cx.tensor(4);
    let indexes = cx.tensor(4).as_dtype(DType::Int);
    let dest = cx.tensor(4);
    let result = src.scatter(indexes, dest).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(src, &[40.0, 30.0, 20.0, 10.0]);
    rt.set_data_i32(indexes, &[3i32, 2, 1, 0]);
    rt.set_data(dest, &[1.0, 2.0, 3.0, 4.0]);
    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(result);
    assert_close(&out, &[10.0, 20.0, 30.0, 40.0], 0.001);
}

// ============================================================================
// Embedding lookup regression: prove the old path was broken
// ============================================================================

/// Demonstrates that embedding lookups were *silently wrong* before MetalEmbed.
///
/// Root cause: MetalMul's MSL kernel declares all inputs as `device float *`.
/// An i32 token ID of 1 (bytes 0x00000001) read as f32 is ~1.4e-45.
/// Multiplied by embed_dim (4) it stays ~0, so the computed flat indices are
/// just arange(0..4) — meaning every token lookup lands on row 0 regardless
/// of the actual token ID.
///
/// This test excludes MetalEmbed to reproduce that broken path and asserts
/// the output is *wrong*, confirming that MetalEmbed fixes a real bug.
#[test]
fn metal_embed_without_fusion_is_wrong() {
    let vocab_size = 8usize;
    let embed_dim = 4usize;
    // Rows 0-7, each with distinct values so row confusion is detectable.
    let embed_data: Vec<f32> = (0..32).map(|i| i as f32 + 1.0).collect();
    // Token IDs 1,2,3,4 — all non-zero, so correct output must differ from row 0.
    let token_data: Vec<i32> = vec![1, 2, 3, 4];
    let seq_len = token_data.len();

    let mut cx = Graph::default();
    let token_ids = cx.tensor(seq_len).as_dtype(DType::Int);
    let embed_table = cx.tensor((vocab_size, embed_dim));
    let output = embed_table
        .gather(
            (token_ids * embed_dim).expand_dim(1, embed_dim)
                + cx.arange(embed_dim as i32).expand_dim(0, seq_len),
        )
        .output();

    // Exclude MetalEmbed — force the old broken path.
    cx.build_search_space_exclude_ops::<MetalRuntime, MetalEmbed>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data_i32(token_ids, &token_data);
    rt.set_data(embed_table, &embed_data);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);

    // Build the correct expected output.
    let mut expected = vec![0.0f32; seq_len * embed_dim];
    for (i, &tid) in token_data.iter().enumerate() {
        for j in 0..embed_dim {
            expected[i * embed_dim + j] = embed_data[tid as usize * embed_dim + j];
        }
    }

    // The old path produces the wrong answer.
    // The exact wrong values are hardware-dependent: MetalMul gives ~0 for each token ID,
    // so MetalAdd produces float values 0.0, 1.0, 2.0, ... which MetalGather reads as i32.
    // The i32 bit-pattern of 1.0f32 is 0x3F800000 = 1_065_353_216 — far out of bounds.
    // Metal returns undefined data for out-of-bounds reads rather than panicking.
    assert_ne!(
        out, expected,
        "expected the old path to give wrong results for non-zero token IDs"
    );
}

// ============================================================================
// Embedding lookup stress tests
// ============================================================================

/// Run a MetalEmbed correctness test with explicit data.
/// MetalGather is excluded so MetalEmbed is always the selected kernel.
fn run_embed_test(vocab_size: usize, embed_dim: usize, token_data: &[i32], embed_data: &[f32]) {
    assert_eq!(embed_data.len(), vocab_size * embed_dim);
    let seq_len = token_data.len();

    let mut cx = Graph::default();
    let token_ids = cx.tensor(seq_len).as_dtype(DType::Int);
    let embed_table = cx.tensor((vocab_size, embed_dim));
    let output = embed_table
        .gather(
            (token_ids * embed_dim).expand_dim(1, embed_dim)
                + cx.arange(embed_dim as i32).expand_dim(0, seq_len),
        )
        .output();

    let mut expected = vec![0.0f32; seq_len * embed_dim];
    for (i, &tid) in token_data.iter().enumerate() {
        for j in 0..embed_dim {
            expected[i * embed_dim + j] = embed_data[tid as usize * embed_dim + j];
        }
    }

    cx.build_search_space_exclude_ops::<MetalRuntime, MetalGather>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data_i32(token_ids, token_data);
    rt.set_data(embed_table, embed_data);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    assert_close(rt.get_f32(output).as_slice(), &expected, 1e-5);
}

#[test]
fn metal_embed_basic() {
    // 8 tokens × 4 dims, token ids [0, 2, 5, 7]
    let embed_data: Vec<f32> = (0..32).map(|i| i as f32 + 1.0).collect();
    run_embed_test(8, 4, &[0, 2, 5, 7], &embed_data);
}

/// First token (id 0) and last token (id vocab_size-1) — boundary rows of the table.
#[test]
fn metal_embed_boundary_tokens() {
    let vocab_size = 16usize;
    let embed_dim = 8usize;
    let embed_data: Vec<f32> = (0..(vocab_size * embed_dim)).map(|i| i as f32).collect();
    run_embed_test(
        vocab_size,
        embed_dim,
        &[0, (vocab_size - 1) as i32, 0, (vocab_size - 1) as i32],
        &embed_data,
    );
}

/// All tokens in the sequence are the same id — tests the repeated-token case.
#[test]
fn metal_embed_repeated_token() {
    let vocab_size = 32usize;
    let embed_dim = 16usize;
    let embed_data: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| -(i as f32))
        .collect();
    let token_data = vec![7i32; 64]; // 64 lookups all to token 7
    run_embed_test(vocab_size, embed_dim, &token_data, &embed_data);
}

/// Single-token sequence — the minimal case.
#[test]
fn metal_embed_single_token() {
    let vocab_size = 4usize;
    let embed_dim = 8usize;
    let embed_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    run_embed_test(vocab_size, embed_dim, &[3], &embed_data);
}

/// Single-element embedding vector (embed_dim = 1) — edge case for the inner loop.
#[test]
fn metal_embed_dim_one() {
    let vocab_size = 16usize;
    let embed_dim = 1usize;
    let embed_data: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 10.0).collect();
    run_embed_test(vocab_size, embed_dim, &[0, 3, 7, 15, 1], &embed_data);
}

/// Non-power-of-2 dimensions — exercises integer div/mod on odd thread counts.
#[test]
fn metal_embed_non_power_of_two() {
    let vocab_size = 13usize;
    let embed_dim = 7usize;
    let embed_data: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| i as f32 + 0.1)
        .collect();
    run_embed_test(vocab_size, embed_dim, &[0, 6, 12, 3, 9], &embed_data);
}

/// LLM-scale stress test: 32 k vocab, 512-dim embeddings, 128 tokens.
/// Exercises large buffers and verifies the kernel handles >1M output elements.
#[test]
fn metal_embed_llm_scale() {
    let vocab_size = 32_768usize;
    let embed_dim = 512usize;
    let seq_len = 128usize;

    let embed_data: Vec<f32> = (0..(vocab_size * embed_dim))
        .map(|i| (i % 1000) as f32 * 0.001)
        .collect();
    let token_data: Vec<i32> = (0..seq_len as i32)
        .map(|i| (i * 257) % vocab_size as i32) // prime-stride walk across vocab
        .collect();

    run_embed_test(vocab_size, embed_dim, &token_data, &embed_data);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn metal_embed_proptest(
        vocab_size in 4usize..64,
        embed_dim in 4usize..32,
        seq_len in 1usize..16,
        token_data in proptest::collection::vec(0i32..16, 1..16),
        embed_vals in proptest::collection::vec(-1.0f32..1.0, 1..2048),
    ) {
        prop_assume!(token_data.len() >= seq_len);
        prop_assume!(embed_vals.len() >= vocab_size * embed_dim);

        let token_data: Vec<i32> = token_data.into_iter()
            .take(seq_len)
            .map(|t| t % vocab_size as i32)
            .collect();
        let embed_data: Vec<f32> = embed_vals.into_iter().take(vocab_size * embed_dim).collect();

        run_embed_test(vocab_size, embed_dim, &token_data, &embed_data);
    }
}
