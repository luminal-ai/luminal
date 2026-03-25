//! Fuzz tests for model-architecture-specific subgraphs (Llama, Gemma, Qwen).
//!
//! Tests many random e-graph extraction variants (genomes) against a candle CPU
//! reference to catch incorrect HLIR kernel fallback rewrites.

use luminal::prelude::*;

use super::utilities::{assert_close, fuzz_genomes, get_cuda_stream, random_f32_vec};
use crate::runtime::CudaRuntime;

/// Number of genomes to fuzz per test (higher than default GENOME_FUZZ_COUNT=20).
const FUZZ_COUNT: usize = 100;

// ============================================================================
// RMSNorm helper (used by all three models)
// ============================================================================

fn rms_norm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let normed = x.std_norm(x.shape.last_axis(), eps);
    normed * weight.expand_lhs(&x.dims()[..x.dims().len() - 1])
}

fn rms_norm_ref(
    x: &candle_core::Tensor,
    weight: &candle_core::Tensor,
    eps: f64,
) -> candle_core::Tensor {
    let dims = x.dims();
    let last_dim = dims[dims.len() - 1];
    let sq_mean = x.sqr().unwrap().mean_keepdim(dims.len() - 1).unwrap();
    let rsqrt = (sq_mean + eps).unwrap().sqrt().unwrap().recip().unwrap();
    let normed = x.broadcast_mul(&rsqrt).unwrap();
    normed
        .broadcast_mul(&weight.reshape((1, last_dim)).unwrap())
        .unwrap()
}

// ============================================================================
// SwiGLU MLP helper (used by all three models)
// ============================================================================

fn swiglu_mlp(
    x: GraphTensor,
    w_gate: GraphTensor,
    w_up: GraphTensor,
    w_down: GraphTensor,
) -> GraphTensor {
    let gate = x.matmul(w_gate.t()).swish();
    let up = x.matmul(w_up.t());
    (gate * up).matmul(w_down.t())
}

fn swiglu_mlp_ref(
    x: &candle_core::Tensor,
    w_gate: &candle_core::Tensor,
    w_up: &candle_core::Tensor,
    w_down: &candle_core::Tensor,
) -> candle_core::Tensor {
    let gate = x.matmul(&w_gate.t().unwrap()).unwrap().silu().unwrap();
    let up = x.matmul(&w_up.t().unwrap()).unwrap();
    (gate * up).unwrap().matmul(&w_down.t().unwrap()).unwrap()
}

// ============================================================================
// Generic test functions
// ============================================================================

/// Test a SwiGLU MLP block at given dimensions with genome fuzzing.
fn fuzz_mlp(seq: usize, hidden: usize, intermediate: usize, seed: u64) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((seq, hidden));
    let w_gate = cx.tensor((intermediate, hidden));
    let w_up = cx.tensor((intermediate, hidden));
    let w_down = cx.tensor((hidden, intermediate));
    let out = swiglu_mlp(input, w_gate, w_up, w_down).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());

    let input_data = random_f32_vec(seq * hidden, seed, -0.5, 0.5);
    let gate_data = random_f32_vec(intermediate * hidden, seed + 1, -0.3, 0.3);
    let up_data = random_f32_vec(intermediate * hidden, seed + 2, -0.3, 0.3);
    let down_data = random_f32_vec(hidden * intermediate, seed + 3, -0.3, 0.3);

    rt.set_data(input, input_data.clone());
    rt.set_data(w_gate, gate_data.clone());
    rt.set_data(w_up, up_data.clone());
    rt.set_data(w_down, down_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input =
        candle_core::Tensor::from_vec(input_data.clone(), (seq, hidden), &device).unwrap();
    let ref_gate =
        candle_core::Tensor::from_vec(gate_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_up =
        candle_core::Tensor::from_vec(up_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_down =
        candle_core::Tensor::from_vec(down_data.clone(), (hidden, intermediate), &device).unwrap();
    let expected = swiglu_mlp_ref(&ref_input, &ref_gate, &ref_up, &ref_down);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2, 1e-2);

    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(input, input_data.clone());
            rt.set_data(w_gate, gate_data.clone());
            rt.set_data(w_up, up_data.clone());
            rt.set_data(w_down, down_data.clone());
        },
        out.id,
        &expected,
        1e-2,
        1e-2,
        FUZZ_COUNT,
        seed,
    );
}

/// Test RMSNorm + matmul projection at given dimensions with genome fuzzing.
fn fuzz_norm_proj(seq: usize, hidden: usize, proj_dim: usize, eps: f32, seed: u64) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((seq, hidden));
    let norm_w = cx.tensor(hidden);
    let proj_w = cx.tensor((proj_dim, hidden));
    let out = rms_norm(input, norm_w, eps).matmul(proj_w.t()).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());

    let input_data = random_f32_vec(seq * hidden, seed, -0.5, 0.5);
    let norm_data: Vec<f32> = random_f32_vec(hidden, seed + 1, -0.5, 0.5)
        .iter()
        .map(|x| x + 1.0)
        .collect();
    let proj_data = random_f32_vec(proj_dim * hidden, seed + 2, -0.3, 0.3);

    rt.set_data(input, input_data.clone());
    rt.set_data(norm_w, norm_data.clone());
    rt.set_data(proj_w, proj_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input =
        candle_core::Tensor::from_vec(input_data.clone(), (seq, hidden), &device).unwrap();
    let ref_norm = candle_core::Tensor::from_vec(norm_data.clone(), hidden, &device).unwrap();
    let ref_proj =
        candle_core::Tensor::from_vec(proj_data.clone(), (proj_dim, hidden), &device).unwrap();
    let normed = rms_norm_ref(&ref_input, &ref_norm, eps as f64);
    let expected = normed.matmul(&ref_proj.t().unwrap()).unwrap();
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2, 1e-2);

    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(input, input_data.clone());
            rt.set_data(norm_w, norm_data.clone());
            rt.set_data(proj_w, proj_data.clone());
        },
        out.id,
        &expected,
        1e-2,
        1e-2,
        FUZZ_COUNT,
        seed,
    );
}

/// Test a full transformer layer (norm -> proj -> norm -> MLP) without attention.
fn fuzz_layer_no_attn(
    seq: usize,
    hidden: usize,
    intermediate: usize,
    proj_dim: usize,
    eps: f32,
    seed: u64,
) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((seq, hidden));
    let attn_norm_w = cx.tensor(hidden);
    let proj_w = cx.tensor((proj_dim, hidden));
    let o_proj_w = cx.tensor((hidden, proj_dim));
    let mlp_norm_w = cx.tensor(hidden);
    let w_gate = cx.tensor((intermediate, hidden));
    let w_up = cx.tensor((intermediate, hidden));
    let w_down = cx.tensor((hidden, intermediate));

    let normed = rms_norm(input, attn_norm_w, eps);
    let proj_out = normed.matmul(proj_w.t()).matmul(o_proj_w.t());
    let x = input + proj_out;
    let mlp_normed = rms_norm(x, mlp_norm_w, eps);
    let mlp_out = swiglu_mlp(mlp_normed, w_gate, w_up, w_down);
    let out = (x + mlp_out).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());

    let input_data = random_f32_vec(seq * hidden, seed, -0.5, 0.5);
    let attn_norm_data: Vec<f32> = random_f32_vec(hidden, seed + 1, -0.5, 0.5)
        .iter()
        .map(|x| x + 1.0)
        .collect();
    let proj_data = random_f32_vec(proj_dim * hidden, seed + 2, -0.3, 0.3);
    let o_proj_data = random_f32_vec(hidden * proj_dim, seed + 3, -0.3, 0.3);
    let mlp_norm_data: Vec<f32> = random_f32_vec(hidden, seed + 4, -0.5, 0.5)
        .iter()
        .map(|x| x + 1.0)
        .collect();
    let gate_data = random_f32_vec(intermediate * hidden, seed + 5, -0.3, 0.3);
    let up_data = random_f32_vec(intermediate * hidden, seed + 6, -0.3, 0.3);
    let down_data = random_f32_vec(hidden * intermediate, seed + 7, -0.3, 0.3);

    rt.set_data(input, input_data.clone());
    rt.set_data(attn_norm_w, attn_norm_data.clone());
    rt.set_data(proj_w, proj_data.clone());
    rt.set_data(o_proj_w, o_proj_data.clone());
    rt.set_data(mlp_norm_w, mlp_norm_data.clone());
    rt.set_data(w_gate, gate_data.clone());
    rt.set_data(w_up, up_data.clone());
    rt.set_data(w_down, down_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    // Candle reference
    let device = candle_core::Device::Cpu;
    let ref_input =
        candle_core::Tensor::from_vec(input_data.clone(), (seq, hidden), &device).unwrap();
    let ref_attn_norm =
        candle_core::Tensor::from_vec(attn_norm_data.clone(), hidden, &device).unwrap();
    let ref_proj =
        candle_core::Tensor::from_vec(proj_data.clone(), (proj_dim, hidden), &device).unwrap();
    let ref_o_proj =
        candle_core::Tensor::from_vec(o_proj_data.clone(), (hidden, proj_dim), &device).unwrap();
    let ref_mlp_norm =
        candle_core::Tensor::from_vec(mlp_norm_data.clone(), hidden, &device).unwrap();
    let ref_gate =
        candle_core::Tensor::from_vec(gate_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_up =
        candle_core::Tensor::from_vec(up_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_down =
        candle_core::Tensor::from_vec(down_data.clone(), (hidden, intermediate), &device).unwrap();

    let normed = rms_norm_ref(&ref_input, &ref_attn_norm, eps as f64);
    let proj_out = normed
        .matmul(&ref_proj.t().unwrap())
        .unwrap()
        .matmul(&ref_o_proj.t().unwrap())
        .unwrap();
    let x_ref = (&ref_input + proj_out).unwrap();
    let mlp_normed = rms_norm_ref(&x_ref, &ref_mlp_norm, eps as f64);
    let mlp_out = swiglu_mlp_ref(&mlp_normed, &ref_gate, &ref_up, &ref_down);
    let expected_t = (x_ref + mlp_out).unwrap();
    let expected: Vec<f32> = expected_t.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 2e-2, 2e-2);

    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(input, input_data.clone());
            rt.set_data(attn_norm_w, attn_norm_data.clone());
            rt.set_data(proj_w, proj_data.clone());
            rt.set_data(o_proj_w, o_proj_data.clone());
            rt.set_data(mlp_norm_w, mlp_norm_data.clone());
            rt.set_data(w_gate, gate_data.clone());
            rt.set_data(w_up, up_data.clone());
            rt.set_data(w_down, down_data.clone());
        },
        out.id,
        &expected,
        2e-2,
        2e-2,
        FUZZ_COUNT,
        seed,
    );
}

/// Test a SwiGLU MLP with HLIR-only to specifically verify
/// the HLIR matmul decomposition (KernelMul + KernelSumReduce).
fn fuzz_mlp_hlir_only(seq: usize, hidden: usize, intermediate: usize, seed: u64) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((seq, hidden));
    let w_gate = cx.tensor((intermediate, hidden));
    let w_up = cx.tensor((intermediate, hidden));
    let w_down = cx.tensor((hidden, intermediate));
    let out = swiglu_mlp(input, w_gate, w_up, w_down).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream.clone());

    let input_data = random_f32_vec(seq * hidden, seed, -0.5, 0.5);
    let gate_data = random_f32_vec(intermediate * hidden, seed + 1, -0.3, 0.3);
    let up_data = random_f32_vec(intermediate * hidden, seed + 2, -0.3, 0.3);
    let down_data = random_f32_vec(hidden * intermediate, seed + 3, -0.3, 0.3);

    rt.set_data(input, input_data.clone());
    rt.set_data(w_gate, gate_data.clone());
    rt.set_data(w_up, up_data.clone());
    rt.set_data(w_down, down_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input =
        candle_core::Tensor::from_vec(input_data.clone(), (seq, hidden), &device).unwrap();
    let ref_gate =
        candle_core::Tensor::from_vec(gate_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_up =
        candle_core::Tensor::from_vec(up_data.clone(), (intermediate, hidden), &device).unwrap();
    let ref_down =
        candle_core::Tensor::from_vec(down_data.clone(), (hidden, intermediate), &device).unwrap();
    let expected = swiglu_mlp_ref(&ref_input, &ref_gate, &ref_up, &ref_down);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2, 1e-2);

    fuzz_genomes::<f32>(
        &cx,
        &stream,
        |rt| {
            rt.set_data(input, input_data.clone());
            rt.set_data(w_gate, gate_data.clone());
            rt.set_data(w_up, up_data.clone());
            rt.set_data(w_down, down_data.clone());
        },
        out.id,
        &expected,
        1e-2,
        1e-2,
        FUZZ_COUNT,
        seed,
    );
}

// ============================================================================
// Llama-specific tests
// Llama 3 8B: HIDDEN=4096, INTERMEDIATE=14336, HEAD_DIM=128
// Using scaled-down dims that preserve architectural ratios
// ============================================================================

mod llama {
    use super::*;

    const SEQ: usize = 4;
    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 896; // ~3.5x hidden, matching 14336/4096
    const PROJ_DIM: usize = 256; // Q_DIM == HIDDEN for llama
    const EPS: f32 = 1e-5;

    #[test]
    fn fuzz_llama_mlp() {
        fuzz_mlp(SEQ, HIDDEN, INTERMEDIATE, 42);
    }

    #[test]
    fn fuzz_llama_norm_proj() {
        fuzz_norm_proj(SEQ, HIDDEN, PROJ_DIM, EPS, 100);
    }

    #[test]
    fn fuzz_llama_layer() {
        fuzz_layer_no_attn(SEQ, HIDDEN, INTERMEDIATE, PROJ_DIM, EPS, 200);
    }

    #[test]
    fn fuzz_llama_mlp_seq1() {
        fuzz_mlp(1, HIDDEN, INTERMEDIATE, 300);
    }

    #[test]
    fn fuzz_llama_mlp_seq7() {
        fuzz_mlp(7, HIDDEN, INTERMEDIATE, 400);
    }

    /// Force HLIR-only (no block ops) to specifically test the fallback path.
    #[test]
    fn fuzz_llama_mlp_hlir_only() {
        fuzz_mlp_hlir_only(SEQ, HIDDEN, INTERMEDIATE, 450);
    }
}

// ============================================================================
// Gemma-specific tests
// Gemma 3 4B: HIDDEN=2560, INTERMEDIATE=10240, HEAD_DIM=256, Q_DIM=2048
// Key difference: Q_DIM != HIDDEN, and 4 extra RMSNorm layers per block
// ============================================================================

mod gemma {
    use super::*;

    const SEQ: usize = 4;
    const HIDDEN: usize = 320; // divisible by 8 (N_HEADS)
    const INTERMEDIATE: usize = 1280; // 4x hidden, matching 10240/2560
    const Q_DIM: usize = 256; // scaled from 2048 (N_HEADS * HEAD_DIM)
    const EPS: f32 = 1e-6;

    #[test]
    fn fuzz_gemma_mlp() {
        fuzz_mlp(SEQ, HIDDEN, INTERMEDIATE, 500);
    }

    #[test]
    fn fuzz_gemma_norm_proj() {
        fuzz_norm_proj(SEQ, HIDDEN, Q_DIM, EPS, 600);
    }

    #[test]
    fn fuzz_gemma_layer() {
        fuzz_layer_no_attn(SEQ, HIDDEN, INTERMEDIATE, Q_DIM, EPS, 700);
    }

    /// Gemma has extra post-attention and post-feedforward norms.
    #[test]
    fn fuzz_gemma_layer_full_norms() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };

        let mut cx = Graph::default();
        let input = cx.tensor((SEQ, HIDDEN));
        let attn_norm_w = cx.tensor(HIDDEN);
        let post_attn_norm_w = cx.tensor(HIDDEN);
        let pre_ff_norm_w = cx.tensor(HIDDEN);
        let post_ff_norm_w = cx.tensor(HIDDEN);
        let proj_w = cx.tensor((Q_DIM, HIDDEN));
        let o_proj_w = cx.tensor((HIDDEN, Q_DIM));
        let w_gate = cx.tensor((INTERMEDIATE, HIDDEN));
        let w_up = cx.tensor((INTERMEDIATE, HIDDEN));
        let w_down = cx.tensor((HIDDEN, INTERMEDIATE));

        let normed = rms_norm(input, attn_norm_w, EPS);
        let proj_out = normed.matmul(proj_w.t()).matmul(o_proj_w.t());
        let attn_normed = rms_norm(proj_out, post_attn_norm_w, EPS);
        let x = input + attn_normed;
        let ff_normed = rms_norm(x, pre_ff_norm_w, EPS);
        let mlp_out = swiglu_mlp(ff_normed, w_gate, w_up, w_down);
        let mlp_normed = rms_norm(mlp_out, post_ff_norm_w, EPS);
        let out = (x + mlp_normed).output();

        cx.build_search_space::<CudaRuntime>();
        let mut rt = CudaRuntime::initialize(stream.clone());

        let seed = 800u64;
        let input_data = random_f32_vec(SEQ * HIDDEN, seed, -0.5, 0.5);
        let attn_norm_data: Vec<f32> = random_f32_vec(HIDDEN, seed + 1, -0.5, 0.5)
            .iter()
            .map(|x| x + 1.0)
            .collect();
        let post_attn_data: Vec<f32> = random_f32_vec(HIDDEN, seed + 2, -0.5, 0.5)
            .iter()
            .map(|x| x + 1.0)
            .collect();
        let pre_ff_data: Vec<f32> = random_f32_vec(HIDDEN, seed + 3, -0.5, 0.5)
            .iter()
            .map(|x| x + 1.0)
            .collect();
        let post_ff_data: Vec<f32> = random_f32_vec(HIDDEN, seed + 4, -0.5, 0.5)
            .iter()
            .map(|x| x + 1.0)
            .collect();
        let proj_data = random_f32_vec(Q_DIM * HIDDEN, seed + 5, -0.3, 0.3);
        let o_proj_data = random_f32_vec(HIDDEN * Q_DIM, seed + 6, -0.3, 0.3);
        let gate_data = random_f32_vec(INTERMEDIATE * HIDDEN, seed + 7, -0.3, 0.3);
        let up_data = random_f32_vec(INTERMEDIATE * HIDDEN, seed + 8, -0.3, 0.3);
        let down_data = random_f32_vec(HIDDEN * INTERMEDIATE, seed + 9, -0.3, 0.3);

        rt.set_data(input, input_data.clone());
        rt.set_data(attn_norm_w, attn_norm_data.clone());
        rt.set_data(post_attn_norm_w, post_attn_data.clone());
        rt.set_data(pre_ff_norm_w, pre_ff_data.clone());
        rt.set_data(post_ff_norm_w, post_ff_data.clone());
        rt.set_data(proj_w, proj_data.clone());
        rt.set_data(o_proj_w, o_proj_data.clone());
        rt.set_data(w_gate, gate_data.clone());
        rt.set_data(w_up, up_data.clone());
        rt.set_data(w_down, down_data.clone());
        rt = cx.search(rt, 5);
        rt.execute(&cx.dyn_map);
        let result = rt.get_f32(out);

        // Candle reference
        let device = candle_core::Device::Cpu;
        let t = |data: &[f32], shape: &[usize]| {
            candle_core::Tensor::from_vec(data.to_vec(), shape, &device).unwrap()
        };
        let ref_input = t(&input_data, &[SEQ, HIDDEN]);
        let ref_attn_norm = t(&attn_norm_data, &[HIDDEN]);
        let ref_post_attn = t(&post_attn_data, &[HIDDEN]);
        let ref_pre_ff = t(&pre_ff_data, &[HIDDEN]);
        let ref_post_ff = t(&post_ff_data, &[HIDDEN]);
        let ref_proj = t(&proj_data, &[Q_DIM, HIDDEN]);
        let ref_o_proj = t(&o_proj_data, &[HIDDEN, Q_DIM]);
        let ref_gate = t(&gate_data, &[INTERMEDIATE, HIDDEN]);
        let ref_up = t(&up_data, &[INTERMEDIATE, HIDDEN]);
        let ref_down = t(&down_data, &[HIDDEN, INTERMEDIATE]);

        let normed = rms_norm_ref(&ref_input, &ref_attn_norm, EPS as f64);
        let proj_out = normed
            .matmul(&ref_proj.t().unwrap())
            .unwrap()
            .matmul(&ref_o_proj.t().unwrap())
            .unwrap();
        let attn_normed = rms_norm_ref(&proj_out, &ref_post_attn, EPS as f64);
        let x_ref = (&ref_input + attn_normed).unwrap();
        let ff_normed = rms_norm_ref(&x_ref, &ref_pre_ff, EPS as f64);
        let mlp_out = swiglu_mlp_ref(&ff_normed, &ref_gate, &ref_up, &ref_down);
        let mlp_normed = rms_norm_ref(&mlp_out, &ref_post_ff, EPS as f64);
        let expected_t = (x_ref + mlp_normed).unwrap();
        let expected: Vec<f32> = expected_t.flatten_all().unwrap().to_vec1().unwrap();

        assert_close(&result, &expected, 2e-2, 2e-2);

        fuzz_genomes::<f32>(
            &cx,
            &stream,
            |rt| {
                rt.set_data(input, input_data.clone());
                rt.set_data(attn_norm_w, attn_norm_data.clone());
                rt.set_data(post_attn_norm_w, post_attn_data.clone());
                rt.set_data(pre_ff_norm_w, pre_ff_data.clone());
                rt.set_data(post_ff_norm_w, post_ff_data.clone());
                rt.set_data(proj_w, proj_data.clone());
                rt.set_data(o_proj_w, o_proj_data.clone());
                rt.set_data(w_gate, gate_data.clone());
                rt.set_data(w_up, up_data.clone());
                rt.set_data(w_down, down_data.clone());
            },
            out.id,
            &expected,
            2e-2,
            2e-2,
            FUZZ_COUNT,
            seed,
        );
    }

    #[test]
    fn fuzz_gemma_mlp_seq1() {
        fuzz_mlp(1, HIDDEN, INTERMEDIATE, 900);
    }

    /// Force HLIR-only to test fallback path with Gemma dimensions.
    #[test]
    fn fuzz_gemma_mlp_hlir_only() {
        fuzz_mlp_hlir_only(SEQ, HIDDEN, INTERMEDIATE, 950);
    }
}

// ============================================================================
// Qwen-specific tests
// Qwen3-4B: HIDDEN=2560, INTERMEDIATE=9728, HEAD_DIM=128, Q_DIM=4096
// Key difference: Q_DIM > HIDDEN, tied embeddings (lm_head = embedding.t())
// ============================================================================

mod qwen {
    use super::*;

    const SEQ: usize = 4;
    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 768; // ~3x hidden, matching 9728/2560
    const Q_DIM: usize = 512; // scaled from 4096 (Q_DIM > HIDDEN)
    const EPS: f32 = 1e-6;

    #[test]
    fn fuzz_qwen_mlp() {
        fuzz_mlp(SEQ, HIDDEN, INTERMEDIATE, 1000);
    }

    #[test]
    fn fuzz_qwen_norm_proj() {
        fuzz_norm_proj(SEQ, HIDDEN, Q_DIM, EPS, 1100);
    }

    #[test]
    fn fuzz_qwen_layer() {
        fuzz_layer_no_attn(SEQ, HIDDEN, INTERMEDIATE, Q_DIM, EPS, 1200);
    }

    /// Qwen uses tied embeddings: lm_head = embedding^T
    #[test]
    fn fuzz_qwen_lm_head() {
        let Some(stream) = get_cuda_stream() else {
            return;
        };

        const VOCAB: usize = 512;

        let mut cx = Graph::default();
        let input = cx.tensor((SEQ, HIDDEN));
        let norm_w = cx.tensor(HIDDEN);
        let embedding = cx.tensor((VOCAB, HIDDEN));
        let out = rms_norm(input, norm_w, EPS).matmul(embedding.t()).output();

        cx.build_search_space::<CudaRuntime>();
        let mut rt = CudaRuntime::initialize(stream.clone());

        let seed = 1300u64;
        let input_data = random_f32_vec(SEQ * HIDDEN, seed, -0.5, 0.5);
        let norm_data: Vec<f32> = random_f32_vec(HIDDEN, seed + 1, -0.5, 0.5)
            .iter()
            .map(|x| x + 1.0)
            .collect();
        let emb_data = random_f32_vec(VOCAB * HIDDEN, seed + 2, -0.3, 0.3);

        rt.set_data(input, input_data.clone());
        rt.set_data(norm_w, norm_data.clone());
        rt.set_data(embedding, emb_data.clone());
        rt = cx.search(rt, 5);
        rt.execute(&cx.dyn_map);
        let result = rt.get_f32(out);

        let device = candle_core::Device::Cpu;
        let ref_input =
            candle_core::Tensor::from_vec(input_data.clone(), (SEQ, HIDDEN), &device).unwrap();
        let ref_norm = candle_core::Tensor::from_vec(norm_data.clone(), HIDDEN, &device).unwrap();
        let ref_emb =
            candle_core::Tensor::from_vec(emb_data.clone(), (VOCAB, HIDDEN), &device).unwrap();
        let normed = rms_norm_ref(&ref_input, &ref_norm, EPS as f64);
        let expected = normed.matmul(&ref_emb.t().unwrap()).unwrap();
        let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

        assert_close(&result, &expected, 1e-2, 1e-2);

        fuzz_genomes::<f32>(
            &cx,
            &stream,
            |rt| {
                rt.set_data(input, input_data.clone());
                rt.set_data(norm_w, norm_data.clone());
                rt.set_data(embedding, emb_data.clone());
            },
            out.id,
            &expected,
            1e-2,
            1e-2,
            FUZZ_COUNT,
            seed,
        );
    }

    #[test]
    fn fuzz_qwen_mlp_seq1() {
        fuzz_mlp(1, HIDDEN, INTERMEDIATE, 1400);
    }

    #[test]
    fn fuzz_qwen_mlp_seq7() {
        fuzz_mlp(7, HIDDEN, INTERMEDIATE, 1500);
    }

    /// Force HLIR-only to test fallback path with Qwen dimensions.
    #[test]
    fn fuzz_qwen_mlp_hlir_only() {
        fuzz_mlp_hlir_only(SEQ, HIDDEN, INTERMEDIATE, 1550);
    }
}
