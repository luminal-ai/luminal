//! Fuzz tests for small transformer models on CUDA.
//!
//! Builds a mini Llama-like transformer (RMSNorm + causal self-attention + SwiGLU MLP)
//! and verifies CUDA execution against a CPU reference implementation using candle.

use luminal::prelude::*;

use super::utilities::{assert_close, get_cuda_stream, random_f32_vec};
use crate::runtime::CudaRuntime;

// ---- Tiny Llama hyperparameters ----
const SEQ: usize = 4;
const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;

// ---- Graph-based mini transformer (Luminal) ----

/// RMSNorm: x * rsqrt(mean(x^2) + eps), optionally scaled by weight
fn rms_norm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let normed = x.std_norm(x.shape.last_axis(), eps);
    normed * weight.expand_lhs(&x.dims()[..x.dims().len() - 1])
}

/// Build self-attention using a simple single-head approach.
/// Input: (seq, hidden), outputs: (seq, hidden)
fn self_attention(
    x: GraphTensor,
    wq: GraphTensor,
    wk: GraphTensor,
    wv: GraphTensor,
    wo: GraphTensor,
) -> GraphTensor {
    // Project to Q, K, V: (seq, hidden) @ (hidden, hidden)^T = (seq, hidden)
    let q = x.matmul(wq.t());
    let k = x.matmul(wk.t());
    let v = x.matmul(wv.t());

    // Simple single-head scaled dot-product attention (no causal mask for simplicity)
    let scale = 1.0 / (HIDDEN as f32).sqrt();
    let scores = q.matmul(k.t()) * scale; // (seq, seq)
    let attn_weights = scores.softmax(1); // softmax over key dim

    // Apply attention to values and output projection
    attn_weights.matmul(v).matmul(wo.t())
}

/// SwiGLU MLP: down(swish(gate(x)) * up(x))
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

/// Build a single transformer layer on the graph.
struct MiniTransformerLayer {
    attn_norm_w: GraphTensor,
    wq: GraphTensor,
    wk: GraphTensor,
    wv: GraphTensor,
    wo: GraphTensor,
    mlp_norm_w: GraphTensor,
    w_gate: GraphTensor,
    w_up: GraphTensor,
    w_down: GraphTensor,
}

impl MiniTransformerLayer {
    fn init(cx: &mut Graph) -> Self {
        Self {
            attn_norm_w: cx.tensor(HIDDEN),
            wq: cx.tensor((HIDDEN, HIDDEN)),
            wk: cx.tensor((HIDDEN, HIDDEN)),
            wv: cx.tensor((HIDDEN, HIDDEN)),
            wo: cx.tensor((HIDDEN, HIDDEN)),
            mlp_norm_w: cx.tensor(HIDDEN),
            w_gate: cx.tensor((INTERMEDIATE, HIDDEN)),
            w_up: cx.tensor((INTERMEDIATE, HIDDEN)),
            w_down: cx.tensor((HIDDEN, INTERMEDIATE)),
        }
    }

    fn forward(&self, x: GraphTensor) -> GraphTensor {
        // Pre-norm attention with residual
        let normed = rms_norm(x, self.attn_norm_w, 1e-5);
        let attn_out = self_attention(normed, self.wq, self.wk, self.wv, self.wo);
        let x = x + attn_out;

        // Pre-norm MLP with residual
        let normed = rms_norm(x, self.mlp_norm_w, 1e-5);
        let mlp_out = swiglu_mlp(normed, self.w_gate, self.w_up, self.w_down);
        x + mlp_out
    }

    /// Return all weight tensors and their sizes for data loading
    fn weights(&self) -> Vec<(GraphTensor, usize)> {
        vec![
            (self.attn_norm_w, HIDDEN),
            (self.wq, HIDDEN * HIDDEN),
            (self.wk, HIDDEN * HIDDEN),
            (self.wv, HIDDEN * HIDDEN),
            (self.wo, HIDDEN * HIDDEN),
            (self.mlp_norm_w, HIDDEN),
            (self.w_gate, INTERMEDIATE * HIDDEN),
            (self.w_up, INTERMEDIATE * HIDDEN),
            (self.w_down, HIDDEN * INTERMEDIATE),
        ]
    }
}

// ---- Candle CPU reference ----

/// CPU reference for RMSNorm using candle
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

/// CPU reference for self-attention (single-head, no causal mask)
fn self_attention_ref(
    x: &candle_core::Tensor,
    wq: &candle_core::Tensor,
    wk: &candle_core::Tensor,
    wv: &candle_core::Tensor,
    wo: &candle_core::Tensor,
) -> candle_core::Tensor {
    let q = x.matmul(&wq.t().unwrap()).unwrap();
    let k = x.matmul(&wk.t().unwrap()).unwrap();
    let v = x.matmul(&wv.t().unwrap()).unwrap();

    let scale = 1.0 / (HIDDEN as f64).sqrt();
    let scores = q.matmul(&k.t().unwrap()).unwrap();
    let scores = (scores * scale).unwrap();

    // Softmax over key dimension (dim 1)
    let max_val = scores.max(1).unwrap().unsqueeze(1).unwrap();
    let shifted = scores.broadcast_sub(&max_val).unwrap();
    let exps = shifted.exp().unwrap();
    let sum_exps = exps.sum(1).unwrap().unsqueeze(1).unwrap();
    let attn_weights = exps.broadcast_div(&sum_exps).unwrap();

    attn_weights
        .matmul(&v)
        .unwrap()
        .matmul(&wo.t().unwrap())
        .unwrap()
}

/// CPU reference for SwiGLU MLP
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

/// CPU reference for one transformer layer
fn transformer_layer_ref(
    x: &candle_core::Tensor,
    attn_norm_w: &candle_core::Tensor,
    wq: &candle_core::Tensor,
    wk: &candle_core::Tensor,
    wv: &candle_core::Tensor,
    wo: &candle_core::Tensor,
    mlp_norm_w: &candle_core::Tensor,
    w_gate: &candle_core::Tensor,
    w_up: &candle_core::Tensor,
    w_down: &candle_core::Tensor,
) -> candle_core::Tensor {
    let normed = rms_norm_ref(x, attn_norm_w, 1e-5);
    let attn_out = self_attention_ref(&normed, wq, wk, wv, wo);
    let x = (x + attn_out).unwrap();

    let normed = rms_norm_ref(&x, mlp_norm_w, 1e-5);
    let mlp_out = swiglu_mlp_ref(&normed, w_gate, w_up, w_down);
    (x + mlp_out).unwrap()
}

// ---- Helper to generate weight data for a layer ----

fn generate_layer_weights(
    layer: &MiniTransformerLayer,
    base_seed: u64,
) -> Vec<(GraphTensor, Vec<f32>)> {
    layer
        .weights()
        .iter()
        .enumerate()
        .map(|(i, (tensor, size))| {
            let data = random_f32_vec(*size, base_seed + i as u64, -0.5, 0.5);
            // RMSNorm weights should be initialized to ~1.0
            let data = if *size == HIDDEN {
                data.iter().map(|x| x + 1.0).collect::<Vec<_>>()
            } else {
                data
            };
            (*tensor, data)
        })
        .collect()
}

fn build_candle_ref(input_data: &[f32], weight_data: &[(GraphTensor, Vec<f32>)]) -> Vec<f32> {
    let device = candle_core::Device::Cpu;
    let ref_input =
        candle_core::Tensor::from_vec(input_data.to_vec(), (SEQ, HIDDEN), &device).unwrap();

    // weight_data: [attn_norm_w, wq, wk, wv, wo, mlp_norm_w, w_gate, w_up, w_down]
    let w = |idx: usize, shape: &[usize]| {
        candle_core::Tensor::from_vec(weight_data[idx].1.clone(), shape, &device).unwrap()
    };
    let ref_attn_norm_w = w(0, &[HIDDEN]);
    let ref_wq = w(1, &[HIDDEN, HIDDEN]);
    let ref_wk = w(2, &[HIDDEN, HIDDEN]);
    let ref_wv = w(3, &[HIDDEN, HIDDEN]);
    let ref_wo = w(4, &[HIDDEN, HIDDEN]);
    let ref_mlp_norm_w = w(5, &[HIDDEN]);
    let ref_w_gate = w(6, &[INTERMEDIATE, HIDDEN]);
    let ref_w_up = w(7, &[INTERMEDIATE, HIDDEN]);
    let ref_w_down = w(8, &[HIDDEN, INTERMEDIATE]);

    let expected = transformer_layer_ref(
        &ref_input,
        &ref_attn_norm_w,
        &ref_wq,
        &ref_wk,
        &ref_wv,
        &ref_wo,
        &ref_mlp_norm_w,
        &ref_w_gate,
        &ref_w_up,
        &ref_w_down,
    );
    expected.flatten_all().unwrap().to_vec1().unwrap()
}

// ---- Tests ----

/// Test a single transformer layer on CUDA against candle CPU reference.
#[test]
fn test_mini_transformer_layer() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((SEQ, HIDDEN));
    let layer = MiniTransformerLayer::init(&mut cx);
    let out = layer.forward(input).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_f32_vec(SEQ * HIDDEN, 42, -0.5, 0.5);
    rt.set_data(input, input_data.clone());

    let weight_data = generate_layer_weights(&layer, 100);
    for (tensor, data) in &weight_data {
        rt.set_data(*tensor, data.clone());
    }

    // Use minimal search iterations to avoid excessive graph rewriting
    // which can cause float drift through softmax/RMSNorm reordering
    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let expected = build_candle_ref(&input_data, &weight_data);
    assert_close(&result, &expected, 1e-2, 1e-2);
}

/// Test a two-layer transformer on CUDA against candle CPU reference.
#[test]
fn test_mini_transformer_two_layers() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((SEQ, HIDDEN));
    let layer1 = MiniTransformerLayer::init(&mut cx);
    let layer2 = MiniTransformerLayer::init(&mut cx);
    let x = layer1.forward(input).graph_break();
    let out = layer2.forward(x).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_f32_vec(SEQ * HIDDEN, 42, -0.5, 0.5);
    rt.set_data(input, input_data.clone());

    let layer1_weights = generate_layer_weights(&layer1, 200);
    let layer2_weights = generate_layer_weights(&layer2, 300);

    for (tensor, data) in layer1_weights.iter().chain(layer2_weights.iter()) {
        rt.set_data(*tensor, data.clone());
    }

    rt = cx.search(rt, 1);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    // Run two layers on CPU reference
    let device = candle_core::Device::Cpu;
    let mut ref_x = candle_core::Tensor::from_vec(input_data, (SEQ, HIDDEN), &device).unwrap();

    for weights in [&layer1_weights, &layer2_weights] {
        let w = |idx: usize, shape: &[usize]| {
            candle_core::Tensor::from_vec(weights[idx].1.clone(), shape, &device).unwrap()
        };
        ref_x = transformer_layer_ref(
            &ref_x,
            &w(0, &[HIDDEN]),
            &w(1, &[HIDDEN, HIDDEN]),
            &w(2, &[HIDDEN, HIDDEN]),
            &w(3, &[HIDDEN, HIDDEN]),
            &w(4, &[HIDDEN, HIDDEN]),
            &w(5, &[HIDDEN]),
            &w(6, &[INTERMEDIATE, HIDDEN]),
            &w(7, &[INTERMEDIATE, HIDDEN]),
            &w(8, &[HIDDEN, INTERMEDIATE]),
        );
    }

    let expected: Vec<f32> = ref_x.flatten_all().unwrap().to_vec1().unwrap();
    // Two layers accumulate more drift
    assert_close(&result, &expected, 2e-2, 2e-2);
}

/// Test the transformer with multiple random data seeds to catch data-dependent bugs.
#[test]
fn test_transformer_multi_seed() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    for seed in [42u64, 99, 777] {
        let mut cx = Graph::default();
        let input = cx.tensor((SEQ, HIDDEN));
        let layer = MiniTransformerLayer::init(&mut cx);
        let out = layer.forward(input).output();

        cx.build_search_space::<CudaRuntime>();
        let mut rt = CudaRuntime::initialize(stream.clone());

        let input_data = random_f32_vec(SEQ * HIDDEN, seed, -0.5, 0.5);
        rt.set_data(input, input_data.clone());

        let weight_data = generate_layer_weights(&layer, seed + 100);
        for (tensor, data) in &weight_data {
            rt.set_data(*tensor, data.clone());
        }

        rt = cx.search(rt, 1);
        rt.execute(&cx.dyn_map);
        let result = rt.get_f32(out);

        let expected = build_candle_ref(&input_data, &weight_data);
        assert_close(&result, &expected, 1e-2, 1e-2);
    }
}

/// Test just the RMSNorm component on CUDA
#[test]
fn test_rms_norm_cuda() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((SEQ, HIDDEN));
    let weight = cx.tensor(HIDDEN);
    let out = rms_norm(input, weight, 1e-5).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_f32_vec(SEQ * HIDDEN, 1, -0.5, 0.5);
    let weight_data: Vec<f32> = random_f32_vec(HIDDEN, 2, -0.5, 0.5)
        .iter()
        .map(|x| x + 1.0)
        .collect();
    rt.set_data(input, input_data.clone());
    rt.set_data(weight, weight_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input = candle_core::Tensor::from_vec(input_data, (SEQ, HIDDEN), &device).unwrap();
    let ref_weight = candle_core::Tensor::from_vec(weight_data, HIDDEN, &device).unwrap();
    let expected = rms_norm_ref(&ref_input, &ref_weight, 1e-5);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3, 1e-3);
}

/// Test just the self-attention on CUDA
#[test]
fn test_self_attention_cuda() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((SEQ, HIDDEN));
    let wq = cx.tensor((HIDDEN, HIDDEN));
    let wk = cx.tensor((HIDDEN, HIDDEN));
    let wv = cx.tensor((HIDDEN, HIDDEN));
    let wo = cx.tensor((HIDDEN, HIDDEN));
    let out = self_attention(input, wq, wk, wv, wo).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_f32_vec(SEQ * HIDDEN, 10, -0.5, 0.5);
    let wq_data = random_f32_vec(HIDDEN * HIDDEN, 11, -0.5, 0.5);
    let wk_data = random_f32_vec(HIDDEN * HIDDEN, 12, -0.5, 0.5);
    let wv_data = random_f32_vec(HIDDEN * HIDDEN, 13, -0.5, 0.5);
    let wo_data = random_f32_vec(HIDDEN * HIDDEN, 14, -0.5, 0.5);

    rt.set_data(input, input_data.clone());
    rt.set_data(wq, wq_data.clone());
    rt.set_data(wk, wk_data.clone());
    rt.set_data(wv, wv_data.clone());
    rt.set_data(wo, wo_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input = candle_core::Tensor::from_vec(input_data, (SEQ, HIDDEN), &device).unwrap();
    let ref_wq = candle_core::Tensor::from_vec(wq_data, (HIDDEN, HIDDEN), &device).unwrap();
    let ref_wk = candle_core::Tensor::from_vec(wk_data, (HIDDEN, HIDDEN), &device).unwrap();
    let ref_wv = candle_core::Tensor::from_vec(wv_data, (HIDDEN, HIDDEN), &device).unwrap();
    let ref_wo = candle_core::Tensor::from_vec(wo_data, (HIDDEN, HIDDEN), &device).unwrap();

    let expected = self_attention_ref(&ref_input, &ref_wq, &ref_wk, &ref_wv, &ref_wo);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-2, 1e-2);
}

/// Test just the SwiGLU MLP on CUDA
#[test]
fn test_swiglu_mlp_cuda() {
    let Some(stream) = get_cuda_stream() else {
        println!("CUDA not available, skipping");
        return;
    };

    let mut cx = Graph::default();
    let input = cx.tensor((SEQ, HIDDEN));
    let w_gate = cx.tensor((INTERMEDIATE, HIDDEN));
    let w_up = cx.tensor((INTERMEDIATE, HIDDEN));
    let w_down = cx.tensor((HIDDEN, INTERMEDIATE));
    let out = swiglu_mlp(input, w_gate, w_up, w_down).output();

    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    let input_data = random_f32_vec(SEQ * HIDDEN, 20, -0.5, 0.5);
    let gate_data = random_f32_vec(INTERMEDIATE * HIDDEN, 21, -0.5, 0.5);
    let up_data = random_f32_vec(INTERMEDIATE * HIDDEN, 22, -0.5, 0.5);
    let down_data = random_f32_vec(HIDDEN * INTERMEDIATE, 23, -0.5, 0.5);

    rt.set_data(input, input_data.clone());
    rt.set_data(w_gate, gate_data.clone());
    rt.set_data(w_up, up_data.clone());
    rt.set_data(w_down, down_data.clone());
    rt = cx.search(rt, 5);
    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(out);

    let device = candle_core::Device::Cpu;
    let ref_input = candle_core::Tensor::from_vec(input_data, (SEQ, HIDDEN), &device).unwrap();
    let ref_gate =
        candle_core::Tensor::from_vec(gate_data, (INTERMEDIATE, HIDDEN), &device).unwrap();
    let ref_up = candle_core::Tensor::from_vec(up_data, (INTERMEDIATE, HIDDEN), &device).unwrap();
    let ref_down =
        candle_core::Tensor::from_vec(down_data, (HIDDEN, INTERMEDIATE), &device).unwrap();

    let expected = swiglu_mlp_ref(&ref_input, &ref_gate, &ref_up, &ref_down);
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    assert_close(&result, &expected, 1e-3, 1e-3);
}
