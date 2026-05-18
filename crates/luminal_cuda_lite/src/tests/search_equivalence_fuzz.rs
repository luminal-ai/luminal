//! End-to-end e-graph search-space equivalence fuzz tests.
//!
//! These tests do not compare against a hand-written reference. They assert the
//! stronger search invariant: every selectable LLIR graph from the same e-graph
//! must produce the same outputs for the same runtime inputs.

#[allow(dead_code)]
#[path = "../../../../examples/llama/src/model.rs"]
mod llama_model;

use half::bf16;
use luminal::{dtype::DType, prelude::*, shape::Expression};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::utilities::{CudaSearchEquivalenceFuzzer, get_cuda_stream, random_f32_vec};

const SEARCH_EQUIV_SAMPLES: usize = 32;

fn random_bf16_vec(n: usize, seed: u64, low: f32, high: f32) -> Vec<bf16> {
    random_f32_vec(n, seed, low, high)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn rms_norm(x: GraphTensor, weight: GraphTensor, eps: f32) -> GraphTensor {
    let normed = x.std_norm(x.shape.last_axis(), eps);
    normed * weight.expand_lhs(&x.dims()[..x.dims().len() - 1])
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (axis, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(axis, *dim);
    }
    let expert_flat_idx = exp_base + exp_within;
    weights.gather(expert_flat_idx)
}

#[test]
fn llama_architecture_search_space_equivalence_fuzz() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    const SEQ: usize = 2;
    const CTX: usize = 3;
    const SLOTS: usize = 4;

    let config = llama_model::LlamaConfig {
        layers: 2,
        hidden: 32,
        intermediate: 64,
        head_dim: 8,
        kv_groups: 2,
        vocab_size: 64,
    };

    let mut cx = Graph::default();
    cx.set_dim('s', SEQ);
    cx.set_dim('c', CTX);

    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let attn_mask = cx.named_tensor("attn_mask", ('s', 'c'));
    let kv_cache = llama_model::KVCache::new_with_config(&mut cx, SLOTS, config);
    let llama = llama_model::Llama::init_with_config(&mut cx, config);

    let (logits, cache_outputs) =
        llama.forward(input, q_pos, scatter_idx, gather_idx, attn_mask, &kv_cache);
    let logits = logits.output();
    let mut fuzzer = CudaSearchEquivalenceFuzzer::new(&mut cx, &stream)
        .seed(0x5EED_1234)
        .samples(SEARCH_EQUIV_SAMPLES)
        .generation_size(8)
        .mutations(3)
        .build_options(BuildSearchSpaceOptions::new().max_memory_mib(512))
        .output_f32(logits.id, "logits", 3e-3, 3e-3);
    for (layer, (k_out, v_out)) in cache_outputs.into_iter().enumerate() {
        let k_out = k_out.output();
        let v_out = v_out.output();
        fuzzer = fuzzer.output_f32(k_out.id, format!("layer{layer}.k_cache"), 3e-3, 3e-3);
        fuzzer = fuzzer.output_f32(v_out.id, format!("layer{layer}.v_cache"), 3e-3, 3e-3);
    }

    let mut rng = StdRng::seed_from_u64(0x11A_AA55);
    fuzzer = fuzzer
        .input_i32(input.id, vec![3, 17])
        .input_i32(q_pos.id, vec![1, 2])
        .input_i32(scatter_idx.id, vec![1, 2])
        .input_i32(gather_idx.id, vec![0, 1, 2])
        .input_f32(attn_mask.id, vec![0.0, 0.0, -1e4, 0.0, 0.0, 0.0]);

    let kv_dim = config.kv_dim();
    for tensor in kv_cache.tensors() {
        fuzzer = fuzzer.input_f32(tensor.id, vec![0.0; SLOTS * kv_dim]);
    }
    for tensor in llama.parameter_tensors() {
        let elements = tensor
            .dims()
            .iter()
            .map(|dim| dim.to_usize().expect("tiny llama test uses static params"))
            .product::<usize>();
        let data = (0..elements)
            .map(|_| rng.random_range(-0.08f32..0.08f32))
            .collect::<Vec<_>>();
        fuzzer = fuzzer.input_f32(tensor.id, data);
    }

    let report = fuzzer.run();
    eprintln!("llama search equivalence fuzz report: {report:?}");
}

#[test]
fn gemma_architecture_search_space_equivalence_fuzz() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    const SEQ: usize = 2;
    const HIDDEN: usize = 32;
    const Q_DIM: usize = 24;
    const INTERMEDIATE: usize = 64;
    const EPS: f32 = 1e-6;

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
    let mlp_out =
        (gemma_gelu(ff_normed.matmul(w_gate.t())) * ff_normed.matmul(w_up.t())).matmul(w_down.t());
    let mlp_normed = rms_norm(mlp_out, post_ff_norm_w, EPS);
    let out = (x + mlp_normed).output();

    let report = CudaSearchEquivalenceFuzzer::new(&mut cx, &stream)
        .seed(0x6E4D_4DAA)
        .samples(SEARCH_EQUIV_SAMPLES)
        .generation_size(8)
        .mutations(3)
        .build_options(BuildSearchSpaceOptions::new().max_memory_mib(512))
        .input_f32(input.id, random_f32_vec(SEQ * HIDDEN, 101, -0.15, 0.15))
        .input_f32(attn_norm_w.id, random_f32_vec(HIDDEN, 102, 0.7, 1.3))
        .input_f32(post_attn_norm_w.id, random_f32_vec(HIDDEN, 103, 0.7, 1.3))
        .input_f32(pre_ff_norm_w.id, random_f32_vec(HIDDEN, 104, 0.7, 1.3))
        .input_f32(post_ff_norm_w.id, random_f32_vec(HIDDEN, 105, 0.7, 1.3))
        .input_f32(proj_w.id, random_f32_vec(Q_DIM * HIDDEN, 106, -0.08, 0.08))
        .input_f32(
            o_proj_w.id,
            random_f32_vec(HIDDEN * Q_DIM, 107, -0.08, 0.08),
        )
        .input_f32(
            w_gate.id,
            random_f32_vec(INTERMEDIATE * HIDDEN, 108, -0.08, 0.08),
        )
        .input_f32(
            w_up.id,
            random_f32_vec(INTERMEDIATE * HIDDEN, 109, -0.08, 0.08),
        )
        .input_f32(
            w_down.id,
            random_f32_vec(HIDDEN * INTERMEDIATE, 110, -0.08, 0.08),
        )
        .output_f32(out.id, "gemma_block", 5e-3, 5e-3)
        .run();
    eprintln!("gemma search equivalence fuzz report: {report:?}");
}

#[test]
fn moe_architecture_search_space_equivalence_fuzz() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    const SEQ: usize = 2;
    const HIDDEN: usize = 16;
    const NUM_EXPERTS: usize = 8;
    const TOP_K: usize = 2;
    const MOE_INTERMEDIATE: usize = 6;
    const EPS: f32 = 1e-6;

    let mut cx = Graph::default();
    let router_input = cx.tensor(('s', HIDDEN));
    let expert_input = cx.tensor(('s', HIDDEN));
    let router_scale = cx.tensor(HIDDEN);
    let router_proj = cx.tensor((NUM_EXPERTS, HIDDEN));
    let per_expert_scale = cx.tensor(NUM_EXPERTS);
    let gate_up_weights = cx
        .tensor((NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = router_input.dims().len();
    let e_dim = *router_proj.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let router_hidden = router_input.std_norm(n - 1, EPS)
        * router_scale.expand_lhs(&router_input.dims()[..n - 1])
        * (HIDDEN as f32).sqrt().recip();
    let routing_weights = router_hidden.matmul(router_proj.t()).softmax(n - 1);

    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);
    let row_offsets = router_input
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_norm = top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);
    let top_k_weights = (top_k_values / top_k_norm) * per_expert_scale.gather(top_k_indices);

    let gate_up_gathered =
        gather_experts(expert_input, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = expert_input.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gemma_gelu(gate) * up;

    let down_gathered = gather_experts(expert_input, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_weights.unsqueeze(top_k_weights.dims().len());
    weights_exp.shape.expand(down_out.dims());
    let out = (down_out * weights_exp).sum(n - 1).output();
    cx.set_dim('s', SEQ);

    let report = CudaSearchEquivalenceFuzzer::new(&mut cx, &stream)
        .seed(0x0DEE_55EE)
        .samples(SEARCH_EQUIV_SAMPLES)
        .generation_size(8)
        .mutations(3)
        .build_options(BuildSearchSpaceOptions::new().max_memory_mib(512))
        .input_f32(
            router_input.id,
            random_f32_vec(SEQ * HIDDEN, 201, -0.15, 0.15),
        )
        .input_f32(
            expert_input.id,
            random_f32_vec(SEQ * HIDDEN, 202, -0.15, 0.15),
        )
        .input_f32(router_scale.id, random_f32_vec(HIDDEN, 203, 0.7, 1.3))
        .input_f32(
            router_proj.id,
            random_f32_vec(NUM_EXPERTS * HIDDEN, 204, -0.2, 0.2),
        )
        .input_f32(
            per_expert_scale.id,
            random_f32_vec(NUM_EXPERTS, 205, 0.5, 1.5),
        )
        .input_bf16(
            gate_up_weights.id,
            random_bf16_vec(NUM_EXPERTS * MOE_INTERMEDIATE * 2 * HIDDEN, 206, -0.1, 0.1),
        )
        .input_bf16(
            down_weights.id,
            random_bf16_vec(NUM_EXPERTS * HIDDEN * MOE_INTERMEDIATE, 207, -0.1, 0.1),
        )
        .output_f32(out.id, "gemma_moe_block", 5e-2, 5e-2)
        .run();
    eprintln!("moe search equivalence fuzz report: {report:?}");
}

#[test]
fn moe_architecture_native_reference_fuzz() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    const SEQ: usize = 2;
    const HIDDEN: usize = 16;
    const NUM_EXPERTS: usize = 8;
    const TOP_K: usize = 2;
    const MOE_INTERMEDIATE: usize = 6;

    let mut cx = Graph::default();
    let input = cx.tensor(('s', HIDDEN));
    let router = cx.tensor((NUM_EXPERTS, HIDDEN));
    let gate_up_weights = cx
        .tensor((NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = input.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(TOP_K);

    let routing_weights = input.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);
    let row_offsets = input
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_weights = top_k_values / top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);

    let gate_up_gathered = gather_experts(input, top_k_indices, gate_up_weights).cast(DType::F32);
    let input_exp = input.expand_dim(n - 1, TOP_K).unsqueeze(n);
    let gate_up_out = input_exp
        .matmul(gate_up_gathered.transpose(2, 3))
        .squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(input, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_weights.unsqueeze(top_k_weights.dims().len());
    weights_exp.shape.expand(down_out.dims());
    let out = (down_out * weights_exp).sum(n - 1).output();
    cx.set_dim('s', SEQ);

    let report = CudaSearchEquivalenceFuzzer::new(&mut cx, &stream)
        .seed(0x51A7_E5ED)
        .samples(SEARCH_EQUIV_SAMPLES)
        .generation_size(8)
        .mutations(3)
        .build_options(BuildSearchSpaceOptions::new().max_memory_mib(512))
        .native_reference()
        .input_f32(input.id, random_f32_vec(SEQ * HIDDEN, 301, -0.15, 0.15))
        .input_f32(
            router.id,
            random_f32_vec(NUM_EXPERTS * HIDDEN, 302, -0.2, 0.2),
        )
        .input_bf16(
            gate_up_weights.id,
            random_bf16_vec(NUM_EXPERTS * MOE_INTERMEDIATE * 2 * HIDDEN, 303, -0.1, 0.1),
        )
        .input_bf16(
            down_weights.id,
            random_bf16_vec(NUM_EXPERTS * HIDDEN * MOE_INTERMEDIATE, 304, -0.1, 0.1),
        )
        .output_f32(out.id, "qwen_swiglu_moe_native_reference", 6e-2, 6e-2)
        .run();
    eprintln!("moe native-reference fuzz report: {report:?}");
}
