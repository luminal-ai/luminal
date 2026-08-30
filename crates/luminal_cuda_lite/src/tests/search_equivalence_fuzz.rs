//! End-to-end e-graph search-space equivalence fuzz tests.
//!
//! These tests do not compare against a hand-written reference. They assert the
//! stronger search invariant: every selectable LLIR graph from the same e-graph
//! must produce finite, numerically close outputs for the same runtime inputs.

use half::bf16;
use luminal::{dtype::DType, prelude::*, shape::Expression};
use luminal_nn::{gather_rows, scatter_rows};
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

fn llama_rope(input: GraphTensor, positions: GraphTensor, head_dim: usize) -> GraphTensor {
    let input = input.split_dims(1, head_dim).transpose(0, 1);
    let freqs = input
        .graph()
        .arange_options(0, head_dim, 2)
        .cast(DType::F32)
        / head_dim as f32;
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let angles = positions
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));
    let x0 = input.slice((.., .., ..head_dim / 2));
    let x1 = input.slice((.., .., head_dim / 2..));
    let cos = angles.cos().expand_dim(0, x0.dims()[0]);
    let sin = angles.sin().expand_dim(0, x0.dims()[0]);
    (x0 * cos - x1 * sin)
        .concat_along(x1 * cos + x0 * sin, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

#[test]
fn llama_architecture_search_space_equivalence_fuzz() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    const LAYERS: usize = 2;
    const SEQ: usize = 2;
    const CTX: usize = 3;
    const SLOTS: usize = 4;
    const HIDDEN: usize = 32;
    const INTERMEDIATE: usize = 64;
    const HEAD_DIM: usize = 8;
    const KV_GROUPS: usize = 2;
    const VOCAB_SIZE: usize = 64;
    const N_HEADS: usize = HIDDEN / HEAD_DIM;
    const N_KV_HEADS: usize = N_HEADS / KV_GROUPS;
    const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
    const EPS: f32 = 1e-5;

    let mut cx = Graph::default();
    cx.set_dim('s', SEQ);
    cx.set_dim('c', CTX);

    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let embedding = cx
        .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
        .persist();
    let mut parameters = vec![embedding];
    let mut cache_inputs = Vec::with_capacity(2 * LAYERS);
    let mut cache_outputs = Vec::with_capacity(LAYERS);
    let mut x = embedding.gather(
        (input * HIDDEN).expand_dim(1, HIDDEN)
            + input
                .graph()
                .arange(HIDDEN)
                .expand_dim(0, Expression::from('s')),
    );

    for layer in 0..LAYERS {
        let parameter = |cx: &mut Graph, suffix: &str, shape: (usize, usize)| {
            cx.named_tensor(format!("model.layers.{layer}.{suffix}"), shape)
                .persist()
        };
        let attn_norm = cx
            .named_tensor(
                format!("model.layers.{layer}.input_layernorm.weight"),
                HIDDEN,
            )
            .persist();
        let mlp_norm = cx
            .named_tensor(
                format!("model.layers.{layer}.post_attention_layernorm.weight"),
                HIDDEN,
            )
            .persist();
        let q_weight = parameter(&mut cx, "self_attn.q_proj.weight", (HIDDEN, HIDDEN));
        let k_weight = parameter(&mut cx, "self_attn.k_proj.weight", (KV_DIM, HIDDEN));
        let v_weight = parameter(&mut cx, "self_attn.v_proj.weight", (KV_DIM, HIDDEN));
        let o_weight = parameter(&mut cx, "self_attn.o_proj.weight", (HIDDEN, HIDDEN));
        let gate_weight = parameter(&mut cx, "mlp.gate_proj.weight", (INTERMEDIATE, HIDDEN));
        let up_weight = parameter(&mut cx, "mlp.up_proj.weight", (INTERMEDIATE, HIDDEN));
        let down_weight = parameter(&mut cx, "mlp.down_proj.weight", (HIDDEN, INTERMEDIATE));
        parameters.extend([
            attn_norm,
            mlp_norm,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            gate_weight,
            up_weight,
            down_weight,
        ]);

        let k_cache = cx
            .named_tensor(format!("kv_cache.{layer}.k"), (SLOTS, KV_DIM))
            .persist();
        let v_cache = cx
            .named_tensor(format!("kv_cache.{layer}.v"), (SLOTS, KV_DIM))
            .persist();
        cache_inputs.extend([k_cache, v_cache]);

        let x_attn = rms_norm(x, attn_norm, EPS);
        let q = llama_rope(x_attn.matmul(q_weight.t()), q_pos, HEAD_DIM);
        let k = llama_rope(x_attn.matmul(k_weight.t()), q_pos, HEAD_DIM);
        let v = x_attn.matmul(v_weight.t());
        let k_out = scatter_rows(k, scatter_idx, k_cache, KV_DIM);
        let v_out = scatter_rows(v, scatter_idx, v_cache, KV_DIM);
        let k_ctx = gather_rows(k_out, gather_idx, KV_DIM);
        let v_ctx = gather_rows(v_out, gather_idx, KV_DIM);

        let q = (q * 1.0).split_dims(1, HEAD_DIM).transpose(0, 1);
        let k_ctx = k_ctx.split_dims(1, HEAD_DIM).permute((1, 2, 0));
        let v_ctx = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1);
        let k_ctx = k_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
        let v_ctx = v_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
        let scores = q.matmul(k_ctx) / (HEAD_DIM as f32).sqrt();
        let context = Expression::from('c');
        let causal = scores.graph().triu(context, 1).cast(scores.dtype) * -1e10;
        let row_offsets = (q_pos * context).expand_dim(1, context);
        let col_offsets = scores
            .graph()
            .arange(context)
            .expand_dim(0, Expression::from('s'));
        let mask = causal.gather(row_offsets + col_offsets);
        let weights = (scores + mask.expand_dim(0, N_HEADS)).softmax(2);
        let attn_out = weights.matmul(v_ctx).transpose(0, 1).merge_dims(1, 2);
        x += attn_out.matmul(o_weight.t());

        let x_mlp = rms_norm(x, mlp_norm, EPS);
        let gated = x_mlp.matmul(gate_weight.t()).swish() * x_mlp.matmul(up_weight.t());
        x += gated.matmul(down_weight.t());
        cache_outputs.push((k_out, v_out));
    }

    let final_norm = cx.named_tensor("model.norm.weight", HIDDEN).persist();
    let lm_head = cx
        .named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN))
        .persist();
    parameters.extend([final_norm, lm_head]);
    let logits = rms_norm(x, final_norm, EPS).matmul(lm_head.t()).output();
    let mut fuzzer = CudaSearchEquivalenceFuzzer::new(&mut cx, &stream)
        .seed(0x5EED_1234)
        .samples(SEARCH_EQUIV_SAMPLES)
        .generation_size(8)
        .mutations(3)
        .output_f32(logits.id, "logits", 5e-2, 5e-2);
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
        .input_i32(gather_idx.id, vec![0, 1, 2]);

    for tensor in cache_inputs {
        fuzzer = fuzzer.input_f32(tensor.id, vec![0.0; SLOTS * KV_DIM]);
    }
    for tensor in parameters {
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
fn moe_architecture_reference_runtime_fuzz() {
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
        .reference_runtime()
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
        .output_f32(out.id, "qwen_swiglu_moe_reference_runtime", 6e-2, 6e-2)
        .run();
    eprintln!("moe reference-runtime fuzz report: {report:?}");
}
