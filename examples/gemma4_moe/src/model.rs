use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::{LayerNorm, gather_rows, scatter_rows};

pub const LAYERS: usize = 30;
pub const HIDDEN: usize = 2816;
pub const INTERMEDIATE: usize = 2112;
pub const MOE_INTERMEDIATE: usize = 704;
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const N_HEADS: usize = 16;
pub const SLIDING_HEAD_DIM: usize = 256;
pub const FULL_HEAD_DIM: usize = 512;
pub const SLIDING_KV_HEADS: usize = 8;
pub const FULL_KV_HEADS: usize = 2;
pub const VOCAB_SIZE: usize = 262144;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const SLIDING_WINDOW_SIZE: usize = 1024;
pub const SLIDING_ROPE_THETA: f32 = 10_000.0;
pub const FULL_ROPE_THETA: f32 = 1_000_000.0;
pub const FULL_PARTIAL_ROTARY_FACTOR: f32 = 0.25;
pub const FINAL_LOGIT_SOFTCAP: f32 = 30.0;

#[derive(Clone, Copy)]
struct LayerSpec {
    is_sliding: bool,
    head_dim: usize,
    q_dim: usize,
    num_kv_heads: usize,
    kv_dim: usize,
    kv_groups: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
    has_v_proj: bool,
}

fn layer_spec(layer: usize) -> LayerSpec {
    if !(layer + 1).is_multiple_of(6) {
        LayerSpec {
            is_sliding: true,
            head_dim: SLIDING_HEAD_DIM,
            q_dim: N_HEADS * SLIDING_HEAD_DIM,
            num_kv_heads: SLIDING_KV_HEADS,
            kv_dim: SLIDING_KV_HEADS * SLIDING_HEAD_DIM,
            kv_groups: N_HEADS / SLIDING_KV_HEADS,
            rope_theta: SLIDING_ROPE_THETA,
            partial_rotary_factor: 1.0,
            has_v_proj: true,
        }
    } else {
        LayerSpec {
            is_sliding: false,
            head_dim: FULL_HEAD_DIM,
            q_dim: N_HEADS * FULL_HEAD_DIM,
            num_kv_heads: FULL_KV_HEADS,
            kv_dim: FULL_KV_HEADS * FULL_HEAD_DIM,
            kv_groups: N_HEADS / FULL_KV_HEADS,
            rope_theta: FULL_ROPE_THETA,
            partial_rotary_factor: FULL_PARTIAL_ROTARY_FACTOR,
            has_v_proj: false,
        }
    }
}

pub fn cache_bytes_for_layer(layer: usize, max_seq: usize) -> usize {
    let spec = layer_spec(layer);
    // Row-paged (max_seq, kv_dim) bf16 pools.
    max_seq * spec.kv_dim * 2
}

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
}

impl KVCache {
    pub fn new(cx: &mut Graph, max_seq: usize) -> Self {
        let mut k_caches = Vec::with_capacity(LAYERS);
        let mut v_caches = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let spec = layer_spec(layer);
            // Row-paged layout (num_slots, kv_dim) in bf16 — the spelling the
            // FlashInfer attention rewrites match.
            k_caches.push(
                persist(cx, format!("kv_cache.{layer}.k"), (max_seq, spec.kv_dim))
                    .as_dtype(DType::Bf16),
            );
            v_caches.push(
                persist(cx, format!("kv_cache.{layer}.v"), (max_seq, spec.kv_dim))
                    .as_dtype(DType::Bf16),
            );
        }
        Self { k_caches, v_caches }
    }
}

pub struct Gemma4MoE {
    embedding: GraphTensor,
    lm_head: GraphTensor,
    layers: Vec<Gemma4Layer>,
    lm_norm: LayerNorm,
}

impl Gemma4MoE {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            embedding: persist(cx, "model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
                .as_dtype(DType::Bf16),
            lm_head: persist(cx, "lm_head.weight", (VOCAB_SIZE, HIDDEN)).as_dtype(DType::Bf16),
            layers: (0..LAYERS)
                .map(|layer| Gemma4Layer::init(cx, layer))
                .collect(),
            lm_norm: gemma4_norm(HIDDEN, "model.norm.weight", cx),
        }
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = token_embedding(self.embedding, token_ids);

        let mut cache_outputs = Vec::with_capacity(LAYERS);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                pos_ids,
                scatter_idx,
                gather_idx,
                kv_cache.k_caches[layer_idx],
                kv_cache.v_caches[layer_idx],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }

        let logits = norm_in_f32(&self.lm_norm, x)
            .matmul(self.lm_head.t())
            .cast(DType::F32);
        let logits = (logits / FINAL_LOGIT_SOFTCAP).tanh() * FINAL_LOGIT_SOFTCAP;
        (logits, cache_outputs)
    }

    /// Forward + on-device sampling: repetition penalty against a persistent
    /// seen-token mask, then argmax. Per-step host I/O is one token id each
    /// way. `new_token` (the previously sampled id, -1 for none) is inserted
    /// into the mask BEFORE the penalty read.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_sampling(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        kv_cache: &KVCache,
        seen_mask: GraphTensor,
        new_token: GraphTensor,
        repetition_penalty: f32,
    ) -> (GraphTensor, GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let (logits, cache_outputs) =
            self.forward(token_ids, pos_ids, scatter_idx, gather_idx, kv_cache);
        let cx = unsafe { &mut *logits.graph_ref };
        let s = logits.dims()[0];

        let one = cx.constant_float(1.0).expand_dim(0, 1);
        let seen_out = one.scatter(new_token, seen_mask);

        let p = repetition_penalty;
        let seen_b = seen_out.expand_dim(0, s);
        let zero = cx.constant_float(0.0).expand_rhs(logits.dims());
        let pos = zero.lt(logits).cast(DType::F32);
        let factor = seen_b * (pos * (1.0 / p - 1.0) + (-pos + 1.0) * (p - 1.0)) + 1.0;
        let penalized = logits * factor;

        let sampled = penalized.argmax(1);
        (sampled, seen_out, cache_outputs)
    }
}

struct Gemma4Layer {
    spec: LayerSpec,
    gate: GraphTensor,
    up: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: Option<GraphTensor>,
    o_proj: GraphTensor,
    q_norm: GraphTensor,
    k_norm: GraphTensor,
    layer_scalar: GraphTensor,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    pre_feedforward_layernorm: LayerNorm,
    post_feedforward_layernorm: LayerNorm,
    post_feedforward_layernorm_1: LayerNorm,
    post_feedforward_layernorm_2: LayerNorm,
    pre_feedforward_layernorm_2: LayerNorm,
    moe: Gemma4SparseMoE,
}

struct Gemma4SparseMoE {
    router_scale: GraphTensor,
    router_proj: GraphTensor,
    per_expert_scale: GraphTensor,
    gate_up_weights: GraphTensor,
    down_weights: GraphTensor,
}

impl Gemma4Layer {
    fn init(cx: &mut Graph, layer: usize) -> Self {
        let spec = layer_spec(layer);
        Self {
            spec,
            gate: layer_weight(cx, layer, "mlp.gate_proj", (INTERMEDIATE, HIDDEN))
                .as_dtype(DType::Bf16),
            up: layer_weight(cx, layer, "mlp.up_proj", (INTERMEDIATE, HIDDEN))
                .as_dtype(DType::Bf16),
            down: layer_weight(cx, layer, "mlp.down_proj", (HIDDEN, INTERMEDIATE))
                .as_dtype(DType::Bf16),
            q_proj: layer_weight(cx, layer, "self_attn.q_proj", (spec.q_dim, HIDDEN))
                .as_dtype(DType::Bf16),
            k_proj: layer_weight(cx, layer, "self_attn.k_proj", (spec.kv_dim, HIDDEN))
                .as_dtype(DType::Bf16),
            v_proj: spec.has_v_proj.then(|| {
                layer_weight(cx, layer, "self_attn.v_proj", (spec.kv_dim, HIDDEN))
                    .as_dtype(DType::Bf16)
            }),
            o_proj: layer_weight(cx, layer, "self_attn.o_proj", (HIDDEN, spec.q_dim))
                .as_dtype(DType::Bf16),
            q_norm: layer_weight(cx, layer, "self_attn.q_norm", spec.head_dim),
            k_norm: layer_weight(cx, layer, "self_attn.k_norm", spec.head_dim),
            layer_scalar: layer_tensor(cx, layer, "layer_scalar", HIDDEN),
            input_layernorm: layer_norm(cx, layer, "input_layernorm"),
            post_attention_layernorm: layer_norm(cx, layer, "post_attention_layernorm"),
            pre_feedforward_layernorm: layer_norm(cx, layer, "pre_feedforward_layernorm"),
            post_feedforward_layernorm: layer_norm(cx, layer, "post_feedforward_layernorm"),
            post_feedforward_layernorm_1: layer_norm(cx, layer, "post_feedforward_layernorm_1"),
            post_feedforward_layernorm_2: layer_norm(cx, layer, "post_feedforward_layernorm_2"),
            pre_feedforward_layernorm_2: layer_norm(cx, layer, "pre_feedforward_layernorm_2"),
            moe: Gemma4SparseMoE {
                router_scale: layer_tensor(cx, layer, "router.scale", HIDDEN),
                router_proj: layer_weight(cx, layer, "router.proj", (NUM_EXPERTS, HIDDEN)),
                per_expert_scale: layer_tensor(cx, layer, "router.per_expert_scale", NUM_EXPERTS),
                gate_up_weights: layer_tensor(
                    cx,
                    layer,
                    "experts.gate_up_proj",
                    (NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN),
                )
                .as_dtype(DType::Bf16),
                down_weights: layer_tensor(
                    cx,
                    layer,
                    "experts.down_proj",
                    (NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE),
                )
                .as_dtype(DType::Bf16),
            },
        }
    }

    pub fn forward(
        &self,
        x: GraphTensor,
        pos_ids: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let residual = x;
        let x_attn = norm_in_f32(&self.input_layernorm, x);
        let q = x_attn.matmul(self.q_proj.t());
        let k_base = x_attn.matmul(self.k_proj.t());
        let v_base = if let Some(v_proj) = self.v_proj {
            x_attn.matmul(v_proj.t())
        } else {
            k_base
        };

        let q_normed = qk_norm(q, self.q_norm, N_HEADS, self.spec.head_dim);
        let k_normed = qk_norm(
            k_base,
            self.k_norm,
            self.spec.num_kv_heads,
            self.spec.head_dim,
        );
        let v_normed = value_norm(v_base, self.spec.head_dim);

        let q_rope = gemma4_rotary_embeddings(
            q_normed,
            pos_ids,
            N_HEADS,
            self.spec.head_dim,
            self.spec.rope_theta,
            self.spec.partial_rotary_factor,
        );
        let k_rope = gemma4_rotary_embeddings(
            k_normed,
            pos_ids,
            self.spec.num_kv_heads,
            self.spec.head_dim,
            self.spec.rope_theta,
            self.spec.partial_rotary_factor,
        );

        // bf16 paged attention: scores/softmax compute in the attention
        // island (FlashInfer when matched), KV cache is bf16.
        let (attn_out, k_cache_out, v_cache_out) = paged_attention(
            q_rope,
            k_rope,
            v_normed,
            k_cache_in,
            v_cache_in,
            scatter_idx,
            gather_idx,
            pos_ids,
            self.spec,
        );

        let attn_proj = attn_out.matmul(self.o_proj.t());
        let x = residual + norm_in_f32(&self.post_attention_layernorm, attn_proj);

        let dense_ff = dense_ffn(
            norm_in_f32(&self.pre_feedforward_layernorm, x),
            self.gate,
            self.up,
            self.down,
        );
        let dense_ff = norm_in_f32(&self.post_feedforward_layernorm_1, dense_ff);

        // Router and expert math in F32 (cast at the block edge), back to
        // bf16 for the residual stream.
        let moe_out = self.moe.forward(
            x.cast(DType::F32),
            norm_in_f32(&self.pre_feedforward_layernorm_2, x).cast(DType::F32),
        );
        let moe_out = norm_in_f32(
            &self.post_feedforward_layernorm_2,
            moe_out.cast(DType::Bf16),
        );

        let ff_out = norm_in_f32(&self.post_feedforward_layernorm, dense_ff + moe_out);
        let x = x + ff_out;
        let x = x * self
            .layer_scalar
            .cast(DType::Bf16)
            .expand_lhs(&x.dims()[..x.dims().len() - 1]);

        (x, k_cache_out, v_cache_out)
    }
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

fn layer_tensor(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    persist(cx, format!("model.layers.{layer}.{suffix}"), shape)
}

fn layer_weight(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    layer_tensor(cx, layer, &format!("{suffix}.weight"), shape)
}

fn layer_norm(cx: &mut Graph, layer: usize, name: &str) -> LayerNorm {
    gemma4_norm(HIDDEN, &format!("model.layers.{layer}.{name}.weight"), cx)
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
    let seq = token_ids.dims1();
    embedding.gather(
        (token_ids * HIDDEN).expand_dim(1, HIDDEN)
            + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
    )
}

fn gemma4_norm(dim: usize, weight_name: &str, cx: &mut Graph) -> LayerNorm {
    LayerNorm::new(dim, Some(weight_name), None, false, RMS_NORM_EPS, cx)
}

/// RMS norm computed in F32 with explicit casts when the input is 16-bit.
fn norm_in_f32(norm: &LayerNorm, x: GraphTensor) -> GraphTensor {
    if x.dtype == DType::F32 {
        norm.forward(x)
    } else {
        norm.forward(x.cast(DType::F32)).cast(x.dtype)
    }
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize, head_dim: usize) -> GraphTensor {
    let dtype = x.dtype;
    let seq = x.dims()[0];
    let x = if dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let reshaped = x.split_dims(1, head_dim);
    let normed = reshaped.std_norm(2, RMS_NORM_EPS);
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    let result = (normed * w).merge_dims(1, 2);
    if dtype == DType::F32 {
        result
    } else {
        result.cast(dtype)
    }
}

fn value_norm(x: GraphTensor, head_dim: usize) -> GraphTensor {
    let dtype = x.dtype;
    let x = if dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let result = x
        .split_dims(1, head_dim)
        .std_norm(2, RMS_NORM_EPS)
        .merge_dims(1, 2);
    if dtype == DType::F32 {
        result
    } else {
        result.cast(dtype)
    }
}

fn gemma4_rotary_embeddings(
    input: GraphTensor,
    pos_ids: GraphTensor,
    n_heads: usize,
    head_dim: usize,
    rope_theta: f32,
    partial_rotary_factor: f32,
) -> GraphTensor {
    let input = input.split_dims(1, head_dim).transpose(0, 1);
    let half_dim = head_dim / 2;
    let rope_angles = ((partial_rotary_factor * head_dim as f32) / 2.0).floor() as usize;

    let rotated = input
        .graph()
        .arange_options(0, rope_angles * 2, 2)
        .cast(DType::F32)
        / head_dim as f32;
    let rotated = rope_theta.pow(rotated).reciprocal();
    let inv_freqs = if rope_angles < half_dim {
        let zeros = input
            .graph()
            .arange(half_dim - rope_angles)
            .cast(DType::F32)
            * 0.0;
        rotated.concat_along(zeros, 0)
    } else {
        rotated
    };

    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    let x0 = input.slice((.., .., ..half_dim));
    let x1 = input.slice((.., .., half_dim..));

    // Angles in F32; rotation in the activation dtype.
    let mut cos = emb.cos();
    let mut sin = emb.sin();
    if x0.dtype != DType::F32 {
        cos = cos.cast(x0.dtype);
        sin = sin.cast(x0.dtype);
    }
    let cos = cos.expand_dim(0, n_heads);
    let sin = sin.expand_dim(0, n_heads);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
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

/// Paged attention in the llama/qwen HLIR spelling (scatter_rows into 2D
/// bf16 caches, gather via a flat row index, GQA broadcast, triu causal
/// mask) — the FlashInfer rewrites match this pattern; the HLIR chain
/// remains the fallback candidate. Gemma deltas vs qwen: scores are
/// SCALE-FREE (attention scaling 1.0), and sliding layers add a window
/// term to the mask: positions older than `q_pos - (W-1)` are blocked.
#[allow(clippy::too_many_arguments)]
fn paged_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    q_pos: GraphTensor,
    spec: LayerSpec,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let head_dim = spec.head_dim;
    let kv_dim = spec.kv_dim;
    let kv_groups = spec.kv_groups;

    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, kv_dim);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, kv_dim);

    let k = gather_rows(k_cache_out, gather_idx, kv_dim);
    let v_ctx = gather_rows(v_cache_out, gather_idx, kv_dim);

    let q = (q_rope * 1.0).split_dims(1, head_dim).transpose(0, 1);
    let k = k.split_dims(1, head_dim).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, head_dim).transpose(0, 1);

    let k = k.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;

    // Gemma 4's text attention leaves the attention scaling at 1.0 — no
    // 1/sqrt(head_dim) scale on the scores.
    let scores = q.matmul(k);
    let ctx = Expression::from('c');
    let seq = q_rope.dims()[0];
    let causal_square = scores.graph().triu(ctx, 1).cast(scores.dtype) * -1e10;
    let row_offsets = (q_pos * ctx).expand_dim(1, ctx);
    let col_offsets = scores.graph().arange(ctx).expand_dim(0, seq);
    let attn_mask = causal_square.gather(row_offsets + col_offsets);

    let attn_mask = if spec.is_sliding {
        // Window: block kv positions older than q_pos - (W-1).
        let q_f = q_pos.cast(DType::F32);
        let win_lo = q_f - (SLIDING_WINDOW_SIZE - 1) as f32;
        let col_f = cx.arange(ctx).cast(DType::F32);
        let too_old = col_f.expand_dim(0, seq).lt(win_lo.expand_dim(1, ctx));
        attn_mask + too_old.cast(scores.dtype) * -1e10
    } else {
        attn_mask
    };

    let masked_scores = scores + attn_mask.expand_dim(0, N_HEADS);
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);

    (attn_out, k_cache_out, v_cache_out)
}

fn dense_ffn(x: GraphTensor, gate: GraphTensor, up: GraphTensor, down: GraphTensor) -> GraphTensor {
    (gemma_gelu(x.matmul(gate.t())) * x.matmul(up.t())).matmul(down.t())
}

impl Gemma4SparseMoE {
    fn forward(&self, router_input: GraphTensor, expert_input: GraphTensor) -> GraphTensor {
        let n = router_input.dims().len();
        let e_dim = *self.router_proj.dims().first().unwrap();
        let k_expr = Expression::from(TOP_K);

        let router_hidden = router_input.std_norm(router_input.dims().len() - 1, RMS_NORM_EPS)
            * self
                .router_scale
                .expand_lhs(&router_input.dims()[..router_input.dims().len() - 1])
            * (HIDDEN as f32).sqrt().recip();
        let routing_weights = router_hidden.matmul(self.router_proj.t()).softmax(n - 1);

        let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);
        let row_offsets = router_input
            .graph()
            .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
        let routing_flat_idx = row_offsets + top_k_indices;
        let top_k_values = routing_weights.gather(routing_flat_idx);
        let top_k_norm = top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);
        let top_k_weights =
            (top_k_values / top_k_norm) * self.per_expert_scale.gather(top_k_indices);

        let gate_up_gathered =
            gather_experts(expert_input, top_k_indices, self.gate_up_weights).cast(DType::F32);
        let x_exp = expert_input.expand_dim(n - 1, TOP_K).unsqueeze(n);
        let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);

        let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
        let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
        let hidden = gemma_gelu(gate) * up;

        let down_gathered =
            gather_experts(expert_input, top_k_indices, self.down_weights).cast(DType::F32);
        let hidden_exp = hidden.unsqueeze(2);
        let down_out = hidden_exp.matmul(down_gathered.transpose(2, 3)).squeeze(2);

        let mut weights_exp = top_k_weights.unsqueeze(top_k_weights.dims().len());
        weights_exp.shape.expand(down_out.dims());
        (down_out * weights_exp).sum(n - 1)
    }
}
