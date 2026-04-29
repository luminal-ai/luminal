use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::LayerNorm;

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
    spec.num_kv_heads * max_seq * spec.head_dim * std::mem::size_of::<f32>()
}

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
    pub max_seq: usize,
}

impl KVCache {
    pub fn new(cx: &mut Graph, max_seq: usize) -> Self {
        let mut k_caches = Vec::with_capacity(LAYERS);
        let mut v_caches = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let spec = layer_spec(layer);
            let k = cx
                .named_tensor(
                    format!("kv_cache.{layer}.k"),
                    (spec.num_kv_heads, max_seq, spec.head_dim),
                )
                .persist();
            let v = cx
                .named_tensor(
                    format!("kv_cache.{layer}.v"),
                    (spec.num_kv_heads, max_seq, spec.head_dim),
                )
                .persist();
            k_caches.push(k);
            v_caches.push(v);
        }
        Self {
            k_caches,
            v_caches,
            max_seq,
        }
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
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let spec = layer_spec(layer);
            let gate = cx
                .named_tensor(
                    format!("model.layers.{layer}.mlp.gate_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                )
                .persist();
            let up = cx
                .named_tensor(
                    format!("model.layers.{layer}.mlp.up_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                )
                .persist();
            let down = cx
                .named_tensor(
                    format!("model.layers.{layer}.mlp.down_proj.weight"),
                    (HIDDEN, INTERMEDIATE),
                )
                .persist();

            let q_proj = cx
                .named_tensor(
                    format!("model.layers.{layer}.self_attn.q_proj.weight"),
                    (spec.q_dim, HIDDEN),
                )
                .persist();
            let k_proj = cx
                .named_tensor(
                    format!("model.layers.{layer}.self_attn.k_proj.weight"),
                    (spec.kv_dim, HIDDEN),
                )
                .persist();
            let v_proj = spec.has_v_proj.then(|| {
                cx.named_tensor(
                    format!("model.layers.{layer}.self_attn.v_proj.weight"),
                    (spec.kv_dim, HIDDEN),
                )
                .persist()
            });
            let o_proj = cx
                .named_tensor(
                    format!("model.layers.{layer}.self_attn.o_proj.weight"),
                    (HIDDEN, spec.q_dim),
                )
                .persist();
            let q_norm = cx
                .named_tensor(
                    format!("model.layers.{layer}.self_attn.q_norm.weight"),
                    spec.head_dim,
                )
                .persist();
            let k_norm = cx
                .named_tensor(
                    format!("model.layers.{layer}.self_attn.k_norm.weight"),
                    spec.head_dim,
                )
                .persist();
            let layer_scalar = cx
                .named_tensor(format!("model.layers.{layer}.layer_scalar"), HIDDEN)
                .persist();

            let router_scale = cx
                .named_tensor(format!("model.layers.{layer}.router.scale"), HIDDEN)
                .persist();
            let router_proj = cx
                .named_tensor(
                    format!("model.layers.{layer}.router.proj.weight"),
                    (NUM_EXPERTS, HIDDEN),
                )
                .persist();
            let per_expert_scale = cx
                .named_tensor(
                    format!("model.layers.{layer}.router.per_expert_scale"),
                    NUM_EXPERTS,
                )
                .persist();
            let gate_up_weights = cx
                .named_tensor(
                    format!("model.layers.{layer}.experts.gate_up_proj"),
                    (NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN),
                )
                .persist()
                .as_dtype(DType::Bf16);
            let down_weights = cx
                .named_tensor(
                    format!("model.layers.{layer}.experts.down_proj"),
                    (NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE),
                )
                .persist()
                .as_dtype(DType::Bf16);

            layers.push(Gemma4Layer {
                spec,
                gate,
                up,
                down,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                layer_scalar,
                input_layernorm: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.input_layernorm.weight"),
                    cx,
                ),
                post_attention_layernorm: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.post_attention_layernorm.weight"),
                    cx,
                ),
                pre_feedforward_layernorm: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.pre_feedforward_layernorm.weight"),
                    cx,
                ),
                post_feedforward_layernorm: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.post_feedforward_layernorm.weight"),
                    cx,
                ),
                post_feedforward_layernorm_1: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.post_feedforward_layernorm_1.weight"),
                    cx,
                ),
                post_feedforward_layernorm_2: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.post_feedforward_layernorm_2.weight"),
                    cx,
                ),
                pre_feedforward_layernorm_2: gemma4_norm(
                    HIDDEN,
                    &format!("model.layers.{layer}.pre_feedforward_layernorm_2.weight"),
                    cx,
                ),
                moe: Gemma4SparseMoE {
                    router_scale,
                    router_proj,
                    per_expert_scale,
                    gate_up_weights,
                    down_weights,
                },
            });
        }

        let embedding = cx
            .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
            .persist();
        let lm_head = cx
            .named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN))
            .persist();
        let lm_norm = gemma4_norm(HIDDEN, "model.norm.weight", cx);

        Self {
            embedding,
            lm_head,
            layers,
            lm_norm,
        }
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let seq = token_ids.dims1();
        let mut x = self.embedding.gather(
            (token_ids * HIDDEN).expand_dim(1, HIDDEN)
                + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
        );

        let mut cache_outputs = Vec::with_capacity(LAYERS);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                pos_ids,
                kv_cache.k_caches[layer_idx],
                kv_cache.v_caches[layer_idx],
                kv_cache.max_seq,
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }

        let logits = self.lm_norm.forward(x).matmul(self.lm_head.t());
        let logits = (logits / FINAL_LOGIT_SOFTCAP).tanh() * FINAL_LOGIT_SOFTCAP;
        (logits, cache_outputs)
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

fn gemma4_norm(dim: usize, weight_name: &str, cx: &mut Graph) -> LayerNorm {
    LayerNorm::new(dim, Some(weight_name), None, false, RMS_NORM_EPS, cx)
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize, head_dim: usize) -> GraphTensor {
    let seq = x.dims()[0];
    let reshaped = x.split_dims(1, head_dim);
    let normed = reshaped.std_norm(2, RMS_NORM_EPS);
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    (normed * w).merge_dims(1, 2)
}

fn value_norm(x: GraphTensor, head_dim: usize) -> GraphTensor {
    x.split_dims(1, head_dim)
        .std_norm(2, RMS_NORM_EPS)
        .merge_dims(1, 2)
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

    let cos = emb.cos().expand_dim(0, n_heads);
    let sin = emb.sin().expand_dim(0, n_heads);
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

fn hlir_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    max_seq: usize,
    spec: LayerSpec,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0];
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    let k_new = k_rope.split_dims(1, spec.head_dim).transpose(0, 1);
    let v_new = v.split_dims(1, spec.head_dim).transpose(0, 1);

    let h_offset = cx.arange(spec.num_kv_heads) * (max_seq * spec.head_dim);
    let p_offset = (cx.arange(seq) + prev) * spec.head_dim;
    let d_offset = cx.arange(spec.head_dim);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, spec.head_dim)
        + p_offset
            .expand_dim(0, spec.num_kv_heads)
            .expand_dim(2, spec.head_dim)
        + d_offset.expand_dim(0, spec.num_kv_heads).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    let k_full = k_cache_out.slice((.., ..total_seq, ..));
    let v_full = v_cache_out.slice((.., ..total_seq, ..));

    let k_3d = k_full.expand_dim(1, spec.kv_groups).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, spec.kv_groups).merge_dims(0, 1);
    let q = q_rope.split_dims(1, spec.head_dim).transpose(0, 1);

    // Gemma 4's text attention uses Q/K normalization and then leaves the
    // attention scaling at 1.0 in the reference implementation.
    let scores = q.matmul(k_3d.transpose(1, 2));

    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let future_mask = k_pos
        .expand_dim(0, seq)
        .gt(q_abs.expand_dim(1, total_seq))
        .cast(DType::F32);

    let mask_2d = if spec.is_sliding {
        let window_start = q_abs - (SLIDING_WINDOW_SIZE - 1) as f32;
        let past_mask = window_start
            .expand_dim(1, total_seq)
            .gt(k_pos.expand_dim(0, seq))
            .cast(DType::F32);
        future_mask + past_mask
    } else {
        future_mask
    };
    let mask_3d = mask_2d.expand_dim(0, N_HEADS);
    let masked_scores = scores + mask_3d * (-1e10f32);

    let attn_weights = masked_scores.softmax(2);
    let attn_out = attn_weights.matmul(v_3d);
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}

impl Gemma4Layer {
    pub fn forward(
        &self,
        x: GraphTensor,
        pos_ids: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
        max_seq: usize,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let residual = x;
        let x_attn = self.input_layernorm.forward(x);
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

        let (attn_out, k_cache_out, v_cache_out) = hlir_attention(
            q_rope, k_rope, v_normed, k_cache_in, v_cache_in, max_seq, self.spec,
        );

        let attn_proj = attn_out.matmul(self.o_proj.t());
        let x = residual + self.post_attention_layernorm.forward(attn_proj);

        let dense_ff = dense_ffn(
            self.pre_feedforward_layernorm.forward(x),
            self.gate,
            self.up,
            self.down,
        );
        let dense_ff = self.post_feedforward_layernorm_1.forward(dense_ff);

        let moe_out = self
            .moe
            .forward(x, self.pre_feedforward_layernorm_2.forward(x));
        let moe_out = self.post_feedforward_layernorm_2.forward(moe_out);

        let ff_out = self.post_feedforward_layernorm.forward(dense_ff + moe_out);
        let x = x + ff_out;
        let x = x * self
            .layer_scalar
            .expand_lhs(&x.dims()[..x.dims().len() - 1]);

        (x, k_cache_out, v_cache_out)
    }
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

        (down_out * top_k_weights.unsqueeze(top_k_weights.dims().len())).sum(n - 1)
    }
}
