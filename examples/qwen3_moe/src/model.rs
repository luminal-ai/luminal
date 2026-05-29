use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::LayerNorm;

// Qwen3-30B-A3B hyperparams
pub const LAYERS: usize = 48;
pub const HIDDEN: usize = 2048;
pub const MOE_INTERMEDIATE: usize = 768;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 4;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // = 8
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // = 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // = 512
pub const VOCAB_SIZE: usize = 151936;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const NUM_EXPERTS: usize = 128;
pub const TOP_K: usize = 8;

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
    pub max_seq: usize,
}

impl KVCache {
    pub fn new(cx: &mut Graph, max_seq: usize) -> Self {
        let mut k_caches = Vec::with_capacity(LAYERS);
        let mut v_caches = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            k_caches.push(persist(
                cx,
                format!("kv_cache.{l}.k"),
                (N_KV_HEADS, max_seq, HEAD_DIM),
            ));
            v_caches.push(persist(
                cx,
                format!("kv_cache.{l}.v"),
                (N_KV_HEADS, max_seq, HEAD_DIM),
            ));
        }
        Self {
            k_caches,
            v_caches,
            max_seq,
        }
    }
}

pub struct Qwen3MoE {
    pub embedding: GraphTensor,
    layers: Vec<Qwen3MoELayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Qwen3MoE {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            embedding: persist(cx, "model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            layers: (0..LAYERS).map(|l| Qwen3MoELayer::init(cx, l)).collect(),
            lm_norm: rms_norm(cx, "model.norm.weight"),
            lm_head: persist(cx, "lm_head.weight", (VOCAB_SIZE, HIDDEN)),
        }
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = token_embedding(self.embedding, token_ids);
        let mut cache_outputs = Vec::with_capacity(LAYERS);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                pos_ids,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
                kv_cache.max_seq,
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        let logits = self.lm_norm.forward(x).matmul(self.lm_head.t());
        (logits, cache_outputs)
    }
}

struct Qwen3MoELayer {
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    q_norm: GraphTensor,
    k_norm: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
    moe: QwenMoE,
}

/// Gated MoE module using standard graph ops (SwiGLU variant).
struct QwenMoE {
    router: GraphTensor,          // [E, H] F32
    gate_up_weights: GraphTensor, // [E, intermediate*2, H] BF16
    down_weights: GraphTensor,    // [E, H, intermediate] BF16
}

impl Qwen3MoELayer {
    fn init(cx: &mut Graph, l: usize) -> Self {
        Self {
            q_proj: layer_weight(cx, l, "self_attn.q_proj", (Q_DIM, HIDDEN)),
            k_proj: layer_weight(cx, l, "self_attn.k_proj", (KV_DIM, HIDDEN)),
            v_proj: layer_weight(cx, l, "self_attn.v_proj", (KV_DIM, HIDDEN)),
            o_proj: layer_weight(cx, l, "self_attn.o_proj", (HIDDEN, Q_DIM)),
            q_norm: layer_weight(cx, l, "self_attn.q_norm", HEAD_DIM),
            k_norm: layer_weight(cx, l, "self_attn.k_norm", HEAD_DIM),
            attn_rms: rms_norm(cx, format!("model.layers.{l}.input_layernorm.weight")),
            mlp_rms: rms_norm(
                cx,
                format!("model.layers.{l}.post_attention_layernorm.weight"),
            ),
            moe: QwenMoE {
                router: layer_weight(cx, l, "mlp.gate", (NUM_EXPERTS, HIDDEN)),
                gate_up_weights: layer_tensor(
                    cx,
                    l,
                    "mlp.gate_up_weights",
                    (NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN),
                )
                .as_dtype(DType::Bf16),
                down_weights: layer_tensor(
                    cx,
                    l,
                    "mlp.down_weights",
                    (NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE),
                )
                .as_dtype(DType::Bf16),
            },
        }
    }

    pub fn forward(
        &self,
        mut x: GraphTensor,
        pos_ids: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
        max_seq: usize,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        // Attention
        let x_attn = self.attn_rms.forward(x);
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        // QK-norm: per-head RMS normalization
        let q_normed = qk_norm(q, self.q_norm, N_HEADS);
        let k_normed = qk_norm(k, self.k_norm, N_KV_HEADS);

        // RoPE
        let q_rope = qwen_rotary_embeddings(q_normed, pos_ids, N_HEADS);
        let k_rope = qwen_rotary_embeddings(k_normed, pos_ids, N_KV_HEADS);

        let (attn_out, k_cache_out, v_cache_out) =
            attention(q_rope, k_rope, v, k_cache_in, v_cache_in, max_seq);
        x += attn_out.matmul(self.o_proj.t());

        // MoE FFN
        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out = self.moe.forward(x_mlp);
        (x + mlp_out, k_cache_out, v_cache_out)
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

fn rms_norm(cx: &mut Graph, weight_name: impl ToString) -> LayerNorm {
    LayerNorm::new(
        HIDDEN,
        Some(&weight_name.to_string()),
        None,
        false,
        RMS_NORM_EPS,
        cx,
    )
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
    let seq = token_ids.dims1();
    embedding.gather(
        (token_ids * HIDDEN).expand_dim(1, HIDDEN)
            + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
    )
}

impl QwenMoE {
    fn forward(&self, x: GraphTensor) -> GraphTensor {
        let n = x.dims().len(); // 2 for [s, H]
        let e_dim = *self.router.dims().first().unwrap(); // E
        let k_expr = Expression::from(TOP_K);

        // 1. Router: softmax(x @ router^T) → [s, E]
        let routing_weights = x.matmul(self.router.t()).softmax(n - 1);

        // 2. TopK expert selection → [s, k] (Int)
        let top_k_indices = routing_weights.topk_indexes(TOP_K, n - 1);

        // 3. Gather top-k routing values → [s, k]
        let row_offsets = x
            .graph()
            .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
        let routing_flat_idx = row_offsets + top_k_indices;
        let top_k_values = routing_weights.gather(routing_flat_idx);
        let top_k_values = top_k_values / top_k_values.sum(n - 1).expand_dim(n - 1, TOP_K);

        // 4. Gather gate_up expert weights → [s, k, intermediate*2, H]
        //    Transpose last two dims → [s, k, H, intermediate*2]
        //    Batched matmul: [s,k,1,H] @ [s,k,H,intermediate*2] → [s,k,1,intermediate*2]
        let gate_up_gathered =
            gather_experts(x, top_k_indices, self.gate_up_weights).cast(DType::F32);
        let x_exp = x.expand_dim(n - 1, TOP_K).unsqueeze(n); // [s, k, 1, H]
        let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n); // [s, k, intermediate*2]

        // 5. SwiGLU: silu(gate) * up → [s, k, intermediate]
        let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
        let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
        let hidden = gate.silu() * up;

        // 6. Gather down expert weights → [s, k, H, intermediate]
        //    Transpose last two dims → [s, k, intermediate, H]
        //    Batched matmul: [s,k,1,intermediate] @ [s,k,intermediate,H] → [s,k,1,H]
        let down_gathered = gather_experts(x, top_k_indices, self.down_weights).cast(DType::F32);
        let hidden_exp = hidden.unsqueeze(2); // [s, k, 1, intermediate]
        let down_out = hidden_exp.matmul(down_gathered.transpose(2, 3)).squeeze(2); // [s, k, H]

        // 7. Weighted sum over k experts → [s, H]
        let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [s, k, 1]
        weights_exp.shape.expand(down_out.dims());
        (down_out * weights_exp).sum(n - 1)
    }
}

/// Gather expert weight matrices using topk indices.
/// weights: [E, d1, d2], top_k_indices: [s, k] → result: [s, k, d1, d2]
fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    // Keep expert gather indices in Int all the way through. Routing them through
    // F32 loses exactness once the flat offsets exceed 2^24, which Qwen's expert
    // tensors do at realistic hidden sizes.
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (i, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(i, *dim);
    }
    let expert_flat_idx = exp_base + exp_within;
    weights.gather(expert_flat_idx)
}

/// Per-head RMS normalization for QK-norm.
fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize) -> GraphTensor {
    let seq = x.dims()[0];
    let reshaped = x.split_dims(1, HEAD_DIM);
    let normed = reshaped.std_norm(2, RMS_NORM_EPS);
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    let result = normed * w;
    result.merge_dims(1, 2)
}

fn qwen_rotary_embeddings(
    mut input: GraphTensor,
    pos_ids: GraphTensor,
    n_heads: usize,
) -> GraphTensor {
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // [n_heads, seq, HEAD_DIM]

    let freqs = input
        .graph()
        .arange_options(0, HEAD_DIM, 2)
        .cast(DType::F32)
        / HEAD_DIM as f32;
    let inv_freqs = 1_000_000_f32.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    let cos = emb.cos().expand_dim(0, n_heads);
    let sin = emb.sin().expand_dim(0, n_heads);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

/// Attention with pre-allocated KV cache using scatter.
fn attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    max_seq: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0];
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    let k_new = k_rope.split_dims(1, HEAD_DIM).transpose(0, 1);
    let v_new = v.split_dims(1, HEAD_DIM).transpose(0, 1);

    let h_offset = cx.arange(N_KV_HEADS) * (max_seq * HEAD_DIM);
    let p_offset = (cx.arange(seq) + prev) * HEAD_DIM;
    let d_offset = cx.arange(HEAD_DIM);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, HEAD_DIM)
        + p_offset.expand_dim(0, N_KV_HEADS).expand_dim(2, HEAD_DIM)
        + d_offset.expand_dim(0, N_KV_HEADS).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    let mut k_full = k_cache_out.slice((.., ..total_seq, ..));
    let mut v_full = v_cache_out.slice((.., ..total_seq, ..));
    // LUM-545: model invariant `prev + seq <= max_seq`, but the frontend
    // cannot yet propagate expression-bound assertions, so `slice` reports
    // `min(max_seq, p+s)`. Normalize the visible cache axis to `total_seq`.
    k_full.shape.dims[1] = total_seq;
    v_full.shape.dims[1] = total_seq;

    // GQA expand
    let k_3d = k_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);

    let q = q_rope.split_dims(1, HEAD_DIM).transpose(0, 1);

    let scores = q.matmul(k_3d.transpose(1, 2)) / (HEAD_DIM as f32).sqrt();

    // Causal mask
    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total_seq));
    let mask_3d = mask.cast(DType::F32).expand_dim(0, N_HEADS);
    let masked_scores = scores + mask_3d * (-1e10f32);

    let attn_weights = masked_scores.softmax(2);
    let attn_out = attn_weights.matmul(v_3d);
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}
