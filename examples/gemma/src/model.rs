use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::LayerNorm;

// Gemma 3 4B hyperparams
pub const LAYERS: usize = 34;
pub const HIDDEN: usize = 2560;
pub const INTERMEDIATE: usize = 10240;
pub const HEAD_DIM: usize = 256;
pub const N_HEADS: usize = 8;
pub const N_KV_HEADS: usize = 4;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // = 2
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // = 2048
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // = 1024
pub const VOCAB_SIZE: usize = 262208;
pub const RMS_NORM_EPS: f32 = 1e-6;

// Attention pattern constants
pub const SLIDING_WINDOW_PATTERN: usize = 6;
pub const SLIDING_WINDOW_SIZE: usize = 1024;
pub const ROPE_THETA_GLOBAL: f32 = 1_000_000.0;
pub const ROPE_THETA_LOCAL: f32 = 10_000.0;

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

// Gemma norms: weights are pre-transformed to (1 + weight) in hf.rs, so we use
// standard LayerNorm (RMS mode) which just multiplies by weight.
fn gemma_norm(dim: usize, weight_name: &str, cx: &mut Graph) -> LayerNorm {
    LayerNorm::new(dim, Some(weight_name), None, false, RMS_NORM_EPS, cx)
}

pub struct Gemma {
    embedding: GraphTensor,
    lm_head: GraphTensor,
    layers: Vec<GemmaLayer>,
    lm_norm: LayerNorm,
}

impl Gemma {
    pub fn init(cx: &mut Graph) -> Self {
        Self {
            embedding: persist(cx, "model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            lm_head: persist(cx, "lm_head.weight", (VOCAB_SIZE, HIDDEN)),
            layers: (0..LAYERS).map(|l| GemmaLayer::init(cx, l)).collect(),
            lm_norm: gemma_norm(HIDDEN, "model.norm.weight", cx),
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

struct GemmaLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    q_norm: GraphTensor,
    k_norm: GraphTensor,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    pre_feedforward_layernorm: LayerNorm,
    post_feedforward_layernorm: LayerNorm,
    is_local: bool,
    rope_theta: f32,
    rope_scaling_factor: f32,
}

impl GemmaLayer {
    fn init(cx: &mut Graph, l: usize) -> Self {
        let is_local = !(l + 1).is_multiple_of(SLIDING_WINDOW_PATTERN);
        Self {
            up: layer_weight(cx, l, "mlp.up_proj", (INTERMEDIATE, HIDDEN)),
            gate: layer_weight(cx, l, "mlp.gate_proj", (INTERMEDIATE, HIDDEN)),
            down: layer_weight(cx, l, "mlp.down_proj", (HIDDEN, INTERMEDIATE)),
            q_proj: layer_weight(cx, l, "self_attn.q_proj", (Q_DIM, HIDDEN)),
            k_proj: layer_weight(cx, l, "self_attn.k_proj", (KV_DIM, HIDDEN)),
            v_proj: layer_weight(cx, l, "self_attn.v_proj", (KV_DIM, HIDDEN)),
            o_proj: layer_weight(cx, l, "self_attn.o_proj", (HIDDEN, Q_DIM)),
            q_norm: layer_weight(cx, l, "self_attn.q_norm", HEAD_DIM),
            k_norm: layer_weight(cx, l, "self_attn.k_norm", HEAD_DIM),
            input_layernorm: layer_norm(cx, l, "input_layernorm"),
            post_attention_layernorm: layer_norm(cx, l, "post_attention_layernorm"),
            pre_feedforward_layernorm: layer_norm(cx, l, "pre_feedforward_layernorm"),
            post_feedforward_layernorm: layer_norm(cx, l, "post_feedforward_layernorm"),
            is_local,
            rope_theta: if is_local {
                ROPE_THETA_LOCAL
            } else {
                ROPE_THETA_GLOBAL
            },
            rope_scaling_factor: if is_local { 1.0 } else { 8.0 },
        }
    }

    pub fn forward(
        &self,
        x: GraphTensor,
        pos_ids: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
        max_seq: usize,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let x_attn = self.input_layernorm.forward(x);
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        let q_rope = gemma_rotary_embeddings(
            qk_norm(q, self.q_norm, N_HEADS),
            pos_ids,
            N_HEADS,
            self.rope_theta,
            self.rope_scaling_factor,
        );
        let k_rope = gemma_rotary_embeddings(
            qk_norm(k, self.k_norm, N_KV_HEADS),
            pos_ids,
            N_KV_HEADS,
            self.rope_theta,
            self.rope_scaling_factor,
        );

        let (attn_out, k_cache_out, v_cache_out) = hlir_attention(
            q_rope,
            k_rope,
            v,
            k_cache_in,
            v_cache_in,
            max_seq,
            self.is_local,
        );

        let attn_proj = attn_out.matmul(self.o_proj.t());
        let x = x + self.post_attention_layernorm.forward(attn_proj);

        let x_ff = self.pre_feedforward_layernorm.forward(x);
        let mlp_out = (gemma_gelu(x_ff.matmul(self.gate.t())) * x_ff.matmul(self.up.t()))
            .matmul(self.down.t());
        (
            x + self.post_feedforward_layernorm.forward(mlp_out),
            k_cache_out,
            v_cache_out,
        )
    }
}

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

fn layer_weight(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    persist(cx, format!("model.layers.{layer}.{suffix}.weight"), shape)
}

fn layer_norm(cx: &mut Graph, layer: usize, name: &str) -> LayerNorm {
    gemma_norm(HIDDEN, &format!("model.layers.{layer}.{name}.weight"), cx)
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor) -> GraphTensor {
    let seq = token_ids.dims1();
    embedding.gather(
        (token_ids * HIDDEN).expand_dim(1, HIDDEN)
            + token_ids.graph().arange(HIDDEN).expand_dim(0, seq),
    )
}

/// GELU using the identity: 0.5*x*(1+tanh(a)) = x*sigmoid(2*a)
/// This produces far fewer e-graph nodes than the tanh-based expansion.
#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))
    //         = x * sigmoid(2 * sqrt(2/pi) * x * (1 + 0.044715 * x^2))
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
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

fn gemma_rotary_embeddings(
    input: GraphTensor,
    pos_ids: GraphTensor,
    n_heads: usize,
    rope_theta: f32,
    rope_scaling_factor: f32,
) -> GraphTensor {
    // Input: [seq, dim] where dim = n_heads * HEAD_DIM
    let input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // [n_heads, seq, HEAD_DIM]

    let freqs = input
        .graph()
        .arange_options(0, HEAD_DIM, 2)
        .cast(DType::F32)
        / HEAD_DIM as f32;
    let inv_freqs = rope_theta.pow(freqs).reciprocal();
    // Apply scaling factor to positions
    let scaled_pos = pos_ids.cast(DType::F32) / rope_scaling_factor;
    let emb = scaled_pos
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    // Split-half rotation
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

/// HLIR attention with scatter-based KV cache.
/// For local layers, applies sliding window mask.
fn hlir_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache_in: GraphTensor,
    v_cache_in: GraphTensor,
    max_seq: usize,
    is_local: bool,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0];
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    // Reshape: [seq, kv_dim] -> [N_KV_HEADS, seq, HEAD_DIM]
    let k_new = k_rope.split_dims(1, HEAD_DIM).transpose(0, 1);
    let v_new = v.split_dims(1, HEAD_DIM).transpose(0, 1);

    // Scatter indices
    let h_offset = cx.arange(N_KV_HEADS) * (max_seq * HEAD_DIM);
    let p_offset = (cx.arange(seq) + prev) * HEAD_DIM;
    let d_offset = cx.arange(HEAD_DIM);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, HEAD_DIM)
        + p_offset.expand_dim(0, N_KV_HEADS).expand_dim(2, HEAD_DIM)
        + d_offset.expand_dim(0, N_KV_HEADS).expand_dim(1, seq);

    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    // Slice to valid range
    let mut k_full = k_cache_out.slice((.., ..total_seq, ..));
    let mut v_full = v_cache_out.slice((.., ..total_seq, ..));
    // LUM-545: model invariant `prev + seq <= max_seq`, but the frontend
    // cannot yet propagate expression-bound assertions, so `slice` reports
    // `min(max_seq, p+s)`. Normalize the visible cache axis to `total_seq`.
    k_full.shape.dims[1] = total_seq;
    v_full.shape.dims[1] = total_seq;

    // GQA expand: [N_KV_HEADS, total_seq, HEAD_DIM] -> [N_HEADS, total_seq, HEAD_DIM]
    let k_3d = k_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);

    // Q: [seq, Q_DIM] -> [N_HEADS, seq, HEAD_DIM]
    let q = q_rope.split_dims(1, HEAD_DIM).transpose(0, 1);

    // Attention scores
    let scores = q.matmul(k_3d.transpose(1, 2)) / (HEAD_DIM as f32).sqrt();

    // Causal mask: mask future positions
    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let future_mask = k_pos
        .expand_dim(0, seq)
        .gt(q_abs.expand_dim(1, total_seq))
        .cast(DType::F32);

    let mask_2d = if is_local {
        // Sliding window: also mask positions too far in the past
        // Mask where q_abs - k_pos >= SLIDING_WINDOW_SIZE (i.e., k_pos < q_abs - window + 1)
        let window_start = q_abs - (SLIDING_WINDOW_SIZE - 1) as f32;
        let past_mask = window_start
            .expand_dim(1, total_seq)
            .gt(k_pos.expand_dim(0, seq))
            .cast(DType::F32);
        // Combine: either future or too far past
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
