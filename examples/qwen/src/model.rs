use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::LayerNorm;

// Qwen3-4B hyperparams
pub const LAYERS: usize = 36;
pub const HIDDEN: usize = 2560;
pub const INTERMEDIATE: usize = 9728;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // = 4
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // = 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // = 1024
pub const VOCAB_SIZE: usize = 151936;
pub const RMS_NORM_EPS: f32 = 1e-6;

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
            let k = cx
                .named_tensor(format!("kv_cache.{l}.k"), (N_KV_HEADS, max_seq, HEAD_DIM))
                .persist();
            let v = cx
                .named_tensor(format!("kv_cache.{l}.v"), (N_KV_HEADS, max_seq, HEAD_DIM))
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

pub struct Qwen {
    pub embedding: GraphTensor,
    layers: Vec<QwenLayer>,
    lm_norm: LayerNorm,
}

impl Qwen {
    pub fn init(cx: &mut Graph) -> Self {
        let mut w = vec![];
        for l in 0..LAYERS {
            let up = cx
                .named_tensor(
                    format!("model.layers.{l}.mlp.up_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                )
                .persist();
            let gate = cx
                .named_tensor(
                    format!("model.layers.{l}.mlp.gate_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                )
                .persist();
            let down = cx
                .named_tensor(
                    format!("model.layers.{l}.mlp.down_proj.weight"),
                    (HIDDEN, INTERMEDIATE),
                )
                .persist();
            let q_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (Q_DIM, HIDDEN),
                )
                .persist();
            let k_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (KV_DIM, HIDDEN),
                )
                .persist();
            let v_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (KV_DIM, HIDDEN),
                )
                .persist();
            let o_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (HIDDEN, Q_DIM),
                )
                .persist();
            let q_norm = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.q_norm.weight"),
                    HEAD_DIM,
                )
                .persist();
            let k_norm = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.k_norm.weight"),
                    HEAD_DIM,
                )
                .persist();
            w.push(QwenLayer {
                up,
                gate,
                down,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                attn_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
            });
        }
        let lm_norm = LayerNorm::new(
            HIDDEN,
            Some("model.norm.weight"),
            None,
            false,
            RMS_NORM_EPS,
            cx,
        );
        let embedding = cx
            .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
            .persist();
        Self {
            embedding,
            layers: w,
            lm_norm,
        }
    }

    #[tracing::instrument(skip_all)]
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
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                pos_ids,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
                kv_cache.max_seq,
            );
            x = x_new.graph_break();
            cache_outputs.push((k_out, v_out));
        }
        // Tied embeddings: lm_head = embedding.t()
        let logits = self.lm_norm.forward(x).matmul(self.embedding.t());
        (logits, cache_outputs)
    }
}

struct QwenLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    q_norm: GraphTensor,
    k_norm: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
}

/// Per-head RMS normalization for QK-norm.
/// Input: [seq, dim] where dim = n_heads * HEAD_DIM
/// split_dims to [seq, n_heads, HEAD_DIM], RMS norm over last axis, multiply by weight, merge back.
fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize) -> GraphTensor {
    let seq = x.dims()[0];
    // [seq, dim] -> [seq, n_heads, HEAD_DIM]
    let reshaped = x.split_dims(1, HEAD_DIM);
    // RMS norm over last axis (HEAD_DIM)
    let normed = reshaped.std_norm(2, RMS_NORM_EPS);
    // weight is [HEAD_DIM], expand to [seq, n_heads, HEAD_DIM] for broadcast
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    let result = normed * w;
    // Back to [seq, dim]
    result.merge_dims(1, 2)
}

fn qwen_rotary_embeddings(
    mut input: GraphTensor,
    pos_ids: GraphTensor,
    n_heads: usize,
) -> GraphTensor {
    // Input: [seq, dim] where dim = n_heads * HEAD_DIM
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // [n_heads, seq, HEAD_DIM]

    // Get freqs with rope_theta = 1,000,000
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

    // Split into first half and second half ("half" rotation convention)
    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    // Apply sin and cos embeddings
    let cos = emb.cos().expand_dim(0, n_heads);
    let sin = emb.sin().expand_dim(0, n_heads);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    // Combine back: [n_heads, seq, HEAD_DIM] -> [seq, n_heads, HEAD_DIM] -> [seq, dim]
    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

/// HLIR attention with pre-allocated KV cache using scatter.
/// Returns (attn_output, k_cache_updated, v_cache_updated).
fn hlir_attention(
    q_rope: GraphTensor,     // [seq, Q_DIM]
    k_rope: GraphTensor,     // [seq, KV_DIM]
    v: GraphTensor,          // [seq, KV_DIM]
    k_cache_in: GraphTensor, // [N_KV_HEADS, max_seq, HEAD_DIM]
    v_cache_in: GraphTensor, // [N_KV_HEADS, max_seq, HEAD_DIM]
    max_seq: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();
    let seq = q_rope.dims()[0]; // Expression 's'
    let prev = Expression::from('p');
    let total_seq = prev + seq;

    // Reshape new K, V: [seq, kv_dim] -> [N_KV_HEADS, seq, HEAD_DIM]
    let k_new = k_rope.split_dims(1, HEAD_DIM).transpose(0, 1);
    let v_new = v.split_dims(1, HEAD_DIM).transpose(0, 1);

    // Build flat scatter indices for cache positions [prev..prev+seq]
    // Cache layout: [N_KV_HEADS, max_seq, HEAD_DIM], flat index = h*max_seq*HEAD_DIM + (prev+s)*HEAD_DIM + d
    let h_offset = cx.arange(N_KV_HEADS) * (max_seq * HEAD_DIM);
    let p_offset = (cx.arange(seq) + prev) * HEAD_DIM;
    let d_offset = cx.arange(HEAD_DIM);
    let scatter_idx = h_offset.expand_dim(1, seq).expand_dim(2, HEAD_DIM)
        + p_offset.expand_dim(0, N_KV_HEADS).expand_dim(2, HEAD_DIM)
        + d_offset.expand_dim(0, N_KV_HEADS).expand_dim(1, seq);

    // Scatter new K/V into cache
    let k_cache_out = k_new.scatter(scatter_idx, k_cache_in);
    let v_cache_out = v_new.scatter(scatter_idx, v_cache_in);

    // Slice to valid range: [N_KV_HEADS, total_seq, HEAD_DIM]
    let k_full = k_cache_out.slice((.., ..total_seq, ..));
    let v_full = v_cache_out.slice((.., ..total_seq, ..));

    // GQA expand: [N_KV_HEADS, total_seq, HEAD_DIM] -> [N_HEADS, total_seq, HEAD_DIM]
    let k_3d = k_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);
    let v_3d = v_full.expand_dim(1, KV_GROUPS).merge_dims(0, 1);

    // Q: [seq, Q_DIM] -> [N_HEADS, seq, HEAD_DIM]
    let q = q_rope.split_dims(1, HEAD_DIM).transpose(0, 1);

    // Attention scores: Q @ K^T / sqrt(d)
    let scores = q.matmul(k_3d.transpose(1, 2)) / (HEAD_DIM as f32).sqrt();

    // Causal mask: mask positions where k_pos > prev + q_local_pos
    let q_abs = cx.arange(seq).cast(DType::F32) + prev;
    let k_pos = cx.arange(total_seq).cast(DType::F32);
    let mask = k_pos.expand_dim(0, seq).gt(q_abs.expand_dim(1, total_seq));
    let mask_3d = mask.cast(DType::F32).expand_dim(0, N_HEADS);
    let masked_scores = scores + mask_3d * (-1e10f32);

    // Softmax along key dimension
    let attn_weights = masked_scores.softmax(2);

    // Weighted sum: [N_HEADS, seq, total_seq] x [N_HEADS, total_seq, HEAD_DIM] -> [N_HEADS, seq, HEAD_DIM]
    let attn_out = attn_weights.matmul(v_3d);

    // Reshape: [N_HEADS, seq, HEAD_DIM] -> [seq, N_HEADS, HEAD_DIM] -> [seq, Q_DIM]
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}

impl QwenLayer {
    pub fn forward(
        &self,
        mut x: GraphTensor,
        pos_ids: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
        max_seq: usize,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
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
            hlir_attention(q_rope, k_rope, v, k_cache_in, v_cache_in, max_seq);
        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
        (x + mlp_out, k_cache_out, v_cache_out)
    }
}
