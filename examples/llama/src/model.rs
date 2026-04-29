use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::LayerNorm;

// Llama 3 8B hyperparams
pub const LAYERS: usize = 32;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 14336;
pub const HEAD_DIM: usize = 128;
pub const KV_GROUPS: usize = 4;
pub const VOCAB_SIZE: usize = 128256;
pub const N_KV_HEADS: usize = HIDDEN / HEAD_DIM / KV_GROUPS; // 8
pub const N_HEADS: usize = HIDDEN / HEAD_DIM; // 32

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

pub struct Llama {
    embedding: GraphTensor,
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Llama {
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
                    (HIDDEN, HIDDEN),
                )
                .persist();
            let k_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                )
                .persist();
            let v_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (HIDDEN / KV_GROUPS, HIDDEN),
                )
                .persist();
            let o_proj = cx
                .named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (HIDDEN, HIDDEN),
                )
                .persist();
            w.push(LlamaLayer {
                up,
                gate,
                down,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                attn_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
            });
        }
        let lm_norm = LayerNorm::new(HIDDEN, Some("model.norm.weight"), None, false, 1e-5, cx);
        let lm_head = cx
            .named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN))
            .persist();
        let embedding = cx
            .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
            .persist();
        Self {
            embedding,
            layers: w,
            lm_head,
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
            x = x_new;
            //x = x_new.graph_break();
            cache_outputs.push((k_out, v_out));
        }
        let logits = self.lm_norm.forward(x).matmul(self.lm_head.t());
        (logits, cache_outputs)
    }
}

struct LlamaLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
}

fn llama_rotary_embeddings(mut input: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
    // Input: [seq, dim]
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1); // n_heads, seq, head_dim

    // Get freqs
    let freqs = input
        .graph()
        .arange_options(0, HEAD_DIM, 2)
        .cast(DType::F32)
        / HEAD_DIM as f32;
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    // Split into first half and second half (Llama "half" rotation convention)
    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    // Apply sin and cos embeddings
    let cos = emb.cos().expand_dim(0, x0.dims()[0]);
    let sin = emb.sin().expand_dim(0, x0.dims()[0]);
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
    q_rope: GraphTensor,     // [seq, HIDDEN]
    k_rope: GraphTensor,     // [seq, HIDDEN/KV_GROUPS]
    v: GraphTensor,          // [seq, HIDDEN/KV_GROUPS]
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

    // Q: [seq, HIDDEN] -> [N_HEADS, seq, HEAD_DIM]
    let q = q_rope.split_dims(1, HEAD_DIM).transpose(0, 1);

    // Attention scores: Q @ K^T / sqrt(d)
    // 3D matmul: [N_HEADS, seq, HEAD_DIM] x [N_HEADS, HEAD_DIM, total_seq] -> [N_HEADS, seq, total_seq]
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

    // Reshape: [N_HEADS, seq, HEAD_DIM] -> [seq, N_HEADS, HEAD_DIM] -> [seq, HIDDEN]
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}

impl LlamaLayer {
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
        let q_rope = llama_rotary_embeddings(q, pos_ids);
        let k_rope = llama_rotary_embeddings(k, pos_ids);
        let (attn_out, k_cache_out, v_cache_out) =
            hlir_attention(q_rope, k_rope, v, k_cache_in, v_cache_in, max_seq);
        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
        (x + mlp_out, k_cache_out, v_cache_out)
    }
}
