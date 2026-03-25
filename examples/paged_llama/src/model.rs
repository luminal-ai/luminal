use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::{gather_rows, scatter_rows, LayerNorm};

// Llama 3 8B hyperparams
pub const LAYERS: usize = 32;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 14336;
pub const HEAD_DIM: usize = 128;
pub const KV_GROUPS: usize = 4;
pub const VOCAB_SIZE: usize = 128256;
pub const N_KV_HEADS: usize = HIDDEN / HEAD_DIM / KV_GROUPS; // 8
pub const N_HEADS: usize = HIDDEN / HEAD_DIM; // 32
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 1024

/// Flat 2D paged KV cache: (num_slots, KV_DIM) per layer.
/// Slots are physical positions; the page table maps logical→physical.
pub struct PagedKVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
}

impl PagedKVCache {
    pub fn new(cx: &mut Graph, num_slots: usize) -> Self {
        let mut k_caches = vec![];
        let mut v_caches = vec![];
        for l in 0..LAYERS {
            k_caches.push(cx.named_tensor(format!("kv_cache.{l}.k"), (num_slots, KV_DIM)));
            v_caches.push(cx.named_tensor(format!("kv_cache.{l}.v"), (num_slots, KV_DIM)));
        }
        Self { k_caches, v_caches }
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
        let mut layers = vec![];
        for l in 0..LAYERS {
            layers.push(LlamaLayer {
                up: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.up_proj.weight"),
                        (INTERMEDIATE, HIDDEN),
                    )
                    .persist(),
                gate: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.gate_proj.weight"),
                        (INTERMEDIATE, HIDDEN),
                    )
                    .persist(),
                down: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.down_proj.weight"),
                        (HIDDEN, INTERMEDIATE),
                    )
                    .persist(),
                q_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.q_proj.weight"),
                        (HIDDEN, HIDDEN),
                    )
                    .persist(),
                k_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.k_proj.weight"),
                        (KV_DIM, HIDDEN),
                    )
                    .persist(),
                v_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.v_proj.weight"),
                        (KV_DIM, HIDDEN),
                    )
                    .persist(),
                o_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.o_proj.weight"),
                        (HIDDEN, HIDDEN),
                    )
                    .persist(),
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
        Self {
            embedding: cx
                .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
                .persist(),
            layers,
            lm_head: cx
                .named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN))
                .persist(),
            lm_norm: LayerNorm::new(HIDDEN, Some("model.norm.weight"), None, false, 1e-5, cx),
        }
    }

    /// Forward pass with paged attention.
    ///
    /// - `input`:       (s,) Int — token IDs
    /// - `q_pos`:       (s,) Int — absolute positions for RoPE
    /// - `scatter_idx`: (s,) Int — physical cache slots to write new KV
    /// - `gather_idx`:  (c,) Int — physical cache slots to read for attention context
    /// - `attn_mask`:   (s, c) F32 — precomputed attention mask (0 or -1e10)
    /// - `kv_cache`:    per-layer caches (consumed each step)
    ///
    /// Returns (logits, cache_outputs):
    /// - logits: (s, VOCAB_SIZE)
    /// - cache_outputs: per-layer (k_cache_out, v_cache_out) — the updated caches
    ///   after scatter. Caller must round-trip these back to kv_cache inputs.
    pub fn forward(
        &self,
        input: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        kv_cache: &PagedKVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let seq = input.dims1();
        let mut x = self.embedding.gather(
            (input * HIDDEN).expand_dim(1, HIDDEN)
                + input.graph().arange(HIDDEN).expand_dim(0, seq),
        );
        let mut cache_outputs = vec![];
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                q_pos,
                scatter_idx,
                gather_idx,
                attn_mask,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new.graph_break();
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
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1);

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

    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    let cos = emb.cos().expand_dim(0, x0.dims()[0]);
    let sin = emb.sin().expand_dim(0, x0.dims()[0]);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

/// Paged attention: scatter new KV into flat cache, gather context, compute attention.
///
/// K and V are scattered with RoPE already applied to K. The gather retrieves
/// pre-RoPE'd K values, so no RoPE is needed on the gathered context.
///
/// The attention mask is precomputed on the CPU and handles:
/// - Causal masking within a sequence
/// - Cross-sequence isolation in supersequence batches
#[allow(clippy::too_many_arguments)]
fn paged_attention(
    q_rope: GraphTensor,      // (s, HIDDEN) — RoPE'd queries
    k_rope: GraphTensor,      // (s, KV_DIM) — RoPE'd keys for new tokens
    v: GraphTensor,           // (s, KV_DIM) — values for new tokens
    k_cache: GraphTensor,     // (num_slots, KV_DIM) — consumed key cache
    v_cache: GraphTensor,     // (num_slots, KV_DIM) — consumed value cache
    scatter_idx: GraphTensor, // (s,) Int — slots to write new KV
    gather_idx: GraphTensor,  // (c,) Int — slots to read for attention
    attn_mask: GraphTensor,   // (s, c) F32 — precomputed mask
) -> (GraphTensor, GraphTensor, GraphTensor) {
    // Phase 1: Scatter new KV into cache (in-place with KernelScatterNoCopy)
    // The input cache buffers are consumed; the scatter outputs are the new caches.
    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, KV_DIM);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, KV_DIM);

    // Phase 2: Gather full context from cache
    let k = gather_rows(k_cache_out, gather_idx, KV_DIM); // (c, KV_DIM)
    let v_ctx = gather_rows(v_cache_out, gather_idx, KV_DIM); // (c, KV_DIM)

    // Phase 3: Multi-head reshape
    // Q: (s, HIDDEN) → (N_HEADS, s, HEAD_DIM)
    let q = (q_rope * 1.0).split_dims(1, HEAD_DIM).transpose(0, 1);

    // K: (c, KV_DIM) → (N_KV_HEADS, HEAD_DIM, c)  [transposed for Q@K^T]
    let k = k.split_dims(1, HEAD_DIM).permute((1, 2, 0));

    // V: (c, KV_DIM) → (N_KV_HEADS, c, HEAD_DIM)
    let v_ctx = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1);

    // GQA broadcast: N_KV_HEADS → N_HEADS (materialize after merge for correct strides)
    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0; // (N_HEADS, HEAD_DIM, c)
    let v_ctx = v_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0; // (N_HEADS, c, HEAD_DIM)

    // Phase 4: Attention
    // Scores: (N_HEADS, s, HEAD_DIM) @ (N_HEADS, HEAD_DIM, c) → (N_HEADS, s, c)
    let scores = q.matmul(k) / (HEAD_DIM as f32).sqrt();

    // Apply mask: (s, c) → (N_HEADS, s, c)
    let mask = attn_mask.expand_dim(0, N_HEADS);
    let masked_scores = scores + mask;

    let weights = masked_scores.softmax(2);

    // Output: (N_HEADS, s, c) @ (N_HEADS, c, HEAD_DIM) → (N_HEADS, s, HEAD_DIM)
    let out = weights.matmul(v_ctx);

    // Phase 5: Reshape → (s, HIDDEN)
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);
    (attn_out, k_cache_out, v_cache_out)
}

impl LlamaLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        mut x: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let x_attn = self.attn_rms.forward(x);
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        // Apply RoPE before scattering into cache
        let q_rope = llama_rotary_embeddings(q, q_pos);
        let k_rope = llama_rotary_embeddings(k, q_pos);

        let (attn_out, k_cache_out, v_cache_out) = paged_attention(
            q_rope,
            k_rope,
            v,
            k_cache,
            v_cache,
            scatter_idx,
            gather_idx,
            attn_mask,
        );

        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
        (x + mlp_out, k_cache_out, v_cache_out)
    }
}
