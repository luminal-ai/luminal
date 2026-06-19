use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
};
use luminal_nn::{LayerNorm, gather_rows, scatter_rows};

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
/// query_pre_attn_scalar = head_dim = 256, so the score scale is 1/sqrt(256).
pub const ATTN_SCALE: f32 = 1.0 / 16.0;

/// Row-paged bf16 cache pool byte size for one layer.
pub fn cache_bytes(max_seq: usize) -> usize {
    max_seq * KV_DIM * 2
}

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
}

impl KVCache {
    pub fn new(cx: &mut Graph, max_seq: usize) -> Self {
        let mut k_caches = Vec::with_capacity(LAYERS);
        let mut v_caches = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            // Row-paged layout (num_slots, kv_dim) in bf16 — the spelling the
            // FlashInfer attention rewrites match.
            k_caches.push(
                persist(cx, format!("kv_cache.{l}.k"), (max_seq, KV_DIM)).as_dtype(DType::Bf16),
            );
            v_caches.push(
                persist(cx, format!("kv_cache.{l}.v"), (max_seq, KV_DIM)).as_dtype(DType::Bf16),
            );
        }
        Self { k_caches, v_caches }
    }
}

// Gemma norms: weights are pre-transformed to (1 + weight) in hf.rs, so we use
// standard LayerNorm (RMS mode) which just multiplies by weight.
fn gemma_norm(dim: usize, weight_name: &str, cx: &mut Graph) -> LayerNorm {
    LayerNorm::new(dim, Some(weight_name), None, false, RMS_NORM_EPS, cx)
}

/// RMS norm computed in F32 with explicit casts when the input is 16-bit. The
/// KernelRMSNorm egglog rule re-fuses the F32 cast → std_norm → cast chain.
fn norm_in_f32(norm: &LayerNorm, x: GraphTensor) -> GraphTensor {
    if x.dtype == DType::F32 {
        norm.forward(x)
    } else {
        norm.forward(x.cast(DType::F32)).cast(x.dtype)
    }
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
            // bf16 table rows feed the layer stack directly; norms stay F32.
            embedding: persist(cx, "model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
                .as_dtype(DType::Bf16),
            lm_head: persist(cx, "lm_head.weight", (VOCAB_SIZE, HIDDEN)).as_dtype(DType::Bf16),
            layers: (0..LAYERS).map(|l| GemmaLayer::init(cx, l)).collect(),
            lm_norm: gemma_norm(HIDDEN, "model.norm.weight", cx),
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
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                pos_ids,
                scatter_idx,
                gather_idx,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        // Final norm in F32, head matmul in bf16, then cast logits to F32 so
        // host-side get_f32 / argmax sampling read F32.
        let logits = norm_in_f32(&self.lm_norm, x)
            .matmul(self.lm_head.t())
            .cast(DType::F32);
        (logits, cache_outputs)
    }

    /// Forward + on-device greedy sampling: repetition penalty against a
    /// persistent seen-token mask, then argmax. `new_token` (the previously
    /// sampled id, -1 for none) is inserted into the mask BEFORE the penalty
    /// read. Per-step host I/O is one token id each way.
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

        // seen[new_token] = 1.0 (in place; no-op for -1).
        let one = cx.constant_float(1.0).expand_dim(0, 1);
        let seen_out = one.scatter(new_token, seen_mask);

        // CPU-equivalent penalty: seen & logit > 0 → /p, seen & logit <= 0 → *p.
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
            up: layer_weight(cx, l, "mlp.up_proj", (INTERMEDIATE, HIDDEN)).as_dtype(DType::Bf16),
            gate: layer_weight(cx, l, "mlp.gate_proj", (INTERMEDIATE, HIDDEN))
                .as_dtype(DType::Bf16),
            down: layer_weight(cx, l, "mlp.down_proj", (HIDDEN, INTERMEDIATE))
                .as_dtype(DType::Bf16),
            q_proj: layer_weight(cx, l, "self_attn.q_proj", (Q_DIM, HIDDEN)).as_dtype(DType::Bf16),
            k_proj: layer_weight(cx, l, "self_attn.k_proj", (KV_DIM, HIDDEN)).as_dtype(DType::Bf16),
            v_proj: layer_weight(cx, l, "self_attn.v_proj", (KV_DIM, HIDDEN)).as_dtype(DType::Bf16),
            o_proj: layer_weight(cx, l, "self_attn.o_proj", (HIDDEN, Q_DIM)).as_dtype(DType::Bf16),
            // q/k norms stay F32 (applied inside the F32 norm sandwich).
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
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        k_cache_in: GraphTensor,
        v_cache_in: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let x_attn = norm_in_f32(&self.input_layernorm, x);
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

        let (attn_out, k_cache_out, v_cache_out) = paged_attention(
            q_rope,
            k_rope,
            v,
            k_cache_in,
            v_cache_in,
            scatter_idx,
            gather_idx,
            pos_ids,
            self.is_local,
        );

        let attn_proj = attn_out.matmul(self.o_proj.t());
        let x = x + norm_in_f32(&self.post_attention_layernorm, attn_proj);

        let x_ff = norm_in_f32(&self.pre_feedforward_layernorm, x);
        let mlp_out = gemma_mlp(x_ff, self.gate, self.up, self.down);
        let mlp_out = norm_in_f32(&self.post_feedforward_layernorm, mlp_out);
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

/// GELU using the identity 0.5*x*(1+tanh(a)) = x*sigmoid(2*a) — fewer e-graph
/// nodes than the tanh expansion.
#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

/// GeGLU MLP. Matmuls in bf16; the gelu + elementwise mul run in F32 per the
/// dtype contract (bf16 division/activation is ambiguous), then cast back.
fn gemma_mlp(x: GraphTensor, gate: GraphTensor, up: GraphTensor, down: GraphTensor) -> GraphTensor {
    let act = x.dtype;
    let gate_out = x.matmul(gate.t());
    let up_out = x.matmul(up.t());
    let glu = if act == DType::F32 {
        gemma_gelu(gate_out) * up_out
    } else {
        (gemma_gelu(gate_out.cast(DType::F32)) * up_out.cast(DType::F32)).cast(act)
    };
    glu.matmul(down.t())
}

/// Per-head RMS normalization for QK-norm, computed in F32.
fn qk_norm(x: GraphTensor, weight: GraphTensor, n_heads: usize) -> GraphTensor {
    let dtype = x.dtype;
    let seq = x.dims()[0];
    let x = if dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let reshaped = x.split_dims(1, HEAD_DIM);
    let normed = reshaped.std_norm(2, RMS_NORM_EPS);
    let w = weight.expand_dim(0, n_heads).expand_dim(0, seq);
    let result = (normed * w).merge_dims(1, 2);
    if dtype == DType::F32 {
        result
    } else {
        result.cast(dtype)
    }
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
    // Apply linear rope scaling to positions (global layers only).
    let scaled_pos = pos_ids.cast(DType::F32) / rope_scaling_factor;
    let emb = scaled_pos
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    // Split-half rotation.
    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

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

/// Paged attention in the llama/qwen/gemma4 HLIR spelling (scatter_rows into 2D
/// bf16 caches, gather via a flat row index, GQA broadcast, triu-gather causal
/// mask) — the FlashInfer rewrites match this pattern; the HLIR chain remains
/// the fallback candidate. Gemma 3 scales scores by `query_pre_attn_scalar`
/// (= 1/sqrt(head_dim)) and the score scale is folded into Q so the QK Sum is
/// scale-free (the gemma FlashInfer rules match scale-free scores). Sliding
/// layers add a window term to the mask: positions older than `q_pos - (W-1)`
/// are blocked.
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
    is_local: bool,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let cx = q_rope.graph();

    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, KV_DIM);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, KV_DIM);

    let k = gather_rows(k_cache_out, gather_idx, KV_DIM);
    let v_ctx = gather_rows(v_cache_out, gather_idx, KV_DIM);

    // Fold the attention scale into Q so the QK Sum is scale-free — matches the
    // gemma scale-free FlashInfer rules (which have a sliding-window variant).
    let q = (q_rope * ATTN_SCALE)
        .split_dims(1, HEAD_DIM)
        .transpose(0, 1);
    let k = k.split_dims(1, HEAD_DIM).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1);

    // GQA expand: `* 1.0` forces contiguous materialization (uniform strides
    // for batched cuBLAS / FlashInfer).
    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;

    // Scale-free scores (the 1/sqrt(head_dim) scale is folded into Q above).
    let scores = q.matmul(k);
    let ctx = Expression::from('c');
    let seq = q_rope.dims()[0];
    let causal_square = scores.graph().triu(ctx, 1).cast(scores.dtype) * -1e10;
    let row_offsets = (q_pos * ctx).expand_dim(1, ctx);
    let col_offsets = scores.graph().arange(ctx).expand_dim(0, seq);
    let attn_mask = causal_square.gather(row_offsets + col_offsets);

    let attn_mask = if is_local {
        // Sliding window: block kv positions older than q_pos - (W-1).
        let q_f = q_pos.cast(DType::F32);
        let win_lo = q_f - (SLIDING_WINDOW_SIZE - 1) as f32;
        let col_f = cx.arange(ctx).cast(DType::F32);
        let too_old = col_f.expand_dim(0, seq).lt(win_lo.expand_dim(1, ctx));
        attn_mask + too_old.cast(scores.dtype) * -1e10
    } else {
        attn_mask
    };

    let masked_scores = scores + attn_mask.expand_dim(0, N_HEADS);
    let attn_weights = masked_scores.softmax(2);
    let attn_out = attn_weights.matmul(v_ctx);
    let out = attn_out.transpose(0, 1).merge_dims(1, 2);

    (out, k_cache_out, v_cache_out)
}
