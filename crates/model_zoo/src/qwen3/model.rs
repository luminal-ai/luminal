//! Qwen3 as PURE LOGICAL OPS (conversion directive 2026-08-10; this
//! crate is the zoo's exemplar): the model is authored entirely through
//! luminal_nn constructs on the native recorder — no residency markers,
//! no HLIR and no backend types. Weight-ness is not authored (a weight is
//! an ordinary named input; storage residency is runtime-binding business).
//! `init_with_parameter_dtype` can keep those inputs in checkpoint-native storage while
//! explicit casts preserve this model's FP32 arithmetic semantics.
//!
//! RoPE is the concat-free pairing-matrix spelling ([`luminal_nn::rotary_apply`])
//! with host-precomputed tables — the rejoin-divergence workaround — and
//! the KV cache is the scatter/gather paged form, so no concat-of-slices
//! road ever forms in the recorded graph.

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, KvCache, LayerNorm, Linear, Namespace, causal_bias,
    paged_attention, rms_norm_heads, rotary_apply,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

/// Architecture hyperparameters. `qwen3_4b` is the real model;
/// `tiny` keeps the identical anatomy at smoke-test scale.
#[derive(Clone)]
pub struct QwenDims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl QwenDims {
    /// Qwen/Qwen3-4B. Note head_dim (128) is DECOUPLED from hidden:
    /// q_dim = 32·128 = 4096 ≠ hidden = 2560.
    pub fn qwen3_4b() -> Self {
        Self {
            vocab: 151_936,
            hidden: 2560,
            intermediate: 9728,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 36,
            rope_theta: 1_000_000.0,
            rms_eps: 1e-6,
        }
    }

    /// Same anatomy (GQA, decoupled head_dim, QK-norm, SwiGLU, tied
    /// head) at scalar-test scale.
    pub fn tiny() -> Self {
        Self {
            vocab: 31,
            hidden: 16,
            intermediate: 24,
            head_dim: 8,
            n_heads: 2,
            n_kv_heads: 1,
            layers: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-6,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

pub struct QwenLayer {
    pub attn_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub ffn_norm: LayerNorm,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl QwenLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward_rope(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let q = rotary_apply(
            rms_norm_heads(self.wq.forward(normed), self.head_dim, self.q_norm, 1e-6),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            rms_norm_heads(self.wk.forward(normed), self.head_dim, self.k_norm, 1e-6),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let context_positions = q.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            q,
            k,
            self.wv.forward(normed),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, context_positions),
            AttentionGeometry::new(self.n_heads, self.n_kv_heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + self.wo.forward(attn);
        let ff_in = self.ffn_norm.forward(x);
        let ff = self
            .down
            .forward(self.gate.forward(ff_in).silu() * self.up.forward(ff_in));
        (x + ff, k_cache, v_cache)
    }
}

/// The model: an embedding (tied to the lm head), a stack of
/// rope-threaded Qwen layers with QK-norm, and the final RMS norm.
/// Every parameter is a named input tensor whose LABEL is its HF
/// checkpoint key — the loader walks `input_specs()` and matches labels
/// against the checkpoint (label-driven staging, ruling 2026-08-13).
/// Qwen3-4B ties lm_head to the embedding, so no lm_head.weight input
/// exists.
pub struct Qwen {
    pub dims: QwenDims,
    pub embed: Embedding,
    pub blocks: Vec<QwenLayer>,
    pub final_norm: LayerNorm,
}

impl Qwen {
    pub fn init(cx: &mut Graph, dims: &QwenDims) -> Self {
        Self::init_with_parameter_dtype(cx, dims, DType::F32)
    }

    pub fn init_with_parameter_dtype(cx: &mut Graph, dims: &QwenDims, dtype: DType) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Self::block(l, dims, dtype, cx))
            .collect();
        Self {
            dims: dims.clone(),
            // HF stores embed_tokens as (vocab, hidden) — the natural
            // Embedding orientation; `reverse` is the tied lm head.
            embed: Embedding::new_with_storage_dtype(
                dims.vocab,
                dims.hidden,
                dtype,
                &Namespace::root().child("model").child("embed_tokens"),
                cx,
            ),
            blocks,
            final_norm: LayerNorm::new_with_storage_dtype(
                dims.hidden,
                true,
                false,
                false,
                dims.rms_eps,
                dtype,
                &Namespace::root().child("model").child("norm"),
                cx,
            ),
        }
    }

    /// Qwen3 uses decoupled head dimensions and learned norm weights.
    /// `Linear::new` records Luminal's canonical
    /// (in, out) orientation; checkpoint staging transposes HF weights.
    fn block(l: usize, d: &QwenDims, dtype: DType, cx: &mut Graph) -> QwenLayer {
        let ns = Namespace::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        QwenLayer {
            attn_norm: LayerNorm::new_with_storage_dtype(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                dtype,
                &ns.child("input_layernorm"),
                cx,
            ),
            wq: Linear::new_with_storage_dtype(
                d.hidden,
                d.q_dim(),
                false,
                dtype,
                &attn.child("q_proj"),
                cx,
            ),
            wk: Linear::new_with_storage_dtype(
                d.hidden,
                d.kv_dim(),
                false,
                dtype,
                &attn.child("k_proj"),
                cx,
            ),
            wv: Linear::new_with_storage_dtype(
                d.hidden,
                d.kv_dim(),
                false,
                dtype,
                &attn.child("v_proj"),
                cx,
            ),
            wo: Linear::new_with_storage_dtype(
                d.q_dim(),
                d.hidden,
                false,
                dtype,
                &attn.child("o_proj"),
                cx,
            ),
            q_norm: cx
                .named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim, dtype)
                .cast(DType::F32),
            k_norm: cx
                .named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim, dtype)
                .cast(DType::F32),
            ffn_norm: LayerNorm::new_with_storage_dtype(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                dtype,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
            gate: Linear::new_with_storage_dtype(
                d.hidden,
                d.intermediate,
                false,
                dtype,
                &mlp.child("gate_proj"),
                cx,
            ),
            up: Linear::new_with_storage_dtype(
                d.hidden,
                d.intermediate,
                false,
                dtype,
                &mlp.child("up_proj"),
                cx,
            ),
            down: Linear::new_with_storage_dtype(
                d.intermediate,
                d.hidden,
                false,
                dtype,
                &mlp.child("down_proj"),
                cx,
            ),
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
        }
    }

    /// One decode step over the paged cache. `tokens`/`q_pos` are (s,)
    /// Int; rope tables (s, head_dim) and the pairing matrix are
    /// host-built inputs; caches are one (slots, kv_dim) pair per layer.
    /// Returns (logits (s, vocab), per-layer cache outs).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        assert_eq!(caches.len(), self.blocks.len());
        let mut x = self.embed.forward(tokens);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope(
                x,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                q_pos,
                rope_cos,
                rope_sin,
                rope_rot,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
