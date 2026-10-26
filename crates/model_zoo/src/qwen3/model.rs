//! Qwen3 as PURE LOGICAL OPS (conversion directive 2026-08-10; this
//! crate is the zoo's exemplar): the model is authored entirely through
//! luminal_nn constructs on the native recorder — no residency markers,
//! no HLIR and no backend types. Weight-ness is not authored (a weight is
//! an ordinary named input; storage residency is runtime-binding business).
//! Weight ports declare FP32 arithmetic. Applications can attach narrower
//! checkpoint sources with `Graph::input_source`, which records explicit casts.
//!
//! RoPE is the concat-free pairing-matrix spelling ([`luminal_nn::rotary_apply`])
//! with host-precomputed tables — the rejoin-divergence workaround — and
//! the KV cache is the scatter/gather paged form, so no concat-of-slices
//! road ever forms in the recorded graph.

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, causal_bias, paged_attention, rms_norm_heads,
    rotary_apply,
};

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
    pub attn_norm_weight: Option<GraphTensor>,
    pub attn_norm_bias: Option<GraphTensor>,
    pub attn_norm_epsilon: f32,
    pub wq_weight: GraphTensor,
    pub wq_bias: Option<GraphTensor>,
    pub wk_weight: GraphTensor,
    pub wk_bias: Option<GraphTensor>,
    pub wv_weight: GraphTensor,
    pub wv_bias: Option<GraphTensor>,
    pub wo_weight: GraphTensor,
    pub wo_bias: Option<GraphTensor>,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub ffn_norm_weight: Option<GraphTensor>,
    pub ffn_norm_bias: Option<GraphTensor>,
    pub ffn_norm_epsilon: f32,
    pub gate_weight: GraphTensor,
    pub gate_bias: Option<GraphTensor>,
    pub up_weight: GraphTensor,
    pub up_bias: Option<GraphTensor>,
    pub down_weight: GraphTensor,
    pub down_bias: Option<GraphTensor>,
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
        let normed = luminal_nn::rms_norm(
            x,
            self.attn_norm_weight,
            self.attn_norm_bias,
            self.attn_norm_epsilon,
        );
        let q = rotary_apply(
            rms_norm_heads(
                luminal_nn::linear(normed, self.wq_weight, self.wq_bias),
                self.head_dim,
                self.q_norm,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            rms_norm_heads(
                luminal_nn::linear(normed, self.wk_weight, self.wk_bias),
                self.head_dim,
                self.k_norm,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let context_positions = q.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            q,
            k,
            luminal_nn::linear(normed, self.wv_weight, self.wv_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, context_positions),
            AttentionGeometry::new(self.n_heads, self.n_kv_heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + luminal_nn::linear(attn, self.wo_weight, self.wo_bias);
        let ff_in = luminal_nn::rms_norm(
            x,
            self.ffn_norm_weight,
            self.ffn_norm_bias,
            self.ffn_norm_epsilon,
        );
        let ff = luminal_nn::linear(
            luminal_nn::linear(ff_in, self.gate_weight, self.gate_bias).silu()
                * luminal_nn::linear(ff_in, self.up_weight, self.up_bias),
            self.down_weight,
            self.down_bias,
        );
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
    pub embed: GraphTensor,
    pub blocks: Vec<QwenLayer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
}

impl Qwen {
    pub fn init(cx: &mut Graph, dims: &QwenDims) -> Self {
        let dtype = DType::F32;
        let blocks = (0..dims.layers)
            .map(|l| Self::block(l, dims, dtype, cx))
            .collect();
        Self {
            dims: dims.clone(),
            // HF stores embed_tokens as (vocab, hidden) — the natural
            // Embedding orientation; embedding_projection is the tied lm head.
            embed: cx.named_tensor(
                Namespace::root()
                    .child("model")
                    .child("embed_tokens")
                    .leaf("weight"),
                vec![dims.vocab, dims.hidden],
                dtype,
            ),
            blocks,
            final_norm_weight: Some(
                cx.named_tensor(
                    Namespace::root()
                        .child("model")
                        .child("norm")
                        .leaf("weight"),
                    vec![dims.hidden],
                    dtype,
                ),
            ),
            final_norm_bias: None,
            final_norm_epsilon: dims.rms_eps,
        }
    }

    /// Qwen3 uses decoupled head dimensions and learned norm weights.
    /// Weight ports use Luminal's canonical (in, out) orientation;
    /// checkpoint staging transposes HF weights.
    fn block(l: usize, d: &QwenDims, dtype: DType, cx: &mut Graph) -> QwenLayer {
        let ns = Namespace::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        QwenLayer {
            attn_norm_weight: Some(cx.named_tensor(
                ns.child("input_layernorm").leaf("weight"),
                vec![d.hidden],
                dtype,
            )),
            attn_norm_bias: None,
            attn_norm_epsilon: d.rms_eps,
            wq_weight: cx.named_tensor(
                attn.child("q_proj").leaf("weight"),
                vec![d.hidden, d.q_dim()],
                dtype,
            ),
            wq_bias: None,
            wk_weight: cx.named_tensor(
                attn.child("k_proj").leaf("weight"),
                vec![d.hidden, d.kv_dim()],
                dtype,
            ),
            wk_bias: None,
            wv_weight: cx.named_tensor(
                attn.child("v_proj").leaf("weight"),
                vec![d.hidden, d.kv_dim()],
                dtype,
            ),
            wv_bias: None,
            wo_weight: cx.named_tensor(
                attn.child("o_proj").leaf("weight"),
                vec![d.q_dim(), d.hidden],
                dtype,
            ),
            wo_bias: None,
            q_norm: cx
                .named_tensor(attn.child("q_norm").leaf("weight"), vec![d.head_dim], dtype)
                .cast(DType::F32),
            k_norm: cx
                .named_tensor(attn.child("k_norm").leaf("weight"), vec![d.head_dim], dtype)
                .cast(DType::F32),
            ffn_norm_weight: Some(cx.named_tensor(
                ns.child("post_attention_layernorm").leaf("weight"),
                vec![d.hidden],
                dtype,
            )),
            ffn_norm_bias: None,
            ffn_norm_epsilon: d.rms_eps,
            gate_weight: cx.named_tensor(
                mlp.child("gate_proj").leaf("weight"),
                vec![d.hidden, d.intermediate],
                dtype,
            ),
            gate_bias: None,
            up_weight: cx.named_tensor(
                mlp.child("up_proj").leaf("weight"),
                vec![d.hidden, d.intermediate],
                dtype,
            ),
            up_bias: None,
            down_weight: cx.named_tensor(
                mlp.child("down_proj").leaf("weight"),
                vec![d.intermediate, d.hidden],
                dtype,
            ),
            down_bias: None,
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
        let mut x = luminal_nn::embedding(tokens, self.embed);
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
        let logits = luminal_nn::embedding_projection(
            luminal_nn::rms_norm(
                x,
                self.final_norm_weight,
                self.final_norm_bias,
                self.final_norm_epsilon,
            ),
            self.embed,
        );
        (logits, caches_out)
    }
}
