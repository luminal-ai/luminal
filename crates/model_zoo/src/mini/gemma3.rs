//! MiniGemma3 — the gemma family's mini model (rulings 2026-08-07/10,
//! relocation 2026-08-13): the model DEFINITION lives here in the
//! example crate, including its model-specific decoder layer.

use crate::namespace::Namespace;
use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, causal_bias, paged_attention, rms_norm_heads,
    rotary_apply, sliding_window_bias,
};

pub struct MiniGemma3Layer {
    pub input_norm_weight: Option<GraphTensor>,
    pub input_norm_bias: Option<GraphTensor>,
    pub input_norm_epsilon: f32,
    pub post_attn_norm_weight: Option<GraphTensor>,
    pub post_attn_norm_bias: Option<GraphTensor>,
    pub post_attn_norm_epsilon: f32,
    pub pre_ff_norm_weight: Option<GraphTensor>,
    pub pre_ff_norm_bias: Option<GraphTensor>,
    pub pre_ff_norm_epsilon: f32,
    pub post_ff_norm_weight: Option<GraphTensor>,
    pub post_ff_norm_bias: Option<GraphTensor>,
    pub post_ff_norm_epsilon: f32,
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
    pub gate_weight: GraphTensor,
    pub gate_bias: Option<GraphTensor>,
    pub up_weight: GraphTensor,
    pub up_bias: Option<GraphTensor>,
    pub down_weight: GraphTensor,
    pub down_bias: Option<GraphTensor>,
    pub local: bool,
    pub window: usize,
    pub rope_theta: f32,
    pub pos_scale: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl MiniGemma3Layer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        local: bool,
        window: usize,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |segment: &str, cx: &mut Graph| {
            (
                Some(cx.named_tensor(ns.child(segment).leaf("weight"), vec![d], DType::F32)),
                None,
                1e-6,
            )
        };
        {
            let (input_norm_weight, input_norm_bias, input_norm_epsilon) =
                rms("input_layernorm", cx);
            let (post_attn_norm_weight, post_attn_norm_bias, post_attn_norm_epsilon) =
                rms("post_attention_layernorm", cx);
            let (pre_ff_norm_weight, pre_ff_norm_bias, pre_ff_norm_epsilon) =
                rms("pre_feedforward_layernorm", cx);
            let (post_ff_norm_weight, post_ff_norm_bias, post_ff_norm_epsilon) =
                rms("post_feedforward_layernorm", cx);
            Self {
                input_norm_weight,
                input_norm_bias,
                input_norm_epsilon,
                post_attn_norm_weight,
                post_attn_norm_bias,
                post_attn_norm_epsilon,
                pre_ff_norm_weight,
                pre_ff_norm_bias,
                pre_ff_norm_epsilon,
                post_ff_norm_weight,
                post_ff_norm_bias,
                post_ff_norm_epsilon,
                wq_weight: cx.named_tensor(
                    attn.child("q_proj").leaf("weight"),
                    vec![d, q_dim],
                    DType::F32,
                ),
                wq_bias: None,
                wk_weight: cx.named_tensor(
                    attn.child("k_proj").leaf("weight"),
                    vec![d, kv_dim],
                    DType::F32,
                ),
                wk_bias: None,
                wv_weight: cx.named_tensor(
                    attn.child("v_proj").leaf("weight"),
                    vec![d, kv_dim],
                    DType::F32,
                ),
                wv_bias: None,
                wo_weight: cx.named_tensor(
                    attn.child("o_proj").leaf("weight"),
                    vec![q_dim, d],
                    DType::F32,
                ),
                wo_bias: None,
                q_norm: cx.named_tensor(
                    attn.child("q_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
                k_norm: cx.named_tensor(
                    attn.child("k_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
                gate_weight: cx.named_tensor(
                    mlp.child("gate_proj").leaf("weight"),
                    vec![d, ff],
                    DType::F32,
                ),
                gate_bias: None,
                up_weight: cx.named_tensor(
                    mlp.child("up_proj").leaf("weight"),
                    vec![d, ff],
                    DType::F32,
                ),
                up_bias: None,
                down_weight: cx.named_tensor(
                    mlp.child("down_proj").leaf("weight"),
                    vec![ff, d],
                    DType::F32,
                ),
                down_bias: None,
                local,
                window,
                rope_theta: if local { 10_000.0 } else { 1_000_000.0 },
                pos_scale: if local { 1.0 } else { 1.0 / 8.0 },
                n_heads,
                n_kv_heads,
                head_dim,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = luminal_nn::rms_norm(
            x,
            self.input_norm_weight.map(|weight| weight + 1.0),
            self.input_norm_bias,
            self.input_norm_epsilon,
        );
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q = rotary_apply(
            rms_norm_heads(
                luminal_nn::linear(normed, self.wq_weight, self.wq_bias),
                self.head_dim,
                self.q_norm + 1.0,
                1e-6,
            ) * scale,
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            rms_norm_heads(
                luminal_nn::linear(normed, self.wk_weight, self.wk_bias),
                self.head_dim,
                self.k_norm + 1.0,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let query_positions = q.graph().iota(vec![q.dims()[0]], |c| c[0] + prev_seq);
        let context_positions = q.graph().arange(gather_idx.dims1());
        let score_bias = if self.local {
            sliding_window_bias(query_positions, context_positions, self.window)
        } else {
            causal_bias(query_positions, context_positions)
        };
        let result = paged_attention(
            q,
            k,
            luminal_nn::linear(normed, self.wv_weight, self.wv_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            score_bias,
            AttentionGeometry::with_scale(self.n_heads, self.n_kv_heads, self.head_dim, 1.0),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + luminal_nn::rms_norm(
            luminal_nn::linear(attn, self.wo_weight, self.wo_bias),
            self.post_attn_norm_weight.map(|weight| weight + 1.0),
            self.post_attn_norm_bias,
            self.post_attn_norm_epsilon,
        );
        let ff_in = luminal_nn::rms_norm(
            x,
            self.pre_ff_norm_weight.map(|weight| weight + 1.0),
            self.pre_ff_norm_bias,
            self.pre_ff_norm_epsilon,
        );
        let gated = luminal_nn::linear(ff_in, self.gate_weight, self.gate_bias)
            .gelu_fast_tanh_approximation();
        let ff = luminal_nn::linear(
            gated * luminal_nn::linear(ff_in, self.up_weight, self.up_bias),
            self.down_weight,
            self.down_bias,
        );
        (
            x + luminal_nn::rms_norm(
                ff,
                self.post_ff_norm_weight.map(|weight| weight + 1.0),
                self.post_ff_norm_bias,
                self.post_ff_norm_epsilon,
            ),
            k_cache,
            v_cache,
        )
    }
}

/// MiniGemma3 — the gemma family at FULL ANATOMY (ruling 2026-08-10:
/// minis exercise every architectural construct, shrinking only
/// shapes). Beyond the skeleton: √d EMBEDDING SCALING in-graph with the
/// tied lm_head left UNSCALED; alternating LOCAL (sliding-window,
/// θ=10k) and GLOBAL (full-context, θ=1M, pos·⅛ scaling) layers; and
/// per layer the whole Gemma construct set — sandwich norms,
/// decoupled head_dim, QK-norm, scale-folded-into-Q, in-graph
/// split-half RoPE, GeGLU. `pattern` = every pattern-th layer is
/// global (gemma uses 6; the ratio is a shape, the alternation the
/// construct).
pub struct MiniGemma3 {
    pub embed: GraphTensor,
    pub blocks: Vec<MiniGemma3Layer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
    d: usize,
}

impl MiniGemma3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        layers: usize,
        window: usize,
        pattern: usize,
        cx: &mut Graph,
    ) -> Self {
        {
            let (final_norm_weight, final_norm_bias, final_norm_epsilon) = (
                Some(
                    cx.named_tensor(
                        Namespace::root()
                            .child("model")
                            .child("norm")
                            .leaf("weight"),
                        vec![d],
                        DType::F32,
                    ),
                ),
                None,
                1e-6,
            );
            Self {
                embed: cx.named_tensor(
                    Namespace::root()
                        .child("model")
                        .child("embed_tokens")
                        .leaf("weight"),
                    vec![vocab, d],
                    DType::F32,
                ),
                blocks: (0..layers)
                    .map(|layer| {
                        let local = (layer + 1) % pattern != 0;
                        let layer_ns = Namespace::root()
                            .child("model")
                            .child("layers")
                            .index(layer);
                        MiniGemma3Layer::new(
                            d, ff, n_heads, n_kv_heads, head_dim, local, window, &layer_ns, cx,
                        )
                    })
                    .collect(),
                final_norm_weight,
                final_norm_bias,
                final_norm_epsilon,
                d,
            }
        }
    }

    /// ids (s,) Int, per-layer caches, per-layer rope tables (cos, sin)
    /// — host-built from each block's role theta/pos_scale — plus the
    /// shared split-half pairing matrix → (logits, caches'). Embeddings
    /// scale by √d in-graph; the tied logits head reads the UNSCALED
    /// table (gemma's convention).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
        rope: &[(GraphTensor, GraphTensor)],
        rope_rot: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = luminal_nn::embedding(ids, self.embed) * (self.d as f32).sqrt();
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, kc, vc) = block.forward(
                x,
                caches[layer].0,
                caches[layer].1,
                gather_idx,
                scatter_idx,
                prev_seq,
                rope[layer].0,
                rope[layer].1,
                rope_rot,
            );
            x = next;
            caches_out.push((kc, vc));
        }
        let logits = luminal_nn::embedding_projection(
            luminal_nn::rms_norm(
                x,
                self.final_norm_weight.map(|weight| weight + 1.0),
                self.final_norm_bias,
                self.final_norm_epsilon,
            ),
            self.embed,
        );
        (logits, caches_out)
    }
}
