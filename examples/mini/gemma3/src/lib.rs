//! MiniGemma3 — the gemma family's mini model (rulings 2026-08-07/10,
//! relocation 2026-08-13): the model DEFINITION lives here in the
//! example crate, including its model-specific decoder layer.

#[path = "../../../common/model_support.rs"]
mod model_support;

use crate::model_support::{
    causal_bias, paged_attention, rms_norm_heads, rotary_apply, sliding_window_bias,
    AttentionGeometry, CacheAccess, Embedding, KvCache, LayerNorm, Linear, Namespace,
};
use luminal::prelude::*;
use luminal::shape::IntExpr;

pub struct MiniGemma3Layer {
    pub input_norm: LayerNorm,
    pub post_attn_norm: LayerNorm,
    pub pre_ff_norm: LayerNorm,
    pub post_ff_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
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
            LayerNorm::new(
                d,
                true,
                false,
                false,
                1e-6,
                DType::F32,
                &ns.child(segment),
                cx,
            )
            .with_unit_offset()
        };
        Self {
            input_norm: rms("input_layernorm", cx),
            post_attn_norm: rms("post_attention_layernorm", cx),
            pre_ff_norm: rms("pre_feedforward_layernorm", cx),
            post_ff_norm: rms("post_feedforward_layernorm", cx),
            wq: Linear::new(d, q_dim, false, DType::F32, &attn.child("q_proj"), cx),
            wk: Linear::new(d, kv_dim, false, DType::F32, &attn.child("k_proj"), cx),
            wv: Linear::new(d, kv_dim, false, DType::F32, &attn.child("v_proj"), cx),
            wo: Linear::new(q_dim, d, false, DType::F32, &attn.child("o_proj"), cx),
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), head_dim, DType::F32),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), head_dim, DType::F32),
            gate: Linear::new(d, ff, false, DType::F32, &mlp.child("gate_proj"), cx),
            up: Linear::new(d, ff, false, DType::F32, &mlp.child("up_proj"), cx),
            down: Linear::new(ff, d, false, DType::F32, &mlp.child("down_proj"), cx),
            local,
            window,
            rope_theta: if local { 10_000.0 } else { 1_000_000.0 },
            pos_scale: if local { 1.0 } else { 1.0 / 8.0 },
            n_heads,
            n_kv_heads,
            head_dim,
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
        let normed = self.input_norm.forward(x);
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q = rotary_apply(
            rms_norm_heads(
                self.wq.forward(normed),
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
                self.wk.forward(normed),
                self.head_dim,
                self.k_norm + 1.0,
                1e-6,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let query_positions = q.graph().iota(q.dims()[0], |c| c[0] + prev_seq);
        let context_positions = q.graph().arange(gather_idx.dims1());
        let score_bias = if self.local {
            sliding_window_bias(query_positions, context_positions, self.window)
        } else {
            causal_bias(query_positions, context_positions)
        };
        let result = paged_attention(
            q,
            k,
            self.wv.forward(normed),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            score_bias,
            AttentionGeometry::with_scale(self.n_heads, self.n_kv_heads, self.head_dim, 1.0),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + self.post_attn_norm.forward(self.wo.forward(attn));
        let ff_in = self.pre_ff_norm.forward(x);
        let gated = self.gate.forward(ff_in).gelu_fast_tanh_approximation();
        let ff = self.down.forward(gated * self.up.forward(ff_in));
        (x + self.post_ff_norm.forward(ff), k_cache, v_cache)
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
    pub embed: Embedding,
    pub blocks: Vec<MiniGemma3Layer>,
    pub final_norm: LayerNorm,
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
        Self {
            embed: Embedding::new(
                vocab,
                d,
                DType::F32,
                &Namespace::root().child("model").child("embed_tokens"),
                cx,
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
            final_norm: LayerNorm::new(
                d,
                true,
                false,
                false,
                1e-6,
                DType::F32,
                &Namespace::root().child("model").child("norm"),
                cx,
            )
            .with_unit_offset(),
            d,
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
        let mut x = self.embed.forward(ids) * (self.d as f32).sqrt();
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
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
