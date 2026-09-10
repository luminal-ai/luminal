//! unsloth/gemma-3-4b-it (text tower) as pure logical ops — 100% of
//! THIS checkpoint's features (ruling 2026-08-12): 34 layers in the
//! 5-local:1-global alternation (globals at l+1 ≡ 0 mod 6), sliding
//! window 1024 on locals (enforced by mask — no eviction), dual rope
//! thetas (10k local / 1M global) with linear position scaling /8 on
//! globals only, per-head QK-norm (learned, eps 1e-6) before rope,
//! attention scale 1/16 FOLDED INTO Q (query_pre_attn_scalar =
//! head_dim), sandwich norms (four per layer, the Gemma (1+w) pattern
//! pre-baked host-side at combine), GeGLU FFN, TIED embeddings with
//! the sqrt(hidden) normalizer IN-GRAPH over the unscaled table (the
//! parked example pre-scaled the table and duplicated an unscaled head
//! copy — same math, this spelling keeps one table and matches the HF
//! config's tie + normalizer directly). No logit softcaps (Gemma 3
//! dropped them for QK-norm).

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, KvCache, KvCachePool, LayerNorm, Linear, Namespace,
    causal_bias, paged_attention, rms_norm_heads, rotary_apply, sliding_window_bias,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

#[derive(Clone)]
pub struct Gemma3Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub window: usize,
    pub sliding_pattern: usize,
    pub rms_eps: f32,
}

impl Gemma3Dims {
    pub fn gemma3_4b() -> Self {
        Self {
            vocab: 262_208,
            hidden: 2560,
            intermediate: 10_240,
            head_dim: 256,
            n_heads: 8,
            n_kv_heads: 4,
            layers: 34,
            window: 1024,
            sliding_pattern: 6,
            rms_eps: 1e-6,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab: 31,
            hidden: 16,
            intermediate: 24,
            head_dim: 4,
            n_heads: 4,
            n_kv_heads: 2,
            layers: 2, // layer 0 local, layer 1 global (pattern 2)
            window: 3,
            sliding_pattern: 2,
            rms_eps: 1e-6,
        }
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    #[allow(clippy::manual_is_multiple_of)] // Keep rust-version 1.85 compatibility.
    pub fn is_local(&self, layer: usize) -> bool {
        (layer + 1) % self.sliding_pattern != 0
    }
}

pub struct Gemma3Layer {
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
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Gemma3Layer {
    #[allow(clippy::too_many_arguments)]
    fn forward_positional(
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
        let context_positions = q.graph().arange(gather_idx.dims1());
        let score_bias = if self.local {
            sliding_window_bias(q_pos, context_positions, self.window)
        } else {
            causal_bias(q_pos, context_positions)
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

pub struct Gemma3 {
    pub dims: Gemma3Dims,
    pub embed: Embedding,
    pub blocks: Vec<Gemma3Layer>,
    pub final_norm: LayerNorm,
}

impl Gemma3 {
    pub fn init(cx: &mut Graph, dims: &Gemma3Dims) -> Self {
        let text = Namespace::root().child("language_model").child("model");
        let blocks = (0..dims.layers).map(|l| Self::block(l, dims, cx)).collect();
        Self {
            dims: dims.clone(),
            embed: Embedding::new(
                dims.vocab,
                dims.hidden,
                DType::F32,
                &text.child("embed_tokens"),
                cx,
            ),
            blocks,
            final_norm: LayerNorm::new(
                dims.hidden,
                true,
                false,
                false,
                dims.rms_eps,
                DType::F32,
                &text.child("norm"),
                cx,
            )
            .with_unit_offset(),
        }
    }

    fn block(l: usize, d: &Gemma3Dims, cx: &mut Graph) -> Gemma3Layer {
        let local = d.is_local(l);
        let ns = Namespace::root()
            .child("language_model")
            .child("model")
            .child("layers")
            .index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |ns: &Namespace, cx: &mut Graph| {
            LayerNorm::new(d.hidden, true, false, false, d.rms_eps, DType::F32, ns, cx)
                .with_unit_offset()
        };
        Gemma3Layer {
            input_norm: rms(&ns.child("input_layernorm"), cx),
            post_attn_norm: rms(&ns.child("post_attention_layernorm"), cx),
            pre_ff_norm: rms(&ns.child("pre_feedforward_layernorm"), cx),
            post_ff_norm: rms(&ns.child("post_feedforward_layernorm"), cx),
            wq: Linear::new(
                d.hidden,
                d.n_heads * d.head_dim,
                false,
                DType::F32,
                &attn.child("q_proj"),
                cx,
            ),
            wk: Linear::new(
                d.hidden,
                d.kv_dim(),
                false,
                DType::F32,
                &attn.child("k_proj"),
                cx,
            ),
            wv: Linear::new(
                d.hidden,
                d.kv_dim(),
                false,
                DType::F32,
                &attn.child("v_proj"),
                cx,
            ),
            wo: Linear::new(
                d.n_heads * d.head_dim,
                d.hidden,
                false,
                DType::F32,
                &attn.child("o_proj"),
                cx,
            ),
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim, DType::F32),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim, DType::F32),
            gate: Linear::new(
                d.hidden,
                d.intermediate,
                false,
                DType::F32,
                &mlp.child("gate_proj"),
                cx,
            ),
            up: Linear::new(
                d.hidden,
                d.intermediate,
                false,
                DType::F32,
                &mlp.child("up_proj"),
                cx,
            ),
            down: Linear::new(
                d.intermediate,
                d.hidden,
                false,
                DType::F32,
                &mlp.child("down_proj"),
                cx,
            ),
            local,
            window: d.window,
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
        }
    }

    /// One decode step. Local and global layers consume their own rope
    /// tables (dual theta + position scaling), fed as two (s, head_dim)
    /// pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_local: (GraphTensor, GraphTensor),
        rope_global: (GraphTensor, GraphTensor),
        rope_rot: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        // The HF gemma normalizer: hidden states scale by sqrt(hidden)
        // right after the (unscaled) embedding lookup.
        let mut x = self.embed.forward(tokens) * (self.dims.hidden as f32).sqrt();
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (cos, sin) = if block.local { rope_local } else { rope_global };
            let (next, k_cache, v_cache) = block.forward_positional(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
                gather_idx,
                scatter_idx,
                q_pos,
                cos,
                sin,
                rope_rot,
            );
            x = next;
            caches_out.push((k_cache, v_cache));
        }
        // Tied head over the unscaled table.
        let logits = self.embed.reverse(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
