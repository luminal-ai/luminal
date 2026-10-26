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

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, KvCachePool, causal_bias, paged_attention,
    rms_norm_heads, rotary_apply, sliding_window_bias,
};

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
        let context_positions = q.graph().arange(gather_idx.dims1());
        let score_bias = if self.local {
            sliding_window_bias(q_pos, context_positions, self.window)
        } else {
            causal_bias(q_pos, context_positions)
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

pub struct Gemma3 {
    pub dims: Gemma3Dims,
    pub embed: GraphTensor,
    pub blocks: Vec<Gemma3Layer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
}

impl Gemma3 {
    pub fn init(cx: &mut Graph, dims: &Gemma3Dims) -> Self {
        let dtype = DType::F32;
        let text = Namespace::root().child("language_model").child("model");
        let blocks = (0..dims.layers)
            .map(|l| Self::block(l, dims, dtype, cx))
            .collect();
        {
            let (final_norm_weight, final_norm_bias, final_norm_epsilon) = (
                Some(cx.named_tensor(text.child("norm").leaf("weight"), vec![dims.hidden], dtype)),
                None,
                dims.rms_eps,
            );
            Self {
                dims: dims.clone(),
                embed: cx.named_tensor(
                    text.child("embed_tokens").leaf("weight"),
                    vec![dims.vocab, dims.hidden],
                    dtype,
                ),
                blocks,
                final_norm_weight,
                final_norm_bias,
                final_norm_epsilon,
            }
        }
    }

    fn block(l: usize, d: &Gemma3Dims, dtype: DType, cx: &mut Graph) -> Gemma3Layer {
        let local = d.is_local(l);
        let ns = Namespace::root()
            .child("language_model")
            .child("model")
            .child("layers")
            .index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        let rms = |ns: &Namespace, cx: &mut Graph| {
            (
                Some(cx.named_tensor((ns).leaf("weight"), vec![d.hidden], dtype)),
                None,
                d.rms_eps,
            )
        };
        {
            let (input_norm_weight, input_norm_bias, input_norm_epsilon) =
                rms(&ns.child("input_layernorm"), cx);
            let (post_attn_norm_weight, post_attn_norm_bias, post_attn_norm_epsilon) =
                rms(&ns.child("post_attention_layernorm"), cx);
            let (pre_ff_norm_weight, pre_ff_norm_bias, pre_ff_norm_epsilon) =
                rms(&ns.child("pre_feedforward_layernorm"), cx);
            let (post_ff_norm_weight, post_ff_norm_bias, post_ff_norm_epsilon) =
                rms(&ns.child("post_feedforward_layernorm"), cx);
            Gemma3Layer {
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
                    vec![d.hidden, d.n_heads * d.head_dim],
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
                    vec![d.n_heads * d.head_dim, d.hidden],
                    dtype,
                ),
                wo_bias: None,
                q_norm: cx
                    .named_tensor(attn.child("q_norm").leaf("weight"), vec![d.head_dim], dtype)
                    .cast(DType::F32),
                k_norm: cx
                    .named_tensor(attn.child("k_norm").leaf("weight"), vec![d.head_dim], dtype)
                    .cast(DType::F32),
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
                local,
                window: d.window,
                n_heads: d.n_heads,
                n_kv_heads: d.n_kv_heads,
                head_dim: d.head_dim,
            }
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
        let mut x = luminal_nn::embedding(tokens, self.embed) * (self.dims.hidden as f32).sqrt();
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
