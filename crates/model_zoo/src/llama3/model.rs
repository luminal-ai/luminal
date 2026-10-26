//! Meta-Llama-3-8B-Instruct as PURE LOGICAL OPS — the second zoo
//! conversion (per-HF-model fidelity ruling 2026-08-12: this example
//! is exactly `NousResearch/Meta-Llama-3-8B-Instruct`, every
//! architectural feature of that checkpoint and no other's). Anatomy:
//! 32 pre-RMSNorm layers (LEARNED norm weights, eps 1e-5), GQA 32/8
//! heads at head_dim 128, split-half rope at theta 500 000 with NO
//! scaling (this checkpoint's rope_scaling is null — the 3.1-family
//! frequency ramp belongs to the fp8 example), SwiGLU FFN, UNTIED
//! lm_head, and a caller-supplied slot-pool KV cache — cache
//! structure is model definition; this model uses the position-slots
//! form). Projections are UNFUSED: the old [v;q;k]/[gate;up] byte
//! fusion was a GEMV batching choice, not architecture, and the
//! unfused spelling keeps original HF tensor names and no slice
//! triple. Rope is the concat-free pairing-matrix spelling.

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, KvCachePool, causal_bias, paged_attention,
    rotary_apply,
};

#[derive(Clone)]
pub struct Llama3Dims {
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

impl Llama3Dims {
    /// NousResearch/Meta-Llama-3-8B-Instruct.
    pub fn llama3_8b() -> Self {
        Self {
            vocab: 128_256,
            hidden: 4096,
            intermediate: 14_336,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 32,
            rope_theta: 500_000.0,
            rms_eps: 1e-5,
        }
    }

    /// Same anatomy (GQA, learned norms, untied head, SwiGLU) at
    /// smoke-test scale.
    pub fn tiny() -> Self {
        Self {
            vocab: 29,
            hidden: 16,
            intermediate: 24,
            head_dim: 4,
            n_heads: 4,
            n_kv_heads: 2,
            layers: 2,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        }
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

pub struct Llama3Layer {
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

impl Llama3Layer {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rope(
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
            luminal_nn::linear(normed, self.wq_weight, self.wq_bias),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            luminal_nn::linear(normed, self.wk_weight, self.wk_bias),
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

    #[allow(clippy::too_many_arguments)]
    pub fn forward_rope_masked(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        mask: GraphTensor,
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
            luminal_nn::linear(normed, self.wq_weight, self.wq_bias),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            luminal_nn::linear(normed, self.wk_weight, self.wk_bias),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let result = paged_attention(
            q,
            k,
            luminal_nn::linear(normed, self.wv_weight, self.wv_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            mask,
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

pub struct Llama3 {
    pub dims: Llama3Dims,
    pub embed: GraphTensor,
    pub blocks: Vec<Llama3Layer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
    /// UNTIED head — a real (vocab, hidden) checkpoint tensor.
    pub lm_head_weight: GraphTensor,
    pub lm_head_bias: Option<GraphTensor>,
}

impl Llama3 {
    pub fn init(cx: &mut Graph, dims: &Llama3Dims) -> Self {
        let dtype = DType::F32;
        let blocks = (0..dims.layers)
            .map(|l| Self::block(l, dims, dtype, cx))
            .collect();
        Self {
            dims: dims.clone(),
            // HF stores embed_tokens as (vocab, hidden) — the natural
            // Embedding orientation.
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
            lm_head_weight: cx.named_tensor(
                Namespace::root().child("lm_head").leaf("weight"),
                vec![dims.hidden, dims.vocab],
                dtype,
            ),
            lm_head_bias: None,
        }
    }

    /// Literal construction: learned per-layer norm weights at this
    /// checkpoint's eps 1e-5, NO qk-norm (a Qwen3 feature, absent
    /// here), with checkpoint linears transposed to canonical (in, out) at staging.
    fn block(l: usize, d: &Llama3Dims, dtype: DType, cx: &mut Graph) -> Llama3Layer {
        let ns = Namespace::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        Llama3Layer {
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

    /// One decode step over the slot pool; see the qwen exemplar for
    /// the step-invariant contract (everything per-step is DATA).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        tokens: GraphTensor,
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
        pool: &KvCachePool,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        assert_eq!(tokens.dtype, DType::Int);
        assert_eq!(pool.layers.len(), self.blocks.len());
        let mut x = luminal_nn::embedding(tokens, self.embed);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward_rope(
                x,
                pool.layers[layer].0,
                pool.layers[layer].1,
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
        let logits = luminal_nn::linear(
            luminal_nn::rms_norm(
                x,
                self.final_norm_weight,
                self.final_norm_bias,
                self.final_norm_epsilon,
            ),
            self.lm_head_weight,
            self.lm_head_bias,
        );
        (logits, caches_out)
    }
}
