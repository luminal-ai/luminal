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

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, KvCache, KvCachePool, LayerNorm, Linear, Namespace,
    causal_bias, paged_attention, rotary_apply,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

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
    pub attn_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub ffn_norm: LayerNorm,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
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
        let normed = self.attn_norm.forward(x);
        let q = rotary_apply(
            self.wq.forward(normed),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            self.wk.forward(normed),
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
        let normed = self.attn_norm.forward(x);
        let q = rotary_apply(
            self.wq.forward(normed),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = rotary_apply(
            self.wk.forward(normed),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let result = paged_attention(
            q,
            k,
            self.wv.forward(normed),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            mask,
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

pub struct Llama3 {
    pub dims: Llama3Dims,
    pub embed: Embedding,
    pub blocks: Vec<Llama3Layer>,
    pub final_norm: LayerNorm,
    /// UNTIED head — a real (vocab, hidden) checkpoint tensor.
    pub lm_head: Linear,
}

impl Llama3 {
    pub fn init(cx: &mut Graph, dims: &Llama3Dims) -> Self {
        let blocks = (0..dims.layers).map(|l| Self::block(l, dims, cx)).collect();
        Self {
            dims: dims.clone(),
            // HF stores embed_tokens as (vocab, hidden) — the natural
            // Embedding orientation.
            embed: Embedding::new(
                dims.vocab,
                dims.hidden,
                DType::F32,
                &Namespace::root().child("model").child("embed_tokens"),
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
                &Namespace::root().child("model").child("norm"),
                cx,
            ),
            lm_head: Linear::new(
                dims.hidden,
                dims.vocab,
                false,
                DType::F32,
                &Namespace::root().child("lm_head"),
                cx,
            ),
        }
    }

    /// Literal construction: learned per-layer norm weights at this
    /// checkpoint's eps 1e-5, NO qk-norm (a Qwen3 feature, absent
    /// here), with checkpoint linears transposed to canonical (in, out) at staging.
    fn block(l: usize, d: &Llama3Dims, cx: &mut Graph) -> Llama3Layer {
        let ns = Namespace::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        Llama3Layer {
            attn_norm: LayerNorm::new(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                DType::F32,
                &ns.child("input_layernorm"),
                cx,
            ),
            wq: Linear::new(
                d.hidden,
                d.q_dim(),
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
                d.q_dim(),
                d.hidden,
                false,
                DType::F32,
                &attn.child("o_proj"),
                cx,
            ),
            ffn_norm: LayerNorm::new(
                d.hidden,
                true,
                false,
                false,
                d.rms_eps,
                DType::F32,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
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
        let mut x = self.embed.forward(tokens);
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
        let logits = self.lm_head.forward(self.final_norm.forward(x));
        (logits, caches_out)
    }
}
