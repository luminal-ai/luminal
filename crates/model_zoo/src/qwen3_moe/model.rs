//! Qwen/Qwen3-30B-A3B as pure logical ops — 100% of THIS checkpoint
//! (ruling 2026-08-12): 48 layers, hidden 2048, GQA 32/4 at head_dim
//! 128, per-head QK-norm (learned, eps 1e-6) before rope, rope theta
//! 1e6 unscaled, UNTIED lm_head, and the MoE FFN on every layer — 128
//! experts, top-8, NO shared expert, router in F32 with the Qwen3
//! scoring order (softmax over all experts FIRST, then top-k, then
//! renormalize — norm_topk_prob).
//!
//! THE EXPERT BANKS ARE TWO RANK-3 INPUTS PER LAYER, not 3×E rank-2
//! ones: `mlp.experts.gate_up_proj` [E, 2I, H] and
//! `mlp.experts.down_proj` [E, H, I] — the fused form transformers v5
//! gives this model, and the form `gemma4_moe`'s checkpoint stores
//! natively. The Qwen3 checkpoint on disk stores every expert
//! separately (`mlp.experts.{e}.gate_proj|up_proj|down_proj.weight`),
//! so the LOADER stacks them, a pure byte relayout with no arithmetic
//! and no dtype change: `gate_up_proj[e, ..I, :]` = expert e's gate
//! rows, `gate_up_proj[e, I.., :]` = its up rows, `down_proj[e]` = its
//! down matrix. (Stacking in-graph instead would put 3×E inputs on the
//! boundary and build the bank by pad+add.)

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, KvCache, KvCachePool, LayerNorm, Linear, Namespace,
    TopKRoutes, causal_bias, paged_attention,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

#[derive(Clone)]
pub struct Qwen3MoeDims {
    pub vocab: usize,
    pub hidden: usize,
    pub moe_intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub experts: usize,
    pub top_k: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl Qwen3MoeDims {
    pub fn qwen3_30b_a3b() -> Self {
        Self {
            vocab: 151_936,
            hidden: 2048,
            moe_intermediate: 768,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 4,
            layers: 48,
            experts: 128,
            top_k: 8,
            rope_theta: 1_000_000.0,
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

/// Qwen3's model-specific routed SwiGLU feed-forward network over the
/// stacked expert banks (see the module doc for their layout).
pub struct Qwen3MoeFfn {
    pub router: Linear,
    /// `[E, 2I, H]`: gate rows then up rows per expert.
    pub gate_up: GraphTensor,
    /// `[E, H, I]`.
    pub down: GraphTensor,
    pub top_k: usize,
    pub intermediate: usize,
}

impl Qwen3MoeFfn {
    fn new(mlp: &Namespace, d: &Qwen3MoeDims, cx: &mut Graph) -> Self {
        let experts = mlp.child("experts");
        Self {
            router: Linear::new(
                d.hidden,
                d.experts,
                false,
                DType::F32,
                &mlp.child("gate"),
                cx,
            ),
            gate_up: cx.named_tensor(
                experts.leaf("gate_up_proj"),
                (d.experts, 2 * d.moe_intermediate, d.hidden),
                DType::F32,
            ),
            down: cx.named_tensor(
                experts.leaf("down_proj"),
                (d.experts, d.hidden, d.moe_intermediate),
                DType::F32,
            ),
            top_k: d.top_k,
            intermediate: d.moe_intermediate,
        }
    }

    fn forward(&self, input: GraphTensor) -> GraphTensor {
        let probabilities = self.router.forward(input).softmax(1);
        let expert_ids = probabilities.topk_indexes(self.top_k, 1);
        let routes = TopKRoutes::from_scores(probabilities, expert_ids).normalize();

        let gate_up = routes.select(self.gate_up);
        let projected = routes
            .dispatch(input)
            .expand_dim(2, 1)
            .matmul(gate_up.permute((0, 1, 3, 2)))
            .squeeze(2);
        let gate = projected.slice_along(..self.intermediate, 2);
        let up = projected.slice_along(self.intermediate.., 2);
        let hidden_states = gate.silu() * up;

        let down = routes.select(self.down);
        let routed_output = hidden_states
            .expand_dim(2, 1)
            .matmul(down.permute((0, 1, 3, 2)))
            .squeeze(2);
        routes.combine(routed_output)
    }
}

pub struct Qwen3MoeBlock {
    pub attn_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub q_norm: GraphTensor,
    pub k_norm: GraphTensor,
    pub ffn_norm: LayerNorm,
    pub moe: Qwen3MoeFfn,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Qwen3MoeBlock {
    fn new(l: usize, d: &Qwen3MoeDims, cx: &mut Graph) -> Self {
        let ns = Namespace::root().child("model").child("layers").index(l);
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        Self {
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
            q_norm: cx.named_tensor(attn.child("q_norm").leaf("weight"), d.head_dim, DType::F32),
            k_norm: cx.named_tensor(attn.child("k_norm").leaf("weight"), d.head_dim, DType::F32),
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
            moe: Qwen3MoeFfn::new(&mlp, d, cx),
            n_heads: d.n_heads,
            n_kv_heads: d.n_kv_heads,
            head_dim: d.head_dim,
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
        q_pos: GraphTensor,
        rope_cos: GraphTensor,
        rope_sin: GraphTensor,
        rope_rot: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let q = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(self.wq.forward(normed), self.head_dim, self.q_norm, 1e-6),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
            luminal_nn::rms_norm_heads(self.wk.forward(normed), self.head_dim, self.k_norm, 1e-6),
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
        let ff = self.moe.forward(self.ffn_norm.forward(x));
        (x + ff, k_cache, v_cache)
    }
}

pub struct Qwen3Moe {
    pub dims: Qwen3MoeDims,
    pub embed: Embedding,
    pub blocks: Vec<Qwen3MoeBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
}

impl Qwen3Moe {
    pub fn init(cx: &mut Graph, dims: &Qwen3MoeDims) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Qwen3MoeBlock::new(l, dims, cx))
            .collect();
        Self {
            dims: dims.clone(),
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
        let mut x = self.embed.forward(tokens);
        let mut caches_out = Vec::with_capacity(self.blocks.len());
        for (layer, block) in self.blocks.iter().enumerate() {
            let (next, k_cache, v_cache) = block.forward(
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
