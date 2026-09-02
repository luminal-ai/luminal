//! MiniGemma4Moe — the gemma4_moe family mini (rulings 2026-08-07/10,
//! relocated out of luminal_nn 2026-08-13): one small, runnable
//! representative of the gemma4_moe example model, named for the model
//! it represents. The model definition and decoder layer live here so
//! the family's constructs are visible in one place.

#[path = "../../../common/model_support.rs"]
mod model_support;

use crate::model_support::{
    causal_bias, paged_attention, AttentionGeometry, CacheAccess, Embedding, KvCache, LayerNorm,
    Linear, Namespace, TopKRoutes,
};
use luminal::prelude::*;
use luminal::shape::IntExpr;

pub struct MiniLinearMoe {
    pub expert_weights: GraphTensor,
    pub router: GraphTensor,
    pub top_k: usize,
}

impl MiniLinearMoe {
    fn forward(&self, input: GraphTensor) -> GraphTensor {
        let expert_axis = input.rank() - 1;
        let scores = input.matmul(self.router).softmax(expert_axis);
        let expert_ids = scores.topk_indexes(self.top_k, expert_axis);
        let routes = TopKRoutes::from_scores(scores, expert_ids);
        let routed_input = routes.dispatch(input);
        let selected_weights = routes.select(self.expert_weights);
        let matrix_axis = routes.route_axis() + 1;
        let routed_output = routed_input
            .unsqueeze(matrix_axis)
            .matmul(selected_weights)
            .squeeze(matrix_axis);
        routes.combine(routed_output)
    }
}

pub struct MiniGemma4MoeLayer {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub moe: Box<MiniLinearMoe>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MiniGemma4MoeLayer {
    fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let query = self.wq.forward(x);
        let query_positions = query.graph().iota(query.dims()[0], |c| c[0] + prev_seq);
        let context_positions = query.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            query,
            self.wk.forward(x),
            self.wv.forward(x),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(query_positions, context_positions),
            AttentionGeometry::new(self.n_heads, self.n_heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + self.wo.forward(attn);
        (x + self.moe.forward(x), k_cache, v_cache)
    }
}

/// Shared MoE-decoder assembly: embed → N decoder layers with the MoE
/// feed-forward → final LayerNorm → tied logits.
fn moe_lm_new(
    vocab: usize,
    d: usize,
    experts: usize,
    top_k: usize,
    n_heads: usize,
    layers: usize,
    cx: &mut Graph,
) -> (Embedding, Vec<MiniGemma4MoeLayer>, LayerNorm) {
    let model = Namespace::root().child("model");
    (
        Embedding::new(vocab, d, DType::F32, &model.child("embed_tokens"), cx),
        (0..layers)
            .map(|l| {
                let a = model.child("layers").index(l).child("self_attn");
                let mlp_ns = model.child("layers").index(l).child("mlp");
                MiniGemma4MoeLayer {
                    wq: Linear::new(d, d, false, DType::F32, &a.child("q_proj"), cx),
                    wk: Linear::new(d, d, false, DType::F32, &a.child("k_proj"), cx),
                    wv: Linear::new(d, d, false, DType::F32, &a.child("v_proj"), cx),
                    wo: Linear::new(d, d, false, DType::F32, &a.child("o_proj"), cx),
                    moe: Box::new(MiniLinearMoe {
                        expert_weights: cx.named_tensor(
                            mlp_ns.leaf("experts"),
                            (experts, d, d),
                            DType::F32,
                        ),
                        router: cx.named_tensor(mlp_ns.leaf("router"), (d, experts), DType::F32),
                        top_k,
                    }),
                    n_heads,
                    head_dim: d / n_heads,
                }
            })
            .collect(),
        LayerNorm::new(
            d,
            false,
            false,
            true,
            1e-5,
            DType::F32,
            &model.child("norm"),
            cx,
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn moe_lm_forward(
    embed: &Embedding,
    blocks: &[MiniGemma4MoeLayer],
    final_norm: &LayerNorm,
    ids: GraphTensor,
    caches: &[(GraphTensor, GraphTensor)],
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: IntExpr,
) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
    let mut x = embed.forward(ids);
    let mut caches_out = Vec::with_capacity(blocks.len());
    for (layer, block) in blocks.iter().enumerate() {
        let (next, kc, vc) = block.forward(
            x,
            caches[layer].0,
            caches[layer].1,
            gather_idx,
            scatter_idx,
            prev_seq,
        );
        x = next;
        caches_out.push((kc, vc));
    }
    let logits = embed.reverse(final_norm.forward(x));
    (logits, caches_out)
}

/// MiniGemma4Moe — the gemma4_moe family: the MoE decoder plus the
/// construct only this family has, FINAL LOGIT SOFT-CAPPING:
/// `tanh(logits / cap) · cap`. HONEST GAP: sandwich norms and QK-norm
/// from the example ride the pending fidelity ruling.
pub struct MiniGemma4Moe {
    pub embed: Embedding,
    pub blocks: Vec<MiniGemma4MoeLayer>,
    pub final_norm: LayerNorm,
    pub logit_softcap: f32,
}

impl MiniGemma4Moe {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        d: usize,
        experts: usize,
        top_k: usize,
        n_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) = moe_lm_new(vocab, d, experts, top_k, n_heads, layers, cx);
        Self {
            embed,
            blocks,
            final_norm,
            logit_softcap: 30.0,
        }
    }

    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let (logits, caches_out) = moe_lm_forward(
            &self.embed,
            &self.blocks,
            &self.final_norm,
            ids,
            caches,
            gather_idx,
            scatter_idx,
            prev_seq,
        );
        let capped = (logits * (1.0 / self.logit_softcap)).tanh() * self.logit_softcap;
        (capped, caches_out)
    }
}
