//! MiniQwen3Moe — the qwen3_moe family mini (mini relocation, 2026-08-13):
//! the model definition and decoder layer live here in the example crate.

use crate::namespace::Namespace;
use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, TopKRoutes, causal_bias, paged_attention,
};

pub struct MiniLinearMoe {
    pub expert_weights: GraphTensor,
    pub router: GraphTensor,
    pub top_k: usize,
}

impl MiniLinearMoe {
    fn forward(&self, input: GraphTensor) -> GraphTensor {
        let expert_axis = input.rank() - 1;
        let scores = input.matmul(self.router).softmax(vec![expert_axis]);
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

pub struct MiniQwen3MoeLayer {
    pub wq_weight: GraphTensor,
    pub wq_bias: Option<GraphTensor>,
    pub wk_weight: GraphTensor,
    pub wk_bias: Option<GraphTensor>,
    pub wv_weight: GraphTensor,
    pub wv_bias: Option<GraphTensor>,
    pub wo_weight: GraphTensor,
    pub wo_bias: Option<GraphTensor>,
    pub moe: Box<MiniLinearMoe>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl MiniQwen3MoeLayer {
    fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let query = luminal_nn::linear(x, self.wq_weight, self.wq_bias);
        let query_positions = query
            .graph()
            .iota(vec![query.dims()[0]], |c| c[0] + prev_seq);
        let context_positions = query.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            query,
            luminal_nn::linear(x, self.wk_weight, self.wk_bias),
            luminal_nn::linear(x, self.wv_weight, self.wv_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(query_positions, context_positions),
            AttentionGeometry::new(self.n_heads, self.n_heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + luminal_nn::linear(attn, self.wo_weight, self.wo_bias);
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
) -> (GraphTensor, Vec<MiniQwen3MoeLayer>, f32) {
    let model = Namespace::root().child("model");
    (
        cx.named_tensor(
            model.child("embed_tokens").leaf("weight"),
            vec![vocab, d],
            DType::F32,
        ),
        (0..layers)
            .map(|l| {
                let a = model.child("layers").index(l).child("self_attn");
                let mlp_ns = model.child("layers").index(l).child("mlp");
                MiniQwen3MoeLayer {
                    wq_weight: cx.named_tensor(
                        a.child("q_proj").leaf("weight"),
                        vec![d, d],
                        DType::F32,
                    ),
                    wq_bias: None,
                    wk_weight: cx.named_tensor(
                        a.child("k_proj").leaf("weight"),
                        vec![d, d],
                        DType::F32,
                    ),
                    wk_bias: None,
                    wv_weight: cx.named_tensor(
                        a.child("v_proj").leaf("weight"),
                        vec![d, d],
                        DType::F32,
                    ),
                    wv_bias: None,
                    wo_weight: cx.named_tensor(
                        a.child("o_proj").leaf("weight"),
                        vec![d, d],
                        DType::F32,
                    ),
                    wo_bias: None,
                    moe: Box::new(MiniLinearMoe {
                        expert_weights: cx.named_tensor(
                            mlp_ns.leaf("experts"),
                            vec![experts, d, d],
                            DType::F32,
                        ),
                        router: cx.named_tensor(
                            mlp_ns.leaf("router"),
                            vec![d, experts],
                            DType::F32,
                        ),
                        top_k,
                    }),
                    n_heads,
                    head_dim: d / n_heads,
                }
            })
            .collect(),
        1e-5,
    )
}

#[allow(clippy::too_many_arguments)]
fn moe_lm_forward(
    embed: &GraphTensor,
    blocks: &[MiniQwen3MoeLayer],
    final_norm: &(Option<GraphTensor>, Option<GraphTensor>, f32),
    ids: GraphTensor,
    caches: &[(GraphTensor, GraphTensor)],
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: IntExpr,
) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
    let mut x = luminal_nn::embedding(ids, *embed);
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
    let logits = luminal_nn::embedding_projection(
        luminal_nn::layer_norm(x, final_norm.0, final_norm.1, final_norm.2),
        *embed,
    );
    (logits, caches_out)
}

/// MiniQwen3Moe — the qwen3_moe family: MoE decoder blocks over the
/// paged cache. HONEST GAP: the example also carries QK-norm and
/// top-k=8 routing with renormalized weights (mini routes k=1); those
/// ride the pending fidelity ruling.
pub struct MiniQwen3Moe {
    pub embed: GraphTensor,
    pub blocks: Vec<MiniQwen3MoeLayer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
}

impl MiniQwen3Moe {
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
        let (embed, blocks, final_norm_epsilon) =
            moe_lm_new(vocab, d, experts, top_k, n_heads, layers, cx);
        {
            let final_norm_weight = None;
            let final_norm_bias = None;
            Self {
                embed,
                blocks,
                final_norm_weight,
                final_norm_bias,
                final_norm_epsilon,
            }
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
        moe_lm_forward(
            &self.embed,
            &self.blocks,
            &(
                self.final_norm_weight,
                self.final_norm_bias,
                self.final_norm_epsilon,
            ),
            ids,
            caches,
            gather_idx,
            scatter_idx,
            prev_seq,
        )
    }
}
