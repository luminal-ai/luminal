//! MiniGemma4Moe — the gemma4_moe family mini (rulings 2026-08-07/10,
//! relocated out of luminal_nn 2026-08-13): one small, runnable
//! representative of the gemma4_moe example model, named for the model
//! it represents. luminal_nn keeps the building blocks; the model
//! definition lives here so the family's constructs are visible in one
//! place.

use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::{DecoderBlock, Embedding, FeedForward, LayerNorm, Linear, MoE};

/// Shared MoE-decoder assembly (the [`MoE`] layer itself lives in
/// luminal_nn — the minis only assemble it): embed → N × DecoderBlock
/// with the MoE feed-forward → final LayerNorm → tied logits.
fn moe_lm_new(
    vocab: usize,
    d: usize,
    experts: usize,
    top_k: usize,
    n_heads: usize,
    layers: usize,
    cx: &mut Graph,
) -> (Embedding, Vec<DecoderBlock>, LayerNorm) {
    let model = Ns::root().child("model");
    (
        Embedding::new(vocab, d, &model.child("embed_tokens"), cx),
        (0..layers)
            .map(|l| {
                let a = model.child("layers").index(l).child("self_attn");
                let mlp_ns = model.child("layers").index(l).child("mlp");
                DecoderBlock {
                    wq: Linear::new(d, d, false, &a.child("q_proj"), cx),
                    wk: Linear::new(d, d, false, &a.child("k_proj"), cx),
                    wv: Linear::new(d, d, false, &a.child("v_proj"), cx),
                    wo: Linear::new(d, d, false, &a.child("o_proj"), cx),
                    ff: FeedForward::Moe(Box::new(MoE {
                        expert_weights: cx.named_tensor(mlp_ns.leaf("experts"), (experts, d, d)),
                        router: cx.named_tensor(mlp_ns.leaf("router"), (d, experts)),
                        k: top_k,
                    })),
                    n_heads,
                    n_kv_heads: n_heads,
                    head_dim: d / n_heads,
                }
            })
            .collect(),
        LayerNorm::new(d, false, false, true, 1e-5, &model.child("norm"), cx),
    )
}

fn moe_lm_forward(
    embed: &Embedding,
    blocks: &[DecoderBlock],
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
    pub blocks: Vec<DecoderBlock>,
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
