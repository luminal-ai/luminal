//! MiniQwen3Moe — the qwen3_moe family mini (mini relocation, 2026-08-13):
//! the model DEFINITION lives here in its example crate; luminal_nn keeps
//! only the building blocks ([`MoE`], [`DecoderBlock`], [`Embedding`], ...).

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

/// MiniQwen3Moe — the qwen3_moe family: MoE decoder blocks over the
/// paged cache. HONEST GAP: the example also carries QK-norm and
/// top-k=8 routing with renormalized weights (mini routes k=1); those
/// ride the pending fidelity ruling.
pub struct MiniQwen3Moe {
    pub embed: Embedding,
    pub blocks: Vec<DecoderBlock>,
    pub final_norm: LayerNorm,
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
        let (embed, blocks, final_norm) = moe_lm_new(vocab, d, experts, top_k, n_heads, layers, cx);
        Self {
            embed,
            blocks,
            final_norm,
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
            &self.final_norm,
            ids,
            caches,
            gather_idx,
            scatter_idx,
            prev_seq,
        )
    }
}
