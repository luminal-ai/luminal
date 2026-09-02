//! MiniLlama3 — one small, runnable representative of the
//! llama/paged_llama example-model family (mini relocation ruling
//! 2026-08-13: model definitions live in their example crates).

#[path = "../../../common/model_support.rs"]
pub mod model_support;

use crate::model_support::{
    causal_bias, paged_attention, rms_norm_heads, AttentionGeometry, CacheAccess, Embedding,
    KvCache, LayerNorm, Linear, Namespace,
};
use luminal::prelude::*;
use luminal::shape::IntExpr;

pub struct MiniLlama3Layer {
    pub attn_norm: LayerNorm,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub qk_norm: Option<(GraphTensor, GraphTensor)>,
    pub ffn_norm: LayerNorm,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl MiniLlama3Layer {
    pub fn new(
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        ns: &Namespace,
        cx: &mut Graph,
    ) -> Self {
        let head_dim = d / n_heads;
        let kv_dim = n_kv_heads * head_dim;
        let attn = ns.child("self_attn");
        let mlp = ns.child("mlp");
        Self {
            attn_norm: LayerNorm::new(
                d,
                false,
                false,
                false,
                1e-5,
                DType::F32,
                &ns.child("input_layernorm"),
                cx,
            ),
            wq: Linear::new(d, d, false, DType::F32, &attn.child("q_proj"), cx),
            wk: Linear::new(d, kv_dim, false, DType::F32, &attn.child("k_proj"), cx),
            wv: Linear::new(d, kv_dim, false, DType::F32, &attn.child("v_proj"), cx),
            wo: Linear::new(d, d, false, DType::F32, &attn.child("o_proj"), cx),
            qk_norm: None,
            ffn_norm: LayerNorm::new(
                d,
                false,
                false,
                false,
                1e-5,
                DType::F32,
                &ns.child("post_attention_layernorm"),
                cx,
            ),
            gate: Linear::new(d, ff, false, DType::F32, &mlp.child("gate_proj"), cx),
            up: Linear::new(d, ff, false, DType::F32, &mlp.child("up_proj"), cx),
            down: Linear::new(ff, d, false, DType::F32, &mlp.child("down_proj"), cx),
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = self.attn_norm.forward(x);
        let mut q = self.wq.forward(normed);
        let mut k = self.wk.forward(normed);
        if let Some((q_weight, k_weight)) = self.qk_norm {
            q = rms_norm_heads(q, self.head_dim, q_weight, 1e-6);
            k = rms_norm_heads(k, self.head_dim, k_weight, 1e-6);
        }
        let query_positions = q.graph().iota(q.dims()[0], |c| c[0] + prev_seq);
        let context_positions = q.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            q,
            k,
            self.wv.forward(normed),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(query_positions, context_positions),
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

/// Shared GQA-decoder assembly behind the family minis: embed →
/// N decoder layers (paged KV cache) → final RMSNorm → tied logits. Each
/// family keeps its own NAMED front door (ruling 2026-08-10: minis are
/// named for the model they represent, not parameterized as llama) so
/// family-specific constructs accrete in one visible place.
fn gqa_lm_new(
    vocab: usize,
    d: usize,
    ff: usize,
    n_heads: usize,
    n_kv_heads: usize,
    layers: usize,
    cx: &mut Graph,
) -> (Embedding, Vec<MiniLlama3Layer>, LayerNorm) {
    let model = Namespace::root().child("model");
    let blocks = (0..layers)
        .map(|l| {
            let layer_ns = model.child("layers").index(l);
            MiniLlama3Layer::new(d, ff, n_heads, n_kv_heads, &layer_ns, cx)
        })
        .collect();
    (
        Embedding::new(vocab, d, DType::F32, &model.child("embed_tokens"), cx),
        blocks,
        LayerNorm::new(
            d,
            false,
            false,
            false,
            1e-5,
            DType::F32,
            &model.child("norm"),
            cx,
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn gqa_lm_forward(
    embed: &Embedding,
    blocks: &[MiniLlama3Layer],
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

/// MiniLlama3 — the llama/paged_llama family: RMS pre-norms, GQA over a
/// paged KV cache, SwiGLU. (RoPE deferred by ruling, as everywhere.)
pub struct MiniLlama3 {
    pub embed: Embedding,
    pub blocks: Vec<MiniLlama3Layer>,
    pub final_norm: LayerNorm,
}

impl MiniLlama3 {
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm) = gqa_lm_new(vocab, d, ff, n_heads, n_kv_heads, layers, cx);
        Self {
            embed,
            blocks,
            final_norm,
        }
    }

    /// ids (s,) Int + one (k, v) cache pair per layer → (logits, caches').
    pub fn forward(
        &self,
        ids: GraphTensor,
        caches: &[(GraphTensor, GraphTensor)],
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        gqa_lm_forward(
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
