//! MiniQwen3 — the qwen family mini (rulings 2026-08-07/10, relocated
//! out of luminal_nn 2026-08-13): one small, runnable representative of
//! the qwen example family, named for the model it represents. The
//! model definition and decoder layer live here so the family's
//! constructs are visible in one place.

use crate::namespace::Namespace;
use luminal::prelude::*;
use luminal::shape::IntExpr;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, causal_bias, paged_attention, rms_norm_heads,
};

pub struct MiniQwen3Layer {
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
    pub qk_norm: Option<(GraphTensor, GraphTensor)>,
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

impl MiniQwen3Layer {
    fn new(
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
            attn_norm_weight: None,
            attn_norm_bias: None,
            attn_norm_epsilon: 1e-5,
            wq_weight: cx.named_tensor(attn.child("q_proj").leaf("weight"), vec![d, d], DType::F32),
            wq_bias: None,
            wk_weight: cx.named_tensor(
                attn.child("k_proj").leaf("weight"),
                vec![d, kv_dim],
                DType::F32,
            ),
            wk_bias: None,
            wv_weight: cx.named_tensor(
                attn.child("v_proj").leaf("weight"),
                vec![d, kv_dim],
                DType::F32,
            ),
            wv_bias: None,
            wo_weight: cx.named_tensor(attn.child("o_proj").leaf("weight"), vec![d, d], DType::F32),
            wo_bias: None,
            qk_norm: Some((
                cx.named_tensor(
                    attn.child("q_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
                cx.named_tensor(
                    attn.child("k_norm").leaf("weight"),
                    vec![head_dim],
                    DType::F32,
                ),
            )),
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            ffn_norm_epsilon: 1e-5,
            gate_weight: cx.named_tensor(
                mlp.child("gate_proj").leaf("weight"),
                vec![d, ff],
                DType::F32,
            ),
            gate_bias: None,
            up_weight: cx.named_tensor(
                mlp.child("up_proj").leaf("weight"),
                vec![d, ff],
                DType::F32,
            ),
            up_bias: None,
            down_weight: cx.named_tensor(
                mlp.child("down_proj").leaf("weight"),
                vec![ff, d],
                DType::F32,
            ),
            down_bias: None,
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    fn forward(
        &self,
        x: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
        gather_idx: GraphTensor,
        scatter_idx: GraphTensor,
        prev_seq: IntExpr,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let normed = luminal_nn::rms_norm(
            x,
            self.attn_norm_weight,
            self.attn_norm_bias,
            self.attn_norm_epsilon,
        );
        let mut q = luminal_nn::linear(normed, self.wq_weight, self.wq_bias);
        let mut k = luminal_nn::linear(normed, self.wk_weight, self.wk_bias);
        if let Some((q_weight, k_weight)) = self.qk_norm {
            q = rms_norm_heads(q, self.head_dim, q_weight, 1e-6);
            k = rms_norm_heads(k, self.head_dim, k_weight, 1e-6);
        }
        let query_positions = q.graph().iota(vec![q.dims()[0]], |c| c[0] + prev_seq);
        let context_positions = q.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            q,
            k,
            luminal_nn::linear(normed, self.wv_weight, self.wv_bias),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(query_positions, context_positions),
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
) -> (GraphTensor, Vec<MiniQwen3Layer>, f32) {
    let model = Namespace::root().child("model");
    let blocks = (0..layers)
        .map(|l| {
            let layer_ns = model.child("layers").index(l);
            MiniQwen3Layer::new(d, ff, n_heads, n_kv_heads, &layer_ns, cx)
        })
        .collect();
    (
        cx.named_tensor(
            model.child("embed_tokens").leaf("weight"),
            vec![vocab, d],
            DType::F32,
        ),
        blocks,
        1e-5,
    )
}

#[allow(clippy::too_many_arguments)]
fn gqa_lm_forward(
    embed: &GraphTensor,
    blocks: &[MiniQwen3Layer],
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
        luminal_nn::rms_norm(x, final_norm.0, final_norm.1, final_norm.2),
        *embed,
    );
    (logits, caches_out)
}

/// MiniQwen3 — the qwen family: the llama-3 skeleton plus Qwen3's
/// per-head QK RMSNorm on q/k (the construct the qwen example adds).
pub struct MiniQwen3 {
    pub embed: GraphTensor,
    pub blocks: Vec<MiniQwen3Layer>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
}

impl MiniQwen3 {
    pub fn new(
        vocab: usize,
        d: usize,
        ff: usize,
        n_heads: usize,
        n_kv_heads: usize,
        layers: usize,
        cx: &mut Graph,
    ) -> Self {
        let (embed, blocks, final_norm_epsilon) =
            gqa_lm_new(vocab, d, ff, n_heads, n_kv_heads, layers, cx);
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
