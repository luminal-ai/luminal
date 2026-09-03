//! nvidia/Llama-3.1-8B-Instruct-FP8 as pure logical ops. Fidelity notes
//! (ruling 2026-08-12 — every feature of THIS checkpoint): Llama-3.1
//! anatomy = Llama-3's plus the llama3-type ROPE SCALING RAMP (factor
//! 8.0, low_freq 1.0, high_freq 4.0, original_max 8192 — the parked
//! example omitted this; we don't) and per-tensor E4M3FN quantization
//! of every layer linear with static input scales (modelopt
//! calibration). Quantization is MODEL DEFINITION: the weights stage
//! and store as F8E4M3 buffers and the quantize/dequant chain is
//! spelled in model text ([`luminal_nn::fp8_linear`]). Embeddings and
//! the (untied) lm_head are bf16 in the checkpoint — numeric tensors,
//! not quantization — staged f32 like every zoo example.

use crate::model_support::{
    AttentionGeometry, CacheAccess, Embedding, Fp8Linear, KvCache, KvCachePool, LayerNorm, Linear,
    Namespace, causal_bias, paged_attention,
};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;

#[derive(Clone)]
pub struct Fp8Dims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub layers: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_low_freq_factor: f32,
    pub rope_high_freq_factor: f32,
    pub rope_original_max: f32,
    pub rms_eps: f32,
}

impl Fp8Dims {
    /// nvidia/Llama-3.1-8B-Instruct-FP8.
    pub fn llama3_1_8b_fp8() -> Self {
        Self {
            vocab: 128_256,
            hidden: 4096,
            intermediate: 14_336,
            head_dim: 128,
            n_heads: 32,
            n_kv_heads: 8,
            layers: 32,
            rope_theta: 500_000.0,
            rope_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max: 8192.0,
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

/// One fp8 layer: the Llama block anatomy with every projection an
/// [`Fp8Linear`]. Example-level for now — promoted to luminal_nn when a
/// second fp8 model lands.
pub struct Fp8Block {
    pub attn_norm: LayerNorm,
    pub wq: Fp8Linear,
    pub wk: Fp8Linear,
    pub wv: Fp8Linear,
    pub wo: Fp8Linear,
    pub ffn_norm: LayerNorm,
    pub gate: Fp8Linear,
    pub up: Fp8Linear,
    pub down: Fp8Linear,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Fp8Block {
    fn new(l: usize, d: &Fp8Dims, cx: &mut Graph) -> Self {
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
            wq: Fp8Linear::new(d.hidden, d.q_dim(), &attn.child("q_proj"), cx),
            wk: Fp8Linear::new(d.hidden, d.kv_dim(), &attn.child("k_proj"), cx),
            wv: Fp8Linear::new(d.hidden, d.kv_dim(), &attn.child("v_proj"), cx),
            wo: Fp8Linear::new(d.q_dim(), d.hidden, &attn.child("o_proj"), cx),
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
            gate: Fp8Linear::new(d.hidden, d.intermediate, &mlp.child("gate_proj"), cx),
            up: Fp8Linear::new(d.hidden, d.intermediate, &mlp.child("up_proj"), cx),
            down: Fp8Linear::new(d.intermediate, d.hidden, &mlp.child("down_proj"), cx),
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
            self.wq.forward(normed),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
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
}

pub struct Llama31Fp8 {
    pub dims: Fp8Dims,
    pub embed: Embedding,
    pub blocks: Vec<Fp8Block>,
    pub final_norm: LayerNorm,
    pub lm_head: Linear,
}

impl Llama31Fp8 {
    pub fn init(cx: &mut Graph, dims: &Fp8Dims) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Fp8Block::new(l, dims, cx))
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
