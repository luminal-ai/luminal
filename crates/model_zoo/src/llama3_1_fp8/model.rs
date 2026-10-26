//! nvidia/Llama-3.1-8B-Instruct-FP8 as pure logical ops. Fidelity notes
//! (ruling 2026-08-12 — every feature of THIS checkpoint): Llama-3.1
//! anatomy = Llama-3's plus the llama3-type ROPE SCALING RAMP (factor
//! 8.0, low_freq 1.0, high_freq 4.0, original_max 8192 — the parked
//! example omitted this; we don't) and per-tensor E4M3FN quantization
//! of every layer linear with static input scales (modelopt
//! calibration). Quantization is MODEL DEFINITION: the weights stage
//! and store as F8E4M3FN buffers and the quantize/dequant chain is
//! spelled in model text ([`luminal_nn::fp8_linear`]). Embeddings and
//! the (untied) lm_head are bf16 in the checkpoint — numeric tensors,
//! not quantization — staged f32 like every zoo example.

use crate::namespace::Namespace;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::GraphTensor;
use luminal_nn::{
    AttentionGeometry, CacheAccess, KvCache, KvCachePool, causal_bias, paged_attention,
};

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
/// call to [`luminal_nn::fp8_linear`] with explicit weight and scale tensors.
pub struct Fp8Block {
    pub attn_norm_weight: Option<GraphTensor>,
    pub attn_norm_bias: Option<GraphTensor>,
    pub attn_norm_epsilon: f32,
    pub wq_weight: GraphTensor,
    pub wq_input_scale: GraphTensor,
    pub wq_weight_scale: GraphTensor,
    pub wk_weight: GraphTensor,
    pub wk_input_scale: GraphTensor,
    pub wk_weight_scale: GraphTensor,
    pub wv_weight: GraphTensor,
    pub wv_input_scale: GraphTensor,
    pub wv_weight_scale: GraphTensor,
    pub wo_weight: GraphTensor,
    pub wo_input_scale: GraphTensor,
    pub wo_weight_scale: GraphTensor,
    pub ffn_norm_weight: Option<GraphTensor>,
    pub ffn_norm_bias: Option<GraphTensor>,
    pub ffn_norm_epsilon: f32,
    pub gate_weight: GraphTensor,
    pub gate_input_scale: GraphTensor,
    pub gate_weight_scale: GraphTensor,
    pub up_weight: GraphTensor,
    pub up_input_scale: GraphTensor,
    pub up_weight_scale: GraphTensor,
    pub down_weight: GraphTensor,
    pub down_input_scale: GraphTensor,
    pub down_weight_scale: GraphTensor,
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
            attn_norm_weight: Some(cx.named_tensor(
                ns.child("input_layernorm").leaf("weight"),
                vec![d.hidden],
                DType::F32,
            )),
            attn_norm_bias: None,
            attn_norm_epsilon: d.rms_eps,
            wq_weight: cx.named_tensor(
                attn.child("q_proj").leaf("weight"),
                vec![d.q_dim(), d.hidden],
                DType::F8E4M3FN,
            ),
            wq_input_scale: cx.named_tensor(
                attn.child("q_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wq_weight_scale: cx.named_tensor(
                attn.child("q_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wk_weight: cx.named_tensor(
                attn.child("k_proj").leaf("weight"),
                vec![d.kv_dim(), d.hidden],
                DType::F8E4M3FN,
            ),
            wk_input_scale: cx.named_tensor(
                attn.child("k_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wk_weight_scale: cx.named_tensor(
                attn.child("k_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wv_weight: cx.named_tensor(
                attn.child("v_proj").leaf("weight"),
                vec![d.kv_dim(), d.hidden],
                DType::F8E4M3FN,
            ),
            wv_input_scale: cx.named_tensor(
                attn.child("v_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wv_weight_scale: cx.named_tensor(
                attn.child("v_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wo_weight: cx.named_tensor(
                attn.child("o_proj").leaf("weight"),
                vec![d.hidden, d.q_dim()],
                DType::F8E4M3FN,
            ),
            wo_input_scale: cx.named_tensor(
                attn.child("o_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            wo_weight_scale: cx.named_tensor(
                attn.child("o_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            ffn_norm_weight: Some(cx.named_tensor(
                ns.child("post_attention_layernorm").leaf("weight"),
                vec![d.hidden],
                DType::F32,
            )),
            ffn_norm_bias: None,
            ffn_norm_epsilon: d.rms_eps,
            gate_weight: cx.named_tensor(
                mlp.child("gate_proj").leaf("weight"),
                vec![d.intermediate, d.hidden],
                DType::F8E4M3FN,
            ),
            gate_input_scale: cx.named_tensor(
                mlp.child("gate_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            gate_weight_scale: cx.named_tensor(
                mlp.child("gate_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            up_weight: cx.named_tensor(
                mlp.child("up_proj").leaf("weight"),
                vec![d.intermediate, d.hidden],
                DType::F8E4M3FN,
            ),
            up_input_scale: cx.named_tensor(
                mlp.child("up_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            up_weight_scale: cx.named_tensor(
                mlp.child("up_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            down_weight: cx.named_tensor(
                mlp.child("down_proj").leaf("weight"),
                vec![d.hidden, d.intermediate],
                DType::F8E4M3FN,
            ),
            down_input_scale: cx.named_tensor(
                mlp.child("down_proj").leaf("input_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
            down_weight_scale: cx.named_tensor(
                mlp.child("down_proj").leaf("weight_scale"),
                Vec::<usize>::new(),
                DType::F32,
            ),
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
        let normed = luminal_nn::rms_norm(
            x,
            self.attn_norm_weight,
            self.attn_norm_bias,
            self.attn_norm_epsilon,
        );
        let q = luminal_nn::rotary_apply(
            luminal_nn::fp8_linear(
                normed,
                self.wq_weight,
                self.wq_input_scale,
                self.wq_weight_scale,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let k = luminal_nn::rotary_apply(
            luminal_nn::fp8_linear(
                normed,
                self.wk_weight,
                self.wk_input_scale,
                self.wk_weight_scale,
            ),
            self.head_dim,
            rope_cos,
            rope_sin,
            rope_rot,
        );
        let context_positions = q.graph().arange(gather_idx.dims1());
        let result = paged_attention(
            q,
            k,
            luminal_nn::fp8_linear(
                normed,
                self.wv_weight,
                self.wv_input_scale,
                self.wv_weight_scale,
            ),
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, context_positions),
            AttentionGeometry::new(self.n_heads, self.n_kv_heads, self.head_dim),
        );
        let (attn, k_cache, v_cache) = (result.output, result.cache.keys, result.cache.values);
        let x = x + luminal_nn::fp8_linear(
            attn,
            self.wo_weight,
            self.wo_input_scale,
            self.wo_weight_scale,
        );
        let ff_in = luminal_nn::rms_norm(
            x,
            self.ffn_norm_weight,
            self.ffn_norm_bias,
            self.ffn_norm_epsilon,
        );
        let ff = luminal_nn::fp8_linear(
            luminal_nn::fp8_linear(
                ff_in,
                self.gate_weight,
                self.gate_input_scale,
                self.gate_weight_scale,
            )
            .silu()
                * luminal_nn::fp8_linear(
                    ff_in,
                    self.up_weight,
                    self.up_input_scale,
                    self.up_weight_scale,
                ),
            self.down_weight,
            self.down_input_scale,
            self.down_weight_scale,
        );
        (x + ff, k_cache, v_cache)
    }
}

pub struct Llama31Fp8 {
    pub dims: Fp8Dims,
    pub embed: GraphTensor,
    pub blocks: Vec<Fp8Block>,
    pub final_norm_weight: Option<GraphTensor>,
    pub final_norm_bias: Option<GraphTensor>,
    pub final_norm_epsilon: f32,
    pub lm_head_weight: GraphTensor,
    pub lm_head_bias: Option<GraphTensor>,
}

impl Llama31Fp8 {
    pub fn init(cx: &mut Graph, dims: &Fp8Dims) -> Self {
        let blocks = (0..dims.layers)
            .map(|l| Fp8Block::new(l, dims, cx))
            .collect();
        Self {
            dims: dims.clone(),
            embed: cx.named_tensor(
                Namespace::root()
                    .child("model")
                    .child("embed_tokens")
                    .leaf("weight"),
                vec![dims.vocab, dims.hidden],
                DType::F32,
            ),
            blocks,
            final_norm_weight: Some(
                cx.named_tensor(
                    Namespace::root()
                        .child("model")
                        .child("norm")
                        .leaf("weight"),
                    vec![dims.hidden],
                    DType::F32,
                ),
            ),
            final_norm_bias: None,
            final_norm_epsilon: dims.rms_eps,
            lm_head_weight: cx.named_tensor(
                Namespace::root().child("lm_head").leaf("weight"),
                vec![dims.hidden, dims.vocab],
                DType::F32,
            ),
            lm_head_bias: None,
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
        let mut x = luminal_nn::embedding(tokens, self.embed);
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
