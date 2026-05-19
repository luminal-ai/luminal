use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
};
use luminal_nn::{LayerNorm, gather_rows, scatter_rows};

// Llama 3 8B hyperparams
pub const LAYERS: usize = 32;
pub const HIDDEN: usize = 4096;
pub const INTERMEDIATE: usize = 14336;
pub const HEAD_DIM: usize = 128;
pub const KV_GROUPS: usize = 4;
pub const VOCAB_SIZE: usize = 128256;
pub const N_KV_HEADS: usize = HIDDEN / HEAD_DIM / KV_GROUPS; // 8
#[allow(dead_code)]
pub const N_HEADS: usize = HIDDEN / HEAD_DIM; // 32
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // 1024

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaConfig {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub head_dim: usize,
    pub kv_groups: usize,
    pub vocab_size: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            layers: LAYERS,
            hidden: HIDDEN,
            intermediate: INTERMEDIATE,
            head_dim: HEAD_DIM,
            kv_groups: KV_GROUPS,
            vocab_size: VOCAB_SIZE,
        }
    }
}

impl LlamaConfig {
    pub fn n_heads(self) -> usize {
        self.hidden / self.head_dim
    }

    pub fn n_kv_heads(self) -> usize {
        self.hidden / self.head_dim / self.kv_groups
    }

    pub fn kv_dim(self) -> usize {
        self.n_kv_heads() * self.head_dim
    }
}

pub struct KVCache {
    pub k_caches: Vec<GraphTensor>,
    pub v_caches: Vec<GraphTensor>,
}

impl KVCache {
    pub fn new(cx: &mut Graph, num_slots: usize) -> Self {
        Self::new_with_config(cx, num_slots, LlamaConfig::default())
    }

    pub fn new_with_config(cx: &mut Graph, num_slots: usize, config: LlamaConfig) -> Self {
        let kv_dim = config.kv_dim();
        let mut k_caches = Vec::with_capacity(config.layers);
        let mut v_caches = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            k_caches.push(cx.named_tensor(format!("kv_cache.{l}.k"), (num_slots, kv_dim)));
            v_caches.push(cx.named_tensor(format!("kv_cache.{l}.v"), (num_slots, kv_dim)));
        }
        Self { k_caches, v_caches }
    }

    #[allow(dead_code)]
    pub fn tensors(&self) -> Vec<GraphTensor> {
        self.k_caches
            .iter()
            .chain(self.v_caches.iter())
            .copied()
            .collect()
    }
}

pub struct Llama {
    config: LlamaConfig,
    embedding: GraphTensor,
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Llama {
    pub fn init(cx: &mut Graph) -> Self {
        Self::init_with_config(cx, LlamaConfig::default())
    }

    pub fn init_fp8(cx: &mut Graph) -> Self {
        Self::init_with_config_and_fp8(cx, LlamaConfig::default(), true)
    }

    pub fn init_with_config(cx: &mut Graph, config: LlamaConfig) -> Self {
        Self::init_with_config_and_fp8(cx, config, false)
    }

    pub fn init_with_config_and_fp8(
        cx: &mut Graph,
        config: LlamaConfig,
        fp8_linears: bool,
    ) -> Self {
        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            layers.push(LlamaLayer {
                config,
                up: linear_weight(
                    cx,
                    format!("model.layers.{l}.mlp.up_proj"),
                    (config.intermediate, config.hidden),
                    fp8_linears,
                ),
                up_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.mlp.up_proj"),
                    fp8_linears,
                ),
                gate: linear_weight(
                    cx,
                    format!("model.layers.{l}.mlp.gate_proj"),
                    (config.intermediate, config.hidden),
                    fp8_linears,
                ),
                gate_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.mlp.gate_proj"),
                    fp8_linears,
                ),
                down: linear_weight(
                    cx,
                    format!("model.layers.{l}.mlp.down_proj"),
                    (config.hidden, config.intermediate),
                    fp8_linears,
                ),
                down_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.mlp.down_proj"),
                    fp8_linears,
                ),
                q_proj: linear_weight(
                    cx,
                    format!("model.layers.{l}.self_attn.q_proj"),
                    (config.hidden, config.hidden),
                    fp8_linears,
                ),
                q_proj_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.self_attn.q_proj"),
                    fp8_linears,
                ),
                k_proj: linear_weight(
                    cx,
                    format!("model.layers.{l}.self_attn.k_proj"),
                    (config.kv_dim(), config.hidden),
                    fp8_linears,
                ),
                k_proj_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.self_attn.k_proj"),
                    fp8_linears,
                ),
                v_proj: linear_weight(
                    cx,
                    format!("model.layers.{l}.self_attn.v_proj"),
                    (config.kv_dim(), config.hidden),
                    fp8_linears,
                ),
                v_proj_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.self_attn.v_proj"),
                    fp8_linears,
                ),
                o_proj: linear_weight(
                    cx,
                    format!("model.layers.{l}.self_attn.o_proj"),
                    (config.hidden, config.hidden),
                    fp8_linears,
                ),
                o_proj_scales: fp8_linear_scales(
                    cx,
                    format!("model.layers.{l}.self_attn.o_proj"),
                    fp8_linears,
                ),
                attn_rms: LayerNorm::new(
                    config.hidden,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    config.hidden,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    1e-5,
                    cx,
                ),
            });
        }
        Self {
            config,
            embedding: cx
                .named_tensor(
                    "model.embed_tokens.weight",
                    (config.vocab_size, config.hidden),
                )
                .persist(),
            layers,
            lm_head: cx
                .named_tensor("lm_head.weight", (config.vocab_size, config.hidden))
                .persist(),
            lm_norm: LayerNorm::new(
                config.hidden,
                Some("model.norm.weight"),
                None,
                false,
                1e-5,
                cx,
            ),
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(
        &self,
        input: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let seq = input.dims1();
        let hidden = self.config.hidden;
        let mut x = self.embedding.gather(
            (input * hidden).expand_dim(1, hidden)
                + input.graph().arange(hidden).expand_dim(0, seq),
        );
        let mut cache_outputs = Vec::with_capacity(self.config.layers);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                q_pos,
                scatter_idx,
                gather_idx,
                attn_mask,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        let logits = self.lm_norm.forward(x).matmul(self.lm_head.t());
        (logits, cache_outputs)
    }

    #[allow(dead_code)]
    pub fn parameter_tensors(&self) -> Vec<GraphTensor> {
        let mut tensors = Vec::new();
        tensors.push(self.embedding);
        for layer in &self.layers {
            tensors.extend(layer.parameter_tensors());
        }
        if let Some(weight) = self.lm_norm.weight {
            tensors.push(weight);
        }
        if let Some(bias) = self.lm_norm.bias {
            tensors.push(bias);
        }
        tensors.push(self.lm_head);
        tensors
    }
}

struct LlamaLayer {
    config: LlamaConfig,
    up: GraphTensor,
    up_scales: Option<Fp8LinearScales>,
    gate: GraphTensor,
    gate_scales: Option<Fp8LinearScales>,
    down: GraphTensor,
    down_scales: Option<Fp8LinearScales>,
    q_proj: GraphTensor,
    q_proj_scales: Option<Fp8LinearScales>,
    k_proj: GraphTensor,
    k_proj_scales: Option<Fp8LinearScales>,
    v_proj: GraphTensor,
    v_proj_scales: Option<Fp8LinearScales>,
    o_proj: GraphTensor,
    o_proj_scales: Option<Fp8LinearScales>,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
}

#[derive(Clone, Copy)]
struct Fp8LinearScales {
    input: GraphTensor,
    weight: GraphTensor,
}

fn linear_weight(
    cx: &mut Graph,
    prefix: impl ToString,
    shape: impl luminal::prelude::ToShape,
    fp8: bool,
) -> GraphTensor {
    let tensor = cx.named_tensor(format!("{}.weight", prefix.to_string()), shape);
    if fp8 {
        tensor.as_dtype(DType::F8E4M3).persist()
    } else {
        tensor.persist()
    }
}

fn fp8_linear_scales(cx: &mut Graph, prefix: impl ToString, fp8: bool) -> Option<Fp8LinearScales> {
    if !fp8 {
        return None;
    }
    let prefix = prefix.to_string();
    Some(Fp8LinearScales {
        input: cx
            .named_tensor(format!("{prefix}.input_scale"), ())
            .persist(),
        weight: cx
            .named_tensor(format!("{prefix}.weight_scale"), ())
            .persist(),
    })
}

fn expand_scalar(scale: GraphTensor, like: GraphTensor) -> GraphTensor {
    scale.expand_rhs(like.dims())
}

fn linear_matmul(
    input: GraphTensor,
    weight: GraphTensor,
    scales: Option<Fp8LinearScales>,
) -> GraphTensor {
    if let Some(scales) = scales {
        let input_scale = expand_scalar(scales.input, input);
        let scaled_input = input / input_scale;
        let output = scaled_input
            .cast(DType::F8E4M3)
            .matmul(weight.t())
            .cast(DType::F32);
        let output_scale = expand_scalar(scales.input * scales.weight, output);
        output * output_scale
    } else {
        input.matmul(weight.t())
    }
}

fn llama_rotary_embeddings(
    mut input: GraphTensor,
    pos_ids: GraphTensor,
    config: LlamaConfig,
) -> GraphTensor {
    input = input.split_dims(1, config.head_dim).transpose(0, 1);

    let freqs = input
        .graph()
        .arange_options(0, config.head_dim, 2)
        .cast(DType::F32)
        / config.head_dim as f32;
    let inv_freqs = 500_000_f32.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    let x0 = input.slice((.., .., ..config.head_dim / 2));
    let x1 = input.slice((.., .., config.head_dim / 2..));

    let cos = emb.cos().expand_dim(0, x0.dims()[0]);
    let sin = emb.sin().expand_dim(0, x0.dims()[0]);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

struct AttentionInputs {
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    attn_mask: GraphTensor,
}

fn attention(
    AttentionInputs {
        q_rope,
        k_rope,
        v,
        k_cache,
        v_cache,
        scatter_idx,
        gather_idx,
        attn_mask,
    }: AttentionInputs,
    config: LlamaConfig,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = config.kv_dim();
    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, kv_dim);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, kv_dim);

    let k = gather_rows(k_cache_out, gather_idx, kv_dim);
    let v_ctx = gather_rows(v_cache_out, gather_idx, kv_dim);

    let q = (q_rope * 1.0)
        .split_dims(1, config.head_dim)
        .transpose(0, 1);
    let k = k.split_dims(1, config.head_dim).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, config.head_dim).transpose(0, 1);

    let k = k.expand_dim(1, config.kv_groups).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, config.kv_groups).merge_dims(0, 1) * 1.0;

    let scores = q.matmul(k) / (config.head_dim as f32).sqrt();
    let masked_scores = scores + attn_mask.expand_dim(0, config.n_heads());
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);

    (attn_out, k_cache_out, v_cache_out)
}

impl LlamaLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        mut x: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let x_attn = self.attn_rms.forward(x);
        let q = linear_matmul(x_attn, self.q_proj, self.q_proj_scales);
        let k = linear_matmul(x_attn, self.k_proj, self.k_proj_scales);
        let v = linear_matmul(x_attn, self.v_proj, self.v_proj_scales);

        let q_rope = llama_rotary_embeddings(q, q_pos, self.config);
        let k_rope = llama_rotary_embeddings(k, q_pos, self.config);

        let (attn_out, k_cache_out, v_cache_out) = attention(
            AttentionInputs {
                q_rope,
                k_rope,
                v,
                k_cache,
                v_cache,
                scatter_idx,
                gather_idx,
                attn_mask,
            },
            self.config,
        );
        x += linear_matmul(attn_out, self.o_proj, self.o_proj_scales);

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out = linear_matmul(x_mlp, self.gate, self.gate_scales).swish()
            * linear_matmul(x_mlp, self.up, self.up_scales);
        let mlp_out = linear_matmul(mlp_out, self.down, self.down_scales);
        (x + mlp_out, k_cache_out, v_cache_out)
    }

    #[allow(dead_code)]
    fn parameter_tensors(&self) -> Vec<GraphTensor> {
        let mut tensors = vec![
            self.up,
            self.gate,
            self.down,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
        ];
        for scales in [
            self.up_scales,
            self.gate_scales,
            self.down_scales,
            self.q_proj_scales,
            self.k_proj_scales,
            self.v_proj_scales,
            self.o_proj_scales,
        ]
        .into_iter()
        .flatten()
        {
            tensors.push(scales.input);
            tensors.push(scales.weight);
        }
        if let Some(weight) = self.attn_rms.weight {
            tensors.push(weight);
        }
        if let Some(bias) = self.attn_rms.bias {
            tensors.push(bias);
        }
        if let Some(weight) = self.mlp_rms.weight {
            tensors.push(weight);
        }
        if let Some(bias) = self.mlp_rms.bias {
            tensors.push(bias);
        }
        tensors
    }
}
