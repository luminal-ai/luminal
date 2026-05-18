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

    pub fn init_with_config(cx: &mut Graph, config: LlamaConfig) -> Self {
        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            layers.push(LlamaLayer {
                config,
                up: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.up_proj.weight"),
                        (config.intermediate, config.hidden),
                    )
                    .persist(),
                gate: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.gate_proj.weight"),
                        (config.intermediate, config.hidden),
                    )
                    .persist(),
                down: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.down_proj.weight"),
                        (config.hidden, config.intermediate),
                    )
                    .persist(),
                q_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.q_proj.weight"),
                        (config.hidden, config.hidden),
                    )
                    .persist(),
                k_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.k_proj.weight"),
                        (config.kv_dim(), config.hidden),
                    )
                    .persist(),
                v_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.v_proj.weight"),
                        (config.kv_dim(), config.hidden),
                    )
                    .persist(),
                o_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.o_proj.weight"),
                        (config.hidden, config.hidden),
                    )
                    .persist(),
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
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
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
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

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
        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
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
