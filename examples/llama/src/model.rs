use luminal::{
    dtype::DType,
    graph::Graph,
    prelude::{F32Pow, GraphTensor},
    shape::Expression,
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
pub enum LlamaPrecision {
    /// F32 weights and activations. Used only by the search-equivalence fuzz
    /// test fixture (`init_with_config`); the example ships bf16/fp8.
    #[allow(dead_code)]
    F32,
    /// Bf16 weights and activations; norms computed in F32 via explicit
    /// casts, logits cast to F32 at the head.
    Bf16,
    /// F8E4M3 linear weights with Bf16 activations, KV cache and attention.
    /// Linears run x.cast(F32)/scale → cast(F8) → matmul → cast(F32) × scale
    /// → cast(Bf16); everything else matches the Bf16 pipeline.
    Fp8,
}

impl LlamaPrecision {
    fn weight_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Fp8 => DType::F8E4M3,
            Self::Bf16 => DType::Bf16,
        }
    }

    fn act_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 | Self::Fp8 => DType::Bf16,
        }
    }

    fn is_fp8(self) -> bool {
        matches!(self, Self::Fp8)
    }
}

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
    pub fn new_dtype(cx: &mut Graph, num_slots: usize, dtype: DType) -> Self {
        Self::new_with_config_dtype(cx, num_slots, LlamaConfig::default(), dtype)
    }

    // Used by the cuda_lite search-equivalence fuzz (which path-includes this
    // model); the example binary itself only calls `new_dtype`.
    #[allow(dead_code)]
    pub fn new_with_config(cx: &mut Graph, num_slots: usize, config: LlamaConfig) -> Self {
        Self::new_with_config_dtype(cx, num_slots, config, DType::F32)
    }

    pub fn new_with_config_dtype(
        cx: &mut Graph,
        num_slots: usize,
        config: LlamaConfig,
        dtype: DType,
    ) -> Self {
        let kv_dim = config.kv_dim();
        let mut k_caches = Vec::with_capacity(config.layers);
        let mut v_caches = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            k_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.k"), (num_slots, kv_dim))
                    .as_dtype(dtype),
            );
            v_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.v"), (num_slots, kv_dim))
                    .as_dtype(dtype),
            );
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
    precision: LlamaPrecision,
    embedding: GraphTensor,
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
    lm_head: GraphTensor,
}

impl Llama {
    pub fn init_fp8(cx: &mut Graph) -> Self {
        Self::init_with_precision(cx, LlamaConfig::default(), LlamaPrecision::Fp8)
    }

    pub fn init_bf16(cx: &mut Graph) -> Self {
        Self::init_with_precision(cx, LlamaConfig::default(), LlamaPrecision::Bf16)
    }

    /// F32 reference build at a custom config. The example ships bf16/fp8 only;
    /// this is retained as the numerically-stable fixture for the backend
    /// search-equivalence fuzz test (`luminal_cuda_lite`), which references this
    /// module via `#[path]`.
    #[allow(dead_code)]
    pub fn init_with_config(cx: &mut Graph, config: LlamaConfig) -> Self {
        Self::init_with_precision(cx, config, LlamaPrecision::F32)
    }

    pub fn init_with_precision(
        cx: &mut Graph,
        config: LlamaConfig,
        precision: LlamaPrecision,
    ) -> Self {
        // Embedding and lm_head share the activation dtype (bf16 table rows
        // feed the layer stack directly); norm weights stay F32.
        let table_dtype = precision.act_dtype();
        Self {
            config,
            precision,
            embedding: persist(
                cx,
                "model.embed_tokens.weight",
                (config.vocab_size, config.hidden),
            )
            .as_dtype(table_dtype),
            layers: (0..config.layers)
                .map(|l| LlamaLayer::init(cx, l, config, precision))
                .collect(),
            lm_head: persist(cx, "lm_head.weight", (config.vocab_size, config.hidden))
                .as_dtype(table_dtype),
            lm_norm: rms_norm(cx, config.hidden, "model.norm.weight"),
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(
        &self,
        input: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let mut x = token_embedding(self.embedding, input, self.config.hidden);
        let mut cache_outputs = Vec::with_capacity(self.config.layers);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                q_pos,
                scatter_idx,
                gather_idx,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }
        let normed = norm_in_f32(&self.lm_norm, x, self.precision.act_dtype());
        let mut logits = normed.matmul(self.lm_head.t());
        if logits.dtype != DType::F32 {
            logits = logits.cast(DType::F32);
        }
        (logits, cache_outputs)
    }

    /// Forward pass plus GPU-side greedy sampling with repetition penalty.
    ///
    /// `seen_mask` is a persistent `(vocab,)` F32 buffer of 0/1 flags;
    /// `new_token` is a `(1,)` Int holding the previously sampled token id
    /// (or -1 before the first sample — the scatter bounds-check skips it).
    /// The mask update is ordered BEFORE the penalty read, matching the CPU
    /// sampler (token inserted right after sampling): penalty at step N sees
    /// every token sampled through step N-1. Because the scatter is the only
    /// reader of `seen_mask`, it updates the buffer in place, so no host-side
    /// buffer swap is needed.
    ///
    /// Returns `(token_ids (s,) Int, seen_out, cache_outputs)` — the host
    /// reads one i32 per row instead of a vocab-sized logit row.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_sampling(
        &self,
        input: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        kv_cache: &KVCache,
        seen_mask: GraphTensor,
        new_token: GraphTensor,
        repetition_penalty: f32,
    ) -> (GraphTensor, GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let (logits, cache_outputs) = self.forward(input, q_pos, scatter_idx, gather_idx, kv_cache);
        let cx = unsafe { &mut *logits.graph_ref };
        let s = logits.dims()[0];
        let vocab = self.config.vocab_size;

        // seen[new_token] = 1.0 (in place; no-op for -1).
        let one = cx.constant_float(1.0).expand_dim(0, 1);
        let seen_out = one.scatter(new_token, seen_mask);

        // CPU-equivalent penalty: seen & logit > 0 → /p, seen & logit <= 0 → *p.
        let p = repetition_penalty;
        let seen_b = seen_out.expand_dim(0, s);
        let zero = cx.constant_float(0.0).expand_rhs(logits.dims());
        let pos = zero.lt(logits).cast(DType::F32);
        let factor = seen_b * (pos * (1.0 / p - 1.0) + (-pos + 1.0) * (p - 1.0)) + 1.0;
        let penalized = logits * factor;

        let _ = vocab;
        let token_ids = penalized.argmax(1);
        (token_ids, seen_out, cache_outputs)
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
    precision: LlamaPrecision,
    /// Fused [v; q; k] projection, 16-bit pipeline only.
    qkv: Option<GraphTensor>,
    /// Calibration scales for the fused projection (fp8 weights only).
    qkv_scales: Option<Fp8LinearScales>,
    /// Fused [gate; up] projection, 16-bit pipeline only.
    gate_up: Option<GraphTensor>,
    /// Calibration scales for the fused projection (fp8 weights only).
    gate_up_scales: Option<Fp8LinearScales>,
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

impl LlamaLayer {
    fn init(cx: &mut Graph, l: usize, config: LlamaConfig, precision: LlamaPrecision) -> Self {
        let fp8 = precision.is_fp8();
        // The 16-bit pipelines use fused projections (one GEMM for [v;q;k],
        // one for [gate;up]) written by the weight converters. RoPE reads q/k
        // head groups straight out of the fused QKV row via pitch/offset.
        // The fp8 converter requantizes the parts to one shared per-fusion
        // scale, so the fused weight carries a single scale pair.
        let fused = matches!(precision, LlamaPrecision::Bf16 | LlamaPrecision::Fp8);
        let (qkv, gate_up) = if fused {
            (
                Some(layer_linear_weight(
                    cx,
                    l,
                    "self_attn.vqk_proj",
                    (config.hidden + 2 * config.kv_dim(), config.hidden),
                    precision,
                )),
                Some(layer_linear_weight(
                    cx,
                    l,
                    "mlp.gate_up_proj",
                    (2 * config.intermediate, config.hidden),
                    precision,
                )),
            )
        } else {
            (None, None)
        };
        let unfused_weight = |cx: &mut Graph, suffix: &str, shape: (usize, usize)| {
            if fused {
                // Placeholder: unnamed, unreachable, never loaded.
                cx.tensor((1, 1)).as_dtype(precision.weight_dtype())
            } else {
                layer_linear_weight(cx, l, suffix, shape, precision)
            }
        };
        Self {
            config,
            precision,
            qkv,
            qkv_scales: layer_linear_scales(cx, l, "self_attn.vqk_proj", fp8 && fused),
            gate_up,
            gate_up_scales: layer_linear_scales(cx, l, "mlp.gate_up_proj", fp8 && fused),
            up: unfused_weight(cx, "mlp.up_proj", (config.intermediate, config.hidden)),
            up_scales: layer_linear_scales(cx, l, "mlp.up_proj", fp8 && !fused),
            gate: unfused_weight(cx, "mlp.gate_proj", (config.intermediate, config.hidden)),
            gate_scales: layer_linear_scales(cx, l, "mlp.gate_proj", fp8 && !fused),
            down: layer_linear_weight(
                cx,
                l,
                "mlp.down_proj",
                (config.hidden, config.intermediate),
                precision,
            ),
            down_scales: layer_linear_scales(cx, l, "mlp.down_proj", fp8),
            q_proj: unfused_weight(cx, "self_attn.q_proj", (config.hidden, config.hidden)),
            q_proj_scales: layer_linear_scales(cx, l, "self_attn.q_proj", fp8 && !fused),
            k_proj: unfused_weight(cx, "self_attn.k_proj", (config.kv_dim(), config.hidden)),
            k_proj_scales: layer_linear_scales(cx, l, "self_attn.k_proj", fp8 && !fused),
            v_proj: unfused_weight(cx, "self_attn.v_proj", (config.kv_dim(), config.hidden)),
            v_proj_scales: layer_linear_scales(cx, l, "self_attn.v_proj", fp8 && !fused),
            o_proj: layer_linear_weight(
                cx,
                l,
                "self_attn.o_proj",
                (config.hidden, config.hidden),
                precision,
            ),
            o_proj_scales: layer_linear_scales(cx, l, "self_attn.o_proj", fp8),
            attn_rms: rms_norm(
                cx,
                config.hidden,
                format!("model.layers.{l}.input_layernorm.weight"),
            ),
            mlp_rms: rms_norm(
                cx,
                config.hidden,
                format!("model.layers.{l}.post_attention_layernorm.weight"),
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        mut x: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let act = self.precision.act_dtype();
        // The fp8 fused path quantizes inside the norm (and SwiGLU) kernels,
        // so the plain bf16 norm output only exists on the other paths.
        let x_attn = if self.qkv_scales.is_some() {
            None
        } else {
            Some(norm_in_f32(&self.attn_rms, x, act))
        };

        let (q_rope, k_rope, v) = if let Some(qkv) = self.qkv {
            // Fused path: one GEMV for [q;k;v]; rope reads the q/k head
            // groups straight out of the fused row, v is a strided slice
            // consumed by the cache scatter.
            let hidden = self.config.hidden;
            let kv_dim = self.config.kv_dim();
            // Angles are built per layer so the layer bodies stay textually
            // identical for loop rolling (cross-layer references to computed
            // nodes break the roller's window hashing; references to Input
            // nodes like q_pos are fine). The e-graph CSEs the duplicates.
            // Layout [v | q | k]: v is a start-0 strided view; q/k are
            // offset slices (lowered to Gather materializations) fed to the
            // pure-HLIR rotary chain, which the KernelRoPE rule re-fuses.
            let xqkv = if let Some(scales) = self.qkv_scales {
                let x_norm = norm_in_f32(&self.attn_rms, x, act);
                let xq = quant_f8(x_norm, scales.input);
                linear_matmul_prequant(xq, scales.input, qkv, scales.weight, act)
            } else {
                linear_matmul(
                    x_attn.expect("non-fp8 fused path computes the norm"),
                    qkv,
                    None,
                )
            };
            let v = xqkv.slice((.., ..kv_dim));
            let q = xqkv.slice((.., kv_dim..kv_dim + hidden));
            let k = xqkv.slice((.., kv_dim + hidden..));
            let q_rope = llama_rotary_embeddings(q, q_pos, self.config);
            let k_rope = llama_rotary_embeddings(k, q_pos, self.config);
            (q_rope, k_rope, v)
        } else {
            let x_attn = x_attn.expect("unfused path computes the norm");
            let q = linear_matmul(x_attn, self.q_proj, self.q_proj_scales);
            let k = linear_matmul(x_attn, self.k_proj, self.k_proj_scales);
            let v = linear_matmul(x_attn, self.v_proj, self.v_proj_scales);
            // Pure-HLIR rotary for every precision; the KernelRoPE rule
            // re-fuses the bf16 chain into one kernel.
            let q_rope = llama_rotary_embeddings(q, q_pos, self.config);
            let k_rope = llama_rotary_embeddings(k, q_pos, self.config);
            (q_rope, k_rope, v)
        };

        let (mut attn_out, k_cache_out, v_cache_out) = attention(
            AttentionInputs {
                q_rope,
                k_rope,
                v,
                k_cache,
                v_cache,
                scatter_idx,
                gather_idx,
                q_pos,
            },
            self.config,
        );
        if act != DType::F32 {
            // The attention output is a transpose+merge view whose symbolic
            // strides the cuBLASLt rewrites can't match, dropping the
            // o-projection onto the generic matmul fallback (~38µs vs ~8µs
            // measured). A `* 1.0` materialization barrier (one ~2µs fused
            // region) makes it contiguous so cuBLASLt matches.
            attn_out *= 1.0;
        }
        x += linear_matmul(attn_out, self.o_proj, self.o_proj_scales);

        let intermediate = self.config.intermediate;
        let mlp_out = if let (Some(gate_up), Some(scales)) = (self.gate_up, self.gate_up_scales) {
            // fp8 MLP in pure HLIR: norm → quant → gate_up GEMM → swiglu →
            // quant → down GEMM. The quant chains match KernelQuantF8 /
            // cuBLASLt-scaled rules.
            let x_mlp = norm_in_f32(&self.mlp_rms, x, act);
            let xq = quant_f8(x_mlp, scales.input);
            let xgu = linear_matmul_prequant(xq, scales.input, gate_up, scales.weight, act);
            let gate = xgu.slice((.., ..intermediate));
            let up = xgu.slice((.., intermediate..));
            let hidden_act = gate.swish() * up;
            let down_scales = self
                .down_scales
                .expect("fp8 fused path always has down-projection scales");
            let gq = quant_f8(hidden_act, down_scales.input);
            linear_matmul_prequant(gq, down_scales.input, self.down, down_scales.weight, act)
        } else {
            let x_mlp = norm_in_f32(&self.mlp_rms, x, act);
            let mlp_out = if let Some(gate_up) = self.gate_up {
                let xgu = linear_matmul(x_mlp, gate_up, self.gate_up_scales);
                let gate = xgu.slice((.., ..intermediate));
                let up = xgu.slice((.., intermediate..));
                gate.swish() * up
            } else {
                linear_matmul(x_mlp, self.gate, self.gate_scales).swish()
                    * linear_matmul(x_mlp, self.up, self.up_scales)
            };
            linear_matmul(mlp_out, self.down, self.down_scales)
        };
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

fn persist(
    cx: &mut Graph,
    name: impl ToString,
    shape: impl luminal::prelude::ToShape,
) -> GraphTensor {
    cx.named_tensor(name, shape).persist()
}

fn linear_weight(
    cx: &mut Graph,
    prefix: impl ToString,
    shape: impl luminal::prelude::ToShape,
    precision: LlamaPrecision,
) -> GraphTensor {
    cx.named_tensor(format!("{}.weight", prefix.to_string()), shape)
        .as_dtype(precision.weight_dtype())
        .persist()
}

fn layer_linear_weight(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    shape: impl luminal::prelude::ToShape,
    precision: LlamaPrecision,
) -> GraphTensor {
    linear_weight(
        cx,
        format!("model.layers.{layer}.{suffix}"),
        shape,
        precision,
    )
}

/// Norms always compute in F32 per the dtype contract. The F32 pipeline uses
/// the decomposed spelling; the 16-bit pipeline calls the fused RMSNorm
/// kernel (16-bit rows in/out, F32 accumulation and weights inside — the
/// same semantics the explicit cast → std_norm → cast spelling expresses,
/// in one launch instead of ~7).
fn norm_in_f32(norm: &LayerNorm, x: GraphTensor, act: DType) -> GraphTensor {
    if act == DType::F32 {
        norm.forward(x)
    } else {
        // Pure-HLIR F32 norm sandwich — the KernelRMSNorm egglog rule
        // matches this chain and fuses it back into one kernel.
        norm.forward(x.cast(DType::F32)).cast(act)
    }
}

/// Pure-HLIR per-linear fp8 quantization: `cast_f8(x_f32 / scale)` — the
/// spelling the KernelQuantF8 / cuBLASLt-scaled rules match.
fn quant_f8(x: GraphTensor, scale: GraphTensor) -> GraphTensor {
    let x_f32 = if x.dtype == DType::F32 {
        x
    } else {
        x.cast(DType::F32)
    };
    let scale = expand_scalar(scale, x_f32);
    (x_f32 / scale).cast(DType::F8E4M3)
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

fn layer_linear_scales(
    cx: &mut Graph,
    layer: usize,
    suffix: &str,
    fp8: bool,
) -> Option<Fp8LinearScales> {
    fp8_linear_scales(cx, format!("model.layers.{layer}.{suffix}"), fp8)
}

const RMS_NORM_EPS: f32 = 1e-5;

fn rms_norm(cx: &mut Graph, dim: usize, weight_name: impl ToString) -> LayerNorm {
    LayerNorm::new(
        dim,
        Some(&weight_name.to_string()),
        None,
        false,
        RMS_NORM_EPS,
        cx,
    )
}

fn token_embedding(embedding: GraphTensor, token_ids: GraphTensor, hidden: usize) -> GraphTensor {
    let seq = token_ids.dims1();
    embedding.gather(
        (token_ids * hidden).expand_dim(1, hidden)
            + token_ids.graph().arange(hidden).expand_dim(0, seq),
    )
}

fn expand_scalar(scale: GraphTensor, like: GraphTensor) -> GraphTensor {
    scale.expand_rhs(like.dims())
}

/// GEMM + dequant for an activation already quantized to F8E4M3 by a fused
/// custom op (norm+quant or swiglu+quant) carrying `in_scale`. Spelled with
/// the exact dequant chain the scaled-fp8 GEMM rewrites match:
/// `Cast(Bf16)(Cast(F32)(q @ W^T) * (in_scale * weight_scale))`.
fn linear_matmul_prequant(
    q: GraphTensor,
    in_scale: GraphTensor,
    weight: GraphTensor,
    weight_scale: GraphTensor,
    act: DType,
) -> GraphTensor {
    let out = q.matmul(weight.t()).cast(DType::F32);
    let output_scale = expand_scalar(in_scale * weight_scale, out);
    let out = out * output_scale;
    if act == DType::F32 {
        out
    } else {
        out.cast(act)
    }
}

fn linear_matmul(
    input: GraphTensor,
    weight: GraphTensor,
    scales: Option<Fp8LinearScales>,
) -> GraphTensor {
    if let Some(scales) = scales {
        let act = input.dtype;
        let input = if act == DType::F32 {
            input
        } else {
            input.cast(DType::F32)
        };
        let input_scale = expand_scalar(scales.input, input);
        let scaled_input = input / input_scale;
        let output = scaled_input
            .cast(DType::F8E4M3)
            .matmul(weight.t())
            .cast(DType::F32);
        let output_scale = expand_scalar(scales.input * scales.weight, output);
        let output = output * output_scale;
        if act == DType::F32 {
            output
        } else {
            output.cast(act)
        }
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

    // RoPE angles are computed in F32 for accuracy; cast to the activation
    // dtype before the elementwise rotation so the kernels stay uniform.
    let mut cos = emb.cos();
    let mut sin = emb.sin();
    if x0.dtype != DType::F32 {
        cos = cos.cast(x0.dtype);
        sin = sin.cast(x0.dtype);
    }
    let cos = cos.expand_dim(0, x0.dims()[0]);
    let sin = sin.expand_dim(0, x0.dims()[0]);
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
    q_pos: GraphTensor,
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
        q_pos,
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
    let ctx = Expression::from('c');
    let causal_square = scores.graph().triu(ctx, 1).cast(scores.dtype) * -1e10;
    let row_offsets = (q_pos * ctx).expand_dim(1, ctx);
    let col_offsets = scores
        .graph()
        .arange(ctx)
        .expand_dim(0, Expression::from('s'));
    let attn_mask = causal_square.gather(row_offsets + col_offsets);
    let masked_scores = scores + attn_mask.expand_dim(0, config.n_heads());
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);

    (attn_out, k_cache_out, v_cache_out)
}
