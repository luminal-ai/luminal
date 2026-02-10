use luminal::{
    graph::Graph,
    op::{CustomOp, LLIROp},
    prelude::GraphTensor,
    shape::{flatten_mul_strides, Expression, ToShape},
};
use luminal_cuda::{
    block::{cstruct::CStruct, BlockOp},
    cudarc::driver::{CudaSlice, CudaStream, DevicePtr},
};
use luminal_nn::LayerNorm;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::{fmt::Debug, path::Path, sync::Arc};

// Qwen3-4B hyperparams
pub const LAYERS: usize = 36;
pub const HIDDEN: usize = 2560;
pub const INTERMEDIATE: usize = 9728;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // = 4
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // = 4096
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // = 1024
pub const VOCAB_SIZE: usize = 151936;
pub const RMS_NORM_EPS: f32 = 1e-6;

pub struct Qwen {
    pub embedding: GraphTensor,
    layers: Vec<QwenLayer>,
    lm_norm: LayerNorm,
}

impl Qwen {
    pub fn init(cx: &mut Graph) -> Self {
        let mut w = vec![];
        for l in 0..LAYERS {
            w.push(QwenLayer {
                up: cx.named_tensor(
                    format!("model.layers.{l}.mlp.up_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                gate: cx.named_tensor(
                    format!("model.layers.{l}.mlp.gate_proj.weight"),
                    (INTERMEDIATE, HIDDEN),
                ),
                down: cx.named_tensor(
                    format!("model.layers.{l}.mlp.down_proj.weight"),
                    (HIDDEN, INTERMEDIATE),
                ),
                q_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.q_proj.weight"),
                    (Q_DIM, HIDDEN),
                ),
                k_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.k_proj.weight"),
                    (KV_DIM, HIDDEN),
                ),
                v_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.v_proj.weight"),
                    (KV_DIM, HIDDEN),
                ),
                o_proj: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.o_proj.weight"),
                    (HIDDEN, Q_DIM),
                ),
                attn_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
            });
        }
        let lm_norm = LayerNorm::new(
            HIDDEN,
            Some("model.norm.weight"),
            None,
            false,
            RMS_NORM_EPS,
            cx,
        );
        Self {
            embedding: cx.named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            layers: w,
            lm_norm,
        }
    }

    pub fn forward(
        &self,
        token_ids: GraphTensor,
        kv_cache: &KVCache,
        norm_bufs: &NormWeightBuffers,
    ) -> GraphTensor {
        let batch = token_ids.dims1();
        let mut x = self.embedding.gather(
            (token_ids * HIDDEN).expand_dim(1, HIDDEN)
                + token_ids.graph().arange(HIDDEN).expand_dim(0, batch),
        );
        for (layer, ((k_cache, v_cache), (q_norm_buf, k_norm_buf))) in self
            .layers
            .iter()
            .zip(kv_cache.layers.iter().zip(norm_bufs.layers.iter()))
        {
            x = layer
                .forward(
                    x,
                    k_cache.device_ptr(v_cache.stream()).0,
                    v_cache.device_ptr(k_cache.stream()).0,
                    q_norm_buf.device_ptr(k_cache.stream()).0,
                    k_norm_buf.device_ptr(k_cache.stream()).0,
                )
                .graph_break();
        }
        self.lm_norm.forward(x).matmul(self.embedding.t())
    }
}

struct QwenLayer {
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

impl QwenLayer {
    pub fn forward(
        &self,
        mut x: GraphTensor,
        k_cache: u64,
        v_cache: u64,
        q_norm_ptr: u64,
        k_norm_ptr: u64,
    ) -> GraphTensor {
        let x_attn = self.attn_rms.forward(x);
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        // 3 graph inputs (Q, K, V) + norm weights via payload pointers
        let attn_out = x.graph().custom_op(
            QwenAttention::new(
                k_cache,
                v_cache,
                q_norm_ptr,
                k_norm_ptr,
                q.dims()[0],
                'p'.into(),
            ),
            (q, k, v),
            q.shape,
            q.dtype,
        );
        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
        x + mlp_out
    }
}

// ---------------------------------------------------------------------------
// KV Cache + Norm Weight Buffers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct KVCache {
    pub layers: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
}

impl KVCache {
    pub fn new(stream: &Arc<CudaStream>, capacity: usize) -> Self {
        Self {
            layers: (0..LAYERS)
                .map(|_| {
                    (
                        stream
                            .alloc_zeros(N_KV_HEADS * HEAD_DIM * capacity * size_of::<f32>())
                            .unwrap(),
                        stream
                            .alloc_zeros(N_KV_HEADS * HEAD_DIM * capacity * size_of::<f32>())
                            .unwrap(),
                    )
                })
                .collect(),
        }
    }

    pub fn reset(&mut self) {
        for (k_cache, v_cache) in &mut self.layers {
            v_cache.stream().memset_zeros(k_cache).unwrap();
            k_cache.stream().memset_zeros(v_cache).unwrap();
        }
    }
}

/// Pre-allocated GPU buffers for QK-norm weights per layer.
/// Weights are copied here from the runtime after loading safetensors.
#[derive(Debug, Clone)]
pub struct NormWeightBuffers {
    pub layers: Vec<(CudaSlice<u8>, CudaSlice<u8>)>, // (q_norm, k_norm) per layer
}

impl NormWeightBuffers {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            layers: (0..LAYERS)
                .map(|_| {
                    (
                        stream.alloc_zeros(HEAD_DIM * size_of::<f32>()).unwrap(),
                        stream.alloc_zeros(HEAD_DIM * size_of::<f32>()).unwrap(),
                    )
                })
                .collect(),
        }
    }

    /// Load QK-norm weights directly from a safetensors file into GPU buffers.
    pub fn load_from_safetensors(&mut self, weights_path: &Path) {
        let f = std::fs::File::open(weights_path).unwrap();
        let mmap = unsafe { MmapOptions::new().map(&f).unwrap() };
        let st = SafeTensors::deserialize(&mmap).unwrap();
        for i in 0..LAYERS {
            let q_name = format!("model.layers.{i}.self_attn.q_norm.weight");
            let k_name = format!("model.layers.{i}.self_attn.k_norm.weight");
            let q_data = st.tensor(&q_name).unwrap().data();
            let k_data = st.tensor(&k_name).unwrap().data();
            let stream = self.layers[i].0.stream().clone();
            stream.memcpy_htod(q_data, &mut self.layers[i].0).unwrap();
            stream.memcpy_htod(k_data, &mut self.layers[i].1).unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// QwenAttention: Fused QK-Norm + RoPE + Causal Attention with KV Cache
// ---------------------------------------------------------------------------
// Graph inputs: Q (seq, Q_DIM), K (seq, KV_DIM), V (seq, KV_DIM)
// Payload pointers: k_cache, v_cache, q_norm_weights, k_norm_weights
// Output: (seq, Q_DIM) - attention output

#[derive(Debug, Clone)]
pub struct QwenAttention {
    range: Vec<Expression>,
    head_dim: Expression,
    cur_seq: Expression,
    kv_row_stride: Expression,
    q_stride: Vec<Expression>,
    k_stride: Vec<Expression>,
    v_stride: Vec<Expression>,
    o_stride: Vec<Expression>,
    prev_seq: Expression,
    k_cache: u64,
    v_cache: u64,
    q_norm_ptr: u64,
    k_norm_ptr: u64,
}

impl QwenAttention {
    fn new(
        k_cache: u64,
        v_cache: u64,
        q_norm_ptr: u64,
        k_norm_ptr: u64,
        seq: Expression,
        prev_seq: Expression,
    ) -> Self {
        Self {
            range: (N_KV_HEADS, KV_GROUPS, seq).to_shape(),
            head_dim: HEAD_DIM.into(),
            cur_seq: seq,
            kv_row_stride: KV_DIM.into(),
            q_stride: (HEAD_DIM * KV_GROUPS, HEAD_DIM, Q_DIM).to_shape(),
            k_stride: (HEAD_DIM, 0, 0).to_shape(),
            v_stride: (HEAD_DIM, 0, 0).to_shape(),
            o_stride: (HEAD_DIM * KV_GROUPS, HEAD_DIM, Q_DIM).to_shape(),
            prev_seq,
            k_cache,
            v_cache,
            q_norm_ptr,
            k_norm_ptr,
        }
    }
}

impl CustomOp for QwenAttention {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for QwenAttention {
    fn op_name(&self) -> &'static str {
        "QwenAttention"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.head_dim
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        let mut q = vec![true; self.range.len()];
        q[self.range.len() - 1] = false;
        let mut k = vec![true; self.range.len()];
        k[self.range.len() - 1] = false;
        let mut v = vec![true; self.range.len()];
        v[self.range.len() - 1] = false;
        vec![q, k, v]
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        let mut q_pos_stride = vec![0.into(); self.range.len()];
        q_pos_stride[self.range.len() - 1] = 1.into();
        let mut group_pos_stride = vec![0.into(); self.range.len()];
        group_pos_stride[self.range.len() - 2] = 1.into();
        let mut head_pos_stride = vec![0.into(); self.range.len()];
        head_pos_stride[self.range.len() - 3] = 1.into();
        payload
            .expr("head_size", self.head_dim)
            .expr("cur_seq", self.cur_seq)
            .expr("kv_row_stride", self.kv_row_stride)
            .expr("q", flatten_mul_strides(&self.range, &self.q_stride))
            .expr("k", flatten_mul_strides(&self.range, &self.k_stride))
            .expr("v", flatten_mul_strides(&self.range, &self.v_stride))
            .expr("out", flatten_mul_strides(&self.range, &self.o_stride))
            .ptr_mut_f32("key_cache", self.k_cache as *mut f32)
            .ptr_mut_f32("val_cache", self.v_cache as *mut f32)
            .ptr_const_f32("q_norm_weights", self.q_norm_ptr as *const f32)
            .ptr_const_f32("k_norm_weights", self.k_norm_ptr as *const f32)
            .expr("prev_seq", self.prev_seq)
            .expr(
                "q_pos_stride",
                flatten_mul_strides(&self.range, &q_pos_stride),
            )
            .expr(
                "group_pos_stride",
                flatten_mul_strides(&self.range, &group_pos_stride),
            )
            .expr(
                "head_pos_stride",
                flatten_mul_strides(&self.range, &head_pos_stride),
            )
    }

    fn cuda_function(&self) -> String {
        "
            __shared__ float shared[32];
            __shared__ float q_buf[128];
            __shared__ float k_buf[128];
            __shared__ float rnorm_s;

            auto warp_reduce_sum = [](float val) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                return val;
            };

            auto block_reduce_sum = [&](float val) {
                int lane = threadIdx.x & 31;
                int wid  = threadIdx.x >> 5;
                val = warp_reduce_sum(val);
                if (lane == 0) shared[wid] = val;
                __syncthreads();
                val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
                if (wid == 0) val = warp_reduce_sum(val);
                return val;
            };

            // 3 graph inputs
            const float* q_raw = source_ptrs[0] + eval_expression(payload.q, current);
            const float* k_base = source_ptrs[1] + eval_expression(payload.k, current);
            const float* v_base = source_ptrs[2] + eval_expression(payload.v, current);
            // Norm weights from payload (pre-allocated GPU buffers)
            const float* q_weights = payload.q_norm_weights;
            const float* k_weights = payload.k_norm_weights;

            float* out = out_ptr + eval_expression(payload.out, current);
            int q_pos_local = eval_expression(payload.q_pos_stride, current);
            const int group_pos_local = eval_expression(payload.group_pos_stride, current);
            const int head_pos_local = eval_expression(payload.head_pos_stride, current);

            const int d             = eval_expression(payload.head_size, 0);
            float* __restrict__ K_cache = payload.key_cache + head_pos_local * d;
            float* __restrict__ V_cache = payload.val_cache + head_pos_local * d;

            const int S             = eval_expression(payload.cur_seq, 0);
            const int kv_row_stride = eval_expression(payload.kv_row_stride, 0);
            const int prev          = eval_expression(payload.prev_seq, 0);

            const float* __restrict__ K_cur = k_base;
            const float* __restrict__ V_cur = v_base;
            float* __restrict__ O = out;

            if (q_pos_local >= S) q_pos_local = S - 1;
            if (q_pos_local < 0)  q_pos_local = 0;

            const int q_pos_total = prev + q_pos_local;
            const float scale = rsqrtf((float)d);

            const int half = d / 2;
            const float eps = 1e-6f;
            const float rope_base = 1000000.0f;

            // ================================================================
            // Step 1: QK-Norm + RoPE for this Q row
            // ================================================================
            {
                float sum_sq = 0.0f;
                for (int i = t; i < d; i += blockDim.x) {
                    float val = q_raw[i];
                    sum_sq += val * val;
                }
                sum_sq = block_reduce_sum(sum_sq);
                if (t == 0) rnorm_s = rsqrtf(sum_sq / (float)d + eps);
                __syncthreads();
                float rn = rnorm_s;

                for (int i = t; i < d; i += blockDim.x) {
                    q_buf[i] = q_raw[i] * rn * q_weights[i];
                }
                __syncthreads();

                int pos = prev + q_pos_local;
                for (int i = t; i < half; i += blockDim.x) {
                    float freq = powf(rope_base, -2.0f * (float)i / (float)d);
                    float theta = (float)pos * freq;
                    float cos_t, sin_t;
                    __sincosf(theta, &sin_t, &cos_t);
                    float x0 = q_buf[i];
                    float x1 = q_buf[i + half];
                    q_buf[i]        = x0 * cos_t - x1 * sin_t;
                    q_buf[i + half] = x1 * cos_t + x0 * sin_t;
                }
                __syncthreads();
            }

            // ================================================================
            // Step 2: First group writes K (norm+rope) + V to cache
            // ================================================================
            if (group_pos_local == 0) {
                for (int r = 0; r < S; ++r) {
                    const float* __restrict__ srcK = K_cur + r * kv_row_stride;
                    const float* __restrict__ srcV = V_cur + r * kv_row_stride;
                    float* __restrict__ dstK = K_cache + (prev + r) * kv_row_stride;
                    float* __restrict__ dstV = V_cache + (prev + r) * kv_row_stride;

                    // Copy V directly
                    for (int u = t; u < d; u += blockDim.x) {
                        dstV[u] = srcV[u];
                    }

                    // K: QK-Norm
                    float k_sum = 0.0f;
                    for (int u = t; u < d; u += blockDim.x) {
                        float val = srcK[u];
                        k_sum += val * val;
                    }
                    k_sum = block_reduce_sum(k_sum);
                    if (t == 0) rnorm_s = rsqrtf(k_sum / (float)d + eps);
                    __syncthreads();
                    float k_rn = rnorm_s;

                    for (int u = t; u < d; u += blockDim.x) {
                        k_buf[u] = srcK[u] * k_rn * k_weights[u];
                    }
                    __syncthreads();

                    // K: Split-half RoPE -> write to cache
                    int k_pos = prev + r;
                    for (int i = t; i < half; i += blockDim.x) {
                        float freq = powf(rope_base, -2.0f * (float)i / (float)d);
                        float theta = (float)k_pos * freq;
                        float cos_t, sin_t;
                        __sincosf(theta, &sin_t, &cos_t);
                        float kx0 = k_buf[i];
                        float kx1 = k_buf[i + half];
                        dstK[i]        = kx0 * cos_t - kx1 * sin_t;
                        dstK[i + half] = kx1 * cos_t + kx0 * sin_t;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();

            // ================================================================
            // Step 3: Online softmax attention
            //   rows < prev  : K from cache (normed+rotated), V from cache
            //   rows >= prev : K norm+rope on-the-fly from source, V from source
            // ================================================================

            __shared__ float att_m;
            __shared__ float att_corr;
            __shared__ float att_w;
            float att_d = 0.0f;

            for (int j = t; j < d; j += blockDim.x) {
                O[j] = 0.0f;
            }
            if (t == 0) att_m = -__int_as_float(0x7f800000);
            __syncthreads();

            for (int r = 0; r <= q_pos_total; ++r) {
                const float* __restrict__ k_row;
                const float* __restrict__ v_row;

                if (r < prev) {
                    k_row = K_cache + r * kv_row_stride;
                    v_row = V_cache + r * kv_row_stride;
                } else {
                    int r_local = r - prev;
                    const float* __restrict__ srcK = K_cur + r_local * kv_row_stride;
                    v_row = V_cur + r_local * kv_row_stride;

                    // K: QK-Norm on the fly
                    float k_sum = 0.0f;
                    for (int u = t; u < d; u += blockDim.x) {
                        float val = srcK[u];
                        k_sum += val * val;
                    }
                    k_sum = block_reduce_sum(k_sum);
                    if (t == 0) rnorm_s = rsqrtf(k_sum / (float)d + eps);
                    __syncthreads();
                    float k_rn = rnorm_s;

                    for (int u = t; u < d; u += blockDim.x) {
                        k_buf[u] = srcK[u] * k_rn * k_weights[u];
                    }
                    __syncthreads();

                    // K: Split-half RoPE
                    for (int i = t; i < half; i += blockDim.x) {
                        float freq = powf(rope_base, -2.0f * (float)i / (float)d);
                        float theta = (float)r * freq;
                        float cos_t, sin_t;
                        __sincosf(theta, &sin_t, &cos_t);
                        float kx0 = k_buf[i];
                        float kx1 = k_buf[i + half];
                        k_buf[i]        = kx0 * cos_t - kx1 * sin_t;
                        k_buf[i + half] = kx1 * cos_t + kx0 * sin_t;
                    }
                    __syncthreads();

                    k_row = k_buf;
                }

                // Dot product: q . k
                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {
                    partial += q_buf[u] * k_row[u];
                }
                float dot_qk = block_reduce_sum(partial);

                // Online softmax update
                if (t == 0) {
                    float logit = dot_qk * scale;
                    float m_old = att_m;
                    float m_new = fmaxf(m_old, logit);
                    float corr = __expf(m_old - m_new);
                    float w = __expf(logit - m_new);
                    att_d = att_d * corr + w;
                    att_m = m_new;
                    att_corr = corr;
                    att_w = w;
                }
                __syncthreads();

                float corr = att_corr;
                float w = att_w;

                for (int j = t; j < d; j += blockDim.x) {
                    O[j] = O[j] * corr + w * v_row[j];
                }
                __syncthreads();
            }

            // Final normalization
            if (t == 0) att_w = 1.0f / att_d;
            __syncthreads();
            float inv_d = att_w;

            for (int j = t; j < d; j += blockDim.x) {
                O[j] *= inv_d;
            }
        "
        .to_string()
    }
}
