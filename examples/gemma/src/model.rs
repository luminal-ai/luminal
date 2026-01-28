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
use std::{fmt::Debug, sync::Arc};

// Gemma 3 4B hyperparams
pub const LAYERS: usize = 34;
pub const HIDDEN: usize = 2560;
pub const INTERMEDIATE: usize = 10240;
pub const HEAD_DIM: usize = 256;
pub const N_HEADS: usize = 8; // Number of attention heads for Q
pub const N_KV_HEADS: usize = 4; // Number of KV heads
pub const KV_GROUPS: usize = N_HEADS / N_KV_HEADS; // = 2
pub const Q_DIM: usize = N_HEADS * HEAD_DIM; // = 2048
pub const KV_DIM: usize = N_KV_HEADS * HEAD_DIM; // = 1024
pub const VOCAB_SIZE: usize = 262208;
pub const RMS_NORM_EPS: f32 = 1e-6;

/// Gemma-specific RMSNorm
/// Note: weights are pre-transformed to (1 + weight) in setup.py
pub struct GemmaRMSNorm {
    pub weight: GraphTensor,
    epsilon: f32,
}

impl GemmaRMSNorm {
    pub fn new(dim: usize, weight_name: &str, epsilon: f32, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor(weight_name, dim),
            epsilon,
        }
    }

    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        // RMS normalize, then multiply by weight (already has +1 applied in setup.py)
        let normalized = input.std_norm(input.shape.last_axis(), self.epsilon);
        let scale = self
            .weight
            .expand_lhs(&input.dims()[..input.dims().len() - 1]);
        normalized * scale
    }
}

pub struct Gemma {
    embedding: GraphTensor,
    lm_head: GraphTensor,
    layers: Vec<GemmaLayer>,
    lm_norm: GemmaRMSNorm,
}

impl Gemma {
    pub fn init(cx: &mut Graph) -> Self {
        let mut w = vec![];
        for l in 0..LAYERS {
            w.push(GemmaLayer {
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
                input_layernorm: GemmaRMSNorm::new(
                    HIDDEN,
                    &format!("model.layers.{l}.input_layernorm.weight"),
                    RMS_NORM_EPS,
                    cx,
                ),
                post_attention_layernorm: GemmaRMSNorm::new(
                    HIDDEN,
                    &format!("model.layers.{l}.post_attention_layernorm.weight"),
                    RMS_NORM_EPS,
                    cx,
                ),
                pre_feedforward_layernorm: GemmaRMSNorm::new(
                    HIDDEN,
                    &format!("model.layers.{l}.pre_feedforward_layernorm.weight"),
                    RMS_NORM_EPS,
                    cx,
                ),
                post_feedforward_layernorm: GemmaRMSNorm::new(
                    HIDDEN,
                    &format!("model.layers.{l}.post_feedforward_layernorm.weight"),
                    RMS_NORM_EPS,
                    cx,
                ),
                q_norm: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.q_norm.weight"),
                    HEAD_DIM,
                ),
                k_norm: cx.named_tensor(
                    format!("model.layers.{l}.self_attn.k_norm.weight"),
                    HEAD_DIM,
                ),
            });
        }
        let lm_norm = GemmaRMSNorm::new(HIDDEN, "model.norm.weight", RMS_NORM_EPS, cx);
        Self {
            embedding: cx.named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN)),
            lm_head: cx.named_tensor("lm_head.weight", (VOCAB_SIZE, HIDDEN)),
            layers: w,
            lm_norm,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn forward(
        &self,
        token_ids: GraphTensor,
        pos_ids: GraphTensor,
        kv_cache: &KVCache,
    ) -> GraphTensor {
        let batch = token_ids.dims1();
        // Gemma uses pre-scaled embeddings (scaled by sqrt(hidden_size) in setup.py)
        let mut x = self.embedding.gather(
            (token_ids * HIDDEN).expand_dim(1, HIDDEN)
                + token_ids.graph().arange(HIDDEN).expand_dim(0, batch),
        );
        for (layer, (k_cache, v_cache)) in self.layers.iter().zip(&kv_cache.layers) {
            x = layer.forward(
                x,
                pos_ids,
                k_cache.device_ptr(v_cache.stream()).0,
                v_cache.device_ptr(k_cache.stream()).0,
            );
        }
        // Use separate lm_head (not tied to embeddings)
        self.lm_norm.forward(x).matmul(self.lm_head.t())
    }
}

struct GemmaLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    input_layernorm: GemmaRMSNorm,
    post_attention_layernorm: GemmaRMSNorm,
    pre_feedforward_layernorm: GemmaRMSNorm,
    post_feedforward_layernorm: GemmaRMSNorm,
    q_norm: GraphTensor,
    k_norm: GraphTensor,
}

/// Fused QK-Norm + RoPE custom operation (interleaved format for Gemma)
#[derive(Debug, Clone)]
pub struct GemmaQKNormRoPE {
    range: Vec<Expression>,
    inp_stride: Vec<Expression>,
    row_width: Expression,
}

impl GemmaQKNormRoPE {
    fn new(seq: Expression, row_width: Expression) -> Self {
        Self {
            range: vec![seq],
            inp_stride: vec![row_width],
            row_width,
        }
    }
}

impl CustomOp for GemmaQKNormRoPE {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for GemmaQKNormRoPE {
    fn op_name(&self) -> &'static str {
        "GemmaQKNormRoPE"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![
            vec![true; self.range.len()],
            vec![true; self.range.len()],
            vec![true; self.range.len()],
        ]
    }

    fn cuda_function(&self) -> String {
        format!(
            r#"
        __shared__ float rms_scale_shared;

        const float* inp = source_ptrs[0] + eval_expression(payload.inp, current);
        float*       out = out_ptr + eval_expression(payload.out, current);
        const float* weight = source_ptrs[1];
        const int* token_ids = (const int*)source_ptrs[2] + eval_expression(payload.token_ids, current);

        const int D_total = eval_expression(payload.row_width, 0);
        const int d_head  = {HEAD_DIM};
        const int n_heads = D_total / d_head;

        const int pos  = token_ids[0];
        const float base = 1000000.0f;
        const float eps = {RMS_NORM_EPS}f;

        const int half = d_head / 2;

        // Process each head
        for (int h = 0; h < n_heads; ++h) {{
            const float* head_in  = inp + h * d_head;
            float*       head_out = out + h * d_head;

            // Step 1: Compute sum of squares for RMS norm
            if (t == 0) {{
                float sum_sq = 0.0f;
                for (int k = 0; k < d_head; ++k) {{
                    float val = head_in[k];
                    sum_sq += val * val;
                }}
                rms_scale_shared = rsqrtf(sum_sq / (float)d_head + eps);
            }}
            __syncthreads();
            float rms_scale = rms_scale_shared;

            // Step 2: Apply RMS norm + weight and RoPE (interleaved format)
            for (int k = t; k < half; k += blockDim.x) {{
                // Interleaved indexing: pairs are (0,1), (2,3), (4,5), ...
                const int j_even = 2 * k;
                const int j_odd  = 2 * k + 1;

                // Apply RMS norm and weight (weights have 1+w pre-applied in setup.py)
                float x_even = head_in[j_even] * rms_scale * weight[j_even];
                float x_odd  = head_in[j_odd]  * rms_scale * weight[j_odd];

                // Compute RoPE rotation
                const float exponent = -(2.0f * (float)k) / (float)d_head;
                const float theta = (float)pos * __powf(base, exponent);

                float s, c;
                __sincosf(theta, &s, &c);

                // Rotation
                head_out[j_even] = x_even * c - x_odd * s;
                head_out[j_odd]  = x_even * s + x_odd * c;
            }}
            __syncthreads();
        }}
        "#,
            HEAD_DIM = HEAD_DIM,
            RMS_NORM_EPS = RMS_NORM_EPS
        )
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr("inp", flatten_mul_strides(&self.range, &self.inp_stride))
            .expr("out", flatten_mul_strides(&self.range, &self.inp_stride))
            .expr("row_width", self.row_width)
            .expr("weight", 0)
            .expr("token_ids", 'z')
    }
}

impl GemmaLayer {
    pub fn forward(
        &self,
        x: GraphTensor,
        pos_ids: GraphTensor,
        k_cache: u64,
        v_cache: u64,
    ) -> GraphTensor {
        // 1. Pre-attention norm
        let x_attn = self.input_layernorm.forward(x);

        // 2. Q/K/V projections
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        // 3. Apply QK-Norm + RoPE using fused custom kernel
        let q_rope = x.graph().custom_op(
            GemmaQKNormRoPE::new(q.dims()[0], q.dims()[1]),
            (q, self.q_norm, pos_ids),
            q.shape,
            q.dtype,
        );
        let k_rope = x.graph().custom_op(
            GemmaQKNormRoPE::new(k.dims()[0], k.dims()[1]),
            (k, self.k_norm, pos_ids),
            k.shape,
            k.dtype,
        );

        // 4. Attention
        let attn_out = x.graph().custom_op(
            GemmaAttention::new(k_cache, v_cache, q_rope.dims()[0], 'p'.into()),
            (q_rope, k_rope, v),
            q_rope.shape,
            q_rope.dtype,
        );

        // 5. O projection + post-attention norm + residual
        let attn_proj = attn_out.matmul(self.o_proj.t());
        let attn_normed = self.post_attention_layernorm.forward(attn_proj);
        let x = x + attn_normed;

        // 6. Pre-feedforward norm
        let x_ff = self.pre_feedforward_layernorm.forward(x);

        // 7. MLP with SwiGLU
        // MLP with gated activation (using swish for now - TODO: verify correct activation)
        let mlp_out =
            (x_ff.matmul(self.gate.t()).swish() * x_ff.matmul(self.up.t())).matmul(self.down.t());

        // 8. Post-feedforward norm + residual
        let mlp_normed = self.post_feedforward_layernorm.forward(mlp_out);
        x + mlp_normed
    }
}

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

#[derive(Debug, Clone)]
pub struct GemmaAttention {
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
}

impl GemmaAttention {
    fn new(k_cache: u64, v_cache: u64, seq: Expression, prev_seq: Expression) -> Self {
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
        }
    }
}

impl CustomOp for GemmaAttention {
    fn to_llir_op(&self) -> luminal::op::LLIROp {
        LLIROp::new::<dyn BlockOp>(Box::new(self.clone()))
    }
}

impl BlockOp for GemmaAttention {
    fn op_name(&self) -> &'static str {
        "GemmaAttention"
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
            // shared buffer for block-wide reduction
            __shared__ float shared[32]; // max 32 warps per block

            // warp-level reduction
            auto warp_reduce_sum = [](float val) {
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                return val;
            };

            // block-level reduction (sum valid only in thread 0)
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

            const float* q   = source_ptrs[0] + eval_expression(payload.q, current);
            const float* k   = source_ptrs[1] + eval_expression(payload.k, current);
            const float* v   = source_ptrs[2] + eval_expression(payload.v, current);
            float*       out = out_ptr + eval_expression(payload.out, current);
            int q_pos_local = eval_expression(payload.q_pos_stride, current);
            const int group_pos_local = eval_expression(payload.group_pos_stride, current);
            const int head_pos_local = eval_expression(payload.head_pos_stride, current);

            const int d             = eval_expression(payload.head_size, 0);
            const float* __restrict__ K_cache = payload.key_cache + head_pos_local * d;
            const float* __restrict__ V_cache = payload.val_cache + head_pos_local * d;

            const int S             = eval_expression(payload.cur_seq, 0);
            const int kv_row_stride = eval_expression(payload.kv_row_stride, 0);
            const int prev          = eval_expression(payload.prev_seq, 0);

            const float* __restrict__ K_cur = k;
            const float* __restrict__ V_cur = v;
            float*       __restrict__ O     = out;

            if (q_pos_local >= S) q_pos_local = S - 1;
            if (q_pos_local < 0)  q_pos_local = 0;

            const int q_pos_total = prev + q_pos_local;

            const float scale = rsqrtf((float)d);

            __shared__ float max_l_shared;
            __shared__ float inv_s_shared;
            __shared__ float w_shared;

            if (group_pos_local == 0 && K_cache != nullptr && V_cache != nullptr) {
                for (int r = 0; r < S; ++r) {
                    const float* __restrict__ srcK = K_cur + r * kv_row_stride;
                    const float* __restrict__ srcV = V_cur + r * kv_row_stride;
                          float* __restrict__ dstK = const_cast<float*>(K_cache) + (prev + r) * kv_row_stride;
                          float* __restrict__ dstV = const_cast<float*>(V_cache) + (prev + r) * kv_row_stride;

                    for (int u = t; u < d; u += blockDim.x) {
                        dstK[u] = srcK[u];
                        dstV[u] = srcV[u];
                    }
                }
            }
            __syncthreads();

            if (t == 0) max_l_shared = -__int_as_float(0x7f800000);
            __syncthreads();

            for (int r = 0; r <= q_pos_total; ++r) {
                const float* __restrict__ k_row;
                if (r < prev) {
                    k_row = K_cache + r * kv_row_stride;
                } else {
                    int r_local = r - prev;
                    k_row = K_cur + r_local * kv_row_stride;
                }

                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {
                    partial += q[u] * k_row[u];
                }
                float dot_qk = block_reduce_sum(partial);

                if (t == 0) {
                    float logit = dot_qk * scale;
                    max_l_shared = fmaxf(max_l_shared, logit);
                }
                __syncthreads();
            }

            __syncthreads();
            float max_l = max_l_shared;

            for (int j = t; j < d; j += blockDim.x) {
                O[j] = 0.0f;
            }

            float s_local = 0.0f;
            __syncthreads();

            for (int r = 0; r <= q_pos_total; ++r) {
                const float* __restrict__ k_row;
                const float* __restrict__ v_row;

                if (r < prev) {
                    k_row = K_cache + r * kv_row_stride;
                    v_row = V_cache + r * kv_row_stride;
                } else {
                    int r_local = r - prev;
                    k_row = K_cur + r_local * kv_row_stride;
                    v_row = V_cur + r_local * kv_row_stride;
                }

                float partial = 0.0f;
                for (int u = t; u < d; u += blockDim.x) {
                    partial += q[u] * k_row[u];
                }
                float dot_qk = block_reduce_sum(partial);

                if (t == 0) {
                    float logit = dot_qk * scale;
                    float w     = __expf(logit - max_l);
                    s_local    += w;
                    w_shared    = w;
                }
                __syncthreads();

                float w = w_shared;

                for (int j = t; j < d; j += blockDim.x) {
                    O[j] += w * v_row[j];
                }
                __syncthreads();
            }

            if (t == 0) {
                inv_s_shared = 1.0f / s_local;
            }
            __syncthreads();

            float inv_s = inv_s_shared;

            for (int j = t; j < d; j += blockDim.x) {
                O[j] *= inv_s;
            }
        "
        .to_string()
    }
}
