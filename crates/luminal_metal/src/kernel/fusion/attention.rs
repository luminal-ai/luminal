use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, F64, OP_KIND},
    },
    op::*,
    prelude::*,
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};
#[cfg(feature = "debug")]
use objc::runtime::Object;

use crate::kernel::{MetalEncodeContext, MetalKernelOp, ops::compile_shader};
#[cfg(feature = "debug")]
use crate::kernel::{pop_debug_group, push_debug_group, set_objc_label};

fn single_dyn_var(expr: Expression) -> Option<char> {
    let vars = expr.dyn_vars();
    if vars.len() == 1 { Some(vars[0]) } else { None }
}

fn resolve_dim(expr: Expression, symbol: Option<char>, dyn_map: &FxHashMap<char, usize>) -> usize {
    symbol
        .and_then(|symbol| dyn_map.get(&symbol).copied())
        .or_else(|| expr.exec(dyn_map))
        .expect("FusedPostRopeAttention dynamic dimension not set")
}

#[derive(Debug, Clone, Default)]
/// Fuses paged attention after Q/K have already been RoPE-rotated.
/// q [seq, hidden], k_cache/v_cache [max_context, kv_dim] -> out [seq, hidden]
/// scores = q @ k_cache[k_gather_idx].T * scale + mask
/// out = softmax(scores) @ v_cache[v_gather_idx]
pub struct MetalFusedPostRopeAttention {
    seq: Expression,
    context: Expression,
    seq_symbol: Option<char>,
    context_symbol: Option<char>,
    hidden: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_context: usize,
    scale: f32,
    decode_attention_pipeline: std::sync::OnceLock<ComputePipelineState>,
}

impl MetalFusedPostRopeAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        seq: Expression,
        context: Expression,
        hidden: usize,
        kv_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_context: usize,
        scale: f32,
    ) -> Self {
        let seq_symbol = single_dyn_var(seq);
        let context_symbol = single_dyn_var(context);
        Self {
            seq,
            context,
            seq_symbol,
            context_symbol,
            hidden,
            kv_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            max_context,
            scale,
            decode_attention_pipeline: std::sync::OnceLock::new(),
        }
    }

    fn seq_size_expr(&self) -> Expression {
        self.seq_symbol.map(Expression::from).unwrap_or(self.seq)
    }

    fn resolved_dims(
        &self,
        dyn_map: &FxHashMap<char, usize>,
    ) -> (u32, u32, u32, u32, u32, u32, u32) {
        let seq = resolve_dim(self.seq, self.seq_symbol, dyn_map) as u32;
        let context = resolve_dim(self.context, self.context_symbol, dyn_map) as u32;
        assert!(
            context as usize <= self.max_context,
            "FusedPostRopeAttention context {context} exceeds compiled max_context {}",
            self.max_context
        );
        (
            seq,
            context,
            self.hidden as u32,
            self.kv_dim as u32,
            self.n_heads as u32,
            self.n_kv_heads as u32,
            self.head_dim as u32,
        )
    }

    fn attention_source(&self) -> String {
        let max_context = self.max_context;
        let scale = self.scale;
        format!(
            r#"
                #include <metal_stdlib>
                using namespace metal;

                #define THREADS_PER_GROUP 256
                #define MAX_CONTEXT {max_context}
                #define SCALE {scale:.9e}f

                kernel void fused_attention(
                    const device float *q [[buffer(0)]],
                    const device float *k_cache [[buffer(1)]],
                    const device float *v_cache [[buffer(2)]],
                    const device int *k_gather_idx [[buffer(3)]],
                    const device int *v_gather_idx [[buffer(4)]],
                    const device float *attn_mask [[buffer(5)]],
                    device float *out [[buffer(6)]],
                    constant uint &seq [[buffer(7)]],
                    constant uint &context_len [[buffer(8)]],
                    constant uint &hidden [[buffer(9)]],
                    constant uint &kv_dim [[buffer(10)]],
                    constant uint &n_heads [[buffer(11)]],
                    constant uint &n_kv_heads [[buffer(12)]],
                    constant uint &head_dim [[buffer(13)]],
                    uint2 gid [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]
                ) {{
                    const uint head = gid.x;
                    const uint token = gid.y;
                    if (head >= n_heads || token >= seq || context_len > MAX_CONTEXT) {{
                        return;
                    }}

                    threadgroup float scores[MAX_CONTEXT];
                    threadgroup float partials[THREADS_PER_GROUP];

                    const uint kv_group = n_heads / n_kv_heads;
                    const uint kv_head = head / kv_group;
                    const uint q_base = token * hidden + head * head_dim;

                    float local_max = -INFINITY;
                    for (uint j = tid; j < context_len; j += THREADS_PER_GROUP) {{
                        const int flat = k_gather_idx[j * kv_dim];
                        float score = -INFINITY;
                        if (flat >= 0) {{
                            const uint pos = uint(flat) / kv_dim;
                            const uint k_base = pos * kv_dim + kv_head * head_dim;
                            float dot = 0.0f;
                            for (uint d = 0; d < head_dim; ++d) {{
                                dot += q[q_base + d] * k_cache[k_base + d];
                            }}
                            score = dot * SCALE + attn_mask[token * context_len + j];
                        }}
                        scores[j] = score;
                        local_max = max(local_max, score);
                    }}

                    partials[tid] = local_max;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                        if (tid < stride) {{
                            partials[tid] = max(partials[tid], partials[tid + stride]);
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}
                    const float max_score = partials[0];

                    float local_sum = 0.0f;
                    for (uint j = tid; j < context_len; j += THREADS_PER_GROUP) {{
                        const float w = exp2((scores[j] - max_score) * 1.4426950408889634f);
                        scores[j] = w;
                        local_sum += w;
                    }}

                    partials[tid] = local_sum;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                        if (tid < stride) {{
                            partials[tid] += partials[tid + stride];
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}
                    const float inv_sum = 1.0f / partials[0];

                    if (tid < head_dim) {{
                        float acc = 0.0f;
                        for (uint j = 0; j < context_len; ++j) {{
                            const int flat = v_gather_idx[j * kv_dim];
                            if (flat >= 0) {{
                                const uint pos = uint(flat) / kv_dim;
                                const uint v_base = pos * kv_dim + kv_head * head_dim;
                                acc += scores[j] * inv_sum * v_cache[v_base + tid];
                            }}
                        }}
                        out[head * seq * head_dim + token * head_dim + tid] = acc;
                    }}
                }}
                "#
        )
    }

    fn decode_heads_per_group(&self) -> usize {
        let kv_group = self.n_heads / self.n_kv_heads;
        if kv_group.is_multiple_of(4) && self.max_context <= 512 && self.head_dim * 4 <= 256 {
            4
        } else if kv_group.is_multiple_of(2) && self.head_dim * 2 <= 256 {
            2
        } else {
            1
        }
    }

    fn decode_attention_source(&self) -> String {
        let max_context = self.max_context;
        let hidden = self.hidden;
        let kv_dim = self.kv_dim;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let kv_group = self.n_heads / self.n_kv_heads;
        let heads_per_group = self.decode_heads_per_group();
        let scale = self.scale;
        format!(
            r#"
                #include <metal_stdlib>
                using namespace metal;

                #define THREADS_PER_GROUP 256
                #define MAX_CONTEXT {max_context}
                #define HIDDEN {hidden}
                #define KV_DIM {kv_dim}
                #define N_HEADS {n_heads}
                #define N_KV_HEADS {n_kv_heads}
                #define HEAD_DIM {head_dim}
                #define KV_GROUP {kv_group}
                #define HEADS_PER_GROUP {heads_per_group}
                #define SCALE {scale:.9e}f

                kernel void fused_decode_attention(
                    const device float *q [[buffer(0)]],
                    const device float *k_cache [[buffer(1)]],
                    const device float *v_cache [[buffer(2)]],
                    const device int *k_gather_idx [[buffer(3)]],
                    const device int *v_gather_idx [[buffer(4)]],
                    const device float *attn_mask [[buffer(5)]],
                    device float *out [[buffer(6)]],
                    constant uint &context_len [[buffer(7)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]
                ) {{
                    const uint base_head = gid * HEADS_PER_GROUP;
                    if (base_head >= N_HEADS || context_len > MAX_CONTEXT) {{
                        return;
                    }}

                    threadgroup float scores[HEADS_PER_GROUP * MAX_CONTEXT];
                    threadgroup float partials[HEADS_PER_GROUP * THREADS_PER_GROUP];

                    const uint kv_head = base_head / KV_GROUP;
                    const uint out_lane_count = HEADS_PER_GROUP * HEAD_DIM;

                    float local_max[HEADS_PER_GROUP];
                    for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                        local_max[h] = -INFINITY;
                    }}

                    for (uint j = tid; j < context_len; j += THREADS_PER_GROUP) {{
                        const int flat = k_gather_idx[j * KV_DIM];
                        for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                            const uint head = base_head + h;
                            float score = -INFINITY;
                            if (head < N_HEADS && flat >= 0) {{
                                const uint pos = uint(flat) / KV_DIM;
                                const uint q_base = head * HEAD_DIM;
                                const uint k_base = pos * KV_DIM + kv_head * HEAD_DIM;
                                float dot = 0.0f;
                                for (uint d = 0; d < HEAD_DIM; ++d) {{
                                    dot += q[q_base + d] * k_cache[k_base + d];
                                }}
                                score = dot * SCALE + attn_mask[j];
                            }}
                            scores[h * MAX_CONTEXT + j] = score;
                            local_max[h] = max(local_max[h], score);
                        }}
                    }}

                    for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                        partials[h * THREADS_PER_GROUP + tid] = local_max[h];
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                        if (tid < stride) {{
                            for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                                partials[h * THREADS_PER_GROUP + tid] =
                                    max(partials[h * THREADS_PER_GROUP + tid],
                                        partials[h * THREADS_PER_GROUP + tid + stride]);
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}

                    float max_score[HEADS_PER_GROUP];
                    for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                        max_score[h] = partials[h * THREADS_PER_GROUP];
                    }}

                    float local_sum[HEADS_PER_GROUP];
                    for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                        local_sum[h] = 0.0f;
                    }}

                    for (uint j = tid; j < context_len; j += THREADS_PER_GROUP) {{
                        for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                            const float w =
                                exp2((scores[h * MAX_CONTEXT + j] - max_score[h]) *
                                     1.4426950408889634f);
                            scores[h * MAX_CONTEXT + j] = w;
                            local_sum[h] += w;
                        }}
                    }}

                    for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                        partials[h * THREADS_PER_GROUP + tid] = local_sum[h];
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                        if (tid < stride) {{
                            for (uint h = 0; h < HEADS_PER_GROUP; ++h) {{
                                partials[h * THREADS_PER_GROUP + tid] +=
                                    partials[h * THREADS_PER_GROUP + tid + stride];
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}

                    if (tid < out_lane_count) {{
                        const uint h = tid / HEAD_DIM;
                        const uint d = tid - h * HEAD_DIM;
                        const uint head = base_head + h;
                        if (head < N_HEADS) {{
                            const float inv_sum =
                                1.0f / partials[h * THREADS_PER_GROUP];
                            float acc = 0.0f;
                            for (uint j = 0; j < context_len; ++j) {{
                                const int flat = v_gather_idx[j * KV_DIM];
                                if (flat >= 0) {{
                                    const uint pos = uint(flat) / KV_DIM;
                                    const uint v_base =
                                        pos * KV_DIM + kv_head * HEAD_DIM;
                                    acc += scores[h * MAX_CONTEXT + j] * inv_sum *
                                           v_cache[v_base + d];
                                }}
                            }}
                            out[head * HEAD_DIM + d] = acc;
                        }}
                    }}
                }}
                "#
        )
    }
}

impl EgglogOp for MetalFusedPostRopeAttention {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "MetalFusedPostRopeAttention",
            &[
                ("num_qo_heads", EXPRESSION),
                ("num_kv_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("hidden", EXPRESSION),
                ("seq", EXPRESSION),
                ("context", EXPRESSION),
                ("max_context", EXPRESSION),
                ("scale", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        6
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("post_rope_attention.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;

        let mut expr_at =
            |idx: usize| extract_expr(egraph, kind_children[idx], expr_cache).unwrap();
        let static_usize = |expr: Expression, name: &str| {
            expr.exec(&FxHashMap::default())
                .unwrap_or_else(|| panic!("MetalFusedPostRopeAttention {name} must be static"))
        };
        let n_heads = static_usize(expr_at(0), "num_qo_heads");
        let n_kv_heads = static_usize(expr_at(1), "num_kv_heads");
        let head_dim = static_usize(expr_at(2), "head_dim");
        let hidden = static_usize(expr_at(3), "hidden");
        let seq = expr_at(4);
        let context = expr_at(5);
        let max_context = static_usize(expr_at(6), "max_context");
        let scale = egraph.enodes[kind_children[7]]
            .0
            .replace('\"', "")
            .parse::<f32>()
            .expect("MetalFusedPostRopeAttention scale must be a float");

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self::new(
                seq,
                context,
                hidden,
                n_kv_heads * head_dim,
                n_heads,
                n_kv_heads,
                head_dim,
                max_context,
                scale,
            ))),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalFusedPostRopeAttention {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "FusedPostRopeAttention"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        assert_eq!(
            output_dtype,
            DType::F32,
            "FusedPostRopeAttention output must be F32"
        );
        assert!(
            input_dtypes.len() >= 6,
            "FusedPostRopeAttention received too few inputs"
        );
        assert_eq!(
            input_dtypes[0],
            DType::F32,
            "FusedPostRopeAttention q must be F32"
        );
        assert_eq!(
            input_dtypes[1],
            DType::F32,
            "FusedPostRopeAttention k cache must be F32"
        );
        assert_eq!(
            input_dtypes[2],
            DType::F32,
            "FusedPostRopeAttention v cache must be F32"
        );
        assert_eq!(
            input_dtypes[3],
            DType::Int,
            "FusedPostRopeAttention k gather_idx must be Int"
        );
        assert_eq!(
            input_dtypes[4],
            DType::Int,
            "FusedPostRopeAttention v gather_idx must be Int"
        );
        assert_eq!(
            input_dtypes[5],
            DType::F32,
            "FusedPostRopeAttention mask must be F32"
        );

        Some(compile_shader(
            device,
            &self.attention_source(),
            "fused_attention",
        ))
        .inspect(|_| {
            let _ = self.decode_attention_pipeline.set(compile_shader(
                device,
                &self.decode_attention_source(),
                "fused_decode_attention",
            ));
        })
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.seq_size_expr() * Expression::from(self.hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        context: &mut MetalEncodeContext<'_>,
        pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let pipeline = pipeline.expect("FusedPostRopeAttention compute pipeline not compiled");
        let (seq, context_len, hidden, kv_dim, n_heads, n_kv_heads, head_dim) =
            self.resolved_dims(dyn_map);

        let encoder = context.command_buffer.new_compute_command_encoder();
        #[cfg(feature = "debug")]
        {
            let label = self.label();
            set_objc_label(encoder.as_ptr() as *mut Object, label);
            push_debug_group(encoder.as_ptr() as *mut Object, label);
        }
        if seq == 1 {
            let decode_pipeline = self
                .decode_attention_pipeline
                .get()
                .expect("FusedPostRopeAttention decode pipeline not compiled");
            self.encode_decode_attention(encoder, decode_pipeline, inputs, output, context_len);
        } else {
            self.encode_attention(
                encoder,
                pipeline,
                inputs,
                output,
                seq,
                context_len,
                hidden,
                kv_dim,
                n_heads,
                n_kv_heads,
                head_dim,
            );
        }
        #[cfg(feature = "debug")]
        pop_debug_group(encoder.as_ptr() as *mut Object, self.label());
        encoder.end_encoding();
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let (seq, context, hidden, kv_dim, n_heads, n_kv_heads, head_dim) =
            self.resolved_dims(dyn_map);
        if seq == 1 {
            let decode_pipeline = self
                .decode_attention_pipeline
                .get()
                .expect("FusedPostRopeAttention decode pipeline not compiled");
            self.encode_decode_attention(encoder, decode_pipeline, inputs, output, context);
        } else {
            self.encode_attention(
                encoder, pipeline, inputs, output, seq, context, hidden, kv_dim, n_heads,
                n_kv_heads, head_dim,
            );
        }
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let seq = self.seq.exec(dyn_map).unwrap_or(0);
        let context = self.context.exec(dyn_map).unwrap_or(0);
        let qk = seq * self.n_heads * context * self.head_dim * 2 * std::mem::size_of::<f32>();
        let av = seq * self.n_heads * context * self.head_dim * std::mem::size_of::<f32>();
        qk + av
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let seq = self.seq.exec(dyn_map).unwrap_or(0);
        seq * self.hidden * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let seq = self.seq.exec(dyn_map).unwrap_or(0);
        let context = self.context.exec(dyn_map).unwrap_or(0);
        let qk = seq * self.n_heads * context * self.head_dim * 2;
        let av = seq * self.n_heads * context * self.head_dim * 2;
        let softmax = seq * self.n_heads * context * 4;
        qk + av + softmax
    }
}

impl MetalFusedPostRopeAttention {
    #[allow(clippy::too_many_arguments)]
    fn encode_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        seq: u32,
        context: u32,
        hidden: u32,
        kv_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) {
        let thread_group_size = MTLSize::new(256, 1, 1);

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0); // Q, post-RoPE
        encoder.set_buffer(1, Some(inputs[1]), 0); // K cache
        encoder.set_buffer(2, Some(inputs[2]), 0); // V cache
        encoder.set_buffer(3, Some(inputs[3]), 0); // K gather flat indices
        encoder.set_buffer(4, Some(inputs[4]), 0); // V gather flat indices
        encoder.set_buffer(5, Some(inputs[5]), 0); // additive mask
        encoder.set_buffer(6, Some(output), 0);
        let constants_start = 7;
        encoder.set_bytes(
            constants_start,
            std::mem::size_of::<u32>() as u64,
            &seq as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 1,
            std::mem::size_of::<u32>() as u64,
            &context as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 2,
            std::mem::size_of::<u32>() as u64,
            &hidden as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 3,
            std::mem::size_of::<u32>() as u64,
            &kv_dim as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 4,
            std::mem::size_of::<u32>() as u64,
            &n_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 5,
            std::mem::size_of::<u32>() as u64,
            &n_kv_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            constants_start + 6,
            std::mem::size_of::<u32>() as u64,
            &head_dim as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, seq as u64, 1),
            thread_group_size,
        );
    }

    fn encode_decode_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        context: u32,
    ) {
        let thread_group_size = MTLSize::new(256, 1, 1);
        let heads_per_group = self.decode_heads_per_group() as u64;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(inputs[2]), 0);
        encoder.set_buffer(3, Some(inputs[3]), 0);
        encoder.set_buffer(4, Some(inputs[4]), 0);
        encoder.set_buffer(5, Some(inputs[5]), 0);
        encoder.set_buffer(6, Some(output), 0);
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &context as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((self.n_heads as u64).div_ceil(heads_per_group), 1, 1),
            thread_group_size,
        );
    }
}
