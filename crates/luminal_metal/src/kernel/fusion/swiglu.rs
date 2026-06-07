use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{ELIST, EXPRESSION, IR},
    },
    op::*,
    prelude::*,
    shape::flatten_strides,
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use crate::kernel::{
    MetalKernelOp,
    ops::{compile_shader, lower_expression_for_metal},
};

/// Fuses the elementwise SwiGLU activation into one kernel:
/// `gate * sigmoid(gate) * up`. The rewrite removes the decomposed neg/exp2,
/// add, reciprocal, and multiply chain after the gate and up projections have
/// already been materialized.
#[derive(Debug, Clone, Default)]
pub struct MetalSwiGLU {
    shape: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalSwiGLU {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalSwiGLU",
            &[
                ("shape", ELIST),
                ("gate", IR),
                ("up", IR),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            r#"
            (rule
                (
                    (= ?neg_one (MetalConstant -1.000000))
                    (= ?log2e (MetalConstant 1.442695))
                    (= ?one (MetalConstant 1.000000))
                    (= ?neg_gate (Op
                        (MetalMul ?shape ?gate_stride ?neg_one_stride ?neg_gate_stride)
                        (ICons ?gate (ICons ?neg_one (INil)))))
                    (= ?scaled (Op
                        (MetalMul ?shape ?neg_gate_stride ?log2e_stride ?scaled_stride)
                        (ICons ?neg_gate (ICons ?log2e (INil)))))
                    (= ?exp (Op
                        (MetalExp2 ?shape ?scaled_stride ?exp_stride)
                        (ICons ?scaled (INil))))
                    (= ?denom (Op
                        (MetalAdd ?shape ?exp_stride ?one_stride ?denom_stride)
                        (ICons ?exp (ICons ?one (INil)))))
                    (= ?sigmoid (Op
                        (MetalRecip ?shape ?denom_stride ?sigmoid_stride)
                        (ICons ?denom (INil))))
                    (= ?swish (Op
                        (MetalMul ?shape ?gate_stride ?sigmoid_stride ?swish_stride)
                        (ICons ?gate (ICons ?sigmoid (INil)))))
                    (= ?out (Op
                        (MetalMul ?shape ?swish_stride ?up_stride ?out_stride)
                        (ICons ?swish (ICons ?up (INil)))))
                )
                (
                    (let ?fused (MetalSwiGLU ?shape ?gate ?up ?out_stride))
                    (union ?out ?fused)
                    (set (dtype ?fused) (F32))
                )
                :ruleset matmul_backend
                :name "metal-fused-swiglu"
            )

            (rule
                (
                    (= ?fused (MetalSwiGLU ?shape ?gate ?up ?out_stride))
                    (= ?neg_one (MetalConstant -1.000000))
                    (= ?log2e (MetalConstant 1.442695))
                    (= ?one (MetalConstant 1.000000))
                    (= ?neg_gate (Op
                        (MetalMul ?shape ?gate_stride ?neg_one_stride ?neg_gate_stride)
                        (ICons ?gate (ICons ?neg_one (INil)))))
                    (= ?scaled (Op
                        (MetalMul ?shape ?neg_gate_stride ?log2e_stride ?scaled_stride)
                        (ICons ?neg_gate (ICons ?log2e (INil)))))
                    (= ?exp (Op
                        (MetalExp2 ?shape ?scaled_stride ?exp_stride)
                        (ICons ?scaled (INil))))
                    (= ?denom (Op
                        (MetalAdd ?shape ?exp_stride ?one_stride ?denom_stride)
                        (ICons ?exp (ICons ?one (INil)))))
                    (= ?sigmoid (Op
                        (MetalRecip ?shape ?denom_stride ?sigmoid_stride)
                        (ICons ?denom (INil))))
                    (= ?swish (Op
                        (MetalMul ?shape ?gate_stride ?sigmoid_stride ?swish_stride)
                        (ICons ?gate (ICons ?sigmoid (INil)))))
                    (= ?out (Op
                        (MetalMul ?shape ?swish_stride ?up_stride ?out_stride)
                        (ICons ?swish (ICons ?up (INil)))))
                )
                (
                    (delete (Op
                        (MetalMul ?shape ?swish_stride ?up_stride ?out_stride)
                        (ICons ?swish (ICons ?up (INil)))))
                    (delete (Op
                        (MetalMul ?shape ?gate_stride ?sigmoid_stride ?swish_stride)
                        (ICons ?gate (ICons ?sigmoid (INil)))))
                    (delete (Op
                        (MetalRecip ?shape ?denom_stride ?sigmoid_stride)
                        (ICons ?denom (INil))))
                    (delete (Op
                        (MetalAdd ?shape ?exp_stride ?one_stride ?denom_stride)
                        (ICons ?exp (ICons ?one (INil)))))
                    (delete (Op
                        (MetalExp2 ?shape ?scaled_stride ?exp_stride)
                        (ICons ?scaled (INil))))
                    (delete (Op
                        (MetalMul ?shape ?neg_gate_stride ?log2e_stride ?scaled_stride)
                        (ICons ?neg_gate (ICons ?log2e (INil)))))
                    (delete (Op
                        (MetalMul ?shape ?gate_stride ?neg_one_stride ?neg_gate_stride)
                        (ICons ?gate (ICons ?neg_one (INil)))))
                )
                :ruleset cleanup
                :name "delete-decomposed-swiglu"
            )
            "#,
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[2]],
        )
    }
}

impl MetalKernelOp for MetalSwiGLU {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "SwiGLU"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        if input_dtypes.first().copied().unwrap_or(DType::F32) != DType::F32
            || input_dtypes.get(1).copied().unwrap_or(DType::F32) != DType::F32
            || output_dtype != DType::F32
        {
            return None;
        }

        let out_index =
            lower_expression_for_metal(&flatten_strides(&self.shape, &self.output_strides), "idx");
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define LOG2E 1.4426950408889634f

            kernel void swiglu_kernel(
                const device float *gate [[buffer(0)]],
                const device float *up [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant int *dyn [[buffer(3)]],
                constant uint &n_elements [[buffer(4)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                (void)dyn;
                if (idx >= n_elements) {{
                    return;
                }}

                const float g = gate[idx];
                const float sigmoid = 1.0f / (1.0f + fast::exp2(-g * LOG2E));
                out[{out_index}] = g * sigmoid * up[idx];
            }}
            "#,
            out_index = out_index,
        );
        Some(compile_shader(device, &source, "swiglu_kernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        2 * n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        6 * self.output_size().exec(dyn_map).unwrap_or(0)
    }
}

/// Fuses the decode-time gate and up GEMVs with the SwiGLU activation:
/// `(lhs @ gate.T) * sigmoid(lhs @ gate.T) * (lhs @ up.T)`. Each SIMD group
/// accumulates matching gate/up columns together, so the intermediate
/// projection buffers and standalone SwiGLU kernel disappear from the decode
/// MLP path.
#[derive(Debug, Clone, Default)]
pub struct MetalFusedSwiGLUGemv {
    shape: Vec<Expression>,
    m: Expression,
    n: Expression,
    k: Expression,
    output_strides: Vec<Expression>,
}

impl MetalFusedSwiGLUGemv {
    fn static_nk(&self) -> (Option<usize>, Option<usize>) {
        let static_dims = FxHashMap::default();
        (self.n.exec(&static_dims), self.k.exec(&static_dims))
    }

    fn cols_per_simdgroup(&self) -> usize {
        let (static_n, static_k) = self.static_nk();
        if matches!((static_n, static_k), (Some(8_192), Some(2_048))) {
            2
        } else if static_n.is_some_and(|n| n >= 32_768) || static_k.is_some_and(|k| k >= 8_192) {
            8
        } else {
            4
        }
    }
}

impl EgglogOp for MetalFusedSwiGLUGemv {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalFusedSwiGLUGemv",
            &[
                ("shape", ELIST),
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("gate_rhs", IR),
                ("up_rhs", IR),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            r#"
            (rule
                (
                    (= ?gate (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?gate_rhs))
                    (= ?up (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?up_rhs))
                    (= ?out (MetalSwiGLU ?shape ?gate ?up ?out_stride))
                )
                (
                    (let ?fused (MetalFusedSwiGLUGemv ?shape ?m ?n ?k ?lhs ?gate_rhs ?up_rhs ?out_stride))
                    (union ?out ?fused)
                    (set (dtype ?fused) (F32))
                )
                :ruleset matmul_backend
                :name "metal-fused-swiglu-gemv"
            )

            (rule
                (
                    (= ?out (MetalSwiGLU ?shape ?gate ?up ?out_stride))
                    (= ?out (MetalFusedSwiGLUGemv ?shape ?m ?n ?k ?lhs ?gate_rhs ?up_rhs ?out_stride))
                )
                (
                    (subsume (MetalSwiGLU ?shape ?gate ?up ?out_stride))
                )
                :ruleset cleanup
                :name "metal-fused-swiglu-gemv-subsumes-standalone-swiglu"
            )
            "#,
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                m: extract_expr(egraph, children[1], expr_cache).unwrap(),
                n: extract_expr(egraph, children[2], expr_cache).unwrap(),
                k: extract_expr(egraph, children[3], expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[4], children[5], children[6]],
        )
    }
}

impl MetalKernelOp for MetalFusedSwiGLUGemv {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "FusedSwiGLUGemv"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        assert_eq!(
            input_dtypes.first().copied().unwrap_or(DType::F32),
            DType::F32,
            "MetalFusedSwiGLUGemv lhs must be F32"
        );
        assert_eq!(
            input_dtypes.get(1).copied().unwrap_or(DType::F32),
            DType::F32,
            "MetalFusedSwiGLUGemv gate weights must be F32"
        );
        assert_eq!(
            input_dtypes.get(2).copied().unwrap_or(DType::F32),
            DType::F32,
            "MetalFusedSwiGLUGemv up weights must be F32"
        );
        assert_eq!(
            output_dtype,
            DType::F32,
            "MetalFusedSwiGLUGemv output must be F32"
        );

        let (static_n, static_k) = self.static_nk();
        let accumulator_count = self.cols_per_simdgroup();
        let out_index = lower_expression_for_metal(
            &flatten_strides(&self.shape, &self.output_strides),
            "elem_idx",
        );
        let sum_decls = (0..accumulator_count)
            .flat_map(|j| {
                [
                    format!("            float gate_sum{j} = 0.0f;"),
                    format!("            float up_sum{j} = 0.0f;"),
                ]
            })
            .collect::<Vec<_>>()
            .join("\n");
        let dot_body = (0..accumulator_count)
            .map(|j| {
                format!(
                    "                if (col_base + {j} < n) {{\n                    const uint rhs_offset{j} = (col_base + {j}) * k + i;\n                    gate_sum{j} += dot(float4(lhs_vec), load_rhs4(gate_rhs, rhs_offset{j}));\n                    up_sum{j} += dot(float4(lhs_vec), load_rhs4(up_rhs, rhs_offset{j}));\n                }}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let tail_body = (0..accumulator_count)
            .map(|j| {
                format!(
                    "                if (col_base + {j} < n) {{\n                    const uint rhs_offset{j} = (col_base + {j}) * k + i;\n                    gate_sum{j} += lhs_val * gate_rhs[rhs_offset{j}];\n                    up_sum{j} += lhs_val * up_rhs[rhs_offset{j}];\n                }}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let reduce_body = (0..accumulator_count)
            .flat_map(|j| {
                [
                    format!("            gate_sum{j} = simd_sum(gate_sum{j});"),
                    format!("            up_sum{j} = simd_sum(up_sum{j});"),
                ]
            })
            .collect::<Vec<_>>()
            .join("\n");
        let store_body = (0..accumulator_count)
            .map(|j| {
                format!(
                    "                if (col_base + {j} < n) {{\n                    const float gate = gate_sum{j};\n                    const float sigmoid = 1.0f / (1.0f + fast::exp2(-gate * LOG2E));\n                    const uint elem_idx = row * n + col_base + {j};\n                    out[{out_index}] = gate * sigmoid * up_sum{j};\n                }}"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let function_name = match (static_n, static_k) {
            (Some(n), Some(k)) => format!("swiglu_gemv_n{n}_k{k}"),
            _ => "swiglu_gemv_dyn".to_string(),
        };

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 128
            #define SIMD_GROUP_SIZE 32
            #define SIMD_GROUPS_PER_TG 4
            #define COLS_PER_SIMDGROUP {accumulator_count}
            #define COLS_PER_GROUP (SIMD_GROUPS_PER_TG * COLS_PER_SIMDGROUP)
            #define LOG2E 1.4426950408889634f

            inline float4 load_rhs4(const device float *rhs, uint offset) {{
                const packed_float4 v = *(const device packed_float4 *)(rhs + offset);
                return float4(v);
            }}

            kernel void {function_name}(
                const device float *lhs [[buffer(0)]],
                const device float *gate_rhs [[buffer(1)]],
                const device float *up_rhs [[buffer(2)]],
                device float *out [[buffer(3)]],
                constant int *dyn [[buffer(4)]],
                constant uint &m [[buffer(5)]],
                constant uint &n [[buffer(6)]],
                constant uint &k [[buffer(7)]],
                uint2 gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]],
                uint simd_group [[simdgroup_index_in_threadgroup]]
            ) {{
                (void)dyn;
                (void)tid;

                const uint row = gid.y;
                const uint col_base = gid.x * COLS_PER_GROUP + simd_group * COLS_PER_SIMDGROUP;
                if (row >= m) {{
                    return;
                }}

                const uint lhs_base = row * k;
{sum_decls}

                const uint k4 = (k / 4) * 4;

                for (uint i = lane * 4; i < k4; i += SIMD_GROUP_SIZE * 4) {{
                    const packed_float4 lhs_vec = *(const device packed_float4 *)(lhs + lhs_base + i);
{dot_body}
                }}

                for (uint i = k4 + lane; i < k; i += SIMD_GROUP_SIZE) {{
                    const float lhs_val = lhs[lhs_base + i];
{tail_body}
                }}

{reduce_body}

                if (lane == 0) {{
{store_body}
                }}
            }}
            "#,
            accumulator_count = accumulator_count,
            function_name = function_name,
            sum_decls = sum_decls,
            dot_body = dot_body,
            tail_body = tail_body,
            reduce_body = reduce_body,
            store_body = store_body,
        );

        Some(compile_shader(device, &source, &function_name))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let m = self.m.exec(dyn_map).unwrap() as u32;
        let n = self.n.exec(dyn_map).unwrap() as u32;
        let k = self.k.exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(inputs[2]), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &n as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );

        let cols_per_group = if n == 8_192 && k == 2_048 {
            8
        } else if n >= 32_768 || k >= 8_192 {
            32
        } else {
            16
        };
        let thread_group_size = MTLSize::new(128, 1, 1);
        let thread_groups = MTLSize::new((n as u64).div_ceil(cols_per_group), m as u64, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        (m * k + 2 * n * k) * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        4 * m * n * k + 6 * m * n
    }

    fn is_matmul(&self) -> bool {
        true
    }
}
