use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{ELIST, EXPRESSION, F64, IR},
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

/// Fuses the decomposed RMSNorm subgraph into one row-wise kernel:
/// compute `scale = rsqrt(mean(x * x) + eps)` for each row, then write
/// `x * scale * weight` with the requested output strides. This replaces the
/// separate square/sum/scale/rsqrt/multiply kernels used in decoder blocks.
#[derive(Debug, Clone, Default)]
pub struct MetalRmsNorm {
    rows: Expression,
    cols: Expression,
    eps: f32,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalRmsNorm {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalRmsNorm",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("eps", F64),
                ("input", IR),
                ("weight", IR),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            r#"
            (rule
                (
                    (= ?eps_const (MetalConstant ?eps))
                    (= ?sum_sq
                        (GenericMatmul
                            (ECons ?rows (ENil))
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?cols
                            ?input
                            ?input_lhs_strides
                            ?input
                            ?input_rhs_strides
                            ?sum_input_strides
                            ?sum_iter_stride
                            ?sum_out_strides))
                    (= ?mean (Op
                        (MetalMul
                            (ECons ?rows (ENil))
                            ?sum_out_strides
                            ?mean_scale_strides
                            ?mean_out_strides)
                        (ICons ?sum_sq (ICons ?mean_scale (INil)))))
                    (= ?with_eps (Op
                        (MetalAdd
                            (ECons ?rows (ENil))
                            ?mean_out_strides
                            ?eps_strides
                            ?eps_out_strides)
                        (ICons ?mean (ICons ?eps_const (INil)))))
                    (= ?sqrt (Op
                        (MetalSqrt
                            (ECons ?rows (ENil))
                            ?eps_out_strides
                            ?sqrt_out_strides)
                        (ICons ?with_eps (INil))))
                    (= ?inv (Op
                        (MetalRecip
                            (ECons ?rows (ENil))
                            ?sqrt_out_strides
                            ?inv_out_strides)
                        (ICons ?sqrt (INil))))
                    (= ?normed (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?inv_broadcast_strides
                            ?norm_input_strides
                            ?norm_out_strides)
                        (ICons ?inv (ICons ?input (INil)))))
                    (= ?weighted (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?norm_out_strides
                            (ECons (MNum 0) (ECons (MIter) (ENil)))
                            ?weighted_out_strides)
                        (ICons ?normed (ICons ?weight (INil)))))
                )
                (
                    (let ?fused (MetalRmsNorm ?rows ?cols ?eps ?input ?weight ?weighted_out_strides))
                    (union ?weighted ?fused)
                    (set (dtype ?fused) (F32))
                )
                :ruleset matmul_backend
                :name "metal-fused-weighted-rms-norm"
            )

            (rule
                (
                    (= ?eps_const (MetalConstant ?eps))
                    (= ?fused (MetalRmsNorm ?rows ?cols ?eps ?input ?weight ?weighted_out_strides))
                    (= ?sum_sq
                        (GenericMatmul
                            (ECons ?rows (ENil))
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?cols
                            ?input
                            ?input_lhs_strides
                            ?input
                            ?input_rhs_strides
                            ?sum_input_strides
                            ?sum_iter_stride
                            ?sum_out_strides))
                    (= ?mean (Op
                        (MetalMul
                            (ECons ?rows (ENil))
                            ?sum_out_strides
                            ?mean_scale_strides
                            ?mean_out_strides)
                        (ICons ?sum_sq (ICons ?mean_scale (INil)))))
                    (= ?with_eps (Op
                        (MetalAdd
                            (ECons ?rows (ENil))
                            ?mean_out_strides
                            ?eps_strides
                            ?eps_out_strides)
                        (ICons ?mean (ICons ?eps_const (INil)))))
                    (= ?sqrt (Op
                        (MetalSqrt
                            (ECons ?rows (ENil))
                            ?eps_out_strides
                            ?sqrt_out_strides)
                        (ICons ?with_eps (INil))))
                    (= ?inv (Op
                        (MetalRecip
                            (ECons ?rows (ENil))
                            ?sqrt_out_strides
                            ?inv_out_strides)
                        (ICons ?sqrt (INil))))
                    (= ?normed (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?inv_broadcast_strides
                            ?norm_input_strides
                            ?norm_out_strides)
                        (ICons ?inv (ICons ?input (INil)))))
                    (= ?weighted (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?norm_out_strides
                            (ECons (MNum 0) (ECons (MIter) (ENil)))
                            ?weighted_out_strides)
                        (ICons ?normed (ICons ?weight (INil)))))
                )
                (
                    (delete (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?norm_out_strides
                            (ECons (MNum 0) (ECons (MIter) (ENil)))
                            ?weighted_out_strides)
                        (ICons ?normed (ICons ?weight (INil)))))
                    (delete (Op
                        (MetalMul
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?inv_broadcast_strides
                            ?norm_input_strides
                            ?norm_out_strides)
                        (ICons ?inv (ICons ?input (INil)))))
                    (delete (Op
                        (MetalRecip
                            (ECons ?rows (ENil))
                            ?sqrt_out_strides
                            ?inv_out_strides)
                        (ICons ?sqrt (INil))))
                    (delete (Op
                        (MetalSqrt
                            (ECons ?rows (ENil))
                            ?eps_out_strides
                            ?sqrt_out_strides)
                        (ICons ?with_eps (INil))))
                    (delete (Op
                        (MetalAdd
                            (ECons ?rows (ENil))
                            ?mean_out_strides
                            ?eps_strides
                            ?eps_out_strides)
                        (ICons ?mean (ICons ?eps_const (INil)))))
                    (delete (Op
                        (MetalMul
                            (ECons ?rows (ENil))
                            ?sum_out_strides
                            ?mean_scale_strides
                            ?mean_out_strides)
                        (ICons ?sum_sq (ICons ?mean_scale (INil)))))
                    (delete
                        (GenericMatmul
                            (ECons ?rows (ENil))
                            (ECons ?rows (ECons ?cols (ENil)))
                            ?cols
                            ?input
                            ?input_lhs_strides
                            ?input
                            ?input_rhs_strides
                            ?sum_input_strides
                            ?sum_iter_stride
                            ?sum_out_strides))
                )
                :ruleset cleanup
                :name "delete-decomposed-weighted-rms-norm"
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

        let eps = egraph.enodes[children[2]]
            .0
            .replace('\"', "")
            .parse::<f32>()
            .expect("MetalRmsNorm eps must be a float");

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                rows: extract_expr(egraph, children[0], expr_cache).unwrap(),
                cols: extract_expr(egraph, children[1], expr_cache).unwrap(),
                eps,
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[3], children[4]],
        )
    }
}

impl MetalKernelOp for MetalRmsNorm {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "RmsNorm"
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

        let shape = vec![self.rows, self.cols];
        let out_flat = flatten_strides(&shape, &self.output_strides);
        let out_index = lower_expression_for_metal(&out_flat, "idx");
        let output_body = if out_index == "idx" {
            r#"
                for (uint col = tid * 4; col < cols4; col += THREADS_PER_GROUP * 4) {
                    const uint idx = row_base + col;
                    const packed_float4 x = *(const device packed_float4 *)(input + idx);
                    const packed_float4 w = *(const device packed_float4 *)(weight + col);
                    const float4 y = float4(x) * scale * float4(w);
                    *(device packed_float4 *)(out + idx) = packed_float4(y);
                }
                for (uint col = cols4 + tid; col < cols; col += THREADS_PER_GROUP) {
                    const uint idx = row_base + col;
                    out[idx] = input[idx] * scale * weight[col];
                }
            "#
            .to_string()
        } else {
            format!(
                r#"
                for (uint col = tid; col < cols; col += THREADS_PER_GROUP) {{
                    const uint idx = row_base + col;
                    out[{out_index}] = input[idx] * scale * weight[col];
                }}
            "#
            )
        };
        let eps = self.eps;
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 128
            #define SIMD_GROUP_SIZE 32
            #define SIMD_GROUPS_PER_TG 4
            #define EPS {eps:.9e}f

            kernel void rms_norm_kernel(
                const device float *input [[buffer(0)]],
                const device float *weight [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant int *dyn [[buffer(3)]],
                constant uint &rows [[buffer(4)]],
                constant uint &cols [[buffer(5)]],
                uint row [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]],
                uint simd_group [[simdgroup_index_in_threadgroup]]
            ) {{
                (void)dyn;
                if (row >= rows) {{
                    return;
                }}

                threadgroup float partials[SIMD_GROUPS_PER_TG];
                threadgroup float row_scale;
                const uint row_base = row * cols;

                float sum = 0.0f;
                const uint cols4 = (cols / 4) * 4;
                for (uint col = tid * 4; col < cols4; col += THREADS_PER_GROUP * 4) {{
                    const packed_float4 v = *(const device packed_float4 *)(input + row_base + col);
                    const float4 vf = float4(v);
                    sum += dot(vf, vf);
                }}
                for (uint col = cols4 + tid; col < cols; col += THREADS_PER_GROUP) {{
                    const float v = input[row_base + col];
                    sum += v * v;
                }}

                sum = simd_sum(sum);
                if (lane == 0) {{
                    partials[simd_group] = sum;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (simd_group == 0) {{
                    float total = lane < SIMD_GROUPS_PER_TG ? partials[lane] : 0.0f;
                    total = simd_sum(total);
                    if (lane == 0) {{
                        row_scale = fast::rsqrt(total / float(cols) + EPS);
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const float scale = row_scale;
{output_body}
            }}
            "#,
            eps = eps,
            output_body = output_body,
        );
        Some(compile_shader(device, &source, "rms_norm_kernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        (self.rows * self.cols).max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let rows = self.rows.exec(dyn_map).unwrap() as u32;
        let cols = self.cols.exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &rows as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &cols as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(128, 1, 1);
        let thread_groups = MTLSize::new(rows as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        3 * n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        4 * n
    }
}
