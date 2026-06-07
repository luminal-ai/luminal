use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, Term as EggTerm, app, eq, i64 as lit_i64, rule, sort, union, v},
        base::{EXPRESSION, IR, SORTS, cons, dtype, ilist, iter, mul, nil, num, op_term},
    },
    op::*,
    prelude::*,
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use crate::kernel::{
    MetalKernelOp,
    ops::{MetalMul, MetalSumReduce, compile_shader},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSMatrixLayout {
    RowMajor,
    TransposedRowMajor,
}

/// Specialized row-major lhs by transposed row-major rhs matmul for decode GEMV.
/// Cleanup keeps this candidate only for single-row/decode buckets (m < 2);
/// multi-row buckets prefer MPSMatmul.
/// [m, k] x [n, k].T -> [m, n]
/// out[i, j] = sum(lhs[i, t] * rhs[j, t], t in 0..k)
#[derive(Debug, Default, Clone)]
pub struct MetalTransposedRhsGemv {
    m: Expression,
    n: Expression,
    k: Expression,
}

impl EgglogOp for MetalTransposedRhsGemv {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalTransposedRhsGemv",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("rhs", IR),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let zero = num(lit_i64(0));
        let z = iter();
        let expr_list = |terms: Vec<EggTerm>| {
            terms
                .into_iter()
                .rev()
                .fold(nil(), |tail, head| cons(head, tail))
        };

        let m = v("?metal_transposed_rhs_gemv_m");
        let n = v("?metal_transposed_rhs_gemv_n");
        let k = v("?metal_transposed_rhs_gemv_k");
        let lhs = v("?metal_transposed_rhs_gemv_lhs");
        let rhs = v("?metal_transposed_rhs_gemv_rhs");
        let out_row_stride = mul(n.clone(), z.clone());
        let mul_op = op_term(
            MetalMul::default().sort().call([
                (
                    "shape",
                    cons(m.clone(), cons(n.clone(), cons(k.clone(), nil()))),
                ),
                (
                    "a_strides",
                    expr_list(vec![mul(k.clone(), z.clone()), zero.clone(), z.clone()]),
                ),
                (
                    "b_strides",
                    expr_list(vec![zero.clone(), mul(k.clone(), z.clone()), z.clone()]),
                ),
                (
                    "out_strides",
                    v("?metal_transposed_rhs_gemv_mul_output_strides"),
                ),
            ]),
            ilist(vec![lhs.clone(), rhs.clone()]),
        );
        let sum_op = op_term(
            MetalSumReduce::default().sort().call([
                ("shape", cons(m.clone(), cons(n.clone(), nil()))),
                ("iters", k.clone()),
                ("strides", v("?metal_transposed_rhs_gemv_sum_input_strides")),
                ("iter_stride", z.clone()),
                (
                    "out_strides",
                    cons(out_row_stride.clone(), cons(z.clone(), nil())),
                ),
            ]),
            ilist(vec![mul_op.clone()]),
        );
        let gemv_op = Self::default().sort().call([
            ("m", m.clone()),
            ("n", n.clone()),
            ("k", k.clone()),
            ("lhs", lhs.clone()),
            ("rhs", rhs.clone()),
        ]);

        vec![
            rule(union(sum_op.clone(), gemv_op.clone()))
                .set(dtype(gemv_op), app(&SORTS.f32_dt, vec![]))
                .fact(eq(app(&SORTS.f32_dt, vec![]), dtype(sum_op)))
                .ruleset("matmul_backend")
                .name("metal-transposed-rhs-gemv-row-transposed-rhs"),
            // Materialize the row-count predicate so interval simplification can
            // prove whether this is (m < 2) or multi-row.
            Rule::raw(
                "(rule
                    ((= ?matmul (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs)))
                    ((union (MGte ?m (MNum 2)) (MGte ?m (MNum 2))))
                    :ruleset matmul_backend
                    :name \"materialize-metal-transposed-rhs-gemv-row-count-check\"
                )",
            ),
            // Once the fused GEMV exists, remove the decomposed broadcast
            // multiply + sum that produced the same matmul.
            Rule::raw(
                "(rule
                    ((= ?mul (Op (MetalMul ?shape ?as ?bs ?os) ?inputs))
                     (= ?sum (Op (MetalSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (= ?sum (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs)))
                    ((delete (Op (MetalSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (delete (Op (MetalMul ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                    :name \"delete-broadcast-mul-sum-when-metal-transposed-rhs-gemv-exists\"
                )",
            ),
            // Prefer the specialized transposed-RHS GEMV over the generic
            // matmul fallback whenever both describe the same computation.
            Rule::raw(
                "(rule
                    ((= ?matmul (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos))
                     (= ?matmul (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs)))
                    ((subsume (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    :ruleset cleanup
                    :name \"metal-transposed-rhs-gemv-subsumes-generic-matmul\"
                )",
            ),
            // For single-row/decode matmul, keep the custom GEMV candidate and
            // remove the equivalent MPS matmul candidate.
            Rule::raw(
                "(rule
                    ((= ?matmul (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs))
                     (= ?matmul (MPSMatmul ?m ?n ?k ?lhs ?lhsrs ?rhs ?rhsrs ?ors ?tl ?tr))
                     (= (MGte ?m (MNum 2)) (MNum 0)))
                    ((subsume (MPSMatmul ?m ?n ?k ?lhs ?lhsrs ?rhs ?rhsrs ?ors ?tl ?tr)))
                    :ruleset cleanup
                    :name \"metal-transposed-rhs-gemv-subsumes-single-row-mps-matmul\"
                )",
            ),
            // When this is not a single-row/decode matmul, remove the custom
            // GEMV candidate and keep the MPS matmul candidate.
            Rule::raw(
                "(rule
                    ((= ?matmul (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs))
                     (= ?matmul (MPSMatmul ?m ?n ?k ?lhs ?lhsrs ?rhs ?rhsrs ?ors ?tl ?tr))
                     (= (MGte ?m (MNum 2)) (MNum 1)))
                    ((subsume (MetalTransposedRhsGemv ?m ?n ?k ?lhs ?rhs)))
                    :ruleset cleanup
                    :name \"mps-matmul-subsumes-multi-row-metal-transposed-rhs-gemv\"
                )",
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                m: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                n: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                k: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
            })),
            vec![kind_children[3], kind_children[4]],
        )
    }
}

impl MetalKernelOp for MetalTransposedRhsGemv {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "MetalTransposedRhsGemv"
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
            "MetalTransposedRhsGemv lhs must be F32"
        );
        assert_eq!(
            input_dtypes.get(1).copied().unwrap_or(DType::F32),
            DType::F32,
            "MetalTransposedRhsGemv rhs must be F32"
        );
        assert_eq!(
            output_dtype,
            DType::F32,
            "MetalTransposedRhsGemv output must be F32"
        );
        let source = r#"
        #include <metal_stdlib>
        using namespace metal;

        #define THREADS_PER_GROUP 64
        #define SIMD_GROUP_SIZE 32
        #define SIMD_GROUPS_PER_TG 2
        #define COLS_PER_SIMDGROUP 4
        #define COLS_PER_GROUP (SIMD_GROUPS_PER_TG * COLS_PER_SIMDGROUP)

        kernel void gemv16_kernel(
            const device float *lhs [[buffer(0)]],
            const device float *rhs [[buffer(1)]],
            device float *out [[buffer(2)]],
            constant int *dyn [[buffer(3)]],
            constant uint &m [[buffer(4)]],
            constant uint &n [[buffer(5)]],
            constant uint &k [[buffer(6)]],
            uint2 gid [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint lane [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            (void)dyn;
            (void)tid;

            const uint row = gid.y;
            const uint col_base = gid.x * COLS_PER_GROUP + simd_group * COLS_PER_SIMDGROUP;
            if (row >= m) {
                return;
            }

            const uint lhs_base = row * k;
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            const uint k4 = (k / 4) * 4;

            for (uint i = lane * 4; i < k4; i += SIMD_GROUP_SIZE * 4) {
                const packed_float4 lhs_vec = *(const device packed_float4 *)(lhs + lhs_base + i);
                if (col_base + 0 < n) {
                    const packed_float4 rhs_vec = *(const device packed_float4 *)(rhs + (col_base + 0) * k + i);
                    sum0 += dot(float4(lhs_vec), float4(rhs_vec));
                }
                if (col_base + 1 < n) {
                    const packed_float4 rhs_vec = *(const device packed_float4 *)(rhs + (col_base + 1) * k + i);
                    sum1 += dot(float4(lhs_vec), float4(rhs_vec));
                }
                if (col_base + 2 < n) {
                    const packed_float4 rhs_vec = *(const device packed_float4 *)(rhs + (col_base + 2) * k + i);
                    sum2 += dot(float4(lhs_vec), float4(rhs_vec));
                }
                if (col_base + 3 < n) {
                    const packed_float4 rhs_vec = *(const device packed_float4 *)(rhs + (col_base + 3) * k + i);
                    sum3 += dot(float4(lhs_vec), float4(rhs_vec));
                }
            }

            for (uint i = k4 + lane; i < k; i += SIMD_GROUP_SIZE) {
                const float lhs_val = lhs[lhs_base + i];
                if (col_base + 0 < n) {
                    sum0 += lhs_val * rhs[(col_base + 0) * k + i];
                }
                if (col_base + 1 < n) {
                    sum1 += lhs_val * rhs[(col_base + 1) * k + i];
                }
                if (col_base + 2 < n) {
                    sum2 += lhs_val * rhs[(col_base + 2) * k + i];
                }
                if (col_base + 3 < n) {
                    sum3 += lhs_val * rhs[(col_base + 3) * k + i];
                }
            }

            sum0 = simd_sum(sum0);
            sum1 = simd_sum(sum1);
            sum2 = simd_sum(sum2);
            sum3 = simd_sum(sum3);

            if (lane == 0) {
                const uint out_base = row * n + col_base;
                if (col_base + 0 < n) {
                    out[out_base + 0] = sum0;
                }
                if (col_base + 1 < n) {
                    out[out_base + 1] = sum1;
                }
                if (col_base + 2 < n) {
                    out[out_base + 2] = sum2;
                }
                if (col_base + 3 < n) {
                    out[out_base + 3] = sum3;
                }
            }
        }
        "#;
        Some(compile_shader(device, source, "gemv16_kernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        (self.m * self.n).max(Expression::from(1))
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
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &n as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(64, 1, 1);
        let thread_groups = MTLSize::new((n as u64).div_ceil(8), m as u64, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        (m * k + n * k) * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * m * n * k
    }

    fn is_matmul(&self) -> bool {
        true
    }
}
