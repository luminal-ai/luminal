//! Grouped MoE expert GEMV: per-(token, top-k slot) matrix-vector products
//! against the selected expert's weights, read directly from the full expert
//! weight tensor via on-device routing indices.
//!
//! `out[s, k, o] = Σ_d x[s, (k,) d] · W[topk[s,k], o, d]`
//!
//! The egglog rewrite matches the pure-HLIR gather-experts spelling the MoE
//! examples use (and that the GLUMoE host-op rules also match):
//!
//! ```text
//! flat_idx = topk·(O·D) + iota(z, (O, D))          ; expert offset + within
//! gathered = Gather(flat_idx, W[E,O,D])  [s,k,O,D] ; BF16
//! f32      = Cast(F32)(gathered)
//! mm       = Sum_d(Mul(x, f32^T))        [s,k,1,O] ; batched mat-vec
//! ```
//!
//! and unions this kernel into the matmul's eclass. Compared to extracting
//! that chain directly, the kernel streams each selected expert's BF16 rows
//! exactly once (no [s,k,O,D] gathered intermediate, no BF16→F32 weight-cast
//! round-trip), reads the routing indices on-device, and captures into the
//! decode CUDA graph — unlike the GLUMoE host op, which copies the routing
//! tensors to the host and drives a per-expert cuBLASLt loop with a stream
//! sync per layer.
//!
//! Both x layouts are handled by one rule: the x-operand strides over the
//! mul shape `[s,k,1,O,D]` are carried into the op, so a shared `(s, D)`
//! input (gate_up; k-stride 0) and a per-slot `(s, k, D)` input (down)
//! differ only in the embedded stride constants.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::dtype_includes, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, EXPRESSION, OP_KIND},
        extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

const TPB: usize = 256;
const WARPS_PER_BLOCK: usize = TPB / 32;

#[derive(Default, Debug, Clone)]
pub struct KernelMoEGemv {
    /// Matmul output shape `[s, k, 1, O]` (s may be dynamic).
    out_shape: Vec<Expression>,
    /// Reduce dimension (= D, the expert row length).
    k_dim: Expression,
    /// Expert weight tensor shape `[E, O, D]`.
    w_shape: Vec<Expression>,
    /// x-operand strides over the mul shape `[s, k, 1, O, D]`.
    x_strides: Vec<Expression>,
    /// topk-operand strides over `(s, k)`. NOT necessarily contiguous: the
    /// routing indices are typically a `..k` slice VIEW of the full `(s, E)`
    /// argsort output, so the row stride is E, not k. Reading the buffer as
    /// contiguous (s, k) silently picks bottom-ranked experts for rows ≥ 1.
    topk_strides: Vec<Expression>,
}

impl EgglogOp for KernelMoEGemv {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMoEGemv",
            &[
                ("out_shape", ELIST),
                ("k_dim", EXPRESSION),
                ("w_shape", ELIST),
                ("x_strides", ELIST),
                ("topk_strides", ELIST),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    ; expert flat index: topk · (O·D) + within-expert iota.
                    ; ?mb_a carries the topk read strides over (s, k) — the
                    ; indices are usually a slice view of the (s, E) argsort.
                    (= ?iota_base (Op (Iota ?io ?iota_base_range) (INil)))
                    (= ?mul_base (Op (Mul ?mb_shape ?mb_a ?mb_b ?mb_o)
                        (ICons ?topk_idx (ICons ?iota_base (INil)))))
                    (= ?mb_a (ECons ?ts_s (ECons ?ts_k (ENil))))
                    (= ?iota_within (Op (Iota (MIter) ?iota_within_range) (INil)))
                    (= ?add_idx (Op (Add ?ai_shape ?ai_a ?ai_b ?ai_o)
                        (ICons ?mul_base (ICons ?iota_within (INil)))))

                    ; gather the selected experts' BF16 weights, widen to F32
                    (= ?gathered (Op (Gather ?g_idx_shape ?g_idx_stride ?g_data_shape ?g_data_stride)
                        (ICons ?add_idx (ICons ?weights (INil)))))
                    (= (Bf16) (dtype ?weights))
                    (= ?f32 (Op (Cast ?f32_size (F32)) (ICons ?gathered (INil))))

                    ; batched mat-vec against the transposed gathered view
                    (= ?mm_mul (Op (Mul ?mm_shape ?x_strides ?w_strides ?mm_mul_out)
                        (ICons ?x (ICons ?f32 (INil)))))
                    (= (F32) (dtype ?x))
                    (= ?mm (Op (Sum ?out_shape ?k_dim ?mm_in_stride ?mm_k_stride ?mm_out_stride)
                        (ICons ?mm_mul (INil))))

                    ; weights are (E, O, D); reduce dim must be D
                    (= ?g_data_shape (ECons ?e (ECons ?o (ECons ?d (ENil)))))
                    (= ?d ?k_dim)
                    ; matmul output is (s, k, 1, O)
                    (= ?out_shape (ECons ?s (ECons ?k (ECons (MNum 1) (ECons ?o2 (ENil))))))
                    (= ?o ?o2)
                    ; gathered operand reads (s,k,O,D) through the transpose
                    ; view: O-dim stride z·D, D-dim stride z
                    (= ?w_strides (ECons ?ws_s (ECons ?ws_k (ECons ?ws_1
                        (ECons (MMul (MIter) ?d3) (ECons (MIter) (ENil)))))))
                    (= ?d3 ?d)
                    ; x reads its row contiguously and broadcasts over O
                    (= ?x_strides (ECons ?xs_s (ECons ?xs_k (ECons ?xs_1
                        (ECons (MNum 0) (ECons (MIter) (ENil)))))))
                )
                (
                    (let ?kmoe (Op (KernelMoEGemv ?out_shape ?k_dim ?g_data_shape ?x_strides ?mb_a)
                        (ICons ?x (ICons ?topk_idx (ICons ?weights (INil))))))
                    (union ?mm ?kmoe)
                    (set (dtype ?kmoe) (F32))
                )
                :ruleset matmul_backend
                :name \"kernel moe gemv from expert gather\"
            )",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                k_dim: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                w_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                x_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                topk_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelMoEGemv {
    /// (s, top_k, d_out, d_in, n_experts, x s-stride, x k-stride,
    /// topk s-stride, topk k-stride)
    #[allow(clippy::type_complexity)]
    fn dims(
        &self,
    ) -> (
        Expression,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let s = self.out_shape[0];
        let top_k = self.out_shape[1].to_usize().expect("MoE top_k is static");
        let d_out = self.out_shape[3].to_usize().expect("MoE d_out is static");
        let d_in = self.k_dim.to_usize().expect("MoE d_in is static");
        let n_experts = self.w_shape[0].to_usize().expect("MoE E is static");
        let resolve = |e: Expression| -> usize {
            e.substitute('z', Expression::from(1usize))
                .simplify()
                .to_usize()
                .expect("MoE x stride is static")
        };
        let xs_s = resolve(self.x_strides[0]);
        let xs_k = resolve(self.x_strides[1]);
        let ts_s = resolve(self.topk_strides[0]);
        let ts_k = resolve(self.topk_strides[1]);
        (s, top_k, d_out, d_in, n_experts, xs_s, xs_k, ts_s, ts_k)
    }
}

impl KernelOp for KernelMoEGemv {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let (s, k, o, d, e, xs_s, xs_k, ts_s, ts_k) = self.dims();
        let includes = dtype_includes(&[DType::Bf16]);

        let vars: FxHashSet<char> = s.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let s_expr = s.to_kernel();

        // 16-byte lane loads (8 bf16 + 2×float4) when D is 8-aligned.
        let body = if d % 8 == 0 {
            format!(
                r#"
        float acc = 0.0f;
        for (int i = lane * 8; i < {d}; i += 32 * 8) {{
            uint4 wv = __ldcs((const uint4*)(wr + i));
            const __nv_bfloat16* we = (const __nv_bfloat16*)&wv;
            float4 xv0 = *(const float4*)(xr + i);
            float4 xv1 = *(const float4*)(xr + i + 4);
            acc += (float)we[0] * xv0.x + (float)we[1] * xv0.y
                 + (float)we[2] * xv0.z + (float)we[3] * xv0.w
                 + (float)we[4] * xv1.x + (float)we[5] * xv1.y
                 + (float)we[6] * xv1.z + (float)we[7] * xv1.w;
        }}"#
            )
        } else {
            format!(
                r#"
        float acc = 0.0f;
        for (int i = lane; i < {d}; i += 32) {{
            acc += (float)wr[i] * xr[i];
        }}"#
            )
        };

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void moe_gemv_k(float *out, const float *x, const int *topk, const __nv_bfloat16 *w{dyn_dims_param}) {{
        const long long total = (long long)({s_expr}) * {k} * {o};
        long long gw = (long long)blockIdx.x * {WARPS_PER_BLOCK} + (threadIdx.x >> 5);
        if (gw >= total) return;
        int lane = threadIdx.x & 31;
        int o_idx = (int)(gw % {o});
        long long sk = gw / {o};
        long long s_idx = sk / {k};
        long long kk = sk % {k};

        int expert = topk[s_idx * {ts_s} + kk * {ts_k}];
        if (expert < 0 || expert >= {e}) {{
            if (lane == 0) out[gw] = 0.0f;
            return;
        }}
        const __nv_bfloat16* wr = w + ((long long)expert * {o} + o_idx) * {d};
        const float* xr = x + s_idx * {xs_s} + kk * {xs_k};
{body}

        #pragma unroll
        for (int sft = 16; sft > 0; sft >>= 1) {{
            acc += __shfl_down_sync(0xffffffff, acc, sft);
        }}
        if (lane == 0) {{
            out[gw] = acc;
        }}
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("moe_gemv_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let (s, k, o, ..) = self.dims();
        (
            func,
            module,
            kernel,
            (
                (s * k * o).ceil_div(WARPS_PER_BLOCK),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        let (s, k, o, ..) = self.dims();
        s * k * o
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape[0].dyn_vars().into_iter().collect()
    }

    fn bytes_loaded(&self) -> Expression {
        // Each (s, k) slot streams one expert's O×D BF16 rows plus its input
        // row and routing index.
        let (s, k, o, d, ..) = self.dims();
        s * k * (o * d * 2 + d * 4 + 4)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let (s, k, o, d, ..) = self.dims();
        s * k * o * d * 2
    }

    fn kernel_name(&self) -> &'static str {
        "MoEGemv"
    }
}
