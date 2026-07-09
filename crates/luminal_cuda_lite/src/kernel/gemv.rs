//! Warp-per-row GEMV for M=1 decode matmuls.
//!
//! Matches the same row-major × column-major `GenericMatmul` pattern as the
//! cuBLASLt RmCm rewrite, restricted to m == 1 and 16-bit dtypes, and unions
//! a lean kernel into the same eclass — the GA search then picks cuBLASLt or
//! this per shape. Rationale (measured on Llama 3 8B decode): the nvjet GEMV
//! kernels carry ~5µs of fixed launch/tail cost per call, which dominates the
//! small projections (o-proj: 13.2µs vs an 8.3µs traffic floor). One warp per
//! output row with vectorized 16-byte loads and F32 accumulation has ~1µs of
//! fixed cost and no splitK reduction pass.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{KernelOp, hlir::dtype_includes, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
#[allow(unused_imports)]
use luminal::dtype::DType;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::*,
    prelude::*,
};

const TPB: usize = 256;
const WARPS_PER_BLOCK: usize = TPB / 32;

#[derive(Default, Debug, Clone)]
pub struct KernelGemv {
    n: Expression,
    k: Expression,
    dtype: DType,
}

impl EgglogOp for KernelGemv {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGemv",
            &[("n", EXPRESSION), ("k", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Same structural anchors as "cublaslt row-major × column-major",
        // restricted to m == 1 (decode GEMV) and 16-bit dtypes.
        // Two m-shape variants per dtype: a literal (MNum 1) for static
        // graphs, and an interval-pinned dyn dim (the decode bucket proves
        // s in [1,1]; `lower`/`upper` have no rows for literals).
        let m_variants: [(&str, &str); 2] = [
            (
                "static",
                "(= ?out_shape (ECons (MNum 1) (ECons ?n (ENil))))",
            ),
            (
                "dyn",
                "(= ?out_shape (ECons ?m (ECons ?n (ENil))))
                            (= ?m_lower (lower ?m))
                            (= ?m_upper (upper ?m))
                            (> ?m_lower 0)
                            (< ?m_upper 2)",
            ),
        ];
        m_variants
            .into_iter()
            .flat_map(|(variant, m_cond)| ["Bf16", "F16"].map(move |dt| (variant, m_cond, dt)))
            .map(|(variant, m_cond, dt)| {
                Rule::raw(format!(
                    "(rule
                        (
                            (= ?sum (Op (GenericMatmul
                                ?out_shape ?mul_shape ?k
                                ?a_stride ?b_stride
                                ?sum_in_stride ?k_stride ?sum_out_stride
                                ?matmul_dtype)
                                (ICons ?a (ICons ?b (INil)))))

                            {m_cond}
                            (!= ?n (MNum 0))
                            (!= ?k (MNum 1))

                            (= ?a_stride (ECons ?a_m_stride (ECons ?a_n_stride (ECons ?a_k_stride (ENil)))))
                            (= ?b_stride (ECons ?b_m_stride (ECons ?b_n_stride (ECons ?b_k_stride (ENil)))))
                            (= ?k_stride (MIter))

                            ; A (the activation row) is row-major [1, k]
                            (= ?a_m_stride (MMul (MIter) ?k))
                            (= ?a_n_stride (MNum 0))
                            (= ?a_k_stride (MIter))

                            ; B is column-major [k, n] — i.e. the weight stored
                            ; row-major [n, k], read row by row.
                            (= ?b_m_stride (MNum 0))
                            (= ?b_n_stride (MMul (MIter) ?k))
                            (= ?b_k_stride (MIter))

                            (= ?dt (dtype ?a))
                            (= ?dt (dtype ?b))
                            (= ?dt ({dt}))
                        )
                        (
                            (let ?gemv (Op (KernelGemv ?n ?k ({dt})) (ICons ?a (ICons ?b (INil)))))
                            (union ?sum ?gemv)
                            (set (dtype ?gemv) ({dt}))
                        )
                        :ruleset matmul_backend
                        :name \"kernel gemv m1 {dt} {variant}\"
                    )"
                ))
            })
            .collect()
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
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                n: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                k: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelGemv {
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
        let vars = self
            .n
            .dyn_vars()
            .into_iter()
            .chain(self.k.dyn_vars())
            .collect::<FxHashSet<_>>();
        let ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n = self.n.to_kernel();
        let k = self.k.to_kernel();

        // 16-byte vectorized lane loads (8 × 16-bit) when K is statically a
        // multiple of 8; scalar loop otherwise. F32 accumulation throughout.
        let vectorized = self.k.to_usize().map(|k| k % 8 == 0).unwrap_or(false);
        let body = if vectorized {
            format!(
                r#"
        const uint4* xr = (const uint4*)x;
        const uint4* wr = (const uint4*)(w + row * ({k}));
        long long chunks = ({k}) / 8;
        float acc = 0.0f;
        for (long long c = lane; c < chunks; c += 32) {{
            uint4 xv = xr[c];
            uint4 wv = wr[c];
            const {ty}* xe = (const {ty}*)&xv;
            const {ty}* we = (const {ty}*)&wv;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                acc += (float)xe[e] * (float)we[e];
            }}
        }}"#
            )
        } else {
            format!(
                r#"
        const {ty}* wr = w + row * ({k});
        float acc = 0.0f;
        for (long long i = lane; i < ({k}); i += 32) {{
            acc += (float)x[i] * (float)wr[i];
        }}"#
            )
        };

        let kernel = format!(
            "{includes}
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void gemv_k({ty} *out, const {ty} *x, const {ty} *w{dyn_dims_param}) {{
        long long row = (long long)blockIdx.x * {WARPS_PER_BLOCK} + (threadIdx.x >> 5);
        if (row >= ({n})) return;
        int lane = threadIdx.x & 31;
{body}

        #pragma unroll
        for (int s = 16; s > 0; s /= 2) {{
            acc += __shfl_down_sync(FULL_MASK, acc, s);
        }}
        if (lane == 0) {{
            out[row] = ({ty})acc;
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gemv_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (self.n.ceil_div(WARPS_PER_BLOCK), 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.n
    }

    fn output_bytes(&self) -> Expression {
        (self.n * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        ((self.n * self.k + self.k) * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.n * self.k * 2
    }

    fn kernel_name(&self) -> &'static str {
        "Gemv"
    }
}

// ═══════════════════════════════════════════════════════════
// Tensor-core scaled FP8 GEMV:
//   out_bf16[n] = (Σ_k x_f8[k]·w_f8[n,k]) · in_scale·w_scale
//
// Matches the scaled-fp8 linear chain (KernelQuantF8 activation →
// GenericMatmul → Cast(F32) → ×(in_scale·w_scale) → Cast(Bf16)) at m == 1
// and fuses GEMM + dequant + output cast into one mma.sync kernel.
//
// Why tensor cores: a CUDA-core fp8 GEMV is cvt-ALU-bound (fp8→f16→f32 per
// element) at ~2.2× the traffic floor; mma.sync.m16n8k32.e4m3 contracts
// 32 fp8 per instruction with f32 accumulators, leaving the kernel purely
// bandwidth-bound (75-83% of floor measured vs nvjet's 51-67%).
//
// Layout (the M=1 k-permutation trick): the contraction only requires the A
// and B fragments to carry the SAME physical k in matching slots, so each
// lane feeds the mma from plain contiguous 16-byte loads — no ldmatrix, no
// shared-memory staging. Lane l of a warp covers row n = l/4 and k-slot
// group lg = l%4: per 128-k step it loads two uint4 at
// w[n][k0 + 32·lg (+16)], x replicated across A rows (D is row-replicated;
// lanes 0-3 hold the result row). 8 warps per block split K over the same
// 8 rows (grid = N/8 keeps all SMs busy at small N), reduced through smem.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelGemvF8 {
    n: Expression,
    k: Expression,
}

impl EgglogOp for KernelGemvF8 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGemvF8",
            &[("n", EXPRESSION), ("k", EXPRESSION)],
        )
    }

    fn n_inputs(&self) -> usize {
        4
    }

    fn rewrites(&self) -> Vec<Rule> {
        let m_variants: [(&str, &str); 2] = [
            (
                "static",
                "(= ?out_shape (ECons (MNum 1) (ECons ?n (ENil))))",
            ),
            (
                "dyn",
                "(= ?out_shape (ECons ?m (ECons ?n (ENil))))
                        (= ?m_lower (lower ?m))
                        (= ?m_upper (upper ?m))
                        (> ?m_lower 0)
                        (< ?m_upper 2)",
            ),
        ];
        // A-operand variants: the standalone fused quant kernel, plus the
        // fused norm+quant / swiglu+quant F8 custom ops. CONVENTION: f8
        // custom ops put the quantization input scale as their LAST input —
        // these patterns match by arity and read the scale from that slot.
        let a_variants: [(&str, &str); 3] = [
            (
                "quant",
                "(= ?a (Op (KernelQuantF8 ?q_size) (ICons ?x (ICons ?in_scale (INil)))))",
            ),
            (
                "custom3",
                "(= ?a (Op (CustomOpKind ?cid (F8E4M3)) (ICons ?cx (ICons ?cw (ICons ?in_scale (INil))))))",
            ),
            (
                "custom2",
                "(= ?a (Op (CustomOpKind ?cid (F8E4M3)) (ICons ?cx (ICons ?in_scale (INil)))))",
            ),
        ];
        m_variants
            .into_iter()
            .flat_map(|m| a_variants.into_iter().map(move |a| (m, a)))
            .map(|((variant, m_cond), (a_variant, a_cond))| {
                Rule::raw(format!(
                    "(rule
                    (
                        ; A is a pre-quantized f8 activation with its input
                        ; scale recoverable from the producing op.
                        {a_cond}

                        (= ?sum (Op (GenericMatmul
                            ?out_shape ?mul_shape ?k
                            ?a_stride ?b_stride
                            ?sum_in_stride ?k_stride ?sum_out_stride
                            ?matmul_dtype)
                            (ICons ?a (ICons ?b (INil)))))

                        {m_cond}
                        (!= ?n (MNum 0))
                        (!= ?k (MNum 1))

                        (= ?a_stride (ECons ?a_m_stride (ECons ?a_n_stride (ECons ?a_k_stride (ENil)))))
                        (= ?b_stride (ECons ?b_m_stride (ECons ?b_n_stride (ECons ?b_k_stride (ENil)))))
                        (= ?k_stride (MIter))
                        (= ?a_m_stride (MMul (MIter) ?k))
                        (= ?a_n_stride (MNum 0))
                        (= ?a_k_stride (MIter))
                        (= ?b_m_stride (MNum 0))
                        (= ?b_n_stride (MMul (MIter) ?k))
                        (= ?b_k_stride (MIter))
                        (= (F8E4M3) (dtype ?b))

                        ; dequant: Cast(F32) → × (in_scale · w_scale) → Cast(Bf16)
                        (= ?cast (Op (Cast ?c_size (F32)) (ICons ?sum (INil))))
                        (= ?scale_product (Op (Mul (ENil) (ENil) (ENil) (ENil))
                            (ICons ?in_scale2 (ICons ?w_scale (INil)))))
                        (= ?in_scale ?in_scale2)
                        (= ?scaled (Op (Mul
                            ?s_shape ?s_a_strides
                            (ECons (MNum 0) (ECons (MNum 0) (ENil)))
                            ?s_out_strides)
                            (ICons ?cast (ICons ?scale_product (INil)))))
                        (= ?out_bf16 (Op (Cast ?o_size (Bf16)) (ICons ?scaled (INil))))
                    )
                    (
                        (let ?gemv (Op (KernelGemvF8 ?n ?k)
                            (ICons ?a (ICons ?b (ICons ?in_scale (ICons ?w_scale (INil)))))))
                        (union ?out_bf16 ?gemv)
                        (set (dtype ?gemv) (Bf16))
                    )
                    :ruleset matmul_backend
                    :name \"kernel gemv f8 tensor core m1 {variant} {a_variant}\"
                )"
                ))
            })
            .collect()
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
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                n: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                k: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

const GEMV_F8_NWARPS: usize = 8;

impl KernelOp for KernelGemvF8 {
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
        let vars = self
            .n
            .dyn_vars()
            .into_iter()
            .chain(self.k.dyn_vars())
            .collect::<FxHashSet<_>>();
        let includes = dtype_includes(&[DType::Bf16, DType::F8E4M3]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n = self.n.to_kernel();
        let k = self.k.to_kernel();

        // Aligned shapes (k % 128 == 0, n % 8 == 0 — every llama projection)
        // take the branch-free loop; otherwise loads are bounds-padded with
        // zero bytes (fp8 0x00 == 0.0 contributes nothing) and stores are
        // predicated.
        let aligned = self.k.to_usize().map(|k| k % 128 == 0).unwrap_or(false)
            && self.n.to_usize().map(|n| n % 8 == 0).unwrap_or(false);

        let load_w = if aligned {
            "uint4 wv = __ldcs((const uint4*)(wrow + k0 + 16 * v));"
        } else {
            "uint4 wv = make_uint4(0, 0, 0, 0);
                if (row_valid) {
                    long long rem = ({k}) - (k0 + 16 * v);
                    if (rem >= 16) {
                        wv = __ldcs((const uint4*)(wrow + k0 + 16 * v));
                    } else if (rem > 0) {
                        unsigned char* wb = (unsigned char*)&wv;
                        for (int t = 0; t < rem; t++) wb[t] = ((const unsigned char*)wrow)[k0 + 16 * v + t];
                    }
                }"
        };
        let load_x = if aligned {
            "uint4 xv = __ldg((const uint4*)(x + k0 + 16 * v));"
        } else {
            "uint4 xv = make_uint4(0, 0, 0, 0);
                {
                    long long rem = ({k}) - (k0 + 16 * v);
                    if (rem >= 16) {
                        xv = __ldg((const uint4*)(x + k0 + 16 * v));
                    } else if (rem > 0) {
                        unsigned char* xb = (unsigned char*)&xv;
                        for (int t = 0; t < rem; t++) xb[t] = ((const unsigned char*)x)[k0 + 16 * v + t];
                    }
                }"
        };
        let store_guard = if aligned {
            ""
        } else {
            "if (row0 + lane >= ({n})) return;"
        };

        let load_w = if aligned {
            load_w.to_string()
        } else {
            load_w.replace("{k}", &k)
        };
        let load_x = if aligned {
            load_x.to_string()
        } else {
            load_x.replace("{k}", &k)
        };
        let store_guard = store_guard.replace("{n}", &n);

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gemv_f8_tc_k(__nv_bfloat16 *out, const __nv_fp8_e4m3 *x, const __nv_fp8_e4m3 *w, const float *in_scale, const float *w_scale{dyn_dims_param}) {{
        __shared__ float partial[{GEMV_F8_NWARPS}][8];
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int lg = lane & 3;
        const int n_in_tile = lane >> 2;
        const long long row0 = (long long)blockIdx.x * 8;
        const long long row = row0 + n_in_tile;
        const bool row_valid = row < ({n});
        const __nv_fp8_e4m3* wrow = w + (row_valid ? row : 0) * (long long)({k});

        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        const long long kstep = 128LL * {GEMV_F8_NWARPS};
        for (long long kb = warp * 128LL; kb < ({k}); kb += kstep) {{
            const long long k0 = kb + 32 * lg;
            #pragma unroll
            for (int v = 0; v < 2; v++) {{
                {load_w}
                {load_x}
                asm volatile(
                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \"
                    \"{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\\n\"
                    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)
                    : \"r\"(xv.x), \"r\"(xv.x), \"r\"(xv.y), \"r\"(xv.y), \"r\"(wv.x), \"r\"(wv.y));
                asm volatile(
                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \"
                    \"{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\\n\"
                    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)
                    : \"r\"(xv.z), \"r\"(xv.z), \"r\"(xv.w), \"r\"(xv.w), \"r\"(wv.z), \"r\"(wv.w));
            }}
        }}
        // D is row-replicated; lanes 0-3 hold result row cols {{2lg, 2lg+1}}.
        if (lane < 4) {{
            partial[warp][2 * lg] = d0;
            partial[warp][2 * lg + 1] = d1;
        }}
        __syncthreads();
        if (warp == 0 && lane < 8) {{
            {store_guard}
            float acc = 0.f;
            #pragma unroll
            for (int wp = 0; wp < {GEMV_F8_NWARPS}; wp++) acc += partial[wp][lane];
            out[row0 + lane] = (__nv_bfloat16)(acc * in_scale[0] * w_scale[0]);
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gemv_f8_tc_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (self.n.ceil_div(8), 1.into(), 1.into()),
            ((GEMV_F8_NWARPS * 32).into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.n
    }

    fn output_bytes(&self) -> Expression {
        self.n * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn bytes_loaded(&self) -> Expression {
        self.n * self.k + self.k + 8
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.n * self.k * 2
    }

    fn kernel_name(&self) -> &'static str {
        "GemvF8"
    }
}
