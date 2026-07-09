//! Fused RMSNorm: `x * rsqrt(mean(x²) + eps) * w` in one kernel.
//!
//! Replaces the decomposed norm sandwich (cast → square-mul → mean-reduce →
//! +eps → sqrt → recip → mul → weight-mul → cast: ~6-8 graph nodes) that the
//! 16-bit pipeline spells as `norm_in_f32`. Per the dtype contract the norm
//! computes in F32: the kernel loads `dtype` rows, accumulates the mean of
//! squares in F32, and rounds once at the store — the same semantics the
//! explicit-cast spelling expresses, minus the per-op intermediate roundings
//! (the decomposed path computes entirely in F32 between the casts too).
//!
//! Layout: x `(rows, cols)` contiguous in `dtype` with dynamic `rows`;
//! w `(cols,)` F32. One block per row; F32 warp + block reduction.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

const TPB: usize = 1024;

#[derive(Debug, Clone)]
pub struct RMSNormKernel {
    pub rows: Expression,
    pub cols: usize,
    pub eps: f32,
    pub dtype: DType,
}

impl KernelOp for RMSNormKernel {
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
        let cols = self.cols;
        let eps = self.eps;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        let kernel = format!(
            r#"{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
extern "C" __global__ void rms_norm_k(
    {ty}* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ w
) {{
    const int COLS = {cols};
    __shared__ float warp_sums[{TPB} / WARP_SIZE];
    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    const {ty}* xr = x + row * COLS;
    {ty}* yr = out + row * COLS;

    float partial = 0.0f;
#if {cols} % 8 == 0
    {{
        const uint4* xv = (const uint4*)xr;
        for (int c = tid; c < COLS / 8; c += {TPB}) {{
            uint4 chunk = xv[c];
            const {ty}* xe = (const {ty}*)&chunk;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                float v = (float)xe[e];
                partial += v * v;
            }}
        }}
    }}
#else
    for (int i = tid; i < COLS; i += {TPB}) {{
        float v = (float)xr[i];
        partial += v * v;
    }}
#endif

    #pragma unroll
    for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
        partial += __shfl_down_sync(FULL_MASK, partial, s);
    }}
    if (lane_id == 0) {{
        warp_sums[warp_id] = partial;
    }}
    __syncthreads();

    if (warp_id == 0) {{
        int cnt = {TPB} / WARP_SIZE;
        float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int s = cnt / 2; s > 0; s /= 2) {{
            block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
        }}
        if (tid == 0) {{
            warp_sums[0] = rsqrtf(block_sum / (float)COLS + {eps:.10}f);
        }}
    }}
    __syncthreads();
    float rinv = warp_sums[0];

    for (int i = tid; i < COLS; i += {TPB}) {{
        yr[i] = ({ty})((float)xr[i] * rinv * w[i]);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rms_norm_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rms_norm_k".to_string(),
            (
                self.rows,
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
        self.rows * self.cols
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        // Two passes over x plus the weight row.
        (self.rows * self.cols * self.dtype.bits() * 2).ceil_div(8) + self.cols * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.cols * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RMSNorm"
    }
}

#[derive(Debug, Clone)]
pub struct RMSNormCustom(pub RMSNormKernel);

impl CustomOp for RMSNormCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `x * rsqrt(mean(x², last axis) + eps) * w`.
///
/// `x` is `(rows, cols)` in any float dtype (F32 accumulation inside),
/// `w` is `(cols,)` F32. Returns `(rows, cols)` in `x`'s dtype.
pub fn fused_rms_norm(x: GraphTensor, w: GraphTensor, eps: f32) -> GraphTensor {
    assert_eq!(w.dtype, DType::F32, "RMSNorm weight must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");
    assert_eq!(
        w.dims()[0].to_usize().expect("RMSNorm weight dim"),
        cols,
        "RMSNorm weight length mismatch"
    );

    let kern = RMSNormKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RMSNormCustom(kern), vec![x, w], (rows, cols), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Fused RMSNorm + static-scale FP8 quantization
//
// `q = (f8)(x · rsqrt(mean(x²) + eps) · w / in_scale)` in one kernel — the
// norm output's only consumer in the fp8 pipeline is the per-linear
// quantization, so the rounding to bf16 between them is pure overhead.
//
// CONVENTION (load-bearing for the e-graph): F8E4M3-producing custom ops put
// the quantization input scale as their LAST input. The scaled-fp8 GEMM
// rewrites (cuBLASLt + tensor-core GEMV) match `(CustomOpKind ?id (F8E4M3))`
// by arity and read the scale from that slot.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RMSNormQuantKernel {
    pub rows: Expression,
    pub cols: usize,
    pub eps: f32,
    /// Input dtype (16-bit); output is always F8E4M3.
    pub dtype: DType,
}

impl KernelOp for RMSNormQuantKernel {
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
        let cols = self.cols;
        let eps = self.eps;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype, DType::F8E4M3]);
        let kernel = format!(
            r#"{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
extern "C" __global__ void rms_norm_quant_k(
    __nv_fp8_e4m3* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ in_scale
) {{
    const int COLS = {cols};
    __shared__ float warp_sums[{TPB} / WARP_SIZE];
    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    const {ty}* xr = x + row * COLS;
    __nv_fp8_e4m3* yr = out + row * COLS;

    float partial = 0.0f;
#if {cols} % 8 == 0
    {{
        const uint4* xv = (const uint4*)xr;
        for (int c = tid; c < COLS / 8; c += {TPB}) {{
            uint4 chunk = xv[c];
            const {ty}* xe = (const {ty}*)&chunk;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                float v = (float)xe[e];
                partial += v * v;
            }}
        }}
    }}
#else
    for (int i = tid; i < COLS; i += {TPB}) {{
        float v = (float)xr[i];
        partial += v * v;
    }}
#endif

    #pragma unroll
    for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
        partial += __shfl_down_sync(FULL_MASK, partial, s);
    }}
    if (lane_id == 0) {{
        warp_sums[warp_id] = partial;
    }}
    __syncthreads();

    if (warp_id == 0) {{
        int cnt = {TPB} / WARP_SIZE;
        float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int s = cnt / 2; s > 0; s /= 2) {{
            block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
        }}
        if (tid == 0) {{
            warp_sums[0] = rsqrtf(block_sum / (float)COLS + {eps:.10}f);
        }}
    }}
    __syncthreads();
    float rinv = warp_sums[0] * (1.0f / in_scale[0]);

    for (int i = tid; i < COLS; i += {TPB}) {{
        yr[i] = (__nv_fp8_e4m3)((float)xr[i] * rinv * w[i]);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rms_norm_quant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rms_norm_quant_k".to_string(),
            (
                self.rows,
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
        self.rows * self.cols
    }

    fn output_bytes(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        DType::F8E4M3
    }

    fn bytes_loaded(&self) -> Expression {
        (self.rows * self.cols * self.dtype.bits() * 2).ceil_div(8) + self.cols * 4 + 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.cols * 5
    }

    fn kernel_name(&self) -> &'static str {
        "RMSNormQuant"
    }
}

#[derive(Debug, Clone)]
pub struct RMSNormQuantCustom(pub RMSNormQuantKernel);

impl CustomOp for RMSNormQuantCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `(f8)(rms_norm(x, w, eps) / in_scale)`.
///
/// Inputs `(x, w, in_scale)` — the scale is LAST per the f8 custom-op
/// convention the scaled-GEMM rewrites rely on. `x` is `(rows, cols)`
/// 16-bit, `w` `(cols,)` F32, `in_scale` a scalar F32 tensor. Returns
/// `(rows, cols)` F8E4M3.
pub fn fused_rms_norm_quant(
    x: GraphTensor,
    w: GraphTensor,
    eps: f32,
    in_scale: GraphTensor,
) -> GraphTensor {
    assert_eq!(w.dtype, DType::F32, "RMSNorm weight must be F32");
    assert_eq!(in_scale.dtype, DType::F32, "quant scale must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");

    let kern = RMSNormQuantKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(
        RMSNormQuantCustom(kern),
        vec![x, w, in_scale],
        (rows, cols),
        DType::F8E4M3,
    )
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched RMSNorm: unions the fused kernel into the decomposed HLIR
// chain the models spell (per the pure-HLIR rule):
//
//   Cast(F32)(x_bf16) → Mul(x,x) → Sum(last) → ×Recip(Iota(cols)) → +eps
//   → Sqrt → Recip → ×x → ×w → Cast(Bf16)
//
// Two variants: 2-D (s, cols) plain norms and 3-D (s, h, d) per-head QK
// norms. The 3-D input is a split view of a contiguous (s, h·d) buffer, so
// the same row-per-block kernel serves both with rows = product of the
// non-reduced dims.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRMSNorm {
    out_shape: Vec<Expression>,
    eps: f64,
}

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, F64, OP_KIND},
        extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

impl EgglogOp for KernelRMSNorm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRMSNorm",
            &[("out_shape", ELIST), ("eps", F64)],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Two relation-staged parts (pre → late): the rinv core (anchored by
        // the rare Sqrt→Recip pair and the eps Constant) emits a fact; the
        // weight-mul tail joins with ?rin/?xf bound, so each variant's pins
        // are cheap. A monolithic join explodes on rolled bodies with
        // several distinct layer instances.
        let core = "(relation rms_rinv (IR IR IR f64 Expression))
            (rule
                (
                    ; bf16 → f32 sandwich entry
                    (= ?xf (Op (Cast ?xf_size (F32)) (ICons ?xb (INil))))
                    (= (Bf16) (dtype ?xb))

                    ; sum of squares over the last axis
                    (= ?sq (Op (Mul ?sq_shape ?sq_a ?sq_b ?sq_o)
                        (ICons ?xf (ICons ?xf2 (INil)))))
                    (= ?xf ?xf2)
                    (= ?sum (Op (Sum ?sum_shape ?cols ?sum_in (MIter) ?sum_out)
                        (ICons ?sq (INil))))

                    ; mean: × recip(cols) — the divisor iota must carry the
                    ; reduce dim itself
                    (= ?mean (Op (Mul ?mn_shape ?mn_a ?mn_b ?mn_o)
                        (ICons ?sum (ICons ?rcpn (INil)))))
                    (= ?rcpn (Op (Recip ?rn_shape ?rn_in ?rn_out) (ICons ?ncast (INil))))
                    (= ?ncast (Op (Cast ?nc_size (F32)) (ICons ?ncst (INil))))
                    (= ?ncst (Op (Iota ?cols3 ?nc_range) (INil)))
                    (= ?cols ?cols3)

                    ; + eps → sqrt → recip
                    (= ?pe (Op (Add ?pe_shape ?pe_a ?pe_b ?pe_o)
                        (ICons ?mean (ICons ?epsc (INil)))))
                    (= ?epsc (Op (Constant ?eps) (INil)))
                    (= ?sqr (Op (Sqrt ?sq2_shape ?sq2_in ?sq2_out) (ICons ?pe (INil))))
                    (= ?rin (Op (Recip ?ri_shape ?ri_in ?ri_out) (ICons ?sqr (INil))))
                )
                (
                    (rms_rinv ?rin ?xf ?xb ?eps ?cols)
                )
                :ruleset kernel_fuse_late_pre
                :name \"rms rinv core\"
            )"
        .to_string();

        // (variant name, out-shape destructure + w stride pins)
        let variants: [(&str, &str); 2] = [
            (
                "2d",
                "(= ?wg_shape (ECons ?rows (ECons ?cols2 (ENil))))
                        (= ?cols ?cols2)
                        ; w broadcast over rows, contiguous within
                        (= ?wg_b (ECons (MNum 0) (ECons (MIter) (ENil))))",
            ),
            (
                "3d",
                "(= ?wg_shape (ECons ?d0 (ECons ?d1 (ECons ?cols2 (ENil)))))
                        (= ?cols ?cols2)
                        (= ?wg_b (ECons (MNum 0) (ECons (MNum 0) (ECons (MIter) (ENil)))))",
            ),
        ];
        let tails = variants
            .into_iter()
            .map(|(variant, shape_pins)| {
                format!(
                    "(rule
                    (
                        (rms_rinv ?rin ?xf ?xb ?eps ?cols)

                        ; × x → × w → bf16
                        (= ?nrm (Op (Mul ?nr_shape ?nr_a ?nr_b ?nr_o)
                            (ICons ?rin (ICons ?xf3 (INil)))))
                        (= ?xf ?xf3)
                        (= ?wgt (Op (Mul ?wg_shape ?wg_a ?wg_b ?wg_o)
                            (ICons ?nrm (ICons ?w (INil)))))
                        (= (F32) (dtype ?w))
                        (= ?out (Op (Cast ?o_size (Bf16)) (ICons ?wgt (INil))))

                        {shape_pins}
                    )
                    (
                        (let ?krms (Op (KernelRMSNorm ?wg_shape ?eps)
                            (ICons ?xb (ICons ?w (INil)))))
                        (union ?out ?krms)
                        (set (dtype ?krms) (Bf16))
                    )
                    :ruleset kernel_fuse_late
                    :name \"kernel rms norm bf16 {variant}\"
                )"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        vec![Rule::raw(format!("{core}\n{tails}"))]
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
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let eps: f64 = egraph.enodes[kind_children[1]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let cols = out_shape
            .last()
            .and_then(|c| c.to_usize())
            .expect("RMSNorm cols must be static");
        let rows = out_shape[..out_shape.len() - 1]
            .iter()
            .copied()
            .product::<Expression>();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(RMSNormKernel {
                rows,
                cols,
                eps: eps as f32,
                dtype: DType::Bf16,
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched fused RMSNorm + fp8 quant: the rms_rinv staged core plus
// the weight-mul tail and the per-linear quant spelling
// `cast_f8(norm_bf16_f32 · recip(scale))`, re-fused into the existing
// RMSNormQuantKernel (norm + quantize in one pass).
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRMSNormQuant {
    out_shape: Vec<Expression>,
    eps: f64,
}

impl EgglogOp for KernelRMSNormQuant {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRMSNormQuant",
            &[("out_shape", ELIST), ("eps", F64)],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    (rms_rinv ?rin ?xf ?xb ?eps ?cols)

                    ; × x → × w → bf16 (the 2D norm tail)
                    (= ?nrm (Op (Mul ?nr_shape ?nr_a ?nr_b ?nr_o)
                        (ICons ?rin (ICons ?xf3 (INil)))))
                    (= ?xf ?xf3)
                    (= ?wgt (Op (Mul ?wg_shape ?wg_a ?wg_b ?wg_o)
                        (ICons ?nrm (ICons ?w (INil)))))
                    (= (F32) (dtype ?w))
                    (= ?out (Op (Cast ?o_size (Bf16)) (ICons ?wgt (INil))))
                    (= ?wg_shape (ECons ?rows (ECons ?cols2 (ENil))))
                    (= ?cols ?cols2)
                    (= ?wg_b (ECons (MNum 0) (ECons (MIter) (ENil))))

                    ; quant tail: cast_f8(out_f32 · recip(scale))
                    (= ?qf (Op (Cast ?qf_size (F32)) (ICons ?out (INil))))
                    (= ?qrecip (Op (Recip ?qr_sh ?qr_in ?qr_out)
                        (ICons ?scale (INil))))
                    (= ?qmul (Op (Mul ?qm_sh ?qm_a ?qm_b ?qm_o)
                        (ICons ?qf (ICons ?qrecip (INil)))))
                    (= ?q (Op (Cast ?q_size (F8E4M3)) (ICons ?qmul (INil))))
                )
                (
                    (let ?krq (Op (KernelRMSNormQuant ?wg_shape ?eps)
                        (ICons ?xb (ICons ?w (ICons ?scale (INil))))))
                    (union ?q ?krq)
                    (set (dtype ?krq) (F8E4M3))
                )
                :ruleset kernel_fuse_late
                :name \"kernel rms norm quant f8\"
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
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let eps: f64 = egraph.enodes[kind_children[1]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let kern = RMSNormQuantKernel {
            rows: out_shape[0],
            cols: out_shape[1].to_usize().expect("rms cols must be static"),
            eps: eps as f32,
            dtype: DType::Bf16,
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(kern) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}
