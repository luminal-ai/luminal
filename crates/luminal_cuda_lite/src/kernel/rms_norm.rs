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

use std::{fmt::Display, sync::Arc};

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType,
    op::{CustomOp, HLIROp, LLIROp},
    prelude::{FxHashMap, GraphTensor, NodeIndex, ShapeTracker, Symbol},
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
    /// Storage dtype; conversion to `dtype` happens before either reduction or scaling.
    pub input_dtype: DType,
    pub threads: usize,
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let cols = self.cols;
        let eps = self.eps;
        let tpb = self.threads;
        assert!(matches!(self.dtype, DType::F32 | DType::F16 | DType::Bf16));
        assert!(matches!(tpb, 128 | 256 | 512 | 1024));
        assert!(
            self.input_dtype == self.dtype
                || (self.input_dtype == DType::F32
                    && matches!(self.dtype, DType::F16 | DType::Bf16))
        );
        let input_ty = crate::cuda_dtype(self.input_dtype);
        // Keep the logical reduction groups unchanged when absorbing a cast.
        // F32 storage needs two 16-byte loads for eight rounded 16-bit values.
        let elements = 128 / self.dtype.bits();
        let vectors = self.input_dtype.bits() / self.dtype.bits();
        let loads = (0..vectors)
            .map(|i| format!("xv[c * {vectors} + {i}]"))
            .collect::<Vec<_>>()
            .join(", ");
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype, self.input_dtype]);
        let kernel = format!(
            r#"{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
extern "C" __global__ void rms_norm_k(
    {ty}* __restrict__ out,
    const {input_ty}* __restrict__ x,
    const float* __restrict__ w
) {{
    const int COLS = {cols};
    __shared__ float warp_sums[{tpb} / WARP_SIZE];
    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    const {input_ty}* xr = x + row * COLS;
    {ty}* yr = out + row * COLS;

    float partial = 0.0f;
#if {cols} % {elements} == 0
    {{
        const uint4* xv = (const uint4*)xr;
        for (int c = tid; c < COLS / {elements}; c += {tpb}) {{
            const uint4 chunks[{vectors}] = {{{loads}}};
            const {input_ty}* xe = (const {input_ty}*)chunks;
            #pragma unroll
            for (int e = 0; e < {elements}; e++) {{
                float v = (float)({ty})xe[e];
                partial += v * v;
            }}
        }}
    }}
#else
    for (int i = tid; i < COLS; i += {tpb}) {{
        float v = (float)({ty})xr[i];
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
        int cnt = {tpb} / WARP_SIZE;
        float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int s = cnt / 2; s > 0; s /= 2) {{
            block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
        }}
        if (tid == 0) {{
            warp_sums[0] = rsqrtf(block_sum / (float)COLS + {eps:e}f);
        }}
    }}
    __syncthreads();
    float rinv = warp_sums[0];

    for (int i = tid; i < COLS; i += {tpb}) {{
        yr[i] = ({ty})((float)({ty})xr[i] * rinv * w[i]);
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
            kernel,
            (
                self.rows,
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(tpb),
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
        (self.rows * self.cols * self.input_dtype.bits() * 2).ceil_div(8) + self.cols * 4
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
    assert_eq!(w.dims().len(), 1, "RMSNorm weight must be a vector");
    assert!(
        matches!(x.dtype, DType::F32 | DType::F16 | DType::Bf16),
        "RMSNorm requires F32, F16 or BF16 input"
    );
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");
    assert!(cols > 0, "RMSNorm reduction must be nonempty");
    assert_eq!(
        w.dims()[0].to_usize().expect("RMSNorm weight dim"),
        cols,
        "RMSNorm weight length mismatch"
    );

    let contiguous = |t: GraphTensor| {
        if t.shape.is_contiguous() {
            return t;
        }
        let indices = t.graph().arange(t.shape.n_elements());
        let mut gathered = t.gather(indices);
        gathered.shape = ShapeTracker::new_with_element_bits(t.dims(), t.dtype.bits());
        gathered
    };
    let x = contiguous(x);
    let w = contiguous(w);
    let kern = RMSNormKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
        input_dtype: x.dtype,
        threads: TPB,
    };
    let cx = unsafe { &mut *x.graph_ref };
    let id = cx.add_op(kern, &[x.id, w.id]);
    GraphTensor::from_id(
        id,
        ShapeTracker::new_with_element_bits((rows, cols), x.dtype.bits()),
        cx,
        x.dtype,
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
                :ruleset kernel_fuse_late_pre_rms
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
                        ; Once the fused kernel is legal, the decomposed norm
                        ; only adds large temporaries and launch latency.
                        (delete (Op (Cast ?o_size (Bf16)) (ICons ?wgt (INil))))
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
                input_dtype: DType::Bf16,
                threads: TPB,
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

// The explicit primitive is a normal searchable IR node. It keeps rsqrt and
// conversion boundaries independent of the decomposed sqrt/reciprocal pattern.
impl Default for RMSNormKernel {
    fn default() -> Self {
        Self {
            rows: 1.into(),
            cols: 1,
            eps: 1e-5,
            dtype: DType::F32,
            input_dtype: DType::F32,
            threads: TPB,
        }
    }
}

impl Display for RMSNormKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FusedRMSNorm")
    }
}

impl HLIROp for RMSNormKernel {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        use luminal::egglog_utils::list_to_egglog;
        assert_eq!(inputs.len(), 2);
        format!(
            "(Op (FusedRMSNorm {} {} {} {:?} ({:?}) ({:?}) {}) {})",
            self.rows.to_egglog(),
            Expression::from(self.cols).to_egglog(),
            ShapeTracker::new((self.rows, self.cols))
                .n_elements()
                .to_egglog(),
            self.eps as f64,
            self.dtype,
            self.input_dtype,
            Expression::from(self.threads).to_egglog(),
            list_to_egglog(&[&inputs[0].1, &inputs[1].1], "ICons", "INil")
        )
    }

    fn output_dtype(&self, _: &[DType]) -> DType {
        self.dtype
    }
}

impl EgglogOp for RMSNormKernel {
    fn sort(&self) -> SortDef {
        use luminal::egglog_utils::base::{DTYPE, EXPRESSION};
        sort(
            OP_KIND,
            "FusedRMSNorm",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("size", EXPRESSION),
                ("eps", F64),
                ("dtype", DTYPE),
                ("input_dtype", DTYPE),
                ("threads", EXPRESSION),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }
    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = vec![Rule::raw(
            r#"
            (rule ((= ?out (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt ?input_dt ?threads) ?inputs)))
                ((set (dtype ?out) ?dt)) :ruleset dtype_prop :name "explicit rmsnorm dtype")
            (rule (
                (= ?out (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt ?dt ?threads)
                    (ICons ?cast (ICons ?w (INil)))))
                (= ?cast (Op (Cast ?size ?dt) (ICons ?x (INil))))
                (= (dtype ?x) (F32))
                (!= ?dt (F32))
            ) (
                (let ?fused (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads)
                    (ICons ?x (ICons ?w (INil)))))
                (union ?out ?fused)
                (set (dtype ?fused) ?dt)
            ) :ruleset kernel_fuse_late :name "explicit rmsnorm absorb rounded input cast")
        "#
            .to_string(),
        )];
        rules.extend([128, 256, 512].map(|threads| Rule::raw(format!(r#"
            (rule ((= ?out (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt ?input_dt (MNum 1024)) ?inputs)))
                ((let ?tuned (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt ?input_dt (MNum {threads})) ?inputs))
                 (union ?out ?tuned) (set (dtype ?tuned) ?dt))
                :ruleset kernel_fuse_late :name "explicit rmsnorm {threads} threads")
        "#))));
        rules
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_dtype, extract_expr};
        let mut expr = |i| extract_expr(egraph, kind_children[i], expr_cache).unwrap();
        let kernel = Self {
            rows: expr(0),
            cols: expr(1).to_usize().unwrap(),
            eps: egraph.enodes[kind_children[3]]
                .0
                .replace('"', "")
                .parse::<f64>()
                .unwrap() as f32,
            dtype: extract_dtype(egraph, kind_children[4]),
            input_dtype: extract_dtype(egraph, kind_children[5]),
            threads: expr(6).to_usize().unwrap(),
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(kernel) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

#[cfg(test)]
#[path = "rms_norm_tests.rs"]
mod tests;
