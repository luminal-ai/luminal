//! Fused SwiGLU reading both halves of a fused gate_up GEMM row.
//!
//! `out[r, c] = silu(x[r, c]) * x[r, I + c]` for `x (rows, 2I)`. Avoids the
//! offset-slice problem (tensors don't have offsets, so `x[:, I:]` lowers to
//! a Gather materialization) by indexing the second half inside the kernel,
//! and replaces gather + swish/mul region with one launch. F32 math, storage
//! dtype in/out — same compute precision as the widened decomposed chain,
//! rounded once at the store.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

const TPB: usize = 256;

#[derive(Debug, Clone)]
pub struct SwigluKernel {
    pub rows: Expression,
    pub intermediate: usize,
    pub dtype: DType,
}

impl KernelOp for SwigluKernel {
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
        let i = self.intermediate;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        // 2D grid: x = rows (dynamic, carried by the grid expression only),
        // y = static column tiles — one block per (row, tile) so a single
        // decode row still spreads across the GPU instead of one SM.
        let col_tiles = i.div_ceil(TPB);
        let kernel = format!(
            r#"{includes}
extern "C" __global__ void swiglu_k(
    {ty}* __restrict__ out,
    const {ty}* __restrict__ x
) {{
    const long long I = {i};
    long long row = blockIdx.x;
    long long col = (long long)blockIdx.y * {TPB} + threadIdx.x;
    if (col >= I) return;
    const {ty}* xr = x + row * (2 * I);
    float g = (float)xr[col];
    float u = (float)xr[I + col];
    float silu = g / (1.0f + expf(-g));
    out[row * I + col] = ({ty})(silu * u);
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("swiglu_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "swiglu_k".to_string(),
            (
                self.rows,
                Expression::from(col_tiles),
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
        self.rows * self.intermediate
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        (self.rows * self.intermediate * 2 * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.intermediate * 6
    }

    fn kernel_name(&self) -> &'static str {
        "Swiglu"
    }
}

#[derive(Debug, Clone)]
pub struct SwigluCustom(pub SwigluKernel);

impl CustomOp for SwigluCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// `silu(x[:, :I]) * x[:, I:]` for a fused `(rows, 2I)` gate_up projection.
pub fn fused_swiglu(x: GraphTensor, intermediate: usize) -> GraphTensor {
    let x_dims = x.dims();
    assert_eq!(
        x_dims.len(),
        2,
        "swiglu x must be 2-D (rows, 2*intermediate)"
    );
    let rows = x_dims[0];
    assert_eq!(
        x_dims[1].to_usize().expect("swiglu cols must be static"),
        2 * intermediate,
        "swiglu expects [gate | up] halves"
    );
    let kern = SwigluKernel {
        rows,
        intermediate,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(SwigluCustom(kern), vec![x], (rows, intermediate), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Fused SwiGLU + static-scale FP8 quantization
//
// `q[r, c] = (f8)(silu(x[r, c]) · x[r, I + c] / in_scale)` — the SwiGLU
// output's only consumer in the fp8 pipeline is the down-projection's
// quantization, so the bf16 round-trip between them is pure overhead.
//
// CONVENTION (load-bearing for the e-graph): F8E4M3-producing custom ops put
// the quantization input scale as their LAST input — the scaled-fp8 GEMM
// rewrites match `(CustomOpKind ?id (F8E4M3))` by arity and read the scale
// from that slot.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct SwigluQuantKernel {
    pub rows: Expression,
    pub intermediate: usize,
    /// Input dtype (16-bit); output is always F8E4M3.
    pub dtype: DType,
}

impl KernelOp for SwigluQuantKernel {
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
        let i = self.intermediate;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype, DType::F8E4M3]);
        let col_tiles = i.div_ceil(TPB);
        let kernel = format!(
            r#"{includes}
extern "C" __global__ void swiglu_quant_k(
    __nv_fp8_e4m3* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ in_scale
) {{
    const long long I = {i};
    long long row = blockIdx.x;
    long long col = (long long)blockIdx.y * {TPB} + threadIdx.x;
    if (col >= I) return;
    const {ty}* xr = x + row * (2 * I);
    float g = (float)xr[col];
    float u = (float)xr[I + col];
    float silu = g / (1.0f + expf(-g));
    out[row * I + col] = (__nv_fp8_e4m3)(silu * u * (1.0f / in_scale[0]));
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("swiglu_quant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "swiglu_quant_k".to_string(),
            (
                self.rows,
                Expression::from(col_tiles),
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
        self.rows * self.intermediate
    }

    fn output_bytes(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        DType::F8E4M3
    }

    fn bytes_loaded(&self) -> Expression {
        (self.rows * self.intermediate * 2 * self.dtype.bits()).ceil_div(8) + 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.intermediate * 7
    }

    fn kernel_name(&self) -> &'static str {
        "SwigluQuant"
    }
}

#[derive(Debug, Clone)]
pub struct SwigluQuantCustom(pub SwigluQuantKernel);

impl CustomOp for SwigluQuantCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `(f8)(silu(x[:, :I]) * x[:, I:] / in_scale)`.
///
/// Inputs `(x, in_scale)` — scale LAST per the f8 custom-op convention.
pub fn fused_swiglu_quant(
    x: GraphTensor,
    intermediate: usize,
    in_scale: GraphTensor,
) -> GraphTensor {
    assert_eq!(in_scale.dtype, DType::F32, "quant scale must be F32");
    let x_dims = x.dims();
    assert_eq!(
        x_dims.len(),
        2,
        "swiglu x must be 2-D (rows, 2*intermediate)"
    );
    let rows = x_dims[0];
    assert_eq!(
        x_dims[1].to_usize().expect("swiglu cols must be static"),
        2 * intermediate,
        "swiglu expects [gate | up] halves"
    );
    let kern = SwigluQuantKernel {
        rows,
        intermediate,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(
        SwigluQuantCustom(kern),
        vec![x, in_scale],
        (rows, intermediate),
        DType::F8E4M3,
    )
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched fused SwiGLU (+quant): the pure-HLIR spelling
//   gate = x[:, :I] (view), up = x[:, I:] (offset slice → Gather)
//   silu = gate · recip(1 + exp2(gate · (−1) · log2e))
//   out  = silu · up                       [bf16]
//   q    = cast_f8(out_f32 · recip(scale)) [quant variant]
// re-fused into the existing one-kernel implementations. Constants are
// matched as raw Cast(Bf16)(Constant v) chains — the `const_like` relation
// lives in the FlashInfer host-op egg, which loads after kernel ops.
// ═══════════════════════════════════════════════════════════

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

fn swiglu_chain_atoms() -> &'static str {
    "
                    ; silu(gate): gate is the (rows, I) start-0 view of x
                    (= ?ng (Op (Mul ?ng_sh
                        (ECons (MMul (MIter) ?e_2i) (ECons (MIter) (ENil)))
                        ?ng_b ?ng_o)
                        (ICons ?x (ICons ?negb (INil)))))
                    (= ?negb (Op (Cast ?nb_size (Bf16)) (ICons ?negc (INil))))
                    (= ?negc (Op (Constant -1.000000) (INil)))
                    (= ?sc (Op (Mul ?sc_sh ?sc_a ?sc_b ?sc_o)
                        (ICons ?ng (ICons ?l2eb (INil)))))
                    (= ?l2eb (Op (Cast ?l2eb_size (Bf16)) (ICons ?l2ec (INil))))
                    (= ?l2ec (Op (Constant 1.442695) (INil)))
                    (= ?ex (Op (Exp2 ?ex_sh ?ex_in ?ex_out) (ICons ?sc (INil))))
                    (= ?pl1 (Op (Add ?p1_sh ?p1_a ?p1_b ?p1_o)
                        (ICons ?ex (ICons ?oneb (INil)))))
                    (= ?oneb (Op (Cast ?ob_size (Bf16)) (ICons ?onec (INil))))
                    (= ?onec (Op (Constant 1.000000) (INil)))
                    (= ?sig (Op (Recip ?sg_sh ?sg_in ?sg_out) (ICons ?pl1 (INil))))
                    (= ?silu (Op (Mul ?si_sh
                        (ECons (MMul (MIter) ?e_2i2) (ECons (MIter) (ENil)))
                        ?si_b ?si_o)
                        (ICons ?x2 (ICons ?sig (INil)))))
                    (= ?x ?x2)

                    ; · up, where up is the offset-slice gather of x
                    (= ?out (Op (Mul ?o_sh ?o_a ?o_b ?o_o)
                        (ICons ?silu (ICons ?up (INil)))))
                    (= ?o_sh (ECons ?rows (ECons ?i (ENil))))
                    (= ?up (Op (Gather ?up_osh ?up_ostr ?up_dsh
                        (ECons (MMul (MIter) ?e_2i3) (ECons (MIter) (ENil))))
                        (ICons ?upidx (ICons ?x3 (INil)))))
                    (= ?x ?x3)
                    (= ?upidx (Op (Iota
                        (MAdd (MAdd (MMod (MIter) ?e_i) ?e_i)
                              (MMul (MDiv (MIter) ?e_i) ?e_2i4))
                        ?up_range) (INil)))
                    (= ?i ?e_i)

                    (= (Bf16) (dtype ?x))"
}

#[derive(Default, Debug, Clone)]
pub struct KernelSwiglu {
    /// Output shape `(rows, intermediate)`; rows may be dynamic.
    out_shape: Vec<Expression>,
}

impl EgglogOp for KernelSwiglu {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelSwiglu", &[("out_shape", ELIST)])
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(format!(
            "(rule
                ({}
                )
                (
                    (let ?ks (Op (KernelSwiglu ?o_sh) (ICons ?x (INil))))
                    (union ?out ?ks)
                    (set (dtype ?ks) (Bf16))
                )
                :ruleset kernel_fuse_late
                :name \"kernel swiglu bf16\"
            )",
            swiglu_chain_atoms()
        ))]
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
        let kern = SwigluKernel {
            rows: out_shape[0],
            intermediate: out_shape[1].to_usize().expect("swiglu I must be static"),
            dtype: DType::Bf16,
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(kern) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSwigluQuant {
    out_shape: Vec<Expression>,
}

impl EgglogOp for KernelSwigluQuant {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelSwigluQuant", &[("out_shape", ELIST)])
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(format!(
            "(rule
                ({}

                    ; quant tail: cast_f8(out_f32 · recip(scale))
                    (= ?qf (Op (Cast ?qf_size (F32)) (ICons ?out (INil))))
                    (= ?qrecip (Op (Recip ?qr_sh ?qr_in ?qr_out)
                        (ICons ?scale (INil))))
                    (= ?qmul (Op (Mul ?qm_sh ?qm_a ?qm_b ?qm_o)
                        (ICons ?qf (ICons ?qrecip (INil)))))
                    (= ?q (Op (Cast ?q_size (F8E4M3)) (ICons ?qmul (INil))))
                )
                (
                    (let ?ksq (Op (KernelSwigluQuant ?o_sh) (ICons ?x (ICons ?scale (INil)))))
                    (union ?q ?ksq)
                    (set (dtype ?ksq) (F8E4M3))
                )
                :ruleset kernel_fuse_late
                :name \"kernel swiglu quant f8\"
            )",
            swiglu_chain_atoms()
        ))]
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
        let kern = SwigluQuantKernel {
            rows: out_shape[0],
            intermediate: out_shape[1].to_usize().expect("swiglu I must be static"),
            dtype: DType::Bf16,
        };
        (
            LLIROp::new::<dyn KernelOp>(Box::new(kern) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}
