//! Fused argmax over the last axis.
//!
//! Matches the frontend `argmax(axis)` decomposition (src/frontend/unary.rs):
//!
//! ```text
//! mx      = Max(x, axis)                       broadcast back over axis
//! ne      = Cast(F32)(LessThan(x, mx)) + Cast(F32)(LessThan(mx, x))
//! eq      = Cast(Bool)(ne * -1 + 1)
//! one_hot = Cast(Int)(eq)
//! out     = Max(one_hot * Iota(cols), axis)
//! ```
//!
//! and unions a single kernel into the final Max's eclass. Output dtype is
//! Int, exactly like the decomposed chain. Tie rule preserved: the highest
//! index among equal maxima wins (the decomposition takes a max over
//! `index * one_hot`). Used for GPU-side greedy sampling: the host reads one
//! i32 per row instead of a vocab-sized logit row.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{KernelOp, hlir::dtype_includes},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

// Wide blocks: one block scans a whole vocab row (≈128K elements), so
// maximize per-block parallelism.
const TPB: usize = 1024;

#[derive(Default, Debug, Clone)]
pub struct KernelArgmax {
    out_shape: Vec<Expression>,
    cols: Expression,
    dtype: DType,
}

impl EgglogOp for KernelArgmax {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelArgmax",
            &[("shape", ELIST), ("cols", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // The scalar constants (-1, +1) sit behind frontend-emitted identity
        // casts; the kernel_lower fold unions a KernelConstant into each
        // Cast(Constant) eclass (F32 included), which is matched directly
        // here (const_like lives in the flashinfer host-op file, which loads
        // after kernel ops).
        vec![Rule::raw(
            "(rule
                (
                    ; final Max over cols of (one_hot * iota)
                    (= ?out (Op (Max ?out_shape ?cols ?out_in_strides (MIter) ?out_out_strides)
                        (ICons ?prod (INil))))
                    (= ?prod (Op (Mul ?prod_shape ?onehot_strides ?iota_strides ?prod_out_strides)
                        (ICons ?one_hot (ICons ?iota (INil)))))
                    (= ?iota (Op (Iota (MIter) ?cols2) (INil)))

                    ; one_hot = Cast(Int)(Cast(Bool)(ne * -1 + 1))
                    (= ?one_hot (Op (Cast ?oh_size (Int)) (ICons ?eq (INil))))
                    (= ?eq (Op (Cast ?eq_size (Bool)) (ICons ?plus1 (INil))))
                    (= ?plus1 (Op (Add ?p1_shape ?neg_strides ?one_strides ?p1_out_strides)
                        (ICons ?neg (ICons ?one (INil)))))
                    (= ?one (Op (KernelConstant 1.000000 (F32)) (INil)))
                    (= ?neg (Op (Mul ?neg_shape ?ne_strides ?negone_strides ?neg_out_strides)
                        (ICons ?ne (ICons ?negone (INil)))))
                    (= ?negone (Op (KernelConstant -1.000000 (F32)) (INil)))

                    ; ne = Cast(F32)(x < mx) + Cast(F32)(mx < x)
                    (= ?ne (Op (Add ?ne_shape ?lt1_strides ?lt2_strides ?ne_out_strides)
                        (ICons ?lt1f (ICons ?lt2f (INil)))))
                    (= ?lt1f (Op (Cast ?lt1_size (F32)) (ICons ?lt1 (INil))))
                    (= ?lt2f (Op (Cast ?lt2_size (F32)) (ICons ?lt2 (INil))))
                    (= ?lt1 (Op (LessThan ?lt_shape ?x_strides1 ?mx_strides1 ?lt1_out_strides)
                        (ICons ?x (ICons ?mx (INil)))))
                    (= ?lt2 (Op (LessThan ?lt_shape2 ?mx_strides2 ?x_strides2 ?lt2_out_strides)
                        (ICons ?mx (ICons ?x (INil)))))

                    ; mx = Max(x) over the same axis (broadcast back via strides)
                    (= ?mx (Op (Max ?mx_shape ?cols3 ?mx_in_strides (MIter) ?mx_out_strides)
                        (ICons ?x (INil))))

                    (= ?dt (dtype ?x))
                )
                (
                    (let ?am (Op (KernelArgmax ?out_shape ?cols ?dt) (ICons ?x (INil))))
                    (union ?out ?am)
                    (set (dtype ?am) (Int))
                )
                :ruleset kernel_specialize
                :name \"kernel argmax last axis\"
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
                cols: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelArgmax {
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
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.cols.dyn_vars())
            .collect::<FxHashSet<_>>();
        let ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let cols = self.cols.to_kernel();
        let n_rows: Expression = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);

        // Tie rule: highest index wins, matching both the decomposed chain
        // (max over index*one_hot) and the CPU sampler's `max_by` (last max).
        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void argmax_k(int *out, const {ty} *in{dyn_dims_param}) {{
        __shared__ float warp_vals[{TPB} / WARP_SIZE];
        __shared__ int warp_idxs[{TPB} / WARP_SIZE];
        long long const_z = blockIdx.x;
        long long cols = {cols};
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        const {ty} *row = in + const_z * cols;
        float best = NEG_INF_F;
        int best_idx = 0;
        for (long long i = tid; i < cols; i += {TPB}) {{
            float v = (float)row[i];
            if (v > best || (v == best && (int)i > best_idx)) {{
                best = v;
                best_idx = (int)i;
            }}
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            float ov = __shfl_down_sync(FULL_MASK, best, s);
            int oi = __shfl_down_sync(FULL_MASK, best_idx, s);
            if (ov > best || (ov == best && oi > best_idx)) {{
                best = ov;
                best_idx = oi;
            }}
        }}
        if (lane_id == 0) {{
            warp_vals[warp_id] = best;
            warp_idxs[warp_id] = best_idx;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = {TPB} / WARP_SIZE;
            best = tid < cnt ? warp_vals[tid] : NEG_INF_F;
            best_idx = tid < cnt ? warp_idxs[tid] : 0;
            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                float ov = __shfl_down_sync(FULL_MASK, best, s);
                int oi = __shfl_down_sync(FULL_MASK, best_idx, s);
                if (ov > best || (ov == best && oi > best_idx)) {{
                    best = ov;
                    best_idx = oi;
                }}
            }}
            if (tid == 0) {{
                out[const_z] = best_idx;
            }}
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("argmax_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_rows, 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            64.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.cols * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.cols
    }

    fn kernel_name(&self) -> &'static str {
        "Argmax"
    }
}
