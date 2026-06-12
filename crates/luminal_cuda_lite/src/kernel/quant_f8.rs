//! Fused FP8 quantization: `(f8)((float)x * (1/scale))` in one kernel.
//!
//! Matches the per-linear quantization chain the fp8 pipeline spells as
//! `x.cast(F32) / input_scale` followed by `.cast(F8E4M3)`:
//!
//! ```text
//! q = Cast(F8E4M3)( Mul( Cast(F32)(x_bf16), Recip(scale) ) )
//! ```
//!
//! and unions a single kernel into the outer Cast's eclass (dtype F8E4M3,
//! unchanged). The bf16→f32 widening is exact and the f32 multiply/round
//! sequence is identical, so the fused kernel computes the same values as
//! the three-kernel chain. The scaled-fp8 cuBLASLt rewrite consumes the same
//! eclass as its A input, so this slots in front of every fp8 GEMM.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::dtype_includes, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::*,
    prelude::*,
};

const TPB: usize = 256;

#[derive(Default, Debug, Clone)]
pub struct KernelQuantF8 {
    size: Expression,
}

impl EgglogOp for KernelQuantF8 {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelQuantF8", &[("size", EXPRESSION)])
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    (= ?xf (Op (Cast ?xf_size (F32)) (ICons ?x (INil))))
                    (= (Bf16) (dtype ?x))
                    (= ?recip (Op (Recip ?r_shape ?r_in_strides ?r_out_strides)
                        (ICons ?scale (INil))))
                    (= ?mul (Op (Mul ?m_shape ?xf_strides ?recip_strides ?m_out_strides)
                        (ICons ?xf (ICons ?recip (INil)))))
                    (= ?q (Op (Cast ?q_size (F8E4M3)) (ICons ?mul (INil))))
                )
                (
                    (let ?kq (Op (KernelQuantF8 ?q_size) (ICons ?x (ICons ?scale (INil)))))
                    (union ?q ?kq)
                    (set (dtype ?kq) (F8E4M3))
                )
                :ruleset kernel_specialize
                :name \"kernel quant f8 from bf16\"
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
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                size: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelQuantF8 {
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
        let vars = self.size.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let includes = dtype_includes(&[DType::Bf16, DType::F8E4M3]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let size = self.size.to_kernel();

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void quant_f8_k(__nv_fp8_e4m3 *out, const __nv_bfloat16 *x, const float *scale{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {size}) return;
        float rs = 1.0f / scale[0];
        out[const_z] = (__nv_fp8_e4m3)((float)x[const_z] * rs);
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("quant_f8_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (self.size.ceil_div(TPB), 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn output_bytes(&self) -> Expression {
        self.size
    }

    fn output_dtype(&self) -> DType {
        DType::F8E4M3
    }

    fn bytes_loaded(&self) -> Expression {
        self.size * 2 + 4
    }

    fn bytes_stored(&self) -> Expression {
        self.size
    }

    fn flops(&self) -> Expression {
        self.size
    }

    fn kernel_name(&self) -> &'static str {
        "QuantF8"
    }
}
