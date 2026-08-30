//! Warp-per-row GEMV for M=1 decode matmuls.
//!
//! Matches the same row-major × column-major `GenericMatmul` pattern as the
//! cuBLASLt RmCm rewrite, restricted to m == 1 and 16-bit dtypes, and unions
//! a lean kernel into the same eclass — measured search then chooses among
//! cuBLASLt, GenericMatmul, the lowered reduction, and this kernel per shape.
//! Rationale (measured on Llama 3 8B decode): the nvjet GEMV
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
                "(= ?out_shape (ECons (MNum 1) (ECons ?n (ENil))))
                            (= ?semantic_m (MNum 1))",
            ),
            (
                "dyn",
                "(= ?out_shape (ECons ?m (ECons ?n (ENil))))
                            (= ?semantic_m ?m)
                            (= ?m_lower (lower ?m))
                            (= ?m_upper (upper ?m))
                            (> ?m_lower 0)
                            (< ?m_upper 2)",
            ),
        ];
        m_variants
            .into_iter()
            .flat_map(|(variant, m_cond)| {
                ["Bf16", "F16"].map(move |dt| (variant, m_cond, dt))
            })
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
                            (generic_matmul_exact_2d
                                ?sum ?semantic_m ?n ?k ({dt}))
                            (= ?matmul_dtype ({dt}))
                            (!= ?n (MNum 0))
                            (!= ?k (MNum 1))

                            (= ?a_stride (ECons ?a_m_stride (ECons ?a_n_stride (ECons ?a_k_stride (ENil)))))
                            (= ?b_stride (ECons ?b_m_stride (ECons ?b_n_stride (ECons ?b_k_stride (ENil)))))
                            (= ?k_stride (MIter))

                            ; A (the activation row) reads as a contiguous
                            ; [k] vector. Both m variants guarantee m <= 1,
                            ; so the m stride is never advanced and carries
                            ; no layout constraint: a reshaped view (e.g. an
                            ; attention output merged from [heads, head_dim])
                            ; may spell it as any value.
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
        FxHashMap<Symbol, CudaSlice<u8>>,
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
