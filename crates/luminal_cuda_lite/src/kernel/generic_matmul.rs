use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{
        KernelOp,
        hlir::{dtype_includes, generate_dyn_dims_defines},
    },
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
    shape::flatten_strides,
};

const MATMUL_BACKEND_RELATION_DECLARATIONS: &str = "(relation generic_matmul_exact_2d
        (IR Expression Expression Expression DType))
     (relation generic_matmul_exact_3d
        (IR Expression Expression Expression Expression DType))
     (relation low_precision_matmul_dtype (DType))
     (low_precision_matmul_dtype (F16))
     (low_precision_matmul_dtype (Bf16))";

#[derive(Default, Debug, Clone)]
pub struct GenericMatmul {
    out_shape: Vec<Expression>,
    mul_shape: Vec<Expression>,
    k: Expression,
    lhs_strides: Vec<Expression>,
    rhs_strides: Vec<Expression>,
    sum_input_strides: Vec<Expression>,
    sum_iter_stride: Expression,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for GenericMatmul {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GenericMatmul",
            &[
                ("out_shape", ELIST),
                ("mul_shape", ELIST),
                ("k", EXPRESSION),
                ("lhs_strides", ELIST),
                ("rhs_strides", ELIST),
                ("sum_input_strides", ELIST),
                ("sum_iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn egglog_declarations(&self) -> Vec<String> {
        vec![MATMUL_BACKEND_RELATION_DECLARATIONS.to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // The exact-layout witnesses below pattern-match the canonical
            // nested stride spellings, but HLIR serialization and constant
            // folding often leave only the folded product in the e-graph
            // (e.g. z*8388608 instead of (z*4096)*2048), so a semantically
            // canonical layout fails the syntactic match. Seed the canonical
            // spellings for every GenericMatmul: the expr fold rules union a
            // seeded spelling into the actual stride's e-class exactly when
            // the two are numerically equal, which lets the witness match
            // without ever equating unequal strides.
            Rule::raw(
                "(rule
                    (
                        (= ?sum (Op (GenericMatmul
                            (ECons ?m (ECons ?n (ENil)))
                            (ECons ?m (ECons ?n (ECons ?k (ENil))))
                            ?k
                            ?lhs_strides ?rhs_strides
                            ?sum_input_strides ?sum_iter_stride ?out_strides
                            ?dt)
                            ?inputs))
                    )
                    (
                        (let ?canonical_sum_m (MMul (MMul (MIter) ?k) ?n))
                        (let ?canonical_sum_n (MMul (MIter) ?k))
                        (let ?canonical_out_m (MMul (MIter) ?n))
                    )
                    :ruleset matmul_backend
                    :name \"seed canonical 2d matmul stride spellings\"
                )

                (rule
                    (
                        (= ?sum (Op (GenericMatmul
                            (ECons ?batch (ECons ?m (ECons ?n (ENil))))
                            (ECons ?batch (ECons ?m (ECons ?n (ECons ?k (ENil)))))
                            ?k
                            ?lhs_strides ?rhs_strides
                            ?sum_input_strides ?sum_iter_stride ?out_strides
                            ?dt)
                            ?inputs))
                    )
                    (
                        (let ?canonical_sum_batch (MMul (MMul (MMul (MIter) ?k) ?n) ?m))
                        (let ?canonical_sum_m (MMul (MMul (MIter) ?k) ?n))
                        (let ?canonical_sum_n (MMul (MIter) ?k))
                        (let ?canonical_out_batch (MMul (MMul (MIter) ?n) ?m))
                        (let ?canonical_out_m (MMul (MIter) ?n))
                    )
                    :ruleset matmul_backend
                    :name \"seed canonical 3d matmul stride spellings\"
                )",
            ),
            Rule::raw(
                "; cuBLASLt and GEMV implement only the canonical materialized
             ; matmul reduction. Operand broadcast strides alone are not a
             ; sufficient witness: a movement-only view between Mul and Sum
             ; can permute the reduction's surviving axes while leaving both
             ; operand strides unchanged. Stage the complete semantic layout
             ; here so every optimized consumer shares the same proof.
             (rule
                (
                    (= ?sum (Op (GenericMatmul
                        (ECons ?m (ECons ?n (ENil)))
                        (ECons ?m (ECons ?n (ECons ?k (ENil))))
                        ?k
                        ?lhs_strides ?rhs_strides
                        (ECons
                            (MMul (MMul (MIter) ?k) ?n)
                            (ECons (MMul (MIter) ?k) (ENil)))
                        (MIter)
                        (ECons (MMul (MIter) ?n) (ECons (MIter) (ENil)))
                        ?dt)
                        ?inputs))
                )
                ((generic_matmul_exact_2d ?sum ?m ?n ?k ?dt))
                :ruleset matmul_backend
                :name \"prove exact materialized 2d matmul reduction\"
             )

             (rule
                (
                    (= ?sum (Op (GenericMatmul
                        (ECons ?batch (ECons ?m (ECons ?n (ENil))))
                        (ECons ?batch (ECons ?m (ECons ?n (ECons ?k (ENil)))))
                        ?k
                        ?lhs_strides ?rhs_strides
                        (ECons
                            (MMul (MMul (MMul (MIter) ?k) ?n) ?m)
                            (ECons
                                (MMul (MMul (MIter) ?k) ?n)
                                (ECons (MMul (MIter) ?k) (ENil))))
                        (MIter)
                        (ECons
                            (MMul (MMul (MIter) ?n) ?m)
                            (ECons (MMul (MIter) ?n) (ECons (MIter) (ENil))))
                        ?dt)
                        ?inputs))
                )
                ((generic_matmul_exact_3d ?sum ?batch ?m ?n ?k ?dt))
                :ruleset matmul_backend
                :name \"prove exact materialized 3d matmul reduction\"
             )

             (rule
                    (
                        (= ?mul (Op (Mul ?mul_shape ?lhs_strides ?rhs_strides ?mul_out_strides)
                            (ICons ?lhs (ICons ?rhs (INil)))))
                        (= ?sum (Op (Sum ?out_shape ?k ?sum_input_strides ?sum_iter_stride ?out_strides)
                            (ICons ?mul (INil))))
                        (= ?dt (dtype ?sum))
                    )
                    (
                        (let ?generic (Op (GenericMatmul
                            ?out_shape
                            ?mul_shape
                            ?k
                            ?lhs_strides
                            ?rhs_strides
                            ?sum_input_strides
                            ?sum_iter_stride
                            ?out_strides
                            ?dt)
                            (ICons ?lhs (ICons ?rhs (INil)))))
                        (union ?sum ?generic)
                        (set (dtype ?generic) ?dt)
                    )
                    :ruleset matmul_backend
                    :name \"generic-matmul-cuda-mul-sum\"
                )",
            ),

            // A low-precision materialized Mul followed by a separate reduction is
            // not the floating-point matmul contract implemented by GenericMatmul,
            // GEMV, or cuBLASLt: it rounds every product to F16/BF16 before the
            // reduction instead of accumulating products in F32. Keep search wide
            // across the F32-accumulating backends, not across a numerically
            // different product-materialization algorithm. F32 keeps its
            // decomposed alternative because there is no low-precision product
            // rounding to eliminate.
            //
            // GraphTensor::matmul is already decomposed to Mul + Sum at this IR
            // boundary, so there is no separate provenance marker to distinguish
            // it from the same contraction authored directly. The shared
            // GenericMatmul e-class is therefore the witness that this contraction
            // has the backend matmul accumulator contract. A future first-class
            // contraction marker could make that provenance explicit.
            //
            // Both forms must be covered because cleanup can observe either the
            // original HLIR Sum or its CUDA KernelSum lowering depending on rule
            // scheduling.
            Rule::raw(
                "(rule
                    (
                        (= ?mul (Op (Mul ?mul_shape ?lhs_strides ?rhs_strides ?mul_out_strides)
                            (ICons ?lhs (ICons ?rhs (INil)))))
                        (= ?sum (Op (Sum ?out_shape ?k ?sum_input_strides ?sum_iter_stride ?out_strides)
                            (ICons ?mul (INil))))
                        (= ?sum (Op (GenericMatmul
                            ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?dt)
                            ?generic_inputs))
                        (= ?dt (dtype ?sum))
                        (= ?dt (dtype ?mul))
                        (low_precision_matmul_dtype ?dt)
                    )
                    (
                        (delete (Op (Sum ?out_shape ?k ?sum_input_strides ?sum_iter_stride ?out_strides)
                            (ICons ?mul (INil))))
                    )
                    :ruleset cleanup
                    :name \"delete-low-precision-sum-when-wide-accum-matmul-exists\"
                )

                (rule
                    (
                        (= ?kernel_sum (Op (KernelSum
                            ?out_shape ?k ?sum_input_strides ?sum_iter_stride ?out_strides ?dt)
                            ?sum_inputs))
                        (= ?kernel_sum (Op (GenericMatmul
                            ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?dt)
                            ?generic_inputs))
                        (= ?dt (dtype ?kernel_sum))
                        (low_precision_matmul_dtype ?dt)
                    )
                    (
                        (delete (Op (KernelSum
                            ?out_shape ?k ?sum_input_strides ?sum_iter_stride ?out_strides ?dt)
                            ?sum_inputs))
                    )
                    :ruleset cleanup
                    :name \"delete-low-precision-kernel-sum-when-wide-accum-matmul-exists\"
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
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                mul_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                k: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                lhs_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                rhs_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                sum_input_strides: extract_expr_list(
                    egraph,
                    kind_children[5],
                    list_cache,
                    expr_cache,
                )
                .unwrap(),
                sum_iter_stride: extract_expr(egraph, kind_children[6], expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[7], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[8]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for GenericMatmul {
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
        let vars = self.all_dyn_vars();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let n_outputs = self.output_size();
        let sum_base_idx = flatten_strides(&self.out_shape, &self.sum_input_strides).to_kernel();
        let iter_offset = self.sum_iter_stride.to_kernel_with_index("i");
        let lhs_idx =
            flatten_strides(&self.mul_shape, &self.lhs_strides).to_kernel_with_index("mul_idx");
        let rhs_idx =
            flatten_strides(&self.mul_shape, &self.rhs_strides).to_kernel_with_index("mul_idx");
        let out_idx = flatten_strides(&self.out_shape, &self.out_strides).to_kernel();
        let k = self.k.to_kernel();

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void generic_matmul({dtype} *out, const {dtype} *lhs, const {dtype} *rhs{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        if (const_z >= {n_outputs}) return;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long base_idx = {sum_base_idx};
        long long iters = {k};

        float partial = 0.0f;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            long long mul_idx = base_idx + {iter_offset};
            partial += static_cast<float>(lhs[{lhs_idx}]) * static_cast<float>(rhs[{rhs_idx}]);
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            float block_sum = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = (THREADS_PER_BLOCK / WARP_SIZE) / 2; s > 0; s >>= 1) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_idx}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            n_outputs = n_outputs.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_matmul").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            32.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn all_dyn_vars(&self) -> FxHashSet<Symbol> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.mul_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.k.dyn_vars())
            .chain(self.lhs_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.rhs_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.sum_input_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.sum_iter_stride.dyn_vars())
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.k * self.dtype.bits() * 2).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.k * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericMatmul"
    }
}
