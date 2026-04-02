use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines, kernel_rewrite},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

pub type Ops = (
    KernelMeanReduce,
    KernelBatchMatVec,
    KernelBatchMatMul,
    KernelScatterNoCopy,
    KernelSoftmax,
    KernelExp,
    KernelSigmoid,
);

#[derive(Default, Debug, Clone)]

pub struct KernelMeanReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMeanReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMean",
            &[
                ("shape", ELIST),
                ("iters", EXPRESSION),
                ("strides", ELIST),
                ("iter_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Disabled: the e-graph union introduced by this rule can cause the search
        // to select genomes with accumulated FP precision issues over many layers.
        // The unfused Sum + Mul(Recip(Cast(Iota))) path produces equivalent results.
        vec![]
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
            {
                let out_shape =
                    extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
                let iters = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
                let in_stride =
                    extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
                let iter_stride = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
                let out_stride =
                    extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
                let dtype = extract_dtype(egraph, kind_children[5]);
                LLIROp::new::<dyn KernelOp>(Box::new(Self {
                    out_shape,
                    iters,
                    in_stride,
                    iter_stride,
                    out_stride,
                    dtype,
                }) as Box<dyn KernelOp>)
            },
            input_enodes,
        )
    }
}

impl KernelOp for KernelMeanReduce {
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_mean_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = blockIdx.x;
        long long n_elements = {n_outputs};
        if (const_z >= n_elements) return;

        long long in_start = {in_index};
        long long iters = {iters};
        long long iter_stride = {iter_stride};

        {dtype} sum = 0;
        for (long long i = 0; i < iters; i++) {{
            sum += in[in_start + i * iter_stride];
        }}

        out[{out_index}] = ({dtype})(sum / ({dtype})iters);
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            n_outputs = n_outputs.to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self
                .iter_stride
                .substitute('z', Expression::from(1))
                .simplify()
                .to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_mean_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()), // grid
            (1.into(), 1.into(), 1.into()),  // blocks (single-threaded)
            0.into(),                        // shmem size
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        n_outputs * self.iters + n_outputs
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "MeanReduce"
    }
}

// =============================================================================
// KernelScatterNoCopy: In-place scatter that writes directly to dest buffer
// without copying. The output buffer aliases the dest buffer.
// =============================================================================

#[derive(Debug, Clone)]
pub struct KernelScatterNoCopy {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl Default for KernelScatterNoCopy {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            dtype: DType::F32,
        }
    }
}

impl EgglogOp for KernelScatterNoCopy {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelScatterNoCopy",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn ir_defs(&self) -> Vec<String> {
        vec!["(ConsumedBuffer IR)".to_string()]
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match KernelScatter and rewrite to KernelScatterNoCopy with ConsumedBuffer on dest.
        // ConsumedBuffer wraps dest to signal in-place modification.
        //
        // Two-phase resolution:
        // 1. During (run): cleanup rules delete ConsumedBuffer if dest is shared (another op uses it)
        // 2. During (saturate base_cleanup): surviving ConsumedBuffers are valid — union with
        //    source and delete. This merges the ConsumedBuffer eclass into the source eclass,
        //    making KernelScatterNoCopy's input resolve directly to the source buffer.
        //
        // If ConsumedBuffer was deleted (shared case), cascade cleanup removes the dependent
        // ICons and KernelScatterNoCopy Op, leaving only KernelScatter.
        let mut rules = vec![
            // Rewrite: KernelScatter -> KernelScatterNoCopy with ConsumedBuffer
            Rule::raw(
                "(rule
                    (
                        (= ?scatter (Op (KernelScatter ?ds ?dst ?is ?istr ?ss ?os ?dt)
                            (ICons ?dest (ICons ?indexes (ICons ?src (INil))))))
                        (= ?dty (dtype ?src))
                    )
                    (
                        (let ?consumed (ConsumedBuffer ?dest))
                        (let ?nocopy (Op (KernelScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os ?dt)
                            (ICons ?consumed (ICons ?indexes (ICons ?src (INil))))))
                        (union ?scatter ?nocopy)
                        (set (dtype ?nocopy) ?dty)
                    )
                    :name \"scatter to scatter-no-copy\"
                )",
            ),
            // Dtype propagation for ConsumedBuffer
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?dt (dtype ?a)))
                    ((set (dtype ?cb) ?dt))
                    :name \"consumed-buffer-dtype\"
                )",
            ),
        ];
        // Cleanup: delete ConsumedBuffer when inner buffer is used by a DIFFERENT Op.
        rules.push(Rule::raw(
            "(rule
                ((= ?cb (ConsumedBuffer ?a))
                 (= ?op1 (Op ?k1 ?ilist1))
                 (= ?ilist1 (ICons ?cb ?rest1))
                 (= ?op2 (Op ?k2 ?ilist2))
                 (!= ?op1 ?op2)
                 (= ?ilist2 (ICons ?a ?t2)))
                ((delete (ConsumedBuffer ?a)))
                :ruleset cleanup
                :name \"consumed-buffer-cleanup-pos\"
            )",
        ));
        // Surviving ConsumedBuffers are valid — union with source and delete.
        // Runs in base_cleanup (after all (run) iterations).
        // TODO: figure out how to validate this is a valid ConsumedBuffer independantly so we can run it in the cleanup ruleset, rather than base_cleanup
        rules.push(Rule::raw(
            "(rule
                ((= ?cb (ConsumedBuffer ?a)))
                ((union ?cb ?a)
                 (delete (ConsumedBuffer ?a)))
                :ruleset base_cleanup
                :name \"consumed-buffer-resolve\"
            )",
        ));
        rules
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
                dest_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                dest_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[5], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[6]),
            })),
            input_enodes, // dest, indexes, src
        )
    }
}

impl KernelOp for KernelScatterNoCopy {
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
        let all_vars: FxHashSet<char> = self
            .dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
        let dyn_dims_param = if all_vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let n_src_elements = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let n_dest_elements = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let scatter_idx_idx = flatten_strides(&self.index_shape, &self.index_strides).to_kernel();
        let scatter_src_idx = flatten_strides(&self.index_shape, &self.src_strides).to_kernel();
        let scatter_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter_nocopy({dtype} *dest, const int *indexes, const {dtype} *src{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_src_elements}) return;
        int idx = indexes[{scatter_idx_idx}];
        if (idx >= 0 && idx < {n_dest_elements}) {{
            dest[idx] = src[{scatter_src_idx}];
        }}
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&scatter_kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx =
                compile_module_image_for_current_device(stream.context(), &scatter_kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("scatter_nocopy").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let n_src: Expression = self.index_shape.iter().copied().product();
        (
            func,
            module,
            scatter_kernel,
            (n_src, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.dest_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter output_bytes: {other:?}"),
        }
        .into();
        self.output_size() * elem_size
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // scatter_nocopy kernel: (dest, indexes, src [, dyn_dims])
        // Write directly to dest buffer (input_ptrs[0]), NOT to output_ptr
        let mut params = vec![input_ptrs[0], input_ptrs[1], input_ptrs[2]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn bytes_loaded(&self) -> Expression {
        let data_elem_size: Expression = match self.dtype {
            DType::F64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter bytes_loaded: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // Only load indices + src (no dest copy!)
        n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
        let data_elem_size: Expression = match self.dtype {
            DType::F64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter bytes_stored: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // Only store the scattered elements
        n_src * data_elem_size
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0) // output aliases dest (input 0)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "ScatterNoCopy"
    }
}

// =============================================================================
// KernelBatchMatVec: Fused batched matrix-vector product for attention
// Matches: Mul(broadcast) + Sum pattern for [B, 1, K] x [B, K, N] -> [B, 1, N]
// or [B, M, K] x [B, K, N] -> [B, M, N] with small M
// Replaces the broadcast KernelMul + single-threaded KernelSumReduce pipeline
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelBatchMatVec {
    // Output shape: the final reduced shape [B..., M, N]
    out_shape: Vec<Expression>,
    // K: the reduction dimension (was the Sum iters)
    k_dim: Expression,
    // Strides for input A (with K dim removed)
    a_stride: Vec<Expression>,
    a_k_stride: Expression,
    // Strides for input B (with K dim removed)
    b_stride: Vec<Expression>,
    b_k_stride: Expression,
    // Output strides
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelBatchMatVec {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelBatchMatVec",
            &[
                ("out_shape", ELIST),
                ("k_dim", EXPRESSION),
                ("a_stride", ELIST),
                ("a_k_stride", EXPRESSION),
                ("b_stride", ELIST),
                ("b_k_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    ; Match Mul node (broadcast multiply)
                    (= ?mul (Op (Mul ?mul_shape ?a_stride ?b_stride ?mul_out_stride) (ICons ?a (ICons ?b (INil)))))

                    ; Match Sum that reduces the Mul (k dimension)
                    (= ?sum (Op (Sum ?out_shape ?k ?sum_in_stride ?k_stride ?sum_out_stride) (ICons ?mul (INil))))

                    ; Output shape must have 3+ dimensions (batched)
                    (= ?out_shape (ECons ?batch_or_d0 (ECons ?d1 (ECons ?d2 ?rest))))

                    ; k_stride must be contiguous
                    (= ?k_stride (MIter))

                    ; Get A's k-dimension stride (second from end in Mul's a_stride)
                    (= ?a_k_stride (nth_from_end ?a_stride 1))

                    ; Get B's k-dimension stride (second from end in Mul's b_stride)
                    (= ?b_k_stride (nth_from_end ?b_stride 1))

                    ; A's k stride must be contiguous (row-major A)
                    (= ?a_k_stride (MIter))

                    ; B's k stride must be contiguous (col-major B)
                    (= ?b_k_stride (MIter))

                    ; Must be F32
                    (= (F32) (dtype ?a))
                    (= (F32) (dtype ?b))
                )
                (
                    ; Remove the k-dimension from A strides for the kernel
                    (let ?a_kern_stride (RemoveNthFromEnd ?a_stride 1))
                    ; Remove the k-dimension from B strides
                    (let ?b_kern_stride (RemoveNthFromEnd ?b_stride 1))

                    (let ?bmv (Op (KernelBatchMatVec
                        ?out_shape ?k
                        ?a_kern_stride ?a_k_stride
                        ?b_kern_stride ?b_k_stride
                        ?sum_out_stride (F32)) (ICons ?a (ICons ?b (INil)))))
                    (union ?sum ?bmv)
                    (set (dtype ?bmv) (F32))
                )
                :name \"batch mat-vec\"
            )"
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
                a_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                a_k_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                b_k_stride: extract_expr(egraph, kind_children[5], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[7]),
            })),
            input_enodes, // A, B
        )
    }
}

impl KernelOp for KernelBatchMatVec {
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
        let vars: FxHashSet<char> = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.k_dim.dyn_vars())
            .chain(self.a_k_stride.dyn_vars())
            .chain(self.b_k_stride.dyn_vars())
            .collect();

        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        // Each output element is a dot product of length K.
        // We launch one block of 256 threads per output element.
        // Threads cooperatively reduce K using warp shuffles.
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let k_expr = self.k_dim.to_kernel();
        let a_k_stride_expr = self
            .a_k_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let b_k_stride_expr = self
            .b_k_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();

        let kernel = format!(
            "
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void batch_matvec(float *out, const float *A, const float *B{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long a_base = {a_idx};
        long long b_base = {b_idx};
        long long K = {k_expr};
        long long a_k_stride = {a_k_stride_expr};
        long long b_k_stride = {b_k_stride_expr};

        float partial = 0.0f;
        for (long long k = tid; k < K; k += THREADS_PER_BLOCK) {{
            partial += A[a_base + k * a_k_stride] * B[b_base + k * b_k_stride];
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_idx}] = block_sum;
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
            let func = module.load_function("batch_matvec").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()), // grid: one block per output
            (256.into(), 1.into(), 1.into()), // block: 256 threads
            32.into(),                       // shared mem for warp_sums
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let n = self.output_size();
        // Each output loads K elements from A and K elements from B
        n * self.k_dim * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // Each output: K multiply-adds = 2*K FLOPs
        self.output_size() * self.k_dim * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "BatchMatVec"
    }
}

// =============================================================================
// KernelBatchMatMul: General batched matmul with arbitrary strides
// Like KernelBatchMatVec but handles non-contiguous K strides (e.g., transposed
// inputs) and non-uniform batch strides (e.g., GQA expansion). One block of 256
// threads per output element; threads cooperatively reduce along K.
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelBatchMatMul {
    out_shape: Vec<Expression>,
    k_dim: Expression,
    a_stride: Vec<Expression>,
    a_k_stride: Expression,
    b_stride: Vec<Expression>,
    b_k_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelBatchMatMul {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelBatchMatMul",
            &[
                ("out_shape", ELIST),
                ("k_dim", EXPRESSION),
                ("a_stride", ELIST),
                ("a_k_stride", EXPRESSION),
                ("b_stride", ELIST),
                ("b_k_stride", EXPRESSION),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    ; Match Mul node (broadcast multiply)
                    (= ?mul (Op (Mul ?mul_shape ?a_stride ?b_stride ?mul_out_stride) (ICons ?a (ICons ?b (INil)))))

                    ; Match Sum that reduces the Mul (k dimension)
                    (= ?sum (Op (Sum ?out_shape ?k ?sum_in_stride ?k_stride ?sum_out_stride) (ICons ?mul (INil))))

                    ; Output shape must have 3+ dimensions (batched)
                    (= ?out_shape (ECons ?batch_or_d0 (ECons ?d1 (ECons ?d2 ?rest))))

                    ; k_stride must be contiguous in the Sum output
                    (= ?k_stride (MIter))

                    ; K must be > 1 (K=1 is a degenerate outer product, not a real matmul)
                    (!= ?k (MNum 1))

                    ; Get A's and B's k-dimension strides (no contiguity requirement)
                    (= ?a_k_stride (nth_from_end ?a_stride 1))
                    (= ?b_k_stride (nth_from_end ?b_stride 1))

                    ; One of A's non-k strides must be 0 (broadcast along n)
                    (= (MNum 0) (nth_from_end ?a_stride 0))

                    ; One of B's non-k strides must be 0 (broadcast along m)
                    (= (MNum 0) (nth_from_end ?b_stride 2))

                    ; Must be F32
                    (= (F32) (dtype ?a))
                    (= (F32) (dtype ?b))
                )
                (
                    (let ?a_kern_stride (RemoveNthFromEnd ?a_stride 1))
                    (let ?b_kern_stride (RemoveNthFromEnd ?b_stride 1))

                    (let ?bmm (Op (KernelBatchMatMul
                        ?out_shape ?k
                        ?a_kern_stride ?a_k_stride
                        ?b_kern_stride ?b_k_stride
                        ?sum_out_stride (F32)) (ICons ?a (ICons ?b (INil)))))
                    (union ?sum ?bmm)
                    (set (dtype ?bmm) (F32))
                )
                :name \"batch matmul\"
            )"
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
                a_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                a_k_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                b_k_stride: extract_expr(egraph, kind_children[5], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[7]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelBatchMatMul {
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
        let vars: FxHashSet<char> = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.k_dim.dyn_vars())
            .chain(self.a_k_stride.dyn_vars())
            .chain(self.b_k_stride.dyn_vars())
            .collect();

        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let k_expr = self.k_dim.to_kernel();
        let a_k_stride_expr = self
            .a_k_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let b_k_stride_expr = self
            .b_k_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();

        let kernel = format!(
            "
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void batch_matmul(float *out, const float *A, const float *B{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long a_base = {a_idx};
        long long b_base = {b_idx};
        long long K = {k_expr};
        long long a_k_stride = {a_k_stride_expr};
        long long b_k_stride = {b_k_stride_expr};

        float partial = 0.0f;
        for (long long k = tid; k < K; k += THREADS_PER_BLOCK) {{
            partial += A[a_base + k * a_k_stride] * B[b_base + k * b_k_stride];
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_idx}] = block_sum;
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
            let func = module.load_function("batch_matmul").unwrap();
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
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let n = self.output_size();
        n * self.k_dim * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.k_dim * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "BatchMatMul"
    }
}

// =============================================================================
// KernelSoftmax: Fused softmax over last dimension
// Matches: Mul(Recip(Sum(Exp2(Sub(x, Max(x))))), Exp2(Sub(x, Max(x))))
// Replaces 5+ kernel launches with a single fused kernel
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelSoftmax {
    out_shape: Vec<Expression>,  // shape of output (same as input)
    in_stride: Vec<Expression>,  // input strides
    out_stride: Vec<Expression>, // output strides
    reduce_dim: Expression,      // size of the softmax dimension (last dim)
    reduce_stride: Expression,   // stride along softmax dimension in input
}

impl EgglogOp for KernelSoftmax {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSoftmax",
            &[
                ("shape", ELIST),
                ("in_strides", ELIST),
                ("out_strides", ELIST),
                ("reduce_dim", EXPRESSION),
                ("reduce_stride", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            kernel_rewrite::<luminal::hlir::Softmax, Self>(),
            // Also add a direct rewrite that assumes F32 dtype, in case dtype
            // propagation hasn't reached the Softmax node yet.
            Rule::raw(
                "(rule
                (
                    (= ?sm (Op (Softmax ?shape ?in_strides ?out_strides ?reduce_dim ?reduce_stride) ?inputs))
                )
                (
                    (let ?ksm (Op (KernelSoftmax ?shape ?in_strides ?out_strides ?reduce_dim ?reduce_stride (F32)) ?inputs))
                    (union ?sm ?ksm)
                    (set (dtype ?ksm) (F32))
                )
                :name \"softmax-to-kernel-f32\"
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
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let in_stride =
            extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap();
        let out_stride =
            extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
        let reduce_dim = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let reduce_stride = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape,
                in_stride,
                out_stride,
                reduce_dim,
                reduce_stride,
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelSoftmax {
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
        let vars: FxHashSet<char> = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.reduce_dim.dyn_vars())
            .chain(self.reduce_stride.dyn_vars())
            .collect();

        // n_rows = product of all dims except the last (reduce dim)
        let n_rows: Expression = self.out_shape[..self.out_shape.len() - 1]
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        // Each block handles one row. 256 threads cooperatively compute softmax.
        let in_idx = flatten_strides(
            &self.out_shape[..self.out_shape.len() - 1],
            &self.in_stride[..self.in_stride.len() - 1],
        )
        .to_kernel();
        let out_idx = flatten_strides(
            &self.out_shape[..self.out_shape.len() - 1],
            &self.out_stride[..self.out_stride.len() - 1],
        )
        .to_kernel();
        let reduce_dim_expr = self.reduce_dim.to_kernel();
        let in_reduce_stride = self
            .reduce_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let out_reduce_stride = self.out_stride[self.out_stride.len() - 1]
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();

        let kernel = format!(
            "
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void fused_softmax(float *out, const float *inp{dyn_dims_param}) {{
        __shared__ float shared[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_base = {in_idx};
        long long out_base = {out_idx};
        long long N = {reduce_dim_expr};
        long long in_stride = {in_reduce_stride};
        long long out_stride = {out_reduce_stride};

        // Pass 1: find max
        float max_val = NEG_INF_F;
        for (long long i = tid; i < N; i += THREADS_PER_BLOCK) {{
            max_val = fmaxf(max_val, inp[in_base + i * in_stride]);
        }}
        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            max_val = fmaxf(max_val, __shfl_down_sync(FULL_MASK, max_val, s));
        }}
        if (lane_id == 0) shared[warp_id] = max_val;
        __syncthreads();
        if (warp_id == 0) {{
            max_val = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? shared[tid] : NEG_INF_F;
            #pragma unroll
            for (int s = (THREADS_PER_BLOCK / WARP_SIZE) / 2; s > 0; s /= 2) {{
                max_val = fmaxf(max_val, __shfl_down_sync(FULL_MASK, max_val, s));
            }}
            shared[0] = max_val;
        }}
        __syncthreads();
        max_val = shared[0];

        // Pass 2: compute exp2 and sum
        float sum_val = 0.0f;
        for (long long i = tid; i < N; i += THREADS_PER_BLOCK) {{
            float v = exp2f((inp[in_base + i * in_stride] - max_val) * 1.4426950408889634f);
            out[out_base + i * out_stride] = v;  // store exp temporarily
            sum_val += v;
        }}
        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            sum_val += __shfl_down_sync(FULL_MASK, sum_val, s);
        }}
        if (lane_id == 0) shared[warp_id] = sum_val;
        __syncthreads();
        if (warp_id == 0) {{
            sum_val = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? shared[tid] : 0.0f;
            #pragma unroll
            for (int s = (THREADS_PER_BLOCK / WARP_SIZE) / 2; s > 0; s /= 2) {{
                sum_val += __shfl_down_sync(FULL_MASK, sum_val, s);
            }}
            shared[0] = sum_val;
        }}
        __syncthreads();
        float inv_sum = 1.0f / shared[0];

        // Pass 3: normalize
        for (long long i = tid; i < N; i += THREADS_PER_BLOCK) {{
            out[out_base + i * out_stride] *= inv_sum;
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("fused_softmax").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_rows, 1.into(), 1.into()),     // grid: one block per row
            (256.into(), 1.into(), 1.into()), // block: 256 threads
            32.into(),                        // shared mem
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        // 3 passes over input (max, exp+sum, normalize reads from output)
        self.output_size() * 4 * 3
    }

    fn bytes_stored(&self) -> Expression {
        // 2 writes: exp values, then normalized values
        self.output_size() * 4 * 2
    }

    fn flops(&self) -> Expression {
        // Per element: sub, exp2, add (sum), div = ~4 ops
        self.output_size() * 4
    }

    fn kernel_name(&self) -> &'static str {
        "Softmax"
    }
}

// KernelExp: native exp (uses expf instead of exp2f * constant)
// Single-kernel alternative to the 3-kernel Constant+Mul+Exp2 path.
// Improves numerical precision by avoiding the truncated log2(e) constant.

#[derive(Default, Debug, Clone)]
pub struct KernelExp {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelExp {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelExp",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // Match Exp2(Mul(x, log2e_constant)) directly.
            // This matches the pattern created by frontend exp() = (self * (1/ln(2))).exp2()
            Rule::raw(
                "(rule
                (
                    (= ?mul (Op (Mul ?shape ?x_stride ?const_stride ?inter_stride) (ICons ?x (ICons ?exp_const (INil)))))
                    (= ?exp2 (Op (Exp2 ?shape ?inter_stride ?out_stride) (ICons ?mul (INil))))
                    (= ?dt (dtype ?x))
                    (= ?cv (Op (Constant ?val) (INil)))
                    (= ?exp_const ?cv)
                    (> ?val 1.44)
                    (< ?val 1.45)
                )
                (
                    (let ?kexp (Op (KernelExp ?shape ?x_stride ?out_stride ?dt) (ICons ?x (INil))))
                    (union ?exp2 ?kexp)
                    (set (dtype ?kexp) ?dt)
                )
                :name \"direct-exp-fusion\"
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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelExp {
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
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void exp_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = expf(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("exp_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Exp"
    }
}

// KernelSigmoid: fused sigmoid = 1/(1+exp(-x))
// Single-kernel alternative to the 5-kernel Neg+Exp+Const+Add+Recip path.

#[derive(Default, Debug, Clone)]
pub struct KernelSigmoid {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelSigmoid {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSigmoid",
            &[
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // Match the HLIR pattern directly: Recip(Add(Exp2(Mul(Mul(x, -1), log2e)), 1))
            Rule::raw(
                "(rule
                (
                    (= ?neg1 (Op (Constant ?nv) (INil)))
                    (< ?nv -0.99)
                    (> ?nv -1.01)
                    (= ?neg_x (Op (Mul ?shape ?x_stride ?neg_stride ?neg_out_stride) (ICons ?x (ICons ?neg1 (INil)))))
                    (= ?log2e (Op (Constant ?lv) (INil)))
                    (> ?lv 1.44)
                    (< ?lv 1.45)
                    (= ?scaled (Op (Mul ?shape ?neg_out_stride ?log2e_stride ?scaled_stride) (ICons ?neg_x (ICons ?log2e (INil)))))
                    (= ?exp2 (Op (Exp2 ?shape ?scaled_stride ?exp_stride) (ICons ?scaled (INil))))
                    (= ?one (Op (Constant ?ov) (INil)))
                    (> ?ov 0.99)
                    (< ?ov 1.01)
                    (= ?plus_one (Op (Add ?shape ?exp_stride ?one_stride ?add_stride) (ICons ?exp2 (ICons ?one (INil)))))
                    (= ?sig_out (Op (Recip ?shape ?add_stride ?out_stride) (ICons ?plus_one (INil))))
                    (= ?dt (dtype ?x))
                )
                (
                    (let ?ksig (Op (KernelSigmoid ?shape ?x_stride ?out_stride ?dt) (ICons ?x (INil))))
                    (union ?sig_out ?ksig)
                    (set (dtype ?ksig) ?dt)
                )
                :name \"direct-sigmoid-fusion\"
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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[3]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelSigmoid {
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
            .shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.shape, &self.out_strides).to_kernel();
        let in_idx = flatten_strides(&self.shape, &self.in_strides).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void sigmoid_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = 1.0f / (1.0f + expf(-in[{in_idx}]));
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("sigmoid_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        // neg + exp + add + recip = ~4 ops per element
        self.shape.iter().copied().product::<Expression>() * 4
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Sigmoid"
    }
}
