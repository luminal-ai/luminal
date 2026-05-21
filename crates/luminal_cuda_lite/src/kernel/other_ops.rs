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
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND, STRING},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
    prelude::*,
};

pub type Ops = (KernelMeanReduce, KernelScatterNoCopy, KernelSoftmax);

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
        let threads_per_block: usize = 256; // 8 warps per block
        let n_warps = threads_per_block / 32;
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

        float thread_sum = 0.0f;
        for (long long i = threadIdx.x; i < iters; i += {threads_per_block})
            thread_sum += (float)in[in_start + i * iter_stride];

        for (int offset = 16; offset > 0; offset >>= 1)
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

        __shared__ float warp_sums[{n_warps}];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) warp_sums[warp] = thread_sum;
        __syncthreads();

        if (threadIdx.x == 0) {{
            float sum = 0.0f;
            for (int w = 0; w < {n_warps}; w++) sum += warp_sums[w];
            out[{out_index}] = ({dtype})(sum / (float)iters);
        }}
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
            threads_per_block = threads_per_block,
            n_warps = n_warps,
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
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // block
            0.into(),                                       // shmem size
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
        // This is only valid when the destination buffer can also represent
        // the scatter output layout. If dest is a strided/broadcast view,
        // regular Scatter must first materialize a contiguous output copy.
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
            Rule::raw("(relation consumed_buffer_ilist_contains (IList IR))"),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail)))
                    ((consumed_buffer_ilist_contains ?list ?head))
                    :ruleset cleanup
                    :name \"consumed-buffer-ilist-contains-head\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail))
                     (consumed_buffer_ilist_contains ?tail ?item))
                    ((consumed_buffer_ilist_contains ?list ?item))
                    :ruleset cleanup
                    :name \"consumed-buffer-ilist-contains-tail\"
                )",
            ),
            // Rewrite: KernelScatter -> KernelScatterNoCopy with ConsumedBuffer
            Rule::raw(
                "(rule
                    (
                        (= ?scatter (Op (KernelScatter ?ds ?dst ?is ?istr ?ss ?os ?dt)
                            (ICons ?dest (ICons ?indexes (ICons ?src (INil))))))
                        (= ?dst ?os)
                        (= ?dty (dtype ?src))
                    )
                    (
                        (let ?consumed (ConsumedBuffer ?dest))
                        (let ?nocopy (Op (KernelScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os ?dt)
                            (ICons ?consumed (ICons ?indexes (ICons ?src (INil))))))
                        (union ?scatter ?nocopy)
                        (set (dtype ?nocopy) ?dty)
                    )
                    :ruleset buffer_reuse
                    :name \"scatter to scatter-no-copy\"
                )",
            ),
            // Dtype propagation for ConsumedBuffer
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?dt (dtype ?a)))
                    ((set (dtype ?cb) ?dt))
                    :ruleset dtype_prop
                    :name \"consumed-buffer-dtype\"
                )",
            ),
        ];
        // Cleanup: delete ConsumedBuffer when inner buffer is used by a DIFFERENT Op.
        rules.push(Rule::raw(
            "(rule
                ((= ?cb (ConsumedBuffer ?a))
                 (= ?op1 (Op ?k1 ?ilist1))
                 (consumed_buffer_ilist_contains ?ilist1 ?cb)
                 (= ?op2 (Op ?k2 ?ilist2))
                 (!= ?op1 ?op2)
                 (consumed_buffer_ilist_contains ?ilist2 ?a))
                ((delete (ConsumedBuffer ?a)))
                :ruleset cleanup
                :name \"consumed-buffer-cleanup-shared-op-use\"
            )",
        ));
        // If a valid no-copy scatter survives cleanup, it dominates the copying scatter.
        // This must run before base_cleanup resolves ConsumedBuffer back to the destination.
        rules.push(Rule::raw(
            "(rule
                ((= ?cb (ConsumedBuffer ?dest))
                 (= ?scatter (Op (KernelScatter ?ds ?dst ?is ?istr ?ss ?os ?dt)
                     (ICons ?dest (ICons ?indexes (ICons ?src (INil))))))
                 (= ?nocopy (Op (KernelScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os ?dt)
                     (ICons ?cb (ICons ?indexes (ICons ?src (INil)))))))
                ((delete (Op (KernelScatter ?ds ?dst ?is ?istr ?ss ?os ?dt)
                     (ICons ?dest (ICons ?indexes (ICons ?src (INil)))))))
                :ruleset post_cleanup
                :name \"scatter-no-copy-dominates-valid-consumed-buffer\"
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
            (n_src.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
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
                :ruleset kernel_lower
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
#define LOG2E 1.4426950408889634f

extern \"C\" {{
    // Online normalizer calculation for softmax (Milakov & Gimelshein 2018).

    // Merge two partial (max, sum) pairs using the online softmax rule.
    __device__ __forceinline__ void merge_md(float *m, float *d, float m2, float d2) {{
        float new_m = fmaxf(*m, m2);
        *d = *d * exp2f((*m - new_m) * LOG2E) + d2 * exp2f((m2 - new_m) * LOG2E);
        *m = new_m;
    }}

    __global__ void fused_softmax(float *out, const float *inp{dyn_dims_param}) {{
        __shared__ float sh_m[THREADS_PER_BLOCK / WARP_SIZE];
        __shared__ float sh_d[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_base = {in_idx};
        long long out_base = {out_idx};
        long long N = {reduce_dim_expr};
        long long in_stride = {in_reduce_stride};
        long long out_stride = {out_reduce_stride};

        // Pass 1: one read of inp produces (global_max, global_sum).
        float m = NEG_INF_F, d = 0.0f;
        for (long long i = tid; i < N; i += THREADS_PER_BLOCK) {{
            merge_md(&m, &d, inp[in_base + i * in_stride], 1.0f);
        }}
        // Warp reduce: collapse 32 threads within each warp down to lane 0.
        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            merge_md(&m, &d, __shfl_down_sync(FULL_MASK, m, s), __shfl_down_sync(FULL_MASK, d, s));
        }}
        if (lane_id == 0) {{ sh_m[warp_id] = m; sh_d[warp_id] = d; }}
        __syncthreads();
        // Block reduce: warp 0 collapses the 8 warp results down to one.
        if (warp_id == 0) {{
            m = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? sh_m[tid] : NEG_INF_F;
            d = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? sh_d[tid] : 0.0f;
            #pragma unroll
            for (int s = (THREADS_PER_BLOCK / WARP_SIZE) / 2; s > 0; s /= 2) {{
                merge_md(&m, &d, __shfl_down_sync(FULL_MASK, m, s), __shfl_down_sync(FULL_MASK, d, s));
            }}
            sh_m[0] = m;
            sh_d[0] = d;
        }}
        __syncthreads();
        float global_max = sh_m[0];
        float inv_sum = 1.0f / sh_d[0];

        // Pass 2: write final softmax values.
        for (long long i = tid; i < N; i += THREADS_PER_BLOCK) {{
            out[out_base + i * out_stride] = exp2f((inp[in_base + i * in_stride] - global_max) * LOG2E) * inv_sum;
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
