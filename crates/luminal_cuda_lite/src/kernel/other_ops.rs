use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
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

pub type Ops = (KernelMeanReduce, KernelScatterNoCopy);

// Pure storage aliases register here so scatter's late ownership proof can
// reject destinations whose storage has uses through another logical value.
pub(crate) const SCATTER_ALIAS_DECLARATION: &str = "(relation cuda-scatter-alias (IR))";

/// Run after structural fusion. The proof deliberately requires exclusive use
/// across the entire egraph; even safe ordered prior reads fall back to copying.
/// No-copy chains remain safe because every old version is exclusively used.
/// RoPE-scatter specialization inherits this proof from its no-copy input.
pub(crate) fn scatter_reuse_late_pass() -> luminal::egglog_utils::LateEgglogPass {
    luminal::egglog_utils::LateEgglogPass::new(
        "",
        "(seq
            (saturate cuda_scatter_uses)
            cuda_scatter_reuse
            kernel_fuse_late2_rope
            (saturate dtype_prop))",
    )
}

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
        FxHashMap<Symbol, CudaSlice<u8>>,
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
    pub(crate) dest_shape: Vec<Expression>,
    pub(crate) dest_strides: Vec<Expression>,
    pub(crate) index_shape: Vec<Expression>,
    pub(crate) index_strides: Vec<Expression>,
    pub(crate) src_strides: Vec<Expression>,
    pub(crate) out_strides: Vec<Expression>,
    pub(crate) dtype: DType,
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

    fn egglog_declarations(&self) -> Vec<String> {
        vec![
            SCATTER_ALIAS_DECLARATION.to_string(),
            include_str!("scatter_reuse.egg").to_string(),
        ]
    }

    fn n_inputs(&self) -> usize {
        3
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let all_vars: FxHashSet<Symbol> = self
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

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        for expression in self
            .dest_shape
            .iter()
            .chain(&self.dest_strides)
            .chain(&self.index_shape)
            .chain(&self.index_strides)
            .chain(&self.src_strides)
            .chain(&self.out_strides)
        {
            expression.collect_dyn_vars_into(vars);
        }
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F64 | DType::I64 => 8,
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
            DType::F64 | DType::I64 => 8,
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
            DType::F64 | DType::I64 => 8,
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

    fn mutates_aliased_input(&self) -> bool {
        true
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "ScatterNoCopy"
    }
}
