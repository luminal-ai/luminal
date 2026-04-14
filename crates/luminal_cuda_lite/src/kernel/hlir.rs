use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{CudaFunctionExt, KernelOp},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, app, eq, rule, set, sort, union, v},
        base::{DTYPE, ELIST, EXPRESSION, F64, OP_KIND, SORTS, dtype, ilist, op_term},
        extract_dtype, extract_expr, extract_expr_list,
    },
    hlir::{Add, Exp2, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Scatter, Sin, Sqrt, SumReduce},
    op::*,
    prelude::*,
};

/// Generates CUDA include directives based on the dtypes used in a kernel
pub fn dtype_includes(dtypes: &[DType]) -> String {
    let needs_fp16 = dtypes.iter().any(|d| matches!(d, DType::F16));
    let needs_bf16 = dtypes.iter().any(|d| matches!(d, DType::Bf16));
    let needs_fp8 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0));
    let needs_fp6 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F6E2M3 | DType::F6E3M2));
    let needs_fp4 = dtypes.iter().any(|d| matches!(d, DType::F4E2M1));
    let mut s = String::new();
    if needs_fp16 {
        s.push_str("#include <cuda_fp16.h>\n");
    }
    if needs_bf16 {
        s.push_str("#include <cuda_bf16.h>\n");
    }
    if needs_fp8 {
        s.push_str("#include <cuda_fp8.h>\n");
    }
    if needs_fp6 {
        s.push_str("#include <cuda_fp6.h>\n");
    }
    if needs_fp4 {
        s.push_str("#include <cuda_fp4.h>\n");
    }
    s
}

pub type Ops = (
    KernelAdd,
    KernelMul,
    KernelMod,
    KernelLessThan,
    KernelIota,
    KernelGather,
    KernelScatter,
    KernelSumReduce,
    KernelMaxReduce,
    KernelExp2,
    KernelLog2,
    KernelSin,
    KernelRecip,
    KernelSqrt,
    KernelConstant,
    KernelCast,
    KernelEmbed,
);

/// Build a rewrite that matches an HLIR op, reads dtype(s) from the given source fields,
/// and unions with a kernel op that has the same fields plus the dtype(s) appended.
pub fn kernel_rewrite<H: Default + EgglogOp, L: Default + EgglogOp>() -> Rule {
    let hlir = H::default().sort();
    let llir = L::default().sort();
    let (mut args, hlir_kind_term) = hlir.new_call();
    let inputs = v("?__inputs");
    let hlir_op = op_term(hlir_kind_term, inputs.clone());
    let dt = v("?__dt");
    args.add("dtype", dt.clone());
    let llir_kind_term = llir.call(&args);
    let llir_op = op_term(llir_kind_term, inputs);
    rule(union(hlir_op.clone(), llir_op)).fact(eq(dt, dtype(hlir_op)))
}

#[derive(Default, Debug, Clone)]

pub struct KernelMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMaxReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMax",
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
        vec![kernel_rewrite::<MaxReduce, Self>()]
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
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelMaxReduce {
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

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        __shared__ {dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {dtype} max_value = NEG_INF_F;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            max_value = fmaxf(max_value, in[in_start + {iter_stride_of_i}]);
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            max_value = fmaxf(max_value, __shfl_down_sync(FULL_MASK, max_value, s));
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = max_value;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {dtype} block_max = tid < cnt ? warp_sums[tid] : NEG_INF_F;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_max = fmaxf(block_max, __shfl_down_sync(FULL_MASK, block_max, s));
            }}

            if (tid == 0) {{
                out[{out_index}] = block_max;
            }}
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride_of_i = iter_stride_of_i,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_max_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
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
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "MaxReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelSumReduce {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSum",
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
        vec![kernel_rewrite::<SumReduce, Self>()]
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

impl KernelOp for KernelSumReduce {
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

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in_data{dyn_dims_param}) {{
        __shared__ {dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {dtype} partial = 0;
        {dtype} comp = 0;   // Kahan compensation
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            {dtype} y = in_data[in_start + {iter_stride_of_i}] - comp;
            {dtype} t = partial + y;
            comp = (t - partial) - y;
            partial = t;
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
            {dtype} block_sum = tid < cnt ? warp_sums[tid] : ({dtype})0;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = block_sum;
            }}
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride_of_i = iter_stride_of_i,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks (warp-parallel)
            32.into(),                                      // shmem for warp_sums
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
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "SumReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelAdd {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelAdd {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelAdd",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Add, Self>()]
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
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelAdd {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        // Add dyn_dims parameter if we have dynamic dimensions
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void add_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] + B[{b_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("add_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        // Return empty constants map - we now use shared dyn_dims buffer
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(), // No per-module constants needed
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Add"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelMul {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMul {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMul",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Mul, Self>()]
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
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelMul {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mul_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] * B[{b_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mul_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
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
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Mul"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGather {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelGather",
            &[
                ("out_shape", ELIST),
                ("index_strides", ELIST),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Gather (now in Op format) and rewrite to KernelGather
        let hlir_gather = luminal::hlir::Gather::default().sort();
        let (gather_args, gather_kind_term) = hlir_gather.new_call();
        // HLIR Gather inputs: [indexes, data] (n_inputs=2)
        let indexes = v("?__indexes");
        let data = v("?__data");
        let gather_inputs = ilist(vec![indexes.clone(), data.clone()]);
        let gather_op = op_term(gather_kind_term, gather_inputs);

        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), gather_args["index_shape"].clone()));
        let dt = v("?__dt");
        let kernel_kind_args = [
            ("out_shape".to_string(), gather_args["index_shape"].clone()),
            (
                "index_strides".to_string(),
                gather_args["index_strides"].clone(),
            ),
            ("data_shape".to_string(), gather_args["data_shape"].clone()),
            (
                "data_strides".to_string(),
                gather_args["data_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        let kernel_kind_term = self.sort().call(kernel_kind_args);
        let kernel_op = op_term(kernel_kind_term, ilist(vec![indexes, data.clone()]));
        vec![rule(union(gather_op, kernel_op)).fact(eq(dt, dtype(data)))]
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
                index_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[5]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelGather {
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
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
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
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let idx_idx = flatten_strides(&self.out_shape, &self.index_stride).to_kernel();
        let data_idx = flatten_strides(&self.data_shape, &self.data_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        {dtype}* out = C + {out_idx};
        const_z = indexes[{idx_idx}];
        *out = data[{data_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Data + indices (indices are always int32)
        (self.output_size() * self.dtype.bits()).ceil_div(8) + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Gather"
    }
}

// KernelScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Two-phase: memcpy graph node copies dest→output, then scatter kernel runs in same CUDA graph.
#[derive(Debug, Clone)]
pub struct KernelScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl Default for KernelScatter {
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

impl EgglogOp for KernelScatter {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelScatter",
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

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Scatter (now in Op format) and rewrite to KernelScatter
        let hlir_scatter = luminal::hlir::Scatter::default().sort();
        let (scatter_args, scatter_kind_term) = hlir_scatter.new_call();
        // HLIR Scatter inputs: [dest, indexes, src] (n_inputs=3)
        let dest = v("?__dest");
        let indexes = v("?__indexes");
        let src = v("?__src");
        let scatter_inputs = ilist(vec![dest.clone(), indexes.clone(), src.clone()]);
        let scatter_op = op_term(scatter_kind_term, scatter_inputs);

        let out_strides = SORTS
            .row_major
            .call(("list".to_string(), scatter_args["dest_shape"].clone()));
        let dt = v("?__dt");
        let kernel_kind_args = [
            ("dest_shape".to_string(), scatter_args["dest_shape"].clone()),
            (
                "dest_strides".to_string(),
                scatter_args["dest_strides"].clone(),
            ),
            (
                "index_shape".to_string(),
                scatter_args["index_shape"].clone(),
            ),
            (
                "index_strides".to_string(),
                scatter_args["index_strides"].clone(),
            ),
            (
                "src_strides".to_string(),
                scatter_args["src_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
            ("dtype".to_string(), dt.clone()),
        ];
        let kernel_kind_term = self.sort().call(kernel_kind_args);
        let kernel_op = op_term(kernel_kind_term, ilist(vec![dest, indexes, src.clone()]));
        vec![rule(union(scatter_op, kernel_op)).fact(eq(dt, dtype(src)))]
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

impl KernelOp for KernelScatter {
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

        // Single-kernel scatter: copy dest→output then scatter src→output[indexes]
        // Launched as 1 block of 1024 threads with __syncthreads() barrier.
        // Uses float4 vectorized copy (4x throughput) for the copy phase.
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
    __global__ void scatter(
        {dtype} *out, const {dtype} *dest, const int *indexes, const {dtype} *src{dyn_dims_param}
    ) {{
        int tid = threadIdx.x;
        long long n_dest = {n_dest_elements};
        long long n_src = {n_src_elements};
        // Phase 1: vectorized copy dest → output (float4 = 4 elements per op)
        long long n_vec = n_dest / 4;
        float4 *out4 = (float4 *)out;
        const float4 *dest4 = (const float4 *)dest;
        for (long long i = tid; i < n_vec; i += blockDim.x) {{
            out4[i] = dest4[i];
        }}
        // Handle remaining elements
        long long remainder_start = n_vec * 4;
        for (long long i = remainder_start + tid; i < n_dest; i += blockDim.x) {{
            out[i] = dest[i];
        }}
        __syncthreads();
        // Phase 2: scatter src → output[indexes[i]]
        for (long long const_z = tid; const_z < n_src; const_z += blockDim.x) {{
            int idx = indexes[{scatter_idx_idx}];
            if (idx >= 0 && idx < n_dest) {{
                out[idx] = src[{scatter_src_idx}];
            }}
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
            let func = module.load_function("scatter").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            scatter_kernel,
            (1.into(), 1.into(), 1.into()),    // grid: 1 block
            (1024.into(), 1.into(), 1.into()), // block: 1024 threads
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
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // params: (out, dest, indexes, src [, dyn_dims])
        // input_ptrs: [dest, indexes, src]
        let mut params = vec![output_ptr, input_ptrs[0], input_ptrs[1], input_ptrs[2]];
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
        // dest (copy) + indices + src
        self.output_size() * data_elem_size + n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_data_input(&self) -> Option<usize> {
        Some(0) // output is derived from dest (input 0): copy dest→output then scatter
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Scatter"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for KernelIota {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn n_inputs(&self) -> usize {
        0
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_iota_kind) = luminal::hlir::Iota::default().sort().new_call();
        let hlir_inputs = v("?__inputs");
        let hlir_op = op_term(hlir_iota_kind, hlir_inputs.clone());
        let kernel_kind = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind, hlir_inputs);
        vec![
            rule(union(hlir_op, kernel_op.clone()))
                .set(dtype(kernel_op), app(&SORTS.int_dt, vec![])),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                expr: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelIota {
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
        let vars = self.expr.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void iota_k(int *C{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[const_z] = {};
    }}
}}",
            self.expr.to_kernel(),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("iota_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.range, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn output_bytes(&self) -> Expression {
        // Iota always outputs int32 (4 bytes)
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn kernel_name(&self) -> &'static str {
        "Iota"
    }
}

// =============================================================================
// Unary Operations: Exp2, Log2, Sin, Recip, Sqrt
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelExp2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelExp2 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelExp2",
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
        vec![kernel_rewrite::<Exp2, Self>()]
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

impl KernelOp for KernelExp2 {
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
    __global__ void exp2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = exp2f(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("exp2_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
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
        "Exp2"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLog2 {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelLog2 {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelLog2",
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
        vec![kernel_rewrite::<Log2, Self>()]
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

impl KernelOp for KernelLog2 {
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
    __global__ void log2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = log2f(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("log2_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
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
        "Log2"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSin {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelSin {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSin",
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
        vec![kernel_rewrite::<Sin, Self>()]
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

impl KernelOp for KernelSin {
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
    __global__ void sin_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = sinf(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("sin_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
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
        "Sin"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelRecip {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelRecip {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRecip",
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
        vec![kernel_rewrite::<Recip, Self>()]
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

impl KernelOp for KernelRecip {
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
    __global__ void recip_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = 1.0f / in[{in_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("recip_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
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
        "Recip"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSqrt {
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelSqrt {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelSqrt",
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
        vec![kernel_rewrite::<Sqrt, Self>()]
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

impl KernelOp for KernelSqrt {
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
    __global__ void sqrt_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        out[{out_idx}] = sqrtf(in[{in_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("sqrt_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
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
        "Sqrt"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelMod {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMod {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelMod",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![kernel_rewrite::<Mod, Self>()]
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
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelMod {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
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
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mod_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = fmodf(A[{a_idx}], B[{b_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mod_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
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
        // Both inputs have same dtype
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Mod"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLessThan {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelLessThan {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelLessThan",
            &[
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        let hlir = LessThan::default().sort();
        let (mut args, hlir_kind_term) = hlir.new_call();
        // LessThan's dtype is Bool (output type), but the kernel needs the INPUT dtype
        // HLIR LessThan inputs: [inp_a, inp_b]
        let inp_a = v("?__inp_a");
        let inp_b = v("?__inp_b");
        let hlir_inputs = ilist(vec![inp_a.clone(), inp_b.clone()]);
        let hlir_op = op_term(hlir_kind_term, hlir_inputs.clone());
        let dt = v("?__dt");
        args.add("dtype", dt.clone());
        let kernel_kind_term = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind_term, hlir_inputs);
        vec![rule(union(hlir_op, kernel_op)).fact(eq(dt, dtype(inp_a)))]
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
                a_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelLessThan {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void less_than_k(unsigned char *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] < B[{b_idx}] ? 1 : 0;
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("less_than_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        // LessThan outputs Bool (unsigned char, 1 byte per element)
        self.output_size()
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        DType::Bool
    }

    fn kernel_name(&self) -> &'static str {
        "LessThan"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelConstant {
    value: f32,
}

impl EgglogOp for KernelConstant {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelConstant", &[("value", F64)])
    }

    fn n_inputs(&self) -> usize {
        0
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, const_kind) = luminal::hlir::Constant::default().sort().new_call();
        let hlir_inputs = v("?__inputs");
        let hlir_op = op_term(const_kind, hlir_inputs.clone());
        let kernel_kind = self.sort().call(&args);
        let kernel_op = op_term(kernel_kind, hlir_inputs);
        vec![
            rule(union(hlir_op, kernel_op.clone()))
                .set(dtype(kernel_op), app(&SORTS.f32_dt, vec![])),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                value: egraph.enodes[kind_children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl KernelOp for KernelConstant {
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
        let value_str = if self.value.is_nan() {
            "__int_as_float(0x7fc00000)".to_string()
        } else if self.value.is_infinite() {
            if self.value > 0.0 {
                "__int_as_float(0x7f800000)".to_string()
            } else {
                "__int_as_float(0xff800000)".to_string()
            }
        } else {
            format!("{:.10}f", self.value)
        };
        let kernel = format!(
            "
extern \"C\" {{
    __global__ void constant_k(float *out) {{
        out[0] = {value_str};
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("constant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (1.into(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        1.into()
    }

    fn output_bytes(&self) -> Expression {
        // Constant always outputs F32
        4.into()
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Constant"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelCast {
    size: Expression,
    in_dtype: DType,
    out_dtype: DType,
}

impl EgglogOp for KernelCast {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelCast",
            &[("size", EXPRESSION), ("dtype", DTYPE), ("src_dtype", DTYPE)],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Match HLIR Cast and rewrite to KernelCast
        let hlir_cast = luminal::hlir::Cast::default().sort();
        let (mut cast_args, cast_kind_term) = hlir_cast.new_call();
        let inp = v("?__inp");
        let cast_inputs = ilist(vec![inp.clone()]);
        let cast_op = op_term(cast_kind_term, cast_inputs.clone());

        let out_dty = cast_args.remove("dtype");
        let in_dty = v("?__in_dt");
        cast_args.add("dtype", in_dty.clone());
        cast_args.add("src_dtype", out_dty);
        let kernel_kind_term = self.sort().call(&cast_args);
        let kernel_op = op_term(kernel_kind_term, cast_inputs);
        vec![rule(union(cast_op, kernel_op)).fact(eq(in_dty, dtype(inp)))]
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
                size: extract_expr(egraph, kind_children[0], expr_cache).unwrap_or_default(),
                in_dtype: extract_dtype(egraph, kind_children[1]),
                out_dtype: extract_dtype(egraph, kind_children[2]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for KernelCast {
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
        let out_dtype = cuda_dtype(self.out_dtype);
        let includes = dtype_includes(&[self.in_dtype, self.out_dtype]);

        let kernel = if self.in_dtype.bits() < 8 {
            // Sub-byte packed types: multiple values packed per byte.
            // Extract the correct bits using bit-level addressing.
            let bits = self.in_dtype.bits();
            let in_cuda_type = cuda_dtype(self.in_dtype);
            let mask = (1u32 << bits) - 1;
            format!(
                "{includes}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const unsigned char *in_raw) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long bit_offset = idx * {bits};
        long long byte_idx = bit_offset >> 3;
        int bit_pos = (int)(bit_offset & 7);
        unsigned short raw = (unsigned short)in_raw[byte_idx];
        if (bit_pos + {bits} > 8) raw |= ((unsigned short)in_raw[byte_idx + 1]) << 8;
        {in_cuda_type} val;
        val.__x = (unsigned char)((raw >> bit_pos) & {mask}u);
        out[idx] = ({out_dtype})val;
    }}
}}"
            )
        } else {
            let in_dtype = cuda_dtype(self.in_dtype);
            format!(
                "{includes}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[const_z] = ({out_dtype})in[const_z];
    }}
}}"
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.size, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn output_bytes(&self) -> Expression {
        (self.size * self.out_dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.size * self.in_dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.out_dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Cast"
    }
}

/// Thread-local global dim ordering override. When set, `generate_dyn_dims_defines`
/// uses this ordering for buffer indices instead of the kernel's local ordering.
/// This ensures all kernels in a CudaGraphOp use consistent indices into the shared
/// dyn_dims buffer.
thread_local! {
    static GLOBAL_DYN_DIMS: std::cell::RefCell<Option<Vec<char>>> = const { std::cell::RefCell::new(None) };
}

/// Set the global dyn dims ordering for subsequent kernel compilations.
pub fn set_global_dyn_dims(dims: Vec<char>) {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = Some(dims));
}

/// Clear the global dyn dims ordering.
pub fn clear_global_dyn_dims() {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = None);
}

/// Get the current global dyn dims ordering.
pub fn get_global_dyn_dims() -> Option<Vec<char>> {
    GLOBAL_DYN_DIMS.with(|g| g.borrow().clone())
}

/// Generate #define macros for dynamic dimensions that read from a shared dyn_dims buffer.
/// The buffer layout is alphabetically sorted by dim char for consistency.
/// Returns (defines_string, sorted_dims) where sorted_dims gives the order of dims in the buffer.
///
/// When a global dyn dims ordering is set (via `set_global_dyn_dims`), indices are based
/// on the global ordering to ensure consistency across kernels sharing a dyn_dims buffer.
pub fn generate_dyn_dims_defines(vars: &FxHashSet<char>) -> (String, Vec<char>) {
    if vars.is_empty() {
        return (String::new(), Vec::new());
    }
    // Check for global ordering override
    let global = GLOBAL_DYN_DIMS.with(|g| g.borrow().clone());
    if let Some(mut global_order) = global {
        // Use global ordering for indices - each dim gets its position in the global list
        // Dynamically extend the ordering if a kernel uses a dim not in the pre-scan
        let mut extended = false;
        for dim in vars.iter().sorted() {
            if !global_order.contains(dim) {
                global_order.push(*dim);
                extended = true;
            }
        }
        if extended {
            global_order.sort();
            // Update the thread-local so subsequent kernels see the extended ordering
            set_global_dyn_dims(global_order.clone());
        }
        let defines = vars
            .iter()
            .sorted()
            .map(|dim| {
                let idx = global_order
                    .iter()
                    .position(|d| d == dim)
                    .expect("Dim must be in global ordering after extension");
                format!("#define const_{dim} dyn_dims[{idx}]")
            })
            .collect::<Vec<_>>()
            .join("\n");
        return (defines, global_order);
    }
    // Default: local ordering
    let mut sorted_dims: Vec<char> = vars.iter().copied().collect();
    sorted_dims.sort();
    let defines = sorted_dims
        .iter()
        .enumerate()
        .map(|(idx, dim)| format!("#define const_{dim} dyn_dims[{idx}]"))
        .collect::<Vec<_>>()
        .join("\n");
    (defines, sorted_dims)
}

/// Get the offset for a dynamic dimension in the shared dyn_dims buffer.
/// Returns None if the dim is not in the set.
pub fn get_dyn_dim_offset(dim: char, sorted_dims: &[char]) -> Option<usize> {
    sorted_dims.iter().position(|&d| d == dim)
}

#[derive(Default, Debug, Clone)]
pub struct KernelEmbed {
    batch_shape: Vec<Expression>,  // batch dimensions (e.g., [seq_len])
    token_stride: Vec<Expression>, // stride for token_ids input
    out_stride: Vec<Expression>,   // stride for output
    embed_dim: Expression,         // embedding dimension
}

impl EgglogOp for KernelEmbed {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelEmbed",
            &[
                ("batch_shape", ELIST),
                ("token_stride", ELIST),
                ("out_stride", ELIST),
                ("embed_dim", EXPRESSION),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // Match Gather with Add(Mul(Cast(token_ids), const), Iota) indices
            // Now uses (Op (OpKind ...) (ICons ...)) format
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= (len ?idx_shape) 2)
                    (= ?indices (Op (Add ?add_shape ?mul_stride ?iota_stride ?add_out_stride) (ICons ?mul_result (ICons ?iota_result (INil)))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_cast_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids_cast (ICons ?mul_const (INil)))))
                    (= ?token_ids_cast (Op (Cast ?cast_size ?cast_dtype) (ICons ?token_ids (INil))))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_cast_stride ?out_stride_batch ?embed_dim) (ICons ?token_ids_cast (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with cast mul\"
            )"),
            // Match Gather with Add(Iota, Mul(Cast(token_ids), const)) indices (reversed order)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= (len ?idx_shape) 2)
                    (= ?indices (Op (Add ?add_shape ?iota_stride ?mul_stride ?add_out_stride) (ICons ?iota_result (ICons ?mul_result (INil)))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_cast_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids_cast (ICons ?mul_const (INil)))))
                    (= ?token_ids_cast (Op (Cast ?cast_size ?cast_dtype) (ICons ?token_ids (INil))))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_cast_stride ?out_stride_batch ?embed_dim) (ICons ?token_ids_cast (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with cast mul reversed\"
            )"),
            // Match Gather with Add(Mul(token_ids, const), Iota) indices (no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= (len ?idx_shape) 2)
                    (= ?indices (Op (Add ?add_shape ?mul_stride ?iota_stride ?add_out_stride) (ICons ?mul_result (ICons ?iota_result (INil)))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids (ICons ?mul_const (INil)))))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_stride ?out_stride_batch ?embed_dim) (ICons ?token_ids (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with mul\"
            )"),
            // Match Gather with Add(Iota, Mul(token_ids, const)) indices (reversed order, no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Op (Gather ?idx_shape ?idx_stride ?embed_shape ?embed_stride) (ICons ?indices (ICons ?embed_table (INil)))))
                    (= (len ?idx_shape) 2)
                    (= ?indices (Op (Add ?add_shape ?iota_stride ?mul_stride ?add_out_stride) (ICons ?iota_result (ICons ?mul_result (INil)))))
                    (= ?mul_result (Op (Mul ?mul_shape ?token_stride ?mul_const_stride ?mul_out_stride) (ICons ?token_ids (ICons ?mul_const (INil)))))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?ke (Op (KernelEmbed ?batch_shape ?token_stride ?out_stride_batch ?embed_dim) (ICons ?token_ids (ICons ?embed_table (INil)))))
                    (union ?gather ?ke)
                    (set (dtype ?ke) (F32))
                )
                :name \"kernel embed with mul reversed\"
            )"),
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
                batch_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                token_stride: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                embed_dim: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
            })),
            input_enodes, // token_ids, embedding_table
        )
    }
}

impl KernelOp for KernelEmbed {
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
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        let vars = self
            .batch_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.token_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.embed_dim.dyn_vars())
            .collect::<FxHashSet<_>>();
        let token_offset_expr = flatten_strides(&self.batch_shape, &self.token_stride).to_kernel();
        let out_offset_expr = flatten_strides(&self.batch_shape, &self.out_stride).to_kernel();
        let embed_dim_expr = self.embed_dim.to_kernel();
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void embed(float *out, const int *token_ids, const float *embed_table) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        long long embed_dim = {embed_dim_expr};
        long long batch_idx = idx / embed_dim;
        long long embed_idx = idx % embed_dim;
        long long const_z = batch_idx;
        long long token_offset = {token_offset_expr};
        long long out_offset = {out_offset_expr};
        int token_id = token_ids[token_offset];
        out[out_offset + embed_idx] = embed_table[(long long)token_id * embed_dim + embed_idx];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("embed").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        let total_threads = batch_size * self.embed_dim;
        (
            func,
            module,
            kernel,
            (total_threads, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
            * self.embed_dim
    }

    fn output_bytes(&self) -> Expression {
        // Embed outputs F32
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        // Load: 1 token ID (4 bytes) per batch + 1 embedding row (embed_dim * 4 bytes) per batch
        batch_size * (4 + self.embed_dim * 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store: 1 embedding row per batch element
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // No FLOPs - just memory copy
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Embed"
    }
}
