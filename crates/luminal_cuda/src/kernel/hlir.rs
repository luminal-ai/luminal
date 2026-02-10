use std::sync::Arc;

use crate::{cuda_dtype, kernel::KernelOp};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::{CompileOptions, compile_ptx},
};
use itertools::Itertools;
use luminal::{
    egglog_utils::{extract_dtype, extract_expr, extract_expr_list},
    op::OpParam::*,
    op::*,
    prelude::*,
};

pub type Ops = (
    KernelAdd,
    KernelMul,
    KernelMod,
    KernelLessThan,
    KernelIota,
    KernelGather,
    KernelSumReduce,
    KernelMaxReduce,
    KernelExp2,
    KernelLog2,
    KernelSin,
    KernelRecip,
    KernelSqrt,
    KernelConstant,
    KernelCast,
);

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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMax".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Max ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelMax ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride ?dty))
    )
    :name \"kernel max reduce\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            }) as Box<dyn KernelOp>),
            vec![children[2]],
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
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "
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
        long long iter_stride = {iter_stride};

        {dtype} max_value = NEG_INF_F;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            max_value = fmaxf(max_value, in[in_start + i * iter_stride]);
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
            in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self.iter_stride.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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

    fn bytes_loaded(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelSum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Sum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelSum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride ?dty))
    )
    :name \"kernel sum reduce\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            }) as Box<dyn KernelOp>),
            vec![children[2]],
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
        let n_outputs: Expression = self.out_shape.iter().copied().product();
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
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = blockIdx.x;

        long long in_start = {in_index};
        long long iters = {iters};
        long long iter_stride = {iter_stride};

        {dtype} sum = 0;
        for (long long i = 0; i < iters; i++) {{
            sum += in[in_start + i * iter_stride];
        }}

        out[{out_index}] = sum;
    }}
}}",
            dtype = dtype,
            in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self.iter_stride.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_sum_k").unwrap();
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

    fn bytes_loaded(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
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
    b_dtype: DType,
}

impl EgglogOp for KernelAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Add ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
        (= ?b_dty (dtype ?inp_b))
    )
    (
        (union ?a (KernelAdd ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty ?b_dty))
    )
    :name \"kernel add\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
                b_dtype: extract_dtype(egraph, children[7]),
            })),
            vec![children[1], children[3]],
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
        let b_dtype = cuda_dtype(self.b_dtype);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        // Add dyn_dims parameter if we have dynamic dimensions
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void add_k({dtype} *C, const {dtype} *A, const {b_dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] + ({dtype})B[{}];
    }}
}}",
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(), // No per-module constants needed
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
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
    b_dtype: DType,
}

impl EgglogOp for KernelMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Mul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
        (= ?b_dty (dtype ?inp_b))
    )
    (
        (union ?a (KernelMul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty ?b_dty))
    )
    :name \"kernel mul\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
                b_dtype: extract_dtype(egraph, children[7]),
            })),
            vec![children[1], children[3]],
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
        let b_dtype = cuda_dtype(self.b_dtype);
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
    __global__ void mul_k({dtype} *C, const {dtype} *A, const {b_dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] * ({dtype})B[{}];
    }}
}}",
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel(),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Gather ?indexes ?out_shape ?index_strides ?data ?data_shape ?data_strides))
        (= ?dty (dtype ?data))
    )
    (
        (let ?out_strides (RowMajor ?out_shape))
        (union ?a (KernelGather ?out_shape ?indexes ?index_strides ?data ?data_shape ?data_strides ?out_strides ?dty))
    )
    :name \"kernel gather\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                data_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[6], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[7]),
            })),
            vec![children[1], children[3]],
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
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        {dtype}* out = C + {};
        const_z = indexes[{}];
        *out = data[{}];
    }}
}}",
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.index_stride).to_kernel(),
            flatten_mul_strides(&self.data_shape, &self.data_stride).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.out_shape.iter().copied().product(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Gather"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for KernelIota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelIota".to_string(), vec![Expr, Expr])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Iota ?expr ?range))
    )
    (
        (let ?kernel_iota (KernelIota ?expr ?range))
        (union ?a ?kernel_iota)
        (set (dtype ?kernel_iota) (Int))
    )
    :name \"kernel iota\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                expr: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
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
            let ptx = compile_ptx(&kernel).unwrap();
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

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        0.into()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelExp2".to_string(),
            vec![EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Exp2 ?shape ?inp ?in_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelExp2 ?shape ?inp ?in_strides ?out_strides ?dty))
    )
    :name \"kernel exp2\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
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
    __global__ void exp2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[{}] = exp2f(in[{}]);
    }}
}}",
            flatten_mul_strides(&self.shape, &self.out_strides).to_kernel(),
            flatten_mul_strides(&self.shape, &self.in_strides).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelLog2".to_string(),
            vec![EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Log2 ?shape ?inp ?in_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelLog2 ?shape ?inp ?in_strides ?out_strides ?dty))
    )
    :name \"kernel log2\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
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
    __global__ void log2_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[{}] = log2f(in[{}]);
    }}
}}",
            flatten_mul_strides(&self.shape, &self.out_strides).to_kernel(),
            flatten_mul_strides(&self.shape, &self.in_strides).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelSin".to_string(),
            vec![EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Sin ?shape ?inp ?in_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelSin ?shape ?inp ?in_strides ?out_strides ?dty))
    )
    :name \"kernel sin\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
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
    __global__ void sin_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[{}] = sinf(in[{}]);
    }}
}}",
            flatten_mul_strides(&self.shape, &self.out_strides).to_kernel(),
            flatten_mul_strides(&self.shape, &self.in_strides).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelRecip".to_string(),
            vec![EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Recip ?shape ?inp ?in_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelRecip ?shape ?inp ?in_strides ?out_strides ?dty))
    )
    :name \"kernel recip\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
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
    __global__ void recip_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[{}] = 1.0f / in[{}];
    }}
}}",
            flatten_mul_strides(&self.shape, &self.out_strides).to_kernel(),
            flatten_mul_strides(&self.shape, &self.in_strides).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelSqrt".to_string(),
            vec![EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Sqrt ?shape ?inp ?in_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?a (KernelSqrt ?shape ?inp ?in_strides ?out_strides ?dty))
    )
    :name \"kernel sqrt\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, children[4]),
            })),
            vec![children[1]],
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
    __global__ void sqrt_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[{}] = sqrtf(in[{}]);
    }}
}}",
            flatten_mul_strides(&self.shape, &self.out_strides).to_kernel(),
            flatten_mul_strides(&self.shape, &self.in_strides).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "Sqrt"
    }
}

// =============================================================================
// Binary Operations: Mod, LessThan
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelMod {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelMod {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMod".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Mod ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (union ?a (KernelMod ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty))
    )
    :name \"kernel mod\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            })),
            vec![children[1], children[3]],
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
    __global__ void mod_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = fmodf(A[{}], B[{}]);
    }}
}}",
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
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
    b_dtype: DType,
}

impl EgglogOp for KernelLessThan {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelLessThan".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (LessThan ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
        (= ?b_dty (dtype ?inp_b))
    )
    (
        (union ?a (KernelLessThan ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty ?b_dty))
    )
    :name \"kernel less_than\"
)"
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
                b_dtype: extract_dtype(egraph, children[7]),
            })),
            vec![children[1], children[3]],
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
        let b_dtype = cuda_dtype(self.b_dtype);
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
    __global__ void less_than_k(unsigned char *C, const {dtype} *A, const {b_dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] < ({dtype})B[{}] ? 1 : 0;
    }}
}}",
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "LessThan"
    }
}

// =============================================================================
// Special Operations: Constant, Cast
// =============================================================================

#[derive(Default, Debug, Clone)]
pub struct KernelConstant {
    value: f32,
}

impl EgglogOp for KernelConstant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelConstant".to_string(), vec![Float])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Constant ?value))
    )
    (
        (let ?kernel_const (KernelConstant ?value))
        (union ?a ?kernel_const)
        (set (dtype ?kernel_const) (F32))
    )
    :name \"kernel constant\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
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
        let kernel = format!(
            "
extern \"C\" {{
    __global__ void constant_k(float *out) {{
        out[0] = {:.10}f;
    }}
}}",
            self.value
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        4.into()
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
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelCast".to_string(), vec![Input, Expr, Dty, Dty])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "
(rule
    (
        (= ?a (Cast ?inp ?size ?out_dty))
        (= ?in_dty (dtype ?inp))
    )
    (
        (union ?a (KernelCast ?inp ?size ?in_dty ?out_dty))
    )
    :name \"kernel cast\"
)"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                size: extract_expr(egraph, children[1], expr_cache).unwrap_or_default(),
                in_dtype: extract_dtype(egraph, children[2]),
                out_dtype: extract_dtype(egraph, children[3]),
            })),
            vec![children[0]],
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
        let in_dtype = cuda_dtype(self.in_dtype);
        let out_dtype = cuda_dtype(self.out_dtype);

        let kernel = format!(
            "
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        out[const_z] = ({out_dtype})in[const_z];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
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

    fn bytes_loaded(&self) -> Expression {
        self.size * self.in_dtype.sizeof()
    }

    fn bytes_stored(&self) -> Expression {
        self.size * self.out_dtype.sizeof()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Cast"
    }
}

/// Generate #define macros for dynamic dimensions that read from a shared dyn_dims buffer.
/// The buffer layout is alphabetically sorted by dim char for consistency.
/// Returns (defines_string, sorted_dims) where sorted_dims gives the order of dims in the buffer.
pub fn generate_dyn_dims_defines(vars: &FxHashSet<char>) -> (String, Vec<char>) {
    if vars.is_empty() {
        return (String::new(), Vec::new());
    }
    let mut sorted_dims: Vec<char> = vars.iter().copied().collect();
    sorted_dims.sort();
    let defines = sorted_dims
        .iter()
        .enumerate()
        .map(|(idx, dim)| format!("#define const_{dim} (dyn_dims + {idx})"))
        .collect::<Vec<_>>()
        .join("\n");
    (defines, sorted_dims)
}

/// Get the offset for a dynamic dimension in the shared dyn_dims buffer.
/// Returns None if the dim is not in the set.
pub fn get_dyn_dim_offset(dim: char, sorted_dims: &[char]) -> Option<usize> {
    sorted_dims.iter().position(|&d| d == dim)
}
