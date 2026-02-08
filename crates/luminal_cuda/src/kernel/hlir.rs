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
    ComplexElementwiseOp
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
        true // TODO: this is very bad! its here so we dont have the Mul -> SumReduce pattern which can exceed memory but we should analytically eliminate that from search sapce!
        // a slightly better way than this will be to subsume in the cublas match rule
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

// Either an input buffer or a temp from a previous step in ComplexElementwiseOps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ref {
    Input(usize),
    Temp(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElementwiseKind {
    Add,
    Mul,
}

impl ElementwiseKind {
    pub fn to_step_code(self) -> u32 {
        match self {
            ElementwiseKind::Add => 0,
            ElementwiseKind::Mul => 1,
        }
    }

    pub fn from_step_code(n: u32) -> Option<ElementwiseKind> {
        match n {
            0 => Some(ElementwiseKind::Add),
            1 => Some(ElementwiseKind::Mul),
            _ => None,
        }
    }
}

/// ElementwiseStep = One elementwise operation in a fused chain (kind + refs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ElementwiseStep {
    pub kind: ElementwiseKind,
    pub lhs: Ref,
    pub rhs: Ref,
}

#[derive(Default, Debug, Clone)]
pub struct ComplexElementwiseOp {
    out_shape: Vec<Expression>,
    input_strides: Vec<Vec<Expression>>,
    out_stride: Vec<Expression>,
    steps: Vec<ElementwiseStep>,
    /// One pair of stride vectors `(inner_out_strides, outer_in_strides)` per fusion boundary.
    /// boundary_remaps[k] sits between step k and step k+1 to remap the stirdes when fusing two elementwise ops.
    boundary_remaps: Vec<(Vec<Expression>, Vec<Expression>)>,
    dtype: DType,
}

impl EgglogOp for ComplexElementwiseOp {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "ComplexElementwiseOp".to_string(),
            // out_shape, inputs (IList), all_strides_flat, out_strides, steps, remaps, dtype
            vec![EList, IList, EList, EList, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        // The biggest assumption for fusion of elementwise ops is currently the linear chain assumption
        // This means that input for the elementwise_op{i+1} in the chain is the temp{i} (output of elementwise_op{i})
        // and the input{i+1}. Currently, this is done for simplicity reasons.
        // TODO: This should be generalized to support more general patterns (e.g. inputs are both temps)
        let add_rule = r#"
(rule
    (
        (= ?a (Add ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (let ?inputs (ICons ?inp_a (ICons ?inp_b (INil))))
        (let ?all_strides (Concat ?inp_a_strides ?inp_b_strides))
        (union ?a (ComplexElementwiseOp ?out_shape ?inputs ?all_strides ?out_strides (ECons (MNum 0) (ENil)) (ENil) ?dty))
    )
    :name "elementwise add to ComplexElementwiseOp"
)"#;
        let mul_rule = r#"
(rule
    (
        (= ?a (Mul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (let ?inputs (ICons ?inp_a (ICons ?inp_b (INil))))
        (let ?all_strides (Concat ?inp_a_strides ?inp_b_strides))
        (union ?a (ComplexElementwiseOp ?out_shape ?inputs ?all_strides ?out_strides (ECons (MNum 1) (ENil)) (ENil) ?dty))
    )
    :name "elementwise mul to ComplexElementwiseOp"
)"#;
        let fusion_rule = r#"
(rule
    (
        (= ?outer (ComplexElementwiseOp ?shape (ICons ?inner ?outer_rest) ?outer_all_strides ?outer_out_strides ?outer_steps ?outer_remaps ?dty))
        (= ?inner (ComplexElementwiseOp ?shape ?inner_inputs ?inner_strides ?inner_out_strides ?inner_steps ?inner_remaps ?dty))
        (= ?ndim (len ?shape))
    )
    (
        (let ?fused_inputs (IConcat ?inner_inputs ?outer_rest))
        (let ?outer_in_strides (Take ?outer_all_strides ?ndim))
        (let ?boundary_remap (Concat ?inner_out_strides ?outer_in_strides))
        (let ?outer_strides_tail (Drop ?outer_all_strides ?ndim))
        (let ?fused_strides (Concat ?inner_strides ?outer_strides_tail))
        (let ?fused_steps (Concat ?inner_steps ?outer_steps))
        (let ?fused_remaps (Concat ?inner_remaps (Concat ?boundary_remap ?outer_remaps)))
        (union ?outer (ComplexElementwiseOp ?shape ?fused_inputs ?fused_strides ?outer_out_strides ?fused_steps ?fused_remaps ?dty))
    )
    :name "fuse two ComplexElementwiseOp"
)"#;
        vec![
            add_rule.to_string(),
            mul_rule.to_string(),
            fusion_rule.to_string(),
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
        // children: [0]=out_shape, [1]=inputs(IList), [2]=all_strides_flat, [3]=out_strides, [4]=steps, [5]=remaps, [6]=dtype
        let out_shape = extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap();
        let ndim = out_shape.len();
        // Skip children[1] — IList sources are resolved generically by graph.rs
        let all_strides = extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap();
        let input_strides: Vec<Vec<Expression>> =
            all_strides.chunks(ndim).map(|c| c.to_vec()).collect();
        let out_stride = extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap();
        let steps_list = extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap();
        let remaps_flat = extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap();
        let dtype = extract_dtype(egraph, children[6]);

        let boundary_remaps: Vec<(Vec<Expression>, Vec<Expression>)> = if remaps_flat.is_empty() {
            vec![]
        } else {
            remaps_flat
                .chunks(2 * ndim)
                .map(|c| (c[..ndim].to_vec(), c[ndim..].to_vec()))
                .collect()
        };

        let steps: Vec<ElementwiseStep> = steps_list
            .into_iter()
            .filter_map(|e| {
                e.as_num()
                    .and_then(|n| ElementwiseKind::from_step_code(n as u32))
            })
            .enumerate()
            .map(|(i, kind)| ElementwiseStep {
                kind,
                lhs: if i == 0 {
                    Ref::Input(0)
                } else {
                    Ref::Temp(i - 1)
                },
                rhs: Ref::Input(i + 1),
            })
            .collect();
        assert_eq!(
            input_strides.len(),
            steps.len() + 1,
            "ComplexElementwiseOp linear chain: #inputs must equal #steps + 1"
        );

        (
            LLIROp::new::<dyn KernelOp>(Box::new(ComplexElementwiseOp {
                out_shape,
                input_strides,
                out_stride,
                steps,
                boundary_remaps,
                dtype,
            }) as Box<dyn KernelOp>),
            vec![], // sources come from IList walking in graph.rs
        )
    }
}

impl KernelOp for ComplexElementwiseOp {
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
            .chain(
                self.input_strides
                    .iter()
                    .flat_map(|s| s.iter().flat_map(|e| e.dyn_vars())),
            )
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.boundary_remaps.iter().flat_map(|(inner_out, outer_in)| {
                inner_out
                    .iter()
                    .chain(outer_in.iter())
                    .flat_map(|e| e.dyn_vars())
            }))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        // Compute z values for each step.
        // For N steps (0..N-1), there are N-1 boundaries.
        // z_values[N-1] = z (outermost), z_values[k] = remap(z_values[k+1]) via boundary k.
        // Input(0), Input(1) -> step 0 -> z_values[0]
        // Input(k+1) for k>=1 -> step k -> z_values[k]
        let n_steps = self.steps.len();
        let n_boundaries = self.boundary_remaps.len();
        assert!(
            n_boundaries == 0 || n_boundaries == n_steps - 1,
            "Expected 0 or n_steps-1 boundary remaps, got {n_boundaries} for {n_steps} steps"
        );

        let z_per_step: Vec<Expression> = if n_boundaries == 0 {
            vec![Expression::from('z'); n_steps]
        } else {
            let mut z_vals = vec![Expression::from(0); n_steps];
            z_vals[n_steps - 1] = Expression::from('z');
            let rm = row_major_strides(&self.out_shape);
            for k in (0..n_boundaries).rev() {
                // boundary k sits between step k and step k+1
                let (ref inner_out, ref outer_in) = self.boundary_remaps[k];
                let z_outer = z_vals[k + 1];

                let z_inner = if inner_out == outer_in { // no remapping if strides are equal
                    z_outer
                } else if *inner_out == rm { // if inner_out is row-major, then its offset == z_inner
                    let m_template = flatten_mul_strides(&self.out_shape, outer_in);
                    m_template.substitute('z', z_outer).simplify()
                } else { // in general, we do z_outer -> offset -> z_inner
                    let m_template = flatten_mul_strides(&self.out_shape, outer_in);
                    let m = m_template.substitute('z', z_outer).simplify();
                    let inv_template =
                        inverse_flatten_mul_strides(&self.out_shape, inner_out);
                    inv_template.substitute('z', m).simplify()
                };
                z_vals[k] = z_inner;
            }
            z_vals
        };

        // Inputs 0 and 1 -> step 0, Input k+1 -> step k (for k >= 1)
        let input_z: Vec<Expression> = (0..self.input_strides.len())
            .map(|i| {
                let step_idx = i.saturating_sub(1);
                z_per_step[step_idx]
            })
            .collect();

        let out_idx = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel();
        let input_idxs: Vec<String> = self
            .input_strides
            .iter()
            .enumerate()
            .map(|(i, strides)| {
                let idx_template = flatten_mul_strides(&self.out_shape, strides);
                idx_template
                    .substitute('z', input_z[i])
                    .simplify()
                    .to_kernel()
            })
            .collect();

        let ref_to_expr = |r: Ref| -> String {
            match r {
                Ref::Input(i) => format!("in{}[{}]", i, input_idxs[i]),
                Ref::Temp(i) => format!("t{i}"),
            }
        };

        let scalar_ty = cuda_dtype(self.dtype);
        let body: String = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, step)| {
                let lhs_expr = ref_to_expr(step.lhs);
                let rhs_expr = ref_to_expr(step.rhs);
                let expr = match step.kind {
                    ElementwiseKind::Add => format!("{} + {}", lhs_expr, rhs_expr),
                    ElementwiseKind::Mul => format!("{} * {}", lhs_expr, rhs_expr),
                };
                if i == self.steps.len() - 1 {
                    format!("out[{}] = {};", out_idx, expr)
                } else {
                    format!("{} t{} = {};", scalar_ty, i, expr)
                }
            })
            .join("\n        ");

        let input_params: String = (0..self.input_strides.len())
            .map(|i| format!(", const {dtype} *in{i}"))
            .collect::<String>();

        let kernel = format!(
            "
{constants}
extern \"C\" {{
    __global__ void complex_elementwise_k({dtype} *out{input_params}) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        {body}
    }}
}}",
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_ptx(&kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("complex_elementwise_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4 * self.input_strides.len() as i32
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn kernel_name(&self) -> &'static str {
        "ComplexElementwise"
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
