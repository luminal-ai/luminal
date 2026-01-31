use std::sync::Arc;

use crate::{cuda_dtype, kernel::KernelOp};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::{CompileOptions, compile_ptx},
};
use itertools::Itertools;
use luminal::{
    graph::{extract_dtype, extract_expr, extract_expr_list},
    op::OpParam::*,
    op::*,
    prelude::*,
};

pub type Ops = (
    KernelAdd,
    KernelMul,
    KernelIota,
    KernelGather,
    KernelSumReduce,
    KernelMaxReduce,
    KernelMeanReduce,
    KernelArgsort,
);

#[derive(Default, Debug, Clone)]
pub struct KernelArgsort {
    descending: bool,
    shape: Vec<Expression>,
    in_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    iters: Expression,
    dtype: DType,
}
impl EgglogOp for KernelArgsort {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelArgsort".to_string(),
            // Input, descending, shape, in_strides_3d, out_strides, iters, dtype
            vec![Input, Int, EList, EList, EList, Expr, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        let inverse_perm_pattern = "
            (= ?sum_cmp (Sum ?sum_shape ?sum_iters ?cmp ?sum_in_strides ?sum_iter_stride ?sum_out_strides))
            (= ?cast (Cast ?sum_cmp (Int)))
            (= ?cand_iota (Iota ?cand_expr ?cand_size))
            (= ?pos_iota (Iota ?pos_expr ?pos_size))
            (= ?lt1 (LessThan ?lt1_shape ?cast ?cast_str1 ?cand_iota ?cand_str1 ?lt1_out))
            (= ?lt2 (LessThan ?lt2_shape ?cand_iota ?cand_str2 ?cast ?cast_str2 ?lt2_out))
            (= ?ne (Add ?ne_shape ?lt1 ?lt1_str ?lt2 ?lt2_str ?ne_out))
            (= ?neg1 (Constant -1.000000))
            (= ?neg_ne (Mul ?neg_shape ?ne ?ne_str ?neg1 ?neg1_str ?neg_out))
            (= ?one (Constant 1.000000))
            (= ?eq (Add ?eq_shape ?neg_ne ?neg_str ?one ?one_str ?eq_out))
            (= ?mul_pos (Mul ?mul_shape ?eq ?eq_str ?pos_iota ?pos_str ?mul_out))
            (= ?result (Sum ?final_shape ?final_iters ?mul_pos ?mul_strides ?mul_iter_stride ?out_strides))
            (= ?dty (dtype ?inp))
        ";

        let ascending_rule = format!("
    (rule
        (
            ; Ascending: LessThan(add_eps, input) means a.gt(b) in Rust
            (= ?add_eps (Add ?add_shape ?inp ?inp_str1 ?eps ?eps_str ?add_out))
            (= ?cmp (LessThan ?cmp_shape ?add_eps ?add_str ?inp ?inp_str2 ?cmp_out))
            {inverse_perm_pattern}
        )
        (
            (union ?result (KernelArgsort ?inp 0 ?final_shape ?inp_str2 ?out_strides ?sum_iters ?dty))
        )
        :name \"kernel argsort ascending\"
    )");

        let descending_rule = format!("
    (rule
        (
            ; Descending: LessThan(input, add_eps) means a.lt(b) in Rust
            (= ?add_eps (Add ?add_shape ?inp ?inp_str1 ?eps ?eps_str ?add_out))
            (= ?cmp (LessThan ?cmp_shape ?inp ?inp_str2 ?add_eps ?add_str ?cmp_out))
            {inverse_perm_pattern}
        )
        (
            (union ?result (KernelArgsort ?inp 1 ?final_shape ?inp_str2 ?out_strides ?sum_iters ?dty))
        )
        :name \"kernel argsort descending\"
    )");

        vec![ascending_rule, descending_rule]
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
                descending: egraph.enodes[children[1]].0.parse::<i32>().unwrap() != 0,
                shape: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, children[5], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            }) as Box<dyn KernelOp>),
            vec![children[0]],
        )
    }
}

impl KernelOp for KernelArgsort {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let sort_axis = self
            .shape
            .iter()
            .position(|&s| s == self.iters)
            .unwrap_or(self.shape.len() - 1);

        // Remove extra dim from in_strides to get the original input strides.
        let in_strides: Vec<Expression> = self
            .in_strides
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != sort_axis + 1)
            .map(|(_, &s)| s)
            .collect();

        let iter_stride = in_strides[sort_axis];
        let out_stride = self.out_strides[sort_axis];

        // derive the batch dimensions from the sort_axis
        let batch_shape: Vec<Expression> = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != sort_axis)
            .map(|(_, &s)| s)
            .collect();
        let batch_in_strides: Vec<Expression> = in_strides
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != sort_axis)
            .map(|(_, &s)| s)
            .collect();
        let batch_out_strides: Vec<Expression> = self
            .out_strides
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != sort_axis)
            .map(|(_, &s)| s)
            .collect();

        let vars = batch_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(batch_in_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(batch_out_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(out_stride.dyn_vars())
            .chain(self.iters.dyn_vars())
            .chain(iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let n_blocks: Expression = batch_shape.iter().copied().product();
        let in_index = flatten_mul_strides(&batch_shape, &batch_in_strides);
        let out_index = flatten_mul_strides(&batch_shape, &batch_out_strides);

        let threads_per_block = 1024;
        let shmem_limit = 4096; // safe shmem limit for values + indices

        let kernel = format!(
            "
#define THREADS_PER_BLOCK {threads_per_block}
#define SHMEM_LIMIT {shmem_limit}
#define ASCENDING {ascending}
{constants}
extern \"C\" {{
    __global__ void argsort_k(int *output, const {dtype} *data) {{
        extern __shared__ char shared_mem[];
        {dtype} *shmem_vals = ({dtype}*)shared_mem;
        int *shmem_idx = (int*)(shmem_vals + SHMEM_LIMIT);

        int const_z = blockIdx.x;
        int tid = threadIdx.x;

        int in_base = {in_index};
        int out_base = {out_index};
        int N = {iters};
        int in_stride = {iter_stride};
        int out_stride = {out_stride};

        {dtype} *vals;
        int *idx;
        int idx_stride;

        int use_shmem = (N <= SHMEM_LIMIT);
        if (use_shmem) {{
            vals = shmem_vals;
            idx = shmem_idx;
            for (int i = tid; i < N; i += THREADS_PER_BLOCK) {{
                vals[i] = data[in_base + i * in_stride];
            }}
            idx_stride = 1;
            in_stride = 1;
        }} else {{
            vals = ({dtype}*)(data + in_base);
            idx = output + out_base;
            idx_stride = out_stride;
        }}

        // init indices
        for (int i = tid; i < N; i += THREADS_PER_BLOCK) {{
            idx[i * idx_stride] = i;
        }}
        __syncthreads();

        // odd even transposition sort
        // https://www.geeksforgeeks.org/dsa/odd-even-transposition-sort-brick-sort-using-pthreads/
        for (int phase = 0; phase < N; phase++) {{
            int p2 = phase % 2;
            for (int i = tid; i < N / 2; i += THREADS_PER_BLOCK) {{
                int left = 2 * i + p2;
                int right = 2 * i + 1 + p2;

                if (right < N) {{
                    int idx_l = idx[left * idx_stride];
                    int idx_r = idx[right * idx_stride];
                    {dtype} val_l = vals[idx_l * in_stride];
                    {dtype} val_r = vals[idx_r * in_stride];
                    bool cond = ASCENDING ? val_l > val_r : val_l < val_r;

                    if (cond) {{
                        idx[left * idx_stride] = idx_r;
                        idx[right * idx_stride] = idx_l;
                    }}
                }}
            }}
            __syncthreads();
        }}

        // only need to write back if using shmem
        if (use_shmem) {{
            for (int i = tid; i < N; i += THREADS_PER_BLOCK) {{
                output[out_base + i * out_stride] = idx[i];
            }}
        }}
    }}
}}",
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            dtype = dtype,
            in_index = in_index.to_kernel(),
            out_index = out_index.to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = iter_stride.to_kernel(),
            out_stride = out_stride.to_kernel(),
            threads_per_block = threads_per_block,
            shmem_limit = shmem_limit,
            ascending = if self.descending { 0 } else { 1 },
        );

        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("argsort_k").unwrap();

        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();

        let shmem_bytes = shmem_limit * 8; // safe for 4 bytes dtype (f32) + 4 bytes int indices

        (
            func,
            module,
            kernel,
            (n_blocks, 1.into(), 1.into()),                 // grid
            (threads_per_block.into(), 1.into(), 1.into()), // threads
            shmem_bytes.into(),                             // shmem
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }
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

        let kernel = format!(
            "
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{constants}
extern \"C\" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in) {{
        __shared__ {dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        int const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        int in_start = {in_index};
        int iters = {iters};
        int iter_stride = {iter_stride};

        {dtype} max_value = NEG_INF_F;
        for (int i = tid; i < iters; i += THREADS_PER_BLOCK) {{
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
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            dtype = dtype,
            in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self.iter_stride.to_kernel(),
        );

        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("reduce_max_k").unwrap();

        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
            constants,
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

pub struct KernelMeanReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl EgglogOp for KernelMeanReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMean".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?sum (Sum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?sum_out_stride))
        (= ?iota (Iota ?iters ?one))
        (= ?cast (Cast ?iota (F32)))
        (= ?recip (Recip ?r_shape ?cast ?r_in_strides ?r_out_strides))
        (= ?result (Mul ?shape ?sum ?sum_strides ?recip ?recip_strides ?out_strides))
        (= ?dty (dtype ?inp))
    )
    (
        (union ?result (KernelMean ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_strides ?dty))
    )
    :name \"kernel mean reduce\"
)
".to_string()]
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

impl KernelOp for KernelMeanReduce {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
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

        let kernel = format!(
            "
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{constants}
extern \"C\" {{
    __global__ void reduce_mean_k({dtype} *out, const {dtype} *in) {{
        __shared__ {dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        int const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        int in_start = {in_index};
        int iters = {iters};
        int iter_stride = {iter_stride};

        {dtype} sum = 0;
        for (int i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            sum += in[in_start + i * iter_stride];
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            sum += __shfl_down_sync(FULL_MASK, sum, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = sum;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {dtype} block_sum = tid < cnt ? warp_sums[tid] : 0;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})(block_sum / (float)iters);
            }}
        }}
    }}
}}",
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            dtype = dtype,
            in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self.iter_stride.to_kernel(),
        );

        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("reduce_mean_k").unwrap();

        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
            constants,
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
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        n_outputs * self.iters + n_outputs
    }

    fn kernel_name(&self) -> &'static str {
        "MeanReduce"
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

        let kernel = format!(
            "
{constants}
extern \"C\" {{
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in) {{
        int const_z = blockIdx.x;

        int in_start = {in_index};
        int iters = {iters};
        int iter_stride = {iter_stride};

        {dtype} sum = 0;
        for (int i = 0; i < iters; i++) {{
            sum += in[in_start + i * iter_stride];
        }}

        out[{out_index}] = sum;
    }}
}}",
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            dtype = dtype,
            in_index = flatten_mul_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self.iter_stride.to_kernel(),
        );

        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("reduce_sum_k").unwrap();

        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()), // grid
            (1.into(), 1.into(), 1.into()),  // blocks (single-threaded)
            0.into(),                        // shmem size
            constants,
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
}

impl EgglogOp for KernelAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Add ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (union ?a (KernelAdd ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty))
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
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelAdd {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
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
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void add_k({dtype} *C, const {dtype} *A, const {dtype} *B) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] + B[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("add_k").unwrap();
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
}

impl EgglogOp for KernelMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["
(rule
    (
        (= ?a (Mul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides))
        (= ?dty (dtype ?inp_a))
    )
    (
        (union ?a (KernelMul ?out_shape ?inp_a ?inp_a_strides ?inp_b ?inp_b_strides ?out_strides ?dty))
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
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelMul {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
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
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void mul_k({dtype} *C, const {dtype} *A, const {dtype} *B) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[{}] = A[{}] * B[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.a_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.b_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("mul_k").unwrap();
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
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

impl EgglogOp for KernelGather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "KernelGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Dty],
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
        (union ?a (KernelGather ?out_shape ?indexes ?index_strides ?data ?data_strides ?out_strides ?dty))
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
                data_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                dtype: extract_dtype(egraph, children[6]),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl KernelOp for KernelGather {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
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
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        {dtype}* out = C + {};
        const_z = indexes[{}];
        *out = data[{}];
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            flatten_mul_strides(&self.out_shape, &self.out_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.index_stride).to_kernel(),
            flatten_mul_strides(&self.out_shape, &self.data_stride).to_kernel()
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("gather").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            module,
            kernel,
            (self.out_shape.iter().copied().product(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
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
        let kernel = format!(
            "
{}
extern \"C\" {{
    __global__ void iota_k(int *C) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
        C[const_z] = {};
    }}
}}",
            vars.iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            self.expr.to_kernel(),
        );
        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("iota_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();
        (
            func,
            module,
            kernel,
            (self.range, 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            constants,
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
