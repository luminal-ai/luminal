use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use num_traits::float::FloatConst;
use rustc_hash::FxHashMap;
use std::{any::Any, marker::PhantomData, mem::size_of, sync::Arc};

use petgraph::visit::EdgeRef;

use luminal::{
    op::{ConstantValue, InputTensor, Operator},
    prelude::*,
};

use crate::{
    binary::CudaSub,
    compile_and_load_kernel, constant, cuda_unary_op, get_buffer_from_tensor, get_idx_valid_exps,
    input_dyn_dims,
    prim::{
        CudaAdd, CudaConstant, CudaContiguous, CudaExp2, CudaMaxReduce, CudaMul, CudaRecip,
        CudaSin, CudaSqrt, CudaSumReduce,
    },
    render_dyn_dim_inputs, CudaData, CudaFloat,
};

/// Special kernel for efficient mean reduction
#[derive(Clone)]
pub struct CudaMeanReduce<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    pub dim: usize,
    pub dyn_symbols: Vec<char>,
    pub dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}
crate::debug_type!(CudaMeanReduce);

impl<T> PartialEq for CudaMeanReduce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim
    }
}

impl<T: CudaFloat> CudaMeanReduce<T> {
    fn new(
        dev: Arc<CudaContext>,
        dim: usize,
        shape: ShapeTracker,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let type_name = T::type_name();
        let mut code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(const {type_name} *inp, {type_name} *out, int n_elements, int front_size, int back_size, int dim_size{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_ < n_elements) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid_exp}) != 0) {{
                reduce_value += (float)inp[{idx_exp}];
            }}
        }}
        out[i_] = ({type_name})(reduce_value / (float)dim_size);
    }}
}}");
        code = code.replace("mkernel", "kernel_mean_reduce");

        Self {
            function: compile_and_load_kernel(code, &dev),
            device: dev,
            dim,
            dyn_symbols,
            dyn_map,
            _phantom: Default::default(),
        }
    }
}

impl<T: CudaFloat> Operator for CudaMeanReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Setup buffers
        let mut sh = tensors[0].1;
        sh.remove_dim(self.dim);
        let inp_size = sh.n_elements().to_usize().unwrap();
        let inp_size_int = inp_size as i32;
        let front_size = tensors[0]
            .1
            .dims()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>() as i32;
        let back_size = tensors[0]
            .1
            .dims()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>() as i32;
        let dim_size = tensors[0].1.dims()[self.dim].to_usize().unwrap() as i32;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(inp_size).unwrap();
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(get_buffer_from_tensor::<T>(&tensors[0].0));
        launch_args.arg(&front_size);
        launch_args.arg(&back_size);
        launch_args.arg(&dim_size);
        launch_args.arg(&inp_size_int);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }
        vec![Tensor::new(CudaData(out))]
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct MeanReduceCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for MeanReduceCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))
        let fake_sum_reduce = op::<CudaConstant<T>>();
        let sum_reduce = op::<CudaSumReduce<T>>();
        let mul = binary::<CudaMul<T>>(
            sum_reduce.clone(),
            unary::<CudaRecip<T>>(fake_sum_reduce.clone()),
        );
        let mut s = mul.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[mul.id]) {
                // An intermediate node can't be deleted
                continue;
            }
            let (sum_reduce, mul) = (s.get(&sum_reduce), s.get(&mul));
            let dim = graph.get_op::<CudaSumReduce<T>>(sum_reduce).dim;
            // Insert MeanReduce op
            let src = graph.get_sources(sum_reduce)[0];
            let mean_reduce = graph
                .add_op(CudaMeanReduce::<T>::new(
                    dev.clone(),
                    dim,
                    src.2,
                    &graph.dyn_map,
                ))
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            move_outgoing_edge(mul, mean_reduce, graph);
            remap(mul, mean_reduce, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(mul);
            s.try_delete();
        }
    }
}

/// Special kernel for efficient std norming
#[derive(Clone)]
pub struct CudaStdNorm<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    epsilon: f32, // Epsilon
    _phantom: PhantomData<T>,
}
crate::debug_type!(CudaStdNorm);

impl<T> PartialEq for CudaStdNorm<T> {
    fn eq(&self, other: &Self) -> bool {
        self.epsilon == other.epsilon
    }
}

impl<T: CudaFloat> CudaStdNorm<T> {
    fn new(epsilon: f32, device: Arc<CudaContext>) -> Self {
        let type_name = T::type_name();
        let kernel_code = format!("
#include \"cuda_fp16.h\"
typedef struct __align__(8) {{
    __half x;
    __half y;
    __half z;
    __half w;
 }} __half4;
__device__ float warp_sum(float val) {{
    const unsigned int mask = 0xffffffff;

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {{
        val += __shfl_down_sync(mask, val, offset);
    }}

    return __shfl_sync(mask, val, 0);
}}
extern \"C\" __global__ void kernel(const  {type_name} * src0, {type_name} * dst, const int row_size, const float eps) {{
    int threadgroup_position_in_grid = blockIdx.x;
    int thread_position_in_threadgroup = threadIdx.x;
    int simdgroup_index_in_threadgroup = thread_position_in_threadgroup / 32; // 32 threads in warp
    int thread_index_in_simdgroup = thread_position_in_threadgroup % 32;
    int threads_per_threadgroup = blockDim.x;

    extern __shared__ float buf[];
    const {type_name}4 * x = (const {type_name}4 *) (src0 + threadgroup_position_in_grid * row_size);

    float sumf = 0.;

    // parallel sum
    for (int i = thread_position_in_threadgroup; i < row_size/4; i += threads_per_threadgroup) {{
        sumf += (float)x[i].x * (float)x[i].x;
        sumf += (float)x[i].y * (float)x[i].y;
        sumf += (float)x[i].z * (float)x[i].z;
        sumf += (float)x[i].w * (float)x[i].w;
    }}
    float all_sum = sumf;
    all_sum = warp_sum(all_sum);

    if (threads_per_threadgroup > 32) {{
        if (simdgroup_index_in_threadgroup == 0) {{
            buf[thread_index_in_simdgroup] = 0.0f;
        }}

        __syncthreads();

        if (thread_index_in_simdgroup == 0) {{
            buf[simdgroup_index_in_threadgroup] = all_sum;
        }}

        __syncthreads();

        all_sum = buf[thread_index_in_simdgroup];
        all_sum = warp_sum(all_sum);
    }}

    const float mean  = all_sum / row_size;
    const float scale = rsqrt(mean + eps);

    {type_name}4 * y = ({type_name}4 *) (dst + threadgroup_position_in_grid * row_size);
    for (int i = thread_position_in_threadgroup; i < row_size/4; i += threads_per_threadgroup) {{
        y[i].x = ({type_name})((float)x[i].x * scale);
        y[i].y = ({type_name})((float)x[i].y * scale);
        y[i].z = ({type_name})((float)x[i].z * scale);
        y[i].w = ({type_name})((float)x[i].w * scale);
    }}
}}");

        Self {
            function: compile_and_load_kernel(kernel_code, &device),
            device,
            epsilon,
            _phantom: Default::default(),
        }
    }
}

impl<T: CudaFloat> Operator for CudaStdNorm<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let row_size = tensors[0].1.dims().last().unwrap().to_usize().unwrap();
        let row_size_int = row_size as i32;
        let stream = self.device.default_stream();
        let mut out = stream
            .alloc_zeros::<T>(tensors[0].1.n_elements().to_usize().unwrap())
            .unwrap();
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(get_buffer_from_tensor::<T>(&tensors[0].0));
        launch_args.arg(&mut out);
        launch_args.arg(&row_size_int);
        launch_args.arg(&self.epsilon);
        let batch_size = tensors[0]
            .1
            .dims()
            .into_iter()
            .take(tensors[0].1.len() - 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>();
        let mut nth = 32; // SIMD width
        while nth < row_size / 4 && nth < 1024 {
            nth *= 2;
        }
        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (batch_size as u32, 1, 1),
                    block_dim: (nth as u32, 1, 1),
                    shared_mem_bytes: 32 * size_of::<f32>() as u32,
                })
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

/// Replace the mean reduce pattern with a special kernel. This is meant to be ran **after** the FakeSumReduceCompiler.
#[derive(Default, Debug)]
pub struct StdNormCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for StdNormCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the RMSNorm pattern
        // mul(recip(sqrt(add(mean_reduce(mul(x, x)), 1e-6))), x)

        let mut eps = op::<CudaConstant<T>>();
        eps.check(|op, _| {
            if let Some(c) = op.as_any().downcast_ref::<CudaConstant<T>>() {
                if let ConstantValue::Float(v) = c.value {
                    v <= 1e-2 && v > 0.0
                } else {
                    false
                }
            } else {
                false
            }
        });
        let inp = node();
        let square = unary::<CudaMul<T>>(inp.clone()); // This should check both inputs! For some reason doesn't work
        let mean = unary::<CudaMeanReduce<T>>(square.clone());
        let add = binary::<CudaAdd<T>>(mean.clone(), eps.clone());
        let mul = unary::<CudaMul<T>>(unary::<CudaRecip<T>>(unary::<CudaSqrt<T>>(add.clone())));

        let mut s = mul.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[mul.id, inp.id]) {
                // An intermediate node can't be deleted
                continue;
            }
            let ConstantValue::Float(epsilon_num) =
                graph.get_op::<CudaConstant<T>>(s.get(&eps)).value
            else {
                continue;
            };
            let (mut x, _, mut sh) = graph.get_sources(s.get(&square))[0];
            if let Some(mean_reduce) = graph.try_get_op::<CudaMeanReduce<T>>(s.get(&mean)) {
                if mean_reduce.dim != sh.len() - 1 {
                    continue;
                }
            }
            if sh
                .dims()
                .last()
                .unwrap()
                .to_usize()
                .map(|i| i % 32 != 0 || i < 32)
                .unwrap_or(true)
            {
                continue;
            }
            if !graph
                .get_sources(s.get(&mul))
                .iter()
                .any(|(i, _, _)| *i == x)
            {
                continue;
            }

            // Input must be contiguous
            if sh.is_reshaped() {
                x = graph
                    .add_op(CudaContiguous::<T>::new(sh, dev.clone(), &graph.dyn_map))
                    .input(x, 0, sh)
                    .finish();
                sh = sh.contiguous();
            }

            // Insert RMSNorm op
            let rms_norm = graph
                .add_op(CudaStdNorm::<T>::new(epsilon_num, dev.clone()))
                .input(x, 0, sh)
                .finish();

            // Create edges to dests
            let mul = s.get(&mul);
            move_outgoing_edge(mul, rms_norm, graph);
            remap(mul, rms_norm, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(mul);
            s.try_delete();
        }
    }
}

cuda_unary_op!("exp", CudaExp);

#[derive(Default, Debug)]
pub struct CudaExpCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CudaExpCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the exp pattern
        // exp2(mul(x, const))

        let inp = node();
        let mul = binary::<CudaMul<T>>(inp.clone(), constant::<T>(1.0 / f32::ln(2.)));
        let exp2 = unary::<CudaExp2<T>>(mul.clone());
        let mut s = exp2.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[exp2.id]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert exp op
            let (_, _, src_shape) = graph
                .edges_connecting(s.get(&inp), s.get(&mul))
                .next()
                .unwrap()
                .weight()
                .as_data()
                .unwrap();
            let exp = graph
                .add_op(CudaExp::<T>::new(src_shape, dev.clone(), &graph.dyn_map))
                .input(s.get(&inp), 0, src_shape)
                .finish();

            // Create edges to dests
            let exp2 = s.get(&exp2);
            move_outgoing_edge(exp2, exp, graph);
            remap(exp2, exp, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(exp2);
            s.try_delete();
        }
    }
}

// Special kernel for cos
cuda_unary_op!("cos", CudaCos);

#[derive(Default, Debug)]
pub struct CudaCosCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CudaCosCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the cos pattern
        // sin(add(mul(const_neg_one, x), const_pi_over_2))

        let const_pi = constant::<T>(f32::PI() / 2.);
        let inp = node();
        let sub = binary::<CudaSub<T>>(inp.clone(), const_pi.clone());
        let sin = unary::<CudaSin<T>>(sub.clone());
        let mut s = sin.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[sin.id]) {
                // An intermediate node can't be deleted
                continue;
            }

            // Insert cos op
            let shape = graph
                .edges_directed(s.get(&sub), petgraph::Direction::Incoming)
                .filter(|e| !e.weight().is_schedule())
                .find(|e| e.source() != s.get(&const_pi))
                .unwrap()
                .weight()
                .as_data()
                .unwrap()
                .2;
            let cos = graph
                .add_op(CudaCos::<T>::new(shape, dev.clone(), &graph.dyn_map))
                .input(s.get(&inp), 0, shape)
                .finish();

            // Create edges to dests
            let sin = s.get(&sin);
            move_outgoing_edge(sin, cos, graph);
            remap(sin, cos, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(sin);
            s.try_delete();
        }
    }
}

/// Special kernel for efficient softmax. Currently only works on the last dim
#[derive(Clone)]
pub struct CudaSoftmax<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
}
crate::debug_type!(CudaSoftmax);

impl<T: CudaFloat> CudaSoftmax<T> {
    fn new(device: Arc<CudaContext>) -> Self {
        let type_name = T::type_name();
        Self {
            function: compile_and_load_kernel(
                format!(
                    "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel(const {type_name} * x, {type_name} * dst, const int ncols) {{
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    {type_name} max_val = -__int_as_float(0x7f800000);

    for (int col = tid; col < ncols; col += block_size) {{
        const int i = row*ncols + col;
        max_val = fmaxf(max_val, x[i]);
    }}

    // find the max value in the block
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {{
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }}

    {type_name} tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {{
        const int i = row*ncols + col;
        const {type_name} val = exp(x[i] - max_val);
        tmp += static_cast<{type_name}>(val);
        dst[i] = val;
    }}

    // sum up partial sums
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {{
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }}

    const {type_name} inv_tmp = ({type_name})1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {{
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }}
}}
",
                ),
                &device,
            ),
            device,
            _phantom: Default::default(),
        }
    }
}

impl<T: CudaFloat> Operator for CudaSoftmax<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Setup buffers
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let batch_size = tensors[0]
            .1
            .dims()
            .iter()
            .take(tensors[0].1.len() - 1)
            .map(|i| i.to_usize().unwrap())
            .product::<usize>()
            .max(1);
        let axis_size = tensors[0].1.dims().last().unwrap().to_usize().unwrap();
        let axis_size_int = axis_size as i32;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(inp_size).unwrap();

        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(get_buffer_from_tensor::<T>(&tensors[0].0));
        launch_args.arg(&mut out);
        launch_args.arg(&axis_size_int);
        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (batch_size as u32, 1, 1),
                    block_dim: (1, 32, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

/// Replace the softmax pattern with a special kernel.
#[derive(Default, Debug)]
pub struct SoftmaxCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for SoftmaxCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the mean-reduce pattern
        // mul(recip(fake_sum_reduce(const_ones)), sum_reduce(x))

        let max_reduce = op::<CudaMaxReduce<T>>();
        let mul = unary::<CudaMul<T>>(unary::<CudaRecip<T>>(unary::<CudaSumReduce<T>>(unary::<
            CudaExp<T>,
        >(
            unary::<CudaSub<T>>(max_reduce.clone()),
        ))));

        let mut s = mul.clone().search(graph);
        while s.next_match() {
            if s.check_no_delete(&[mul.id]) {
                // An intermediate node can't be deleted
                continue;
            }
            // Insert Softmax op
            let src = graph.get_sources(s.get(&max_reduce))[0];
            let mean_reduce = graph
                .add_op(CudaSoftmax::<T>::new(dev.clone()))
                .input(src.0, 0, src.2)
                .finish();

            // Create edges to dests
            let mul = s.get(&mul);
            move_outgoing_edge(mul, mean_reduce, graph);
            remap(mul, mean_reduce, &mut ids, graph);

            // Remove the old ops
            graph.remove_node(mul);
            s.try_delete();
        }
    }
}

#[cfg(test)]
mod tests {
    use luminal::prelude::*;

    use crate::tests::assert_op_in_graph;

    use super::{CudaMeanReduce, CudaStdNorm};
    #[test]
    fn test_norms() {
        let mut cx = Graph::new();
        let a = cx.tensor(32).set([0.; 32]);
        let mut b = a.layer_norm(0, 1e-5).retrieve();

        cx.compile(
            <(
                GenericCompiler,
                crate::prim::PrimitiveCompiler<f16>,
                crate::SpecialOpsCompiler<f16>,
            )>::default(),
            &mut b,
        );

        assert_op_in_graph::<CudaStdNorm<f16>>(&cx);
        assert_op_in_graph::<CudaMeanReduce<f16>>(&cx);
    }
}
