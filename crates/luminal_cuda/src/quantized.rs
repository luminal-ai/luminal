use std::{marker::PhantomData, sync::Arc};

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use petgraph::visit::EdgeRef;

use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
};

use crate::{
    binary::CudaGather, compile_and_load_kernel, get_buffer_from_tensor, CudaData, CudaFloat,
};

/// Multiplies a BxMxK matrix with a KxN matrix, resulting in a BxMxN matrix. This expects the first input to be a quantized 2D matrix
#[derive(Clone)]
pub struct QuantizedMatmul<T> {
    matvec_function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
}
crate::debug_type!(QuantizedMatmul);

impl<T: CudaFloat> QuantizedMatmul<T> {
    fn new(device: Arc<CudaContext>) -> Self {
        let type_name = T::type_name();
        Self {
            matvec_function: compile_and_load_kernel(format!("
#include \"cuda_fp16.h\"
typedef struct {{
    half    d;         // delta
    char  qs[32]; // quants
}} block_q8_0;

__inline__ __device__ float warpReduceSum(float val) {{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {{
        val += __shfl_down_sync(0xffffffff, val, offset);
    }}
    return val;
}}

extern \"C\" __global__ void kernel(
    const block_q8_0* x, // Quantized 2D matrix
    const {type_name}* y, // Float src vector
    {type_name}* dst, // Float dest vector
    int src_vec_size, // Matrix n cols (src vector size) (Must be >= 32)
    int dest_vec_size, // Matrix n rows (dest vector size) (Must be >= 4)
    int mat_batch_stride, // Matrix batch stride
    int vec_batch_stride // Vector batch stride
) {{
    int threadgroup_position_in_grid_x = blockIdx.x;
    int threadgroup_position_in_grid_z = blockIdx.z;
    int thread_index_in_simdgroup = (threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
    int simdgroup_index_in_threadgroup = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize; // 2 simdgroups in a threadgroup
    const int num_rows = 4;
    const int num_simdgroups_per_threadgroup = 2;

    int num_quants_per_row = src_vec_size / 32; // Number of quants per row

    // This is the first row the simdgroup will work on (each simdgroup handles a block of 4 rows)
    int first_row = (threadgroup_position_in_grid_x * num_simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * num_rows;

    // Offsets
    // x += first_row * num_quants_per_row + threadgroup_position_in_grid_z * (mat_batch_stride / 32); // Batch offset
    x += first_row * num_quants_per_row; // No batch offset
    y += threadgroup_position_in_grid_z * vec_batch_stride;
    dst += (threadgroup_position_in_grid_z * dest_vec_size);

    // thread-local cache of vector values to work on. This thread must only work on 8 at a time
    {type_name} yl[8];
    // thread-local cache of 4 row sums
    float sumf[num_rows] = {{0.0}};

    int ix = thread_index_in_simdgroup / 4;
    int il = thread_index_in_simdgroup % 4;

    y += thread_index_in_simdgroup * 8;

    // each thread in a SIMD group deals with 8 quants at a time
    // we start at 0-7 (ix) depending on the simdgroup index, and jump 8 indexes each time
    for (int ib = ix; ib < num_quants_per_row; ib += 8) {{ // ib: current column position
        // Load vector values into the cache
        for (int i = 0; i < 8; ++i) {{
            yl[i] = y[i];
        }}

        // Loop through 4 matrix rows
        for (int row = 0; row < 4; ++row) {{
            // Get pointer to matrix data
            const char* qs = x[ib + row * num_quants_per_row].qs + il * 8;
            float sumq = 0.0; // Partial sum
            // Loop through 8 columns
            for (int iq = 0; iq < 8; ++iq) {{
                sumq += (float)qs[iq] * (float)yl[iq]; // Multiply int with vector value (auto converts to float?)
            }}
            sumf[row] += sumq * (float)x[ib + row * num_quants_per_row].d; // multiply by delta (scaling factor)
        }}
        y += 256; // Jump by 256
    }}

    // each simdgroup is responsible for saving 4 final vector values (n rows)
    for (int row = 0; row < num_rows; ++row) {{
        // warp sum
        float sum = warpReduceSum(sumf[row]);
        if (thread_index_in_simdgroup == 0 && first_row + row < dest_vec_size) {{
            dst[first_row + row] = ({type_name})sum;
        }}
    }}
}}"), &device),
            device,
            _phantom: Default::default(),
        }
    }
}

impl<T: 'static + CudaFloat> Operator for QuantizedMatmul<T> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        assert!(
            !inp[1].1.is_contiguous(),
            "Weight matrix must be column-major"
        );
        let (a_shape, b_shape) = (
            inp[0]
                .1
                .dims()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
            inp[1]
                .1
                .dims()
                .into_iter()
                .map(|i| i.to_usize().unwrap())
                .collect::<Vec<_>>(),
        );
        let a_dims = a_shape.len();
        let b_dims = b_shape.len();
        let batch_size = a_shape.iter().take(a_dims - 2).product::<usize>().max(1);
        let m = a_shape[a_dims - 2];
        let k = b_shape[b_dims - 2];
        let n = b_shape[b_dims - 1];
        let stream = self.device.default_stream();

        let mut out = unsafe { stream.alloc::<T>(batch_size * m * n).unwrap() };

        // Matvec
        let mut launch_args = stream.launch_builder(&self.matvec_function);
        launch_args.arg(get_buffer_from_tensor::<u8>(&inp[1].0)); // Matrix
        launch_args.arg(get_buffer_from_tensor::<T>(&inp[0].0)); // Vector
        launch_args.arg(&mut out); // Dest vector
        launch_args.arg(&k); // Src vec size
        launch_args.arg(&n); // Dest vec size
        launch_args.arg(&0); // Matrix batch stride
        launch_args.arg(&k); // Vector batch stride

        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (n.div_ceil(8) as u32, 1, (m * batch_size) as u32),
                    block_dim: (8, 8, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Clone)]
pub struct QuantizedGather<T> {
    pipeline: CudaFunction,
    device: Arc<CudaContext>,
    embed_dim: usize,
    _phantom: PhantomData<T>,
}
crate::debug_type!(QuantizedGather);

impl<T: CudaFloat> QuantizedGather<T> {
    fn new(device: Arc<CudaContext>, embed_dim: usize) -> Self {
        let type_name = T::type_name();
        Self {pipeline: compile_and_load_kernel(format!(
            "
#include \"cuda_fp16.h\"
#define QK8_0 32
typedef struct {{
    half    d;         // delta
    char  qs[QK8_0]; // quants
}} block_q8_0;

extern \"C\" __global__ void kernel(const float* inp, const block_q8_0* weights, {type_name}* out, int n_embeddings, int embedding_dim) {{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (pos_x < n_embeddings && pos_y < embedding_dim) {{
        int block_idx = ((int)inp[pos_x] * embedding_dim + pos_y) / QK8_0;
        out[pos_x * embedding_dim + pos_y] = ({type_name})weights[block_idx].qs[pos_y % QK8_0] * ({type_name})weights[block_idx].d;
    }}
}}"), &device), device, embed_dim, _phantom: Default::default()}
    }
}

impl<T: CudaFloat> Operator for QuantizedGather<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Setup buffers
        let indexes = tensors[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        let stream = self.device.default_stream();
        let mut index_buffer = unsafe { stream.alloc::<f32>(indexes.len()).unwrap() };
        stream.memcpy_htod(indexes, &mut index_buffer).unwrap();

        let mut out = unsafe { stream.alloc::<T>(indexes.len() * self.embed_dim).unwrap() };

        // Set inputs
        let indexes_len = indexes.len() as i32;
        let mut launch_args = stream.launch_builder(&self.pipeline);
        launch_args.arg(&index_buffer);
        launch_args.arg(get_buffer_from_tensor::<u8>(&tensors[1].0));
        launch_args.arg(&mut out);
        launch_args.arg(&indexes_len);
        launch_args.arg(&self.embed_dim);

        unsafe {
            launch_args
                .launch(LaunchConfig {
                    grid_dim: (indexes.len() as u32, self.embed_dim as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                })
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Default, Debug)]
pub struct CudaQuantizedCompiler<T>(Vec<NodeIndex>, PhantomData<T>);

impl<T> CudaQuantizedCompiler<T> {
    pub fn new<To: ToIds>(weights: To) -> Self {
        Self(weights.to_ids(), Default::default())
    }
}

impl<T: CudaFloat + Default> Compiler for CudaQuantizedCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let device = crate::shared_cuda_context(0);
        let weight_ids = self.0.clone();
        // Modify ops directly downstream of weights
        for weight in downstream(&weight_ids, graph) {
            for (target, (inp_ind, _, _)) in graph
                .edges_directed(weight, petgraph::Direction::Outgoing)
                .filter_map(|e| e.weight().as_data().map(|i| (e.target(), i)))
                .collect::<Vec<_>>()
            {
                assert_eq!(
                    inp_ind, 1,
                    "Quantized weight {target:?} is the wrong input!",
                );
                let op_node = graph.node_weight_mut(target).unwrap();
                if let Some(gather) = op_node.as_any().downcast_ref::<CudaGather<T>>() {
                    *op_node =
                        Box::new(QuantizedGather::<T>::new(device.clone(), gather.embed_dim));
                } else if op_node.as_any().is::<super::matmul::Matmul<T>>() {
                    *op_node = Box::new(QuantizedMatmul::<T>::new(device.clone()));
                } else {
                    panic!("Quantized weight {target:?} is an input to a node that isn't a matmul or gather ({op_node:?})");
                }
            }
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use std::sync::Arc;

//     use cudarc::driver::CudaContext;
//     use dfdx::{
//         tensor::TensorFromVec,
//         tensor_ops::{PermuteTo, TryMatMul},
//     };
//     use luminal::{
//         prelude::*,
//         tests::{assert_close, assert_close_precision, random_vec_rng},
//     };
//     use rand::{thread_rng, Rng};

//     use crate::{CudaData, CudaQuantizedCompiler};

//     #[repr(C, packed)]
//     struct BlockQ8_0 {
//         _d: f16,
//         _qs: [i8; 32],
//     }

//     fn quantized_buffer(weights: &[BlockQ8_0], dev: &Arc<CudaContext>) -> Tensor {
//         let n_bytes = std::mem::size_of_val(weights);
//         let buffer = dev
//             .htod_copy(unsafe {
//                 Vec::<u8>::from_raw_parts(weights.as_ptr() as *mut u8, n_bytes, n_bytes)
//             })
//             .unwrap();
//         Tensor::new(CudaData(buffer))
//     }

//     #[test]
//     fn test_quantized_matvec() {
//         let mut rng = thread_rng();
//         let mat_data: Vec<i8> = (0..1024 * 512).map(|_| rng.gen_range(0..5)).collect();
//         let vec_data = random_vec_rng(1024, &mut rng);
//         let mut cx = Graph::new();
//         let weights = cx.tensor::<R2<512, 1024>>().keep();
//         let vec = cx.tensor::<R1<1024>>().set(vec_data.clone());
//         let mut out = vec.matmul(weights.permute()).retrieve();

//         // "Load" weights in 8bit
//         let blocks = mat_data
//             .chunks_exact(32)
//             .map(|chunk| {
//                 let mut array = [0; 32];
//                 for (i, n) in chunk.iter().enumerate() {
//                     array[i] = *n;
//                 }
//                 BlockQ8_0 {
//                     _d: f16::from_f32(1.0),
//                     _qs: array,
//                 }
//             })
//             .collect::<Vec<_>>();
//         cx.tensors.insert(
//             (weights.id, 0),
//             quantized_buffer(&blocks, &CudaContext::new(0).unwrap()),
//         );

//         cx.compile(
//             CudaQuantizedCompiler::<f32>::new(vec![weights.id]),
//             &mut out,
//         );
//         cx.execute();

//         let mut cx1 = Graph::new();
//         let weights = cx1
//             .tensor::<R2<512, 1024>>()
//             .set(mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>())
//             .keep();
//         let vec = cx1.tensor::<R1<1024>>().set(vec_data);
//         let out_32 = vec.matmul(weights.permute()).retrieve();
//         cx1.execute();

//         assert_close(&out.data(), &out_32.data());
//         blocks.leak(); // Segfaults without this
//     }

//     #[test]
//     fn test_quantized_matmul() {
//         let mut rng = thread_rng();
//         let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
//         let inp_mat_data = random_vec_rng(1024 * 16, &mut rng);
//         let mut cx = Graph::new();
//         let weights = cx.tensor::<R2<512, 1024>>().keep();
//         let inp_mat = cx.tensor::<R2<16, 1024>>().set(inp_mat_data.clone());
//         let mut out = inp_mat.matmul(weights.permute()).retrieve();

//         // "Load" weights in 8bit
//         let blocks = mat_data
//             .chunks_exact(32)
//             .map(|chunk| {
//                 let mut array = [0; 32];
//                 for (i, n) in chunk.iter().enumerate() {
//                     array[i] = *n;
//                 }
//                 BlockQ8_0 {
//                     _d: f16::from_f32(1.0),
//                     _qs: array,
//                 }
//             })
//             .collect::<Vec<_>>();
//         let dev = CudaContext::new(0).unwrap();
//         cx.tensors
//             .insert((weights.id, 0), quantized_buffer(&blocks, &dev));

//         cx.compile(
//             CudaQuantizedCompiler::<f32>::new(vec![weights.id]),
//             &mut out,
//         );
//         cx.execute();

//         let cpu = dfdx::tensor::Cpu::default();
//         let d_a = cpu.tensor_from_vec(
//             mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
//             (dfdx::shapes::Const::<512>, dfdx::shapes::Const::<1024>),
//         );
//         let d_b = cpu.tensor_from_vec(
//             inp_mat_data,
//             (dfdx::shapes::Const::<16>, dfdx::shapes::Const::<1024>),
//         );
//         let d_c = d_b.matmul(d_a.permute());
//         assert_close(&out.data(), &d_c.as_vec());
//         blocks.leak(); // Segfaults without this
//     }

//     #[test]
//     fn test_quantized_matmul_fp16() {
//         let mut rng = thread_rng();
//         let mat_data: Vec<i8> = (0..(1024 * 512)).map(|_| rng.gen_range(0..5)).collect();
//         let inp_mat_data = random_vec_rng(1024 * 16, &mut rng);
//         let mut cx = Graph::new();
//         let weights = cx.tensor::<R2<512, 1024>>().keep();
//         let inp_mat = cx.tensor::<R2<16, 1024>>().set(inp_mat_data.clone());
//         let mut out = inp_mat.matmul(weights.permute()).retrieve();

//         // "Load" weights in 8bit
//         let blocks = mat_data
//             .chunks_exact(32)
//             .map(|chunk| {
//                 let mut array = [0; 32];
//                 for (i, n) in chunk.iter().enumerate() {
//                     array[i] = *n;
//                 }
//                 BlockQ8_0 {
//                     _d: f16::from_f32(1.0),
//                     _qs: array,
//                 }
//             })
//             .collect::<Vec<_>>();
//         let dev = CudaContext::new(0).unwrap();
//         cx.tensors
//             .insert((weights.id, 0), quantized_buffer(&blocks, &dev));

//         cx.compile(
//             CudaQuantizedCompiler::<f16>::new(vec![weights.id]),
//             &mut out,
//         );
//         cx.execute();

//         let cpu = dfdx::tensor::Cpu::default();
//         let d_a = cpu.tensor_from_vec(
//             mat_data.into_iter().map(|i| i as f32).collect::<Vec<_>>(),
//             (dfdx::shapes::Const::<512>, dfdx::shapes::Const::<1024>),
//         );
//         let d_b = cpu.tensor_from_vec(
//             inp_mat_data,
//             (dfdx::shapes::Const::<16>, dfdx::shapes::Const::<1024>),
//         );
//         let d_c = d_b.matmul(d_a.permute());
//         assert_close_precision(&out.data(), &d_c.as_vec(), 1.0);
//         // This is imprecise currently because we accumulate in fp16 in the matmul. TODO: accumulate in fp32 and convert before saving to dest

//         blocks.leak(); // Segfaults without this
//     }
// }
