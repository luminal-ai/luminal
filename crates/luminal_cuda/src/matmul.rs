use std::{marker::PhantomData, sync::Arc};

use cudarc::{
    cublas::{sys::cublasOperation_t::*, CudaBlas},
    driver::{CudaContext, DevicePtr},
};

use crate::{
    get_buffer_from_tensor,
    prim::{CudaMul, CudaSumReduce},
    CudaData, CudaFloat,
};
use luminal::{
    op::{InputTensor, Operator},
    prelude::*,
};

#[derive(Clone)]
pub struct Matmul<T>(Arc<CudaBlas>, Arc<CudaContext>, PhantomData<T>);
crate::debug_type!(Matmul);

impl<T: CudaFloat> Operator for Matmul<T> {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let stream = self.1.default_stream();
        let (a_shape, b_shape) = (inp[0].1.dims(), inp[1].1.dims());
        let (batch_size, m, k, n) = (
            a_shape
                .iter()
                .take(a_shape.len() - 2)
                .map(|i| i.to_usize().unwrap())
                .product::<usize>() as i32,
            a_shape[a_shape.len() - 2].to_usize().unwrap() as i32,
            a_shape[a_shape.len() - 1].to_usize().unwrap() as i32,
            b_shape[b_shape.len() - 1].to_usize().unwrap() as i32,
        );
        let a = get_buffer_from_tensor::<T>(&inp[0].0);
        let b = get_buffer_from_tensor::<T>(&inp[1].0);
        let out = self
            .1
            .default_stream()
            .alloc_zeros::<T>((m * n * batch_size) as usize)
            .unwrap();
        let (a_row_major, b_row_major) = (
            inp[0].1.indexes[inp[0].1.len() - 1] > inp[0].1.indexes[inp[0].1.len() - 2],
            inp[1].1.indexes[inp[1].1.len() - 1] > inp[1].1.indexes[inp[1].1.len() - 2],
        );
        let (transa, transb) = match (a_row_major, b_row_major) {
            (true, true) => (CUBLAS_OP_N, CUBLAS_OP_N),
            (false, false) => (CUBLAS_OP_T, CUBLAS_OP_T),
            (false, true) => (CUBLAS_OP_N, CUBLAS_OP_T),
            (true, false) => (CUBLAS_OP_T, CUBLAS_OP_N),
        };

        let a_dims = inp[0].1.fake.iter().filter(|f| !**f).count();
        let b_dims = inp[1].1.fake.iter().filter(|f| !**f).count();
        let (a_ptr, _a) = a.device_ptr(&stream);
        let (b_ptr, _b) = b.device_ptr(&stream);
        let (out_ptr, _out) = out.device_ptr(&stream);
        if T::is_f32() {
            unsafe {
                cudarc::cublas::result::sgemm_strided_batched(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &1.0_f32 as *const f32,
                    b_ptr as *const f32,
                    if b_row_major { n } else { k },
                    if b_dims == 2 { 0 } else { (n * k) as i64 },
                    a_ptr as *const f32,
                    if a_row_major { k } else { m },
                    if a_dims == 2 { 0 } else { (m * k) as i64 },
                    &0.0_f32 as *const f32,
                    out_ptr as *mut f32,
                    n,
                    (m * n) as i64,
                    batch_size,
                )
                .unwrap();
            }
        } else {
            unsafe {
                cudarc::cublas::result::hgemm_strided_batched(
                    *self.0.handle(),
                    transa,
                    transb,
                    n,
                    m,
                    k,
                    &f16::from_f32(1.0) as *const f16,
                    b_ptr as *const f16,
                    if b_row_major { n } else { k },
                    if b_dims == 2 { 0 } else { (n * k) as i64 },
                    a_ptr as *const f16,
                    if a_row_major { k } else { m },
                    if a_dims == 2 { 0 } else { (m * k) as i64 },
                    &f16::from_f32(0.0) as *const f16,
                    out_ptr as *mut f16,
                    n,
                    (m * n) as i64,
                    batch_size,
                )
                .unwrap();
            }
        }
        drop(_out);
        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Default)]
pub struct MatMulCompiler<T>(PhantomData<T>);

impl<T: CudaFloat + 'static> Compiler for MatMulCompiler<T>
where
    CudaData<T>: Data,
{
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = crate::shared_cuda_context(0);
        // Look for the matmul pattern
        // Mul ([A, C(fake), B] | [A(fake), C, B]) -> SumReduce(2) -> [A, C]
        // Actually starts at [A,B] | [B, C]
        let mut mul2d = op::<CudaMul<T>>();
        mul2d.shapes([['M', 'N', 'K'], ['M', 'N', 'K']]);
        mul2d.fakes([
            [None, Some(true), Some(false)],
            [Some(true), Some(false), Some(false)],
        ]);
        let mut sr2d = op::<CudaSumReduce<T>>();
        sr2d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                o.dim == 2
            } else {
                false
            }
        });
        let mut s2d = mul2d.clone().connect(sr2d.clone()).search(graph);
        let mut mul3d = op::<CudaMul<T>>();
        mul3d.shapes([['D', 'A', 'C', 'B'], ['D', 'A', 'C', 'B']]);
        mul3d.fakes([
            [Some(false), Some(false), Some(true), Some(false)],
            [None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr3d = op::<CudaSumReduce<T>>();
        sr3d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                o.dim == 3
            } else {
                false
            }
        });
        let mut s3d = mul3d.clone().connect(sr3d.clone()).search(graph);
        let mut mul4d = op::<CudaMul<T>>();
        mul4d.shapes([['E', 'D', 'A', 'C', 'B'], ['E', 'D', 'A', 'C', 'B']]);
        mul4d.fakes([
            [
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
            ],
            [None, None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr4d = op::<CudaSumReduce<T>>();
        sr4d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                o.dim == 4
            } else {
                false
            }
        });
        let mut s4d = mul4d.clone().connect(sr4d.clone()).search(graph);
        let mut mul5d = op::<CudaMul<T>>();
        mul5d.shapes([
            ['F', 'E', 'D', 'A', 'C', 'B'],
            ['F', 'E', 'D', 'A', 'C', 'B'],
        ]);
        mul5d.fakes([
            [
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(false),
            ],
            [None, None, None, Some(true), Some(false), Some(false)],
        ]);
        let mut sr5d = op::<CudaSumReduce<T>>();
        sr5d.check(|o, _| {
            if let Some(o) = o.as_any().downcast_ref::<CudaSumReduce<T>>() {
                o.dim == 5
            } else {
                false
            }
        });
        let mut s5d = mul5d.clone().connect(sr5d.clone()).search(graph);
        while s2d.next_match() || s3d.next_match() || s4d.next_match() || s5d.next_match() {
            let (mul, sum_reduce) = if s2d.matched {
                (s2d.get(&mul2d), s2d.get(&sr2d))
            } else if s3d.matched {
                (s3d.get(&mul3d), s3d.get(&sr3d))
            } else if s4d.matched {
                (s4d.get(&mul4d), s4d.get(&sr4d))
            } else {
                (s5d.get(&mul5d), s5d.get(&sr5d))
            };
            if graph.no_delete.contains(&mul) {
                // The intermediate mul can't be deleted
                continue;
            }
            // Insert Matmul op
            let srcs = graph.get_sources(mul);
            let (src1, mut src1_shape) = (srcs[0].0, srcs[0].2);
            let (src2, mut src2_shape) = (srcs[1].0, srcs[1].2);
            // Undo expansions and permute
            src1_shape.remove_dim(src1_shape.len() - 2);
            src2_shape.remove_dim(src2_shape.len() - 3);
            let mut dims = (0..src2_shape.len()).collect::<Vec<_>>();
            dims.swap(src2_shape.len() - 2, src2_shape.len() - 1);
            src2_shape.permute(&dims);
            let new_op = graph
                .add_op(Matmul::<T>(
                    Arc::new(CudaBlas::new(dev.default_stream()).unwrap()),
                    dev.clone(),
                    Default::default(),
                ))
                .input(src1, 0, src1_shape)
                .input(src2, 0, src2_shape)
                .finish();

            // Create edges to dests
            move_outgoing_edge(sum_reduce, new_op, &mut graph.graph);
            remap(sum_reduce, new_op, &mut ids, graph);
            remap(mul, new_op, &mut ids, graph);

            // Remove the old ops
            graph.graph.remove_node(mul);
            graph.graph.remove_node(sum_reduce);
        }
    }
}
