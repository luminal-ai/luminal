mod matmul;
mod ops;
pub use matmul::*;
pub use ops::*;

use luminal::dtype::DType;
use luminal::op::EgglogOp;
use luminal::prelude::*;
use metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Device,
    foreign_types::ForeignTypeRef, mps,
};
use objc::rc::StrongPtr;
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};
use std::cell::RefCell;

pub const DYN_SLOT_COUNT: usize = 26;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MpsMatrixDescriptorKey {
    rows: usize,
    cols: usize,
    row_bytes: u64,
    data_type: isize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MpsMatmulKey {
    transpose_lhs: bool,
    transpose_rhs: bool,
    m: usize,
    n: usize,
    k: usize,
    alpha: u64,
    beta: u64,
}

#[derive(Default)]
pub struct MpsKernelCache {
    matrix_descriptors: FxHashMap<MpsMatrixDescriptorKey, StrongPtr>,
    matmul_kernels: FxHashMap<MpsMatmulKey, StrongPtr>,
}

impl MpsKernelCache {
    pub(crate) fn matrix_descriptor(
        &mut self,
        rows: usize,
        cols: usize,
        row_bytes: u64,
        dtype: DType,
    ) -> *mut Object {
        let key = MpsMatrixDescriptorKey {
            rows,
            cols,
            row_bytes,
            data_type: Self::mps_data_type(dtype),
        };
        let descriptor = self
            .matrix_descriptors
            .entry(key)
            .or_insert_with(|| unsafe {
                let descriptor: *mut Object = msg_send![
                    class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows: rows
                    columns: cols
                    rowBytes: row_bytes as usize
                    dataType: key.data_type
                ];
                StrongPtr::retain(descriptor)
            });
        **descriptor
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matrix_multiplication(
        &mut self,
        command_buffer: &CommandBufferRef,
        transpose_lhs: bool,
        transpose_rhs: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: f64,
        beta: f64,
    ) -> *mut Object {
        let key = MpsMatmulKey {
            transpose_lhs,
            transpose_rhs,
            m,
            n,
            k,
            alpha: alpha.to_bits(),
            beta: beta.to_bits(),
        };
        let kernel = self.matmul_kernels.entry(key).or_insert_with(|| unsafe {
            let device: *mut Object = msg_send![command_buffer.as_ptr(), device];
            let kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let kernel: *mut Object = msg_send![
                kernel,
                initWithDevice: device
                transposeLeft: transpose_lhs
                transposeRight: transpose_rhs
                resultRows: m
                resultColumns: n
                interiorColumns: k
                alpha: alpha
                beta: beta
            ];
            StrongPtr::new(kernel)
        });
        **kernel
    }

    fn mps_data_type(dtype: DType) -> isize {
        match dtype {
            DType::F32 | DType::TF32 => mps::MPSDataType::Float32 as isize,
            DType::F16 => mps::MPSDataType::Float16 as isize,
            unsupported => panic!("MPSMatmul does not support dtype {unsupported:?}"),
        }
    }
}

pub struct MetalEncodeContext<'a> {
    pub(crate) command_buffer: &'a CommandBufferRef,
    pub(crate) dyn_buffer: &'a Buffer,
    pub(crate) mps_cache: &'a RefCell<MpsKernelCache>,
}

#[derive(Debug, Clone)]
pub struct MetalMulInfo {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub output_strides: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct MetalSumReduceInfo {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
}

pub trait MetalKernelOp: EgglogOp {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState>;

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression;

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    );

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        context: &mut MetalEncodeContext<'_>,
        pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let pipeline = pipeline.expect("compute pipeline not compiled");
        let encoder = context.command_buffer.new_compute_command_encoder();
        let dyn_idx = inputs.len() as u64 + 1;
        encoder.set_buffer(dyn_idx, Some(context.dyn_buffer), 0);
        self.encode_compute(encoder, pipeline, inputs, output, dyn_map);
        encoder.end_encoding();
    }

    // ========================================================================
    // Performance Metrics for MBU/MFU Calculation
    // ========================================================================

    fn bytes_loaded(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn bytes_stored(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn mul_info(&self) -> Option<MetalMulInfo> {
        None
    }

    fn sum_reduce_info(&self) -> Option<MetalSumReduceInfo> {
        None
    }

    fn output_aliases_input(&self) -> Option<usize> {
        None
    }

    fn is_matmul(&self) -> bool {
        false
    }
}

luminal::impl_into_ops!(MetalKernelOp);
