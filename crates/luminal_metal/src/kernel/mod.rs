mod matmul;
mod ops;
pub use matmul::*;
pub use ops::*;

use luminal::dtype::DType;
use luminal::op::EgglogOp;
use luminal::prelude::*;
use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Device};

pub const DYN_SLOT_COUNT: usize = 26;

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
        command_buffer: &CommandBufferRef,
        pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        dyn_buffer: &Buffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let pipeline = pipeline.expect("compute pipeline not compiled");
        let encoder = command_buffer.new_compute_command_encoder();
        let dyn_idx = inputs.len() as u64 + 1;
        encoder.set_buffer(dyn_idx, Some(dyn_buffer), 0);
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
