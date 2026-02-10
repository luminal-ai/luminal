mod ops;
pub use ops::*;

use luminal::op::EgglogOp;
use luminal::prelude::*;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device};

pub const DYN_BUFFER_INDEX: u64 = 30;
pub const DYN_SLOT_COUNT: usize = 26;

pub trait MetalKernelOp: EgglogOp {
    fn compile(&self, device: &Device) -> ComputePipelineState;

    fn output_size(&self) -> Expression;

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    );

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
}

luminal::impl_into_ops!(MetalKernelOp);
