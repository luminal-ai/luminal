mod ops;
pub use ops::*;

use luminal::op::EgglogOp;
use luminal::prelude::*;
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device};
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
}

luminal::impl_into_ops!(MetalKernelOp);
