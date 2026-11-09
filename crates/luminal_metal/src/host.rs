//! Synchronous host operations between completed Metal command buffers.
use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::layouts::DecodedLayout;
pub trait HostOp: BufferTensorIrOp {
    fn execute(
        &self,
        input: &[u8],
        layout: &DecodedLayout,
        dims: &luminal::shape::DynMap,
    ) -> anyhow::Result<Vec<u8>>;
}
