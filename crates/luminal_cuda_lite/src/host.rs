//! Opaque host-launched operations, such as cuBLASLt library calls.

use luminal::buffer_tensor_ir::BufferTensorIrOp;

/// A device buffer or arena subrange bound by the executor.
/// Raw pointers allow simultaneously live ranges within one arena allocation.
/// Calls use the executor's single stream, which orders access to those ranges.
#[derive(Debug, Clone, Copy)]
pub struct DeviceRange {
    pub ptr: u64,
    pub bytes: usize,
}

/// Bindings and layout descriptors for a single-destination host operation.
#[cfg(feature = "device")]
pub struct HostOpContext<'a> {
    pub stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
    /// Read operands, excluding the destination appended by DPS lowering.
    pub inputs: &'a [DeviceRange],
    pub dest: DeviceRange,
    /// Plan-order descriptors, including the appended destination operand.
    pub operand_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
    pub result_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
}

/// An operation that launches work from the host on the executor's CUDA stream.
/// The trait remains available in device-free builds for operation selection.
pub trait HostOp: BufferTensorIrOp {
    /// Execute using the bufferizer's bindings and declared memory effects.
    ///
    /// # Safety
    /// The device ranges must be live on `ctx.stream`'s device, large enough
    /// for their descriptors, and aliased only as permitted by the op's
    /// bufferization contract. They must remain live until the work completes.
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &HostOpContext<'_>) -> anyhow::Result<()>;
}
