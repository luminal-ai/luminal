//! Opaque library operations compiled into captured CUDA child graphs.
use luminal::buffer_tensor_ir::BufferTensorIrOp;

#[derive(Debug, Clone, Copy)]
pub struct DeviceRange {
    pub ptr: u64,
    pub bytes: usize,
}

#[cfg(feature = "device")]
pub struct HostOpContext<'a> {
    pub stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
    pub inputs: &'a [DeviceRange],
    pub dest: DeviceRange,
    pub workspace: DeviceRange,
    pub dims: &'a luminal::shape::DynMap,
    pub operand_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
    pub result_info: &'a [luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>],
}

/// Created only inside an active graph capture. Recording never executes model work.
#[cfg(feature = "device")]
pub struct CaptureCtx<'a> {
    pub(crate) stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
}
#[cfg(feature = "device")]
impl CaptureCtx<'_> {
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        self.stream
    }
}

#[cfg(feature = "device")]
pub trait PreparedHostOp {
    /// Record GPU work into the active capture. Preparation must already have
    /// loaded modules and resolved host-side descriptors/algorithms.
    /// # Safety
    /// Bound device ranges must remain live until the last graph replay finishes.
    unsafe fn record(&self, capture: &CaptureCtx<'_>) -> anyhow::Result<()>;
}

pub trait HostOp: BufferTensorIrOp {
    /// Scratch reserved in the runtime's shared arena. A host op must not retain
    /// invocation state in scratch across graph launches or bucket switches.
    /// All device scratch must come from this reservation.
    fn workspace_bytes(&self, _bounds: &crate::symbolic::Bounds) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Dimensions affecting captured library work. None conservatively means
    /// all dimensions; Some(empty) describes a shape-independent capture.
    /// Include every dimension affecting descriptors, pointers, or recorded work.
    fn capture_dims(&self) -> Option<Vec<luminal::shape::Symbol>> {
        None
    }

    /// Prepare a capture-compatible library call without executing it.
    /// # Safety
    /// ctx's ranges must be valid, sized for their descriptors, and obey this
    /// operation's alias contract. Returned state may retain their addresses.
    #[cfg(feature = "device")]
    unsafe fn prepare(&self, ctx: &HostOpContext<'_>) -> anyhow::Result<Box<dyn PreparedHostOp>>;
}
