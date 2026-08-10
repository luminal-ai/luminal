use std::{fmt::Debug, sync::Arc};

use crate::cudarc::driver::{CudaStream, DriverError, result};
#[doc(hidden)]
pub use crate::resource::{HostDeviceMemoryPlan, ResourceViolation, SharedDeviceMemoryAllocation};
use luminal::{op::EgglogOp, prelude::*};
pub(crate) mod cublaslt;
pub mod flashinfer;
pub mod moe;

pub type Ops = (
    cublaslt::CuBlasLt,
    cublaslt::CuBlasLtScaled,
    moe::GLUMoE,
    moe::fused::FusedMoE,
    flashinfer::FlashInferAttention,
    flashinfer::sink_attention::SinkAttention,
);

#[cfg(test)]
pub(crate) type CublasLtTypeTuple = (
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    luminal::dtype::DType,
    &'static str,
    luminal::dtype::DType,
);

#[cfg(test)]
pub(crate) fn cublaslt_type_tuple(op: &dyn HostOp) -> Option<CublasLtTypeTuple> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::type_tuple)
}

#[cfg(test)]
pub(crate) type CublasLtScaleValues = (f64, f64);

#[cfg(test)]
pub(crate) fn cublaslt_scale_values(op: &dyn HostOp) -> Option<CublasLtScaleValues> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::scale_values)
}

#[cfg(test)]
pub(crate) fn cublaslt_epilogue(op: &dyn HostOp) -> Option<&'static str> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::epilogue)
}

#[cfg(test)]
pub(crate) type CublasLtMatrixOrders = (&'static str, &'static str, &'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_matrix_orders(op: &dyn HostOp) -> Option<CublasLtMatrixOrders> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::matrix_orders)
}

#[cfg(test)]
pub(crate) type CublasLtTransposeOps = (&'static str, &'static str);

#[cfg(test)]
pub(crate) fn cublaslt_transpose_ops(op: &dyn HostOp) -> Option<CublasLtTransposeOps> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::transpose_ops)
}

#[cfg(test)]
pub(crate) fn cublaslt_c_d_layouts_match(op: &dyn HostOp) -> Option<bool> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::c_d_layouts_match)
}

#[cfg(test)]
pub(crate) type CublasLtTensorScaleInputs = (bool, bool);

#[cfg(test)]
pub(crate) fn cublaslt_tensor_scale_inputs(op: &dyn HostOp) -> Option<CublasLtTensorScaleInputs> {
    op.as_any()
        .downcast_ref::<cublaslt::CuBlasLt>()
        .map(cublaslt::CuBlasLt::tensor_scale_inputs)
}

/// Non-owning device buffer handle used by host operations.
///
/// Runtime-owned intermediates may be a whole `CudaSlice`, a subregion inside
/// the reusable arena, or an external pointer. Host ops only need the pointer
/// and the logical byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    ptr: u64,
    len: usize,
}

impl DeviceBuffer {
    pub fn new(ptr: u64, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn ptr(self) -> u64 {
        self.ptr
    }

    pub fn len(self) -> usize {
        self.len
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn clone_dtoh(self, stream: &Arc<CudaStream>) -> Result<Vec<u8>, DriverError> {
        let mut host = vec![0u8; self.len];
        unsafe {
            result::memcpy_dtoh_async(&mut host, self.ptr, stream.cu_stream())?;
        }
        stream.synchronize()?;
        Ok(host)
    }
}

/// Host operations that execute on the CPU but orchestrate GPU work.
///
/// This includes operations like cuBLAS calls and CUDA graph executions.
pub trait HostOp: Debug + as_any::AsAny + EgglogOp {
    /// Execute the operation with access to buffers via a map.
    ///
    /// # Arguments
    /// * `stream` - The CUDA stream to execute on
    /// * `self_node` - The NodeIndex of this op in the llir_graph (used as output buffer)
    /// * `inputs` - NodeIndices of input nodes (in edge order from the graph)
    /// * `buffers` - Map from NodeIndex to device buffer for all allocated nodes
    /// * `dyn_map` - Dynamic dimension values
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., CudaGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns the output buffer size in bytes (accounts for dtype).
    fn output_bytes(&self) -> Expression;

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For CudaGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns relative lifetimes for extra buffer nodes within this host op.
    ///
    /// The tuple is `(node, first_step, last_step)`, where steps are local to
    /// this host op's execution. Returning `None` tells the runtime to treat
    /// every extra buffer as live for the whole host op.
    fn extra_buffer_lifetimes(&self) -> Option<Vec<(NodeIndex, usize, usize)>> {
        None
    }

    /// Returns pairs of extra buffer nodes that must not share arena storage.
    ///
    /// This refines `extra_buffer_lifetimes` for host ops with internal DAGs:
    /// two buffers may have disjoint positions in one topological order while
    /// still being unordered by real dependencies, so CUDA could overlap them.
    fn extra_buffer_conflicts(&self) -> Option<Vec<(NodeIndex, NodeIndex)>> {
        None
    }

    /// Returns buffer size requirements for extra nodes (node -> size in elements).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For CudaGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Device allocations owned by, temporarily created by, or globally
    /// shared by this host operation beyond its graph-visible output and the
    /// runtime intermediate arena. Planning is pointer-free: `buffer_lengths`
    /// contains logical byte lengths only, and implementations must not
    /// allocate device memory or read device contents during this call.
    fn device_memory_plan(
        &self,
        _self_node: NodeIndex,
        _inputs: &[NodeIndex],
        _buffer_lengths: &FxHashMap<NodeIndex, usize>,
        _dyn_map: &FxHashMap<char, usize>,
    ) -> Result<HostDeviceMemoryPlan, ResourceViolation> {
        Ok(HostDeviceMemoryPlan::default())
    }

    /// Graph-visible buffers whose logical byte lengths can change the result
    /// of `device_memory_plan`.
    ///
    /// Pointer identity and payload contents are never resource facts. Most
    /// HostOps derive their resource requirements entirely from `dyn_map` and
    /// therefore return no nodes here. An op that consults `buffer_lengths`
    /// must return exactly those inputs so the runtime can invalidate its
    /// hard-resource validation cache when (and only when) their lengths
    /// change.
    fn resource_buffer_nodes(&self, _inputs: &[NodeIndex]) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }
}
