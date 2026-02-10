use std::{fmt::Debug, sync::Arc};

use crate::cudarc::driver::{CudaSlice, CudaStream};
use luminal::{op::EgglogOp, prelude::*};
mod cublas;

pub type Ops = (cublas::CuBlasSgemmV2,);

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
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()>;

    /// Returns the output buffer size in elements.
    /// Return 0 if this op doesn't have a single output buffer (e.g., CudaGraphOp).
    fn output_size(&self) -> Expression;

    /// Returns additional nodes (beyond graph edges) that this op needs buffers for.
    ///
    /// For most ops, this returns empty (buffers determined by graph edges).
    /// For CudaGraphOp, this returns all internal kernel nodes.
    fn extra_buffer_nodes(&self) -> Vec<NodeIndex> {
        vec![]
    }

    /// Returns buffer size requirements for extra nodes (node -> size in elements).
    ///
    /// Called during buffer allocation to ensure all required buffers exist.
    /// For CudaGraphOp, this returns sizes for all internal kernel output buffers.
    fn extra_buffer_sizes(&self) -> FxHashMap<NodeIndex, Expression> {
        FxHashMap::default()
    }

    /// Returns the name of this host op for stats reporting, or None if not reportable.
    fn stats_name(&self) -> Option<&'static str> {
        None
    }
}
