use luminal::prelude::*;
#[cfg(feature = "cuda")]
use luminal_cuda_lite::cudarc::driver::{CudaContext, CudaStream};
#[cfg(feature = "cuda")]
use luminal_cuda_lite::runtime::CudaRuntime;
use rustc_hash::FxHashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Enum wrapper for runtime backends allowing runtime selection.
pub enum RuntimeBackend {
    Native(NativeRuntime),
    #[cfg(feature = "cuda")]
    Cuda(Box<CudaRuntime>),
}

impl RuntimeBackend {
    /// Set input data for a tensor node.
    pub fn set_data(&mut self, node: NodeIndex, data: Vec<f32>) {
        match self {
            RuntimeBackend::Native(rt) => rt.set_data(node, data),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.set_data(node, data),
        }
    }

    /// Execute the compiled graph.
    pub fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        match self {
            RuntimeBackend::Native(rt) => rt.execute(dyn_map),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.execute(dyn_map),
        }
    }

    /// Get output data from a tensor node.
    pub fn get_f32(&self, node: NodeIndex) -> Vec<f32> {
        match self {
            RuntimeBackend::Native(rt) => rt.get_f32(node).to_vec(),
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(rt) => rt.get_f32(node),
        }
    }

    /// Get the name of the active backend.
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeBackend::Native(_) => "native",
            #[cfg(feature = "cuda")]
            RuntimeBackend::Cuda(_) => "cuda",
        }
    }
}

// ============================================================================
// Two-phase initialization for CUDA (required because profiling executes graph)
// ============================================================================

/// Prepare CUDA runtime: build search space and create runtime, but don't search yet.
/// Returns the unoptimized runtime that can have data set on it.
///
/// Use this with `finalize_cuda` for proper CUDA initialization:
/// 1. Call `prepare_cuda` to get the runtime
/// 2. Set data on the runtime using `rt.set_data(node_id, data)`
/// 3. Call `finalize_cuda` to run profiling with data available
///

#[cfg(feature = "cuda")]
pub fn prepare_cuda(context: &mut Graph) -> Result<(CudaRuntime, Arc<CudaStream>), String> {
    let cuda_ctx =
        CudaContext::new(0).map_err(|e| format!("Failed to init CUDA context: {}", e))?;
    let stream = cuda_ctx.default_stream();
    context.build_search_space::<CudaRuntime>();
    let rt = CudaRuntime::initialize(stream.clone());
    Ok((rt, stream))
}

/// Finalize CUDA runtime: run search with data already set.
#[cfg(feature = "cuda")]
pub fn finalize_cuda(context: &mut Graph, rt: CudaRuntime) -> RuntimeBackend {
    let optimized_rt = context.search(rt, 10);
    RuntimeBackend::Cuda(Box::new(optimized_rt))
}

/// Initialize a native (CPU) runtime using single-phase approach.
/// NativeRuntime validates Input nodes, so we must search first, then set data.
pub fn initialize_native(context: &mut Graph) -> Result<RuntimeBackend, String> {
    context.build_search_space::<NativeRuntime>();
    let rt = context.search(NativeRuntime::default(), 10);
    Ok(RuntimeBackend::Native(rt))
}
