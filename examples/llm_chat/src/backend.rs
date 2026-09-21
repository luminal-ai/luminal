//! Runtime adapters for the shared chat session. Each adapter owns its
//! runtime's binding statement and device representation.
use crate::Inputs;
use anyhow::Result;

pub trait Backend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<()>;
}

#[cfg(feature = "luminal_cuda_lite")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;

// Both adapters may be built for tests. The CLI requires exactly one
// execution backend; CUDA wins this type alias only to keep all-features
// builds valid.
#[cfg(any(
    feature = "cuda_lite",
    all(feature = "luminal_cuda_lite", not(feature = "metal"))
))]
pub use cuda::CudaBackend as GpuBackend;
#[cfg(any(
    feature = "cuda_lite",
    all(feature = "luminal_cuda_lite", not(feature = "metal"))
))]
pub use luminal_cuda_lite::CompileOptions;
#[cfg(all(feature = "metal", not(feature = "cuda_lite")))]
pub use luminal_metal::CompileOptions;
#[cfg(all(feature = "metal", not(feature = "cuda_lite")))]
pub use metal::MetalBackend as GpuBackend;
