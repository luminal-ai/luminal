use crate::Inputs;
use anyhow::Result;

/// The chat loop uses this contract regardless of the model or device.
pub trait Backend {
    fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<()>;
}

#[cfg(any(feature = "cuda_lite", feature = "metal"))]
mod gpu;
#[cfg(any(feature = "cuda_lite", feature = "metal"))]
pub use gpu::{CompileOptions, GpuBackend, harness_search_options};
