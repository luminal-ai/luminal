pub mod dyn_backend;
pub mod kernel;
mod memory_analysis;
pub mod runtime;

#[cfg(test)]
mod tests;

pub use runtime::{WebGpuBuffer, WebGpuRuntime};

// Re-export kernel ops
pub use kernel::WebGpuOps;
