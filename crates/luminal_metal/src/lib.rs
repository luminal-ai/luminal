#[cfg(not(feature = "native"))]
pub mod dyn_backend;
#[cfg(not(feature = "native"))]
pub mod kernel;
#[cfg(not(feature = "native"))]
mod memory_analysis;
#[cfg(not(feature = "native"))]
pub mod runtime;

#[cfg(test)]
#[cfg(not(feature = "native"))]
mod tests;

#[cfg(not(feature = "native"))]
pub use metal::{Buffer, Device, MTLResourceOptions};
#[cfg(not(feature = "native"))]
pub use objc::rc::autoreleasepool;
#[cfg(not(feature = "native"))]
pub use runtime::MetalRuntime;

// Re-export kernel ops
#[cfg(not(feature = "native"))]
pub use kernel::MetalOps;

#[cfg(feature = "native")]
pub mod native;
#[cfg(all(feature = "native", target_os = "macos"))]
pub use native::MetalRuntime;
