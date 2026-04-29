pub mod dyn_backend;
pub mod kernel;
pub mod runtime;

#[cfg(test)]
mod tests;

pub use metal::{Buffer, Device, MTLResourceOptions};
pub use objc::rc::autoreleasepool;
pub use runtime::MetalRuntime;

// Re-export kernel ops
pub use kernel::MetalOps;
