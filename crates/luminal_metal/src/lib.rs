//! Metal execution for the native logical graph and bufferized core IR.
//! Load a graph, bind shapes, search, stage inputs, execute, and read outputs.
pub mod arena;
pub mod bindings;
pub mod finalists;
pub mod heuristic;
pub mod host_buffer;
pub mod kernels;
pub mod lattice;
pub mod layouts;
pub mod op;
pub mod ops;
pub mod runtime;
mod saturation;
pub mod search;
mod storage;
pub mod symbolic;
pub use luminal::extraction as extractor;
#[cfg(target_os = "macos")]
pub mod device;
#[cfg(target_os = "macos")]
pub mod profile;
pub use bindings::MetalBindings;
pub use host_buffer::HostBuffer;
pub use kernels::KernelOp;
pub use layouts::MetalPlan;
#[cfg(target_os = "macos")]
pub use metal::{Buffer, Device, MTLResourceOptions};
#[cfg(target_os = "macos")]
pub use objc::rc::autoreleasepool;
pub use op::{MetalOpInterface, as_kernel_op};
pub use ops::{RegisteredOp, metal_registry, metal_registry_filtered};
pub use runtime::MetalRuntime;
pub use search::{CompileOptions, Evaluator, SearchOutcome, harness_search_options};
/// A view can be claimed without a kernel only when its effects prove it folds.
pub fn plan_transparent(op: &dyn luminal::layout_ir::LayoutIrOp) -> bool {
    use luminal::layout_ir::{AliasInfo, Sharing};
    op.alias_info()
        == [AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
        && !op.operand_reads_memory(0)
        && !op.result_writes_memory(0)
        && op.to_dps().is_none()
}
pub fn metal_allow_list() -> Vec<&'static str> {
    MetalRuntime::allow_list()
        .into_iter()
        .map(|s| s.strip_prefix("LayoutTensorOp").unwrap_or(s))
        .collect()
}
