//! Shared resident input allocation with CUDA scratch sizing.
pub use luminal::resident::{ResidentBindings, ResidentHome};
pub type ResidentPlan = luminal::resident::ResidentPlan<crate::symbolic::Bounds>;
pub type ResidentAllocation = luminal::resident::ResidentAllocation<crate::symbolic::Bounds>;
/// `bindings` states which boundary buffers the arena keeps between
/// executions and which are the caller's own device memory (and so get no
/// slab range at all).
pub fn allocate(
    plans: Vec<(crate::layouts::CudaPlan, crate::symbolic::Bounds)>,
    bindings: ResidentBindings,
) -> anyhow::Result<ResidentAllocation> {
    luminal::resident::allocate(
        plans,
        bindings,
        crate::storage::plan_resident,
        crate::symbolic::capacity_bytes,
    )
}
