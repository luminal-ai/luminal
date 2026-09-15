//! Shared resident input allocation with CUDA scratch sizing.
pub use luminal::resident::{ResidentBindings, ResidentHome};
pub type ResidentPlan = luminal::resident::ResidentPlan<crate::symbolic::Bounds>;
pub type ResidentAllocation = luminal::resident::ResidentAllocation<crate::symbolic::Bounds>;
/// `external_outputs` selects whether final output buffers are caller-owned at
/// execution time (and so excluded from the slab) or arena-resident.
pub fn allocate(
    plans: Vec<(crate::layouts::CudaPlan, crate::symbolic::Bounds)>,
    bindings: ResidentBindings,
    external_outputs: bool,
) -> anyhow::Result<ResidentAllocation> {
    luminal::resident::allocate(
        plans,
        bindings,
        move |plan, bounds, bindings| {
            crate::storage::plan_resident(plan, bounds, bindings, external_outputs)
        },
        crate::symbolic::capacity_bytes,
    )
}
