use crate::{
    arena::ArenaPlan,
    layouts::MetalPlan,
    symbolic::{Bounds, capacity_bytes},
};
use anyhow::{Result, anyhow};
pub(crate) fn plan_storage(
    plan: &MetalPlan,
    bounds: &Bounds,
    bindings: &std::collections::BTreeSet<i64>,
) -> Result<ArenaPlan> {
    // A buffer the bindings declared External is the caller's own device
    // memory and reserves no slab range.
    let external_buffers: luminal::prelude::FxHashSet<luminal::bufferize::BufferId> = plan
        .buffers
        .values()
        .filter(|buffer| buffer.lit.is_some_and(|lit| bindings.contains(&lit)))
        .map(|buffer| buffer.id.clone())
        .collect();
    crate::arena::plan_external_over(
        plan,
        |buffer| capacity_bytes(&buffer.layout, bounds),
        |_| Ok(0),
        bounds
            .len()
            .checked_mul(8)
            .ok_or_else(|| anyhow!("parameter size overflow"))?
            .max(8),
        crate::arena::issue_order(plan)?,
        &external_buffers,
    )
}
