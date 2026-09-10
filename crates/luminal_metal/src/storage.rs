use crate::{
    arena::{ArenaPlan, plan_with_workspace},
    layouts::MetalPlan,
    symbolic::{Bounds, capacity_bytes},
};
use anyhow::{Result, anyhow};
pub(crate) fn plan(plan: &MetalPlan, bounds: &Bounds) -> Result<ArenaPlan> {
    let parameter_bytes = bounds
        .len()
        .checked_mul(8)
        .ok_or_else(|| anyhow!("parameter size overflow"))?
        .max(8);
    plan_with_workspace(
        plan,
        |buffer| capacity_bytes(&buffer.layout, bounds),
        |_| Ok(0),
        parameter_bytes,
    )
}
