//! Opt-in device-resident input boundaries. Names are runtime buffer/slot
//! IDs; this layer knows nothing about models. A resident input may be
//! MUTATED in place: a `.output_into()` output shares the input's BufferId,
//! so its writes land in the home and need no host readback — the state is
//! still SSA, only its boundary storage is shared.
use crate::arena::{ArenaPlan, ArenaSlice};
use crate::{bufferize::BufferIrGraph, layouts::DecodedLayout};
use anyhow::{Result, anyhow, ensure};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Default)]
pub struct ResidentBindings {
    pub inputs: BTreeSet<i64>,
}
/// Session-lived storage for one resident input boundary.
#[derive(Clone, Debug)]
pub struct ResidentHome {
    pub data: ArenaSlice,
    pub dtype: crate::dtype::PlanDtype,
    pub shape: Vec<usize>,
}
pub struct ResidentPlan<B> {
    pub plan: BufferIrGraph<DecodedLayout>,
    pub storage: ArenaPlan,
    pub bounds: B,
}
pub struct ResidentAllocation<B> {
    pub plans: Vec<ResidentPlan<B>>,
    pub homes: BTreeMap<i64, ResidentHome>,
    pub bytes: usize,
}
/// One physical planning implementation for native GPU executors. Resident
/// ranges are identical across buckets; temporaries overlay the largest plan.
pub fn allocate<B>(
    plans: Vec<(BufferIrGraph<DecodedLayout>, B)>,
    bindings: ResidentBindings,
    plan_storage: impl Fn(&BufferIrGraph<DecodedLayout>, &B, &ResidentBindings) -> Result<ArenaPlan>,
    capacity_bytes: impl Fn(&DecodedLayout, &B) -> Result<usize>,
) -> Result<ResidentAllocation<B>> {
    let mut installed = plans
        .into_iter()
        .map(|(plan, bounds)| {
            let storage = plan_storage(&plan, &bounds, &bindings)?;
            Ok(ResidentPlan {
                plan,
                storage,
                bounds,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut bytes = installed
        .iter()
        .map(|p| p.storage.slab_bytes)
        .max()
        .unwrap_or(1);
    let mut residents = BTreeMap::new();
    for &lit in &bindings.inputs {
        let mut geometry = None;
        for bucket in &installed {
            for buffer in bucket.plan.buffers.values().filter(|b| b.lit == Some(lit)) {
                ensure!(
                    buffer.freed_by == crate::layout_ir::FreedBy::Caller,
                    "resident input {lit} must be caller-owned"
                );
                ensure!(
                    buffer
                        .layout
                        .has::<crate::layouts::RightMajorContiguousElementLayout>(),
                    "resident input {lit} needs contiguous row-major storage"
                );
                let shape = buffer
                    .layout
                    .literal_extents()
                    .ok_or_else(|| anyhow!("resident input {lit} must have static dimensions"))?;
                let dtype = buffer
                    .layout
                    .dtype
                    .ok_or_else(|| anyhow!("resident dtype missing"))?;
                let size = capacity_bytes(&buffer.layout, &bucket.bounds)?;
                let current = (shape, dtype, size);
                if let Some(previous) = &geometry {
                    ensure!(
                        previous == &current,
                        "resident input {lit} changes geometry across buckets"
                    );
                }
                geometry = Some(current);
            }
        }
        let (shape, dtype, size) =
            geometry.ok_or_else(|| anyhow!("resident input {lit} is absent from plans"))?;
        bytes = bytes
            .checked_add(crate::arena::ARENA_ALIGN - 1)
            .ok_or_else(|| anyhow!("resident alignment overflow"))?
            / crate::arena::ARENA_ALIGN
            * crate::arena::ARENA_ALIGN;
        let data = ArenaSlice {
            offset: bytes,
            bytes: size,
        };
        bytes = bytes
            .checked_add(data.reserved())
            .ok_or_else(|| anyhow!("resident size overflow"))?;
        residents.insert(lit, ResidentHome { data, dtype, shape });
    }
    for bucket in &mut installed {
        for (id, buffer) in &bucket.plan.buffers {
            if let Some(home) = buffer.lit.and_then(|lit| residents.get(&lit)) {
                bucket.storage.slices.insert(id.clone(), home.data);
            }
        }
    }
    Ok(ResidentAllocation {
        plans: installed,
        homes: residents,
        bytes,
    })
}
