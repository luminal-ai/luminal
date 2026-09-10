//! Opt-in device-resident input boundaries and output-to-input feedback.
//! Names are runtime buffer/slot IDs; this layer knows nothing about models.
use crate::{
    arena::{ArenaPlan, ArenaSlice},
    layouts::CudaPlan,
    symbolic::Bounds,
};
use anyhow::{Result, anyhow, ensure};
use luminal::bufferize::BufferNode;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Default)]
pub struct ResidentBindings {
    pub inputs: BTreeSet<i64>,
    /// Output slot -> resident input BufferLit. Every destination is unique.
    pub feedback: BTreeMap<usize, i64>,
}
/// Session-lived storage; `next` snapshots feedback before old state is retired.
#[derive(Clone, Debug)]
pub struct ResidentHome {
    pub data: ArenaSlice,
    pub next: Option<ArenaSlice>,
    pub dtype: luminal::dtype::PlanDtype,
    pub shape: Vec<usize>,
}
pub struct ResidentPlan {
    pub plan: CudaPlan,
    pub storage: ArenaPlan,
    pub bounds: Bounds,
}
pub struct ResidentAllocation {
    pub plans: Vec<ResidentPlan>,
    pub homes: BTreeMap<i64, ResidentHome>,
    pub bytes: usize,
    pub feedback: BTreeMap<usize, i64>,
}
/// One physical planning implementation for native GPU executors. Resident
/// ranges are identical across buckets; temporaries overlay the largest plan.
pub fn allocate(
    plans: Vec<(CudaPlan, Bounds)>,
    bindings: ResidentBindings,
) -> Result<ResidentAllocation> {
    let mut installed = plans
        .into_iter()
        .map(|(plan, bounds)| {
            let storage = if bindings.inputs.is_empty() {
                crate::storage::plan(&plan, &bounds)?
            } else {
                crate::storage::plan_resident(&plan, &bounds, &bindings)?
            };
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
    let mut feedback_inputs = BTreeSet::new();
    for lit in bindings.feedback.values() {
        ensure!(
            bindings.inputs.contains(lit) && feedback_inputs.insert(*lit),
            "invalid/duplicate resident feedback destination {lit}"
        );
    }
    for &lit in &bindings.inputs {
        let mut geometry = None;
        for bucket in &installed {
            for buffer in bucket.plan.buffers.values().filter(|b| b.lit == Some(lit)) {
                ensure!(
                    buffer.access == luminal::layout_ir::Access::ReadOnly
                        && buffer.freed_by == luminal::layout_ir::FreedBy::Caller,
                    "resident input {lit} must be read-only and caller-owned"
                );
                ensure!(
                    buffer
                        .layout
                        .has::<luminal::layouts::RightMajorContiguousElementLayout>(),
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
                let size = crate::symbolic::capacity_bytes(&buffer.layout, &bucket.bounds)?;
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
        let next = if feedback_inputs.contains(&lit) {
            let next = ArenaSlice {
                offset: bytes,
                bytes: size,
            };
            bytes = bytes
                .checked_add(next.reserved())
                .ok_or_else(|| anyhow!("feedback size overflow"))?;
            Some(next)
        } else {
            None
        };
        residents.insert(
            lit,
            ResidentHome {
                data,
                next,
                dtype,
                shape,
            },
        );
    }
    for bucket in &mut installed {
        for (id, buffer) in &bucket.plan.buffers {
            if let Some(home) = buffer.lit.and_then(|lit| residents.get(&lit)) {
                bucket.storage.slices.insert(id.clone(), home.data);
            }
        }
        let mut seen = BTreeSet::new();
        for node in bucket.plan.dag.node_weights() {
            if let BufferNode::BufferOutput { slots } = node {
                for slot in slots {
                    if let Some(lit) = bindings.feedback.get(&slot.index) {
                        let home = &residents[lit];
                        ensure!(
                            slot.layout
                                .has::<luminal::layouts::RightMajorContiguousElementLayout>()
                                && slot.layout.literal_extents().as_ref() == Some(&home.shape)
                                && slot.layout.dtype == Some(home.dtype),
                            "feedback slot {} must match its static contiguous input boundary",
                            slot.index
                        );
                        ensure!(
                            crate::symbolic::capacity_bytes(
                                &bucket.plan.buffers[&slot.buffer].layout,
                                &bucket.bounds
                            )? == home.data.bytes,
                            "feedback backing span differs from input"
                        );
                        seen.insert(slot.index);
                    }
                }
            }
        }
        ensure!(
            seen.len() == bindings.feedback.len(),
            "feedback output absent from a bucket"
        );
    }
    Ok(ResidentAllocation {
        plans: installed,
        homes: residents,
        bytes,
        feedback: bindings.feedback,
    })
}
