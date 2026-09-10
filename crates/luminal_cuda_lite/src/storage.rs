//! Capacity layouts overlay every bucket in one device arena. Boundary buffers
//! are private device copies: returned outputs are owned host bytes, so those
//! ranges can also be reused by the next invocation/bucket.
use crate::{
    arena::{ARENA_ALIGN, ArenaPlan, ArenaSlice, plan_arena},
    layouts::CudaPlan,
    symbolic::{Bounds, capacity_bytes},
};
use anyhow::{Result, anyhow, ensure};

#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) struct StoragePlan {
    pub arena: ArenaPlan,
    pub workspace: ArenaSlice,
    pub staging_bytes: usize,
}
pub(crate) fn plan(plan: &CudaPlan, bounds: &Bounds) -> Result<StoragePlan> {
    let mut arena = plan_arena(plan, |b| capacity_bytes(&b.layout, bounds))?;
    let mut end = arena.slab_bytes;
    let aligned = |v: usize| {
        v.checked_add(ARENA_ALIGN - 1)
            .map(|v| v / ARENA_ALIGN * ARENA_ALIGN)
            .ok_or_else(|| anyhow!("arena size overflow"))
    };
    // Stable order makes per-bucket pointer layouts reproducible.
    let mut ids: Vec<_> = arena
        .standalone
        .iter()
        .chain(&arena.donated)
        .cloned()
        .collect();
    ids.sort_by_key(|id| format!("{id:?}"));
    for id in ids {
        let bytes = capacity_bytes(&plan.buffers[&id].layout, bounds)?;
        let offset = aligned(end)?;
        end = offset
            .checked_add(bytes.max(1))
            .ok_or_else(|| anyhow!("arena size overflow"))?;
        arena.slices.insert(id, ArenaSlice { offset, bytes });
    }
    let mut scratch = 0;
    for node in plan.dag.node_weights() {
        if let luminal::bufferize::BufferNode::Compute { op, .. } = node
            && let Some(host) = crate::as_host_op(op.as_ref())
        {
            scratch = scratch.max(host.workspace_bytes(bounds)?);
        }
        if let luminal::bufferize::BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                let buffer = plan
                    .buffers
                    .get(&slot.buffer)
                    .ok_or_else(|| anyhow!("unknown output buffer"))?;
                ensure!(
                    buffer.freed_by == luminal::layout_ir::FreedBy::Caller,
                    "output slot {} has NON-ESCAPING buffer {} (FreedBy::Program)",
                    slot.index,
                    buffer.label
                );
            }
        }
    }
    let workspace = ArenaSlice {
        offset: aligned(end)?,
        bytes: scratch,
    };
    arena.slab_bytes = workspace
        .offset
        .checked_add(scratch)
        .ok_or_else(|| anyhow!("arena workspace overflow"))?
        .max(1);
    let prefix = aligned(
        bounds
            .len()
            .checked_mul(8)
            .ok_or_else(|| anyhow!("parameter size overflow"))?
            .max(8),
    )?;
    for slice in arena.slices.values_mut() {
        slice.offset = slice
            .offset
            .checked_add(prefix)
            .ok_or_else(|| anyhow!("arena offset overflow"))?;
    }
    let workspace = ArenaSlice {
        offset: workspace
            .offset
            .checked_add(prefix)
            .ok_or_else(|| anyhow!("arena offset overflow"))?,
        bytes: workspace.bytes,
    };
    arena.slab_bytes = arena
        .slab_bytes
        .checked_add(prefix)
        .ok_or_else(|| anyhow!("arena size overflow"))?;
    let mut staging_bytes = (bounds.len() * 8).max(8);
    let mut reserve = |bytes: usize| -> Result<()> {
        staging_bytes = staging_bytes
            .checked_add(bytes.max(1))
            .ok_or_else(|| anyhow!("staging size overflow"))?;
        Ok(())
    };
    for (id, buffer) in &plan.buffers {
        if buffer.lit.is_some() {
            reserve(arena.slices[id].bytes)?;
        }
    }
    for node in plan.dag.node_weights() {
        if let luminal::bufferize::BufferNode::BufferOutput { slots } = node {
            for slot in slots {
                reserve(arena.slices[&slot.buffer].bytes)?;
            }
        }
    }
    Ok(StoragePlan {
        arena,
        workspace,
        staging_bytes,
    })
}
