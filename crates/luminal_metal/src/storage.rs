use crate::{
    arena::ArenaPlan,
    layouts::MetalPlan,
    symbolic::{Bounds, capacity_bytes},
};
use anyhow::{Result, anyhow};
pub(crate) fn plan(plan: &MetalPlan, bounds: &Bounds) -> Result<ArenaPlan> {
    plan_resident(plan, bounds, &Default::default())
}
pub(crate) fn plan_resident(
    plan: &MetalPlan,
    bounds: &Bounds,
    bindings: &luminal::resident::ResidentBindings,
) -> Result<ArenaPlan> {
    // Output slots whose buffer IS a resident input are mutation sinks: the
    // binding put them on the input's buffer, so their writes already land in
    // the arena home and they reserve no pinned staging.
    let device_outputs: std::collections::BTreeSet<usize> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            luminal::bufferize::BufferNode::BufferOutput { slots } => Some(slots),
            _ => None,
        })
        .flatten()
        .filter(|slot| {
            plan.buffers[&slot.buffer]
                .lit
                .is_some_and(|lit| bindings.inputs.contains(&lit))
        })
        .map(|slot| slot.index)
        .collect();
    // A buffer the bindings declared External is the caller's own device
    // memory and reserves no slab range.
    let external_buffers: luminal::prelude::FxHashSet<luminal::bufferize::BufferId> = plan
        .buffers
        .values()
        .filter(|buffer| {
            buffer
                .lit
                .is_some_and(|lit| bindings.externals.contains(&lit))
        })
        .map(|buffer| buffer.id.clone())
        .collect();
    crate::arena::plan_resident_over(
        plan,
        |buffer| capacity_bytes(&buffer.layout, bounds),
        |_| Ok(0),
        bounds
            .len()
            .checked_mul(8)
            .ok_or_else(|| anyhow!("parameter size overflow"))?
            .max(8),
        crate::arena::issue_order(plan)?,
        &bindings.inputs,
        &device_outputs,
        &external_buffers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::MetalBindings;
    use crate::{MetalRuntime, harness_search_options};
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::{arena::ArenaStep, prelude::*, resident::ResidentBindings};

    #[test]
    #[cfg_attr(
        not(target_os = "macos"),
        ignore = "candidate search requires a Metal device"
    )]
    fn resident_ranges_and_mutation_sinks_share_the_arena_across_buckets() {
        let mut graph = Graph::new();
        let weights = graph.tensor(4, DType::F32);
        let state = graph.tensor(4, DType::F32);
        let input = graph.tensor('n', DType::F32);
        let next = state + weights * input.sum(0).expand_dim(0, 4);
        let previous = state.sum(0);
        // `next` writes the state input's storage: the alias is a binding.
        let mut bindings = MetalBindings::dense(&graph.logical, &[previous.id]);
        let home = bindings.buffer_of_input(state.id).unwrap();
        bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
        bindings.output_on(next.id, home);
        let mut runtime =
            MetalRuntime::load_with(&graph, bindings, crate::ops::metal_registry()).unwrap();
        runtime
            .bind_dim_buckets(
                'n',
                vec![
                    luminal::graph::DimBucket::new(1, 1),
                    luminal::graph::DimBucket::new(2, 4),
                ],
            )
            .unwrap();
        // Search executes each bucket at its representative shape. Only the
        // dynamic input varies; weights and mutable state are shared.
        let profiles = [1usize, 3]
            .into_iter()
            .map(|n| {
                (
                    [('n'.into(), n)].into_iter().collect(),
                    [(input.id, vec![1f32; n].into())].into_iter().collect(),
                )
            })
            .collect::<Vec<_>>();
        runtime
            .search_with_profile_inputs(
                &[
                    (weights.id, vec![1f32; 4].into()),
                    (state.id, vec![0f32; 4].into()),
                ]
                .into_iter()
                .collect(),
                &profiles,
                &harness_search_options(),
            )
            .unwrap();
        let weights = runtime.input_buffer(weights.id).unwrap();
        let state = runtime.input_buffer(state.id).unwrap();
        let bindings = ResidentBindings {
            inputs: [weights, state].into_iter().collect(),
            ..Default::default()
        };
        let plans = || {
            runtime
                .bucket_plans()
                .iter()
                .map(|p| (p.plan.clone(), p.ranges.clone()))
                .collect()
        };
        let allocation =
            luminal::resident::allocate(plans(), bindings.clone(), plan_resident, capacity_bytes)
                .unwrap();
        let scratch = allocation
            .plans
            .iter()
            .map(|p| p.storage.slab_bytes)
            .max()
            .unwrap();
        assert!(allocation.bytes > scratch);
        for bucket in &allocation.plans {
            for (id, buffer) in &bucket.plan.buffers {
                if let Some(home) = buffer.lit.and_then(|lit| allocation.homes.get(&lit)) {
                    assert_eq!(bucket.storage.slices[id], home.data);
                    assert!(home.data.offset >= scratch);
                    assert!(
                        !bucket
                            .storage
                            .steps
                            .iter()
                            .any(|s| matches!(s, ArenaStep::Upload { buffer, .. } if buffer == id))
                    );
                }
            }
            assert!(
                bucket.storage.steps.iter().any(
                    |s| matches!(s, ArenaStep::Download { staging, .. } if staging.bytes == 0)
                ),
                "a resident mutation sink does not reserve host staging"
            );
        }
        let mut invalid = bindings;
        invalid
            .inputs
            .insert(runtime.input_buffer(input.id).unwrap());
        assert!(
            luminal::resident::allocate(plans(), invalid, plan_resident, capacity_bytes).is_err(),
            "resident geometry must be static across buckets"
        );
    }
}
