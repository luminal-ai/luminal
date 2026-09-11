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
    // Output slots whose buffer IS a resident input are mutation sinks:
    // `.output_into()` pinned them to the input's buffer, so their writes
    // already land in the arena home and they reserve no pinned staging.
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
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalRuntime, harness_search_options};
    use luminal::{arena::ArenaStep, prelude::*, resident::ResidentBindings};

    #[test]
    fn resident_ranges_and_mutation_sinks_share_the_arena_across_buckets() {
        let mut graph = Graph::new();
        let weights = graph.tensor(4, DType::F32);
        let state = graph.tensor(4, DType::F32);
        let input = graph.tensor('n', DType::F32);
        (state + weights * input.sum(0).expand_dim(0, 4)).output_into(&state);
        state.sum(0).output();
        let mut runtime = MetalRuntime::load(&graph).unwrap();
        runtime
            .bind_dim_buckets(
                'n',
                vec![
                    luminal::graph::DimBucket::new(1, 1),
                    luminal::graph::DimBucket::new(2, 4),
                ],
            )
            .unwrap();
        runtime
            .search(&Default::default(), &harness_search_options())
            .unwrap();
        let weights = runtime.input_buffer(weights.id).unwrap();
        let state = runtime.input_buffer(state.id).unwrap();
        let bindings = ResidentBindings {
            inputs: [weights, state].into_iter().collect(),
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
