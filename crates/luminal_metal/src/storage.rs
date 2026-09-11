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
        &bindings.feedback.keys().copied().collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalRuntime, harness_search_options};
    use luminal::{arena::ArenaStep, prelude::*, resident::ResidentBindings};

    #[test]
    fn resident_ranges_and_feedback_share_the_arena_across_buckets() {
        let mut graph = Graph::new();
        let weights = graph.tensor(4, DType::F32);
        let state = graph.tensor(4, DType::F32);
        let input = graph.tensor('n', DType::F32);
        let next = (state + weights * input.sum(0).expand_dim(0, 4)).output();
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
        let next = runtime.output_slot_index(next.id).unwrap();
        let bindings = ResidentBindings {
            inputs: [weights, state].into_iter().collect(),
            feedback: [(next, state)].into_iter().collect(),
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
        assert!(allocation.homes[&state].next.is_some());
        assert!(allocation.homes[&weights].next.is_none());
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
                "feedback does not reserve host staging"
            );
        }
        let mut invalid = bindings.clone();
        invalid
            .inputs
            .insert(runtime.input_buffer(input.id).unwrap());
        assert!(
            luminal::resident::allocate(plans(), invalid, plan_resident, capacity_bytes).is_err(),
            "resident geometry must be static across buckets"
        );
        let mut invalid = bindings;
        invalid.feedback.insert(next + 100, state);
        assert!(
            luminal::resident::allocate(plans(), invalid, plan_resident, capacity_bytes).is_err(),
            "two feedback outputs cannot overwrite the same state"
        );
    }
}
