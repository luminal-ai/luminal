//! Shared search defaults and representative inputs for every chat backend.
use crate::{Inputs, graph::LlmGraph};
use anyhow::Result;
use luminal::bucketing::BucketSpec;
use luminal::shape::{DimensionBounds, DynMap};

pub const DEFAULT_SEARCH_GENERATIONS: usize = 10;
pub const DEFAULT_SEARCH_POPULATION: usize = 10;
pub const DEFAULT_PREFILL_CHUNK: usize = 128;
pub const PROFILE_CONTEXT: usize = 128;

/// Decode covers one query token. Prefill covers all larger chunks, including
/// a short final chunk, and is profiled at 128 tokens when capacity permits.
pub fn buckets(graph: &LlmGraph) -> Result<Vec<BucketSpec>> {
    let mut queries = vec![(1, 1, 1)];
    if graph.chunk_size > 1 {
        queries.push((
            2,
            graph.chunk_size,
            DEFAULT_PREFILL_CHUNK.min(graph.chunk_size),
        ));
    }
    queries
        .into_iter()
        .map(|(lo, hi, query)| {
            BucketSpec::new(
                DimensionBounds::from_ranges([
                    ('q'.into(), (lo, hi)),
                    ('c'.into(), (1, graph.capacity)),
                ])?,
                [
                    ('q'.into(), query),
                    ('c'.into(), context_representative(graph)),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect()
}

/// Application-owned diagnostics for one independent search.
pub struct BucketPlan<P, O> {
    pub ranges: std::collections::BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: DynMap,
    pub plan: P,
    pub outcome: O,
    pub finalist_rank: usize,
    pub slab_bytes: usize,
}

pub fn context_representative(graph: &LlmGraph) -> usize {
    PROFILE_CONTEXT.min(graph.capacity)
}

/// Build valid positions, gather/scatter maps, last-row indices, and RoPE
/// tables at each representative. Resizing another bucket's payload would
/// leave these dependent values incorrect (notably the last-row index).
pub fn profile_inputs(graph: &LlmGraph) -> Result<Vec<(DynMap, Inputs)>> {
    let context = context_representative(graph);
    buckets(graph)?
        .into_iter()
        .map(|bucket| {
            let query = bucket.profile_dims()[&'q'.into()];
            let dims = [('q'.into(), query), ('c'.into(), context)]
                .into_iter()
                .collect();
            Ok((dims, graph.step_inputs(&vec![0; query], context - query)?))
        })
        .collect()
}
