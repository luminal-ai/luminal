//! Shared search defaults and representative inputs for every chat backend.
use crate::{Inputs, graph::LlmGraph};
use anyhow::Result;
use luminal::{graph::DimBucket, shape::DynMap};

pub const DEFAULT_SEARCH_GENERATIONS: usize = 10;
pub const DEFAULT_SEARCH_POPULATION: usize = 10;
pub const DEFAULT_PREFILL_CHUNK: usize = 128;
pub const PROFILE_CONTEXT: usize = 128;

/// Decode covers one query token. Prefill covers all larger chunks, including
/// a short final chunk, and is profiled at 128 tokens when capacity permits.
pub fn query_buckets(graph: &LlmGraph) -> Vec<DimBucket> {
    let mut buckets = vec![DimBucket::new(1, 1)];
    if graph.chunk_size > 1 {
        buckets.push(
            DimBucket::new(2, graph.chunk_size)
                .representative(DEFAULT_PREFILL_CHUNK.min(graph.chunk_size)),
        );
    }
    buckets
}

pub fn context_representative(graph: &LlmGraph) -> usize {
    PROFILE_CONTEXT.min(graph.capacity)
}

/// Build valid positions, gather/scatter maps, last-row indices, and RoPE
/// tables at each representative. Resizing another bucket's payload would
/// leave these dependent values incorrect (notably the last-row index).
pub fn profile_inputs(graph: &LlmGraph) -> Result<Vec<(DynMap, Inputs)>> {
    let context = context_representative(graph);
    query_buckets(graph)
        .into_iter()
        .map(|bucket| {
            let query = bucket.representative_value();
            let dims = [('q'.into(), query), ('c'.into(), context)]
                .into_iter()
                .collect();
            Ok((dims, graph.step_inputs(&vec![0; query], context - query)?))
        })
        .collect()
}
