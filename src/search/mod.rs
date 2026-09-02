//! Everything after egglog saturation.
//!
//! Core stops at a [`SearchSpace`]: one saturated e-graph per dynamic-dim
//! bucket combination. Choosing a program from it is the runtime's job
//! ([`crate::op::Runtime::compile`]); this module is the toolkit a runtime
//! can build that choice from, and core never invokes any of it:
//!
//! - [`extract_one`] / [`LlirExtractor`]: extraction. A runtime that does
//!   not rank candidates (the reference runtime) needs nothing else.
//! - [`GeneticSearch`]: Luminal's genetic search as a pull-style state
//!   machine. The runtime asks for the next candidate, evaluates it however
//!   it likes (profile it, statically cost it, look it up in a cache), and
//!   reports the [`Outcome`]. The state machine owns generations, mutation,
//!   dedup, budgets, and progress output.
//! - [`Finalists`]: lazy re-extraction of ranked genomes into deployment
//!   graphs under a hard filter.
//! - [`BucketLattice`]: best-first selection of one finalist per bucket
//!   under an aggregate constraint.
//! - [`genetic_search`]: closure sugar composing the three above into the
//!   stock strategy, for runtimes with nothing special to do between steps.

pub mod diagnostics;
pub mod finalist;
pub mod genetic;
pub mod lattice;
pub mod packed;
#[cfg(test)]
mod tests;
pub mod unroll;

use std::sync::Arc;

use rand::RngCore;
use rustc_hash::FxHashMap;

use crate::egglog_utils::SerializedEGraph;
pub use crate::egglog_utils::{IndexedChoiceSet, LlirExtractor};
pub use crate::graph::{BucketLLIR, BucketLLIRRef, DimBucket};
use crate::graph::{CompileOptions, LLIRGraph};
use crate::op::{EgglogOp, LLIROp};
use crate::shape::{DynDimIntervals, DynMap, Symbol};
pub use finalist::{Finalist, FinalistValidator, Finalists, PendingFinalist};
pub use genetic::{Candidate, CandidateId, GeneticSearch, Outcome, Ranked};
pub use lattice::{AggregateFn, BucketLattice, BucketSet};
pub use packed::{LlirFingerprint, PackedLLIRGraph};
pub use unroll::{collapse_loops_to_first_iter, unroll_loops_in_llir, unroll_packed_llir};

pub struct SelectedProgram {
    pub bucket_indices: DynMap,
    pub representative_dyn_map: DynMap,
    pub genome: IndexedChoiceSet,
    pub llir: LLIRGraph,
}

impl SelectedProgram {
    pub fn into_bucket_llir(self) -> BucketLLIR {
        (self.bucket_indices, self.representative_dyn_map, self.llir)
    }
}

/// What core hands the runtime: the saturated e-graph of every bucket
/// combination plus everything needed to extract programs from them.
#[derive(Debug)]
pub struct SearchSpace {
    /// One entry per bucket combination; exactly one when unbucketed.
    pub buckets: Vec<BucketSearchSpace>,
    /// Registered egglog ops (backend ops plus HLIR), used for extraction.
    pub ops: Vec<Arc<Box<dyn EgglogOp>>>,
    /// [`crate::graph::Graph::custom_ops`] resolved to LLIR at build time,
    /// indexed by custom-op id.
    pub custom_ops: Vec<LLIROp>,
    /// Bucket definitions the space was built with. Empty when unbucketed.
    pub dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
}

/// The saturated e-graph for one bucket combination.
#[derive(Debug)]
pub struct BucketSearchSpace {
    pub egraph: SerializedEGraph,
    /// dim → index into `SearchSpace::dim_buckets[dim]`. Empty when
    /// unbucketed.
    pub bucket_indices: DynMap,
    /// Dynamic-dim intervals saturation assumed for this bucket.
    pub intervals: DynDimIntervals,
}

/// One bucket of a [`SearchSpace`] together with the dynamic-dim values a
/// search should treat as representative of it.
#[derive(Clone)]
pub struct BucketContext<'a> {
    pub space: &'a SearchSpace,
    pub index: usize,
    /// The base dyn map (after `search_dims`) with this bucket's
    /// representative values applied.
    pub representative_dyn_map: DynMap,
}

impl<'a> BucketContext<'a> {
    pub fn bucket(&self) -> &'a BucketSearchSpace {
        &self.space.buckets[self.index]
    }

    pub fn egraph(&self) -> &'a SerializedEGraph {
        &self.bucket().egraph
    }

    pub fn bucket_indices(&self) -> &'a DynMap {
        &self.bucket().bucket_indices
    }

    pub fn dim_buckets(&self) -> &'a FxHashMap<Symbol, Vec<DimBucket>> {
        &self.space.dim_buckets
    }

    pub fn is_bucketed(&self) -> bool {
        !self.space.dim_buckets.is_empty()
    }

    /// `(bucket index, bucket count)` for progress display when bucketed.
    pub fn progress(&self) -> Option<(usize, usize)> {
        self.is_bucketed()
            .then_some((self.index, self.space.buckets.len()))
    }

    /// Human-readable label such as `s=1, c=[1,4096]@512`.
    pub fn label(&self) -> String {
        format_bucket_label(&self.space.dim_buckets, self.bucket_indices())
    }

    /// The representative dyn map with `options.profile_dims` applied — the
    /// values a profiling search should execute candidates with.
    pub fn profile_dyn_map(&self, options: &CompileOptions) -> DynMap {
        let mut profile_dyn_map = self.representative_dyn_map.clone();
        for (&dim, &value) in &options.profile_dims {
            profile_dyn_map.insert(dim, value);
        }
        profile_dyn_map
    }

    /// Borrowed view of a selected LLIR for this bucket.
    pub fn llir_ref<'l>(&'l self, llir: &'l LLIRGraph) -> BucketLLIRRef<'l> {
        BucketLLIRRef {
            bucket_indices: self.bucket_indices(),
            representative_dyn_map: &self.representative_dyn_map,
            llir,
        }
    }
}

impl SearchSpace {
    /// One context per bucket, in bucket order. Representative dyn maps are
    /// derived here, at search time, so `base_dyn_map` (the graph's dyn map
    /// after `search_dims`) provides the values for unbucketed dimensions and
    /// bucket representatives override it per bucket.
    pub fn bucket_contexts(&self, base_dyn_map: &DynMap) -> Vec<BucketContext<'_>> {
        self.buckets
            .iter()
            .enumerate()
            .map(|(index, bucket)| {
                let mut representative_dyn_map = base_dyn_map.clone();
                let mut dims: Vec<_> = bucket.bucket_indices.iter().collect();
                dims.sort_by_key(|(dim, _)| **dim);
                for (dim, &bucket_idx) in dims {
                    representative_dyn_map.insert(
                        *dim,
                        self.dim_buckets[dim][bucket_idx].representative_value(),
                    );
                }
                BucketContext {
                    space: self,
                    index,
                    representative_dyn_map,
                }
            })
            .collect()
    }
}

/// Cartesian product of bucket indices over all bucketed dims, in a stable
/// order (dims sorted by name, earlier dims outermost). One empty map when
/// there are no buckets.
pub fn bucket_index_combinations(dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>) -> Vec<DynMap> {
    let mut dims: Vec<(Symbol, &Vec<DimBucket>)> =
        dim_buckets.iter().map(|(c, b)| (*c, b)).collect();
    dims.sort_by_key(|(c, _)| *c);

    let mut combos: Vec<DynMap> = vec![FxHashMap::default()];
    for (dim, buckets) in &dims {
        let mut new_combos = Vec::new();
        for existing in &combos {
            for bucket_idx in 0..buckets.len() {
                let mut indices = existing.clone();
                indices.insert(*dim, bucket_idx);
                new_combos.push(indices);
            }
        }
        combos = new_combos;
    }
    combos
}

/// Format a human-readable label for a bucket combination.
pub fn format_bucket_label(
    dim_buckets: &FxHashMap<Symbol, Vec<DimBucket>>,
    bucket_indices: &DynMap,
) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut dims: Vec<_> = bucket_indices.iter().collect();
    dims.sort_by_key(|(c, _)| **c);
    for (dim, &idx) in dims {
        let bucket = &dim_buckets[dim][idx];
        if bucket.min == bucket.max {
            parts.push(format!("{}={}", dim, bucket.min));
        } else {
            parts.push(format!(
                "{}=[{},{}]@{}",
                dim,
                bucket.min,
                bucket.max,
                bucket.representative_value()
            ));
        }
    }
    parts.join(", ")
}

/// Extract one valid deployment graph from a bucket: a cycle-repaired random
/// choice set, extracted and fully unrolled. Every extractable program is
/// semantically equivalent by contract, so this is a complete "search" for a
/// runtime that has nothing to rank on.
pub fn extract_one(
    space: &SearchSpace,
    ctx: &BucketContext<'_>,
    rng: &mut dyn RngCore,
) -> LLIRGraph {
    extract_one_selected(space, ctx, rng).llir
}

pub fn extract_one_selected(
    space: &SearchSpace,
    ctx: &BucketContext<'_>,
    rng: &mut dyn RngCore,
) -> SelectedProgram {
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let genome = extractor.random_indexed_choice(rng);
    let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &space.custom_ops));
    SelectedProgram {
        bucket_indices: ctx.bucket_indices().clone(),
        representative_dyn_map: ctx.representative_dyn_map.clone(),
        genome,
        llir,
    }
}

/// The stock strategy: genetic search in every bucket, lazy finalists under
/// `validate_finalist`, and best-first bucket-set selection under
/// `validate_set`. Returns the selected LLIR per bucket; the caller loads it.
///
/// `state` is threaded through the closures so a runtime can pass `self`
/// without three closures competing for one mutable borrow. Panics inside
/// `evaluate`, `validate_finalist`, and `validate_set` are caught and
/// treated as rejections, as the search always has.
///
/// Panics when no viable program (or bucket set) exists, with the same
/// diagnostics the search has always produced.
#[allow(clippy::too_many_arguments)]
pub fn genetic_search<S, M>(
    space: &SearchSpace,
    dyn_map: &DynMap,
    options: &CompileOptions,
    rng: &mut dyn RngCore,
    state: &mut S,
    mut evaluate: impl FnMut(&mut S, &mut Candidate<M>, &BucketContext<'_>) -> Outcome<M>,
    mut validate_finalist: impl FnMut(
        &mut S,
        &PendingFinalist<M>,
        &BucketContext<'_>,
    ) -> Result<(), String>,
    mut validate_set: impl FnMut(&mut S, &[BucketLLIRRef<'_>]) -> Result<(), String>,
    aggregate: impl Fn(&[M]) -> M,
) -> Vec<BucketLLIR>
where
    M: PartialOrd + Clone + std::fmt::Debug,
{
    let search_started_at = std::time::Instant::now();
    let contexts = space.bucket_contexts(dyn_map);
    let mut finalists = Vec::with_capacity(contexts.len());
    for ctx in &contexts {
        let mut search = GeneticSearch::new(space, ctx, options, search_started_at);
        while let Some(mut candidate) = search.next_candidate(rng) {
            let outcome = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                evaluate(state, &mut candidate, ctx)
            })) {
                Ok(outcome) => outcome,
                Err(payload) => {
                    crate::mask_events::CANDIDATE_PANIC
                        .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
                    diagnostics::dump_failed_candidate(&candidate, &ctx.representative_dyn_map);
                    Outcome::Rejected("candidate compile panicked".to_string())
                }
            };
            search.report(candidate, outcome);
        }
        finalists.push(Finalists::new(
            search.into_ranked(),
            space,
            ctx,
            options,
            search_started_at,
        ));
    }

    let mut lattice = BucketLattice::new(finalists, aggregate, options, search_started_at);
    loop {
        let Some(set) = lattice.next(&mut |pending, ctx| validate_finalist(state, pending, ctx))
        else {
            panic!("{}", lattice.failure_message());
        };
        let refs = lattice.llirs(&set);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| validate_set(state, &refs)))
                .unwrap_or_else(|_| Err("aggregate bucket candidate filter panicked".to_string()));
        drop(refs);
        let result = if lattice.timed_out(&set) {
            Err(format!(
                "candidate timeout expired while filtering aggregate bucket set {}",
                lattice.attempts()
            ))
        } else {
            result
        };
        match result {
            Ok(()) => return lattice.select(set),
            Err(reason) => {
                lattice.reject(set, reason, &mut |pending, ctx| {
                    validate_finalist(state, pending, ctx)
                });
            }
        }
    }
}
