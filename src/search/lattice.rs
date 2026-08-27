//! Best-first selection of one finalist per bucket under an aggregate
//! constraint.
//!
//! Each bucket's finalists are individually viable, but a runtime may have
//! resources (persistent buffers, retained kernels) that every retained
//! bucket shares. [`BucketLattice`] walks the Cartesian lattice of per-bucket
//! finalist ranks fastest-aggregate-first: [`BucketLattice::next`] proposes
//! a set, the runtime validates it, and [`BucketLattice::reject`] opens the
//! set's one-coordinate-slower successors — materializing deeper finalists
//! only when a rejection actually reaches them.

use std::fmt::Debug;
use std::time::Instant;

use colored::Colorize;
use rustc_hash::FxHashSet;

use super::finalist::{FinalistValidator, Finalists};
use super::{BucketLLIR, BucketLLIRRef};
use crate::graph::CompileOptions;

/// Combines one metric per bucket into the value the lattice walk minimizes.
pub type AggregateFn<'a, M> = Box<dyn Fn(&[M]) -> M + 'a>;

/// One finalist rank per bucket, proposed by [`BucketLattice::next`].
pub struct BucketSet {
    /// Finalist index per bucket, in bucket order.
    pub indices: Vec<usize>,
    proposed_at: Instant,
}

pub struct BucketLattice<'a, M> {
    buckets: Vec<Finalists<'a, M>>,
    aggregate: AggregateFn<'a, M>,
    options: &'a CompileOptions,
    search_started_at: Instant,
    search_log: bool,
    initialized: bool,
    frontier: Vec<(M, Vec<usize>)>,
    visited: FxHashSet<Vec<usize>>,
    attempts: usize,
    rejections: usize,
    last_rejection: Option<String>,
    stopped_reason: Option<String>,
}

impl<'a, M: PartialOrd + Clone + Debug> BucketLattice<'a, M> {
    /// `aggregate` combines one metric per bucket into the value the walk
    /// minimizes. It must be coordinate-monotone: replacing any input with a
    /// greater metric must not make the aggregate compare less.
    pub fn new(
        buckets: Vec<Finalists<'a, M>>,
        aggregate: impl Fn(&[M]) -> M + 'a,
        options: &'a CompileOptions,
        search_started_at: Instant,
    ) -> Self {
        assert!(
            !buckets.is_empty(),
            "bucket lattice needs at least one bucket"
        );
        Self {
            buckets,
            aggregate: Box::new(aggregate),
            options,
            search_started_at,
            search_log: options.search_log_enabled(),
            initialized: false,
            frontier: Vec::new(),
            visited: FxHashSet::default(),
            attempts: 0,
            rejections: 0,
            last_rejection: None,
            stopped_reason: None,
        }
    }

    /// Bucket sets proposed so far.
    pub fn attempts(&self) -> usize {
        self.attempts
    }

    pub fn rejections(&self) -> usize {
        self.rejections
    }

    pub fn buckets(&self) -> &[Finalists<'a, M>] {
        &self.buckets
    }

    fn label(&self, bucket_idx: usize) -> String {
        self.buckets[bucket_idx].bucket_context().label()
    }

    fn metrics(&self, indices: &[usize]) -> M {
        let metrics: Vec<M> = self
            .buckets
            .iter()
            .zip(indices)
            .map(|(bucket, &idx)| bucket.get(idx).metric.clone())
            .collect();
        (self.aggregate)(&metrics)
    }

    /// Materialize the fastest individually viable finalist of every bucket
    /// to seed the walk. Slower finalists are extracted only when an
    /// aggregate rejection makes their coordinate reachable.
    fn initialize(&mut self, validate: &mut FinalistValidator<'_, M>) -> bool {
        self.initialized = true;
        for bucket_idx in 0..self.buckets.len() {
            if !self.buckets[bucket_idx].ensure(0, validate) {
                let message = self.buckets[bucket_idx].failure_message();
                self.stopped_reason = Some(
                    if self.buckets[bucket_idx].bucket_context().is_bucketed() {
                        format!(
                            "Failed to find a viable final graph for bucket {bucket_idx} ({}): {message}",
                            self.label(bucket_idx)
                        )
                    } else {
                        format!("Failed to find a viable final graph: {message}")
                    },
                );
                return false;
            }
        }
        let initial = vec![0usize; self.buckets.len()];
        self.frontier = vec![(self.metrics(&initial), initial.clone())];
        self.visited.insert(initial);
        true
    }

    /// Propose the fastest unvisited bucket set. `validate` hard-filters
    /// finalists that have to be materialized to get there. `None` when no
    /// viable set remains or the search time limit expired; see
    /// [`BucketLattice::failure_message`].
    pub fn next(&mut self, validate: &mut FinalistValidator<'_, M>) -> Option<BucketSet> {
        if !self.initialized && !self.initialize(validate) {
            return None;
        }
        if self.attempts > 0 && self.search_started_at.elapsed() >= self.options.search_time_limit {
            self.stopped_reason =
                Some("search time limit expired during aggregate bucket finalization".to_string());
            return None;
        }
        if self.frontier.is_empty() {
            return None;
        }
        let best_pos = (1..self.frontier.len()).fold(0, |best, candidate| {
            if self.frontier[candidate]
                .0
                .partial_cmp(&self.frontier[best].0)
                .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
            {
                candidate
            } else {
                best
            }
        });
        let (_, indices) = self.frontier.swap_remove(best_pos);
        self.attempts += 1;
        Some(BucketSet {
            indices,
            proposed_at: Instant::now(),
        })
    }

    /// The proposed set's LLIRs, in bucket order.
    pub fn llirs(&self, set: &BucketSet) -> Vec<BucketLLIRRef<'_>> {
        self.buckets
            .iter()
            .zip(&set.indices)
            .map(|(bucket, &idx)| bucket.bucket_context().llir_ref(&bucket.get(idx).llir))
            .collect()
    }

    /// Whether `candidate_timeout` expired since the set was proposed.
    pub fn timed_out(&self, set: &BucketSet) -> bool {
        self.options
            .candidate_timeout
            .is_some_and(|timeout| set.proposed_at.elapsed() >= timeout)
    }

    /// The set is not viable together. Opens its one-coordinate successors.
    pub fn reject(
        &mut self,
        set: BucketSet,
        reason: String,
        validate: &mut FinalistValidator<'_, M>,
    ) {
        self.rejections += 1;
        crate::mask_events::AGGREGATE_REJECT.record();
        if self.search_log {
            println!(
                "   {:>6}  aggregate reject per-bucket finalist ranks {:?}: {reason}",
                "Search".yellow().bold(),
                set.indices,
            );
        }
        self.last_rejection = Some(reason);

        // Any one-coordinate successor is the next possible slower
        // combination. A visited set prevents duplicate lattice paths.
        for bucket_idx in 0..self.buckets.len() {
            let mut successor = set.indices.clone();
            successor[bucket_idx] += 1;
            if !self.visited.insert(successor.clone()) {
                continue;
            }
            if !self.buckets[bucket_idx].ensure(successor[bucket_idx], validate) {
                let bucket = &self.buckets[bucket_idx];
                if bucket.stopped_reason.is_some()
                    || bucket
                        .last_rejection
                        .as_deref()
                        .is_some_and(|reason| reason.contains("timeout"))
                {
                    let message = bucket.failure_message();
                    let label = self.label(bucket_idx);
                    self.stopped_reason.get_or_insert_with(|| {
                        format!("bucket {bucket_idx} ({label}) fallback stopped: {message}")
                    });
                }
                continue;
            }
            let metric = self.metrics(&successor);
            self.frontier.push((metric, successor));
        }
    }

    /// The set is viable. Takes the selected finalists (dumping them under
    /// `LLIR_DUMP_DIR`) as `(bucket_indices, representative_dyn_map, llir)`
    /// per bucket.
    pub fn select(self, set: BucketSet) -> Vec<BucketLLIR> {
        if self.search_log && self.rejections > 0 {
            println!(
                "   {:>6}  aggregate fallback: selected per-bucket finalist ranks {:?} after {} rejection(s)",
                "Search".yellow().bold(),
                set.indices,
                self.rejections,
            );
        }
        self.buckets
            .into_iter()
            .zip(set.indices)
            .map(|(bucket, idx)| {
                let ctx = bucket.bucket_context();
                let finalist = bucket.take(idx);
                (
                    ctx.bucket_indices().clone(),
                    ctx.representative_dyn_map.clone(),
                    finalist.llir,
                )
            })
            .collect()
    }

    /// Why no viable set could be proposed.
    pub fn failure_message(&self) -> String {
        if !self.initialized
            || self.attempts == 0 && self.frontier.is_empty() && self.rejections == 0
        {
            return self
                .stopped_reason
                .clone()
                .unwrap_or_else(|| "no viable final graph".to_string());
        }
        let reason = self
            .stopped_reason
            .clone()
            .or_else(|| self.last_rejection.clone())
            .unwrap_or_else(|| "no aggregate candidate combinations remain".to_string());
        format!(
            "Failed to find a viable aggregate bucket set after {} rejections: {reason}",
            self.rejections
        )
    }
}
