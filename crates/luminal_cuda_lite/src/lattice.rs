//! THE BUCKET LATTICE — a best-first walk over per-bucket finalist
//! ranks, minimizing a coordinate-monotone aggregate.
//!
//! PHASE 5 OF THE #420/#422 REJOIN (2026-09-03), the other half of
//! Austin's D8 "then add". Re-expressed from main's
//! `src/search/lattice.rs` for a branch with no LLIR.
//!
//! # The problem it solves
//!
//! A bucketed search produces ONE ranked list of candidates per bucket
//! ([`crate::finalists::Finalists`]). Installing each bucket's own
//! winner is right only while nothing constrains the buckets JOINTLY.
//! This runtime has exactly one such constraint — the serving slab is
//! grown once and sized to the largest installed plan, so a caller's
//! device budget applies to the SET, not to any one bucket — and the
//! moment such a constraint exists, "each bucket's best" can be
//! infeasible while a set that is one rank slower in ONE bucket fits.
//!
//! The candidate space is the Cartesian product of the buckets' ranks: a
//! point is a vector of indices, one per bucket, and its cost is
//! `aggregate` over the buckets' metrics at those ranks. The walk is
//! best-first over that lattice.
//!
//! # Why one-coordinate-slower successors are enough
//!
//! `aggregate` must be COORDINATE-MONOTONE: raising any one coordinate
//! (to a slower finalist) must not lower the aggregate. Σ of
//! nonnegative metrics is, which is the aggregate this runtime uses.
//! Under that assumption every point's cost is ≥ the cost of each of its
//! one-step predecessors, so expanding a point into its `n`
//! one-coordinate-slower successors and always popping the frontier's
//! strictly-smallest aggregate enumerates the lattice in nondecreasing
//! cost order — the ordinary best-first argument. A `visited` set keeps
//! a point from entering the frontier along two different paths, so no
//! set is ever proposed twice.
//!
//! # Laziness
//!
//! `initialize` materializes RANK 0 of every bucket and nothing else.
//! Rank k+1 of bucket i is extracted and hard-filtered only when a
//! rejected set's successor actually reaches it, through
//! [`crate::finalists::Finalists::ensure`]. A search that validates on
//! the first try therefore pays for exactly one finalist per bucket.
//!
//! # The single-bucket case
//!
//! An UNBUCKETED search runs the same code path over a one-bucket
//! lattice — main's "one designed difference" from the pre-#420
//! behaviour, adopted here for the same reason: the aggregate check is a
//! property of what gets installed, and an unbucketed install is a set of
//! one. It costs nothing when no budget is set (rank 0 validates and the
//! first `next` wins) and it means there is ONE selection path to reason
//! about rather than two.

use crate::finalists::{Finalists, PendingFinalist};
use luminal::prelude::FxHashSet;

/// One point of the lattice: a finalist index per bucket, in bucket
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketSet {
    pub indices: Vec<usize>,
}

/// The aggregate: a coordinate-monotone function of the per-bucket
/// metrics. A plain `fn` pointer, not a trait and not a boxed closure —
/// there is one of these in this crate and it is [`sum_metrics`].
pub type AggregateFn = fn(&[u128]) -> u128;

/// Σ of the buckets' metrics, saturating. Coordinate-monotone because
/// the metrics are nonnegative.
pub fn sum_metrics(metrics: &[u128]) -> u128 {
    metrics
        .iter()
        .fold(0u128, |total, metric| total.saturating_add(*metric))
}

/// What a completed walk installs: the selected finalist per bucket, how
/// many sets were rejected on the way, and the installed ranks (1-based,
/// for the fallback log line).
#[derive(Debug)]
pub struct Installed {
    pub selected: Vec<(usize, PendingFinalist)>,
    pub rejections: usize,
    pub ranks: Vec<usize>,
}

/// Best-first selection over the buckets' finalist ranks.
pub struct BucketLattice<'a> {
    buckets: Vec<Finalists<'a>>,
    aggregate: AggregateFn,
    /// `(aggregate, indices)` — the open set. Small by construction (at
    /// most one entry per rejection per bucket), so a linear scan for
    /// the minimum beats a heap and keeps first-come tie-breaking
    /// obvious.
    frontier: Vec<(u128, Vec<usize>)>,
    visited: FxHashSet<Vec<usize>>,
    initialized: bool,
    /// Sets PROPOSED so far — `next` returning a set counts one.
    attempts: usize,
    /// Sets the caller REJECTED (the set-level constraint failed).
    rejections: usize,
    /// Why the walk stopped, when it stopped for a nameable reason.
    stopped_reason: Option<String>,
    last_rejection: Option<String>,
}

impl<'a> BucketLattice<'a> {
    /// At least one bucket; the aggregate must be coordinate-monotone
    /// (see the module note).
    pub fn new(buckets: Vec<Finalists<'a>>, aggregate: AggregateFn) -> Self {
        assert!(
            !buckets.is_empty(),
            "a bucket lattice needs at least one bucket"
        );
        Self {
            buckets,
            aggregate,
            frontier: Vec::new(),
            visited: FxHashSet::default(),
            initialized: false,
            attempts: 0,
            rejections: 0,
            stopped_reason: None,
            last_rejection: None,
        }
    }

    /// How many sets the caller rejected — what
    /// `SearchOutcome::lattice_rejections` reports.
    pub fn rejections(&self) -> usize {
        self.rejections
    }

    /// The next set to try, cheapest first, or `None` when the walk is
    /// out of options (see [`BucketLattice::failure_message`] for why).
    pub fn next(
        &mut self,
        validate: &mut dyn FnMut(&PendingFinalist) -> Result<(), String>,
    ) -> Option<BucketSet> {
        if !self.initialized && !self.initialize(validate) {
            return None;
        }
        if self.frontier.is_empty() {
            return None;
        }
        // The strictly smallest aggregate; ties go to the entry that
        // entered the frontier first.
        let mut best = 0usize;
        for (position, entry) in self.frontier.iter().enumerate().skip(1) {
            if entry.0 < self.frontier[best].0 {
                best = position;
            }
        }
        let (_, indices) = self.frontier.remove(best);
        self.attempts += 1;
        Some(BucketSet { indices })
    }

    /// RANK 0 EVERYWHERE. A bucket that cannot supply even one viable
    /// finalist stops the whole walk, naming itself.
    fn initialize(
        &mut self,
        validate: &mut dyn FnMut(&PendingFinalist) -> Result<(), String>,
    ) -> bool {
        self.initialized = true;
        for index in 0..self.buckets.len() {
            if !self.buckets[index].ensure(0, validate) {
                self.stopped_reason = Some(format!(
                    "failed to find a viable final plan for {}: {}",
                    self.buckets[index].label(),
                    self.buckets[index].failure_message()
                ));
                return false;
            }
        }
        let origin = vec![0usize; self.buckets.len()];
        let cost = self.aggregate_at(&origin);
        self.visited.insert(origin.clone());
        self.frontier.push((cost, origin));
        true
    }

    /// The aggregate over an already-materialized point.
    fn aggregate_at(&self, indices: &[usize]) -> u128 {
        let metrics: Vec<u128> = indices
            .iter()
            .enumerate()
            .map(|(bucket, index)| {
                self.buckets[bucket]
                    .get(*index)
                    .map(|finalist| finalist.metric)
                    .unwrap_or(u128::MAX)
            })
            .collect();
        (self.aggregate)(&metrics)
    }

    /// Every bucket's arena high-water mark at `set`, in bucket order —
    /// the numbers the caller's set-level constraint reads. Returned by
    /// VALUE so the caller can hold them across a `reject`.
    pub fn slab_bytes(&self, set: &BucketSet) -> Vec<usize> {
        set.indices
            .iter()
            .enumerate()
            .map(|(bucket, index)| {
                self.buckets[bucket]
                    .get(*index)
                    .map(|finalist| finalist.arena.slab_bytes)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// The finalist ranks (1-based) at `set`, for reporting.
    pub fn ranks(&self, set: &BucketSet) -> Vec<usize> {
        set.indices
            .iter()
            .enumerate()
            .map(|(bucket, index)| {
                self.buckets[bucket]
                    .get(*index)
                    .map(|finalist| finalist.rank)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// THE SET FAILED the caller's constraint: open its
    /// one-coordinate-slower successors, materializing the finalists
    /// they name.
    pub fn reject(
        &mut self,
        set: &BucketSet,
        reason: impl Into<String>,
        validate: &mut dyn FnMut(&PendingFinalist) -> Result<(), String>,
    ) {
        self.rejections += 1;
        self.last_rejection = Some(reason.into());
        for bucket in 0..self.buckets.len() {
            let mut successor = set.indices.clone();
            successor[bucket] += 1;
            if self.visited.contains(&successor) {
                continue;
            }
            if !self.buckets[bucket].ensure(successor[bucket], validate) {
                // This coordinate cannot go one step slower. That is not
                // itself a failure — another coordinate may still open —
                // but it IS the reason the walk ends if none does, so
                // the first such exhaustion is remembered.
                if self.stopped_reason.is_none() {
                    self.stopped_reason = Some(format!(
                        "{} ran out of finalists: {}",
                        self.buckets[bucket].label(),
                        self.buckets[bucket].failure_message()
                    ));
                }
                continue;
            }
            let cost = self.aggregate_at(&successor);
            self.visited.insert(successor.clone());
            self.frontier.push((cost, successor));
        }
    }

    /// THE INSTALL: consume the lattice and hand back, per bucket, its
    /// selected `(bucket index, plan, arena plan)`.
    pub fn select(self, set: &BucketSet) -> Vec<(usize, PendingFinalist)> {
        let mut selected = Vec::with_capacity(self.buckets.len());
        for (bucket, finalists) in self.buckets.into_iter().enumerate() {
            let index = set.indices[bucket];
            let finalist = finalists
                .take(index)
                .expect("a proposed set names only materialized finalists");
            selected.push((bucket, finalist));
        }
        selected
    }

    /// DRIVE THE WALK TO AN INSTALLED SET: propose the cheapest untried
    /// set, check the caller's set-level constraint over its slabs,
    /// install it or reject it and let the lattice open the
    /// one-coordinate-slower successors. `Err` carries
    /// [`BucketLattice::failure_message`].
    pub fn drive(
        mut self,
        validate: &mut dyn FnMut(&PendingFinalist) -> Result<(), String>,
        validate_set: &mut dyn FnMut(&[usize]) -> Result<(), String>,
    ) -> Result<Installed, String> {
        loop {
            let Some(set) = self.next(validate) else {
                return Err(self.failure_message());
            };
            // Owned numbers, so the immutable borrow of the lattice ends
            // before a rejection takes it mutably.
            let slabs = self.slab_bytes(&set);
            match validate_set(&slabs) {
                Ok(()) => {
                    let rejections = self.rejections();
                    let ranks = self.ranks(&set);
                    return Ok(Installed {
                        selected: self.select(&set),
                        rejections,
                        ranks,
                    });
                }
                Err(reason) => self.reject(&set, reason, validate),
            }
        }
    }

    /// Why nothing viable was found — the text `search` refuses with.
    ///
    /// MAIN'S SPLIT, kept: when NOTHING was ever proposed the failure is
    /// the bucket's (no viable finalist at all), so its message is the
    /// whole answer; once a set HAS been proposed and rejected, the
    /// caller's own rejection reason is what the reader needs — it names
    /// the constraint that was not met — and the exhaustion is appended
    /// as the reason no slower set was tried.
    pub fn failure_message(&self) -> String {
        if self.attempts == 0 {
            return format!(
                "no viable plan set: {}",
                self.stopped_reason
                    .clone()
                    .unwrap_or_else(|| "no reason recorded".to_string())
            );
        }
        let reason = self
            .last_rejection
            .clone()
            .or_else(|| self.stopped_reason.clone())
            .unwrap_or_else(|| "no reason recorded".to_string());
        let mut message = format!(
            "no viable plan set after {} proposal(s) and {} rejection(s): {reason}",
            self.attempts, self.rejections
        );
        if let Some(stopped) = &self.stopped_reason {
            message.push_str(&format!("; no slower set is available ({stopped})"));
        }
        message
    }
}

#[cfg(test)]
mod tests {
    //! The walk over SYNTHETIC finalists: `(metric, slab_bytes)` pairs
    //! written down, so every precondition holds by construction rather
    //! than by what a seeded genetic sample happened to contain.

    use super::*;
    use luminal::prelude::egraph_serialize;

    fn bucket<'a>(
        label: &str,
        egraph: &'a egraph_serialize::EGraph,
        ranked: &[(u128, usize)],
    ) -> Finalists<'a> {
        Finalists::synthetic(label, egraph, ranked)
    }

    fn budget(bytes: usize) -> impl FnMut(&[usize]) -> Result<(), String> {
        move |slabs: &[usize]| {
            let peak = slabs.iter().copied().max().unwrap_or(0);
            if peak > bytes {
                Err(format!("peak {peak} over the {bytes}-byte budget"))
            } else {
                Ok(())
            }
        }
    }

    fn accept_all(_: &PendingFinalist) -> Result<(), String> {
        Ok(())
    }

    /// A budget one byte under the winning set's slab rejects it, and the
    /// walk installs the cheapest one-coordinate-slower set that fits.
    #[test]
    fn a_budget_under_the_winner_installs_the_cheapest_slower_set_that_fits() {
        let egraph = egraph_serialize::EGraph::default();
        let buckets = vec![
            bucket("bucket 0", &egraph, &[(10, 100), (12, 50)]),
            bucket("bucket 1", &egraph, &[(5, 80)]),
        ];
        let lattice = BucketLattice::new(buckets, sum_metrics);
        let Installed {
            selected: installed,
            rejections,
            ranks,
        } = lattice
            .drive(&mut accept_all, &mut budget(99))
            .expect("the slower set fits");
        assert_eq!(rejections, 1, "exactly the winning set was rejected");
        assert_eq!(
            ranks,
            vec![2, 1],
            "bucket 0 fell back one rank, bucket 1 kept its winner"
        );
        let slabs: Vec<usize> = installed.iter().map(|(_, f)| f.arena.slab_bytes).collect();
        assert_eq!(slabs, vec![50, 80]);
        assert!(slabs.iter().all(|s| *s <= 99));
    }

    /// No budget: the origin set installs, nothing is rejected.
    #[test]
    fn unconstrained_installs_every_bucket_winner() {
        let egraph = egraph_serialize::EGraph::default();
        let buckets = vec![
            bucket("bucket 0", &egraph, &[(10, 100), (12, 50)]),
            bucket("bucket 1", &egraph, &[(5, 80), (6, 10)]),
        ];
        let Installed {
            selected: installed,
            rejections,
            ranks,
        } = BucketLattice::new(buckets, sum_metrics)
            .drive(&mut accept_all, &mut |_: &[usize]| Ok(()))
            .expect("the winners install");
        assert_eq!(rejections, 0);
        assert_eq!(ranks, vec![1, 1]);
        assert_eq!(installed.len(), 2);
    }

    /// A budget nothing meets: every set is rejected, the walk runs out,
    /// and the message names the constraint and the exhaustion.
    #[test]
    fn a_budget_nothing_meets_runs_out_and_names_it() {
        let egraph = egraph_serialize::EGraph::default();
        let buckets = vec![
            bucket("bucket 0", &egraph, &[(10, 100), (12, 50)]),
            bucket("bucket 1", &egraph, &[(5, 80)]),
        ];
        let err = BucketLattice::new(buckets, sum_metrics)
            .drive(&mut accept_all, &mut budget(10))
            .expect_err("nothing fits in 10 bytes");
        assert!(
            err.contains("no viable plan set after 2 proposal(s) and 2 rejection(s)"),
            "{err}"
        );
        assert!(err.contains("over the 10-byte budget"), "{err}");
        assert!(err.contains("ran out of finalists"), "{err}");
    }

    /// Sets are proposed in nondecreasing aggregate order and never twice.
    #[test]
    fn sets_are_proposed_cheapest_first_and_once() {
        let egraph = egraph_serialize::EGraph::default();
        let metrics: Vec<Vec<u128>> = vec![vec![10, 11, 20], vec![5, 9]];
        let buckets = vec![
            bucket("bucket 0", &egraph, &[(10, 1), (11, 1), (20, 1)]),
            bucket("bucket 1", &egraph, &[(5, 1), (9, 1)]),
        ];
        let mut lattice = BucketLattice::new(buckets, sum_metrics);
        let mut proposed: Vec<Vec<usize>> = Vec::new();
        while let Some(set) = lattice.next(&mut accept_all) {
            proposed.push(set.indices.clone());
            lattice.reject(&set, "recording", &mut accept_all);
        }
        assert_eq!(
            proposed.len(),
            6,
            "every point of the 3x2 lattice was proposed exactly once"
        );
        let aggregates: Vec<u128> = proposed
            .iter()
            .map(|idx| idx.iter().enumerate().map(|(b, i)| metrics[b][*i]).sum())
            .collect();
        assert!(
            aggregates.windows(2).all(|w| w[0] <= w[1]),
            "not cheapest-first: {aggregates:?}"
        );
        let mut unique = proposed.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), proposed.len());
    }
}
