//! Choose per-bucket finalists under one aggregate arena budget.

use crate::finalists::{Finalists, PendingFinalist};
use luminal::prelude::FxHashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BucketSet {
    pub indices: Vec<usize>,
}

pub type AggregateFn = fn(&[u128]) -> u128;

pub fn sum_metrics(metrics: &[u128]) -> u128 {
    metrics
        .iter()
        .fold(0u128, |total, metric| total.saturating_add(*metric))
}

pub struct BucketLattice<'a> {
    buckets: Vec<Finalists<'a>>,
    aggregate: AggregateFn,
    frontier: Vec<(u128, Vec<usize>)>,
    visited: FxHashSet<Vec<usize>>,
    initialized: bool,
    attempts: usize,
    rejections: usize,
    stopped_reason: Option<String>,
    last_rejection: Option<String>,
}

impl<'a> BucketLattice<'a> {
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

    pub fn rejections(&self) -> usize {
        self.rejections
    }

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
