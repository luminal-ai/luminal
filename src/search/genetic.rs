//! Luminal's genetic search over one bucket's e-graph, as a pull-style
//! state machine.
//!
//! The runtime drives it: [`GeneticSearch::next_candidate`] hands out a
//! fully-unrolled deployment graph, the runtime evaluates it by whatever
//! means it has, and [`GeneticSearch::report`] feeds the [`Outcome`] back.
//! The state machine owns everything else the search has always owned —
//! initial-genome retries, generations and mutation, stagnation kicks and
//! resampling, genome and program dedup, the graph limit, the search time
//! limit, the candidate timeout, the early-stop hint, progress bars, and the
//! `LLIR_DUMP_DIR` / `LUMINAL_LOG_LLIR` / `LUMINAL_CANDIDATE_OPS` hooks.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::Instant;

use colored::Colorize;
use rand::RngCore;
use rustc_hash::{FxHashMap, FxHashSet};

use super::diagnostics::{
    ProgressBars, log_best_llir, log_candidate_ops, panic_initial_filter_limit,
};
use super::packed::LlirFingerprint;
use super::unroll::unroll_packed_llir;
use super::{BucketContext, SearchSpace};
use crate::egglog_utils::{IndexedChoiceSet, LlirExtractor, count_choice_sets_up_to};
use crate::graph::{CompileOptions, LLIRGraph};
use crate::shape::DynMap;

/// Identifies a handed-out [`Candidate`] so it can only be reported once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidateId(u64);

/// A deployment graph the search wants evaluated.
pub struct Candidate<M> {
    pub id: CandidateId,
    /// Fully unrolled — the graph that would be deployed if selected.
    pub llir: LLIRGraph,
    /// Dyn values to evaluate with: the bucket representative with
    /// `profile_dims` applied.
    pub profile_dyn_map: DynMap,
    /// `Some((best_metric, factor))` once a best candidate exists: an
    /// evaluator may stop early once this candidate's running metric
    /// exceeds `best * factor`. The truncated metric is ranked normally.
    pub early_stop: Option<(M, f64)>,
    pre_collapse: Option<LLIRGraph>,
    timer: Instant,
}

impl<M> Candidate<M> {
    /// Restart the clock the search checks `candidate_timeout` against. By
    /// default it runs from hand-out to report; a runtime whose evaluation
    /// has a phase the timeout should not cover (e.g. compilation) restarts
    /// it before the phase it should.
    pub fn restart_timer(&mut self) {
        self.timer = Instant::now();
    }

    /// The rolled graph before loop unrolling, retained only when
    /// `LLIR_DUMP_DIR` is set, for post-mortem dumps.
    pub fn pre_collapse(&self) -> Option<&LLIRGraph> {
        self.pre_collapse.as_ref()
    }
}

/// What happened to a [`Candidate`].
#[derive(Debug, Clone)]
pub enum Outcome<M> {
    /// Evaluated. Counts toward the graph limit and is ranked by the metric
    /// unless the candidate timeout expired.
    Measured(M, String),
    /// Not viable before evaluation (failed to compile, over a resource cap).
    /// Does not count toward the graph limit.
    Rejected(String),
    /// Evaluation started but produced no usable metric. Counts toward the
    /// graph limit; never ranked.
    Invalid(String),
}

/// Every measured genome of a finished search, fastest first.
pub type Ranked<M> = Vec<(M, IndexedChoiceSet)>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Looking for the first viable genome.
    Initial,
    /// Evolving from the initial genome.
    Evolving,
    Done,
}

const MAX_INVALID_INITIAL_ATTEMPTS: usize = 100;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Refinement {
    Family(u32, usize),
    Recombination,
}

pub struct GeneticSearch<'a, M> {
    space: &'a SearchSpace,
    ctx: &'a BucketContext<'a>,
    options: &'a CompileOptions,
    extractor: LlirExtractor<'a>,
    profile_dyn_map: DynMap,
    search_limit: usize,
    started_at: Instant,
    search_started_at: Instant,
    search_log: bool,
    bars: ProgressBars,
    keep_pre_collapse: bool,

    prev_selected: FxHashSet<u64>,
    explored_llir_hashes: FxHashSet<LlirFingerprint>,
    phase: Phase,
    next_id: u64,
    outstanding: Option<(CandidateId, IndexedChoiceSet)>,

    // Initial phase.
    initial_seed: Option<IndexedChoiceSet>,
    invalid_attempts: usize,
    filter_fails: usize,
    max_filter_fails: usize,
    last_filter_rejection: Option<String>,
    n_timed_out: usize,
    n_invalid_profile: usize,

    // Evolving phase.
    pending: VecDeque<IndexedChoiceSet>,
    refinements: VecDeque<(Refinement, VecDeque<IndexedChoiceSet>)>,
    prefer_refinement: bool,
    generation_open: bool,
    ranked: Ranked<M>,
    parents: Vec<(M, IndexedChoiceSet)>,
    best_metric: Option<M>,
    family_best: FxHashMap<(u32, usize), M>,
    n_graphs: usize,
    resample_generation: bool,
    stagnant_generations: usize,
    generation_found_non_timeout: bool,
    generation_found_new_best: bool,
    slower_since_faster: usize,
    slower_line_visible: bool,
}

impl<'a, M: PartialOrd + Clone + Debug> GeneticSearch<'a, M> {
    /// `search_started_at` is the start of the whole compile, shared by
    /// every bucket, against which `options.search_time_limit` is checked.
    pub fn new(
        space: &'a SearchSpace,
        ctx: &'a BucketContext<'a>,
        options: &'a CompileOptions,
        search_started_at: Instant,
    ) -> Self {
        let search_log = options.search_log_enabled();
        let bucket_progress = ctx.progress();
        if search_log {
            if let Some((index, n_buckets)) = bucket_progress {
                println!(
                    "   {:>6}  Group {}/{}: {}",
                    "Search".cyan().bold(),
                    index + 1,
                    n_buckets,
                    ctx.label(),
                );
            }
        }
        let limit = options
            .bucket_limits
            .get(&ctx.index)
            .copied()
            .unwrap_or(options.limit);
        let max_filter_fails = limit
            .max(1)
            .saturating_mul(options.generation_size.max(1))
            .saturating_mul(100)
            .max(10_000);
        let initial_seed = options
            .seed_schedule
            .as_ref()
            .and_then(|schedule| schedule.seed_for_bucket(ctx));
        if search_log && options.seed_schedule.is_some() {
            println!(
                "Search seed bucket {}: {}",
                ctx.index,
                if initial_seed.is_some() {
                    "accepted; revalidating and remeasuring"
                } else {
                    "incompatible; ordinary initialization"
                }
            );
        }
        Self {
            space,
            ctx,
            options,
            extractor: LlirExtractor::new(ctx.egraph(), &space.ops),
            profile_dyn_map: ctx.profile_dyn_map(options),
            search_limit: count_choice_sets_up_to(ctx.egraph(), limit),
            started_at: Instant::now(),
            search_started_at,
            search_log,
            bars: ProgressBars::new(bucket_progress),
            keep_pre_collapse: std::env::var_os("LLIR_DUMP_DIR").is_some(),
            prev_selected: FxHashSet::default(),
            explored_llir_hashes: FxHashSet::default(),
            phase: Phase::Initial,
            next_id: 0,
            outstanding: None,
            initial_seed,
            invalid_attempts: 0,
            filter_fails: 0,
            max_filter_fails,
            last_filter_rejection: None,
            n_timed_out: 0,
            n_invalid_profile: 0,
            pending: VecDeque::new(),
            refinements: VecDeque::new(),
            prefer_refinement: false,
            generation_open: false,
            ranked: Vec::new(),
            parents: Vec::new(),
            best_metric: None,
            family_best: FxHashMap::default(),
            n_graphs: 0,
            resample_generation: false,
            stagnant_generations: 0,
            generation_found_non_timeout: false,
            generation_found_new_best: false,
            slower_since_faster: 0,
            slower_line_visible: false,
        }
    }

    pub fn bucket_context(&self) -> &'a BucketContext<'a> {
        self.ctx
    }

    /// Start from a previously selected program when its e-graph and bucket
    /// contracts match. It enters the ordinary validation/profiling path and
    /// consumes one measured-candidate slot; no old fitness is reused. A
    /// rejected seed falls back to random initialization, and mutation remains
    /// free to replace any of its choices.
    pub fn seed_schedule(&mut self, schedule: &crate::graph::SelectedSchedule) -> bool {
        assert!(
            self.phase == Phase::Initial
                && self.outstanding.is_none()
                && self.initial_seed.is_none()
                && self.invalid_attempts == 0
                && self.filter_fails == 0,
            "seed a search only before requesting its first candidate"
        );
        self.initial_seed = schedule.seed_for_bucket(self.ctx);
        self.initial_seed.is_some()
    }

    /// Dyn values candidates should be evaluated with.
    pub fn profile_dyn_map(&self) -> &DynMap {
        &self.profile_dyn_map
    }

    /// Best metric measured so far.
    pub fn best(&self) -> Option<&M> {
        self.best_metric.as_ref()
    }

    /// Candidates measured so far (the count the graph limit applies to).
    pub fn measured(&self) -> usize {
        self.n_graphs
    }

    fn time_limit_reached(&self) -> bool {
        self.search_started_at.elapsed() >= self.options.search_time_limit
    }

    /// The next candidate to evaluate, or `None` once the graph limit, the
    /// search time limit, or the space itself is exhausted. The previous
    /// candidate must have been reported.
    pub fn next_candidate(&mut self, rng: &mut dyn RngCore) -> Option<Candidate<M>> {
        assert!(
            self.outstanding.is_none(),
            "report the outstanding candidate before requesting another"
        );
        loop {
            match self.phase {
                Phase::Done => return None,
                Phase::Initial => {
                    let genome = self.initial_seed.take().or_else(|| {
                        self.extractor
                            .random_indexed_generation(1, &mut self.prev_selected, rng)
                            .pop()
                    });
                    let Some(genome) = genome else {
                        panic_initial_filter_limit(
                            self.filter_fails,
                            self.last_filter_rejection.as_deref(),
                        );
                    };
                    match self.extract(&genome) {
                        Ok(Some((_, llir))) => return Some(self.hand_out(genome, llir, None)),
                        Ok(None) => continue,
                        Err(()) => {
                            self.invalid_attempts += 1;
                            if self.invalid_attempts > MAX_INVALID_INITIAL_ATTEMPTS {
                                panic!(
                                    "Failed to find a viable initial genome after {MAX_INVALID_INITIAL_ATTEMPTS} invalid attempts"
                                );
                            }
                            continue;
                        }
                    }
                }
                Phase::Evolving => {
                    // Recombination can add proposals during a generation.
                    // The measured-candidate budget applies before every handout.
                    if self.n_graphs >= self.search_limit || self.time_limit_reached() {
                        if self.generation_open {
                            self.close_generation();
                        }
                        self.finish();
                        return None;
                    }
                    if self.pending.is_empty()
                        && (!self.prefer_refinement || self.refinements.is_empty())
                    {
                        if self.generation_open {
                            self.close_generation();
                        }
                        self.breed(rng);
                        if self.pending.is_empty() && self.refinements.is_empty() {
                            self.finish();
                            return None;
                        }
                        self.generation_open = true;
                    }
                    if self.time_limit_reached() {
                        self.close_generation();
                        self.finish();
                        return None;
                    }
                    let genome = self.take_pending().unwrap();
                    match self.extract(&genome) {
                        Ok(Some((pre_collapse, llir))) => {
                            return Some(self.hand_out(genome, llir, pre_collapse));
                        }
                        Ok(None) => continue,
                        Err(()) => {
                            if self.search_log {
                                self.bars.redraw(self.n_graphs, self.search_limit);
                            }
                            continue;
                        }
                    }
                }
            }
        }
    }

    // Alternate exploration with refinement. A continually improving losing
    // family or winner recombination must not delay every other constructor.
    // Within refinement, visit families round-robin while prioritizing each
    // family's newest measured improvement. Every proposal uses the same budget.
    fn take_pending(&mut self) -> Option<IndexedChoiceSet> {
        if !self.refinements.is_empty() && (self.prefer_refinement || self.pending.is_empty()) {
            self.prefer_refinement = false;
            let (key, mut queue) = self.refinements.pop_front().unwrap();
            let genome = queue.pop_front();
            if !queue.is_empty() {
                self.refinements.push_back((key, queue));
            }
            genome
        } else {
            self.prefer_refinement = true;
            self.pending.pop_front()
        }
    }

    fn queue_refinement(&mut self, key: Refinement, neighbors: Vec<IndexedChoiceSet>) {
        if neighbors.is_empty() {
            return;
        }
        if let Some((_, queue)) = self.refinements.iter_mut().find(|(k, _)| *k == key) {
            for neighbor in neighbors.into_iter().rev() {
                queue.push_front(neighbor);
            }
        } else {
            self.refinements.push_back((key, neighbors.into()));
        }
    }

    /// Extract and unroll a genome. `Ok(None)` when the program was already
    /// explored; `Err` when extraction or unrolling panicked.
    #[allow(clippy::type_complexity)]
    fn extract(
        &mut self,
        genome: &IndexedChoiceSet,
    ) -> Result<Option<(Option<LLIRGraph>, LLIRGraph)>, ()> {
        let keep_pre_collapse = self.keep_pre_collapse && self.phase == Phase::Evolving;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let packed = self
                .extractor
                .extract_indexed_packed(genome, &self.space.custom_ops);
            if !self.explored_llir_hashes.insert(packed.fingerprint()) {
                return None;
            }
            let pre_collapse = keep_pre_collapse.then(|| packed.to_stable());
            // Profile the deployment graph itself: fully unrolled. Every
            // scaled-down proxy (collapsed bodies, trip-count differencing)
            // leaked family-dependent costs and inverted rankings; measuring
            // the real graph is slower per candidate but cannot misorder
            // families.
            Some((pre_collapse, unroll_packed_llir(packed)))
        }));
        match result {
            Ok(extracted) => Ok(extracted),
            Err(payload) => {
                if self.phase == Phase::Evolving {
                    crate::mask_events::CANDIDATE_PANIC
                        .record_with(|| crate::mask_events::panic_payload(payload.as_ref()));
                }
                Err(())
            }
        }
    }

    fn hand_out(
        &mut self,
        genome: IndexedChoiceSet,
        llir: LLIRGraph,
        pre_collapse: Option<LLIRGraph>,
    ) -> Candidate<M> {
        let id = CandidateId(self.next_id);
        self.next_id += 1;
        self.outstanding = Some((id, genome));
        // Losers are the most expensive candidates to evaluate: a candidate
        // whose running metric is already `factor ×` worse than the best can
        // stop early — its partial metric is ranked normally and cannot win.
        let early_stop = self.best_metric.clone().zip(self.options.early_stop_factor);
        Candidate {
            id,
            llir,
            profile_dyn_map: self.profile_dyn_map.clone(),
            early_stop,
            pre_collapse,
            timer: Instant::now(),
        }
    }

    /// Report the outcome of the outstanding candidate.
    pub fn report(&mut self, candidate: Candidate<M>, outcome: Outcome<M>) {
        let (id, genome) = self
            .outstanding
            .take()
            .expect("no outstanding candidate to report");
        assert_eq!(
            id, candidate.id,
            "reported candidate is not the outstanding one"
        );
        let timed_out = self
            .options
            .candidate_timeout
            .is_some_and(|timeout| candidate.timer.elapsed() >= timeout);
        super::diagnostics::log_candidate_choices(|| {
            serde_json::json!({
                "search_started": format!("{:?}", self.search_started_at),
                "bucket": self.ctx.index,
                "candidate": candidate.id.0,
                "candidate_seconds": candidate.timer.elapsed().as_secs_f64(),
                "timed_out": timed_out,
                "outcome": format!("{outcome:?}"),
                "choices": self.extractor.named_choices(&genome),
            })
        });
        match self.phase {
            Phase::Initial => self.report_initial(genome, candidate, outcome, timed_out),
            Phase::Evolving => self.report_evolving(genome, candidate, outcome, timed_out),
            Phase::Done => panic!("report after the search finished"),
        }
    }

    fn report_initial(
        &mut self,
        genome: IndexedChoiceSet,
        candidate: Candidate<M>,
        outcome: Outcome<M>,
        timed_out: bool,
    ) {
        match outcome {
            // Runtime-rejected candidates are dry failures, not searched
            // graphs: they are never ranked and do not count toward the
            // graph search limit.
            Outcome::Rejected(reason) => {
                self.filter_fails += 1;
                // Rejections are otherwise silent until the 10k-fail panic;
                // surface them early — a structural rejection (e.g. every
                // candidate over the memory cap) loops here for hours
                // looking like a hang.
                if self.filter_fails <= 5 || self.filter_fails % 100 == 0 {
                    eprintln!(
                        "   Search  initial-genome filter reject #{}: {reason}",
                        self.filter_fails
                    );
                }
                self.last_filter_rejection = Some(reason);
                if self.filter_fails >= self.max_filter_fails {
                    panic_initial_filter_limit(
                        self.filter_fails,
                        self.last_filter_rejection.as_deref(),
                    );
                }
                return;
            }
            Outcome::Measured(metric, display) if !timed_out => {
                log_best_llir(&candidate.llir, &format!("candidate=0 {display}"));
                self.best_metric = Some(metric.clone());
                self.ranked = vec![(metric.clone(), genome.clone())];
                self.parents = vec![(metric, genome)];
                self.n_graphs = 1;
                if self.search_log {
                    println!("   {:>6} {}", "Start".cyan().bold(), display);
                    self.bars.render(self.n_graphs, self.search_limit);
                }
                self.phase = Phase::Evolving;
                return;
            }
            Outcome::Measured(..) => self.n_timed_out += 1,
            Outcome::Invalid(_) => self.n_invalid_profile += 1,
        }
        self.invalid_attempts += 1;
        if self.invalid_attempts > MAX_INVALID_INITIAL_ATTEMPTS {
            panic!(
                "Failed to find a viable initial genome after {MAX_INVALID_INITIAL_ATTEMPTS} invalid attempts \
                 (candidate_timed_out={} invalid_profile={})",
                self.n_timed_out, self.n_invalid_profile
            );
        }
    }

    fn report_evolving(
        &mut self,
        genome: IndexedChoiceSet,
        candidate: Candidate<M>,
        outcome: Outcome<M>,
        timed_out: bool,
    ) {
        let (new_metric, display_metric) = match outcome {
            Outcome::Rejected(_) => return,
            Outcome::Invalid(_) => {
                self.n_graphs += 1;
                if self.search_log {
                    self.bars.redraw(self.n_graphs, self.search_limit);
                }
                return;
            }
            Outcome::Measured(metric, display) => {
                self.n_graphs += 1;
                if timed_out {
                    if self.search_log {
                        self.bars.redraw(self.n_graphs, self.search_limit);
                    }
                    return;
                }
                self.generation_found_non_timeout = true;
                (metric, display)
            }
        };

        let new_best = self
            .best_metric
            .as_ref()
            .is_some_and(|best| best.gt(&new_metric));
        if new_best {
            // A winner may descend from an older parent and lose independent
            // improvements in the previous incumbent. Test their combinations
            // immediately instead of waiting for mutation to rediscover them.
            let combinations = self.extractor.recombine_reachable_choices(
                &genome,
                &self.ranked[0].1,
                &mut self.prev_selected,
            );
            self.queue_refinement(Refinement::Recombination, combinations);
        }

        let rank = self
            .ranked
            .iter()
            .position(|(metric, _)| {
                new_metric
                    .partial_cmp(metric)
                    .is_some_and(|ordering| ordering == std::cmp::Ordering::Less)
            })
            .unwrap_or(self.ranked.len());
        self.ranked
            .insert(rank, (new_metric.clone(), genome.clone()));

        if !new_best {
            for family in self
                .extractor
                .alternate_families(&genome, &self.ranked[0].1)
            {
                let improved = self
                    .family_best
                    .get(&family)
                    .is_none_or(|best| new_metric.lt(best));
                if improved {
                    self.family_best.insert(family, new_metric.clone());
                    let neighbors = self.extractor.argument_neighbors(
                        &genome,
                        family.0,
                        &mut self.prev_selected,
                    );
                    self.queue_refinement(Refinement::Family(family.0, family.1), neighbors);
                }
            }
        }

        // Update parents list (keep top-N for next generation)
        let dominated_by_all = self.parents.len() >= self.options.keep_best
            && !self.parents.last().unwrap().0.gt(&new_metric);
        if !dominated_by_all {
            let pos = self
                .parents
                .iter()
                .position(|(m, _)| {
                    new_metric
                        .partial_cmp(m)
                        .is_some_and(|o| o == std::cmp::Ordering::Less)
                })
                .unwrap_or(self.parents.len());
            self.parents.insert(pos, (new_metric.clone(), genome));
            if self.parents.len() > self.options.keep_best {
                self.parents.truncate(self.options.keep_best);
            }
        }

        log_candidate_ops(
            &candidate.llir,
            &format!("cand={} {display_metric}", self.n_graphs),
        );
        if new_best {
            self.generation_found_new_best = true;
            self.best_metric = Some(new_metric);
            log_best_llir(
                &candidate.llir,
                &format!("candidate={} {display_metric}", self.n_graphs),
            );
        }

        let msg = if new_best {
            self.slower_since_faster = 0;
            format!("   {:>6} {display_metric}", "Faster".green().bold())
        } else {
            self.slower_since_faster += 1;
            format!(
                "   {:>6} x{}",
                "Slower".yellow().bold(),
                self.slower_since_faster
            )
        };
        if self.search_log {
            self.bars
                .print_message(&msg, self.slower_line_visible && !new_best);
            self.slower_line_visible = !new_best;
            self.bars.render(self.n_graphs, self.search_limit);
        }
    }

    /// End-of-generation bookkeeping: stagnation tracking and whether the
    /// next generation resamples from fresh random genomes.
    fn close_generation(&mut self) {
        if self.generation_found_new_best {
            self.stagnant_generations = 0;
        } else {
            self.stagnant_generations += 1;
        }
        // Every other stagnant generation past the threshold explores from
        // fresh random genomes instead of the converged parents.
        let stagnation_resample = self.options.restart_stagnation > 0
            && self.stagnant_generations >= self.options.restart_stagnation
            && self.stagnant_generations % 2 == 0;
        self.resample_generation = !self.generation_found_non_timeout || stagnation_resample;
        self.generation_found_non_timeout = false;
        self.generation_found_new_best = false;
        self.generation_open = false;
    }

    /// Generate the next generation's offspring from all parents, dividing
    /// the remaining budget evenly.
    fn breed(&mut self, rng: &mut dyn RngCore) {
        let options = self.options;
        let budget = (self.search_limit - self.n_graphs).min(options.generation_size);
        let offspring = if self.resample_generation {
            self.extractor
                .random_indexed_generation(budget, &mut self.prev_selected, rng)
        } else {
            let per_parent = budget.div_ceil(self.parents.len());
            let mut offspring = Vec::new();
            for (_, parent_genome) in &self.parents {
                let remaining = budget.saturating_sub(offspring.len());
                if remaining == 0 {
                    break;
                }
                // Stagnation kick: escaping a family basin needs multi-gene
                // jumps, so mutation counts escalate with consecutive
                // stagnant generations (capped 16x).
                let kick = if options.restart_stagnation > 0
                    && self.stagnant_generations >= options.restart_stagnation
                {
                    (1 + self.stagnant_generations - options.restart_stagnation).min(16)
                } else {
                    1
                };
                offspring.extend(self.extractor.extract_reachable_indexed_generation(
                    parent_genome,
                    per_parent.min(remaining),
                    options.mutations * kick,
                    &mut self.prev_selected,
                    rng,
                ));
            }
            offspring
        };
        self.pending.extend(offspring);
    }

    fn finish(&mut self) {
        if self.phase == Phase::Done {
            return;
        }
        let was_evolving = self.phase == Phase::Evolving;
        self.phase = Phase::Done;
        if self.search_log && was_evolving {
            self.bars.clear();
            println!(
                "   {:>6}  in {}",
                "Searched".green().bold(),
                pretty_duration::pretty_duration(&self.started_at.elapsed(), None)
            );
        }
    }

    /// Finish the search (if the caller stopped early) and return every
    /// measured genome, fastest first.
    pub fn into_ranked(mut self) -> Ranked<M> {
        assert!(
            self.outstanding.is_none(),
            "report the outstanding candidate before finishing the search"
        );
        if self.phase == Phase::Evolving && self.generation_open {
            self.close_generation();
        }
        self.finish();
        std::mem::take(&mut self.ranked)
    }
}

#[cfg(test)]
mod family_tests {
    use super::*;
    use crate::egglog_utils::proposal_tests::{
        choices_fixture, independent_choices_fixture, paired_arguments_fixture,
    };
    use crate::search::BucketSearchSpace;
    use rand::SeedableRng;

    fn family_space(width: usize, variants: usize) -> SearchSpace {
        let mut graph = paired_arguments_fixture(width, false);
        let old = choices_fixture(variants);
        let root = graph.roots[0].clone();
        graph
            .eclasses
            .get_mut(&root)
            .unwrap()
            .1
            .extend(old.eclasses[&root].1.clone());
        graph.enodes.extend(old.enodes);
        graph.node_to_class.extend(old.node_to_class);
        for (class, value) in old.eclasses {
            if class != root {
                graph.eclasses.insert(class, value);
            }
        }
        SearchSpace {
            buckets: vec![BucketSearchSpace {
                egraph: graph,
                bucket_indices: Default::default(),
                intervals: Default::default(),
            }],
            ops: vec![],
            custom_ops: vec![],
            dim_buckets: Default::default(),
        }
    }

    #[test]
    fn losing_family_climbs_multiple_arguments_within_candidate_budget() {
        let space = family_space(2, 0);
        let ctx = BucketContext {
            space: &space,
            index: 0,
            representative_dyn_map: Default::default(),
        };
        let options = CompileOptions::default()
            .search_log(false)
            .search_graph_limit(5);
        let mut search = GeneticSearch::<usize>::new(&space, &ctx, &options, Instant::now());
        // Extraction is irrelevant to this state-machine test. Each fake
        // evaluation reports the metric for an ordinary legal genome.
        let incumbent = search
            .extractor
            .index_seed_choices(&[("root".into(), "op-0".into())]);
        fn mark_seen(search: &mut GeneticSearch<usize>, genome: &IndexedChoiceSet) {
            let bindings: Vec<_> = search
                .extractor
                .named_choices(genome)
                .into_iter()
                .map(|(c, n)| {
                    (
                        egraph_serialize::ClassId::from(c),
                        egraph_serialize::NodeId::from(n),
                    )
                })
                .collect();
            search
                .prev_selected
                .insert(crate::egglog_utils::hash_choice_set(
                    &bindings.iter().map(|(c, n)| (c, n)).collect(),
                ));
        }
        mark_seen(&mut search, &incumbent);
        let candidate = search.hand_out(incumbent, LLIRGraph::default(), None);
        search.report(candidate, Outcome::Measured(100, "incumbent".into()));
        let seed = search
            .extractor
            .index_seed_choices(&[("root".into(), "pair-1-1".into())]);
        mark_seen(&mut search, &seed);
        let candidate = search.hand_out(seed, LLIRGraph::default(), None);
        search.report(candidate, Outcome::Measured(120, "new family".into()));
        assert_eq!(search.best(), Some(&100));
        assert_eq!(
            search
                .refinements
                .iter()
                .map(|(_, q)| q.len())
                .sum::<usize>(),
            2
        );
        while let Some(genome) = search.take_pending() {
            let named = search.extractor.named_choices(&genome);
            let node = &named.iter().find(|(class, _)| class == "root").unwrap().1;
            let metric = if node == "pair-0-0" { 80 } else { 110 };
            let candidate = search.hand_out(genome, LLIRGraph::default(), None);
            search.report(candidate, Outcome::Measured(metric, metric.to_string()));
        }
        assert_eq!(
            search.best(),
            Some(&80),
            "two individually losing changes must compose"
        );
        assert_eq!(search.measured(), 5);
        assert!(
            search
                .next_candidate(&mut rand::rngs::StdRng::seed_from_u64(993))
                .is_none()
        );
        assert_eq!(
            search.measured(),
            5,
            "family exploration consumes the ordinary budget"
        );
    }

    #[test]
    fn improving_losing_family_does_not_starve_queued_algorithms() {
        let space = family_space(16, 3);
        let ctx = BucketContext {
            space: &space,
            index: 0,
            representative_dyn_map: Default::default(),
        };
        let options = CompileOptions::default()
            .search_log(false)
            .search_graph_limit(8);
        let mut search = GeneticSearch::<usize>::new(&space, &ctx, &options, Instant::now());
        let incumbent = search
            .extractor
            .index_seed_choices(&[("root".into(), "op-0".into())]);
        let candidate = search.hand_out(incumbent, LLIRGraph::default(), None);
        search.report(candidate, Outcome::Measured(100, "incumbent".into()));
        let family = search
            .extractor
            .index_seed_choices(&[("root".into(), "pair-15-15".into())]);
        let candidate = search.hand_out(family, LLIRGraph::default(), None);
        search.report(candidate, Outcome::Measured(200, "losing family".into()));
        // These proposals were already generated before refinement began. Only
        // the last is a winner; improvements within the large losing family
        // must leave enough of the same eight-evaluation budget to discover it.
        for i in 1..=3 {
            let genome = search
                .extractor
                .index_seed_choices(&[("root".into(), format!("op-{i}"))]);
            search.pending.push_back(genome);
        }
        let mut tried = Vec::new();
        while search.measured() < 8 {
            let genome = search.take_pending().expect("unmeasured proposals remain");
            let named = search.extractor.named_choices(&genome);
            let node = named
                .iter()
                .find(|(class, _)| class == "root")
                .unwrap()
                .1
                .clone();
            // Every visited family refinement improves, but still loses to the
            // incumbent. This reproduced exhaustion of the serving search budget.
            let metric = if node == "op-3" {
                80
            } else {
                200 - search.measured()
            };
            tried.push(node);
            let candidate = search.hand_out(genome, LLIRGraph::default(), None);
            search.report(candidate, Outcome::Measured(metric, metric.to_string()));
        }
        assert!(
            tried.iter().any(|node| node.starts_with("pair-")),
            "refinement remains useful"
        );
        assert_eq!(
            search.best(),
            Some(&80),
            "queued algorithms must be tested before the budget is exhausted: {tried:?}"
        );
        assert!(
            search
                .next_candidate(&mut rand::rngs::StdRng::seed_from_u64(993))
                .is_none()
        );
        assert_eq!(search.measured(), 8);
    }

    #[test]
    fn winner_recombination_does_not_starve_queued_algorithms() {
        let (graph, roots) = independent_choices_fixture(8, 1);
        let space = SearchSpace {
            buckets: vec![BucketSearchSpace {
                egraph: graph,
                bucket_indices: Default::default(),
                intervals: Default::default(),
            }],
            ops: vec![],
            custom_ops: vec![],
            dim_buckets: Default::default(),
        };
        let ctx = BucketContext {
            space: &space,
            index: 0,
            representative_dyn_map: Default::default(),
        };
        let options = CompileOptions::default()
            .search_log(false)
            .search_graph_limit(4);
        let mut search = GeneticSearch::<usize>::new(&space, &ctx, &options, Instant::now());
        let bindings = |which: fn(usize) -> bool| {
            roots
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    (
                        c.to_string(),
                        format!("value-{i}-variant-{}", usize::from(which(i))),
                    )
                })
                .collect::<Vec<_>>()
        };
        let incumbent = search.extractor.index_seed_choices(&bindings(|i| i < 4));
        let candidate = search.hand_out(incumbent, LLIRGraph::default(), None);
        search.report(candidate, Outcome::Measured(100, "incumbent".into()));
        let probe = search.extractor.index_seed_choices(&bindings(|_| true));
        search.pending.push_back(probe);
        let winner = search.extractor.index_seed_choices(&bindings(|i| i >= 4));
        let candidate = search.hand_out(winner, LLIRGraph::default(), None);
        search.report(
            candidate,
            Outcome::Measured(90, "independent improvements".into()),
        );
        // The two parents disagree at eight sites, generating more combinations
        // than the remaining budget. The queued all-improved program must still
        // be evaluated, while combination proposals retain a share of the work.
        let mut combinations = 0;
        while search.measured() < 4 {
            let genome = search.take_pending().unwrap();
            let all_improved = search
                .extractor
                .named_choices(&genome)
                .iter()
                .filter(|(class, node)| {
                    class.starts_with("value-class-") && node.ends_with("variant-1")
                })
                .count()
                == 8;
            combinations += usize::from(!all_improved);
            let cost = if all_improved { 1 } else { 95 };
            let candidate = search.hand_out(genome, LLIRGraph::default(), None);
            search.report(candidate, Outcome::Measured(cost, cost.to_string()));
        }
        assert_eq!(search.best(), Some(&1));
        assert_eq!(combinations, 1);
        assert!(
            search
                .next_candidate(&mut rand::rngs::StdRng::seed_from_u64(997))
                .is_none()
        );
        assert_eq!(search.measured(), 4);
    }
}
