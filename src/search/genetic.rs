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
use rustc_hash::FxHashSet;

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
    invalid_attempts: usize,
    filter_fails: usize,
    max_filter_fails: usize,
    last_filter_rejection: Option<String>,
    n_timed_out: usize,
    n_invalid_profile: usize,

    // Evolving phase.
    pending: VecDeque<IndexedChoiceSet>,
    generation_open: bool,
    ranked: Ranked<M>,
    parents: Vec<(M, IndexedChoiceSet)>,
    best_metric: Option<M>,
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
        let max_filter_fails = options
            .limit
            .max(1)
            .saturating_mul(options.generation_size.max(1))
            .saturating_mul(100)
            .max(10_000);
        Self {
            space,
            ctx,
            options,
            extractor: LlirExtractor::new(ctx.egraph(), &space.ops),
            profile_dyn_map: ctx.profile_dyn_map(options),
            search_limit: count_choice_sets_up_to(ctx.egraph(), options.limit),
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
            invalid_attempts: 0,
            filter_fails: 0,
            max_filter_fails,
            last_filter_rejection: None,
            n_timed_out: 0,
            n_invalid_profile: 0,
            pending: VecDeque::new(),
            generation_open: false,
            ranked: Vec::new(),
            parents: Vec::new(),
            best_metric: None,
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
                    let mut generation =
                        self.extractor
                            .random_indexed_generation(1, &mut self.prev_selected, rng);
                    let Some(genome) = generation.pop() else {
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
                    if self.pending.is_empty() {
                        if self.generation_open {
                            self.close_generation();
                        }
                        if self.n_graphs >= self.search_limit || self.time_limit_reached() {
                            self.finish();
                            return None;
                        }
                        self.breed(rng);
                        if self.pending.is_empty() {
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
                    let genome = self.pending.pop_front().unwrap();
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
        let new_best = self
            .best_metric
            .as_ref()
            .is_some_and(|best| best.gt(&new_metric));
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
