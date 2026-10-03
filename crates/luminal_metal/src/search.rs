//! Device-measured implementation search. Candidate selection requires a GPU.

use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::extractor::{self, Genome};
use luminal::bufferize::BufferIrGraph;
use luminal::prelude::FxHashMap;
use luminal::prelude::egraph_serialize;

pub use luminal::search_support::{
    CaptureAwareStderr, ProducerIndex, RefusalBreakdown, SearchProgress, SearchTimings,
    bufferize_cycle_tripwire, early_stop_exceeded, log_channel_enabled, mutate_genome,
    mutate_genome_reporting, mutate_genome_with_seed, sample_genome, sample_genome_correlated,
    sample_genome_reporting, sample_genome_with_preferred_family, sample_genome_with_seed,
};

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub generations: usize,
    pub generation_size: usize,
    pub mutations: usize,
    pub trials: usize,
    pub seed: u64,
    pub search_log: bool,
    pub candidate_timeout: Option<Duration>,
    pub keep_finalists: usize,
    /// Maximum arena size, capped by the device's maximum buffer length.
    /// Applied before search to materializations and to candidate allocations.
    pub device_budget_bytes: Option<usize>,
    /// Prune individual intermediate materializations above this size before
    /// extraction, preserving boundary storage and zero-copy view alternatives.
    pub max_intermediate_bytes: Option<usize>,
    /// Custom edits after serialization and before mandatory memory pruning.
    pub serialized_graph_passes: Vec<crate::egraph_postpass::SerializedGraphPostPass>,
    /// Cumulative matches per ring expansion rule; None requests exhaustive saturation.
    pub algebra_match_budget: Option<usize>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            generations: 8,
            generation_size: 8,
            mutations: 2,
            trials: 3,
            seed: 0,
            search_log: true,
            candidate_timeout: None,
            keep_finalists: 4,
            device_budget_bytes: None,
            max_intermediate_bytes: None,
            serialized_graph_passes: Vec::new(),
            algebra_match_budget: Some(crate::saturation::DEFAULT_ALGEBRA_MATCH_BUDGET),
        }
    }
}

impl CompileOptions {
    pub fn search_log(mut self, enabled: bool) -> Self {
        self.search_log = enabled;
        self
    }

    fn search_log_enabled(&self) -> bool {
        log_channel_enabled(self.search_log, "SEARCH_LOG")
    }
}

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub best_plan: BufferIrGraph<luminal::layouts::DecodedLayout>,
    pub best_genome: Genome,
    pub memory_pruning: crate::egraph_postpass::MemoryPruning,
    pub best_nanos: u128,
    pub plans_profiled: usize,
    pub fingerprint_hits: usize,
    pub timings: SearchTimings,
    pub refusal_breakdown: RefusalBreakdown,
    pub ranked: Vec<(u128, Genome)>,
    pub finalist_rejections: usize,
    /// Rank of the finalized program in the measured candidate ordering.
    pub finalist_rank: usize,
}

/// Borrowed device and input data for one compilation.
pub enum Evaluator<'a> {
    #[cfg(target_os = "macos")]
    Device {
        device: &'a mut crate::device::MetalExecutable,
        staged: &'a FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
    },
    #[cfg(not(target_os = "macos"))]
    #[doc(hidden)]
    NoDevice(std::marker::PhantomData<&'a ()>),
}

impl Evaluator<'_> {
    fn arena_budget(&self, requested: Option<usize>) -> Result<usize> {
        #[cfg(target_os = "macos")]
        {
            let Self::Device { device, .. } = self;
            let available = device.available_arena_bytes()?;
            Ok(requested.map_or(available, |limit| limit.min(available)))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = requested;
            Err(anyhow!("device search requires a GPU"))
        }
    }

    fn measure(
        &mut self,
        plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
        options: &CompileOptions,
        shapes: &crate::symbolic::ShapeEnv,
        best_nanos: Option<u128>,
    ) -> Priced {
        #[cfg(target_os = "macos")]
        {
            let Self::Device { device, staged } = self;
            let measured = crate::profile::profile_candidate_at(
                device,
                plan,
                staged,
                options.trials,
                best_nanos,
                options.candidate_timeout,
                shapes,
                options.device_budget_bytes,
            );
            device.uninstall();
            match measured {
                Ok(crate::profile::Measurement::Timed { mean_nanos, .. }) => {
                    Priced::Cost(mean_nanos)
                }
                Ok(crate::profile::Measurement::TimedOut {
                    elapsed_nanos,
                    completed_trials,
                }) => Priced::TimedOut(format!(
                    "candidate exceeded the timed-run budget after \
                     {completed_trials} trial(s), {:.3} ms elapsed",
                    elapsed_nanos as f64 / 1e6
                )),
                Err(crate::profile::ProfileFailure::Prepare(err)) => {
                    Priced::PrepareFailed(format!("{err:#}"))
                }
                Err(crate::profile::ProfileFailure::Execute(err)) => {
                    Priced::ExecuteFailed(format!("{err:#}"))
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (plan, options, shapes, best_nanos);
            Priced::PrepareFailed("candidate search requires macOS and a Metal GPU".into())
        }
    }

    pub fn reborrow(&mut self) -> Evaluator<'_> {
        match self {
            #[cfg(target_os = "macos")]
            Evaluator::Device { device, staged } => Evaluator::Device { device, staged },
            #[cfg(not(target_os = "macos"))]
            Evaluator::NoDevice(marker) => Evaluator::NoDevice(*marker),
        }
    }

    fn is_device(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            matches!(self, Evaluator::Device { .. })
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

fn rank_insert(ranked: &mut Vec<(u128, Genome)>, nanos: u128, genome: &Genome, keep: usize) {
    let keep = keep.max(1);
    if ranked.len() >= keep && ranked.last().is_some_and(|(worst, _)| nanos >= *worst) {
        return; // cannot displace anyone
    }
    let position = ranked
        .iter()
        .position(|(metric, _)| nanos < *metric)
        .unwrap_or(ranked.len());
    ranked.insert(position, (nanos, genome.clone()));
    ranked.truncate(keep);
}

struct Best {
    nanos: u128,
    genome: Genome,
    plan: BufferIrGraph<luminal::layouts::DecodedLayout>,
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
enum Priced {
    Cost(u128),
    TimedOut(String),
    PrepareFailed(String),
    ExecuteFailed(String),
}

pub fn search_implementations(
    egraph: &mut egraph_serialize::EGraph,
    program: &SearchProgram,
    shapes: &crate::symbolic::ShapeEnv,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: &[Box<dyn luminal::layout_ir::OpMatcher>],
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))] mut evaluator: Evaluator<'_>,
) -> Result<SearchOutcome> {
    ensure!(
        evaluator.is_device(),
        "candidate search requires macOS and a Metal GPU"
    );
    let mut timings = SearchTimings::default();
    let analysis_start = Instant::now();
    let decoders = luminal::egglog_snippet::decoder_registry_for(matchers)?;
    let arena_budget_bytes = evaluator.arena_budget(options.device_budget_bytes)?;
    let mut resolved_options = options.clone();
    resolved_options.device_budget_bytes = Some(arena_budget_bytes);
    let options = &resolved_options;
    let memory_pruning = crate::egraph_postpass::run(
        egraph,
        &crate::egraph_postpass::PostPassContext {
            decoders: &decoders,
            bounds: &shapes.bounds,
            arena_budget_bytes,
            max_intermediate_bytes: options.max_intermediate_bytes,
            matchers,
        },
        &options.serialized_graph_passes,
    )?;
    if options.search_log_enabled() && memory_pruning.oversized_tensors > 0 {
        eprintln!(
            "Arena memory pass: removed {} oversized tensors and {} producer classes ({} nodes), budget {:.2} GiB",
            memory_pruning.oversized_tensors,
            memory_pruning.producer_classes,
            memory_pruning.removed_nodes,
            arena_budget_bytes as f64 / 1073741824.0
        );
    }

    let allow = allow_override;
    if options.search_log_enabled() {
        eprintln!("Metal: building producer index");
    }
    let mut session =
        extractor::ExtractionSession::new_with_matcher_set(egraph, allow.as_deref(), matchers);
    let index = session.producer_index();
    if options.search_log_enabled() {
        eprintln!("Metal: producer index ready ({} classes)", index.len());
    }
    timings.analysis_nanos = analysis_start.elapsed().as_nanos();
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    let space = session.sampling_space(&index);
    if options.search_log_enabled() {
        eprintln!("Metal: sampling space ready");
    }

    let families: Vec<_> = index
        .values()
        .flatten()
        .map(|(family, _)| family.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut family_cursor = 0usize;

    let random_genome = |rng: &mut StdRng| sample_genome(&index, &space, rng);
    let mutate = |parent: &Genome, rng: &mut StdRng, count: usize| {
        mutate_genome(parent, &index, &space, &classes, rng, count)
    };

    let mut cache: FxHashMap<u64, u128> = FxHashMap::default();
    let view = luminal::egglog_utils::eclass::EGraphView::new(egraph, &decoders);
    let mut layout_cache = luminal::layouts::LayoutDecodeCache::new();
    let mut plans_profiled = 0usize;
    let mut fingerprint_hits = 0usize;
    let mut refusals: Vec<String> = Vec::new();
    let mut breakdown = RefusalBreakdown::default();
    let mut ranked: Vec<(u128, Genome)> = Vec::new();
    let mut best: Option<Best> = None;
    let mut progress = options
        .search_log_enabled()
        .then(|| SearchProgress::new(CaptureAwareStderr));

    for generation in 0..options.generations {
        let mut candidates: Vec<Genome> = Vec::with_capacity(options.generation_size);
        match &best {
            None => {
                while candidates.len() < options.generation_size {
                    candidates.push(
                        if candidates.len().is_multiple_of(2) && !families.is_empty() {
                            let preferred = &families[family_cursor % families.len()];
                            family_cursor += 1;
                            sample_genome_with_preferred_family(&index, &space, &mut rng, preferred)
                        } else {
                            random_genome(&mut rng)
                        },
                    );
                }
            }
            Some(incumbent) => {
                let parent = incumbent.genome.clone();
                for _ in 0..options.generation_size {
                    candidates.push(mutate(&parent, &mut rng, options.mutations));
                }
            }
        }

        for genome in candidates {
            let extract_start = Instant::now();
            let extracted = session.extract_with_genome(&genome);
            timings.extract_nanos += extract_start.elapsed().as_nanos();
            let graph = match extracted {
                Ok(Some(graph)) => graph,
                Ok(None) => {
                    breakdown.extract_refusals += 1;
                    if refusals.len() < 8 {
                        refusals.push("extract: no boundary reached".to_string());
                    }
                    continue;
                }
                Err(err) => {
                    breakdown.extract_refusals += 1;
                    let (cycle, dead_end, summary) = session.failure_breakdown();
                    if cycle {
                        breakdown.with_choice_cycles += 1;
                        let edges = space.chosen_edges(&index, &genome);
                        ensure!(
                            extractor::edges_have_cycle(&edges),
                            "sampler invariant violated: choice cycle in a sampled genome \
                             whose chosen-edge graph is acyclic — the sampler's candidate \
                             inputs disagree with the extractor's; {summary}"
                        );
                    }
                    if dead_end {
                        breakdown.with_dead_ends += 1;
                    }
                    if breakdown.exemplars.len() < 4 {
                        breakdown.exemplars.push(summary.clone());
                    }
                    if refusals.len() < 8 {
                        refusals.push(format!("extract: {err:#}; {summary}"));
                    }
                    continue;
                }
            };
            let fingerprint = extractor::plan_fingerprint(&graph);
            let nanos = match cache.get(&fingerprint) {
                Some(nanos) => {
                    fingerprint_hits += 1;
                    *nanos
                }
                None => {
                    let build_start = Instant::now();
                    let dps = luminal::dps::dps_rewrite(&graph);
                    let built = luminal::layouts::decode_layout_table(
                        &view,
                        &dps,
                        "implementation search",
                        &mut layout_cache,
                    )
                    .and_then(|table| luminal::bufferize::bufferize(&dps, &table));
                    timings.plan_build_nanos += build_start.elapsed().as_nanos();
                    let plan = match built {
                        Ok(plan) => plan,
                        Err(err) => {
                            bufferize_cycle_tripwire(&err, &index, &space, &genome)?;
                            breakdown.plan_build_refusals += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("bufferize: {err:#}"));
                            }
                            continue;
                        }
                    };
                    let profile_start = Instant::now();
                    if options.search_log_enabled() && best.is_none() {
                        eprintln!(
                            "Metal: profiling candidate ({} buffers)",
                            plan.buffers.len()
                        );
                    }
                    let priced = evaluator.measure(
                        &plan,
                        options,
                        shapes,
                        best.as_ref().map(|incumbent| incumbent.nanos),
                    );
                    timings.profile_nanos += profile_start.elapsed().as_nanos();
                    let nanos = match priced {
                        Priced::Cost(nanos) => nanos,
                        Priced::TimedOut(note) => {
                            breakdown.timed_out += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("timed out: {note}"));
                            }
                            continue;
                        }
                        Priced::PrepareFailed(note) => {
                            breakdown.plan_build_refusals += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("device prepare: {note}"));
                            }
                            continue;
                        }
                        Priced::ExecuteFailed(note) => {
                            breakdown.execute_refusals += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("execute: {note}"));
                            }
                            continue;
                        }
                    };
                    cache.insert(fingerprint, nanos);
                    plans_profiled += 1;
                    rank_insert(&mut ranked, nanos, &genome, options.keep_finalists);
                    let improved = best
                        .as_ref()
                        .is_none_or(|incumbent| nanos < incumbent.nanos);
                    if let Some(progress) = progress.as_mut() {
                        if plans_profiled == 1 {
                            progress.start(nanos);
                        } else {
                            progress.report(improved, nanos);
                        }
                    }
                    if improved {
                        best = Some(Best {
                            nanos,
                            genome: genome.clone(),
                            plan,
                        });
                    }
                    continue;
                }
            };
            if best
                .as_ref()
                .is_none_or(|incumbent| nanos < incumbent.nanos)
            {
                let build_start = Instant::now();
                let dps = luminal::dps::dps_rewrite(&graph);
                let built = luminal::layouts::decode_layout_table(
                    &view,
                    &dps,
                    "implementation search",
                    &mut layout_cache,
                )
                .and_then(|table| luminal::bufferize::bufferize(&dps, &table));
                timings.plan_build_nanos += build_start.elapsed().as_nanos();
                let plan = match built {
                    Ok(plan) => plan,
                    Err(err) => {
                        bufferize_cycle_tripwire(&err, &index, &space, &genome)?;
                        continue;
                    }
                };
                rank_insert(&mut ranked, nanos, &genome, options.keep_finalists);
                best = Some(Best {
                    nanos,
                    genome: genome.clone(),
                    plan,
                });
            }
        }

        if best.is_none() && generation + 1 == options.generations {
            break;
        }
    }

    if let Some(progress) = progress.as_mut() {
        progress.finish();
    }
    let best = best.ok_or_else(|| {
        anyhow!("no candidate genome produced an executable plan; refusals: {refusals:#?}")
    })?;
    let _ = program; // Boundary metadata is retained by the runtime.
    Ok(SearchOutcome {
        best_plan: best.plan,
        best_genome: best.genome,
        memory_pruning,
        best_nanos: best.nanos,
        plans_profiled,
        fingerprint_hits,
        timings,
        refusal_breakdown: breakdown,
        ranked,
        finalist_rejections: 0,
        finalist_rank: 1,
    })
}

pub fn finalist_validate(
    pending: &crate::finalists::PendingFinalist,
    _options: &CompileOptions,
    evaluator: &mut Evaluator<'_>,
) -> Result<(), String> {
    crate::kernels::validate_plan(&pending.plan)
        .map_err(|error| format!("Metal code generation: {error:#}"))?;
    #[cfg(target_os = "macos")]
    {
        let Evaluator::Device { device, staged } = evaluator;
        let ran = crate::profile::warmup_candidate(
            device,
            &pending.plan,
            staged,
            &pending.shapes,
            _options.device_budget_bytes,
        );
        device.uninstall();
        ran.map_err(|err| format!("device warmup of ranked #{}: {err:#}", pending.rank))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (pending, evaluator);
        Err("finalist validation requires macOS and a Metal GPU".into())
    }
}

/// Select one program's first feasible ranked candidate. Budgets belong to
/// this search; applications compose independently selected programs.
pub fn select_finalist(
    mut finalists: crate::finalists::Finalists<'_>,
    options: &CompileOptions,
    evaluator: &mut Evaluator<'_>,
) -> Result<(crate::finalists::PendingFinalist, usize)> {
    let mut validate = |candidate: &crate::finalists::PendingFinalist| {
        if let Some(budget) = options.device_budget_bytes
            && candidate.arena.slab_bytes > budget
        {
            return Err(format!(
                "program requires {} arena bytes, budget {budget}",
                candidate.arena.slab_bytes
            ));
        }
        finalist_validate(candidate, options, evaluator)
    };
    if !finalists.ensure(0, &mut validate) {
        return Err(anyhow!(finalists.failure_message()));
    }
    let rejections = finalists.rejections();
    Ok((finalists.take(0).expect("validated finalist"), rejections))
}

#[derive(Debug, Clone)]
pub struct SearchProgram {
    pub text: String,
    pub inputs: Vec<crate::bindings::Bound>,
    pub outputs: Vec<crate::bindings::Bound>,
}

/// Small deterministic search configuration for tests and examples.
pub fn harness_search_options() -> CompileOptions {
    CompileOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
        ..CompileOptions::default()
    }
}
