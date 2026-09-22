//! Device-measured implementation search. Candidate selection requires a GPU.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use colored::Colorize;
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
    sample_genome_reporting, sample_genome_with_seed,
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
    pub shapes: crate::symbolic::ShapeEnv,
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
            shapes: Default::default(),
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
    pub lattice_rejections: usize,
}

/// Representative-specific host payloads, borrowed without copying model weights.
pub type ProfileInputs<'a> = (
    luminal::shape::DynMap,
    FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
);

#[cfg(target_os = "macos")]
fn staged_at<'a>(
    staged: &FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
    profiles: &[ProfileInputs<'a>],
    dims: &luminal::shape::DynMap,
) -> Result<FxHashMap<i64, &'a crate::host_buffer::HostBuffer>> {
    let mut data = staged.clone();
    if !profiles.is_empty() {
        let mut matches = profiles.iter().filter(|(shape, _)| shape == dims);
        let (_, inputs) = matches
            .next()
            .ok_or_else(|| anyhow!("no profiling inputs for {dims:?}"))?;
        ensure!(
            matches.next().is_none(),
            "ambiguous profiling inputs for {dims:?}"
        );
        data.extend(inputs.iter().map(|(id, data)| (*id, *data)));
    }
    Ok(data)
}

/// A persistent device and borrowed inputs for candidate measurements.
pub enum Evaluator<'a> {
    #[cfg(target_os = "macos")]
    Device {
        device: &'a mut crate::device::MetalDevice,
        staged: &'a FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
        residents: &'a luminal::resident::ResidentBindings,
        profile_inputs: &'a [ProfileInputs<'a>],
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
        best_nanos: Option<u128>,
    ) -> Priced {
        #[cfg(target_os = "macos")]
        {
            let Self::Device {
                device,
                staged,
                residents,
                profile_inputs,
            } = self;
            let staged = match staged_at(staged, profile_inputs, &options.shapes.values) {
                Ok(staged) => staged,
                Err(error) => return Priced::PrepareFailed(format!("{error:#}")),
            };
            let measured = crate::profile::profile_candidate_at(
                device,
                plan,
                &staged,
                residents,
                options.trials,
                best_nanos,
                options.candidate_timeout,
                &options.shapes,
                options.device_budget_bytes,
            );
            device.release_slab();
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
            let _ = (plan, options, best_nanos);
            Priced::PrepareFailed("candidate search requires macOS and a Metal GPU".into())
        }
    }

    pub fn reborrow(&mut self) -> Evaluator<'_> {
        match self {
            #[cfg(target_os = "macos")]
            Evaluator::Device {
                device,
                staged,
                residents,
                profile_inputs,
            } => Evaluator::Device {
                device,
                staged,
                residents,
                profile_inputs,
            },
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
            bounds: &options.shapes.bounds,
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
    let mut session =
        extractor::ExtractionSession::new_with_matcher_set(egraph, allow.as_deref(), matchers);
    let index = session.producer_index();
    timings.analysis_nanos = analysis_start.elapsed().as_nanos();
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    let space = session.sampling_space(&index);

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
                    candidates.push(if candidates.len().is_multiple_of(2) {
                        sample_genome_correlated(&index, &space, &mut rng)
                    } else {
                        random_genome(&mut rng)
                    });
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
                    let priced = evaluator.measure(
                        &plan,
                        options,
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
    let _ = program; // binding tables travel with the caller; kept for future bucket plumbing
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
        lattice_rejections: 0,
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
        let Evaluator::Device {
            device,
            staged,
            residents,
            profile_inputs,
        } = evaluator;
        let ran = staged_at(staged, profile_inputs, &pending.shapes.values).and_then(|staged| {
            crate::profile::prepare_candidate(
                device,
                &pending.plan,
                &staged,
                residents,
                &pending.shapes,
                _options.device_budget_bytes,
            )
            .and_then(|staged| {
                let borrowed = staged.iter().map(|(k, v)| (*k, v.as_ref())).collect();
                device.execute(0, &borrowed, &pending.shapes.values)
            })
        });
        device.release_slab();
        ran.map_err(|err| format!("device warmup of ranked #{}: {err:#}", pending.rank))?;
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (pending, evaluator);
        Err("finalist validation requires macOS and a Metal GPU".into())
    }
}

fn validate_set(slab_bytes: &[usize], options: &CompileOptions) -> Result<(), String> {
    let Some(budget) = options.device_budget_bytes else {
        return Ok(());
    };
    let peak = slab_bytes.iter().copied().max().unwrap_or(0);
    if peak > budget {
        return Err(format!(
            "the set's plans need an arena slab of {peak} bytes (per-bucket \
             {slab_bytes:?}), over the {budget}-byte device budget"
        ));
    }
    Ok(())
}

pub fn select_finalist_set(
    buckets: Vec<crate::finalists::Finalists<'_>>,
    options: &CompileOptions,
    evaluator: &mut Evaluator<'_>,
) -> Result<(Vec<(usize, crate::finalists::PendingFinalist)>, usize)> {
    let mut lattice = crate::lattice::BucketLattice::new(buckets, crate::lattice::sum_metrics);
    let mut validate = |pending: &crate::finalists::PendingFinalist| -> Result<(), String> {
        finalist_validate(pending, options, evaluator)
    };
    loop {
        let Some(set) = lattice.next(&mut validate) else {
            return Err(anyhow!("{}", lattice.failure_message()));
        };
        let slabs = lattice.slab_bytes(&set);
        match validate_set(&slabs, options) {
            Ok(()) => {
                let rejections = lattice.rejections();
                if rejections > 0 && options.search_log_enabled() {
                    eprintln!(
                        "   {} finalist ranks {:?} after {rejections} rejection(s)",
                        "Fallback".yellow().bold(),
                        lattice.ranks(&set)
                    );
                }
                if options.search_log_enabled() {
                    eprintln!("Metal: selected arena capacities {slabs:?} bytes");
                }
                return Ok((lattice.select(&set), rejections));
            }
            Err(reason) => lattice.reject(&set, reason, &mut validate),
        }
    }
}

/// The program a search runs: its text, plus the boundary bindings the
/// tensor-keyed caller data maps through.
#[derive(Debug, Clone)]
pub struct SearchProgram {
    pub text: String,
    pub inputs: Vec<crate::bindings::Bound>,
    pub outputs: Vec<crate::bindings::Bound>,
}

#[derive(Debug)]
pub struct BucketPlan {
    pub ranges: BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: luminal::shape::DynMap,
    pub program: SearchProgram,
    pub outcome: SearchOutcome,
    pub plan: crate::layouts::MetalPlan,
    pub finalist_rank: usize,
    pub slab_bytes: usize,
}

pub struct BucketAssembly<'a> {
    pub assembled_program: &'a str,
    pub prefix: &'a str,
    pub binding_seeds: &'a str,
    pub schedule: &'a str,
    pub post_checks: &'a str,
    pub inputs: &'a [crate::bindings::Bound],
    pub outputs: &'a [crate::bindings::Bound],
    pub base_dims: &'a luminal::shape::DynMap,
    pub decoders: &'a luminal::egglog_utils::eclass::ConstructorRegistry,
}

pub fn bucketed_search_implementations(
    assembly: &BucketAssembly<'_>,
    dim_buckets: &BTreeMap<luminal::shape::Symbol, Vec<luminal::graph::DimBucket>>,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: &[Box<dyn luminal::layout_ir::OpMatcher>],
    mut evaluator: Evaluator<'_>,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");
    let mut egraphs: Vec<egraph_serialize::EGraph> = Vec::new();
    let mut searched: Vec<SearchedBucket> = Vec::new();
    for (ranges, representative, program) in bucket_renders(assembly, dim_buckets)? {
        if options.search_log_enabled() {
            eprintln!(
                "Searching bucket {ranges:?}, representative {representative:?}: {} x {} candidate attempts",
                options.generations, options.generation_size
            );
        }
        let mut bucket_options = options.clone();
        bucket_options.shapes.values = representative.clone();
        bucket_options
            .shapes
            .bounds
            .extend(ranges.iter().map(|(k, v)| (*k, *v)));
        let text = format!("{}\n\n{}", assembly.assembled_program, program.text);
        if options.search_log_enabled() {
            eprintln!("Metal: saturating bucket {ranges:?}");
        }
        if let Ok(path) = std::env::var("LUMINAL_METAL_DUMP_PROGRAM") {
            std::fs::write(path, &text)?;
        }
        let saturation_start = Instant::now();
        let mut egraph = luminal::egglog_snippet::new_egraph();
        crate::saturation::run_program(&mut egraph, &text, options.algebra_match_budget)
            .map_err(|err| anyhow!("bucket {ranges:?} range render fails: {err}"))?;
        assembly.decoders.check(&egraph)?;
        if options.search_log_enabled() {
            eprintln!(
                "Metal: bucket {ranges:?} saturated in {:.2}s ({} tuples)",
                saturation_start.elapsed().as_secs_f64(),
                egraph.num_tuples()
            );
        }

        let mut serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let outcome = search_implementations(
            &mut serialized,
            &program,
            &bucket_options,
            allow_override.clone(),
            matchers,
            evaluator.reborrow(),
        )?;
        egraphs.push(serialized);
        searched.push((ranges, representative, program, outcome));
    }

    let (selected, rejections) = {
        let buckets: Vec<crate::finalists::Finalists<'_>> = searched
            .iter()
            .zip(&egraphs)
            .enumerate()
            .map(|(index, ((ranges, representative, _, outcome), egraph))| {
                let mut shapes = options.shapes.clone();
                shapes.bounds.extend(ranges.iter().map(|(k, v)| (*k, *v)));
                shapes.values = representative.clone();
                crate::finalists::Finalists::new(
                    bucket_label(index, ranges),
                    egraph,
                    allow_override.clone(),
                    matchers,
                    outcome.ranked.clone(),
                    Some(outcome.best_plan.clone()),
                )
                .with_shapes(shapes)
            })
            .collect();
        select_finalist_set(buckets, options, &mut evaluator)?
    };

    let mut installed: BTreeMap<usize, crate::finalists::PendingFinalist> =
        selected.into_iter().collect();
    let mut plans = Vec::new();
    for (index, (ranges, representative, program, mut outcome)) in searched.into_iter().enumerate()
    {
        let finalist = installed
            .remove(&index)
            .ok_or_else(|| anyhow!("the lattice selected no plan for bucket {index}"))?;
        outcome.lattice_rejections = rejections;
        plans.push(BucketPlan {
            ranges,
            representative,
            program,
            outcome,
            slab_bytes: finalist.arena.slab_bytes,
            finalist_rank: finalist.rank,
            plan: finalist.plan,
        });
    }
    Ok(plans)
}

type SearchedBucket = (
    BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    luminal::shape::DynMap,
    SearchProgram,
    SearchOutcome,
);

pub(crate) fn bucket_label(
    index: usize,
    ranges: &BTreeMap<luminal::shape::Symbol, (usize, usize)>,
) -> String {
    let dims: Vec<String> = ranges
        .iter()
        .map(|(dim, (min, max))| format!("{dim} in [{min}, {max}]"))
        .collect();
    format!("bucket {index} ({})", dims.join(", "))
}

type BucketRender = (
    BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    luminal::shape::DynMap,
    SearchProgram,
);

fn bucket_renders(
    assembly: &BucketAssembly<'_>,
    dim_buckets: &BTreeMap<luminal::shape::Symbol, Vec<luminal::graph::DimBucket>>,
) -> Result<Vec<BucketRender>> {
    let seeds_text = |seeds: &BTreeMap<luminal::shape::Symbol, (u64, u64)>| {
        let mut text = String::new();
        for (var, (lower, upper)) in seeds {
            text.push_str(&format!(
                "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
                 (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
            ));
        }
        text
    };
    let assemble = |seeds: &BTreeMap<luminal::shape::Symbol, (u64, u64)>| SearchProgram {
        text: format!(
            "{}{}{}{}{}",
            assembly.prefix,
            assembly.binding_seeds,
            seeds_text(seeds),
            assembly.schedule,
            assembly.post_checks
        ),
        inputs: assembly.inputs.to_vec(),
        outputs: assembly.outputs.to_vec(),
    };

    let dims: Vec<&luminal::shape::Symbol> = dim_buckets.keys().collect();
    let mut combos: Vec<Vec<usize>> = vec![Vec::new()];
    for dim in &dims {
        let count = dim_buckets[*dim].len();
        combos = combos
            .into_iter()
            .flat_map(|combo| {
                (0..count).map(move |index| {
                    let mut next = combo.clone();
                    next.push(index);
                    next
                })
            })
            .collect();
    }

    let mut renders = Vec::new();
    for combo in combos {
        let mut ranges = BTreeMap::new();
        let mut representative = assembly.base_dims.clone();
        for (dim, bucket_index) in dims.iter().zip(&combo) {
            let bucket = &dim_buckets[*dim][*bucket_index];
            ranges.insert(**dim, (bucket.min, bucket.max));
            representative.insert(**dim, bucket.representative_value());
        }

        let mut validation_seeds: BTreeMap<luminal::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, (min, max)) in &ranges {
            validation_seeds.insert(*dim, (*min as u64, *max as u64));
        }
        let validation = assemble(&validation_seeds);

        renders.push((ranges, representative, validation));
    }
    Ok(renders)
}

pub fn select_bucket<'a>(
    plans: &'a [BucketPlan],
    dims: &luminal::shape::DynMap,
) -> Option<&'a BucketPlan> {
    plans.iter().find(|plan| {
        plan.ranges.iter().all(|(dim, (min, max))| {
            dims.get(dim)
                .is_some_and(|value| value >= min && value <= max)
        })
    })
}

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
