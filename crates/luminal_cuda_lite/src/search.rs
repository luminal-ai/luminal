//! Device-measured implementation search. Candidate selection requires a GPU.

use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::extractor::{self, Genome};
use luminal::bufferize::BufferIrGraph;
use luminal::prelude::egraph_serialize;
use luminal::prelude::{FxHashMap, FxHashSet};

// The pieces that decide nothing, in core since Phase 8. Re-exported
// under this module's own name: every public path this crate used to
// offer still resolves (`crate::search::early_stop_exceeded` is
// `profile.rs`'s, and the tests read `RefusalBreakdown` off the
// outcome).
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
    /// Point mutations per offspring. Mutations hit ANY producer class —
    /// dead rows included, deliberately: a dead-row mutation is free now and
    /// pre-stages the choice a later route flip lands on.
    pub mutations: usize,
    pub trials: usize,
    pub seed: u64,
    /// Print live search progress to stderr (`Start` / `Faster` /
    /// `Slower x{n}`). ON by default, matching main's
    /// `CompileOptions::search_log`; overridden by `SEARCH_LOG=0`/`1`
    /// or `LUMINAL_LOG=1`.
    pub search_log: bool,
    /// PER-CANDIDATE BUDGET FOR THE TIMED RUN — and for nothing else
    /// (ruling, 2026-09-03: *"timeout should just cover run"*). The
    /// clock starts at the first TIMED trial, after the candidate has
    /// been compiled and warmed, and is checked BETWEEN trials; a
    /// candidate that exceeds it is not ranked and is counted under
    /// [`RefusalBreakdown::timed_out`]. `None` = no budget.
    ///
    /// It is deliberately NOT a compile budget: NVRTC time is paid once
    /// per distinct kernel source across the whole search (the device's
    /// persistent module cache), so charging it to whichever candidate
    /// happened to hit a cold cache would time out plans for a cost
    /// their successors do not pay.
    pub candidate_timeout: Option<Duration>,
    /// Ranked candidates retained for per-program fallback. Additional plans
    /// are extracted only when an earlier candidate fails final validation.
    pub keep_finalists: usize,
    /// Maximum bytes for the entire arena. Search caps this by available CUDA
    /// memory (or uses that capacity when None), prunes individually impossible
    /// materializations, and rejects candidate arenas over the limit before
    /// allocation. Final validation enforces the same per-program budget.
    pub device_budget_bytes: Option<usize>,
    /// Prune individual intermediate materializations above this size before
    /// extraction, preserving boundary storage and zero-copy view alternatives.
    pub max_intermediate_bytes: Option<usize>,
    /// Custom edits after serialization and before mandatory memory pruning.
    pub serialized_graph_passes: Vec<crate::egraph_postpass::SerializedGraphPostPass>,
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
        }
    }
}

impl CompileOptions {
    /// Enable or disable live search progress logging — main's
    /// `CompileOptions::search_log(enabled)` builder, re-expressed.
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
    /// Plans actually profiled (distinct fingerprints).
    pub plans_profiled: usize,
    /// Proposed genomes rejected because their extracted plan fingerprint was
    /// already admitted earlier in the search. Duplicate phenotypes never
    /// consume a generation slot.
    pub fingerprint_hits: usize,
    /// Wall-clock attribution across the pipeline stages — the
    /// programmatic answer to "what is the search time actually spent
    /// on" (no env vars; read it from the outcome).
    pub timings: SearchTimings,
    /// What rejected genomes were rejected FOR (diagnosis ruling
    /// 2026-08-07: understand the breakdown, no auto-repair).
    pub refusal_breakdown: RefusalBreakdown,
    /// Measured candidates, fastest first. `best_*` identifies the finalized
    /// candidate at `finalist_rank`, which can differ from the first measurement.
    pub ranked: Vec<(u128, Genome)>,
    /// Candidates refused during finalization of this program.
    pub finalist_rejections: usize,
    /// Rank of the finalized program in the measured candidate ordering.
    pub finalist_rank: usize,
}

/// Borrowed device and input data for one compilation.
pub enum Evaluator<'a> {
    /// ON-DEVICE MEASUREMENT ([`crate::profile::profile_candidate`]) on
    /// a persistent device — the module cache and the slab are the
    /// runtime's, so kernel compilation is paid once per distinct source
    /// across the search rather than once per candidate.
    ///
    /// The profiling harness borrows `staged` by BufferLit id and explicitly
    /// uploads it before timed execution. It is a map of
    /// references and not of payloads on purpose: a full-size model's
    /// weights are gigabytes on the host and the search must not hold a
    /// second copy of them.
    #[cfg(feature = "device")]
    Device {
        device: &'a mut crate::device::CudaExecutable,
        staged: &'a FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
    },
    /// The lifetime placeholder for builds WITHOUT the `device` feature,
    /// so [`search_implementations`]'s signature is the same in both.
    /// Searching with this placeholder always returns an error.
    #[cfg(not(feature = "device"))]
    #[doc(hidden)]
    NoDevice(std::marker::PhantomData<&'a ()>),
}

impl Evaluator<'_> {
    fn arena_budget(&self, requested: Option<usize>) -> Result<usize> {
        #[cfg(feature = "device")]
        {
            let Self::Device { device, .. } = self;
            let available = device.available_arena_bytes()?;
            Ok(requested.map_or(available, |limit| limit.min(available)))
        }
        #[cfg(not(feature = "device"))]
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
        #[cfg(feature = "device")]
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
        #[cfg(not(feature = "device"))]
        {
            let _ = (plan, options, shapes, best_nanos);
            Priced::PrepareFailed(
                "candidate search requires the `device` feature and a CUDA GPU".into(),
            )
        }
    }

    /// Lend this evaluator to the search driver (which runs
    /// one search per Cartesian combination and must hand the SAME
    /// device to each).
    pub fn reborrow(&mut self) -> Evaluator<'_> {
        match self {
            #[cfg(feature = "device")]
            Evaluator::Device { device, staged } => Evaluator::Device { device, staged },
            #[cfg(not(feature = "device"))]
            Evaluator::NoDevice(marker) => Evaluator::NoDevice(*marker),
        }
    }

    /// Does this evaluator measure on a device?
    fn is_device(&self) -> bool {
        #[cfg(feature = "device")]
        {
            matches!(self, Evaluator::Device { .. })
        }
        #[cfg(not(feature = "device"))]
        {
            false
        }
    }
}

/// INSERT ONE MEASURED CANDIDATE INTO THE RANKING (Phase 5), keeping
/// `ranked` sorted fastest-first and no longer than `keep`.
///
/// TIES GO AFTER (main's `report_evolving` rule, `genetic.rs:469-479`):
/// the insertion point is the FIRST position whose metric the newcomer
/// strictly beats, so an equal-cost incumbent keeps the better rank.
/// That is what makes `ranked[0]` the same genome the incumbent logic
/// crowns — the incumbent is only replaced on a strict improvement too.
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

/// The fastest measured candidate and its extracted plan.
struct Best {
    nanos: u128,
    genome: Genome,
    plan: BufferIrGraph<luminal::layouts::DecodedLayout>,
}

/// What pricing one candidate produced. A cost is ranked; the other
/// three are accounted and the candidate is dropped.
///
/// Only the device evaluator can produce the last three, so a
/// device-free build constructs `Cost` alone.
#[cfg_attr(not(feature = "device"), allow(dead_code))]
enum Priced {
    Cost(u128),
    /// The timed run exceeded [`CompileOptions::candidate_timeout`].
    TimedOut(String),
    /// Compile / stage / warmup failed — an ordinary unfit candidate
    /// (D10), counted with the bufferize refusals.
    PrepareFailed(String),
    /// A timed trial failed after the warmup had succeeded.
    ExecuteFailed(String),
}

/// PLACEMENT FEASIBILITY of one genome's plan. An output bound External
/// sits on the caller's own buffer: a plan that elects that slot as a view
/// of ANOTHER caller buffer, or lets two External slots bound on different
/// buffers share one escape cell, cannot be installed on this boundary —
/// the caller allocated distinct storage. That is a refusal of THIS genome,
/// and the search tries others (a materializing candidate, where the
/// e-graph offers one). The runtime's retarget keeps the same rule as its
/// final tripwire.
fn external_placement_feasible(
    plan: &BufferIrGraph<luminal::layouts::DecodedLayout>,
    outputs: &[crate::bindings::Bound],
) -> Result<(), String> {
    let mut cell_owner: FxHashMap<luminal::bufferize::BufferId, i64> = FxHashMap::default();
    for node in plan.dag.node_weights() {
        let luminal::bufferize::BufferNode::BufferOutput { slots } = node else {
            continue;
        };
        for slot in slots {
            let Some(bound) = outputs.get(slot.index) else {
                continue;
            };
            if bound.placement != crate::bindings::Placement::External {
                continue;
            }
            let Some(cell) = plan.buffers.get(&slot.buffer) else {
                continue;
            };
            match cell.lit {
                Some(lit) if lit == bound.buffer => {}
                Some(other) => {
                    return Err(format!(
                        "output v{} is bound External on buffer {} but this plan elects it \
                         as a view of caller buffer {other}",
                        bound.value.index(),
                        bound.buffer
                    ));
                }
                None => {
                    if let Some(first) = cell_owner.insert(slot.buffer.clone(), bound.buffer)
                        && first != bound.buffer
                    {
                        return Err(format!(
                            "outputs bound External on buffers {first} and {} share one \
                             escape cell {:?}",
                            bound.buffer, slot.buffer
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// THE SELECTION LOOP for this backend: the caller supplies its OWN
/// matcher vocabulary and its OWN allow list — both are properties of
/// the runtime INSTANCE, chosen when it was loaded (see
/// [`crate::CudaRuntime::load_with_registry`]), not of this crate. The
/// vocabulary is BORROWED: one instance runs many extractions (every
/// genome), and `dyn OpMatcher` is
/// not clonable, so the list lives in the runtime and is lent here.
/// Deterministic for a fixed seed.
///
/// Candidates are always compiled, warmed and measured on the device.
// The evaluator is mutated (reborrowed per candidate) only by the
// device arm, which a device-free build compiles out.
#[cfg_attr(not(feature = "device"), allow(unused_mut))]
pub fn search_implementations(
    egraph: &mut egraph_serialize::EGraph,
    program: &SearchProgram,
    shapes: &crate::symbolic::ShapeEnv,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: &[Box<dyn luminal::layout_ir::OpMatcher>],
    mut evaluator: Evaluator<'_>,
) -> Result<SearchOutcome> {
    ensure!(
        evaluator.is_device(),
        "candidate search requires the `device` feature and a CUDA GPU"
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

    // The allow list narrows the caller's matcher set; None = the whole set.
    let allow = allow_override;
    let mut session =
        extractor::ExtractionSession::new_with_matcher_set(egraph, allow.as_deref(), matchers);
    let index = session.producer_index();
    timings.analysis_nanos = analysis_start.elapsed().as_nanos();
    // An empty index is NOT an error: a graph with no searchable producer
    // classes (pure identity — every output is an input value) has a
    // one-point genome space, the empty genome. The search still profiles
    // that single candidate; the fingerprint cache collapses the rest.
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    // TWO-PHASE SAMPLING over the candidate graph's COMPONENTS
    // (ruling 2026-08-07, generalized 2026-09-02). See
    // [`extractor::SamplingSpace`] for the criterion: a choice cycle
    // can only close inside a strongly connected component of the
    // candidate graph, so components ARE the re-description groups —
    // Copy⟷Copy layout welds and the cuBLASLt collapse's
    // `x ≡ Tᵀ(x)` two-logical-value 2-cycle alike, with no op-name
    // pattern list anywhere. Generation 0 samples a FOREST inside each
    // component (see [`sample_genome`]); mutation admits a flip only
    // when it closes no cycle (see [`mutate_genome`]). Everything
    // outside a component is sampled freely: its edges leave the
    // component and the condensation is a DAG.
    let space = session.sampling_space(&index);

    let random_genome = |rng: &mut StdRng| sample_genome(&index, &space, rng);
    let mutate = |parent: &Genome, rng: &mut StdRng, count: usize| {
        mutate_genome(parent, &index, &space, &classes, rng, count)
    };

    // fingerprint → measured nanos (the dedup cache).
    let mut cache: FxHashMap<u64, u128> = FxHashMap::default();
    // THE VIEW the layout decoders read through: this e-graph plus this
    // matcher set's `(sort, constructor)` decoders. Built once — the
    // view indexes classes through the serialized graph's own
    // `classes()` cache, so every later class lookup is a map hit.
    let view = luminal::egglog_utils::eclass::EGraphView::new(egraph, &decoders);
    // THE DECODED-LAYOUT CACHE, one per search and CALLER-OWNED
    // (`decode_layout_table` takes it by `&mut`). Decoding is a pure
    // function of `(layout class, dtype)`, and the table is VALUE-keyed
    // per candidate — so a cache that did not span candidates would
    // re-decode every distinct layout class once per candidate.
    // `egraph` is fixed for this loop, which is what makes the
    // `ClassId` keys comparable across candidates.
    let mut layout_cache = luminal::layouts::LayoutDecodeCache::new();
    let mut plans_profiled = 0usize;
    let mut fingerprint_hits = 0usize;
    // Refusal accounting, minimal form (Step 5 down-payment): keep the
    // first few refusal reasons so a fully-refused search names its
    // causes instead of shrugging.
    let mut refusals: Vec<String> = Vec::new();
    let mut breakdown = RefusalBreakdown::default();
    // THE RANKED MEASURED GENOMES (Phase 5): the finalist fallback list.
    // Only NEWLY PROFILED candidates enter it — a fingerprint cache hit
    // is the same plan under a different genome, and two identical
    // finalists at two ranks would waste the retained candidates on one
    // choice.
    let mut ranked: Vec<(u128, Genome)> = Vec::new();
    let mut best: Option<Best> = None;
    // Live progress (#391), on stderr (via the capture-aware adapter, so
    // test output stays clean) and never on a caller's stdout.
    // `None` = the option (or `SEARCH_LOG`) says quiet.
    let mut progress = options
        .search_log_enabled()
        .then(|| SearchProgress::new(CaptureAwareStderr));

    for generation in 0..options.generations {
        // A generation contains feasible PHENOTYPES, not merely proposed
        // genomes. Mutation is transactional: retain the parent, propose a
        // child, extract and arena-plan it, and admit it only after the exact
        // physical-lifetime plan is within budget. Rejection does not repair
        // the child into some other graph; it simply draws another mutation.
        // This keeps genotype diversity meaningful because duplicate plans do
        // not consume population slots either.
        let parent = best.as_ref().map(|incumbent| incumbent.genome.clone());
        let max_attempts = options.generation_size.saturating_mul(32).max(1);
        let mut attempts = 0usize;
        let mut admitted = 0usize;
        let mut generation_fingerprints: FxHashSet<u64> = FxHashSet::default();
        while admitted < options.generation_size && attempts < max_attempts {
            let genome = match &parent {
                None if attempts.is_multiple_of(2) => {
                    sample_genome_correlated(&index, &space, &mut rng)
                }
                None => random_genome(&mut rng),
                Some(parent) => mutate(parent, &mut rng, options.mutations),
            };
            attempts += 1;
            // Extraction failure = invalid genome (cycle, contract breach):
            // discard; a fresh proposal is the repair.
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
                        // THE SAMPLER INVARIANT (2026-09-02). Sampling
                        // and mutation both keep the genome's
                        // chosen-edge graph acyclic, so the ONLY way a
                        // sampled genome can reach the extractor with a
                        // choice cycle is the documented full-list
                        // fallback (a component position with no
                        // acyclic option at all — its own diagnosis).
                        // A choice cycle on an ACYCLIC chosen-edge
                        // graph means the sampler's notion of a
                        // candidate's inputs has drifted from the
                        // planner's: a bug, and it stops the search
                        // instead of quietly costing genomes.
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
                        // The breakdown names WHY (dead-end classes carry
                        // the unproven-Int-op note and its attestation
                        // door), not just that extraction failed.
                        refusals.push(format!("extract: {err:#}; {summary}"));
                    }
                    continue;
                }
            };
            let fingerprint = extractor::plan_fingerprint(&graph);
            if cache.contains_key(&fingerprint) || !generation_fingerprints.insert(fingerprint) {
                fingerprint_hits += 1;
                continue;
            }
            let build_start = Instant::now();
            // Decode the elected layouts and bufferize before admission: the
            // arena planner consumes this exact plan, so its certificate is
            // the same one device installation will enforce.
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
            if let Err(why) = external_placement_feasible(&plan, &program.outputs) {
                breakdown.plan_build_refusals += 1;
                if refusals.len() < 8 {
                    refusals.push(format!("placement: {why}"));
                }
                continue;
            }
            let arena =
                match crate::storage::plan_storage(&plan, &shapes.bounds, &Default::default()) {
                    Ok(arena) => arena,
                    Err(err) => {
                        breakdown.plan_build_refusals += 1;
                        if refusals.len() < 8 {
                            refusals.push(format!("arena planning: {err:#}"));
                        }
                        continue;
                    }
                };
            if arena.slab_bytes > arena_budget_bytes {
                breakdown.arena_refusals += 1;
                if refusals.len() < 8 {
                    refusals.push(format!(
                        "arena: {} bytes exceeds budget {arena_budget_bytes}",
                        arena.slab_bytes
                    ));
                }
                continue;
            }

            admitted += 1;
            let profile_start = Instant::now();
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
        // Final validation has not run yet; the runtime fills these fields.
        finalist_rejections: 0,
        finalist_rank: 1,
    })
}

// ===========================================================================
// PHASE 5: FINALIST VALIDATION — how a searched winner becomes
// an INSTALLED plan.
// ===========================================================================

/// Compile, stage and execute the finalist once on the device. The candidate
/// timeout covers timed trials only, so it does not apply to this warmup.
pub fn finalist_validate(
    pending: &crate::finalists::PendingFinalist,
    _options: &CompileOptions,
    evaluator: &mut Evaluator<'_>,
) -> Result<(), String> {
    #[cfg(feature = "device")]
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
    #[cfg(not(feature = "device"))]
    {
        let _ = (pending, evaluator);
        Err("finalist validation requires the `device` feature and a CUDA GPU".into())
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
