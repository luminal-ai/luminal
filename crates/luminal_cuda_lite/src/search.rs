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
    /// HOW MANY RANKED GENOMES THE SEARCH KEEPS (Phase 5, 2026-09-03),
    /// fastest first, for [`crate::finalists::Finalists`] to fall back
    /// through. 1 reproduces the pre-Phase-5 behaviour exactly (only the
    /// winner is ever installable); the default 4 gives the bucket
    /// lattice three runners-up per bucket to walk into when a set-level
    /// constraint refuses the fastest.
    ///
    /// It costs NOTHING when nothing refuses: finalists past rank 0 are
    /// extracted only if the walk reaches them.
    pub keep_finalists: usize,
    /// Maximum bytes for the entire arena. Search caps this by available CUDA
    /// memory (or uses that capacity when None), prunes individually impossible
    /// materializations, and rejects candidate arenas over the limit before
    /// allocation. The finalist lattice also enforces the requested set budget.
    pub device_budget_bytes: Option<usize>,
    /// Prune individual intermediate materializations above this size before
    /// extraction, preserving boundary storage and zero-copy view alternatives.
    pub max_intermediate_bytes: Option<usize>,
    /// Custom edits after serialization and before mandatory memory pruning.
    pub serialized_graph_passes: Vec<crate::egraph_postpass::SerializedGraphPostPass>,
    /// Shape environment for lower-level search callers. The runtime fills this from bindings.
    pub shapes: crate::symbolic::ShapeEnv,
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
    /// Candidates answered from the fingerprint cache without re-profiling.
    pub fingerprint_hits: usize,
    /// Wall-clock attribution across the pipeline stages — the
    /// programmatic answer to "what is the search time actually spent
    /// on" (no env vars; read it from the outcome).
    pub timings: SearchTimings,
    /// What rejected genomes were rejected FOR (diagnosis ruling
    /// 2026-08-07: understand the breakdown, no auto-repair).
    pub refusal_breakdown: RefusalBreakdown,
    /// THE RANKED MEASURED GENOMES (Phase 5), fastest first, at most
    /// `CompileOptions::keep_finalists` of them. `ranked[0]` is the
    /// winner — the same genome as `best_genome`, by construction.
    ///
    /// This is the raw material for [`crate::finalists::Finalists`]: the
    /// runner-ups a set-level constraint can fall back to. Genomes, not
    /// plans, because a plan is the expensive half and the walk may
    /// never need it.
    pub ranked: Vec<(u128, Genome)>,
    /// HOW MANY PLAN SETS THE BUCKET LATTICE REJECTED before installing
    /// one (Phase 5). 0 means the search's own winner (per bucket) was
    /// installed unchanged — which is what every unconstrained search
    /// reports. Stamped by the runtime after the lattice runs; the
    /// genetic search itself always leaves it 0.
    pub lattice_rejections: usize,
}

/// Representative-specific host payloads, borrowed without copying model weights.
pub type ProfileInputs<'a> = (
    luminal::shape::DynMap,
    FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
);

#[cfg(feature = "device")]
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
    /// ON-DEVICE MEASUREMENT ([`crate::profile::profile_candidate`]) on
    /// a persistent device — the module cache and the slab are the
    /// runtime's, so kernel compilation is paid once per distinct source
    /// across the search rather than once per candidate.
    ///
    /// `staged` is BORROWED, by BufferLit id, exactly as
    /// [`crate::device::execute_plan`] wants it. It is a map of
    /// references and not of payloads on purpose: a full-size model's
    /// weights are gigabytes on the host and the search must not hold a
    /// second copy of them.
    #[cfg(feature = "device")]
    Device {
        device: &'a mut crate::device::CudaDevice,
        staged: &'a FxHashMap<i64, &'a crate::host_buffer::HostBuffer>,
        residents: &'a luminal::resident::ResidentBindings,
        profile_inputs: &'a [ProfileInputs<'a>],
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
        best_nanos: Option<u128>,
    ) -> Priced {
        #[cfg(feature = "device")]
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
        #[cfg(not(feature = "device"))]
        {
            let _ = (plan, options, best_nanos);
            Priced::PrepareFailed(
                "candidate search requires the `device` feature and a CUDA GPU".into(),
            )
        }
    }

    /// Lend this evaluator to a nested search (the bucketed entry runs
    /// one search per Cartesian combination and must hand the SAME
    /// device to each).
    pub fn reborrow(&mut self) -> Evaluator<'_> {
        match self {
            #[cfg(feature = "device")]
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
/// genome, and with buckets every combination), and `dyn OpMatcher` is
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
    // finalists at two ranks would waste the lattice's depth on one
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
        let mut candidates: Vec<Genome> = Vec::with_capacity(options.generation_size);
        match &best {
            None => {
                for attempt in 0..options.generation_size {
                    candidates.push(if attempt % 2 == 0 {
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
            // Extraction failure = invalid genome (cycle, contract breach):
            // discard; the next generation's fresh mutations are the repair.
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
            let nanos = match cache.get(&fingerprint) {
                Some(nanos) => {
                    fingerprint_hits += 1;
                    *nanos
                }
                None => {
                    let build_start = Instant::now();
                    // Decode the elected layouts (the runtime's hook; a
                    // refusal rejects THIS genome, loudly accounted, and
                    // the search tries others), then bufferize under the
                    // decoded table.
                    // The table is VALUE-keyed (corrected contract), so it
                    // must be built over the graph bufferize sees — the
                    // POST-DPS one, whose poison destinations are fresh
                    // values. They clone their tied result's layout class
                    // AND dtype fact, so every poison is a decoder-cache
                    // HIT: value-keying costs no extra decoder calls.
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
                            // THE BUFFERIZE TRIPWIRE (2026-09-02): a
                            // cyclic extracted graph from a SAMPLED
                            // genome is a sampler bug, not a refusal.
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
                            // NOT a refusal: nothing failed, the plan is
                            // just too slow to finish measuring. Counted
                            // apart so the zero-refusal ladder
                            // acceptance stays about failures.
                            breakdown.timed_out += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("timed out: {note}"));
                            }
                            continue;
                        }
                        Priced::PrepareFailed(note) => {
                            // D10: a plan the device cannot compile,
                            // stage or warm up is an ordinary unfit
                            // candidate — accounted with the other
                            // plan-build refusals, never fatal.
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
                        // The FIRST profiled plan IS the baseline, so it
                        // reports as `Start`, never as an improvement on
                        // itself; everything after it is Faster/Slower.
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
                if external_placement_feasible(&plan, &program.outputs).is_err() {
                    continue;
                }
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
        // The lattice has not run yet; the runtime stamps this once it
        // has (see [`select_finalist_set`]).
        lattice_rejections: 0,
    })
}

// ===========================================================================
// PHASE 5: FINALISTS AND THE BUCKET LATTICE — how a searched winner becomes
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
    #[cfg(not(feature = "device"))]
    {
        let _ = (pending, evaluator);
        Err("finalist validation requires the `device` feature and a CUDA GPU".into())
    }
}

/// Bound the maximum resident arena across the installed bucket set. Each
/// bucket includes temporaries, boundary device copies, metadata, and HostOp
/// scratch. Their offsets overlay one allocation because launches are serial.
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

/// RUN THE BUCKET LATTICE and return the installed finalist per bucket,
/// plus how many sets were rejected on the way.
///
/// The driver loop is main's (`§2.7`), minus the LLIR compile step this
/// branch does not have: propose the cheapest untried set, check the
/// aggregate constraint over it, install it or reject it and let the
/// lattice open the one-coordinate-slower successors.
///
/// UNBUCKETED CALLERS PASS ONE BUCKET. That is main's "one designed
/// difference" and it is deliberate here too: there is one selection
/// path, and an unbucketed install is a set of one.
pub fn select_finalist_set(
    buckets: Vec<crate::finalists::Finalists<'_>>,
    options: &CompileOptions,
    evaluator: &mut Evaluator<'_>,
) -> Result<(Vec<(usize, crate::finalists::PendingFinalist)>, usize)> {
    let lattice = crate::lattice::BucketLattice::new(buckets, crate::lattice::sum_metrics);
    let mut validate = |pending: &crate::finalists::PendingFinalist| -> Result<(), String> {
        finalist_validate(pending, options, evaluator)
    };
    let mut validate_slabs = |slabs: &[usize]| validate_set(slabs, options);
    let crate::lattice::Installed {
        selected,
        rejections,
        ranks,
    } = lattice
        .drive(&mut validate, &mut validate_slabs)
        .map_err(|message| anyhow!("{message}"))?;
    if rejections > 0 && options.search_log_enabled() {
        // MAIN'S FALLBACK LINE ("aggregate fallback: selected per-bucket
        // finalist ranks …"): the one moment the installed plan is NOT
        // the search's winner is worth saying out loud.
        eprintln!(
            "   {} finalist ranks {:?} after {rejections} rejection(s)",
            "Fallback".yellow().bold(),
            ranks
        );
    }
    Ok((selected, rejections))
}

/// The program a search runs: its text, plus the boundary bindings the
/// tensor-keyed caller data maps through.
#[derive(Debug, Clone)]
pub struct SearchProgram {
    pub text: String,
    pub inputs: Vec<crate::bindings::Bound>,
    pub outputs: Vec<crate::bindings::Bound>,
}

/// One bucket combination's finished search: the dim ranges it covers, the
/// representative pins it was searched at, and the plan the bucket
/// lattice INSTALLED for it.
#[derive(Debug)]
pub struct BucketPlan {
    pub ranges: BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: luminal::shape::DynMap,
    pub program: SearchProgram,
    /// This bucket's own genetic search — its winner, its accounting,
    /// its ranked finalists. It is the SEARCH's report and is left
    /// exactly as the search wrote it.
    pub outcome: SearchOutcome,
    /// THE INSTALLED PLAN (Phase 5): the finalist the bucket lattice
    /// selected. It is `outcome.best_plan` whenever nothing refused the
    /// search's winner — which is every unconstrained search — and a
    /// runner-up when the aggregate device budget refused the faster
    /// set. `execute` loads THIS.
    pub plan: crate::layouts::CudaPlan,
    /// The installed finalist's 1-BASED rank in `outcome.ranked`. 1 =
    /// the search's own winner.
    pub finalist_rank: usize,
    /// The installed plan's arena high-water mark, in bytes — what the
    /// budget was checked against.
    pub slab_bytes: usize,
}

/// The pre-search program parts a bucketed search re-renders from — the
/// runtime's own `load`-time capture. The MODEL TEXT never changes
/// across buckets; only the bounds seeds do, which is the whole point of
/// the bucket model.
pub struct BucketAssembly<'a> {
    /// The runtime's assembled egglog preamble (matchers + registry).
    pub assembled_program: &'a str,
    /// The bound program before the schedule: model text plus boundary.
    pub prefix: &'a str,
    /// The caller's own `bind_*` seeds — for the dims that are NOT
    /// bucketed. Buckets and range bindings refuse each other in BOTH
    /// orders (a range-bound dim is refused buckets, a bucketed dim is
    /// refused a range binding), so these never collide with the
    /// per-bucket seeds appended after them.
    pub binding_seeds: &'a str,
    /// The runtime's schedule text.
    pub schedule: &'a str,
    /// The authoring-contract checks. THEY RUN IN THE BUCKET-WIDE
    /// VALIDATION RENDER TOO: the base logical program must be valid over
    /// the WHOLE interval, not merely at the representative (Austin,
    /// 2026-09-03).
    pub post_checks: &'a str,
    pub inputs: &'a [crate::bindings::Bound],
    pub outputs: &'a [crate::bindings::Bound],
    /// The runtime's placement statement — which boundary buffers the arena
    /// keeps and which are the caller's device memory. Every bucket's
    /// finalists are sized under it, so a bucket's budget and its installed
    /// plan agree.
    pub residents: &'a crate::resident::ResidentBindings,
    /// Values for profiling non-bucket dimensions. These never narrow the
    /// range facts already present in binding_seeds.
    pub base_dims: &'a luminal::shape::DynMap,
    /// The runtime's `(sort, constructor)` decoders — what every
    /// bucket's saturated program is checked against by the assembly
    /// tripwire, and what its layout decodes read through.
    pub decoders: &'a luminal::egglog_utils::eclass::ConstructorRegistry,
}

/// Search one range-seeded e-graph per Cartesian bucket combination. Both
/// authoring checks and implementation selection use the entire interval.
/// Representatives are used only to measure candidates; the installed plans
/// retain symbolic geometry and are capacity-planned over their full bounds.
pub fn bucketed_search_implementations(
    assembly: &BucketAssembly<'_>,
    dim_buckets: &BTreeMap<luminal::shape::Symbol, Vec<luminal::graph::DimBucket>>,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: &[Box<dyn luminal::layout_ir::OpMatcher>],
    mut evaluator: Evaluator<'_>,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");
    // The per-combination e-graphs are kept ALIVE across the whole
    // routine, not dropped at the end of each search: Phase 5's
    // finalists re-extract from them once every bucket has been
    // searched. They are locals rather than fields of [`BucketPlan`] so
    // they go away when this function returns — a serialized e-graph for
    // a real model is large, and nothing after selection reads it.
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
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} range render fails: {err}"))?;
        assembly.decoders.check(&egraph)?;
        let mut serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let outcome = search_implementations(
            &mut serialized,
            &program,
            &bucket_options,
            allow_override.clone(),
            matchers,
            // Every bucket's search prices on the SAME device: the
            // module cache and the staged payloads are shared across
            // combinations, so the evaluator is lent, not rebuilt.
            evaluator.reborrow(),
        )?;
        egraphs.push(serialized);
        searched.push((ranges, representative, program, outcome));
    }

    // PHASE 5: the buckets' ranked genomes become finalists, and the
    // lattice picks one SET of them under the aggregate budget.
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
                .with_resident_bindings(assembly.residents.clone())
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

/// One combination after its genetic search, before the lattice has
/// chosen which of its finalists to install: `(ranges, representative,
/// range-valid render, the search's report)`.
type SearchedBucket = (
    BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    luminal::shape::DynMap,
    SearchProgram,
    SearchOutcome,
);

/// How a bucket names itself in a lattice failure message —
/// `"bucket 0 (a in [2, 4])"`.
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

/// One combination's ranges, profiling dimensions, and range-valid program.
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

    // Cartesian combinations, dims in sorted order.
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

        // BUCKET-WIDE SOUNDNESS: the range-seeded render must run its
        // whole fixpoint over the interval.
        let mut validation_seeds: BTreeMap<luminal::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, (min, max)) in &ranges {
            validation_seeds.insert(*dim, (*min as u64, *max as u64));
        }
        let validation = assemble(&validation_seeds);
        let text = format!("{}\n\n{}", assembly.assembled_program, validation.text);
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} fails bucket-wide validation: {err}"))?;

        // Extract from the range-valid fixpoint; pinning here can select an
        // implementation whose guards do not hold at other bucket dimensions.
        renders.push((ranges, representative, validation));
    }
    Ok(renders)
}

/// The covering bucket plan for a concrete dim assignment, if any.
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

/// The test/example harness's search budget — the SAME genetic algorithm
/// as the module-level ladder tests, sized for a suite of hundreds of
/// graphs. Deterministic (fixed seed); 2 generations x 4 genomes
/// exercises random genomes plus the mutation step without profiling 64
/// candidates per differential.
///
/// Moved out of core `test_support` with the search itself: it is a
/// PRODUCTION-PATH helper (this crate's examples call it), not a test
/// fixture. Duplicated from `luminal_reference::search` per the
/// duplication ruling.
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
