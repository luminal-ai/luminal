//! THE IMPLEMENTATION SEARCH — the reference runtime's selection.
//!
//! There is NO search at the logical level: saturation discovers the
//! implementations, and this module only SELECTS among them, pricing
//! every candidate by EXECUTING its bufferized plan on this runtime.
//!
//! WHAT STAYS HERE, AND WHY. The loop must price a plan in the middle of
//! every iteration, and pricing is what this runtime is — a fresh
//! `ReferenceRuntime`, the plan loaded, the caller's data staged, one
//! warm-up and `trials` timed executes (see
//! [`profile_on_reference_runtime`]). Nothing generic can do that for
//! us, and the 2026-09-03 ruling closed the seam that used to let core
//! call back out ("SearchSpace can live wherever... duplication is fine
//! for now. Don't worry about doing this generic thing" — the
//! `PlanProfiler` trait is GONE). So the loop lives here, and with it
//! this crate's option knobs, outcome shape, allow-list defaults and
//! bucketed driver.
//!
//! WHAT DOES NOT: drawing genomes, counting refusals, attributing
//! wall-clock and printing progress decide nothing, were byte-identical
//! in every copy, and are [`luminal::search_support`] (#420/#422 rejoin
//! Phase 8). The names this module used to define are re-exported below
//! so callers read the same.
//!
//! A mutation-only hill climb over per-value producer genomes — luminal's
//! search shape (no cost models, profile the real thing, keep the best,
//! mutate) over our genome representation. Genomes that fail to extract
//! (cycles, contract violations) are discarded and replaced with fresh
//! random rolls — the repair strategy. Many genomes build the same plan
//! (dead rows are unread), so every built plan is fingerprinted and
//! duplicates reuse the cached measurement instead of burning profile
//! time (the plan-hash dedup ruling, 2026-07-27).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;

use crate::typed_buffer::TypedBuffer;

type InterruptCheck = Box<dyn Fn() -> Result<()>>;

#[derive(Debug)]
pub struct SearchInterrupted(pub anyhow::Error);

impl std::fmt::Display for SearchInterrupted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for SearchInterrupted {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

thread_local! {
    static INTERRUPT_CHECK: RefCell<Option<InterruptCheck>> = RefCell::new(None);
}

/// Run a search with a caller-owned interruption check on the current thread.
/// The reference runtime also checks it between operations while profiling.
pub fn with_interrupt_check<T>(
    check: impl Fn() -> Result<()> + 'static,
    run: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<InterruptCheck>);
    impl Drop for Restore {
        fn drop(&mut self) {
            INTERRUPT_CHECK.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let previous = INTERRUPT_CHECK.with(|slot| slot.borrow_mut().replace(Box::new(check)));
    let _restore = Restore(previous);
    run()
}

pub(crate) fn check_interrupt() -> Result<()> {
    INTERRUPT_CHECK.with(|slot| match slot.borrow().as_ref() {
        Some(check) => check().map_err(|err| SearchInterrupted(err).into()),
        None => Ok(()),
    })
}
use luminal::bufferize::BufferIrGraph;
use luminal::layouts::DecodedLayout;
use luminal::prelude::egraph_serialize;

use crate::extractor::{self, Genome};
use crate::runtime::{ReferenceRuntime, reference_allow_list};

// The pieces that decide nothing, in core since Phase 8. Re-exported
// under this module's own name: every public path this crate used to
// offer still resolves.
pub use luminal::search_support::{
    CaptureAwareStderr, ProducerIndex, RefusalBreakdown, SearchProgress, SearchTimings,
    bufferize_cycle_tripwire, early_stop_exceeded, log_channel_enabled, mutate_genome,
    mutate_genome_reporting, mutate_genome_with_seed, sample_genome, sample_genome_reporting,
    sample_genome_with_seed,
};

#[derive(Debug, Clone)]
pub struct CompileOptions {
    /// Maximum size of one non-boundary buffer, checked before profiling or execution.
    pub max_intermediate_bytes: usize,
    /// Aggregate live tensor payload and kernel scratch ceiling.
    pub memory_budget_bytes: usize,
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
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            max_intermediate_bytes: crate::runtime::DEFAULT_MAX_INTERMEDIATE_BYTES,
            memory_budget_bytes: crate::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
            generations: 8,
            generation_size: 8,
            mutations: 2,
            trials: 3,
            seed: 0,
            search_log: true,
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
}

/// THE REFERENCE EVALUATOR — the search's price for one candidate plan:
/// build a FRESH `ReferenceRuntime`, load the plan, stage the caller's
/// data, run once for warmup and validity, then time `trials` executes.
///
/// THE METRIC IS THE MEAN over trials (ruling 2, 2026-09-02), not the
/// best-of-trials minimum it used to be. A minimum can still fall on a
/// later trial, so truncating a minimum is a heuristic that can promote
/// a candidate whose truncated metric flatters it; a mean only rises as
/// trials accumulate, which is what makes #386's early stop an exact
/// argument rather than a guess. Every reader of `best_nanos` is
/// reading a mean.
///
/// THE INPUT CLONE is deliberate and unavoidable today: each candidate
/// gets its OWN runtime (no state carried between candidates), and
/// `set_data_buffer` takes ownership of the payload. A borrowing stage
/// API would remove it; that is a runtime-surface change, not a Phase 1
/// one.
fn profile_on_reference_runtime(
    plan: &BufferIrGraph<DecodedLayout>,
    input_data: &FxHashMap<i64, TypedBuffer>,
    dims: &luminal::shape::DynMap,
    trials: usize,
    best_so_far: Option<u128>,
    memory_budget_bytes: usize,
) -> Result<u128> {
    let input_bytes = input_data.values().try_fold(0usize, |n, v| {
        n.checked_add(v.byte_len())
            .ok_or_else(|| anyhow::anyhow!("input size overflow"))
    })?;
    ensure!(
        input_bytes <= memory_budget_bytes,
        "reference live memory budget exceeded: inputs require {input_bytes} bytes, budget={memory_budget_bytes}"
    );

    let mut runtime = ReferenceRuntime::default();
    runtime.set_memory_budget_bytes(memory_budget_bytes);
    runtime.load_plan(plan.clone());
    // The plan's spans/extents may be SYMBOLIC (`Var("a")`), so the
    // profiling runtime must hold the representative assignment before it
    // can allocate or index anything. This is the whole point of pricing a
    // symbolic plan: it is a valid plan for the whole bucket, and the
    // representative is only where it gets measured.
    for (symbol, value) in dims {
        runtime.set_dim(*symbol, *value);
    }
    for (id, data) in input_data {
        runtime.set_data_buffer(*id, data.clone());
    }
    runtime.execute()?; // warmup + validity
    let total = trials.max(1);
    let mut sum = 0u128;
    for trial in 0..total {
        let start = Instant::now();
        runtime.execute()?;
        sum += start.elapsed().as_nanos();
        let completed = trial + 1;
        // EARLY STOP (#386). The cutoff is applied to a LOWER BOUND on
        // this candidate's FINAL mean — the trials so far averaged over
        // ALL of them, i.e. assuming every remaining trial costs zero.
        // Once even that bound exceeds the incumbent, no continuation
        // can win and the remaining trials are pure waste. Factor 1.0:
        // main's margin knob (`early_stop_factor`) is a device-runtime
        // tuning parameter; here the exact bound is available, so the
        // stop is taken exactly when the candidate has provably lost.
        // The partial mean returned is >= that bound, so it is still a
        // loss when ranked, exactly as on main.
        if completed < total
            && best_so_far.is_some_and(|best| early_stop_exceeded(sum / total as u128, best, 1.0))
        {
            return Ok(sum / completed as u128);
        }
    }
    Ok(sum / total as u128)
}

/// THE SELECTION LOOP, defaulted to this crate's registry: search the
/// saturated e-graph for the fastest executable plan on the reference
/// runtime, profiling with the given caller data. Deterministic for a
/// fixed seed. `allow_override` narrows the matcher set to a runtime's
/// ALLOWABLE-OPS inventory (M3 Step 2: per-runtime, unstandardized);
/// `None` keeps the reference runtime's own allow list.
pub fn search_implementations_with_ops(
    egraph: &mut egraph_serialize::EGraph,
    program: &SearchProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
    dims: &luminal::shape::DynMap,
    capacity_dims: &luminal::shape::DynMap,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    let pruning = crate::egraph_postpass::run(
        egraph,
        capacity_dims,
        options.max_intermediate_bytes,
        options.memory_budget_bytes,
    )?;
    if options.search_log_enabled() && pruning.oversized_tensors > 0 {
        eprintln!(
            "Reference memory pass: removed {} oversized tensors and {} producer classes ({} nodes)",
            pruning.oversized_tensors, pruning.producer_classes, pruning.removed_nodes
        );
    }
    let matchers = crate::ops::built_in_matchers();
    let allow_override = allow_override.or_else(|| Some(reference_allow_list()));
    // Tensor-keyed at the boundary (the retired-HLIR-keyspace design);
    // buffer-keyed internally via the program's slots.
    let buffer_data: FxHashMap<i64, crate::typed_buffer::TypedBuffer> = input_data
        .iter()
        .map(|(tensor, data)| {
            let bound = program
                .inputs
                .iter()
                .find(|bound| bound.value == *tensor)
                .unwrap_or_else(|| panic!("tensor {tensor:?} is not a bound input"));
            (bound.buffer, data.clone())
        })
        .collect();
    let input_data = &buffer_data;

    let mut timings = SearchTimings::default();
    let analysis_start = Instant::now();
    // The allow list narrows this crate's matcher set; None = the whole set.
    let allow = allow_override;
    let mut session =
        extractor::ExtractionSession::new_with_matcher_set(egraph, allow.as_deref(), &matchers);
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
    // THE VIEW the layout decoders read through: this e-graph plus the
    // reference registry's `(sort, constructor)` decoders. Built once —
    // the view indexes classes through the serialized graph's own
    // `classes()` cache, so every later class lookup is a map hit.
    let view = luminal::egglog_utils::eclass::EGraphView::new(egraph, crate::decoder_registry());
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
    let mut best: Option<(u128, Genome, BufferIrGraph<luminal::layouts::DecodedLayout>)> = None;
    // Live progress (#391), on stderr (via the capture-aware adapter, so
    // test output stays clean) and never on a caller's stdout.
    // `None` = the option (or `SEARCH_LOG`) says quiet.
    let mut progress = options
        .search_log_enabled()
        .then(|| SearchProgress::new(CaptureAwareStderr));

    for generation in 0..options.generations {
        check_interrupt()?;
        let mut candidates: Vec<Genome> = Vec::with_capacity(options.generation_size);
        match &best {
            None => {
                for _ in 0..options.generation_size {
                    candidates.push(random_genome(&mut rng));
                }
            }
            Some((_, parent, _)) => {
                let parent = parent.clone();
                for _ in 0..options.generation_size {
                    candidates.push(mutate(&parent, &mut rng, options.mutations));
                }
            }
        }

        for genome in candidates {
            check_interrupt()?;
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
                    let profile_start = Instant::now();
                    // The incumbent's metric is the early-stop cutoff
                    // (#386). `None` on the first candidate: it IS the
                    // baseline, so there is nothing to have lost to.
                    let best_so_far = best.as_ref().map(|(best_nanos, _, _)| *best_nanos);
                    let profiled = profile_on_reference_runtime(
                        &plan,
                        input_data,
                        dims,
                        options.trials,
                        best_so_far,
                        options.memory_budget_bytes,
                    );
                    timings.profile_nanos += profile_start.elapsed().as_nanos();
                    check_interrupt()?;
                    let nanos = match profiled {
                        Ok(nanos) => nanos,
                        Err(err) if err.is::<SearchInterrupted>() => return Err(err),
                        Err(err) => {
                            breakdown.execute_refusals += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("execute: {err:#}"));
                            }
                            continue;
                        }
                    };
                    cache.insert(fingerprint, nanos);
                    plans_profiled += 1;
                    let improved = best
                        .as_ref()
                        .is_none_or(|(best_nanos, _, _)| nanos < *best_nanos);
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
                        best = Some((nanos, genome.clone(), plan));
                    }
                    continue;
                }
            };
            if best
                .as_ref()
                .is_none_or(|(best_nanos, _, _)| nanos < *best_nanos)
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
                best = Some((nanos, genome.clone(), plan));
            }
        }

        if best.is_none() && generation + 1 == options.generations {
            break;
        }
    }

    if let Some(progress) = progress.as_mut() {
        progress.finish();
    }
    let (best_nanos, best_genome, best_plan) = best.ok_or_else(|| {
        anyhow!(
            "no candidate genome produced an executable plan after memory pruning \
             (max_intermediate_bytes={}, memory_budget_bytes={}, removed {} oversized tensors \
             and {} producer classes, largest pruned tensor {} bytes); refusals: {refusals:#?}",
            options.max_intermediate_bytes,
            options.memory_budget_bytes,
            pruning.oversized_tensors,
            pruning.producer_classes,
            pruning.largest_tensor_bytes,
        )
    })?;
    let _ = program; // binding tables travel with the caller; kept for future bucket plumbing
    Ok(SearchOutcome {
        best_plan,
        best_genome,
        best_nanos,
        plans_profiled,
        fingerprint_hits,
        timings,
        refusal_breakdown: breakdown,
    })
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
/// representative pins it was searched at, and the winning plan.
#[derive(Debug)]
pub struct BucketPlan {
    pub ranges: BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: luminal::shape::DynMap,
    pub program: SearchProgram,
    pub outcome: SearchOutcome,
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
    /// Dim values the runtime already holds, carried into every bucket's
    /// representative map so a plan records the full pin it was searched
    /// at.
    pub base_dims: &'a luminal::shape::DynMap,
}

/// Range-seeded bucketed search: one Cartesian combination of
/// `DimBucket`s per search, each combination run as a bucket-wide
/// RANGE-seeded render whose WHOLE FIXPOINT (authoring checks included)
/// must pass, proving the base logical program valid over the entire
/// interval. The EXTRACTION comes from that range-valid fixpoint (matching
/// CUDA Lite), so the winning plan's spans and extents stay expressions
/// (`Var("a")`) and it executes at every value in the bucket without a
/// re-search — the representative only selects the plan by bucket coverage
/// and prices it during profiling. [`select_bucket`] picks the covering
/// plan at execute time.
pub fn bucketed_search_implementations(
    assembly: &BucketAssembly<'_>,
    dim_buckets: &BTreeMap<luminal::shape::Symbol, Vec<luminal::graph::DimBucket>>,
    input_data: impl Fn(&luminal::shape::DynMap) -> FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");
    let mut plans = Vec::new();
    for (ranges, representative, program) in bucket_renders(assembly, dim_buckets)? {
        check_interrupt()?;
        let text = format!("{}\n\n{}", assembly.assembled_program, program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        check_interrupt()?;
        crate::decoder_registry().check(&egraph)?;
        let mut serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let data = input_data(&representative);
        let mut capacity_dims = representative.clone();
        for (symbol, (min, max)) in &ranges {
            // PyTorch can encode an absent upper bound as i64::MAX-1, or
            // derive another enormous finite upper bound from it. Such
            // extents cannot fit the live arena and can overflow products
            // with other dimensions. Use the minimum for pruning there;
            // concrete allocations remain guarded at execution.
            let capacity = if *max > options.memory_budget_bytes {
                *min
            } else {
                *max
            };
            capacity_dims.insert(*symbol, capacity);
        }
        let outcome = search_implementations_with_ops(
            &mut serialized,
            &program,
            &data,
            &representative,
            &capacity_dims,
            options,
            allow_override.clone(),
        )?;
        plans.push(BucketPlan {
            ranges,
            representative,
            program,
            outcome,
        });
    }
    Ok(plans)
}

/// One bucket combination's `(ranges, representative pins, pinned
/// render)`, in sorted-dim Cartesian order. Each combination's
/// BUCKET-WIDE VALIDATION render runs here, before its pinned render is
/// handed back to be searched: the range-seeded program's whole fixpoint
/// — authoring-contract checks included — must pass, which is what makes
/// "the base logical program is valid over the whole bucket" a checked
/// claim rather than an assumption. Ranges are seeded as intervals and
/// do NOT collapse; only the representative render pins `[n, n]`.
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
        for (dim, value) in &representative {
            validation_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        for (dim, (min, max)) in &ranges {
            validation_seeds.insert(*dim, (*min as u64, *max as u64));
        }
        let validation = assemble(&validation_seeds);
        let text = format!("{}\n\n{}", assembly.assembled_program, validation.text);
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} fails bucket-wide validation: {err}"))?;

        // Extract from the range-valid fixpoint (matching CUDA Lite): the
        // bucket's dimensional seeds stay INTERVALS, so the winning plan's
        // spans and extents remain expressions (`Var("a")`) and one plan
        // serves the whole bucket. Pinning here would collapse them to the
        // representative's literals and reintroduce the static-plan limit.
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
/// graphs (ruling 2026-08-06: everything in the main tree runs the
/// genetic implementation search — there is no plain-walk bypass).
/// Deterministic (fixed seed); 2 generations x 4 genomes exercises
/// random genomes plus the mutation step without profiling 64 candidates
/// per differential.
///
/// Moved out of core `test_support` with the search itself: it is a
/// PRODUCTION-PATH helper (the CL examples call it), not a test fixture.
pub fn harness_search_options() -> CompileOptions {
    CompileOptions {
        max_intermediate_bytes: crate::runtime::DEFAULT_MAX_INTERMEDIATE_BYTES,
        memory_budget_bytes: crate::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
        generations: 2,
        generation_size: 1,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
    }
}

/// Search the saturated e-graph for the fastest executable plan on the
/// reference runtime, profiling with the given caller data.
/// Deterministic for a fixed seed. No dimension assignment: literal-only
/// plans evaluate with an empty map (see
/// [`search_implementations_with_ops`] for the bucketed/symbolic path).
pub fn search_implementations(
    egraph: &mut egraph_serialize::EGraph,
    program: &SearchProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
    options: &CompileOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(
        egraph,
        program,
        input_data,
        &luminal::shape::DynMap::default(),
        &luminal::shape::DynMap::default(),
        options,
        None,
    )
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use luminal::prelude::egglog::SerializeConfig;
    use rustc_hash::FxHashMap;

    use luminal::dtype::DType;
    use luminal::graph::Graph;

    use super::{CompileOptions, SearchProgram, search_implementations};
    use crate::ReferenceRuntime;

    #[test]
    fn interrupt_check_is_scoped_and_restored() {
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        super::with_interrupt_check(
            move || {
                observed.set(observed.get() + 1);
                anyhow::bail!("interrupted")
            },
            || {
                let err = super::check_interrupt().unwrap_err();
                assert!(err.is::<super::SearchInterrupted>());
                assert_eq!(err.to_string(), "interrupted");
            },
        );
        assert_eq!(calls.get(), 1);
        assert!(super::check_interrupt().is_ok());
    }

    #[test]
    fn python_search_preset_prices_one_candidate_per_iteration() {
        assert_eq!(super::harness_search_options().generation_size, 1);
    }

    /// A REAL selection space (x+y and x*y from shared inputs offers the
    /// fused kernel vs the pair, plus commuted and mutating variants): the
    /// search must return a numerically correct plan, and the fingerprint
    /// cache must absorb duplicate plans.
    #[test]
    fn search_returns_a_correct_plan_and_dedups_duplicate_plans() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(4, DType::F32);
            let y = cx.tensor(4, DType::F32);
            let a = x + y;
            let m = x * y;
            (cx, x, y, a, m)
        };
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0];

        // GOLDEN (pinned: x + y and x * y on the fixed data).
        let their_a = vec![11.0, 22.0, 33.0, 44.0];
        let their_m = vec![10.0, 40.0, 90.0, 160.0];

        // Our search.
        let (cx2, x2, y2, a2, m2) = build();
        let bound = crate::ReferenceBindings::leaves(&cx2.logical)
            .bind(&cx2.logical)
            .expect("native program");
        let program = SearchProgram {
            text: bound.text(),
            inputs: bound.inputs.clone(),
            outputs: bound.outputs.clone(),
        };
        let text = format!("{}\n\n{}", crate::assembled_program(), program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let mut serialized = egraph.serialize(SerializeConfig::default()).egraph;

        let mut inputs = FxHashMap::default();
        inputs.insert(x2.id, x_data.clone().into());
        inputs.insert(y2.id, y_data.clone().into());
        let outcome = search_implementations(
            &mut serialized,
            &program,
            &inputs,
            &CompileOptions::default(),
        )
        .expect("search finds an executable plan");

        assert!(
            outcome.fingerprint_hits > 0,
            "small space, many genomes: the plan cache must fire \
             (profiled {}, hits {})",
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );

        let mut runtime = ReferenceRuntime::default();
        runtime.stage_bindings(&bound.inputs, &bound.outputs);
        runtime.load_plan(outcome.best_plan.clone());
        runtime.set_data(x2.id, x_data);
        runtime.set_data(y2.id, y_data);
        runtime.execute().expect("best plan executes");
        let ours_a = runtime.get_f32(a2.id).unwrap();
        let ours_m = runtime.get_f32(m2.id).unwrap();
        for (ours, theirs) in [(ours_a, &their_a), (ours_m, &their_m)] {
            assert_eq!(ours.len(), theirs.len());
            for (index, (lhs, rhs)) in ours.iter().zip(theirs).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "element {index}: ours {lhs} vs theirs {rhs}"
                );
            }
        }
    }
}
