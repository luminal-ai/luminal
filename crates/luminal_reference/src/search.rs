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

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;

use crate::typed_buffer::TypedBuffer;
use luminal::bufferize::BufferIrGraph;
use luminal::graph::LogicalProgram;
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
    trials: usize,
    best_so_far: Option<u128>,
) -> Result<u128> {
    let mut runtime = ReferenceRuntime::default();
    runtime.load_plan(plan.clone());
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
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    let matchers = crate::ops::built_in_matchers();
    let allow_override = allow_override.or_else(|| Some(reference_allow_list()));
    // Tensor-keyed at the boundary (the retired-HLIR-keyspace design);
    // buffer-keyed internally via the program's slots.
    let buffer_data: FxHashMap<i64, crate::typed_buffer::TypedBuffer> = input_data
        .iter()
        .map(|(tensor, data)| {
            let slot = program
                .input_slots
                .iter()
                .find(|slot| slot.tensor == *tensor)
                .unwrap_or_else(|| panic!("tensor {tensor:?} is not a bound input"));
            (slot.buffer, data.clone())
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
                        options.trials,
                        best_so_far,
                    );
                    timings.profile_nanos += profile_start.elapsed().as_nanos();
                    let nanos = match profiled {
                        Ok(nanos) => nanos,
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
        anyhow!("no candidate genome produced an executable plan; refusals: {refusals:#?}")
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

/// One bucket combination's finished search: the dim ranges it covers, the
/// representative pins it was searched at, and the winning plan.
#[derive(Debug)]
pub struct BucketPlan {
    pub ranges: BTreeMap<luminal::shape::Symbol, (usize, usize)>,
    pub representative: luminal::shape::DynMap,
    pub program: LogicalProgram,
    pub outcome: SearchOutcome,
}

/// The pre-search program parts a bucketed search re-renders from — the
/// runtime's own `load`-time capture. The MODEL TEXT never changes
/// across buckets; only the bounds seeds do, which is the whole point of
/// the bucket model.
pub struct BucketAssembly<'a> {
    /// The runtime's assembled egglog preamble (matchers + registry).
    pub assembled_program: &'a str,
    /// The recorded model, before the schedule.
    pub pre_schedule: &'a str,
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
    pub input_slots: &'a [luminal::graph::InputSlot],
    pub output_slots: &'a [luminal::graph::OutputSlot],
    /// Dim values the runtime already holds, carried into every bucket's
    /// representative map so a plan records the full pin it was searched
    /// at.
    pub base_dims: &'a luminal::shape::DynMap,
}

/// Range-seeded bucketed search: one Cartesian combination of
/// `DimBucket`s per search, each combination run TWICE — a bucket-wide
/// RANGE-seeded render whose WHOLE FIXPOINT (authoring checks included)
/// must pass, proving the base logical program valid over the entire
/// interval, then a representative-pinned render that is searched.
/// [`select_bucket`] picks the covering plan at execute time.
///
/// THE LIMITATION, stated rather than solved (Phase 1 scope): each
/// winning plan is STATIC at its representative — plans carry LITERAL
/// spans, so a plan searched at `a = 3` allocates and indexes for `a =
/// 3` and nothing else. Executing a bucket's plan at any OTHER value
/// inside that bucket is REFUSED loudly by the runtime, naming the
/// representative; it is never silently run. Lifting this needs symbolic
/// plans (spans as expressions) and the capacity contract that goes with
/// them — the open item this note points at.
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
        let text = format!("{}\n\n{}", assembly.assembled_program, program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        crate::decoder_registry().check(&egraph)?;
        let serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let data = input_data(&representative);
        let outcome = search_implementations_with_ops(
            &serialized,
            &program,
            &data,
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
    LogicalProgram,
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
    let assemble = |seeds: &BTreeMap<luminal::shape::Symbol, (u64, u64)>| LogicalProgram {
        text: format!(
            "{}{}{}{}{}",
            assembly.pre_schedule,
            assembly.binding_seeds,
            seeds_text(seeds),
            assembly.schedule,
            assembly.post_checks
        ),
        input_slots: assembly.input_slots.to_vec(),
        output_slots: assembly.output_slots.to_vec(),
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

        // Representative render: pinned via tight bounds.
        let mut pin_seeds: BTreeMap<luminal::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            pin_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        renders.push((ranges, representative, assemble(&pin_seeds)));
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
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
    }
}

/// Search the saturated e-graph for the fastest executable plan on the
/// reference runtime, profiling with the given caller data.
/// Deterministic for a fixed seed.
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, TypedBuffer>,
    options: &CompileOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(egraph, program, input_data, options, None)
}

#[cfg(test)]
mod tests {
    use luminal::prelude::egglog::SerializeConfig;
    use rustc_hash::FxHashMap;

    use luminal::dtype::DType;
    use luminal::graph::Graph;

    use super::{CompileOptions, search_implementations};
    use crate::ReferenceRuntime;

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
            let a = (x + y).output();
            let m = (x * y).output();
            (cx, x, y, a, m)
        };
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![10.0, 20.0, 30.0, 40.0];

        // GOLDEN (pinned: x + y and x * y on the fixed data).
        let their_a = vec![11.0, 22.0, 33.0, 44.0];
        let their_m = vec![10.0, 40.0, 90.0, 160.0];

        // Our search.
        let (cx2, x2, y2, a2, m2) = build();
        let program = cx2
            .logical
            .bound_program(&crate::ReferenceBindings)
            .expect("native program");
        let text = format!("{}\n\n{}", crate::assembled_program(), program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;

        let mut inputs = FxHashMap::default();
        inputs.insert(x2.id, x_data.clone().into());
        inputs.insert(y2.id, y_data.clone().into());
        let outcome =
            search_implementations(&serialized, &program, &inputs, &CompileOptions::default())
                .expect("search finds an executable plan");

        assert!(
            outcome.fingerprint_hits > 0,
            "small space, many genomes: the plan cache must fire \
             (profiled {}, hits {})",
            outcome.plans_profiled,
            outcome.fingerprint_hits
        );

        let mut runtime = ReferenceRuntime::default();
        runtime.stage_slots(&program.input_slots, &program.output_slots);
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
