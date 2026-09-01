//! IMPLEMENTATION search (renamed from logical_search, ruling 2026-07-30):
//! there is NO search at the logical level — saturation discovers
//! implementations; this module only SELECTS among them, pricing every
//! candidate by executing its bufferized plan on the runtime.
//!
//! A mutation-only hill climb over
//! per-value producer genomes, profiled on the real `ReferenceRuntime` —
//! luminal's search shape (no cost models, profile the real thing, keep the
//! best, mutate) over our genome representation.
//!
//! Genomes that fail to extract (cycles, contract violations) are discarded
//! and replaced with fresh random rolls — the repair strategy. Many genomes
//! build the same plan (dead rows are unread), so every built plan is
//! fingerprinted and duplicates reuse the cached measurement instead of
//! burning profile time (the plan-hash dedup ruling, 2026-07-27).

use std::time::Instant;

use anyhow::{Result, anyhow, ensure};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashMap;
use std::collections::BTreeMap;

use crate::bufferize::BufferIrGraph;
use crate::extractor::{self, Genome};
use crate::graph::LogicalProgram;
use crate::reference::ReferenceRuntime;

#[derive(Debug, Clone)]
pub struct ImplementationSearchOptions {
    pub generations: usize,
    pub generation_size: usize,
    /// Point mutations per offspring. Mutations hit ANY producer class —
    /// dead rows included, deliberately: a dead-row mutation is free now and
    /// pre-stages the choice a later route flip lands on.
    pub mutations: usize,
    pub trials: usize,
    pub seed: u64,
}

impl Default for ImplementationSearchOptions {
    fn default() -> Self {
        Self {
            generations: 8,
            generation_size: 8,
            mutations: 2,
            trials: 3,
            seed: 0,
        }
    }
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub best_plan: BufferIrGraph,
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

/// Aggregated classification of rejected genomes across one search.
#[derive(Debug, Clone, Default)]
pub struct RefusalBreakdown {
    /// Genomes whose extraction produced no plan for some output.
    pub extract_refusals: usize,
    /// ...of those, how many involved a CHOICE-CYCLE (the genome's
    /// chosen producers block on each other).
    pub with_choice_cycles: usize,
    /// ...and how many involved a DEAD-END (an unplanned class with no
    /// candidate at all).
    pub with_dead_ends: usize,
    /// Genomes that extracted but failed bufferize / execute.
    pub plan_build_refusals: usize,
    pub execute_refusals: usize,
    /// First few classified summaries, verbatim.
    pub exemplars: Vec<String>,
}

impl RefusalBreakdown {
    pub fn summary(&self) -> String {
        format!(
            "extract refusals {} (choice-cycles {}, dead-ends {}), bufferize {}, execute {}",
            self.extract_refusals,
            self.with_choice_cycles,
            self.with_dead_ends,
            self.plan_build_refusals,
            self.execute_refusals
        )
    }
}

/// Stage wall-clock totals for one search, in nanoseconds. Saturation
/// and serialization happen in the runtime's `search` wrapper and are
/// stamped there; the rest accumulate inside the selection loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchTimings {
    /// egglog parse + saturation to fixpoint (one per search).
    pub saturation_nanos: u128,
    /// e-graph serialization (one per search).
    pub serialize_nanos: u128,
    /// ExtractionSession::new + producer index + viability fixpoint.
    pub analysis_nanos: u128,
    /// All genome extractions (extract_with_genome, cumulative).
    pub extract_nanos: u128,
    /// All DPS rewrites + bufferizations (cumulative).
    pub plan_build_nanos: u128,
    /// All candidate executions: warmup + timed trials (cumulative) —
    /// the part that shrinks with a faster runtime.
    pub profile_nanos: u128,
}

impl SearchTimings {
    /// A compact human-readable ms breakdown for test logs.
    pub fn summary(&self) -> String {
        let ms = |n: u128| n as f64 / 1e6;
        format!(
            "saturation {:.0}ms, serialize {:.0}ms, analysis {:.0}ms, extract {:.0}ms, \
             plan-build {:.0}ms, profile-exec {:.0}ms",
            ms(self.saturation_nanos),
            ms(self.serialize_nanos),
            ms(self.analysis_nanos),
            ms(self.extract_nanos),
            ms(self.plan_build_nanos),
            ms(self.profile_nanos)
        )
    }
}

/// Search the saturated e-graph for the fastest executable plan, profiling
/// with the given caller data. Deterministic for a fixed seed.
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
) -> Result<SearchOutcome> {
    search_implementations_with_ops(egraph, program, input_data, options, None)
}

/// As [`search_implementations`], with the runtime's ALLOWABLE-OPS
/// inventory made explicit (M3 Step 2: per-runtime, unstandardized).
/// How search prices a candidate plan. Runtimes supply their own
/// (ruling 2026-08-17: every runtime owns its execution — including
/// how its candidates are timed); the in-core implementations are the
/// reference host executor (the historical behavior) and a static
/// ranker that never executes.
pub trait PlanProfiler {
    /// Best-of-`trials` cost for the plan with buffer-keyed inputs;
    /// smaller wins. `heuristic_cost` is the extracted graph's summed
    /// bytes-moved estimate, for profilers that rank without running.
    fn profile(
        &mut self,
        plan: &crate::bufferize::BufferIrGraph,
        input_data: &FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer>,
        trials: usize,
        heuristic_cost: u64,
    ) -> Result<u128>;
}

/// The historical profiler: execute on the reference host runtime.
#[derive(Default)]
pub struct ReferenceProfiler;

impl PlanProfiler for ReferenceProfiler {
    fn profile(
        &mut self,
        plan: &crate::bufferize::BufferIrGraph,
        input_data: &FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer>,
        trials: usize,
        _heuristic_cost: u64,
    ) -> Result<u128> {
        let mut runtime = ReferenceRuntime::default();
        runtime.load_plan(plan.clone());
        for (id, data) in input_data {
            runtime.set_data_buffer(*id, data.clone());
        }
        runtime.execute()?; // warmup + validity
        let mut best_nanos = u128::MAX;
        for _ in 0..trials.max(1) {
            let start = Instant::now();
            runtime.execute()?;
            best_nanos = best_nanos.min(start.elapsed().as_nanos());
        }
        Ok(best_nanos)
    }
}

/// Rank by the heuristic byte-move estimate without executing —
/// deterministic, device-free search for runtimes that cannot (or
/// need not) run candidates on the searching host.
#[derive(Default)]
pub struct StaticProfiler;

impl PlanProfiler for StaticProfiler {
    fn profile(
        &mut self,
        _plan: &crate::bufferize::BufferIrGraph,
        _input_data: &FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer>,
        _trials: usize,
        heuristic_cost: u64,
    ) -> Result<u128> {
        Ok(u128::from(heuristic_cost).saturating_add(1))
    }
}

pub fn search_implementations_with_ops(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
) -> Result<SearchOutcome> {
    search_implementations_with_runtime(
        egraph,
        program,
        input_data,
        options,
        allow_override,
        None,
        &mut ReferenceProfiler,
    )
}

/// The runtime-owned search entry (ruling 2026-08-17): the caller
/// supplies its OWN matcher set (None = the in-core reference
/// registry, which Step B moves out) and its OWN profiler.
pub fn search_implementations_with_runtime(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: Option<Vec<Box<dyn crate::layout_ir::OpMatcher>>>,
    profiler: &mut dyn PlanProfiler,
) -> Result<SearchOutcome> {
    // Tensor-keyed at the boundary (the retired-HLIR-keyspace design);
    // buffer-keyed internally via the program's slots.
    let buffer_data: FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer> = input_data
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
    let allow = allow_override.unwrap_or_else(crate::reference::reference_allow_list);
    let mut timings = SearchTimings::default();
    let analysis_start = Instant::now();
    let (mut session, index) = match matchers {
        Some(matchers) => {
            let session =
                extractor::ExtractionSession::new_with_matcher_set(egraph, Some(&allow), matchers);
            let index = session.producer_index();
            (session, index)
        }
        None => (
            extractor::ExtractionSession::new(egraph, Some(&allow)),
            extractor::producer_index_with_ops(egraph, Some(&allow)),
        ),
    };
    timings.analysis_nanos = analysis_start.elapsed().as_nanos();
    // An empty index is NOT an error: a graph with no searchable producer
    // classes (pure identity — every output is an input value) has a
    // one-point genome space, the empty genome. The search still profiles
    // that single candidate; the fingerprint cache collapses the rest.
    let classes: Vec<_> = index.keys().cloned().collect();
    let mut rng = StdRng::seed_from_u64(options.seed);

    // TWO-PHASE SAMPLING (ruling 2026-08-07): the genome searches
    // PROGRESSING candidates — ops whose inputs are other logical
    // values — freely; sibling-sourced candidates (layout copies within
    // one logical value) are subordinate plumbing. Generation 0 elects
    // one progressing PRIMARY per sibling group; the other members
    // choose uniformly between their progressing candidates and copies
    // sourced from the primary. Mutation preserves the same invariant
    // locally: a sibling-sourced choice needs its source to currently
    // progress, and a member some sibling currently copies must keep
    // progressing. Copy⟷Copy welds and copy chains are therefore
    // unconstructible, while recompute-at-every-layout and
    // copy-from-boundary-input plans stay in the space. A class whose
    // allowed set comes up empty falls back to its full candidate list
    // (same space as the old free sampler); the refusal accounting
    // stays the loud diagnosis for that corner.
    let space = extractor::sampling_space(egraph, &index);
    let group_of: FxHashMap<_, usize> = space
        .groups
        .iter()
        .enumerate()
        .flat_map(|(group, members)| members.iter().map(move |member| (member.clone(), group)))
        .collect();

    let random_genome = |rng: &mut StdRng| {
        let mut genome = Genome::default();
        let mut primary_of: FxHashMap<_, _> = FxHashMap::default();
        for members in &space.groups {
            let eligible: Vec<_> = members
                .iter()
                .filter(|member| {
                    space.sibling_sources[*member]
                        .iter()
                        .any(|sources| sources.is_empty())
                })
                .cloned()
                .collect();
            if eligible.is_empty() {
                continue; // copy-only group: full-list fallback below
            }
            let primary = eligible[rng.random_range(0..eligible.len())].clone();
            for member in members {
                primary_of.insert(member.clone(), primary.clone());
            }
        }
        for (class, candidates) in &index {
            let sources_per_candidate = &space.sibling_sources[class];
            let allowed: Vec<usize> = match primary_of.get(class) {
                Some(primary) if class == primary => (0..candidates.len())
                    .filter(|&position| sources_per_candidate[position].is_empty())
                    .collect(),
                Some(primary) => (0..candidates.len())
                    .filter(|&position| {
                        sources_per_candidate[position]
                            .iter()
                            .all(|source| source == primary)
                    })
                    .collect(),
                None => (0..candidates.len()).collect(),
            };
            let position = if allowed.is_empty() {
                rng.random_range(0..candidates.len())
            } else {
                allowed[rng.random_range(0..allowed.len())]
            };
            genome
                .choices
                .insert(class.clone(), candidates[position].1.clone());
        }
        genome
    };
    // Whether a class's CURRENT choice progresses (boundary classes
    // without genome rows are leaves — they trivially progress).
    let progresses = |child: &Genome, class: &egraph_serialize::ClassId| -> bool {
        let (Some(choice), Some(candidates)) = (child.choices.get(class), index.get(class)) else {
            return true;
        };
        candidates
            .iter()
            .position(|(_, candidate)| candidate == choice)
            .is_none_or(|position| space.sibling_sources[class][position].is_empty())
    };
    let mutate = |parent: &Genome, rng: &mut StdRng, count: usize| {
        let mut child = parent.clone();
        if classes.is_empty() {
            return child; // one-point genome space: nothing to mutate
        }
        for _ in 0..count {
            let class = &classes[rng.random_range(0..classes.len())];
            let candidates = &index[class];
            let sources_per_candidate = &space.sibling_sources[class];
            let copied_by_sibling = group_of.get(class).is_some_and(|group| {
                space.groups[*group].iter().any(|member| {
                    member != class
                        && child.choices.get(member).is_some_and(|choice| {
                            index[member]
                                .iter()
                                .position(|(_, candidate)| candidate == choice)
                                .is_some_and(|position| {
                                    space.sibling_sources[member][position]
                                        .iter()
                                        .any(|source| source == class)
                                })
                        })
                })
            });
            let allowed: Vec<usize> = (0..candidates.len())
                .filter(|&position| {
                    let sources = &sources_per_candidate[position];
                    sources.is_empty()
                        || (!copied_by_sibling
                            && sources
                                .iter()
                                .all(|source| source != class && progresses(&child, source)))
                })
                .collect();
            let position = if allowed.is_empty() {
                rng.random_range(0..candidates.len())
            } else {
                allowed[rng.random_range(0..allowed.len())]
            };
            child
                .choices
                .insert(class.clone(), candidates[position].1.clone());
        }
        child
    };

    // fingerprint → measured nanos (the dedup cache).
    let mut cache: FxHashMap<u64, u128> = FxHashMap::default();
    let mut plans_profiled = 0usize;
    let mut fingerprint_hits = 0usize;
    // Refusal accounting, minimal form (Step 5 down-payment): keep the
    // first few refusal reasons so a fully-refused search names its
    // causes instead of shrugging.
    let mut refusals: Vec<String> = Vec::new();
    let mut breakdown = RefusalBreakdown::default();
    let mut best: Option<(u128, Genome, BufferIrGraph)> = None;

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
                    let built = crate::bufferize::bufferize(&crate::dps::dps_rewrite(&graph));
                    timings.plan_build_nanos += build_start.elapsed().as_nanos();
                    let plan = match built {
                        Ok(plan) => plan,
                        Err(err) => {
                            breakdown.plan_build_refusals += 1;
                            if refusals.len() < 8 {
                                refusals.push(format!("bufferize: {err:#}"));
                            }
                            continue;
                        }
                    };
                    let heuristic_total: u64 = graph
                        .dag
                        .node_weights()
                        .map(|node| match node {
                            crate::layout_ir::ExtractedNode::LayoutOp(op) => op.heuristic_cost,
                            _ => 0,
                        })
                        .sum();
                    let profile_start = Instant::now();
                    let profiled =
                        profiler.profile(&plan, input_data, options.trials, heuristic_total);
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
                    if best
                        .as_ref()
                        .is_none_or(|(best_nanos, _, _)| nanos < *best_nanos)
                    {
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
                let built = crate::bufferize::bufferize(&crate::dps::dps_rewrite(&graph));
                timings.plan_build_nanos += build_start.elapsed().as_nanos();
                let Ok(plan) = built else {
                    continue;
                };
                best = Some((nanos, genome.clone(), plan));
            }
        }

        if best.is_none() && generation + 1 == options.generations {
            break;
        }
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
    pub ranges: BTreeMap<crate::shape::Symbol, (usize, usize)>,
    pub representative: crate::shape::DynMap,
    pub program: LogicalProgram,
    pub outcome: SearchOutcome,
}

/// Range-seeded bucketed search, mirroring their per-bucket model: one
/// Cartesian combination of `DimBucket`s per search, each combination run
/// TWICE — a bucket-wide RANGE-seeded render whose fixpoint checks prove
/// the model sound over the whole interval (validation only; ranges do not
/// collapse), then a representative-pinned render that is searched and
/// profiled. `select_bucket` picks the covering plan at runtime.
///
/// Slice note (documented divergence from their symbolic LLIR): each
/// winning plan is STATIC at its representative; executing at another pin
/// re-renders — genome transfer across renders is future work.
pub fn bucketed_search_implementations(
    graph: &crate::graph::Graph,
    dim_buckets: &BTreeMap<crate::shape::Symbol, Vec<crate::graph::DimBucket>>,
    input_data: impl Fn(
        &crate::shape::DynMap,
    )
        -> FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");

    // M3 Topic C: buckets assemble NATIVELY — one recorder model, per-
    // bucket binding seeds (ranges for the bucket-wide validation render,
    // tight [n,n] pins for the representative render). The model text
    // never changes across buckets; only the binding does.
    let (pre, input_slots, output_slots, post, _labeled) = graph
        .logical
        .native_parts()
        .map_err(|reason| anyhow!("native load refused: {reason}"))?;
    let seeds_text = |seeds: &BTreeMap<crate::shape::Symbol, (u64, u64)>| {
        let mut text = String::new();
        for (var, (lower, upper)) in seeds {
            text.push_str(&format!(
                "(set (lower-bound-of (IntVar \"{var}\")) (bigint {lower}))\n\
                 (set (upper-bound-of (IntVar \"{var}\")) (bigint {upper}))\n"
            ));
        }
        text
    };
    let assemble =
        |seeds: &BTreeMap<crate::shape::Symbol, (u64, u64)>| crate::graph::LogicalProgram {
            text: format!(
                "{pre}{}{}{post}",
                seeds_text(seeds),
                crate::reference_binding::SCHEDULE
            ),
            input_slots: input_slots.clone(),
            output_slots: output_slots.clone(),
        };

    // Cartesian combinations, dims in sorted order (their bucket_combinations).
    let dims: Vec<&crate::shape::Symbol> = dim_buckets.keys().collect();
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

    let mut plans = Vec::new();
    for combo in combos {
        let mut ranges = BTreeMap::new();
        let mut representative = graph.dyn_map.clone();
        for (dim, bucket_index) in dims.iter().zip(&combo) {
            let bucket = &dim_buckets[*dim][*bucket_index];
            ranges.insert(**dim, (bucket.min, bucket.max));
            representative.insert(**dim, bucket.representative_value());
        }

        // Bucket-wide soundness: the range-seeded render must run its whole
        // fixpoint (authoring-contract checks included) over the interval.
        let mut validation_seeds: BTreeMap<crate::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            validation_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        for (dim, (min, max)) in &ranges {
            validation_seeds.insert(*dim, (*min as u64, *max as u64));
        }
        let validation = assemble(&validation_seeds);
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            validation.text
        );
        crate::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} fails bucket-wide validation: {err}"))?;

        // Representative render: pinned via tight bounds, searched, profiled.
        let mut pin_seeds: BTreeMap<crate::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            pin_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        let program = assemble(&pin_seeds);
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let data = input_data(&representative);
        let outcome = search_implementations(&serialized, &program, &data, options)?;
        plans.push(BucketPlan {
            ranges,
            representative,
            program,
            outcome,
        });
    }
    Ok(plans)
}

/// The covering bucket plan for a concrete dim assignment, if any.
pub fn select_bucket<'a>(
    plans: &'a [BucketPlan],
    dims: &crate::shape::DynMap,
) -> Option<&'a BucketPlan> {
    plans.iter().find(|plan| {
        plan.ranges.iter().all(|(dim, (min, max))| {
            dims.get(dim)
                .is_some_and(|value| value >= min && value <= max)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use egglog::SerializeConfig;

    /// A REAL selection space (x+y and x*y from shared inputs offers the
    /// fused kernel vs the pair, plus commuted and mutating variants): the
    /// search must return a numerically correct plan, and the fingerprint
    /// cache must absorb duplicate plans.
    #[test]
    fn search_returns_a_correct_plan_and_dedups_duplicate_plans() {
        let build = || {
            let mut cx = Graph::new();
            let x = cx.tensor(4);
            let y = cx.tensor(4);
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
        let program = cx2.logical.native_program().expect("native program");
        let text = format!(
            "{}\n\n{}",
            crate::egglog_snippet::assembled_program(),
            program.text
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;

        let mut inputs = FxHashMap::default();
        inputs.insert(x2.id, x_data.clone().into());
        inputs.insert(y2.id, y_data.clone().into());
        let outcome = search_implementations(
            &serialized,
            &program,
            &inputs,
            &ImplementationSearchOptions::default(),
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

    /// BUCKETED: two buckets over 'a', each validated bucket-wide
    /// (range seeds) and searched at its representative; selection covers
    /// runtime dims; each bucket's plan agrees with ReferenceRuntime at its
    /// representative.
    #[test]
    fn bucketed_search_validates_searches_and_selects() {
        use crate::graph::DimBucket;

        let build = |dim: usize| {
            let mut cx = Graph::new();
            cx.set_dim('a', dim);
            let x = cx.tensor(('a', 2));
            let y = cx.tensor(('a', 2));
            let out = (x * y).output();
            (cx, x, y, out)
        };

        let buckets: BTreeMap<crate::shape::Symbol, Vec<DimBucket>> = [(
            crate::shape::Symbol::from('a'),
            vec![DimBucket::new(2, 4), DimBucket::new(5, 9)],
        )]
        .into();

        // The graph used for SEARCH: built at any pin (per-bucket renders
        // re-pin), sharing the same HLIR shape.
        let (cx, x, y, _out) = build(3);
        let data_for = |rep: &crate::shape::DynMap| {
            let n = rep[&crate::shape::Symbol::from('a')] * 2;
            let mut data = FxHashMap::default();
            data.insert(
                x.id,
                (0..n).map(|v| v as f32 + 1.0).collect::<Vec<f32>>().into(),
            );
            data.insert(
                y.id,
                (0..n).map(|v| v as f32 * 0.5).collect::<Vec<f32>>().into(),
            );
            data
        };
        let plans = bucketed_search_implementations(
            &cx,
            &buckets,
            data_for,
            &ImplementationSearchOptions::default(),
        )
        .expect("bucketed search completes");
        assert_eq!(plans.len(), 2, "one plan per bucket");

        // Selection covers each bucket; out-of-range dims select nothing.
        let mut dims = FxHashMap::default();
        dims.insert(crate::shape::Symbol::from('a'), 3usize);
        assert!(
            select_bucket(&plans, &dims).unwrap().ranges[&crate::shape::Symbol::from('a')]
                == (2, 4)
        );
        dims.insert(crate::shape::Symbol::from('a'), 7usize);
        assert!(
            select_bucket(&plans, &dims).unwrap().ranges[&crate::shape::Symbol::from('a')]
                == (5, 9)
        );
        dims.insert(crate::shape::Symbol::from('a'), 20usize);
        assert!(select_bucket(&plans, &dims).is_none());

        // Numeric agreement at each bucket's representative.
        for plan in &plans {
            let rep = plan.representative[&crate::shape::Symbol::from('a')];
            // GOLDEN (computed: out = x * y with x[i] = i+1, y[i] = i*0.5
            // — the data_for closure's values at this representative).
            let n = rep * 2;
            let expected: Vec<f32> = (0..n)
                .map(|v| (v as f32 + 1.0) * (v as f32 * 0.5))
                .collect();
            let data = data_for(&plan.representative);

            let mut runtime = ReferenceRuntime::default();
            runtime.stage_slots(&plan.program.input_slots, &plan.program.output_slots);
            runtime.load_plan(plan.outcome.best_plan.clone());
            for (id, values) in &data {
                runtime.set_data(*id, values.clone());
            }
            runtime
                .execute()
                .expect("bucket plan executes at representative");
            let ours = runtime
                .get_f32(plan.program.output_slots[0].tensor)
                .unwrap();
            assert_eq!(ours.len(), expected.len());
            for (index, (lhs, rhs)) in ours.iter().zip(&expected).enumerate() {
                assert!(
                    (lhs - rhs).abs() <= 1e-5 * rhs.abs().max(1.0),
                    "bucket {:?} element {index}: ours {lhs} vs theirs {rhs}",
                    plan.ranges
                );
            }
        }
    }
}
