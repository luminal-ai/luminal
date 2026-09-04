//! THE IMPLEMENTATION SEARCH — the CUDA-lite runtime's copy.
//!
//! There is NO search at the logical level: saturation discovers the
//! implementations, and this module only SELECTS among them.
//!
//! RUNTIME-OWNED (ruling 2026-09-03, #420/#422 rejoin Phase 1). This was
//! `luminal::implementation_search`, generic over a `PlanProfiler` trait
//! core defined and two runtimes implemented. Both the loop and the
//! extractor it drives now live in each runtime, duplicated rather than
//! generalized ("just put this search in the cuda lite runtime\'s crate.
//! it\'s fine."), and the profiler trait is GONE: this copy ranks
//! candidates inline with the DEVICE-FREE HEURISTIC in
//! [`crate::heuristic`] — a weak static prior, never a measurement.
//!
//! Tests live with the reference copy (`luminal_reference::search`).

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{anyhow, ensure, Result};
use colored::Colorize;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::extractor::{self, Genome, ProducerChoice, SamplingSpace};
use luminal::bufferize::BufferIrGraph;
use luminal::graph::LogicalProgram;
use luminal::prelude::egraph_serialize::{self, ClassId};
use luminal::prelude::{FxHashMap, FxHashSet};

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

fn parse_log_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Main's `log_channel_enabled` (its `src/egglog_utils/mod.rs`), which
/// this branch has no counterpart for: `LUMINAL_LOG=1` forces every
/// channel on, an explicit channel variable overrides the programmatic
/// setting, and otherwise the option stands.
pub fn log_channel_enabled(option_enabled: bool, channel_env: &str) -> bool {
    if std::env::var("LUMINAL_LOG").is_ok_and(|value| parse_log_flag(&value)) {
        return true;
    }
    if let Ok(value) = std::env::var(channel_env) {
        return parse_log_flag(&value);
    }
    option_enabled
}

/// A profiled candidate's cost, as the progress lines spell it.
fn display_nanos(nanos: u128) -> String {
    format!("{:.3} ms", nanos as f64 / 1e6)
}

/// The production sink for [`SearchProgress`].
///
/// Writing to `std::io::stderr()` directly bypasses libtest's output
/// capture, so every test that leaves `search_log` on would leak
/// progress lines (including unterminated transient `Slower` rows)
/// into the harness output. Routing the same bytes through the
/// `eprint!` macro goes through the capture-aware path instead, so the
/// suites stay silent unless run with `--nocapture` while real runs
/// still print to stderr.
pub(crate) struct CaptureAwareStderr;

impl std::io::Write for CaptureAwareStderr {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        eprint!("{}", String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        // `eprint!` already writes through on each call; nothing buffered here.
        Ok(())
    }
}

/// Live search progress, re-expressing main's #391 three-state report
/// (`Start` once, a permanent `Faster` per improvement, one transient
/// `Slower x{n}`) for this branch's selection loop.
///
/// THE ONE DIVERGENCE from main: main draws progress bars under the
/// report and walks the cursor up over them (`\x1b[1A` per bar row)
/// before printing. This branch's search draws no bars, so all of that
/// cursor arithmetic is dropped; the transient `Slower` line is instead
/// written WITHOUT a newline and every subsequent line starts by
/// clearing it in place (`\r\x1b[2K`). A `Faster` line therefore
/// replaces the pending `Slower` line rather than being appended below
/// it, and ends with a newline, so improvements accumulate as
/// scrollback exactly as on main.
pub(crate) struct SearchProgress<W: std::io::Write> {
    out: W,
    /// The baseline has been announced.
    started: bool,
    /// Consecutive non-improving candidates since the last improvement.
    slower_since_faster: usize,
    /// A transient `Slower` line is currently on screen, unterminated.
    slower_line_visible: bool,
}

impl<W: std::io::Write> SearchProgress<W> {
    pub(crate) fn new(out: W) -> Self {
        Self {
            out,
            started: false,
            slower_since_faster: 0,
            slower_line_visible: false,
        }
    }

    /// The BASELINE: the first profiled plan, announced once. (Main
    /// renamed this label from `Search` to `Start` in the same commit:
    /// the first line reports the baseline, not a search result.)
    pub(crate) fn start(&mut self, nanos: u128) {
        if self.started {
            return;
        }
        self.started = true;
        let _ = writeln!(
            self.out,
            "   {:>6} {}",
            "Start".cyan().bold(),
            display_nanos(nanos)
        );
        let _ = self.out.flush();
    }

    /// One profiled candidate after the baseline: a permanent `Faster`
    /// line carrying the new best, or the transient `Slower x{n}`
    /// counter (reset to zero by every improvement).
    pub(crate) fn report(&mut self, improved: bool, nanos: u128) {
        let _ = write!(self.out, "\r\x1b[2K");
        if improved {
            self.slower_since_faster = 0;
            self.slower_line_visible = false;
            let _ = writeln!(
                self.out,
                "   {:>6} {}",
                "Faster".green().bold(),
                display_nanos(nanos)
            );
        } else {
            self.slower_since_faster += 1;
            self.slower_line_visible = true;
            let _ = write!(
                self.out,
                "   {:>6} x{}",
                "Slower".yellow().bold(),
                self.slower_since_faster
            );
        }
        let _ = self.out.flush();
    }

    /// End of search: clear a pending transient `Slower` line so it
    /// does not survive as a half-written row.
    pub(crate) fn finish(&mut self) {
        if self.slower_line_visible {
            let _ = write!(self.out, "\r\x1b[2K");
            self.slower_line_visible = false;
        }
        let _ = self.out.flush();
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

/// Aggregated classification of rejected genomes across one search.
#[derive(Debug, Clone, Default)]
pub struct RefusalBreakdown {
    /// Genomes whose extraction produced no plan for some output.
    pub extract_refusals: usize,
    /// ...of those, how many involved a CHOICE-CYCLE (the genome's
    /// chosen producers block on each other).
    ///
    /// INVARIANT (2026-09-02): sampling and mutation both keep a
    /// genome's chosen-edge graph acyclic, so for a SAMPLED genome this
    /// can only be nonzero through the sampler's documented full-list
    /// fallback — a component position with no acyclic option at all.
    /// A choice cycle on an acyclic chosen-edge graph is a sampler bug
    /// and stops the search (see `search_implementations_with_runtime`).
    /// Genomes assembled by hand (the election boards) are of course
    /// still free to name cycles, and are still counted here.
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

/// Producer index shorthand: class -> its candidate `(constructor
/// name, choice)` entries, in the index's own deterministic order.
pub type ProducerIndex = BTreeMap<ClassId, Vec<(String, ProducerChoice)>>;

/// Pick a position: uniformly among `allowed`, or — when nothing is
/// admissible — uniformly over the FULL candidate list.
///
/// THE FALLBACK is deliberate and unchanged from the 2026-08-07
/// sampler: a class every one of whose candidates sources from a member
/// of its own component that is not yet assigned has no acyclic option
/// at this position, which means its component holds no acyclic genome
/// at all through this class. Refusing to choose would silently shrink
/// the space; choosing anyway keeps the refusal accounting (and the
/// blockage anatomy) as the loud diagnosis for that corner.
fn choose_position(rng: &mut StdRng, allowed: &[usize], total: usize) -> usize {
    if allowed.is_empty() {
        rng.random_range(0..total)
    } else {
        allowed[rng.random_range(0..allowed.len())]
    }
}

/// GENERATION-0 SAMPLING: a FOREST inside every component.
///
/// Each component's members are assigned in a random order built one
/// step at a time. A member may elect a candidate with
/// intra-component sources only when ALL of them are already assigned,
/// so every chosen intra-component edge points BACKWARD along the
/// order — acyclic by construction.
///
/// THE ORDER IS DRAWN FROM THE ADMISSIBLE ONES: at each step the next
/// member is picked uniformly among those that still HAVE an
/// admissible candidate, so the first member always progresses and no
/// member is forced into the fallback by an unlucky shuffle. (A
/// uniformly random order over all members would sometimes lead with a
/// member that cannot progress — e.g. a layout copy whose only sources
/// are its own component — and then the fallback could weld a cycle
/// the 2026-08-07 sampler would have refused. Restricting to
/// admissible orders removes no genome: see COVERAGE.)
///
/// COVERAGE. This admits chains and forests, not just the 2026-08-07
/// star (one elected primary, everyone else copying it): for any
/// acyclic assignment of intra-component candidates there is a
/// topological order of the chosen edges, at every prefix of which the
/// next member's own chosen candidate is admissible — so that member
/// is in the pool, and the assignment is reachable. The only genomes
/// excluded are the cyclic ones. Classes outside every component are
/// sampled freely: their candidates' edges all leave the component,
/// and the condensation of the SCC decomposition is a DAG.
pub fn sample_genome(index: &ProducerIndex, space: &SamplingSpace, rng: &mut StdRng) -> Genome {
    sample_genome_reporting(index, space, rng).0
}

/// [`sample_genome`] plus the classes that had to take the full-list
/// fallback, in the order they took it — the sampler's own account of
/// where its acyclicity invariant was unenforceable. An EMPTY list is
/// the guarantee: the genome's chosen-edge graph is acyclic.
pub fn sample_genome_reporting(
    index: &ProducerIndex,
    space: &SamplingSpace,
    rng: &mut StdRng,
) -> (Genome, Vec<ClassId>) {
    let mut genome = Genome::default();
    let mut fallbacks: Vec<ClassId> = Vec::new();
    for members in &space.components {
        // Per member, per candidate: how many of its intra-component
        // sources are still unassigned. Zero = admissible now. (Sources
        // are deduplicated, so one decrement each.)
        let mut pending: Vec<Vec<usize>> = members
            .iter()
            .map(|class| {
                space.intra_sources[class]
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<usize>>()
            })
            .collect();
        // How many admissible candidates each member currently has.
        let mut admissible: Vec<usize> = pending
            .iter()
            .map(|per_candidate| per_candidate.iter().filter(|left| **left == 0).count())
            .collect();
        // source class -> the (member, candidate) counters it releases.
        let mut dependents: std::collections::BTreeMap<&ClassId, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (member, class) in members.iter().enumerate() {
            for (candidate, sources) in space.intra_sources[class].iter().enumerate() {
                for source in sources {
                    dependents
                        .entry(source)
                        .or_default()
                        .push((member, candidate));
                }
            }
        }

        let mut assigned = vec![false; members.len()];
        for _ in 0..members.len() {
            // The next member: uniform over those that can still choose
            // admissibly, or — when none can — over whoever is left,
            // which is the full-list fallback.
            let mut pool: Vec<usize> = (0..members.len())
                .filter(|member| !assigned[*member] && admissible[*member] > 0)
                .collect();
            let forced = pool.is_empty();
            if forced {
                pool = (0..members.len())
                    .filter(|member| !assigned[*member])
                    .collect();
            }
            let member = pool[rng.random_range(0..pool.len())];
            let class = &members[member];
            let candidates = &index[class];
            let allowed: Vec<usize> = (0..candidates.len())
                .filter(|position| pending[member][*position] == 0)
                .collect();
            debug_assert_eq!(allowed.is_empty(), forced);
            if forced {
                fallbacks.push(class.clone());
            }
            let position = choose_position(rng, &allowed, candidates.len());
            genome
                .choices
                .insert(class.clone(), candidates[position].1.clone());
            assigned[member] = true;
            for (other, candidate) in dependents.get(class).into_iter().flatten() {
                pending[*other][*candidate] -= 1;
                if pending[*other][*candidate] == 0 {
                    admissible[*other] += 1;
                }
            }
        }
    }
    for (class, candidates) in index {
        if genome.choices.contains_key(class) {
            continue;
        }
        let position = rng.random_range(0..candidates.len());
        genome
            .choices
            .insert(class.clone(), candidates[position].1.clone());
    }
    (genome, fallbacks)
}

/// Would routing `class` through `sources` close a cycle in the genome's
/// chosen intra-component edge graph? A DFS from each source over the
/// OTHER members' current choices; reaching `class` again — or a source
/// that IS `class` — is the cycle. Intra-component edges never leave the
/// component, so the walk is component-local and small.
fn flip_closes_cycle(
    index: &ProducerIndex,
    space: &SamplingSpace,
    genome: &Genome,
    class: &ClassId,
    sources: &[ClassId],
) -> bool {
    let mut stack: Vec<ClassId> = sources.to_vec();
    let mut seen: FxHashSet<ClassId> = FxHashSet::default();
    while let Some(node) = stack.pop() {
        if &node == class {
            return true;
        }
        if !seen.insert(node.clone()) {
            continue;
        }
        let Some(position) = space.chosen_position(index, genome, &node) else {
            continue;
        };
        if let Some(next) = space
            .intra_sources
            .get(&node)
            .and_then(|per_candidate| per_candidate.get(position))
        {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

/// POINT MUTATION under the same invariant: a flip of one class to one
/// candidate is admissible exactly when the resulting chosen-edge graph
/// closes no cycle through that class (see [`flip_closes_cycle`]).
/// Progressing candidates have no intra-component sources and so are
/// always admissible; a candidate sourcing from its own class never is.
/// Mutations hit ANY producer class, dead rows included (deliberately —
/// a dead-row mutation is free now and pre-stages the choice a later
/// route flip lands on).
pub fn mutate_genome(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    classes: &[ClassId],
    rng: &mut StdRng,
    count: usize,
) -> Genome {
    mutate_genome_reporting(parent, index, space, classes, rng, count).0
}

/// [`mutate_genome`] plus the classes whose flip found NO admissible
/// candidate and took the full-list fallback (see
/// [`sample_genome_reporting`]).
pub fn mutate_genome_reporting(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    classes: &[ClassId],
    rng: &mut StdRng,
    count: usize,
) -> (Genome, Vec<ClassId>) {
    let mut child = parent.clone();
    let mut fallbacks: Vec<ClassId> = Vec::new();
    if classes.is_empty() {
        return (child, fallbacks); // one-point genome space: nothing to mutate
    }
    for _ in 0..count {
        let class = &classes[rng.random_range(0..classes.len())];
        let candidates = &index[class];
        let sources = &space.intra_sources[class];
        let allowed: Vec<usize> = (0..candidates.len())
            .filter(|&position| !flip_closes_cycle(index, space, &child, class, &sources[position]))
            .collect();
        if allowed.is_empty() {
            fallbacks.push(class.clone());
        }
        let position = choose_position(rng, &allowed, candidates.len());
        child
            .choices
            .insert(class.clone(), candidates[position].1.clone());
    }
    (child, fallbacks)
}

/// [`sample_genome_reporting`] from a seed — the entry the
/// sampler-invariant boards use, so a test crate needs no `rand` of its
/// own. Returns the genome and its fallback classes.
pub fn sample_genome_with_seed(
    index: &ProducerIndex,
    space: &SamplingSpace,
    seed: u64,
) -> (Genome, Vec<ClassId>) {
    sample_genome_reporting(index, space, &mut StdRng::seed_from_u64(seed))
}

/// [`mutate_genome_reporting`] from a seed (see
/// [`sample_genome_with_seed`]).
pub fn mutate_genome_with_seed(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    count: usize,
    seed: u64,
) -> (Genome, Vec<ClassId>) {
    let classes: Vec<ClassId> = index.keys().cloned().collect();
    mutate_genome_reporting(
        parent,
        index,
        space,
        &classes,
        &mut StdRng::seed_from_u64(seed),
        count,
    )
}

/// THE BUFFERIZE TRIPWIRE (2026-09-02), the second half of the sampler
/// invariant. Sampling and mutation keep a genome's chosen intra-component
/// edges acyclic, and the extractor plans an input terminal from the
/// boundary rather than from any producer, so no sampled or mutated genome
/// can extract a graph with a cycle in it. If one reaches bufferize anyway,
/// the sampler's candidate graph has drifted from the plan the extractor
/// actually emits: the search STOPS and names the genome's own chosen
/// intra-component edges alongside the cycle bufferize found, instead of
/// quietly losing that genome as an ordinary refusal. Every other
/// bufferize refusal (dead ends, unsupported ownership, unschedulable
/// anti-edges) passes through untouched.
fn bufferize_cycle_tripwire(
    err: &anyhow::Error,
    index: &ProducerIndex,
    space: &SamplingSpace,
    genome: &Genome,
) -> Result<()> {
    let text = format!("{err:#}");
    if !text.contains(luminal::bufferize::EXTRACTED_GRAPH_CYCLE) {
        return Ok(());
    }
    let intra = space.chosen_intra_edges(index, genome);
    let intra: Vec<String> = intra
        .iter()
        .map(|(class, sources)| format!("{class} <- {sources:?}"))
        .collect();
    Err(anyhow!(
        "sampler invariant violated: a sampled genome extracted a CYCLIC graph \
         (bufferize refused it). Sampling and mutation keep the chosen \
         intra-component edges acyclic and input terminals are planned from the \
         boundary, so this means the sampler's candidate graph disagrees with the \
         plan the extractor emitted. Chosen intra-component edges: [{}]. {text}",
        intra.join("; ")
    ))
}

/// THE SELECTION LOOP for this backend: the caller supplies its OWN
/// matcher vocabulary (the cuBLASLt marker set is an instance option)
/// and its OWN allow list. Deterministic for a fixed seed.
///
/// NO CALLER DATA (D6, 2026-09-03): candidates are ranked by
/// [`crate::heuristic::heuristic_cost_of`], which never runs anything,
/// so there is nothing to stage. The ladder's `search` still takes the
/// caller's payloads and still checks them against the program's bound
/// inputs — it just does not hand them here.
pub fn search_implementations(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> Result<SearchOutcome> {
    let mut timings = SearchTimings::default();
    let analysis_start = Instant::now();
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
                        egraph,
                        &dps,
                        "implementation search",
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
                    // DEVICE-FREE RANKING (D6): a static prior over the
                    // extracted graph, never a measurement. See
                    // [`crate::heuristic`] for what it does and does not
                    // claim.
                    let profiled: Result<u128> = Ok(crate::heuristic::heuristic_cost_of(&graph));
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
                let built =
                    luminal::layouts::decode_layout_table(egraph, &dps, "implementation search")
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
    /// bucketed. A bucketed dim is refused a range binding, so these
    /// never collide with the per-bucket seeds appended after them.
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
    options: &CompileOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: impl Fn() -> Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> Result<Vec<BucketPlan>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");
    let mut plans = Vec::new();
    for (ranges, representative, program) in bucket_renders(assembly, dim_buckets)? {
        let text = format!("{}\n\n{}", assembly.assembled_program, program.text);
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        let serialized = egraph
            .serialize(luminal::prelude::egglog::SerializeConfig::default())
            .egraph;
        let outcome = search_implementations(
            &serialized,
            &program,
            options,
            allow_override.clone(),
            matchers(),
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
    }
}
