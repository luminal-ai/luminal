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
use colored::Colorize;
use egraph_serialize::ClassId;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::BTreeMap;

use crate::bufferize::BufferIrGraph;
use crate::extractor::{self, Genome, ProducerChoice, SamplingSpace};
use crate::graph::LogicalProgram;

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
    /// Print live search progress to stderr (`Start` / `Faster` /
    /// `Slower x{n}`). ON by default, matching main's
    /// `CompileOptions::search_log`; overridden by `SEARCH_LOG=0`/`1`
    /// or `LUMINAL_LOG=1`.
    pub search_log: bool,
}

impl Default for ImplementationSearchOptions {
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

impl ImplementationSearchOptions {
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

#[cfg(test)]
mod progress_tests {
    use super::SearchProgress;

    /// Read the WORDS whatever `colored` decided about this terminal:
    /// drop every escape sequence, keep the carriage returns (they are
    /// the transient-line mechanism the test is pinning).
    fn strip_ansi(text: &str) -> String {
        let mut out = String::new();
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                for c in chars.by_ref() {
                    if ('\x40'..='\x7e').contains(&c) && c != '[' {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn progress_prints_start_once_faster_per_improvement_and_a_resetting_slower_counter() {
        let mut sink: Vec<u8> = Vec::new();
        {
            let mut progress = SearchProgress::new(&mut sink);
            progress.start(4_000_000);
            // The baseline is announced exactly once, even if the loop
            // were to ask twice.
            progress.start(9_000_000);
            progress.report(false, 9_000_000);
            progress.report(false, 8_000_000);
            progress.report(true, 3_000_000);
            progress.report(false, 5_000_000);
            progress.finish();
        }
        let raw = String::from_utf8(sink).expect("utf8");
        let text = strip_ansi(&raw);

        assert_eq!(text.matches("Start").count(), 1, "Start once: {text:?}");
        assert!(text.contains("Start 4.000 ms"), "baseline nanos: {text:?}");

        assert_eq!(
            text.matches("Faster").count(),
            1,
            "one improvement: {text:?}"
        );
        assert!(
            text.contains("Faster 3.000 ms"),
            "the new best rides the Faster line: {text:?}"
        );

        // The counter climbs while nothing improves and RESETS on the
        // improvement, so the last candidate is x1 again, not x3.
        assert!(text.contains("Slower x1"), "{text:?}");
        assert!(text.contains("Slower x2"), "{text:?}");
        assert!(!text.contains("Slower x3"), "counter must reset: {text:?}");
        assert_eq!(text.matches("Slower x1").count(), 2, "{text:?}");

        // Every report (and the finish) rewrites the transient line in
        // place: no bar-relative cursor math, just \r + erase-line.
        assert_eq!(raw.matches("\r\x1b[2K").count(), 5, "{raw:?}");
        // Faster/Start lines are permanent (newline-terminated); the
        // pending Slower line is not, and finish() clears it.
        assert!(raw.ends_with("\r\x1b[2K"), "{raw:?}");
    }
}

#[derive(Debug)]
pub struct SearchOutcome<L: crate::bufferize::PlanLayout> {
    pub best_plan: BufferIrGraph<L>,
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

/// Search the saturated e-graph for the fastest executable plan, profiling
/// with the given caller data. Deterministic for a fixed seed.
/// As [`search_implementations`], with the runtime's ALLOWABLE-OPS
/// inventory made explicit (M3 Step 2: per-runtime, unstandardized).
/// How search prices a candidate plan. Runtimes supply their own
/// (ruling 2026-08-17: every runtime owns its execution — including
/// how its candidates are timed); the in-core implementations are the
/// reference host executor (the historical behavior) and a static
/// ranker that never executes.
pub trait PlanProfiler<L: crate::bufferize::PlanLayout> {
    /// The plan's cost over `trials` with buffer-keyed inputs; smaller
    /// wins. `heuristic_cost` is the extracted graph's summed
    /// bytes-moved estimate, for profilers that rank without running.
    ///
    /// `best_so_far` is the INCUMBENT's metric in nanos — `None` while
    /// the search holds no candidate yet, since the first candidate IS
    /// the baseline. It is the early-stop cutoff (#386, ruling 3 of
    /// 2026-09-02: the cutoff crosses this seam as a fifth POSITIONAL
    /// argument, not a struct). A profiler may use it to stop trialing
    /// a candidate that has already lost; whatever it then returns is
    /// still ranked normally, so early stop never changes which
    /// candidates are eligible — only how much time is spent timing
    /// ones already out of contention. Ignoring it is always correct.
    fn profile(
        &mut self,
        plan: &crate::bufferize::BufferIrGraph<L>,
        input_data: &FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer>,
        trials: usize,
        heuristic_cost: u64,
        best_so_far: Option<u128>,
    ) -> Result<u128>;
}

/// Main's `early_stop_exceeded` (its `src/op.rs`), retyped from
/// `Duration` to this branch's u128 nanos: true once a candidate's mean
/// trial cost exceeds `best * factor`, i.e. the candidate has already
/// lost by at least that margin and further trials can only refine a
/// metric that is out of contention.
///
/// The in-core caller ([`crate::implementation_search`]'s reference
/// profiler, in `luminal_reference`) applies it at `factor = 1.0` to a
/// LOWER BOUND on the candidate's final mean, which makes the stop
/// exact rather than heuristic: a candidate whose best conceivable
/// final mean already exceeds the incumbent cannot win. The factor
/// survives because it is main's semantics and main's tuning knob
/// (`CompileOptions::early_stop_factor`), and a device profiler
/// mirroring this design will want it.
pub fn early_stop_exceeded(mean_nanos: u128, best_nanos: u128, factor: f64) -> bool {
    mean_nanos as f64 > best_nanos as f64 * factor
}

#[cfg(test)]
mod early_stop_tests {
    use super::early_stop_exceeded;

    /// Main's `test_early_stop_exceeded` (`src/op.rs`), retyped from
    /// `Duration` to nanos.
    #[test]
    fn early_stop_exceeded_keeps_mains_margin_semantics() {
        const MS: u128 = 1_000_000;
        let best = 5 * MS;
        // 2x cutoff: 10ms mean is at the boundary, not over it.
        assert!(!early_stop_exceeded(10 * MS, best, 2.0));
        assert!(early_stop_exceeded(11 * MS, best, 2.0));
        // A candidate faster than best never stops early.
        assert!(!early_stop_exceeded(4 * MS, best, 2.0));
        // Factor 1.0 stops anything slower than best.
        assert!(early_stop_exceeded(6 * MS, best, 1.0));
        // ...and NOT a tie: the incumbent keeps its seat, but a tied
        // candidate is still worth finishing (it is not yet losing).
        assert!(!early_stop_exceeded(5 * MS, best, 1.0));
    }
}

/// Rank by the heuristic byte-move estimate without executing —
/// deterministic, device-free search for runtimes that cannot (or
/// need not) run candidates on the searching host.
#[derive(Default)]
pub struct StaticProfiler;

impl<L: crate::bufferize::PlanLayout> PlanProfiler<L> for StaticProfiler {
    fn profile(
        &mut self,
        _plan: &crate::bufferize::BufferIrGraph<L>,
        _input_data: &FxHashMap<i64, crate::buffer_tensor_ir::TypedBuffer>,
        _trials: usize,
        heuristic_cost: u64,
        // Ruling 4 (2026-09-02): CL must eventually profile ON DEVICE,
        // mirroring the reference profiler's design; until it does, the
        // static ranker runs no trials, so there is nothing to cut
        // short and the cutoff is accepted and ignored.
        _best_so_far: Option<u128>,
    ) -> Result<u128> {
        Ok(u128::from(heuristic_cost).saturating_add(1))
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
    if !text.contains(crate::bufferize::EXTRACTED_GRAPH_CYCLE) {
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

/// The runtime-owned search entry (ruling 2026-08-17): the caller
/// supplies its OWN matcher set (None = the in-core reference
/// registry, which Step B moves out) and its OWN profiler.
#[expect(
    clippy::too_many_arguments,
    reason = "the public runtime boundary keeps each independently owned search input explicit"
)]
pub fn search_implementations_with_runtime<L: crate::bufferize::PlanLayout>(
    egraph: &egraph_serialize::EGraph,
    program: &LogicalProgram,
    input_data: &FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    allow_override: Option<Vec<&'static str>>,
    matchers: Vec<Box<dyn crate::layout_ir::OpMatcher>>,
    layout_decoder: &dyn crate::layout_ir::LayoutDecoder<L>,
    profiler: &mut dyn PlanProfiler<L>,
) -> Result<SearchOutcome<L>> {
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
    // Decoded layouts are pure functions of (layout class, dtype fact)
    // — the decoder cache contract — so one cache serves every genome.
    let mut layout_cache: std::collections::HashMap<
        (egraph_serialize::ClassId, Option<crate::dtype::PlanDtype>),
        L,
    > = std::collections::HashMap::new();
    let mut plans_profiled = 0usize;
    let mut fingerprint_hits = 0usize;
    // Refusal accounting, minimal form (Step 5 down-payment): keep the
    // first few refusal reasons so a fully-refused search names its
    // causes instead of shrugging.
    let mut refusals: Vec<String> = Vec::new();
    let mut breakdown = RefusalBreakdown::default();
    let mut best: Option<(u128, Genome, BufferIrGraph<L>)> = None;
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
                    let dps = crate::dps::dps_rewrite(&graph);
                    let built = extractor::decoded_layout_table(
                        egraph,
                        &dps,
                        layout_decoder,
                        &mut layout_cache,
                    )
                    .and_then(|table| crate::bufferize::bufferize(&dps, &table));
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
                    let heuristic_total: u64 = graph
                        .dag
                        .node_weights()
                        .map(|node| match node {
                            crate::layout_ir::ExtractedNode::LayoutOp(op) => op.heuristic_cost,
                            _ => 0,
                        })
                        .sum();
                    let profile_start = Instant::now();
                    // The incumbent's metric is the early-stop cutoff
                    // (#386). `None` on the first candidate: it IS the
                    // baseline, so there is nothing to have lost to.
                    let best_so_far = best.as_ref().map(|(best_nanos, _, _)| *best_nanos);
                    let profiled = profiler.profile(
                        &plan,
                        input_data,
                        options.trials,
                        heuristic_total,
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
                let dps = crate::dps::dps_rewrite(&graph);
                let built = extractor::decoded_layout_table(
                    egraph,
                    &dps,
                    layout_decoder,
                    &mut layout_cache,
                )
                .and_then(|table| crate::bufferize::bufferize(&dps, &table));
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
pub struct BucketPlan<L: crate::bufferize::PlanLayout> {
    pub ranges: BTreeMap<crate::shape::Symbol, (usize, usize)>,
    pub representative: crate::shape::DynMap,
    pub program: LogicalProgram,
    pub outcome: SearchOutcome<L>,
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
#[expect(
    clippy::too_many_arguments,
    reason = "bucket search composes independently owned graph, runtime, matcher, and profiling inputs"
)]
pub fn bucketed_search_implementations<L: crate::bufferize::PlanLayout>(
    graph: &crate::graph::Graph,
    dim_buckets: &BTreeMap<crate::shape::Symbol, Vec<crate::graph::DimBucket>>,
    input_data: impl Fn(
        &crate::shape::DynMap,
    )
        -> FxHashMap<petgraph::graph::NodeIndex, crate::buffer_tensor_ir::TypedBuffer>,
    options: &ImplementationSearchOptions,
    assembled_program: &str,
    matchers: impl Fn() -> Vec<Box<dyn crate::layout_ir::OpMatcher>>,
    layout_decoder: &dyn crate::layout_ir::LayoutDecoder<L>,
    profiler: &mut dyn PlanProfiler<L>,
    bindings: &dyn crate::runtime_binding::RuntimeBindingsGenerator,
) -> Result<Vec<BucketPlan<L>>> {
    ensure!(!dim_buckets.is_empty(), "no dim buckets supplied");

    // M3 Topic C: buckets assemble NATIVELY — one recorder model, per-
    // bucket binding seeds (ranges for the bucket-wide validation render,
    // tight [n,n] pins for the representative render). The model text
    // never changes across buckets; only the binding does.
    let (pre, input_slots, output_slots, post, _labeled) = graph
        .logical
        .bound_parts(bindings)
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
            text: format!("{pre}{}{}{post}", seeds_text(seeds), bindings.schedule()),
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
        let text = format!("{}\n\n{}", assembled_program, validation.text);
        crate::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} fails bucket-wide validation: {err}"))?;

        // Representative render: pinned via tight bounds, searched, profiled.
        let mut pin_seeds: BTreeMap<crate::shape::Symbol, (u64, u64)> = BTreeMap::new();
        for (dim, value) in &representative {
            pin_seeds.insert(*dim, (*value as u64, *value as u64));
        }
        let program = assemble(&pin_seeds);
        let text = format!("{}\n\n{}", assembled_program, program.text);
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .map_err(|err| anyhow!("bucket {ranges:?} representative render fails: {err}"))?;
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
        let data = input_data(&representative);
        let outcome = search_implementations_with_runtime(
            &serialized,
            &program,
            &data,
            options,
            None,
            matchers(),
            layout_decoder,
            profiler,
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

/// The covering bucket plan for a concrete dim assignment, if any.
pub fn select_bucket<'a, L: crate::bufferize::PlanLayout>(
    plans: &'a [BucketPlan<L>],
    dims: &crate::shape::DynMap,
) -> Option<&'a BucketPlan<L>> {
    plans.iter().find(|plan| {
        plan.ranges.iter().all(|(dim, (min, max))| {
            dims.get(dim)
                .is_some_and(|value| value >= min && value <= max)
        })
    })
}

#[cfg(test)]
mod tests {
    // DEP-WORLD tests (Step B): these suites drive the reference runtime
    // through the luminal_reference dev-dependency, so every luminal type
    // they touch must come from the `luminal::` build that crate links,
    // never `crate` (the cyclic dev-dependency compiles the library
    // twice, and the two builds' types do not unify).
    use std::collections::BTreeMap;

    use egglog::SerializeConfig;
    use rustc_hash::FxHashMap;

    use luminal::dtype::DType;
    use luminal::graph::Graph;
    use luminal::implementation_search::{
        ImplementationSearchOptions, bucketed_search_implementations, select_bucket,
    };
    use luminal_reference::{ReferenceRuntime, search_implementations};

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
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("native program");
        let text = format!(
            "{}\n\n{}",
            luminal_reference::assembled_program(),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
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

    /// Main's `search_passes_best_so_far_to_profile_early_stop`
    /// (`src/graph.rs`, #386), re-expressed against this branch's seam:
    /// the selection loop passes `None` for the FIRST profiled
    /// candidate — it is the baseline — and the incumbent's metric for
    /// every later one. Main's version asserts `Some((best, factor))`;
    /// here the cutoff is the bare incumbent (ruling 3, 2026-09-02).
    #[test]
    fn search_passes_the_incumbent_metric_to_every_later_profile_call() {
        #[derive(Default)]
        struct RecordingProfiler {
            seen: Vec<Option<u128>>,
        }

        impl luminal::implementation_search::PlanProfiler<luminal_reference::RefLayout>
            for RecordingProfiler
        {
            fn profile(
                &mut self,
                _plan: &luminal::bufferize::BufferIrGraph<luminal_reference::RefLayout>,
                _input_data: &FxHashMap<i64, luminal::buffer_tensor_ir::TypedBuffer>,
                _trials: usize,
                _heuristic_cost: u64,
                best_so_far: Option<u128>,
            ) -> anyhow::Result<u128> {
                self.seen.push(best_so_far);
                // Strictly increasing metrics (main's trick): the first
                // candidate scores 0 and stays the incumbent, so every
                // later call must see exactly that.
                Ok(self.seen.len() as u128 - 1)
            }
        }

        let mut cx = Graph::new();
        let x = cx.tensor(4, DType::F32);
        let y = cx.tensor(4, DType::F32);
        let _sum = (x + y).output();
        let _product = (x * y).output();
        let program = cx
            .logical
            .bound_program(&luminal_reference::ReferenceBindings)
            .expect("native program");
        let text = format!(
            "{}\n\n{}",
            luminal_reference::assembled_program(),
            program.text
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &text)
            .expect("program runs");
        let serialized = egraph.serialize(SerializeConfig::default()).egraph;

        let mut inputs = FxHashMap::default();
        inputs.insert(x.id, vec![1.0f32, 2.0, 3.0, 4.0].into());
        inputs.insert(y.id, vec![10.0f32, 20.0, 30.0, 40.0].into());

        let options = ImplementationSearchOptions {
            generations: 3,
            generation_size: 4,
            mutations: 2,
            trials: 1,
            seed: 0,
            search_log: false,
        };
        let mut profiler = RecordingProfiler::default();
        let outcome = luminal::implementation_search::search_implementations_with_runtime(
            &serialized,
            &program,
            &inputs,
            &options,
            Some(luminal_reference::reference_allow_list()),
            luminal_reference::ops::built_in_matchers(),
            &luminal_reference::ReferenceLayoutDecoder,
            &mut profiler,
        )
        .expect("search finds an executable plan");

        assert!(
            profiler.seen.len() > 1,
            "the search must profile more than the baseline candidate: {:?}",
            profiler.seen
        );
        assert_eq!(
            profiler.seen.len(),
            outcome.plans_profiled,
            "one profile call per profiled plan (the rest are cache hits)"
        );
        assert_eq!(
            profiler.seen[0], None,
            "no incumbent exists before the first candidate is profiled"
        );
        for (index, seen) in profiler.seen.iter().enumerate().skip(1) {
            assert_eq!(
                *seen,
                Some(0),
                "candidate {index} must receive the incumbent's metric"
            );
        }
        assert_eq!(
            outcome.best_nanos, 0,
            "the first candidate's metric is the running best"
        );
    }

    /// BUCKETED: two buckets over 'a', each validated bucket-wide
    /// (range seeds) and searched at its representative; selection covers
    /// runtime dims; each bucket's plan agrees with ReferenceRuntime at its
    /// representative.
    #[test]
    fn bucketed_search_validates_searches_and_selects() {
        use luminal::graph::DimBucket;

        let build = |dim: usize| {
            let mut cx = Graph::new();
            cx.set_dim('a', dim);
            let x = cx.tensor(('a', 2), DType::F32);
            let y = cx.tensor(('a', 2), DType::F32);
            let out = (x * y).output();
            (cx, x, y, out)
        };

        let buckets: BTreeMap<luminal::shape::Symbol, Vec<DimBucket>> = [(
            luminal::shape::Symbol::from('a'),
            vec![DimBucket::new(2, 4), DimBucket::new(5, 9)],
        )]
        .into();

        // The graph used for SEARCH: built at any pin (per-bucket renders
        // re-pin), sharing the same HLIR shape.
        let (cx, x, y, _out) = build(3);
        let data_for = |rep: &luminal::shape::DynMap| {
            let n = rep[&luminal::shape::Symbol::from('a')] * 2;
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
            luminal_reference::assembled_program(),
            luminal_reference::ops::built_in_matchers,
            &luminal_reference::ReferenceLayoutDecoder,
            &mut luminal_reference::ReferenceProfiler,
            &luminal_reference::ReferenceBindings,
        )
        .expect("bucketed search completes");
        assert_eq!(plans.len(), 2, "one plan per bucket");

        // Selection covers each bucket; out-of-range dims select nothing.
        let mut dims = FxHashMap::default();
        dims.insert(luminal::shape::Symbol::from('a'), 3usize);
        assert!(
            select_bucket(&plans, &dims).unwrap().ranges[&luminal::shape::Symbol::from('a')]
                == (2, 4)
        );
        dims.insert(luminal::shape::Symbol::from('a'), 7usize);
        assert!(
            select_bucket(&plans, &dims).unwrap().ranges[&luminal::shape::Symbol::from('a')]
                == (5, 9)
        );
        dims.insert(luminal::shape::Symbol::from('a'), 20usize);
        assert!(select_bucket(&plans, &dims).is_none());

        // Numeric agreement at each bucket's representative.
        for plan in &plans {
            let rep = plan.representative[&luminal::shape::Symbol::from('a')];
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

/// THE SAMPLER-INVARIANT UNIT BOARD (2026-09-02): components, forest
/// sampling and the mutation cycle check on hand-built candidate
/// graphs — no e-graph, no runtime, so the rule itself is under test
/// rather than a graph that happens to exercise it.
///
/// CRATE-LOCAL types only (`crate::`, never `luminal::`): the sibling
/// `mod tests` above is a DEP-WORLD suite driving the reference runtime
/// through the cyclic dev-dependency, which compiles this library twice
/// and does not unify the two builds' types. These tests touch neither.
#[cfg(test)]
mod sampler_tests {
    use std::collections::BTreeMap;

    use egraph_serialize::{ClassId, NodeId};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::extractor::{Genome, ProducerChoice, SamplingSpace, edges_have_cycle};

    use super::{ProducerIndex, bufferize_cycle_tripwire, mutate_genome, sample_genome};

    fn class(name: &str) -> ClassId {
        ClassId::from(name)
    }

    /// One candidate of one class: `(candidate name, input classes)`.
    type CandidateRow<'a> = (&'a str, &'a [&'a str]);
    /// One class of a hand-built board: `(class, its candidates)`.
    type BoardRow<'a> = (&'a str, &'a [CandidateRow<'a>]);

    /// A producer index and its candidate graph from one table:
    /// `(class, [(candidate name, [input classes])])`. Candidate
    /// positions line up between the two by construction — the same
    /// parallelism `SamplingSpace` assumes of the real index.
    fn build(table: &[BoardRow<'_>]) -> (ProducerIndex, SamplingSpace) {
        let mut index: ProducerIndex = BTreeMap::new();
        let mut inputs: BTreeMap<ClassId, Vec<Vec<ClassId>>> = BTreeMap::new();
        for (owner, candidates) in table {
            index.insert(
                class(owner),
                candidates
                    .iter()
                    .map(|(name, _)| {
                        (
                            (*name).to_string(),
                            ProducerChoice {
                                enode: NodeId::from(format!("{owner}::{name}")),
                                output_index: 0,
                            },
                        )
                    })
                    .collect(),
            );
            inputs.insert(
                class(owner),
                candidates
                    .iter()
                    .map(|(_, sources)| sources.iter().map(|s| class(s)).collect())
                    .collect(),
            );
        }
        let space = SamplingSpace::from_candidate_inputs(inputs);
        (index, space)
    }

    /// The genome electing the named candidate per class — the hand-built
    /// genome a FREE sampler (uniform over every candidate, no invariant)
    /// is free to draw.
    fn genome_electing(index: &ProducerIndex, picks: &[(&str, &str)]) -> Genome {
        let mut genome = Genome::default();
        for (owner, name) in picks {
            let (_, choice) = index[&class(owner)]
                .iter()
                .find(|(candidate, _)| candidate == name)
                .expect("the board names this candidate");
            genome.choices.insert(class(owner), choice.clone());
        }
        genome
    }

    /// The candidate NAME the genome elected per class, sorted by class
    /// — a readable genome identity for coverage assertions.
    fn spelling(index: &ProducerIndex, genome: &Genome) -> Vec<String> {
        index
            .iter()
            .map(|(owner, candidates)| {
                let choice = &genome.choices[owner];
                let (name, _) = candidates
                    .iter()
                    .find(|(_, candidate)| candidate == choice)
                    .expect("the genome names an index entry");
                format!("{owner}={name}")
            })
            .collect()
    }

    fn cyclic(index: &ProducerIndex, space: &SamplingSpace, genome: &Genome) -> bool {
        edges_have_cycle(&space.chosen_edges(index, genome))
    }

    /// (i) TWO CLASSES RE-DESCRIBING EACH OTHER — the cuBLASLt
    /// collapse's shape in miniature. One component of size 2; over 200
    /// seeds the (intra, intra) pair never appears and all three acyclic
    /// combinations do.
    #[test]
    fn mutual_re_description_is_one_component_and_only_the_cycle_is_excluded() {
        let (index, space) = build(&[
            ("a", &[("prog", &[]), ("reads_b", &["b"])]),
            ("b", &[("prog", &[]), ("reads_a", &["a"])]),
        ]);
        assert_eq!(
            space.components,
            vec![vec![class("a"), class("b")]],
            "two classes reading each other are one non-trivial component"
        );

        let mut seen: std::collections::BTreeSet<Vec<String>> = std::collections::BTreeSet::new();
        for seed in 0..200u64 {
            let genome = sample_genome(&index, &space, &mut StdRng::seed_from_u64(seed));
            assert!(
                !cyclic(&index, &space, &genome),
                "seed {seed} sampled a cyclic genome: {:?}",
                spelling(&index, &genome)
            );
            seen.insert(spelling(&index, &genome));
        }
        let expected: std::collections::BTreeSet<Vec<String>> = [
            vec!["a=prog".to_string(), "b=prog".to_string()],
            vec!["a=prog".to_string(), "b=reads_a".to_string()],
            vec!["a=reads_b".to_string(), "b=prog".to_string()],
        ]
        .into_iter()
        .collect();
        assert_eq!(
            seen, expected,
            "exactly the three acyclic genomes are reachable"
        );
    }

    /// (ii) COVERAGE OF CHAINS: a 3-cycle of re-descriptions
    /// (a<-c, b<-a, c<-b) with a progressing option each. The CHAIN
    /// genome — a progresses, b reads a, c reads b — is a forest, not
    /// a star, and the 2026-08-07 sampler could not build it. It must
    /// be reachable; the all-intra 3-cycle must not be.
    #[test]
    fn chains_inside_a_component_are_reachable() {
        let (index, space) = build(&[
            ("a", &[("prog", &[]), ("reads_c", &["c"])]),
            ("b", &[("prog", &[]), ("reads_a", &["a"])]),
            ("c", &[("prog", &[]), ("reads_b", &["b"])]),
        ]);
        assert_eq!(
            space.components,
            vec![vec![class("a"), class("b"), class("c")]],
            "the 3-cycle is one component"
        );

        let chain = vec![
            "a=prog".to_string(),
            "b=reads_a".to_string(),
            "c=reads_b".to_string(),
        ];
        let mut chain_seen = false;
        for seed in 0..200u64 {
            let genome = sample_genome(&index, &space, &mut StdRng::seed_from_u64(seed));
            assert!(
                !cyclic(&index, &space, &genome),
                "seed {seed} sampled a cyclic genome: {:?}",
                spelling(&index, &genome)
            );
            chain_seen |= spelling(&index, &genome) == chain;
        }
        assert!(
            chain_seen,
            "the chain genome (a progresses, b reads a, c reads b) must be sampled"
        );
    }

    /// (iii) A COMPONENT WITH NO PROGRESSING MEMBER: every candidate of
    /// every member is intra-component, so the first member processed
    /// has no admissible candidate and takes the documented full-list
    /// fallback. The sampler must produce a total genome and not panic;
    /// the resulting cycle is the refusal accounting's business.
    #[test]
    fn a_component_with_no_progressing_member_falls_back() {
        let (index, space) = build(&[("a", &[("reads_b", &["b"])]), ("b", &[("reads_a", &["a"])])]);
        assert_eq!(space.components, vec![vec![class("a"), class("b")]]);
        for seed in 0..32u64 {
            let genome = sample_genome(&index, &space, &mut StdRng::seed_from_u64(seed));
            assert_eq!(genome.choices.len(), 2, "the genome stays total");
            assert!(
                cyclic(&index, &space, &genome),
                "the fallback is the ONLY route to a cyclic sample, and here it is forced"
            );
        }
    }

    /// MUTATION keeps the invariant: from every acyclic parent, 500
    /// mutated children over the two-class and three-class boards are
    /// acyclic — and the flip check is not merely refusing everything,
    /// since the children do move.
    #[test]
    fn mutation_never_closes_a_cycle() {
        for table in [
            &[
                (
                    "a",
                    &[("prog", &[] as &[&str]), ("reads_b", &["b"] as &[&str])]
                        as &[(&str, &[&str])],
                ),
                ("b", &[("prog", &[]), ("reads_a", &["a"])]),
            ] as &[(&str, &[(&str, &[&str])])],
            &[
                ("a", &[("prog", &[]), ("reads_c", &["c"])]),
                ("b", &[("prog", &[]), ("reads_a", &["a"])]),
                ("c", &[("prog", &[]), ("reads_b", &["b"])]),
            ],
        ] {
            let (index, space) = build(table);
            let classes: Vec<ClassId> = index.keys().cloned().collect();
            let mut moved = 0usize;
            for seed in 0..500u64 {
                let mut rng = StdRng::seed_from_u64(seed);
                let parent = sample_genome(&index, &space, &mut rng);
                let child = mutate_genome(&parent, &index, &space, &classes, &mut rng, 2);
                assert!(
                    !cyclic(&index, &space, &child),
                    "seed {seed}: mutation closed a cycle: {:?} -> {:?}",
                    spelling(&index, &parent),
                    spelling(&index, &child)
                );
                if spelling(&index, &child) != spelling(&index, &parent) {
                    moved += 1;
                }
            }
            assert!(moved > 0, "mutation must actually move the genome");
        }
    }

    /// A class whose candidate sources from ITSELF is a self-loop: a
    /// one-member component, and the sampler must never elect it.
    #[test]
    fn a_self_sourcing_candidate_is_never_elected() {
        let (index, space) = build(&[("a", &[("prog", &[]), ("reads_self", &["a"])])]);
        assert_eq!(
            space.components,
            vec![vec![class("a")]],
            "a self-loop is a non-trivial component of one"
        );
        let classes: Vec<ClassId> = index.keys().cloned().collect();
        for seed in 0..200u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let genome = sample_genome(&index, &space, &mut rng);
            assert_eq!(spelling(&index, &genome), vec!["a=prog".to_string()]);
            let child = mutate_genome(&genome, &index, &space, &classes, &mut rng, 3);
            assert_eq!(spelling(&index, &child), vec!["a=prog".to_string()]);
        }
    }

    /// THE BUFFERIZE TRIPWIRE fires on the cyclic-graph refusal and on
    /// nothing else, and it prints the genome's own chosen
    /// intra-component edges — the evidence for "the sampler's candidate
    /// graph disagrees with the plan the extractor emitted".
    #[test]
    fn the_bufferize_tripwire_fires_only_on_the_cycle_refusal() {
        let (index, space) = build(&[
            ("a", &[("prog", &[]), ("reads_b", &["b"])]),
            ("b", &[("prog", &[]), ("reads_a", &["a"])]),
        ]);
        let genome = genome_electing(&index, &[("a", "reads_b"), ("b", "prog")]);

        let ordinary = anyhow::anyhow!("op declares result 0 as internally allocated");
        assert!(
            bufferize_cycle_tripwire(&ordinary, &index, &space, &genome).is_ok(),
            "every other bufferize refusal stays an ordinary refusal"
        );

        let cyclic = anyhow::anyhow!(
            "{}; cannot bufferize: the cycle runs through 2 node(s): [Copy -> a | Copy -> b]",
            crate::bufferize::EXTRACTED_GRAPH_CYCLE
        );
        let err = bufferize_cycle_tripwire(&cyclic, &index, &space, &genome)
            .expect_err("a cyclic extracted graph from a sampled genome stops the search");
        let text = format!("{err:#}");
        assert!(text.contains("sampler invariant violated"), "{text}");
        assert!(
            text.contains("a <- [ClassId(\"b\")]") && text.contains("b <- []"),
            "the bail names the genome's chosen intra-component edges: {text}"
        );
        assert!(
            text.contains("the cycle runs through 2 node(s)"),
            "and carries bufferize's cycle members: {text}"
        );
    }

    /// Classes OUTSIDE every component are free: a plain DAG of
    /// candidates yields no components and full-list sampling.
    #[test]
    fn an_acyclic_candidate_graph_has_no_components() {
        let (index, space) = build(&[
            ("a", &[("prog", &[])]),
            ("b", &[("prog", &[]), ("reads_a", &["a"])]),
            ("c", &[("reads_a", &["a"]), ("reads_b", &["b"])]),
        ]);
        assert!(space.components.is_empty());
        for per_candidate in space.intra_sources.values() {
            for sources in per_candidate {
                assert!(sources.is_empty(), "no component, no intra sources");
            }
        }
        let mut seen: std::collections::BTreeSet<Vec<String>> = std::collections::BTreeSet::new();
        for seed in 0..200u64 {
            seen.insert(spelling(
                &index,
                &sample_genome(&index, &space, &mut StdRng::seed_from_u64(seed)),
            ));
        }
        assert_eq!(seen.len(), 4, "all 1x2x2 combinations stay reachable");
    }
}
