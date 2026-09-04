//! SEARCH SUPPORT — the parts of implementation search that decide
//! nothing.
//!
//! Every runtime that searches draws genomes the same way, counts
//! refusals the same way, attributes wall-clock the same way, and
//! prints the same three-state progress report. None of that chooses
//! an implementation, so none of it is runtime-specific: it lives here
//! beside [`crate::extraction`], and each runtime's `search` module
//! imports it.
//!
//! WHAT IS NOT HERE, and why. The GA loop stays in each runtime,
//! because it must PRICE a plan in the middle of every iteration — the
//! reference runtime executes the candidate and times it, CUDA-lite
//! reads a heuristic or profiles on the device — and pricing is the
//! decision. Core would have to call back out to get it, which is the
//! seam the 2026-09-03 no-trait ruling closed. So the loop, the option
//! knobs, the outcome shape, the evaluators, the bucketed drivers and
//! CUDA-lite's finalist/lattice policy all stay runtime-local; what
//! moved here is what was byte-identical in every copy.
//!
//! (#420/#422 rejoin Phase 8, 2026-09-04. The sampler had three
//! identical copies — `luminal_reference::search`,
//! `luminal_cuda_lite::search`, `test_runtime::sampler` — and the
//! reporting plumbing two.)

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use colored::Colorize;
use egraph_serialize::ClassId;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

use crate::extraction::{Genome, ProducerChoice, SamplingSpace};

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
pub struct CaptureAwareStderr;

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
pub struct SearchProgress<W: std::io::Write> {
    out: W,
    /// The baseline has been announced.
    started: bool,
    /// Consecutive non-improving candidates since the last improvement.
    slower_since_faster: usize,
    /// A transient `Slower` line is currently on screen, unterminated.
    slower_line_visible: bool,
}

impl<W: std::io::Write> SearchProgress<W> {
    pub fn new(out: W) -> Self {
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
    pub fn start(&mut self, nanos: u128) {
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
    pub fn report(&mut self, improved: bool, nanos: u128) {
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
    pub fn finish(&mut self) {
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
    /// and stops the search (see each runtime's selection loop, and
    /// [`bufferize_cycle_tripwire`]).
    /// Genomes assembled by hand (the election boards) are of course
    /// still free to name cycles, and are still counted here.
    pub with_choice_cycles: usize,
    /// ...and how many involved a DEAD-END (an unplanned class with no
    /// candidate at all).
    pub with_dead_ends: usize,
    /// Genomes that extracted but failed bufferize / execute.
    ///
    /// UNDER DEVICE PROFILING (Phase 4) the plan-build count also
    /// carries candidates whose PREPARE step failed — NVRTC compilation,
    /// module load, staging, or the warmup execution. That is a
    /// deliberate classification (D10: *"runtimes can choose how to
    /// handle failures at different points"*): a plan the device cannot
    /// compile is an ordinary unfit candidate, indistinguishable in kind
    /// from one bufferize refused, and it must NOT fail the ladder.
    pub plan_build_refusals: usize,
    /// Failures in a TIMED trial, after the warmup already succeeded.
    pub execute_refusals: usize,
    /// Candidates whose TIMED RUN exceeded the runtime's candidate
    /// timeout, where it has one. NOT a refusal in the
    /// execute sense — nothing failed, the plan is simply too slow to
    /// finish measuring — so it is counted apart and is not part of the
    /// zero-refusal ladder acceptance.
    pub timed_out: usize,
    /// First few classified summaries, verbatim.
    pub exemplars: Vec<String>,
}

impl RefusalBreakdown {
    pub fn summary(&self) -> String {
        format!(
            "extract refusals {} (choice-cycles {}, dead-ends {}), bufferize {}, execute {}, \
             timed out {}",
            self.extract_refusals,
            self.with_choice_cycles,
            self.with_dead_ends,
            self.plan_build_refusals,
            self.execute_refusals,
            self.timed_out
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

/// Main's `early_stop_exceeded` (its `src/op.rs`), retyped from
/// `Duration` to this branch's u128 nanos: true once a candidate's mean
/// trial cost exceeds `best * factor`, i.e. the candidate has already
/// lost by at least that margin and further trials can only refine a
/// metric that is out of contention.
///
/// `luminal_reference`'s host profiler applies it at `factor = 1.0` to
/// a LOWER BOUND on the candidate's final mean, which makes the stop
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
pub fn bufferize_cycle_tripwire(
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

/// THE SAMPLER-INVARIANT UNIT BOARD (2026-09-02): components, forest
/// sampling and the mutation cycle check on hand-built candidate
/// graphs — no e-graph, no runtime, so the rule itself is under test
/// rather than a graph that happens to exercise it.
///
/// These boards are HAND-BUILT: they construct a producer index and a
/// candidate graph directly, so the sampling rule itself is under test
/// rather than a graph that happens to exercise it.
#[cfg(test)]
mod sampler_tests {
    use std::collections::BTreeMap;

    use egraph_serialize::{ClassId, NodeId};
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::extraction::{Genome, ProducerChoice, SamplingSpace, edges_have_cycle};

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
            luminal::bufferize::EXTRACTED_GRAPH_CYCLE
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
