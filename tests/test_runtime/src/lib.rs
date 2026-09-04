//! TestRuntime — the tests-side runtime vocabulary (rehoming ruling
//! 2026-08-13).
//!
//! THE SPLIT this crate exists to hold: the reference runtime is
//! functional, out-of-place and view-free. Its kernel table carries only
//! `*FunctionalDps` types and `reference_allow_list()` is *derived* from
//! that table, so a mutating or view-shaped op registered there is
//! extractable but can never be selected — dead weight paid for on every
//! saturation. The op shapes that exercise `Bufferizable` / `ToDps` /
//! the bufferizer's aliasing machinery live HERE instead, and this crate
//! owns them outright: instance, DPS form, matcher and `.egg` rewrites,
//! one folder per op under [`ops`].
//!
//! It depends on NO other runtime crate for its op vocabulary — not even
//! the reference one. The 22 plain functional ops are forked here too,
//! kernels omitted. These are two different runtimes with two different
//! jobs, and their vocabularies are EXPECTED to diverge: this op list
//! will be whittled down to the shapes the bufferizer contracts actually
//! need, while the reference registry moves toward
//! canonical-layout-only. Nothing is kept in sync on purpose.
//!
//! Egglog assembly is runtime-injectable
//! (`luminal::egglog_snippet::assembled_program_for`), so beyond that
//! this crate is a MATCHER LIST, its own op folders, and fixture
//! runners. It is plan-level only — no kernels, no executor: everything
//! it asserts is a property of an `ExtractedGraph` or a `BufferIrGraph`.
//!
//! ONE EXCEPTION, as of the #420/#422 rejoin Phase 1 (2026-09-03):
//! post-saturation search is runtime-owned, so [`extractor`] and
//! [`sampler`] are THIS crate's copies of what used to be
//! `luminal::extractor` and `luminal::implementation_search`. The
//! charter above says it plainly — "anything it turns out to want from
//! another runtime gets REPLICATED here, never borrowed" — so the
//! duplication is the charter working, not a hole in it.

pub mod extractor;
pub mod sampler;

pub mod bindings;
pub mod ops;
pub mod test_equality;

pub use bindings::TestRuntimeBindings;
pub use ops::{
    AddMulFused, AddMulFusedDps, AddMulFusedMatcher, IndexMapApplyView, IndexMapApplyViewMatcher,
};

/// The cuBLASLt marker estate — REHOMED (Train 3 op-ownership move) to
/// `luminal_cuda_lite::ops::cublaslt`, its executing runtime. The board
/// keeps its `test_runtime::cublaslt_marker::…` spelling through this
/// re-export; the rule text, extractor, and election core are the SAME
/// items, never forked.
///
/// This is the ONE runtime crate the TestRuntime still reaches into, and
/// it is deliberate: the marker estate asserts on a real backend's
/// election, so forking it here would be asserting on a copy.
pub use luminal_cuda_lite::ops::cublaslt as cublaslt_marker;

use std::path::PathBuf;

use luminal::layout_ir::ExtractedGraph;
use luminal::layout_ir::OpMatcher;

/// THE TestRuntime vocabulary, every op entry owned by this crate: 22
/// forked functional ops, the metadata view op, the fused add+mul pair,
/// and the 12 mutating forms — plus the cuBLASLt markers, which stay
/// borrowed from their executing runtime (see [`cublaslt_marker`]).
/// Tests here extract and assemble against exactly this list.
pub fn matchers() -> Vec<Box<dyn OpMatcher>> {
    let mut matchers = ops::functional::functional_matchers();
    matchers.push(Box::new(IndexMapApplyViewMatcher));
    matchers.push(Box::new(AddMulFusedMatcher));
    matchers.extend(ops::mutating::mutating_matchers());
    for matcher in cublaslt_marker::all_matchers() {
        matchers.push(Box::new(matcher));
    }
    matchers
}

/// A fixture owned by THIS crate, under `fixtures/`.
///
/// Every fixture this runtime uses lives here, including forked copies of
/// scripts the reference corpus also carries. The fork is deliberate: the
/// reference copy cannot name a mutating or view constructor (an
/// undeclared constructor is an egglog PARSE failure), and the two
/// vocabularies are expected to diverge. Nothing here reaches into the
/// core script tree.
pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

/// [`extract_fixture`] on one of this crate's own fixtures, by file name.
pub fn extract_fixture_by_name(name: &str) -> ExtractedGraph {
    let source = std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|_| panic!("fixture script {name} readable"));
    extract_fixture(&source)
}

/// [`extract_fixture_by_name`] restricted to an allow-list of
/// `LayoutTensorOp` constructor names — forces extraction through
/// specific implementations so a test can pin one spelling end to end.
/// Runs over THIS runtime's matcher set.
pub fn extract_fixture_with_ops(name: &str, allowed: &[&str]) -> ExtractedGraph {
    try_extract_fixture_with_ops(name, allowed)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {name} produced no extracted graph"))
}

/// The fallible form of [`extract_fixture_with_ops`] — used to assert
/// that an unsatisfiable filter refuses, rather than silently widening.
pub fn try_extract_fixture_with_ops(
    name: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    let source = std::fs::read_to_string(fixture_path(name))
        .unwrap_or_else(|_| panic!("fixture script {name} readable"));
    try_extract_text_with_ops(&source, allowed)
}

fn try_extract_text_with_ops(
    script_text: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    let serialized = serialize_fixture(script_text);
    crate::extractor::extract_layout_ir_with_ops_and_matchers(
        &serialized,
        Some(allowed),
        matchers(),
    )
}

/// The assembled egglog program for this runtime's vocabulary, plus the
/// fixture script, run to saturation and serialized — the raw material
/// for every extraction below. Panics on any failure: these are fixtures.
pub fn serialize_fixture(script_text: &str) -> luminal::prelude::egraph_serialize::EGraph {
    use egglog::SerializeConfig;

    let preamble = luminal::egglog_snippet::assembled_program_for(&matchers());
    let program = format!("{preamble}\n\n{script_text}");
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture: {err}"));
    egraph.serialize(SerializeConfig::default()).egraph
}

/// Deterministic (min-cost) extraction of a fixture script on this
/// runtime's vocabulary.
pub fn extract_fixture(script_text: &str) -> ExtractedGraph {
    let serialized = serialize_fixture(script_text);
    crate::extractor::extract_layout_ir_with_matchers(&serialized, matchers())
        .expect("extraction succeeds")
        .expect("fixture produced no extracted graph")
}

// ---------------------------------------------------------------------------
// The round-11 ELECTION CORE was rehomed with the marker estate (Train 3:
// `luminal_cuda_lite::ops::cublaslt::election`). Semantics are identical;
// the core's vocabulary is now a parameter, and these wrappers keep the
// board's original signatures by passing THIS runtime's `matchers()`.
// ---------------------------------------------------------------------------

type ProducerOrdering<'a> =
    &'a dyn Fn(&[(String, crate::extractor::ProducerChoice)], usize) -> Vec<usize>;

/// See `luminal_cuda_lite::ops::cublaslt::election::genome_preferring` —
/// this wrapper binds the board vocabulary.
pub fn genome_preferring(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    preferences: &[&str],
) -> crate::extractor::Genome {
    adopt_genome(
        luminal_cuda_lite::ops::cublaslt::election::genome_preferring(
            egraph,
            &matchers(),
            preferences,
        ),
    )
}

/// THE COPY BOUNDARY. The election core stays with the marker estate in
/// CUDA-lite (the charter's one borrowing exception), and since Phase 1
/// of the #420/#422 rejoin that crate has its OWN extractor copy — so
/// its `Genome` is a DIFFERENT TYPE from ours with the same two fields.
/// Re-keying it is structural, total and unambiguous: `ClassId` and
/// `NodeId` come from the shared `egraph-serialize` crate, and a
/// producer choice is nothing but a pair of them.
fn adopt_genome(other: luminal_cuda_lite::extractor::Genome) -> crate::extractor::Genome {
    let mut genome = crate::extractor::Genome::default();
    for (class, choice) in other.choices {
        genome.choices.insert(class, adopt_choice(&choice));
    }
    genome
}

/// [`adopt_genome`] for one choice (see there).
fn adopt_choice(
    other: &luminal_cuda_lite::extractor::ProducerChoice,
) -> crate::extractor::ProducerChoice {
    crate::extractor::ProducerChoice {
        enode: other.enode.clone(),
        output_index: other.output_index,
    }
}

/// Re-export: the strictness-level admission predicate (rehomed core).
pub use luminal_cuda_lite::ops::cublaslt::election::level_admits;

/// The viability-aware election core (round 11) — rehomed
/// (`luminal_cuda_lite::ops::cublaslt::election::genome_with_ordering`);
/// this wrapper binds the board vocabulary and keeps the original
/// signature for the board's call sites.
pub fn genome_with_ordering(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    ordered: ProducerOrdering<'_>,
) -> crate::extractor::Genome {
    // The ordering closure speaks OUR `ProducerChoice`; the core hands
    // it ITS own (see [`adopt_genome`]). The bridge re-keys the
    // candidate list and passes the indices straight back — `ordered`
    // returns POSITIONS in the list it was given, which the re-keying
    // preserves exactly.
    let bridge = |candidates: &[(String, luminal_cuda_lite::extractor::ProducerChoice)],
                  level: usize|
     -> Vec<usize> {
        let ours: Vec<(String, crate::extractor::ProducerChoice)> = candidates
            .iter()
            .map(|(name, choice)| (name.clone(), adopt_choice(choice)))
            .collect();
        ordered(&ours, level)
    };
    adopt_genome(
        luminal_cuda_lite::ops::cublaslt::election::genome_with_ordering(
            egraph,
            &matchers(),
            &bridge,
        ),
    )
}

/// Genome-driven fixture extraction (the selection adapter's walk) plus the
/// plan fingerprint the search dedups on.
pub fn extract_fixture_with_genome(
    script_text: &str,
    preferences: &[&str],
) -> (ExtractedGraph, u64) {
    let serialized = serialize_fixture(script_text);
    let genome = genome_preferring(&serialized, preferences);
    let graph = crate::extractor::extract_layout_ir_with_genome_and_matchers(
        &serialized,
        &genome,
        matchers(),
    )
    .expect("genome extraction runs")
    .expect("genome extraction reaches the boundary");
    let fingerprint = crate::extractor::plan_fingerprint(&graph);
    (graph, fingerprint)
}
