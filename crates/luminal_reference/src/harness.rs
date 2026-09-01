//! The reference-flavored test/run harness — the fixture path (real
//! egglog scripts through the real extractor) and the `run_reference`
//! ladder every model crate drives fidelity through. Moved from core
//! `test_support` in Step B (ruling 2026-08-17): these helpers default
//! the matcher set to the reference registry, which core no longer owns.
//! Core keeps only the runtime-neutral pieces (`TestGraph`, the mocks,
//! `harness_search_options`).

use std::fs;

use luminal::extractor;
use luminal::layout_ir::ExtractedGraph;

/// Fixture scripts live in the WORKSPACE-ROOT Egglog core tree; this crate
/// runs two directories below it.
fn fixture_path(script: &str) -> String {
    format!(
        "{}/../../src/egglog_core/test_scripts/{script}",
        env!("CARGO_MANIFEST_DIR")
    )
}

// =============================================================================
// The reference-defaulted extraction conveniences (were free functions in
// core `extractor`; now wrappers over its `_with_matchers` seams).
// =============================================================================

/// Deterministic (min-cost, tie-broken) extraction over the reference
/// registry — tooling for fixtures and goldens, not the selection
/// mechanism. The search path is [`extract_layout_ir_with_genome`].
pub fn extract_layout_ir(
    egraph: &egraph_serialize::EGraph,
) -> anyhow::Result<Option<ExtractedGraph>> {
    extractor::extract_layout_ir_with_matchers(egraph, crate::ops::built_in_matchers())
}

/// Genome-driven extraction (the selection adapter's walk) over the
/// reference registry.
pub fn extract_layout_ir_with_genome(
    egraph: &egraph_serialize::EGraph,
    genome: &extractor::Genome,
) -> anyhow::Result<Option<ExtractedGraph>> {
    extractor::extract_layout_ir_with_genome_and_matchers(
        egraph,
        genome,
        crate::ops::built_in_matchers(),
    )
}

/// Every LayoutTensor class's candidate producers over the reference
/// registry, optionally restricted to an implementation allow-list —
/// the raw material genome construction and mutation draw from.
pub fn producer_index_with_ops(
    egraph: &egraph_serialize::EGraph,
    allowed_ops: Option<&[&str]>,
) -> std::collections::BTreeMap<egraph_serialize::ClassId, Vec<(String, extractor::ProducerChoice)>>
{
    extractor::ExtractionSession::new_with_matcher_set(
        egraph,
        allowed_ops,
        crate::ops::built_in_matchers(),
    )
    .producer_index()
}

// =============================================================================
// Fixture path: real egglog scripts through the real extractor
// =============================================================================

/// Run `test_scripts/<script>` through egglog (with the full preamble) and
/// hand back the serialized e-graph — the selection tooling's raw material.
pub fn serialize_fixture(script: &str) -> egraph_serialize::EGraph {
    use egglog::SerializeConfig;

    let preamble = crate::assembled_program();
    let source = fs::read_to_string(fixture_path(script))
        .unwrap_or_else(|_| panic!("fixture script {script} readable"));
    let program = format!("{preamble}\n\n{source}");

    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(Some(script.to_string()), &program)
        .unwrap_or_else(|err| panic!("egglog failed on fixture {script}: {err}"));
    egraph.serialize(SerializeConfig::default()).egraph
}

/// Build a TOTAL genome over a fixture's produced classes: each class takes
/// the first preference (an implementation constructor name) it can satisfy,
/// falling back to its first candidate — the producer index is
/// deterministically sorted, so the same preferences always build the same
/// genome.
pub fn genome_preferring(
    egraph: &egraph_serialize::EGraph,
    preferences: &[&str],
) -> extractor::Genome {
    let index = producer_index_with_ops(egraph, None);
    let mut genome = extractor::Genome::default();
    for (class, candidates) in index {
        let pick = preferences
            .iter()
            .find_map(|preferred| {
                candidates
                    .iter()
                    .find(|(name, _)| name.as_str() == *preferred)
            })
            .or_else(|| candidates.first())
            .expect("produced classes have candidates");
        genome.choices.insert(class, pick.1.clone());
    }
    genome
}

/// Genome-driven fixture extraction (the selection adapter's walk) plus the
/// plan fingerprint the search dedups on.
pub fn extract_fixture_with_genome(script: &str, preferences: &[&str]) -> (ExtractedGraph, u64) {
    let egraph = serialize_fixture(script);
    let genome = genome_preferring(&egraph, preferences);
    let graph = extract_layout_ir_with_genome(&egraph, &genome)
        .expect("genome extraction runs")
        .expect("genome extraction reaches the boundary");
    let fingerprint = extractor::plan_fingerprint(&graph);
    (graph, fingerprint)
}

// The TESTRUNTIME vocabulary used to be assembled HERE, as the reference
// registry plus the op variants the reference runtime does not implement.
// It moved out with those variants (runtime-split, PR #425): the test
// runtime now owns its whole op set in its own crate — `test_runtime::
// matchers()` — and this harness is once again about the reference
// runtime alone. Nothing was lost: the two entry points that lived here
// (`test_runtime_matchers`, `extract_fixture_on_test_runtime`) are
// `test_runtime::matchers` and `test_runtime::extract_fixture_by_name`.

/// Run `test_scripts/<script>` through egglog (with the full preamble) and the
/// real extractor, returning the extracted graph. Panics on any failure — these
/// are test fixtures.
pub fn extract_fixture(script: &str) -> ExtractedGraph {
    let serialized = serialize_fixture(script);
    extract_layout_ir(&serialized)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {script} produced no extracted graph"))
}

/// [`extract_fixture`] restricted to an allow-list of LayoutTensorOp
/// constructor names — forces extraction through specific implementations so
/// a test can exercise one op end to end. Panics if the program is not
/// implementable within the list; use [`try_extract_fixture_with_ops`] to
/// assert that failure itself.
pub fn extract_fixture_with_ops(script: &str, allowed: &[&str]) -> ExtractedGraph {
    try_extract_fixture_with_ops(script, allowed)
        .expect("extraction succeeds")
        .unwrap_or_else(|| panic!("fixture {script} produced no extracted graph"))
}

/// The fallible form of [`extract_fixture_with_ops`].
pub fn try_extract_fixture_with_ops(
    script: &str,
    allowed: &[&str],
) -> anyhow::Result<Option<ExtractedGraph>> {
    let serialized = serialize_fixture(script);
    extractor::extract_layout_ir_with_ops_and_matchers(
        &serialized,
        Some(allowed),
        crate::ops::built_in_matchers(),
    )
}

// =============================================================================
// The run_reference ladder: load → bind → search → execute
// =============================================================================

/// DIAGNOSIS-ONLY (test-extractor exemption): does the plain no-genome
/// extraction produce an executable plan for this graph? Separates "no
/// plan exists" (structural gap) from "random genomes cannot find one"
/// (genome-space density). Never used by the main path — everything real
/// runs the genetic search.
pub fn plain_plan_exists(cx: &luminal::graph::Graph) -> anyhow::Result<()> {
    let program = cx
        .logical
        .bound_program(&crate::bindings::ReferenceBindings)
        .map_err(|reason| anyhow::anyhow!("recorder: {reason}"))?;
    let text = format!("{}\n\n{}", crate::assembled_program(), program.text);
    let mut egraph = luminal::egglog_snippet::new_egraph();
    let start = std::time::Instant::now();
    egraph
        .parse_and_run_program(None, &text)
        .map_err(|err| anyhow::anyhow!("saturation: {err}"))?;
    eprintln!("[plain-plan] saturation {:?}", start.elapsed());
    let start = std::time::Instant::now();
    let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;
    eprintln!(
        "[plain-plan] serialize {:?} ({} nodes, {} classes)",
        start.elapsed(),
        serialized.nodes.len(),
        serialized.classes().len()
    );
    let allow = crate::runtime::reference_allow_list();
    let start = std::time::Instant::now();
    let extracted = extractor::extract_layout_ir_with_ops_and_matchers(
        &serialized,
        Some(&allow),
        crate::ops::built_in_matchers(),
    )?
    .ok_or_else(|| anyhow::anyhow!("no output boundary reached"))?;
    eprintln!("[plain-plan] extract {:?}", start.elapsed());
    let start = std::time::Instant::now();
    // VALUE-keyed table: render over the POST-DPS graph bufferize sees.
    let dps = luminal::dps::dps_rewrite(&extracted);
    let layouts = extractor::rendered_layout_table(
        &serialized,
        &dps,
        &crate::layouts::ReferenceLayoutRenderer,
        &mut std::collections::HashMap::new(),
    )?;
    luminal::bufferize::bufferize(&dps, &layouts)?;
    eprintln!("[plain-plan] dps+bufferize {:?}", start.elapsed());
    Ok(())
}

/// M3 Step 4b: the NATIVE test harness — recorder model + reference
/// binding + dyn pins as tight [n,n] bounds seeds, saturated, then the
/// GENETIC IMPLEMENTATION SEARCH picks the winning plan (executing every
/// candidate with the given data), which executes and stays loaded for
/// output reads. The frontend candle differentials and the reference
/// differentials both run through here — the same load → bind → search →
/// execute ladder as the nn module tests, on the harness budget
/// (`luminal::test_support::harness_search_options`).
pub fn run_reference(
    cx: &luminal::graph::Graph,
    inputs: &[(
        petgraph::graph::NodeIndex,
        luminal::buffer_tensor_ir::TypedBuffer,
    )],
) -> crate::runtime::ReferenceRuntime {
    run_reference_with_ranges(cx, inputs, &[])
}

/// [`run_reference`] with VALUE-RANGE ATTESTATIONS (typed-buffers landing D):
/// plain Int arithmetic is proof-gated, so a graph doing arithmetic
/// over caller Int data implements only when the caller attests the
/// data's range — no attestation, no proof, and the search refuses
/// loudly. `ranges` entries are (tensor, lower, upper), seeded via
/// `bind_value_range` between load and search.
pub fn run_reference_with_ranges(
    cx: &luminal::graph::Graph,
    inputs: &[(
        petgraph::graph::NodeIndex,
        luminal::buffer_tensor_ir::TypedBuffer,
    )],
    ranges: &[(petgraph::graph::NodeIndex, i64, i64)],
) -> crate::runtime::ReferenceRuntime {
    let mut rt =
        crate::runtime::ReferenceRuntime::load(cx).expect("recorder clean for a covered graph");
    let mut vars: Vec<_> = cx.dyn_map.iter().collect();
    vars.sort();
    for (var, value) in vars {
        rt.bind_dyn_range(*var, *value as u64, *value as u64)
            .expect("dyn pin binds");
    }
    for (tensor, lower, upper) in ranges {
        rt.bind_value_range(*tensor, *lower, *upper)
            .expect("value range binds");
    }
    let data: rustc_hash::FxHashMap<_, _> = inputs.iter().cloned().collect();
    rt.search(&data, &luminal::test_support::harness_search_options())
        .expect("search finds a plan");
    for (node, values) in inputs {
        rt.set_data(*node, values.clone());
    }
    rt.execute().expect("winner executes");
    rt
}
