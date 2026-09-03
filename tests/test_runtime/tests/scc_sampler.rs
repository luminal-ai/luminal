//! THE SAMPLER-INVARIANT BOARDS on real e-graphs (ruling 2026-09-02).
//!
//! Genome sampling used to keep re-description candidates subordinate
//! by grouping index classes that instantiate the SAME LOGICAL VALUE
//! (the 2026-08-07 two-phase form, built for Copy⟷Copy layout welds).
//! It now groups them by STRONGLY CONNECTED COMPONENT of the candidate
//! graph, which is the mechanism itself: a choice cycle can only close
//! inside a component, whoever's logical value the members carry.
//!
//! Three boards here:
//!
//!  * `coverage_*` — the cross-check, in two halves. A FREE sampler
//!    (uniform over every candidate — the pre-2026-08-07 space) is run
//!    500 times and "the forest rule admits this genome" must agree
//!    EXACTLY with "its chosen-edge graph is acyclic"; then every
//!    component's candidate combinations are ENUMERATED and the set the
//!    forest sampler reaches is compared with the acyclic set. Equality
//!    is the two-sided claim: nothing cyclic survives, and nothing
//!    acyclic is eliminated.
//!
//!  * `marker_*` — the cuBLASLt double-transpose collapse on a 2D
//!    matmul, CPU-side. Its components hold members of DIFFERENT
//!    logical values, which is exactly what the same-logical-value
//!    grouping could not see.
//!
//!  * `union_*` — Austin's worked example. Inputs x, y, z; a = x + y;
//!    b = a - x; c = z + b; outputs (a, c) — then the recorded script
//!    is told what is true of it, `(union v7 v1)`: b IS y. The FINDING
//!    is recorded there: because b = y is a BOUND INPUT, and an
//!    input-terminal class is planned unconditionally, the weld is not
//!    a cycle at all and the sampler is free.
//!
//!  * `an_input_terminal_*` — the leaf rule itself (2026-09-02). The
//!    double-transpose collapse gives the marker's BOUND INPUTS
//!    producers of their own, and a zero-byte view used to win the
//!    input class on the `plan_label` tie-break — 11 of 64 sampled
//!    genomes then extracted a plan that computed an input from a value
//!    reading it back, which only bufferize's toposort caught. An input
//!    terminal is a leaf: no genome row, no candidate, and every sample
//!    reads it as a `BufferInput`.

use std::collections::{BTreeMap, BTreeSet};

use luminal::dtype::DType;
use luminal::extractor::{edges_have_cycle, ExtractionSession, Genome, SamplingSpace};
use luminal::graph::Graph;
use luminal::implementation_search::{
    mutate_genome_with_seed, sample_genome_with_seed, ProducerIndex,
};
use luminal::layout_ir::ExtractedNode;
use luminal::prelude::egraph_serialize::{ClassId, EGraph};

// ---------------------------------------------------------------------------
// Shared machinery.
// ---------------------------------------------------------------------------

/// SplitMix64 — a self-contained deterministic stream, so these boards
/// need no `rand` of their own and the free sampler is reproducible.
struct Stream(u64);

impl Stream {
    fn next(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % bound as u64) as usize
    }
}

/// THE FREE SAMPLER: uniform over every candidate of every class, with
/// no invariant at all — the space the forest rule carves out of.
fn free_genome(index: &ProducerIndex, stream: &mut Stream) -> Genome {
    let mut genome = Genome::default();
    for (class, candidates) in index {
        let position = stream.next(candidates.len());
        genome
            .choices
            .insert(class.clone(), candidates[position].1.clone());
    }
    genome
}

/// Is there an order of each component's members that makes every one
/// of this genome's chosen candidates admissible under the forest rule
/// (all its intra-component sources already assigned)? Greedy answers
/// it exactly: admissibility only ever grows as members are assigned,
/// so a member that is ever admissible stays admissible.
fn forest_admits(index: &ProducerIndex, space: &SamplingSpace, genome: &Genome) -> bool {
    for members in &space.components {
        let mut assigned: BTreeSet<ClassId> = BTreeSet::new();
        let mut remaining: Vec<ClassId> = members.clone();
        while let Some(at) = remaining.iter().position(|class| {
            let position = space
                .chosen_position(index, genome, class)
                .expect("the genome names an index entry for every component member");
            space.intra_sources[class][position]
                .iter()
                .all(|source| assigned.contains(source))
        }) {
            assigned.insert(remaining.remove(at));
        }
        if !remaining.is_empty() {
            return false;
        }
    }
    true
}

fn cyclic(index: &ProducerIndex, space: &SamplingSpace, genome: &Genome) -> bool {
    edges_have_cycle(&space.chosen_edges(index, genome))
}

/// Every combination of candidate positions for one component's
/// members, as a `Vec<usize>` parallel to `members` — small by
/// construction (re-description groups are pairs and triples), capped
/// so a pathological component cannot blow the board up.
fn component_combinations(index: &ProducerIndex, members: &[ClassId]) -> Option<Vec<Vec<usize>>> {
    let widths: Vec<usize> = members.iter().map(|class| index[class].len()).collect();
    let total: usize = widths
        .iter()
        .try_fold(1usize, |acc, w| acc.checked_mul(*w))?;
    if total > 4096 {
        return None;
    }
    let mut combos = vec![Vec::new()];
    for width in &widths {
        combos = combos
            .into_iter()
            .flat_map(|prefix| {
                (0..*width).map(move |position| {
                    let mut next = prefix.clone();
                    next.push(position);
                    next
                })
            })
            .collect();
    }
    Some(combos)
}

/// Is one component-local combination acyclic over the chosen INTRA
/// edges? (A cycle in a genome's chosen-edge graph always lies inside
/// one component: inter-component edges follow the condensation, which
/// is a DAG.)
fn combination_is_acyclic(
    space: &SamplingSpace,
    members: &[ClassId],
    combination: &[usize],
) -> bool {
    let edges: BTreeMap<ClassId, Vec<ClassId>> = members
        .iter()
        .zip(combination)
        .map(|(class, position)| (class.clone(), space.intra_sources[class][*position].clone()))
        .collect();
    !edges_have_cycle(&edges)
}

/// The component-local combination a genome elected.
fn combination_of(
    index: &ProducerIndex,
    space: &SamplingSpace,
    genome: &Genome,
    members: &[ClassId],
) -> Vec<usize> {
    members
        .iter()
        .map(|class| {
            space
                .chosen_position(index, genome, class)
                .expect("the genome names an index entry for every component member")
        })
        .collect()
}

/// The two-sided coverage claim on one e-graph — see the module note.
/// Returns `(components, exactness-checked components)`.
///
/// MEASURED 2026-09-02 on the fixtures below: every component admits
/// acyclic combinations and NO sample takes the fallback (0/2000 on
/// each board). Where a component admits no acyclic combination at
/// all, the documented full-list fallback is what runs and the
/// assertion is only that it still produces a total genome — the same
/// corner, and the same behaviour, as the 2026-08-07 sampler's
/// copy-only groups.
fn cross_check(name: &str, egraph: &EGraph) -> (usize, usize) {
    let session = ExtractionSession::new_with_matcher_set(egraph, None, test_runtime::matchers());
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    assert!(!index.is_empty(), "{name}: nothing to sample");

    let mut stream = Stream(0xC0FF_EE00);
    let (mut free_acyclic, mut free_cyclic) = (0usize, 0usize);
    for sample in 0..500 {
        let genome = free_genome(&index, &mut stream);
        let closes = cyclic(&index, &space, &genome);
        assert_eq!(
            forest_admits(&index, &space, &genome),
            !closes,
            "{name} free sample {sample}: the forest rule must admit EXACTLY the acyclic \
             genomes, and this one is {}",
            if closes { "cyclic" } else { "acyclic" }
        );
        if closes {
            free_cyclic += 1;
        } else {
            free_acyclic += 1;
        }
    }

    let mut fell_back = 0usize;
    let mut mutated = 0usize;
    let mut sampled: Vec<BTreeSet<Vec<usize>>> = vec![BTreeSet::new(); space.components.len()];
    // THE E[copies] MEASUREMENT (not re-tuned here, ruling 2026-09-02 —
    // just measured): across every component member of every sample,
    // how often does the sampler elect a RE-DESCRIBING candidate (one
    // with intra-component sources) rather than a progressing one? The
    // 2026-08-07 star fixed this at one primary per group; the forest
    // lets it float, so it is reported, never asserted.
    let (mut member_choices, mut intra_choices) = (0usize, 0usize);
    for seed in 0..2000u64 {
        let (genome, fallbacks) = sample_genome_with_seed(&index, &space, seed);
        assert_eq!(
            genome.choices.len(),
            index.len(),
            "{name} seed {seed}: the genome must stay total"
        );
        if fallbacks.is_empty() {
            assert!(
                !cyclic(&index, &space, &genome),
                "{name} seed {seed}: a fallback-free sample closed a cycle"
            );
        } else {
            fell_back += 1;
        }
        for (component, members) in space.components.iter().enumerate() {
            let combination = combination_of(&index, &space, &genome, members);
            for (class, position) in members.iter().zip(&combination) {
                member_choices += 1;
                intra_choices += usize::from(!space.intra_sources[class][*position].is_empty());
            }
            sampled[component].insert(combination);
        }
        // MUTATION under the same invariant: two point mutations off
        // this genome, and the child may not close a cycle either
        // (unless the flip itself had to fall back).
        let (child, mutation_fallbacks) =
            mutate_genome_with_seed(&genome, &index, &space, 2, seed ^ 0x5EED);
        if fallbacks.is_empty() && mutation_fallbacks.is_empty() {
            assert!(
                !cyclic(&index, &space, &child),
                "{name} seed {seed}: a fallback-free mutation closed a cycle"
            );
        }
        mutated += usize::from(child.choices != genome.choices);
    }
    assert!(
        mutated > 0,
        "{name}: mutation must actually move the genome somewhere"
    );

    let mut enumerated = 0usize;
    let mut checked_exactly = 0usize;
    for (component, members) in space.components.iter().enumerate() {
        let Some(combos) = component_combinations(&index, members) else {
            println!("SCC-COVERAGE {name}: component {component} too wide to enumerate");
            continue;
        };
        enumerated += 1;
        let acyclic: BTreeSet<Vec<usize>> = combos
            .iter()
            .filter(|combination| combination_is_acyclic(&space, members, combination))
            .cloned()
            .collect();
        println!(
            "SCC-COVERAGE {name}: component {component} {members:?} — {} combinations, \
             {} acyclic, {} sampled",
            combos.len(),
            acyclic.len(),
            sampled[component].len()
        );
        if acyclic.is_empty() {
            assert!(
                !sampled[component].is_empty(),
                "{name} component {component}: no acyclic combination exists, so the \
                 documented fallback must still produce one"
            );
            continue;
        }
        // SOUNDNESS, always: nothing the sampler reaches may be cyclic.
        assert!(
            sampled[component].is_subset(&acyclic),
            "{name} component {component}: the forest sampler reached a CYCLIC \
             combination"
        );
        // COMPLETENESS, where 2000 seeds can plausibly cover the space:
        // nothing acyclic may be unreachable. A component with more
        // acyclic combinations than that gets the soundness half only —
        // its coverage is the enumerated small components' business.
        if acyclic.len() <= 128 {
            checked_exactly += 1;
            assert_eq!(
                sampled[component], acyclic,
                "{name} component {component}: the forest sampler must reach EXACTLY \
                 the acyclic combinations — no acyclic shape may be unreachable"
            );
        }
    }

    println!(
        "SCC-COVERAGE {name}: {} genome classes, {} components ({enumerated} enumerated, \
         {checked_exactly} exactness-checked), free samples {free_acyclic} acyclic / \
         {free_cyclic} cyclic, forest samples with a fallback {fell_back}/2000, \
         mutations that moved {mutated}/2000, re-describing choices \
         {intra_choices}/{member_choices}",
        index.len(),
        space.components.len()
    );
    (space.components.len(), checked_exactly)
}

// ---------------------------------------------------------------------------
// B. Coverage cross-check on real fixture e-graphs.
// ---------------------------------------------------------------------------

#[test]
fn coverage_on_the_basic_program_fixture() {
    let source = std::fs::read_to_string(test_runtime::fixture_path("basic_program.egg"))
        .expect("fixture readable");
    let egraph = test_runtime::serialize_fixture(&source);
    let (components, checked_exactly) = cross_check("basic_program", &egraph);
    assert_eq!(
        components, 2,
        "the fixture's layout copies weld into two components — the 2026-08-07 \
         Copy⟷Copy shape, found now by the component criterion rather than by \
         same-logical-value grouping"
    );
    assert_eq!(
        checked_exactly, 2,
        "both components must be small enough to enumerate AND offer acyclic \
         combinations, or the exactness half of the cross-check is vacuous here"
    );
}

#[test]
fn coverage_on_the_re_description_program() {
    let egraph = test_runtime::serialize_fixture(&re_description_program());
    let (components, _) = cross_check("b_is_y_union", &egraph);
    assert_eq!(
        components, 0,
        "b = y welds through a BOUND INPUT, which is a leaf — see \
         `union_of_b_and_y_is_a_leaf_weld_not_a_cycle`"
    );
}

/// THE CROSS-LOGICAL CASE, CPU-side: the cuBLASLt marker estate's
/// double-transpose collapse on the 2D canonical matmul. Its components
/// hold members instantiating DIFFERENT logical values — invisible to
/// the 2026-08-07 same-logical-value grouping, which saw both arms of
/// each weld as "progressing" and let genomes elect them together.
#[test]
fn marker_matmul_components_span_different_logical_values() {
    let egraph = test_runtime::serialize_fixture(&marker_matmul_program());
    let session = ExtractionSession::new_with_matcher_set(&egraph, None, test_runtime::matchers());
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    assert!(
        !space.components.is_empty(),
        "the collapse must weld re-description components"
    );

    let logical_of = |class: &ClassId| -> BTreeSet<String> {
        egraph
            .nodes
            .values()
            .filter(|node| &node.eclass == class && node.op == "LayoutTensorLit")
            .map(|node| egraph.nodes[&node.children[0]].eclass.to_string())
            .collect()
    };
    let mut cross_logical = 0usize;
    for members in &space.components {
        let logicals: Vec<BTreeSet<String>> = members.iter().map(&logical_of).collect();
        println!("SCC-MARKER component {members:?} logical values {logicals:?}");
        let spans = logicals
            .iter()
            .enumerate()
            .any(|(i, a)| logicals[i + 1..].iter().any(|b| a.is_disjoint(b)));
        cross_logical += usize::from(spans);
        // Every member must have somewhere to go, or the sampler is
        // forced into the fallback on a graph that has a way out.
        for member in members {
            assert!(
                space.intra_sources[member]
                    .iter()
                    .any(|sources| sources.is_empty()),
                "{member} has no progressing candidate in its own component"
            );
        }
    }
    assert!(
        cross_logical > 0,
        "at least one component must span two DIFFERENT logical values — that is \
         the case the same-logical-value grouping could not group"
    );
    cross_check("marker_matmul_2d", &egraph);
}

/// The 2D canonical matmul the round-11 marker matches: A[4,8] . B[8,3].
fn marker_matmul_program() -> String {
    let mut cx = Graph::new();
    let a = cx.tensor((4usize, 8usize), DType::F32);
    let b = cx.tensor((8usize, 3usize), DType::F32);
    let _out = a.matmul(b).output();
    cx.logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text
}

// ---------------------------------------------------------------------------
// C. Austin's example: b = y, told to the e-graph as a union.
// ---------------------------------------------------------------------------

/// x, y, z inputs; a = x + y; b = a - x; c = z + b; outputs (a, c) —
/// recorded through the TestRuntime bindings, then handed the fact the
/// recorder cannot know: `b` and `y` are the same value.
///
/// The recorder's SSA names are read off the emitted text the way
/// `cublaslt_bias_leftmajor_premise.rs` reads `natout{key}`: the
/// recorder names every value `v{node index}`, so `y` is `v1` and `b`
/// is `v7` = `(LogicalAdd v3 v6)`, a + (-1) * x. The union goes in
/// BEFORE the schedule so saturation sees it.
fn re_description_program() -> String {
    let mut cx = Graph::new();
    let x = cx.tensor(4usize, DType::F32);
    let y = cx.tensor(4usize, DType::F32);
    let z = cx.tensor(4usize, DType::F32);
    let a = x + y;
    let b = a - x;
    let c = z + b;
    let _ = a.output();
    let _ = c.output();
    let text = cx
        .logical
        .bound_program(&test_runtime::TestRuntimeBindings)
        .expect("recorder clean")
        .text;
    let b_name = format!("v{}", b.id.index());
    let y_name = format!("v{}", y.id.index());
    assert!(
        text.contains(&format!("(let {b_name} (LogicalAdd ")),
        "b is the recorder's {b_name}: {text}"
    );
    assert!(
        text.contains(&format!("(let {y_name} (LogicalTensorInputLit ")),
        "y is the recorder's {y_name}: {text}"
    );
    let at = text
        .find("(run-schedule")
        .expect("the recorder emits a schedule");
    format!(
        "{}\n; b IS y — the fact the recorder cannot know.\n(union {b_name} {y_name})\n\n{}",
        &text[..at],
        &text[at..]
    )
}

/// THE FINDING, recorded rather than wished away: after the union the
/// candidate graph has NO cycle at all, so there is no component and
/// the sampler is unconstrained here.
///
/// The weld is real — class(a) reads class(b = y) and class(b = y)'s
/// `a + (-1) * x` producer reads class(a) — but `b = y`'s class is a
/// BOUND INPUT: it is produced by nothing, so it holds no genome row,
/// offers no candidate, and is planned from the boundary on
/// `relax_to_fixpoint`'s first pass. It is not a node of the candidate
/// graph at all, and a class that demands nothing contracts to nothing
/// for its readers either, so no edge and no cycle. The cross-logical
/// re-description that DOES make a component is on the `marker_*`
/// board, where neither member is a boundary value.
#[test]
fn union_of_b_and_y_is_a_leaf_weld_not_a_cycle() {
    let egraph = test_runtime::serialize_fixture(&re_description_program());
    let session = ExtractionSession::new_with_matcher_set(&egraph, None, test_runtime::matchers());
    let index = session.producer_index();
    let space = session.sampling_space(&index);

    assert!(
        space.components.is_empty(),
        "the weld runs through a bound input, which is a leaf: {:?}",
        space.components
    );
    for per_candidate in space.intra_sources.values() {
        for sources in per_candidate {
            assert!(
                sources.is_empty(),
                "no component, no intra-component sources"
            );
        }
    }
    // The union itself must still be there, or this board proves
    // nothing: some LOGICAL class holds both the boundary input `y` and
    // the computed `b`, which is what `(union v7 v1)` asserts.
    let mut welded: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    for node in egraph.nodes.values() {
        welded
            .entry(node.eclass.to_string())
            .or_default()
            .insert(node.op.as_str());
    }
    assert!(
        welded
            .values()
            .any(|ops| ops.contains("LogicalTensorInputLit") && ops.contains("LogicalAdd")),
        "b = y must put the input and the `a + (-1) * x` add in ONE logical class — \
         otherwise the union never reached the e-graph"
    );
}

/// THE ACCEPTANCE: sampling this program 200 ways never produces a
/// choice cycle, and `a` is always computed from x and y.
///
/// ROUTES, honestly: only ONE of the two routes for `c` is electable,
/// and the reason is not the sampler. `b = y`'s class is a bound INPUT,
/// so it is a LEAF — no genome row, no candidate, planned from the
/// boundary at `heuristic_cost` 0 whatever the genome says. The
/// recomputed `a - x` is not something `is_better` could prefer; it is
/// not offered at all. Every plan here is therefore two adds — `a` from
/// x and y, and `c` from z and the same class — with `b`'s -1
/// constant/mul chain never planned.
#[test]
fn union_extraction_never_cycles() {
    let egraph = test_runtime::serialize_fixture(&re_description_program());
    let mut session =
        ExtractionSession::new_with_matcher_set(&egraph, None, test_runtime::matchers());
    let index = session.producer_index();
    let space = session.sampling_space(&index);

    let mut shapes: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    for seed in 0..200u64 {
        let (genome, _fallbacks) = sample_genome_with_seed(&index, &space, seed);
        match session.extract_with_genome(&genome) {
            Ok(Some(graph)) => {
                let mut labels: Vec<String> = graph
                    .dag
                    .node_weights()
                    .filter_map(|node| match node {
                        ExtractedNode::LayoutOp(op) => Some(op.op.label().to_string()),
                        _ => None,
                    })
                    .collect();
                labels.sort();
                // `a` IS computed from x and y: all three bound inputs
                // are read, so neither add reads a recomputed `b`
                // (which would need the -1 constant/mul chain, and
                // more ops than the label check below allows).
                let inputs = graph
                    .dag
                    .node_weights()
                    .filter(|node| matches!(node, ExtractedNode::BufferInput(_)))
                    .count();
                assert_eq!(
                    inputs, 3,
                    "seed {seed}: every plan reads x, y and z — got {inputs} inputs \
                     alongside {labels:?}"
                );
                *shapes.entry(labels).or_default() += 1;
            }
            Ok(None) => panic!("seed {seed}: extraction reached no boundary"),
            Err(err) => {
                let (cycle, dead_end, summary) = session.failure_breakdown();
                panic!(
                    "seed {seed}: extraction refused (choice-cycle {cycle}, dead-end \
                     {dead_end}; {summary}): {err:#}"
                );
            }
        }
    }
    for (labels, count) in &shapes {
        println!("SCC-UNION plan x{count}: {labels:?}");
    }
    assert!(shapes.len() > 1, "the sampler explores more than one plan");
    for labels in shapes.keys() {
        assert_eq!(
            labels.len(),
            2,
            "a = x + y and c = z + (b = y): two adds, no recompute of b — got {labels:?}"
        );
        for label in labels {
            assert!(
                label.starts_with("Add"),
                "both ops are adds (b's -1 constant/mul chain is never planned): {labels:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// D. AN INPUT TERMINAL IS A LEAF: no genome row, no producer, no cycle.
// ---------------------------------------------------------------------------

/// The extractor's own input-terminal rule replicated on the raw
/// e-graph (`collect_input_terminals`): the LayoutTensor child of every
/// `BufferTensorLit` whose BufferTensor class is one of the program's
/// bound inputs.
fn input_terminal_classes(egraph: &EGraph) -> BTreeSet<ClassId> {
    let buffer_list = |root_op: &str| -> BTreeSet<ClassId> {
        let mut out = BTreeSet::new();
        for node in egraph.nodes.values().filter(|node| node.op == root_op) {
            let mut cursor = node
                .children
                .first()
                .and_then(|id| egraph.nodes.get(id))
                .map(|child| child.eclass.clone());
            while let Some(list) = cursor {
                let Some(cons) = egraph
                    .nodes
                    .values()
                    .find(|node| node.eclass == list && node.op == "BufferTensorCons")
                else {
                    break;
                };
                if let Some(head) = cons.children.first().and_then(|id| egraph.nodes.get(id)) {
                    out.insert(head.eclass.clone());
                }
                cursor = cons
                    .children
                    .get(1)
                    .and_then(|id| egraph.nodes.get(id))
                    .map(|child| child.eclass.clone());
            }
        }
        out
    };
    let inputs = buffer_list("BufferInputLit");
    let outputs = buffer_list("BufferOutputLit");
    let mut terminals = BTreeSet::new();
    for node in egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferTensorLit")
    {
        if if inputs.is_empty() {
            outputs.contains(&node.eclass)
        } else {
            !inputs.contains(&node.eclass)
        } {
            continue;
        }
        if let Some(tensor) = node.children.first().and_then(|id| egraph.nodes.get(id)) {
            terminals.insert(tensor.eclass.clone());
        }
    }
    terminals
}

/// The implementation op names that PRODUCE `class` in the e-graph: for
/// every `LayoutTensorOpLit` spec whose output list contains it, the
/// `LayoutTensorOp*` spellings of the op class it belongs to. This is
/// the board's non-vacuity witness — the rows the producer index used
/// to carry for an input terminal.
fn producer_op_names(egraph: &EGraph, class: &ClassId) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for node in egraph
        .nodes
        .values()
        .filter(|node| node.op == "LayoutTensorOpLit")
    {
        let Some(outputs) = node.children.get(1).and_then(|id| egraph.nodes.get(id)) else {
            continue;
        };
        let mut cursor = Some(outputs.eclass.clone());
        let mut produces = false;
        while let Some(list) = cursor {
            let Some(cons) = egraph
                .nodes
                .values()
                .find(|node| node.eclass == list && node.op == "LayoutTensorCons")
            else {
                break;
            };
            if let Some(head) = cons.children.first().and_then(|id| egraph.nodes.get(id)) {
                produces |= &head.eclass == class;
            }
            cursor = cons
                .children
                .get(1)
                .and_then(|id| egraph.nodes.get(id))
                .map(|child| child.eclass.clone());
        }
        if !produces {
            continue;
        }
        for sibling in egraph
            .nodes
            .values()
            .filter(|sibling| sibling.eclass == node.eclass && !sibling.subsumed)
        {
            if sibling.op.starts_with("LayoutTensorOp") && sibling.op != "LayoutTensorOpLit" {
                names.insert(sibling.op.clone());
            }
        }
    }
    names
}

/// THE GAP-A BOARD (2026-09-02). The double-transpose collapse gives
/// the marker matmul's bound INPUTS producers of their own — `x` is in
/// the class of `Tᵀ(T(x))`, whose producer is a view. A view moves no
/// bytes, so its plan ties the input plan at `heuristic_cost` 0 and the
/// `plan_label` tie-break ("IndexMapApplyViewGeneric" sorts before
/// "Input:…") HANDED THE INPUT CLASS TO THE VIEW: the extractor emitted
/// a plan in which a program input is computed from a value that reads
/// it back, and only bufferize's toposort noticed.
///
/// An input terminal is a leaf by definition — its value exists at
/// launch. So: no genome row for it (the producer index drops them), no
/// candidate for it (`candidates_for_class` returns none), and every
/// sampled genome extracts a graph that reads it as a `BufferInput` and
/// bufferizes.
#[test]
fn an_input_terminal_keeps_its_buffer_input_and_never_a_producer() {
    let egraph = test_runtime::serialize_fixture(&marker_matmul_program());
    let terminals = input_terminal_classes(&egraph);
    assert_eq!(
        terminals.len(),
        2,
        "the 2D matmul binds two inputs: {terminals:?}"
    );

    // NON-VACUITY: the collapse really does offer these classes producers.
    let mut with_producers = 0usize;
    for terminal in &terminals {
        let names = producer_op_names(&egraph, terminal);
        println!("SCC-TERMINAL {terminal} producers {names:?}");
        with_producers += usize::from(!names.is_empty());
    }
    assert!(
        with_producers > 0,
        "the board is vacuous unless some input terminal has a producer spelling"
    );

    let mut session =
        ExtractionSession::new_with_matcher_set(&egraph, None, test_runtime::matchers());
    let index = session.producer_index();
    for terminal in &terminals {
        assert!(
            !index.contains_key(terminal),
            "{terminal} is an input terminal: the producer index must carry NO genome \
             row for it (a row is dead weight the extractor must never honour)"
        );
    }
    let space = session.sampling_space(&index);

    for seed in 0..64u64 {
        let (genome, _fallbacks) = sample_genome_with_seed(&index, &space, seed);
        let graph = match session.extract_with_genome(&genome) {
            Ok(Some(graph)) => graph,
            Ok(None) => panic!("seed {seed}: extraction reached no boundary"),
            Err(err) => {
                let (cycle, dead_end, summary) = session.failure_breakdown();
                panic!(
                    "seed {seed}: extraction refused (choice-cycle {cycle}, dead-end \
                     {dead_end}; {summary}): {err:#}"
                );
            }
        };
        let read_as_inputs: BTreeSet<ClassId> = graph
            .dag
            .node_weights()
            .filter_map(|node| match node {
                ExtractedNode::BufferInput(input) => Some(input.value.eclass.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            read_as_inputs, terminals,
            "seed {seed}: every bound input must reach the plan as a BufferInput, never \
             as the result of a producer"
        );
        // AND THE PLAN IS ACYCLIC: the cycle this bug produced was only
        // ever caught here, by bufferize's toposort.
        let dps = luminal::dps::dps_rewrite(&graph);
        luminal::test_support::bufferize_mock(&dps)
            .unwrap_or_else(|err| panic!("seed {seed}: bufferize refused the plan: {err:#}"));
    }
}
