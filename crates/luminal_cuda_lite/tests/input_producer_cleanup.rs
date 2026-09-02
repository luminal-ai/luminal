//! THE INPUT-PRODUCER CLEANUP STRATUM (Austin's ruling 2026-09-02).
//!
//! A boundary input is a LEAF: it exists at launch. Nothing in the
//! program produces it, so no producer of an input's class is ever
//! NEEDED, and none can ever be cheaper than reading the leaf. Sound
//! unions nevertheless mint such producers: the cuBLASLt marker's
//! double-transpose collapse unions `apply(apply(x, t), t)` INTO x's
//! logical class, and forward-layout rules then land a matched view op's
//! output in a LAYOUT-TENSOR class over that logical value. The
//! extractor seeds an input terminal at `heuristic_cost` 0 and a view
//! producer also costs 0, so `is_better` tied on cost and broke on
//! `plan_label` ("IndexMapApplyViewGeneric" < "Input:..."): the
//! BufferInput vanished and bufferize received a CYCLIC graph.
//!
//! The `cleanup` ruleset (preamble) marks every `LayoutTensorOp` whose
//! OUTPUT list holds an input's LEAF layout tensor — the one a
//! `BufferInputLit` binding names — with the `input-producer` relation,
//! and subsumes the generic `LayoutTensorOpLit` spelling for hygiene.
//! Subsume alone is NOT the mechanism: each runtime op's
//! `match_functional.egg` unions its own constructor into that same
//! class, and a core rule cannot name them all — so the fact is
//! enode-anchored and every consumer reads THE SAME spelling.
//!
//! BELT AND BRACES: PR #444 landed the extractor-side rule (an input
//! terminal keeps no producer row, and offers no candidate), which is
//! what actually cleared the seven marker minis' refusals. This stratum
//! is the estate half — one spelling for every consumer — plus the
//! `Extractor::new` TRIPWIRE that fires if the two halves ever disagree
//! about what a launch-time leaf is.
//!
//! The premise is the BOUND LEAF, not "any layout tensor over an input's
//! logical value": a different layout over the same input value is a
//! different arrangement of bytes that genuinely needs producing (see
//! the preamble's refutation note).
//!
//! Both halves are CPU-side: the e-graph the search reads
//! (`CudaRuntime::saturated_egraph`) and the search itself.

use std::collections::{BTreeMap, BTreeSet};

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::egraph_serialize::{ClassId, EGraph, Node};
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

/// Deterministic values (the shared example seeding discipline).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// THE 2D MARKER GRAPH: `A[4,8] · B[8,3] -> out[4,3]`, the canonical
/// form the round-11 marker matches — and therefore the form whose
/// transpose-sandwich + double-transpose collapse produced the welds.
fn marker_matmul() -> (Graph, NodeIndex, NodeIndex) {
    let mut cx = Graph::new();
    let a = cx.tensor((4usize, 8usize), DType::F32);
    let b = cx.tensor((8usize, 3usize), DType::F32);
    let _out = a.matmul(b).output();
    (cx, a.id, b.id)
}

fn child_class(egraph: &EGraph, node: &Node, index: usize) -> Option<ClassId> {
    node.children
        .get(index)
        .map(|id| egraph.nodes[id].eclass.clone())
}

/// The elements of a cons list, following `{Head}Cons` / `{Head}Nil`.
fn list_items(egraph: &EGraph, cons_op: &str, mut list: ClassId) -> Vec<ClassId> {
    let mut items = Vec::new();
    for _ in 0..64 {
        let Some(cons) = egraph
            .nodes
            .values()
            .find(|n| n.eclass == list && n.op == cons_op)
        else {
            break;
        };
        let (Some(head), Some(tail)) = (child_class(egraph, cons, 0), child_class(egraph, cons, 1))
        else {
            break;
        };
        items.push(head);
        list = tail;
    }
    items
}

/// The LAUNCH-TIME LEAVES, computed independently of the estate: the
/// layout tensors the `BufferInputLit` binding's `BufferTensorLit`s
/// name, whose logical class holds `LogicalTensorInputLit`. This is the
/// Rust mirror of the preamble's `input-leaf-layout-tensor` premise, so
/// the estate assertion is a real cross-check and not a tautology.
fn input_leaf_layout_tensors(egraph: &EGraph) -> BTreeSet<ClassId> {
    boundary_layout_tensors(egraph, "BufferInputLit")
        .into_iter()
        .filter(|layout_tensor| {
            egraph
                .nodes
                .values()
                .filter(|n| n.eclass == *layout_tensor && n.op == "LayoutTensorLit")
                .filter_map(|n| child_class(egraph, n, 0))
                .any(|logical| {
                    egraph
                        .nodes
                        .values()
                        .any(|n| n.eclass == logical && n.op == "LogicalTensorInputLit")
                })
        })
        .collect()
}

/// The `input-producer` relation, read exactly as the extractor reads it:
/// one serialized node per row whose op is the relation name and whose
/// child 0 is the marked `LayoutTensorOp` class.
fn input_producer_ops(egraph: &EGraph) -> BTreeSet<ClassId> {
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "input-producer")
        .filter_map(|n| child_class(egraph, n, 0))
        .collect()
}

/// The BufferTensor classes named by a boundary list root.
fn buffer_list(egraph: &EGraph, root_op: &str) -> BTreeSet<ClassId> {
    let mut out = BTreeSet::new();
    for node in egraph.nodes.values().filter(|n| n.op == root_op) {
        let Some(list) = child_class(egraph, node, 0) else {
            continue;
        };
        out.extend(list_items(egraph, "BufferTensorCons", list));
    }
    out
}

/// The LayoutTensor classes a boundary list's BufferTensorLits name.
fn boundary_layout_tensors(egraph: &EGraph, root_op: &str) -> BTreeSet<ClassId> {
    let buffers = buffer_list(egraph, root_op);
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "BufferTensorLit" && buffers.contains(&n.eclass))
        .filter_map(|n| child_class(egraph, n, 0))
        .collect()
}

/// Every `LayoutTensorOpLit` node's (op class, output LayoutTensor classes).
fn op_lit_outputs(egraph: &EGraph) -> Vec<(ClassId, Vec<ClassId>)> {
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LayoutTensorOpLit")
        .filter_map(|n| {
            let outs = child_class(egraph, n, 1)?;
            Some((
                n.eclass.clone(),
                list_items(egraph, "LayoutTensorCons", outs),
            ))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// (a) THE ESTATE: the fact marks exactly the input producers, and the
//     extractor's producer index no longer offers them.
// ---------------------------------------------------------------------------

#[test]
fn cleanup_marks_exactly_the_ops_that_produce_an_input_layout_tensor() {
    let (cx, _a, _b) = marker_matmul();
    let rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");

    let marked = input_producer_ops(&egraph);
    let lits = op_lit_outputs(&egraph);
    assert!(
        !lits.is_empty(),
        "the marker graph must assemble LayoutTensorOpLit terms"
    );

    // FORWARD: every op whose output list holds an input's LEAF layout
    // tensor carries the fact.
    let leaves = input_leaf_layout_tensors(&egraph);
    assert_eq!(leaves.len(), 2, "the pin binds two boundary inputs");
    let mut expected: BTreeSet<ClassId> = BTreeSet::new();
    for (op_class, outputs) in &lits {
        if outputs.iter().any(|out| leaves.contains(out)) {
            expected.insert(op_class.clone());
        }
    }
    assert!(
        !expected.is_empty(),
        "the 2D marker graph must MINT input producers (the double-transpose \
         collapse puts a view op's output into an input's own class) — if this \
         is empty the measurement premise is gone, not the bug"
    );
    assert_eq!(
        expected, marked,
        "the `input-producer` relation must mark EXACTLY the ops whose output \
         list holds a bound input leaf"
    );

    // HYGIENE: the generic spelling is subsumed on every marked class.
    for (op_class, _) in &lits {
        if !marked.contains(op_class) {
            continue;
        }
        let all_subsumed = egraph
            .nodes
            .values()
            .filter(|n| n.eclass == *op_class && n.op == "LayoutTensorOpLit")
            .all(|n| n.subsumed);
        assert!(
            all_subsumed,
            "cleanup must subsume the generic LayoutTensorOpLit spelling on a \
             marked class ({op_class:?})"
        );
    }

    // ...and subsume alone is NOT enough: the runtime constructor
    // spellings unioned into the same class survive it. That is WHY the
    // extractor reads the fact.
    let live_runtime_spellings: BTreeMap<ClassId, BTreeSet<String>> = marked
        .iter()
        .map(|op_class| {
            let ops = egraph
                .nodes
                .values()
                .filter(|n| n.eclass == *op_class && !n.subsumed)
                .filter(|n| n.op.starts_with("LayoutTensorOp"))
                .map(|n| n.op.clone())
                .collect::<BTreeSet<String>>();
            (op_class.clone(), ops)
        })
        .collect();
    println!(
        "CLEANUP marked {} op classes; live runtime spellings per class: {:?}",
        marked.len(),
        live_runtime_spellings
    );
    assert!(
        live_runtime_spellings.values().any(|ops| !ops.is_empty()),
        "subsume alone would suffice only if no runtime constructor survived on \
         a marked class; the extractor's fact read is load-bearing precisely \
         because they do"
    );
}

/// ONE LEAF NOTION (belt and braces, ruling 2026-09-02). PR #444 landed
/// the EXTRACTOR-side rule — `Extractor::new` retains no producer row for
/// a class it seeds as an input terminal — so the producer index has no
/// rows for a terminal REGARDLESS of the estate. What this stratum adds
/// is one SPELLING every consumer reads (the `input-producer` fact, so
/// the estate and any future index agree) plus the tripwire that makes a
/// disagreement loud. The check that keeps the two halves honest is
/// therefore CONTAINMENT: every class produced by a marked op must be an
/// input terminal, i.e. the fact set adds nothing the extractor's own
/// retain would not already take.
#[test]
fn marked_ops_produce_only_input_terminals() {
    let (cx, _a, _b) = marker_matmul();
    let rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");

    let marked = input_producer_ops(&egraph);
    assert!(
        !marked.is_empty(),
        "the marker graph must mint input producers"
    );
    let leaves = input_leaf_layout_tensors(&egraph);
    for (op_class, outputs) in op_lit_outputs(&egraph) {
        if !marked.contains(&op_class) {
            continue;
        }
        for out in outputs {
            assert!(
                leaves.contains(&out),
                "marked op {op_class:?} produces {out:?}, which is not an input \
                 leaf — the estate and the extractor disagree about what a \
                 launch-time leaf is"
            );
        }
    }
}

#[test]
fn the_producer_index_offers_no_producer_for_an_input_terminal() {
    let (cx, _a, _b) = marker_matmul();
    let rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");

    let index = luminal::extractor::producer_index_with_matchers(
        &egraph,
        luminal_cuda_lite::ops::cuda_matchers_with_cublaslt(),
    );

    // Holds under #444's retain alone; pinned here because the stratum
    // must never make it FALSE (marking an op whose output the extractor
    // still plans as produced would resurrect the cycle).
    let inputs = input_leaf_layout_tensors(&egraph);
    assert_eq!(inputs.len(), 2, "the pin binds two boundary inputs");
    for terminal in &inputs {
        assert!(
            !index.contains_key(terminal),
            "input terminal {terminal:?} must have NO producer row — it is a \
             launch-time leaf; rows: {:?}",
            index.get(terminal)
        );
    }

    // A class that is NOT an input keeps its rows: the program's output.
    let outputs = boundary_layout_tensors(&egraph, "BufferOutputLit");
    assert_eq!(outputs.len(), 1, "the pin binds one boundary output");
    for produced in &outputs {
        let rows = index
            .get(produced)
            .unwrap_or_else(|| panic!("output {produced:?} lost its producer rows"));
        assert!(
            !rows.is_empty(),
            "output {produced:?} must keep its producers: {rows:?}"
        );
        println!(
            "CLEANUP output class {produced:?} keeps {} producer rows: {:?}",
            rows.len(),
            rows.iter().map(|(name, _)| name).collect::<Vec<_>>()
        );
    }
}

// ---------------------------------------------------------------------------
// (b) THE WELD MEASUREMENT: no sampled genome hands bufferize a cyclic
//     graph any more.
// ---------------------------------------------------------------------------

#[test]
fn sampled_genomes_never_hand_bufferize_a_cyclic_graph() {
    const SEEDS: u64 = 20;
    let mut bufferize_refusals = 0usize;
    let mut extract_refusals = 0usize;
    let mut cycle_mentions = 0usize;
    let mut died = 0usize;

    for seed in 0..SEEDS {
        let (cx, a, b) = marker_matmul();
        let mut rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
        let data: FxHashMap<NodeIndex, TypedBuffer> =
            [(a, weights(32, 1).into()), (b, weights(24, 2).into())]
                .into_iter()
                .collect();
        let mut options = luminal::test_support::harness_search_options();
        options.seed = seed;
        match rt.search(&data, &options) {
            Ok(outcome) => {
                let breakdown = &outcome.refusal_breakdown;
                bufferize_refusals += breakdown.plan_build_refusals;
                extract_refusals += breakdown.extract_refusals;
            }
            Err(err) => {
                // A search that finds no plan at the 2x4 budget is a
                // BUDGET finding, reported not asserted (the election
                // pin runs at 12x16). Its message carries the verbatim
                // refusal strings, which is where a cycle would show.
                died += 1;
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("no candidate genome produced an executable plan"),
                    "seed {seed} died for an unexpected reason: {msg}"
                );
                cycle_mentions += msg.matches("extracted graph has a cycle").count();
            }
        }
    }

    println!(
        "CLEANUP WELD: searches={SEEDS} genomes={} budget-exhausted={died} \
         bufferize_refusals={bufferize_refusals} extract_refusals={extract_refusals} \
         cycle_mentions={cycle_mentions}",
        SEEDS * 2 * 4
    );
    assert_eq!(
        bufferize_refusals, 0,
        "no sampled genome may reach bufferize with a broken plan; the cleanup \
         stratum removes the producer rows that made the BufferInput lose its \
         own class"
    );
    assert_eq!(
        cycle_mentions, 0,
        "no refusal may name `extracted graph has a cycle`"
    );
}

// ---------------------------------------------------------------------------
// (c) NAMES SURVIVE THE STRIPPING (Austin's ruling 2026-09-02: "make it
//     not remove a named tensor, so that if the input was named that tool
//     would still work").
// ---------------------------------------------------------------------------

/// The authored names on the NAMED marker fixture. Both inputs and the
/// output carry one, so the pin covers both name-bearing spellings:
/// `LogicalTensorInputLit`'s own `LogicalId` (an input's name lives IN
/// its declaration) and the `LogicalTensorNamed` annotation a
/// `.output_named()` unions in.
const NAMED_INPUTS: [&str; 2] = ["blocks.0.attn.q_proj.weight", "hidden_states"];
const NAMED_OUTPUT: &str = "logits";

/// [`marker_matmul`]'s shapes, authored with NAMES — the same cuBLASLt
/// transpose collapse, so the same input producers are minted.
fn named_marker_matmul() -> Graph {
    let mut cx = Graph::new();
    let a = cx.named_tensor(NAMED_INPUTS[0], (4usize, 8usize), DType::F32);
    let b = cx.named_tensor(NAMED_INPUTS[1], (8usize, 3usize), DType::F32);
    let _out = a.matmul(b).output_named(NAMED_OUTPUT);
    cx
}

/// Saturate a program with an ARBITRARY schedule — the seam this pin
/// needs to measure a WITHOUT-CLEANUP baseline. It mirrors
/// `CudaRuntime::assemble_and_saturate` over the same public parts
/// (`bound_parts` + the cuBLASLt matcher vocabulary), so the ONLY
/// difference from [`CudaRuntime::saturated_egraph`] is the schedule.
fn saturate_with_schedule(cx: &Graph, schedule: &str) -> EGraph {
    let (pre_schedule, _inputs, _outputs, post_checks, _labeled) = cx
        .logical
        .bound_parts(&luminal_cuda_lite::CudaBindings)
        .expect("bound parts");
    let full = format!(
        "{}\n\n{pre_schedule}{schedule}{post_checks}",
        luminal::egglog_snippet::assembled_program_for(
            &luminal_cuda_lite::ops::cuda_matchers_with_cublaslt()
        )
    );
    let mut egraph = luminal::egglog_snippet::new_egraph();
    egraph
        .parse_and_run_program(None, &full)
        .expect("saturation");
    egraph
        .serialize(luminal::prelude::egglog::SerializeConfig::default())
        .egraph
}

/// The name a `LogicalId` class carries, read the way the renderer reads
/// it (`ClassRenderer::logical_name_from_logical`): the `LogicalIdLit`
/// node's String child, unquoted. Also reports whether that spelling is
/// subsumed.
fn logical_id_name(egraph: &EGraph, id_class: &ClassId) -> Option<(String, bool)> {
    let lit = egraph
        .nodes
        .values()
        .find(|n| n.eclass == *id_class && n.op == "LogicalIdLit")?;
    let text = lit.children.first().map(|c| egraph.nodes[c].op.clone())?;
    Some((text.trim_matches('"').to_string(), lit.subsumed))
}

/// Every name-bearing spelling: (name, carrier op, carrier class,
/// subsumed anywhere on the name's route — carrier or `LogicalIdLit`).
fn name_bearing_spellings(egraph: &EGraph) -> Vec<(String, String, ClassId, bool)> {
    egraph
        .nodes
        .values()
        .filter(|n| n.op == "LogicalTensorInputLit" || n.op == "LogicalTensorNamed")
        .filter_map(|n| {
            let id_class = child_class(egraph, n, 0)?;
            let (name, id_subsumed) = logical_id_name(egraph, &id_class)?;
            Some((
                name,
                n.op.clone(),
                n.eclass.clone(),
                n.subsumed || id_subsumed,
            ))
        })
        .collect()
}

/// The ops of every subsumed node — the measurement (d) compares across
/// the with/without-cleanup schedules. Node and class ids shift when a
/// ruleset mints relations, so OPS are the stable currency.
fn subsumed_ops(egraph: &EGraph) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in egraph.nodes.values().filter(|n| n.subsumed) {
        *counts.entry(node.op.clone()).or_default() += 1;
    }
    counts
}

/// A NAMED INPUT KEEPS ITS NAME (ruling 2026-09-02).
///
/// The cleanup stratum's job is to STRIP a boundary input's producer
/// spellings: it subsumes the generic `LayoutTensorOpLit` on every op
/// whose output list holds the input's leaf layout tensor. Names travel
/// on a DIFFERENT route — `LogicalTensorInputLit`'s own `LogicalId` for
/// an input, the `LogicalTensorNamed` annotation for a designated
/// output, plus the serialized `class_data.extra["let"]` the recorder
/// stamps on each SSA value — and every naming tool reads that route:
///
///  * LET-NAME LOOKUP (`ClassRenderer::class_let_name`, and the
///    `by_let` idiom the estate tests use to address a class without
///    ids) resolves a class to its authored `v{k}` binder.
///  * THE RENDER PATH (`ClassRenderer::logical_name_from_logical` /
///    `logical_name_from_layout_tensor`) resolves a layout tensor back
///    to the AUTHORED tensor name — which is how a plan, a refusal
///    message, or the visualizer says "blocks.0.attn.q_proj.weight"
///    instead of an opaque class id.
///
/// A cleanup rule that widened from "the op that produces the leaf" to
/// "anything in the leaf's neighbourhood" would silently retire the very
/// node those tools read, and the failure would be INVISIBLE at the
/// plan level: the search would still elect, and only the names would
/// turn into class ids. This pin therefore asserts the two routes stay
/// LIVE while the stripping happens, and — against the WITHOUT-CLEANUP
/// baseline (the same program on the same schedule minus
/// `(saturate (run cleanup))`) — that the ONLY spelling the stratum
/// newly subsumes anywhere in the e-graph is `LayoutTensorOpLit`.
#[test]
fn named_input_keeps_its_name_bearing_spellings() {
    let cx = named_marker_matmul();
    let rt = CudaRuntime::load_with_cublaslt(&cx).expect("load");
    let egraph = rt.saturated_egraph().expect("saturation");

    // ---- (a) the name-bearing spellings are present and LIVE. ----
    let spellings = name_bearing_spellings(&egraph);
    let mut by_name: BTreeMap<String, (String, ClassId, bool)> = BTreeMap::new();
    for (name, op, class, subsumed) in spellings {
        by_name.insert(name, (op, class, subsumed));
    }
    println!("NAMED spellings: {by_name:?}");
    for name in NAMED_INPUTS {
        let (op, _, subsumed) = by_name.get(name).unwrap_or_else(|| {
            panic!("named input {name:?} lost its name-bearing spelling: {by_name:?}")
        });
        assert_eq!(
            op, "LogicalTensorInputLit",
            "an input's name lives IN its declaration"
        );
        assert!(
            !subsumed,
            "the cleanup stratum must not subsume the name route of input {name:?}"
        );
    }
    let (out_op, _, out_subsumed) = by_name
        .get(NAMED_OUTPUT)
        .unwrap_or_else(|| panic!("named output {NAMED_OUTPUT:?} lost its annotation"));
    assert_eq!(out_op, "LogicalTensorNamed");
    assert!(
        !out_subsumed,
        "the cleanup stratum must not subsume the output's name annotation"
    );

    // ---- (b) the recorder's let-names are intact. ----
    // A named input is still an SSA value with a binder, and `by_let`
    // (the estate tests' id-free addressing) must still find it.
    let let_names: BTreeMap<String, ClassId> = egraph
        .class_data
        .iter()
        .filter_map(|(class, data)| Some((data.extra.get("let")?.clone(), class.clone())))
        .collect();
    for name in NAMED_INPUTS {
        let (_, class, _) = &by_name[name];
        let binder = egraph
            .class_data
            .get(class)
            .and_then(|data| data.extra.get("let"))
            .unwrap_or_else(|| {
                panic!("named input {name:?} class {class:?} lost its let-name; binders: {let_names:?}")
            });
        assert!(
            let_names.get(binder) == Some(class),
            "let-name {binder:?} must resolve back to input {name:?}'s class"
        );
        println!("NAMED input {name:?} -> let {binder:?} -> {class:?}");
    }

    // ---- (c) the cleanup DID run on this graph. ----
    // Without this the rest is vacuous: nothing was stripped, so of
    // course nothing was over-stripped.
    let marked = input_producer_ops(&egraph);
    let leaves = input_leaf_layout_tensors(&egraph);
    assert_eq!(leaves.len(), 2, "the named pin binds two boundary inputs");
    let mut expected: BTreeSet<ClassId> = BTreeSet::new();
    for (op_class, outputs) in op_lit_outputs(&egraph) {
        if outputs.iter().any(|out| leaves.contains(out)) {
            expected.insert(op_class);
        }
    }
    assert!(
        !expected.is_empty(),
        "the NAMED marker graph must mint input producers too — naming must \
         not change which welds appear, or this pin measures nothing"
    );
    assert_eq!(
        expected, marked,
        "the `input-producer` relation must mark EXACTLY the ops whose output \
         list holds a bound input leaf, names or no names"
    );

    // ---- (d) the stratum subsumes NOTHING but LayoutTensorOpLit. ----
    // Direct check first: no name-bearing or boundary spelling anywhere
    // in the e-graph is retired.
    const MUST_STAY_LIVE: [&str; 4] = [
        "LayoutTensorLit",
        "LogicalTensorInputLit",
        "LogicalTensorNamed",
        "LogicalIdLit",
    ];
    let with_cleanup = subsumed_ops(&egraph);
    for op in MUST_STAY_LIVE {
        assert!(
            !with_cleanup.contains_key(op),
            "cleanup subsumed a {op} node ({} of them) — a naming tool reads \
             that spelling; full subsumed census: {with_cleanup:?}",
            with_cleanup[op]
        );
    }

    // ...and the BASELINE delta: run the identical program on the
    // identical schedule minus the cleanup stratum, and diff.
    let without =
        luminal_cuda_lite::CudaBindings::SCHEDULE.replace("(saturate (run cleanup)) ", "");
    assert_ne!(
        without,
        luminal_cuda_lite::CudaBindings::SCHEDULE,
        "the baseline must actually drop the cleanup stratum — if the schedule \
         was reworded this pin silently stops measuring anything"
    );
    let baseline = saturate_with_schedule(&cx, &without);
    assert!(
        input_producer_ops(&baseline).is_empty(),
        "the baseline must NOT run the cleanup stratum"
    );
    let without_cleanup = subsumed_ops(&baseline);
    println!("SUBSUMED with cleanup:    {with_cleanup:?}");
    println!("SUBSUMED without cleanup: {without_cleanup:?}");

    let newly_subsumed: BTreeSet<&String> = with_cleanup
        .keys()
        .filter(|op| !without_cleanup.contains_key(*op))
        .collect();
    assert_eq!(
        newly_subsumed,
        ["LayoutTensorOpLit".to_string()].iter().collect(),
        "the cleanup stratum may retire the generic LayoutTensorOpLit spelling \
         and NOTHING else; with={with_cleanup:?} without={without_cleanup:?}"
    );
    for (op, before) in &without_cleanup {
        let after = with_cleanup.get(op).copied().unwrap_or(0);
        assert!(
            after >= *before,
            "cleanup must only ADD subsumptions, but {op} went {before} -> {after}"
        );
    }
    assert!(
        with_cleanup["LayoutTensorOpLit"] > 0,
        "the stratum must actually retire generic op spellings on this graph"
    );
}
