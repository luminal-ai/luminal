//! The round-11 ELECTION CORE, rehomed with the marker estate (Train 3
//! op-ownership move): viability-aware genome election over a serialized
//! e-graph — the deterministic test-side walk the cuBLASLt marker board
//! elects with. Semantics are IDENTICAL to the test_runtime original;
//! the only delta of the move is that the runtime VOCABULARY (the
//! matcher list the producer index is built over) is now a parameter —
//! the core serves any runtime's matcher set, and `test_runtime` passes
//! its own board vocabulary through thin wrappers with the original
//! signatures.
//!
//! The REAL search needs none of this: sampled genomes that reach no
//! plan are simply unfit; only deterministic test helpers must land on a
//! viable plan first try.

use luminal::layout_ir::OpMatcher;

type ProducerOrdering<'a> =
    &'a dyn Fn(&[(String, crate::extractor::ProducerChoice)], usize) -> Vec<usize>;

/// Build a TOTAL genome over a fixture's produced classes: each class takes
/// the first preference (an implementation constructor name) it can satisfy,
/// falling back to its first candidate — the producer index is
/// deterministically sorted, so the same preferences always build the same
/// genome. (Adapted from `luminal_reference::harness::genome_preferring`, which
/// is public but hard-wired to the built-in producer index; this one runs
/// over the CALLER's matcher set via the genome seam.)
pub fn genome_preferring(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    matchers: &[Box<dyn OpMatcher>],
    preferences: &[&str],
) -> crate::extractor::Genome {
    let preferences: Vec<String> = preferences.iter().map(|s| s.to_string()).collect();
    let ordered =
        |candidates: &[(String, crate::extractor::ProducerChoice)], level: usize| -> Vec<usize> {
            let admitted = level_admits(level);
            let mut order: Vec<usize> = Vec::new();
            for preferred in &preferences {
                for (i, (name, _)) in candidates.iter().enumerate() {
                    if name == preferred && admitted(name) && !order.contains(&i) {
                        order.push(i);
                    }
                }
            }
            for (i, (name, _)) in candidates.iter().enumerate() {
                if !name.contains("Copy") && admitted(name) && !order.contains(&i) {
                    order.push(i);
                }
            }
            for (i, (name, _)) in candidates.iter().enumerate() {
                if admitted(name) && !order.contains(&i) {
                    order.push(i);
                }
            }
            order
        };
    genome_with_ordering(egraph, matchers, &ordered)
}

/// STRICTNESS LEVELS for viability-aware election: level 0 = no
/// Materialize and no Copy (pure compute + views), level 1 = Materialize
/// allowed, level 2 = everything. Election runs level 0 first and
/// escalates only when no viable subtree exists — materializes and copies
/// are elected only when they are the ONLY route (the cycle-anatomy
/// doctrine, extended to the round-11 transpose-view re-description
/// cycles).
pub fn level_admits(level: usize) -> impl Fn(&str) -> bool {
    move |name: &str| match level {
        0 => !name.contains("Materialize") && !name.contains("Copy"),
        1 => !name.contains("Copy"),
        _ => true,
    }
}

/// The viability-aware election core (round 11): `ordered` supplies the
/// caller's preference order per (candidate list, strictness level); the
/// genome takes, per class, the first candidate in that order whose
/// chosen subtree reaches input terminals without revisiting a class —
/// exactly the real extractor's demand walk. Escalates the strictness
/// level only when nothing viable exists below; falls back to the
/// caller's level-2 order head for classes with no viable subtree at all
/// (dead rows are legal and free).
pub fn genome_with_ordering(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    matchers: &[Box<dyn OpMatcher>],
    ordered: ProducerOrdering<'_>,
) -> crate::extractor::Genome {
    use luminal::prelude::egraph_serialize::ClassId;
    use std::collections::{BTreeSet, HashMap};

    let index = crate::extractor::producer_index_with_matchers(egraph, matchers);

    // ------------------------------------------------------------------
    // ROUND 11: VIABILITY-AWARE primary election. The cycle-anatomy
    // ruling (2026-08-07) already demoted Copy below any non-Copy
    // producer because materializing-copy PAIRS between sibling layout
    // tensors of one logical value form 2-cycles. The round-11 transpose
    // views reproduce the same anatomy one level up: a transpose-view
    // value pair (x, x^T) carries mutually-derived layout tensors (each
    // frame's fresh right-major layout is the other frame's composed
    // view layout), so the VIEW op family now also forms 2-cycles of
    // pure re-descriptions, and a name-preference pick that cannot see
    // the walk deterministically elects them (observed: fixture1 dies
    // with "outputs with no plan" because the first CublasLt candidate
    // for the sibling's claim reads the views' fresh right-major
    // layouts, whose only preferred producers view each other).
    //
    // The fix keeps the SAME preference semantics (explicit preferences
    // in order, then any non-Copy, then Copy) but only accepts a
    // candidate whose chosen subtree can actually reach input terminals
    // without revisiting a class — exactly the real extractor's demand
    // walk. Where the old pick succeeded it was viable, so this elects
    // identically there; it differs only where the old pick had no plan.
    // The REAL search needs none of this: sampled genomes that reach no
    // plan are simply unfit; only this deterministic test helper must
    // land on a viable plan first try.
    // ------------------------------------------------------------------

    // INPUT TERMINALS, replicating the extractor's own rule
    // (extractor.rs collect_input_terminals): a BufferTensorLit whose
    // BufferTensor class is an explicit input (BufferInputLit list) when
    // explicit inputs exist, else any BufferTensorLit not reachable from
    // the BufferOutputLit list. These classes are leaves: they
    // take the extractor's Input plan, and the producer index carries no
    // rows for them at all (Extractor::new drops every input-terminal
    // key), so no genome row over one is even constructible here. This
    // helper's terminal set therefore exists for ONE purpose: the demand
    // walk's leaf check, which stops a candidate's subtree at a bound
    // input instead of declaring it unreachable.
    let buffer_list_classes = |root_op: &str| -> BTreeSet<ClassId> {
        let mut out: BTreeSet<ClassId> = BTreeSet::new();
        for node in egraph.nodes.values() {
            if node.op != root_op {
                continue;
            }
            let mut cur = node
                .children
                .first()
                .and_then(|id| egraph.nodes.get(id))
                .map(|c| c.eclass.clone());
            let mut guard = 0;
            while let Some(list) = cur {
                guard += 1;
                if guard > 64 {
                    break;
                }
                if egraph
                    .nodes
                    .values()
                    .any(|m| m.eclass == list && m.op == "BufferTensorNil")
                {
                    break;
                }
                let Some(cons) = egraph
                    .nodes
                    .values()
                    .find(|m| m.eclass == list && m.op == "BufferTensorCons")
                else {
                    break;
                };
                if let Some(h) = cons.children.first().and_then(|id| egraph.nodes.get(id)) {
                    out.insert(h.eclass.clone());
                }
                cur = cons
                    .children
                    .get(1)
                    .and_then(|id| egraph.nodes.get(id))
                    .map(|c| c.eclass.clone());
            }
        }
        out
    };
    let input_buffer_classes = buffer_list_classes("BufferInputLit");
    let output_buffer_classes = buffer_list_classes("BufferOutputLit");
    let has_explicit_inputs = !input_buffer_classes.is_empty();
    let mut terminals: BTreeSet<ClassId> = BTreeSet::new();
    for node in egraph.nodes.values() {
        if node.op != "BufferTensorLit" {
            continue;
        }
        if has_explicit_inputs {
            if !input_buffer_classes.contains(&node.eclass) {
                continue;
            }
        } else if output_buffer_classes.contains(&node.eclass) {
            continue;
        }
        let Some(lt) = node.children.first().and_then(|id| egraph.nodes.get(id)) else {
            continue;
        };
        terminals.insert(lt.eclass.clone());
    }

    // The Lit input lists of the op class a candidate enode belongs to.
    let lit_input_lists = |op_class: &ClassId| -> Vec<Vec<ClassId>> {
        let mut lists = Vec::new();
        for n in egraph.nodes.values() {
            if n.eclass != *op_class || n.op != "LayoutTensorOpLit" {
                continue;
            }
            let mut items = Vec::new();
            let mut cur = n
                .children
                .first()
                .and_then(|id| egraph.nodes.get(id))
                .map(|c| c.eclass.clone());
            let mut guard = 0;
            while let Some(list) = cur {
                guard += 1;
                if guard > 16 {
                    break;
                }
                if egraph
                    .nodes
                    .values()
                    .any(|m| m.eclass == list && m.op == "LayoutTensorNil")
                {
                    break;
                }
                let Some(cons) = egraph
                    .nodes
                    .values()
                    .find(|m| m.eclass == list && m.op == "LayoutTensorCons")
                else {
                    break;
                };
                if let Some(h) = cons.children.first().and_then(|id| egraph.nodes.get(id)) {
                    items.push(h.eclass.clone());
                }
                cur = cons
                    .children
                    .get(1)
                    .and_then(|id| egraph.nodes.get(id))
                    .map(|c| c.eclass.clone());
            }
            lists.push(items);
        }
        lists
    };

    #[derive(Clone, PartialEq)]
    enum Outcome {
        Viable(crate::extractor::ProducerChoice),
        Dead,
    }

    #[allow(clippy::too_many_arguments)]
    fn choose(
        class: &ClassId,
        level: usize,
        index: &std::collections::BTreeMap<
            ClassId,
            Vec<(String, crate::extractor::ProducerChoice)>,
        >,
        egraph: &luminal::prelude::egraph_serialize::EGraph,
        terminals: &BTreeSet<ClassId>,
        ordered: ProducerOrdering<'_>,
        lit_input_lists: &dyn Fn(&ClassId) -> Vec<Vec<ClassId>>,
        path: &mut Vec<ClassId>,
        memo: &mut HashMap<(ClassId, usize), Outcome>,
    ) -> Option<Outcome> {
        // None = failed BECAUSE of the current path (cycle) — not memoized.
        if terminals.contains(class) {
            return Some(Outcome::Dead); // terminal: needs no producer
        }
        if let Some(hit) = memo.get(&(class.clone(), level)) {
            return Some(hit.clone());
        }
        if path.contains(class) {
            return None; // cycle through the current path
        }
        let Some(candidates) = index.get(class) else {
            memo.insert((class.clone(), level), Outcome::Dead);
            return Some(Outcome::Dead);
        };
        path.push(class.clone());
        let mut cyclic_failure = false;
        let mut result: Option<crate::extractor::ProducerChoice> = None;
        'cands: for i in ordered(candidates, level) {
            let (_, choice) = &candidates[i];
            let Some(op_class) = egraph.nodes.get(&choice.enode).map(|n| n.eclass.clone()) else {
                continue;
            };
            let lists = lit_input_lists(&op_class);
            if lists.is_empty() {
                result = Some(choice.clone());
                break 'cands;
            }
            for list in &lists {
                let mut all_ok = true;
                for input in list {
                    match choose(
                        input,
                        level,
                        index,
                        egraph,
                        terminals,
                        ordered,
                        lit_input_lists,
                        path,
                        memo,
                    ) {
                        Some(Outcome::Viable(_)) | Some(Outcome::Dead)
                            if terminals.contains(input) =>
                        {
                            // terminal input: fine
                        }
                        Some(Outcome::Viable(_)) => {}
                        Some(Outcome::Dead) => {
                            all_ok = false;
                            break;
                        }
                        None => {
                            cyclic_failure = true;
                            all_ok = false;
                            break;
                        }
                    }
                }
                if all_ok {
                    result = Some(choice.clone());
                    break 'cands;
                }
            }
        }
        path.pop();
        match result {
            Some(choice) => {
                let outcome = Outcome::Viable(choice);
                memo.insert((class.clone(), level), outcome.clone());
                Some(outcome)
            }
            None if cyclic_failure => None, // path-dependent: do not memoize
            None => {
                memo.insert((class.clone(), level), Outcome::Dead);
                Some(Outcome::Dead)
            }
        }
    }

    let mut memo: HashMap<(ClassId, usize), Outcome> = HashMap::new();
    let mut genome = crate::extractor::Genome::default();
    for (class, candidates) in &index {
        // Escalate strictness only when nothing viable exists below.
        let mut pick: Option<crate::extractor::ProducerChoice> = None;
        for level in 0..3usize {
            let mut path = Vec::new();
            if let Some(Outcome::Viable(choice)) = choose(
                class,
                level,
                &index,
                egraph,
                &terminals,
                &ordered,
                &lit_input_lists,
                &mut path,
                &mut memo,
            ) {
                pick = Some(choice);
                break;
            }
        }
        // Fall back to the caller's unrestricted (level-2) preference
        // head where no viable subtree exists (dead rows are legal and
        // free — this preserves the pre-round-11 name-preference pick).
        let pick = pick.unwrap_or_else(|| {
            ordered(candidates, 2)
                .first()
                .map(|i| candidates[*i].1.clone())
                .unwrap_or_else(|| {
                    candidates
                        .first()
                        .expect("produced classes have candidates")
                        .1
                        .clone()
                })
        });
        genome.choices.insert(class.clone(), pick);
    }
    genome
}
