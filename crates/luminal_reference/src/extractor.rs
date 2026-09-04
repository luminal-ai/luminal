//! THE EXTRACTOR — the reference runtime's copy.
//!
//! Post-saturation selection is RUNTIME-OWNED (ruling 2026-09-03,
//! #420/#422 rejoin Phase 1): the e-graph walk that turns a saturated
//! program plus a genome into an [`luminal::layout_ir::ExtractedGraph`]
//! left core and was duplicated, verbatim, into every runtime that
//! searches. Austin's ruling on the duplication: "move extractor.rs to
//! each runtime. Maybe if there are some core utilities, they can belong
//! in core, but for now just do a simple duplication in each runtime."
//!
//! THE TESTS FOR ALL THREE COPIES LIVE HERE (the reference copy):
//! `render_memo_tests` below. The sibling copies carry none.
//!
//! Sibling copies: `luminal_reference::extractor`,
//! `luminal_cuda_lite::extractor`, `test_runtime::extractor`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use anyhow::{Context, Result, bail};
use egraph_serialize::{ClassId, EGraph, Node, NodeId};
use petgraph::graph::{DiGraph, NodeIndex};

use luminal::layout_ir::{
    Access, BufferInfo, ExtractedDag, ExtractedEdge, ExtractedGraph, ExtractedNode, ExtractionSite,
    FreedBy, InputNode, LayoutInfo, LayoutIrOp, LayoutTensorInfo, LazyText, LogicalInfo, OpInput,
    OpMatcher, OpNode, OutputNode, OutputSlot,
};
use luminal::logical_op::{LogicalRender, logical_op_for};

type Bounds = (Option<i128>, Option<i128>);
type BoundsIndex = HashMap<ClassId, Bounds>;

#[derive(Debug)]
struct Extractor<'a> {
    egraph: &'a EGraph,
    /// The op registry: egglog constructor name → registered matcher, built
    /// from [`built_in_matchers`] (optionally filtered by the allow-list —
    /// the test/debug lever that forces extraction through specific ops;
    /// structural plumbing — inputs, outputs, buffer lists — is never
    /// filtered). This registry is the ONLY dispatch: an enode whose label
    /// has no entry here simply offers no implementation candidate.
    matchers: HashMap<&'static str, Box<dyn OpMatcher>>,
    class_nodes: HashMap<ClassId, Vec<NodeId>>,
    /// The shared rendering state: the render-time class index and the
    /// per-(class, depth, preference) render memo, behind an `Rc` so the
    /// lazy text closures the extraction hands out can keep it alive
    /// after this borrow of the caller's e-graph ends.
    render: Rc<RenderCtx>,
    op_specs: HashMap<ClassId, Vec<OpSpec>>,
    producer_index: HashMap<ClassId, Vec<ProducerRef>>,
    input_terminals: HashMap<ClassId, InputInfo>,
    /// The search genome, when this walk is genome-driven (see [`Genome`]).
    /// `None` = the deterministic fixture extractor (min-cost tooling).
    genome: Option<Genome>,
    memo: HashMap<ClassId, Option<Plan>>,
    /// Post-relaxation blockage record (diagnosis, ruling 2026-08-07:
    /// UNDERSTAND refusals, no auto-repair): unplanned class → its
    /// candidates' unplanned children, plus the unplanned classes with
    /// no candidates at all. Cleared per extraction.
    blocked: HashMap<ClassId, Vec<ClassId>>,
    no_candidates: Vec<ClassId>,
    /// GENOME-INDEPENDENT op-instance cache: matcher `extract()` parses
    /// only enode METADATA (index maps, iota expressions, coordinate
    /// ranks — the deep backtracking walks over saturated classes), which
    /// never depends on the genome. Filled once per session, NEVER
    /// cleared by `extract_with_genome` — measured 2026-08-06: re-parsing
    /// per genome was 98% of a 378s MLP search (5.8s × 64 genomes).
    op_cache: std::cell::RefCell<HashMap<NodeId, Box<dyn LayoutIrOp>>>,
    /// GENOME-INDEPENDENT pricing caches for the bytes-moved heuristic
    /// (ruling 2026-08-10). `bounds_index` is built once by scanning the
    /// serialized `lower-bound-of` / `upper-bound-of` rows: IntExpr class →
    /// (lower, upper). `tensor_bytes_cache` memoizes the per-LayoutTensor
    /// byte size derived from its layout's shape and bit width.
    bounds_index: std::cell::RefCell<Option<BoundsIndex>>,
    tensor_bytes_cache: std::cell::RefCell<HashMap<ClassId, u64>>,
    /// GENOME-INDEPENDENT dtype index (typed-buffers landing A,
    /// 2026-08-11): serialized `dtype-of` rows, scanned once —
    /// LogicalTensor class → the plan dtype. Same row encoding as the
    /// bounds index (op = function name, children[0] = the argument
    /// node, the row's own eclass holds the value member); pinned by
    /// `dtype_index_reads_serialized_rows` in test_support.
    dtype_index: std::cell::RefCell<Option<HashMap<ClassId, luminal::dtype::PlanDtype>>>,
    /// GENOME-INDEPENDENT stable-key memo (measured 2026-08-10: eager
    /// per-comparison rendering was 99% of deep extraction — 7395/7471
    /// sampled stacks inside `is_better`). The rendered form of an enode
    /// never changes within a session, so one cache serves every genome.
    stable_key_cache: std::cell::RefCell<HashMap<NodeId, std::rc::Rc<str>>>,
}

#[derive(Debug, Clone)]
struct InputInfo {
    buffer_tensor_class: ClassId,
    buffer_tensor_enode: NodeId,
    buffer_id_class: ClassId,
    logical_name: String,
}

#[derive(Debug, Clone)]
struct Plan {
    /// Bytes-moved estimate for the subplan (ruling 2026-08-10): the sum
    /// over each op of the byte sizes of the tensors it reads and the
    /// results it writes, with symbolic dimensions priced at the midpoint
    /// of their seeded interval bounds. HEURISTIC by name and by nature:
    /// it assembles one plan per genome and orders the plain path; it
    /// never ranks genomes (profiling does) and never overrides the
    /// genome's op choice.
    heuristic_cost: u64,
    source_eclass: Option<ClassId>,
    source_enode: Option<NodeId>,
    selected_output_index: Option<usize>,
    input_list: Vec<ClassId>,
    output_list: Vec<ClassId>,
    kind: PlanKind,
    children: Vec<PlanChild>,
    metadata: Vec<PlanMeta>,
}

#[derive(Debug, Clone)]
struct PlanChild {
    port: String,
    class: ClassId,
}

#[derive(Debug, Clone)]
struct PlanMeta {
    name: &'static str,
    class: ClassId,
}

/// Everything `ClassRenderer::op_tooltip` reads out of a [`Plan`], cloned
/// at emission time so the DEFERRED tooltip can render long after the
/// plan (an extractor-internal value) is gone.
#[derive(Debug, Clone)]
struct OpTooltipSeed {
    class: ClassId,
    source_eclass: Option<ClassId>,
    source_enode: Option<NodeId>,
    selected_output_index: Option<usize>,
    heuristic_cost: u64,
    input_list: Vec<ClassId>,
    output_list: Vec<ClassId>,
    metadata: Vec<PlanMeta>,
}

/// Extractor-internal ONLY. `PlanKind` is the selection/cost IR at *e-graph*
/// granularity — it includes plumbing (buffer-list cons/nil, boundary literals)
/// that has no place in the clean dataflow output. It is deliberately private and
/// must never leak out of this module: the public artifact is [`ExtractedGraph`]
/// (whose nodes are [`ExtractedNode`]), which every `Plan` is lowered into by
/// `build_extracted_graph`. If you find yourself wanting to expose a `PlanKind`
/// (or `Plan`) across the module boundary, lower it to an `ExtractedNode` instead.
#[derive(Debug, Clone)]
enum PlanKind {
    Input(InputInfo),
    BufferOutputLit,
    BufferTensorCons,
    BufferTensorNil,
    BufferTensorLit {
        buffer_id_class: ClassId,
        logical_name: String,
    },
    LayoutIr(Box<dyn LayoutIrOp>),
}

#[derive(Debug, Clone)]
struct OpSpec {
    inputs: Vec<ClassId>,
    outputs: Vec<ClassId>,
}

#[derive(Debug, Clone)]
struct ProducerRef {
    op_class: ClassId,
    spec_index: usize,
    output_index: usize,
}

/// [`extract_layout_ir`] with an explicit runtime matcher set (the
/// TestRuntime seam — see `Extractor::new_with_matchers`).
pub fn extract_layout_ir_with_matchers(
    egraph: &EGraph,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> Result<Option<ExtractedGraph>> {
    Extractor::new_with_matchers(egraph, None, None, matchers).extract()
}

/// A reusable extraction session: the immutable analysis (class maps, op
/// specs, the runtime-viability fixpoint) is computed ONCE, and genomes
/// are swapped in per extraction. The implementation search runs dozens
/// of genome extractions per call — reconstructing the analysis each
/// time doubled suite wall time (measured 2026-08-05).
pub struct ExtractionSession<'a> {
    extractor: Extractor<'a>,
}

impl<'a> ExtractionSession<'a> {
    /// The runtime-owned constructor (ruling 2026-08-17): extraction
    /// over the CALLER's matcher set, intersected with its allow list.
    pub fn new_with_matcher_set(
        egraph: &'a EGraph,
        allowed_ops: Option<&[&str]>,
        matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
    ) -> Self {
        let allowed = allowed_ops.map(|ops| ops.iter().map(|op| op.to_string()).collect());
        let mut extractor = Extractor::new_with_matchers(egraph, allowed, None, matchers);
        extractor.apply_viability_filter();
        Self { extractor }
    }

    /// The genome sampling index over this session's matcher set —
    /// derivable without consuming the session, so runtime callers
    /// need not supply their matchers twice.
    pub fn producer_index(
        &self,
    ) -> std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>> {
        producer_index_from(&self.extractor)
    }

    /// The [`SamplingSpace`] over this session's producer index — the
    /// candidate graph's strongly connected components, built from the
    /// extractor's OWN per-candidate input enumeration (so the sampler
    /// and the planner agree on what an input is, by construction).
    ///
    /// ONE LEAF NOTION: a class with no genome row. Edges are the
    /// candidate's input classes FILTERED to the classes that hold a
    /// row; a row-less input is simply dropped, never contracted
    /// through. That is sound because after `apply_viability_filter`
    /// the genome index's keys ARE the producer index's keys, so a
    /// row-less class an `OpSpec` input names is either an INPUT
    /// TERMINAL — produced by nothing, planned from the boundary on
    /// `relax_to_fixpoint`'s first pass, never blocked, and holding no
    /// row for exactly that reason (see
    /// `Extractor::candidates_for_class`) — or a DEAD END with no
    /// candidate at all, which no cycle can run through.
    pub fn sampling_space(
        &self,
        index: &std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>>,
    ) -> SamplingSpace {
        let candidate_inputs = index
            .iter()
            .map(|(class, entries)| {
                let per_candidate = entries
                    .iter()
                    .map(|(_, choice)| {
                        let mut inputs = self.extractor.choice_input_classes(class, choice);
                        inputs.retain(|input| index.contains_key(input));
                        inputs
                    })
                    .collect();
                (class.clone(), per_candidate)
            })
            .collect();
        SamplingSpace::from_candidate_inputs(candidate_inputs)
    }

    /// Classify the last failed extraction's blockage (diagnosis ruling
    /// 2026-08-07: understand refusals, never auto-repair). Returns
    /// (has_choice_cycle, has_dead_end, summary): choice-cycles are
    /// strongly-connected components among unplanned classes whose chosen
    /// producers block on each other; dead-ends are unplanned classes
    /// with no candidate at all (a genome naming a producer whose route
    /// left the viable set, or a contract violation).
    pub fn failure_breakdown(&self) -> (bool, bool, String) {
        let extractor = &self.extractor;
        if extractor.blocked.is_empty() && extractor.no_candidates.is_empty() {
            return (false, false, "no blockage recorded".to_string());
        }
        // SUBSUMED SPELLINGS ARE NOT CANDIDATES (cleanup stratum,
        // 2026-09-02): the diagnosis names what the walk could have
        // chosen, so a retired term must never appear as one. The
        // last-resort arm keeps a label for a class that is entirely
        // subsumed rather than printing "?".
        let live_ops = |class: &ClassId| {
            extractor
                .class_nodes
                .get(class)
                .into_iter()
                .flatten()
                .filter_map(|id| extractor.egraph.nodes.get(id))
                .filter(|node| !node.subsumed)
                .map(|node| node.op.as_str())
        };
        let class_label = |class: &ClassId| -> String {
            live_ops(class)
                .find(|op| op.starts_with("LayoutTensorOp"))
                .or_else(|| live_ops(class).next())
                .or_else(|| {
                    extractor
                        .class_nodes
                        .get(class)
                        .into_iter()
                        .flatten()
                        .filter_map(|id| extractor.egraph.nodes.get(id))
                        .map(|node| node.op.as_str())
                        .next()
                })
                .unwrap_or("?")
                .to_string()
        };
        // Tarjan over the blocked graph via petgraph.
        let mut graph = petgraph::graph::DiGraph::<ClassId, ()>::new();
        let mut nodes: HashMap<ClassId, petgraph::graph::NodeIndex> = HashMap::new();
        for class in extractor.blocked.keys() {
            let index = graph.add_node(class.clone());
            nodes.insert(class.clone(), index);
        }
        for (class, blockers) in &extractor.blocked {
            for blocker in blockers {
                if let (Some(&a), Some(&b)) = (nodes.get(class), nodes.get(blocker)) {
                    graph.add_edge(a, b, ());
                }
            }
        }
        let mut cycles: Vec<Vec<String>> = Vec::new();
        for scc in petgraph::algo::tarjan_scc(&graph) {
            let is_cycle = scc.len() > 1 || (scc.len() == 1 && graph.contains_edge(scc[0], scc[0]));
            if is_cycle {
                let mut labels: Vec<String> = scc
                    .iter()
                    .map(|index| class_label(&graph[*index]))
                    .collect();
                labels.sort();
                labels.dedup();
                cycles.push(labels);
            }
        }
        // Landing D refusal visibility: a dead-end whose logical member
        // is a proof-gated Int op did not fail for lack of a kernel —
        // it failed for lack of a PROOF. Name that, and name the door.
        let bounded: std::collections::HashSet<ClassId> = extractor
            .egraph
            .nodes
            .values()
            .filter(|node| node.op == "value-lower-bound-of")
            .filter_map(|node| {
                node.children
                    .first()
                    .and_then(|id| extractor.egraph.nodes.get(id))
                    .map(|arg| arg.eclass.clone())
            })
            .collect();
        const PROOF_GATED: [&str; 5] = [
            "LogicalAdd",
            "LogicalMul",
            "LogicalReduceSum",
            "LogicalTruncDiv",
            "LogicalTruncRem",
        ];
        let unproven_note = |class: &ClassId| -> Option<String> {
            let (logical, _layout) = extractor.layout_tensor_parts(class)?;
            let gated_op = extractor
                .class_nodes
                .get(&logical)?
                .iter()
                .filter_map(|id| extractor.egraph.nodes.get(id))
                .map(|node| node.op.as_str())
                .find(|op| PROOF_GATED.contains(op))?;
            let dtype = extractor.with_dtype_index(|index| index.get(&logical).copied())?;
            if !matches!(
                dtype,
                luminal::dtype::PlanDtype::Int | luminal::dtype::PlanDtype::Int64
            ) {
                return None;
            }
            Some(if bounded.contains(&logical) {
                format!(
                    "{gated_op} on {dtype:?} has value bounds but they do not \
                     discharge the proof obligations (width, or a divisor \
                     range admitting zero) — tighten the attested input \
                     ranges (non-wrapping ruling 2026-08-11)"
                )
            } else {
                format!(
                    "{gated_op} on {dtype:?} is UNPROVEN — no value-bounds \
                     fact reached it; attest the input data's range with \
                     bind_value_range (non-wrapping ruling 2026-08-11)"
                )
            })
        };
        let dead_ends: Vec<String> = extractor
            .no_candidates
            .iter()
            .map(|class| match unproven_note(class) {
                Some(note) => format!("{} ({class}) — {note}", class_label(class)),
                None => format!("{} ({class})", class_label(class)),
            })
            .collect();
        let summary = format!(
            "{} unplanned classes; choice-cycles: {} {:?}; dead-ends: {} {:?}",
            extractor.blocked.len() + extractor.no_candidates.len(),
            cycles.len(),
            cycles.iter().take(3).collect::<Vec<_>>(),
            dead_ends.len(),
            dead_ends.iter().take(5).collect::<Vec<_>>()
        );
        (!cycles.is_empty(), !dead_ends.is_empty(), summary)
    }

    /// Deep anatomy of the last failure's cyclic SCCs, for the
    /// semantics question "what ARE these welded pairs": per member, the
    /// LayoutTensorLit children (logical class, layout class) and every
    /// candidate with its input classes' own (logical, layout) pairs —
    /// so same-logical/different-layout structure is visible directly.
    pub fn blockage_anatomy(&self) -> String {
        use std::fmt::Write as _;
        let ex = &self.extractor;
        let lit_children = |class: &ClassId| -> Option<(ClassId, ClassId)> {
            let node_id = ex.class_nodes.get(class)?.iter().find(|id| {
                ex.egraph
                    .nodes
                    .get(*id)
                    .is_some_and(|n| n.op == "LayoutTensorLit")
            })?;
            let node = ex.egraph.nodes.get(node_id)?;
            let logical = ex.egraph.nodes.get(node.children.first()?)?.eclass.clone();
            let layout = ex.egraph.nodes.get(node.children.get(1)?)?.eclass.clone();
            Some((logical, layout))
        };
        let mut out = String::new();
        let mut graph = petgraph::graph::DiGraph::<ClassId, ()>::new();
        let mut nodes: HashMap<ClassId, petgraph::graph::NodeIndex> = HashMap::new();
        for class in ex.blocked.keys() {
            nodes.insert(class.clone(), graph.add_node(class.clone()));
        }
        for (class, blockers) in &ex.blocked {
            for blocker in blockers {
                if let (Some(&a), Some(&b)) = (nodes.get(class), nodes.get(blocker)) {
                    graph.add_edge(a, b, ());
                }
            }
        }
        let mut dumped = 0;
        for scc in petgraph::algo::tarjan_scc(&graph) {
            let cyclic = scc.len() > 1 || (scc.len() == 1 && graph.contains_edge(scc[0], scc[0]));
            if !cyclic || dumped >= 3 {
                continue;
            }
            dumped += 1;
            let _ = writeln!(out, "SCC (size {}):", scc.len());
            for index in &scc {
                let class = &graph[*index];
                let children = lit_children(class);
                let _ = writeln!(
                    out,
                    "  member {class}: logical={:?} layout={:?}",
                    children.as_ref().map(|(l, _)| l.to_string()),
                    children.as_ref().map(|(_, layout)| layout.to_string())
                );
                for candidate in self.extractor.candidates_for_class(class) {
                    let op = candidate
                        .source_enode
                        .as_ref()
                        .and_then(|id| ex.egraph.nodes.get(id))
                        .map(|n| n.op.clone())
                        .unwrap_or_else(|| "?".to_string());
                    let inputs: Vec<String> = candidate
                        .children
                        .iter()
                        .map(|child| {
                            let pair = lit_children(&child.class);
                            format!(
                                "{} (logical={:?} layout={:?})",
                                child.class,
                                pair.as_ref().map(|(l, _)| l.to_string()),
                                pair.as_ref().map(|(_, layout)| layout.to_string())
                            )
                        })
                        .collect();
                    let _ = writeln!(out, "    candidate {op}: inputs {inputs:?}");
                }
            }
        }
        if out.is_empty() {
            out.push_str("no cyclic SCCs recorded");
        }
        out
    }

    pub fn extract_with_genome(&mut self, genome: &Genome) -> Result<Option<ExtractedGraph>> {
        self.extractor.genome = Some(genome.clone());
        self.extractor.memo.clear();
        self.extractor.blocked.clear();
        self.extractor.no_candidates.clear();
        self.extractor.extract()
    }
}

/// [`extract_layout_ir_with_matchers`] restricted to an allow-list of
/// LayoutTensorOp constructor names — the test/debug lever for exercising
/// a specific implementation. `None` allows every op; a program not
/// implementable within the list fails extraction loudly.
///
/// This is the DETERMINISTIC FIXTURE extractor (min-cost, tie-broken) —
/// tooling for fixtures and goldens, not the selection mechanism. The
/// search path is [`extract_layout_ir_with_genome_and_matchers`].
pub fn extract_layout_ir_with_ops_and_matchers(
    egraph: &EGraph,
    allowed_ops: Option<&[&str]>,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> Result<Option<ExtractedGraph>> {
    let allowed = allowed_ops.map(|ops| ops.iter().map(|op| op.to_string()).collect());
    let mut extractor = Extractor::new_with_matchers(egraph, allowed, None, matchers);
    extractor.extract()
}

/// One genome choice: the concrete implementation enode that produces the
/// keyed LayoutTensor class, and which of its output slots carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducerChoice {
    pub enode: NodeId,
    pub output_index: usize,
}

/// The search genome: a per-LayoutTensor-class producer selection. The
/// genome is the ONLY authority under [`extract_layout_ir_with_genome`] —
/// it replaces both the cost choice and first-emission slot claiming. The
/// contract is TOTALITY over produced classes: a demanded class that has
/// producers but no entry fails extraction loudly (no silent substitution).
/// Entries for classes the walk never demands are dead rows — legal and
/// free (the reachability-kill semantics).
#[derive(Debug, Clone, Default)]
pub struct Genome {
    pub choices: HashMap<ClassId, ProducerChoice>,
}

/// Genome-driven extraction with an EXPLICIT runtime matcher set — the
/// selection adapter's walk (ruling 2026-08-13; deletes the tests-side
/// vendored-source workaround). Starts from the binding outputs and
/// instantiates exactly the genome's chosen producer per demanded class;
/// multi-output instances dedup by enode; output slots the genome does
/// NOT assign to their instance write anonymous waste destinations
/// (fresh synthetic values, allocated and freed unread — waste-allowed,
/// priced by profiling).
pub fn extract_layout_ir_with_genome_and_matchers(
    egraph: &EGraph,
    genome: &Genome,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> Result<Option<ExtractedGraph>> {
    let mut extractor = Extractor::new_with_matchers(egraph, None, Some(genome), matchers);
    extractor.apply_viability_filter();
    extractor.extract()
}

/// Every LayoutTensor class's candidate producers over an EXPLICIT
/// runtime matcher set (the TestRuntime seam), as
/// `(implementation constructor name, choice)` pairs sorted for
/// determinism — the raw material genome construction and mutation draw
/// from. Classes with no producers (boundary inputs) are absent.
pub fn producer_index_with_matchers(
    egraph: &EGraph,
    matchers: Vec<Box<dyn luminal::layout_ir::OpMatcher>>,
) -> std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>> {
    let mut extractor = Extractor::new_with_matchers(egraph, None, None, matchers);
    extractor.apply_viability_filter();
    producer_index_from(&extractor)
}

fn producer_index_from(
    extractor: &Extractor<'_>,
) -> std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>> {
    let mut index = std::collections::BTreeMap::new();
    for (class, producers) in &extractor.producer_index {
        let mut entries: Vec<(String, ProducerChoice)> = Vec::new();
        for producer in producers {
            let Some(node_ids) = extractor.class_nodes.get(&producer.op_class) else {
                continue;
            };
            for node_id in node_ids {
                let Some(node) = extractor.egraph.nodes.get(node_id) else {
                    continue;
                };
                if node.subsumed || !extractor.matchers.contains_key(node.op.as_str()) {
                    continue;
                }
                entries.push((
                    node.op.clone(),
                    ProducerChoice {
                        enode: node_id.clone(),
                        output_index: producer.output_index,
                    },
                ));
            }
        }
        if !entries.is_empty() {
            entries.sort_by_key(|(name, choice)| {
                (name.clone(), choice.enode.to_string(), choice.output_index)
            });
            index.insert(class.clone(), entries);
        }
    }
    index
}

/// The RE-DESCRIPTION anatomy of a producer index — what two-phase
/// sampling ("elect the progressing producers first, insert the
/// re-describing ones as plumbing", ruling 2026-08-07, generalized
/// 2026-09-02) consumes.
///
/// THE CANDIDATE GRAPH. One node per class holding a genome row; one
/// edge `C -> D` for every candidate of `C` whose extractor-demanded
/// inputs include `D` (see [`Extractor::choice_input_classes`] — the
/// SAME `OpSpec::inputs` enumeration `producer_candidates_for_output`
/// hands the planner, never a second notion of "input"). A class the
/// genome index has no row for is not a node, and an input naming one
/// is simply dropped from the edge list. Nothing is contracted through
/// such a class, and nothing needs to be: after
/// `Extractor::apply_viability_filter` the genome index's keys are the
/// producer index's keys, so a row-less input class is either an INPUT
/// TERMINAL — produced by nothing, planned from the boundary on
/// `relax_to_fixpoint`'s first pass, never blocked, and so never part
/// of a blocking cycle — or a DEAD END with no candidate at all. Both
/// are leaves; the row-less-ness is the same fact as the leaf-ness,
/// not a second rule.
///
/// THE COMPONENT CRITERION. A choice cycle is possible only where the
/// candidate graph has a cycle, so the re-description groups are
/// exactly its NON-TRIVIAL strongly connected components (size >= 2,
/// or a single class with a self-loop). A candidate's INTRA-COMPONENT
/// SOURCES are its inputs inside its own component; a candidate with
/// none PROGRESSES — every route it takes leaves the component, and
/// the condensation of an SCC decomposition is a DAG, so progressing
/// choices can never close a cycle no matter how they combine.
///
/// WHAT THIS SUBSUMES. The 2026-08-07 form grouped classes by SHARED
/// LOGICAL VALUE, on the (then true) premise that the logical graph is
/// a DAG so only layout siblings of one value could re-describe each
/// other — the measured Copy⟷Copy welds. The cuBLASLt double-transpose
/// collapse `(union ?w ?x)` broke that premise: two DIFFERENT logical
/// values re-describe each other through `LogicalIndexMapApply`
/// transposes, and both spellings looked "progressing" to the
/// logical-value grouping. Components are the mechanism itself — no op
/// name list, no logical-value heuristic — and same-logical sibling
/// groups fall out of them automatically wherever the welds are real.
pub struct SamplingSpace {
    /// class → per-candidate input classes that hold genome rows,
    /// parallel to the class's producer-index entry. These are the
    /// candidate-graph edges out of each candidate.
    pub candidate_inputs: std::collections::BTreeMap<ClassId, Vec<Vec<ClassId>>>,
    /// class → per-candidate INTRA-COMPONENT source classes, parallel
    /// to the class's producer-index entry. An empty inner vec marks a
    /// PROGRESSING candidate. Always a subset of `candidate_inputs`;
    /// identical to it only inside a component, empty everywhere else.
    pub intra_sources: std::collections::BTreeMap<ClassId, Vec<Vec<ClassId>>>,
    /// The non-trivial components: members sorted, components ordered —
    /// a deterministic processing order for the sampler.
    pub components: Vec<Vec<ClassId>>,
    /// class → its index in [`SamplingSpace::components`], for the
    /// classes that belong to a non-trivial one.
    pub component_of: std::collections::BTreeMap<ClassId, usize>,
}

impl SamplingSpace {
    /// The component decomposition of a candidate graph given directly
    /// as class → per-candidate input classes (already restricted to
    /// classes holding genome rows). Deterministic: nodes in sorted
    /// class order, edges in candidate order, components sorted.
    pub fn from_candidate_inputs(
        candidate_inputs: std::collections::BTreeMap<ClassId, Vec<Vec<ClassId>>>,
    ) -> Self {
        let mut graph = DiGraph::<ClassId, ()>::new();
        let mut node_of: std::collections::BTreeMap<ClassId, NodeIndex> =
            std::collections::BTreeMap::new();
        for class in candidate_inputs.keys() {
            node_of.insert(class.clone(), graph.add_node(class.clone()));
        }
        for (class, per_candidate) in &candidate_inputs {
            let from = node_of[class];
            let mut added: std::collections::BTreeSet<ClassId> = std::collections::BTreeSet::new();
            for inputs in per_candidate {
                for input in inputs {
                    let Some(&to) = node_of.get(input) else {
                        continue; // boundary input: a leaf, never an edge
                    };
                    if added.insert(input.clone()) {
                        graph.add_edge(from, to, ());
                    }
                }
            }
        }
        let mut components: Vec<Vec<ClassId>> = petgraph::algo::tarjan_scc(&graph)
            .into_iter()
            .filter(|scc| scc.len() > 1 || graph.contains_edge(scc[0], scc[0]))
            .map(|scc| {
                let mut members: Vec<ClassId> =
                    scc.iter().map(|index| graph[*index].clone()).collect();
                members.sort();
                members
            })
            .collect();
        components.sort();
        let mut component_of: std::collections::BTreeMap<ClassId, usize> =
            std::collections::BTreeMap::new();
        for (component, members) in components.iter().enumerate() {
            for member in members {
                component_of.insert(member.clone(), component);
            }
        }
        let intra_sources = candidate_inputs
            .iter()
            .map(|(class, per_candidate)| {
                let component = component_of.get(class).copied();
                let per_candidate = per_candidate
                    .iter()
                    .map(|inputs| match component {
                        None => Vec::new(),
                        Some(component) => inputs
                            .iter()
                            .filter(|input| component_of.get(*input) == Some(&component))
                            .cloned()
                            .collect(),
                    })
                    .collect();
                (class.clone(), per_candidate)
            })
            .collect();
        Self {
            candidate_inputs,
            intra_sources,
            components,
            component_of,
        }
    }

    /// The candidate position a genome selected for `class`, if the
    /// genome names one of the class's index entries.
    pub fn chosen_position(
        &self,
        index: &std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>>,
        genome: &Genome,
        class: &ClassId,
    ) -> Option<usize> {
        let choice = genome.choices.get(class)?;
        index
            .get(class)?
            .iter()
            .position(|(_, candidate)| candidate == choice)
    }

    /// The genome's CHOSEN-EDGE graph: class → the genome-row input
    /// classes of the one candidate the genome elected for it. This is
    /// the graph the sampler keeps acyclic, and the graph the
    /// extractor's blocking follows.
    pub fn chosen_edges(
        &self,
        index: &std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>>,
        genome: &Genome,
    ) -> std::collections::BTreeMap<ClassId, Vec<ClassId>> {
        self.candidate_inputs
            .iter()
            .map(|(class, per_candidate)| {
                let edges = self
                    .chosen_position(index, genome, class)
                    .and_then(|position| per_candidate.get(position))
                    .cloned()
                    .unwrap_or_default();
                (class.clone(), edges)
            })
            .collect()
    }

    /// The genome's chosen INTRA-COMPONENT edges: component members
    /// only, each mapped to the intra-component sources of the one
    /// candidate the genome elected. This is the sub-graph the forest
    /// sampler and the mutation check actually keep acyclic, so it is
    /// what a tripwire prints when a genome reaches the planner with a
    /// cycle anyway.
    pub fn chosen_intra_edges(
        &self,
        index: &std::collections::BTreeMap<ClassId, Vec<(String, ProducerChoice)>>,
        genome: &Genome,
    ) -> std::collections::BTreeMap<ClassId, Vec<ClassId>> {
        self.component_of
            .keys()
            .map(|class| {
                let sources = self
                    .chosen_position(index, genome, class)
                    .and_then(|position| {
                        self.intra_sources
                            .get(class)
                            .and_then(|per_candidate| per_candidate.get(position))
                    })
                    .cloned()
                    .unwrap_or_default();
                (class.clone(), sources)
            })
            .collect()
    }
}

/// Does a chosen-edge graph close a cycle? (Iterative three-colour DFS;
/// the graphs are the sampler's own component-local edges.)
pub fn edges_have_cycle(edges: &std::collections::BTreeMap<ClassId, Vec<ClassId>>) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Colour {
        Open,
        Done,
    }
    let mut colour: HashMap<&ClassId, Colour> = HashMap::new();
    for root in edges.keys() {
        if colour.contains_key(root) {
            continue;
        }
        // (class, next child index) frames — no recursion, the graphs
        // can be as deep as the program.
        let mut stack: Vec<(&ClassId, usize)> = vec![(root, 0)];
        colour.insert(root, Colour::Open);
        while let Some((class, cursor)) = stack.pop() {
            let children = edges.get(class).map_or(&[][..], Vec::as_slice);
            if cursor >= children.len() {
                colour.insert(class, Colour::Done);
                continue;
            }
            stack.push((class, cursor + 1));
            let child = &children[cursor];
            match colour.get(child) {
                Some(Colour::Open) => return true,
                Some(Colour::Done) => {}
                None => {
                    colour.insert(child, Colour::Open);
                    stack.push((child, 0));
                }
            }
        }
    }
    false
}

/// A stable fingerprint of a plan's SHAPE: the chosen instances (enode +
/// claimed slots) and the dataflow between them. Many genomes map to one
/// plan (dead rows are unread), so the search hashes the built plan and
/// skips re-profiling duplicates (ruling 2026-07-27).
#[allow(dead_code)] // selection-adapter API: test harness here; lib export in the luminal graft
pub fn plan_fingerprint(graph: &ExtractedGraph) -> u64 {
    use petgraph::visit::EdgeRef;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for index in graph.dag.node_indices() {
        match &graph.dag[index] {
            ExtractedNode::LayoutOp(op) => {
                "op".hash(&mut hasher);
                op.op.label().hash(&mut hasher);
                if let luminal::layout_ir::Provenance::Extracted {
                    source_enode,
                    selected_output_index,
                    ..
                } = &op.provenance
                {
                    source_enode.to_string().hash(&mut hasher);
                    selected_output_index.hash(&mut hasher);
                }
                for output in &op.outputs {
                    output.eclass.to_string().hash(&mut hasher);
                }
            }
            ExtractedNode::BufferInput(input) => {
                "in".hash(&mut hasher);
                input.value.eclass.to_string().hash(&mut hasher);
            }
            ExtractedNode::BufferOutput(output) => {
                "out".hash(&mut hasher);
                for slot in &output.slots {
                    slot.index.hash(&mut hasher);
                    slot.value.to_string().hash(&mut hasher);
                }
            }
        }
    }
    for edge in graph.dag.edge_references() {
        edge.source().index().hash(&mut hasher);
        edge.target().index().hash(&mut hasher);
        edge.weight().port.hash(&mut hasher);
        edge.weight().value.to_string().hash(&mut hasher);
    }
    hasher.finish()
}

impl<'a> Extractor<'a> {
    /// The runtime-injectable constructor (the TestRuntime seam, ruling
    /// 2026-08-13): extraction consumes THE GIVEN runtime's matcher set —
    /// the reference registry is just the default caller.
    fn new_with_matchers(
        egraph: &'a EGraph,
        allowed_ops: Option<HashSet<String>>,
        genome: Option<&Genome>,
        matcher_set: Vec<Box<dyn OpMatcher>>,
    ) -> Self {
        let matchers: HashMap<&'static str, Box<dyn OpMatcher>> = matcher_set
            .into_iter()
            .filter(|matcher| {
                allowed_ops
                    .as_ref()
                    .is_none_or(|allowed| allowed.contains(matcher.egglog_constructor()))
            })
            .map(|matcher| (matcher.egglog_constructor(), matcher))
            .collect();
        let class_nodes = class_nodes(egraph);
        let render = Rc::new(RenderCtx::new(egraph));
        let (op_specs, mut producer_index) = collect_op_specs(egraph, &render.class_nodes);
        let output_buffer_classes = collect_output_buffer_classes(egraph, &class_nodes);
        let input_buffer_classes = collect_input_buffer_classes(egraph, &class_nodes);
        let input_terminals =
            collect_input_terminals(&render, &output_buffer_classes, &input_buffer_classes);
        // AN INPUT TERMINAL HAS NO PRODUCERS (2026-09-02). Its value
        // exists at launch: `relax_to_fixpoint` plans it from the
        // boundary, and `candidates_for_class` offers it no candidate at
        // all (see the leaf note there). Keeping producer rows for such
        // a class would leave DEAD WEIGHT in every genome — rows the
        // extractor must never honour — and the genome index built from
        // this map is exactly what the sampler's candidate graph has
        // nodes for, so dropping them here gives the component analysis
        // ONE leaf notion: a class with no genome row. Producers of
        // OTHER classes that READ a terminal are untouched — that is how
        // copy-from-a-boundary-input plans exist.
        producer_index.retain(|class, _| !input_terminals.contains_key(class));

        // ONE LEAF NOTION, CHECKED (cleanup stratum, ruling 2026-09-02).
        // The `cleanup` ruleset marks with `input-producer` every
        // LayoutTensorOp whose OUTPUT list holds a boundary input's LEAF
        // layout tensor, and the estate strips those spellings. The
        // extractor independently refuses producers for a class it seeds
        // as an input terminal (the retain above). Both are kept — belt
        // and braces — but they must agree about what a leaf IS, so the
        // fact set may add nothing the retain would not already take:
        // every class produced by a marked op has to be an input
        // terminal. A violation means the estate marked an op whose
        // output the extractor does NOT read as a launch-time leaf (or
        // vice versa), which is a representation disagreement, not a
        // cost accident. One pass, only when the relation is non-empty.
        let input_producer_ops = collect_input_producer_ops(egraph);
        if !input_producer_ops.is_empty() {
            for (class, producers) in &producer_index {
                for producer in producers {
                    assert!(
                        !input_producer_ops.contains(&producer.op_class),
                        "input-producer tripwire: op class {:?} carries the `input-producer` \
                         fact but its output class {:?} is not an input terminal — the estate \
                         and the extractor disagree about what a launch-time leaf is",
                        producer.op_class,
                        class,
                    );
                }
            }
        }

        Self {
            egraph,
            matchers,
            class_nodes,
            render,
            op_specs,
            producer_index,
            input_terminals,
            genome: genome.cloned(),
            memo: HashMap::new(),
            blocked: HashMap::new(),
            no_candidates: Vec::new(),
            op_cache: Default::default(),
            bounds_index: Default::default(),
            tensor_bytes_cache: Default::default(),
            dtype_index: Default::default(),
            stable_key_cache: Default::default(),
        }
    }

    /// RUNTIME-VIABILITY FILTER (Austin's ruling, 2026-08-05): restrict
    /// the producer index to ops the runtime can actually realize — a
    /// matched, unsubsumed implementation whose operand LayoutTensors
    /// are all transitively realizable from the program inputs. Any
    /// genome over the filtered index assembles an executable graph;
    /// residual discards are choice-cycles. LAZY: only genome-driven
    /// extraction needs this (the plain cost walk backtracks past dead
    /// candidates on its own), and the fixpoint is too expensive to pay
    /// on every plain extraction.
    fn apply_viability_filter(&mut self) {
        let op_matched: HashMap<&ClassId, bool> = self
            .op_specs
            .keys()
            .map(|op_class| {
                let matched = self
                    .class_nodes
                    .get(op_class)
                    .into_iter()
                    .flatten()
                    .any(|node_id| {
                        self.egraph.nodes.get(node_id).is_some_and(|node| {
                            !node.subsumed && self.matchers.contains_key(node.op.as_str())
                        })
                    });
                (op_class, matched)
            })
            .collect();
        let mut viable: HashSet<ClassId> = self.input_terminals.keys().cloned().collect();
        loop {
            let mut changed = false;
            for (op_class, specs) in &self.op_specs {
                if !op_matched.get(op_class).copied().unwrap_or(false) {
                    continue;
                }
                for spec in specs {
                    if spec.inputs.iter().all(|class| viable.contains(class)) {
                        for output in &spec.outputs {
                            if viable.insert(output.clone()) {
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let op_matched: HashMap<ClassId, bool> = op_matched
            .into_iter()
            .map(|(k, v)| (k.clone(), v))
            .collect();
        for producers in self.producer_index.values_mut() {
            producers.retain(|producer| {
                op_matched.get(&producer.op_class).copied().unwrap_or(false)
                    && self
                        .op_specs
                        .get(&producer.op_class)
                        .into_iter()
                        .flatten()
                        .any(|spec| spec.inputs.iter().all(|class| viable.contains(class)))
            });
        }
        self.producer_index
            .retain(|_, producers| !producers.is_empty());
    }

    fn extract(&mut self) -> Result<Option<ExtractedGraph>> {
        let roots = output_root_classes(self.egraph);
        if roots.is_empty() {
            return Ok(None);
        }

        self.relax_to_fixpoint(&roots);
        for root in &roots {
            if self.memo.get(root).cloned().flatten().is_none() {
                // Refusal accounting: the output list is ONE class — walk
                // its cons spine and test each element so the refusal
                // names WHICH outputs (binding order) have no plan.
                let mut failing: Vec<usize> = Vec::new();
                let spine_nodes = self.class_nodes.get(root).cloned().unwrap_or_default();
                if let Some(list_node) = spine_nodes.iter().find_map(|id| {
                    self.egraph
                        .nodes
                        .get(id)
                        .filter(|n| n.op == "BufferOutputLit")
                }) {
                    let mut spine = list_node
                        .children
                        .first()
                        .and_then(|c| self.egraph.nodes.get(c).map(|n| n.eclass.clone()));
                    let mut index = 0usize;
                    while let Some(class) = spine {
                        let Some(cons) = self
                            .class_nodes
                            .get(&class)
                            .into_iter()
                            .flatten()
                            .find_map(|id| {
                                self.egraph
                                    .nodes
                                    .get(id)
                                    .filter(|n| n.op == "BufferTensorCons")
                            })
                        else {
                            break;
                        };
                        if let Some(element) = cons
                            .children
                            .first()
                            .and_then(|c| self.egraph.nodes.get(c).map(|n| n.eclass.clone()))
                        {
                            if self.memo.get(&element).cloned().flatten().is_none() {
                                failing.push(index);
                            }
                        }
                        index += 1;
                        spine = cons
                            .children
                            .get(1)
                            .and_then(|c| self.egraph.nodes.get(c).map(|n| n.eclass.clone()));
                    }
                }
                bail!(
                    "failed to extract LayoutIR graph from BufferOutputLit eclass {root}; \
                     outputs with no plan (binding order): {failing:?}"
                );
            }
        }

        Ok(Some(self.build_extracted_graph(&roots)?))
    }

    /// The candidate set for one class under the current genome (or the
    /// full enumeration on the plain path). Pure construction — no
    /// recursion, no planning.
    ///
    /// GENOME AUTHORITY (the selection adapter): when a genome drives the
    /// walk, a class with producers is produced by EXACTLY its chosen
    /// enode/slot — never by cost, never by first-emission claiming. A
    /// produced class missing from the genome violates the total-genome
    /// contract: candidates empty out and extraction fails loudly at the
    /// root (fail-open, no silent substitution). The choice DIRECTS
    /// construction (2026-08-06): candidates are built for the chosen
    /// enode only, never built-then-discarded per spelling.
    fn candidates_for_class(&self, class: &ClassId) -> Vec<Candidate> {
        // AN INPUT TERMINAL IS A LEAF BY DEFINITION (2026-09-02): its
        // value exists at launch, so it is PRODUCED BY NOTHING.
        // `relax_to_fixpoint` seeds its plan from the boundary on pass
        // one at cost 0 with no children; offering candidates here let a
        // zero-cost producer (a VIEW moves no bytes, so it ties the
        // input plan at 0 and wins the `plan_label` tie-break —
        // "IndexMapApplyViewGeneric" sorts before "Input:…") take the
        // class over and EMIT a plan in which a program input is
        // computed from something that reads it back: a cyclic extracted
        // graph only bufferize rejects. Producers of OTHER classes that
        // read this one stay available — copies and views out of a
        // boundary input are exactly how such plans are written.
        if self.is_input_terminal(class) {
            return Vec::new();
        }
        let genome_choice = self.genome.as_ref().and_then(|genome| {
            self.producer_index
                .contains_key(class)
                .then(|| genome.choices.get(class))
        });
        let mut candidates = Vec::new();
        match genome_choice {
            Some(Some(choice)) => {
                if let Some(node) = self.egraph.nodes.get(&choice.enode) {
                    let node_id = choice.enode.clone();
                    if let Some(candidate) = self.candidate_for_node(&node_id, node) {
                        candidates.push(candidate);
                    }
                }
                candidates.extend(self.producer_candidates_for_output(class));
                candidates.retain(|candidate| {
                    candidate.source_enode.as_ref() == Some(&choice.enode)
                        && candidate.selected_output_index == Some(choice.output_index)
                });
            }
            Some(None) => {} // total-genome contract violated: no candidates
            None => {
                let node_ids = self.class_nodes.get(class).cloned().unwrap_or_default();
                for node_id in node_ids {
                    let Some(node) = self.egraph.nodes.get(&node_id) else {
                        continue;
                    };
                    if let Some(candidate) = self.candidate_for_node(&node_id, node) {
                        candidates.push(candidate);
                    }
                }
                candidates.extend(self.producer_candidates_for_output(class));
            }
        }
        candidates
    }

    /// BOTTOM-UP FIXPOINT RELAXATION (2026-08-07) — replaces the
    /// recursive walk. The DFS's cycle guard made `None` contextual:
    /// not caching it re-explored whole subtrees exponentially on
    /// cycle-rich e-graphs (the 2-layer decoder hang — 15k nodes,
    /// more than 150s), and caching it produced wrong refusals. Relaxation has
    /// neither problem: a class's plan materializes the pass after all
    /// of some candidate's children have plans, costs only improve
    /// monotonically, and cycles simply never enable — no guard, no
    /// taint, polynomial by construction. The memo fills exactly as the
    /// walk would have filled it; `build_extracted_graph` reads it
    /// unchanged.
    fn relax_to_fixpoint(&mut self, roots: &[ClassId]) {
        // Discover the class universe reachable through candidates.
        // Progress reporting is PATHOLOGY-GATED: healthy extractions say
        // nothing; anything slow narrates itself every few seconds.
        let discovery_start = std::time::Instant::now();
        let mut last_report = discovery_start;
        let mut discovered: HashSet<ClassId> = HashSet::new();
        let mut universe: Vec<ClassId> = Vec::new();
        let mut candidate_lists: Vec<Vec<Candidate>> = Vec::new();
        let mut queue: std::collections::VecDeque<ClassId> = roots.iter().cloned().collect();
        while let Some(class) = queue.pop_front() {
            if !discovered.insert(class.clone()) {
                continue;
            }
            let candidates = self.candidates_for_class(&class);
            for candidate in &candidates {
                for child in &candidate.children {
                    if !discovered.contains(&child.class) {
                        queue.push_back(child.class.clone());
                    }
                }
            }
            if last_report.elapsed().as_secs() >= 5 {
                last_report = std::time::Instant::now();
                eprintln!(
                    "[extract] SLOW DISCOVERY {:?}: {} classes so far, {} queued, last class {}",
                    discovery_start.elapsed(),
                    universe.len(),
                    queue.len(),
                    class
                );
            }
            universe.push(class);
            candidate_lists.push(candidates);
        }
        let total_candidates: usize = candidate_lists.iter().map(Vec::len).sum();
        if discovery_start.elapsed().as_secs() >= 2 {
            eprintln!(
                "[extract] discovery {:?}: {} classes, {} candidates",
                discovery_start.elapsed(),
                universe.len(),
                total_candidates
            );
        }

        let mut passes = 0usize;
        let relax_start = std::time::Instant::now();
        let mut last_report = relax_start;
        loop {
            passes += 1;
            assert!(
                passes <= 100_000,
                "extraction fixpoint did not converge after {passes} passes over {} classes",
                universe.len()
            );
            if last_report.elapsed().as_secs() >= 5 {
                last_report = std::time::Instant::now();
                let planned = self.memo.values().filter(|plan| plan.is_some()).count();
                eprintln!(
                    "[extract] SLOW RELAX {:?}: pass {passes}, {} planned / {} classes",
                    relax_start.elapsed(),
                    planned,
                    universe.len()
                );
            }
            let mut changed = false;
            for (class, candidates) in universe.iter().zip(&candidate_lists) {
                let mut best = self.input_terminals.get(class).map(|input| Plan {
                    heuristic_cost: 0,
                    source_eclass: None,
                    source_enode: None,
                    selected_output_index: None,
                    input_list: Vec::new(),
                    output_list: Vec::new(),
                    kind: PlanKind::Input(input.clone()),
                    children: Vec::new(),
                    metadata: Vec::new(),
                });
                'candidates: for candidate in candidates {
                    let mut heuristic_cost = self.candidate_heuristic_cost(candidate);
                    let mut child_plans = Vec::with_capacity(candidate.children.len());
                    for child in &candidate.children {
                        let Some(Some(child_plan)) = self.memo.get(&child.class) else {
                            continue 'candidates;
                        };
                        // Saturating: child costs are memoized per CLASS but
                        // accumulated per plan EDGE, so a deep graph with shared
                        // subgraphs (whisper's decode loop) counts paths, not
                        // nodes, and overflows u64 (wrapped silently in release,
                        // panicked in debug). Saturation stops the panic; the
                        // path-vs-node cost model itself is a recorded follow-up.
                        heuristic_cost = heuristic_cost.saturating_add(child_plan.heuristic_cost);
                        child_plans.push(child.clone());
                    }
                    let plan = Plan {
                        heuristic_cost,
                        source_eclass: candidate
                            .source_eclass
                            .clone()
                            .or_else(|| Some(class.clone())),
                        source_enode: candidate.source_enode.clone(),
                        selected_output_index: candidate.selected_output_index,
                        input_list: candidate.input_list.clone(),
                        output_list: candidate.output_list.clone(),
                        kind: candidate.kind.clone(),
                        children: child_plans,
                        metadata: candidate.metadata.clone(),
                    };
                    if self.is_better(&plan, best.as_ref()) {
                        best = Some(plan);
                    }
                }
                let current = self.memo.get(class).cloned().flatten();
                let improved = match (&best, &current) {
                    (Some(new), Some(old)) => self.is_better(new, Some(old)),
                    (Some(_), None) => true,
                    (None, _) => false,
                };
                if improved {
                    self.memo.insert(class.clone(), best);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        // Classes that never planned: record the definitive None so the
        // failure diagnostics (and build-side plan()) read a settled memo.
        for class in &universe {
            self.memo.entry(class.clone()).or_insert(None);
        }
        // Blockage record for the refusal breakdown: for each unplanned
        // class, which of its candidates' children are also unplanned
        // (the enablement blockers), and which have no candidates at all.
        for (class, candidates) in universe.iter().zip(&candidate_lists) {
            if self.memo.get(class).cloned().flatten().is_some() {
                continue;
            }
            if candidates.is_empty() {
                self.no_candidates.push(class.clone());
                continue;
            }
            let blockers: Vec<ClassId> = candidates
                .iter()
                .flat_map(|candidate| candidate.children.iter().map(|child| child.class.clone()))
                .filter(|child| self.memo.get(child).cloned().flatten().is_none())
                .collect();
            self.blocked.insert(class.clone(), blockers);
        }
    }

    fn candidate_for_node(&self, node_id: &NodeId, node: &Node) -> Option<Candidate> {
        if node.subsumed || node.op == "[...]" {
            return None;
        }

        let op = node.op.as_str();
        match op {
            "BufferOutputLit" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferOutputLit,
                self.children(node, &[("outputs", 0)])?,
            )),
            "BufferTensorCons" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferTensorCons,
                self.children(node, &[("head", 0), ("tail", 1)])?,
            )),
            "BufferTensorNil" => Some(Candidate::structural(
                node_id,
                PlanKind::BufferTensorNil,
                vec![],
            )),
            "BufferTensorLit" => {
                let layout_tensor_class = self.child_class(node, 0)?;
                let buffer_id_class = self.child_class(node, 1)?;
                Some(Candidate::structural(
                    node_id,
                    PlanKind::BufferTensorLit {
                        buffer_id_class: buffer_id_class.clone(),
                        logical_name: self
                            .logical_name_from_layout_tensor(&layout_tensor_class)
                            .unwrap_or_else(|| layout_tensor_class.to_string()),
                    },
                    vec![PlanChild {
                        port: "tensor".to_string(),
                        class: layout_tensor_class,
                    }],
                ))
            }
            _ => None,
        }
    }

    /// Is this class planned straight from a boundary input? Such a
    /// class is plannable unconditionally (see the leaf note in
    /// `ExtractionSession::sampling_space`).
    fn is_input_terminal(&self, class: &ClassId) -> bool {
        self.input_terminals.contains_key(class)
    }

    /// The input classes the extractor would DEMAND for one genome
    /// choice on `class`: the union of `OpSpec::inputs` over every
    /// producer entry the choice resolves to — literally what
    /// [`Extractor::producer_candidates_for_output`] turns into a
    /// candidate's `children` (via `op_children`) and what
    /// `relax_to_fixpoint` blocks on. The sampler's candidate graph is
    /// built from THIS, so "input" means one thing in both places.
    ///
    /// A union rather than a per-spec list because a `ProducerChoice`
    /// names (enode, output slot) only: when one op class carries
    /// several distinct input lists at that slot, the planner enables
    /// the class as soon as SOME of them plans, so taking the union is
    /// the conservative reading — every edge the planner might follow
    /// is present, and a union-acyclic genome is acyclic under each
    /// spec individually.
    fn choice_input_classes(&self, class: &ClassId, choice: &ProducerChoice) -> Vec<ClassId> {
        let mut inputs: Vec<ClassId> = Vec::new();
        let Some(producers) = self.producer_index.get(class) else {
            return inputs;
        };
        let Some(node) = self.egraph.nodes.get(&choice.enode) else {
            return inputs;
        };
        if node.subsumed || node.op == "[...]" || !self.matchers.contains_key(node.op.as_str()) {
            return inputs;
        }
        for producer in producers {
            if producer.output_index != choice.output_index {
                continue;
            }
            if !self
                .class_nodes
                .get(&producer.op_class)
                .is_some_and(|nodes| nodes.contains(&choice.enode))
            {
                continue;
            }
            let Some(spec) = self
                .op_specs
                .get(&producer.op_class)
                .and_then(|specs| specs.get(producer.spec_index))
            else {
                continue;
            };
            inputs.extend(spec.inputs.iter().cloned());
        }
        inputs.sort();
        inputs.dedup();
        inputs
    }

    fn producer_candidates_for_output(&self, output_class: &ClassId) -> Vec<Candidate> {
        let Some(producers) = self.producer_index.get(output_class) else {
            return Vec::new();
        };

        let mut candidates = Vec::new();
        for producer in producers {
            let Some(spec) = self
                .op_specs
                .get(&producer.op_class)
                .and_then(|specs| specs.get(producer.spec_index))
            else {
                continue;
            };
            let Some(node_ids) = self.class_nodes.get(&producer.op_class) else {
                continue;
            };

            for node_id in node_ids {
                let Some(node) = self.egraph.nodes.get(node_id) else {
                    continue;
                };
                if let Some(candidate) = self.candidate_for_layout_op(producer, spec, node_id, node)
                {
                    candidates.push(candidate);
                }
            }
        }

        candidates
    }

    fn candidate_for_layout_op(
        &self,
        producer: &ProducerRef,
        spec: &OpSpec,
        node_id: &NodeId,
        node: &Node,
    ) -> Option<Candidate> {
        if node.subsumed || node.op == "[...]" {
            return None;
        }

        // The registry IS the dispatch: an enode whose constructor has no
        // registered matcher (unknown, or excluded by the allow-list — which
        // filters IMPLEMENTATIONS only) offers no candidate here. A MATCHED
        // enode, by contrast, is extractable by construction (the match rules
        // discharged applicability in egglog — see the OpMatcher validity
        // contract), so a metadata slot that fails to resolve is schema
        // drift between the matcher's slot spec and the preamble's
        // constructor arity: a bug, and it panics rather than silently
        // shrinking the candidate space.
        let matcher = self.matchers.get(node.op.as_str())?;
        let metadata = self.metadata(node, matcher.metadata_slots()).unwrap_or_else(|| {
            panic!(
                "schema drift: {} enode {node_id} does not satisfy its matcher's metadata slots {:?}",
                node.op,
                matcher.metadata_slots(),
            )
        });
        // THE BORROW ENDS BEFORE THE MISS PATH, explicitly. Spelling
        // this as `if let ... = self.op_cache.borrow().get(..) { .. }
        // else { ..borrow_mut().. }` reads correctly and panics in
        // edition 2021, where the scrutinee temporary outlives the
        // `else` arm; edition 2024 shortened exactly that scope, and
        // since Phase 7 all three copies live in 2024 crates, so that
        // spelling would compile-and-run correctly here too. It is
        // still not used: a lookup in its own statement drops the
        // `Ref` at the semicolon, so the guard is released before the
        // miss path writes in EVERY edition, and these three files are
        // hand-kept copies whose correctness should not depend on the
        // edition of whichever crate is holding them.
        let cached = self.op_cache.borrow().get(node_id).cloned();
        let op: Box<dyn LayoutIrOp> = match cached {
            Some(cached) => cached,
            None => {
                let op = matcher.extract(&ExtractionSite {
                    egraph: self.egraph,
                    node_id,
                    node,
                });
                self.op_cache
                    .borrow_mut()
                    .insert(node_id.clone(), op.clone());
                op
            }
        };

        let children = self.op_children(&spec.inputs, op.as_ref());
        Some(Candidate {
            source_eclass: Some(producer.op_class.clone()),
            source_enode: Some(node_id.clone()),
            selected_output_index: Some(producer.output_index),
            input_list: spec.inputs.clone(),
            output_list: spec.outputs.clone(),
            kind: PlanKind::LayoutIr(op),
            children,
            metadata,
        })
    }

    fn op_children(&self, inputs: &[ClassId], op: &dyn LayoutIrOp) -> Vec<PlanChild> {
        inputs
            .iter()
            .enumerate()
            .map(|(index, class)| PlanChild {
                port: op.operand_name(index),
                class: class.clone(),
            })
            .collect()
    }

    fn children(&self, node: &Node, ports: &[(&'static str, usize)]) -> Option<Vec<PlanChild>> {
        ports
            .iter()
            .map(|(port, index)| {
                Some(PlanChild {
                    port: (*port).to_string(),
                    class: self.child_class(node, *index)?,
                })
            })
            .collect()
    }

    fn metadata(&self, node: &Node, ports: &[(&'static str, usize)]) -> Option<Vec<PlanMeta>> {
        ports
            .iter()
            .map(|(name, index)| {
                Some(PlanMeta {
                    name,
                    class: self.child_class(node, *index)?,
                })
            })
            .collect()
    }

    fn child_class(&self, node: &Node, index: usize) -> Option<ClassId> {
        let child_id = node.children.get(index)?;
        self.egraph
            .nodes
            .get(child_id)
            .map(|child| child.eclass.clone())
    }

    // The remaining renderer delegates are the ones the EXTRACTION path
    // still reads. Everything the tooltips used moved onto
    // `ClassRenderer` with them (see `Extractor::lazy_text`).

    fn layout_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        self.renderer().layout_tensor_parts(class)
    }

    fn class_let_name(&self, class: &ClassId) -> Option<String> {
        self.renderer().class_let_name(class)
    }

    fn logical_children(&self, class: &ClassId) -> Vec<(&'static str, ClassId)> {
        self.renderer().logical_children(class)
    }

    /// Plan preference: (cost, copies, label) as before, then a CONTENT-based
    /// stable key. The e-graph unions commutative variants (`IntAdd(x,y)` =
    /// `IntAdd(y,x)`) into one class; without a content key, which variant wins
    /// depends on hash-iteration order and flips run to run. Rendering the
    /// source e-node resolves children to let-names/literals, which are stable
    /// across runs — making the (user-blessed) arbitrary tie-break deterministic.
    /// LAZY evaluation of the same total order (2026-08-10, semantics
    /// identical): cost decides almost every comparison, so labels are
    /// only built on cost ties and the rendered stable key only on label
    /// ties — and each enode's key renders once per session (memo).
    /// Eagerly building the full tuple rendered BOTH plans to depth 3 on
    /// EVERY comparison: 99% of deep-extraction wall time.
    fn is_better(&self, plan: &Plan, best: Option<&Plan>) -> bool {
        let Some(best) = best else {
            return true;
        };
        if plan.heuristic_cost != best.heuristic_cost {
            return plan.heuristic_cost < best.heuristic_cost;
        }
        let (plan_label_key, best_label_key) = (plan_label(plan), plan_label(best));
        if plan_label_key != best_label_key {
            return plan_label_key < best_label_key;
        }
        self.stable_key(plan) < self.stable_key(best)
    }

    fn stable_key(&self, plan: &Plan) -> std::rc::Rc<str> {
        let Some(enode) = plan.source_enode.as_ref() else {
            return std::rc::Rc::from("");
        };
        if let Some(key) = self.stable_key_cache.borrow().get(enode) {
            return key.clone();
        }
        let key: std::rc::Rc<str> = std::rc::Rc::from(self.renderer().render_node(enode, 3));
        self.stable_key_cache
            .borrow_mut()
            .insert(enode.clone(), key.clone());
        key
    }

    fn renderer(&self) -> ClassRenderer<'_> {
        self.render.renderer()
    }

    /// Defer one display-text field (ruling 2026-09-01). The closure
    /// captures an `Rc<RenderCtx>` — which owns its e-graph — so it can
    /// still run after this `Extractor`'s borrow of the caller's e-graph
    /// is gone, which is the normal case: the fixture harnesses drop the
    /// deserialized e-graph the moment extraction returns, and the
    /// visualizer reads the text long afterwards. `build` must be the
    /// SAME code the eager version ran, so the deferred string is the
    /// eager string.
    fn lazy_text(&self, build: impl Fn(&ClassRenderer<'_>) -> String + 'static) -> LazyText {
        let ctx = Rc::clone(&self.render);
        LazyText::deferred(move || build(&ctx.renderer()))
    }

    // ---- bytes-moved heuristic pricing (ruling 2026-08-10): the
    // heuristic_cost of a candidate is the bytes its op moves — operand
    // bytes for every declared READ plus result bytes for every declared
    // WRITE. Symbolic dims price at the midpoint of their seeded interval
    // bounds. Loud on broken invariants: a value tensor with no readable
    // shape/width, or a dim with neither literal nor bounds, names itself
    // and panics rather than silently distorting the search.

    /// The candidate's own contribution to `heuristic_cost` (children add
    /// theirs during relaxation). Structural plumbing and inputs are 0;
    /// a view (reads nothing, writes nothing) is honestly free.
    fn candidate_heuristic_cost(&self, candidate: &Candidate) -> u64 {
        match &candidate.kind {
            PlanKind::Input(_)
            | PlanKind::BufferOutputLit
            | PlanKind::BufferTensorCons
            | PlanKind::BufferTensorNil
            | PlanKind::BufferTensorLit { .. } => 0,
            PlanKind::LayoutIr(op) => {
                let reads = candidate
                    .input_list
                    .iter()
                    .enumerate()
                    .filter(|(operand, _)| op.operand_reads_memory(*operand))
                    .map(|(_, class)| self.tensor_bytes(class))
                    .fold(0u64, u64::saturating_add);
                let writes = candidate
                    .output_list
                    .iter()
                    .enumerate()
                    .filter(|(result, _)| op.result_writes_memory(*result))
                    .map(|(_, class)| self.tensor_bytes(class))
                    .fold(0u64, u64::saturating_add);
                reads.saturating_add(writes)
            }
        }
    }

    /// Byte size of a LayoutTensor class, memoized: product of its
    /// layout's extents times the element bit width, rounded up to bytes.
    fn tensor_bytes(&self, layout_tensor: &ClassId) -> u64 {
        if let Some(&bytes) = self.tensor_bytes_cache.borrow().get(layout_tensor) {
            return bytes;
        }
        let bytes = self.compute_tensor_bytes(layout_tensor);
        self.tensor_bytes_cache
            .borrow_mut()
            .insert(layout_tensor.clone(), bytes);
        bytes
    }

    fn compute_tensor_bytes(&self, layout_tensor: &ClassId) -> u64 {
        let renderer = self.renderer();
        let lit = renderer
            .node_with_op(layout_tensor, "LayoutTensorLit")
            .unwrap_or_else(|| {
                panic!("heuristic cost: class {layout_tensor} has no LayoutTensorLit spelling")
            });
        let node = self.egraph.nodes.get(lit).expect("lit node resolvable");
        let layout = child_class(self.egraph, node, 1).unwrap_or_else(|| {
            panic!("heuristic cost: LayoutTensorLit in {layout_tensor} has no layout child")
        });
        let dims = self.estimated_layout_dims(&layout).unwrap_or_else(|| {
            panic!(
                "heuristic cost: layout {layout} (of tensor {layout_tensor}) has no \
                 readable ShapeLit shape"
            )
        });
        let bits = renderer.numeric_layout_bits(&layout).unwrap_or_else(|| {
            panic!(
                "heuristic cost: layout {layout} (of tensor {layout_tensor}) has no \
                 literal bit width"
            )
        });
        let elements = dims
            .iter()
            .fold(1u128, |product, &dim| product.saturating_mul(dim as u128));
        let total_bits = elements.saturating_mul(bits.max(0) as u128);
        total_bits.div_ceil(8).min(u64::MAX as u128) as u64
    }

    /// The layout's extents with symbolic dims estimated (mirrors the
    /// renderer's `numeric_layout_dims`, but symbolic-tolerant).
    fn estimated_layout_dims(&self, class: &ClassId) -> Option<Vec<u64>> {
        let renderer = self.renderer();
        for node_id in renderer.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            let shape_child = match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit"
                | "LeftMajorContiguousElementLayoutLit"
                | "StridedElementLayoutLit" => 0,
                "ElementOffsetExpressionLayoutLit" | "BitOffsetExpressionLayoutLit" => 1,
                _ => continue,
            };
            let shape_class = child_class(self.egraph, node, shape_child)?;
            let shape_node_id = renderer.node_with_op(&shape_class, "ShapeLit")?;
            let shape_node = self.egraph.nodes.get(shape_node_id)?;
            let mut current = child_class(self.egraph, shape_node, 0)?;
            let mut dims = Vec::new();
            loop {
                if let Some(cons_id) = renderer.node_with_op(&current, "IntExprCons").cloned() {
                    let cons = self.egraph.nodes.get(&cons_id)?;
                    dims.push(self.dim_estimate(&child_class(self.egraph, cons, 0)?));
                    current = child_class(self.egraph, cons, 1)?;
                } else if renderer.node_with_op(&current, "IntExprNil").is_some() {
                    return Some(dims);
                } else {
                    return None;
                }
            }
        }
        None
    }

    /// One extent: a literal dim exactly; a symbolic dim at the MIDPOINT
    /// of its seeded interval (ruling 2026-08-10: halfway between the
    /// bounds). A dim with neither is a broken seeding contract — loud.
    fn dim_estimate(&self, dim: &ClassId) -> u64 {
        if let Some(value) = self.renderer().numeric_int_expr(dim) {
            return value.max(0) as u64;
        }
        self.with_bounds_index(|index| {
            let (lower, upper) = index.get(dim).copied().unwrap_or((None, None));
            match (lower, upper) {
                (Some(lower), Some(upper)) => ((lower + upper) / 2).max(0) as u64,
                _ => panic!(
                    "heuristic cost: dim class {dim} has neither a literal value nor \
                     complete seeded bounds (lower {lower:?}, upper {upper:?}) — the \
                     bounds-seeding contract is broken"
                ),
            }
        })
    }

    /// The serialized interval rows, indexed once: IntExpr class →
    /// (lower, upper). Rows encode as op `lower-bound-of`/`upper-bound-of`
    /// with the argument node as child 0 and the BigInt value as the row
    /// node's own eclass (observed encoding, probe 2026-08-10). Multiple
    /// rows per class merge tightest, mirroring the lattice's `:merge`.
    fn with_bounds_index<R>(
        &self,
        read: impl FnOnce(&HashMap<ClassId, (Option<i128>, Option<i128>)>) -> R,
    ) -> R {
        let mut slot = self.bounds_index.borrow_mut();
        if slot.is_none() {
            let mut index: HashMap<ClassId, (Option<i128>, Option<i128>)> = HashMap::new();
            for node in self.egraph.nodes.values() {
                let is_lower = node.op == "lower-bound-of";
                if !is_lower && node.op != "upper-bound-of" {
                    continue;
                }
                let Some(arg) = node
                    .children
                    .first()
                    .and_then(|id| self.egraph.nodes.get(id))
                else {
                    continue;
                };
                let Some(value) = self.bigint_value(&node.eclass) else {
                    continue;
                };
                let entry = index.entry(arg.eclass.clone()).or_insert((None, None));
                if is_lower {
                    entry.0 = Some(entry.0.map_or(value, |held: i128| held.max(value)));
                } else {
                    entry.1 = Some(entry.1.map_or(value, |held: i128| held.min(value)));
                }
            }
            *slot = Some(index);
        }
        read(slot.as_ref().expect("bounds index just built"))
    }

    /// The serialized `dtype-of` rows, indexed once: LogicalTensor
    /// class → [`luminal::dtype::PlanDtype`]. `dtype-of` is `:no-merge`
    /// in the preamble (a dtype divergence is a saturation panic), so
    /// at most one dtype per class can survive to serialization — a
    /// second, different row here means the invariant broke upstream
    /// and we refuse loudly rather than pick one.
    fn with_dtype_index<R>(
        &self,
        read: impl FnOnce(&HashMap<ClassId, luminal::dtype::PlanDtype>) -> R,
    ) -> R {
        let mut slot = self.dtype_index.borrow_mut();
        if slot.is_none() {
            let mut index: HashMap<ClassId, luminal::dtype::PlanDtype> = HashMap::new();
            for node in self.egraph.nodes.values() {
                if node.op != "dtype-of" {
                    continue;
                }
                let Some(arg) = node
                    .children
                    .first()
                    .and_then(|id| self.egraph.nodes.get(id))
                else {
                    continue;
                };
                let Some(dtype) = self.plan_dtype_value(&node.eclass) else {
                    continue;
                };
                match index.entry(arg.eclass.clone()) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(dtype);
                    }
                    std::collections::hash_map::Entry::Occupied(entry) => {
                        assert!(
                            *entry.get() == dtype,
                            "dtype-of divergence for class {}: {:?} vs {:?} \
                             (the :no-merge tripwire should have caught this \
                             at saturation)",
                            arg.eclass,
                            entry.get(),
                            dtype
                        );
                    }
                }
            }
            *slot = Some(index);
        }
        read(slot.as_ref().expect("dtype index just built"))
    }

    /// A `Dtype` class: the childless member whose op is one of the
    /// egglog dtype spellings.
    fn plan_dtype_value(&self, class: &ClassId) -> Option<luminal::dtype::PlanDtype> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.children.is_empty() {
                if let Some(dtype) = luminal::dtype::PlanDtype::from_egglog_name(&node.op) {
                    return Some(dtype);
                }
            }
        }
        None
    }

    /// A BigInt primitive class: the childless member whose op is the
    /// decimal literal.
    fn bigint_value(&self, class: &ClassId) -> Option<i128> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.children.is_empty() {
                if let Ok(value) = node.op.parse::<i128>() {
                    return Some(value);
                }
            }
        }
        None
    }

    fn logical_name_from_layout_tensor(&self, class: &ClassId) -> Option<String> {
        self.renderer().logical_name_from_layout_tensor(class)
    }
}

#[derive(Debug)]
struct Candidate {
    source_eclass: Option<ClassId>,
    source_enode: Option<NodeId>,
    selected_output_index: Option<usize>,
    input_list: Vec<ClassId>,
    output_list: Vec<ClassId>,
    kind: PlanKind,
    children: Vec<PlanChild>,
    metadata: Vec<PlanMeta>,
}

impl Candidate {
    fn structural(source_enode: &NodeId, kind: PlanKind, children: Vec<PlanChild>) -> Self {
        Self {
            source_eclass: None,
            source_enode: Some(source_enode.clone()),
            selected_output_index: None,
            input_list: Vec::new(),
            output_list: Vec::new(),
            kind,
            children,
            metadata: Vec::new(),
        }
    }
}

/// The render memo's key: an e-class, the remaining render depth, and the
/// preferred constructor. `preferred_op` is `&'static str` because every
/// caller names a constructor literally (or goes through
/// [`metadata_preferred_op`]), so a lookup allocates nothing; `ClassId` is
/// an `Arc<str>`, whose clone is a refcount bump.
type RenderKey = (ClassId, usize, Option<&'static str>);

/// The rendering state every renderer view shares: the e-graph, the
/// render-time class index (subsumed nodes INCLUDED — see
/// [`render_class_nodes`]), and the render memo.
///
/// A MEMO, NEVER A SKIP (ruling 2026-09-01). Rendered text is not only
/// display: `stable_key` renders a plan's source e-node to break
/// selection ties, so a cached render must return EXACTLY the string the
/// uncached computation would have produced. A class's members never
/// change within a session, so one entry serves every genome — the ctx
/// outlives `ExtractionSession::extract_with_genome`'s cache clearing,
/// exactly like `op_cache` and `stable_key_cache`.
///
/// The e-graph here is OWNED (a clone of the caller's). Lazy text
/// closures hold an `Rc<RenderCtx>` and are forced after extraction
/// returns, by which time the caller's `&EGraph` is frequently gone (the
/// fixture harnesses drop the deserialized e-graph on return), so a
/// borrow could not carry them.
#[derive(Debug)]
struct RenderCtx {
    egraph: EGraph,
    class_nodes: HashMap<ClassId, Vec<NodeId>>,
    memo: RefCell<HashMap<RenderKey, Rc<str>>>,
}

impl RenderCtx {
    fn new(egraph: &EGraph) -> Self {
        Self {
            class_nodes: render_class_nodes(egraph),
            egraph: egraph.clone(),
            memo: RefCell::new(HashMap::new()),
        }
    }

    fn renderer(&self) -> ClassRenderer<'_> {
        ClassRenderer {
            egraph: &self.egraph,
            class_nodes: &self.class_nodes,
            memo: &self.memo,
        }
    }
}

struct ClassRenderer<'a> {
    egraph: &'a EGraph,
    class_nodes: &'a HashMap<ClassId, Vec<NodeId>>,
    memo: &'a RefCell<HashMap<RenderKey, Rc<str>>>,
}

/// The renderer's implementation of the [`LogicalRender`] callbacks: the
/// bridge each [`luminal::logical_op::LogicalOp`] formats itself through.
/// Carries the recursion guard (`visiting`) so `child_expr` cycles fall back
/// to labels exactly as direct recursion did.
struct LogicalRenderCtx<'r, 'a, 'v> {
    renderer: &'r ClassRenderer<'a>,
    visiting: &'v mut HashSet<ClassId>,
    /// Remaining expansion depth — the logical value graph is a
    /// CONVERGENT DAG (residual connections reuse values), so unbounded
    /// readable-expression expansion is exponential (2026-08-07: the
    /// 2-layer decoder build hang). Past the cap, children render as
    /// short labels.
    depth: usize,
}

impl LogicalRender for LogicalRenderCtx<'_, '_, '_> {
    fn child_expr(&mut self, node: &Node, index: usize) -> String {
        child_class(self.renderer.egraph, node, index)
            .map(|class| {
                self.renderer
                    .readable_logical_expr_depth(&class, self.visiting, self.depth)
            })
            .unwrap_or_else(|| "?".to_string())
    }

    fn child_short(
        &mut self,
        node: &Node,
        index: usize,
        depth: usize,
        prefer: Option<&'static str>,
    ) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .map(|class| self.renderer.render_class_prefer(&class, depth, prefer))
    }

    fn child_shape(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .and_then(|class| self.renderer.readable_shape(&class))
    }

    fn child_index_map(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .and_then(|class| self.renderer.readable_index_map(&class))
    }

    fn child_int_expr(&mut self, node: &Node, index: usize) -> Option<String> {
        child_class(self.renderer.egraph, node, index)
            .map(|class| self.renderer.readable_expr(&class, &mut HashSet::new()))
    }
}

impl<'a> ClassRenderer<'a> {
    fn class_let_name(&self, class: &ClassId) -> Option<String> {
        self.egraph
            .class_data
            .get(class)
            .and_then(|data| data.extra.get("let"))
            .cloned()
    }

    fn class_type(&self, class: &ClassId) -> Option<String> {
        self.egraph
            .class_data
            .get(class)
            .and_then(|data| data.typ.clone())
    }

    /// MEMOIZED per (class, depth, preference) for the session
    /// (2026-09-01). Rendering recurses into every child class at
    /// `depth - 1`, so on a convergent DAG the uncached walk re-rendered
    /// shared subterms once per path — exponential in depth, and the
    /// extraction wall on the deep model fixtures. A memo is the only
    /// legal fix: the text feeds `stable_key`, so skipping or truncating
    /// a render would change plan election. Returns the stored string
    /// verbatim.
    fn render_class_prefer(
        &self,
        class: &ClassId,
        depth: usize,
        preferred_op: Option<&'static str>,
    ) -> String {
        if depth == 0 {
            return class.to_string();
        }

        let key = (class.clone(), depth, preferred_op);
        // Two statements on purpose: the `Ref` must die at this
        // semicolon, because `render_node` below recurses straight back
        // into this function and a live borrow would panic.
        let hit = self.memo.borrow().get(&key).cloned();
        if let Some(hit) = hit {
            return hit.to_string();
        }

        let rendered = match self
            .class_nodes
            .get(class)
            .and_then(|node_ids| choose_render_node(self.egraph, node_ids, preferred_op))
        {
            Some(node_id) => self.render_node(node_id, depth),
            None => class.to_string(),
        };
        self.memo
            .borrow_mut()
            .insert(key, Rc::from(rendered.as_str()));
        rendered
    }

    fn render_class_with_op(&self, class: &ClassId, depth: usize, op: &str) -> Option<String> {
        let node_id = self.node_with_op(class, op)?;
        Some(self.render_node(node_id, depth))
    }

    fn node_with_op(&self, class: &ClassId, op: &str) -> Option<&NodeId> {
        self.class_nodes.get(class)?.iter().find(|node_id| {
            self.egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == op)
        })
    }

    fn render_node(&self, node_id: &NodeId, depth: usize) -> String {
        if depth == 0 {
            return self
                .egraph
                .nodes
                .get(node_id)
                .map(|node| node.eclass.to_string())
                .unwrap_or_else(|| node_id.to_string());
        }

        let Some(node) = self.egraph.nodes.get(node_id) else {
            return node_id.to_string();
        };
        if node.children.is_empty() {
            return node.op.clone();
        }

        let args = node
            .children
            .iter()
            .filter_map(|child_id| self.egraph.nodes.get(child_id))
            .map(|child| self.render_class_prefer(&child.eclass, depth - 1, None))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({args})", node.op)
    }

    fn display_name(&self, class: &ClassId, fallback: impl FnOnce() -> String) -> String {
        self.class_let_name(class).unwrap_or_else(fallback)
    }

    fn logical_name_from_layout_tensor(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LayoutTensorLit" {
                continue;
            }
            let logical_class = child_class(self.egraph, node, 0)?;
            if let Some(name) = self.logical_name_from_logical(&logical_class) {
                return Some(name);
            }
        }
        None
    }

    fn logical_name_from_logical(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" && node.op != "LogicalTensorNamed" {
                continue;
            }
            let id_class = child_class(self.egraph, node, 0)?;
            return Some(self.render_class_prefer(&id_class, 2, Some("LogicalIdLit")));
        }
        None
    }

    fn layout_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LayoutTensorLit" {
                continue;
            }
            return Some((
                child_class(self.egraph, node, 0)?,
                child_class(self.egraph, node, 1)?,
            ));
        }
        None
    }

    fn buffer_tensor_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "BufferTensorLit" {
                continue;
            }
            return Some((
                child_class(self.egraph, node, 0)?,
                child_class(self.egraph, node, 1)?,
            ));
        }
        None
    }

    fn layout_tensor_label(&self, class: &ClassId) -> String {
        self.display_name(class, || {
            self.layout_tensor_parts(class)
                .map(|(logical, _)| self.logical_label(&logical))
                .or_else(|| self.logical_name_from_layout_tensor(class))
                .unwrap_or_else(|| class.to_string())
        })
    }

    fn layout_tensor_summary(&self, class: &ClassId) -> String {
        let label = self.layout_tensor_label(class);
        let Some((logical, layout)) = self.layout_tensor_parts(class) else {
            return label;
        };
        format!(
            "{label}(logical={}, layout={})",
            self.logical_label(&logical),
            self.canonical_layout(&layout)
        )
    }

    fn logical_label(&self, class: &ClassId) -> String {
        self.display_name(class, || {
            let Some(node_id) = self.choose_logical_node(class) else {
                return class.to_string();
            };
            let Some(node) = self.egraph.nodes.get(node_id) else {
                return class.to_string();
            };
            match logical_op_for(node.op.as_str()) {
                Some(op) => op.display_label(
                    node,
                    &mut LogicalRenderCtx {
                        renderer: self,
                        visiting: &mut HashSet::new(),
                        depth: 8,
                    },
                ),
                None => self.render_node(node_id, 8),
            }
        })
    }

    fn logical_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = self.class_details(class);
        details.push((
            "expr".to_string(),
            self.readable_logical_expr(class, &mut HashSet::new()),
        ));
        if let Some(shape) = self.logical_shape(class) {
            details.push(("shape".to_string(), shape));
        }
        if let Some(dtype) = self.logical_dtype(class) {
            details.push(("dtype".to_string(), dtype));
        }
        details
    }

    fn logical_children(&self, class: &ClassId) -> Vec<(&'static str, ClassId)> {
        let Some(node_id) = self.choose_logical_node(class) else {
            return Vec::new();
        };
        let Some(node) = self.egraph.nodes.get(node_id) else {
            return Vec::new();
        };

        let ports: &[(&str, usize)] = logical_op_for(node.op.as_str())
            .map(|op| op.child_ports())
            .unwrap_or(&[]);

        ports
            .iter()
            .filter_map(|(port, index)| {
                let child = child_class(self.egraph, node, *index)?;
                if child == *class {
                    None
                } else {
                    Some((*port, child))
                }
            })
            .collect()
    }

    fn logical_op_name(&self, class: &ClassId) -> Option<String> {
        let node_id = self.choose_logical_node(class)?;
        let node = self.egraph.nodes.get(node_id)?;
        Some(node.op.clone())
    }

    fn choose_logical_node(&self, class: &ClassId) -> Option<&NodeId> {
        let node_ids = self.class_nodes.get(class)?;
        for op in luminal::logical_op::built_in_logical_ops() {
            if let Some(node_id) = node_ids.iter().find(|node_id| {
                self.egraph
                    .nodes
                    .get(*node_id)
                    .is_some_and(|node| node.op == op.egglog_constructor())
            }) {
                return Some(node_id);
            }
        }
        choose_render_node(self.egraph, node_ids, None)
    }

    /// ONE LEVEL OF INDIRECTION (Austin's ruling 2026-08-07): a logical
    /// value renders as its op over its CHILDREN'S LABELS (let-names or
    /// stable e-class ids) — "id-123 = sqrt(id-122)" — never the nested
    /// tree. The logical value graph is a convergent DAG; full-tree
    /// expansion was exponential (the 2-layer decoder build hang).
    fn readable_logical_expr(&self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> String {
        self.readable_logical_expr_depth(class, visiting, 1)
    }

    fn readable_logical_expr_depth(
        &self,
        class: &ClassId,
        visiting: &mut HashSet<ClassId>,
        depth: usize,
    ) -> String {
        if depth == 0 || !visiting.insert(class.clone()) {
            return self.logical_label(class);
        }

        let rendered = self
            .choose_logical_node(class)
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                Some(match logical_op_for(node.op.as_str()) {
                    Some(op) => op.readable_expr(
                        node,
                        &mut LogicalRenderCtx {
                            renderer: self,
                            visiting,
                            depth: depth - 1,
                        },
                    ),
                    None => self.render_node(node_id, 16),
                })
            })
            .unwrap_or_else(|| class.to_string());

        visiting.remove(class);
        rendered
    }

    fn layout_label(&self, class: &ClassId) -> String {
        let canonical = self.canonical_layout_short_label(class);
        match self.class_let_name(class) {
            Some(name) if name != canonical => format!("{name}\n{canonical}"),
            Some(name) => name,
            None => canonical,
        }
    }

    fn canonical_layout_short_label(&self, class: &ClassId) -> String {
        if let Some(summary) = self.contiguous_layout_summary(class) {
            summary
        } else if let Some(summary) = self.left_major_layout_summary(class) {
            summary
        } else if let Some(summary) = self.strided_layout_summary(class) {
            summary
        } else if self
            .node_with_op(class, "ElementOffsetExpressionLayoutLit")
            .is_some()
        {
            "ElementOffset".to_string()
        } else if self
            .node_with_op(class, "BitOffsetExpressionLayoutLit")
            .is_some()
        {
            "BitOffset".to_string()
        } else {
            self.render_class_prefer(class, 6, None)
        }
    }

    fn canonical_layout(&self, class: &ClassId) -> String {
        if let Some(summary) = self.contiguous_layout_inline(class) {
            return summary;
        }
        if let Some(summary) = self.left_major_layout_inline(class) {
            return summary;
        }
        if let Some(summary) = self.strided_layout_inline(class) {
            return summary;
        }
        for op in LAYOUT_RENDER_OPS {
            if let Some(rendered) = self.render_class_with_op(class, 16, op) {
                return rendered;
            }
        }
        self.render_class_prefer(class, 16, None)
    }

    fn layout_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = self.class_details(class);
        details.push(("canonical".to_string(), self.canonical_layout(class)));
        if let Some((shape, bits)) = self.contiguous_layout_shape_bits(class) {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        } else if let Some((shape, bits)) = self.left_major_layout_shape_bits(class) {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        } else if let Some((shape, _strides, bits)) = self.strided_layout_shape_strides_bits(class)
        {
            details.push(("shape".to_string(), shape));
            details.push(("bits".to_string(), bits));
        }
        if let Some(contiguous) =
            self.render_class_with_op(class, 16, "RightMajorContiguousElementLayoutLit")
        {
            details.push(("right_major_contiguous".to_string(), contiguous));
        }
        if let Some(left_major) =
            self.render_class_with_op(class, 16, "LeftMajorContiguousElementLayoutLit")
        {
            details.push(("left_major_contiguous".to_string(), left_major));
        }
        if let Some(strided) = self.strided_layout_inline(class) {
            details.push(("strided".to_string(), strided));
        }
        if let Some(element_offset) =
            self.render_class_with_op(class, 16, "ElementOffsetExpressionLayoutLit")
        {
            details.push(("element_offset".to_string(), element_offset));
        }
        details.push((
            "bit_offset".to_string(),
            self.render_class_with_op(class, 32, "BitOffsetExpressionLayoutLit")
                .unwrap_or_else(|| "<none>".to_string()),
        ));
        details
    }

    fn contiguous_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.contiguous_layout_shape_bits(class)?;
        Some(format!("RightMajorContiguous\n{shape}\n{bits}b"))
    }

    fn contiguous_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.contiguous_layout_shape_bits(class)?;
        Some(format!("RightMajorContiguous(shape={shape}, bits={bits})"))
    }

    fn contiguous_layout_shape_bits(&self, class: &ClassId) -> Option<(String, String)> {
        let node_id = self.node_with_op(class, "RightMajorContiguousElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let bits_class = child_class(self.egraph, node, 1)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn left_major_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.left_major_layout_shape_bits(class)?;
        Some(format!("LeftMajorContiguous\n{shape}\n{bits}b"))
    }

    fn left_major_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, bits) = self.left_major_layout_shape_bits(class)?;
        Some(format!("LeftMajorContiguous(shape={shape}, bits={bits})"))
    }

    fn left_major_layout_shape_bits(&self, class: &ClassId) -> Option<(String, String)> {
        let node_id = self.node_with_op(class, "LeftMajorContiguousElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let bits_class = child_class(self.egraph, node, 1)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn strided_layout_summary(&self, class: &ClassId) -> Option<String> {
        let (shape, strides, bits) = self.strided_layout_shape_strides_bits(class)?;
        Some(format!("Strided\n{shape}\n{strides}\n{bits}b"))
    }

    fn strided_layout_inline(&self, class: &ClassId) -> Option<String> {
        let (shape, strides, bits) = self.strided_layout_shape_strides_bits(class)?;
        Some(format!(
            "Strided(shape={shape}, strides={strides}, bits={bits})"
        ))
    }

    fn strided_layout_shape_strides_bits(
        &self,
        class: &ClassId,
    ) -> Option<(String, String, String)> {
        let node_id = self.node_with_op(class, "StridedElementLayoutLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let shape_class = child_class(self.egraph, node, 0)?;
        let strides_class = child_class(self.egraph, node, 1)?;
        let bits_class = child_class(self.egraph, node, 2)?;
        Some((
            self.readable_shape(&shape_class)
                .unwrap_or_else(|| self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))),
            self.readable_expr_list_display(&strides_class)
                .unwrap_or_else(|| {
                    self.render_class_prefer(&strides_class, 16, Some("IntAffineExprCons"))
                }),
            self.readable_bit_width(&bits_class),
        ))
    }

    fn readable_shape(&self, class: &ClassId) -> Option<String> {
        let node_id = self.node_with_op(class, "ShapeLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let dims_class = child_class(self.egraph, node, 0)?;
        self.readable_expr_list_display(&dims_class)
    }

    // Widths are wrapped terms (BitWidthLit i64); labels want the bare
    // number, so unwrap one level before rendering.
    fn readable_bit_width(&self, class: &ClassId) -> String {
        self.node_with_op(class, "BitWidthLit")
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                let value_class = child_class(self.egraph, node, 0)?;
                Some(self.render_class_prefer(&value_class, 2, None))
            })
            .unwrap_or_else(|| self.render_class_prefer(class, 4, None))
    }

    fn readable_index_map(&self, class: &ClassId) -> Option<String> {
        let node_id = self.node_with_op(class, "IndexMapLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let exprs_class = child_class(self.egraph, node, 0)?;
        self.readable_expr_list_display(&exprs_class)
    }

    fn readable_expr_list_display(&self, class: &ClassId) -> Option<String> {
        let exprs = self.readable_expr_list(class, &mut HashSet::new())?;
        Some(format!("[{}]", exprs.join(", ")))
    }

    fn readable_expr_list(
        &self,
        class: &ClassId,
        visiting: &mut HashSet<ClassId>,
    ) -> Option<Vec<String>> {
        if !visiting.insert(class.clone()) {
            return None;
        }

        let result = if self.node_with_op(class, "IntExprNil").is_some() {
            Some(Vec::new())
        } else {
            let cons_id = self.node_with_op(class, "IntExprCons")?;
            let cons = self.egraph.nodes.get(cons_id)?;
            let head_class = child_class(self.egraph, cons, 0)?;
            let tail_class = child_class(self.egraph, cons, 1)?;
            let mut dims = vec![self.readable_expr(&head_class, &mut HashSet::new())];
            dims.extend(self.readable_expr_list(&tail_class, visiting)?);
            Some(dims)
        };

        visiting.remove(class);
        result
    }

    fn readable_expr(&self, class: &ClassId, visiting: &mut HashSet<ClassId>) -> String {
        if !visiting.insert(class.clone()) {
            return class.to_string();
        }

        let rendered = self
            .node_with_op(class, "IntLit")
            .and_then(|node_id| {
                let node = self.egraph.nodes.get(node_id)?;
                let value_class = child_class(self.egraph, node, 0)?;
                Some(self.render_class_prefer(&value_class, 2, None))
            })
            .or_else(|| {
                // A dynamic-dimension symbol renders as its bare name. After
                // IntLit: a pinned IntVar's class holds both, and the concrete
                // value reads better.
                self.node_with_op(class, "IntVar").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let name_class = child_class(self.egraph, node, 0)?;
                    Some(
                        self.render_class_prefer(&name_class, 2, None)
                            .trim_matches('"')
                            .to_string(),
                    )
                })
            })
            .or_else(|| {
                // Compact per user: v<axis> — the owner shape rides in
                // tooltips, not labels (a coordinate variable's identity is
                // (shape, axis); child 0 is the owner, child 1 the axis).
                self.node_with_op(class, "CoordVar").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let axis_class = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "v{}",
                        self.render_class_prefer(&axis_class, 2, None)
                    ))
                })
            })
            .or_else(|| {
                self.node_with_op(class, "IntAdd").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let lhs = child_class(self.egraph, node, 0)?;
                    let rhs = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "({} + {})",
                        self.readable_expr(&lhs, visiting),
                        self.readable_expr(&rhs, visiting)
                    ))
                })
            })
            .or_else(|| {
                self.node_with_op(class, "IntMul").and_then(|node_id| {
                    let node = self.egraph.nodes.get(node_id)?;
                    let lhs = child_class(self.egraph, node, 0)?;
                    let rhs = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "({} * {})",
                        self.readable_expr(&lhs, visiting),
                        self.readable_expr(&rhs, visiting)
                    ))
                })
            })
            .or_else(|| {
                // The division family and lattice pair render function-style:
                // the rounding mode / lattice direction is the constructor's
                // identity, so it must stay visible.
                [
                    "IntTruncDiv",
                    "IntTruncRem",
                    "IntCeilDiv",
                    "IntMin",
                    "IntMax",
                ]
                .iter()
                .zip(["tdiv", "trem", "ceildiv", "min", "max"])
                .find_map(|(op, name)| {
                    let node_id = self.node_with_op(class, op)?;
                    let node = self.egraph.nodes.get(node_id)?;
                    let dividend = child_class(self.egraph, node, 0)?;
                    let divisor = child_class(self.egraph, node, 1)?;
                    Some(format!(
                        "{name}({}, {})",
                        self.readable_expr(&dividend, visiting),
                        self.readable_expr(&divisor, visiting)
                    ))
                })
            })
            .unwrap_or_else(|| self.render_class_prefer(class, 8, None));

        visiting.remove(class);
        rendered
    }

    fn class_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let mut details = Vec::new();
        if let Some(typ) = self.class_type(class) {
            details.push(("type".to_string(), typ));
        }
        if let Some(name) = self.class_let_name(class) {
            details.push(("let".to_string(), name));
        }
        details
    }

    /// The `LayoutTensorLit` member the DETAILS table describes: the
    /// first one whose logical and layout children both read. Distinct
    /// from [`Self::layout_tensor_parts`], which stops at the first
    /// `LayoutTensorLit` even when a child is missing; sharing this one
    /// selector is what keeps [`Self::layout_tensor_shape_dtype`]
    /// byte-identical to the `find_detail` lookup it replaced.
    fn layout_tensor_detail_parts(&self, class: &ClassId) -> Option<(ClassId, ClassId)> {
        for node_id in self.class_nodes.get(class)? {
            let Some(node) = self.egraph.nodes.get(node_id) else {
                continue;
            };
            if node.op != "LayoutTensorLit" {
                continue;
            }
            let Some(logical_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            let Some(layout_class) = child_class(self.egraph, node, 1) else {
                continue;
            };
            return Some((logical_class, layout_class));
        }
        None
    }

    /// The `shape` and `dtype` a value's info carries — EXACTLY what
    /// `find_detail(&layout_tensor_details(class), "shape"/"dtype")`
    /// returned: the same member node, the logical side first, the
    /// layout side only when the logical side offers none. Split out so
    /// the two eager fields no longer force the whole details table
    /// (whose `canonical`/`bit_offset` entries are full-depth renders).
    fn layout_tensor_shape_dtype(&self, class: &ClassId) -> (Option<String>, Option<String>) {
        let Some((logical_class, layout_class)) = self.layout_tensor_detail_parts(class) else {
            return (None, None);
        };
        (
            self.logical_shape(&logical_class)
                .or_else(|| self.layout_shape(&layout_class)),
            self.logical_dtype(&logical_class)
                .or_else(|| self.layout_dtype(&layout_class)),
        )
    }

    fn layout_tensor_details(&self, class: &ClassId) -> Vec<(String, String)> {
        let Some((logical_class, layout_class)) = self.layout_tensor_detail_parts(class) else {
            return Vec::new();
        };

        {
            let mut details = self.class_details(class);
            details.push(("logical".to_string(), self.logical_label(&logical_class)));
            details.push(("logical_eclass".to_string(), logical_class.to_string()));
            if let Some(shape) = self.logical_shape(&logical_class) {
                details.push(("shape".to_string(), shape));
            }
            if let Some(dtype) = self.logical_dtype(&logical_class) {
                details.push(("dtype".to_string(), dtype));
            }
            if !details.iter().any(|(key, _)| key == "shape") {
                if let Some(shape) = self.layout_shape(&layout_class) {
                    details.push(("shape".to_string(), shape));
                }
            }
            if !details.iter().any(|(key, _)| key == "dtype") {
                if let Some(dtype) = self.layout_dtype(&layout_class) {
                    details.push(("dtype".to_string(), dtype));
                }
            }
            details.push(("layout".to_string(), self.canonical_layout(&layout_class)));
            details.push(("layout_eclass".to_string(), layout_class.to_string()));
            details
        }
    }

    fn logical_shape(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" {
                continue;
            }
            let shape_class = child_class(self.egraph, node, 1)?;
            return Some(
                self.readable_shape(&shape_class).unwrap_or_else(|| {
                    self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))
                }),
            );
        }
        None
    }

    fn logical_dtype(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if node.op != "LogicalTensorInputLit" {
                continue;
            }
            let dtype_class = child_class(self.egraph, node, 2)?;
            return Some(self.render_class_prefer(&dtype_class, 4, None));
        }
        None
    }

    // ---- numeric geometry (the executor/translator surface): literal shapes
    // walked straight off the e-graph terms, never parsed from rendered
    // strings. `None` = symbolic or absent — consumers bail loudly.

    /// A primitive i64 class: any member node whose op parses as an integer.
    fn numeric_i64(&self, class: &ClassId) -> Option<i64> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            if let Ok(value) = node.op.parse::<i64>() {
                return Some(value);
            }
        }
        None
    }

    /// An IntExpr class holding a literal: the IntLit member's value.
    fn numeric_int_expr(&self, class: &ClassId) -> Option<i64> {
        let node_id = self.node_with_op(class, "IntLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        let value_class = child_class(self.egraph, node, 0)?;
        self.numeric_i64(&value_class)
    }

    /// A fully-literal IntExprList, walked cons-by-cons BY E-CLASS (a class's
    /// serialized representative may be a function node — the R8 lesson).
    fn numeric_expr_list(&self, class: &ClassId) -> Option<Vec<i64>> {
        let mut dims = Vec::new();
        let mut current = class.clone();
        loop {
            if let Some(node_id) = self.node_with_op(&current, "IntExprCons").cloned() {
                let node = self.egraph.nodes.get(&node_id)?;
                dims.push(self.numeric_int_expr(&child_class(self.egraph, node, 0)?)?);
                current = child_class(self.egraph, node, 1)?;
            } else if self.node_with_op(&current, "IntExprNil").is_some() {
                return Some(dims);
            } else {
                return None;
            }
        }
    }

    /// A Shape class with fully-literal dims.
    fn numeric_shape(&self, class: &ClassId) -> Option<Vec<i64>> {
        let node_id = self.node_with_op(class, "ShapeLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        self.numeric_expr_list(&child_class(self.egraph, node, 0)?)
    }

    /// The numeric `BufferLit` value of a BufferId class, when literal.
    fn numeric_buffer_lit(&self, class: &ClassId) -> Option<i64> {
        let node_id = self.node_with_op(class, "BufferLit")?;
        let node = self.egraph.nodes.get(node_id)?;
        self.numeric_i64(&child_class(self.egraph, node, 0)?)
    }

    /// The layout class's extents, numerically (mirrors [`Self::layout_shape`]).
    fn numeric_layout_dims(&self, class: &ClassId) -> Option<Vec<i64>> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            let shape_child = match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit"
                | "LeftMajorContiguousElementLayoutLit"
                | "StridedElementLayoutLit" => 0,
                "ElementOffsetExpressionLayoutLit" | "BitOffsetExpressionLayoutLit" => 1,
                _ => continue,
            };
            let shape_class = child_class(self.egraph, node, shape_child)?;
            return self.numeric_shape(&shape_class);
        }
        None
    }

    /// The layout class's element bit width, numerically (mirrors
    /// [`Self::layout_dtype`]'s constructor positions).
    fn numeric_layout_bits(&self, class: &ClassId) -> Option<i64> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            let bits_child = match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit" | "LeftMajorContiguousElementLayoutLit" => 1,
                "StridedElementLayoutLit"
                | "ElementOffsetExpressionLayoutLit"
                | "BitOffsetExpressionLayoutLit" => 2,
                _ => continue,
            };
            let bits_class = child_class(self.egraph, node, bits_child)?;
            let lit = self.node_with_op(&bits_class, "BitWidthLit")?;
            let lit_node = self.egraph.nodes.get(lit)?;
            return self.numeric_i64(&child_class(self.egraph, lit_node, 0)?);
        }
        None
    }

    fn layout_shape(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit"
                | "LeftMajorContiguousElementLayoutLit"
                | "StridedElementLayoutLit" => {
                    let shape_class = child_class(self.egraph, node, 0)?;
                    return Some(self.readable_shape(&shape_class).unwrap_or_else(|| {
                        self.render_class_prefer(&shape_class, 16, Some("ShapeLit"))
                    }));
                }
                _ => {}
            }
        }
        None
    }

    fn layout_dtype(&self, class: &ClassId) -> Option<String> {
        for node_id in self.class_nodes.get(class)? {
            let node = self.egraph.nodes.get(node_id)?;
            match node.op.as_str() {
                "RightMajorContiguousElementLayoutLit" | "LeftMajorContiguousElementLayoutLit" => {
                    let bits_class = child_class(self.egraph, node, 1)?;
                    return Some(self.readable_bit_width(&bits_class));
                }
                "StridedElementLayoutLit" => {
                    let bits_class = child_class(self.egraph, node, 2)?;
                    return Some(self.readable_bit_width(&bits_class));
                }
                _ => {}
            }
        }
        None
    }

    // ---- tooltip builders (ported from the former GraphBuilder; they
    // live on the renderer, not the extractor, so a DEFERRED text field
    // can run them with nothing but the render ctx — see
    // `Extractor::lazy_text`).

    fn source_lines(&self, eclass: Option<&ClassId>, enode: Option<&NodeId>) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(eclass) = eclass {
            push_detail(&mut lines, "eclass", eclass);
            if let Some(typ) = self.class_type(eclass) {
                push_detail(&mut lines, "type", typ);
            }
            if let Some(name) = self.class_let_name(eclass) {
                push_detail(&mut lines, "let", name);
            }
        }
        if let Some(enode) = enode {
            push_detail(&mut lines, "enode", enode);
        }
        lines
    }

    fn render_buffer_id(&self, class: &ClassId) -> String {
        self.render_class_prefer(class, 3, Some("BufferLit"))
    }

    fn render_layout_tensor_list(&self, classes: &[ClassId]) -> String {
        let items = classes
            .iter()
            .enumerate()
            .map(|(index, class)| format!("{index}:{}", self.layout_tensor_summary(class)))
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{items}]")
    }

    fn layout_tensor_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &self.layout_tensor_details(class));
        join_tooltip(lines)
    }

    fn logical_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &self.logical_details(class));
        join_tooltip(lines)
    }

    fn layout_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_details(&mut lines, &self.layout_details(class));
        join_tooltip(lines)
    }

    fn buffer_tensor_tooltip(
        &self,
        class: &ClassId,
        source_enode: Option<&NodeId>,
        tensor: &ClassId,
        buffer_id: &ClassId,
    ) -> String {
        let mut lines = self.source_lines(Some(class), source_enode);
        push_detail(&mut lines, "tensor_eclass", tensor);
        push_detail(&mut lines, "buffer_id_eclass", buffer_id);
        if let Some((literal_tensor, literal_buffer_id)) = self.buffer_tensor_parts(class) {
            push_detail(&mut lines, "literal_tensor_eclass", literal_tensor);
            push_detail(&mut lines, "literal_buffer_id_eclass", literal_buffer_id);
        }
        push_detail(&mut lines, "buffer_id", self.render_buffer_id(buffer_id));
        push_details(&mut lines, &self.layout_tensor_details(tensor));
        join_tooltip(lines)
    }

    fn buffer_id_tooltip(&self, class: &ClassId) -> String {
        let mut lines = self.source_lines(Some(class), None);
        push_detail(&mut lines, "value", self.render_buffer_id(class));
        join_tooltip(lines)
    }

    fn output_tooltip(&self, class: &ClassId, source_enode: Option<&NodeId>) -> String {
        join_tooltip(self.source_lines(Some(class), source_enode))
    }

    /// The op tooltip, from a [`OpTooltipSeed`] rather than the `Plan`:
    /// the plan is extractor-internal and gone by the time a deferred
    /// tooltip renders, so `ensure_value` clones the handful of fields
    /// this reads.
    fn op_tooltip(&self, seed: &OpTooltipSeed) -> String {
        let mut lines = Vec::new();
        push_detail(&mut lines, "selected_output_eclass", &seed.class);
        if let Some(op_eclass) = &seed.source_eclass {
            push_detail(&mut lines, "op_eclass", op_eclass);
        }
        if let Some(enode) = &seed.source_enode {
            push_detail(&mut lines, "concrete_enode", enode);
        }
        if let Some(index) = seed.selected_output_index {
            push_detail(&mut lines, "selected_output_index", index);
        }
        push_detail(&mut lines, "heuristic_cost", seed.heuristic_cost);
        push_details(&mut lines, &self.layout_tensor_details(&seed.class));
        if !seed.input_list.is_empty() {
            push_detail(
                &mut lines,
                "input_layout_tensors",
                self.render_layout_tensor_list(&seed.input_list),
            );
        }
        if !seed.output_list.is_empty() {
            push_detail(
                &mut lines,
                "output_layout_tensors",
                self.render_layout_tensor_list(&seed.output_list),
            );
        }
        for meta in &seed.metadata {
            let value = if is_layout_metadata(meta.name) {
                self.canonical_layout(&meta.class)
            } else if meta.name == "shape" {
                self.readable_shape(&meta.class).unwrap_or_else(|| {
                    self.render_class_prefer(
                        &meta.class,
                        metadata_render_depth(meta.name),
                        metadata_preferred_op(meta.name),
                    )
                })
            } else if meta.name == "index_map" {
                self.readable_index_map(&meta.class).unwrap_or_else(|| {
                    self.render_class_prefer(
                        &meta.class,
                        metadata_render_depth(meta.name),
                        metadata_preferred_op(meta.name),
                    )
                })
            } else {
                self.render_class_prefer(
                    &meta.class,
                    metadata_render_depth(meta.name),
                    metadata_preferred_op(meta.name),
                )
            };
            push_detail(&mut lines, meta.name, value);
        }
        join_tooltip(lines)
    }
}

impl<'a> Extractor<'a> {
    fn plan(&self, class: &ClassId) -> Result<&Plan> {
        self.memo
            .get(class)
            .and_then(Option::as_ref)
            .with_context(|| format!("no extracted plan for eclass {class}"))
    }

    fn build_extracted_graph(&self, roots: &[ClassId]) -> Result<ExtractedGraph> {
        let mut builder = IrBuilder {
            extractor: self,
            dag: DiGraph::new(),
            value_producer: HashMap::new(),
            op_nodes: HashMap::new(),
        };
        let mut outputs = Vec::with_capacity(roots.len());
        for root in roots {
            outputs.push(builder.emit_output(root)?);
        }
        Ok(ExtractedGraph {
            dag: builder.dag,
            outputs,
        })
    }

    // ---- structured info builders for the Layout IR DAG ----

    fn layout_tensor_info(&self, class: &ClassId) -> LayoutTensorInfo {
        let renderer = self.renderer();
        let label = renderer.layout_tensor_label(class);
        // `shape`/`dtype` stay eager but no longer force the details
        // table: `layout_tensor_shape_dtype` reproduces exactly what
        // `find_detail(&details, ..)` returned.
        let (shape, dtype) = renderer.layout_tensor_shape_dtype(class);
        let tooltip = {
            let class = class.clone();
            self.lazy_text(move |renderer| renderer.layout_tensor_tooltip(&class))
        };

        let (logical, layout) = match self.layout_tensor_parts(class) {
            Some((logical_class, layout_class)) => (
                self.logical_info(&logical_class, &mut HashSet::new(), 4),
                self.layout_info(&layout_class),
            ),
            None => (
                LogicalInfo {
                    eclass: class.clone(),
                    label: LazyText::eager(class.to_string()),
                    tooltip: LazyText::default(),
                    op: None,
                    children: Vec::new(),
                },
                LayoutInfo {
                    eclass: class.clone(),
                    label: LazyText::eager(class.to_string()),
                    tooltip: LazyText::default(),
                },
            ),
        };

        let (dims, element_bits, dtype_enum) = match self.layout_tensor_parts(class) {
            Some((logical_class, layout_class)) => {
                (
                    renderer.numeric_layout_dims(&layout_class),
                    renderer.numeric_layout_bits(&layout_class),
                    // Dtype comes from the LOGICAL side (dtype-of rows) —
                    // never the layout, which is pure placement by the
                    // preamble contract.
                    self.with_dtype_index(|index| index.get(&logical_class).copied()),
                )
            }
            None => (None, None, None),
        };

        LayoutTensorInfo {
            eclass: class.clone(),
            label,
            tooltip,
            shape,
            dtype,
            dtype_enum,
            dims,
            element_bits,
            logical,
            layout,
        }
    }

    /// DEPTH-CAPPED (2026-08-07): this is RENDERING data (labels +
    /// tooltips), and the logical value graph is a convergent DAG —
    /// residual connections reuse values, so eager full-ancestry TREE
    /// expansion is exponential in depth (the 2-layer decoder build hang:
    /// every op output re-expanded the whole model history). Local
    /// context is what a tooltip needs; the cap bounds the walk.
    fn logical_info(
        &self,
        class: &ClassId,
        visiting: &mut HashSet<ClassId>,
        depth: usize,
    ) -> LogicalInfo {
        let label = {
            let class = class.clone();
            self.lazy_text(move |renderer| renderer.logical_label(&class))
        };
        let tooltip = {
            let class = class.clone();
            self.lazy_text(move |renderer| renderer.logical_tooltip(&class))
        };
        let children = if depth > 0 && visiting.insert(class.clone()) {
            let children = self
                .logical_children(class)
                .into_iter()
                .map(|(port, child)| {
                    (
                        port.to_string(),
                        self.logical_info(&child, visiting, depth - 1),
                    )
                })
                .collect();
            visiting.remove(class);
            children
        } else {
            Vec::new()
        };
        let op = if children.is_empty() {
            None
        } else {
            self.logical_op_name(class)
        };
        LogicalInfo {
            eclass: class.clone(),
            label,
            tooltip,
            op,
            children,
        }
    }

    fn logical_op_name(&self, class: &ClassId) -> Option<String> {
        self.renderer().logical_op_name(class)
    }

    fn layout_info(&self, class: &ClassId) -> LayoutInfo {
        LayoutInfo {
            eclass: class.clone(),
            label: {
                let class = class.clone();
                self.lazy_text(move |renderer| renderer.layout_label(&class))
            },
            tooltip: {
                let class = class.clone();
                self.lazy_text(move |renderer| renderer.layout_tooltip(&class))
            },
        }
    }

    fn buffer_info(
        &self,
        buffer_tensor_class: &ClassId,
        buffer_tensor_enode: Option<&NodeId>,
        tensor_class: &ClassId,
        buffer_id_class: &ClassId,
    ) -> BufferInfo {
        let renderer = self.renderer();
        let tensor_label = renderer
            .class_let_name(buffer_tensor_class)
            .unwrap_or_else(|| buffer_tensor_class.to_string());
        let tensor_tooltip = {
            let (class, enode, tensor, buffer_id) = (
                buffer_tensor_class.clone(),
                buffer_tensor_enode.cloned(),
                tensor_class.clone(),
                buffer_id_class.clone(),
            );
            self.lazy_text(move |renderer| {
                renderer.buffer_tensor_tooltip(&class, enode.as_ref(), &tensor, &buffer_id)
            })
        };
        let rendered = renderer.render_buffer_id(buffer_id_class);
        let id_label = match renderer.class_let_name(buffer_id_class) {
            Some(name) if name != rendered => format!("{name}\n{rendered}"),
            Some(name) => name,
            None => rendered,
        };
        let id_tooltip = {
            let class = buffer_id_class.clone();
            self.lazy_text(move |renderer| renderer.buffer_id_tooltip(&class))
        };
        let lit = renderer.numeric_buffer_lit(buffer_id_class);
        BufferInfo {
            tensor_eclass: buffer_tensor_class.clone(),
            tensor_label,
            tensor_tooltip,
            id_eclass: buffer_id_class.clone(),
            id_label,
            id_tooltip,
            access: self.buffer_access(buffer_id_class),
            freed_by: self.buffer_freed_by(buffer_id_class),
            lit,
        }
    }

    /// Look up the buffer's contents permission via the `buffer-access-of`
    /// function: its entries serialize as `buffer-access-of` nodes (child 0 =
    /// the BufferId) living in the e-class of their Access value. `None` =
    /// the program declared nothing, which input-program validation rejects
    /// for every buffer — declarations are always explicit.
    fn buffer_access(&self, buffer_id_class: &ClassId) -> Option<Access> {
        for (node_id, node) in &self.egraph.nodes {
            if node.subsumed || node.op != "buffer-access-of" {
                continue;
            }
            let Some(arg_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            if &arg_class != buffer_id_class {
                continue;
            }
            let access_class = self.egraph.nid_to_cid(node_id);
            if self
                .renderer()
                .node_with_op(access_class, "ReadOnly")
                .is_some()
            {
                return Some(Access::ReadOnly);
            }
            return Some(Access::ReadWrite);
        }
        None
    }

    /// Look up storage deallocation responsibility via `buffer-freed-by`.
    /// `None` = undeclared, which input-program validation rejects for every
    /// buffer — there is deliberately no default.
    fn buffer_freed_by(&self, buffer_id_class: &ClassId) -> Option<FreedBy> {
        for (node_id, node) in &self.egraph.nodes {
            if node.subsumed || node.op != "buffer-freed-by" {
                continue;
            }
            let Some(arg_class) = child_class(self.egraph, node, 0) else {
                continue;
            };
            if &arg_class != buffer_id_class {
                continue;
            }
            let freed_class = self.egraph.nid_to_cid(node_id);
            if self
                .renderer()
                .node_with_op(freed_class, "ProgramFrees")
                .is_some()
            {
                return Some(FreedBy::Program);
            }
            return Some(FreedBy::Caller);
        }
        None
    }

    fn output_tooltip(&self, class: &ClassId, source_enode: Option<&NodeId>) -> String {
        self.renderer().output_tooltip(class, source_enode)
    }
}

/// Walks the memoized plans into the [`ExtractedGraph`] DAG. Nodes are ops plus
/// input/output boundaries; edges carry the LayoutTensor value flowing between
/// a producer and a consumer.
struct IrBuilder<'e, 'a> {
    extractor: &'e Extractor<'a>,
    dag: ExtractedDag,
    /// Value e-class -> the node that produces it (an op or an input boundary).
    value_producer: HashMap<ClassId, NodeIndex>,
    /// Op identity -> its node, so multi-output ops are emitted exactly once.
    op_nodes: HashMap<(ClassId, NodeId), NodeIndex>,
}

impl<'e, 'a> IrBuilder<'e, 'a> {
    /// Whether an instantiated op's output slot BELONGS to its output class.
    /// Without a genome every slot is claimed (the deterministic extractor's
    /// first-emission behavior). Under a genome, a slot is claimed only if
    /// the genome maps that class to exactly this enode and slot — the
    /// genome, not emission order, decides ownership.
    fn slot_claimed(&self, output: &ClassId, enode: &NodeId, slot: usize) -> bool {
        match self.extractor.genome.as_ref() {
            None => true,
            Some(genome) => genome
                .choices
                .get(output)
                .is_some_and(|choice| &choice.enode == enode && choice.output_index == slot),
        }
    }

    fn ensure_value(&mut self, class: &ClassId) -> Result<NodeIndex> {
        if let Some(index) = self.value_producer.get(class) {
            return Ok(*index);
        }
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::Input(info) => {
                let value = self.extractor.layout_tensor_info(class);
                let buffer = self.extractor.buffer_info(
                    &info.buffer_tensor_class,
                    Some(&info.buffer_tensor_enode),
                    class,
                    &info.buffer_id_class,
                );
                let index = self
                    .dag
                    .add_node(ExtractedNode::BufferInput(Box::new(InputNode {
                        value,
                        buffer,
                    })));
                self.value_producer.insert(class.clone(), index);
                Ok(index)
            }
            PlanKind::LayoutIr(op) => {
                let op_eclass = plan
                    .source_eclass
                    .clone()
                    .with_context(|| format!("op plan for {class} missing source eclass"))?;
                let source_enode = plan
                    .source_enode
                    .clone()
                    .with_context(|| format!("op plan for {class} missing source enode"))?;
                let key = (op_eclass.clone(), source_enode.clone());
                if let Some(index) = self.op_nodes.get(&key) {
                    self.value_producer.insert(class.clone(), *index);
                    return Ok(*index);
                }

                let outputs = plan
                    .output_list
                    .iter()
                    .enumerate()
                    .map(|(slot, output)| {
                        let mut info = self.extractor.layout_tensor_info(output);
                        if !self.slot_claimed(output, &source_enode, slot) {
                            // WASTE DESTINATION (genome walks only): this
                            // instance computes the slot, but the genome
                            // assigned the class to a different producer. A
                            // fresh synthetic value identity (the poison-id
                            // idiom) makes bufferize allocate scratch instead
                            // of double-writing the class's real home.
                            info.eclass =
                                ClassId::from(format!("genome$waste${source_enode}${slot}"));
                            info.label = format!("{} (unclaimed)", info.label);
                        }
                        info
                    })
                    .collect::<Vec<_>>();
                let inputs = plan
                    .children
                    .iter()
                    .map(|child| OpInput {
                        port: child.port.clone(),
                        value: child.class.clone(),
                    })
                    .collect::<Vec<_>>();
                let tooltip = {
                    let seed = OpTooltipSeed {
                        class: class.clone(),
                        source_eclass: plan.source_eclass.clone(),
                        source_enode: plan.source_enode.clone(),
                        selected_output_index: plan.selected_output_index,
                        heuristic_cost: plan.heuristic_cost,
                        input_list: plan.input_list.clone(),
                        output_list: plan.output_list.clone(),
                        metadata: plan.metadata.clone(),
                    };
                    self.extractor
                        .lazy_text(move |renderer| renderer.op_tooltip(&seed))
                };
                let node = OpNode {
                    op: op.clone(),
                    provenance: luminal::layout_ir::Provenance::Extracted {
                        op_eclass,
                        source_enode: source_enode.clone(),
                        selected_output_index: plan.selected_output_index.unwrap_or(0),
                    },
                    inputs,
                    outputs,
                    tooltip,
                    heuristic_cost: plan.heuristic_cost,
                };
                let index = self.dag.add_node(ExtractedNode::LayoutOp(node));
                self.op_nodes.insert(key, index);
                for (slot, output) in plan.output_list.iter().enumerate() {
                    if self.slot_claimed(output, &source_enode, slot) {
                        self.value_producer.insert(output.clone(), index);
                    }
                }
                self.value_producer.insert(class.clone(), index);

                for child in &plan.children {
                    let producer = self.ensure_value(&child.class)?;
                    self.dag.add_edge(
                        producer,
                        index,
                        ExtractedEdge {
                            value: child.class.clone(),
                            port: child.port.clone(),
                        },
                    );
                }
                Ok(index)
            }
            other => bail!("expected value-producing plan at {class}, found {other:?}"),
        }
    }

    fn emit_output(&mut self, class: &ClassId) -> Result<NodeIndex> {
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::BufferOutputLit => {
                let outputs_list = only_child(&plan, "outputs")?;
                let mut slots = Vec::new();
                self.collect_output_buffers(&outputs_list, 0, &mut slots)?;

                let label = self
                    .extractor
                    .class_let_name(class)
                    .unwrap_or_else(|| class.to_string());
                let tooltip = self
                    .extractor
                    .output_tooltip(class, plan.source_enode.as_ref());
                let output_slots = slots
                    .iter()
                    .map(|(index, value, _, buffer)| OutputSlot {
                        index: *index,
                        value: value.clone(),
                        buffer: buffer.clone(),
                    })
                    .collect();
                let output_index = self.dag.add_node(ExtractedNode::BufferOutput(OutputNode {
                    eclass: class.clone(),
                    label,
                    tooltip,
                    slots: output_slots,
                }));
                for (index, value, producer, _) in &slots {
                    self.dag.add_edge(
                        *producer,
                        output_index,
                        ExtractedEdge {
                            value: value.clone(),
                            port: format!("out {index}"),
                        },
                    );
                }
                Ok(output_index)
            }
            other => bail!("expected BufferOutputLit at extracted root {class}, found {other:?}"),
        }
    }

    fn collect_output_buffers(
        &mut self,
        list_class: &ClassId,
        index: usize,
        slots: &mut Vec<(usize, ClassId, NodeIndex, BufferInfo)>,
    ) -> Result<usize> {
        let plan = self.extractor.plan(list_class)?.clone();
        match &plan.kind {
            PlanKind::BufferTensorCons => {
                let head = only_child(&plan, "head")?;
                let tail = only_child(&plan, "tail")?;
                self.emit_output_buffer(&head, index, slots)?;
                self.collect_output_buffers(&tail, index + 1, slots)
            }
            PlanKind::BufferTensorNil => Ok(index),
            other => bail!("expected BufferTensorList at {list_class}, found {other:?}"),
        }
    }

    fn emit_output_buffer(
        &mut self,
        class: &ClassId,
        index: usize,
        slots: &mut Vec<(usize, ClassId, NodeIndex, BufferInfo)>,
    ) -> Result<()> {
        let plan = self.extractor.plan(class)?.clone();
        match &plan.kind {
            PlanKind::BufferTensorLit {
                buffer_id_class, ..
            } => {
                let tensor = only_child(&plan, "tensor")?;
                let producer = self.ensure_value(&tensor)?;
                let buffer = self.extractor.buffer_info(
                    class,
                    plan.source_enode.as_ref(),
                    &tensor,
                    buffer_id_class,
                );
                slots.push((index, tensor, producer, buffer));
                Ok(())
            }
            other => bail!("expected BufferTensorLit at output {class}, found {other:?}"),
        }
    }
}

fn only_child(plan: &Plan, port: &str) -> Result<ClassId> {
    plan.children
        .iter()
        .find(|child| child.port == port)
        .map(|child| child.class.clone())
        .with_context(|| format!("missing {port} child for {:?}", plan.kind))
}

fn push_detail(lines: &mut Vec<String>, key: &str, value: impl ToString) {
    let value = tooltip_value(value);
    if value.is_empty() {
        return;
    }
    let line = format!("{key}={value}");
    if !lines.contains(&line) {
        lines.push(line);
    }
}

fn push_details(lines: &mut Vec<String>, details: &[(String, String)]) {
    for (key, value) in details {
        push_detail(lines, key, value);
    }
}

fn join_tooltip(lines: Vec<String>) -> String {
    lines.join("\n")
}

fn tooltip_value(value: impl ToString) -> String {
    const MAX_FIELD_CHARS: usize = 2_000;

    let mut value = value
        .to_string()
        .replace('"', "'")
        .replace(['\n', '\r', '\t'], " ");
    if value.chars().count() <= MAX_FIELD_CHARS {
        return value;
    }

    value = value.chars().take(MAX_FIELD_CHARS).collect();
    value.push_str("...");
    value
}

fn metadata_preferred_op(name: &str) -> Option<&'static str> {
    match name {
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout" => {
            Some("RightMajorContiguousElementLayoutLit")
        }
        "shape" => Some("ShapeLit"),
        "index_map" => Some("IndexMapLit"),
        "buffer_id" => Some("BufferLit"),
        _ => None,
    }
}

fn metadata_render_depth(name: &str) -> usize {
    match name {
        "index_map" => 32,
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout" | "shape" => 16,
        "axis" | "buffer_id" => 4,
        _ => 12,
    }
}

fn is_layout_metadata(name: &str) -> bool {
    matches!(
        name,
        "layout" | "out_layout" | "add_out_layout" | "mul_out_layout"
    )
}

fn plan_label(plan: &Plan) -> String {
    match &plan.kind {
        PlanKind::Input(input) => format!("Input:{}", input.logical_name),
        PlanKind::BufferOutputLit => "BufferOutputLit".to_string(),
        PlanKind::BufferTensorCons => "BufferTensorCons".to_string(),
        PlanKind::BufferTensorNil => "BufferTensorNil".to_string(),
        PlanKind::BufferTensorLit { logical_name, .. } => format!("BufferTensorLit:{logical_name}"),
        PlanKind::LayoutIr(op) => op.label().to_string(),
    }
}

fn class_nodes(egraph: &EGraph) -> HashMap<ClassId, Vec<NodeId>> {
    let mut classes: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for (node_id, node) in &egraph.nodes {
        if node.subsumed || node.op == "[...]" {
            continue;
        }
        classes
            .entry(node.eclass.clone())
            .or_default()
            .push(node_id.clone());
    }
    classes
}

fn render_class_nodes(egraph: &EGraph) -> HashMap<ClassId, Vec<NodeId>> {
    let mut classes: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for (node_id, node) in &egraph.nodes {
        if node.op == "[...]" {
            continue;
        }
        classes
            .entry(node.eclass.clone())
            .or_default()
            .push(node_id.clone());
    }
    classes
}

/// THE INPUT-PRODUCER FACT (cleanup stratum, ruling 2026-09-02): the
/// LayoutTensorOp classes the `cleanup` ruleset marked — every op whose
/// OUTPUT list holds a boundary input's LEAF layout tensor, the one a
/// `BufferInputLit` binding names. Those are precisely the classes
/// `collect_input_terminals` seeds as `PlanKind::Input`, so the
/// invariant is "a class the plan reads as a launch-time leaf offers no
/// producer".
///
/// A boundary input is a LEAF: it exists at launch. No producer of that
/// class is ever needed, and none can ever be cheaper than reading the
/// leaf — but sound unions do mint such producers (the cuBLASLt
/// double-transpose collapse puts a view op's output into the input's
/// own layout-tensor class), and a zero-cost view then wins the
/// `is_better` tie against the zero-cost terminal on `plan_label`,
/// erasing the BufferInput and handing bufferize a cyclic graph.
///
/// The extractor does not filter on this fact: `Extractor::new_with_matchers`
/// already drops every producer row of a class it seeds as an input
/// terminal, which is the same cut expressed in the extractor's own
/// vocabulary. The fact is read as a TRIPWIRE instead — one leaf notion,
/// checked — so that an estate that marks an op the extractor still
/// plans as produced fails loudly rather than silently re-minting the
/// cycle.
///
/// The relation serializes like every other egglog fact: one node per
/// row whose op is the relation name and whose child 0 is the argument.
/// We read the FACT — enode-anchored and class-invariant — and never the
/// spelling: `(subsume (LayoutTensorOpLit ...))` retires only the
/// generic term, while each runtime op's `match_functional.egg` unions
/// its own constructor into that same class and a core rule cannot name
/// them all.
fn collect_input_producer_ops(egraph: &EGraph) -> HashSet<ClassId> {
    let mut ops = HashSet::new();
    for node in egraph.nodes.values() {
        if node.subsumed || node.op != "input-producer" {
            continue;
        }
        if let Some(op_class) = child_class(egraph, node, 0) {
            ops.insert(op_class);
        }
    }
    ops
}

fn collect_op_specs(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> (
    HashMap<ClassId, Vec<OpSpec>>,
    HashMap<ClassId, Vec<ProducerRef>>,
) {
    let mut op_specs: HashMap<ClassId, Vec<OpSpec>> = HashMap::new();
    let mut producer_index: HashMap<ClassId, Vec<ProducerRef>> = HashMap::new();

    for (op_class, node_ids) in class_nodes {
        for node_id in node_ids {
            let Some(node) = egraph.nodes.get(node_id) else {
                continue;
            };
            if node.op != "LayoutTensorOpLit" {
                continue;
            }

            let Some(input_list_class) = child_class(egraph, node, 0) else {
                continue;
            };
            let Some(output_list_class) = child_class(egraph, node, 1) else {
                continue;
            };
            let Some(inputs) = layout_tensor_list_items(
                egraph,
                class_nodes,
                &input_list_class,
                &mut HashSet::new(),
            ) else {
                continue;
            };
            let Some(outputs) = layout_tensor_list_items(
                egraph,
                class_nodes,
                &output_list_class,
                &mut HashSet::new(),
            ) else {
                continue;
            };

            let specs = op_specs.entry(op_class.clone()).or_default();
            if specs
                .iter()
                .any(|spec| spec.inputs == inputs && spec.outputs == outputs)
            {
                continue;
            }

            let spec_index = specs.len();
            specs.push(OpSpec {
                inputs,
                outputs: outputs.clone(),
            });

            for (output_index, output_class) in outputs.into_iter().enumerate() {
                producer_index
                    .entry(output_class)
                    .or_default()
                    .push(ProducerRef {
                        op_class: op_class.clone(),
                        spec_index,
                        output_index,
                    });
            }
        }
    }

    (op_specs, producer_index)
}

fn layout_tensor_list_items(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
    list_class: &ClassId,
    visiting: &mut HashSet<ClassId>,
) -> Option<Vec<ClassId>> {
    if !visiting.insert(list_class.clone()) {
        return None;
    }

    let node_ids = class_nodes.get(list_class)?;
    for node_id in node_ids {
        let node = egraph.nodes.get(node_id)?;
        match node.op.as_str() {
            "LayoutTensorNil" => {
                visiting.remove(list_class);
                return Some(Vec::new());
            }
            "LayoutTensorCons" => {
                let head = child_class(egraph, node, 0)?;
                let tail = child_class(egraph, node, 1)?;
                let mut items = layout_tensor_list_items(egraph, class_nodes, &tail, visiting)?;
                items.insert(0, head);
                visiting.remove(list_class);
                return Some(items);
            }
            _ => {}
        }
    }

    visiting.remove(list_class);
    None
}

fn output_root_classes(egraph: &EGraph) -> Vec<ClassId> {
    let mut roots = egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferOutputLit")
        .map(|node| node.eclass.clone())
        .collect::<Vec<_>>();
    roots.sort_by_key(ToString::to_string);
    roots.dedup();
    roots
}

fn collect_output_buffer_classes(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> HashSet<ClassId> {
    let mut output_buffers = HashSet::new();
    let mut visited_lists = HashSet::new();

    for node in egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferOutputLit")
    {
        let Some(list_class) = child_class(egraph, node, 0) else {
            continue;
        };
        collect_buffer_list(
            egraph,
            class_nodes,
            &list_class,
            &mut visited_lists,
            &mut output_buffers,
        );
    }

    output_buffers
}

fn collect_input_buffer_classes(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
) -> HashSet<ClassId> {
    let mut input_buffers = HashSet::new();
    let mut visited_lists = HashSet::new();

    for node in egraph
        .nodes
        .values()
        .filter(|node| !node.subsumed && node.op == "BufferInputLit")
    {
        let Some(list_class) = child_class(egraph, node, 0) else {
            continue;
        };
        collect_buffer_list(
            egraph,
            class_nodes,
            &list_class,
            &mut visited_lists,
            &mut input_buffers,
        );
    }

    input_buffers
}

fn collect_buffer_list(
    egraph: &EGraph,
    class_nodes: &HashMap<ClassId, Vec<NodeId>>,
    list_class: &ClassId,
    visited_lists: &mut HashSet<ClassId>,
    buffers: &mut HashSet<ClassId>,
) {
    if !visited_lists.insert(list_class.clone()) {
        return;
    }

    let Some(node_ids) = class_nodes.get(list_class) else {
        return;
    };

    for node_id in node_ids {
        let Some(node) = egraph.nodes.get(node_id) else {
            continue;
        };
        if node.op != "BufferTensorCons" {
            continue;
        }
        if let Some(buffer_class) = child_class(egraph, node, 0) {
            buffers.insert(buffer_class);
        }
        if let Some(tail_class) = child_class(egraph, node, 1) {
            collect_buffer_list(egraph, class_nodes, &tail_class, visited_lists, buffers);
        }
    }
}

fn collect_input_terminals(
    render: &RenderCtx,
    output_buffer_classes: &HashSet<ClassId>,
    input_buffer_classes: &HashSet<ClassId>,
) -> HashMap<ClassId, InputInfo> {
    let mut terminals = HashMap::new();
    let egraph = &render.egraph;
    // Through the session's renderer, so these session-start renders
    // populate (and later hit) the same memo as everything else.
    let renderer = render.renderer();
    let has_explicit_inputs = !input_buffer_classes.is_empty();

    for (node_id, node) in egraph
        .nodes
        .iter()
        .filter(|(_, node)| !node.subsumed && node.op == "BufferTensorLit")
    {
        if has_explicit_inputs {
            if !input_buffer_classes.contains(&node.eclass) {
                continue;
            }
        } else if output_buffer_classes.contains(&node.eclass) {
            continue;
        }
        let Some(layout_tensor_class) = child_class(egraph, node, 0) else {
            continue;
        };
        let Some(buffer_id_class) = child_class(egraph, node, 1) else {
            continue;
        };
        terminals
            .entry(layout_tensor_class.clone())
            .or_insert_with(|| InputInfo {
                buffer_tensor_class: node.eclass.clone(),
                buffer_tensor_enode: node_id.clone(),
                buffer_id_class: buffer_id_class.clone(),
                logical_name: renderer
                    .logical_name_from_layout_tensor(&layout_tensor_class)
                    .unwrap_or_else(|| layout_tensor_class.to_string()),
            });
    }

    terminals
}

fn choose_render_node<'a>(
    egraph: &'a EGraph,
    node_ids: &'a [NodeId],
    preferred_op: Option<&str>,
) -> Option<&'a NodeId> {
    if let Some(preferred_op) = preferred_op {
        if let Some(node_id) = node_ids.iter().find(|node_id| {
            egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == preferred_op)
        }) {
            return Some(node_id);
        }
    }

    if let Some(node_id) = node_ids.iter().find(|node_id| {
        egraph
            .nodes
            .get(*node_id)
            .is_some_and(|node| node.children.is_empty() && is_simple_literal(&node.op))
    }) {
        return Some(node_id);
    }

    for render_op in RENDER_PREFERRED_OPS {
        if let Some(node_id) = node_ids.iter().find(|node_id| {
            egraph
                .nodes
                .get(*node_id)
                .is_some_and(|node| node.op == *render_op)
        }) {
            return Some(node_id);
        }
    }

    node_ids.iter().min_by_key(|node_id| {
        egraph
            .nodes
            .get(*node_id)
            .map(|node| node.op.as_str())
            .unwrap_or_default()
    })
}

const RENDER_PREFERRED_OPS: &[&str] = &[
    "BufferLit",
    "LogicalIdLit",
    "LayoutTensorLit",
    "LogicalTensorInputLit",
    "LogicalTensorNamed",
    "RightMajorContiguousElementLayoutLit",
    "LeftMajorContiguousElementLayoutLit",
    "StridedElementLayoutLit",
    "IndexMapLit",
    "ShapeLit",
    "IntExprCons",
    "IntExprNil",
    "IntLit",
    "IntVar",
    "CoordVar",
    "F32",
    "F64",
    "Int",
    "Bool",
];

const LAYOUT_RENDER_OPS: &[&str] = &[
    "RightMajorContiguousElementLayoutLit",
    "LeftMajorContiguousElementLayoutLit",
    "StridedElementLayoutLit",
    "ElementOffsetExpressionLayoutLit",
    "BitOffsetExpressionLayoutLit",
];

fn is_simple_literal(op: &str) -> bool {
    op.parse::<i64>().is_ok() || op.starts_with('"')
}

/// Per-axis stride classification destructured from a chain-carrying
/// StridedElementLayoutLit (Austin's Vec<Option<...>> contract,
/// 2026-08-04). Outermost axis first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainStride {
    /// Live axis: the stride is this IntExpr class (literal or symbolic).
    Expr(ClassId),
    /// Stride-1 axis (the x*1 fold collapses the summand to its bare
    /// coordinate, so no stride node exists to point at).
    Unit,
    /// Zero contribution on an axis NOT provably extent-1: a DETERMINED
    /// broadcast — the stride must be 0, this is not a choice.
    Zero,
}

/// Walk a layout class's chain-strided spelling into per-axis strides.
/// `None` per slot = the axis is provably extent-1, so the stride is a
/// FREE PARAMETER: each consumer picks its own default (the e-graph
/// never answers this question — the impossibility result). Whole-call
/// `None` = the class has no chain-strided spelling, ranks disagree, or
/// a slot is opaque (e.g. an accumulated diagonal summand) — fail
/// closed, never guess.
pub fn chain_strides(egraph: &EGraph, layout: &ClassId) -> Option<Vec<Option<ChainStride>>> {
    let mut class_nodes: HashMap<ClassId, Vec<NodeId>> = HashMap::new();
    for node_id in egraph.nodes.keys() {
        class_nodes
            .entry(egraph.nid_to_cid(node_id).clone())
            .or_default()
            .push(node_id.clone());
    }
    let find = |class: &ClassId, op: &str| -> Option<NodeId> {
        class_nodes
            .get(class)?
            .iter()
            .find(|node_id| egraph.nodes.get(*node_id).is_some_and(|node| node.op == op))
            .cloned()
    };
    let numeric = |class: &ClassId| -> Option<i64> {
        let lit = find(class, "IntLit")?;
        let value_class = child_class(egraph, egraph.nodes.get(&lit)?, 0)?;
        class_nodes
            .get(&value_class)?
            .iter()
            .find_map(|node_id| egraph.nodes.get(node_id)?.op.parse::<i64>().ok())
    };

    let strided = find(layout, "StridedElementLayoutLit")?;
    let strided_node = egraph.nodes.get(&strided)?;
    let shape_class = child_class(egraph, strided_node, 0)?;
    let chain_class = child_class(egraph, strided_node, 1)?;

    // Dims, outermost first.
    let shape_lit = find(&shape_class, "ShapeLit")?;
    let mut dims = Vec::new();
    let mut cursor = child_class(egraph, egraph.nodes.get(&shape_lit)?, 0)?;
    loop {
        if let Some(cons) = find(&cursor, "IntExprCons") {
            let node = egraph.nodes.get(&cons)?;
            dims.push(child_class(egraph, node, 0)?);
            cursor = child_class(egraph, node, 1)?;
        } else if find(&cursor, "IntExprNil").is_some() {
            break;
        } else {
            return None;
        }
    }

    // Chain, outermost first, in lockstep with dims.
    let mut out = Vec::new();
    let mut cursor = chain_class;
    let mut axis = 0usize;
    loop {
        if let Some(cons) = find(&cursor, "IntAffineExprCons") {
            let node = egraph.nodes.get(&cons)?;
            let summand = child_class(egraph, node, 0)?;
            let extent = dims.get(axis).and_then(numeric);
            let slot = if numeric(&summand) == Some(0) {
                // Zero contribution. Provably-1 axis: free parameter.
                // Otherwise the broadcast stride 0 is determined.
                if extent == Some(1) {
                    None
                } else {
                    Some(ChainStride::Zero)
                }
            } else if find(&summand, "CoordVar").is_some() {
                Some(ChainStride::Unit)
            } else if let Some(stride) = class_nodes.get(&summand).and_then(|nodes| {
                nodes.iter().find_map(|node_id| {
                    let candidate = egraph.nodes.get(node_id)?;
                    if candidate.op != "IntMul" {
                        return None;
                    }
                    let coord = child_class(egraph, candidate, 0)?;
                    if find(&coord, "CoordVar").is_some() {
                        child_class(egraph, candidate, 1)
                    } else {
                        None
                    }
                })
            }) {
                Some(ChainStride::Expr(stride))
            } else {
                return None; // opaque slot: fail closed
            };
            out.push(slot);
            axis += 1;
            cursor = child_class(egraph, node, 1)?;
        } else if find(&cursor, "IntAffineExprNil").is_some() {
            break;
        } else {
            return None;
        }
    }
    if out.len() != dims.len() {
        return None;
    }
    Some(out)
}

fn child_class(egraph: &EGraph, node: &Node, index: usize) -> Option<ClassId> {
    let child_id = node.children.get(index)?;
    egraph.nodes.get(child_id).map(|child| child.eclass.clone())
}

#[cfg(test)]
mod render_memo_tests {
    //! The render memo's two obligations (ruling 2026-09-01): it must
    //! return the string the uncached walk would have returned, and it
    //! must not deadlock on its own recursion.

    use egraph_serialize::{ClassId, EGraph, Node, NodeId};

    use super::{ClassRenderer, RenderCtx};

    /// A chain `a4(a3(a2(a1(leaf, leaf), leaf), leaf), leaf)` — every level
    /// shares one leaf class, which is exactly the shape (a convergent DAG)
    /// that made the uncached renderer exponential in depth.
    fn shared_child_chain() -> EGraph {
        let mut egraph = EGraph::default();
        let mut add = |id: &str, op: &str, children: Vec<&str>| {
            egraph.add_node(
                NodeId::from(id),
                Node {
                    op: op.to_string(),
                    children: children.into_iter().map(NodeId::from).collect(),
                    eclass: ClassId::from(format!("class-{id}")),
                    cost: ordered_float::NotNan::new(1.0).unwrap(),
                    subsumed: false,
                },
            );
        };
        add("leaf", "Leaf", vec![]);
        add("a1", "A1", vec!["leaf", "leaf"]);
        add("a2", "A2", vec!["a1", "leaf"]);
        add("a3", "A3", vec!["a2", "leaf"]);
        add("a4", "A4", vec!["a3", "leaf"]);
        egraph
    }

    /// The memo returns the SAME text on the second call, and adds no
    /// entries — the second call is pure lookup. (Identity is what the
    /// ruling turns on: this text feeds `stable_key`, which breaks plan
    /// selection ties.)
    #[test]
    fn memo_is_output_identical_and_does_no_second_walk() {
        let ctx = RenderCtx::new(&shared_child_chain());
        let renderer = ctx.renderer();
        let root = ClassId::from("class-a4");

        let first = renderer.render_class_prefer(&root, 3, None);
        let filled = ctx.memo.borrow().len();
        let second = renderer.render_class_prefer(&root, 3, None);

        assert_eq!(first, second, "a memo hit must reproduce the render");
        assert_eq!(
            ctx.memo.borrow().len(),
            filled,
            "the second call walked the graph again instead of hitting the memo"
        );
        assert!(filled > 0, "nothing was memoized");
        // Spelled out, so a change to the render grammar has to face this
        // test rather than silently re-electing plans.
        assert_eq!(first, "A4(A3(A2(class-a1, class-leaf), Leaf), Leaf)");
    }

    /// Depth is part of the key: the same class at a shallower depth is a
    /// DIFFERENT string, and a memo that dropped depth would serve the
    /// deep answer to a shallow caller.
    #[test]
    fn memo_keys_on_depth_and_preference() {
        let ctx = RenderCtx::new(&shared_child_chain());
        let renderer = ctx.renderer();
        let root = ClassId::from("class-a4");

        assert_eq!(
            renderer.render_class_prefer(&root, 1, None),
            "A4(class-a3, class-leaf)"
        );
        assert_eq!(
            renderer.render_class_prefer(&root, 2, None),
            "A4(A3(class-a2, class-leaf), Leaf)"
        );
        assert_eq!(renderer.render_class_prefer(&root, 0, None), "class-a4");
    }

    /// `render_class_prefer` recurses into itself through `render_node`.
    /// Holding the memo's `Ref` across that call is a `BorrowMutError` at
    /// runtime, not a compile error, so a deep render is the guard.
    #[test]
    fn deep_render_does_not_double_borrow_the_memo() {
        let ctx = RenderCtx::new(&shared_child_chain());
        let rendered = ctx
            .renderer()
            .render_class_prefer(&ClassId::from("class-a4"), 32, None);
        assert!(rendered.starts_with("A4("));
    }

    /// Both renderer views over one ctx share the memo — the guarantee
    /// that a deferred tooltip rendered later reuses the session's work.
    #[test]
    fn views_over_one_ctx_share_the_memo() {
        let ctx = RenderCtx::new(&shared_child_chain());
        let root = ClassId::from("class-a4");
        let first = ClassRenderer::render_class_prefer(&ctx.renderer(), &root, 3, None);
        let filled = ctx.memo.borrow().len();
        let second = ClassRenderer::render_class_prefer(&ctx.renderer(), &root, 3, None);
        assert_eq!(first, second);
        assert_eq!(ctx.memo.borrow().len(), filled);
    }
}

/// THE STRIDE-DESTRUCTURING CONTRACT, moved here with `chain_strides`
/// (was `luminal::test_support::stage4b_probes`, #420/#422 rejoin Phase
/// 1): the extractor is runtime-owned now, so its contracts are pinned
/// beside the copy that holds them.
#[cfg(test)]
mod chain_stride_tests {
    /// Austin's stride-destructuring contract (chain world, 2026-08-04):
    /// live axes yield their stride class, stride-1 axes yield Unit, a
    /// zero slot on a live axis is the DETERMINED broadcast Some(Zero),
    /// and a provably extent-1 slot is None — the free parameter each
    /// consumer resolves for itself.
    #[test]
    fn chain_strides_destructure_contract() {
        use super::{ChainStride, chain_strides};
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil))))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 2) (IntExprCons (CoordVar osh 0) (IntExprNil))) psh) osh))
(let dsh (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 2) (IntExprNil)))))
(let d (RightMajorContiguousElementLayoutLit dsh (bits-of (F32))))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{body}", crate::assembled_program());
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &full)
            .expect("program runs");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;

        let by_let = |name: &str| {
            serialized
                .class_data
                .iter()
                .find(|(_, data)| data.extra.get("let").map(String::as_str) == Some(name))
                .map(|(class, _)| class.clone())
                .unwrap_or_else(|| panic!("let {name} not found"))
        };

        // Parent (2,3) right-major: strides [3, 1].
        let p = chain_strides(&serialized, &by_let("p")).expect("parent destructures");
        assert_eq!(p.len(), 2);
        assert!(matches!(p[0], Some(ChainStride::Expr(_))), "{p:?}");
        assert_eq!(p[1], Some(ChainStride::Unit), "{p:?}");

        // Degenerate (1,2): the extent-1 slot is the FREE parameter.
        let d = chain_strides(&serialized, &by_let("d")).expect("degenerate destructures");
        assert_eq!(
            d[0], None,
            "extent-1 slot must be the consumer's choice: {d:?}"
        );
        assert_eq!(d[1], Some(ChainStride::Unit), "{d:?}");

        // Broadcast view (2,5,3): [3, DETERMINED 0, 1].
        let v_logical = by_let("v");
        let view_layout = serialized
            .nodes
            .iter()
            .find_map(|(_, node)| {
                if node.op != "LayoutTensorLit" {
                    return None;
                }
                let logical = node.children.first()?;
                if serialized.nid_to_cid(logical) != &v_logical {
                    return None;
                }
                // The materializing copy pairs the SAME logical with an
                // RM target layout — skip it; the view's own layout is
                // the composed (non-contiguous) one.
                let layout = serialized.nid_to_cid(node.children.get(1)?).clone();
                let is_rm = serialized.nodes.iter().any(|(nid2, n2)| {
                    n2.op == "RightMajorContiguousElementLayoutLit"
                        && serialized.nid_to_cid(nid2) == &layout
                });
                if is_rm { None } else { Some(layout) }
            })
            .expect("view LayoutTensor exists");
        let v = chain_strides(&serialized, &view_layout).expect("view destructures");
        assert!(matches!(v[0], Some(ChainStride::Expr(_))), "{v:?}");
        assert_eq!(
            v[1],
            Some(ChainStride::Zero),
            "broadcast axis is determined: {v:?}"
        );
        assert_eq!(v[2], Some(ChainStride::Unit), "{v:?}");
    }
}
