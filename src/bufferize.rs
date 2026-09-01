//! Bufferization planner (one-shot-bufferization, in the spirit of MLIR).
//!
//! The [`ExtractedGraph`](crate::layout_ir::ExtractedGraph) is a single-valued
//! dataflow DAG whose interior `LayoutOp` values are *not yet backed by storage*,
//! and whose `BufferInput` / `BufferOutput` boundaries are already pinned to
//! caller-specified buffers. [`bufferize`] assigns a [`BufferId`] to *every*
//! value and lowers the graph — out of place, source untouched — to a
//! [`BufferIrGraph`] of storage events.
//!
//! # The conflict engine
//!
//! The analysis ([`Analyzer`]) walks ops bottom-up (MLIR's default heuristic)
//! and, for every operand an op *declares* an in-place candidate (a `Must`
//! edge in [`Bufferizable::alias_info`]), admits or rejects reuse (the DPS
//! rewrite pass
//! gives every built-in a destination-tied form, so real plans mix in-place
//! chains, seeded outputs, and out-of-place copies):
//!
//! * **Writability** — writing storage whose alias set includes a
//!   `Access::ReadOnly` binding is vetoed (`wouldCreateWriteToNonWritableBuffer`).
//! * **Read-after-write** — a read of any value aliasing the candidate must
//!   provably happen before the overwrite. Ordering is **DAG reachability**,
//!   never a chosen schedule: unordered ops are conservative conflicts, exactly
//!   like MLIR's dominance reasoning.
//! * **The same-USE exemption** — the candidate operand's own read never
//!   conflicts with its own write ("a use cannot conflict with itself"): a
//!   read-modify-write through one operand (an accumulator) is the canonical
//!   in-place case, its intra-op safety being the op's declared contract.
//! * **The cross-operand permit** — a same-op read through a *different*
//!   operand of aliasing storage is excused only by the op's own
//!   may-share permit (a `Sharing::May` edge in `alias_info`): unconditional
//!   and TRUSTED (the engine checks no layouts; ops discharge their
//!   preconditions at egglog match time). Cross-op reads are never excused
//!   this way.
//! * **Every rejection repairs** — a rejected tie's result gets its own
//!   fresh storage, initialized by the kernel's write (if the op writes it),
//!   preceded by a copy of the operand's bytes iff the result's pre-write
//!   contents must equal them (the op reads the operand, or writes nothing —
//!   a view). Every tie is repairable, so rejection is never an error.
//!
//! Cohabitation is sound because boundary values pinned to one buffer are
//! **pre-unioned** in the alias union-find (a prerequisite, not an
//! optimization). Undefined (poison) values are never READ at all: a program
//! in which any op reads an undefined operand, or any output slot binds an
//! undefined value, is rejected by [`validate_input_program`] as ill-formed —
//! so the conflict checks owe undefined contents nothing and carry no
//! exemptions. Undefined-ness remains a property of the *value*, never of
//! the buffer.
//!
//! # Rewrite
//!
//! Assignment gives equivalence classes their shared buffer (in-place) or fresh
//! system allocations. Rejected candidates get MLIR's `resolveConflicts`
//! treatment: the operand is **retargeted** to its tied result's fresh buffer,
//! with the old contents copied in first iff the op reads the operand. Output
//! slots materialize through boundary copies (skipped when the value already
//! lives in its destination), and every copy's overwrite is ordered after
//! unordered readers of its destination via **anti-dependency (WAR) edges**.
//!
//! The interface defaults still make every undeclared op fully out-of-place;
//! the DPS rewrite pass gives every built-in a destination-tied form, so real
//! plans mix in-place chains, seeded outputs, and copies. The unit tests
//! exercise every in-place path with mock ops; several are direct ports of
//! MLIR's `one-shot-bufferize-analysis.mlir` cases (non-writable function
//! args → the `Access::ReadOnly` tests). One deliberate DIVERGENCE from MLIR:
//! where MLIR exempts reads of undefined values (`read_of_undef`), we reject
//! the program in [`validate_input_program`] instead — undefined values are
//! write-targets only ([`test_support::EmptyOp`] tests pin the rejection).

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use egraph_serialize::ClassId;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};

use crate::layout_ir::{
    Access, ExtractedGraph, ExtractedNode, FreedBy, LayoutIrOp, must_ties, permits_sharing,
};

// =============================================================================
// Buffers and the bufferized graph
// =============================================================================

/// Storage identity. Boundary buffers are pinned by the program (identified by
/// the BufferId e-class they came from); interior buffers are minted by the
/// planner at compile time and owned by the system.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BufferId {
    /// Caller-specified buffer at an input/output boundary.
    Boundary(ClassId),
    /// Planner-minted, system-owned interior allocation.
    Allocated(u32),
}

/// Who is responsible for a buffer's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    /// Provided/consumed by the caller; never freed by the planner.
    Caller,
    /// Minted by the planner; freed by liveness after its last use.
    System,
}

/// A concrete buffer in the plan.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub id: BufferId,
    pub access: Access,
    /// Storage deallocation responsibility (declared for boundary buffers;
    /// forced to `Program` for planner-minted storage).
    pub freed_by: FreedBy,
    pub owner: Owner,
    pub label: String,
    /// Numeric extents of the values this buffer backs (annotated after
    /// lowering from the extraction's literal geometry; `None` = symbolic).
    pub dims: Option<Vec<i64>>,
    /// Element bit width (same provenance and contract as `dims`).
    pub element_bits: Option<i64>,
    /// The plan dtype, from the logical side's `dtype-of` row (same
    /// provenance/consumer contract as `dims`). The executor dispatches
    /// storage on THIS, never on width alone — `bits-of(Int)` equals
    /// `bits-of(F32)`, so width cannot carry the type (typed-buffers
    /// landing A, 2026-08-11).
    pub dtype: Option<crate::dtype::PlanDtype>,
    /// The numeric `BufferLit` id for boundary buffers — the key runtimes
    /// bind caller data by.
    pub lit: Option<i64>,
}

/// Where a program output value ends up: its value and the pinned buffer it was
/// materialized into.
#[derive(Debug, Clone)]
pub struct OutputBinding {
    pub index: usize,
    pub value: ClassId,
    pub buffer: BufferId,
}

/// A program input slot: a value pinned to a caller-owned buffer. Slot order
/// is declaration order (the vec position is the index).
#[derive(Debug, Clone)]
pub struct InputBinding {
    pub value: ClassId,
    pub buffer: BufferId,
}

/// How an edge constrains execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// A value flows from producer to consumer (dataflow dependence).
    Data,
    /// An anti-dependency (write-after-read): the target overwrites a buffer
    /// the source still reads, so the source must run first. Carries no value.
    Anti,
}

/// An edge in the buffer IR: the buffer flowing from producer to consumer, plus
/// the consumer's port label. Every value of the extracted graph has become a
/// buffer here (a BufferTensor); [`BufferIrGraph::value_buffer`] still records the
/// value→buffer map for cross-reference with the source graph.
#[derive(Debug, Clone)]
pub struct BufferEdge {
    pub buffer: BufferId,
    pub port: String,
    pub kind: EdgeKind,
}

/// A node in the buffer IR. `Compute` nodes are the original ops, now reading and
/// writing buffers; `BufferCopy` is the only genuinely new operation (bufferizer-
/// inserted materialization); the boundaries are pinned buffers.
#[derive(Debug, Clone)]
pub enum BufferNode {
    /// The program input boundary: every input slot's value pinned to its
    /// caller-owned buffer — ONE node, mirroring [`BufferNode::BufferOutput`]
    /// (and `BtNode::Input` upstream), so the two boundaries read the same.
    BufferInput { slots: Vec<InputBinding> },
    /// A lowered compute op, reading/writing buffers in operand / result order.
    Compute {
        op: Box<dyn crate::buffer_tensor_ir::BufferTensorIrOp>,
        reads: Vec<BufferId>,
        writes: Vec<BufferId>,
        /// Must-tie pairs `(operand, result)`, carried from the BufferTensor
        /// node for rendering: the plan surface must not query analysis-time
        /// contracts (`alias_info`), and slot names come from `OpSlotNames`.
        ties: Vec<(usize, usize)>,
    },
    /// A bufferizer-inserted copy materializing `src` into `dst`.
    BufferCopy { src: BufferId, dst: BufferId },
    /// A program output: each slot's value pinned into its destination buffer.
    BufferOutput { slots: Vec<OutputBinding> },
}

/// The out-of-place bufferization result: a new, independent dataflow graph whose
/// values are all backed by buffers and whose copies are first-class nodes. The
/// source [`ExtractedGraph`] is left untouched.
#[derive(Debug, Clone)]
pub struct BufferIrGraph {
    pub dag: DiGraph<BufferNode, BufferEdge>,
    /// Every distinct buffer in the plan, by id.
    pub buffers: HashMap<BufferId, Buffer>,
    /// The buffer holding each value (values collapse onto buffers via reuse).
    pub value_buffer: HashMap<ClassId, BufferId>,
    /// The `BufferOutput` node(s).
    pub outputs: Vec<NodeIndex>,
}

impl BufferIrGraph {
    fn buffer_name(&self, id: &BufferId) -> String {
        let label = self
            .buffers
            .get(id)
            .map(|buffer| buffer.label.replace('\n', " / "))
            .unwrap_or_default();
        match id {
            BufferId::Boundary(_) => format!("pinned[{label}]"),
            BufferId::Allocated(n) => {
                if label.is_empty() {
                    format!("alloc#{n}")
                } else {
                    format!("alloc#{n}[{label}]")
                }
            }
        }
    }

    fn sort_key(id: &BufferId) -> String {
        match id {
            BufferId::Boundary(eclass) => format!("0:{eclass}"),
            BufferId::Allocated(n) => format!("1:{n:06}"),
        }
    }

    fn names(&self, ids: &[BufferId]) -> String {
        ids.iter()
            .map(|id| self.buffer_name(id))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A deterministic, human-readable dump of the plan. Nodes are listed in
    /// creation (topological) order.
    pub fn summary(&self) -> String {
        let mut out = String::new();

        let mut buffers: Vec<&Buffer> = self.buffers.values().collect();
        buffers.sort_by_key(|buffer| Self::sort_key(&buffer.id));
        out.push_str(&format!("buffers ({}):\n", buffers.len()));
        for buffer in buffers {
            let owner = match buffer.owner {
                Owner::Caller => "caller",
                Owner::System => "system",
            };
            let writable = if buffer.access == Access::ReadWrite {
                "rw"
            } else {
                "ro"
            };
            out.push_str(&format!(
                "  {} [{owner}, {writable}]\n",
                self.buffer_name(&buffer.id)
            ));
        }

        let mut op_lines = Vec::new();
        let mut output_lines = Vec::new();
        for idx in self.dag.node_indices() {
            match &self.dag[idx] {
                BufferNode::BufferInput { .. } => {}
                BufferNode::Compute {
                    op, reads, writes, ..
                } => op_lines.push(format!(
                    "  {}: [{}] -> [{}]",
                    op.label(),
                    self.names(reads),
                    self.names(writes),
                )),
                BufferNode::BufferCopy { src, dst } => op_lines.push(format!(
                    "  BufferCopy: [{}] -> [{}]",
                    self.buffer_name(src),
                    self.buffer_name(dst),
                )),
                BufferNode::BufferOutput { slots } => {
                    let mut slots: Vec<&OutputBinding> = slots.iter().collect();
                    slots.sort_by_key(|slot| slot.index);
                    for slot in slots {
                        output_lines.push(format!(
                            "  out {} -> {}",
                            slot.index,
                            self.buffer_name(&slot.buffer)
                        ));
                    }
                }
            }
        }

        out.push_str(&format!("ops ({}):\n", op_lines.len()));
        for line in op_lines {
            out.push_str(&line);
            out.push('\n');
        }
        // Anti-dependency (WAR) ordering edges — part of the plan's semantics
        // (an executor must honor them), so part of its golden fingerprint.
        use petgraph::visit::EdgeRef;
        let mut anti_lines = Vec::new();
        for edge in self.dag.edge_references() {
            if edge.weight().kind != EdgeKind::Anti {
                continue;
            }
            let name = |idx: NodeIndex| match &self.dag[idx] {
                BufferNode::Compute { op, .. } => op.label().to_string(),
                BufferNode::BufferCopy { .. } => "BufferCopy".to_string(),
                BufferNode::BufferInput { .. } => "input".to_string(),
                BufferNode::BufferOutput { .. } => "output".to_string(),
            };
            anti_lines.push(format!(
                "  {} -> {} [{}]",
                name(edge.source()),
                name(edge.target()),
                self.buffer_name(&edge.weight().buffer)
            ));
        }
        out.push_str(&format!("anti ({}):\n", anti_lines.len()));
        for line in anti_lines {
            out.push_str(&line);
            out.push('\n');
        }

        out.push_str(&format!("outputs ({}):\n", output_lines.len()));
        for line in output_lines {
            out.push_str(&line);
            out.push('\n');
        }

        out
    }

    /// Render the buffer IR to Graphviz dot. Compute nodes show their op;
    /// edges are labeled with the buffer flowing along them (identity is
    /// spelled out, never color-coded — per the shared visual grammar).
    pub fn to_dot(&self) -> String {
        use petgraph::visit::EdgeRef;

        let mut out = String::from(
            "digraph BufferIR {\n  rankdir=LR;\n  graph [fontname=\"Helvetica\"];\n  node [fontname=\"Helvetica\"];\n  edge [fontname=\"Helvetica\"];\n",
        );

        // The shared visual grammar (see VisualKind::style): ops are squares,
        // everything else rounded; rose = buffer domain, blue = boundary.
        // Identity is never encoded in outlines or edge colors — buffer
        // names are spelled out in labels.
        //
        // Compute nodes render as slot tables: one slot per operand in
        // signature order, destination slots annotated with the result they
        // are tied to. Edges dock at the slot they feed (see the edge loop).
        for idx in self.dag.node_indices() {
            let n = idx.index();
            let (label, style, fill, border) = match &self.dag[idx] {
                BufferNode::BufferInput { .. } => {
                    // One boundary box, like Output — WHICH buffer each line
                    // supplies is the edge's label, not the box's.
                    ("Input".to_string(), "rounded,filled", "#dbeafe", "#2563eb")
                }
                BufferNode::Compute {
                    op, reads, ties, ..
                } => {
                    // Single-source rules only (<HR/> separators, borderless
                    // cells): a CELLBORDER next to the outer BORDER renders
                    // as an ugly double line.
                    let mut slots = String::new();
                    for operand in 0..reads.len() {
                        let port = op.operand_name(operand);
                        let tie = ties
                            .iter()
                            .find(|(o, _)| *o == operand)
                            .map(|(_, result)| format!(" ↔ out{result}"))
                            .unwrap_or_default();
                        if operand > 0 {
                            slots.push_str("<HR/>");
                        }
                        slots.push_str(&format!(
                            "<TR><TD PORT=\"{}\" ALIGN=\"LEFT\">{}{}</TD></TR>",
                            crate::layout_ir::slot_port(&port),
                            crate::layout_ir::html_escape(&port),
                            crate::layout_ir::html_escape(&tie),
                        ));
                    }
                    let body = if slots.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "<HR/><TR><TD CELLPADDING=\"0\"><TABLE BORDER=\"0\" CELLBORDER=\"0\" \
                             CELLSPACING=\"0\" CELLPADDING=\"3\">{slots}</TABLE></TD></TR>"
                        )
                    };
                    out.push_str(&format!(
                        "  n{n} [shape=\"plain\", label=<<TABLE BORDER=\"1\" CELLBORDER=\"0\" \
                         CELLSPACING=\"0\" CELLPADDING=\"2\" COLOR=\"#be185d\" BGCOLOR=\"#fce7f3\">\
                         <TR><TD CELLPADDING=\"4\"><B>{}</B></TD></TR>{body}</TABLE>>];\n",
                        crate::layout_ir::html_escape(op.label()),
                    ));
                    continue;
                }
                BufferNode::BufferCopy { src, dst } => (
                    // An op: a square, like every other op.
                    format!(
                        "BufferCopy\n{} → {}",
                        self.buffer_name(src),
                        self.buffer_name(dst)
                    ),
                    "filled",
                    "#fce7f3",
                    "#be185d",
                ),
                BufferNode::BufferOutput { .. } => {
                    ("Output".to_string(), "rounded,filled", "#dbeafe", "#2563eb")
                }
            };
            out.push_str(&format!(
                "  n{n} [label=\"{}\", shape=\"box\", style=\"{}\", fillcolor=\"{}\", color=\"{}\"];\n",
                dot_escape(&label),
                style,
                fill,
                border,
            ));
        }

        for edge in self.dag.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();
            let weight = edge.weight();
            // Dependency ordering (Anti edges and the alloc -> first-toucher
            // edge, whose port marks it) renders dashed; bytes render solid.
            let is_dependency = weight.kind == EdgeKind::Anti || weight.port == "alloc";
            let (c, style) = if is_dependency {
                ("#000000", "dashed")
            } else {
                ("#000000", "solid")
            };
            // Data edges dock at operand slots: the head at the consuming op's
            // slot for this port (the slot names the operand, so the label
            // keeps only the buffer), the tail at the producing op's slot for
            // the destination tied to the written buffer. Anti edges are
            // ordering constraints between whole ops — never docked.
            let mut head = String::new();
            let mut tail = String::new();
            let mut label = if is_dependency {
                dot_escape(&self.buffer_name(&weight.buffer))
            } else {
                format!(
                    "{}\\n{}",
                    dot_escape(&weight.port),
                    dot_escape(&self.buffer_name(&weight.buffer))
                )
            };
            if weight.kind == EdgeKind::Data {
                if let BufferNode::Compute { op, reads, .. } = &self.dag[edge.target()] {
                    if (0..reads.len()).any(|i| op.operand_name(i) == weight.port) {
                        head = format!(":{}:w", crate::layout_ir::slot_port(&weight.port));
                        label = dot_escape(&self.buffer_name(&weight.buffer));
                    }
                }
                if let BufferNode::Compute {
                    op, writes, ties, ..
                } = &self.dag[edge.source()]
                {
                    if let Some(result) = writes.iter().position(|b| b == &weight.buffer) {
                        if let Some(&(operand, _)) = ties.iter().find(|(_, r)| *r == result) {
                            tail = format!(
                                ":{}:e",
                                crate::layout_ir::slot_port(&op.operand_name(operand))
                            );
                        }
                    }
                }
            }
            out.push_str(&format!(
                "  n{source}{tail} -> n{target}{head} [label=\"{label}\", color=\"{c}\", \
                 fontcolor=\"{c}\", style=\"{style}\"];\n",
            ));
        }

        out.push_str("}\n");
        out
    }
}

fn dot_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

// =============================================================================
// Union-find over values (ClassId)
// =============================================================================

/// A value-keyed disjoint-set forest. Unknown values are lazily interned as their
/// own singleton, so callers never have to pre-register the universe.
#[derive(Default)]
struct UnionFind {
    parent: HashMap<ClassId, ClassId>,
}

impl UnionFind {
    fn find(&mut self, value: &ClassId) -> ClassId {
        let parent = match self.parent.get(value) {
            Some(parent) => parent.clone(),
            None => {
                self.parent.insert(value.clone(), value.clone());
                return value.clone();
            }
        };
        if &parent == value {
            return parent;
        }
        let root = self.find(&parent);
        self.parent.insert(value.clone(), root.clone());
        root
    }

    fn union(&mut self, a: &ClassId, b: &ClassId) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Deterministic representative (smaller id wins), independent of the order
        // unions happen in — keeps buffer assignment reproducible.
        let (keep, drop) = if ra <= rb { (ra, rb) } else { (rb, ra) };
        self.parent.insert(drop, keep);
    }

    fn same(&mut self, a: &ClassId, b: &ClassId) -> bool {
        self.find(a) == self.find(b)
    }
}

// =============================================================================
// Analysis (phase 1): decide in-place vs. out-of-place per operand
// =============================================================================

/// One op as the analyzer sees it: a position in the (forward) schedule, the
/// interface that declares its memory behavior, and its operand / result values.
/// Decoupled from the heavy [`ExtractedGraph`] node types so the analysis can be
/// unit-tested against mock interfaces.
struct AnalysisOp<'a> {
    position: usize,
    iface: &'a dyn LayoutIrOp,
    operands: Vec<ClassId>,
    results: Vec<ClassId>,
}

/// A read of a value, tagged with where it happens. Ordering against a write is
/// decided by [`Analyzer::happens_before`] via DAG reachability, so the `site` is
/// all we need: `None` marks a boundary-output read (logically at the end of the
/// program, after every op).
#[derive(Clone)]
struct ReadUse {
    value: ClassId,
    /// The op + operand doing the read, or `None` for a boundary-output read.
    site: Option<(usize, usize)>,
}

/// An in-place write COMMITTED by an earlier (bottom-up) decision: the operand
/// value whose storage it overwrites, tagged with its site. Kept so a later
/// candidate can re-check already-committed writers against the alias/read
/// sets its own union is about to merge — MLIR's interference check walks the
/// reads AND writes of both alias sets; a reads-only scan admits two unordered
/// writers of one storage as long as neither admission can see the other
/// (review-confirmed miscompile: a dead second writer of a shared destination
/// rides the merged class into the output buffer).
#[derive(Clone)]
struct WriteUse {
    value: ClassId,
    site: (usize, usize),
}

/// Per-value seeds the analyzer needs from the boundaries.
#[derive(Default)]
struct ValueFacts {
    /// Values whose buffer is read-only (a `Access::ReadOnly` boundary input).
    read_only: HashSet<ClassId>,
    /// Values read at the output boundary (live to `END_OF_PROGRAM`).
    output_values: Vec<ClassId>,
    /// The pinned storage identity of each boundary *input* value (BufferId
    /// e-class). Values sharing a pinned buffer cohabit storage, which the
    /// analyzer must know: they are pre-unioned in the alias union-find.
    /// Output slot values are obligations, not residents — never pinned here.
    pinned: HashMap<ClassId, ClassId>,
}

/// A destination-seeding proposal — the strictly-value-SSA replacement for
/// MLIR's EmptyTensorElimination, relocated into the assignment domain. An
/// output slot's value chains backward through dest ties (aliasing whose
/// result the op writes — descriptor-exact by the in-place convention) to a
/// chain-root poison destination; the proposal is "let that poison's
/// storage BE the slot's buffer", so the chain computes directly into the
/// output and the boundary copy disappears.
///
/// The proposal is evaluated, not trusted: pinning the poison to the slot's
/// buffer (pre-analysis) hands the cohabitation to the analyzer — the alias
/// pre-union makes the buffer's residents visible to every conflict check —
/// and the seed is applied (post-analysis) only if every `hops` decision was
/// admitted in place. A rejected seed changes nothing: the pin only ever
/// ADDED conservatism for other decisions. Monotonic by construction.
struct Seed {
    /// The chain-root destination value (produced by a poison/alloc-like op).
    poison: ClassId,
    /// The slot's pinned buffer: id e-class, declarations, display label.
    buffer_eclass: ClassId,
    access: Access,
    freed_by: FreedBy,
    buffer_label: String,
    /// The (op position, operand) decisions that must ALL be admitted in place
    /// for the seed to apply. Empty (the slot value is itself the poison)
    /// applies vacuously: the output is declared undefined.
    hops: Vec<(usize, usize)>,
}

/// Walk each output slot backward to its chain-root poison, if any. Seeds are
/// returned in slot order (ties on one poison resolve to the LOWEST slot) and
/// deduplicated per poison: one poison must never carry two proposals, or the
/// analysis could evaluate cohabitation with one buffer while assignment binds
/// the other, hiding the first buffer's resident conflicts.
fn find_seeds(
    graph: &ExtractedGraph,
    order: &[NodeIndex],
    analysis_ops: &[AnalysisOp],
) -> Vec<Seed> {
    let mut producer: HashMap<&ClassId, (usize, usize)> = HashMap::new();
    for op in analysis_ops {
        for (result, value) in op.results.iter().enumerate() {
            producer.insert(value, (op.position, result));
        }
    }

    let mut seeds: Vec<Seed> = Vec::new();
    let mut seen_poisons: HashSet<ClassId> = HashSet::new();
    for index in order {
        let ExtractedNode::BufferOutput(output) = &graph.dag[*index] else {
            continue;
        };
        for slot in &output.slots {
            // Seeding writes the slot's buffer through the whole chain; a Read
            // grant forbids every such write (only pass-through is legal).
            if slot.buffer.access() == Access::ReadOnly {
                continue;
            }
            let mut current = slot.value.clone();
            let mut hops = Vec::new();
            let poison = loop {
                let Some(&(position, result)) = producer.get(&current) else {
                    break None; // reached a boundary input: no chain root
                };
                let op = &analysis_ops[position];
                let is_root = op.operands.is_empty()
                    && !op.results.is_empty()
                    && (0..op.results.len()).all(|r| op.iface.result_is_undefined(r));
                if is_root {
                    break Some(current);
                }
                // Only a DEST tie may be crossed — one whose result the op
                // WRITES. Dest ties are descriptor-exact (same layout, same
                // extent as the operand) by the in-place convention, so
                // pinning the slot's storage backward through them stays
                // exact. A non-writing tie (a view) ends the chain
                // conservatively: the parent's extent may exceed the slot's.
                if !op.iface.result_writes_memory(result) {
                    break None;
                }
                // Follow the unique tie into this result; zero or several
                // ties (ill-formed — the analyzer's matching guard rejects
                // them later) end the chain conservatively.
                let ties = must_ties(op.iface);
                let mut tied = ties.iter().filter(|&&(_, tied)| tied == result);
                let (Some(&(operand, _)), None) = (tied.next(), tied.next()) else {
                    break None;
                };
                // Defensive: find_seeds runs BEFORE the analyzer's matching
                // guard, so an out-of-range tie ends the chain conservatively
                // here and errors loudly there (never a panic).
                let Some(value) = op.operands.get(operand) else {
                    break None;
                };
                hops.push((position, operand));
                current = value.clone();
            };
            if let Some(poison) = poison {
                if seen_poisons.insert(poison.clone()) {
                    seeds.push(Seed {
                        poison,
                        buffer_eclass: slot.buffer.id_eclass.clone(),
                        access: slot.buffer.access(),
                        freed_by: slot.buffer.freed_by(),
                        buffer_label: slot.buffer.id_label.clone(),
                        hops,
                    });
                }
            }
        }
    }
    seeds
}

/// The analysis outcome: the storage relation and the per-operand in-place
/// decisions.
pub(crate) struct Analysis {
    /// Values that WILL share an allocation: unioned on every ADMITTED aliasing
    /// decision — a DPS result joins its destination's class, a view rides its
    /// parent's class under its own layout. Buffer identity is a class lookup;
    /// nothing directional and no tie kind exists in analysis state (op
    /// contracts are consulted where exactness matters: `find_seeds` crosses
    /// only dest ties — aliasing whose result the op writes — and the planner
    /// folds metadata ops by their declared memory effects). This is the
    /// DECISION relation only — the
    /// analyzer's internal `alias` union-find additionally carries
    /// conservative proposal unions (boundary cohabitation, seed pins) that
    /// must never become storage: a rejected seed's poison falls back to a
    /// fresh allocation.
    storage: UnionFind,
    /// `(op position, operand index) -> bufferized in place?` Entries exist only
    /// for operands that declared aliasing (in-place candidates).
    pub(crate) in_place: HashMap<(usize, usize), bool>,
    /// Number of ops analyzed — lets the planner assert its recomputed op
    /// positions line up with the analysis positions.
    pub(crate) op_count: usize,
}

/// Runs the one-shot in-place analysis over the op DAG.
///
/// Invariant: `ops[i].position == i`. Positions are the analyzer's stable op ids,
/// used both as `ReadUse` sites and as indices into the reachability table.
struct Analyzer<'a> {
    ops: &'a [AnalysisOp<'a>],
    facts: &'a ValueFacts,
    reads: Vec<ReadUse>,
    /// `reachable[a]` = ops reachable *from* op `a` through dataflow dependence
    /// edges (i.e. ops that must run strictly after `a`). The DAG analogue of
    /// dominance: `a` happens-before `b` iff `b` is reachable from `a`.
    reachable: Vec<HashSet<usize>>,
    alias: UnionFind,
    /// The decision relation exported as [`Analysis::storage`].
    storage: UnionFind,
    in_place: HashMap<(usize, usize), bool>,
    /// Writes committed by earlier in-place admissions (see [`WriteUse`]).
    committed_writes: Vec<WriteUse>,
}

impl<'a> Analyzer<'a> {
    fn new(ops: &'a [AnalysisOp<'a>], facts: &'a ValueFacts) -> Self {
        debug_assert!(
            ops.iter().enumerate().all(|(i, op)| op.position == i),
            "AnalysisOp positions must be dense and match their slice index"
        );

        // Every read in the program. Operand reads happen at their op's position;
        // boundary-output reads happen at END_OF_PROGRAM (so a pinned output keeps
        // its value live to the very end).
        let mut reads = Vec::new();
        for op in ops {
            for (operand, value) in op.operands.iter().enumerate() {
                if op.iface.operand_reads_memory(operand) {
                    reads.push(ReadUse {
                        value: value.clone(),
                        site: Some((op.position, operand)),
                    });
                }
            }
        }
        for value in &facts.output_values {
            reads.push(ReadUse {
                value: value.clone(),
                site: None,
            });
        }

        let reachable = Self::compute_reachability(ops);

        // SOUNDNESS PREREQUISITE: boundary values pinned to the same buffer
        // cohabit storage, but no in-place *decision* ever relates them, so the
        // alias union-find would never learn it. Pre-union them — in the ALIAS
        // union-find only, never `storage` (storage drives buffer assignment;
        // cohabiting values are not the same value, and seed pins are
        // proposals that may be rejected). Without this,
        // an in-place write into a cohabited buffer slips past every RaW and
        // read-only check that keys on `alias.same` (verified miscompile).
        let mut alias = UnionFind::default();
        let mut first_in_buffer: HashMap<&ClassId, &ClassId> = HashMap::new();
        for (value, buffer) in &facts.pinned {
            match first_in_buffer.get(buffer) {
                Some(prior) => alias.union(prior, value),
                None => {
                    first_in_buffer.insert(buffer, value);
                }
            }
        }

        Self {
            ops,
            facts,
            reads,
            reachable,
            alias,
            storage: UnionFind::default(),
            in_place: HashMap::new(),
            committed_writes: Vec::new(),
        }
    }

    /// Transitive closure of the dataflow dependence graph: there is an edge
    /// `producer(value) -> consumer` for every operand, and `reachable[a]` is
    /// everything reachable from `a`. Two ops with no path between them are
    /// *unordered* — neither happens-before the other — which the conflict check
    /// then treats conservatively.
    fn compute_reachability(ops: &[AnalysisOp]) -> Vec<HashSet<usize>> {
        let n = ops.len();
        let mut producer: HashMap<ClassId, usize> = HashMap::new();
        for op in ops {
            for result in &op.results {
                producer.insert(result.clone(), op.position);
            }
        }
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        for op in ops {
            for operand in &op.operands {
                if let Some(&from) = producer.get(operand) {
                    adjacency[from].push(op.position);
                }
            }
        }
        let mut reachable: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for start in 0..n {
            let mut stack = adjacency[start].clone();
            while let Some(node) = stack.pop() {
                if reachable[start].insert(node) {
                    stack.extend(adjacency[node].iter().copied());
                }
            }
        }
        reachable
    }

    /// Does the op at `reader_site` provably run before the op at `writer_op`?
    /// True only when the reader's op is *reachable-before* the writer in the DAG;
    /// a boundary-output read (`None`) is at the end of the program, so it never
    /// happens before any op.
    fn happens_before(&self, reader_site: Option<(usize, usize)>, writer_op: usize) -> bool {
        match reader_site {
            None => false,
            Some((reader_op, _)) => {
                reader_op != writer_op && self.reachable[reader_op].contains(&writer_op)
            }
        }
    }

    fn run(mut self) -> Result<Analysis> {
        // CONTRACT WELL-FORMEDNESS: the tie set must be a PARTIAL MATCHING —
        // at most one operand per result, one result per operand, indices in
        // range. Two operands tied to one result would, if both admitted,
        // fuse both operands' storage into one class; with the operands
        // pinned to different boundary buffers, first-wins binding then
        // silently violates the op's own reads[dest] == writes[tied] contract
        // (executed PoC: the analyzer admits both, assignment mis-binds by
        // toposort order, and the validator cannot see write-only operands).
        // One allocation per storage class cannot represent either-or
        // aliasing (MLIR's select-style multi-operand aliasing); reject the
        // contract at the door.
        for op in self.ops {
            let info = op.iface.alias_info();
            for edge in &info {
                anyhow::ensure!(
                    edge.operand < op.operands.len() && edge.result < op.results.len(),
                    "op at position {} declares aliasing out of range \
                     (operand {}, result {})",
                    op.position,
                    edge.operand,
                    edge.result
                );
            }
            let ties = must_ties(op.iface);
            for (index, &(operand, result)) in ties.iter().enumerate() {
                for &(other_operand, other_result) in &ties[index + 1..] {
                    anyhow::ensure!(
                        other_result != result,
                        "op at position {} declares overlapping ties: the tie \
                         set must be a partial matching — at most one operand \
                         per result (one allocation per storage class; \
                         either-or aliasing is unrepresentable)",
                        op.position
                    );
                    // Strictly ascending operand order subsumes one-result-
                    // per-operand AND makes declaration order canonical: it
                    // is the intra-op decision order (each admission mutates
                    // the alias/storage state the next tie sees) and the
                    // planner's copy-insertion order.
                    anyhow::ensure!(
                        other_operand > operand,
                        "op at position {} must declare ties in strictly \
                         ascending operand order (one result per operand; \
                         declaration order is the intra-op decision order)",
                        op.position
                    );
                }
            }
        }

        // ONE SWEEP over the ties, in the order the (replaceable) policy
        // chose. Soundness never depends on that order: every rejection
        // repairs, and the committed-writer re-check covers cross-admission
        // interactions under any sequence.
        for (position, tie) in Self::decision_order(self.ops) {
            self.decide_tie(position, tie);
        }
        Ok(Analysis {
            storage: self.storage,
            in_place: self.in_place,
            op_count: self.ops.len(),
        })
    }

    /// THE DECISION-ORDER POLICY — the one swappable function. Its CONTRACT:
    ///
    ///  * INPUT: the analyzed op table. OUTPUT: a sequence of genuine
    ///    `(position, must-edge-of-ops[position])` pairs — nothing else.
    ///  * Any ORDER is sound: admissions are checked against current state,
    ///    the committed-writer re-check covers writers whose conflicts only
    ///    become visible through a later union, rejections only shrink
    ///    sharing, every rejection repairs at plan build, and the validator
    ///    certifies the final artifact regardless.
    ///  * OMISSION = rejection: an edge never decided has no `in_place`
    ///    entry, and every consumer treats missing as rejected — the repair
    ///    fires. DUPLICATES are ignored: the first decision wins
    ///    (`decide_tie` is first-wins), since unions cannot be unwound.
    ///  * The ONLY thing the order changes is plan QUALITY: which of two
    ///    contending candidates wins storage first-come, hence where copies
    ///    land. A replacement may search orders, randomize (testing), or
    ///    consume feedback — it must merely stay deterministic per input if
    ///    goldens are to stay pinned.
    ///
    /// Current policy: bottom-up (ops nearest the outputs decide first —
    /// MLIR's default heuristic, so result-feeding ops get first claim on
    /// reused storage), with non-writing ties (views) before writing ties —
    /// a view's admission is free, and merging its alias set early keeps
    /// later writers honest, while a writer rejected afterwards has a cheap
    /// repair.
    fn decision_order(ops: &[AnalysisOp]) -> Vec<(usize, (usize, usize))> {
        let mut order = Vec::new();
        for op in ops.iter().rev() {
            for &(operand, result) in &must_ties(op.iface) {
                if !op.iface.result_writes_memory(result) {
                    order.push((op.position, (operand, result)));
                }
            }
        }
        for op in ops.iter().rev() {
            for &(operand, result) in &must_ties(op.iface) {
                if op.iface.result_writes_memory(result) {
                    order.push((op.position, (operand, result)));
                }
            }
        }
        order
    }

    /// Decide one tie: admit (the result lives in the operand's storage) or
    /// reject (the generic repair relocates it at plan-build time).
    /// First-wins: a tie already decided is never re-decided (unions cannot
    /// be unwound), which makes duplicate policy entries harmless.
    fn decide_tie(&mut self, position: usize, (operand, result): (usize, usize)) {
        let op = &self.ops[position];
        if self.in_place.contains_key(&(op.position, operand)) {
            return;
        }
        let decided_in_place = self.try_in_place(op, operand, result);
        self.in_place
            .insert((op.position, operand), decided_in_place);
        if decided_in_place {
            // Commit: the tied result now shares the operand's buffer.
            let operand_value = &op.operands[operand];
            let result_value = &op.results[result];
            if op.iface.result_writes_memory(result) {
                // Only admissions that actually WRITE the shared
                // storage become committed writers; a pure view's
                // admission merges alias sets without writing a byte.
                self.committed_writes.push(WriteUse {
                    value: operand_value.clone(),
                    site: (op.position, operand),
                });
            }
            self.alias.union(operand_value, result_value);
            // The storage relation records the ADMISSION itself —
            // there is no tie kind to record. A DPS result becomes
            // a cohabitant of its destination's class (same
            // descriptor, by the op's contract); a view rides its
            // parent's class under its own layout (in-bounds by
            // index-map legality). Assignment is a class lookup
            // either way.
            self.storage.union(operand_value, result_value);
        }
    }

    /// Can `tie` of `op` bufferize in place? In-place only if every veto
    /// clears: writability, the read+write-same-buffer door, and read-after-write
    /// interference. A NON-writing candidate (a pure view: the tied result is
    /// not written) introduces no write, so the writer-side checks (1)–(2) don't
    /// apply — but its union still merges alias sets, so the committed-writer
    /// interference check (3) always runs.
    fn try_in_place(&mut self, op: &AnalysisOp, operand: usize, result: usize) -> bool {
        let operand_value = op.operands[operand].clone();
        let result_value = op.results[result].clone();
        let introduces_write = op.iface.result_writes_memory(result);

        // (1) Writability (`wouldCreateWriteToNonWritableBuffer`): in-placing
        // writes the operand's buffer. If any alias of it is physically read-only
        // (a constant / weights / read-only input), the write is illegal,
        // regardless of liveness. A view of read-only storage is legal — it
        // writes nothing.
        if introduces_write && self.alias_set_is_read_only(&operand_value) {
            return false;
        }

        // (2) Read-after-write (`wouldCreateReadAfterWriteInterference`).
        // In-placing makes this op overwrite the operand's buffer. Any read of a
        // value still aliasing that buffer — other than the result we are writing
        // there — must provably happen before this op; otherwise it could observe
        // the overwritten contents. Ordering is DAG reachability, so an *unordered*
        // reader (no dependence path to this op) is a conflict, not a free pass.
        // Skipped entirely for non-writing candidates: there is no new write to
        // interfere with anything.
        let raw_scan = if introduces_write {
            self.reads.clone()
        } else {
            Vec::new()
        };
        for read in raw_scan {
            if read.site == Some((op.position, operand)) {
                // "A use cannot conflict with itself. Note: just being the same
                // op is not enough. It has to be the same use." (MLIR) — the
                // candidate operand's own read never conflicts with its own
                // write: a read-modify-write through ONE operand (an
                // accumulator) is the canonical in-place case, and its intra-op
                // safety is the op's declared contract. Reads through OTHER
                // operands of this op are handled below.
                continue;
            }
            if read.value == result_value {
                // Reads the new contents we are placing here — the intended
                // consumer, not a conflict.
                continue;
            }
            if !self.alias.same(&read.value, &operand_value) {
                continue; // Reads a different buffer — irrelevant.
            }
            if self.happens_before(read.site, op.position) {
                continue; // Provably reads the old value before we overwrite it.
            }
            if let Some((reader_op, read_idx)) = read.site {
                if reader_op == op.position {
                    // Same-op read through a DIFFERENT operand, of storage
                    // aliasing the candidate. Excused only by the op's own
                    // `isNotConflicting` assertion — an unconditional,
                    // TRUSTED per-pair contract. The engine performs no
                    // layout checking here: an op that can only tolerate the
                    // aliasing under preconditions must discharge them where
                    // it is matched (egglog), not declare the permit. The
                    // permit is scoped to same-op reads ONLY — cross-op reads
                    // stay governed by `happens_before` above (a blanket
                    // weakening is a known miscompile).
                    if permits_sharing(op.iface, read_idx, operand) {
                        continue;
                    }
                }
            }
            return false;
        }

        // (3) Committed-writer interference (the write side of MLIR's
        // `hasReadAfterWriteInterference`, which walks reads AND writes of
        // both alias sets). Admitting this candidate merges the operand's and
        // results' alias sets; a write COMMITTED by an earlier bottom-up
        // decision may only now come to alias a read it was never checked
        // against — e.g. a second writer of this operand's storage, admitted
        // while the end-of-program read of OUR result was not yet in its
        // alias set (review-confirmed miscompile: the dead writer's bytes
        // land in the output buffer). Re-check every committed write against
        // every read of the post-union set, with the writer's own exemptions.
        let post_union_aliases = |uf: &mut Self, value: &ClassId| {
            uf.alias.same(value, &operand_value)
                || uf.alias.same(value, &result_value)
                || *value == result_value
        };
        for write in self.committed_writes.clone() {
            if !post_union_aliases(self, &write.value) {
                continue;
            }
            let writer = &self.ops[write.site.0];
            for read in self.reads.clone() {
                if !post_union_aliases(self, &read.value) {
                    continue;
                }
                if read.site == Some(write.site) {
                    continue; // a use cannot conflict with itself (RMW operand)
                }
                if writer.results.contains(&read.value) {
                    continue; // reads the contents that write defines
                }
                if self.happens_before(read.site, write.site.0) {
                    continue; // provably reads before the committed write
                }
                if let Some((reader_op, read_idx)) = read.site {
                    if reader_op == write.site.0 {
                        // Same-op pair, judged by the WRITER's contract (the
                        // same permit its own admission would apply).
                        if permits_sharing(writer.iface, read_idx, write.site.1) {
                            continue;
                        }
                    }
                }
                return false;
            }
        }

        true
    }

    /// Is any value in the operand's current alias set backed by read-only storage?
    fn alias_set_is_read_only(&mut self, operand_value: &ClassId) -> bool {
        // The read-only seeds are few; check each against the operand's set.
        let read_only: Vec<ClassId> = self.facts.read_only.iter().cloned().collect();
        read_only
            .iter()
            .any(|ro| self.alias.same(ro, operand_value))
    }
}

// =============================================================================
// Rewrite (phase 2): assign buffers and emit storage ops
// =============================================================================

/// Bufferize the extracted graph **out of place**: analyze in-place
/// opportunities, assign a buffer to every value, then build a new, independent
/// INPUT-PROGRAM VALIDATION — the program-validity gate, run before any
/// analysis or decision is made. Everything it checks is a property of the
/// extracted program alone (the ops' declared memory effects and the caller's
/// boundary bindings), never of a planning choice, so a rejection means "this
/// program", not "this plan".
///
/// Two kinds of rejection, with distinct prefixes:
///  * `invalid input program` — the program is ill-formed in itself: something
///    CONSUMES undefined contents (an op reads a poison operand, or an output
///    slot binds a poison value). Undefined values are write-targets only.
///  * `unsupported program` — the program is meaningful but outside the
///    planner's supported class at buffer granularity: a buffer demanded to
///    deliver two or more DISTINCT final values can only do so as
///    pass-throughs of its own input values (one buffer holds one written
///    value; region-level support would relax this — see the parked
///    aliased-views-in-place fixture).
///
/// TIMING: this is as early as these checks can be made precisely. Value
/// identity is e-class identity, which equality saturation refines — checked
/// against the source before egglog runs, "two distinct values" could be one
/// e-class in disguise, so a pre-egglog check would be conservative. And the
/// poison checks read each op's per-IMPLEMENTATION read declarations, which
/// exist only once extraction has chosen implementations. Op contract
/// well-formedness (the alias_info matching rules) is different: it is a
/// per-TYPE property re-checked per instance at `Analyzer::run`.
fn validate_input_program(graph: &ExtractedGraph) -> Result<()> {
    // Undefined (poison) values: results their producing op declares undefined.
    let mut undefined: HashSet<ClassId> = HashSet::new();
    for node in graph.dag.node_weights() {
        if let ExtractedNode::LayoutOp(op) = node {
            for (result, output) in op.outputs.iter().enumerate() {
                if op.op.result_is_undefined(result) {
                    undefined.insert(output.eclass.clone());
                }
            }
        }
    }

    // Undefined values are write-targets only: no op may READ one. (The DPS
    // rewrite's poison destinations are declared write-only, so they pass.)
    for node in graph.dag.node_weights() {
        let ExtractedNode::LayoutOp(op) = node else {
            continue;
        };
        for (operand, input) in op.inputs.iter().enumerate() {
            if undefined.contains(&input.value) && op.op.operand_reads_memory(operand) {
                anyhow::bail!(
                    "invalid input program: op {} reads undefined contents \
                     through operand {}: undefined (poison) values are \
                     write-targets only",
                    op.op.label(),
                    operand
                );
            }
        }
    }

    // Boundary DECLARATIONS must be explicit: every buffer states its access
    // level AND who frees it — the boundary contract is load-bearing, so no
    // part of it may be implied. Checked binding by binding, in graph order.
    let check_declarations = |info: &crate::layout_ir::BufferInfo| -> Result<()> {
        if info.access.is_none() {
            anyhow::bail!(
                "invalid input program: buffer {} declares no access level — \
                 set (buffer-access-of ...) to (ReadOnly) or (ReadWrite)",
                info.id_label
            );
        }
        if info.freed_by.is_none() {
            anyhow::bail!(
                "invalid input program: buffer {} declares no deallocation \
                 responsibility — set (buffer-freed-by ...) to (CallerFrees) \
                 or (ProgramFrees)",
                info.id_label
            );
        }
        Ok(())
    };
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => check_declarations(&input.buffer)?,
            ExtractedNode::BufferOutput(output) => {
                for slot in &output.slots {
                    check_declarations(&slot.buffer)?;
                }
            }
            ExtractedNode::LayoutOp(_) => {}
        }
    }

    // Boundary bindings, buffer by buffer (in graph order, so the first
    // violation reported is deterministic).
    let mut input_values: HashMap<ClassId, HashSet<ClassId>> = HashMap::new();
    let mut input_layouts: HashMap<ClassId, ClassId> = HashMap::new();
    for node in graph.dag.node_weights() {
        if let ExtractedNode::BufferInput(input) = node {
            input_values
                .entry(input.buffer.id_eclass.clone())
                .or_default()
                .insert(input.value.eclass.clone());
            input_layouts.insert(
                input.value.eclass.clone(),
                input.value.layout.eclass.clone(),
            );
        }
    }
    let mut output_buffers: Vec<ClassId> = Vec::new();
    let mut output_values: HashMap<ClassId, Vec<ClassId>> = HashMap::new();
    for node in graph.dag.node_weights() {
        let ExtractedNode::BufferOutput(output) = node else {
            continue;
        };
        for slot in &output.slots {
            if undefined.contains(&slot.value) {
                anyhow::bail!(
                    "invalid input program: output slot {} binds the undefined \
                     (poison) value {}: a program returning undefined contents \
                     is ill-formed",
                    slot.index,
                    slot.value
                );
            }
            let buffer = slot.buffer.id_eclass.clone();
            if !output_values.contains_key(&buffer) {
                output_buffers.push(buffer.clone());
            }
            let demanded = output_values.entry(buffer).or_default();
            if !demanded.contains(&slot.value) {
                demanded.push(slot.value.clone());
            }
        }
    }

    // The buffer-granular support rule: two or more distinct final values on
    // one buffer are deliverable only as pass-throughs of that buffer's own
    // inputs (nothing may be WRITTEN into a buffer that must also preserve a
    // second value to the end of the program).
    for buffer in &output_buffers {
        let demanded = &output_values[buffer];
        if demanded.len() < 2 {
            continue;
        }
        for value in demanded {
            let is_pass_through = input_values
                .get(buffer)
                .is_some_and(|inputs| inputs.contains(value));
            if !is_pass_through {
                anyhow::bail!(
                    "unsupported program: output buffer {} is demanded to hold \
                     {} distinct final values, and {} is not one of that \
                     buffer's input values — at buffer granularity, multiple \
                     outputs per buffer are supported only as pass-throughs of \
                     the buffer's own inputs",
                    buffer,
                    demanded.len(),
                    value
                );
            }
        }
        // Declaration satisfiability: two pass-through demands on one buffer
        // that view the SAME region (equal layout e-classes) but carry
        // DIFFERENT values are contradictory — the buffer cannot end holding
        // both. Layout equality here is declaration CONSISTENCY (the caller
        // named the same layout twice), not region analysis.
        for (i, a) in demanded.iter().enumerate() {
            for b in &demanded[i + 1..] {
                if input_layouts.contains_key(a) && input_layouts.get(a) == input_layouts.get(b) {
                    anyhow::bail!(
                        "invalid input program: output buffer {} is demanded \
                         to hold two different values ({} and {}) under the \
                         same layout — contradictory declarations are \
                         unsatisfiable",
                        buffer,
                        a,
                        b
                    );
                }
            }
        }
    }
    Ok(())
}

/// [`BufferIrGraph`] whose values are buffers and whose copies are real nodes. The
/// source `graph` is borrowed and left untouched.
pub fn bufferize(graph: &ExtractedGraph) -> Result<BufferIrGraph> {
    let mut plan = lower(buffer_tensor_plan(graph)?)?;
    annotate_buffer_geometry(&mut plan, graph)?;
    Ok(plan)
}

/// Thread the extraction's literal geometry (dims, element bits) and the
/// boundary `BufferLit` keys onto the plan's buffers — the sizing/binding
/// surface the `ReferenceRuntime` executes from. Purely additive; `None`
/// stays `None` for symbolic geometry, and numeric consumers bail loudly.
fn annotate_buffer_geometry(plan: &mut BufferIrGraph, graph: &ExtractedGraph) -> Result<()> {
    use std::collections::HashMap as Map;
    type Geometry = (
        Option<Vec<i64>>,
        Option<i64>,
        Option<crate::dtype::PlanDtype>,
    );
    let mut value_geometry: Map<ClassId, Geometry> = Map::new();
    let mut boundary_lits: Map<ClassId, i64> = Map::new();
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => {
                value_geometry.entry(input.value.eclass.clone()).or_insert((
                    input.value.dims.clone(),
                    input.value.element_bits,
                    input.value.dtype_enum,
                ));
                if let Some(lit) = input.buffer.lit {
                    boundary_lits.insert(input.buffer.id_eclass.clone(), lit);
                }
            }
            ExtractedNode::LayoutOp(op) => {
                for output in &op.outputs {
                    value_geometry.entry(output.eclass.clone()).or_insert((
                        output.dims.clone(),
                        output.element_bits,
                        output.dtype_enum,
                    ));
                }
            }
            ExtractedNode::BufferOutput(output) => {
                for slot in &output.slots {
                    if let Some(lit) = slot.buffer.lit {
                        boundary_lits.insert(slot.buffer.id_eclass.clone(), lit);
                    }
                }
            }
        }
    }
    for (value, id) in &plan.value_buffer {
        if let Some((dims, bits, dtype)) = value_geometry.get(value) {
            if let Some(buffer) = plan.buffers.get_mut(id) {
                // Dims/bits joins are CHECKED, order-independent lattice
                // joins (None ∨ x = x; equal ∨ equal = equal; different
                // knowns BAIL) — the old first-wins-by-hash-order join
                // let a buffer shared by values of different numel take
                // its geometry nondeterministically (found 2026-08-12 by
                // the forced view-admission probe; fix ruled 2026-08-13).
                // Dims stay FIRST-WINS for now — and that is a documented
                // hole, not a design: folded views legitimately cohabit a
                // buffer with the parent at DIFFERENT numel (matmul expand
                // reads (2,3) through (2,4,3); slice reads (2,2) through
                // (2); scalar broadcast reads () through (3,5)), so a
                // checked numel join refuses valid plans. The sound join
                // needs WRITER identity — storage geometry = the resident
                // value that supplies the bytes (staged input or writing
                // kernel), view readers skipped. That resident-geometry
                // annotation is the prerequisite for admitting views on
                // real backends and lands with the M4 re-seat (ruling
                // 2026-08-13); until then hash-order decides ties exactly
                // as before, deterministic per build.
                if buffer.dims.is_none() {
                    buffer.dims = dims.clone();
                }
                match (buffer.element_bits, *bits) {
                    (None, known) => buffer.element_bits = known,
                    (Some(held), Some(new)) if held != new => anyhow::bail!(
                        "buffer {} backs values of conflicting element                          widths {held} and {new} bits",
                        buffer.label
                    ),
                    _ => {}
                }
                // Dtype join is CHECKED, not first-wins: two values
                // cohabiting one buffer with different dtypes would be a
                // silent reinterpretation — exactly the smuggling the
                // typed-buffers ruling forbids. (Width-compatible reuse
                // across dtypes must instead refuse here, loudly.)
                match (buffer.dtype, dtype) {
                    (None, Some(dtype)) => buffer.dtype = Some(*dtype),
                    (Some(held), Some(dtype)) if held != *dtype => anyhow::bail!(
                        "buffer {} backs values of conflicting dtypes \
                         {held:?} and {dtype:?} — dtype-blind buffer reuse \
                         is not executable",
                        buffer.label
                    ),
                    _ => {}
                }
            }
        }
    }
    // Consistency tripwire: a buffer's layout width must agree with its
    // dtype's egglog bits-of row (the always-mint rule guarantees this
    // for auto-minted layouts; a mismatch means a padded/foreign layout
    // reached execution, which has no typed-storage story yet).
    for buffer in plan.buffers.values() {
        if let (Some(bits), Some(dtype)) = (buffer.element_bits, buffer.dtype) {
            anyhow::ensure!(
                bits == dtype.egglog_bits(),
                "buffer {} is annotated {dtype:?} (bits-of = {}) but its \
                 layout width is {bits} bits",
                buffer.label,
                dtype.egglog_bits()
            );
        }
    }
    for buffer in plan.buffers.values_mut() {
        if let BufferId::Boundary(eclass) = &buffer.id {
            buffer.lit = boundary_lits.get(eclass).copied();
        }
    }
    // A delivery copy's destination inherits its source's geometry (same
    // value, same shape — the dst had no value of its own to join on).
    let copy_pairs: Vec<(BufferId, BufferId)> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::BufferCopy { src, dst } => Some((src.clone(), dst.clone())),
            _ => None,
        })
        .collect();
    for (src, dst) in copy_pairs {
        let Some(source) = plan.buffers.get(&src) else {
            continue;
        };
        let (dims, bits, dtype) = (source.dims.clone(), source.element_bits, source.dtype);
        if let Some(buffer) = plan.buffers.get_mut(&dst) {
            if buffer.dims.is_none() {
                buffer.dims = dims;
            }
            if buffer.element_bits.is_none() {
                buffer.element_bits = bits;
            }
            if buffer.dtype.is_none() {
                buffer.dtype = dtype;
            }
        }
    }
    Ok(())
}

/// The PLANNING half of [`bufferize`]: validate the input program, analyze,
/// assign, build the BufferTensor graph, install anti-dependence ordering,
/// certify, and run the storage-level rewrites — everything semantic. The
/// returned graph is the finished BufferTensor program (optimized: poisons
/// folded, dead buffers dropped) and the audit artifact `main` renders;
/// [`lower`] erases it into the executable plan.
pub(crate) fn buffer_tensor_plan(
    graph: &ExtractedGraph,
) -> Result<crate::buffer_tensor_ir::BufferTensorIrGraph> {
    validate_input_program(graph)?;
    let order = toposort(&graph.dag, None)
        .map_err(|_| anyhow::anyhow!("extracted graph has a cycle; cannot bufferize"))?;

    // Collect the ops (in topo order, one dense position each) and the boundary
    // facts the analyzer needs.
    let mut op_nodes: Vec<&crate::layout_ir::OpNode> = Vec::new();
    let mut facts = ValueFacts::default();
    for index in &order {
        match &graph.dag[*index] {
            ExtractedNode::BufferInput(input) => {
                if input.buffer.access() == Access::ReadOnly {
                    facts.read_only.insert(input.value.eclass.clone());
                }
                facts
                    .pinned
                    .insert(input.value.eclass.clone(), input.buffer.id_eclass.clone());
            }
            ExtractedNode::LayoutOp(op) => {
                for result in 0..op.outputs.len() {
                    if op.op.result_is_allocated_internally(result) {
                        anyhow::bail!(
                            "op {} declares result {result} as internally \
                             allocated: op-owned storage is not yet supported \
                             by the planner (needs the Alloc/Free ownership \
                             machinery)",
                            op.op.label()
                        );
                    }
                }
                op_nodes.push(op);
            }
            ExtractedNode::BufferOutput(output) => {
                for slot in &output.slots {
                    facts.output_values.push(slot.value.clone());
                }
            }
        }
    }

    let analysis_ops: Vec<AnalysisOp> = op_nodes
        .iter()
        .enumerate()
        .map(|(position, op)| AnalysisOp {
            position,
            iface: op.op.as_ref(),
            operands: op.inputs.iter().map(|input| input.value.clone()).collect(),
            results: op
                .outputs
                .iter()
                .map(|output| output.eclass.clone())
                .collect(),
        })
        .collect();

    // DESTINATION SEEDING, step 1 — proposal as constraints. Pin each output
    // slot's chain-root poison to the slot's buffer, so the ONE analysis run
    // evaluates the cohabitation for real: the pre-union makes the buffer's
    // residents visible to every RaW check, and the shared pin feeds the
    // conflict checks. Whether the proposal is KEPT is decided
    // after the run (step 2, in `Bufferizer::assign`), gated on the in-place
    // verdicts along the chain; the pins themselves are analysis-only facts
    // (assignment and graph building never read `facts`).
    let seeds = find_seeds(graph, &order, &analysis_ops);
    for seed in &seeds {
        facts
            .pinned
            .insert(seed.poison.clone(), seed.buffer_eclass.clone());
    }

    let mut analysis = Analyzer::new(&analysis_ops, &facts).run()?;
    let assignment = Bufferizer::assign(graph, &order, &mut analysis, &seeds);
    let mut bt =
        crate::buffer_tensor_ir::build_buffer_tensor_ir(graph, &order, assignment, &analysis)?;
    crate::buffer_tensor_ir::install_anti_edges(&mut bt);
    // The certificate runs AFTER the storage-lifetime pass: its lifetime arms
    // certify what `optimize` constructs (allocs, frees, their ordering
    // edges), and the residency arms only gain edges by the reorder — the
    // pass adds ordering and removes nothing a consumer reads.
    let bt = crate::buffer_tensor_ir::optimize(bt);
    crate::buffer_tensor_ir::validate(&bt)?;
    Ok(bt)
}

// -----------------------------------------------------------------------------
// Buffer assignment: value -> BufferId
// -----------------------------------------------------------------------------

/// Assigns a buffer to every value and interns every boundary buffer.
#[derive(Default)]
pub(crate) struct Bufferizer {
    pub(crate) buffers: HashMap<BufferId, Buffer>,
    pub(crate) value_buffer: HashMap<ClassId, BufferId>,
    /// Buffer chosen for each storage-class representative (so all values an
    /// admitted decision placed in one allocation — an in-place producer and
    /// its operand, a view and its parent — share one buffer).
    rep_buffer: HashMap<ClassId, BufferId>,
    next_alloc: u32,
}

impl Bufferizer {
    fn assign(
        graph: &ExtractedGraph,
        order: &[NodeIndex],
        analysis: &mut Analysis,
        seeds: &[Seed],
    ) -> Self {
        let mut this = Bufferizer::default();

        // Intern the INPUT boundary buffers first: `intern_boundary` is
        // first-wins on declarations/label, and an input's declarations are
        // the storage's real contract. A seed interning the same buffer
        // earlier (with the output slot's declarations) could otherwise
        // launder a read-only input buffer into a writable one (review
        // finding).
        for index in order {
            if let ExtractedNode::BufferInput(input) = &graph.dag[*index] {
                this.intern_boundary(
                    &input.buffer.id_eclass,
                    input.buffer.access(),
                    input.buffer.freed_by(),
                    input.buffer.id_label.clone(),
                );
            }
        }

        // DESTINATION SEEDING, step 2 — apply the admitted proposals. A seed
        // whose every hop the analysis admitted in place binds its poison's
        // STORAGE CLASS to the slot's buffer before the walk: the chain's
        // results then find the buffer via `rep_buffer` and compute straight
        // into the output storage (the boundary sees src == dest, no copy).
        // Seeds arrive in slot order, so a contested class goes to the lowest
        // slot; the class-level binding is the whole mechanism — pre-binding
        // `value_buffer` directly is a verified miscompile. A rejected seed
        // binds nothing: the class falls through to a fresh allocation and
        // the plan degrades to exactly the unseeded one.
        for seed in seeds {
            let admitted = seed
                .hops
                .iter()
                .all(|hop| analysis.in_place.get(hop).copied().unwrap_or(false));
            if !admitted {
                continue;
            }
            let rep = analysis.storage.find(&seed.poison);
            if this.rep_buffer.contains_key(&rep) {
                continue;
            }
            let id = this.intern_boundary(
                &seed.buffer_eclass,
                seed.access,
                seed.freed_by,
                seed.buffer_label.clone(),
            );
            // The interned buffer keeps the FIRST declarations it was seen
            // with (an input's, per the pre-pass). If that access is
            // ReadOnly, the storage is not writable and the seed must not
            // apply, no matter what the output slot claimed.
            if this.buffers[&id].access == Access::ReadOnly {
                continue;
            }
            this.rep_buffer.insert(rep, id);
        }

        for index in order {
            match &graph.dag[*index] {
                ExtractedNode::BufferInput(input) => {
                    let id = this.intern_boundary(
                        &input.buffer.id_eclass,
                        input.buffer.access(),
                        input.buffer.freed_by(),
                        input.buffer.id_label.clone(),
                    );
                    this.bind(analysis, &input.value.eclass, id);
                }
                ExtractedNode::LayoutOp(op) => {
                    for output in &op.outputs {
                        let rep = analysis.storage.find(&output.eclass);
                        // If this result shares a storage class with an
                        // already-bound value — its in-place destination, or
                        // the parent a view derives (both bound earlier in
                        // topo order; chained views resolve through the one
                        // shared class) — reuse that buffer; otherwise mint a
                        // fresh, system-owned allocation (out of place).
                        let id = match this.rep_buffer.get(&rep) {
                            Some(existing) => existing.clone(),
                            None => this.allocate(output.label.clone()),
                        };
                        this.bind(analysis, &output.eclass, id);
                    }
                }
                ExtractedNode::BufferOutput(output) => {
                    // Intern each pinned destination so `buffers` is complete; the
                    // wiring (and any needed copy) happens during graph building.
                    for slot in &output.slots {
                        this.intern_boundary(
                            &slot.buffer.id_eclass,
                            slot.buffer.access(),
                            slot.buffer.freed_by(),
                            slot.buffer.id_label.clone(),
                        );
                    }
                }
            }
        }
        this
    }

    /// Record `value`'s buffer and remember it on the value's storage rep so
    /// later values of the same class reuse the same storage.
    fn bind(&mut self, analysis: &mut Analysis, value: &ClassId, id: BufferId) {
        let rep = analysis.storage.find(value);
        self.rep_buffer.entry(rep).or_insert_with(|| id.clone());
        self.value_buffer.insert(value.clone(), id);
    }

    fn intern_boundary(
        &mut self,
        eclass: &ClassId,
        access: Access,
        freed_by: FreedBy,
        label: String,
    ) -> BufferId {
        let id = BufferId::Boundary(eclass.clone());
        self.buffers.entry(id.clone()).or_insert_with(|| Buffer {
            id: id.clone(),
            access,
            freed_by,
            owner: Owner::Caller,
            label,
            dims: None,
            element_bits: None,
            dtype: None,
            lit: None,
        });
        id
    }

    fn allocate(&mut self, label: String) -> BufferId {
        let id = BufferId::Allocated(self.next_alloc);
        self.next_alloc += 1;
        self.buffers.insert(
            id.clone(),
            Buffer {
                id: id.clone(),
                access: Access::ReadWrite,
                // Planner-minted storage must not outlive the call: the
                // program frees it, always (the escape cell is uninhabited).
                freed_by: FreedBy::Program,
                owner: Owner::System,
                label,
                dims: None,
                element_bits: None,
                dtype: None,
                lit: None,
            },
        );
        id
    }
}

// -----------------------------------------------------------------------------
// Plan validation: the lowering tripwires
// -----------------------------------------------------------------------------

/// THE LOWERING TRIPWIRES — mechanical structural checks on the lowered plan.
/// The SEMANTIC certificate (the residency rule) runs on the BufferTensor
/// graph before lowering, where values still exist
/// ([`crate::buffer_tensor_ir::validate`]); what it cannot see is whether the
/// lowering performed its folds, because from the BufferTensor side the folds
/// have not happened yet. These arms certify exactly that:
///
///  * no self-copy — a phantom writer the WAR machinery would hang spurious
///    Anti edges on (the fold becomes mandatory the day a pass can produce
///    one);
///  * no surviving poison-shaped node — undefined-contents producers must
///    lower to nothing (a future Alloc node takes that seat);
///  * no surviving view-shaped node — views must lower to a producer
///    redirect, never to compute.
fn validate_plan(dag: &DiGraph<BufferNode, BufferEdge>) -> Result<()> {
    for index in dag.node_indices() {
        match &dag[index] {
            BufferNode::BufferCopy { src, dst } if src == dst => {
                anyhow::bail!(
                    "plan validation failed: self-copy of {src:?} — a no-op \
                     writer must be folded before the WAR scan"
                );
            }
            BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
            } => {
                if reads.is_empty()
                    && !writes.is_empty()
                    && (0..writes.len()).all(|result| op.result_is_undefined(result))
                    && writes
                        .iter()
                        .any(|buffer| matches!(buffer, BufferId::Boundary(_)))
                {
                    anyhow::bail!(
                        "plan validation failed: alloc-like producer ({}) on \
                         caller storage — caller buffers are never \
                         program-allocated, so a seeded poison escaped the \
                         storage-lifetime pass",
                        op.label()
                    );
                }
                let derives = |result: usize| ties.iter().any(|(_, r)| *r == result);
                if !reads.is_empty()
                    && !writes.is_empty()
                    && (0..reads.len()).all(|operand| !op.operand_reads_memory(operand))
                    && (0..writes.len())
                        .all(|result| !op.result_writes_memory(result) && derives(result))
                {
                    anyhow::bail!(
                        "plan validation failed: unfolded view ({}) — metadata \
                         views must lower to a producer redirect, never to \
                         compute",
                        op.label()
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Lowering: BufferTensorIR -> BufferIR (the forgetting step)
// -----------------------------------------------------------------------------

/// Lower the BufferTensor graph into the executable [`BufferIrGraph`] — the
/// step that erases the value half of every BufferTensor. The folds happen
/// HERE, as lowering rules recognized from declared memory effects:
///
///  * a view-shaped op (no reads, no writes, every result tied) lowers to a
///    producer REDIRECT — consumers resolve to the parent residence's
///    producer, which is the parent's own producer for an admitted view and
///    the repair copy for a rejected one: the BufferTensor structure absorbed
///    that distinction at construction, so one rule serves both. This fold
///    CANNOT move earlier: a view is a value-level distinction with no
///    storage-level content, so it becomes nothing only when values do;
///  * a transport (one operand, one result, same value — the structural
///    signature of [`crate::buffer_tensor_ir::BufferCopy`]) lowers to a
///    [`BufferNode::BufferCopy`];
///  * everything else erases to a [`BufferNode::Compute`] over bare buffers.
///
/// Everything semantic already happened upstream: the Anti (WAR) edges were
/// installed on the BufferTensor graph and are TRANSFERRED here, the
/// residency certificate ran there, and the storage-level rewrites (poison
/// fold, buffer DCE) ran in [`crate::buffer_tensor_ir::optimize`]. What
/// follows the walk is the schedulability check and the lowering tripwires
/// ([`validate_plan`]).
pub(crate) fn lower(bt: crate::buffer_tensor_ir::BufferTensorIrGraph) -> Result<BufferIrGraph> {
    use crate::buffer_tensor_ir::{BtNode, BufferTensor, BufferTensorIrGraph};
    let BufferTensorIrGraph {
        dag: bt_dag,
        buffers,
        value_buffer,
    } = bt;

    use petgraph::visit::EdgeRef;

    let mut dag: DiGraph<BufferNode, BufferEdge> = DiGraph::new();
    // The lowered node producing each RESIDENCE (value, buffer) — a copy
    // gives one value a second residence with a distinct producer, so the
    // pair is the key, never the value alone.
    let mut producer: HashMap<(ClassId, BufferId), NodeIndex> = HashMap::new();
    // BT node -> lowered node, for transferring Anti edges (folded nodes have
    // no entry).
    let mut lowered: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut outputs = Vec::new();

    let residence = |tensor: &BufferTensor| (tensor.value.clone(), tensor.buffer.clone());

    for index in bt_dag.node_indices() {
        match &bt_dag[index] {
            BtNode::Input { slots } => {
                let node = dag.add_node(BufferNode::BufferInput {
                    slots: slots
                        .iter()
                        .map(|slot| InputBinding {
                            value: slot.value.clone(),
                            buffer: slot.buffer.clone(),
                        })
                        .collect(),
                });
                lowered.insert(index, node);
                for slot in slots {
                    producer.insert(residence(slot), node);
                }
            }
            BtNode::Op {
                op,
                operands,
                results,
                ties,
            } => {
                // Alloc-shaped nodes (no operands, every result undefined)
                // are BufferAllocs — `optimize` converted every System-buffer
                // poison and folded every caller-storage (seeded) one. One
                // reaching here on a Boundary buffer means optimize was
                // skipped: caller storage is never program-allocated.
                let is_alloc_shaped = operands.is_empty()
                    && !results.is_empty()
                    && (0..results.len()).all(|r| op.result_is_undefined(r));
                if is_alloc_shaped {
                    if let Some(tensor) = results
                        .iter()
                        .find(|t| matches!(t.buffer, BufferId::Boundary(_)))
                    {
                        anyhow::bail!(
                            "lowering failed: alloc-shaped op ({}) produces \
                             caller storage {:?} — caller buffers are never \
                             program-allocated (was `optimize` skipped?)",
                            op.label(),
                            tensor.buffer,
                        );
                    }
                }

                // FOLD metadata views: no bytes move, so no plan node and no
                // events — just the producer redirect through the parent
                // residence. (`result_is_allocated_internally` needs no check
                // here: declaring it is still a loud error at bufferize
                // entry, so no such op reaches lowering.)
                let derives =
                    |result: usize| ties.iter().find(|(_, r)| *r == result).map(|(o, _)| *o);
                let is_view = !operands.is_empty()
                    && !results.is_empty()
                    && (0..operands.len()).all(|o| !op.operand_reads_memory(o))
                    && (0..results.len())
                        .all(|r| !op.result_writes_memory(r) && derives(r).is_some());
                if is_view {
                    for (result, tensor) in results.iter().enumerate() {
                        let parent = &operands[derives(result).expect("checked by is_view")];
                        if let Some(&from) = producer.get(&residence(parent)) {
                            producer.insert(residence(tensor), from);
                        }
                    }
                    continue;
                }

                // TRANSPORT (value preserved, buffer changed): a real copy.
                if operands.len() == 1
                    && results.len() == 1
                    && operands[0].value == results[0].value
                {
                    let src = &operands[0];
                    let dst = &results[0];
                    let copy = dag.add_node(BufferNode::BufferCopy {
                        src: src.buffer.clone(),
                        dst: dst.buffer.clone(),
                    });
                    if let Some(&from) = producer.get(&residence(src)) {
                        dag.add_edge(
                            from,
                            copy,
                            BufferEdge {
                                buffer: src.buffer.clone(),
                                port: "in".to_string(),
                                kind: EdgeKind::Data,
                            },
                        );
                    }
                    producer.insert(residence(dst), copy);
                    lowered.insert(index, copy);
                    continue;
                }

                // COMPUTE: erase values, keep buffers.
                let reads: Vec<BufferId> = operands.iter().map(|t| t.buffer.clone()).collect();
                let writes: Vec<BufferId> = results.iter().map(|t| t.buffer.clone()).collect();
                let node = dag.add_node(BufferNode::Compute {
                    op: op.clone(),
                    reads: reads.clone(),
                    writes: writes.clone(),
                    ties: ties.clone(),
                });
                for (idx, tensor) in operands.iter().enumerate() {
                    if let Some(&from) = producer.get(&residence(tensor)) {
                        dag.add_edge(
                            from,
                            node,
                            BufferEdge {
                                buffer: reads[idx].clone(),
                                port: op.operand_name(idx),
                                kind: EdgeKind::Data,
                            },
                        );
                    }
                }
                for tensor in results {
                    producer.insert(residence(tensor), node);
                }
                lowered.insert(index, node);
            }
            BtNode::Output { slots } => {
                let bindings: Vec<OutputBinding> = slots
                    .iter()
                    .enumerate()
                    .map(|(index, slot)| OutputBinding {
                        index,
                        value: slot.value.clone(),
                        buffer: slot.buffer.clone(),
                    })
                    .collect();
                let out_node = dag.add_node(BufferNode::BufferOutput { slots: bindings });
                outputs.push(out_node);
                lowered.insert(index, out_node);
                // DELIVERY COPY: a slot demanding its value in a buffer
                // nothing wrote it to (an input passed straight to output
                // gets a fresh output buffer, never the input's ReadOnly
                // one). Materialize with a bufferizer copy from the value's
                // current residence.
                for slot in slots.iter() {
                    if producer.contains_key(&residence(slot)) {
                        continue;
                    }
                    let source = producer
                        .iter()
                        .find(|((value, buffer), _)| *value == slot.value && *buffer != slot.buffer)
                        .map(|((_, buffer), node)| (buffer.clone(), *node));
                    if let Some((src_buffer, from)) = source {
                        let copy = dag.add_node(BufferNode::BufferCopy {
                            src: src_buffer.clone(),
                            dst: slot.buffer.clone(),
                        });
                        dag.add_edge(
                            from,
                            copy,
                            BufferEdge {
                                buffer: src_buffer,
                                port: "in".to_string(),
                                kind: EdgeKind::Data,
                            },
                        );
                        producer.insert(residence(slot), copy);
                    }
                }
                for (index, slot) in slots.iter().enumerate() {
                    if let Some(&from) = producer.get(&residence(slot)) {
                        dag.add_edge(
                            from,
                            out_node,
                            BufferEdge {
                                buffer: slot.buffer.clone(),
                                port: format!("out {index}"),
                                kind: EdgeKind::Data,
                            },
                        );
                    }
                }
            }
        }
    }

    // A synthesized BufferAlloc's ordering edge (alloc -> first toucher)
    // carries a poison the toucher does not list as an operand, so operand
    // derivation cannot reproduce it — transfer it explicitly, as the
    // buffer flowing to its first writer.
    for edge in bt_dag.edge_references() {
        let crate::buffer_tensor_ir::BtEdge::Data { value } = edge.weight() else {
            continue;
        };
        let source_is_alloc = matches!(
            &bt_dag[edge.source()],
            BtNode::Op { op, operands, results, .. }
                if operands.is_empty()
                    && !results.is_empty()
                    && (0..results.len()).all(|r| op.result_is_undefined(r))
        );
        if !source_is_alloc {
            continue;
        }
        let operand_backed = match &bt_dag[edge.target()] {
            BtNode::Op { operands, .. } => operands.iter().any(|t| &t.value == value),
            _ => true, // boundary consumption is always operand-backed
        };
        if operand_backed {
            continue; // the operand-derived edge already exists
        }
        let (Some(&from), Some(&to)) = (lowered.get(&edge.source()), lowered.get(&edge.target()))
        else {
            continue;
        };
        let buffer = match &bt_dag[edge.source()] {
            BtNode::Op { results, .. } => results[0].buffer.clone(),
            _ => unreachable!("alloc-shaped is an Op"),
        };
        dag.add_edge(
            from,
            to,
            BufferEdge {
                buffer,
                port: "alloc".to_string(),
                kind: EdgeKind::Data,
            },
        );
    }

    // WAR (anti-dependence) ordering: installed on the BufferTensor graph by
    // `install_anti_edges` (the residency rule, uniform over all writers) and
    // TRANSFERRED here — an Anti edge's endpoints are always consumers or
    // writers of real storage, which never fold, so every endpoint has a
    // lowered node.
    for edge in bt_dag.edge_references() {
        let crate::buffer_tensor_ir::BtEdge::Anti { buffer } = edge.weight() else {
            continue;
        };
        let (Some(&reader), Some(&writer)) =
            (lowered.get(&edge.source()), lowered.get(&edge.target()))
        else {
            anyhow::bail!(
                "lowering lost an anti-dependence endpoint (buffer {buffer:?}): \
                 an Anti edge must never touch a folded node"
            );
        };
        dag.add_edge(
            reader,
            writer,
            BufferEdge {
                buffer: buffer.clone(),
                port: "war".to_string(),
                kind: EdgeKind::Anti,
            },
        );
    }

    // The added ordering must still admit a schedule.
    if toposort(&dag, None).is_err() {
        anyhow::bail!("anti-dependency edges made the plan unschedulable (ordering cycle)");
    }

    // THE LOWERING TRIPWIRES: the semantic certificate ran on the
    // BufferTensor graph before lowering (`buffer_tensor_ir::validate`); what
    // remains to check HERE is that the lowering itself did its job — see
    // [`validate_plan`]. Runs last, after every plan-mutating step.
    validate_plan(&dag)?;

    Ok(BufferIrGraph {
        dag,
        buffers,
        value_buffer,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout_ir::{AliasInfo, Sharing};
    use crate::test_support::{EmptyOp, MockOp};

    fn cid(s: &str) -> ClassId {
        ClassId::from(s)
    }

    /// A write-only destination whose operand value has no other reader: the old
    /// contents are dead, so in-place is admitted.
    #[test]
    fn write_only_destination_last_use_goes_in_place() {
        let op = MockOp::write_only_dest();
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("dest")],
            results: vec![cid("res")],
        }];
        let facts = ValueFacts {
            read_only: HashSet::new(),
            output_values: vec![cid("res")], // result is the program output
            ..Default::default()
        };
        let mut analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&true));
        assert!(analysis.storage.same(&cid("dest"), &cid("res")));
    }

    /// Same op, but the operand value is read by a *sibling* op that is unordered
    /// with the writer. Reachability cannot prove the read happens first, so the
    /// in-place write is refused (this is the case a topological order would have
    /// wrongly allowed).
    #[test]
    fn unordered_reader_of_operand_forces_out_of_place() {
        let producer = MockOp::write_only_dest();
        // A second op that also reads `dest`, with no dependence on the producer.
        let reader = MockOp {
            reads: vec![true],
            in_place_operand: None,
            ..Default::default()
        };
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &producer,
                operands: vec![cid("dest")],
                results: vec![cid("res")],
            },
            AnalysisOp {
                position: 1,
                iface: &reader,
                operands: vec![cid("dest")],
                results: vec![cid("res2")],
            },
        ];
        let facts = ValueFacts {
            read_only: HashSet::new(),
            output_values: vec![cid("res"), cid("res2")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&false));
    }

    /// A reader of the operand value that *provably* runs before the writer (the
    /// writer consumes the reader's result) does NOT block in-place: reachability
    /// orders it before the overwrite. This is what distinguishes the
    /// reachability check from "any other reader is a conflict".
    #[test]
    fn reader_before_writer_allows_in_place() {
        // op0 reads x, produces y.
        let reader = MockOp {
            reads: vec![true],
            in_place_operand: None,
            ..Default::default()
        };
        // op1 takes (x as write-only dest, y) and writes its result into x. Taking
        // y as an operand makes op1 depend on op0, so op0 happens-before op1.
        let writer = MockOp {
            reads: vec![false, false],
            in_place_operand: Some(0),
            ..Default::default()
        };
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &reader,
                operands: vec![cid("x")],
                results: vec![cid("y")],
            },
            AnalysisOp {
                position: 1,
                iface: &writer,
                operands: vec![cid("x"), cid("y")],
                results: vec![cid("z")],
            },
        ];
        let facts = ValueFacts {
            read_only: HashSet::new(),
            output_values: vec![cid("z")],
            ..Default::default()
        };
        let mut analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(1, 0)), Some(&true));
        assert!(analysis.storage.same(&cid("x"), &cid("z")));
    }

    /// Read-modify-write through ONE operand (an accumulator updating its own
    /// destination) is the canonical in-place case: the operand's own read is
    /// the same USE as its write and never self-conflicts (MLIR's exemption).
    #[test]
    fn same_operand_rmw_is_admitted() {
        let op = MockOp {
            reads: vec![true], // reads AND (in place) writes its operand's buffer
            in_place_operand: Some(0),
            ..Default::default()
        };
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x")],
            results: vec![cid("y")],
        }];
        let facts = ValueFacts {
            read_only: HashSet::new(),
            output_values: vec![cid("y")],
            ..Default::default()
        };
        let mut analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&true));
        assert!(analysis.storage.same(&cid("x"), &cid("y")));
    }

    /// The exemption is per-USE, not per-op: the SAME value read through a
    /// DIFFERENT operand of the same op is still a conflict (unless the op
    /// declares the unconditional permit — MockOp declares none here).
    #[test]
    fn cross_operand_read_of_same_value_still_rejected() {
        let op = MockOp {
            reads: vec![true, false], // operand 0 reads x; operand 1 is the candidate
            in_place_operand: Some(1),
            ..Default::default()
        };
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x"), cid("x")], // same value, two uses
            results: vec![cid("y")],
        }];
        let facts = ValueFacts {
            read_only: HashSet::new(),
            output_values: vec![cid("y")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 1)), Some(&false));
    }

    /// INPUT-PROGRAM VALIDATION: an op reading an undefined value (produced
    /// by an alloc-like op) is rejected before any analysis runs. This is a
    /// deliberate divergence from MLIR's `read_of_undef_is_not_a_conflict`,
    /// which EXEMPTS such reads from RaW instead: undefined values here are
    /// write-targets only, so the program itself is ill-formed and every
    /// downstream check gets to treat all reads uniformly.
    #[test]
    fn input_validation_rejects_read_of_undefined_value() {
        use crate::test_support::TestGraph;
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let x = g.op(
            Box::new(MockOp {
                reads: vec![true], // reads the (undefined) "e" — ill-formed
                ..Default::default()
            }),
            &[&e],
            &[("x", "rm")],
        )[0]
        .clone();
        g.output(&x, "D");
        let err = bufferize(&g.build()).unwrap_err();
        assert!(
            err.to_string().contains("reads undefined contents"),
            "{err}"
        );
    }

    /// A read-only input donated as the in-place destination: writing it is
    /// physically illegal, so the analysis must keep the op out of place.
    #[test]
    fn read_only_buffer_forces_out_of_place() {
        let op = MockOp::write_only_dest();
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("weights")],
            results: vec![cid("res")],
        }];
        let mut read_only = HashSet::new();
        read_only.insert(cid("weights"));
        let facts = ValueFacts {
            read_only,
            output_values: vec![cid("res")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&false));
    }

    /// The built-in functional ops take the conservative out-of-place
    /// defaults: no ties, results written into planner-allocated storage
    /// (never op-internal allocations), and never undefined contents.
    #[test]
    fn builtin_ops_declare_out_of_place_defaults() {
        use crate::reference::ops::{
            AddFunctional, IndexMapApplyMaterialize, ReduceSum, SqrtFunctional,
        };
        let ops: Vec<Box<dyn LayoutIrOp>> = vec![
            Box::new(AddFunctional),
            Box::new(SqrtFunctional),
            Box::new(ReduceSum { axis: 0 }),
            Box::new(IndexMapApplyMaterialize { entries: None }),
        ];
        for op in &ops {
            assert!(op.alias_info().is_empty());
            assert!(!op.result_is_allocated_internally(0));
            assert!(op.result_writes_memory(0));
            assert!(!op.result_is_undefined(0));
        }
    }

    /// SOUNDNESS REGRESSION for the pinned-buffer pre-union: an op reads x and
    /// declares a write-only in-place destination d, where x and d cohabit
    /// buffer B (both pinned). In-placing d writes B while the op reads B
    /// through a DIFFERENT operand — a conflict only visible via the pre-union.
    /// Deleting the pre-union in Analyzer::new makes this test fail (admit).
    #[test]
    fn pre_union_rejects_in_place_into_cohabited_buffer() {
        let op = MockOp {
            reads: vec![true, false], // reads x; dest is write-only
            in_place_operand: Some(1),
            ..Default::default()
        };
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x"), cid("d")],
            results: vec![cid("r")],
        }];
        let mut pinned = HashMap::new();
        pinned.insert(cid("x"), cid("bufB"));
        pinned.insert(cid("d"), cid("bufB"));
        let facts = ValueFacts {
            pinned,
            output_values: vec![cid("r")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 1)), Some(&false));
    }

    /// The read-only veto must fire THROUGH the pre-union: the candidate d is
    /// not itself read-only, but it cohabits B with read-only x. No reads at
    /// all, so rejection can only come from alias_set_is_read_only.
    #[test]
    fn read_only_veto_fires_through_cohabitation() {
        let op = MockOp {
            reads: vec![false, false],
            in_place_operand: Some(1),
            ..Default::default()
        };
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x"), cid("d")],
            results: vec![cid("r")],
        }];
        let mut pinned = HashMap::new();
        pinned.insert(cid("x"), cid("bufB"));
        pinned.insert(cid("d"), cid("bufB"));
        let mut read_only = HashSet::new();
        read_only.insert(cid("x"));
        let facts = ValueFacts {
            pinned,
            read_only,
            output_values: vec![cid("r")],
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 1)), Some(&false));
    }

    /// INPUT-PROGRAM VALIDATION: every boundary buffer must DECLARE who
    /// frees it — there is no implied default.
    #[test]
    fn input_validation_rejects_undeclared_freed_by() {
        use crate::test_support::TestGraph;
        let mut g = TestGraph::new();
        let x = g.input_binding("x", "B", Some(Access::ReadWrite), None, "rm");
        g.output(&x, "B");
        let err = bufferize(&g.build()).unwrap_err();
        assert!(
            err.to_string().contains("no deallocation responsibility"),
            "{err}"
        );
    }

    /// INPUT-PROGRAM VALIDATION: every boundary buffer must DECLARE its
    /// access level — the whole boundary contract is explicit.
    #[test]
    fn input_validation_rejects_undeclared_access() {
        use crate::test_support::TestGraph;
        let mut g = TestGraph::new();
        let x = g.input_binding("x", "B", None, Some(FreedBy::Caller), "rm");
        let y = g.op(
            Box::new(crate::test_support::MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&y, "D");
        let err = bufferize(&g.build()).unwrap_err();
        assert!(
            err.to_string().contains("declares no access level"),
            "{err}"
        );
    }

    /// INPUT-PROGRAM VALIDATION, output arm (user-directed): a boundary
    /// output slot binding an undefined value — "allocate and return an
    /// undefined buffer" — is rejected as ill-formed rather than planned.
    #[test]
    fn input_validation_rejects_poison_valued_output() {
        use crate::test_support::TestGraph;
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        g.output(&e, "D");
        let err = bufferize(&g.build()).unwrap_err();
        assert!(
            err.to_string().contains("returning undefined contents"),
            "{err}"
        );
    }

    /// Cross-operand gate helper: an op reading x through operand 0 while
    /// in-placing destination d through operand 1, x and d cohabiting buffer B.
    fn permit_case(not_conflicting: bool) -> Option<bool> {
        let op = MockOp {
            reads: vec![true, false],
            in_place_operand: Some(1),
            not_conflicting,
        };
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x"), cid("d")],
            results: vec![cid("r")],
        }];
        let mut pinned = HashMap::new();
        pinned.insert(cid("x"), cid("bufB"));
        pinned.insert(cid("d"), cid("bufB"));
        let facts = ValueFacts {
            pinned,
            output_values: vec![cid("r")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        analysis.in_place.get(&(0, 1)).copied()
    }

    /// THE CROSS-OPERAND PERMIT: a same-op read through a different operand
    /// of cohabiting storage is a conflict UNLESS the op declares the
    /// unconditional may-share permit. The engine checks no
    /// layouts — ops whose safety depends on preconditions discharge them at
    /// egglog match time and are trusted here.
    #[test]
    fn cross_operand_permit_is_unconditional_and_trusted() {
        // no permit -> rejected (resolved by relocation, not layout checks)
        assert_eq!(permit_case(false), Some(false));
        // permit declared -> admitted, trusted
        assert_eq!(permit_case(true), Some(true));
    }

    /// CONTRACT WELL-FORMEDNESS: an op declaring TWO operands tied to ONE
    /// result is rejected at the door. If both were admitted, both operands'
    /// storage would fuse into one class; with the operands pinned to
    /// different boundary buffers, first-wins binding silently violates the
    /// op's own reads[dest] == writes[tied] contract (review PoC: the
    /// analyzer admits both, assignment mis-binds by toposort order, and the
    /// validator cannot see write-only operands). One allocation per storage
    /// class cannot represent either-or aliasing.
    #[test]
    fn two_operands_tied_to_one_result_is_rejected() {
        #[derive(Debug, Clone)]
        struct DoubleTie;
        impl crate::buffer_tensor_ir::OpSlotNames for DoubleTie {}
        impl crate::buffer_tensor_ir::BufferTensorIrOp for DoubleTie {
            fn label(&self) -> &str {
                "DoubleTie"
            }
            fn operand_reads_memory(&self, _operand: usize) -> bool {
                false
            }
        }
        impl crate::layout_ir::Bufferizable for DoubleTie {
            fn alias_info(&self) -> Vec<AliasInfo> {
                vec![
                    AliasInfo {
                        operand: 0,
                        result: 0,
                        sharing: Sharing::Must,
                    },
                    AliasInfo {
                        operand: 1,
                        result: 0,
                        sharing: Sharing::Must,
                    },
                ]
            }
        }
        impl crate::layout_ir::ToDps for DoubleTie {
            fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
                None
            }
        }
        impl LayoutIrOp for DoubleTie {}
        let op = DoubleTie;
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("i1"), cid("i2")],
            results: vec![cid("t")],
        }];
        let facts = ValueFacts {
            output_values: vec![cid("t")],
            ..Default::default()
        };
        let err = match Analyzer::new(&ops, &facts).run() {
            Err(err) => err,
            Ok(_) => panic!("double-tied contract should be rejected"),
        };
        assert!(err.to_string().contains("at most one operand"), "{err}");
    }

    /// NO LAYOUT CHECKING IN THE ENGINE: a same-op read through a view of the
    /// in-place destination is a conflict absent an unconditional permit —
    /// even when the view is identity-like (same layout e-class as the
    /// parent). The engine does not inspect layouts to discover coincidence;
    /// an op safe under such aliasing must be matched with that precondition
    /// in egglog and declare the may-share permit. Here the consumer declares
    /// nothing, so the dest tie is rejected and relocates (out of place).
    #[test]
    fn read_through_view_of_dest_is_rejected_without_permit() {
        use crate::test_support::MockView;
        let view = MockView;
        let consumer = MockOp {
            reads: vec![true, false], // reads the view; dest is the parent
            in_place_operand: Some(1),
            ..Default::default()
        };
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &view,
                operands: vec![cid("x")],
                results: vec![cid("v")],
            },
            AnalysisOp {
                position: 1,
                iface: &consumer,
                operands: vec![cid("v"), cid("x")],
                results: vec![cid("r")],
            },
        ];
        let facts = ValueFacts {
            output_values: vec![cid("r")],
            ..Default::default()
        };
        let mut analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&true), "view admitted");
        assert_eq!(
            analysis.in_place.get(&(1, 1)),
            Some(&false),
            "no permit, no layout inspection: the dest tie relocates"
        );
        assert!(analysis.storage.same(&cid("x"), &cid("v")));
        assert!(!analysis.storage.same(&cid("x"), &cid("r")));
    }

    /// THE DECISION-ORDER POLICY PIN: non-writing ties (views) decide before
    /// writing ties, regardless of op position. A view and a writer contend
    /// for the same operand's storage; the view (earlier position, which
    /// bottom-up alone would decide LAST) still commits first under the
    /// policy, and the writer then sees the view's reader through the merged
    /// alias set and yields to its repair. Swapping the policy changes which
    /// side pays a copy — never soundness.
    #[test]
    fn non_writing_ties_decide_before_writers() {
        use crate::test_support::MockView;
        let view = MockView;
        let writer = MockOp::write_only_dest();
        let reader = MockOp {
            reads: vec![true],
            ..Default::default()
        };
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &view,
                operands: vec![cid("x")],
                results: vec![cid("v")],
            },
            AnalysisOp {
                position: 1,
                iface: &writer,
                operands: vec![cid("x")],
                results: vec![cid("rW")],
            },
            AnalysisOp {
                position: 2,
                iface: &reader,
                operands: vec![cid("v")],
                results: vec![cid("s")],
            },
        ];
        let facts = ValueFacts {
            output_values: vec![cid("rW"), cid("s")],
            ..Default::default()
        };
        let mut analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(
            analysis.in_place.get(&(0, 0)),
            Some(&true),
            "view admits first"
        );
        assert_eq!(
            analysis.in_place.get(&(1, 0)),
            Some(&false),
            "writer sees the view's unordered reader and yields to its repair"
        );
        assert!(analysis.storage.same(&cid("x"), &cid("v")));
        assert!(!analysis.storage.same(&cid("x"), &cid("rW")));
    }

    /// THE GENERIC REPAIR for a kernel-less tie — the soundness backstop that
    /// makes EVERY decision-order policy valid: a REJECTED view cannot be
    /// filled by any kernel, so its fresh buffer is initialized by copying
    /// the parent's bytes (a view over a copy of the buffer, layout
    /// unchanged). The current policy never rejects a view (non-writing ties
    /// decide first, before anything is committed), so the rejection is
    /// forced by hand-building the Analysis; the plan — and its certificate,
    /// which runs inside the lowering — must still hold.
    #[test]
    fn rejected_view_repairs_by_copying_the_parent() {
        use crate::test_support::{MockView, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input("x", "D", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row")])[0].clone();
        g.output(&v, "E");
        let graph = g.build();
        let order = toposort(&graph.dag, None).unwrap();
        let mut analysis = Analysis {
            storage: UnionFind::default(), // no union: the tie was rejected
            in_place: HashMap::from([((0, 0), false)]),
            op_count: 1,
        };
        let assignment = Bufferizer::assign(&graph, &order, &mut analysis, &[]);
        let bt =
            crate::buffer_tensor_ir::build_buffer_tensor_ir(&graph, &order, assignment, &analysis)
                .expect("construction never errors on a rejected view");
        let plan = lower(bt).expect("a rejected view repairs, never errors");
        // The repair copy (x's buffer -> fresh alloc) plus the boundary copy
        // (fresh alloc -> slot E).
        let copies = plan
            .dag
            .node_indices()
            .filter(|&i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 2, "repair + boundary:\n{}", plan.summary());
        assert!(
            matches!(plan.value_buffer[&v], BufferId::Allocated(_)),
            "the view got its own storage:\n{}",
            plan.summary()
        );
    }

    /// RANK 1 (mutation-verified hole): an in-place candidate whose OPERAND
    /// value is itself a program output must be rejected — the boundary reads
    /// it at end-of-program, and a boundary read never happens-before anything.
    /// Pins the `site: None => false` arm of happens_before.
    #[test]
    fn operand_live_at_output_boundary_rejects_in_place() {
        let op = MockOp::write_only_dest();
        let ops = vec![AnalysisOp {
            position: 0,
            iface: &op,
            operands: vec![cid("x")],
            results: vec![cid("res")],
        }];
        let facts = ValueFacts {
            output_values: vec![cid("x"), cid("res")], // x itself live to end
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&false));
    }

    /// RANK 4 (mutation-verified hole): the cross-operand permit excuses
    /// SAME-OP reads only. An unordered CROSS-OP reader of cohabiting
    /// storage must still veto, even when the writer declares the permit for
    /// all pairs — a blanket weakening is a known miscompile.
    #[test]
    fn permit_never_excuses_cross_op_reads() {
        let writer = MockOp {
            reads: vec![false],
            in_place_operand: Some(0),
            not_conflicting: true,
        };
        let reader = MockOp {
            reads: vec![true],
            ..Default::default()
        };
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &writer,
                operands: vec![cid("d")],
                results: vec![cid("r")],
            },
            AnalysisOp {
                position: 1,
                iface: &reader,
                operands: vec![cid("x")],
                results: vec![cid("y")],
            },
        ];
        let mut pinned = HashMap::new();
        pinned.insert(cid("x"), cid("bufB"));
        pinned.insert(cid("d"), cid("bufB"));
        let facts = ValueFacts {
            pinned,
            output_values: vec![cid("r"), cid("y")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(analysis.in_place.get(&(0, 0)), Some(&false));
    }

    /// RANK 5: bottom-up commit order under contention — two unordered ops both
    /// declare the same value as their write-only destination, both results
    /// live. The output-nearer (higher-position) op decides FIRST and wins;
    /// the earlier op is then vetoed by the committed union (the winner's
    /// result now aliases the operand and is read at end-of-program).
    #[test]
    fn contention_resolved_bottom_up() {
        let a = MockOp::write_only_dest();
        let b = MockOp::write_only_dest();
        let ops = vec![
            AnalysisOp {
                position: 0,
                iface: &a,
                operands: vec![cid("v")],
                results: vec![cid("ra")],
            },
            AnalysisOp {
                position: 1,
                iface: &b,
                operands: vec![cid("v")],
                results: vec![cid("rb")],
            },
        ];
        let facts = ValueFacts {
            output_values: vec![cid("ra"), cid("rb")],
            ..Default::default()
        };
        let analysis = Analyzer::new(&ops, &facts).run().unwrap();
        assert_eq!(
            analysis.in_place.get(&(1, 0)),
            Some(&true),
            "output-nearer op wins"
        );
        assert_eq!(
            analysis.in_place.get(&(0, 0)),
            Some(&false),
            "loser vetoed by commit"
        );
    }

    /// RANK 9: NO built-in declares the unconditional sharing permit — ops
    /// whose in-place safety depends on preconditions get matched with those
    /// preconditions in egglog (the Mutating tier) instead of asserting a
    /// blanket permit the engine would have to trust.
    #[test]
    fn builtin_ops_declare_no_unconditional_permits() {
        use crate::reference::ops::{
            AddFunctional, AddMutating, DivFunctional, ExpFunctional, IndexMapApplyMaterialize,
            MaterializeLayoutCopy, MulFunctional, ReduceMax, ReduceSum, SqrtFunctional,
            SqrtMutating,
        };
        use crate::test_support::test_ops::AddMulFused;
        let ops: Vec<Box<dyn LayoutIrOp>> = vec![
            Box::new(SqrtFunctional),
            Box::new(ExpFunctional),
            Box::new(AddFunctional),
            Box::new(MulFunctional),
            Box::new(DivFunctional),
            Box::new(SqrtMutating),
            Box::new(AddMutating),
            Box::new(AddMulFused),
            Box::new(MaterializeLayoutCopy),
            Box::new(ReduceSum { axis: 0 }),
            Box::new(ReduceMax { axis: 0 }),
            Box::new(IndexMapApplyMaterialize { entries: None }),
        ];
        for op in &ops {
            assert!(
                op.alias_info()
                    .iter()
                    .all(|info| info.sharing == Sharing::Must),
                "{}",
                op.label()
            );
        }

        // The ONE deliberate May declarer: its egglog match requires all
        // layouts equal, discharging the permit's precondition at match time.
        // rhs may share the mutated storage; the reverse direction (reading
        // the mutated operand against... nothing ties operand 1) is no permit.
        let alias_safe = crate::reference::ops::AddMutatingInputAliasSafe;
        assert!(permits_sharing(&alias_safe, 1, 0));
        assert!(!permits_sharing(&alias_safe, 0, 1));
    }

    // -------------------------------------------------------------------------
    // Lowering tripwires: the mechanical arms of validate_plan, proven to
    // REJECT. (The SEMANTIC certificate is proven to reject in
    // `buffer_tensor_ir`'s tests, on hand-built BufferTensor graphs.)
    // -------------------------------------------------------------------------

    fn vbuf(name: &str) -> BufferId {
        BufferId::Boundary(cid(name))
    }

    /// Canonicality tripwire: a self-copy is a phantom writer that must have
    /// been folded before the WAR machinery ran.
    #[test]
    fn validator_rejects_self_copy() {
        let d = vbuf("D");
        let mut dag = DiGraph::new();
        dag.add_node(BufferNode::BufferCopy {
            src: d.clone(),
            dst: d.clone(),
        });
        let err = validate_plan(&dag).unwrap_err();
        assert!(err.to_string().contains("self-copy"), "{err}");
    }

    /// Canonicality tripwire: an alloc-like producer writing CALLER storage
    /// means a seeded poison escaped the storage-lifetime pass — caller
    /// buffers are never program-allocated. (On program-minted storage the
    /// same shape IS the BufferAlloc, and legal.)
    #[test]
    fn validator_rejects_alloc_on_caller_storage() {
        let mut dag = DiGraph::new();
        dag.add_node(BufferNode::Compute {
            op: Box::new(EmptyOp),
            reads: Vec::new(),
            writes: vec![vbuf("D")],
            ties: Vec::new(),
        });
        let err = validate_plan(&dag).unwrap_err();
        assert!(err.to_string().contains("caller storage"), "{err}");
    }

    /// Canonicality tripwire: a view surviving as compute means the lowering
    /// skipped its producer-redirect fold.
    #[test]
    fn validator_rejects_unfolded_view() {
        use crate::test_support::MockView;
        let mut dag = DiGraph::new();
        dag.add_node(BufferNode::Compute {
            op: Box::new(MockView),
            reads: vec![vbuf("D")],
            writes: vec![vbuf("D")],
            ties: vec![(0, 0)],
        });
        let err = validate_plan(&dag).unwrap_err();
        assert!(err.to_string().contains("unfolded view"), "{err}");
    }
}
