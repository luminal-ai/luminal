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

use std::collections::{BTreeMap, HashMap, HashSet};

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

/// The OPAQUE layout parameter's bound (resident-geometry cleanup,
/// ruling 2026-08-31; PartialEq dropped by amendment, same ruling):
/// layouts are opaque to the bufferizer — it CLONES and TRANSPORTS
/// layout values, and it never interprets, composes, validates,
/// classifies, sizes, or even COMPARES them (layout equality is
/// enforced in the e-graph, where all spellings of a layout class
/// denote one function — no tripwire is re-derived here). There is no
/// layout vocabulary in core: the CALLER provides the concrete type
/// when it builds plans (both shipped runtimes pass core's decoded
/// struct — see [`crate::layouts::decode_layout_table`]), and backends
/// are free to use layouts core has never heard of.
/// Blanket-implemented: the bound IS the whole contract.
pub trait PlanLayout: Clone + std::fmt::Debug {}
impl<T: Clone + std::fmt::Debug> PlanLayout for T {}

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
///
/// CORRECTED CONTRACT (Austin, 2026-08-31): "The bufferizer is only
/// responsible for determining which buffer IDs back which layout
/// tensors." A buffer row is therefore the ASSIGNMENT's buffer side —
/// identity, ownership/lifetime declarations, the boundary binding key,
/// and WHICH TENSOR the buffer backs — and nothing else. There is no
/// plan-carried geometry (dims/bits/dtype all left the vocabulary):
/// allocation is `span-of-layout` bytes for the backed tensor's layout,
/// which the runtime already knows (it elected it) and which Option B
/// ADDITIONALLY carries here as `layout`.
#[derive(Debug, Clone)]
pub struct Buffer<L: PlanLayout> {
    pub id: BufferId,
    pub access: Access,
    /// Storage deallocation responsibility (declared for boundary buffers;
    /// `Program` for planner-minted interior storage, `Caller` for minted
    /// storage that backs an output and ESCAPES to the user — the
    /// escape-and-disclose cell, ruling 2026-08-27).
    pub freed_by: FreedBy,
    pub owner: Owner,
    pub label: String,
    /// The numeric `BufferLit` id for boundary buffers — the key runtimes
    /// bind caller data by.
    pub lit: Option<i64>,
    /// THE ASSIGNMENT, buffer side: the tensor whose bytes this buffer
    /// holds — the value it was minted for (interior) or pinned to
    /// (boundary). Allocation = look up THIS tensor's layout (in the
    /// runtime's own table, or in `layout` right here) and allocate its
    /// span. Cohabiting values (in-place chains) share the buffer by
    /// e-graph-checked precondition; no voting, no fixpoint, no
    /// contradiction detection reconstructs what assignment already says.
    ///
    /// WHICH cohabitant this names is a SIZING answer, not an identity
    /// claim. On a DPS chain the buffer is minted for the poison
    /// DESTINATION, so `backs` names the poison — which sizes correctly
    /// (a poison clones its tied result's shape/dtype/layout) and says
    /// nothing about which of the cohabitants is "the" value. In
    /// particular: `backs != slot.value` does NOT mean an operand reads
    /// through a fold. A runtime that needs that distinction compares
    /// LAYOUTS on its own type — a cohabitant carries the layout the
    /// buffer was allocated for, a view carries a different function
    /// over the same bytes. Core never makes that comparison (no
    /// `PartialEq` on [`PlanLayout`]); the runtime owns it.
    pub backs: ClassId,
    /// PROTOTYPE (Option B): the backed tensor's elected layout — the `L`
    /// the runtime's decoder minted for `backs`, carried VERBATIM.
    /// Informationally redundant for a live runtime (it knew every layout
    /// before it called bufferize); carried so plans are SELF-CONTAINED
    /// for `load_plan`/hand-built callers, who may read this instead of a
    /// table they never had. Core never interprets, composes, sizes, or
    /// compares it — transport only.
    pub layout: L,
}

/// Where a program output value ends up: its value and the buffer backing
/// it — the ASSIGNMENT's output-boundary rows. VIEW OUTPUTS ARE
/// COMPLETELY LEGAL and fulfilled STRUCTURALLY (corrected contract,
/// 2026-08-31): a view is no-work-same-buffer, so a view-elected slot's
/// `buffer` is its parent's backing storage (escaped or repaired —
/// zero-copy by construction), never a refusal and never a forced dense
/// delivery. The backing buffer may be caller storage or an ESCAPING
/// program allocation (`Owner::System` + `FreedBy::Caller`).
///
/// The runtime hands the user this buffer and may attach the layout it
/// already knows ("maybe it can be typed — not essential"); Option B
/// carries that layout here so externally loaded plans can do the same.
#[derive(Debug, Clone)]
pub struct OutputBinding<L: PlanLayout> {
    pub index: usize,
    pub value: ClassId,
    pub buffer: BufferId,
    /// PROTOTYPE (Option B): the output value's elected layout, carried
    /// VERBATIM from the runtime's decoded table — for a view election,
    /// the view's COMPOSED layout as the e-graph minted it, already
    /// addressing the backing buffer's bytes. Redundant for a live
    /// runtime; self-containment for loaded plans.
    pub layout: L,
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

// The hop machinery (`AccessHop` / `ComposedAccess`) is DELETED
// (corrected contract, 2026-08-31): recording per-slot view-fold chains
// was the planner COMPOSING layout knowledge — the e-graph already mints
// every view value's composed layout at view creation, and the runtime's
// decoded `L` for that value IS the read path. The planner records
// assignment, never access expressions.

/// Per-slot descriptor on a plan node: the VALUE occupying an
/// operand/result slot, the buffer backing it — the ASSIGNMENT restated
/// per slot — plus the Option-B carried layout. Identity is filled at
/// lowering from the BufferTensor operands/results. Whatever else an op's
/// kernel wants to remember from its claimed site (extents, dtypes,
/// masks) is the OP RECORD's private business (its matcher captured it) —
/// never plan vocabulary.
#[derive(Debug, Clone)]
pub struct SlotDescriptor<L: PlanLayout> {
    /// The value occupying this slot (operand read / result written).
    pub value: ClassId,
    /// The buffer backing the slot (same entry as `reads`/`writes`).
    pub buffer: BufferId,
    /// PROTOTYPE (Option B): the slot value's elected layout, carried
    /// VERBATIM from the runtime's decoded table (TOTAL: every
    /// LayoutTensor carries a layout by construction; a missing row bails
    /// at lowering, never defaults). For an operand reading through
    /// folded views this is the view's COMPOSED layout as the e-graph
    /// minted it, addressing the residence's bytes directly.
    pub layout: L,
}

/// A node in the buffer IR. `Compute` nodes are the original ops, now reading and
/// writing buffers; `BufferCopy` is the only genuinely new operation (bufferizer-
/// inserted materialization); the boundaries are pinned buffers.
#[derive(Debug, Clone)]
pub enum BufferNode<L: PlanLayout> {
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
        /// Per-operand descriptors, parallel to `reads`.
        operand_info: Vec<SlotDescriptor<L>>,
        /// Per-result descriptors, parallel to `writes`.
        result_info: Vec<SlotDescriptor<L>>,
    },
    /// A bufferizer-inserted copy of `src`'s bytes into `dst`.
    ///
    /// # THE BUFFERCOPY CONTRACT (Austin, ruled 2026-08-31 — stated, never implied)
    ///
    /// **The node carries ONLY the pair `{src, dst}`.** There is no third
    /// field. (The former `value` existed solely to feed the
    /// writer-identity dims join, and that join is deleted with the plan's
    /// geometry vocabulary.)
    ///
    /// **Semantics: a DUMB EXACT-SIZE WHOLE-BUFFER COPY.** Verbatim:
    /// "semantically it is a dumb exact size copy. If a runtime chooses to
    /// do resource reuse and do unequal sized buffer that is an entirely
    /// runtime owned choice." No layout awareness, no element walk, no
    /// partial ranges, no shape conversion. `dst` ends holding `src`'s
    /// bytes.
    ///
    /// **ORDERING IS THE RUNTIME OBLIGATION.** Verbatim: "the runtime is
    /// responsible for analyzing which ops depend on the Copy and making
    /// sure that appropriate ordering is maintained." The plan supplies the
    /// DEPENDENCY STRUCTURE — the dag, including [`EdgeKind::Anti`]
    /// write-after-read edges — and nothing more; scheduling, hazard
    /// handling, stream/queue placement and any barrier the hardware needs
    /// are runtime-side. The bufferizer does not choose an execution order.
    ///
    /// **The three causes a copy is minted** (each documented again at its
    /// mint site — in [`lower`] and in
    /// [`crate::buffer_tensor_ir::build_buffer_tensor_ir`], which is where
    /// the decision to copy is actually taken):
    ///
    /// 1. **Residence conflict repair** — an in-place candidate was
    ///    rejected by the conflict engine, so the result gets fresh storage
    ///    and the operand's bytes are copied in first.
    /// 2. **Boundary placement** — a tensor is bound to a SPECIFIC caller
    ///    buffer whose producing residence is elsewhere, so its bytes are
    ///    moved into the caller's storage.
    /// 3. **Lifetime repair** — a value must outlive the storage it
    ///    currently occupies (an escape, or a buffer about to be reused),
    ///    so it is relocated into storage with the right lifetime.
    ///
    /// A FOLDED resident is copied by copying its BASE STORAGE (the parent
    /// buffer) whole — copying the base buffer COUNTS AS DELIVERY — and
    /// re-rooting the fold onto the copy; a copy materialized INTO a
    /// specific layout is a different LAYOUTTENSOR candidate in the
    /// e-graph, discovered via search, never a copy mode. Repair
    /// destinations are always FRESH single-writer buffers minted by the
    /// bufferizer; the e-graph never represents buffer-level choices —
    /// repairs are deterministic bufferizer insertions ("we don't search
    /// over buffer tensors in the egraph... We're just going to insert
    /// buffer copies as needed to repair").
    BufferCopy { src: BufferId, dst: BufferId },
    /// A program output: each slot's value pinned into its destination buffer.
    BufferOutput { slots: Vec<OutputBinding<L>> },
}

// PROTOTYPE (Option B): `walk_layout_index` — the hop-chain host walker —
// LEFT CORE (Austin's ruling: it "probably gets deleted / moved to
// something called 'test equality' or something in the testing crate").
// Element readback through a returned layout is a TEST concern: the
// testing crate's `test_equality` module evaluates the runtime's own
// layout vocabulary ((buffer, layout) pairs, mirror-struct expressions),
// with no core involvement. Core transports layouts; it never reads
// through them.

/// The out-of-place bufferization result: a new, independent dataflow graph whose
/// values are all backed by buffers and whose copies are first-class nodes. The
/// source [`ExtractedGraph`] is left untouched.
#[derive(Debug, Clone)]
pub struct BufferIrGraph<L: PlanLayout> {
    pub dag: DiGraph<BufferNode<L>, BufferEdge>,
    /// Every distinct buffer in the plan, by id.
    pub buffers: HashMap<BufferId, Buffer<L>>,
    /// The buffer holding each value (values collapse onto buffers via reuse).
    /// A `BTreeMap` deliberately: residual iterations (diagnostics) walk it
    /// in value order, so their messages are stable run-to-run — never
    /// std-HashMap hash order.
    pub value_buffer: BTreeMap<ClassId, BufferId>,
    /// The `BufferOutput` node(s).
    pub outputs: Vec<NodeIndex>,
}

impl<L: PlanLayout> BufferIrGraph<L> {
    /// THE ASSIGNMENT, queried buffer-side: which tensor does `buffer`
    /// back? Allocation = look up this tensor's layout (the runtime's own
    /// table, or [`Buffer::layout`] carried alongside) and allocate its
    /// span — no walk, no voting.
    pub fn backed_tensor(&self, buffer: &BufferId) -> Option<&ClassId> {
        self.buffers.get(buffer).map(|b| &b.backs)
    }

    /// THE ASSIGNMENT, queried tensor-side: which buffer backs `value`?
    /// (Boundary tensors included — the boundary-condition query.)
    pub fn buffer_of(&self, value: &ClassId) -> Option<&BufferId> {
        self.value_buffer.get(value)
    }
}

impl<L: PlanLayout> BufferIrGraph<L> {
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

        let mut buffers: Vec<&Buffer<L>> = self.buffers.values().collect();
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
                BufferNode::BufferCopy { src, dst, .. } => op_lines.push(format!(
                    "  BufferCopy: [{}] -> [{}]",
                    self.buffer_name(src),
                    self.buffer_name(dst),
                )),
                BufferNode::BufferOutput { slots } => {
                    let mut slots: Vec<&OutputBinding<L>> = slots.iter().collect();
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
                BufferNode::BufferCopy { src, dst, .. } => (
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

    // Undefinedness PROPAGATES across non-writing Must ties (views): a view's
    // contents ARE its operand's contents, so a view of an undefined value is
    // itself undefined. Without this, a poison behind a view slips both doors
    // below and only surfaces as a deep plan-construction failure — and once
    // destination seeding can cross views, it would surface as uninitialized
    // bytes delivered to a caller. Fixpoint so chains propagate hop-by-hop.
    // DPS destination ties are untouched: their ops WRITE the tied result, so
    // the !result_writes_memory guard excludes them. (Ruling 2026-08-26.)
    loop {
        let mut grew = false;
        for node in graph.dag.node_weights() {
            let ExtractedNode::LayoutOp(op) = node else {
                continue;
            };
            for (operand, result) in must_ties(op.op.as_ref()) {
                if !op.op.result_writes_memory(result)
                    && op
                        .inputs
                        .get(operand)
                        .is_some_and(|input| undefined.contains(&input.value))
                    && op
                        .outputs
                        .get(result)
                        .is_some_and(|output| !undefined.contains(&output.eclass))
                {
                    undefined.insert(op.outputs[result].eclass.clone());
                    grew = true;
                }
            }
        }
        if !grew {
            break;
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
///
/// `layouts` is the decoded-layout table, keyed by LAYOUT e-class (each
/// value's `LayoutTensorInfo::layout.eclass`): the runtime's decoder
/// produced one opaque `L` per layout class it elected (see
/// [`crate::layouts::decode_layout_table`]). The table must cover
/// every value's layout class — a miss is a loud error, never a default
/// (every LayoutTensor carries a Layout by construction, so a missing
/// row means the decoder refused and the plan must too). Keying by
/// layout class also covers the DPS poisons for free: `dps_rewrite`
/// clones the tied result's layout (e-class included) onto each poison
/// destination, so the poison's `L` IS the result's.
pub fn bufferize<L: PlanLayout>(
    graph: &ExtractedGraph,
    layouts: &HashMap<ClassId, L>,
) -> Result<BufferIrGraph<L>> {
    // `layouts` is keyed by VALUE e-class (the runtime's decoded table,
    // [`crate::layouts::decode_layout_table`]).
    let value_layouts = extraction_layouts(graph, layouts)?;
    let mut plan = lower(buffer_tensor_plan(graph, &value_layouts)?, &value_layouts)?;
    annotate_boundary_lits(&mut plan, graph);
    Ok(plan)
}

/// Every extraction VALUE's decoded layout, keyed by value e-class —
/// TOTAL: every LayoutTensor carries a Layout by construction, so a value
/// with no decoded row is a loud refusal, never a default. This is the
/// ONE per-value fact the planner transports; dims, element bits, and
/// dtype are the runtime's own extraction-side knowledge and never enter
/// the plan (corrected contract, 2026-08-31).
pub(crate) fn extraction_layouts<L: PlanLayout>(
    graph: &ExtractedGraph,
    layouts: &HashMap<ClassId, L>,
) -> Result<HashMap<ClassId, L>> {
    let mut value_layouts: HashMap<ClassId, L> = HashMap::new();
    let record = |value: &crate::layout_ir::LayoutTensorInfo,
                  table: &mut HashMap<ClassId, L>|
     -> Result<()> {
        if table.contains_key(&value.eclass) {
            return Ok(());
        }
        let layout = layouts.get(&value.eclass).cloned().ok_or_else(|| {
            anyhow::anyhow!(
                "value {} (layout class {}) has no row in the decoded \
                 layout table — every LayoutTensor carries a Layout by \
                 construction, so the decoder must cover it or the plan \
                 must refuse",
                value.eclass,
                value.layout.eclass,
            )
        })?;
        table.insert(value.eclass.clone(), layout);
        Ok(())
    };
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => record(&input.value, &mut value_layouts)?,
            ExtractedNode::LayoutOp(op) => {
                for output in &op.outputs {
                    record(output, &mut value_layouts)?;
                }
            }
            ExtractedNode::BufferOutput(_) => {}
        }
    }
    Ok(value_layouts)
}

/// Thread the boundary `BufferLit` keys onto the plan's buffers — the
/// binding surface runtimes stage caller data by. This is the WHOLE of
/// post-lowering annotation now (corrected contract, 2026-08-31): there
/// is NO sizing walk. Allocation is an assignment lookup — for each plan
/// buffer, `backs` names the tensor it holds, that tensor's layout (the
/// runtime's own knowledge, or the carried [`Buffer::layout`]) gives the
/// span, and every consumer takes the BufferId blindly: no voting, no
/// fixpoint, no contradiction detection — "all the necessary
/// preconditions have been checked in the egraph."
///
/// ALLOCATION DOCTRINE (ruling 2026-08-27): every allocation site's bytes
/// are its LAYOUT'S REQUIRED SPAN. CAPACITY CHECKS ARE AN OPEN ITEM,
/// deliberately not implemented: provided (caller) buffers are
/// blanket-assumed adequate for what the plan lands in them.
fn annotate_boundary_lits<L: PlanLayout>(plan: &mut BufferIrGraph<L>, graph: &ExtractedGraph) {
    let mut boundary_lits: HashMap<ClassId, i64> = HashMap::new();
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => {
                if let Some(lit) = input.buffer.lit {
                    boundary_lits.insert(input.buffer.id_eclass.clone(), lit);
                }
            }
            ExtractedNode::LayoutOp(_) => {}
            ExtractedNode::BufferOutput(output) => {
                for slot in &output.slots {
                    if let Some(lit) = slot.buffer.lit {
                        boundary_lits.insert(slot.buffer.id_eclass.clone(), lit);
                    }
                }
            }
        }
    }
    for buffer in plan.buffers.values_mut() {
        if let BufferId::Boundary(eclass) = &buffer.id {
            buffer.lit = boundary_lits.get(eclass).copied();
        }
    }
}

/// The stable marker of the cyclic-extracted-graph refusal — matched by
/// the implementation search's sampler tripwire, which treats a cycle
/// reaching bufferize as a SAMPLER BUG (the genome's chosen edges were
/// supposed to be acyclic) rather than as an ordinary refusal.
pub const EXTRACTED_GRAPH_CYCLE: &str = "extracted graph has a cycle";

/// NAME THE CYCLE (2026-09-02). `toposort` hands back one node it could
/// not order; the cycle is inside that node's strongly connected
/// component, so report the whole component's nodes — a plan whose input
/// is computed from something that reads it back says so by name instead
/// of by "there is a cycle somewhere".
fn describe_cycle(graph: &ExtractedGraph, seed: petgraph::graph::NodeIndex) -> String {
    let label = |index: petgraph::graph::NodeIndex| -> String {
        match &graph.dag[index] {
            ExtractedNode::BufferInput(input) => {
                format!("input {}", input.value.label)
            }
            ExtractedNode::LayoutOp(op) => {
                let outputs: Vec<&str> = op
                    .outputs
                    .iter()
                    .map(|output| output.label.as_str())
                    .collect();
                format!("{} -> {}", op.op.label(), outputs.join(", "))
            }
            ExtractedNode::BufferOutput(output) => format!("output {}", output.label),
        }
    };
    let members: Vec<String> = petgraph::algo::tarjan_scc(&graph.dag)
        .into_iter()
        .find(|scc| scc.contains(&seed))
        .unwrap_or_else(|| vec![seed])
        .into_iter()
        .map(label)
        .collect();
    format!(
        "the cycle runs through {} node(s): [{}]",
        members.len(),
        members.join(" | ")
    )
}

/// The PLANNING half of [`bufferize`]: validate the input program, analyze,
/// assign, build the BufferTensor graph, install anti-dependence ordering,
/// certify, and run the storage-level rewrites — everything semantic. The
/// returned graph is the finished BufferTensor program (optimized: poisons
/// folded, dead buffers dropped) and the audit artifact `main` renders;
/// [`lower`] erases it into the executable plan.
pub(crate) fn buffer_tensor_plan<L: PlanLayout>(
    graph: &ExtractedGraph,
    value_layouts: &HashMap<ClassId, L>,
) -> Result<crate::buffer_tensor_ir::BufferTensorIrGraph<L>> {
    validate_input_program(graph)?;
    let order = toposort(&graph.dag, None).map_err(|cycle| {
        anyhow::anyhow!(
            "{EXTRACTED_GRAPH_CYCLE}; cannot bufferize: {}",
            describe_cycle(graph, cycle.node_id())
        )
    })?;

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
    let assignment = Bufferizer::assign(graph, &order, &mut analysis, &seeds, value_layouts)?;
    let mut bt = crate::buffer_tensor_ir::build_buffer_tensor_ir(
        graph,
        &order,
        assignment,
        &analysis,
        value_layouts,
    )?;
    crate::buffer_tensor_ir::install_anti_edges(&mut bt);
    // The certificate runs AFTER the storage-lifetime pass: its lifetime arms
    // certify what `optimize` constructs (allocs, frees, their ordering
    // edges), and the residency arms only gain edges by the reorder — the
    // pass adds ordering and removes nothing a consumer reads.
    let bt = crate::buffer_tensor_ir::optimize(bt)?;
    crate::buffer_tensor_ir::validate(&bt)?;
    Ok(bt)
}

// -----------------------------------------------------------------------------
// Buffer assignment: value -> BufferId
// -----------------------------------------------------------------------------

/// Assigns a buffer to every value and interns every boundary buffer.
pub(crate) struct Bufferizer<L: PlanLayout> {
    pub(crate) buffers: HashMap<BufferId, Buffer<L>>,
    pub(crate) value_buffer: BTreeMap<ClassId, BufferId>,
    /// Buffer chosen for each storage-class representative (so all values an
    /// admitted decision placed in one allocation — an in-place producer and
    /// its operand, a view and its parent — share one buffer).
    rep_buffer: HashMap<ClassId, BufferId>,
    /// The next fresh `Allocated` id. `pub(crate)` so BT construction can
    /// keep minting from the same sequence (repair destinations and
    /// escape repairs are fresh single-writer buffers, never reused ids).
    pub(crate) next_alloc: u32,
}

/// Manual `Default` (a derive would demand `L: Default`; empty maps need
/// no layout values).
impl<L: PlanLayout> Default for Bufferizer<L> {
    fn default() -> Self {
        Bufferizer {
            buffers: HashMap::new(),
            value_buffer: BTreeMap::new(),
            rep_buffer: HashMap::new(),
            next_alloc: 0,
        }
    }
}

impl<L: PlanLayout> Bufferizer<L> {
    fn assign(
        graph: &ExtractedGraph,
        order: &[NodeIndex],
        analysis: &mut Analysis,
        seeds: &[Seed],
        value_layouts: &HashMap<ClassId, L>,
    ) -> Result<Self> {
        let mut this = Bufferizer::default();
        // THE ASSIGNMENT SEED: the value each buffer is minted for (or a
        // boundary is pinned to) is the tensor the buffer BACKS, and that
        // tensor supplies the carried layout — mint-time facts, final.
        // There is no later writer join overriding them: cohabitants
        // share storage by e-graph-checked precondition. The table is
        // total over graph values, so a miss is a planner bug and bails
        // loudly.
        let layout_of = |value: &ClassId| -> Result<L> {
            value_layouts.get(value).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "value {value} has no decoded layout — every graph \
                     value records one before assignment"
                )
            })
        };

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
                    input.value.eclass.clone(),
                    layout_of(&input.value.eclass)?,
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
                seed.poison.clone(),
                layout_of(&seed.poison)?,
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
                        input.value.eclass.clone(),
                        layout_of(&input.value.eclass)?,
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
                            None => this.allocate(
                                output.label.clone(),
                                output.eclass.clone(),
                                layout_of(&output.eclass)?,
                            ),
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
                            slot.value.clone(),
                            layout_of(&slot.value)?,
                        );
                    }
                }
            }
        }
        Ok(this)
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
        backs: ClassId,
        layout: L,
    ) -> BufferId {
        let id = BufferId::Boundary(eclass.clone());
        self.buffers.entry(id.clone()).or_insert_with(|| Buffer {
            id: id.clone(),
            access,
            freed_by,
            owner: Owner::Caller,
            label,
            lit: None,
            backs,
            layout,
        });
        id
    }

    fn allocate(&mut self, label: String, backs: ClassId, layout: L) -> BufferId {
        let id = BufferId::Allocated(self.next_alloc);
        self.next_alloc += 1;
        self.buffers.insert(
            id.clone(),
            Buffer {
                id: id.clone(),
                access: Access::ReadWrite,
                // Planner-minted storage defaults to program-freed; the
                // BT-level residence split (ruling 2026-08-27) flips a
                // buffer backing an elected view output to
                // `FreedBy::Caller` — the ESCAPE cell: the buffer is
                // handed to the caller to manage, and optimize mints no
                // free for it.
                freed_by: FreedBy::Program,
                owner: Owner::System,
                label,
                lit: None,
                backs,
                layout,
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
fn validate_plan<L: PlanLayout>(dag: &DiGraph<BufferNode<L>, BufferEdge>) -> Result<()> {
    for index in dag.node_indices() {
        match &dag[index] {
            BufferNode::BufferCopy { src, dst, .. } if src == dst => {
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
                ..
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
pub(crate) fn lower<L: PlanLayout>(
    bt: crate::buffer_tensor_ir::BufferTensorIrGraph<L>,
    value_layouts: &HashMap<ClassId, L>,
) -> Result<BufferIrGraph<L>> {
    use crate::buffer_tensor_ir::{BtNode, BufferTensor, BufferTensorIrGraph};
    let BufferTensorIrGraph {
        dag: bt_dag,
        buffers,
        value_buffer,
    } = bt;

    use petgraph::visit::EdgeRef;

    // Per-slot descriptor: the assignment restated per slot — (value,
    // buffer) identity from the BufferTensor pair, plus the Option-B
    // carried layout from the decoded table.
    let describe = |tensor: &BufferTensor| -> Result<SlotDescriptor<L>> {
        // PROTOTYPE (Option B): the slot's layout is TOTAL. Every
        // EXTRACTION value has a decoded-layout row; the one legal miss
        // is a planner-SYNTHESIZED undefined value (a BufferAlloc poison
        // minted by `optimize` — never read, never returned), whose slot
        // carries its residence buffer's mint-seed layout (also total).
        // Anything else is a loud bail, never a default.
        let layout = match value_layouts.get(&tensor.value) {
            Some(layout) => layout.clone(),
            None => buffers
                .get(&tensor.buffer)
                .map(|buffer: &Buffer<L>| buffer.layout.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "lowering: value {} occupies a slot with neither a \
                         decoded-layout row nor an interned buffer — a \
                         layoutless slot is unrepresentable (fail-closed)",
                        tensor.value
                    )
                })?,
        };
        Ok(SlotDescriptor {
            value: tensor.value.clone(),
            buffer: tensor.buffer.clone(),
            layout,
        })
    };

    let mut dag: DiGraph<BufferNode<L>, BufferEdge> = DiGraph::new();
    // The lowered node producing each RESIDENCE (value, buffer) — a copy
    // gives one value a second residence with a distinct producer, so the
    // pair is the key, never the value alone.
    let mut producer: HashMap<(ClassId, BufferId), NodeIndex> = HashMap::new();
    // BT node -> lowered node, for transferring Anti edges (folded nodes have
    // no entry).
    let mut lowered: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // Each folded view value's ROOT parent — the non-view resident whose
    // bytes the fold chain ultimately addresses. A base-storage copy of a
    // folded resident (ruling 2026-08-27) transports THIS value: the dumb
    // whole-buffer memcpy moves the parent's bytes, the dims join sizes the
    // destination as the parent, and the fold re-roots onto the copy.
    let mut fold_root: HashMap<ClassId, ClassId> = HashMap::new();
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
                        // The fold records NO access expression (corrected
                        // contract, 2026-08-31): the view value's own
                        // decoded layout — already composed by the
                        // e-graph at view creation — is how its readers
                        // address the parent residence's bytes. Only the
                        // fold ROOT is tracked, for the base-storage copy
                        // and the escape re-root.
                        let root = fold_root
                            .get(&parent.value)
                            .cloned()
                            .unwrap_or_else(|| parent.value.clone());
                        fold_root.insert(tensor.value.clone(), root);
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
                    // MINT SITE — cause 1 (residence conflict repair) and
                    // cause 3 (lifetime repair): the BT graph already
                    // decided a transport was needed; this is where it
                    // becomes a plan node. THE CONTRACT (see
                    // [`BufferNode::BufferCopy`]): a DUMB EXACT-SIZE
                    // WHOLE-BUFFER copy, and ORDERING IS THE RUNTIME'S
                    // OBLIGATION — all this site owes is the dependency
                    // structure (the data edge below, plus the WAR
                    // anti-edges added later).
                    //
                    // A FOLDED resident (the value addresses `src` through
                    // folded views): the copy is dumb and whole-buffer, so
                    // the one legal lowering is the BASE-STORAGE copy —
                    // memcpy the PARENT buffer whole (copying the base
                    // buffer counts as delivery; src and dst are the same
                    // parent-sized storage, exact-size by construction) and
                    // re-root the fold onto the copy: consumers keep their
                    // own composed layouts, now anchored on the copied
                    // buffer. Possibly expensive, and that is fine — search
                    // finds cheaper routes via cost. The bufferizer never
                    // switches to the materialize route (election committed
                    // the genome); a materializing copy is a LayoutTensor
                    // candidate in the e-graph, never a copy mode.
                    if let Some(root) = fold_root.get(&src.value).cloned() {
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
                        // THE RE-ROOT: the parent value gains a residence in
                        // dst (its bytes are what the memcpy moved), and the
                        // folded value's dst residence is produced by the
                        // copy — downstream consumers of (value, dst) hang
                        // off the copy and read the copied parent bytes
                        // through their value's own (unchanged) layout.
                        producer.insert((root, dst.buffer.clone()), copy);
                        producer.insert(residence(dst), copy);
                        lowered.insert(index, copy);
                        continue;
                    }
                    // The unfolded transport: same value, different buffer.
                    // Exact-size by construction (both buffers back the same
                    // tensor's layout); ordering is the runtime's.
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

                // COMPUTE: erase values into buffers — but keep each slot's
                // value + geometry on the node's descriptors (per-node
                // descriptor schema, approved 2026-08-26b).
                let reads: Vec<BufferId> = operands.iter().map(|t| t.buffer.clone()).collect();
                let writes: Vec<BufferId> = results.iter().map(|t| t.buffer.clone()).collect();
                let node = dag.add_node(BufferNode::Compute {
                    op: op.clone(),
                    reads: reads.clone(),
                    writes: writes.clone(),
                    ties: ties.clone(),
                    operand_info: operands.iter().map(&describe).collect::<Result<Vec<_>>>()?,
                    result_info: results.iter().map(&describe).collect::<Result<Vec<_>>>()?,
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
                // EVERY slot carries its elected layout uniformly: a view
                // output is fulfilled STRUCTURALLY (its buffer is its
                // parent's backing storage — zero-copy by construction),
                // and the returned layout is the slot value's own decoded
                // `L`, verbatim. Total, loud on a miss.
                let bindings: Vec<OutputBinding<L>> = slots
                    .iter()
                    .enumerate()
                    .map(|(index, slot)| {
                        let Some(layout) = value_layouts.get(&slot.value) else {
                            anyhow::bail!(
                                "lowering: output slot value {} has no \
                                 decoded-layout row (fail-closed)",
                                slot.value
                            );
                        };
                        Ok(OutputBinding {
                            index,
                            value: slot.value.clone(),
                            buffer: slot.buffer.clone(),
                            layout: layout.clone(),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
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
                    // THE ESCAPE RE-ROOT (ruling 2026-08-27): a folded slot
                    // residence is produced by whatever produced its fold
                    // ROOT's bytes in the same buffer — a donated-residence
                    // output was repaired at BT construction by ONE
                    // base-storage copy transporting the root into the
                    // escaping buffer, and every view of that base (slots
                    // may legally share one escaping buffer) resolves to
                    // that copy here, with no further copy minted.
                    if let Some(root) = fold_root.get(&slot.value) {
                        if let Some(&from) = producer.get(&(root.clone(), slot.buffer.clone())) {
                            producer.insert(residence(slot), from);
                            continue;
                        }
                        // A folded value with neither its own residence nor
                        // its root's bytes in the slot's buffer means BT
                        // construction skipped the residence split — a
                        // planner bug, never a byte-move (which would
                        // silently drop the fold).
                        anyhow::bail!(
                            "lowering invariant broken: output slot {:?} binds folded \
                             view resident {} but neither the view nor its fold root \
                             {} resides there — the BT-level residence split must run \
                             before lowering",
                            slot.buffer,
                            slot.value,
                            root,
                        );
                    }
                    let source = producer
                        .iter()
                        .find(|((value, buffer), _)| *value == slot.value && *buffer != slot.buffer)
                        .map(|((_, buffer), node)| (buffer.clone(), *node));
                    if let Some((src_buffer, from)) = source {
                        // MINT SITE — cause 2 (BOUNDARY PLACEMENT): this
                        // tensor is bound to a SPECIFIC caller buffer, and
                        // its producing residence is elsewhere, so its bytes
                        // move into the caller's storage. Same contract as
                        // every other copy (see [`BufferNode::BufferCopy`]):
                        // dumb, exact-size, whole-buffer; the plan supplies
                        // only the dependency structure (the data edge below
                        // + WAR anti-edges), and the runtime owns ordering.
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

    /// THE CYCLE IS NAMED (2026-09-02). `toposort` only says "there is a
    /// cycle"; the refusal now walks the strongly connected component the
    /// unorderable node sits in and prints its members, so the plan that
    /// computes a value from something that reads it back says WHICH ops
    /// those are. (Since input terminals became leaves the extractor no
    /// longer emits such a graph — this board builds one by hand, which is
    /// also what the implementation search's tripwire watches for.)
    #[test]
    fn a_cyclic_extracted_graph_names_the_cycle_members() {
        use crate::layout_ir::ExtractedEdge;
        use crate::test_support::TestGraph;

        let mut g = TestGraph::new();
        let x = g.input("x", "A", Access::ReadOnly, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        let z = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&y],
            &[("z", "rm")],
        )[0]
        .clone();
        g.output(&z, "D");
        let mut graph = g.build();

        let node_of = |graph: &ExtractedGraph, label: &str| {
            graph
                .dag
                .node_indices()
                .find(|index| match &graph.dag[*index] {
                    ExtractedNode::LayoutOp(op) => op.outputs.iter().any(|out| out.label == label),
                    _ => false,
                })
                .expect("the board built this op")
        };
        let (writes_y, writes_z) = (node_of(&graph, "y"), node_of(&graph, "z"));
        graph.dag.add_edge(
            writes_z,
            writes_y,
            ExtractedEdge {
                value: z.clone(),
                port: "in0".to_string(),
            },
        );

        let err = crate::test_support::bufferize_mock(&graph).unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains(EXTRACTED_GRAPH_CYCLE),
            "the refusal keeps its stable marker: {text}"
        );
        assert!(
            text.contains("-> y") && text.contains("-> z"),
            "the refusal names both ops in the cycle: {text}"
        );
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
        let err = crate::test_support::bufferize_mock(&g.build()).unwrap_err();
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

    // (The registry-contract pins — built-in ops declare out-of-place
    // defaults and no unconditional sharing permits — moved to
    // `luminal_reference::ops` with the registry in Step B.)

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
        let err = crate::test_support::bufferize_mock(&g.build()).unwrap_err();
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
        let err = crate::test_support::bufferize_mock(&g.build()).unwrap_err();
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
        let err = crate::test_support::bufferize_mock(&g.build()).unwrap_err();
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
    ///
    /// RULING 2026-08-27 (escape-and-disclose): the repaired view is
    /// consumed by an INTERIOR compute op here — a folded view bound
    /// straight to an output slot now escapes with its layout disclosed
    /// (see `folded_output_on_minted_residence_escapes_in_place`), so an
    /// output-bound spelling would pin the escape path, not this repair.
    #[test]
    fn rejected_view_repairs_by_copying_the_parent() {
        use crate::test_support::{MockOp, MockView, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input("x", "D", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "E");
        let graph = g.build();
        let order = toposort(&graph.dag, None).unwrap();
        let mut analysis = Analysis {
            storage: UnionFind::default(), // no union: the tie was rejected
            in_place: HashMap::from([((0, 0), false)]),
            op_count: 2,
        };
        let geometry = extraction_layouts(&graph, &crate::test_support::mock_layout_table(&graph))
            .expect("mock layouts are total over the graph");
        let assignment = Bufferizer::assign(&graph, &order, &mut analysis, &[], &geometry)
            .expect("assignment runs");
        let bt = crate::buffer_tensor_ir::build_buffer_tensor_ir(
            &graph, &order, assignment, &analysis, &geometry,
        )
        .expect("construction never errors on a rejected view");
        let plan = lower(bt, &geometry).expect("a rejected view repairs, never errors");
        // The repair copy (x's buffer -> fresh alloc) plus the boundary copy
        // delivering r (r's alloc -> slot E).
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
        let mut dag: DiGraph<BufferNode<crate::test_support::MockLayout>, BufferEdge> =
            DiGraph::new();
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
        let mut dag: DiGraph<BufferNode<crate::test_support::MockLayout>, BufferEdge> =
            DiGraph::new();
        dag.add_node(BufferNode::Compute {
            op: Box::new(EmptyOp),
            reads: Vec::new(),
            writes: vec![vbuf("D")],
            ties: Vec::new(),
            operand_info: Vec::new(),
            result_info: Vec::new(),
        });
        let err = validate_plan(&dag).unwrap_err();
        assert!(err.to_string().contains("caller storage"), "{err}");
    }

    /// Canonicality tripwire: a view surviving as compute means the lowering
    /// skipped its producer-redirect fold.
    #[test]
    fn validator_rejects_unfolded_view() {
        use crate::test_support::MockView;
        let mut dag: DiGraph<BufferNode<crate::test_support::MockLayout>, BufferEdge> =
            DiGraph::new();
        dag.add_node(BufferNode::Compute {
            op: Box::new(MockView),
            reads: vec![vbuf("D")],
            writes: vec![vbuf("D")],
            ties: vec![(0, 0)],
            operand_info: Vec::new(),
            result_info: Vec::new(),
        });
        let err = validate_plan(&dag).unwrap_err();
        assert!(err.to_string().contains("unfolded view"), "{err}");
    }

    /// RULING 2026-08-26 (a): undefinedness propagates across non-writing
    /// Must ties. A view of an undefined value READ by a computing op is
    /// rejected at validation — previously the view's fresh value slipped
    /// the direct poison check.
    #[test]
    fn view_of_undefined_read_is_rejected_at_validation() {
        use crate::test_support::{MockView, TestGraph};
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let v = g.op(Box::new(MockView), &[&e], &[("v", "view")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "out");
        let err = crate::test_support::bufferize_mock(&g.build()).unwrap_err();
        assert!(
            err.to_string().contains("reads undefined contents"),
            "expected the propagated poison door, got: {err:#}"
        );
    }

    /// RULING 2026-08-26 (a)+(b): a view of an undefined value bound
    /// STRAIGHT to an output slot — no reader op at all — is rejected at
    /// validation with a structured error. Previously this slipped both
    /// validation doors and aborted the process at the free-stage expect
    /// in buffer_tensor_ir.rs (now also demoted to a bail as
    /// defense-in-depth).
    #[test]
    fn view_of_undefined_bound_to_output_slot_is_rejected_at_validation() {
        use crate::test_support::{MockView, TestGraph};
        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let v = g.op(Box::new(MockView), &[&e], &[("v", "view")])[0].clone();
        g.output(&v, "out");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::test_support::bufferize_mock(&g.build())
        }));
        match result {
            Ok(Ok(plan)) => panic!(
                "LAUNDERED: undefined bytes delivered to a bound output through a view:\n{}",
                plan.summary()
            ),
            Ok(Err(err)) => assert!(
                err.to_string().contains("binds the undefined"),
                "expected the propagated slot-poison door, got: {err:#}"
            ),
            Err(_) => panic!("panicked — the ruling demands a structured error, never an abort"),
        }
    }

    // -------------------------------------------------------------------------
    // View folds: the slot carries the view value's OWN layout (corrected
    // contract, 2026-08-31 — the hop machinery is deleted; the e-graph
    // mints every view's composed layout, and the decoded `L` is the
    // read path).
    // -------------------------------------------------------------------------

    /// A folded view redirects its consumer to the parent's buffer while
    /// the slot keeps the VIEW's identity: its value, and its own
    /// decoded layout, verbatim from the table.
    #[test]
    fn folded_view_slot_carries_the_views_own_layout() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockViewWithMap, TestGraph};
        let entries = vec![IotaExpr::Coord(0), IotaExpr::Coord(1)];
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: entries.clone(),
            }),
            &[&x],
            &[("v", "row0")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let graph = g.build();
        let plan = crate::test_support::bufferize_mock(&graph).expect("bufferizes");

        let consumer = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::Compute {
                    op,
                    operand_info,
                    result_info,
                    ..
                } if op.label() == "MockOp" => Some((operand_info.clone(), result_info.clone())),
                _ => None,
            })
            .expect("the consumer survives lowering");
        let (operand_info, result_info) = consumer;
        assert_eq!(operand_info.len(), 1);
        assert_eq!(operand_info[0].value, v, "the slot's VALUE is the view's");
        assert_eq!(
            operand_info[0].buffer, plan.value_buffer[&x],
            "the slot's BUFFER is the parent's (the fold's redirect)"
        );
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &operand_info[0].layout, &table[&v],
            "the slot carries the VIEW's own decoded layout, verbatim"
        );
        assert_eq!(
            &result_info[0].layout, &table[&r],
            "a compute result carries its own layout (produced here, never through a fold)"
        );
        let _ = entries;
    }

    /// A two-hop view chain still lowers to ONE producer redirect: the
    /// consumer's slot is anchored on the ROOT parent's buffer, and the
    /// slot's layout is the OUTER view's own decoded layout (the
    /// e-graph minted the composition; the planner records nothing).
    #[test]
    fn two_hop_view_chain_redirects_to_the_root_parent() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockViewWithMap, TestGraph};
        let inner = vec![IotaExpr::Coord(1), IotaExpr::Coord(0)]; // v1 over x
        let outer = vec![IotaExpr::Add(
            Box::new(IotaExpr::Coord(0)),
            Box::new(IotaExpr::Lit(1)),
        )]; // v2 over v1
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v1 = g.op(
            Box::new(MockViewWithMap {
                entries: inner.clone(),
            }),
            &[&x],
            &[("v1", "row0")],
        )[0]
        .clone();
        let v2 = g.op(
            Box::new(MockViewWithMap {
                entries: outer.clone(),
            }),
            &[&v1],
            &[("v2", "row1")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v2],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let graph = g.build();
        let plan = crate::test_support::bufferize_mock(&graph).expect("bufferizes");

        let operand_info = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::Compute {
                    op, operand_info, ..
                } if op.label() == "MockOp" => Some(operand_info.clone()),
                _ => None,
            })
            .expect("the consumer survives lowering");
        assert_eq!(operand_info[0].value, v2);
        assert_eq!(
            operand_info[0].buffer, plan.value_buffer[&x],
            "both folds redirect to x"
        );
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &operand_info[0].layout, &table[&v2],
            "the slot carries the OUTER view's own layout"
        );
        let (_, _, _) = (v1, inner, outer);
    }

    /// A view op WITHOUT a numeric map still folds structurally — the
    /// read path is the view VALUE's own decoded layout, so there is no
    /// per-op map to lose (fail-closure moved to RENDER time: a layout
    /// class the decoder cannot decode refuses the whole plan).
    #[test]
    fn mapless_view_fold_still_redirects_with_the_views_layout() {
        use crate::test_support::{MockView, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let graph = g.build();
        let plan = crate::test_support::bufferize_mock(&graph).expect("bufferizes");

        let operand_info = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::Compute {
                    op, operand_info, ..
                } if op.label() == "MockOp" => Some(operand_info.clone()),
                _ => None,
            })
            .expect("the consumer survives lowering");
        assert_eq!(operand_info[0].value, v);
        assert_eq!(
            operand_info[0].buffer, plan.value_buffer[&x],
            "the fold's redirect"
        );
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &operand_info[0].layout, &table[&v],
            "the view's own layout, verbatim"
        );
    }

    // -------------------------------------------------------------------------
    // ESCAPE-AND-DISCLOSE (ruling 2026-08-27): BufferCopy = dumb whole-buffer
    // memcpy, one meaning; view-elected outputs return their backing buffer
    // plus the elected layout.
    // -------------------------------------------------------------------------

    /// THE VIEW-OUTPUT FIXTURE (corrected contract, 2026-08-31, correction
    /// 5): VIEW OUTPUTS ARE COMPLETELY LEGAL and fulfilled STRUCTURALLY. A
    /// view is no-work-same-buffer, so a view-elected output slot's
    /// ASSIGNED buffer simply IS its parent's backing storage — zero-copy
    /// by construction, never a refusal and never a forced dense delivery.
    /// Here the parent is an INPUT, so the storage is already the caller's:
    /// the slot is backed by the input buffer itself and the declared
    /// output buffer goes unused (dropped by DCE — runtimes never allocate
    /// it). Option B's addition: the binding also CARRIES the view value's
    /// elected layout `L`, verbatim from the decoded table, so an
    /// externally loaded plan can read the delivery geometry without the
    /// table a live runtime already has.
    #[test]
    fn folded_output_on_input_residence_returns_zero_copy() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockViewWithMap, TestGraph};
        let entries = vec![IotaExpr::Coord(0)];
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: entries.clone(),
            }),
            &[&x],
            &[("v", "row0")],
        )[0]
        .clone();
        g.output(&v, "E");
        let graph = g.build();
        let plan = crate::test_support::bufferize_mock(&graph)
            .expect("a view of an input escapes zero-copy");

        assert!(
            !plan
                .dag
                .node_indices()
                .any(|i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. })),
            "zero copies — the storage is already the caller's:\n{}",
            plan.summary()
        );
        let slot = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
                _ => None,
            })
            .expect("one output slot");
        assert_eq!(
            slot.buffer, plan.value_buffer[&x],
            "backed by the INPUT buffer"
        );
        assert!(
            !plan.buffers.contains_key(&BufferId::Boundary(cid("buf$E"))),
            "the declared output buffer is unused and DCE'd:\n{}",
            plan.summary()
        );
        // THE ASSIGNMENT is queryable both ways for the escaping storage.
        assert_eq!(
            plan.backed_tensor(&slot.buffer),
            Some(&x),
            "the input buffer's assignment row still names the tensor it backs"
        );
        // Option B's disclosure: the slot carries the VIEW value's own
        // elected layout, verbatim from the decoded table.
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &slot.layout, &table[&v],
            "the view's own layout rides the binding"
        );
        assert_ne!(
            table[&v], table[&x],
            "the view's layout is a DIFFERENT function from its parent's — \
             the carried L is what makes the zero-copy delivery readable"
        );
        // `entries` shaped the view's layout class in the mock table; the
        // planner never parsed them (it transports L opaquely).
        let _ = &entries;
    }

    /// INTERIOR copy of a folded resident — the default legal lowering is
    /// the BASE-STORAGE copy: the parent buffer is memcpy'd whole (a DUMB
    /// EXACT-SIZE WHOLE-BUFFER copy — the copy contract admits no other
    /// mode) and the fold re-roots onto the copy — the consumer's data edge
    /// hangs off the copy and its operand descriptor keeps the view's own
    /// composed layout, now anchored on the copied buffer. Forced by
    /// hand-rejecting the CONSUMER's dest tie (its operand is the view),
    /// the same discipline as `rejected_view_repairs_by_copying_the_parent`.
    ///
    /// The destination is a FRESHLY MINTED single-writer buffer, never the
    /// tied result's (ruling 2026-08-27, the repair-destination fix): the
    /// base-storage copy is PARENT-shaped while the consumer writes its
    /// result-shaped bytes, so the two must not cohabit. Under the
    /// corrected contract that is an ASSIGNMENT fact — the fresh buffer
    /// `backs` the parent, the result's buffer `backs` the result — not a
    /// geometry join the planner re-derives.
    #[test]
    fn interior_copy_of_folded_resident_copies_the_parent_and_reroots() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockOp, MockViewWithMap, TestGraph};
        let entries = vec![IotaExpr::Coord(0), IotaExpr::Coord(1)];
        let mut g = TestGraph::new();
        let x = g.input("x", "D", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: entries.clone(),
            }),
            &[&x],
            &[("v", "row0")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "E");
        let graph = g.build();
        let order = toposort(&graph.dag, None).unwrap();
        let mut analysis = Analysis {
            storage: {
                let mut storage = UnionFind::default();
                storage.union(&x, &v); // the view tie ADMITTED: v folds onto x
                storage
            },
            in_place: HashMap::from([((0, 0), true), ((1, 0), false)]), // consumer tie REJECTED
            op_count: 2,
        };
        let geometry = extraction_layouts(&graph, &crate::test_support::mock_layout_table(&graph))
            .expect("mock layouts are total over the graph");
        let assignment = Bufferizer::assign(&graph, &order, &mut analysis, &[], &geometry)
            .expect("assignment runs");
        let bt = crate::buffer_tensor_ir::build_buffer_tensor_ir(
            &graph, &order, assignment, &analysis, &geometry,
        )
        .expect("construction never errors on a rejected consumer tie");
        let plan =
            lower(bt, &geometry).expect("an interior folded copy lowers via the base-storage copy");

        // The repair copy moves the fold's BASE STORAGE — the parent's
        // buffer, whole. The node carries only {src, dst}; the copy is
        // identified by its SOURCE being the parent residence, and its
        // destination's ASSIGNMENT row says which tensor landed there.
        let (copy_idx, copy_src, copy_dst) = plan
            .dag
            .node_indices()
            .find_map(|i| match &plan.dag[i] {
                BufferNode::BufferCopy { src, dst } if *src == plan.value_buffer[&x] => {
                    Some((i, src.clone(), dst.clone()))
                }
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "the base-storage copy reads the parent's buffer:\n{}",
                    plan.summary()
                )
            });
        assert_eq!(
            copy_src, plan.value_buffer[&x],
            "copied FROM the parent's buffer"
        );
        assert_eq!(
            plan.backed_tensor(&copy_dst),
            Some(&x),
            "the destination's ASSIGNMENT says the PARENT landed there — \
             parent-shaped storage, sized by the parent's layout:\n{}",
            plan.summary()
        );
        assert!(
            matches!(copy_dst, BufferId::Allocated(_)),
            "interior destination is program storage:\n{}",
            plan.summary()
        );
        assert_ne!(
            copy_dst,
            plan.value_buffer[&r],
            "the repair destination is FRESH, never the tied result's \
             (result-shaped) buffer:\n{}",
            plan.summary()
        );
        let writers_of_dst = plan
            .dag
            .node_indices()
            .filter(|&i| match &plan.dag[i] {
                BufferNode::BufferCopy { dst, .. } => *dst == copy_dst,
                BufferNode::Compute { op, writes, .. } => writes
                    .iter()
                    .enumerate()
                    .any(|(result, id)| *id == copy_dst && op.result_writes_memory(result)),
                _ => false,
            })
            .count();
        assert_eq!(
            writers_of_dst,
            1,
            "the repair copy is the fresh buffer's SOLE writer:\n{}",
            plan.summary()
        );
        assert!(
            !plan.dag.node_indices().any(|i| matches!(
                &plan.dag[i],
                BufferNode::BufferCopy { dst, .. } if plan.backed_tensor(dst) == Some(&v)
            )),
            "no buffer is minted to hold the VIEW value itself — a view is \
             no-work-same-buffer, so only its base storage is ever copied:\n{}",
            plan.summary()
        );

        // THE RE-ROOT: the consumer hangs off the copy and reads the view
        // through its unchanged composed layout, anchored on the copy's dst.
        let consumer = plan
            .dag
            .node_indices()
            .find(|&i| matches!(&plan.dag[i], BufferNode::Compute { op, .. } if op.label() == "MockOp"))
            .expect("the consumer survives lowering");
        assert!(
            plan.dag
                .edges_connecting(copy_idx, consumer)
                .next()
                .is_some(),
            "the consumer's operand edge re-roots onto the copy:\n{}",
            plan.summary()
        );
        let BufferNode::Compute { operand_info, .. } = &plan.dag[consumer] else {
            unreachable!()
        };
        assert_eq!(
            operand_info[0].value, v,
            "the slot's VALUE stays the view's"
        );
        assert_eq!(
            operand_info[0].buffer, copy_dst,
            "…now anchored on the copied buffer"
        );
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &operand_info[0].layout, &table[&v],
            "the consumer reads through the VIEW's own elected layout — \
             unchanged by the re-root, because the layout addresses the \
             residence's bytes and the copy is byte-identical"
        );
        let _ = &entries;
    }

    /// ESCAPE IN PLACE: a view of an INTERIOR (program-minted) resident
    /// bound to an output slot. The residence buffer flips to
    /// `FreedBy::Caller` — the escape cell — optimize mints its alloc and
    /// NO free, the slot is backed by it, and the binding carries the
    /// elected layout. Zero copies; the declared output buffer is unused
    /// and DCE'd.
    #[test]
    fn folded_output_on_minted_residence_escapes_in_place() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockOp, MockViewWithMap, TestGraph};
        let entries = vec![IotaExpr::Coord(0), IotaExpr::Coord(1)];
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let p = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("p", "rm")],
        )[0]
        .clone();
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: entries.clone(),
            }),
            &[&p],
            &[("v", "t")],
        )[0]
        .clone();
        g.output(&v, "E");
        let graph = g.build();
        let plan = crate::test_support::bufferize_mock(&graph)
            .expect("a minted residence escapes in place");

        assert!(
            !plan
                .dag
                .node_indices()
                .any(|i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. })),
            "zero copies — the buffer itself is handed over:\n{}",
            plan.summary()
        );
        let slot = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
                _ => None,
            })
            .expect("one output slot");
        assert_eq!(
            slot.buffer,
            plan.value_buffer[&p],
            "backed by the parent's minted residence:\n{}",
            plan.summary()
        );
        let record = &plan.buffers[&slot.buffer];
        assert_eq!(record.owner, Owner::System, "program-minted storage");
        assert_eq!(
            record.freed_by,
            FreedBy::Caller,
            "…flipped to the ESCAPE cell — the caller manages it"
        );
        assert!(
            !plan.dag.node_indices().any(|i| matches!(
                &plan.dag[i],
                BufferNode::Compute { op, reads, .. }
                    if op.label() == "BufferFree" && reads.contains(&slot.buffer)
            )),
            "optimize mints NO free for an escaping buffer:\n{}",
            plan.summary()
        );
        assert!(
            !plan.buffers.contains_key(&BufferId::Boundary(cid("buf$E"))),
            "the declared output buffer is unused and DCE'd:\n{}",
            plan.summary()
        );
        // Option B's disclosure: the binding carries the VIEW value's own
        // elected layout — the delivery geometry over the escaping buffer.
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(
            &slot.layout, &table[&v],
            "the view's own layout rides the binding"
        );
        assert_eq!(
            plan.backed_tensor(&slot.buffer),
            Some(&p),
            "the escaping storage's ASSIGNMENT names the PRODUCED tensor; the \
             view rides it structurally (zero-copy), and the carried L says how"
        );
        let _ = &entries;
    }

    /// REPAIR ON DONATED RESIDENCE — the one forced repair: donated storage
    /// dies with the call, so the fold's BASE is copied whole (the root
    /// value — copying the base buffer counts as delivery) into a fresh
    /// ESCAPING buffer, the fold re-roots onto the copy, and the donated
    /// buffer backs no slot (it still gets its free).
    #[test]
    fn folded_output_on_donated_residence_repairs_into_fresh_escaping_buffer() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockViewWithMap, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input_binding(
            "x",
            "D",
            Some(Access::ReadWrite),
            Some(FreedBy::Program), // DONATED: the program must destroy it
            "rm",
        );
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: vec![IotaExpr::Coord(0)],
            }),
            &[&x],
            &[("v", "row0")],
        )[0]
        .clone();
        g.output(&v, "E");
        let graph = g.build();
        let plan =
            crate::test_support::bufferize_mock(&graph).expect("a donated residence repairs");

        // ONE dumb whole-base copy (a LIFETIME repair — cause 3) moves the
        // fold's base storage into a fresh escaping buffer. The node
        // carries only {src, dst}; what landed in `dst` is the ASSIGNMENT's
        // business.
        let copies: Vec<(BufferId, BufferId)> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferCopy { src, dst } => Some((src.clone(), dst.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            copies.len(),
            1,
            "one base-storage copy:\n{}",
            plan.summary()
        );
        let (copy_src, copy_dst) = copies[0].clone();
        assert_eq!(
            plan.backed_tensor(&copy_dst),
            Some(&x),
            "the destination's ASSIGNMENT names the fold ROOT — the value \
             whose bytes the whole-buffer memcpy moved"
        );
        assert_eq!(
            copy_src, plan.value_buffer[&x],
            "…from the donated residence"
        );
        assert!(
            matches!(copy_dst, BufferId::Allocated(_)),
            "…into minted storage"
        );
        let record = &plan.buffers[&copy_dst];
        assert_eq!(record.owner, Owner::System);
        assert_eq!(
            record.freed_by,
            FreedBy::Caller,
            "the repair destination ESCAPES"
        );

        let slot = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
                _ => None,
            })
            .expect("one output slot");
        assert_eq!(
            slot.buffer, copy_dst,
            "the slot is backed by the repair copy's buffer"
        );
        assert_ne!(
            slot.buffer,
            plan.value_buffer[&x],
            "donated storage never backs an output slot:\n{}",
            plan.summary()
        );
        assert_eq!(
            &slot.layout,
            &crate::test_support::mock_layout_table(&graph)[&v],
            "the binding discloses the view value's own elected layout"
        );
        // The donated buffer still dies with the call.
        assert!(
            plan.dag.node_indices().any(|i| matches!(
                &plan.dag[i],
                BufferNode::Compute { op, reads, .. }
                    if op.label() == "BufferFree" && reads.contains(&plan.value_buffer[&x])
            )),
            "the donated buffer keeps its free:\n{}",
            plan.summary()
        );
        // The escaping repair buffer does NOT.
        assert!(
            !plan.dag.node_indices().any(|i| matches!(
                &plan.dag[i],
                BufferNode::Compute { op, reads, .. }
                    if op.label() == "BufferFree" && reads.contains(&copy_dst)
            )),
            "no free for the escaping buffer:\n{}",
            plan.summary()
        );
    }

    /// SHARED BASE, minted residence: two view outputs of one parent share
    /// the ONE escaping buffer — both slots point at it, each disclosing
    /// its own layout.
    #[test]
    fn two_view_outputs_of_one_minted_base_share_one_escaping_buffer() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockOp, MockViewWithMap, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let p = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("p", "rm")],
        )[0]
        .clone();
        let row0 = vec![IotaExpr::Coord(0)];
        let transpose = vec![IotaExpr::Coord(0), IotaExpr::Coord(1)];
        let v1 = g.op(
            Box::new(MockViewWithMap {
                entries: row0.clone(),
            }),
            &[&p],
            &[("v1", "row0")],
        )[0]
        .clone();
        let v2 = g.op(
            Box::new(MockViewWithMap {
                entries: transpose.clone(),
            }),
            &[&p],
            &[("v2", "t")],
        )[0]
        .clone();
        g.output(&v1, "D");
        g.output(&v2, "E");
        let graph = g.build();
        let plan =
            crate::test_support::bufferize_mock(&graph).expect("shared-base views escape together");

        let slots: Vec<OutputBinding<crate::test_support::MockLayout>> = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots.clone()),
                _ => None,
            })
            .expect("the output node");
        assert_eq!(slots.len(), 2);
        assert_eq!(
            slots[0].buffer, slots[1].buffer,
            "ONE escaping buffer backs both"
        );
        assert_eq!(slots[0].buffer, plan.value_buffer[&p]);
        assert_eq!(plan.buffers[&slots[0].buffer].freed_by, FreedBy::Caller);
        assert!(
            !plan
                .dag
                .node_indices()
                .any(|i| matches!(&plan.dag[i], BufferNode::BufferCopy { .. })),
            "zero copies:\n{}",
            plan.summary()
        );
        // TWO views, ONE buffer, TWO DIFFERENT carried layouts — this is
        // exactly where the carried `L` earns its keep for an externally
        // loaded plan: the buffer id alone cannot tell the two deliveries
        // apart, and the assignment names only the base tensor.
        let table = crate::test_support::mock_layout_table(&graph);
        assert_eq!(&slots[0].layout, &table[&v1]);
        assert_eq!(&slots[1].layout, &table[&v2]);
        assert_ne!(
            slots[0].layout, slots[1].layout,
            "distinct views, distinct L"
        );
        let _ = (&row0, &transpose);
    }

    /// SHARED BASE, donated residence: the repair is minted ONCE — one
    /// base-storage copy, one escaping buffer — and both view slots ride
    /// it (the certificate resolves each slot's residence through the fold
    /// root).
    #[test]
    fn two_view_outputs_of_one_donated_base_share_one_repair() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockViewWithMap, TestGraph};
        let mut g = TestGraph::new();
        let x = g.input_binding(
            "x",
            "D",
            Some(Access::ReadWrite),
            Some(FreedBy::Program),
            "rm",
        );
        let v1 = g.op(
            Box::new(MockViewWithMap {
                entries: vec![IotaExpr::Coord(0)],
            }),
            &[&x],
            &[("v1", "row0")],
        )[0]
        .clone();
        let v2 = g.op(
            Box::new(MockViewWithMap {
                entries: vec![IotaExpr::Coord(0), IotaExpr::Coord(1)],
            }),
            &[&x],
            &[("v2", "t")],
        )[0]
        .clone();
        g.output(&v1, "E1");
        g.output(&v2, "E2");
        let plan =
            crate::test_support::bufferize_mock(&g.build()).expect("one repair serves both views");

        let copies: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferCopy { dst, .. } => Some(dst.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            copies.len(),
            1,
            "ONE base-storage copy:\n{}",
            plan.summary()
        );
        let slots: Vec<OutputBinding<crate::test_support::MockLayout>> = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots.clone()),
                _ => None,
            })
            .expect("the output node");
        assert_eq!(slots[0].buffer, copies[0], "slot 0 rides the repair");
        assert_eq!(slots[1].buffer, copies[0], "slot 1 rides the same repair");
        assert_eq!(plan.buffers[&copies[0]].freed_by, FreedBy::Caller);
    }

    /// THE ITEM-3 PROBE, RESPELLED AS ASSIGNMENT (corrected contract,
    /// 2026-08-31). Parent x in buffer A; view v = row0 of x; a consumer
    /// whose in-place dst-tie on v is REJECTED. The historical form of this
    /// test carried per-value dims and ran the writer-identity DIMS JOIN,
    /// asserting the repair buffer "votes" `[5,3]` while the result buffer
    /// "votes" `[1,3]`. That whole apparatus is DELETED: there is no sizing
    /// walk, no voting, and no contradiction detection ("all the necessary
    /// preconditions have been checked in the egraph").
    ///
    /// What survives is the fact the join was reconstructing, now stated
    /// directly: THE ASSIGNMENT. The repair destination is a FRESH buffer
    /// that `backs` the PARENT (so a runtime sizes it by the parent's
    /// layout), never the tied result's buffer, which `backs` the result
    /// (sized by the result's layout). One lookup each; nothing to
    /// contradict.
    #[test]
    fn folded_repair_targets_a_fresh_parent_backed_buffer() {
        use crate::index_expr::IotaExpr;
        use crate::test_support::{MockOp, MockViewWithMap, TestGraph};
        let entries = vec![IotaExpr::Lit(0), IotaExpr::Coord(0)];
        let mut g = TestGraph::new();
        let x = g.input("x", "A", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: entries.clone(),
            }),
            &[&x],
            &[("v", "row0")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: Some(0),
                not_conflicting: false,
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "E");
        let graph = g.build();
        let order = toposort(&graph.dag, None).unwrap();
        let mut analysis = Analysis {
            storage: {
                let mut storage = UnionFind::default();
                storage.union(&x, &v); // the view tie ADMITTED: v folds onto x
                storage
            },
            in_place: HashMap::from([((0, 0), true), ((1, 0), false)]), // consumer tie REJECTED
            op_count: 2,
        };
        let mock_geometry =
            extraction_layouts(&graph, &crate::test_support::mock_layout_table(&graph))
                .expect("mock layouts are total over the graph");
        let assignment = Bufferizer::assign(&graph, &order, &mut analysis, &[], &mock_geometry)
            .expect("assignment runs");
        let bt = crate::buffer_tensor_ir::build_buffer_tensor_ir(
            &graph,
            &order,
            assignment,
            &analysis,
            &mock_geometry,
        )
        .expect("construction survives the rejected tie");
        // Lowering takes the decoded layout table directly — there is no
        // second, geometry-bearing table to build, and no annotation pass
        // to run afterwards.
        let plan = lower(bt, &mock_geometry).expect("the base-storage copy lowers");

        let copy_dst = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferCopy { src, dst } if *src == plan.value_buffer[&x] => {
                    Some(dst.clone())
                }
                _ => None,
            })
            .expect("the base-storage repair copy");
        assert_ne!(
            copy_dst, plan.value_buffer[&r],
            "fresh, not the result's buffer"
        );
        assert_eq!(
            plan.backed_tensor(&copy_dst),
            Some(&x),
            "the repair buffer BACKS THE PARENT — a runtime sizes it by the \
             parent's layout, with no walk and no vote:\n{}",
            plan.summary()
        );
        assert_eq!(
            plan.backed_tensor(&plan.value_buffer[&r]),
            Some(&r),
            "the consumer's result buffer backs the RESULT:\n{}",
            plan.summary()
        );
        // Option B: each buffer additionally carries the backed tensor's
        // layout, so a load_plan caller gets the same two answers with no
        // table of its own.
        assert_eq!(&plan.buffers[&copy_dst].layout, &mock_geometry[&x]);
        assert_eq!(
            &plan.buffers[&plan.value_buffer[&r]].layout,
            &mock_geometry[&r]
        );
        let _ = &entries;
    }
}
