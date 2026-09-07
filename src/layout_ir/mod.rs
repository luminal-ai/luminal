//! The extracted Layout IR: a single-valued dataflow DAG of LayoutTensor ops.
//!
//! This is the artifact the extractor hands back after pulling one concrete graph
//! out of the e-graph. It is intentionally *op-centric* (the analogue of an MLIR
//! function body): the graph nodes are dataflow ops plus the program input/output
//! boundaries, and the edges are the LayoutTensor values that flow between them.
//!
//! The bufferization planner will consume this DAG (each [`OpNode`] is the seam
//! where a `Box<dyn LayoutOp>` / `Bufferizable` implementation will eventually
//! live). The visualization ([`ExtractedGraph::to_dot`]) also consumes this DAG,
//! expanding each value into its LayoutTensor / Logical / Layout detail nodes and
//! each boundary into its BufferTensor / BufferId nodes.

// Several fields below are part of the IR's surface for the upcoming
// bufferization layer and are not yet read by the visualization.
#![allow(dead_code)]

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

use egraph_serialize::{ClassId, NodeId};
use petgraph::graph::{DiGraph, NodeIndex};

// =============================================================================
// Deferred display text
// =============================================================================

/// A value that is either already in hand or produced ON FIRST READ and
/// then kept forever.
///
/// WHY (ruling 2026-09-01, "make label, tooltip, and details lazy,
/// leaving only the numeric readers on the hot path"): the extractor
/// builds one info struct per value and per op output slot, on every
/// genome of a search. Their DISPLAY text — tooltips, the logical and
/// layout labels — costs full-depth e-graph rendering, and only the
/// visualizer and the diagnostics ever read it. Deferring it takes that
/// work off the search's hot path without changing a character of the
/// result: `get()` runs the very same builder the eager code ran, so the
/// string is identical whenever anybody asks for it.
///
/// NOT a cache and not a skip — a `Deferred` value is computed exactly
/// once and never recomputed or approximated.
///
/// `Rc`, not `Arc`: an [`ExtractedGraph`] is single-threaded everywhere
/// it travels (the Python holder is `#[pyclass(unsendable)]`), and the
/// deferred builder holds extractor state that is itself `Rc`-shared.
/// Sharing the `Rc` across clones is deliberate: an info cloned into
/// several DAG slots (the DPS rewrite, multi-output ops) renders at most
/// once for all of them.
pub struct Lazy<T>(Rc<LazyInner<T>>);

enum LazyInner<T> {
    Eager(T),
    Deferred {
        cell: OnceCell<T>,
        build: Box<dyn Fn() -> T>,
    },
}

impl<T> Lazy<T> {
    /// A value already computed (synthesized nodes, test fixtures).
    pub fn eager(value: T) -> Self {
        Lazy(Rc::new(LazyInner::Eager(value)))
    }

    /// A value built on first read. `build` must be a pure function of
    /// what it captures: it may run at any later time, and its result is
    /// what every reader sees.
    pub fn deferred(build: impl Fn() -> T + 'static) -> Self {
        Lazy(Rc::new(LazyInner::Deferred {
            cell: OnceCell::new(),
            build: Box::new(build),
        }))
    }

    /// The value, building it if this is the first read.
    pub fn get(&self) -> &T {
        match &*self.0 {
            LazyInner::Eager(value) => value,
            LazyInner::Deferred { cell, build } => cell.get_or_init(build),
        }
    }

    /// Whether the value is already in hand — for tests and diagnostics
    /// that want to observe the deferral without triggering it.
    pub fn is_rendered(&self) -> bool {
        match &*self.0 {
            LazyInner::Eager(_) => true,
            LazyInner::Deferred { cell, .. } => cell.get().is_some(),
        }
    }
}

/// Sharing, not deep-copying: clones observe one another's rendering.
impl<T> Clone for Lazy<T> {
    fn clone(&self) -> Self {
        Lazy(Rc::clone(&self.0))
    }
}

impl<T> Deref for Lazy<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T: fmt::Display> fmt::Display for Lazy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.get(), f)
    }
}

/// DOES NOT force the value: `{:?}` on a half-built graph is a debugging
/// aid, and making it render every deferred tooltip would defeat the
/// deferral exactly where it matters most.
impl<T: fmt::Debug> fmt::Debug for Lazy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            LazyInner::Eager(value) => fmt::Debug::fmt(value, f),
            LazyInner::Deferred { cell, .. } => match cell.get() {
                Some(value) => fmt::Debug::fmt(value, f),
                None => f.write_str("<unrendered>"),
            },
        }
    }
}

impl<T: Default> Default for Lazy<T> {
    fn default() -> Self {
        Lazy::eager(T::default())
    }
}

impl<T> From<T> for Lazy<T> {
    fn from(value: T) -> Self {
        Lazy::eager(value)
    }
}

/// The deferred-text alias every info struct uses.
pub type LazyText = Lazy<String>;

// =============================================================================
// The op interface
// =============================================================================
//
// A LayoutIR op is an open set: any type that implements [`LayoutIrOp`] can be a
// graph node. Every op is required to be [`Bufferizable`] (the type system
// enforces it — you cannot build a `dyn LayoutIrOp` that isn't), so the
// bufferization analysis can always query any op uniformly.

/// One aliasing declaration: how `operand` relates to `result`'s storage.
/// The op's whole aliasing contract — requirements AND permissions — lives
/// in one list ([`Bufferizable::alias_info`]).
///
/// `Sharing::Must` — a REQUIREMENT (a storage tie): the kernel's signature
/// demands `operand` and `result` be backed by one storage. Always honored
/// in every emitted plan; the analyzer's admit/reject decision selects only
/// WHERE that one storage lives — the operand's current home (admitted, zero
/// copy) or a fresh relocation (rejected — the generic repair: the result
/// gets its own buffer, initialized by the kernel's write if the op writes
/// it, preceded by a copy of the operand's bytes iff the result's pre-write
/// contents must equal them: the op reads the operand, or writes nothing at
/// all, as a view). Every tie is repairable, so rejection is never an error.
/// Must-edges form a PARTIAL MATCHING in ascending operand order (guarded
/// loudly at analysis entry — one allocation per storage class cannot
/// represent either-or aliasing or multi-destination results). What a tie
/// MEANS further derives from write effects: a written result is a dest tie
/// (descriptor-exact by the in-place convention; seeding crosses it), an
/// unwritten result is a view (derived storage under its own layout; seeding
/// never crosses it).
///
/// `Sharing::May` — a PERMISSION: reading through `operand` is safe even
/// while this op writes `result`'s storage in place (the two may alias).
/// Unconditional and TRUSTED — the engine checks no layouts; an op whose
/// safety depends on preconditions (equal layouts, injectivity) discharges
/// them where it is MATCHED (egglog) and only then declares the permit.
/// The ABSENCE of a permit is itself a declaration: `restrict` semantics —
/// the engine keeps the pair's storage disjoint by rejection + repair.
///
/// The edge universe is deliberately operand→result, and it is COMPLETE:
///  * operand↔operand permissions route through the written result — in
///    value SSA every write has a result, so "operand a may alias operand
///    b's storage" is `May(a → the result b is Must-tied to)`; read-read
///    aliasing needs no permission at all.
///  * peer-to-peer storage IDENTITY is expressed as dataflow, not edges:
///    two inputs that must arrive co-located are a Pack op's single packed
///    value; two results that share one underlying buffer are ONE
///    storage-producing result plus view ops over it (the resource is
///    reified as a value — the plan's native value/buffer split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AliasInfo {
    /// Index into the op's input list.
    pub operand: usize,
    /// Index into the op's output list.
    pub result: usize,
    pub sharing: Sharing,
}

/// Requirement vs permission — see [`AliasInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sharing {
    /// The slots ARE one storage (kernel requirement).
    Must,
    /// The slots MAY be one storage (kernel tolerance, trusted).
    May,
}

/// The ANALYSIS-time declaration surface — the Rust analogue of MLIR's
/// `BufferizableOpInterface`, minus the memory effects: which operands read
/// and which results write are identity-level facts that outlive the
/// analysis, so they live on [`crate::buffer_tensor_ir::BufferTensorIrOp`]
/// (a supertrait of [`LayoutIrOp`]). What remains here is exactly what only
/// the analyzer consumes: the aliasing contract, and the not-yet-supported
/// op-owned-storage flag. Defaults describe a pure, out-of-place dataflow op,
/// so a plain op implements this with an empty body.
pub trait Bufferizable {
    /// The op's complete aliasing contract — must-share requirements and
    /// may-share permissions in one place (see [`AliasInfo`]). Empty = pure
    /// dataflow op: every operand read-only, every result freshly allocated
    /// by the planner, no aliasing tolerated. This is the op's ONLY aliasing
    /// declaration, and deliberately the trait's only aliasing method: the
    /// engine derives its own views of the contract (the Must ties, the May
    /// permits) at the call sites that consume them, so there is no derived
    /// method an op could override into disagreement with its declaration.
    fn alias_info(&self) -> Vec<AliasInfo> {
        Vec::new()
    }

    /// Does the op allocate this result's storage ITSELF, outside the
    /// planner's control (a wrapped kernel that returns memory it allocated
    /// internally)? The planner must not assign, seed, or reuse such storage,
    /// and ownership/free obligations follow the op's contract. An op
    /// property — NOT a prediction of the planner's decision: a
    /// planner-allocated fresh result (the ordinary out-of-place outcome) is
    /// `false` here. NOT YET SUPPORTED: declaring it is a loud bufferization
    /// error until Alloc/Free ownership machinery lands.
    fn result_is_allocated_internally(&self, _result: usize) -> bool {
        false
    }
}

/// The engine's reading of the Must edges: the declared ties as
/// (operand, result) pairs, in declaration order. One of the two derived
/// views of [`Bufferizable::alias_info`] (with [`permits_sharing`]), provided
/// once beside the declaration so every consumer — analyzer, seed walker,
/// planner, renderers — reads the contract the same way. Deliberately free
/// functions, not trait methods: no op can override a derived view into
/// disagreement with its declaration.
pub(crate) fn must_ties<O: Bufferizable + ?Sized>(op: &O) -> Vec<(usize, usize)> {
    op.alias_info()
        .iter()
        .filter(|edge| edge.sharing == Sharing::Must)
        .map(|edge| (edge.operand, edge.result))
        .collect()
}

/// The engine's reading of an op's `May` permits: does the op permit reading
/// through `read_operand` while it writes in place through `write_operand`?
/// True iff a `May` edge grants `read_operand` sharing with the result that
/// `write_operand` is must-tied to (the permit routes through the written
/// result — the edge universe is operand→result). The other derived view of
/// [`Bufferizable::alias_info`] (see [`must_ties`] for why it lives here).
/// Public so runtime registries can pin their declared contracts against
/// the same reading (Step B).
pub fn permits_sharing<O: Bufferizable + ?Sized>(
    op: &O,
    read_operand: usize,
    write_operand: usize,
) -> bool {
    let info = op.alias_info();
    info.iter().any(|must| {
        must.sharing == Sharing::Must
            && must.operand == write_operand
            && info.iter().any(|may| {
                may.sharing == Sharing::May
                    && may.operand == read_operand
                    && may.result == must.result
            })
    })
}

/// How (and whether) a result-returning op rewrites into destination-passing
/// style. Every LayoutIR op must implement this — the compiler forces each op
/// author to answer explicitly; there is no silent default.
///
/// The rewrite only ever APPENDS write-only, poison-valued destination operands
/// — exactly one per result, as a consecutive trailing range (MLIR's tie
/// invariant), enforced by construction in the pass rather than checked.
/// There is deliberately no "init" concept: an op that READS a destination-like
/// operand (an accumulator) carries that operand as an ordinary input with an
/// ordinary read + aliasing declaration, not as something this trait appends.
///
/// The returned DPS form is the sole owner of its operand layout: its
/// memory-effect declarations (on `BufferTensorIrOp`) and its aliasing
/// contract (on [`Bufferizable`]) spell out, slot by slot, where data ends
/// and destinations begin. Pairing a base op with a DPS form of matching arity is
/// part of the op pair's DEFINITION — deliberately not re-checked at runtime
/// (no redundant bookkeeping to keep in sync).
pub trait ToDps {
    /// The DPS form of this op. `None` = the op has no DPS form (its results
    /// always bufferize to fresh allocations, like MLIR's non-DPS tensor ops) —
    /// or the op already IS a DPS form (which makes the rewrite pass idempotent).
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>>;
}

/// Clone plumbing for `Box<dyn LayoutIrOp>` — the same blanket pattern as
/// [`crate::buffer_tensor_ir::CloneBufferTensorIrOp`], written once for every
/// `Clone` op so no op author implements it by hand. (A separate helper is
/// needed per object type: this one clones to `Box<dyn LayoutIrOp>`.)
pub trait CloneLayoutIrOp {
    fn clone_box(&self) -> Box<dyn LayoutIrOp>;
}

impl<T: LayoutIrOp + Clone + 'static> CloneLayoutIrOp for T {
    fn clone_box(&self) -> Box<dyn LayoutIrOp> {
        Box::new(self.clone())
    }
}

/// The trait every LayoutIR dataflow op implements — a pure marker over its
/// supertrait stack, each layer one surface: identity + memory effects
/// ([`crate::buffer_tensor_ir::BufferTensorIrOp`], which outlives analysis
/// and is all the BufferTensor layer sees), the aliasing contract
/// ([`Bufferizable`], analysis-time only), and the destination-passing story
/// ([`ToDps`]). An op can never appear in the graph without answering all
/// three; the impl block itself is empty.
pub trait LayoutIrOp:
    crate::buffer_tensor_ir::BufferTensorIrOp + Bufferizable + ToDps + CloneLayoutIrOp
{
}

impl Clone for Box<dyn LayoutIrOp> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// =============================================================================
// The matcher interface — the extraction side of an op's definition
// =============================================================================
//
// Every op's definition has two halves with distinct lifetimes. The MATCHER
// half is registered up front: a zero-sized struct, one per egglog
// implementation constructor, that knows the constructor's name, which enode
// children are metadata, what the op contributes to the egglog program, and
// how to build a concrete instance from a matched enode. The INSTANCE half is
// what `extract` produces: the `Box<dyn LayoutIrOp>` that flows through
// bufferization, DPS rewriting, and rendering. Instances are created only by
// their matcher's `extract` (or derived from another instance via `to_dps`) —
// a hygiene convention, deliberately not enforced by privacy.
//
// The point of the split is that adding an op touches exactly one file: the
// op's module defines its instances, their trait impls, AND its matcher(s),
// and the only edit elsewhere is one line in the registration list
// ([`ops::built_in_matchers`]). There is no dispatch table anywhere — the
// extractor resolves candidates by looking the enode's constructor name up in
// a registry built from that list.

/// The matched enode, handed to [`OpMatcher::extract`]. A borrowed view of
/// everything a matcher may read while building an instance. Today's
/// instances are nullary and ignore it; ops whose instances carry extracted
/// data (layouts, axes, index maps) read their children through it. Extending
/// this struct is the sanctioned way to grow the extraction contract —
/// matchers take `&ExtractionSite`, so new fields cost existing impls nothing.
#[derive(Debug, Clone, Copy)]
pub struct ExtractionSite<'a> {
    pub egraph: &'a egraph_serialize::EGraph,
    pub node_id: &'a NodeId,
    pub node: &'a egraph_serialize::Node,
}

impl ExtractionSite<'_> {
    /// The matched enode's child at `index` as a literal f64 (schema-drift
    /// panic contract as [`Self::child_i64`]).
    pub fn child_f64(&self, index: usize) -> f64 {
        let class = self.child_class(index);
        for node in self.egraph.nodes.values() {
            if node.eclass == class {
                if let Ok(value) = node.op.parse::<f64>() {
                    return value;
                }
            }
        }
        panic!(
            "schema drift: {} child {index} is not a literal f64",
            self.node.op
        );
    }

    /// Find a node with the given constructor inside an arbitrary class —
    /// the metadata-walking helper for data-carrying instances.
    pub fn node_in_class(
        &self,
        class: &egraph_serialize::ClassId,
        op: &str,
    ) -> Option<&egraph_serialize::Node> {
        self.egraph
            .nodes
            .values()
            .find(|node| node.eclass == *class && node.op == op && !node.subsumed)
    }

    /// Every node of this op in the class for VALUE PARSING, unsubsumed
    /// spellings first, subsumed ones as fallback. Subsumption is a
    /// MATCHING fence (it steers rules and cost away from non-canonical
    /// spellings); denotationally a subsumed node is still a true member
    /// of its class, and a kernel evaluating any spelling computes the
    /// same function. Saturation can subsume EVERY constructor spelling
    /// of a class (the 2026-08-05 slice_pad regression), so value
    /// readers must not starve behind the fence. Op MATCHERS keep the
    /// strict accessor.
    pub fn nodes_in_class_value<'a>(
        &'a self,
        class: &'a egraph_serialize::ClassId,
        op: &'a str,
    ) -> impl Iterator<Item = &'a egraph_serialize::Node> + 'a {
        self.nodes_in_class(class, op).chain(
            self.egraph
                .nodes
                .values()
                .filter(move |node| node.eclass == *class && node.op == op && node.subsumed),
        )
    }

    /// EVERY non-subsumed node of this op in the class — for parsers that
    /// must backtrack across a saturated class's many representations.
    pub fn nodes_in_class<'a>(
        &'a self,
        class: &'a egraph_serialize::ClassId,
        op: &'a str,
    ) -> impl Iterator<Item = &'a egraph_serialize::Node> + 'a {
        self.egraph
            .nodes
            .values()
            .filter(move |node| node.eclass == *class && node.op == op && !node.subsumed)
    }

    /// Any literal-i64 node inside an arbitrary class.
    pub fn node_in_class_parse_i64(&self, class: &egraph_serialize::ClassId) -> Option<i64> {
        self.egraph
            .nodes
            .values()
            .filter(|node| node.eclass == *class)
            .find_map(|node| node.op.parse::<i64>().ok())
    }

    /// The e-class of an arbitrary node's child.
    pub fn class_of_child(
        &self,
        node: &egraph_serialize::Node,
        index: usize,
    ) -> Option<egraph_serialize::ClassId> {
        let child = node.children.get(index)?;
        Some(self.egraph.nodes.get(child)?.eclass.clone())
    }

    /// The matched enode's child at `index` as a literal i64. Panics when it
    /// is not one — for a constructor whose schema declares a primitive i64
    /// child, anything else is schema drift (see the validity contract).
    pub fn child_i64(&self, index: usize) -> i64 {
        let class = self.child_class(index);
        for node in self.egraph.nodes.values() {
            if node.eclass == class {
                if let Ok(value) = node.op.parse::<i64>() {
                    return value;
                }
            }
        }
        panic!(
            "schema drift: {} child {index} is not a literal i64",
            self.node.op
        );
    }

    /// The e-class of the matched enode's child at `index`. Panics on a
    /// missing child — a matched enode's arity is fixed by its constructor,
    /// so a short child list is schema drift (see the validity contract).
    pub fn child_class(&self, index: usize) -> egraph_serialize::ClassId {
        let child_id = self.node.children.get(index).unwrap_or_else(|| {
            panic!(
                "schema drift: {} enode {} has no child {index}",
                self.node.op, self.node_id
            )
        });
        self.egraph
            .nodes
            .get(child_id)
            .unwrap_or_else(|| panic!("dangling child node {child_id} on enode {}", self.node_id))
            .eclass
            .clone()
    }
}

/// One registered implementation pattern: the bridge from an egglog
/// LayoutTensorOp constructor to concrete [`LayoutIrOp`] instances.
///
/// VALIDITY CONTRACT: an enode carrying a registered constructor is
/// extractable *by construction* — the match rules discharge every
/// applicability condition in egglog before the constructor is ever minted,
/// so presence in the e-graph IS the certificate. A matcher therefore never
/// declines and never re-litigates applicability in Rust: `extract` returns
/// the instance, unconditionally. The only way extraction of a matched enode
/// can fail is schema drift — the Rust-side slot spec disagreeing with the
/// preamble's constructor arity — and that is a bug, surfaced as a loud
/// panic at the metadata pull, never a silent skip.
///
/// Instances are created only by their matcher's `extract`, derived from
/// another instance via `to_dps`, or synthesized by the DPS pass (Poison).
/// That is a hygiene convention, deliberately not enforced by privacy:
/// instance fields stay public, instance structs simply do not derive
/// `Default` and nothing else constructs them.
pub trait OpMatcher: std::fmt::Debug {
    /// The full egglog constructor name this matcher recognizes, e.g.
    /// `"LayoutTensorOpAddFunctionalGeneric"`. This is the registry key: the
    /// extractor offers this matcher exactly the enodes carrying this label.
    /// (Instance labels keep the house policy — constructor name minus the
    /// `LayoutTensorOp` prefix; the full name lives only here.)
    fn egglog_constructor(&self) -> &'static str;

    /// Which enode children are metadata, as (port name, child index) pairs.
    /// The extractor resolves each index to its child's e-class and carries
    /// the named class through the plan (rendering, output layouts). An enode
    /// whose listed child is missing produces no candidate (fail-closed).
    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[]
    }

    /// Everything this implementation contributes to the assembled egglog
    /// program: its `(constructor ...)` declaration and its match rule(s),
    /// one `.egg` file per contribution, spliced by
    /// [`crate::egglog_snippet::assembled_program`]. Every registered
    /// matcher overrides this; the default exists only so hypothetical
    /// rule-less matchers stay expressible.
    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        Vec::new()
    }

    /// Decoders for every constructor this matcher's snippets DECLARE
    /// for a decoded sort. Mirrors [`OpMatcher::snippets`]: a
    /// constructor's declaration and the Rust struct that reads it back
    /// travel together, so the assembly tripwire can prove the pair is
    /// complete. Default empty — no matcher declares a `Layout`
    /// constructor today.
    fn decoders(&self) -> Vec<crate::egglog_utils::eclass::ConstructorDecoder> {
        Vec::new()
    }

    /// Build the concrete instance for a matched enode. Infallible by the
    /// validity contract above.
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp>;
}

// The op INVENTORY does not live here (ruling 2026-08-06): layout_ir
// defines the IR framework — the traits, extraction machinery, and plan
// types — and stays distant from where ops are implemented. The
// reference runtime's inventory is `luminal_reference::ops`; every
// runtime crate brings its own. (Step B closed the crate-split coupling:
// the extractor and egglog assembly take the matcher set as a
// parameter.)

/// The petgraph carrying the dataflow DAG.
pub type ExtractedDag = DiGraph<ExtractedNode, ExtractedEdge>;

/// The extracted Layout IR graph plus its output roots.
#[derive(Debug, Clone)]
pub struct ExtractedGraph {
    pub dag: ExtractedDag,
    /// The `Output` boundary node(s), i.e. the sinks of the dataflow.
    pub outputs: Vec<NodeIndex>,
}

/// A node in the dataflow DAG: a dataflow op or a program boundary.
#[derive(Debug, Clone)]
pub enum ExtractedNode {
    /// A program input boundary: a LayoutTensor backed by a pre-assigned buffer.
    /// Not a `LayoutOp` — it is a (caller-owned) storage specification.
    BufferInput(Box<InputNode>),
    /// A dataflow op producing one or more LayoutTensor outputs.
    LayoutOp(OpNode),
    /// A program output boundary: final LayoutTensors that must be written into
    /// pre-assigned destination buffers. Also a storage specification, not an op.
    BufferOutput(OutputNode),
}

/// An edge carries the LayoutTensor value flowing from a producer to a consumer.
#[derive(Debug, Clone)]
pub struct ExtractedEdge {
    /// E-class identity of the LayoutTensor value flowing along this edge.
    pub value: ClassId,
    /// Display label for the consumer port (e.g. `lhs`, `input`, `out 0`).
    pub port: String,
}

/// A LayoutTensor value: a logical tensor coupled with a chosen layout. Not yet
/// backed by a buffer — that is exactly what bufferization assigns.
#[derive(Debug, Clone)]
pub struct LayoutTensorInfo {
    pub eclass: ClassId,
    /// EAGER: semantic, not decoration — bufferization names allocations
    /// after it and the plan summary goldens pin those names.
    pub label: String,
    /// Deferred: visualizer/diagnostic text only (see [`Lazy`]).
    pub tooltip: LazyText,
    pub shape: Option<String>,
    pub dtype: Option<String>,
    /// The machine-readable plan dtype, from the logical side's
    /// `dtype-of` row (typed-buffers landing A) — unlike `dtype` above,
    /// which is display text and unreliable for interior values. Same
    /// consumer contract as `dims`: numeric consumers bail loudly on
    /// `None`.
    pub dtype_enum: Option<crate::dtype::PlanDtype>,
    /// The layout's extents as literals, walked off the e-graph terms —
    /// `None` when symbolic or underivable. The executor/translator surface;
    /// numeric consumers bail loudly on `None` rather than parse strings.
    pub dims: Option<Vec<i64>>,
    /// The layout's element bit width as a literal (same contract as `dims`).
    pub element_bits: Option<i64>,
    pub logical: LogicalInfo,
    pub layout: LayoutInfo,
}

/// A logical (value-semantic) tensor, with its dataflow predecessors.
#[derive(Debug, Clone)]
pub struct LogicalInfo {
    pub eclass: ClassId,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub label: LazyText,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub tooltip: LazyText,
    /// The logical CONSTRUCTOR producing this tensor (e.g. `LogicalSqrt`),
    /// when it has operands — rendered as the logical op node between this
    /// tensor and its children. `None` for leaves (literals).
    pub op: Option<String>,
    pub children: Vec<(String, LogicalInfo)>,
}

/// A physical layout (strides / offset expression), independent of any buffer.
#[derive(Debug, Clone)]
pub struct LayoutInfo {
    pub eclass: ClassId,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub label: LazyText,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub tooltip: LazyText,
}

/// CONTENTS permission: may the program overwrite this buffer's bytes?
/// `ReadOnly` = the program only loads (weights / constants / masks) — holding
/// it is also the framing guarantee that nobody mutates the storage for the
/// plan's duration. `ReadWrite` = exclusive read/write (transient scribbling
/// included). Preserving a value across the boundary is an *obligation* (an
/// exit binding), never an access property.
///
/// THE TWO BOUNDARY CONTRACTS (M4 Phase 2, ruled by Austin 2026-08-27):
///
/// CONTRACT 1 — BUFFERLIT DISJOINTNESS. Distinct `BufferLit` e-classes
/// warrant DISJOINT storage: each `BufferLit` names a unique,
/// non-overlapping piece of memory. A caller binding two tensors that
/// share underlying storage must bind them to the SAME `BufferLit`,
/// distinguished by their layouts. Binding overlapping pointer ranges to
/// distinct `BufferLit`s violates the plan's aliasing model — every
/// certificate, anti-edge, and executor storage decision keys on
/// `BufferId` equality — and real backends must assert pairwise
/// non-overlap of bound pointer ranges at bind time and refuse loudly.
///
/// CONTRACT 2 — READWRITE EXCLUSIVITY. A buffer bound `ReadWrite` is the
/// program's EXCLUSIVELY for the plan's duration: the caller must neither
/// read nor write it mid-execution, and the program may consume it — a
/// `ReadWrite` input whose value has no surviving later read is a legal
/// in-place destination, including when reached through view chains, and
/// its prior contents are then destroyed. Callers who want their bytes
/// back unchanged bind `ReadOnly`; callers who want results delivered
/// into their storage declare an output binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    ReadWrite,
}

/// STORAGE deallocation responsibility — deliberately orthogonal to
/// [`Access`]: permission to clobber bytes never implies responsibility to
/// destroy storage (a read-write workspace stays the caller's allocation; a
/// read-only tensor can still be donated for destruction). `Caller`: the
/// storage outlives the call and the program must never free it. `Program`
/// (donated): the program must free it exactly once, after its last use.
/// There is NO undeclared default: input-program validation requires every
/// boundary buffer to declare its responsibility explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreedBy {
    Caller,
    Program,
}

/// A runtime buffer binding: a BufferTensor and its BufferId.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub tensor_eclass: ClassId,
    pub tensor_label: String,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub tensor_tooltip: LazyText,
    pub id_eclass: ClassId,
    pub id_label: String,
    /// Deferred: visualizer text only (see [`Lazy`]).
    pub id_tooltip: LazyText,
    /// The declared contents permission — `None` when the program omits the
    /// declaration, which input-program validation rejects for EVERY buffer.
    pub access: Option<Access>,
    /// The declared deallocation responsibility — `None` when the program
    /// omits it, which input-program validation rejects for EVERY buffer.
    pub freed_by: Option<FreedBy>,
    /// The numeric `BufferLit` value when the id is a literal — the key
    /// runtimes bind caller data by (`None` for non-literal ids).
    pub lit: Option<i64>,
}

impl BufferInfo {
    /// The declared contents permission. Input-program validation guarantees
    /// the declaration exists; a `None` reaching this accessor is a harness
    /// bug, not a program error.
    pub fn access(&self) -> Access {
        self.access
            .expect("input-program validation requires buffer-access-of for every buffer")
    }

    /// The declared deallocation responsibility. Input-program validation
    /// guarantees the declaration exists; a `None` reaching this accessor is
    /// a harness bug, not a program error.
    pub fn freed_by(&self) -> FreedBy {
        self.freed_by
            .expect("input-program validation requires buffer-freed-by for every buffer")
    }
}

/// A program input: a value with a pre-assigned, caller-owned buffer.
#[derive(Debug, Clone)]
pub struct InputNode {
    pub value: LayoutTensorInfo,
    pub buffer: BufferInfo,
}

/// One operand of an op: a port name plus the value e-class it consumes.
#[derive(Debug, Clone)]
pub struct OpInput {
    pub port: String,
    pub value: ClassId,
}

/// Where a graph node came from: pulled out of the e-graph by extraction, or
/// synthesized by a later pass (DPS poison destinations). Keeps synthetic nodes
/// out of every map keyed on e-graph identities.
#[derive(Debug, Clone)]
pub enum Provenance {
    Extracted {
        op_eclass: ClassId,
        source_enode: NodeId,
        /// Index (within `outputs`) of the value whose extraction selected this op.
        selected_output_index: usize,
    },
    Synthesized {
        id: u32,
    },
}

/// A dataflow op node.
#[derive(Debug, Clone)]
pub struct OpNode {
    pub op: Box<dyn LayoutIrOp>,
    pub provenance: Provenance,
    pub inputs: Vec<OpInput>,
    pub outputs: Vec<LayoutTensorInfo>,
    /// Deferred: visualizer text only (see [`Lazy`]). The heaviest of
    /// them — it renders the op's whole operand/result table.
    pub tooltip: LazyText,
    pub heuristic_cost: u64,
}

/// One slot of the program output: a final value written into a destination buffer.
#[derive(Debug, Clone)]
pub struct OutputSlot {
    pub index: usize,
    pub value: ClassId,
    pub buffer: BufferInfo,
}

/// A program output boundary.
#[derive(Debug, Clone)]
pub struct OutputNode {
    pub eclass: ClassId,
    pub label: String,
    pub tooltip: String,
    pub slots: Vec<OutputSlot>,
}

impl ExtractedGraph {
    /// Render the dataflow DAG to Graphviz dot, expanding values and boundaries
    /// into their detail nodes.
    ///
    /// Arrow convention (same law as the bufferized view): every arrow points
    /// in the direction information flows toward the program output. Operands
    /// flow into ops and ops into their results (solid — bytes); constructor
    /// ingredients (logical, layout, buffer id) flow into the term they form,
    /// and the boundary lines state slot membership — the BufferTensorLit's
    /// value/buffer wiring and its "out i" line into the Output sink are all
    /// dotted information, not events. Consequence: a dotted arrow INTO a
    /// BufferTensor means the buffer is written; a dotted arrow OUT of one
    /// means it is read.
    pub fn to_dot(&self) -> String {
        let mut emitter = DotEmitter::new();

        // Index every value by e-class so op operand edges can resolve to the
        // LayoutTensor node emitted for the producing op / input.
        let mut values: HashMap<&ClassId, &LayoutTensorInfo> = HashMap::new();
        for node in self.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    values.insert(&input.value.eclass, &input.value);
                }
                ExtractedNode::LayoutOp(op) => {
                    for output in &op.outputs {
                        values.insert(&output.eclass, output);
                    }
                }
                ExtractedNode::BufferOutput(_) => {}
            }
        }

        // Pass A: emit every value's LayoutTensor (+ logical + layout) node, in a
        // stable order driven by node index.
        for node in self.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    emitter.value_node(&input.value);
                }
                ExtractedNode::LayoutOp(op) => {
                    for output in &op.outputs {
                        emitter.value_node(output);
                    }
                }
                ExtractedNode::BufferOutput(_) => {}
            }
        }

        // Pass B: emit boundary and op nodes plus their dataflow / buffer edges.
        for node in self.dag.node_weights() {
            match node {
                ExtractedNode::BufferInput(input) => {
                    let value_id = emitter.value_node(&input.value);
                    let (buffer_tensor_id, _) = emitter.buffer(&input.buffer);
                    // Constitution: the BufferTensorLit term is MADE OF its
                    // LayoutTensor (and its BufferId, wired in buffer()).
                    emitter.info_edge(value_id, buffer_tensor_id);
                }
                ExtractedNode::LayoutOp(op) => {
                    // No per-op special-casing (copies included): every op renders
                    // the same slot table — one input slot per operand in signature
                    // order, one output slot per result, tied pairs as one spanning
                    // row (declared contract only, never a match on the op) — and
                    // each edge docks at its slot, so operand order is visible in
                    // the node itself.
                    let in_slots: Vec<String> =
                        op.inputs.iter().map(|input| input.port.clone()).collect();
                    let out_slots: Vec<String> = (0..op.outputs.len())
                        .map(|index| op.op.result_name(index))
                        .collect();
                    let ties = must_ties(op.op.as_ref());
                    let op_id = emitter.slot_node(
                        op.op.label(),
                        &in_slots,
                        &out_slots,
                        &ties,
                        VisualKind::LayoutIr,
                        &op.tooltip,
                    );
                    let out_ports = tied_out_ports(&in_slots, &out_slots, &ties);
                    for (index, output) in op.outputs.iter().enumerate() {
                        let value_id = emitter.value_node(output);
                        emitter.edge_from_slot(op_id, &out_ports[index], value_id);
                    }
                    for input in &op.inputs {
                        if let Some(value) = values.get(&input.value) {
                            let value_id = emitter.value_node(value);
                            emitter.edge_to_slot(value_id, op_id, &input.port);
                        }
                    }
                }
                ExtractedNode::BufferOutput(output) => {
                    let output_id = emitter.raw_node(
                        format!("Output\n{}", output.label),
                        VisualKind::Output,
                        &output.tooltip,
                    );
                    for slot in &output.slots {
                        let (buffer_tensor_id, _) = emitter.buffer(&slot.buffer);
                        if let Some(value) = values.get(&slot.value) {
                            let value_id = emitter.value_node(value);
                            // Constitution: the destination BufferTensorLit
                            // is made of the final value + its BufferId.
                            emitter.info_edge(value_id, buffer_tensor_id);
                        }
                        // Boundary membership is information too (user-ratified):
                        // the sink line states WHICH slot this pair fills, not a
                        // byte movement — the write is already the value edge.
                        emitter.info_edge_labeled(
                            buffer_tensor_id,
                            output_id,
                            &format!("out {}", slot.index),
                        );
                    }
                }
            }
        }

        emitter.finish()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum VisualKind {
    Output,
    BufferTensor,
    BufferId,
    LayoutTensor,
    LogicalTensor,
    LogicalOp,
    Layout,
    LayoutIr,
}

impl VisualKind {
    /// (shape, style, fillcolor, color)
    ///
    /// One hue per semantic domain, one grammar across every dot view:
    /// TENSORS are rounded boxes, OPS are squares, METADATA (descriptions of
    /// data — a layout, a buffer identity — that no kernel ever flows through)
    /// is a note (the folded-corner page), and the hue names the domain — so
    /// kind never needs to be spelled out in a label:
    ///   rose  = buffer domain  (BufferTensor, BufferId, buffer-tensor ops)
    ///   amber = layout domain  (Layout, LayoutTensor, layout tensor ops)
    ///   violet = logical domain (LogicalTensor, logical ops)
    ///   blue  = program boundary (Input / Output)
    fn style(self) -> (&'static str, &'static str, &'static str, &'static str) {
        match self {
            VisualKind::Output => ("box", "rounded,filled", "#dbeafe", "#2563eb"),
            VisualKind::BufferTensor => ("box", "rounded,filled", "#fce7f3", "#be185d"),
            VisualKind::BufferId => ("note", "filled", "#fce7f3", "#be185d"),
            VisualKind::LayoutTensor => ("box", "rounded,filled", "#fef9c3", "#a16207"),
            VisualKind::LogicalTensor => ("box", "rounded,filled", "#ede9fe", "#7c3aed"),
            VisualKind::LogicalOp => ("box", "filled", "#ede9fe", "#7c3aed"),
            VisualKind::Layout => ("note", "filled", "#fef9c3", "#a16207"),
            VisualKind::LayoutIr => ("box", "filled", "#fef9c3", "#a16207"),
        }
    }
}

/// THE EDGE GRAMMAR (all views): solid = bytes flow (dataflow through op
/// slots), dotted gray = constitution/information ("what a thing is made of"
/// and boundary membership — never an event), dashed = dependency ordering
/// (no bytes, only sequence). Labels carry only facts no node or slot already
/// states.
pub(crate) struct DotEmitter {
    body: String,
    next_id: usize,
    value_ids: HashMap<ClassId, usize>,
    logical_ids: HashMap<ClassId, usize>,
    buffer_tensor_ids: HashMap<ClassId, usize>,
    buffer_id_ids: HashMap<ClassId, usize>,
    /// (value e-class, buffer name) -> node: one box per RESIDENCE, so a
    /// value transported between buffers shows as two boxes sharing one
    /// LogicalTensor (the BufferTensor view's value/storage split).
    residence_ids: HashMap<(ClassId, String), usize>,
}

impl DotEmitter {
    pub(crate) fn new() -> Self {
        Self {
            body: String::new(),
            next_id: 0,
            value_ids: HashMap::new(),
            logical_ids: HashMap::new(),
            buffer_tensor_ids: HashMap::new(),
            buffer_id_ids: HashMap::new(),
            residence_ids: HashMap::new(),
        }
    }

    pub(crate) fn raw_node(&mut self, label: String, kind: VisualKind, tooltip: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let (shape, style, fill, border) = kind.style();
        self.body.push_str(&format!(
            "  n{} [label=\"{}\", shape=\"{}\", style=\"{}\", fillcolor=\"{}\", color=\"{}\", tooltip=\"{}\"];\n",
            id,
            dot_escape(&label),
            shape,
            style,
            fill,
            border,
            dot_escape(tooltip)
        ));
        id
    }

    /// A solid dataflow edge between plain (slotless) nodes — the logical
    /// story's child → op → tensor wiring.
    pub(crate) fn edge(&mut self, from: usize, to: usize, label: &str) {
        self.body.push_str(&format!(
            "  n{} -> n{} [label=\"{}\"];\n",
            from,
            to,
            dot_escape(label)
        ));
    }

    /// An op rendered as a slot table: a header with the op label, the untied
    /// slots in a west column of inputs and an east column of outputs (each in
    /// signature order), then one full-width row per tie — `in ↔ out` — so a
    /// tied pair reads as a single slot the storage threads through: the
    /// operand's edge docks on the row's west side and the tied result's edge
    /// leaves the same row's east side (dock tied results via
    /// [`tied_out_ports`]). Ties come from the op's declared contract only
    /// ([`Bufferizable::alias_info`] upstream, the node's carried `ties`
    /// downstream) — never from matching on the op. Edges dock at the slots
    /// via [`Self::edge_to_slot`] / [`Self::edge_from_slot`], so no edge label
    /// is needed — the slot itself names the operand.
    pub(crate) fn slot_node(
        &mut self,
        title: &str,
        in_slots: &[String],
        out_slots: &[String],
        ties: &[(usize, usize)],
        kind: VisualKind,
        tooltip: &str,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let (_, _, fill, border) = kind.style();
        // An out-of-range declaration renders as untied: the renderer never
        // panics on a contract the analyzer will reject loudly later.
        let ties: Vec<(usize, usize)> = ties
            .iter()
            .copied()
            .filter(|&(operand, result)| operand < in_slots.len() && result < out_slots.len())
            .collect();
        let tied_in = |index: usize| ties.iter().any(|&(operand, _)| operand == index);
        let tied_out = |index: usize| ties.iter().any(|&(_, result)| result == index);
        // Single-source rules only: borderless cells separated by <HR/>/<VR/>,
        // so no two borders ever sit adjacent (CELLBORDER next to the outer
        // BORDER renders as an ugly double line).
        let mut label = format!(
            "<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"2\" \
             COLOR=\"{border}\" BGCOLOR=\"{fill}\">\
             <TR><TD COLSPAN=\"2\" CELLPADDING=\"4\"><B>{}</B></TD></TR>",
            html_escape(title)
        );
        let untied_ins: Vec<&String> = in_slots
            .iter()
            .enumerate()
            .filter(|&(index, _)| !tied_in(index))
            .map(|(_, slot)| slot)
            .collect();
        let untied_outs: Vec<&String> = out_slots
            .iter()
            .enumerate()
            .filter(|&(index, _)| !tied_out(index))
            .map(|(_, slot)| slot)
            .collect();
        if !untied_ins.is_empty() || !untied_outs.is_empty() {
            label.push_str("<HR/><TR>");
            let mut first_column = true;
            for (slots, align) in [(untied_ins, "LEFT"), (untied_outs, "RIGHT")] {
                if !first_column {
                    label.push_str("<VR/>");
                }
                first_column = false;
                if slots.is_empty() {
                    label.push_str("<TD></TD>");
                    continue;
                }
                label.push_str(
                    "<TD CELLPADDING=\"0\"><TABLE BORDER=\"0\" CELLBORDER=\"0\" \
                                CELLSPACING=\"0\" CELLPADDING=\"3\">",
                );
                let mut first_row = true;
                for slot in slots {
                    if !first_row {
                        label.push_str("<HR/>");
                    }
                    first_row = false;
                    label.push_str(&format!(
                        "<TR><TD PORT=\"{}\" ALIGN=\"{align}\">{}</TD></TR>",
                        slot_port(slot),
                        html_escape(slot)
                    ));
                }
                label.push_str("</TABLE></TD>");
            }
            label.push_str("</TR>");
        }
        // Tie rows trail, matching the signature (destinations come last). The
        // row carries ONE port — the operand's — used from both compass sides.
        for &(operand, result) in &ties {
            label.push_str(&format!(
                "<HR/><TR><TD COLSPAN=\"2\" PORT=\"{}\" CELLPADDING=\"3\">{} ↔ {}</TD></TR>",
                slot_port(&in_slots[operand]),
                html_escape(&in_slots[operand]),
                html_escape(&out_slots[result]),
            ));
        }
        label.push_str("</TABLE>");
        self.body.push_str(&format!(
            "  n{id} [shape=\"plain\", label=<{label}>, tooltip=\"{}\"];\n",
            dot_escape(tooltip)
        ));
        id
    }

    /// A dataflow edge docking into an input slot on the target op.
    pub(crate) fn edge_to_slot(&mut self, from: usize, to: usize, slot: &str) {
        self.body
            .push_str(&format!("  n{from} -> n{to}:{}:w;\n", slot_port(slot)));
    }

    /// A dataflow edge leaving an output slot on the source op.
    pub(crate) fn edge_from_slot(&mut self, from: usize, slot: &str, to: usize) {
        self.body
            .push_str(&format!("  n{from}:{}:e -> n{to};\n", slot_port(slot)));
    }

    /// Emit (once per value e-class) a LayoutTensor node with its logical and
    /// layout detail nodes. Layout nodes are intentionally not deduplicated, so
    /// each value shows its own layout box.
    pub(crate) fn value_node(&mut self, value: &LayoutTensorInfo) -> usize {
        if let Some(id) = self.value_ids.get(&value.eclass) {
            return *id;
        }
        let id = self.raw_node(
            value.label.clone(),
            VisualKind::LayoutTensor,
            &value.tooltip,
        );
        self.value_ids.insert(value.eclass.clone(), id);

        let logical_id = self.logical_node(&value.logical);
        // Bare label, like the BufferId notes: the note shape already says
        // "metadata", so spelling the kind would break the no-prefix rule.
        let layout_id = self.raw_node(
            value.layout.label.to_string(),
            VisualKind::Layout,
            &value.layout.tooltip,
        );
        // Constructor ingredients flow into the term they form.
        self.info_edge(logical_id, id);
        self.info_edge(layout_id, id);
        id
    }

    pub(crate) fn logical_node(&mut self, logical: &LogicalInfo) -> usize {
        if let Some(id) = self.logical_ids.get(&logical.eclass) {
            return *id;
        }
        let id = self.raw_node(
            logical.label.to_string(),
            VisualKind::LogicalTensor,
            &logical.tooltip,
        );
        self.logical_ids.insert(logical.eclass.clone(), id);
        // The producing logical CONSTRUCTOR renders as its own op square
        // between the children and this tensor — same tensors-and-ops grammar
        // as every other domain (unary chains and all).
        if let Some(op) = &logical.op {
            let op_id = self.raw_node(op.clone(), VisualKind::LogicalOp, "");
            self.edge(op_id, id, "");
            for (port, child) in &logical.children {
                let child_id = self.logical_node(child);
                self.edge(child_id, op_id, port);
            }
        } else {
            for (port, child) in &logical.children {
                let child_id = self.logical_node(child);
                self.edge(child_id, id, port);
            }
        }
        id
    }

    /// Emit a BufferTensor (deduped by e-class) and its BufferId (deduped
    /// separately, since a single buffer id may back several buffer tensors).
    fn buffer(&mut self, buffer: &BufferInfo) -> (usize, usize) {
        let buffer_id = self.buffer_id_node(buffer);
        if let Some(tensor_id) = self.buffer_tensor_ids.get(&buffer.tensor_eclass) {
            return (*tensor_id, buffer_id);
        }
        let tensor_id = self.raw_node(
            buffer.tensor_label.clone(),
            VisualKind::BufferTensor,
            &buffer.tensor_tooltip,
        );
        self.buffer_tensor_ids
            .insert(buffer.tensor_eclass.clone(), tensor_id);
        self.info_edge(buffer_id, tensor_id);
        (tensor_id, buffer_id)
    }

    fn buffer_id_node(&mut self, buffer: &BufferInfo) -> usize {
        if let Some(id) = self.buffer_id_ids.get(&buffer.id_eclass) {
            return *id;
        }
        let id = self.raw_node(
            buffer.id_label.clone(),
            VisualKind::BufferId,
            &buffer.id_tooltip,
        );
        self.buffer_id_ids.insert(buffer.id_eclass.clone(), id);
        id
    }

    /// Emit (once per (value, buffer) RESIDENCE) a bare BufferTensor node —
    /// the rounded rose box of the buffer domain. `label` is the buffer
    /// tensor's OWN name (a source-program let binding) and is usually empty:
    /// planner-synthesized residences have no name, and copying the value's
    /// name down would only make the box read as a duplicate of the
    /// LayoutTensor beside it. What the node IS (a value paired with storage)
    /// is drawn by the BT view's edges: one to the LayoutTensor it
    /// represents, one to its BufferId storage node.
    pub(crate) fn residence_node(
        &mut self,
        value: &LayoutTensorInfo,
        label: &str,
        buffer_name: &str,
    ) -> usize {
        let key = (value.eclass.clone(), buffer_name.to_string());
        if let Some(id) = self.residence_ids.get(&key) {
            return *id;
        }
        let tooltip = format!("{}\n@ {}", value.tooltip, buffer_name);
        let id = self.raw_node(label.to_string(), VisualKind::BufferTensor, &tooltip);
        self.residence_ids.insert(key, id);
        id
    }

    /// An edge with explicit color and line style (dependency-ordering
    /// rendering).
    pub(crate) fn styled_edge(
        &mut self,
        from: usize,
        to: usize,
        label: &str,
        color: &str,
        style: &str,
    ) {
        self.body.push_str(&format!(
            "  n{from} -> n{to} [label=\"{}\", color=\"{color}\", fontcolor=\"{color}\", style=\"{style}\"];\n",
            dot_escape(label)
        ));
    }

    /// A CONSTITUTION edge: ingredient -> term ("this is made of that").
    /// Dotted, gray, unlabeled — the arrowhead shows the canonical
    /// ingredient -> term direction; the endpoint's hue and shape name the
    /// ingredient, so words would be redundant ink.
    pub(crate) fn info_edge(&mut self, ingredient: usize, term: usize) {
        self.info_edge_labeled(ingredient, term, "");
    }

    /// A constitution edge carrying a fact no node states — the boundary slot
    /// index ("in i" / "out i"). Every other constitution edge is unlabeled.
    pub(crate) fn info_edge_labeled(&mut self, ingredient: usize, term: usize, label: &str) {
        let label = if label.is_empty() {
            String::new()
        } else {
            format!(", label=\"{}\", fontcolor=\"#6b7280\"", dot_escape(label))
        };
        // penwidth is the DOT DIAMETER: the SVG post-processing turns the
        // dotted stroke into round-capped true dots (see write_svg_source).
        self.body.push_str(&format!(
            "  n{ingredient} -> n{term} [style=\"dotted\", color=\"#6b7280\", penwidth=\"1.6\"{label}];\n"
        ));
    }

    pub(crate) fn finish(self) -> String {
        let mut dot = String::from(
            "digraph ExtractedLayoutIR {\n  rankdir=LR;\n  graph [fontname=\"Helvetica\"];\n  node [fontname=\"Helvetica\"];\n  edge [fontname=\"Helvetica\"];\n",
        );
        dot.push_str(&self.body);
        dot.push_str("}\n");
        dot
    }
}

fn dot_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// Escape text for use inside a Graphviz HTML-like label.
pub(crate) fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The port each result's outgoing edge docks at: a tied result leaves from
/// the east side of its tie row (whose single port is the tied operand's
/// slot — see [`DotEmitter::slot_node`]), an untied result from its own slot
/// in the east column.
pub(crate) fn tied_out_ports(
    in_slots: &[String],
    out_slots: &[String],
    ties: &[(usize, usize)],
) -> Vec<String> {
    (0..out_slots.len())
        .map(|result| {
            ties.iter()
                .find(|&&(operand, tied)| tied == result && operand < in_slots.len())
                .map(|&(operand, _)| in_slots[operand].clone())
                .unwrap_or_else(|| out_slots[result].clone())
        })
        .collect()
}

/// A dot-safe port identifier for a slot name: prefixed (so a slot named like
/// a compass point, e.g. "n", can never collide) and restricted to
/// alphanumerics + underscore.
pub(crate) fn slot_port(slot: &str) -> String {
    let sanitized: String = slot
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("p_{sanitized}")
}
