# EClass decoding: an open, per-constructor API replacing the preference-ordered layout decoder

Status: design approved by Austin 2026-09-05 (shape and names). This document is
the refactor map and spec. It is written to be executed by an implementer without
further design judgment; everything marked DECISION was decided here and the
justification is beside it. Everything marked VERIFIED cites a file:line that was
read on trunk `logical-ssa-project` at 1f101d62 (the mapped source files are
byte-identical to c0ee79cd; only Cargo/vendor changed in between). Code blocks
marked ILLUSTRATION show the intended shape; the implementer fills in the
mechanics.

Branch: `feat/eclass-decoder` (this worktree). Standalone PR into
`logical-ssa-project`.

---

## 1. Summary, and the bug this fixes

**Summary.** Core's `luminal::layouts::decode_layout` walks one `Layout` e-class of
the serialized e-graph and returns *one* spelling as a `MirrorLayout` enum, chosen
by a FIXED preference order baked into the decoder (RightMajor > LeftMajor >
Strided > ElementOffset > BitOffset; VERIFIED `src/layouts.rs:427-535`). That
decoder is replaced by a small, open API over the serialized e-graph —
`EGraphView` / `EClass` / `ENode` — where every egglog constructor has a Rust
struct implementing `EgglogConstructor` (decode one e-node of that constructor),
and a caller asks a class for exactly the constructor(s) it cares about:
`first::<C>()`, `has::<C>()`, `require::<C>(who)`, or, for generic code that needs
a fact about a layout it did not author, the erased `spellings::<Layout>()` whose
items expose `LayoutFacts` (shape, width, span, evaluation). Preferences move to
the call sites, where they belong (Austin: "having a decoder that makes choices
about which constructor to look for is too general to be useful. Each call site
will have some preferences that need to be expressed somehow"). Decoders are
registered per `(sort, constructor)` beside the egg snippet that declares the
constructor, and a tripwire at assembly proves every constructor of a decoded sort
has exactly one decoder. `MirrorLayout`, the preference list, and the legacy
`decode_layout` are deleted.

**The bug.** Whisper tiny.en decodes one token, so its projections are
`x[1,384] @ w[384,384] + b[384]`; the cuBLASLt transpose-sandwich sibling's
destination frame is `[m,n] = [384,1]`. At an extent-1 axis the right-major and
left-major contiguous index maps are the same function (`r*1 + c` and `c*384 + r`
with `c ≡ 0`), and the e-graph knows it: the `[n,n]` pin collapse welds the
extent-1 `CoordVar` to `(IntLit 0)`, both stride tuples zip to the same affine
chain class, and the two contiguous-discovery rules
(VERIFIED `src/egglog_core/egglog_preamble.egg:3530-3560`, the `(3a)` arms that
`union ?layout (RightMajorContiguousElementLayoutLit …)` and
`… (LeftMajorContiguousElementLayoutLit …)`) put BOTH literals into ONE class. The
estate's bias decorators mint a bias form only on the LeftMajor spelling
(VERIFIED `crates/luminal_cuda_lite/src/ops/cublaslt/egg/cublaslt_marker_decorate.egg:182`,
premise `(= ?inner_L (LeftMajorContiguousElementLayoutLit ?ishape ?d_bits2))`);
the decoder, asked for "the" layout of that class, returned RightMajor by
preference (`src/layouts.rs:432-444`); `bind_destination` built a ROW descriptor
(`crates/luminal_cuda_lite/src/ops/cublaslt/exec.rs:344-346`); the coherence
fence `assert_bias_destination_order` (`exec.rs:380-397`) refused every candidate
genome, and the search died with "no candidate genome produced an executable
plan". PR #507 (open, branch `fix/cublaslt-degenerate-d-order`, commit c3b4c428)
added a guarded arm `M::RightMajor(_) if bias && (m == 1 || n == 1) => COL` to
`bind_destination`. This design subsumes that arm: the binding asks the class
`require::<LeftMajorContiguousElementLayout>()` and the class answers yes, because
the spelling is there. No arm, no degenerate-extent special case.

---

## 2. The API, final

### 2.1 Where things live (crate ownership)

| Item | File | Crate |
|---|---|---|
| `EGraphView`, `EClass`, `ENode`, `Sort`, `DynFacts`, `EgglogConstructor`, `ConstructorDecoder`, `ConstructorRegistry`, `Spellings` | NEW `src/egglog_core/egglog_utils/eclass.rs`, declared `pub mod eclass;` in `src/egglog_core/egglog_utils/mod.rs` (reachable as `luminal::egglog_utils::eclass::*`; `egglog_utils` is `#[path]`-mounted at `src/lib.rs:7-8`, VERIFIED) | core |
| `Layout` sort marker, `LayoutFacts`, the five constructor structs with their `EgglogConstructor` + `LayoutFacts` impls, the term decoders (`shape_term`, `bit_width`, `affine_chain`, `int_expr`), `layout_decoders()`, `DecodedLayout`, `LayoutDecodeCache`, `decode_layout_table` | `src/layouts.rs` | core |
| `core_decoders()`, `decoder_registry_for(matchers)` (mirrors `assembled_program_for`) | `src/egglog_snippet.rs` | core |
| `OpMatcher::decoders()` default method | `src/layout_ir/mod.rs` (trait at :518-547, VERIFIED) | core |
| `RegisteredOp::decoders()` delegating to its matcher | `crates/luminal_cuda_lite/src/ops/mod.rs` (struct at :74-77) | CUDA-lite |
| `luminal_reference::decoder_registry()` (`OnceLock`, like `assembled_program()` at `crates/luminal_reference/src/lib.rs:42-47`) | `crates/luminal_reference/src/lib.rs` | reference |
| CUDA-lite-declared constructors of decoded sorts | none today (VERIFIED: `cublaslt_marker_constructors.egg` declares datatypes `CublasLt*` and four `LayoutTensorOp*` constructors, no `Layout` constructor; no other CL op egg declares a constructor) — the plumbing exists for when one appears | CUDA-lite |

Doctrine kept throughout: no closed enum over constructors anywhere; diagnostics
print constructor NAMES; refusals are `Result` in the search path and `expect`
only after validation; tests locate by structure, never by class id; delete what
is superseded.

### 2.2 `eclass.rs` — the generic surface

Facts this rests on (VERIFIED in `egraph-serialize-0.3.0/src/lib.rs`): `EGraph {
nodes: IndexMap<NodeId, Node>, class_data: IndexMap<ClassId, ClassData>, .. }`
(:65-74); `Node { op: String, children: Vec<NodeId>, eclass: ClassId, cost,
subsumed: bool }` (:168-177); `ClassData { typ: Option<String>, extra }`
(:197-203); `EGraph::classes(&self) -> &IndexMap<ClassId, Class { id, nodes:
Vec<NodeId> }>` built once over ALL nodes (subsumed included) behind a `OnceCell`
(:107-120); `ClassId(Arc<str>)` with `From<&str>` (used as
`ClassId::from("val$synthetic")` in CL tests). Egglog's serializer (fork rev
1bb30831, identical lines to c2c0f151): a constructor/function row becomes a node
whose `op` is the function name (`src/serialize.rs:218`), whose `eclass` is the
OUTPUT value's class, and whose `children[i]` is a node inside argument i's class
(:204-215); every class touched gets `class_data[..].typ = Some(sort.name())`
(:393); primitives are rendered with `{:?}` of the base value (:353-362), so i64
literals are digit strings and strings are quoted — exactly what the old Reader
parsed (`src/layouts.rs:395-418`).

```rust
// src/egglog_core/egglog_utils/eclass.rs                                — SPEC
use std::any::Any;
use std::sync::Arc;
use anyhow::Result;
use egraph_serialize::{ClassId, EGraph, Node, NodeId};

/// Read-only view of one serialized e-graph plus the decoders that know its
/// constructors. Two references; `Copy`.
#[derive(Clone, Copy)]
pub struct EGraphView<'g> { egraph: &'g EGraph, decoders: &'g ConstructorRegistry }
impl<'g> EGraphView<'g> {
    pub fn new(egraph: &'g EGraph, decoders: &'g ConstructorRegistry) -> Self;
    /// Total: an id with no nodes yields a class whose `nodes()` is empty.
    pub fn class(&self, id: &ClassId) -> EClass<'g>;
    pub fn egraph(&self) -> &'g EGraph;
    pub fn decoders(&self) -> &'g ConstructorRegistry;
}

/// One e-class of the view. Cheap to clone (the id is an `Arc<str>`).
#[derive(Clone)]
pub struct EClass<'g> { view: EGraphView<'g>, id: ClassId }
impl<'g> EClass<'g> {
    pub fn id(&self) -> &ClassId;
    /// `class_data[id].typ` — the egglog sort name the serializer stamped.
    pub fn sort_name(&self) -> Option<&'g str>;
    /// Every e-node in the class: UNSUBSUMED FIRST, then by `NodeId` (the
    /// order the old Reader used, `src/layouts.rs:359-364,388-398`).
    pub fn nodes(&self) -> Vec<ENode<'g>>;
    pub fn nodes_named(&self, op: &str) -> impl Iterator<Item = ENode<'g>> + '_;
    /// Every distinct `op` in the class, in `nodes()` order (diagnostics).
    pub fn ops(&self) -> Vec<String>;
    /// Any node whose op parses as i64 / as a quoted string.
    pub fn i64_literal(&self) -> Option<i64>;
    pub fn string_literal(&self) -> Option<String>;

    // ---- typed, registry-free: the call site names the constructor ----
    /// Every node named `C::NAME` that decodes, in `nodes()` order.
    pub fn all<C: EgglogConstructor>(&self) -> Vec<C>;
    pub fn first<C: EgglogConstructor>(&self) -> Option<C>;
    /// `first::<C>().is_some()` — a DECODABLE spelling is present.
    pub fn has<C: EgglogConstructor>(&self) -> bool;
    /// `first::<C>()` or a refusal naming who asked, the class, its sort,
    /// `C::NAME`, and `present::<C::Sort>()`.
    pub fn require<C: EgglogConstructor>(&self, who: &str) -> Result<C>;
    /// Panics with `why` — only after validation.
    pub fn expect<C: EgglogConstructor>(&self, why: &str) -> C;

    // ---- erased, registry-driven: generic code over a sort ----
    /// Registered constructor names of sort `S` that appear in the class
    /// (≥1 node, decodable or not), in REGISTRY order.
    pub fn present<S: Sort>(&self) -> Vec<&'static str>;
    /// Every registered constructor of `S` present in the class, decoded.
    pub fn spellings<S: Sort>(&self) -> Spellings<S>;
    /// `spellings::<S>()` items, for the common "I just need a fact" shape.
    pub fn facts<S: Sort>(&self) -> impl Iterator<Item = Arc<S::Facts>>;
    /// Ops in the class with no decoder registered under sort `S` —
    /// see §8.3 for what this contains and why it is diagnostics-only.
    pub fn unknown<S: Sort>(&self) -> Vec<String>;
}

/// One e-node: op name + children as classes.
#[derive(Clone, Copy)]
pub struct ENode<'g> { view: EGraphView<'g>, id: &'g NodeId, node: &'g Node }
impl<'g> ENode<'g> {
    pub fn id(&self) -> &'g NodeId;
    pub fn op(&self) -> &'g str;
    pub fn class(&self) -> EClass<'g>;
    pub fn is_subsumed(&self) -> bool;
    pub fn arity(&self) -> usize;
    /// `children[i]` → that node's `eclass` (serialize.rs:204-215 contract).
    pub fn child(&self, index: usize) -> Option<EClass<'g>>;
    pub fn children(&self) -> Vec<EClass<'g>>;
}

/// A decoded sort: its egglog name and the erased fact surface its
/// constructors share (e.g. `Layout: Sort<Facts = dyn LayoutFacts>`).
pub trait Sort: 'static {
    const NAME: &'static str;
    type Facts: ?Sized + DynFacts;
    /// The downcast door, one line per sort (`{ facts }`): the trait
    /// upcast `dyn XFacts` -> `dyn Any`. See §2.5.
    fn upcast_any(facts: &Self::Facts) -> &dyn Any;
}

/// The sort-agnostic erased surface every decoded constructor exposes.
/// Blanket-implemented for every `EgglogConstructor` that is
/// `PartialEq + Debug + Send + Sync`, so constructor structs write NO
/// boilerplate for it. `Any` is a supertrait, which is what makes a
/// fact object upcastable (§2.5) — there is no `as_any()` here.
pub trait DynFacts: Any + std::fmt::Debug + Send + Sync {
    fn constructor(&self) -> &'static str;
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}
impl<T: EgglogConstructor + PartialEq + std::fmt::Debug + Send + Sync> DynFacts for T {
    fn constructor(&self) -> &'static str { T::NAME }
    fn dyn_eq(&self, other: &dyn Any) -> bool { other.downcast_ref::<T>() == Some(self) }
}

/// One egglog constructor mirrored as a Rust struct.
pub trait EgglogConstructor: Sized + 'static {
    const NAME: &'static str;
    type Sort: Sort;
    /// Decode ONE e-node of this constructor. `node.op() == Self::NAME` is
    /// guaranteed by the callers; children are read by index. Failure =
    /// this spelling does not parse (e.g. a foreign-shape coordinate under
    /// the owner-shape guard) — the caller moves on to the next node.
    fn decode(node: &ENode<'_>) -> Result<Self>;
    /// Erase into the sort's shared fact object. One line per struct:
    /// `Arc::new(self)` (the unsizing coercion is only expressible where
    /// the concrete type is known — see §8.2).
    fn erase(self) -> Arc<<Self::Sort as Sort>::Facts>;
}

/// A registered `(sort, constructor)` decoder. `decode` yields a
/// `Box<dyn Any>` that HOLDS an `Arc<<S as Sort>::Facts>` — the
/// convention `Spellings` downcasts back through.
pub struct ConstructorDecoder {
    pub sort: &'static str,
    pub name: &'static str,
    pub decode: fn(&ENode<'_>) -> Result<Box<dyn Any>>,
}
impl ConstructorDecoder {
    pub fn of<C: EgglogConstructor>() -> Self {
        Self {
            sort: <C::Sort as Sort>::NAME,
            name: C::NAME,
            decode: |node| C::decode(node).map(|c| Box::new(c.erase()) as Box<dyn Any>),
        }
    }
}

/// Decoders keyed by `(sort, constructor)`, in insertion order (core
/// built-ins first, then matcher contributions in registry order).
pub struct ConstructorRegistry { entries: Vec<ConstructorDecoder> /* + index */ }
impl ConstructorRegistry {
    /// Refuses a duplicate `(sort, name)`: "exactly one decoder" is the
    /// contract, and a duplicate is a registration bug, not a preference.
    pub fn new(decoders: impl IntoIterator<Item = ConstructorDecoder>) -> Result<Self>;
    pub fn get(&self, sort: &str, name: &str) -> Option<&ConstructorDecoder>;
    pub fn sorts(&self) -> impl Iterator<Item = &'static str>;
    pub fn constructors_of(&self, sort: &str) -> impl Iterator<Item = &'static str>;
    /// THE TRIPWIRE — §3.3.
    pub fn check(&self, egraph: &egglog::EGraph) -> Result<()>;
}

/// The decoded constructor instances of sort `S` found in one class.
pub struct Spellings<S: Sort> {
    decoded: Vec<Arc<S::Facts>>,            // registry order, then node order
    present: Vec<&'static str>,             // names with ≥1 node in the class
    failed: Vec<(&'static str, String)>,    // (name, reason) per undecodable node
}
impl<S: Sort> Spellings<S> {
    pub fn first<C: EgglogConstructor<Sort = S>>(&self) -> Option<&C>;   // S::upcast_any().downcast_ref
    pub fn all<C: EgglogConstructor<Sort = S>>(&self) -> Vec<&C>;
    pub fn has<C: EgglogConstructor<Sort = S>>(&self) -> bool;
    pub fn require<C: EgglogConstructor<Sort = S>>(&self, who: &str) -> Result<&C>;
    pub fn present(&self) -> &[&'static str];
    pub fn failed(&self) -> &[(&'static str, String)];
    pub fn iter(&self) -> impl Iterator<Item = &S::Facts>;
    pub fn any(&self) -> Option<&S::Facts>;   // first decoded, registry order
    pub fn is_empty(&self) -> bool;
}
impl<S: Sort> Clone for Spellings<S> {}                 // Arc clones; hand-written (no S: Clone)
impl<S: Sort> PartialEq for Spellings<S> {}             // same length, pairwise `dyn_eq`
impl<S: Sort> Eq for Spellings<S> {}
impl<S: Sort> std::fmt::Debug for Spellings<S> {}       // prints `present` and each item
```

Contracts, one line each:

* `EClass::nodes()` is the ONLY place ordering is decided; everything above it
  inherits "unsubsumed first, then `NodeId`". Subsumed nodes are kept because a
  subsumed constructor is still a true member of its class (the slice_pad lesson,
  `src/layouts.rs:380-384`).
* `first::<C>()` is registry-free and generic: `nodes_named(C::NAME)` →
  `C::decode` → first `Ok`. This is what the cuBLASLt fence uses.
* `spellings::<S>()` iterates `decoders.constructors_of(S::NAME)` in registry
  order; a `Box<dyn Any>` that does not downcast to `Arc<S::Facts>` is a registry
  bug and panics with the offending `(sort, name)`.
* Nothing in `eclass.rs` knows about layouts.

### 2.3 `layouts.rs` — the `Layout` sort and its five constructors

```rust
// src/layouts.rs                                                          — SPEC
pub struct Layout;
impl Sort for Layout { const NAME: &'static str = "Layout"; type Facts = dyn LayoutFacts; }

/// What every layout constructor discloses. Object-safe (no generics, all
/// `&self`), so `dyn LayoutFacts` is the erased item type.
pub trait LayoutFacts: DynFacts {
    /// The DOMAIN (every constructor carries one).
    fn shape(&self) -> &ShapeTerm;
    fn width(&self) -> BitWidthTerm;
    /// Storage reach in ELEMENTS as an expression, where the constructor
    /// discloses one (the packed ladder: `SpanExpr`); `None` for the
    /// offset-expression forms — nothing here guesses a reach.
    fn span_elements(&self) -> Option<IntExprTerm>;
    /// The constructor's read function evaluated at literal coordinates
    /// (front-indexed), down to the flat ELEMENT index. Runtime
    /// convenience, never planner machinery. Fail-closed on symbolic
    /// extents, foreign rank, out-of-domain coordinates, a mid-element
    /// bit offset, a negative result.
    fn element_index(&self, coords: &[usize]) -> Result<usize>;
}
impl PartialEq for dyn LayoutFacts { fn eq(&self, o: &Self) -> bool { self.dyn_eq(o as &dyn Any) } }
impl Eq for dyn LayoutFacts {}

// The five structs (unchanged fields, `src/layouts.rs:129-167`) each get:
impl EgglogConstructor for RightMajorContiguousElementLayout {
    const NAME: &'static str = "RightMajorContiguousElementLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let shape = shape_term(&node.child(0).ok_or_else(..)?).ok_or_else(..)?;
        let width = bit_width(&node.child(1).ok_or_else(..)?).ok_or_else(..)?;
        Ok(Self { shape, width })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> { Arc::new(self) }
}
impl LayoutFacts for RightMajorContiguousElementLayout { /* shape/width/span (numel)/element_index (row-major fold) */ }
// LeftMajorContiguousElementLayout: children (0 shape, 1 width); span = numel; element_index = col-major fold.
// StridedElementLayout:            children (0 shape, 1 chain, 2 width); chain via `affine_chain(child1, owner_shape = child0)`; span = SpanExpr; element_index = Σ eval(summand).
// ElementOffsetExpressionLayout:   children (0 offset, 1 shape, 2 width); offset via `int_expr(child0, Some(child1))`; span None; element_index = eval(offset).
// BitOffsetExpressionLayout:       children (0 offset, 1 shape, 2 width); span None; element_index = eval(offset)/width with the alignment check.

/// The core preamble's Layout constructors, in this order (it is the
/// `Spellings` iteration order and therefore what `present()` prints).
pub fn layout_decoders() -> Vec<ConstructorDecoder> {
    vec![
        ConstructorDecoder::of::<RightMajorContiguousElementLayout>(),
        ConstructorDecoder::of::<LeftMajorContiguousElementLayout>(),
        ConstructorDecoder::of::<StridedElementLayout>(),
        ConstructorDecoder::of::<ElementOffsetExpressionLayout>(),
        ConstructorDecoder::of::<BitOffsetExpressionLayout>(),
    ]
}

// Term decoders — the old Reader's `decode_shape` (:550-566), `decode_bit_width`
// (:538-548), `decode_affine_chain` (:568-585), `decode_expr_list` (:587-620),
// `parse_int_expr*` (:622-769, memo + cycle-taint rule) MOVED onto `EClass`
// unchanged in logic:
pub fn shape_term(class: &EClass<'_>) -> Option<ShapeTerm>;
pub fn bit_width(class: &EClass<'_>) -> Option<BitWidthTerm>;
pub fn affine_chain(class: &EClass<'_>, owner_shape: &EClass<'_>) -> Option<Vec<IntExprTerm>>;
pub fn int_expr(class: &EClass<'_>, owner_shape: Option<&EClass<'_>>) -> Option<IntExprTerm>;
// `IntExprTerm::eval_at(&self, coords: &[usize]) -> Result<i64>` — the
// coordinate evaluator today duplicated as `eval_term` in
// crates/luminal_cuda_lite/src/layouts.rs:29-64 and
// tests/test_runtime/src/test_equality.rs:23-58; take test_equality's
// (it has the CeilDiv arm), the CL copy becomes a one-line wrapper.
```

### 2.4 `DecodedLayout` — the plan layout both runtimes carry

DECISION: the plan layout carries the decoded SPELLING SET, not just the class
id. Refutation of the alternative (`{class, dtype}` only, facts looked up live):
`arena::buffer_bytes` (`crates/luminal_cuda_lite/src/arena.rs:79-104`), the
reference executor's storage sizing (`crates/luminal_reference/src/runtime.rs:549-570`),
CL codegen (`kernels.rs:365-598`), readback (`layouts.rs:68-150`) and
`bind_destination` all run at EXECUTE time, where `load_plan` callers and every
hand-built test plan (`codegen_identity.rs`, `plan_smoke.rs`,
`cublaslt_contracts_cpu.rs`, `composed_read_families.rs`, `src/test_support.rs:3829-3862`)
have no e-graph; and the bug fix itself needs the WHOLE spelling set of the
destination class (`has::<LeftMajor>()`), not one chosen spelling. So the plan
carries every decoded spelling; the class id is kept for diagnostics and as the
cache key; the dtype fact rides along as today.

```rust
// src/layouts.rs                                                          — SPEC
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedLayout {
    /// The layout e-class this was decoded from (serialized id; diagnostics
    /// and cache key only — never pinned by a test). Hand-built plans use
    /// `ClassId::from("hand-built")`.
    pub class: ClassId,
    pub dtype: Option<crate::dtype::PlanDtype>,
    /// Every registered Layout constructor present in the class, decoded.
    pub spellings: Spellings<Layout>,
}
impl DecodedLayout {
    /// The decoder: refuses a class with ZERO decoded spellings (message
    /// lists `present` and `failed`) and a class whose decoded spellings
    /// DISAGREE on `shape()` or `width()` (layout identity = domain ×
    /// interpretation, preamble :217; a mixed-domain class is a false
    /// union — see §8.5).
    pub fn from_class(class: &EClass<'_>, dtype: Option<PlanDtype>) -> Result<Self>;
    /// Hand-built, one spelling (test fixtures).
    pub fn of(spelling: impl EgglogConstructor<Sort = Layout>, dtype: Option<PlanDtype>) -> Self;
    /// Hand-built, several spellings of ONE function (the degenerate-extent
    /// fixture: RightMajor[384,1] and LeftMajor[384,1]).
    pub fn of_spellings(spellings: Vec<Arc<dyn LayoutFacts>>, dtype: Option<PlanDtype>) -> Self;

    // Class-invariant facts (all spellings denote one function).
    pub fn shape(&self) -> &ShapeTerm;                       // first spelling's
    pub fn width_bits(&self) -> i64;
    pub fn literal_extents(&self) -> Option<Vec<usize>>;     // was MirrorLayout::literal_extents
    pub fn literal_span_elements(&self) -> Option<usize>;    // first spelling that discloses a span, evaluated
    pub fn element_index(&self, coords: &[usize]) -> Result<usize>;   // first spelling's read function
    // Call-site preferences, delegated to `spellings`.
    pub fn first<C: EgglogConstructor<Sort = Layout>>(&self) -> Option<&C>;
    pub fn has<C: EgglogConstructor<Sort = Layout>>(&self) -> bool;
    pub fn require<C: EgglogConstructor<Sort = Layout>>(&self, who: &str) -> Result<&C>;
    pub fn present(&self) -> &[&'static str];
}

pub type LayoutDecodeCache = HashMap<(ClassId, Option<PlanDtype>), DecodedLayout>;   // unchanged

/// Signature change: the view replaces the bare e-graph. Body as today
/// (`src/layouts.rs:830-873`) with `decode_layout(egraph, &class)` replaced by
/// `DecodedLayout::from_class(&view.class(&value.layout.eclass), value.dtype_enum)`.
pub fn decode_layout_table(
    view: &EGraphView<'_>,
    graph: &crate::layout_ir::ExtractedGraph,
    who: &str,
    cache: &mut LayoutDecodeCache,
) -> Result<HashMap<ClassId, DecodedLayout>>;
```

`PlanLayout` requires only `Clone + Debug` (VERIFIED `src/bufferize.rs:99-100`);
`PartialEq` is used by the reference executor's fold check
(`crates/luminal_reference/src/runtime.rs:767`) and the transport pin
(`src/test_support.rs:1497`) — both keep working: equal classes decode to equal
spelling sets, and two different layout classes always differ in at least one
decoded term (constructors are hash-consed, so equal spellings would have been
one class).

### 2.5 The downcast idiom

Recovering a concrete struct from the erased path:

```rust
// (a) explicit as_any — a hand-written door on the fact trait
let lm = facts.as_any().downcast_ref::<LeftMajorContiguousElementLayout>();
// (b) trait upcasting to dyn Any — requires Rust ≥ 1.86
let lm = (facts as &dyn Any).downcast_ref::<LeftMajorContiguousElementLayout>();
```

DECISION (2026-09-05, amended): use (b), trait upcasting. The original decision
was (a), because the workspace declared `rust-version = "1.85"` and upcasting
stabilized in 1.86. That floor was already false — the tree uses let-chains
(stable 1.88) at 27 `&& let` sites — so the same follow-up raises it to
`rust-version = "1.91"` (`Cargo.toml:5`), the oldest toolchain we run (this Mac
1.91.1, the A100 box 1.95). `DynFacts::as_any` is therefore DELETED; `Any` stays a supertrait of
`DynFacts`, which is exactly what makes a fact object upcastable.

One nuance the type system forces, VERIFIED with rustc 1.91.1: an upcast is a
coercion between two KNOWN types, so it cannot be written over the generic
`S::Facts`, which is `?Sized` —

```
error[E0277]: the size for values of type `<S as Sort>::Facts` cannot be known
              at compilation time
   = note: required for the cast from `&<S as Sort>::Facts` to `&(dyn Any + 'static)`
```

— so the upcast is spelled once per SORT, as `Sort::upcast_any`:

```rust
impl Sort for Layout {
    const NAME: &'static str = "Layout";
    type Facts = dyn LayoutFacts;
    fn upcast_any(facts: &Self::Facts) -> &dyn Any { facts }   // the upcast
}
```

— the same shape, and for the same reason, as `EgglogConstructor::erase` (§8.2):
one line where the concrete `dyn` type is known. Constructor structs still write
no boilerplate, and `Spellings::first::<C>()` wraps the door so call sites spell
neither. Concrete sites (`impl PartialEq for dyn LayoutFacts`) upcast directly.

---

## 3. Registration, plumbing, and the tripwire

### 3.1 Registration rides on the matchers

DECISION: decoders travel on `OpMatcher`, not as a new field on `RegisteredOp`.
`OpMatcher::snippets()` (`src/layout_ir/mod.rs:534-542`) is where a constructor's
DECLARATION travels; the decoder for that constructor belongs beside it, and the
matcher is the one type all three registries share (`RegisteredOp.matcher`,
`ReferenceOp.matcher: fn() -> Box<dyn OpMatcher>` at
`crates/luminal_reference/src/ops/mod.rs:144-150`, test_runtime's matcher list).
A `RegisteredOp` field would leave the reference runtime with no path.

```rust
// src/layout_ir/mod.rs, in `pub trait OpMatcher` after `snippets()`      — SPEC
/// Decoders for every constructor this matcher's snippets DECLARE for a
/// decoded sort. Mirrors `snippets()`: the declaration and its decoder
/// travel together. Default empty — no matcher declares one today.
fn decoders(&self) -> Vec<crate::egglog_utils::eclass::ConstructorDecoder> { Vec::new() }

// crates/luminal_cuda_lite/src/ops/mod.rs, impl RegisteredOp             — SPEC
pub fn decoders(&self) -> Vec<ConstructorDecoder> { self.matcher.decoders() }

// src/egglog_snippet.rs, beside `assembled_program_for`                   — SPEC
/// What core's own preamble declares and decodes: the five Layout constructors.
pub fn core_decoders() -> Vec<ConstructorDecoder> { crate::layouts::layout_decoders() }
/// The registry for a runtime's matcher set — the same shape as
/// `assembled_program_for(matchers)`: core built-ins, then each matcher's
/// contribution in registry order. Errors on a duplicate `(sort, name)`.
pub fn decoder_registry_for(matchers: &[Box<dyn OpMatcher>]) -> Result<ConstructorRegistry> {
    ConstructorRegistry::new(core_decoders().into_iter().chain(matchers.iter().flat_map(|m| m.decoders())))
}

// crates/luminal_reference/src/lib.rs, beside `assembled_program()`       — SPEC
pub fn decoder_registry() -> &'static ConstructorRegistry {
    static R: OnceLock<ConstructorRegistry> = OnceLock::new();
    R.get_or_init(|| luminal::egglog_snippet::decoder_registry_for(&ops::built_in_matchers())
        .expect("the built-in reference registry has no duplicate decoders"))
}
```

### 3.2 Every signature that changes, and every caller

| Function | Today | New | Who constructs the view |
|---|---|---|---|
| `luminal::layouts::decode_layout_table` (`src/layouts.rs:830`) | `(egraph: &EGraph, graph, who, cache)` | `(view: &EGraphView<'_>, graph, who, cache)` | callers below |
| `luminal::layouts::decode_layout` / `decode_layout_for` (:343, :773) | exist | DELETED | — |
| `src/test_support.rs:1478` | `decode_layout_table(&egraph, &graph, "dtype row test", &mut LayoutDecodeCache::new())` | `let view = EGraphView::new(&egraph, luminal_reference::decoder_registry()); decode_layout_table(&view, …)` | core test (already depends on `luminal_reference`, :2431) |
| `crates/luminal_reference/src/harness.rs:209` | `(&serialized, &dps, "plain plan", …)` | `EGraphView::new(&serialized, crate::decoder_registry())` | harness |
| `crates/luminal_reference/src/runtime.rs:1684` | same shape | same as above | runtime test |
| `crates/luminal_reference/src/search.rs:355, 426` (inside `search_implementations_with_ops`, :182) | `(egraph, &dps, "implementation search", &mut layout_cache)` | build once at :249 next to `layout_cache`: `let view = EGraphView::new(egraph, crate::decoder_registry());` then `&view` at both sites | search loop; NO change to `search_implementations_with_ops`'s signature |
| `crates/luminal_cuda_lite/src/search.rs:503, 657` (inside `search_implementations`, :328) | same | at :391 next to `layout_cache`: `let decoders = luminal::egglog_snippet::decoder_registry_for(matchers)?; let view = EGraphView::new(egraph, &decoders);` | search loop; NO signature change (`matchers` is already a parameter) |
| `crates/luminal_cuda_lite/src/finalists.rs:245` (`Finalists::build_plan`) | `decode_layout_table(self.egraph, &dps, "finalist", &mut self.layout_cache)` | `Finalists` gains a field `decoders: ConstructorRegistry` built in `new` (:131-155) from `matchers`; `build_plan` does `let view = EGraphView::new(self.egraph, &self.decoders);` | finalists; `Finalists::new` signature unchanged |

That is all 8 callers. The search loops, `Finalists::new`, `CudaRuntime::search`,
`bucketed_search_implementations` and the `BucketAssembly` struct keep their
public signatures; the only new state is `Finalists.decoders` and
`CudaRuntime.decoders` (§3.3).

### 3.3 The tripwire — VERIFIED mechanism

Runs against the LIVE `egglog::EGraph` right after a successful
`parse_and_run_program` and before `serialize`, where every runtime already holds
it. The live schema is reachable read-only — no `&mut`, no text scanning:

* `egglog::EGraph::functions_iter(&self) -> impl Iterator<Item = (&String, &Function)>`
  — fork `src/lib.rs:1236` (also `get_function(&self, name) -> Option<&Function>`,
  :2318; `get_function_names`, :1230).
* `Function::func_type(&self) -> &FuncType` — `src/lib.rs:342`;
  `Function::name()` — :337.
* `pub struct FuncType { pub name, pub subtype: FunctionSubtype, pub input:
  Vec<ArcSort>, pub output: ArcSort }` — `src/typechecking.rs:115-120`;
  re-exported as `egglog::FuncType` (`src/lib.rs:74`).
* `pub enum FunctionSubtype { Constructor, Custom }` — `src/ast/mod.rs:1298-1302`
  (`egglog::ast` is `pub mod`, `src/lib.rs:3`). `(datatype …)` variants and
  `(constructor …)` are `Constructor`; `(function … :no-merge)` rows such as
  `layout-of` (`egglog_preamble.egg:451`, output sort `Layout`) are `Custom` and
  must be skipped — they are not spellings.
* `egglog::sort::Sort::name(&self) -> &str` — `src/sort/mod.rs:52`; `ArcSort =
  Arc<dyn Sort>`.
* NOT usable: `EGraph::type_info(&mut self)` (`src/lib.rs:589`) takes `&mut`;
  `TypeInfo::get_func_type`/`is_constructor` (`typechecking.rs:1146, 1157`) are
  therefore reachable only mutably. The `functions_iter` route is the one to use.
* Fallback (NOT chosen): regex over the assembled text for
  `(constructor NAME (…) SORT`. Fragile (comments, `:cost`, multi-line forms) and
  blind to `datatype` variants. The live schema is authoritative.

```rust
// eclass.rs                                                               — SPEC
impl ConstructorRegistry {
    pub fn check(&self, egraph: &egglog::EGraph) -> Result<()> {
        use egglog::ast::FunctionSubtype;
        let mut problems = Vec::new();
        // (1) every declared constructor of a decoded sort has a decoder
        for (name, func) in egraph.functions_iter() {
            let ft = func.func_type();
            if ft.subtype != FunctionSubtype::Constructor { continue; }
            let sort = ft.output.name();
            if !self.sorts().any(|s| s == sort) { continue; }
            if self.get(sort, name).is_none() {
                problems.push(format!("sort {sort}: constructor `{name}` is declared by the program but has no registered decoder"));
            }
        }
        // (2) every decoder names a constructor the program declares, under the right sort
        for d in &self.entries {
            match egraph.get_function(d.name) {
                None => problems.push(format!("sort {}: decoder for `{}` names a constructor the program does not declare", d.sort, d.name)),
                Some(f) => {
                    let ft = f.func_type();
                    if ft.subtype != FunctionSubtype::Constructor || ft.output.name() != d.sort {
                        problems.push(format!("sort {}: decoder for `{}` is registered under sort {} but the program declares it as {} with output sort {}",
                            d.sort, d.name, d.sort, ft.subtype.label(), ft.output.name()));
                    }
                }
            }
        }
        if problems.is_empty() { return Ok(()); }
        anyhow::bail!(
            "egglog constructor decoders are incomplete for the assembled program:\n  - {}\n\
             Every constructor of a decoded sort needs exactly one decoder, registered beside the \
             snippet that declares it (OpMatcher::decoders); see docs/design/eclass-decoder.md.",
            problems.join("\n  - ")
        )
    }
}
```

Duplicates are refused earlier, at `ConstructorRegistry::new`, with the same
message family ("sort {sort}: two decoders registered for `{name}`").

Where it is called (REQUIRED — the four places a runtime assembles for search;
each has the live `EGraph` in a local right before `serialize`):

1. CL `CudaRuntime::assemble_and_saturate`, `crates/luminal_cuda_lite/src/runtime.rs:432-456`:
   after the `if let Err(err) = egraph.parse_and_run_program(…)` block, before
   `let serialized = egraph.serialize(…)` at :456 — `self.decoders.check(&egraph)?;`.
   `CudaRuntime` gains `decoders: ConstructorRegistry` built in
   `load_with_registry` (:143-146) as
   `luminal::egglog_snippet::decoder_registry_for(&matchers)?` (the matcher column
   is already extracted there). `Default` gets an empty registry, like `matchers`.
2. CL bucketed search, `crates/luminal_cuda_lite/src/search.rs:931-934`: `BucketAssembly`
   gains `decoders: &'a ConstructorRegistry` (set from `&self.decoders` at
   `runtime.rs:622`); call `assembly.decoders.check(&egraph)?` before :934.
3. Reference `ReferenceRuntime::search`, `crates/luminal_reference/src/runtime.rs:332-365`:
   after the error block, before :365 — `crate::decoder_registry().check(&egraph)?;`.
4. Reference bucketed search, `crates/luminal_reference/src/search.rs:536-538`: same
   call before :538.

RECOMMENDED (cheap, keeps fixtures honest): `luminal_reference::harness::serialize_fixture`
(`harness.rs:72-85`), `harness.rs:186-190`, `test_runtime/src/lib.rs:144-146`,
`src/extraction.rs:4309-4311`. Not required for correctness — the fixtures use
core's preamble, which core's own test below already checks.

Core unit tests (in `eclass.rs` / `layouts.rs`):

* `core_preamble_layout_constructors_all_have_decoders`: `new_egraph()` +
  `assembled_program_for(&[])` + `ConstructorRegistry::new(core_decoders())?.check(&egraph)` is `Ok`.
* `a_missing_decoder_is_named`: same program, registry = `core_decoders()` minus
  `BitOffsetExpressionLayoutLit` → `Err` whose message contains
  ``sort Layout: constructor `BitOffsetExpressionLayoutLit` is declared by the program but has no registered decoder``.
* `a_stale_decoder_is_named`: registry with an extra
  `ConstructorDecoder { sort: "Layout", name: "NoSuchLayoutLit", .. }` → `Err`
  containing ``decoder for `NoSuchLayoutLit` names a constructor the program does not declare``.
* `a_duplicate_registration_is_refused`: `ConstructorRegistry::new` on
  `core_decoders()` twice → `Err`.
* CL: `cuda_registry()`'s matchers → `decoder_registry_for(..)?.check(&saturated egraph)` is `Ok`
  (belongs in `crates/luminal_cuda_lite/tests/registry_selection.rs`).

Failure at a runtime site surfaces as the search's error
(`context("cuda-lite saturation failed")` style is NOT wrapped around it — the
message is self-describing).

---

## 4. Consumer-by-consumer refactor map

Legend: SITE = file:line on trunk; NOW = current shape; NEW = replacement. Where
a site is a match over the five `MirrorLayout` variants, "chain" means the
call-site-preference chain of `first::<C>()` calls in the order RightMajor,
LeftMajor, Strided, ElementOffset, BitOffset with a terminal `bail!` that prints
`layout.present()`. That order is chosen HERE, by the codegen, because the most
structured spelling yields the simplest C; it is no longer anyone else's business.

### 4.1 `crates/luminal_cuda_lite/src/kernels.rs` (10 sites)

| SITE | NOW | NEW |
|---|---|---|
| :23 `use luminal::layouts::DecodedLayout;` | — | unchanged (add `use luminal::layouts::{RightMajorContiguousElementLayout as RM, LeftMajorContiguousElementLayout as LM, StridedElementLayout as ST, ElementOffsetExpressionLayout as EO, BitOffsetExpressionLayout as BO};`) |
| :64 `slot.layout.mirror.literal_extents()` in `from_descriptors` | via `.mirror` | `slot.layout.literal_extents()` — message unchanged (`"{label} {role} has symbolic layout extents (no numeric codegen)"`, pinned by `codegen_identity.rs:573,939`) |
| :145 module comment "the runtime's decoded `MirrorLayout`" | text | "the runtime's decoded `DecodedLayout` (every spelling of the elected class)" |
| :365-392 `read_affine` | `use MirrorLayout as M; if layout.mirror.literal_extents()…; match &layout.mirror { M::RightMajor(_) => strides; M::LeftMajor(_) => …; M::Strided(st) => …; M::ElementOffset(eo) => …; M::BitOffset(bo) => … }` | `layout.literal_extents()` for the domain check; then the chain: `if let Some(_) = layout.first::<RM>() { Affine::from_strides(..) } else if let Some(_) = layout.first::<LM>() { .. } else if let Some(st) = layout.first::<ST>() { .. } else if let Some(eo) = layout.first::<EO>() { .. } else if let Some(bo) = layout.first::<BO>() { .. } else { None }` — arm bodies verbatim |
| :517 `use luminal::layouts::MirrorLayout;` | — | delete |
| :539-598 `layout_read_index` offset emission | `match &layout.mirror { … five arms … }` | the same chain; arm bodies VERBATIM (the emitted strings are pinned bit-for-bit at `codegen_identity.rs:405,440,464,491,523`); terminal arm `bail!("operand {operand}: no lowerable layout spelling — present: {:?}", layout.present())` |

Why not one uniform lowering (`LayoutFacts::offset_term()` → `lower_layout_term`)?
It changes the emitted C (`c1 + (c0 * 8LL)` would gain outer parentheses and the
RightMajor form would lose its stride spelling), breaking the five string pins.
Recorded as a follow-up in §8.7; Austin decides whether to re-bless the pins.

### 4.2 `crates/luminal_cuda_lite/src/ops/cublaslt/exec.rs` (5 sites + the #507 arm)

| SITE | NOW | NEW |
|---|---|---|
| :73-88 module doc (bias forms paragraph) | describes the tripwire as the fence | rewrite: the fence is `require::<LeftMajorContiguousElementLayout>` on the destination class; the degenerate-extent coincidence is not a case because the class carries both spellings and `require` reads the class, not one chosen spelling |
| :321 `dest: &luminal::layouts::DecodedLayout` | — | unchanged |
| :324 `use luminal::layouts::MirrorLayout as M;` | — | delete; `use luminal::layouts::{LeftMajorContiguousElementLayout, RightMajorContiguousElementLayout};` |
| :325 `dest.mirror.literal_extents()` | — | `dest.literal_extents()`; message unchanged (`"…SYMBOLIC extents…"`, pinned `cublaslt_contracts_cpu.rs:498`) |
| :344-360 `let desc = match &dest.mirror { M::RightMajor(_) => row, M::LeftMajor(_) => col, other => bail!(…STRIDED/ELEMENT-OFFSET-EXPRESSION/BIT-OFFSET-EXPRESSION…) }` | preference-blind match | the binding below |
| :363-366 `assert_bias_destination_order(call, who)` call | second fence | DELETE this call (the `require` IS the fence); the function itself stays for `device_call.rs:261` — see below |
| #507 arm (branch only, `M::RightMajor(_) if call.bias_operand.is_some() && (call.m == 1 \|\| call.n == 1) => col`) | not on trunk | NEVER lands; see §5 |

The binding, final:

```rust
// exec.rs, bind_destination, replacing :344-366                           — SPEC
let desc = if call.bias_operand.is_some() {
    // THE FENCE. The estate's two bias decorators mint a bias form only
    // when the claimed D class HOLDS the LeftMajor spelling
    // (egg/cublaslt_marker_decorate.egg, premise
    // `(= ?inner_L (LeftMajorContiguousElementLayoutLit ?ishape ?d_bits2))`).
    // Ask the class the same question. A degenerate extent, where the
    // RightMajor spelling shares the class, is not a case: the answer is
    // still yes. A bias form whose class lacks the spelling is unreachable
    // from the estate — a real drift, refused before any descriptor exists.
    dest.require::<LeftMajorContiguousElementLayout>(who).with_context(|| {
        format!(
            "cuBLASLt {who}: unreachable: the bias decorators require a LeftMajor D; a bias form \
             ({}) reached the executor whose destination class holds {:?} and no \
             LeftMajorContiguousElementLayoutLit — the library refuses BIAS/RELU_BIAS on a \
             ROW-order D (CUBLAS_STATUS_NOT_SUPPORTED, measured on the A100 2026-08-28); \
             refused BEFORE dispatch, no bytes move",
            call.form.constructor_name(),
            dest.present()
        )
    })?;
    LtDesc::col(call.m, call.n, call.m.max(1))
} else if dest.has::<LeftMajorContiguousElementLayout>() {
    LtDesc::col(call.m, call.n, call.m.max(1))
} else if dest.has::<RightMajorContiguousElementLayout>() {
    LtDesc::row(call.m, call.n, call.n.max(1))
} else {
    bail!(
        "cuBLASLt {who}: the plan elected a destination layout whose class holds {:?}; this \
         backend writes only the two dense orders cuBLASLt can express (RightMajor -> \
         CUBLASLT_ORDER_ROW, LeftMajor -> CUBLASLT_ORDER_COL). Strided and offset-expression \
         destinations are NOT lowered — a CAPABILITY refusal (the host-call mirror of the \
         codegen path's identity-index write fence), never a guess.",
        dest.present()
    )
};
call.d = desc;
call.c = desc;
Ok(())
```

Non-bias with BOTH contiguous spellings (degenerate extents only) now binds COL
where it bound ROW; same bytes, same reach (#507's memory-identity argument,
`validate_against` passes both). Flagged in §8.9 as a behaviour change Austin
should see.

Test message pins affected: `cublaslt_contracts_cpu.rs:225-234` asserts
`"unreachable"`, `"bias decorators require a LeftMajor D"`, `"Row-order D descriptor"`,
`"refused BEFORE dispatch"`. Keep the first, second and fourth; replace the third
with `msg.contains("RightMajorContiguousElementLayoutLit")` (the class listing) and
`!msg.contains("LeftMajorContiguousElementLayoutLit —")` is unnecessary — the
message's "no LeftMajor…" clause is stable text. `cublaslt_contracts_cpu.rs:464-485`
asserts `"STRIDED"`, `"CAPABILITY refusal"`, `"ELEMENT-OFFSET-EXPRESSION"`: the
new capability message prints the class's constructor names instead of the
shouted kind; retarget to `"StridedElementLayoutLit"` / `"CAPABILITY refusal"` /
`"ElementOffsetExpressionLayoutLit"`.

**`assert_bias_destination_order` (:380-397).** DECISION: keep the function, drop
its call from `bind_destination`, and re-document it as what its remaining caller
makes it — a VENDOR-PRECONDITION check at dispatch
(`crates/luminal_cuda_lite/src/ops/cublaslt/device_call.rs:261`, called on every
`LtCall` however built, including the hand-built direct-dispatch contract tests).
Justification: inside `bind_destination` it restated the function's own
postcondition (the bias branch sets COL or refuses) — category (1) of the module's
own check taxonomy, "OUT". At `device_call.rs:261` it guards the library's
documented refusal for a call that never passed through `bind_destination` —
category (2), "STAYS". A `debug_assert!` would be wrong there: the hand-built path
is a release path. Its doc comment (:368-379) must stop describing itself as the
estate fence.

### 4.3 `crates/luminal_cuda_lite/src/layouts.rs` (5 sites)

| SITE | NOW | NEW |
|---|---|---|
| :1-11 module doc | "core owns the decoder and publishes `DecodedLayout`" | keep; add one sentence: the read-back helpers evaluate through `LayoutFacts::element_index` |
| :29-64 `pub fn eval_term` | local evaluator (no CeilDiv arm) | `pub fn eval_term(expr, coords) -> Result<i64> { expr.eval_at(coords) }` (core's evaluator; keeps the pub name for `view_admission.rs`, `examples/support`, device tests) |
| :68-120 `pub fn element_index(layout: &DecodedLayout, coords)` | `use M; layout.mirror.literal_extents()…; match &layout.mirror { five arms }` | `layout.element_index(coords)` — signature unchanged; the five arms move into core as each struct's `LayoutFacts::element_index` |
| :124-150 `pub fn dense_f32` | `layout.mirror.literal_extents()` | `layout.literal_extents()`; rest unchanged |
| :13 `pub use luminal::layouts::DecodedLayout;` | — | unchanged |

### 4.4 Other CUDA-lite runtime readers

| SITE | NOW | NEW |
|---|---|---|
| `arena.rs:79-104 buffer_bytes` | `buffer.layout.mirror.literal_span_elements()` | `buffer.layout.literal_span_elements()`; messages unchanged |
| `device.rs:319` | `buffer.layout.dtype` | unchanged |
| `device.rs:525-527` | `bind_destination(&mut call, &dest_slot.layout, label)` | unchanged |
| `device_call.rs:261` | `assert_bias_destination_order(call, "dispatch")?` | unchanged (see 4.2) |
| `heuristic.rs` | sums `op.heuristic_cost` | unchanged (reads no layout) |
| `profile.rs:139`, `runtime.rs:65,72,795-816` | `BufferIrGraph<DecodedLayout>` types | unchanged |
| `ops/cublaslt/mod.rs:389-470, 562-700` (`layout_class_of`, `stride_chain`, `leading_dimension`, the `BitOffsetExpressionLayoutLit` view walks at :649, :689) | per-enode SPEC readers over `ExtractionSite`, asking for ONE named constructor and needing its children as CLASSES (symbolic dims are parsed by class) | NOT migrated in this PR: they already have `first::<C>()` semantics (one constructor, no preference) and need class-level children the decoded term structs do not carry. Follow-up §8.8. |
| `ops/cublaslt/election.rs` | genome election by constructor NAME preferences (`genome_preferring`) | reads no layout class; unchanged |

### 4.5 Reference runtime

| SITE | NOW | NEW |
|---|---|---|
| `crates/luminal_reference/src/runtime.rs:549-570` storage sizing | `buffer.layout.mirror.literal_span_elements()` | `buffer.layout.literal_span_elements()`; message unchanged |
| `runtime.rs:767` fold check | `slot.layout != plan.buffers[id].layout` | unchanged (`PartialEq` preserved, §2.4) |
| `runtime.rs:778` operand dims | `slot.layout.mirror.literal_extents()` | `slot.layout.literal_extents()` |
| `runtime.rs:544` comment "own DecodedLayout — width alone cannot pick a variant" | text | keep ("variant" now means a typed-buffer variant, which is what it meant) |
| `layouts.rs:11-18` | doc mentions "shared mirror vocabulary" | "core's `DecodedLayout` (the elected class's decoded spellings plus the dtype fact)" |
| `kernels/**`, `ops/**` | read `operand_dims`/`dests` from `ReferenceKernelCtx`, never a layout (VERIFIED by grep) | unchanged |
| `search.rs:243-249` comment + `layout_cache` | describes `decode_layout` building a fresh Reader per call | rewrite the comment: the view indexes classes once (`EGraph::classes()`); the cache still spans candidates because the table is value-keyed per candidate |

### 4.6 Core

| SITE | NOW | NEW |
|---|---|---|
| `src/layouts.rs:1-35` module doc | five mirror structs + `MirrorLayout` + preference decoder | rewrite per §2.3/§2.4 |
| `src/layouts.rs:176-183 MirrorLayout`, :286-333 `impl MirrorLayout`, :339-345 `decode_layout`, :347-769 `Reader` (+ `ParseMemo`), :771-774 `decode_layout_for`, :999-1015 test `mirror_layout_equality_is_structural` | — | DELETE (term-decoding bodies move to the §2.3 free functions; the equality test becomes `decoded_layout_equality_is_structural` over `DecodedLayout::of`) |
| `src/lib.rs:56-61` comment on `pub mod layouts` | "Convenience mirrors … + decode_layout" | "the `Layout` sort's constructor structs, `LayoutFacts`, `DecodedLayout` and the value-keyed table; THE BUFFERIZER NEVER CALLS ANY OF THIS" |
| `src/lib.rs:25` comment | `layouts::decode_layout_table` | unchanged name |
| `src/bufferize.rs:96, 1538, 1550` comments | reference `decode_layout_table` | unchanged |
| `src/test_support.rs:1461-1469` doc | "`DecodedLayout { mirror, dtype }`" | "`DecodedLayout { class, dtype, spellings }`" |
| `src/test_support.rs:1478-1483` | `decode_layout_table(&egraph, …)` | §3.2 |
| `src/test_support.rs:3829-3862` `rm_layout`, `transpose_strided_layout` | `DecodedLayout { mirror: MirrorLayout::RightMajor(..), dtype }` literals | `DecodedLayout::of(RightMajorContiguousElementLayout { shape, width }, Some(F32))`, `DecodedLayout::of(StridedElementLayout { .. }, Some(F32))` |
| `src/test_support.rs:3983-3986` | `binding.layout.mirror.literal_extents()`; `let MirrorLayout::Strided(strided) = &binding.layout.mirror else { panic!(..) }` | `binding.layout.literal_extents()`; `let strided = binding.layout.first::<StridedElementLayout>().expect("the disclosed layout is the composed strided form");` |
| `src/extraction.rs:1756-1759` (`estimated_layout_dims`), :2375-2441 and :2458/:2480/:2507 (renderer summaries), :2846-2867 (`numeric_layout_dims/bits`), :2882-2904, :3609 (`metadata_preferred_op`), :4016-4018 (`RENDER_PREFERRED_OPS`) | spelling readers of class-INVARIANT facts (domain, width) or render-text preferences | NOT migrated: no preference-bug exposure (the domain is the same in every spelling), and the heuristic's dim estimate needs the dim CLASS for its bounds lookup (`extraction.rs:1786-1800`), which term structs do not carry. Follow-up §8.8. |

### 4.7 `tests/test_runtime/src/test_equality.rs` (5 arms + 3 signatures)

| SITE | NOW | NEW |
|---|---|---|
| :18 `use luminal::layouts::{IntExprTerm, MirrorLayout, ShapeTerm};` | — | `use luminal::layouts::{DecodedLayout, IntExprTerm};` |
| :23-58 `eval_term` | local evaluator (the complete one) | body MOVES to core as `IntExprTerm::eval_at`; keep `pub fn eval_term` as a wrapper |
| :81-143 `element_index(layout: &MirrorLayout, coords)` | five arms | `pub fn element_index(layout: &DecodedLayout, coords: &[usize]) -> Result<usize> { layout.element_index(coords) }` |
| :146 `dense_f32(backing, layout: &MirrorLayout, value_dims)` | — | `&DecodedLayout` |
| :173-174 comparison helper `(&[f32], &MirrorLayout)` | — | `(&[f32], &DecodedLayout)` |
| :192-290 tests constructing `MirrorLayout::*` | — | `DecodedLayout::of(..)` |

### 4.8 CUDA-lite tests (hand-built fixtures and readers)

| File | Sites | NEW |
|---|---|---|
| `tests/codegen_identity.rs` | :35 `reads_flat(layout: &DecodedLayout, …)` (unchanged); :67 and :234 `.mirror.literal_extents()`; :272-320 `typed(mirror)`, `rm_layout`, `strided_layout`, `offset_layout`; :670-680 comment ("`decode_layout` only states a PREFERENCE"); :700-735 the five spellings incl. `MirrorLayout::BitOffset`; :860-880 `MirrorLayout::LeftMajor`; :903-950 `DecodedLayout { mirror: rm(..), dtype }` | `.literal_extents()`; `fn typed(s: impl EgglogConstructor<Sort = Layout>) -> DecodedLayout { DecodedLayout::of(s, Some(F32)) }`; the comment becomes "all spellings of a class denote one function and a caller picks the spelling IT lowers; these tests state one function five ways and require byte-identical CUDA" — the two-spelling proof gets STRONGER: add a sixth case `DecodedLayout::of_spellings(vec![rm, strided_dense], ..)` (both present) and require the same source |
| `tests/cublaslt_contracts_cpu.rs` | :6-9 imports; :358-376 `right_major`, `left_major`; :452-495 strided/offset/symbolic literals; :219-249 fence-message pins (see 4.2) | `DecodedLayout::of(..)`; plus the #507 carry-over tests (§5) |
| `tests/cublaslt_bias_premise.rs` | :151-176 `decode_layout_for(&egraph, class, ..)` + `matches!(mirror, MirrorLayout::LeftMajor(_))` (an assertion ON THE PREFERENCE ORDER: "the decoder's most-structured spelling is LeftMajor (RightMajor is absent…)"); :317 `println!(.. dest.mirror)` | `let view = EGraphView::new(&egraph, &decoders); let c = view.class(class); assert!(c.has::<LeftMajorContiguousElementLayout>()); assert!(!c.has::<RightMajorContiguousElementLayout>(), "a [3,4] left-major over the bytes of a [4,3] right-major is not right-major"); println!("{:?}", c.present::<Layout>())`; `println!("{:?}", dest.present())`. The registry here: `luminal::egglog_snippet::decoder_registry_for(&cuda_registry().into_iter().map(|r| r.matcher).collect::<Vec<_>>())` (`RegisteredOp.matcher` is pub; `CudaRuntime::matchers()` is private, `runtime.rs:202`) — or expose `CudaRuntime::decoders(&self) -> &ConstructorRegistry` (RECOMMENDED, one accessor) |
| `tests/plan_smoke.rs:78-90` | `DecodedLayout { mirror: MirrorLayout::RightMajor(..) }` | `DecodedLayout::of(..)` |
| `tests/composed_read_families.rs:524-552` | `composed`/`rm` literals | `DecodedLayout::of(..)` |
| `tests/view_admission.rs` | :133, :154 `.mirror.literal_extents()`; :216-245 `flat_index(layout, out_coord)` five arms; :283/:323/:364 `layout.mirror.literal_extents()` | `.literal_extents()`; `flat_index` → `layout.element_index(out_coord).expect(..) as i64` (delete the local arms) |
| `tests/finalists_lattice.rs:30,39` | `DecodedLayout` as a type only | unchanged |

Tally of sites in this map: kernels.rs 10; exec.rs 5 (+1 branch-only arm); CL
layouts.rs 5; arena/device/device_call 4; reference runtime/search 6; core
layouts.rs deletions 6 blocks; core test_support 5; extraction/renderer 9 lines
listed as deliberately untouched; test_equality 8; CL tests 7 files / 27 sites;
decode_layout_table callers 8. Total touched: 84 sites across 21 files, plus
the 9 recorded-untouched lines.

---

## 5. What is deleted

* `MirrorLayout` (`src/layouts.rs:176-183`) and `impl MirrorLayout` (:286-333). Its
  four facts (`shape`, `width_bits`, `literal_extents`, `literal_span_elements`)
  become `DecodedLayout` methods (class-invariant, §2.4).
* `DecodedLayout.mirror`: replaced by `spellings: Spellings<Layout>` plus
  `class: ClassId`. Each runtime builds its plan-layout value in ONE place —
  `decode_layout_table` → `DecodedLayout::from_class` — and hand-built plans use
  `DecodedLayout::of`/`of_spellings`. Neither runtime has a runtime-flavored
  layout type (ruling D9, 2026-09-03, unchanged): `CudaPlan` and `ReferencePlan`
  stay `BufferIrGraph<luminal::layouts::DecodedLayout>`.
* `decode_layout` (:343-345), `decode_layout_for` (:771-774), `Reader` and
  `ParseMemo` (:347-769) — the term-decoding bodies move to the §2.3 functions;
  the preference loop is gone.
* The #507 arm and its rationale paragraphs (exec.rs module doc and the arm's
  comment on the branch). DECISION on #507: this PR supersedes it. #507 must not
  merge its `exec.rs` hunk. Its FOUR tests are carried into this PR, retargeted:
  * `cublaslt_bias_premise.rs` `a_degenerate_extent_d_holds_both_contiguous_spellings_in_one_class`
    — keep (it is the coincidence, measured); replace `decode_layout_for` +
    `mirror` prints with `view.class(c).present::<Layout>()` and assert
    `has::<LeftMajor>() && has::<RightMajor>()`. Structural; no class ids pinned.
  * `cublaslt_bias_premise.rs` `a_degenerate_extent_bias_election_binds_col_and_is_not_refused`
    — keep verbatim except the `dest.mirror` print → `dest.present()`.
  * `cublaslt_contracts_cpu.rs` `degenerate_extent_binds_a_bias_form_col_in_both_orientations`
    — the destination becomes
    `DecodedLayout::of_spellings(vec![RightMajor[384,1].erase(), LeftMajor[384,1].erase()], Some(F32))`
    (the class as the e-graph has it); expected COL/ld=m unchanged. ADD the
    negative the design makes true: `right_major(&[384, 1])` ALONE under a bias
    form is REFUSED (message contains "bias decorators require a LeftMajor D") —
    the fix is the class carrying both spellings, not an extent test.
  * `cublaslt_contracts_cpu.rs` `a_non_degenerate_right_major_d_still_trips_the_bias_fence`
    — keep with the §4.2 message retarget; its last assertion ("a bias-free
    degenerate D stays ROW") becomes: bias-free with `right_major(&[384,1])` alone
    → ROW (unchanged), bias-free with both spellings → COL (§4.2 rule).
  * `cublaslt_contracts.rs` (device) `degenerate_extent_bias_plan_matches_decomposed_route_tolerance_based`
    — carry verbatim (touches no layout type).
  If #507 merges first anyway, step S3 below additionally deletes the arm and
  the "ONE EXCEPTION" doc paragraph; the test retargets are the same.
* Any test asserting the preference order: `cublaslt_bias_premise.rs:170-176`
  (`matches!(mirror, MirrorLayout::LeftMajor(_))` with "the decoder's
  most-structured spelling") is the only one; it becomes the `has`/`!has` pair
  above. `codegen_identity.rs:670-680` is a comment, not an assertion.
* The `LeftMajorContiguousElementLayoutLit` premise on the bias decorators is
  NOT touched (`cublaslt_marker_decorate.egg:155-215`). It is exactly what the
  fence now asks the class.

---

## 6. Migration order (each step compile-green with its gate)

Gate vocabulary (Austin 2026-09-03: minimal gate blocks; the full ~10-minute
gate runs detached and its failures are follow-ups). Disk is tight: use
`export CARGO_TARGET_DIR=/Users/austin/Developer/luminal-logical-ssa/.claude/worktrees/rejoin-p1/target`
and check `df -h` before parallel builds.

```
G-build   cargo build -p luminal -p luminal_reference -p luminal_cuda_lite -p test_runtime --all-targets
G-device  cargo check -p luminal_cuda_lite --features device --all-targets      (type-checks device.rs:525 and device_call.rs:261)
G-fmt     cargo fmt --all -- --check
G-clippy  cargo clippy -p luminal -p luminal_reference -p luminal_cuda_lite --lib
G-core    cargo test -p luminal --lib layouts:: egglog_utils::eclass::
G-cl      cargo test -p luminal_cuda_lite --test codegen_identity --test cublaslt_contracts_cpu --test cublaslt_bias_premise --test plan_smoke --test composed_read_families --test view_admission --test finalists_lattice --test registry_selection
G-ref     cargo test -p luminal_reference --lib
G-tr      cargo test -p test_runtime --lib
G-pins    cargo test -p luminal_cuda_lite --test cublaslt_election -- --nocapture 2>&1 | grep -E 'ELECTION|elect|computes' > /tmp/after.txt   — diff against the same capture on trunk: rows must be identical (bit-identical elections); `canonical_2d_matmul_elects_the_marker` must pass
```

**S1 — core: `eclass.rs`.** Add the module and its unit tests over a
hand-built `egraph_serialize::EGraph` (three nodes: a constructor row, its
children) plus the four tripwire tests of §3.3 (these need only
`assembled_program_for(&[])`; the `Layout` sort does not exist yet, so the
tests register a throwaway `Sort`/`EgglogConstructor` pair for the hand-built
graph and use `core_decoders()` = empty until S2a — or land S1 and S2a's
`layout_decoders()` together; DECISION: land S1 with the hand-built-graph tests
only, move the tripwire tests to S2a). Gate: G-build (core only), G-core, G-fmt.

**S2a — core: the vocabulary, shimmed.** In `src/layouts.rs`: `Layout`,
`LayoutFacts`, the five `EgglogConstructor`/`LayoutFacts` impls, the term
decoders on `EClass`, `IntExprTerm::eval_at`, `layout_decoders()`;
`DecodedLayout` gains `class` and `spellings` and KEEPS `mirror` for this step,
populated by a private shim `fn mirror_of(spellings) -> MirrorLayout` that picks
the first present spelling in the OLD order (behaviour identical); add
`DecodedLayout::of/of_spellings/from_class` and the fact methods;
`decode_layout_table(view, …)`; `egglog_snippet::{core_decoders, decoder_registry_for}`;
`OpMatcher::decoders` default; `RegisteredOp::decoders`;
`luminal_reference::decoder_registry()`; the 8 callers of §3.2; every
`DecodedLayout { mirror, dtype }` literal in tests/fixtures → `DecodedLayout::of`
(§4.6-4.8 literal rows only). Tripwire unit tests of §3.3 land here. Gate:
G-build, G-core, G-cl, G-ref, G-tr, G-fmt.

**S2b — consumers off `.mirror`.** kernels.rs (§4.1), CL layouts.rs (§4.3),
arena.rs, reference runtime.rs, test_support.rs:3983-3986, test_equality.rs,
view_admission.rs, codegen_identity.rs readers, bias_premise.rs:151-176 and :317
(§4.8). Gate: G-build, G-cl, G-ref, G-tr, G-pins (first capture), G-fmt.

**S2c — delete.** `mirror` field and the shim, `MirrorLayout`, `impl
MirrorLayout`, `decode_layout`, `decode_layout_for`, `Reader`, `ParseMemo`, the
`mirror_layout_equality_is_structural` test (replaced). `grep -rn MirrorLayout
--include='*.rs' .` must return nothing. Gate: G-build, G-core, G-cl, G-ref,
G-tr, G-clippy, G-fmt.

**S3 — cuBLASLt binding.** exec.rs per §4.2 (binding, fence-call removal,
module doc, `assert_bias_destination_order` re-documentation); the #507 test
carry-over per §5 with the retargets; `cublaslt_contracts_cpu.rs` message pins
retargeted. Gate: G-build, G-device, `cargo test -p luminal_cuda_lite --test
cublaslt_contracts_cpu --test cublaslt_bias_premise`, G-pins (compare to trunk
capture), G-fmt.

**S4 — the tripwire at the four runtime sites** (§3.3), `CudaRuntime.decoders`
+ `CudaRuntime::decoders()` accessor, `BucketAssembly.decoders`,
`Finalists.decoders`; the CL registry test in `registry_selection.rs`. Gate:
G-build, G-device, G-cl, G-ref, `cargo test -p luminal_cuda_lite --test
dim_buckets` (exercises `bucketed_search_implementations`), G-fmt.

**S5 — the demonstration** (§7). Gate: `cargo run -p luminal_cuda_lite --example
eclass_decoder_demo` prints the §7 lines.

**S6 — sweep.** Doc comments listed in §4 (kernels.rs:145, exec.rs module doc,
CL layouts.rs and lib.rs:21, reference layouts.rs, search.rs:243-249 both
runtimes, src/lib.rs:56-61, test_support.rs:1461, codegen_identity.rs:670-680);
`grep -rn "preference" src/layouts.rs` returns nothing; G-clippy, G-fmt; then
the full gate detached (`cargo test --workspace`) — failures become follow-ups,
not blockers, per the 2026-09-03 ruling.

Commit per step; each commit message ends with
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` if written by an agent.

---

## 7. Demonstration plan

Austin: "show me what the call site looks like and demonstrate (don't make a
test, just show) that it fixes the bug we previously saw."

New example `crates/luminal_cuda_lite/examples/eclass_decoder_demo.rs` (CL's
examples already have `luminal_nn` and the model crates as dev-dependencies,
`crates/luminal_cuda_lite/Cargo.toml:37-48`). It re-uses the fixture and helpers
from `tests/cublaslt_bias_premise.rs` (`degenerate_linear_with_bias` on the #507
branch: `x[1,K] @ w[K,N] + b[N]` with `K=8, N=3` through `luminal_nn::linear`;
`d_layout_class`, `report_forms`, the seed sweep `0..6` at 12×16/mutations 4).
Copy them into the example (examples cannot import test files).

Steps and REQUIRED printed lines:

1. `let (cx, x, w, b, _) = degenerate_linear_with_bias(); let rt = CudaRuntime::load(&cx)?; let egraph = rt.saturated_egraph()?;`
   `let decoders = rt.decoders(); let view = EGraphView::new(&egraph, decoders);`
2. For every `LayoutTensorOpCublasLtBias` node's D layout class `c = view.class(&d_layout_class(..))`, print:
   ```
   DEMO D layout class: present = ["RightMajorContiguousElementLayoutLit", "LeftMajorContiguousElementLayoutLit", "StridedElementLayoutLit", "BitOffsetExpressionLayoutLit"]   (order = registry order; the set must contain BOTH contiguous names)
   DEMO first::<LeftMajorContiguousElementLayout>()  = Some(LeftMajorContiguousElementLayout { shape: ShapeTerm([Lit(3), Lit(1)]), width: BitWidthTerm(32) })
   DEMO first::<RightMajorContiguousElementLayout>() = Some(RightMajorContiguousElementLayout { shape: ShapeTerm([Lit(3), Lit(1)]), width: BitWidthTerm(32) })
   DEMO retired preference order would have answered: RightMajorContiguousElementLayoutLit
   ```
   The last line is computed by re-enacting the deleted list —
   `["RightMajor…","LeftMajor…","Strided…","ElementOffset…","BitOffset…"].iter().find(|n| present.contains(n))`
   — so the contrast is shown without keeping the old code alive.
3. Run the seed sweep; on the first plan whose computes contain a
   `CublasLt*Bias` node, `plan_call(functional)` then
   `bind_destination(&mut call, dest, "demo")` and print:
   ```
   DEMO call frame m=3 n=1 k=8 ; destination present = [...]
   DEMO bind_destination -> Ok ; d = LtDesc { rows: 3, cols: 1, ld: 3, order: Col } ; c == d
   DEMO the fence, spelled at the call site: dest.require::<LeftMajorContiguousElementLayout>("demo") = Ok(..)
   ```
4. Print the negative for contrast, on a hand-built destination holding only
   the right-major spelling:
   ```
   DEMO right_major([3,1]) alone under the bias form: Err(cuBLASLt demo: unreachable: the bias decorators require a LeftMajor D; a bias form (LayoutTensorOpCublasLtBias) reached the executor whose destination class holds ["RightMajorContiguousElementLayoutLit"] and no LeftMajorContiguousElementLayoutLit — …)
   ```
5. Exit 0. No assertions; the reader compares lines 2-4 against the whisper
   refusal quoted in §1. Note in the example's header that whisper's real frame
   is `m=384, n=1`; the pin-scale frame is `m=3, n=1` — same coincidence.

---

## 8. Risks and open questions, with recommendations

**8.1 `Sort` collides with `egglog::sort::Sort`.** Austin fixed the name.
`eclass.rs` refers to egglog's trait only through `ArcSort::name()` (a method call
on `Arc<dyn Sort>`; no import needed), so the collision is a reading hazard, not a
compile one. RECOMMEND: keep the name; in the two files that mention both
(`eclass.rs`, `egglog_snippet.rs`) refer to egglog's as `egglog::sort::Sort` in
comments and never `use` it.

**8.2 Object safety and the erase step.** `LayoutFacts` is object-safe (all
methods `&self`, no generics; the generic `first::<C>()` lives on
`Spellings`/`EClass`). The one non-obvious constraint: converting a concrete `C`
into `Arc<<C::Sort as Sort>::Facts>` is an unsizing coercion the compiler only
performs where the concrete type AND the target `dyn` type are both known, so it
cannot be written once generically over `S::Facts`; hence
`EgglogConstructor::erase(self)` with a one-line `Arc::new(self)` per struct
(five lines total). The same constraint applies to the `dyn Any` upcast in the
other direction, hence `Sort::upcast_any` — one line per sort (§2.5). `DynFacts`
is blanket-implemented, so `constructor()` and `dyn_eq()` cost the structs
nothing. `dyn-clone` is in `Cargo.lock`
(line 506) but is not needed: `Arc` gives `Clone` for free, which is why the
items are `Arc<S::Facts>` and not `Box`.

**8.3 `unknown::<S>()` semantics.** A `Layout` class legitimately contains
non-constructor nodes: rows of `Custom` functions whose output sort is `Layout`
(`layout-of`, `egglog_preamble.egg:451`) and, after truncation, the serializer's
`[...]` placeholder (`serialize.rs:331`). `unknown()` = ops in the class with no
decoder registered under `S` — so it lists those rows too. RECOMMEND: keep it
diagnostics-only (it is printed in `from_class`'s zero-spellings refusal and
nowhere else). If a consumer ever needs "constructor I have no decoder for" as a
hard signal, extend `check()` to record `Custom` function names per sort and
subtract them — deferred; the tripwire already guarantees no undecodable
constructor exists in a checked program.

**8.4 Decode failures are skipped, recorded, and only fatal in aggregate.** A
constructor node whose fields do not parse (foreign-shape `CoordVar` under the
owner-shape guard, a cons spine with no parsing spelling, a cycle) is skipped
exactly as the old Reader skipped it (`continue` at `src/layouts.rs:440,454,468-477`),
but now lands in `Spellings::failed` with a reason. `from_class` refuses when
NOTHING decoded, printing `present` and `failed`. `has::<C>()` means decodable;
`present()` means named. The fence therefore refuses a class that holds an
undecodable LeftMajor spelling — correct: a spelling we cannot read is not one we
can bind on.

**8.5 The cross-spelling domain check may fire on a benign class.** `from_class`
refuses when decoded spellings disagree on `shape()` or `width()`. Argument that
it cannot fire benignly: every spelling's shape is a `ShapeLit` e-class; decoded
terms differ only if the classes differ; two different shape classes in one
layout class is the false cross-rank/cross-domain union the preamble comment at
:217 names as a bug. Counter-example attempted: a symbolic extent spelled two
ways (`s` vs `s+0`) — these are unioned by the ring rewrites into one class, so
`shape_term` reads the same class. Not refuted, but the argument rests on
saturation reaching the union. RECOMMEND: implement as a `Result` refusal (a
refusal rejects the genome; if every genome hits it, the search dies loudly with
the class printed — the right outcome for a false union). If it fires on any of
the seven minis in the detached full gate, downgrade to `eprintln!` and file the
false union as its own bug.

**8.6 Equality of `DecodedLayout` is now set equality.** Used by the reference
executor's fold check (`runtime.rs:767`) and the transport pin
(`test_support.rs:1497`). Same-class values decode to identical sets; different
classes differ in at least one term (hash-consing). Refutation attempt: two
right-major `[2,3] f32` values in different layout classes — impossible, the
constructor node `(RightMajorContiguousElementLayoutLit (ShapeLit …) (BitWidthLit 32))`
with equal children is one e-node. Not refuted. `class` participates in the
derived `PartialEq`; hand-built fixtures all use the same sentinel, so fixtures
compare as before.

**8.7 Uniform lowering via `LayoutFacts::offset_term()` (follow-up).** Adding
`fn offset_term(&self) -> IntExprTerm` (the element-offset function as a term;
`BitOffset` yields `TruncDiv(offset, Lit(width))`) would let `read_affine` and
`layout_read_index` lower ANY spelling with zero constructor dispatch — the
literal form of "never depend on spelling". It changes the emitted C for the
contiguous and strided forms (parenthesization, stride spelling), so the five
string pins in `codegen_identity.rs` would need re-blessing. NOT in this PR
(§4.1). Austin decides.

**8.8 Spelling readers deliberately left in place.** (a) the extractor's
heuristic and renderer (`extraction.rs`, §4.6) read the domain — class-invariant,
so the preference bug cannot occur — and need dim CLASSES for the bounds-midpoint
estimate; (b) `cublaslt/mod.rs`'s spec readers ask for one named constructor via
`ExtractionSite` and need entry classes for symbolic dims. Both could move onto
the new API once `EClass` grows a node-level typed query
(`first_node::<C>() -> Option<ENode>`, so children stay classes) and once the
registry records each constructor's INPUT sorts from `FuncType.input` at
`check()` time (then "the Shape child" is the child whose declared sort is
`Shape`, with no per-constructor index table). Both are follow-ups; neither is
touched by the election pins.

**8.9 Behaviour change to flag:** a bias-FREE cuBLASLt destination whose class
holds both contiguous spellings (degenerate extents only) binds COL where it
bound ROW. Same elements, same order, same reach (`validate_against` passes both
per #507's arithmetic). Device-visible only as the descriptor's order attribute.
Austin should see this line.

**8.10 Class ids in the plan.** `DecodedLayout.class` is a serialized-graph id:
random every run by ruling (2026-09-02). It is never compared in a test and never
printed in a pin; it appears only in refusal messages and as the cache key it
already was. Permutation-invariance is preserved.

**8.11 Performance** is explicitly not a concern (Austin). For the record the new
path is cheaper: the old `Reader::new` indexed every node of the serialized graph
per decode (`src/layouts.rs:357-370`); `EGraphView` uses the graph's own
`classes()` index, built once behind a `OnceCell`.

**8.12 #507 sequencing.** Recommend closing #507 in favour of this PR (its
e-graph test is the valuable part and is carried here). If it merges first, S3
also deletes the arm; nothing else changes.

---

## Appendix A — Verified-facts ledger (file:line, trunk 1f101d62 unless noted)

* Decoder preference loop: `src/layouts.rs:427-535`; class index per decode
  :357-370; spelling order :380-398; literal parsing :395-418; `MirrorLayout`
  :176-183 and :286-333; `DecodedLayout` :796-800; cache :806; table :830-873.
* `MirrorLayout` uses outside `src/layouts.rs`: `src/test_support.rs` (3834-3855, 3983-3985),
  `crates/luminal_cuda_lite/src/{layouts.rs:69, kernels.rs:145,366,517,540-584, ops/cublaslt/exec.rs:324}`,
  CL tests (`view_admission.rs:217`, `plan_smoke.rs:78-82`, `codegen_identity.rs:274-315,561,727,863-908`,
  `cublaslt_contracts_cpu.rs:8,360-491`, `composed_read_families.rs:524-548`, `cublaslt_bias_premise.rs:173`),
  `tests/test_runtime/src/test_equality.rs:18,81-282`.
* `decode_layout_table` callers: `src/test_support.rs:1478`,
  `crates/luminal_reference/src/{harness.rs:209, runtime.rs:1684, search.rs:355,426}`,
  `crates/luminal_cuda_lite/src/{finalists.rs:245, search.rs:503,657}`.
* `.mirror` field reads: `arena.rs:82`, `kernels.rs:64,368,371,539`, CL `layouts.rs:71,83,126`,
  `exec.rs:325,344`, reference `runtime.rs:553,778`, `test_support.rs:3984-3985`,
  `codegen_identity.rs:67,234`, `view_admission.rs:133,154,218,283,323,364`, `cublaslt_bias_premise.rs:317`.
* Bias decorator premise: `cublaslt_marker_decorate.egg:182` (and the second decorator's copy at :244).
* Contiguous discovery unions: `egglog_preamble.egg:3530-3560`; `layout-of` function :451; Layout datatype :203-241.
* `bind_destination` callers: `device.rs:525`, tests; `assert_bias_destination_order` callers: `exec.rs:366`, `device_call.rs:261`.
* Saturation→serialize sites: CL `runtime.rs:432-456`, `search.rs:931-934`; reference `runtime.rs:332-365`,
  `search.rs:536-538`, `harness.rs:82-84, 186-190`, `runtime.rs:1672-1674`, `search.rs:737-739`;
  core `extraction.rs:4309-4311`; `test_runtime/src/lib.rs:144-146`.
* `OpMatcher` trait: `src/layout_ir/mod.rs:518-547`; `RegisteredOp`: `crates/luminal_cuda_lite/src/ops/mod.rs:74-77`;
  `ReferenceOp`: `crates/luminal_reference/src/ops/mod.rs:144-150`; `assembled_program_for`: `src/egglog_snippet.rs:142-152`.
* `PlanLayout: Clone + Debug`: `src/bufferize.rs:99-100`. `rust-version = "1.91"`: `Cargo.toml:5`
  (raised from "1.85" on 2026-09-05 with the §2.5 amendment; let-chains already needed 1.88).
* egglog fork rev 1bb30831 (`Cargo.toml:30-31`): `functions_iter` `src/lib.rs:1236`, `get_function` :2318,
  `Function::func_type` :342, `type_info(&mut self)` :589, `pub use typechecking::FuncType` :74,
  `FuncType` fields `src/typechecking.rs:115-120`, `FunctionSubtype` `src/ast/mod.rs:1298-1302`,
  `Sort::name` `src/sort/mod.rs:52`, serializer op/children :204-224, `typ` :393, `[...]` :331,
  `SerializeConfig::default` = no truncation :59-63.
* egraph-serialize 0.3.0: `EGraph` :65-74, `classes()` :107-120, `Node` :168-177, `Class` :190-193, `ClassData` :197-203, `ClassId(Arc<str>)` :21.
* Codegen string pins: `codegen_identity.rs:405,440,464,491,523`; message pins :573,603,939,953,1012; cuBLASLt message pins `cublaslt_contracts_cpu.rs:225-234,464-485,498`.
* PR #507: branch `fix/cublaslt-degenerate-d-order` @ c3b4c428, files `exec.rs` (+61), `cublaslt_bias_premise.rs` (+189), `cublaslt_contracts.rs` (+119), `cublaslt_contracts_cpu.rs` (+129); OPEN, not merged.
