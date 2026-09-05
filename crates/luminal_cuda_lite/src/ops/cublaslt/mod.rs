//! cuBLASLt MARKER prototype (round-11 state) — four contract
//! constructors, slim spelling-independent descriptor terms, UNSWAPPED
//! descriptor roles behind ONE canonical logical form.
//!
//! ROUND 11 (Austin, 2026-08-26): logical rules never carry layout letters
//! (N/T describe LAYOUTS; `CublasLtOperationN`/`T` on the descriptor
//! stratum keep them — that is exactly the distinction). Three
//! canonicalization rewrites fold every recorded matmul spelling into the
//! canonical chain A[m,k], B[k,n] -> out[m,n] (non-canonically-stored
//! operands become rank-2 transpose VIEWS); ONE site rule matches that
//! form; ONE transpose-sandwich rewrite mints the sibling; the
//! double-transpose collapse anchors termination. Layout variety is then
//! matched entirely by the descriptor arms, per operand, off composed
//! layouts within the single logical form.
//!
//! One matcher family over four fixed-arity op constructors (base / Bias /
//! Accumulate / AccumulateBias), each with a Lit arity that is a CONSTANT
//! of its name; one [`LtMatmulSpec`] endpoint for every path (R2).
//!
//! PARSER DISCIPLINE (the per-enode contract): spec data is read ONLY from
//! the matched enode's own children — never from class-mates (a class can
//! hold several op spellings over one shared Lit: dual spellings, weld
//! corners) and never from Lit position (the Lit lives in the CLASS). The
//! C/bias payloads are direct LayoutTensor children of their constructors
//! for exactly this reason.
//!
//! SPELLING INDEPENDENCE (house doctrine): descriptor terms carry pointers
//! and finite classifications only — no numerics. All API numerics are
//! computed here by walks: rows/cols from the logical shapes (class-level
//! `shape-of` facts), ld from the elected enode's own layout tensor.
//!
//! Convention R10 (unswapped-COL, Austin 2026-08-25): descriptor A is the
//! SITE's a operand's COL view, descriptor B is the site's b, D is the
//! claimed output's COL view; call-m/n/k are the site's OWN m/n/k. The
//! round-9 A/B role swap is gone — a recorder right-major-out matmul is
//! served by the SIBLING site the standalone rewrite mints
//! ((B^T)(A^T))^T — see `egg/cublaslt_marker_rewrite.egg`), whose own
//! frame has call-m = the recorder matmul's n, so the emitted call is
//! numerically identical to round 9's.
//!
//! VIEW ADMISSION (round 10): a descriptor's layout tensor may be a VIEW —
//! a LayoutTensorLit whose layout is a composed chain over another layout
//! tensor's bytes, with no BufferTensor of its own. The spec resolves each
//! descriptor's BUFFER by walking view -> parent (through the composition
//! tie: view layout == parent layout's bit expression substituted through
//! the apply's map) until a BufferTensorLit is found; ld comes from the
//! composed chain itself (`leading_dimension` reads either chain
//! orientation). Split sources + Ruling 1: m/n/k AND every ld may be
//! symbolic (bound at execute time from the dyn map); static pitches are
//! read from the layout.
//!
//! CONDITIONAL SOUNDNESS — THE STANDING CAVEAT (recorded 2026-09-04,
//! when the markers became part of the default registry). Electing a
//! marker replaces a decomposed multiply/reduce chain with
//! `cublasLtMatmul`, and the two are NOT bit-identical. cuBLASLt picks
//! its own REDUCTION ORDER (split-k, tile shape and the serialization
//! of partial sums are per-algo and per-shape choices the heuristic
//! makes for us), and it CONTRACTS multiply-add pairs into FMA where
//! our emitted NVRTC kernels need not. So the matmul rewrites this
//! estate mints are equalities only up to float reassociation and
//! contraction: sound for a real-arithmetic reading of the graph,
//! CONDITIONALLY sound for the machine one.
//!
//! The e-graph has no approximate-equality stratum today — every union
//! it holds claims exactness — so nothing here distinguishes the two.
//! Building that stratum (an explicit approximate-rewrite pass carrying
//! its own tolerance account, so an approximate union can never be
//! mistaken for an exact one) is DEFERRED by ruling 2026-09-04. Until
//! it lands, every marker election carries this caveat, and the
//! device-side check is the TOLERANCE comparison in
//! `tests/cublaslt_contracts.rs` — never a bit-for-bit one.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

// ---------------------------------------------------------------------------
// Train-3 wiring (the op-ownership move): the extractor above is the
// rehomed test_runtime `cublaslt_marker.rs`, semantics-identical; these
// siblings are the estate's other rehomed/new halves.
// ---------------------------------------------------------------------------
/// The cudarc result-layer dispatch (device feature only).
#[cfg(feature = "device")]
pub mod device_call;
/// The round-11 election core, rehomed from the test_runtime lib
/// (vocabulary now a parameter; `test_runtime` wraps with its own).
pub mod election;
/// CPU-side host-call planning + the executor-owned validation the
/// library does not provide (ld bounds, descriptor construction).
pub mod exec;

/// HOST-CALL DISPATCHABLE: the allow-list face of the cuBLASLt estate.
/// Kernel-bearing ops are claimable because a codegen row exists;
/// plan-transparent ops because the planner folds them; these are
/// claimable because the executor dispatches them as a HOST LIBRARY
/// CALL (`cublasLtMatmul`). Derived from the registered prototype's
/// concrete type, never from a name list.
pub fn host_dispatchable(op: &dyn LayoutIrOp) -> bool {
    op.as_any().downcast_ref::<CublasLt>().is_some()
        || op.as_any().downcast_ref::<CublasLtDps>().is_some()
}

type ClassId = luminal::prelude::egraph_serialize::ClassId;

// ---------------------------------------------------------------------------
// The four runtime contracts
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasLtForm {
    Base,
    Bias,
    Accumulate,
    AccumulateBias,
}

impl CublasLtForm {
    pub const ALL: [CublasLtForm; 4] = [
        CublasLtForm::Base,
        CublasLtForm::Bias,
        CublasLtForm::Accumulate,
        CublasLtForm::AccumulateBias,
    ];

    pub fn constructor_name(self) -> &'static str {
        match self {
            CublasLtForm::Base => "LayoutTensorOpCublasLt",
            CublasLtForm::Bias => "LayoutTensorOpCublasLtBias",
            CublasLtForm::Accumulate => "LayoutTensorOpCublasLtAccumulate",
            CublasLtForm::AccumulateBias => "LayoutTensorOpCublasLtAccumulateBias",
        }
    }

    /// Lit input arity — a constant of the constructor name.
    pub fn lit_arity(self) -> usize {
        match self {
            CublasLtForm::Base => 2,
            CublasLtForm::Bias | CublasLtForm::Accumulate => 3,
            CublasLtForm::AccumulateBias => 4,
        }
    }

    pub fn has_c(self) -> bool {
        matches!(
            self,
            CublasLtForm::Accumulate | CublasLtForm::AccumulateBias
        )
    }

    pub fn has_bias(self) -> bool {
        matches!(self, CublasLtForm::Bias | CublasLtForm::AccumulateBias)
    }

    /// Child indices: (c_lt, bias_lt, epilogue).
    fn slots(self) -> (Option<usize>, Option<usize>, usize) {
        match self {
            CublasLtForm::Base => (None, None, 4),
            CublasLtForm::Bias => (None, Some(4), 5),
            CublasLtForm::Accumulate => (Some(4), None, 5),
            CublasLtForm::AccumulateBias => (Some(4), Some(5), 6),
        }
    }

    /// Fixed operand-name map — the runtime contract in slot names.
    pub fn operand_names(self) -> &'static [&'static str] {
        match self {
            CublasLtForm::Base => &["a", "b"],
            CublasLtForm::Bias => &["a", "b", "bias"],
            CublasLtForm::Accumulate => &["a", "b", "c"],
            CublasLtForm::AccumulateBias => &["a", "b", "c", "bias"],
        }
    }
}

// ---------------------------------------------------------------------------
// Spec structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CuEpilogue {
    Default,
    Relu,
    Bias,
    ReluBias,
}

/// A geometry value: a literal the whole class equals, or the symbolic
/// IntExpr class handle the executor binds from the dyn map at call time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuDim {
    Literal(i64),
    Symbolic(ClassId),
}

impl CuDim {
    pub fn literal(&self) -> Option<i64> {
        match self {
            CuDim::Literal(v) => Some(*v),
            CuDim::Symbolic(_) => None,
        }
    }
}

impl PartialEq<i64> for CuDim {
    fn eq(&self, other: &i64) -> bool {
        matches!(self, CuDim::Literal(v) if v == other)
    }
}

impl std::fmt::Display for CuDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CuDim::Literal(v) => write!(f, "{v}"),
            CuDim::Symbolic(class) => write!(f, "sym({class:?})"),
        }
    }
}

/// The single Rust endpoint: one struct, every contract and decoration.
#[derive(Debug, Clone, PartialEq)]
pub struct LtMatmulSpec {
    pub form: CublasLtForm,
    pub m: CuDim,
    pub n: CuDim,
    pub k: CuDim,
    pub trans_a: bool,
    pub trans_b: bool,
    /// lds are CuDim like m/n/k (RULING 1): contiguous forms may carry a
    /// symbolic storage extent — the executor binds it at call time.
    pub lda: CuDim,
    pub ldb: CuDim,
    pub ldc: CuDim,
    pub ldd: CuDim,
    pub order_col: bool,
    pub has_c: bool,
    pub has_bias: bool,
    pub epilogue: CuEpilogue,
    // ---- IDENTITY, in TWO namespaces (round-8 E4 naming discipline) ----
    // ROUND 10: the namespaces now COINCIDE — descriptor A carries the
    // site's a, descriptor B the site's b (Austin's unswap ruling). The
    // two prefixes are kept anyway: `logical_*` reads the SITE TRIPLE,
    // `desc_*` reads the DESCRIPTOR TERM, and the round-2 battery pins
    // that they agree —
    //   logical(desc_a_layout_tensor) == logical_a
    //   logical(desc_b_layout_tensor) == logical_b
    // (For a sibling site minted by the transpose-sandwich rewrite, the
    // site's a IS the recorder matmul's b — the swap lives in the LOGICAL
    // rewrite now, not in the descriptor wiring.)
    /// Logical `a` (the site's first operand) = API descriptor **A**.
    pub logical_a: ClassId,
    /// Logical `b` (the site's second operand) = API descriptor **B**.
    pub logical_b: ClassId,
    /// Logical value of the CLAIMED output (the D descriptor's layout
    /// tensor) — the D the executor binds.
    pub logical_out: ClassId,
    /// The site triple's out — identity only; never bind buffers from it.
    pub logical_site_out: ClassId,
    /// API descriptor **A**'s layout tensor — carries the site's *a*.
    pub desc_a_layout_tensor: ClassId,
    /// API descriptor **B**'s layout tensor — carries the site's *b*.
    pub desc_b_layout_tensor: ClassId,
    pub c_tensor: Option<ClassId>,
    pub bias_tensor: Option<ClassId>,
    // ---- VIEW ADMISSION (round 10): resolved buffer identities ----
    // Each descriptor's layout tensor may be a VIEW with no BufferTensor
    // of its own; these are the buffers the view walk grounds them in
    // (None = unresolved, e.g. a fresh intermediate the planner allocs).
    pub desc_a_buffer: Option<ClassId>,
    pub desc_b_buffer: Option<ClassId>,
    pub d_buffer: Option<ClassId>,
}

impl LtMatmulSpec {
    /// Lit-arity contract — a constant of the constructor name.
    pub fn expected_lit_inputs(&self) -> usize {
        self.form.lit_arity()
    }

    pub fn mnk_lits(&self) -> (i64, i64, i64) {
        (
            self.m.literal().expect("m is symbolic"),
            self.n.literal().expect("n is symbolic"),
            self.k.literal().expect("k is symbolic"),
        )
    }

    pub fn input_name(&self, operand: usize) -> Option<&'static str> {
        self.form.operand_names().get(operand).copied()
    }

    fn validate(&self, d_rows: &CuDim, d_cols: &CuDim) {
        assert!(
            self.order_col,
            "cuBLASLt marker spec: non-COL descriptor minted"
        );
        assert!(
            *d_rows == self.m && *d_cols == self.n,
            "cuBLASLt marker spec inconsistent: D is {d_rows}x{d_cols}, call wants {}x{}",
            self.m,
            self.n
        );
        for (name, dim) in [("m", &self.m), ("n", &self.n), ("k", &self.k)] {
            if let Some(v) = dim.literal() {
                assert!(
                    v >= 1,
                    "cuBLASLt marker spec inconsistent: empty geometry {name}={v}"
                );
            }
        }
        assert_eq!(self.has_c, self.form.has_c());
        assert_eq!(self.has_bias, self.form.has_bias());
    }
}

// ---------------------------------------------------------------------------
// Term walking (enode-anchored — see module doc)
// ---------------------------------------------------------------------------

/// ROUND 8b (E1 LANDED): no layout FORM tag. Orientation comes from the
/// descriptor's OPERATION child (rule-proven from the index map), which
/// fixes the column-major view; ld is the row count of that view, unless
/// the layout is PADDED, in which case it is the pitch.
fn parse_i64_expr(site: &ExtractionSite<'_>, class: &ClassId) -> Option<i64> {
    for lit in site.nodes_in_class_value(class, "IntLit") {
        if let Some(value) = site
            .class_of_child(lit, 0)
            .and_then(|c| site.node_in_class_parse_i64(&c))
        {
            return Some(value);
        }
    }
    None
}

/// Class-invariant read: a literal the whole class equals, else symbolic.
fn parse_dim(site: &ExtractionSite<'_>, class: &ClassId) -> CuDim {
    match parse_i64_expr(site, class) {
        Some(v) => CuDim::Literal(v),
        None => CuDim::Symbolic(class.clone()),
    }
}

fn parse_operation(site: &ExtractionSite<'_>, class: &ClassId) -> Option<bool> {
    if site
        .nodes_in_class_value(class, "CublasLtOperationN")
        .next()
        .is_some()
    {
        return Some(false);
    }
    if site
        .nodes_in_class_value(class, "CublasLtOperationT")
        .next()
        .is_some()
    {
        return Some(true);
    }
    None
}

/// One operand reading: (layout tensor, FORM, operation), site
/// backpointer verified. All numerics are computed later by walks.
fn parse_operand_descriptor(
    site: &ExtractionSite<'_>,
    class: &ClassId,
    role_constructor: &str,
    site_class: &ClassId,
) -> Option<(ClassId, bool)> {
    for node in site.nodes_in_class_value(class, role_constructor) {
        let back = site.class_of_child(node, 0)?;
        assert_eq!(
            &back, site_class,
            "cuBLASLt marker spec inconsistent: {role_constructor} points at a different site"
        );
        let lt = site.class_of_child(node, 1)?;
        // The OPERATION is the orientation carrier: the rules prove it
        // from the index map, which the layout cannot supply.
        if let Some(operation) = site
            .class_of_child(node, 2)
            .and_then(|c| parse_operation(site, &c))
        {
            return Some((lt, operation));
        }
    }
    None
}

fn layout_class_of(site: &ExtractionSite<'_>, lt_class: &ClassId) -> Option<ClassId> {
    for node in site.nodes_in_class_value(lt_class, "LayoutTensorLit") {
        if let Some(layout) = site.class_of_child(node, 1) {
            return Some(layout);
        }
    }
    None
}

fn logical_class_of(site: &ExtractionSite<'_>, lt_class: &ClassId) -> Option<ClassId> {
    for node in site.nodes_in_class_value(lt_class, "LayoutTensorLit") {
        if let Some(logical) = site.class_of_child(node, 0) {
            return Some(logical);
        }
    }
    None
}

/// The logical tensor's rank-2 storage extents, via its class-level
/// `shape-of` fact. CANONICAL CHOICE (doctrine point 3): the first
/// ShapeLit spelling is taken; in extent-1 weld corners a shape class can
/// hold role-permuted spellings, and every choice describes the same
/// bytes under the descriptor's form/operation, so the choice is sound.
fn storage_dims(
    site: &ExtractionSite<'_>,
    logical_class: &ClassId,
) -> Option<Vec<(CuDim, ClassId)>> {
    let shape_class = site.egraph.nodes.values().find_map(|node| {
        if node.op != "shape-of" {
            return None;
        }
        let child = node.children.first()?;
        (&site.egraph.nodes.get(child)?.eclass == logical_class).then(|| node.eclass.clone())
    })?;
    let shape_lit = site.nodes_in_class_value(&shape_class, "ShapeLit").next()?;
    let mut list_class = site.class_of_child(shape_lit, 0)?;
    let mut dims = Vec::new();
    loop {
        if site
            .nodes_in_class_value(&list_class, "IntExprNil")
            .next()
            .is_some()
        {
            break;
        }
        let cons = site
            .nodes_in_class_value(&list_class, "IntExprCons")
            .next()?;
        let head = site.class_of_child(cons, 0)?;
        dims.push((parse_dim(site, &head), head));
        list_class = site.class_of_child(cons, 1)?;
    }
    Some(dims)
}

/// The rank-2 stride chain of a strided spelling in this layout class,
/// as entry classes in chain order [outer, inner]. `None` when the class
/// carries no strided spelling (which the E1 hypothesis says cannot
/// happen for layouts we read, and which is rejected loudly by callers).
fn stride_chain(site: &ExtractionSite<'_>, layout_class: &ClassId) -> Option<Vec<ClassId>> {
    for layout in site.nodes_in_class_value(layout_class, "StridedElementLayoutLit") {
        let Some(mut cur) = site.class_of_child(layout, 1) else {
            continue;
        };
        let mut entries = Vec::new();
        loop {
            if site
                .nodes_in_class_value(&cur, "IntAffineExprNil")
                .next()
                .is_some()
            {
                break;
            }
            let Some(cons) = site.nodes_in_class_value(&cur, "IntAffineExprCons").next() else {
                break;
            };
            let Some(entry) = site.class_of_child(cons, 0) else {
                break;
            };
            entries.push(entry);
            let Some(tail) = site.class_of_child(cons, 1) else {
                break;
            };
            cur = tail;
            if entries.len() > 8 {
                break;
            }
        }
        if entries.len() == 2 {
            return Some(entries);
        }
    }
    None
}

/// The discriminated pitch factor CLASS carried by a stride entry: the
/// co-factor of an (IntMul coord pitch) spelling whose OTHER child is a
/// CoordVar. Ambiguity (two factors with disagreeing literal values) is a
/// LOUD REJECTION — measured clean across the estate (distinct pitches
/// live in distinct layout CLASSES, never as rival spellings of one).
fn entry_pitch_class(site: &ExtractionSite<'_>, entry: &ClassId) -> Option<ClassId> {
    let mut factors: Vec<ClassId> = Vec::new();
    for mul in site.nodes_in_class_value(entry, "IntMul") {
        for child in 0..2usize {
            let other = 1 - child;
            let other_is_coord = site
                .class_of_child(mul, other)
                .map(|c| site.nodes_in_class_value(&c, "CoordVar").next().is_some())
                .unwrap_or(false);
            if !other_is_coord {
                continue;
            }
            if let Some(factor) = site.class_of_child(mul, child)
                && !factors.contains(&factor)
            {
                factors.push(factor);
            }
        }
    }
    match factors.as_slice() {
        [] => None,
        [only] => Some(only.clone()),
        many => {
            let lits: std::collections::BTreeSet<i64> = many
                .iter()
                .filter_map(|c| parse_i64_expr(site, c))
                .collect();
            if lits.len() > 1 {
                panic!(
                    "cuBLASLt marker: stride entry has discriminated pitch factors with \
                     disagreeing values {lits:?} — the walk would be spelling-dependent"
                );
            }
            Some(many[0].clone())
        }
    }
}

/// Does this stride entry's class carry a bare-CoordVar spelling (the
/// x*1-subsumed unit stride)? A membership test — binds nothing.
fn entry_is_bare_coord(site: &ExtractionSite<'_>, entry: &ClassId) -> bool {
    site.nodes_in_class_value(entry, "CoordVar")
        .next()
        .is_some()
}

/// The leading dimension of a descriptor, read from the elected enode's
/// own layout tensor (round-8b E1 — no form tag).
///
/// ld = `rows_prime` (the row count the OPERATION fixes) unless the
/// layout is PADDED, in which case ld is the pitch.
///
/// ROUND 10 generalization: the chain may present EITHER orientation —
/// row-major-form (unit stride on the inner axis, pitch on the outer
/// entry: stored operands, padded creator layouts) or col-major-form
/// (unit stride on the outer axis, pitch on the inner entry: the
/// transpose-VIEW layouts the sandwich rewrite routes through descriptor
/// D). The unit axis is found by the bare-CoordVar membership test; the
/// OTHER entry is the pitch source; the contiguous expectation is the
/// UNIT axis's extent class (for a contiguous layout the non-unit stride
/// IS that extent, and then ld == rows_prime by construction).
///
/// PADDED IS A CLASS COMPARISON, not an ordering: a contiguous layout's
/// stride factor is literally the neighbouring extent's e-class; a padded
/// layout carries a DIFFERENT class (the creator's chosen pitch). Class
/// equality decides this for symbolic extents too.
///
/// DEAD AXES (doctrine point 3): an extent-1 axis has its coordinate
/// welded to the zero class, so its stride entry absorbs unrelated
/// spellings and its value is a DON'T-CARE (it multiplies zero). A pitch
/// is never read off a dead axis; ld falls through to rows_prime, which
/// the OPERATION fixes correctly per reading (the gemv pair: op=N/ld=4,
/// op=T/ld=1).
fn leading_dimension(
    site: &ExtractionSite<'_>,
    lt_class: &ClassId,
    rows_prime: &CuDim,
    storage: &[(CuDim, ClassId)],
) -> CuDim {
    let Some(layout_class) = layout_class_of(site, lt_class) else {
        return rows_prime.clone();
    };
    let Some(chain) = stride_chain(site, &layout_class) else {
        return rows_prime.clone();
    };
    if chain.len() != 2 || storage.len() != 2 {
        return rows_prime.clone();
    }
    let dead = |i: usize| storage[i].0 == 1i64;
    // chain[0] <-> axis 1 (outer, extent storage[0]);
    // chain[1] <-> axis 0 (inner, extent storage[1]).
    // (pitch slot, unit slot): row-major-form has the unit on the inner
    // axis; col-major-form on the outer. Prefer the orientation whose
    // unit axis is alive and bare.
    let orientation = if !dead(1) && entry_is_bare_coord(site, &chain[1]) {
        Some((0usize, 1usize)) // row-major-form: pitch on the outer entry
    } else if !dead(0) && entry_is_bare_coord(site, &chain[0]) {
        Some((1usize, 0usize)) // col-major-form: pitch on the inner entry
    } else {
        None
    };
    if let Some((pitch_slot, unit_slot)) = orientation
        && !dead(pitch_slot)
        && let Some(factor) = entry_pitch_class(site, &chain[pitch_slot])
    {
        let dim = parse_dim(site, &factor);
        // A unit factor is a unit stride, never a pitch; a factor
        // equal to the unit axis's extent class is contiguous.
        let expectation = &storage[unit_slot].1;
        let padded = dim != 1i64 && factor != *expectation;
        if padded {
            return dim;
        }
    }
    rows_prime.clone()
}

/// The direct buffer of a layout tensor, if a BufferTensorLit names it.
fn direct_buffer_of(site: &ExtractionSite<'_>, lt_class: &ClassId) -> Option<ClassId> {
    for node in site.egraph.nodes.values() {
        if node.op != "BufferTensorLit" {
            continue;
        }
        let Some(lt) = node
            .children
            .first()
            .and_then(|id| site.egraph.nodes.get(id))
        else {
            continue;
        };
        if &lt.eclass == lt_class {
            return node
                .children
                .get(1)
                .and_then(|id| site.egraph.nodes.get(id))
                .map(|c| c.eclass.clone());
        }
    }
    None
}

/// VIEW ADMISSION (round 10): ground a descriptor's layout tensor in a
/// BUFFER by walking view -> parent. A view layout tensor is
/// (LayoutTensorLit v L) where v is spelled (LogicalIndexMapApply p map _)
/// and L is EXACTLY p's layout composed through map — the tie is checked
/// classwise ((int-subst-of p_expr map) must be L's own bit expression),
/// so a foreign parent or a foreign layout of the same parent can never
/// donate its buffer. Depth-bounded; returns None when no grounding
/// exists (a fresh intermediate the planner may alloc).
fn resolve_buffer(site: &ExtractionSite<'_>, lt_class: &ClassId, depth: usize) -> Option<ClassId> {
    if let Some(buffer) = direct_buffer_of(site, lt_class) {
        return Some(buffer);
    }
    if depth == 0 {
        return None;
    }
    let logical = logical_class_of(site, lt_class)?;
    let layout = layout_class_of(site, lt_class)?;
    // L's own bit expressions (a class can spell several; collect all).
    let l_exprs: Vec<ClassId> = site
        .nodes_in_class_value(&layout, "BitOffsetExpressionLayoutLit")
        .filter_map(|n| site.class_of_child(n, 0))
        .collect();
    if l_exprs.is_empty() {
        return None;
    }
    // Every apply spelling of the view's logical value...
    for apply in site.nodes_in_class_value(&logical, "LogicalIndexMapApply") {
        let Some(parent_logical) = site.class_of_child(apply, 0) else {
            continue;
        };
        let Some(map_class) = site.class_of_child(apply, 1) else {
            continue;
        };
        // ...every layout tensor of that parent...
        for plt in site.egraph.nodes.values() {
            if plt.op != "LayoutTensorLit" {
                continue;
            }
            let Some(pl) = plt
                .children
                .first()
                .and_then(|id| site.egraph.nodes.get(id))
            else {
                continue;
            };
            if pl.eclass != parent_logical {
                continue;
            }
            let plt_class = plt.eclass.clone();
            let Some(p_layout) = plt
                .children
                .get(1)
                .and_then(|id| site.egraph.nodes.get(id))
                .map(|c| c.eclass.clone())
            else {
                continue;
            };
            // ...whose composition through THIS map is L (the tie).
            let tied = site
                .nodes_in_class_value(&p_layout, "BitOffsetExpressionLayoutLit")
                .filter_map(|n| site.class_of_child(n, 0))
                .any(|p_expr| {
                    site.egraph.nodes.values().any(|n| {
                        n.op == "int-subst-of"
                            && n.children.len() >= 2
                            && site
                                .egraph
                                .nodes
                                .get(&n.children[0])
                                .map(|c| c.eclass == p_expr)
                                .unwrap_or(false)
                            && site
                                .egraph
                                .nodes
                                .get(&n.children[1])
                                .map(|c| c.eclass == map_class)
                                .unwrap_or(false)
                            && l_exprs.contains(&n.eclass)
                    })
                });
            if !tied {
                continue;
            }
            if let Some(buffer) = resolve_buffer(site, &plt_class, depth - 1) {
                return Some(buffer);
            }
        }
    }
    None
}

pub fn parse_spec(site: &ExtractionSite<'_>, form: CublasLtForm) -> Option<LtMatmulSpec> {
    let (c_slot, bias_slot, ep_slot) = form.slots();

    // Slot 0: the site — identity triple.
    let site_class = site.child_class(0);
    let (logical_a, logical_b, logical_site_out) = {
        let node = site
            .nodes_in_class_value(&site_class, "CublasLtLogicalMatmulSite")
            .next()?;
        (
            site.class_of_child(node, 0)?,
            site.class_of_child(node, 1)?,
            site.class_of_child(node, 2)?,
        )
    };

    // Slots 1-2: A and B operand readings (site backpointers verified).
    let (desc_a_layout_tensor, trans_a) = parse_operand_descriptor(
        site,
        &site.child_class(1),
        "CublasLtOperandADescriptor",
        &site_class,
    )?;
    let (desc_b_layout_tensor, trans_b) = parse_operand_descriptor(
        site,
        &site.child_class(2),
        "CublasLtOperandBDescriptor",
        &site_class,
    )?;

    // Slot 3: the D reading — the CLAIMED output. No operation child:
    // the API has no transD, and the D arms only ever conclude the
    // row-major reading, so D's view is immediate (see below). That
    // asymmetry is real and deliberate, not an oversight.
    let d_slot_class = site.child_class(3);
    let out_lt_class = {
        let node = site
            .nodes_in_class_value(&d_slot_class, "CublasLtOutputDDescriptor")
            .next()?;
        let back = site.class_of_child(node, 0)?;
        assert_eq!(
            back, site_class,
            "cuBLASLt marker spec inconsistent: D descriptor points at a different site"
        );
        site.class_of_child(node, 1)?
    };
    let logical_out = logical_class_of(site, &out_lt_class)?;

    // THE CALL FRAME, from the SITE's own logical shapes (R10 unswapped):
    // out is [m, n]; the site's a stores a permutation of (m, k); the
    // site's b a permutation of (k, n). For a sibling site minted by the
    // sandwich rewrite these are the recorder operands under new roles,
    // and this frame is the SIBLING's — numerically the round-9 call.
    let a_storage = storage_dims(site, &logical_class_of(site, &desc_a_layout_tensor)?)?;
    let b_storage = storage_dims(site, &logical_class_of(site, &desc_b_layout_tensor)?)?;
    let d_storage = storage_dims(site, &logical_out)?;
    assert_eq!(a_storage.len(), 2, "cuBLASLt marker: rank-2 a expected");
    assert_eq!(b_storage.len(), 2, "cuBLASLt marker: rank-2 b expected");
    assert_eq!(d_storage.len(), 2, "cuBLASLt marker: rank-2 out expected");
    let m = d_storage[0].0.clone();
    let n = d_storage[1].0.clone();
    // k = the a-storage extent that is not m — a CLASS comparison, never
    // an ordering read: the site rules pin a's storage to a permutation
    // of (m, k), so exactly one assignment types (both, when m == k as a
    // class, in which case either choice is the same class).
    let m_class = &d_storage[0].1;
    let k = if &a_storage[0].1 == m_class {
        a_storage[1].0.clone()
    } else if &a_storage[1].1 == m_class {
        a_storage[0].0.clone()
    } else {
        panic!(
            "cuBLASLt marker spec inconsistent: A storage {}x{} shares no extent \
             class with call-m {}",
            a_storage[0].0, a_storage[1].0, m
        );
    };

    // The OPERATION fixes the column-major view (round-8b E1, R10 frame):
    //   A: op(A') must be m x k    B: op(B') must be k x n    D: m x n
    // and ld is that view's row count unless the layout is padded.
    let (a_rows, a_cols) = if trans_a {
        (k.clone(), m.clone())
    } else {
        (m.clone(), k.clone())
    };
    let (b_rows, b_cols) = if trans_b {
        (n.clone(), k.clone())
    } else {
        (k.clone(), n.clone())
    };
    let (d_rows, d_cols) = (m.clone(), n.clone());
    let lda = leading_dimension(site, &desc_a_layout_tensor, &a_rows, &a_storage);
    let ldb = leading_dimension(site, &desc_b_layout_tensor, &b_rows, &b_storage);
    let ldd = leading_dimension(site, &out_lt_class, &d_rows, &d_storage);

    // Defense in depth: the view's extents must be the operand's own
    // storage extents in SOME order (this is the tripwire that the old
    // per-operand orientation walk used to provide).
    let multiset_ok = |rows: &CuDim, cols: &CuDim, storage: &[(CuDim, ClassId)], who: &str| {
        let (s0, s1) = (&storage[0].0, &storage[1].0);
        let ok = (rows == s0 && cols == s1) || (rows == s1 && cols == s0);
        assert!(
            ok,
            "cuBLASLt marker spec inconsistent: {who} view {rows}x{cols} is not a \
             transposition of its storage {s0}x{s1}"
        );
    };
    multiset_ok(&a_rows, &a_cols, &a_storage, "A");
    multiset_ok(&b_rows, &b_cols, &b_storage, "B");
    multiset_ok(&d_rows, &d_cols, &d_storage, "D");

    // VIEW ADMISSION: ground each descriptor in its buffer (view walk).
    let desc_a_buffer = resolve_buffer(site, &desc_a_layout_tensor, 8);
    let desc_b_buffer = resolve_buffer(site, &desc_b_layout_tensor, 8);
    let d_buffer = resolve_buffer(site, &out_lt_class, 8);

    // Payload slots: DIRECT LayoutTensor children (enode-anchored).
    let c_tensor = c_slot.map(|slot| site.child_class(slot));
    let bias_tensor = bias_slot.map(|slot| site.child_class(slot));

    // Epilogue slot: a plain value enum.
    let ep_class = site.child_class(ep_slot);
    let ep_relu = if site
        .nodes_in_class_value(&ep_class, "CublasLtEpilogueDefault")
        .next()
        .is_some()
    {
        false
    } else if site
        .nodes_in_class_value(&ep_class, "CublasLtEpilogueRelu")
        .next()
        .is_some()
    {
        true
    } else {
        return None;
    };

    // C rides the D layout by rule guard; cross-check the layout classes.
    if let Some(c_lt) = &c_tensor {
        let c_layout = layout_class_of(site, c_lt);
        let out_layout = layout_class_of(site, &out_lt_class);
        assert!(
            c_layout.is_some() && c_layout == out_layout,
            "cuBLASLt marker spec inconsistent: C layout class differs from D layout class"
        );
    }

    let epilogue = match (ep_relu, form.has_bias()) {
        (false, false) => CuEpilogue::Default,
        (true, false) => CuEpilogue::Relu,
        (false, true) => CuEpilogue::Bias,
        (true, true) => CuEpilogue::ReluBias,
    };

    let spec = LtMatmulSpec {
        form,
        m,
        n,
        k,
        trans_a,
        trans_b,
        lda,
        ldb,
        ldc: ldd.clone(), // C is guarded onto the D layout; ldc = ldd
        ldd,
        order_col: true, // every reading is COL by convention (R9)
        has_c: form.has_c(),
        has_bias: form.has_bias(),
        epilogue,
        logical_a,
        logical_b,
        logical_out,
        logical_site_out,
        desc_a_layout_tensor,
        desc_b_layout_tensor,
        c_tensor,
        bias_tensor,
        desc_a_buffer,
        desc_b_buffer,
        d_buffer,
    };
    spec.validate(&d_rows, &d_cols);
    Some(spec)
}

// ---------------------------------------------------------------------------
// The op instances (functional + DPS)
// ---------------------------------------------------------------------------

/// One functional op type for all four contracts; the FORM fixes the
/// label, the operand names, and the Lit arity.
#[derive(Debug, Clone, PartialEq)]
pub struct CublasLt {
    pub form: CublasLtForm,
    pub spec: Option<LtMatmulSpec>,
}

impl OpSlotNames for CublasLt {
    fn operand_name(&self, operand: usize) -> String {
        self.form
            .operand_names()
            .get(operand)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("in{operand}"))
    }
}

impl BufferTensorIrOp for CublasLt {
    fn label(&self) -> &str {
        // IR identity = egglog constructor minus the LayoutTensorOp prefix.
        match self.form {
            CublasLtForm::Base => "CublasLt",
            CublasLtForm::Bias => "CublasLtBias",
            CublasLtForm::Accumulate => "CublasLtAccumulate",
            CublasLtForm::AccumulateBias => "CublasLtAccumulateBias",
        }
    }
}

impl Bufferizable for CublasLt {}

impl ToDps for CublasLt {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        // cublasLtMatmul writes into a CALLER-PROVIDED D buffer.
        Some(Box::new(CublasLtDps { op: self.clone() }))
    }
}

impl LayoutIrOp for CublasLt {}

/// Destination-passing form: inputs unchanged, one write-only destination
/// tied to the single result (Must), and — on the Accumulate contracts —
/// the API's C==D same-buffer in-place accumulate expressed as a MAY
/// alias between the C operand and the result. Legality holds because the
/// C-fold rule guards C onto the D layout class (identical layouts), which
/// is exactly the API's C==D precondition. The bufferizer's donation
/// machinery is the intended consumer.
#[derive(Debug, Clone, PartialEq)]
pub struct CublasLtDps {
    pub op: CublasLt,
}

impl CublasLtDps {
    fn dest_index(&self) -> usize {
        self.op.form.lit_arity()
    }
}

impl OpSlotNames for CublasLtDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == self.dest_index() {
            "dest".to_string()
        } else {
            self.op.operand_name(operand)
        }
    }
}

impl BufferTensorIrOp for CublasLtDps {
    fn label(&self) -> &str {
        self.op.label() // DPS forms keep the IR name
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index()
    }
}

impl Bufferizable for CublasLtDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        let mut aliases = vec![AliasInfo {
            operand: self.dest_index(),
            result: 0,
            sharing: Sharing::Must,
        }];
        if self.op.form.has_c() {
            // C sits at Lit slot 2 in the contract order [a, b, c, bias?].
            aliases.push(AliasInfo {
                operand: 2,
                result: 0,
                sharing: Sharing::May,
            });
        }
        aliases
    }
}

impl ToDps for CublasLtDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS
    }
}

impl LayoutIrOp for CublasLtDps {}

// ---------------------------------------------------------------------------
// The matcher family — one matcher per contract; only the Base matcher
// carries the shared egg snippets (they declare the whole vocabulary).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CublasLtMarkerMatcher {
    pub form: CublasLtForm,
}

pub fn all_matchers() -> Vec<CublasLtMarkerMatcher> {
    CublasLtForm::ALL
        .into_iter()
        .map(|form| CublasLtMarkerMatcher { form })
        .collect()
}

impl OpMatcher for CublasLtMarkerMatcher {
    fn egglog_constructor(&self) -> &'static str {
        self.form.constructor_name()
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        if self.form != CublasLtForm::Base {
            return Vec::new(); // vocabulary declared once, by the Base matcher
        }
        vec![
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: include_str!("egg/cublaslt_marker_constructors.egg"),
            },
            EgglogSnippet {
                // ROUND 11: the three canonicalization rewrites (named by
                // logical shapes) fold every recorded spelling into the
                // ONE canonical form A[m,k], B[k,n] -> out[m,n], plus the
                // double-transpose collapse that anchors termination.
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_canonicalize.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_site.egg"),
            },
            EgglogSnippet {
                // The standalone transpose-sandwich rewrite (one rule,
                // canonical form only, round 11) — logical equivalence
                // reasoning; the unswapped arms match the sibling sites
                // it mints.
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_rewrite.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_desc.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_assemble.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("egg/cublaslt_marker_decorate.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        match self.form {
            CublasLtForm::Base => &[
                ("cublaslt_site", 0),
                ("cublaslt_a_descriptor", 1),
                ("cublaslt_b_descriptor", 2),
                ("cublaslt_d_descriptor", 3),
                ("cublaslt_epilogue", 4),
            ],
            CublasLtForm::Bias => &[
                ("cublaslt_site", 0),
                ("cublaslt_a_descriptor", 1),
                ("cublaslt_b_descriptor", 2),
                ("cublaslt_d_descriptor", 3),
                ("cublaslt_bias_tensor", 4),
                ("cublaslt_epilogue", 5),
            ],
            CublasLtForm::Accumulate => &[
                ("cublaslt_site", 0),
                ("cublaslt_a_descriptor", 1),
                ("cublaslt_b_descriptor", 2),
                ("cublaslt_d_descriptor", 3),
                ("cublaslt_c_tensor", 4),
                ("cublaslt_epilogue", 5),
            ],
            CublasLtForm::AccumulateBias => &[
                ("cublaslt_site", 0),
                ("cublaslt_a_descriptor", 1),
                ("cublaslt_b_descriptor", 2),
                ("cublaslt_d_descriptor", 3),
                ("cublaslt_c_tensor", 4),
                ("cublaslt_bias_tensor", 5),
                ("cublaslt_epilogue", 6),
            ],
        }
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(CublasLt {
            form: self.form,
            spec: parse_spec(site, self.form),
        })
    }
}
