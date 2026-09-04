//! Rust mirrors of the five egglog `Layout` constructors — the CONVENIENCE
//! vocabulary runtimes may share (resident-geometry cleanup train; folded
//! into core from the short-lived `luminal_layouts` crate by Austin's
//! amendment: "don't make it separate crate, people can just pull these
//! structs from core for now").
//!
//! THE BUFFERIZER NEVER CALLS ANY OF THIS. Living in core does not make
//! this a core vocabulary: the bufferizer stays generic over an opaque
//! layout type it only clones and transports, and nothing in the planner
//! imports this module. It exists so runtimes can pull the five mirror
//! structs, the [`SpanExpr`] trait, and [`decode_layout`] from one place
//! instead of each respelling them; a backend that wants a different
//! layout vocabulary brings its own type and ignores this module
//! entirely — nothing here is a closed set the planner depends on.
//!
//! Contents:
//!  * [`IntExprTerm`] / [`ShapeTerm`] / [`BitWidthTerm`] — the term
//!    vocabulary the constructor fields are spelled in;
//!  * the five mirror structs, field-for-field with the preamble's
//!    constructors (`RightMajorContiguousElementLayoutLit(Shape, BitWidth)`
//!    and friends), plus the [`MirrorLayout`] sum for decoders;
//!  * [`SpanExpr`] — the span-as-EXPRESSION trait, implemented ONLY where
//!    a span is honest (the packed element ladder: right-major,
//!    left-major, strided). The offset-expression forms deliberately do
//!    NOT implement it: an offset function alone does not disclose its
//!    reach, and nothing here guesses. NOTHING consumes `span()` yet.
//!  * [`decode_layout`] — the spelling decoder: walk one layout e-class
//!    of a serialized e-graph into a [`MirrorLayout`]. Any spelling
//!    present in a class is correct (all spellings of a layout class
//!    denote one function); the walk PREFERS the most-structured spelling
//!    present (RightMajor > LeftMajor > Strided > ElementOffset >
//!    BitOffset) as a decoding preference only. No normalization, no
//!    analysis — a class none of whose spellings parse is a loud error,
//!    never a guess.

use anyhow::{Result, anyhow, bail};
use egraph_serialize::{ClassId, EGraph, Node, NodeId};
use std::collections::HashMap;

// =============================================================================
// Term vocabulary
// =============================================================================

/// An integer expression TERM — the symbolic vocabulary layout fields are
/// decoded into. A direct transliteration of the preamble's `IntExpr`
/// subset that appears inside layouts; construction only, no evaluation
/// and no rewriting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntExprTerm {
    /// `(IntLit n)`.
    Lit(i64),
    /// `(IntVar "name")` — a symbolic dimension.
    Var(String),
    /// `(CoordVar shape axis)` — a coordinate over the owning layout's
    /// domain, axis 0-based FROM THE END (the preamble's de Bruijn
    /// convention). The owner shape is checked at decode time (a foreign
    /// shape's coordinate never silently parses); the term keeps the
    /// axis alone.
    Coord {
        axis_from_end: i64,
    },
    Add(Box<IntExprTerm>, Box<IntExprTerm>),
    Mul(Box<IntExprTerm>, Box<IntExprTerm>),
    /// Truncated (toward-zero) division — `IntTruncDiv`.
    TruncDiv(Box<IntExprTerm>, Box<IntExprTerm>),
    /// Truncated remainder — `IntTruncRem`.
    TruncRem(Box<IntExprTerm>, Box<IntExprTerm>),
    /// `IntCeilDiv`.
    CeilDiv(Box<IntExprTerm>, Box<IntExprTerm>),
    Min(Box<IntExprTerm>, Box<IntExprTerm>),
    Max(Box<IntExprTerm>, Box<IntExprTerm>),
    /// The bool bridge's indicator: `IntCastFromBool(BoolLessThanInt(a, b))`.
    LessThanCast(Box<IntExprTerm>, Box<IntExprTerm>),
}

impl IntExprTerm {
    /// Substitute every coordinate with the given per-axis replacement
    /// (axis keyed FROM THE END). Pure tree map; an axis the caller
    /// supplies no replacement for is a violated mirror invariant and
    /// panics loudly.
    fn substitute_coords(&self, replace: &dyn Fn(i64) -> IntExprTerm) -> IntExprTerm {
        let go = |e: &IntExprTerm| Box::new(e.substitute_coords(replace));
        match self {
            IntExprTerm::Lit(v) => IntExprTerm::Lit(*v),
            IntExprTerm::Var(name) => IntExprTerm::Var(name.clone()),
            IntExprTerm::Coord { axis_from_end } => replace(*axis_from_end),
            IntExprTerm::Add(a, b) => IntExprTerm::Add(go(a), go(b)),
            IntExprTerm::Mul(a, b) => IntExprTerm::Mul(go(a), go(b)),
            IntExprTerm::TruncDiv(a, b) => IntExprTerm::TruncDiv(go(a), go(b)),
            IntExprTerm::TruncRem(a, b) => IntExprTerm::TruncRem(go(a), go(b)),
            IntExprTerm::CeilDiv(a, b) => IntExprTerm::CeilDiv(go(a), go(b)),
            IntExprTerm::Min(a, b) => IntExprTerm::Min(go(a), go(b)),
            IntExprTerm::Max(a, b) => IntExprTerm::Max(go(a), go(b)),
            IntExprTerm::LessThanCast(a, b) => IntExprTerm::LessThanCast(go(a), go(b)),
        }
    }
}

/// The domain: `(ShapeLit IntExprList)`, one extent expression per axis,
/// outermost first (list order).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeTerm(pub Vec<IntExprTerm>);

impl ShapeTerm {
    /// The extent of the axis counted FROM THE END (the CoordVar basis).
    /// Panics on an out-of-range axis — a violated mirror invariant.
    fn extent_from_end(&self, axis_from_end: i64) -> &IntExprTerm {
        let rank = self.0.len();
        usize::try_from(axis_from_end)
            .ok()
            .filter(|&a| a < rank)
            .map(|a| &self.0[rank - 1 - a])
            .unwrap_or_else(|| {
                panic!("axis {axis_from_end} (from end) out of range for rank {rank}")
            })
    }
}

/// The element access width: `(BitWidthLit i64)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitWidthTerm(pub i64);

// =============================================================================
// The five mirror structs (field-for-field with preamble constructors)
// =============================================================================

/// `(RightMajorContiguousElementLayoutLit Shape BitWidth)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RightMajorContiguousElementLayout {
    pub shape: ShapeTerm,
    pub width: BitWidthTerm,
}

/// `(LeftMajorContiguousElementLayoutLit Shape BitWidth)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeftMajorContiguousElementLayout {
    pub shape: ShapeTerm,
    pub width: BitWidthTerm,
}

/// `(StridedElementLayoutLit Shape IntAffineExprList BitWidth)` — the
/// affine CHAIN: one summand per axis FROM THE END, each canonically
/// `(IntMul (CoordVar shape axis) stride)`, with the doctrine's residues
/// (`(IntLit 0)` for a dead axis, the bare coordinate for stride 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedElementLayout {
    pub shape: ShapeTerm,
    pub chain: Vec<IntExprTerm>,
    pub width: BitWidthTerm,
}

/// `(ElementOffsetExpressionLayoutLit IntExpr Shape BitWidth)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementOffsetExpressionLayout {
    pub offset: IntExprTerm,
    pub shape: ShapeTerm,
    pub width: BitWidthTerm,
}

/// `(BitOffsetExpressionLayoutLit IntExpr Shape BitWidth)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitOffsetExpressionLayout {
    pub offset: IntExprTerm,
    pub shape: ShapeTerm,
    pub width: BitWidthTerm,
}

/// The convenience sum decoders produce — one value for "some spelling
/// of this layout class". DECODER CONVENIENCE ONLY: the bufferizer is
/// generic and never sees this type; a backend may decode into its own
/// type instead. NOT a closed vocabulary — and FLAGGED for review
/// (Austin): a sum over the five constructors may already be too
/// enum-ish for a vocabulary that is deliberately open; kept for now as
/// the shared decoders' return type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirrorLayout {
    RightMajor(RightMajorContiguousElementLayout),
    LeftMajor(LeftMajorContiguousElementLayout),
    Strided(StridedElementLayout),
    ElementOffset(ElementOffsetExpressionLayout),
    BitOffset(BitOffsetExpressionLayout),
}

// =============================================================================
// Span — an EXPRESSION, implemented only where honest
// =============================================================================

/// The storage reach of a layout, in ELEMENTS, as an expression over the
/// layout's own terms (symbolic dims stay symbolic — no evaluation).
/// Implemented ONLY for the packed element ladder, where the span is the
/// constructor's own meaning: contiguous forms span their element count;
/// a strided chain spans to its largest addressed element + 1 (offsets
/// are monotone in every coordinate — element strides are non-negative
/// by construction, an element layout has no base offset to point
/// backwards from). The offset-expression forms do NOT implement this
/// trait: nothing here analyzes an arbitrary offset function's reach.
pub trait SpanExpr {
    fn span(&self) -> IntExprTerm;
}

/// numel: the product of the extents, `(IntLit 1)` at rank 0.
fn numel(shape: &ShapeTerm) -> IntExprTerm {
    let mut extents = shape.0.iter();
    let Some(first) = extents.next() else {
        return IntExprTerm::Lit(1);
    };
    extents.fold(first.clone(), |acc, e| {
        IntExprTerm::Mul(Box::new(acc), Box::new(e.clone()))
    })
}

impl SpanExpr for RightMajorContiguousElementLayout {
    fn span(&self) -> IntExprTerm {
        numel(&self.shape)
    }
}

impl SpanExpr for LeftMajorContiguousElementLayout {
    fn span(&self) -> IntExprTerm {
        numel(&self.shape)
    }
}

impl SpanExpr for StridedElementLayout {
    /// `1 + Σ summand[coord_axis := extent_axis − 1]`: each summand
    /// evaluated at the last coordinate of its axis. Handles the chain's
    /// three canonical residues uniformly — `x*stride` becomes
    /// `(extent−1)*stride`, a bare coordinate becomes `extent−1`, and the
    /// zero residue stays zero — because substitution is a pure tree map,
    /// never a pattern match on spellings. `extent − 1` is constructed as
    /// `(IntAdd extent (IntLit -1))` uniformly (symbolic extents have no
    /// other spelling, and folding literal ones would be normalization,
    /// which this crate refuses to do).
    fn span(&self) -> IntExprTerm {
        let shape = &self.shape;
        let last_coord = |axis_from_end: i64| -> IntExprTerm {
            IntExprTerm::Add(
                Box::new(shape.extent_from_end(axis_from_end).clone()),
                Box::new(IntExprTerm::Lit(-1)),
            )
        };
        let mut total = IntExprTerm::Lit(1);
        for summand in &self.chain {
            total = IntExprTerm::Add(
                Box::new(total),
                Box::new(summand.substitute_coords(&last_coord)),
            );
        }
        total
    }
}

// =============================================================================
// Literal readers — RUNTIME convenience, never planner machinery
// =============================================================================

impl IntExprTerm {
    /// Evaluate a closed literal expression (no vars, no coordinates) to
    /// its value. `None` for symbolic/coordinate-bearing terms — callers
    /// bail loudly, never guess. Runtime-side convenience: the planner
    /// never evaluates layout terms.
    pub fn eval_literal(&self) -> Option<i64> {
        let go = |e: &IntExprTerm| e.eval_literal();
        Some(match self {
            IntExprTerm::Lit(v) => *v,
            IntExprTerm::Var(_) | IntExprTerm::Coord { .. } => return None,
            IntExprTerm::Add(a, b) => go(a)?.checked_add(go(b)?)?,
            IntExprTerm::Mul(a, b) => go(a)?.checked_mul(go(b)?)?,
            IntExprTerm::TruncDiv(a, b) => go(a)?.checked_div(go(b)?)?,
            IntExprTerm::TruncRem(a, b) => go(a)?.checked_rem(go(b)?)?,
            IntExprTerm::CeilDiv(a, b) => {
                let (a, b) = (go(a)?, go(b)?);
                if b == 0 {
                    return None;
                }
                // Toward +inf for the non-negative operands layouts use.
                a.checked_add(b - 1)?.checked_div(b)?
            }
            IntExprTerm::Min(a, b) => go(a)?.min(go(b)?),
            IntExprTerm::Max(a, b) => go(a)?.max(go(b)?),
            IntExprTerm::LessThanCast(a, b) => i64::from(go(a)? < go(b)?),
        })
    }
}

impl MirrorLayout {
    /// The layout's DOMAIN shape (every constructor carries one).
    pub fn shape(&self) -> &ShapeTerm {
        match self {
            MirrorLayout::RightMajor(l) => &l.shape,
            MirrorLayout::LeftMajor(l) => &l.shape,
            MirrorLayout::Strided(l) => &l.shape,
            MirrorLayout::ElementOffset(l) => &l.shape,
            MirrorLayout::BitOffset(l) => &l.shape,
        }
    }

    /// The element access width in bits.
    pub fn width_bits(&self) -> i64 {
        match self {
            MirrorLayout::RightMajor(l) => l.width.0,
            MirrorLayout::LeftMajor(l) => l.width.0,
            MirrorLayout::Strided(l) => l.width.0,
            MirrorLayout::ElementOffset(l) => l.width.0,
            MirrorLayout::BitOffset(l) => l.width.0,
        }
    }

    /// The domain extents as literals — `None` if any axis is symbolic.
    pub fn literal_extents(&self) -> Option<Vec<usize>> {
        self.shape()
            .0
            .iter()
            .map(|e| e.eval_literal().and_then(|v| usize::try_from(v).ok()))
            .collect()
    }

    /// The layout's storage reach in ELEMENTS, where the constructor
    /// discloses one ([`SpanExpr`]: the packed ladder) and the terms are
    /// literal. `None` for the offset-expression forms (undisclosed
    /// reach) and for symbolic terms — allocation-sizing callers bail
    /// loudly on `None`, never guess.
    pub fn literal_span_elements(&self) -> Option<usize> {
        let span = match self {
            MirrorLayout::RightMajor(l) => l.span(),
            MirrorLayout::LeftMajor(l) => l.span(),
            MirrorLayout::Strided(l) => l.span(),
            MirrorLayout::ElementOffset(_) | MirrorLayout::BitOffset(_) => return None,
        };
        span.eval_literal().and_then(|v| usize::try_from(v).ok())
    }
}

// =============================================================================
// The spelling decoder
// =============================================================================

/// Decode one layout e-class of a serialized e-graph into a
/// [`MirrorLayout`]. Builds a class index for the walk (one pass over the
/// e-graph), so callers decoding many classes should memoize per class
/// (all spellings of a class denote one function, so a class decodes the
/// same every time). Errors are LOUD and name the class — never a guess.
pub fn decode_layout(egraph: &EGraph, class: &ClassId) -> Result<MirrorLayout> {
    Reader::new(egraph).decode_layout(class)
}

/// The decoder's walk state: the e-graph plus its by-class node index.
struct Reader<'a> {
    egraph: &'a EGraph,
    class_nodes: HashMap<&'a ClassId, Vec<&'a NodeId>>,
}

/// Memo entry for the expression parse: finished, or the in-progress
/// cycle guard (the index_expr discipline — a cycle fails the SPELLING,
/// and a tainted failure is not cached because it is contextual).
enum ParseMemo {
    InProgress,
    Done(Option<IntExprTerm>),
}

impl<'a> Reader<'a> {
    fn new(egraph: &'a EGraph) -> Self {
        let mut class_nodes: HashMap<&'a ClassId, Vec<&'a NodeId>> = HashMap::new();
        for (id, node) in &egraph.nodes {
            class_nodes.entry(&node.eclass).or_default().push(id);
        }
        // Deterministic spelling order regardless of map iteration order.
        for ids in class_nodes.values_mut() {
            ids.sort();
        }
        Self {
            egraph,
            class_nodes,
        }
    }

    /// Every node of `op` in `class` — unsubsumed spellings first,
    /// subsumed ones as fallback (value PARSING reads denotations, and a
    /// subsumed node is still a true member of its class; saturation can
    /// subsume every constructor spelling — the slice_pad lesson).
    fn nodes_in_class_value(
        &self,
        class: &ClassId,
        op: &str,
    ) -> impl Iterator<Item = &'a Node> + '_ {
        let ids = self
            .class_nodes
            .get(class)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let matching = move |want_subsumed: bool| {
            let op = op.to_string();
            ids.iter()
                .filter_map(|id| self.egraph.nodes.get(*id))
                .filter(move |node| node.op == op && node.subsumed == want_subsumed)
        };
        matching(false).chain(matching(true))
    }

    fn class_of_child(&self, node: &Node, index: usize) -> Option<ClassId> {
        let child = node.children.get(index)?;
        Some(self.egraph.nodes.get(child)?.eclass.clone())
    }

    /// Any literal-i64 node inside a class.
    fn parse_i64(&self, class: &ClassId) -> Option<i64> {
        self.class_nodes
            .get(class)?
            .iter()
            .filter_map(|id| self.egraph.nodes.get(*id))
            .find_map(|node| node.op.parse::<i64>().ok())
    }

    /// Any literal-string node inside a class (egraph_serialize renders
    /// string literals as quoted ops).
    fn parse_string(&self, class: &ClassId) -> Option<String> {
        self.class_nodes
            .get(class)?
            .iter()
            .filter_map(|id| self.egraph.nodes.get(*id))
            .find_map(|node| {
                let op = node.op.as_str();
                op.strip_prefix('"')?.strip_suffix('"').map(str::to_string)
            })
    }

    fn decode_layout(&self, class: &ClassId) -> Result<MirrorLayout> {
        // The decoding PREFERENCE (most-structured spelling first); any
        // spelling present is correct, so the first that parses wins and
        // later constructors are backtracking fallbacks only.
        let mut present = Vec::new();
        for node in self.nodes_in_class_value(class, "RightMajorContiguousElementLayoutLit") {
            present.push("RightMajorContiguousElementLayoutLit");
            let (Some(shape), Some(width)) = (
                self.class_of_child(node, 0)
                    .and_then(|c| self.decode_shape(&c)),
                self.class_of_child(node, 1)
                    .and_then(|c| self.decode_bit_width(&c)),
            ) else {
                continue;
            };
            return Ok(MirrorLayout::RightMajor(
                RightMajorContiguousElementLayout { shape, width },
            ));
        }
        for node in self.nodes_in_class_value(class, "LeftMajorContiguousElementLayoutLit") {
            present.push("LeftMajorContiguousElementLayoutLit");
            let (Some(shape), Some(width)) = (
                self.class_of_child(node, 0)
                    .and_then(|c| self.decode_shape(&c)),
                self.class_of_child(node, 1)
                    .and_then(|c| self.decode_bit_width(&c)),
            ) else {
                continue;
            };
            return Ok(MirrorLayout::LeftMajor(LeftMajorContiguousElementLayout {
                shape,
                width,
            }));
        }
        for node in self.nodes_in_class_value(class, "StridedElementLayoutLit") {
            present.push("StridedElementLayoutLit");
            let (Some(shape_class), Some(chain_class), Some(width_class)) = (
                self.class_of_child(node, 0),
                self.class_of_child(node, 1),
                self.class_of_child(node, 2),
            ) else {
                continue;
            };
            let (Some(shape), Some(width)) = (
                self.decode_shape(&shape_class),
                self.decode_bit_width(&width_class),
            ) else {
                continue;
            };
            let Some(chain) = self.decode_affine_chain(&chain_class, &shape_class) else {
                continue;
            };
            return Ok(MirrorLayout::Strided(StridedElementLayout {
                shape,
                chain,
                width,
            }));
        }
        for (constructor, bit_form) in [
            ("ElementOffsetExpressionLayoutLit", false),
            ("BitOffsetExpressionLayoutLit", true),
        ] {
            for node in self.nodes_in_class_value(class, constructor) {
                present.push(constructor);
                let (Some(offset_class), Some(shape_class), Some(width_class)) = (
                    self.class_of_child(node, 0),
                    self.class_of_child(node, 1),
                    self.class_of_child(node, 2),
                ) else {
                    continue;
                };
                let (Some(shape), Some(width)) = (
                    self.decode_shape(&shape_class),
                    self.decode_bit_width(&width_class),
                ) else {
                    continue;
                };
                let mut memo = HashMap::new();
                let Some(offset) =
                    self.parse_int_expr(&offset_class, 64, Some(&shape_class), &mut memo)
                else {
                    continue;
                };
                return Ok(if bit_form {
                    MirrorLayout::BitOffset(BitOffsetExpressionLayout {
                        offset,
                        shape,
                        width,
                    })
                } else {
                    MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
                        offset,
                        shape,
                        width,
                    })
                });
            }
        }
        if present.is_empty() {
            bail!(
                "layout class {class} has no Layout constructor spelling — \
                 nothing to decode (fail-closed, never a guess)"
            );
        }
        bail!(
            "layout class {class} has constructor spellings {present:?} but \
             none parsed into the mirror vocabulary (fail-closed, never a \
             guess)"
        )
    }

    fn decode_bit_width(&self, class: &ClassId) -> Option<BitWidthTerm> {
        for node in self.nodes_in_class_value(class, "BitWidthLit") {
            let Some(bits_class) = self.class_of_child(node, 0) else {
                continue;
            };
            if let Some(bits) = self.parse_i64(&bits_class) {
                return Some(BitWidthTerm(bits));
            }
        }
        None
    }

    fn decode_shape(&self, class: &ClassId) -> Option<ShapeTerm> {
        for node in self.nodes_in_class_value(class, "ShapeLit") {
            let Some(head) = self.class_of_child(node, 0) else {
                continue;
            };
            let mut memo = HashMap::new();
            if let Some(extents) =
                self.decode_expr_list(&head, "IntExprCons", "IntExprNil", 64, None, &mut memo)
            {
                return Some(ShapeTerm(extents));
            }
        }
        None
    }

    /// The strided chain: one summand per axis from-end. Coordinates are
    /// guarded to the layout's OWN shape (a foreign shape's coordinate is
    /// not this domain's and fails that spelling — the owner-shape guard).
    fn decode_affine_chain(
        &self,
        class: &ClassId,
        owner_shape: &ClassId,
    ) -> Option<Vec<IntExprTerm>> {
        let mut memo = HashMap::new();
        self.decode_expr_list(
            class,
            "IntAffineExprCons",
            "IntAffineExprNil",
            64,
            Some(owner_shape),
            &mut memo,
        )
    }

    /// Cons-spine walk, existential at every level (the backtracking
    /// doctrine): a saturated list class holds several cons spellings and
    /// the first may dead-end while a sibling parses fine.
    fn decode_expr_list(
        &self,
        class: &ClassId,
        cons_op: &str,
        nil_op: &str,
        depth: usize,
        owner_shape: Option<&ClassId>,
        memo: &mut HashMap<ClassId, ParseMemo>,
    ) -> Option<Vec<IntExprTerm>> {
        if depth == 0 {
            return None;
        }
        if self.nodes_in_class_value(class, nil_op).next().is_some() {
            return Some(Vec::new());
        }
        for cons in self.nodes_in_class_value(class, cons_op) {
            let Some(element) = self.class_of_child(cons, 0) else {
                continue;
            };
            let Some(tail) = self.class_of_child(cons, 1) else {
                continue;
            };
            let Some(expr) = self.parse_int_expr(&element, 64, owner_shape, memo) else {
                continue;
            };
            if let Some(mut rest) =
                self.decode_expr_list(&tail, cons_op, nil_op, depth - 1, owner_shape, memo)
            {
                rest.insert(0, expr);
                return Some(rest);
            }
        }
        None
    }

    /// Parse one IntExpr class, preferring folded literals; memoized with
    /// the cycle-taint rule (a `None` whose walk touched the in-progress
    /// guard is contextual and not cached; an untainted `None` — every
    /// spelling genuinely outside the subset — caches).
    fn parse_int_expr(
        &self,
        class: &ClassId,
        depth: usize,
        owner_shape: Option<&ClassId>,
        memo: &mut HashMap<ClassId, ParseMemo>,
    ) -> Option<IntExprTerm> {
        self.parse_int_expr_tainting(class, depth, owner_shape, memo, &mut false)
    }

    fn parse_int_expr_tainting(
        &self,
        class: &ClassId,
        depth: usize,
        owner_shape: Option<&ClassId>,
        memo: &mut HashMap<ClassId, ParseMemo>,
        tainted: &mut bool,
    ) -> Option<IntExprTerm> {
        match memo.get(class) {
            Some(ParseMemo::Done(cached)) => return cached.clone(),
            Some(ParseMemo::InProgress) => {
                *tainted = true;
                return None;
            }
            None => {}
        }
        memo.insert(class.clone(), ParseMemo::InProgress);
        let mut local_taint = false;
        let parsed =
            self.parse_int_expr_uncached(class, depth, owner_shape, memo, &mut local_taint);
        if parsed.is_none() && local_taint {
            memo.remove(class);
            *tainted = true;
        } else {
            memo.insert(class.clone(), ParseMemo::Done(parsed.clone()));
        }
        parsed
    }

    fn parse_int_expr_uncached(
        &self,
        class: &ClassId,
        depth: usize,
        owner_shape: Option<&ClassId>,
        memo: &mut HashMap<ClassId, ParseMemo>,
        tainted: &mut bool,
    ) -> Option<IntExprTerm> {
        if depth == 0 {
            return None;
        }
        if let Some(lit) = self.nodes_in_class_value(class, "IntLit").next() {
            let value_class = self.class_of_child(lit, 0)?;
            return Some(IntExprTerm::Lit(self.parse_i64(&value_class)?));
        }
        for var in self.nodes_in_class_value(class, "IntVar") {
            let Some(name_class) = self.class_of_child(var, 0) else {
                continue;
            };
            if let Some(name) = self.parse_string(&name_class) {
                return Some(IntExprTerm::Var(name));
            }
        }
        for coord in self.nodes_in_class_value(class, "CoordVar") {
            // Child 0 is the owner Shape, child 1 the axis (from-end).
            // The owner-shape guard: when the caller names the layout's
            // domain, a CoordVar owned by any OTHER shape is not one of
            // this layout's coordinates and cannot parse.
            if let Some(expected) = owner_shape {
                let Some(owner_class) = self.class_of_child(coord, 0) else {
                    continue;
                };
                if owner_class != *expected {
                    continue;
                }
            }
            let Some(axis_class) = self.class_of_child(coord, 1) else {
                continue;
            };
            return Some(IntExprTerm::Coord {
                axis_from_end: self.parse_i64(&axis_class)?,
            });
        }
        type Build = fn(Box<IntExprTerm>, Box<IntExprTerm>) -> IntExprTerm;
        let binary_kinds: [(&str, Build); 7] = [
            ("IntAdd", IntExprTerm::Add),
            ("IntMul", IntExprTerm::Mul),
            ("IntTruncDiv", IntExprTerm::TruncDiv),
            ("IntTruncRem", IntExprTerm::TruncRem),
            ("IntCeilDiv", IntExprTerm::CeilDiv),
            ("IntMin", IntExprTerm::Min),
            ("IntMax", IntExprTerm::Max),
        ];
        for (kind, build) in binary_kinds {
            for node in self.nodes_in_class_value(class, kind) {
                let Some(lhs_class) = self.class_of_child(node, 0) else {
                    continue;
                };
                let Some(rhs_class) = self.class_of_child(node, 1) else {
                    continue;
                };
                let Some(lhs) =
                    self.parse_int_expr_tainting(&lhs_class, depth - 1, owner_shape, memo, tainted)
                else {
                    continue;
                };
                let Some(rhs) =
                    self.parse_int_expr_tainting(&rhs_class, depth - 1, owner_shape, memo, tainted)
                else {
                    continue;
                };
                return Some(build(Box::new(lhs), Box::new(rhs)));
            }
        }
        for cast in self.nodes_in_class_value(class, "IntCastFromBool") {
            let Some(bool_class) = self.class_of_child(cast, 0) else {
                continue;
            };
            let Some(less_than) = self
                .nodes_in_class_value(&bool_class, "BoolLessThanInt")
                .next()
            else {
                continue;
            };
            let Some(lhs_class) = self.class_of_child(less_than, 0) else {
                continue;
            };
            let Some(rhs_class) = self.class_of_child(less_than, 1) else {
                continue;
            };
            let Some(lhs) =
                self.parse_int_expr_tainting(&lhs_class, depth - 1, owner_shape, memo, tainted)
            else {
                continue;
            };
            let Some(rhs) =
                self.parse_int_expr_tainting(&rhs_class, depth - 1, owner_shape, memo, tainted)
            else {
                continue;
            };
            return Some(IntExprTerm::LessThanCast(Box::new(lhs), Box::new(rhs)));
        }
        None
    }
}

/// Convenience for decoders that must be total: [`decode_layout`] with
/// the error contextualized by who was asking.
pub fn decode_layout_for(egraph: &EGraph, class: &ClassId, who: &str) -> Result<MirrorLayout> {
    decode_layout(egraph, class).map_err(|err| anyhow!("{who}: {err}"))
}

// =============================================================================
// THE DECODED LAYOUT and its table — core's DECODER (ruling D9,
// 2026-09-03: "the core can have a decoder producing the layout struct
// from core; the runtimes should just directly import and use these
// layout structs"). The per-runtime `LayoutDecoder<L>` hook is GONE: it
// bought a genericity nobody exercised (both runtimes decoded the same
// mirror struct plus the same dtype fact), and the search that called it
// now lives in the runtimes anyway.
//
// THE BUFFERIZER STILL NEVER READS THIS. `bufferize` stays generic over
// an opaque `PlanLayout`; this is simply the layout type both shipped
// runtimes choose to instantiate it with.
// =============================================================================

/// One elected value's decoded layout: the mirror layout plus the
/// value's `dtype-of` fact. `dtype: None` is representable (a value with
/// no `dtype-of` row) and bails loudly at USE — staging, allocation
/// typing, readback — never silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedLayout {
    pub mirror: MirrorLayout,
    pub dtype: Option<crate::dtype::PlanDtype>,
}

/// The caller-owned decoded-layout cache: `(layout class, dtype-of fact)`
/// → the decoded layout. Decoding is a PURE function of that key, so one
/// map serves every candidate a search decodes over ONE serialized
/// e-graph — and only that one: `ClassId`s are meaningful only inside the
/// e-graph they came from, so a new render gets a new cache.
pub type LayoutDecodeCache = HashMap<(ClassId, Option<crate::dtype::PlanDtype>), DecodedLayout>;

/// Build the decoded-layout table for one extracted graph, keyed by
/// VALUE e-class: enumerate every elected value and decode its layout
/// class into a [`DecodedLayout`].
///
/// THE CACHE IS THE CALLER'S, keyed by `(layout class, dtype-of fact)` —
/// all spellings of a layout class denote one function, and the dtype
/// fact is the one extraction-side value fact folded into the decoded
/// type, so decoding is a PURE function of that key and one cache serves
/// every value of the graph AND every later graph over the same e-graph.
/// A search loop holds one for its whole run: `decode_layout` builds a
/// fresh `Reader` that walks and sorts every node of the serialized
/// e-graph, so a per-call cache would pay that index once per distinct
/// layout class per CANDIDATE. A one-shot caller passes
/// `&mut HashMap::new()`. A decode error is LOUD and refuses the graph:
/// there is no default layout.
///
/// CALL IT ON THE GRAPH BUFFERIZE WILL SEE — the POST-DPS one. The table
/// is VALUE-keyed, and the DPS rewrite mints fresh poison-destination
/// VALUES, so a pre-DPS table is not total over the post-DPS graph and
/// `extraction_layouts` refuses it loudly. Decoding post-DPS is free:
/// each poison clones its tied result's layout class AND dtype fact, so
/// it hits the `(layout class, dtype)` cache.
pub fn decode_layout_table(
    egraph: &EGraph,
    graph: &crate::layout_ir::ExtractedGraph,
    who: &str,
    cache: &mut LayoutDecodeCache,
) -> Result<HashMap<ClassId, DecodedLayout>> {
    use crate::layout_ir::ExtractedNode;

    let mut table: HashMap<ClassId, DecodedLayout> = HashMap::new();
    let mut decode = |value: &crate::layout_ir::LayoutTensorInfo,
                      table: &mut HashMap<ClassId, DecodedLayout>|
     -> Result<()> {
        if table.contains_key(&value.eclass) {
            return Ok(());
        }
        let key = (value.layout.eclass.clone(), value.dtype_enum);
        let decoded = match cache.get(&key) {
            Some(decoded) => decoded.clone(),
            None => {
                let decoded = DecodedLayout {
                    mirror: decode_layout(egraph, &value.layout.eclass).map_err(|err| {
                        anyhow!(
                            "{who}: decoding the layout of value {} (layout class {}): {err}",
                            value.eclass,
                            value.layout.eclass
                        )
                    })?,
                    dtype: value.dtype_enum,
                };
                cache.insert(key, decoded.clone());
                decoded
            }
        };
        table.insert(value.eclass.clone(), decoded);
        Ok(())
    };
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => decode(&input.value, &mut table)?,
            ExtractedNode::LayoutOp(op) => {
                for output in &op.outputs {
                    decode(output, &mut table)?;
                }
            }
            ExtractedNode::BufferOutput(_) => {}
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(v: i64) -> IntExprTerm {
        IntExprTerm::Lit(v)
    }
    fn coord(axis_from_end: i64) -> IntExprTerm {
        IntExprTerm::Coord { axis_from_end }
    }
    fn add(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Add(Box::new(a), Box::new(b))
    }
    fn mul(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Mul(Box::new(a), Box::new(b))
    }
    fn w32() -> BitWidthTerm {
        BitWidthTerm(32)
    }

    #[test]
    fn right_major_span_is_numel() {
        let layout = RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3), lit(5)]),
            width: w32(),
        };
        assert_eq!(layout.span(), mul(mul(lit(2), lit(3)), lit(5)));
    }

    #[test]
    fn left_major_span_is_numel_and_rank0_is_one() {
        let layout = LeftMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(4), lit(7)]),
            width: w32(),
        };
        assert_eq!(layout.span(), mul(lit(4), lit(7)));
        let scalar = LeftMajorContiguousElementLayout {
            shape: ShapeTerm(Vec::new()),
            width: w32(),
        };
        assert_eq!(scalar.span(), lit(1));
    }

    /// The chain's first canonical residue: `x*stride` — the summand
    /// `(IntMul (CoordVar _ axis) stride)` contributes
    /// `(extent-1)*stride` via coordinate substitution.
    #[test]
    fn strided_span_product_residue() {
        // shape (4,), chain [coord0 * 32]: span = 1 + (4-1)*32.
        let layout = StridedElementLayout {
            shape: ShapeTerm(vec![lit(4)]),
            chain: vec![mul(coord(0), lit(32))],
            width: w32(),
        };
        assert_eq!(
            layout.span(),
            add(lit(1), mul(add(lit(4), lit(-1)), lit(32)))
        );
    }

    /// The second canonical residue: the bare coordinate IS the
    /// stride-1 summand (x*1 subsumes its product node), contributing
    /// `extent-1`.
    #[test]
    fn strided_span_bare_coord_residue() {
        let layout = StridedElementLayout {
            shape: ShapeTerm(vec![lit(6)]),
            chain: vec![coord(0)],
            width: w32(),
        };
        assert_eq!(layout.span(), add(lit(1), add(lit(6), lit(-1))));
    }

    /// The third canonical residue: `(IntLit 0)` for a dead axis —
    /// substitution leaves it alone and it contributes zero.
    #[test]
    fn strided_span_zero_residue() {
        // shape (2, 3), inner axis dead (broadcast): chain from-end is
        // [0, coord1 * 3] — axis 1 (outer) strides by 3, axis 0 is dead.
        let layout = StridedElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            chain: vec![lit(0), mul(coord(1), lit(3))],
            width: w32(),
        };
        assert_eq!(
            layout.span(),
            add(add(lit(1), lit(0)), mul(add(lit(2), lit(-1)), lit(3)))
        );
    }

    /// Symbolic dims stay symbolic: no evaluation, no folding.
    #[test]
    fn spans_with_symbolic_dims() {
        let n = || IntExprTerm::Var("n".to_string());
        let contiguous = RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![n(), lit(128)]),
            width: w32(),
        };
        assert_eq!(contiguous.span(), mul(n(), lit(128)));

        // shape (n, 128), row-major strides spelled as a chain:
        // from-end [coord0, coord1 * 128].
        let strided = StridedElementLayout {
            shape: ShapeTerm(vec![n(), lit(128)]),
            chain: vec![coord(0), mul(coord(1), lit(128))],
            width: w32(),
        };
        assert_eq!(
            strided.span(),
            add(
                add(lit(1), add(lit(128), lit(-1))),
                mul(add(n(), lit(-1)), lit(128))
            )
        );
    }

    /// Mirror structs compare structurally — assertion convenience for
    /// runtime tests (the bufferizer's bound is `Clone + Debug` only;
    /// no planner check compares layouts).
    #[test]
    fn mirror_layout_equality_is_structural() {
        let a = MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            width: w32(),
        });
        let b = MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            width: w32(),
        });
        let c = MirrorLayout::LeftMajor(LeftMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            width: w32(),
        });
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
