//! THE `Layout` SORT, mirrored: its five constructors as Rust structs,
//! the facts they disclose, and the decoded spelling set a plan carries.
//!
//! THE BUFFERIZER NEVER CALLS ANY OF THIS. Living in core does not make
//! this a core vocabulary: the bufferizer stays generic over an opaque
//! layout type it only clones and transports, and nothing in the planner
//! imports this module. It exists so runtimes can pull the constructor
//! structs, the [`SpanExpr`] trait and [`DecodedLayout`] from one place
//! instead of each respelling them; a backend that wants a different
//! layout vocabulary brings its own type and ignores this module
//! entirely — nothing here is a closed set the planner depends on.
//!
//! Contents:
//!  * [`IntExprTerm`] / [`ShapeTerm`] / [`BitWidthTerm`] — the term
//!    vocabulary the constructor fields are spelled in;
//!  * the five constructor structs, field-for-field with the preamble's
//!    constructors (`RightMajorContiguousElementLayoutLit(Shape,
//!    BitWidth)` and friends), each implementing
//!    [`crate::egglog_utils::eclass::EgglogConstructor`] (decode ONE
//!    e-node of that constructor) and [`LayoutFacts`] (what it
//!    discloses);
//!  * [`SpanExpr`] — the span-as-EXPRESSION trait, implemented ONLY
//!    where a span is honest (the packed element ladder: right-major,
//!    left-major, strided). The offset-expression forms deliberately do
//!    NOT implement it: an offset function alone does not disclose its
//!    reach, and nothing here guesses;
//!  * [`layout_decoders`] — the five `(sort, constructor)` decoders core
//!    registers, and [`shape_term`] / [`bit_width`] / [`affine_chain`] /
//!    [`int_expr`], the term decoders they are written in terms of;
//!  * [`DecodedLayout`] — one elected value's layout: EVERY registered
//!    spelling its class holds, plus the value's dtype fact, plus the
//!    class id for diagnostics.
//!
//! THERE IS NO PREFERENCE ORDER. A class holds every spelling the
//! e-graph proved of it, all denoting one function; a caller asks for
//! the spelling IT can lower (`first::<C>()`, `has::<C>()`,
//! `require::<C>(who)`) and states its own preference at its own call
//! site. A class none of whose spellings parse is a loud error, never a
//! guess.

use crate::egglog_utils::eclass::{
    ConstructorDecoder, DynFacts, EClass, EGraphView, ENode, EgglogConstructor, Sort, Spellings,
};
use anyhow::{Result, anyhow, bail, ensure};
use egraph_serialize::ClassId;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// supplies no replacement for is a violated decode invariant and
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
    /// Panics on an out-of-range axis — a violated decode invariant.
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
// The five constructor structs (field-for-field with the preamble)
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

    /// Evaluate at concrete coordinates (FRONT-indexed: `Coord {
    /// axis_from_end }` reads `coords[rank - 1 - axis_from_end]`).
    /// Symbolic vars, out-of-rank axes and zero divisors refuse loudly —
    /// runtime-side convenience, never planner machinery.
    pub fn eval_at(&self, coords: &[usize]) -> Result<i64> {
        let rank = coords.len();
        Ok(match self {
            IntExprTerm::Lit(v) => *v,
            IntExprTerm::Var(name) => {
                bail!("layout read: symbolic dim `{name}` cannot evaluate")
            }
            IntExprTerm::Coord { axis_from_end } => {
                let axis = usize::try_from(*axis_from_end)
                    .ok()
                    .filter(|&a| a < rank)
                    .ok_or_else(|| anyhow!("coordinate axis {axis_from_end} out of rank {rank}"))?;
                coords[rank - 1 - axis] as i64
            }
            IntExprTerm::Add(a, b) => a.eval_at(coords)? + b.eval_at(coords)?,
            IntExprTerm::Mul(a, b) => a.eval_at(coords)? * b.eval_at(coords)?,
            IntExprTerm::TruncDiv(a, b) => {
                let (a, b) = (a.eval_at(coords)?, b.eval_at(coords)?);
                ensure!(b != 0, "division by zero in a layout expression");
                // Rust's `/` on i64 IS truncation toward zero.
                a / b
            }
            IntExprTerm::TruncRem(a, b) => {
                let (a, b) = (a.eval_at(coords)?, b.eval_at(coords)?);
                ensure!(b != 0, "remainder by zero in a layout expression");
                a % b
            }
            IntExprTerm::CeilDiv(a, b) => {
                let (a, b) = (a.eval_at(coords)?, b.eval_at(coords)?);
                ensure!(
                    b > 0,
                    "ceil-div by non-positive divisor in a layout expression"
                );
                a.div_euclid(b) + i64::from(a.rem_euclid(b) != 0)
            }
            IntExprTerm::Min(a, b) => a.eval_at(coords)?.min(b.eval_at(coords)?),
            IntExprTerm::Max(a, b) => a.eval_at(coords)?.max(b.eval_at(coords)?),
            IntExprTerm::LessThanCast(a, b) => i64::from(a.eval_at(coords)? < b.eval_at(coords)?),
        })
    }
}

// =============================================================================
// THE `Layout` SORT — its erased fact surface and its five constructors
// =============================================================================

/// The egglog `Layout` sort, as [`Sort`] names it. `Layout::Facts` is
/// [`LayoutFacts`], so `spellings::<Layout>()` hands generic code the
/// facts every layout constructor discloses without anyone naming a
/// constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout;

impl Sort for Layout {
    const NAME: &'static str = "Layout";
    type Facts = dyn LayoutFacts;
    /// The sort's downcast door: `dyn LayoutFacts` upcasts to `dyn Any`
    /// because `Any` is a supertrait of `DynFacts` (Rust 1.86+).
    fn upcast_any(facts: &Self::Facts) -> &dyn std::any::Any {
        facts
    }
}

/// WHAT EVERY LAYOUT CONSTRUCTOR DISCLOSES. Object-safe (no generics,
/// every method `&self`), so `dyn LayoutFacts` is the erased item type
/// `Spellings<Layout>` carries.
///
/// This is a FACT surface, never a classification: nothing here says
/// "which kind of layout is this". A caller that needs a particular
/// spelling names it (`first::<C>()`); a caller that needs a fact about
/// a layout it did not author asks here.
pub trait LayoutFacts: DynFacts {
    /// The DOMAIN (every constructor carries one).
    fn shape(&self) -> &ShapeTerm;

    /// The element access width in bits.
    fn width(&self) -> BitWidthTerm;

    /// Storage reach in ELEMENTS as an expression, where the constructor
    /// DISCLOSES one (the packed ladder, [`SpanExpr`]). `None` for the
    /// offset-expression forms — an offset function alone does not
    /// disclose its reach and nothing here guesses.
    fn span_elements(&self) -> Option<IntExprTerm>;

    /// The constructor's read function evaluated at literal coordinates
    /// (front-indexed), down to the flat ELEMENT index. RUNTIME
    /// convenience, never planner machinery. Fail-closed on symbolic
    /// extents, foreign rank, out-of-domain coordinates, a mid-element
    /// bit offset, and a negative result.
    fn element_index(&self, coords: &[usize]) -> Result<usize>;
}

impl PartialEq for dyn LayoutFacts {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other as &dyn std::any::Any)
    }
}
impl Eq for dyn LayoutFacts {}

/// The domain extents as literals — `None` if any axis is symbolic.
fn literal_extents_of(shape: &ShapeTerm) -> Option<Vec<usize>> {
    shape
        .0
        .iter()
        .map(|e| e.eval_literal().and_then(|v| usize::try_from(v).ok()))
        .collect()
}

/// The domain check every `element_index` runs first: literal extents,
/// matching rank, every coordinate inside its extent.
fn check_domain(shape: &ShapeTerm, coords: &[usize]) -> Result<Vec<usize>> {
    let extents =
        literal_extents_of(shape).ok_or_else(|| anyhow!("layout read: symbolic extents"))?;
    ensure!(
        coords.len() == extents.len(),
        "{} coordinates for a rank-{} layout",
        coords.len(),
        extents.len()
    );
    for (axis, (&c, &d)) in coords.iter().zip(&extents).enumerate() {
        ensure!(c < d, "coordinate {c} out of extent {d} (axis {axis})");
    }
    Ok(extents)
}

fn element_of(flat: i64) -> Result<usize> {
    usize::try_from(flat).map_err(|_| anyhow!("negative element index {flat}"))
}

impl EgglogConstructor for RightMajorContiguousElementLayout {
    const NAME: &'static str = "RightMajorContiguousElementLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let shape = shape_term(&node.child_or_bail(0)?)
            .ok_or_else(|| anyhow!("child 0 is not a decodable Shape"))?;
        let width = bit_width(&node.child_or_bail(1)?)
            .ok_or_else(|| anyhow!("child 1 is not a decodable BitWidth"))?;
        Ok(Self { shape, width })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

impl LayoutFacts for RightMajorContiguousElementLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    fn span_elements(&self) -> Option<IntExprTerm> {
        Some(self.span())
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        let extents = check_domain(&self.shape, coords)?;
        element_of(
            coords
                .iter()
                .zip(&extents)
                .fold(0usize, |acc, (&c, &d)| acc * d + c) as i64,
        )
    }
}

impl EgglogConstructor for LeftMajorContiguousElementLayout {
    const NAME: &'static str = "LeftMajorContiguousElementLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let shape = shape_term(&node.child_or_bail(0)?)
            .ok_or_else(|| anyhow!("child 0 is not a decodable Shape"))?;
        let width = bit_width(&node.child_or_bail(1)?)
            .ok_or_else(|| anyhow!("child 1 is not a decodable BitWidth"))?;
        Ok(Self { shape, width })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

impl LayoutFacts for LeftMajorContiguousElementLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    fn span_elements(&self) -> Option<IntExprTerm> {
        Some(self.span())
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        let extents = check_domain(&self.shape, coords)?;
        let mut stride = 1usize;
        let mut acc = 0usize;
        for (&c, &d) in coords.iter().zip(&extents) {
            acc += c * stride;
            stride *= d;
        }
        element_of(acc as i64)
    }
}

impl EgglogConstructor for StridedElementLayout {
    const NAME: &'static str = "StridedElementLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let shape_class = node.child_or_bail(0)?;
        let chain_class = node.child_or_bail(1)?;
        let width_class = node.child_or_bail(2)?;
        let shape =
            shape_term(&shape_class).ok_or_else(|| anyhow!("child 0 is not a decodable Shape"))?;
        let width = bit_width(&width_class)
            .ok_or_else(|| anyhow!("child 2 is not a decodable BitWidth"))?;
        // THE OWNER-SHAPE GUARD: a coordinate owned by any OTHER shape
        // is not one of this layout's coordinates and fails the spelling.
        let chain = affine_chain(&chain_class, &shape_class)
            .ok_or_else(|| anyhow!("child 1 is not a decodable affine chain over this domain"))?;
        Ok(Self {
            shape,
            chain,
            width,
        })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

impl LayoutFacts for StridedElementLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    fn span_elements(&self) -> Option<IntExprTerm> {
        Some(self.span())
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        check_domain(&self.shape, coords)?;
        let mut total = 0i64;
        for summand in &self.chain {
            total += summand.eval_at(coords)?;
        }
        element_of(total)
    }
}

impl EgglogConstructor for ElementOffsetExpressionLayout {
    const NAME: &'static str = "ElementOffsetExpressionLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let offset_class = node.child_or_bail(0)?;
        let shape_class = node.child_or_bail(1)?;
        let width_class = node.child_or_bail(2)?;
        let shape =
            shape_term(&shape_class).ok_or_else(|| anyhow!("child 1 is not a decodable Shape"))?;
        let width = bit_width(&width_class)
            .ok_or_else(|| anyhow!("child 2 is not a decodable BitWidth"))?;
        let offset = int_expr(&offset_class, Some(&shape_class))
            .ok_or_else(|| anyhow!("child 0 is not a decodable offset expression"))?;
        Ok(Self {
            offset,
            shape,
            width,
        })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

impl LayoutFacts for ElementOffsetExpressionLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    /// An offset function alone does not disclose its reach.
    fn span_elements(&self) -> Option<IntExprTerm> {
        None
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        check_domain(&self.shape, coords)?;
        element_of(self.offset.eval_at(coords)?)
    }
}

impl EgglogConstructor for BitOffsetExpressionLayout {
    const NAME: &'static str = "BitOffsetExpressionLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let offset_class = node.child_or_bail(0)?;
        let shape_class = node.child_or_bail(1)?;
        let width_class = node.child_or_bail(2)?;
        let shape =
            shape_term(&shape_class).ok_or_else(|| anyhow!("child 1 is not a decodable Shape"))?;
        let width = bit_width(&width_class)
            .ok_or_else(|| anyhow!("child 2 is not a decodable BitWidth"))?;
        let offset = int_expr(&offset_class, Some(&shape_class))
            .ok_or_else(|| anyhow!("child 0 is not a decodable offset expression"))?;
        Ok(Self {
            offset,
            shape,
            width,
        })
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

impl LayoutFacts for BitOffsetExpressionLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    fn span_elements(&self) -> Option<IntExprTerm> {
        None
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        check_domain(&self.shape, coords)?;
        let bits = self.offset.eval_at(coords)?;
        ensure!(self.width.0 > 0, "non-positive bit width {}", self.width.0);
        ensure!(
            bits % self.width.0 == 0,
            "bit offset {bits} is not element-aligned to width {}",
            self.width.0
        );
        element_of(bits / self.width.0)
    }
}

/// The core preamble's `Layout` constructors and their decoders, IN THIS
/// ORDER. It is the order `Spellings<Layout>` iterates and therefore
/// what [`DecodedLayout::present`] prints — a LISTING order, never a
/// preference: no caller is obliged to take the first.
pub fn layout_decoders() -> Vec<ConstructorDecoder> {
    vec![
        ConstructorDecoder::of::<RightMajorContiguousElementLayout>(),
        ConstructorDecoder::of::<LeftMajorContiguousElementLayout>(),
        ConstructorDecoder::of::<StridedElementLayout>(),
        ConstructorDecoder::of::<ElementOffsetExpressionLayout>(),
        ConstructorDecoder::of::<BitOffsetExpressionLayout>(),
    ]
}

// =============================================================================
// Term decoders — one e-class of the serialized graph into a term
// =============================================================================

/// Memo entry for the expression parse: finished, or the in-progress
/// cycle guard (the index_expr discipline — a cycle fails the SPELLING,
/// and a tainted failure is not cached because it is contextual).
enum ParseMemo {
    InProgress,
    Done(Option<IntExprTerm>),
}

/// `(BitWidthLit i64)`.
pub fn bit_width(class: &EClass<'_>) -> Option<BitWidthTerm> {
    for node in class.nodes_named("BitWidthLit") {
        if let Some(bits) = node.child(0).and_then(|c| c.i64_literal()) {
            return Some(BitWidthTerm(bits));
        }
    }
    None
}

/// `(ShapeLit IntExprList)` — one extent expression per axis, outermost
/// first.
pub fn shape_term(class: &EClass<'_>) -> Option<ShapeTerm> {
    for node in class.nodes_named("ShapeLit") {
        let Some(head) = node.child(0) else {
            continue;
        };
        let mut memo = HashMap::new();
        if let Some(extents) = expr_list(&head, "IntExprCons", "IntExprNil", 64, None, &mut memo) {
            return Some(ShapeTerm(extents));
        }
    }
    None
}

/// The strided chain: one summand per axis from-end. Coordinates are
/// guarded to the layout's OWN shape (a foreign shape's coordinate is
/// not this domain's and fails that spelling — the owner-shape guard).
pub fn affine_chain(class: &EClass<'_>, owner_shape: &EClass<'_>) -> Option<Vec<IntExprTerm>> {
    let mut memo = HashMap::new();
    expr_list(
        class,
        "IntAffineExprCons",
        "IntAffineExprNil",
        64,
        Some(owner_shape.id()),
        &mut memo,
    )
}

/// One `IntExpr` class into a term, optionally under the owner-shape
/// guard.
pub fn int_expr(class: &EClass<'_>, owner_shape: Option<&EClass<'_>>) -> Option<IntExprTerm> {
    let owner = owner_shape.map(|c| c.id().clone());
    let mut memo = HashMap::new();
    parse_int_expr(class, 64, owner.as_ref(), &mut memo)
}

/// Cons-spine walk, existential at every level (the backtracking
/// doctrine): a saturated list class holds several cons spellings and
/// the first may dead-end while a sibling parses fine.
fn expr_list(
    class: &EClass<'_>,
    cons_op: &str,
    nil_op: &str,
    depth: usize,
    owner_shape: Option<&ClassId>,
    memo: &mut HashMap<ClassId, ParseMemo>,
) -> Option<Vec<IntExprTerm>> {
    if depth == 0 {
        return None;
    }
    if class.nodes_named(nil_op).next().is_some() {
        return Some(Vec::new());
    }
    for cons in class.nodes_named(cons_op) {
        let Some(element) = cons.child(0) else {
            continue;
        };
        let Some(tail) = cons.child(1) else {
            continue;
        };
        let Some(expr) = parse_int_expr(&element, 64, owner_shape, memo) else {
            continue;
        };
        if let Some(mut rest) = expr_list(&tail, cons_op, nil_op, depth - 1, owner_shape, memo) {
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
    class: &EClass<'_>,
    depth: usize,
    owner_shape: Option<&ClassId>,
    memo: &mut HashMap<ClassId, ParseMemo>,
) -> Option<IntExprTerm> {
    parse_int_expr_tainting(class, depth, owner_shape, memo, &mut false)
}

fn parse_int_expr_tainting(
    class: &EClass<'_>,
    depth: usize,
    owner_shape: Option<&ClassId>,
    memo: &mut HashMap<ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IntExprTerm> {
    match memo.get(class.id()) {
        Some(ParseMemo::Done(cached)) => return cached.clone(),
        Some(ParseMemo::InProgress) => {
            *tainted = true;
            return None;
        }
        None => {}
    }
    memo.insert(class.id().clone(), ParseMemo::InProgress);
    let mut local_taint = false;
    let parsed = parse_int_expr_uncached(class, depth, owner_shape, memo, &mut local_taint);
    if parsed.is_none() && local_taint {
        memo.remove(class.id());
        *tainted = true;
    } else {
        memo.insert(class.id().clone(), ParseMemo::Done(parsed.clone()));
    }
    parsed
}

fn parse_int_expr_uncached(
    class: &EClass<'_>,
    depth: usize,
    owner_shape: Option<&ClassId>,
    memo: &mut HashMap<ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IntExprTerm> {
    if depth == 0 {
        return None;
    }
    if let Some(lit) = class.nodes_named("IntLit").next() {
        return Some(IntExprTerm::Lit(lit.child(0)?.i64_literal()?));
    }
    for var in class.nodes_named("IntVar") {
        let Some(name_class) = var.child(0) else {
            continue;
        };
        if let Some(name) = name_class.string_literal() {
            return Some(IntExprTerm::Var(name));
        }
    }
    for coord in class.nodes_named("CoordVar") {
        // Child 0 is the owner Shape, child 1 the axis (from-end).
        // The owner-shape guard: when the caller names the layout's
        // domain, a CoordVar owned by any OTHER shape is not one of
        // this layout's coordinates and cannot parse.
        if let Some(expected) = owner_shape {
            let Some(owner_class) = coord.child(0) else {
                continue;
            };
            if owner_class.id() != expected {
                continue;
            }
        }
        let Some(axis_class) = coord.child(1) else {
            continue;
        };
        return Some(IntExprTerm::Coord {
            axis_from_end: axis_class.i64_literal()?,
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
        for node in class.nodes_named(kind) {
            let Some(lhs_class) = node.child(0) else {
                continue;
            };
            let Some(rhs_class) = node.child(1) else {
                continue;
            };
            let Some(lhs) =
                parse_int_expr_tainting(&lhs_class, depth - 1, owner_shape, memo, tainted)
            else {
                continue;
            };
            let Some(rhs) =
                parse_int_expr_tainting(&rhs_class, depth - 1, owner_shape, memo, tainted)
            else {
                continue;
            };
            return Some(build(Box::new(lhs), Box::new(rhs)));
        }
    }
    for cast in class.nodes_named("IntCastFromBool") {
        let Some(bool_class) = cast.child(0) else {
            continue;
        };
        let Some(less_than) = bool_class.nodes_named("BoolLessThanInt").next() else {
            continue;
        };
        let Some(lhs_class) = less_than.child(0) else {
            continue;
        };
        let Some(rhs_class) = less_than.child(1) else {
            continue;
        };
        let Some(lhs) = parse_int_expr_tainting(&lhs_class, depth - 1, owner_shape, memo, tainted)
        else {
            continue;
        };
        let Some(rhs) = parse_int_expr_tainting(&rhs_class, depth - 1, owner_shape, memo, tainted)
        else {
            continue;
        };
        return Some(IntExprTerm::LessThanCast(Box::new(lhs), Box::new(rhs)));
    }
    None
}

// =============================================================================
// THE DECODED LAYOUT and its table — core's DECODER (ruling D9,
// 2026-09-03: "the core can have a decoder producing the layout struct
// from core; the runtimes should just directly import and use these
// layout structs"). The per-runtime `LayoutDecoder<L>` hook is GONE: it
// bought a genericity nobody exercised (both runtimes decoded the same
// decoded layout plus the same dtype fact), and the search that called it
// now lives in the runtimes anyway.
//
// THE BUFFERIZER STILL NEVER READS THIS. `bufferize` stays generic over
// an opaque `PlanLayout`; this is simply the layout type both shipped
// runtimes choose to instantiate it with.
// =============================================================================

/// The `class` a hand-built plan layout carries: fixtures have no
/// e-graph, and nothing may pin a class id anyway (2026-09-02 ruling —
/// ids are random every run).
pub const HAND_BUILT_CLASS: &str = "hand-built";

/// One elected value's decoded layout: EVERY registered `Layout`
/// spelling its e-class holds, plus the value's `dtype-of` fact.
///
/// The spelling SET, not one chosen spelling: all spellings of a layout
/// class denote ONE function, and which of them a consumer can lower is
/// that consumer's business (the cuBLASLt bias fence asks for the
/// LeftMajor spelling; the CUDA codegen asks for whichever it emits
/// simplest C for). Nothing here ranks them.
///
/// `dtype: None` is representable (a value with no `dtype-of` row) and
/// bails loudly at USE — staging, allocation typing, readback — never
/// silently.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedLayout {
    /// The layout e-class this was decoded from — DIAGNOSTICS and the
    /// cache key only; never pinned by a test (serialized ids are random
    /// every run). Hand-built plans carry [`HAND_BUILT_CLASS`].
    pub class: ClassId,
    pub dtype: Option<crate::dtype::PlanDtype>,
    /// Every registered `Layout` constructor present in the class,
    /// decoded, in registry order.
    pub spellings: Spellings<Layout>,
}

impl DecodedLayout {
    /// THE DECODER: every registered `Layout` spelling the class holds.
    ///
    /// Refuses a class with ZERO decoded spellings (the message lists
    /// what was named and why each failed), and a class whose decoded
    /// spellings DISAGREE on `shape()` or `width()` — layout identity is
    /// domain x interpretation, so a mixed-domain class is a false union
    /// and never a layout.
    pub fn from_class(class: &EClass<'_>, dtype: Option<crate::dtype::PlanDtype>) -> Result<Self> {
        let spellings = class.spellings::<Layout>();
        let Some(first) = spellings.any() else {
            if spellings.present().is_empty() {
                bail!(
                    "layout class {} has no registered Layout constructor spelling — \
                     nothing to decode (fail-closed, never a guess); ops in the class: {:?}",
                    class.id(),
                    class.ops()
                );
            }
            bail!(
                "layout class {} has constructor spellings {:?} but none parsed \
                 (fail-closed, never a guess): {:?}",
                class.id(),
                spellings.present(),
                spellings.failed()
            );
        };
        for other in spellings.iter() {
            if other.shape() != first.shape() || other.width() != first.width() {
                bail!(
                    "layout class {} unions spellings over DIFFERENT domains: `{}` over \
                     {:?}/{:?} and `{}` over {:?}/{:?}. Layout identity is domain x \
                     interpretation, so this class is a false union, not a layout — \
                     refused, never reconciled.",
                    class.id(),
                    first.constructor(),
                    first.shape(),
                    first.width(),
                    other.constructor(),
                    other.shape(),
                    other.width()
                );
            }
        }
        Ok(Self {
            class: class.id().clone(),
            dtype,
            spellings,
        })
    }

    /// A hand-built layout stating ONE spelling (test fixtures and plan
    /// literals).
    pub fn of<C: EgglogConstructor<Sort = Layout> + LayoutFacts>(
        spelling: C,
        dtype: Option<crate::dtype::PlanDtype>,
    ) -> Self {
        Self::of_spellings(vec![spelling.erase()], dtype)
    }

    /// A hand-built layout stating SEVERAL spellings of ONE function —
    /// the degenerate-extent fixture (a `[384, 1]` frame is both
    /// contiguous orders, and the e-graph puts both literals in one
    /// class).
    ///
    /// Panics on an empty list: a layout with no spelling is not a
    /// layout, and a fixture that writes one is a test bug, not a
    /// runtime refusal.
    pub fn of_spellings(
        spellings: Vec<Arc<dyn LayoutFacts>>,
        dtype: Option<crate::dtype::PlanDtype>,
    ) -> Self {
        assert!(
            !spellings.is_empty(),
            "a hand-built DecodedLayout states at least one spelling"
        );
        Self {
            class: ClassId::from(HAND_BUILT_CLASS),
            dtype,
            spellings: Spellings::from_decoded(spellings),
        }
    }

    // ---- class-INVARIANT facts (all spellings denote one function) ----

    /// The layout's DOMAIN shape.
    pub fn shape(&self) -> &ShapeTerm {
        self.facts().shape()
    }

    /// The element access width in bits.
    pub fn width_bits(&self) -> i64 {
        self.facts().width().0
    }

    /// The domain extents as literals — `None` if any axis is symbolic.
    pub fn literal_extents(&self) -> Option<Vec<usize>> {
        literal_extents_of(self.shape())
    }

    /// The storage reach in ELEMENTS, taken from the FIRST spelling that
    /// discloses one and evaluated. `None` when no spelling discloses a
    /// reach (the offset-expression forms) or the terms are symbolic —
    /// allocation-sizing callers bail loudly on `None`, never guess.
    pub fn literal_span_elements(&self) -> Option<usize> {
        self.spellings
            .iter()
            .find_map(|f| f.span_elements())
            .and_then(|span| span.eval_literal())
            .and_then(|v| usize::try_from(v).ok())
    }

    /// Element `coords` down to the flat ELEMENT index, read through the
    /// first spelling — every spelling of the class denotes the same
    /// function, so this answer is the class's.
    pub fn element_index(&self, coords: &[usize]) -> Result<usize> {
        self.facts().element_index(coords)
    }

    // ---- call-site preferences, delegated to the spelling set ----

    /// The decoded `C` spelling, if the class holds one.
    pub fn first<C: EgglogConstructor<Sort = Layout>>(&self) -> Option<&C> {
        self.spellings.first::<C>()
    }

    /// The class holds a decodable `C` spelling.
    pub fn has<C: EgglogConstructor<Sort = Layout>>(&self) -> bool {
        self.spellings.has::<C>()
    }

    /// [`DecodedLayout::first`] or a refusal naming `who` asked.
    pub fn require<C: EgglogConstructor<Sort = Layout>>(&self, who: &str) -> Result<&C> {
        self.spellings.require::<C>(who)
    }

    /// The constructor NAMES this layout's class holds, in registry
    /// order — what diagnostics print.
    pub fn present(&self) -> &[&'static str] {
        self.spellings.present()
    }

    /// The first spelling's facts. Every constructor of one class
    /// discloses the same domain and width (checked at decode), so the
    /// class-invariant readers above go through here.
    fn facts(&self) -> &dyn LayoutFacts {
        self.spellings
            .any()
            .expect("a DecodedLayout always holds at least one spelling")
    }
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
/// A search loop holds one for its whole run. A one-shot caller passes
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
    view: &EGraphView<'_>,
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
                let decoded =
                    DecodedLayout::from_class(&view.class(&value.layout.eclass), value.dtype_enum)
                        .map_err(|err| {
                            anyhow!(
                                "{who}: decoding the layout of value {} (layout class {}): {err}",
                                value.eclass,
                                value.layout.eclass
                            )
                        })?;
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
    use crate::egglog_utils::eclass::ConstructorRegistry;

    /// THE ASSEMBLY TRIPWIRE, over core's own preamble: every `Layout`
    /// constructor the program declares has exactly one decoder, and
    /// every decoder names a constructor the program declares.
    #[test]
    fn core_preamble_layout_constructors_all_have_decoders() {
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .expect("the core preamble parses");
        ConstructorRegistry::new(crate::egglog_snippet::core_decoders())
            .expect("core's decoders are unique")
            .check(&egraph)
            .expect("core declares exactly the five Layout constructors it decodes");
    }

    /// A CONSTRUCTOR WITH NO DECODER IS NAMED — the failure mode the
    /// tripwire exists for (someone adds a `Layout` constructor to the
    /// preamble and forgets the struct that reads it back).
    #[test]
    fn a_missing_decoder_is_named() {
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .expect("the core preamble parses");
        let short: Vec<ConstructorDecoder> = crate::egglog_snippet::core_decoders()
            .into_iter()
            .filter(|d| d.name != BitOffsetExpressionLayout::NAME)
            .collect();
        let err = ConstructorRegistry::new(short)
            .expect("still unique")
            .check(&egraph)
            .expect_err("a declared constructor with no decoder must be named");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(
                "sort Layout: constructor `BitOffsetExpressionLayoutLit` is declared by \
                 the program but has no registered decoder"
            ),
            "{msg}"
        );
    }

    /// ...AND THE OTHER DIRECTION: a decoder for a constructor the
    /// program does not declare is a stale registration, named too.
    #[test]
    fn a_stale_decoder_is_named() {
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .expect("the core preamble parses");
        let mut decoders = crate::egglog_snippet::core_decoders();
        decoders.push(ConstructorDecoder {
            sort: "Layout",
            name: "NoSuchLayoutLit",
            decode: |_| bail!("this constructor does not exist"),
        });
        let err = ConstructorRegistry::new(decoders)
            .expect("still unique")
            .check(&egraph)
            .expect_err("a decoder naming an undeclared constructor must be named");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(
                "decoder for `NoSuchLayoutLit` names a constructor the program does not declare"
            ),
            "{msg}"
        );
    }

    /// `layout-of` is a `Custom` function row whose OUTPUT sort is
    /// `Layout`. It is not a spelling and must not be demanded of the
    /// registry — the check above passing is that proof; this pins the
    /// row really is there and really is `Custom`.
    #[test]
    fn a_custom_row_over_a_decoded_sort_is_not_a_spelling() {
        use egglog::ast::FunctionSubtype;
        let mut egraph = crate::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &crate::egglog_snippet::assembled_program_for(&[]))
            .expect("the core preamble parses");
        let layout_of = egraph
            .get_function("layout-of")
            .expect("the preamble declares layout-of");
        assert_eq!(layout_of.func_type().output.name(), "Layout");
        assert_eq!(layout_of.func_type().subtype, FunctionSubtype::Custom);
    }

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

    /// A hand-built layout compares structurally on its SPELLING SET —
    /// assertion convenience for runtime tests (the bufferizer's bound
    /// is `Clone + Debug` only; no planner check compares layouts).
    #[test]
    fn decoded_layout_equality_is_structural() {
        let dtype = Some(crate::dtype::PlanDtype::F32);
        let rm = || RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            width: w32(),
        };
        let lm = || LeftMajorContiguousElementLayout {
            shape: ShapeTerm(vec![lit(2), lit(3)]),
            width: w32(),
        };
        let a = DecodedLayout::of(rm(), dtype);
        let b = DecodedLayout::of(rm(), dtype);
        let c = DecodedLayout::of(lm(), dtype);
        assert_eq!(a, b);
        assert_ne!(a, c);
        // A class holding BOTH orders is a THIRD value: same function,
        // more spellings, and the call sites can tell.
        let both = DecodedLayout::of_spellings(vec![rm().erase(), lm().erase()], dtype);
        assert_ne!(both, a);
        assert_ne!(both, c);
        assert!(both.has::<RightMajorContiguousElementLayout>());
        assert!(both.has::<LeftMajorContiguousElementLayout>());
        assert_eq!(
            both.present(),
            [
                "RightMajorContiguousElementLayoutLit",
                "LeftMajorContiguousElementLayoutLit"
            ]
        );
        // Class-invariant facts read the same through either.
        assert_eq!(both.literal_extents(), Some(vec![2, 3]));
        assert_eq!(both.width_bits(), 32);
        assert_eq!(both.literal_span_elements(), Some(6));
        // ...and the read function is the class's: the FIRST spelling.
        assert_eq!(both.element_index(&[1, 2]).unwrap(), 5);
    }

    /// The five constructors evaluate their own read functions, and the
    /// fail-closed cases refuse.
    #[test]
    fn element_index_reads_each_spelling() {
        let dtype = Some(crate::dtype::PlanDtype::F32);
        let shape = |dims: &[i64]| ShapeTerm(dims.iter().map(|&d| lit(d)).collect());
        let rm = DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: shape(&[2, 3]),
                width: w32(),
            },
            dtype,
        );
        assert_eq!(rm.element_index(&[1, 2]).unwrap(), 5);
        assert!(rm.element_index(&[2, 0]).is_err(), "out of domain");
        assert!(rm.element_index(&[0]).is_err(), "foreign rank");

        let lm = DecodedLayout::of(
            LeftMajorContiguousElementLayout {
                shape: shape(&[2, 3]),
                width: w32(),
            },
            dtype,
        );
        assert_eq!(lm.element_index(&[1, 2]).unwrap(), 5);

        let st = DecodedLayout::of(
            StridedElementLayout {
                shape: shape(&[3, 2]),
                chain: vec![mul(coord(0), lit(3)), coord(1)],
                width: w32(),
            },
            dtype,
        );
        assert_eq!(st.element_index(&[2, 1]).unwrap(), 5);

        let eo = DecodedLayout::of(
            ElementOffsetExpressionLayout {
                offset: add(mul(coord(1), lit(3)), coord(0)),
                shape: shape(&[2, 3]),
                width: w32(),
            },
            dtype,
        );
        assert_eq!(eo.element_index(&[1, 2]).unwrap(), 5);
        assert_eq!(eo.literal_span_elements(), None, "no disclosed reach");

        let bo = DecodedLayout::of(
            BitOffsetExpressionLayout {
                offset: mul(add(mul(coord(1), lit(3)), coord(0)), lit(32)),
                shape: shape(&[2, 3]),
                width: w32(),
            },
            dtype,
        );
        assert_eq!(bo.element_index(&[1, 2]).unwrap(), 5);
        let misaligned = DecodedLayout::of(
            BitOffsetExpressionLayout {
                offset: add(mul(coord(0), lit(32)), lit(8)),
                shape: shape(&[4]),
                width: w32(),
            },
            dtype,
        );
        assert!(
            misaligned.element_index(&[1]).is_err(),
            "a mid-element bit offset has no element read"
        );

        let symbolic = DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(vec![IntExprTerm::Var("n".into()), lit(3)]),
                width: w32(),
            },
            dtype,
        );
        assert!(symbolic.element_index(&[0, 0]).is_err());
        assert_eq!(symbolic.literal_extents(), None);
        assert_eq!(symbolic.literal_span_elements(), None);
    }
}
