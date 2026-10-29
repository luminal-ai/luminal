//! Shared translator helpers, ported from the parked HLIR translator's
//! `pt2_util`/`mod` helper surface and re-expressed on the recorder
//! frontend (dims-based, no ShapeTracker).
//!
//! The surface is deliberately complete for the remaining category
//! ports, so some helpers are not yet called by this crate.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, DimSize, Node, SymIntEntry, TensorMeta};

/// A real scalar argument's literal value, in whichever spelling PT2
/// recorded it. A Python bool is the integer 0 or 1 (`False == 0`,
/// `True == 1`), so [`Self::is_zero`] and [`Self::is_one`] answer by value
/// for every literal spelling, not by which accessor happens to match.
/// A symbolic scalar (SymInt/SymFloat/SymBool) is not a literal: its value
/// is a runtime fact, so [`Self::of`] returns `None` for it.
#[derive(Clone, Copy, Debug)]
pub(super) enum ScalarLiteral {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl ScalarLiteral {
    pub(super) fn of(arg: &Argument) -> Option<Self> {
        if let Some(value) = arg.as_float() {
            return Some(Self::Float(value));
        }
        if let Some(value) = arg.as_int() {
            return Some(Self::Int(value));
        }
        arg.as_bool().map(Self::Bool)
    }

    /// The value equals 0. A negative zero is zero (`-0.0 == 0`); NaN is not.
    pub(super) fn is_zero(self) -> bool {
        match self {
            Self::Bool(value) => !value,
            Self::Int(value) => value == 0,
            Self::Float(value) => value == 0.0,
        }
    }

    /// The value equals 1.
    pub(super) fn is_one(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Int(value) => value == 1,
            Self::Float(value) => value == 1.0,
        }
    }
}

pub(super) fn same_dim(lhs: IntExpr, rhs: IntExpr) -> bool {
    lhs == rhs || lhs.simplify() == rhs.simplify() || lhs.egglog_equal(rhs)
}

/// Normalize a potentially negative dimension index.
pub(super) fn normalize_dim(dim: i64, ndim: usize) -> usize {
    if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    }
}

/// Normalize a possibly-negative slice bound against the axis extent.
pub(super) fn normalize_slice_bound(bound: IntExpr, dim_size: IntExpr) -> IntExpr {
    match bound.as_num() {
        Some(n) if n < 0 => (dim_size + IntExpr::from(n as i32)).simplify(),
        _ => bound,
    }
}

/// Right-align two tensors and expand size-1 axes (NumPy/torch broadcast).
pub(super) fn broadcast_binary(
    mut a: GraphTensor,
    mut b: GraphTensor,
) -> (GraphTensor, GraphTensor) {
    let (a_rank, b_rank) = (a.rank(), b.rank());
    for _ in 0..b_rank.saturating_sub(a_rank) {
        a = a.expand_dim(0, 1usize);
    }
    for _ in 0..a_rank.saturating_sub(b_rank) {
        b = b.expand_dim(0, 1usize);
    }
    let rank = a.rank();
    let ad = a.dims();
    let bd = b.dims();
    let mut a_target = Vec::with_capacity(rank);
    let mut b_target = Vec::with_capacity(rank);
    for i in 0..rank {
        if same_dim(ad[i], bd[i]) {
            a_target.push(ad[i]);
            b_target.push(bd[i]);
        } else if ad[i].to_usize() == Some(1) {
            a_target.push(bd[i]);
            b_target.push(bd[i]);
        } else if bd[i].to_usize() == Some(1) {
            a_target.push(ad[i]);
            b_target.push(ad[i]);
        } else {
            panic!("incompatible broadcast dims {i}: {} vs {}", ad[i], bd[i]);
        }
    }
    (a.expand(a_target), b.expand(b_target))
}

/// Torch's promotion lattice over two dtypes (wider wins; floats beat
/// ints; f16/bf16 mixed promotes to f32).
pub(super) fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    match (a, b) {
        (DType::F64, _) | (_, DType::F64) => DType::F64,
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        (DType::F16, DType::Bf16) | (DType::Bf16, DType::F16) => DType::F32,
        (DType::F16, _) | (_, DType::F16) => DType::F16,
        (DType::Bf16, _) | (_, DType::Bf16) => DType::Bf16,
        (DType::U64, _) | (_, DType::U64) => DType::U64,
        (DType::I64, _) | (_, DType::I64) => DType::I64,
        (DType::Int, _) | (_, DType::Int) => DType::Int,
        _ => DType::F32,
    }
}

/// Promote two tensors to their common dtype (see [`promote`]).
/// Narrow-float storage widened to F32 for computing; any other dtype is
/// returned as it is. The widening is exact: every narrow format is a
/// subset of F32. Narrow floats are storage in the translated graph (torch
/// computes them in F32), so a lowering widens them before any arithmetic.
pub(super) fn widen_storage(value: GraphTensor) -> GraphTensor {
    if value.dtype.is_narrow_float() {
        value.cast(DType::F32)
    } else {
        value
    }
}

/// `value` rounded once to `storage` when that is a narrow float (the
/// counterpart of [`widen_storage`]); otherwise `value` as it is.
pub(super) fn round_to_storage(value: GraphTensor, storage: DType) -> GraphTensor {
    if storage.is_narrow_float() && value.dtype != storage {
        value.cast(storage)
    } else {
        value
    }
}

/// One arithmetic step on storage values: `f` over the F32 widenings of
/// `a` and `b`, rounded once to `a`'s storage dtype. For a single add,
/// subtract, multiply or divide this is bit-identical to narrow-precision
/// arithmetic: F32 carries at least `2p + 2` bits for every narrow format
/// of precision `p`, so rounding through F32 lands where rounding once
/// would. Other dtypes apply `f` directly.
pub(super) fn storage_step(
    a: GraphTensor,
    b: GraphTensor,
    f: impl FnOnce(GraphTensor, GraphTensor) -> GraphTensor,
) -> GraphTensor {
    let storage = a.dtype;
    round_to_storage(f(widen_storage(a), widen_storage(b)), storage)
}

pub(super) fn ensure_same_dtype(a: GraphTensor, b: GraphTensor) -> (GraphTensor, GraphTensor) {
    if a.dtype == b.dtype {
        return (a, b);
    }
    let target = promote(a.dtype, b.dtype);
    (
        if a.dtype == target { a } else { a.cast(target) },
        if b.dtype == target { b } else { b.cast(target) },
    )
}

/// Reshape by view: flatten then split each leading target extent off the
/// remaining axis. The total element count must agree (torch view rules).
///
/// The construction below takes every extent it invents from that
/// element-count equality: with an unchanged prefix `P`,
/// `numel(P) * numel(rest of source) == numel(P) * numel(rest of target)`
/// cancels to `numel(rest of source) == numel(rest of target)`, which is
/// what makes each dropped axis extent 1 and each split exact. The
/// cancellation needs `numel(P) != 0`, so it holds for every tensor that
/// holds an element and fails for an empty one: `(0,3) -> (0,)` would
/// squeeze a size-3 axis, `(0,3) -> (3,0)` split by 0, and
/// `(0,) -> (0,1,5)` split 1 by 5. A tensor with a provably zero extent
/// therefore never enters it; [`reshape_empty`] lowers that view.
pub(super) fn reshape_tensor(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if t.dims().iter().any(|d| d.to_usize() == Some(0)) {
        return reshape_empty(t, target);
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    // Preserve the unchanged prefix. Flattening it and splitting it back
    // invents quotient/remainder expressions for axes that never moved,
    // obscuring ordinary per-head reshape/rotation patterns.
    let prefix = t
        .dims()
        .iter()
        .zip(target)
        .take_while(|(a, b)| a == b)
        .count();
    let mut flat = t.view();
    // The target IS the prefix: squeezing the remaining axes is valid iff
    // each has extent 1. The element count forces that (see above) once
    // the source is not provably empty. A static extent is checked here
    // by `squeeze`, which panics on anything but 1 (an invalid view). A
    // symbolic extent may be paired with a symbolic prefix that is 0 at
    // runtime, where torch accepts any extent; `squeeze` records the
    // `extent == 1` contract for it, so the binding's bounds must prove
    // the axis is 1 at saturation rather than this lowering assuming it.
    if prefix == target.len() {
        while flat.rank() > prefix {
            flat = flat.squeeze(prefix);
        }
        return flat.finish();
    }
    if flat.rank() == prefix {
        flat = flat.unsqueeze(prefix);
    }
    while flat.rank() > prefix + 1 {
        flat = flat.merge_dims(prefix, prefix + 1);
    }
    // `split_dims(axis, inner)` leaves `old / inner` at `axis` and inserts
    // `inner` directly after it, so each leading target dim is split off by
    // the product of the REMAINING dims; splitting by the leading dim itself
    // would reverse the target. `product_of_dims` canonicalises the operand
    // order, so a symbolic product compares equal across call sites.
    for i in prefix..target.len().saturating_sub(1) {
        let inner = super::dim_arith::product_of_dims(target[i + 1..].iter().copied());
        flat = flat.split_dims(i, inner);
    }
    flat.finish()
}

/// View of a tensor with a provably zero extent (see [`reshape_tensor`]).
/// It holds no element, so every index map from the target to the source
/// is a correct one, and the target must hold none either (torch's
/// element-count rule). Collapse the source to its single zero-extent
/// axis, place that axis at the target's first zero extent, and broadcast
/// every other target axis around it with `expand_dim`, which needs no
/// divisibility and no unit extent. A target with no static zero extent
/// cannot be stated with the source's zero axis, so it is refused.
fn reshape_empty(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    let source = t.dims();
    let Some(zero_axis) = target.iter().position(|d| d.to_usize() == Some(0)) else {
        panic!("view of the empty {source:?} as {target:?}: the target has no zero extent");
    };
    let mut flat = t.view();
    while flat.rank() > 1 {
        flat = flat.merge_dims(0, 1);
    }
    for (axis, &extent) in target.iter().enumerate() {
        if axis != zero_axis {
            flat = flat.expand_dim(axis, extent);
        }
    }
    flat.finish()
}

/// Value identity: `clone`, `alias` and the `*_copy` ops name a tensor
/// whose contents are the operand's. The recorder has no storage, so the
/// value IS the operand's; which storage a clone gets is the boundary's
/// statement (its own buffer id for a returned clone), and a view node
/// here would say the opposite — that the clone may share its operand's
/// storage — which the planner would then elect.
pub(super) fn materialize_tensor(t: GraphTensor) -> GraphTensor {
    t
}

/// Resolve `-1` in a reshape target against the input's logical dims.
pub(super) fn resolve_neg1_dim(target: &[i64], current_dims: &[IntExpr]) -> Vec<IntExpr> {
    let mut neg1_idx = None;
    let mut known_product: i64 = 1;
    let mut result = Vec::with_capacity(target.len());
    for (i, &s) in target.iter().enumerate() {
        if s == -1 {
            neg1_idx = Some(i);
            result.push(IntExpr::from(0usize));
        } else {
            known_product *= s;
            result.push(IntExpr::from(s as usize));
        }
    }
    if let Some(idx) = neg1_idx {
        let product: IntExpr = current_dims
            .iter()
            .fold(IntExpr::from(1), |acc, d| acc * *d);
        result[idx] = product / IntExpr::from(known_product as usize);
    }
    result
}

/// `-1` resolution for targets already carrying `IntExpr` extents.
pub(super) fn resolve_neg1_dim_exprs(target: &[IntExpr], current_dims: &[IntExpr]) -> Vec<IntExpr> {
    let neg1 = IntExpr::from(-1i32);
    let Some(idx) = target.iter().position(|e| *e == neg1) else {
        return target.to_vec();
    };
    let input_product: IntExpr = current_dims
        .iter()
        .fold(IntExpr::from(1), |acc, d| acc * *d);
    let known_product: IntExpr = target
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .fold(IntExpr::from(1), |acc, (_, d)| acc * *d);
    let mut result = target.to_vec();
    result[idx] = (input_product / known_product).simplify();
    result
}

/// The value a NaN-aware predicate (`is_nan`, `truth`) classifies: the fp8
/// storage formats widened to F32, everything else unchanged. Every fp8 code
/// widens exactly (NaN to NaN, infinities and subnormals kept). In their own
/// dtype neither form works: the fp8 encodings do not follow the IEEE NaN
/// rule (E4M3FN has a finite all-ones exponent, the FNUZ formats spell NaN
/// as the negative zero code), and their comparisons are the NaN-free
/// spelling.
fn widen_storage_float(value: GraphTensor) -> GraphTensor {
    match value.dtype {
        DType::F8E4M3FN | DType::F8E4M3FNUZ | DType::F8E5M2 | DType::F8E5M2FNUZ => {
            value.cast(DType::F32)
        }
        _ => value,
    }
}

impl Translator<'_> {
    /// A rank-0 constant of `dtype`; integer dtypes never route through a
    /// refused float -> int cast.
    pub(super) fn typed_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::U64 => {
                if value < 0.0 {
                    self.cx.constant_i64(value as i64).cast(DType::U64)
                } else {
                    self.cx.constant_u64(value as u64)
                }
            }
            DType::I64 => self.cx.constant_i64(if value.is_nan() {
                0
            } else if value == f64::INFINITY {
                i64::MAX
            } else if value == f64::NEG_INFINITY {
                i64::MIN
            } else {
                value as i64
            }),
            DType::I8 | DType::I16 | DType::U8 => {
                let (minimum, maximum) = match dtype {
                    DType::I8 => (i8::MIN as i64, i8::MAX as i64),
                    DType::I16 => (i16::MIN as i64, i16::MAX as i64),
                    DType::U8 => (u8::MIN as i64, u8::MAX as i64),
                    _ => unreachable!(),
                };
                let value = if value == f64::INFINITY {
                    maximum
                } else if value == f64::NEG_INFINITY {
                    minimum
                } else {
                    value as i64
                };
                self.cx.constant_i32(value).cast(dtype)
            }
            DType::Int => self.cx.constant_i32(if value.is_nan() {
                0
            } else if value == f64::INFINITY {
                i32::MAX as i64
            } else if value == f64::NEG_INFINITY {
                i32::MIN as i64
            } else {
                value as i64
            }),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        }
    }

    /// The Int indicator of a truncated remainder that disagrees with its
    /// divisor's sign: 1 where `rem < 0 < b` or `b < 0 < rem`, else 0.
    /// Python's `//` and `%` move the truncated quotient and remainder by
    /// one divisor exactly there (a zero remainder is never moved). Spelled
    /// as products of Int indicators, not as a comparison of Bools: the
    /// reference runtime has no Bool compare arm, and every operand here is
    /// bounded in [0, 1], so the arithmetic discharges its own proofs.
    pub(super) fn remainder_sign_disagreement(
        &mut self,
        rem: GraphTensor,
        divisor: GraphTensor,
    ) -> GraphTensor {
        let dtype = rem.dtype;
        let zero = self.constant_like(rem, 0.0);
        let rem_negative = rem.lt(zero).cast(dtype);
        let rem_positive = rem.gt(zero).cast(dtype);
        let divisor_negative = divisor.lt(zero).cast(dtype);
        let divisor_positive = divisor.gt(zero).cast(dtype);
        rem_negative * divisor_positive + rem_positive * divisor_negative
    }

    /// A typed scalar constant matching `tensor`'s dtype and dims.
    pub(super) fn constant_like(&mut self, tensor: GraphTensor, value: f64) -> GraphTensor {
        let scalar = self.typed_scalar(value, tensor.dtype);
        scalar.expand_rhs(tensor.dims())
    }

    /// `where(condition, a, b)` over already-broadcast operands.
    ///
    /// Uses the native ternary `LogicalSelect`: the old arithmetic mask
    /// `a*mask + b*(1-mask)` is a sum of products, which feeds the integer
    /// associativity/commutativity/distributivity e-graph closure and can
    /// explode saturation when chained; it is also NaN/inf-unsafe
    /// (`NaN * 0`).
    pub(super) fn select(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = broadcast_binary(a, condition);
        let (a, b) = broadcast_binary(a, b);
        // The false branch may grow the common shape after the condition
        // was aligned with the true branch (scalar predicates included).
        let (condition, _) = broadcast_binary(condition, a);
        let condition = if condition.dtype == DType::Bool {
            condition
        } else {
            condition.cast(DType::Bool)
        };
        condition.select(a, b)
    }

    /// PyTorch truthiness, `bool(x)`: `x != 0` under IEEE.
    ///
    /// Invariant: for floats this is true at every code except the two
    /// signed zeros, NaNs and subnormals included, so it must be a
    /// comparison that is neither ordered-only (an ordered `x != 0` reads
    /// NaN as false) nor lossy (a cast to F32 rounds F64 subnormals and
    /// tiny normals to zero). Core `ne` meets both: it counts the unordered
    /// case explicitly and compares in the operand's own dtype through
    /// `LessThan`, which never rounds. That is why this no longer reads the
    /// encoding: the bit count was the only NaN-correct spelling while `ne`
    /// was ordered-only. Integers compare natively (`x != 0`), exact at
    /// every width. The fp8 storage formats compare on their exact F32
    /// widening ([`widen_storage_float`]).
    pub(super) fn truth(&mut self, value: GraphTensor) -> GraphTensor {
        if value.dtype == DType::Bool {
            return value;
        }
        let value = widen_storage_float(value);
        let zero = self.constant_like(value, 0.0);
        value.ne(zero)
    }

    pub(super) fn bool_or(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = broadcast_binary(a, b);
        self.truth(a).logical_or(self.truth(b))
    }
    pub(super) fn bool_and(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = broadcast_binary(a, b);
        self.truth(a).logical_and(self.truth(b))
    }
    pub(super) fn bool_not(&mut self, value: GraphTensor) -> GraphTensor {
        self.truth(value).logical_not()
    }

    pub(super) fn real_abs(&mut self, value: GraphTensor) -> GraphTensor {
        value.abs()
    }

    /// IEEE finiteness depends only on whether every exponent bit is one.
    /// Inspect the encoding so subnormals and reduced-precision extrema never
    /// depend on the backend's floating arithmetic or scalar conversion.
    pub(super) fn is_finite(&mut self, value: GraphTensor) -> GraphTensor {
        let (start, end) = match value.dtype {
            DType::F16 => (10, 15),
            DType::Bf16 => (7, 15),
            DType::F32 | DType::TF32 => (23, 31),
            DType::F64 => (52, 63),
            _ => {
                return self
                    .cx
                    .constant_i32(1)
                    .cast(DType::Bool)
                    .expand_rhs(value.dims());
            }
        };
        let bits = value.bits_to_bools();
        let axis = bits.rank() - 1;
        let exponent = bits
            .slice_along(start..end, axis)
            .cast(DType::Int)
            .sum(vec![axis]);
        let width = self.cx.constant_i32(end - start).expand_rhs(value.dims());
        exponent.lt(width)
    }

    /// `torch.isnan`: a `Bool` tensor, true exactly at the NaN elements.
    ///
    /// Invariant: a code is NaN iff its exponent field is all ones and its
    /// significand is nonzero. Reading the encoding classifies every code
    /// exactly (subnormals are never flushed, no arithmetic runs in the
    /// operand dtype) and is total on zero-element tensors, whose sliced bit
    /// view spans zero storage. Integer and Bool inputs are all false. The
    /// fp8 storage formats do have NaN codes: they are classified on their
    /// exact F32 widening, which keeps every NaN a NaN.
    ///
    /// Core [`GraphTensor::is_nan`] (`[x < +inf] + [-inf < x] < 1`) is the
    /// same predicate, but routing this helper through it is held back: with
    /// both NaN tests of `logaddexp` (the `is_nan(y)` guard and the closing
    /// passthrough) in that form, the masked `logaddexp` graph extracts a
    /// select whose condition is its F32 operand (`masked.logaddexp`
    /// OpInfo sample 10), an equivalence violation to fix before switching.
    pub(super) fn is_nan(&mut self, value: GraphTensor) -> GraphTensor {
        let value = widen_storage_float(value);
        let (mantissa, end) = match value.dtype {
            DType::F16 => (10, 15),
            DType::Bf16 => (7, 15),
            DType::F32 | DType::TF32 => (23, 31),
            DType::F64 => (52, 63),
            _ => {
                return self
                    .cx
                    .constant_i32(0)
                    .cast(DType::Bool)
                    .expand_rhs(value.dims());
            }
        };
        let bits = value.bits_to_bools();
        let axis = bits.rank() - 1;
        let exp = bits
            .slice_along(mantissa..end, axis)
            .cast(DType::Int)
            .sum(vec![axis]);
        let width = self
            .cx
            .constant_i32(end - mantissa)
            .expand_rhs(value.dims());
        let all = exp.eq(width);
        let fraction = bits
            .slice_along(0..mantissa, axis)
            .cast(DType::Int)
            .sum(vec![axis]);
        let zero = self.cx.constant_i32(0).expand_rhs(value.dims());
        self.bool_and(all, fraction.gt(zero))
    }

    pub(super) fn is_inf(&mut self, value: GraphTensor) -> GraphTensor {
        let largest = match value.dtype {
            DType::F64 => f64::MAX,
            _ => f32::MAX as f64,
        };
        let positive = value.gt(self.constant_like(value, largest));
        let negative = value.lt(self.constant_like(value, -largest));
        self.bool_or(positive, negative)
    }

    /// torch's `x == 0`, elementwise: a `Bool` tensor that is true exactly
    /// at the zeros (both signed zeros, and integer or Bool zero).
    ///
    /// Invariant: NaN is not zero, and neither is any nonzero value however
    /// small. This is `ScalarLiteral::is_zero` for tensors. Callers that
    /// mirror a torch `== 0` or truth test (`bool`, `any`, `nonzero`, the
    /// order-0 norm's count) need exactly this. So does every guard that
    /// substitutes a value at zero (`f(0)` endpoints, safe denominators):
    /// a NaN must fall through to the general formula and propagate.
    /// The ordered spelling `!(x < 0 || x > 0)` reads NaN as zero, because
    /// both comparisons are false for NaN; the IEEE `eq` counts the
    /// unordered case. The test runs at a precision that keeps zero-ness
    /// exact. F64 stays F64, because narrowing to F32 flushes
    /// `|x| <= 2^-150` to zero. Every other dtype widens to F32, where no
    /// nonzero integer, Bool or narrower float rounds to zero and a NaN
    /// stays NaN.
    pub(super) fn is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let x = if value.dtype == DType::F64 {
            value
        } else {
            value.cast(DType::F32)
        };
        let zero = self.constant_like(x, 0.0);
        x.eq(zero)
    }

    /// The IEEE sign bit of a floating `value`, as `torch.signbit` reads it.
    ///
    /// Invariant: for every non-NaN `x`, the sign bit is set iff `x < 0`
    /// (each negative finite value and `-inf`) or `x` is `-0.0`. Ordered
    /// `<` cannot see a zero's sign, so the second clause reads it through
    /// the reciprocal: `1/-0.0 = -inf < 0` while `1/+0.0 = +inf`. Neither
    /// clause suffices alone: `-0.0 < 0` is false, and `1/-inf = -0.0` is
    /// not `< 0` either, so the reciprocal clause by itself reads `-inf`
    /// as positive.
    ///
    /// NaN: both clauses are ordered comparisons, so every NaN reads as
    /// positive. That matches torch for the default (positive) NaN, but not
    /// for a NaN whose sign bit is set, where torch reports `true`. Only the
    /// encoding view (`bits_to_bools`) can observe a NaN's sign, and that
    /// view does not execute on zero-element tensors yet, which this
    /// predicate (and `copysign`/`atan2` through it) must keep supporting.
    pub(super) fn signbit(&mut self, value: GraphTensor) -> GraphTensor {
        // A narrow float's sign reads off its exact F32 widening.
        let value = widen_storage(value);
        let zero = self.constant_like(value, 0.0);
        let negative = value.lt(zero);
        let negative_zero = value.reciprocal().lt(zero);
        negative.logical_or(negative_zero)
    }

    /// The IEEE absolute value of a floating `value`: its sign bit cleared,
    /// so `-0.0` becomes `+0.0`. `abs()` cannot see a zero's sign (it
    /// selects on the ordered `x < 0`) and keeps `-0.0`; this selects on
    /// [`Self::signbit`] instead. Every NaN reads as positive there, so a
    /// NaN passes through unchanged.
    pub(super) fn ieee_abs(&mut self, value: GraphTensor) -> GraphTensor {
        let storage = value.dtype;
        let value = widen_storage(value);
        let negative = self.signbit(value);
        round_to_storage(negative.select(-value, value), storage)
    }

    /// `torch.copysign(magnitude, sign)`: `|magnitude|` carrying the sign
    /// bit of `sign`. Every translator lowering that transfers a sign goes
    /// through this one helper.
    ///
    /// Invariant: the result's magnitude is exactly `|magnitude|` and its
    /// sign bit is exactly `signbit(sign)`, for zeros, subnormals and
    /// infinities in either operand. It is built by selection, never by
    /// multiplying by a sign indicator: `|m| * sign(s)` is `0` for a zero
    /// `s` and NaN for an infinite one (`0 * inf`).
    ///
    /// The magnitude goes through [`Self::ieee_abs`] rather than `abs()`:
    /// `abs(-0.0)` keeps its sign bit, and the two branches below would then
    /// carry the inverted sign (`copysign(-0.0, 1.0)` would be `-0.0`).
    ///
    /// NaN `sign`: `signbit` reads every NaN as positive (see
    /// [`Self::signbit`]), so the result is `+|magnitude|`. That is torch's
    /// answer for the default NaN, but not for a NaN whose sign bit is set.
    pub(super) fn copy_sign(&mut self, magnitude: GraphTensor, sign: GraphTensor) -> GraphTensor {
        let (magnitude, sign) = broadcast_binary(magnitude, sign);
        let positive = self.ieee_abs(magnitude);
        let negative = self.signbit(sign);
        negative.select(-positive, positive)
    }

    /// A floating scalar of the tensor's own dtype (F64 keeps doubles).
    pub(super) fn floating_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            _ => super::convert(self.cx.constant_f32(value as f32), dtype),
        }
    }

    pub(super) fn get_input_tensor(&mut self, node: &Node, idx: usize) -> Result<GraphTensor> {
        let input = node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?;
        self.operand(input)
    }

    pub(super) fn get_int_arg(&self, node: &Node, idx: usize) -> Result<i64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(v) = arg.as_int() {
            return Ok(v);
        }
        if let Some(expr) = self.resolve_arg_as_expression(arg)
            && let Some(v) = expr.to_usize()
        {
            return Ok(v as i64);
        }
        bail!("input {idx} of {} is not an int: {arg:?}", node.target)
    }

    pub(super) fn get_float_arg(&self, node: &Node, idx: usize) -> Result<f64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(f) = arg.as_float() {
            return Ok(f);
        }
        if let Some(i) = arg.as_int() {
            return Ok(i as f64);
        }
        bail!("input {idx} of {} is not a float: {arg:?}", node.target)
    }

    /// A numeric scalar argument that may be serialized as a float, int, or
    /// bool (bools coerce to 1/0). `fill_value` is the motivating case:
    /// `torch.full(shape, True)` serializes the fill as `as_bool`.
    pub(super) fn get_number_arg(&self, node: &Node, idx: usize) -> Result<f64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(f) = arg.as_float() {
            return Ok(f);
        }
        if let Some(i) = arg.as_int() {
            return Ok(i as f64);
        }
        if let Some(b) = arg.as_bool() {
            return Ok(if b { 1.0 } else { 0.0 });
        }
        bail!("input {idx} of {} is not a number: {arg:?}", node.target)
    }

    pub(super) fn get_ints_arg(&self, node: &Node, idx: usize) -> Result<Vec<i64>> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(entries) = arg.as_sym_ints() {
            let mut out = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    SymIntEntry::Int(i) => out.push(i.as_int),
                    SymIntEntry::Name(s) => {
                        let v = self
                            .resolve_sym_int(&s.as_name)
                            .and_then(|e| e.to_usize())
                            .map(|u| u as i64)
                            .with_context(|| {
                                format!("input {idx} of {} has an unresolved sym_int", node.target)
                            })?;
                        out.push(v);
                    }
                }
            }
            return Ok(out);
        }
        arg.as_ints()
            .map(|v| v.to_vec())
            .with_context(|| format!("input {idx} of {} is not an int list", node.target))
    }

    /// Like [`Self::get_ints_arg`], but keeps symbolic entries as `IntExpr`
    /// instead of collapsing them to their hint. A `sym_size` result (or a
    /// bare sym_int) resolves through `sym_int_values` to the recorder symbol
    /// that also names the tensor's own dimension.
    pub(super) fn get_int_exprs_arg(&self, node: &Node, idx: usize) -> Result<Vec<IntExpr>> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(entries) = arg.as_sym_ints() {
            return entries
                .iter()
                .map(|entry| match entry {
                    SymIntEntry::Int(i) => Ok(IntExpr::from(i.as_int)),
                    SymIntEntry::Name(s) => self.resolve_sym_int(&s.as_name).with_context(|| {
                        format!("input {idx} of {} has an unresolved sym_int", node.target)
                    }),
                })
                .collect();
        }
        arg.as_ints()
            .map(|values| values.iter().map(|v| IntExpr::from(*v)).collect())
            .with_context(|| format!("input {idx} of {} is not an int list", node.target))
    }

    pub(super) fn get_bool_arg(&self, node: &Node, idx: usize) -> Result<bool> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        arg.as_bool()
            .with_context(|| format!("input {idx} of {} is not a bool", node.target))
    }

    /// PT2 serializes arguments by schema name and may omit optional defaults.
    /// Resolve absent and explicit None identically without positional indexing.
    pub(super) fn named_tensor_operand(
        &mut self,
        node: &Node,
        name: &str,
    ) -> Result<Option<GraphTensor>> {
        match node.inputs.iter().find(|input| input.name == name) {
            Some(input) => self.optional_tensor_operand(input),
            None => Ok(None),
        }
    }

    pub(super) fn named_int_arg(&self, node: &Node, name: &str) -> Option<i64> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_int_arg(node, idx).ok()
    }

    pub(super) fn named_float_arg(&self, node: &Node, name: &str) -> Option<f64> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_float_arg(node, idx).ok()
    }

    pub(super) fn named_bool_arg(&self, node: &Node, name: &str) -> Option<bool> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_bool_arg(node, idx).ok()
    }

    /// All tensor output names of a node, multi-output ops included.
    pub(super) fn tensor_output_names(node: &Node) -> Vec<String> {
        node.outputs
            .iter()
            .flat_map(|output| {
                output
                    .as_tensors
                    .as_ref()
                    .map(|values| values.iter().map(|value| value.name.clone()).collect())
                    .unwrap_or_else(|| {
                        output
                            .as_tensor
                            .as_ref()
                            .map(|value| vec![value.name.clone()])
                            .unwrap_or_default()
                    })
            })
            .collect()
    }

    /// Shape of the node's first output from its PT2 metadata.
    pub(super) fn output_meta_shape(&self, node: &Node) -> Result<Vec<IntExpr>> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&name)
            .with_context(|| format!("missing tensor meta for output {name}"))?;
        self.tensor_meta_to_shape(meta)
    }

    /// Dtype of the node's first tensor output from its PT2 metadata.
    pub(super) fn output_meta_dtype(&self, node: &Node) -> Result<DType> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&name)
            .with_context(|| format!("missing tensor meta for output {name}"))?;
        super::dtype_of(meta.dtype)
    }

    pub(super) fn tensor_meta_to_shape(&self, meta: &TensorMeta) -> Result<Vec<IntExpr>> {
        meta.sizes
            .iter()
            .map(|s| self.dim_size_to_expr(s))
            .collect()
    }

    pub(super) fn dim_size_to_expr(&self, dim: &DimSize) -> Result<IntExpr> {
        match dim {
            DimSize::Int(i) => Ok(IntExpr::from(i.as_int)),
            DimSize::Expr(e) => self.resolve_expr_value(&e.as_expr).with_context(|| {
                format!("cannot resolve dimension expression {}", e.as_expr.expr_str)
            }),
        }
    }

    /// A full tensor of `value` at `shape`, using the exact-constant path
    /// per dtype (F64 keeps double precision).
    pub(super) fn full_tensor(
        &mut self,
        shape: Vec<IntExpr>,
        dtype: DType,
        value: f64,
    ) -> GraphTensor {
        let scalar = self.typed_scalar(value, dtype);
        scalar.expand_rhs(shape)
    }

    /// Coordinate arange along `axis`, expanded across the other axes.
    pub(super) fn axis_positions(&mut self, full_shape: &[IntExpr], axis: usize) -> GraphTensor {
        let mut positions = self.cx.arange(full_shape[axis]);
        for (dim, size) in full_shape.iter().copied().enumerate() {
            if dim != axis {
                positions = positions.expand_dim(dim, size);
            }
        }
        positions
    }
}

#[cfg(test)]
mod reshape_order_tests {
    use super::*;

    #[test]
    fn reshape_preserves_target_order() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![1usize, 4usize, 64usize], DType::F32);
        let out = reshape_tensor(x, &[IntExpr::from(4usize), IntExpr::from(64usize)]);
        assert_eq!(
            out.dims(),
            vec![IntExpr::from(4usize), IntExpr::from(64usize)],
            "reshape [1,4,64] -> [4,64] must not reverse its target"
        );
    }

    fn exprs(dims: &[usize]) -> Vec<IntExpr> {
        dims.iter().map(|&d| IntExpr::from(d)).collect()
    }

    /// Every target torch accepts for a zero-element source, including the
    /// ones the prefix/merge/split construction cannot state: a target that
    /// is a strict prefix of the source (`(0,3) -> (0,)` would squeeze a
    /// size-3 axis), a zero extent past the first target axis (`(0,3) ->
    /// (3,0)` would split by 0), and a unit extent inside the rest
    /// (`(0,) -> (0,1,5)` would split 1 by 5).
    #[test]
    fn reshape_of_an_empty_tensor_reaches_every_torch_target() {
        for (source, target) in [
            (&[0usize, 3][..], &[0usize][..]),
            (&[0, 3], &[3, 0]),
            (&[0, 6], &[0, 3, 2]),
            (&[2, 0], &[0]),
            (&[0], &[0, 1, 5]),
            (&[2, 0, 3], &[2, 0]),
            (&[3, 0], &[0, 3]),
            (&[0, 3], &[0, 1, 3]),
        ] {
            let mut cx = Graph::new();
            let x = cx.tensor(source.to_vec(), DType::F32);
            let out = reshape_tensor(x, &exprs(target));
            assert_eq!(out.dims(), exprs(target), "{source:?} -> {target:?}");
        }
    }

    /// A tensor with an element keeps the unchanged-prefix lowering: the
    /// prefix axes are never flattened, so squeezing unit axes off it
    /// (`[2,3,1] -> [2,3]`) or splitting the axis after it (`[4,8,16] ->
    /// [4,8,2,8]`, the per-head pattern) records no quotient or remainder.
    #[test]
    fn reshape_of_a_nonempty_tensor_keeps_its_prefix() {
        for (source, target, prefix_only) in [
            (&[2usize, 3, 1][..], &[2usize, 3][..], true),
            (&[2, 1, 1], &[2], true),
            (&[4, 8, 16], &[4, 8, 2, 8], true),
            (&[2, 3, 4], &[2, 12], false),
        ] {
            let mut cx = Graph::new();
            let x = cx.tensor(source.to_vec(), DType::F32);
            let out = reshape_tensor(x, &exprs(target));
            assert_eq!(out.dims(), exprs(target), "{source:?} -> {target:?}");
            if prefix_only {
                let text = cx.logical.render_all().unwrap();
                assert!(
                    !text.contains("IntTruncDiv") && !text.contains("IntTruncRem"),
                    "{source:?} -> {target:?} flattened its unchanged prefix:\n{text}"
                );
            }
        }
    }

    #[test]
    fn reshape_three_dims_stays_left_to_right() {
        let mut cx = Graph::new();
        let x = cx.tensor(vec![2usize, 12usize], DType::F32);
        let out = reshape_tensor(
            x,
            &[
                IntExpr::from(2usize),
                IntExpr::from(3usize),
                IntExpr::from(4usize),
            ],
        );
        assert_eq!(
            out.dims(),
            vec![
                IntExpr::from(2usize),
                IntExpr::from(3usize),
                IntExpr::from(4usize)
            ]
        );
    }
}
