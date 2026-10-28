use crate::prelude::*;
use std::ops::AddAssign;
use std::ops::DivAssign;
use std::ops::MulAssign;
use std::ops::RemAssign;
use std::ops::SubAssign;
use std::ops::{Add, Div, Mul, Rem, Sub};

impl Add for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        assert!(self.dims_agree(&rhs), "Dims must match to add tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to add tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().logical.op(
            LogicalOp::Add,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }
}

impl Add<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn add(self, rhs: GraphTensor) -> Self::Output {
        rhs + self
    }
}

impl<T> AddAssign<T> for GraphTensor
where
    GraphTensor: Add<T, Output = GraphTensor>,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl Sub for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl Sub<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn sub(self, rhs: GraphTensor) -> Self::Output {
        self + -rhs
    }
}

impl<T> SubAssign<T> for GraphTensor
where
    GraphTensor: Sub<T, Output = GraphTensor>,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl Mul for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        assert!(
            self.dims_agree(&rhs),
            "Dims must match to multiply tensors."
        );
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to multiply tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().logical.op(
            LogicalOp::Mul,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }
}

impl Mul<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn mul(self, rhs: GraphTensor) -> Self::Output {
        rhs * self
    }
}

impl<T> MulAssign<T> for GraphTensor
where
    GraphTensor: Mul<T, Output = GraphTensor>,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        if self.dtype == DType::F64 {
            // Binary64 division must round once. A reciprocal may overflow
            // even when the final quotient is finite (e.g. minsub/minsub).
            assert!(self.dims_agree(&rhs));
            assert_eq!(self.dtype, rhs.dtype);
            let dims = self.dims();
            let id = self.graph().logical.op(
                LogicalOp::Div,
                &[(self.id, dims.clone()), (rhs.id, dims.clone())],
                dims.clone(),
                self.dtype,
            );
            GraphTensor::from_id(id, dims, self.graph_ref, self.dtype)
        } else {
            self * rhs.reciprocal()
        }
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;
    fn div(self, rhs: GraphTensor) -> Self::Output {
        if rhs.dtype == DType::F64 {
            let lhs = rhs.graph().constant_f64(self as f64).expand_rhs(rhs.dims());
            lhs / rhs
        } else {
            self * rhs.reciprocal()
        }
    }
}

impl<T> DivAssign<T> for GraphTensor
where
    GraphTensor: Div<T, Output = GraphTensor>,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl Rem<GraphTensor> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: GraphTensor) -> Self::Output {
        assert!(self.dims_agree(&rhs), "Dims must match to mod tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to mod tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        // Mod is the FLOAT remainder (C `fmod`); every runtime implements
        // it that way. Integer remainder is `trunc_rem`, which carries the
        // nonzero-divisor proof Mod has no way to state — recording an
        // integer Mod would build a graph no backend can execute exactly.
        assert!(
            !matches!(
                self.dtype,
                DType::Int
                    | DType::I64
                    | DType::U64
                    | DType::I4
                    | DType::U4
                    | DType::I8
                    | DType::U8
                    | DType::I16
                    | DType::U16
                    | DType::Bool
            ),
            "`%` is the float remainder; use trunc_rem for {:?}",
            self.dtype
        );
        let new_id = self.graph().logical.op(
            LogicalOp::Mod,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }
}

impl<T> RemAssign<T> for GraphTensor
where
    GraphTensor: Rem<T, Output = GraphTensor>,
{
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

impl Add<f32> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: f32) -> Self::Output {
        self + self
            .graph()
            .constant_f32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self
            .graph()
            .constant_i32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self
            .graph()
            .constant_f32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self
            .graph()
            .constant_i32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_f32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self
            .graph()
            .constant_i32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        if self.dtype == DType::F64 {
            let divisor = self
                .graph()
                .constant_f64(rhs as f64)
                .expand_rhs(self.dims());
            return self / divisor;
        }
        self * self
            .graph()
            .constant_f32(rhs.recip())
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self
            .graph()
            .constant_i32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self
            .graph()
            .constant_f32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self
            .graph()
            .constant_i32(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

// Comparisons, all redurn bools (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    /// One strict/trunc binary op recording (typed-buffers landing D).
    fn int_binary(self, rhs: GraphTensor, op: LogicalOp) -> GraphTensor {
        let constructor = op.constructor();
        assert_eq!(self.dims(), rhs.dims(), "{constructor}: dims must match");
        assert_eq!(
            self.dtype, rhs.dtype,
            "{constructor}: operands must share one dtype"
        );
        assert!(
            matches!(self.dtype, DType::Int | DType::I64 | DType::U64),
            "{constructor} is an INTEGER op (got {:?})",
            self.dtype
        );
        let new_id = self.graph().logical.op(
            op,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            self.dtype,
        );
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, self.dtype)
    }

    /// Integer truncated division (toward zero) — proof-gated: implements
    /// only where the divisor's value bounds exclude zero (declare input
    /// ranges with `bind_value_range` when the divisor is caller data).
    pub fn trunc_div(self, rhs: GraphTensor) -> GraphTensor {
        self.int_binary(rhs, LogicalOp::TruncDiv)
    }

    /// Integer truncated remainder; see [`Self::trunc_div`].
    pub fn trunc_rem(self, rhs: GraphTensor) -> GraphTensor {
        self.int_binary(rhs, LogicalOp::TruncRem)
    }

    /// Less than comparison
    pub fn lt(self, rhs: GraphTensor) -> GraphTensor {
        assert!(self.dims_agree(&rhs), "Dims must match to lt tensors.");
        assert_eq!(
            self.dtype, rhs.dtype,
            "Dtypes must match to compare tensors. Got {:?} and {:?}",
            self.dtype, rhs.dtype
        );
        let new_id = self.graph().logical.op(
            LogicalOp::LessThan,
            &[(self.id, self.dims()), (rhs.id, rhs.dims())],
            self.dims(),
            DType::Bool,
        );
        // Comparison operations always output Bool
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, DType::Bool)
    }

    /// Greater than comparison
    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        rhs.lt(self)
    }

    /// IEEE NaN test: a `Bool` tensor, true exactly at the NaN elements.
    ///
    /// Built from `LessThan` alone, on the operand's own dtype. Every ordered
    /// value satisfies `x < +inf` or `-inf < x` (+inf fails only the first,
    /// -inf only the second, a finite value neither), and a NaN, being
    /// unordered, satisfies neither. So the count `[x < +inf] + [-inf < x]`
    /// is 0 exactly at NaN, and `count < 1` is the test. It needs no
    /// bit-slicing and no arithmetic in the operand dtype (so it holds for
    /// the storage-only narrow floats), and it is total on zero-element
    /// tensors.
    ///
    /// Integer, Bool and the NaN-free narrow formats (fp6, fp4) give all
    /// false. The fp8 formats do have NaN codes but are storage formats
    /// (some have no infinity to compare against): widen them first.
    pub fn is_nan(self) -> GraphTensor {
        match self.dtype {
            DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32 => {
                let ordered = self.lt(self.float_splat(f64::INFINITY)).cast(DType::F32)
                    + self
                        .float_splat(f64::NEG_INFINITY)
                        .lt(self)
                        .cast(DType::F32);
                none_of(ordered)
            }
            DType::Int
            | DType::I64
            | DType::U64
            | DType::I4
            | DType::U4
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::Bool
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4E2M1 => {
                let zero = self.graph().constant_f32(0.0).expand_rhs(self.dims());
                zero.lt(zero)
            }
            DType::F8UE8M0
            | DType::F8E4M3FN
            | DType::F8E4M3FNUZ
            | DType::F8E5M2
            | DType::F8E5M2FNUZ => panic!(
                "is_nan on {:?} is not supported: fp8 is a storage format, widen it first",
                self.dtype
            ),
        }
    }

    // THE DERIVED COMPARISONS. `LessThan` is the one comparison primitive and
    // it is IEEE `<`: false whenever either side is NaN. The derived forms
    // must also agree with IEEE (and torch) on unordered pairs, where `!=` is
    // true and `==`, `<=`, `>=` are false. A `!(b < a)` spelling of `a <= b`
    // is true on NaN, so it is correct only for dtypes without NaN, which keep
    // it unchanged. The NaN-carrying dtypes ([`has_ieee_nan`]) count the
    // unordered case explicitly:
    //   a != b  iff  any of   a < b, b < a, isnan(a), isnan(b)
    //   a == b  iff  none of  a < b, b < a, isnan(a), isnan(b)
    //   a <= b  iff  none of  b < a, isnan(a), isnan(b)
    //   a >= b  iff  none of  a < b, isnan(a), isnan(b)
    // Each condition is a sum of {0,1} F32 indicators of `LessThan` results,
    // kept numeric until ONE final comparison lands it in Bool (`any_of`,
    // `none_of`). No comparison is built by negating another's result (eq as
    // !ne): that adds a Bool -> F32 -> Bool round trip, forcing backends
    // without Bool storage to materialize an otherwise internal boolean
    // buffer. The isnan terms are themselves `count < 1` comparisons cast to
    // F32, one level deeper than the lt/gt terms; that is the depth the
    // ordered-only spelling already had inside its `cast(Bool)` projection.

    /// Less than or equal. IEEE `<=`: false when either side is NaN.
    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        if has_ieee_nan(self.dtype) {
            return none_of(self.gt(rhs).cast(DType::F32) + self.unordered_count(rhs));
        }
        (-self.gt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Greater than or equal. IEEE `>=`: false when either side is NaN.
    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        if has_ieee_nan(self.dtype) {
            return none_of(self.lt(rhs).cast(DType::F32) + self.unordered_count(rhs));
        }
        (-self.lt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Not equal. Returns `Bool`, like every other comparison here
    /// (`lt`/`gt`/`le`/`ge`); the indicator sum is internal, not the result
    /// type. IEEE `!=`: true when either side is NaN.
    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        if has_ieee_nan(self.dtype) {
            return any_of(self.inequality_count(rhs));
        }
        (self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32)).cast(DType::Bool)
    }

    /// Equal. IEEE `==`: false when either side is NaN.
    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        if has_ieee_nan(self.dtype) {
            return none_of(self.inequality_count(rhs));
        }
        // Keep the inequality indicator numeric until the final cast. Calling
        // `ne` here would create a Bool -> F32 round trip, forcing backends
        // without Bool storage to materialize an otherwise internal boolean
        // buffer.
        let not_equal = self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32);
        (-not_equal + 1.0).cast(DType::Bool)
    }

    /// `[a < b] + [b < a] + [isnan a] + [isnan b]`, an F32 count that is
    /// nonzero exactly when `a != b` under IEEE.
    fn inequality_count(self, rhs: GraphTensor) -> GraphTensor {
        self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32) + self.unordered_count(rhs)
    }

    /// `[isnan a] + [isnan b]`, an F32 count that is nonzero exactly when the
    /// pair is unordered.
    fn unordered_count(self, rhs: GraphTensor) -> GraphTensor {
        self.is_nan().cast(DType::F32) + rhs.is_nan().cast(DType::F32)
    }

    /// A scalar float literal in this tensor's own float dtype, expanded to
    /// its shape. F64 gets a binary64 constant; the narrower floats are casts
    /// of an F32 constant (exact for the values used here: +-inf).
    fn float_splat(self, value: f64) -> GraphTensor {
        match self.dtype {
            DType::F64 => self.graph().constant_f64(value),
            _ => self.graph().constant_f32(value as f32).cast(self.dtype),
        }
        .expand_rhs(self.dims())
    }

    /// Raise the tensor to a power.
    ///
    /// The general case is the real-valued approximation `exp(e * log|self|)`
    /// (see tinygrad's full impl), which reads the base through `abs` and so
    /// drops the sign of a negative base at every exponent. A compile-time
    /// scalar exponent that is finite and a whole number has exact
    /// multiplication semantics instead, so it is lowered structurally by
    /// exponentiation by squaring: odd powers keep the sign of a negative
    /// base, `x^0` is ones, `x^1` is `x`, and a negative exponent is the
    /// reciprocal of the positive power. That structural path is capped at
    /// `|e| <= 64`, beyond which the chain of multiplies is not worth its
    /// graph size and the approximation is used again; tensor exponents always
    /// take the approximation.
    pub fn pow<T: PowExponent>(self, e: T) -> GraphTensor {
        e.raise(self)
    }

    // Clipping ops (minimum, maximum, clip)

    /// Take the elementwise maximum of two tensors.
    ///
    /// Lowers to the native ternary `Select`, not the old arithmetic mask
    /// `(a<b)*b + (b<=a)*a`: a sum of products feeds the integer
    /// associativity/commutativity/distributivity e-graph closure, which
    /// explodes when such expressions chain. `Select` is also NaN-safe.
    ///
    /// `maximum` PROPAGATES NaN (unlike `fmax`): if either operand is NaN the
    /// result is NaN. `select(a < b, b, a)` already returns a NaN `a`, because
    /// every ordered comparison with NaN is false; it would drop only a NaN
    /// `b`, so one more `Select` returns `b` wherever `b` is NaN.
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        let picked = self.lt(rhs).select(rhs, self);
        rhs.nan_overrides(picked)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        // `constant_f32` always emits F32; cast it to `self.dtype` so the
        // downstream `lt`-based `maximum` doesn't panic when `self` is Int
        // (e.g. `aten.clamp` on Int top-k indices coming out of an MoE
        // router). For Int self the cast floors the bound, which matches
        // PyTorch's `clamp(int_tensor, min=<float>)` semantics.
        self.maximum(
            self.graph()
                .constant_f32(rhs)
                .cast(self.dtype)
                .expand_rhs(self.dims()),
        )
    }

    /// Take the elementwise minimum of two tensors. See [`Self::maximum`]
    /// for why this is a native select and how NaN propagates: here too
    /// `select(b < a, b, a)` keeps a NaN `a`, and only a NaN `b` needs the
    /// extra `Select`.
    pub fn minimum(self, rhs: GraphTensor) -> GraphTensor {
        let picked = self.gt(rhs).select(rhs, self);
        rhs.nan_overrides(picked)
    }

    /// `select(isnan(self), self, picked)` for NaN-carrying dtypes; `picked`
    /// unchanged otherwise.
    fn nan_overrides(self, picked: GraphTensor) -> GraphTensor {
        if has_ieee_nan(self.dtype) {
            self.is_nan().select(self, picked)
        } else {
            picked
        }
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn minimum_f32(self, rhs: f32) -> GraphTensor {
        self.minimum(
            self.graph()
                .constant_f32(rhs)
                .cast(self.dtype)
                .expand_rhs(self.dims()),
        )
    }

    /// Clip (clamp) a tensor into the range [`min`, `max`]
    pub fn clip(self, min: f32, max: f32) -> GraphTensor {
        self.maximum_f32(min).minimum_f32(max)
    }

    /// Return a tensor of elements selected from either self or other, depending on condition. Condition should be a boolean tensor
    pub fn cond(self, cond: GraphTensor, other: GraphTensor) -> GraphTensor {
        assert_eq!(
            self.dtype, other.dtype,
            "self and other need to be the same dtype!"
        );
        // A Bool condition lowers to the native ternary select; a numeric
        // condition keeps the arithmetic mask (its callers may pass {0,1}
        // values). The native path is NaN/inf-safe.
        if cond.dtype == DType::Bool {
            cond.select(self, other)
        } else {
            (cond.cast(self.dtype) * self)
                + ((1.0 - cond.cast(DType::F32)).cast(other.dtype) * other)
        }
    }

    /// Ternary select: `condition.select(if_true, if_false)`.
    ///
    /// The condition must be a `Bool` tensor; the two value branches must
    /// share a shape and dtype, and the output takes both. Unlike [`cond`],
    /// which blends with `c*a + (1-c)*b` and leaks NaN/inf through `0*inf`,
    /// this records one `LogicalSelect` node so each backend can lower it to
    /// a real boolean select.
    pub fn select(self, if_true: GraphTensor, if_false: GraphTensor) -> GraphTensor {
        assert_eq!(
            self.dtype,
            DType::Bool,
            "select condition must be a Bool tensor, got {:?}",
            self.dtype
        );
        assert_eq!(
            if_true.dtype, if_false.dtype,
            "select branches must share a dtype, got {:?} and {:?}",
            if_true.dtype, if_false.dtype
        );
        assert_eq!(
            if_true.dims(),
            if_false.dims(),
            "select branches must share a shape"
        );
        let new_id = self.graph().logical.op(
            LogicalOp::Select,
            &[
                (self.id, self.dims()),
                (if_true.id, if_true.dims()),
                (if_false.id, if_false.dims()),
            ],
            if_true.dims(),
            if_true.dtype,
        );
        GraphTensor::from_id(new_id, if_true.dims(), self.graph_ref, if_true.dtype)
    }
}

/// The largest `|exponent|` [`GraphTensor::pow`] lowers structurally. Past it
/// the multiply chain costs more graph than the approximation is worth.
const MAX_STRUCTURAL_POW: i64 = 64;

/// The exponent side of [`GraphTensor::pow`]: how a given exponent type raises
/// a base. Implemented for `f32` (structural when the scalar is integral) and
/// for `GraphTensor` (always the approximation).
pub trait PowExponent {
    /// Raise `base` to `self`.
    fn raise(self, base: GraphTensor) -> GraphTensor;
}

impl PowExponent for GraphTensor {
    fn raise(self, base: GraphTensor) -> GraphTensor {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        base.abs().log().mul(self).exp()
    }
}

impl PowExponent for f32 {
    fn raise(self, base: GraphTensor) -> GraphTensor {
        if self.is_finite() && self.fract() == 0.0 && self.abs() <= MAX_STRUCTURAL_POW as f32 {
            return integral_pow(base, self as i64);
        }
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        base.abs().log().mul(self).exp()
    }
}

/// `base ^ exponent` by exponentiation by squaring, which is exact and keeps
/// the sign of a negative base at odd exponents. The graph it builds is
/// logarithmic in `|exponent|`.
fn integral_pow(base: GraphTensor, exponent: i64) -> GraphTensor {
    if exponent == 0 {
        return base
            .graph()
            .constant_f32(1.0)
            .cast(base.dtype)
            .expand_rhs(base.dims());
    }
    let mut remaining = exponent.unsigned_abs();
    let mut factor = base;
    let mut result = None;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result = Some(match result {
                Some(value) => value * factor,
                None => factor,
            });
        }
        remaining >>= 1;
        if remaining > 0 {
            factor = factor * factor;
        }
    }
    let result = result.expect("a nonzero exponent sets at least one bit");
    if exponent < 0 {
        result.reciprocal()
    } else {
        result
    }
}

pub trait F32Pow {
    fn pow(self, e: GraphTensor) -> GraphTensor;
}

impl F32Pow for f32 {
    fn pow(self, e: GraphTensor) -> GraphTensor {
        e.mul(self.abs().ln()).exp()
    }
}

/// The dtypes whose values include IEEE NaN and that the frontend computes
/// in (the float sources of `GraphTensor::cast`). Their derived comparisons
/// and `maximum`/`minimum` treat a NaN operand as unordered.
fn has_ieee_nan(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
    )
}

/// Bool `0 < count` for a nonnegative F32 indicator count: any indicator set.
fn any_of(count: GraphTensor) -> GraphTensor {
    count
        .graph()
        .constant_f32(0.0)
        .expand_rhs(count.dims())
        .lt(count)
}

/// Bool `count < 1` for a nonnegative integral F32 indicator count: no
/// indicator set.
fn none_of(count: GraphTensor) -> GraphTensor {
    count.lt(count.graph().constant_f32(1.0).expand_rhs(count.dims()))
}

// #[cfg(test)]
#[cfg(test)]
pub(super) mod tests {
    use crate::tests::{assert_close, random_vec};
    use candle_core::{DType, Device, Tensor};
    use itertools::Itertools;
    use luminal::prelude::*;
    use proptest::prelude::*;

    pub fn identity(v: Vec<f32>) -> Vec<f32> {
        v
    }

    pub fn shift_from_zero(v: Vec<f32>) -> Vec<f32> {
        v.into_iter()
            .map(|x| if x >= 0.0 { x + 1.0 } else { x - 1.0 })
            .collect()
    }

    /// Strictly positive magnitudes: IEEE `pow` answers NaN for a negative
    /// base with a fractional exponent, where ours answers the documented
    /// abs-based approximation.
    pub fn positive_from_zero(v: Vec<f32>) -> Vec<f32> {
        v.into_iter().map(|x| x.abs() + 1.0).collect()
    }

    pub fn test_binary(
        a_shape: impl ToShape,
        b_shape: impl ToShape,
        func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
        ref_func: impl Fn(Tensor, Tensor) -> Tensor,
    ) {
        test_binary_transforms(a_shape, b_shape, func, ref_func, identity, identity);
    }

    pub fn test_binary_transforms(
        a_shape: impl ToShape,
        b_shape: impl ToShape,
        func: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
        ref_func: impl Fn(Tensor, Tensor) -> Tensor,
        lhs_transform: impl Fn(Vec<f32>) -> Vec<f32>,
        rhs_transform: impl Fn(Vec<f32>) -> Vec<f32>,
    ) {
        let a_shape = a_shape
            .to_shape()
            .into_iter()
            .map(|e| e.to_usize().unwrap())
            .collect_vec();
        let b_shape = b_shape
            .to_shape()
            .into_iter()
            .map(|e| e.to_usize().unwrap())
            .collect_vec();
        let mut cx = Graph::new();
        let a = cx.tensor(a_shape.clone(), luminal::dtype::DType::F32);
        let b = cx.tensor(b_shape.clone(), luminal::dtype::DType::F32);
        let c = func(a, b);

        let lhs_values = lhs_transform(random_vec(a_shape.iter().copied().product()));
        let rhs_values = rhs_transform(random_vec(b_shape.iter().copied().product()));
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (a.id, lhs_values.clone().into()),
                (b.id, rhs_values.clone().into()),
            ],
        );

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::from_vec(lhs_values, a_shape, &device).unwrap();
        let ref_b = Tensor::from_vec(rhs_values, b_shape, &device).unwrap();
        let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();

        assert_close(rt.get_f32(c.id).unwrap(), &ref_c.to_vec1::<f32>().unwrap())
    }

    #[test]
    #[should_panic(expected = "Dims must match to add tensors.")]
    fn test_add_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 3], luminal::dtype::DType::F32);
        let b = cx.tensor(vec![1, 3], luminal::dtype::DType::F32);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to multiply tensors.")]
    fn test_mul_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 3], luminal::dtype::DType::F32);
        let b = cx.tensor(vec![1, 3], luminal::dtype::DType::F32);
        let _ = a * b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to mod tensors.")]
    fn test_mod_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 3], luminal::dtype::DType::F32);
        let b = cx.tensor(vec![1, 3], luminal::dtype::DType::F32);
        let _ = a % b;
    }

    #[test]
    #[should_panic(expected = "`%` is the float remainder; use trunc_rem for Int64")]
    fn test_mod_rejects_integer_operands() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![3], luminal::dtype::DType::I64);
        let b = cx.tensor(vec![3], luminal::dtype::DType::I64);
        let _ = a % b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to lt tensors.")]
    fn test_lt_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 3], luminal::dtype::DType::F32);
        let b = cx.tensor(vec![1, 3], luminal::dtype::DType::F32);
        let _ = a.lt(b);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_add(x in 1..100, y in 1..5) {
            test_binary(vec![x], vec![x], |a, b| a + b, |a, b| (&a + &b).unwrap());
            test_binary(vec![y, x], vec![y, x], |a, b| a + b, |a, b| (&a + &b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_sub(x in 1..100, y in 1..5) {
            test_binary(vec![x], vec![x], |a, b| a - b, |a, b| (&a - &b).unwrap());
            test_binary(vec![y, x], vec![y, x], |a, b| a - b, |a, b| (&a - &b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mul(x in 1..100, y in 1..5) {
            test_binary(vec![x], vec![x], |a, b| a * b, |a, b| (&a * &b).unwrap());
            test_binary(
                vec![2, y, x],
                vec![2, y, x],
                |a, b| a * b,
                |a, b| (&a * &b).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_div(x in 1..100) {
            test_binary_transforms(
                vec![x],
                vec![x],
                |a, b| a / b,
                |a, b| (&a / &b).unwrap(),
                identity,
                shift_from_zero,
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_maximum(x in 1..100) {
            test_binary(vec![x], vec![x], |a, b| a.maximum(b), |a, b| a.maximum(&b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_minimum(x in 1..100) {
            test_binary(vec![x], vec![x], |a, b| a.minimum(b), |a, b| a.minimum(&b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mod(size in 1usize..64) {
            test_binary_transforms(
                vec![size],
                vec![size],
                |a, b| a % b,
                |a, b| {
                    let lhs = a.to_vec1::<f32>().unwrap();
                    let rhs = b.to_vec1::<f32>().unwrap();
                    let remainder: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(x, y)| x % y).collect();
                    Tensor::from_vec(remainder, size, &Device::Cpu).unwrap()
                },
                identity,
                shift_from_zero,
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mod_scalar_broadcast(size in 1usize..64) {
            // rank-0 RHS expanded against rank-N LHS, mirroring `x % torch.tensor(c)`.
            test_binary_transforms(
                vec![size],
                vec![] as Vec<usize>,
                |a, b| a % b.expand_rhs(a.dims()),
                |a, b| {
                    let lhs = a.to_vec1::<f32>().unwrap();
                    let rhs_scalar = b.to_scalar::<f32>().unwrap();
                    let remainder: Vec<f32> = lhs.iter().map(|x| x % rhs_scalar).collect();
                    Tensor::from_vec(remainder, size, &Device::Cpu).unwrap()
                },
                identity,
                shift_from_zero,
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_lt(size in 1usize..64) {
            test_binary(
                vec![size],
                vec![size],
                |a, b| a.lt(b).cast(luminal::dtype::DType::F32),
                |a, b| a.lt(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_lt_scalar_broadcast(size in 1usize..64) {
            // rank-0 RHS expanded against rank-N LHS for `lt`.
            test_binary(
                vec![size],
                vec![] as Vec<usize>,
                |a, b| a.lt(b.expand_rhs(a.dims())).cast(luminal::dtype::DType::F32),
                |a, b| {
                    let scalar = b.to_scalar::<f32>().unwrap();
                    let lhs = a.to_vec1::<f32>().unwrap();
                    let result: Vec<f32> = lhs
                        .iter()
                        .map(|x| if *x < scalar { 1.0f32 } else { 0.0f32 })
                        .collect();
                    Tensor::from_vec(result, size, &Device::Cpu).unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_gt(size in 1usize..64) {
            test_binary(
                vec![size],
                vec![size],
                |a, b| a.gt(b).cast(luminal::dtype::DType::F32),
                |a, b| a.gt(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_le(size in 1usize..64) {
            test_binary(
                vec![size],
                vec![size],
                |a, b| a.le(b).cast(luminal::dtype::DType::F32),
                |a, b| a.le(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_ge(size in 1usize..64) {
            test_binary(
                vec![size],
                vec![size],
                |a, b| a.ge(b).cast(luminal::dtype::DType::F32),
                |a, b| a.ge(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    #[test]
    fn test_ne() {
        test_binary(
            vec![27],
            vec![27],
            |a, b| {
                let result = a.ne(b);
                assert_eq!(result.dtype, luminal::dtype::DType::Bool);
                result.cast(luminal::dtype::DType::F32)
            },
            |a, b| a.ne(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_eq() {
        test_binary(
            vec![27],
            vec![27],
            |a, b| a.eq(b).cast(luminal::dtype::DType::F32),
            |a, b| a.eq(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_pow() {
        test_binary_transforms(
            vec![27],
            vec![27],
            |a, _| a.pow(2.5f32),
            |a, _| a.powf(2.5f64).unwrap(),
            positive_from_zero,
            identity,
        );
    }

    /// Run `func` over one fixed input vector, the way `test_binary_transforms`
    /// does, but against an expected vector rather than a candle tensor —
    /// candle's `powf` is IEEE `pow`, which answers NaN exactly where the
    /// sign-preserving cases below need a number.
    fn run_pow(values: Vec<f32>, func: impl Fn(GraphTensor) -> GraphTensor) -> Vec<f32> {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![values.len()], luminal::dtype::DType::F32);
        let b = func(a);
        // `pow(x, 1)` folds to the input itself, which is no leaf: bind the
        // result explicitly whatever it turned out to be.
        let bindings = luminal_reference::ReferenceBindings::dense(&cx.logical, &[b.id]);
        let rt = luminal_reference::harness::run_reference_bound(
            &cx,
            bindings,
            &[(a.id, values.into())],
            &[],
        );
        rt.get_f32(b.id).unwrap().to_vec()
    }

    #[test]
    fn test_pow_integral_scalar_exponent_keeps_sign() {
        let input = vec![-2.0f32, -1.0, -0.5, 0.5, 1.0, 2.0];
        // (-2)^3 = -8, (-2)^2 = 4, (-2)^0 = 1, x^1 = x, (-2)^-1 = -0.5.
        let cases: Vec<(f32, Vec<f32>)> = vec![
            (3.0, input.iter().map(|x| x * x * x).collect()),
            (2.0, input.iter().map(|x| x * x).collect()),
            (0.0, vec![1.0; input.len()]),
            (1.0, input.clone()),
            (-1.0, input.iter().map(|x| 1.0 / x).collect()),
        ];
        for (exponent, expected) in cases {
            assert_close(&run_pow(input.clone(), |a| a.pow(exponent)), &expected);
        }
    }

    #[test]
    fn test_pow_non_integral_scalar_exponent_keeps_the_approximation() {
        let input = vec![-2.0f32, -1.5, 0.5, 1.5];
        // The abs-based approximation, sign dropped, is still what non-whole
        // exponents get.
        let expected = input.iter().map(|x| x.abs().powf(2.5)).collect::<Vec<_>>();
        assert_close(&run_pow(input, |a| a.pow(2.5f32)), &expected);
    }

    #[test]
    fn test_clip() {
        test_binary_transforms(
            vec![27],
            vec![27],
            |a, _| a.clip(-0.25, 0.25),
            |a, _| a.clamp(-0.25, 0.25).unwrap(),
            identity,
            identity,
        );
    }

    #[test]
    fn test_maximum_f32() {
        test_binary_transforms(
            vec![27],
            vec![27],
            |a, _| a.maximum_f32(0.1),
            |a, _| {
                a.maximum(&Tensor::new(vec![0.1f32; 27], &Device::Cpu).unwrap())
                    .unwrap()
            },
            identity,
            identity,
        );
    }

    #[test]
    fn test_minimum_f32() {
        test_binary_transforms(
            vec![27],
            vec![27],
            |a, _| a.minimum_f32(-0.1),
            |a, _| {
                a.minimum(&Tensor::new(vec![-0.1f32; 27], &Device::Cpu).unwrap())
                    .unwrap()
            },
            identity,
            identity,
        );
    }

    #[test]
    fn test_cond() {
        test_binary(
            vec![27],
            vec![27],
            |a, b| {
                // gt() returns Bool, cast to F32 for cond which expects F32
                let cond = a
                    .gt(b.graph().constant_f32(0.0).expand_rhs(a.dims()))
                    .cast(luminal::dtype::DType::F32);
                a.cond(cond, b)
            },
            |a, b| {
                let refer = a.gt(&Tensor::zeros_like(&a).unwrap()).unwrap();
                refer.where_cond(&a, &b).unwrap()
            },
        );
    }

    // ---- IEEE NaN semantics of the comparisons and of maximum/minimum ----

    type LuminalDType = luminal::dtype::DType;

    /// Every ordered pair over NaN, both infinities, both zeros and two
    /// finite values, as (lhs, rhs) columns.
    fn special_pairs() -> (Vec<f64>, Vec<f64>) {
        let specials = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1.5,
            -2.0,
        ];
        specials
            .iter()
            .flat_map(|&a| specials.iter().map(move |&b| (a, b)))
            .unzip()
    }

    fn staged(values: &[f64], dtype: LuminalDType) -> luminal_reference::TypedBuffer {
        use luminal_reference::TypedBuffer;
        match dtype {
            LuminalDType::F32 => TypedBuffer::F32(values.iter().map(|&v| v as f32).collect()),
            LuminalDType::F64 => TypedBuffer::F64(values.to_vec()),
            LuminalDType::F16 => {
                TypedBuffer::F16(values.iter().map(|&v| half::f16::from_f64(v)).collect())
            }
            LuminalDType::Bf16 => {
                TypedBuffer::Bf16(values.iter().map(|&v| half::bf16::from_f64(v)).collect())
            }
            other => panic!("no staging for {other:?}"),
        }
    }

    type Comparison = fn(GraphTensor, GraphTensor) -> GraphTensor;

    /// Runs each comparison on its own copy of the special pairs (distinct
    /// inputs, so no two outputs share an e-class) and reads the Bool rows.
    fn run_comparisons(dtype: LuminalDType, comparisons: &[Comparison]) -> Vec<Vec<bool>> {
        let (lhs, rhs) = special_pairs();
        let mut cx = Graph::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for compare in comparisons {
            let a = cx.tensor(vec![lhs.len()], dtype);
            let b = cx.tensor(vec![rhs.len()], dtype);
            let out = compare(a, b);
            assert_eq!(out.dtype, LuminalDType::Bool);
            inputs.push((a.id, staged(&lhs, dtype)));
            inputs.push((b.id, staged(&rhs, dtype)));
            outputs.push(out.id);
        }
        let bindings = luminal_reference::ReferenceBindings::dense(&cx.logical, &outputs);
        let rt = luminal_reference::harness::run_reference_bound(&cx, bindings, &inputs, &[]);
        outputs
            .iter()
            .map(|&id| {
                rt.get_bool8(id)
                    .unwrap()
                    .iter()
                    .map(|&code| code == 1)
                    .collect()
            })
            .collect()
    }

    /// `LessThan` is IEEE `<`, so the derived comparisons used to be wrong on
    /// unordered pairs: `!(b < a)` made `NaN <= x` true and `!(a<b | b<a)`
    /// made `NaN == x` true. They must agree with IEEE (and torch) for every
    /// float dtype the frontend computes in, the storage-only narrow floats
    /// included.
    #[test]
    fn comparisons_are_ieee_on_unordered_pairs() {
        type FloatCase = (&'static str, Comparison, fn(f64, f64) -> bool);
        let comparisons: [FloatCase; 7] = [
            ("lt", |a, b| a.lt(b), |a, b| a < b),
            ("gt", |a, b| a.gt(b), |a, b| a > b),
            ("le", |a, b| a.le(b), |a, b| a <= b),
            ("ge", |a, b| a.ge(b), |a, b| a >= b),
            ("eq", |a, b| a.eq(b), |a, b| a == b),
            ("ne", |a, b| a.ne(b), |a, b| a != b),
            ("is_nan", |a, _| a.is_nan(), |a, _| a.is_nan()),
        ];
        let (lhs, rhs) = special_pairs();
        for dtype in [
            LuminalDType::F32,
            LuminalDType::F64,
            LuminalDType::F16,
            LuminalDType::Bf16,
        ] {
            let builders: Vec<Comparison> = comparisons.iter().map(|c| c.1).collect();
            let rows = run_comparisons(dtype, &builders);
            for ((name, _, expected), row) in comparisons.iter().zip(&rows) {
                for ((&a, &b), &got) in lhs.iter().zip(&rhs).zip(row) {
                    assert_eq!(got, expected(a, b), "{name} on {dtype:?} at ({a}, {b})");
                }
            }
        }
    }

    /// Integer comparisons keep their NaN-free spelling and stay exact, and
    /// `is_nan` of an integer tensor is all false.
    #[test]
    fn integer_comparisons_and_is_nan_are_unchanged() {
        let values = [i32::MIN, -7, -1, 0, 1, 7, i32::MAX];
        let (lhs, rhs): (Vec<i32>, Vec<i32>) = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .unzip();
        type IntCase = (Comparison, fn(i32, i32) -> bool);
        let comparisons: [IntCase; 5] = [
            (|a, b| a.le(b), |a, b| a <= b),
            (|a, b| a.ge(b), |a, b| a >= b),
            (|a, b| a.eq(b), |a, b| a == b),
            (|a, b| a.ne(b), |a, b| a != b),
            (|a, _| a.is_nan(), |_, _| false),
        ];
        let mut cx = Graph::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (compare, _) in &comparisons {
            let a = cx.tensor(vec![lhs.len()], LuminalDType::Int);
            let b = cx.tensor(vec![rhs.len()], LuminalDType::Int);
            outputs.push(compare(a, b).id);
            inputs.push((a.id, lhs.clone().into()));
            inputs.push((b.id, rhs.clone().into()));
        }
        let bindings = luminal_reference::ReferenceBindings::dense(&cx.logical, &outputs);
        let rt = luminal_reference::harness::run_reference_bound(&cx, bindings, &inputs, &[]);
        for ((_, expected), &id) in comparisons.iter().zip(&outputs) {
            let row = rt.get_bool8(id).unwrap();
            for ((&a, &b), &code) in lhs.iter().zip(&rhs).zip(row) {
                assert_eq!(code == 1, expected(a, b), "({a}, {b})");
            }
        }
    }

    fn same_value(actual: f32, expected: f32) -> bool {
        (actual.is_nan() && expected.is_nan()) || actual == expected
    }

    /// `maximum`/`minimum` propagate a NaN from EITHER side (torch semantics,
    /// unlike `fmax`/`fmin`). The old guard tested `x != x`, which the
    /// ordered-only `ne` made false for NaN, so `maximum(1, NaN)` was 1 while
    /// `maximum(NaN, 1)` was NaN.
    #[test]
    fn maximum_and_minimum_propagate_nan_from_either_side() {
        let (lhs, rhs) = special_pairs();
        let lhs: Vec<f32> = lhs.iter().map(|&v| v as f32).collect();
        let rhs: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut cx = Graph::new();
        let a1 = cx.tensor(vec![lhs.len()], LuminalDType::F32);
        let b1 = cx.tensor(vec![rhs.len()], LuminalDType::F32);
        let a2 = cx.tensor(vec![lhs.len()], LuminalDType::F32);
        let b2 = cx.tensor(vec![rhs.len()], LuminalDType::F32);
        let max = a1.maximum(b1);
        let min = a2.minimum(b2);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (a1.id, lhs.clone().into()),
                (b1.id, rhs.clone().into()),
                (a2.id, lhs.clone().into()),
                (b2.id, rhs.clone().into()),
            ],
        );
        let max_row = rt.get_f32(max.id).unwrap();
        let min_row = rt.get_f32(min.id).unwrap();
        for (i, (&a, &b)) in lhs.iter().zip(&rhs).enumerate() {
            let nan = a.is_nan() || b.is_nan();
            let (want_max, want_min) = if nan {
                (f32::NAN, f32::NAN)
            } else {
                (a.max(b), a.min(b))
            };
            assert!(
                same_value(max_row[i], want_max),
                "maximum({a}, {b}) = {}",
                max_row[i]
            );
            assert!(
                same_value(min_row[i], want_min),
                "minimum({a}, {b}) = {}",
                min_row[i]
            );
        }
    }

    /// A NaN clamp bound makes every element NaN (torch.clamp(x, min=nan)),
    /// and a NaN input stays NaN under finite bounds.
    #[test]
    fn clip_propagates_nan_bounds_and_nan_inputs() {
        let values = vec![
            f32::NAN,
            f32::NEG_INFINITY,
            -1.5,
            -0.0,
            0.5,
            2.0,
            f32::INFINITY,
        ];
        type ClampCase = (fn(GraphTensor) -> GraphTensor, fn(f32) -> f32);
        let clamps: [ClampCase; 5] = [
            (
                |x| x.clip(-1.0, 1.0),
                |v| if v.is_nan() { v } else { v.clamp(-1.0, 1.0) },
            ),
            (|x| x.clip(f32::NAN, 1.0), |_| f32::NAN),
            (|x| x.clip(-1.0, f32::NAN), |_| f32::NAN),
            (|x| x.maximum_f32(f32::NAN), |_| f32::NAN),
            (|x| x.minimum_f32(f32::NAN), |_| f32::NAN),
        ];
        let mut cx = Graph::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for (clamp, _) in &clamps {
            let x = cx.tensor(vec![values.len()], LuminalDType::F32);
            outputs.push(clamp(x).id);
            inputs.push((x.id, values.clone().into()));
        }
        let rt = luminal_reference::harness::run_reference(&cx, &inputs);
        for (case, ((_, expected), &id)) in clamps.iter().zip(&outputs).enumerate() {
            let row = rt.get_f32(id).unwrap();
            for (&v, &got) in values.iter().zip(row) {
                assert!(
                    same_value(got, expected(v)),
                    "clamp case {case} at {v}: {got}"
                );
            }
        }
    }
}
