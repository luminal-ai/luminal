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
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Add,
                &[(self.id, self.dims()), (rhs.id, rhs.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
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
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Mul,
                &[(self.id, self.dims()), (rhs.id, rhs.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
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
        self * rhs.reciprocal()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<GraphTensor> for f32 {
    type Output = GraphTensor;

    fn div(self, rhs: GraphTensor) -> Self::Output {
        self * rhs.reciprocal()
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
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::Mod,
                &[(self.id, self.dims()), (rhs.id, rhs.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
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
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_float(rhs.recip())
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self
            .graph()
            .constant_float(rhs)
            .cast(self.dtype)
            .expand_rhs(self.dims())
    }
}

impl<S: Into<IntExpr>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self
            .graph()
            .constant(rhs)
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
            matches!(self.dtype, DType::Int | DType::I64),
            "{constructor} is an INTEGER op (got {:?})",
            self.dtype
        );
        let new_id = self
            .graph()
            .logical
            .op(
                op,
                &[(self.id, self.dims()), (rhs.id, rhs.dims())],
                self.dims(),
                self.dtype,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
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
        let new_id = self
            .graph()
            .logical
            .op(
                LogicalOp::LessThan,
                &[(self.id, self.dims()), (rhs.id, rhs.dims())],
                self.dims(),
                DType::Bool,
            )
            .unwrap_or_else(crate::graph::unrecorded_value);
        // Comparison operations always output Bool
        GraphTensor::from_id(new_id, self.dims(), self.graph_ref, DType::Bool)
    }

    /// Greater than comparison
    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        rhs.lt(self)
    }

    /// Less than or equal
    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        (-self.gt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Greater than or equal
    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        (-self.lt(rhs).cast(DType::F32) + 1.0).cast(DType::Bool)
    }

    /// Not equal. Returns `Bool`, like every other comparison here
    /// (`lt`/`gt`/`le`/`ge`); the `lt + gt` sum is an internal numeric
    /// indicator, not the result type.
    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32)).cast(DType::Bool)
    }

    /// Equal
    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        // Keep the inequality indicator numeric until the final cast. Calling
        // `ne` here would create a Bool -> F32 round trip, forcing backends
        // without Bool storage to materialize an otherwise internal boolean
        // buffer.
        let not_equal = self.lt(rhs).cast(DType::F32) + self.gt(rhs).cast(DType::F32);
        (-not_equal + 1.0).cast(DType::Bool)
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

    /// Take the elementwise maximum of two tensors
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs).cast(self.dtype) * rhs) + (rhs.le(self).cast(self.dtype) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        // `constant_float` always emits F32; cast it to `self.dtype` so the
        // downstream `lt`/`le` comparisons inside `maximum` don't panic when
        // `self` is Int (e.g. `aten.clamp` on Int top-k indices coming out
        // of an MoE router). For Int self the cast floors the bound, which
        // matches PyTorch's `clamp(int_tensor, min=<float>)` semantics.
        self.maximum(
            self.graph()
                .constant_float(rhs)
                .cast(self.dtype)
                .expand_rhs(self.dims()),
        )
    }

    /// Take the elementwise minimum of two tensors
    pub fn minimum(self, rhs: GraphTensor) -> GraphTensor {
        -(-self).maximum(-rhs)
    }

    /// Take the elementwise minimum of a tensor and a float
    pub fn minimum_f32(self, rhs: f32) -> GraphTensor {
        -(-self).maximum_f32(-rhs)
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
        (cond.cast(self.dtype) * self) + ((1.0 - cond.cast(DType::F32)).cast(other.dtype) * other)
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
            .constant_float(1.0)
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
        let c = func(a, b).output();

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
        let a = cx.tensor((2, 3), luminal::dtype::DType::F32);
        let b = cx.tensor((1, 3), luminal::dtype::DType::F32);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to multiply tensors.")]
    fn test_mul_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3), luminal::dtype::DType::F32);
        let b = cx.tensor((1, 3), luminal::dtype::DType::F32);
        let _ = a * b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to mod tensors.")]
    fn test_mod_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3), luminal::dtype::DType::F32);
        let b = cx.tensor((1, 3), luminal::dtype::DType::F32);
        let _ = a % b;
    }

    #[test]
    #[should_panic(expected = "Dims must match to lt tensors.")]
    fn test_lt_rejects_implicit_broadcast() {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 3), luminal::dtype::DType::F32);
        let b = cx.tensor((1, 3), luminal::dtype::DType::F32);
        let _ = a.lt(b);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_add(x in 1..100, y in 1..5) {
            test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
            test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_sub(x in 1..100, y in 1..5) {
            test_binary(x, x, |a, b| a - b, |a, b| (&a - &b).unwrap());
            test_binary((y, x), (y, x), |a, b| a - b, |a, b| (&a - &b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mul(x in 1..100, y in 1..5) {
            test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
            test_binary(
                (2, y, x),
                (2, y, x),
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
                x,
                x,
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
            test_binary(x, x, |a, b| a.maximum(b), |a, b| a.maximum(&b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_minimum(x in 1..100) {
            test_binary(x, x, |a, b| a.minimum(b), |a, b| a.minimum(&b).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mod(size in 1usize..64) {
            test_binary_transforms(
                size,
                size,
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
                size,
                (),
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
                size,
                size,
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
                size,
                (),
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
                size,
                size,
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
                size,
                size,
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
                size,
                size,
                |a, b| a.ge(b).cast(luminal::dtype::DType::F32),
                |a, b| a.ge(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    #[test]
    fn test_ne() {
        test_binary(
            27,
            27,
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
            27,
            27,
            |a, b| a.eq(b).cast(luminal::dtype::DType::F32),
            |a, b| a.eq(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_pow() {
        test_binary_transforms(
            27,
            27,
            |a, _| a.pow(2.5f32),
            |a, _| a.powf(2.5f64).unwrap(),
            shift_from_zero,
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
        let b = func(a).output();
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, values.into())]);
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
            27,
            27,
            |a, _| a.clip(-0.25, 0.25),
            |a, _| a.clamp(-0.25, 0.25).unwrap(),
            identity,
            identity,
        );
    }

    #[test]
    fn test_maximum_f32() {
        test_binary_transforms(
            27,
            27,
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
            27,
            27,
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
            27,
            27,
            |a, b| {
                // gt() returns Bool, cast to F32 for cond which expects F32
                let cond = a
                    .gt(b.graph().constant_float(0.0).expand_rhs(a.dims()))
                    .cast(luminal::dtype::DType::F32);
                a.cond(cond, b)
            },
            |a, b| {
                let refer = a.gt(&Tensor::zeros_like(&a).unwrap()).unwrap();
                refer.where_cond(&a, &b).unwrap()
            },
        );
    }
}
