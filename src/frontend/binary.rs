use crate::hlir::*;
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
        // assert_eq!(
        //     self.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     rhs.dims()
        //         .into_iter()
        //         .map(|i| i.simplify())
        //         .collect::<Vec<_>>(),
        //     "Dims must match to add tensors."
        // );
        let new_id = self
            .graph()
            .add_op(crate::hlir::Add::default())
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
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
        // assert_eq!(
        //     self.dims(),
        //     rhs.dims(),
        //     "Dims must match to multiply tensors."
        // );
        let new_id = self
            .graph()
            .add_op(crate::hlir::Mul::default())
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
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
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to mod tensors.");
        let new_id = self
            .graph()
            .add_op(Mod::default())
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
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
        self + self.graph().constant_float(rhs).expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Add<S> for GraphTensor {
    type Output = GraphTensor;

    fn add(self, rhs: S) -> Self::Output {
        self + self.graph().constant(rhs).expand_rhs(self.shape)
    }
}

impl Sub<f32> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: f32) -> Self::Output {
        self - self.graph().constant_float(rhs).expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Sub<S> for GraphTensor {
    type Output = GraphTensor;

    fn sub(self, rhs: S) -> Self::Output {
        self - self.graph().constant(rhs).expand_rhs(self.shape)
    }
}

impl Mul<f32> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: f32) -> Self::Output {
        self * self.graph().constant_float(rhs).expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Mul<S> for GraphTensor {
    type Output = GraphTensor;

    fn mul(self, rhs: S) -> Self::Output {
        self * self.graph().constant(rhs).expand_rhs(self.shape)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<f32> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: f32) -> Self::Output {
        self * self
            .graph()
            .constant_float(rhs.recip())
            .expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Div<S> for GraphTensor {
    type Output = GraphTensor;

    fn div(self, rhs: S) -> Self::Output {
        self / self
            .graph()
            .constant(rhs)
            .cast(self.dtype)
            .expand_rhs(self.shape)
    }
}

impl Rem<f32> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: f32) -> Self::Output {
        self % self.graph().constant_float(rhs).expand_rhs(self.shape)
    }
}

impl<S: Into<Expression>> Rem<S> for GraphTensor {
    type Output = GraphTensor;

    fn rem(self, rhs: S) -> Self::Output {
        self % self.graph().constant(rhs).expand_rhs(self.shape)
    }
}

// Comparisons (based on https://github.com/tinygrad/tinygrad/blob/3e0c2d256fe9f4f5f85cd3e4d8733a51d7b4a984/tinygrad/tensor.py#L653)
impl GraphTensor {
    pub fn lt(self, rhs: GraphTensor) -> GraphTensor {
        assert_eq!(self.dims(), rhs.dims(), "Dims must match to lt tensors.");
        let new_id = self
            .graph()
            .add_op(LessThan::default())
            .input(self.id, self.shape)
            .input(rhs.id, rhs.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    pub fn gt(self, rhs: GraphTensor) -> GraphTensor {
        rhs.lt(self)
    }

    pub fn le(self, rhs: GraphTensor) -> GraphTensor {
        -self.gt(rhs) + 1.0
    }

    pub fn ge(self, rhs: GraphTensor) -> GraphTensor {
        -self.lt(rhs) + 1.0
    }

    pub fn ne(self, rhs: GraphTensor) -> GraphTensor {
        self.lt(rhs) + self.gt(rhs)
    }

    pub fn eq(self, rhs: GraphTensor) -> GraphTensor {
        -self.ne(rhs) + 1.0
    }

    /// Raise the tensor to a power
    pub fn pow<T>(self, e: T) -> GraphTensor
    where
        Self: Mul<T, Output = Self>,
    {
        // Approximate, see full impl here: https://github.com/tinygrad/tinygrad/blob/a32c67760140dd26b60d7932268f2e62e96a66e0/tinygrad/tensor.py#L568
        self.abs().log().mul(e).exp()
    }
}

// Clipping ops (minimum, maximum, clip)
impl GraphTensor {
    /// Take the elementwise maximum of two tensors
    pub fn maximum(self, rhs: GraphTensor) -> GraphTensor {
        (self.lt(rhs) * rhs) + (rhs.le(self) * self)
    }

    /// Take the elementwise maximum of a tensor and a float
    pub fn maximum_f32(self, rhs: f32) -> GraphTensor {
        self.maximum(self.graph().constant_float(rhs).expand_rhs(self.shape))
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

    /// Return a tensor of elements selected from either self or other, depending on condition.
    pub fn cond(self, cond: GraphTensor, other: GraphTensor) -> GraphTensor {
        (cond * self) + ((1.0 - cond) * other)
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
    use crate::{
        prelude::*,
        tests::{assert_close, random_vec},
    };
    use candle_core::{DType, Device, Tensor};
    use itertools::Itertools;
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
        let a = cx.tensor(a_shape.clone());
        let b = cx.tensor(b_shape.clone());
        let c = func(a, b).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        let lhs_values = lhs_transform(random_vec(a_shape.iter().copied().product()));
        let rhs_values = rhs_transform(random_vec(b_shape.iter().copied().product()));
        rt.set_data(a.id, lhs_values.clone());
        rt.set_data(b.id, rhs_values.clone());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::from_vec(lhs_values, a_shape, &device).unwrap();
        let ref_b = Tensor::from_vec(rhs_values, b_shape, &device).unwrap();
        let ref_c = ref_func(ref_a, ref_b).flatten_all().unwrap();

        assert_close(rt.get_f32(c.id), &ref_c.to_vec1::<f32>().unwrap())
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
        fn test_lt(size in 1usize..64) {
            test_binary(
                size,
                size,
                |a, b| a.lt(b),
                |a, b| a.lt(&b).unwrap().to_dtype(DType::F32).unwrap(),
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
                |a, b| a.gt(b),
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
                |a, b| a.le(b),
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
                |a, b| a.ge(b),
                |a, b| a.ge(&b).unwrap().to_dtype(DType::F32).unwrap(),
            );
        }
    }

    #[test]
    fn test_ne() {
        test_binary(
            27,
            27,
            |a, b| a.ne(b),
            |a, b| a.ne(&b).unwrap().to_dtype(DType::F32).unwrap(),
        );
    }

    #[test]
    fn test_eq() {
        test_binary(
            27,
            27,
            |a, b| a.eq(b),
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
                let cond = a.gt(b.graph().constant_float(0.0).expand_rhs(a.shape));
                a.cond(cond, b)
            },
            |a, b| {
                let refer = a.gt(&Tensor::zeros_like(&a).unwrap()).unwrap();
                refer.where_cond(&a, &b).unwrap()
            },
        );
    }
}
