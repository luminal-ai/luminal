mod expression;
mod tracker;

pub use expression::*;
pub use tracker::*;

use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

pub fn flatten_strides(range: &[Expression], strides: &[Expression]) -> Expression {
    assert_eq!(range.len(), strides.len());
    let mut current_elem_size = Expression::from(1);
    let mut flat_stride = Expression::from(0);
    for (dim, (range, stride)) in range.iter().zip(strides).enumerate().rev() {
        let div = Expression::from('z') / current_elem_size;
        let m = if dim > 0 { div % range } else { div };
        flat_stride += stride.substitute('z', m);
        current_elem_size *= range;
    }
    flat_stride.simplify()
}

fn get_start_bound<D: Into<Expression> + Copy>(bound: Bound<D>) -> Expression {
    match bound {
        Bound::Included(x) => x.into(),
        Bound::Excluded(x) => x.into() + 1,
        Bound::Unbounded => 0.into(),
    }
}

fn get_end_bound<D: Into<Expression> + Copy>(bound: Bound<D>) -> Expression {
    match bound {
        Bound::Excluded(x) => x.into(),
        Bound::Included(x) => x.into() + 1,
        Bound::Unbounded => Expression::from(i64::MAX),
    }
}

pub trait SliceRange {
    fn bounds(&self) -> (Expression, Expression);
}

impl SliceRange for usize {
    fn bounds(&self) -> (Expression, Expression) {
        (Expression::from(self), Expression::from(self))
    }
}

impl SliceRange for RangeFrom<usize> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeTo<usize> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeToInclusive<usize> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for Range<usize> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeFrom<Expression> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeTo<Expression> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeToInclusive<Expression> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for Range<Expression> {
    fn bounds(&self) -> (Expression, Expression) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeFull {
    fn bounds(&self) -> (Expression, Expression) {
        (0.into(), Expression::from(i64::MAX))
    }
}
impl<R: SliceRange> SliceRange for (R,) {
    fn bounds(&self) -> (Expression, Expression) {
        self.0.bounds()
    }
}

pub trait ToSlice {
    fn to_range_vec(self) -> Vec<(Expression, Expression)>;
}

impl<R: SliceRange> ToSlice for R {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        vec![self.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange> ToSlice for (R1, R2) {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        vec![self.0.bounds(), self.1.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange, R3: SliceRange> ToSlice for (R1, R2, R3) {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        vec![self.0.bounds(), self.1.bounds(), self.2.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange, R3: SliceRange, R4: SliceRange> ToSlice for (R1, R2, R3, R4) {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(),
            self.1.bounds(),
            self.2.bounds(),
            self.3.bounds(),
        ]
    }
}

impl<R1: SliceRange, R2: SliceRange, R3: SliceRange, R4: SliceRange, R5: SliceRange> ToSlice
    for (R1, R2, R3, R4, R5)
{
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            self.0.bounds(),
            self.1.bounds(),
            self.2.bounds(),
            self.3.bounds(),
            self.4.bounds(),
        ]
    }
}

impl<A: Into<Expression>, B: Into<Expression>> ToSlice for Vec<(A, B)> {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        self.into_iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<A: Into<Expression> + Copy, B: Into<Expression> + Copy> ToSlice for &Vec<(A, B)> {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<A: Into<Expression> + Copy, B: Into<Expression> + Copy> ToSlice for &[(A, B)] {
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<const N: usize, A: Into<Expression> + Copy, B: Into<Expression> + Copy> ToSlice
    for &[(A, B); N]
{
    fn to_range_vec(self) -> Vec<(Expression, Expression)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

pub trait ToPad {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)>;
}

impl ToPad for () {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![]
    }
}

impl<S: Into<Expression>, E: Into<Expression>> ToPad for (S, E) {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![(self.0.into(), self.1.into())]
    }
}

impl<S: Into<Expression>, E: Into<Expression>> ToPad for ((S, E),) {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![(self.0.0.into(), self.0.1.into())]
    }
}

impl<S1: Into<Expression>, E1: Into<Expression>, S2: Into<Expression>, E2: Into<Expression>> ToPad
    for ((S1, E1), (S2, E2))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
        ]
    }
}

impl<
    S1: Into<Expression>,
    E1: Into<Expression>,
    S2: Into<Expression>,
    E2: Into<Expression>,
    S3: Into<Expression>,
    E3: Into<Expression>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
        ]
    }
}

impl<
    S1: Into<Expression>,
    E1: Into<Expression>,
    S2: Into<Expression>,
    E2: Into<Expression>,
    S3: Into<Expression>,
    E3: Into<Expression>,
    S4: Into<Expression>,
    E4: Into<Expression>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
            (self.3.0.into(), self.3.1.into()),
        ]
    }
}

impl<
    S1: Into<Expression>,
    E1: Into<Expression>,
    S2: Into<Expression>,
    E2: Into<Expression>,
    S3: Into<Expression>,
    E3: Into<Expression>,
    S4: Into<Expression>,
    E4: Into<Expression>,
    S5: Into<Expression>,
    E5: Into<Expression>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
            (self.3.0.into(), self.3.1.into()),
            (self.4.0.into(), self.4.1.into()),
        ]
    }
}

impl<
    S1: Into<Expression>,
    E1: Into<Expression>,
    S2: Into<Expression>,
    E2: Into<Expression>,
    S3: Into<Expression>,
    E3: Into<Expression>,
    S4: Into<Expression>,
    E4: Into<Expression>,
    S5: Into<Expression>,
    E5: Into<Expression>,
    S6: Into<Expression>,
    E6: Into<Expression>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5), (S6, E6))
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
            (self.3.0.into(), self.3.1.into()),
            (self.4.0.into(), self.4.1.into()),
            (self.5.0.into(), self.5.1.into()),
        ]
    }
}

impl<S: Into<Expression> + Copy, E: Into<Expression> + Copy> ToPad for &[(S, E)] {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<const N: usize, S: Into<Expression> + Copy, E: Into<Expression> + Copy> ToPad
    for &[(S, E); N]
{
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<S: Into<Expression> + Copy, E: Into<Expression> + Copy> ToPad for &Vec<(S, E)> {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<S: Into<Expression>, E: Into<Expression>> ToPad for Vec<(S, E)> {
    fn to_pad_vec(self) -> Vec<(Expression, Expression)> {
        self.into_iter()
            .map(|(s, e)| (s.into(), e.into()))
            .collect()
    }
}

pub trait ToAxes {
    fn to_axes(&self) -> Vec<usize>;
}

impl ToAxes for () {
    fn to_axes(&self) -> Vec<usize> {
        vec![]
    }
}

impl ToAxes for (usize, usize) {
    fn to_axes(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl ToAxes for (usize, usize, usize) {
    fn to_axes(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl ToAxes for (usize, usize, usize, usize) {
    fn to_axes(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl ToAxes for (usize, usize, usize, usize, usize) {
    fn to_axes(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl ToAxes for (usize, usize, usize, usize, usize, usize) {
    fn to_axes(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}

impl ToAxes for usize {
    fn to_axes(&self) -> Vec<usize> {
        vec![*self]
    }
}

impl ToAxes for Vec<usize> {
    fn to_axes(&self) -> Vec<usize> {
        self.clone()
    }
}

impl ToAxes for &[usize] {
    fn to_axes(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> ToAxes for &[usize; N] {
    fn to_axes(&self) -> Vec<usize> {
        self.to_vec()
    }
}

impl ToAxes for &Vec<usize> {
    fn to_axes(&self) -> Vec<usize> {
        self.to_vec()
    }
}

pub trait ToShape {
    fn to_shape(self) -> Vec<Expression>;
}

impl ToShape for () {
    fn to_shape(self) -> Vec<Expression> {
        vec![]
    }
}

impl<A: Into<Expression>> ToShape for (A,) {
    fn to_shape(self) -> Vec<Expression> {
        vec![self.0.into()]
    }
}

impl<A: Into<Expression>, B: Into<Expression>> ToShape for (A, B) {
    fn to_shape(self) -> Vec<Expression> {
        vec![self.0.into(), self.1.into()]
    }
}

impl<A: Into<Expression>, B: Into<Expression>, C: Into<Expression>> ToShape for (A, B, C) {
    fn to_shape(self) -> Vec<Expression> {
        vec![self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<A: Into<Expression>, B: Into<Expression>, C: Into<Expression>, D: Into<Expression>> ToShape
    for (A, B, C, D)
{
    fn to_shape(self) -> Vec<Expression> {
        vec![self.0.into(), self.1.into(), self.2.into(), self.3.into()]
    }
}

impl<
    A: Into<Expression>,
    B: Into<Expression>,
    C: Into<Expression>,
    D: Into<Expression>,
    E: Into<Expression>,
> ToShape for (A, B, C, D, E)
{
    fn to_shape(self) -> Vec<Expression> {
        vec![
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
        ]
    }
}

impl<
    A: Into<Expression>,
    B: Into<Expression>,
    C: Into<Expression>,
    D: Into<Expression>,
    E: Into<Expression>,
    F: Into<Expression>,
> ToShape for (A, B, C, D, E, F)
{
    fn to_shape(self) -> Vec<Expression> {
        vec![
            self.0.into(),
            self.1.into(),
            self.2.into(),
            self.3.into(),
            self.4.into(),
            self.5.into(),
        ]
    }
}

impl<A: Into<Expression> + Copy> ToShape for &[A] {
    fn to_shape(self) -> Vec<Expression> {
        self.iter().map(|i| (*i).into()).collect()
    }
}

impl<const E: usize, A: Into<Expression> + Copy> ToShape for &[A; E] {
    fn to_shape(self) -> Vec<Expression> {
        self.iter().map(|i| (*i).into()).collect()
    }
}

impl<const E: usize, A: Into<Expression>> ToShape for [A; E] {
    fn to_shape(self) -> Vec<Expression> {
        self.into_iter().map(|i| i.into()).collect()
    }
}

impl<A: Into<Expression>> ToShape for Vec<A> {
    fn to_shape(self) -> Vec<Expression> {
        self.into_iter().map(|i| i.into()).collect()
    }
}

impl<A: Into<Expression>> ToShape for A {
    fn to_shape(self) -> Vec<Expression> {
        vec![self.into()]
    }
}

impl ToShape for ShapeTracker {
    fn to_shape(self) -> Vec<Expression> {
        self.dims.to_shape()
    }
}
