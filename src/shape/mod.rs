pub mod symbol;
pub use symbol::{DynMap, Symbol};
mod expression;

pub use expression::*;

use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo, RangeToInclusive};

fn get_start_bound<D: Into<IntExpr> + Copy>(bound: Bound<D>) -> IntExpr {
    match bound {
        Bound::Included(x) => x.into(),
        Bound::Excluded(x) => x.into() + 1,
        Bound::Unbounded => 0.into(),
    }
}

fn get_end_bound<D: Into<IntExpr> + Copy>(bound: Bound<D>) -> IntExpr {
    match bound {
        Bound::Excluded(x) => x.into(),
        Bound::Included(x) => x.into() + 1,
        Bound::Unbounded => IntExpr::from(i64::MAX),
    }
}

pub trait SliceRange {
    fn bounds(&self) -> (IntExpr, IntExpr);
}

impl SliceRange for usize {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (IntExpr::from(self), IntExpr::from(self))
    }
}

impl SliceRange for RangeFrom<usize> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeTo<usize> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeToInclusive<usize> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for Range<usize> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeFrom<IntExpr> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeTo<IntExpr> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeToInclusive<IntExpr> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for Range<IntExpr> {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (
            get_start_bound(self.start_bound()),
            get_end_bound(self.end_bound()),
        )
    }
}
impl SliceRange for RangeFull {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        (0.into(), IntExpr::from(i64::MAX))
    }
}
impl<R: SliceRange> SliceRange for (R,) {
    fn bounds(&self) -> (IntExpr, IntExpr) {
        self.0.bounds()
    }
}

pub trait ToSlice {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)>;
}

impl<R: SliceRange> ToSlice for R {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![self.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange> ToSlice for (R1, R2) {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![self.0.bounds(), self.1.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange, R3: SliceRange> ToSlice for (R1, R2, R3) {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![self.0.bounds(), self.1.bounds(), self.2.bounds()]
    }
}

impl<R1: SliceRange, R2: SliceRange, R3: SliceRange, R4: SliceRange> ToSlice for (R1, R2, R3, R4) {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
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
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![
            self.0.bounds(),
            self.1.bounds(),
            self.2.bounds(),
            self.3.bounds(),
            self.4.bounds(),
        ]
    }
}

impl<A: Into<IntExpr>, B: Into<IntExpr>> ToSlice for Vec<(A, B)> {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.into_iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<A: Into<IntExpr> + Copy, B: Into<IntExpr> + Copy> ToSlice for &Vec<(A, B)> {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<A: Into<IntExpr> + Copy, B: Into<IntExpr> + Copy> ToSlice for &[(A, B)] {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

impl<const N: usize, A: Into<IntExpr> + Copy, B: Into<IntExpr> + Copy> ToSlice for &[(A, B); N] {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter().map(|i| (i.0.into(), i.1.into())).collect()
    }
}

pub trait ToPad {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)>;
}

impl ToPad for () {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![]
    }
}

impl<S: Into<IntExpr>, E: Into<IntExpr>> ToPad for (S, E) {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![(self.0.into(), self.1.into())]
    }
}

impl<S: Into<IntExpr>, E: Into<IntExpr>> ToPad for ((S, E),) {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![(self.0.0.into(), self.0.1.into())]
    }
}

impl<S1: Into<IntExpr>, E1: Into<IntExpr>, S2: Into<IntExpr>, E2: Into<IntExpr>> ToPad
    for ((S1, E1), (S2, E2))
{
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
        ]
    }
}

impl<
    S1: Into<IntExpr>,
    E1: Into<IntExpr>,
    S2: Into<IntExpr>,
    E2: Into<IntExpr>,
    S3: Into<IntExpr>,
    E3: Into<IntExpr>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3))
{
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
        ]
    }
}

impl<
    S1: Into<IntExpr>,
    E1: Into<IntExpr>,
    S2: Into<IntExpr>,
    E2: Into<IntExpr>,
    S3: Into<IntExpr>,
    E3: Into<IntExpr>,
    S4: Into<IntExpr>,
    E4: Into<IntExpr>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4))
{
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        vec![
            (self.0.0.into(), self.0.1.into()),
            (self.1.0.into(), self.1.1.into()),
            (self.2.0.into(), self.2.1.into()),
            (self.3.0.into(), self.3.1.into()),
        ]
    }
}

impl<
    S1: Into<IntExpr>,
    E1: Into<IntExpr>,
    S2: Into<IntExpr>,
    E2: Into<IntExpr>,
    S3: Into<IntExpr>,
    E3: Into<IntExpr>,
    S4: Into<IntExpr>,
    E4: Into<IntExpr>,
    S5: Into<IntExpr>,
    E5: Into<IntExpr>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5))
{
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
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
    S1: Into<IntExpr>,
    E1: Into<IntExpr>,
    S2: Into<IntExpr>,
    E2: Into<IntExpr>,
    S3: Into<IntExpr>,
    E3: Into<IntExpr>,
    S4: Into<IntExpr>,
    E4: Into<IntExpr>,
    S5: Into<IntExpr>,
    E5: Into<IntExpr>,
    S6: Into<IntExpr>,
    E6: Into<IntExpr>,
> ToPad for ((S1, E1), (S2, E2), (S3, E3), (S4, E4), (S5, E5), (S6, E6))
{
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
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

impl<S: Into<IntExpr> + Copy, E: Into<IntExpr> + Copy> ToPad for &[(S, E)] {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<const N: usize, S: Into<IntExpr> + Copy, E: Into<IntExpr> + Copy> ToPad for &[(S, E); N] {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<S: Into<IntExpr> + Copy, E: Into<IntExpr> + Copy> ToPad for &Vec<(S, E)> {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
        self.iter()
            .map(|(s, e)| ((*s).into(), (*e).into()))
            .collect()
    }
}

impl<S: Into<IntExpr>, E: Into<IntExpr>> ToPad for Vec<(S, E)> {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)> {
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
    fn to_shape(self) -> Vec<IntExpr>;
}

impl ToShape for () {
    fn to_shape(self) -> Vec<IntExpr> {
        vec![]
    }
}

impl<A: Into<IntExpr>> ToShape for (A,) {
    fn to_shape(self) -> Vec<IntExpr> {
        vec![self.0.into()]
    }
}

impl<A: Into<IntExpr>, B: Into<IntExpr>> ToShape for (A, B) {
    fn to_shape(self) -> Vec<IntExpr> {
        vec![self.0.into(), self.1.into()]
    }
}

impl<A: Into<IntExpr>, B: Into<IntExpr>, C: Into<IntExpr>> ToShape for (A, B, C) {
    fn to_shape(self) -> Vec<IntExpr> {
        vec![self.0.into(), self.1.into(), self.2.into()]
    }
}

impl<A: Into<IntExpr>, B: Into<IntExpr>, C: Into<IntExpr>, D: Into<IntExpr>> ToShape
    for (A, B, C, D)
{
    fn to_shape(self) -> Vec<IntExpr> {
        vec![self.0.into(), self.1.into(), self.2.into(), self.3.into()]
    }
}

impl<A: Into<IntExpr>, B: Into<IntExpr>, C: Into<IntExpr>, D: Into<IntExpr>, E: Into<IntExpr>>
    ToShape for (A, B, C, D, E)
{
    fn to_shape(self) -> Vec<IntExpr> {
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
    A: Into<IntExpr>,
    B: Into<IntExpr>,
    C: Into<IntExpr>,
    D: Into<IntExpr>,
    E: Into<IntExpr>,
    F: Into<IntExpr>,
> ToShape for (A, B, C, D, E, F)
{
    fn to_shape(self) -> Vec<IntExpr> {
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

impl<A: Into<IntExpr> + Copy> ToShape for &[A] {
    fn to_shape(self) -> Vec<IntExpr> {
        self.iter().map(|i| (*i).into()).collect()
    }
}

impl<const E: usize, A: Into<IntExpr> + Copy> ToShape for &[A; E] {
    fn to_shape(self) -> Vec<IntExpr> {
        self.iter().map(|i| (*i).into()).collect()
    }
}

impl<const E: usize, A: Into<IntExpr>> ToShape for [A; E] {
    fn to_shape(self) -> Vec<IntExpr> {
        self.into_iter().map(|i| i.into()).collect()
    }
}

impl<A: Into<IntExpr>> ToShape for Vec<A> {
    fn to_shape(self) -> Vec<IntExpr> {
        self.into_iter().map(|i| i.into()).collect()
    }
}

impl<A: Into<IntExpr>> ToShape for A {
    fn to_shape(self) -> Vec<IntExpr> {
        vec![self.into()]
    }
}
