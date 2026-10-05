pub mod symbol;
pub use symbol::{DynMap, InvalidSymbolName, Symbol};
mod bounds;
pub use bounds::{DimensionRange, SymbolBounds, program_dimensions};
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

/// An explicit collection of per-axis slice bounds, such as `vec![(0, 4)]`.
pub trait ToSlice {
    fn to_range_vec(self) -> Vec<(IntExpr, IntExpr)>;
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

/// An explicit collection of `(before, after)` padding pairs.
pub trait ToPad {
    fn to_pad_vec(self) -> Vec<(IntExpr, IntExpr)>;
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

/// An explicit axis collection: `vec![0]` selects one axis; `vec![]` selects none.
pub trait ToAxes {
    fn to_axes(&self) -> Vec<usize>;
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

/// An explicit dimension collection, including for rank-one and rank-zero shapes.
/// Use `vec![n]` for one dimension and `vec![] as Vec<usize>` for a scalar shape.
pub trait ToShape {
    fn to_shape(self) -> Vec<IntExpr>;
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
