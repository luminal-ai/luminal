#[cfg(test)]
mod dynamic;
pub mod test_graphs;
#[cfg(test)]
mod test_prim;

use std::fmt::Debug;

#[cfg(test)]
use crate::prelude::*;
use rand::{rng, Rng};

// Integration and other tests

#[test]
fn main() {
    let mut cx = Graph::new();
    let b = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);
    let c = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);
    let g = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);
    let e = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);

    let mut a = (b * c + g).retrieve();
    let mut d = (b * c / e).exp2().log2().retrieve();

    cx.execute();

    let unoptimized_a = a.data();
    let unoptimized_d = d.data();

    cx.compile(GenericCompiler::default(), (&mut a, &mut d));

    cx.execute();
    assert_close(&unoptimized_a, &a.data());
    assert_close(&unoptimized_d, &d.data());
}

#[test]
fn test_matmul() {
    let mut cx = Graph::new();
    let b = cx.tensor((3, 1)).set(vec![1.0, 2.0, 3.0]);
    let c = cx.tensor((1, 4)).set(vec![1.0, 2.0, 3.0, 3.0]);

    let mut a = b.matmul(c).retrieve();

    cx.execute();

    let unoptimized_a = a.data();

    cx.compile(GenericCompiler::default(), &mut a);
    cx.execute();

    assert_exact(&unoptimized_a, &a.data());
}

#[test]
fn test_shapes() {
    let mut cx = Graph::new();
    let a = cx.tensor(4).set(vec![1., 2., 3., 4.]);
    let b = a.reshape((2, 2)).permute((1, 0)).retrieve();
    cx.execute();

    assert_exact(&b.data(), &[1., 3., 2., 4.]);
}

/// Ensure two arrays are nearly equal
pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

/// Ensure two arrays are nearly equal to a decimal place
pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

/// Ensure two arrays are exactly equal
pub fn assert_exact<T: PartialEq + Debug>(a_vec: &[T], b_vec: &[T]) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if a != b {
            panic!("{a:?} is not equal to {b:?}, index {i}");
        }
    }
}

pub fn random_array<const N: usize>() -> [f32; N] {
    let mut rng = rng();
    random_array_rng(&mut rng)
}

pub fn random_array_rng<const N: usize, R: Rng>(rng: &mut R) -> [f32; N] {
    let mut arr = [0.; N];
    for i in &mut arr {
        *i = rng.random_range(-0.5..0.5);
    }
    arr
}

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = rng();
    random_vec_rng(n, &mut rng)
}

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}

#[macro_export]
macro_rules! test_imports {
    () => {
        #[allow(unused_imports)]
        use dfdx::prelude::{
            Axes as DAxes, Axes2 as DAxes2, Axes3 as DAxes3, Axes4 as DAxes4, Axes5 as DAxes5,
            Axis as DAxis, Const as DConst, *,
        };
        #[allow(unused_imports)]
        use $crate::{
            prelude::*,
            tests::{
                assert_close, assert_close_precision, assert_exact, random_array, random_array_rng,
                random_vec, random_vec_rng, test_graphs,
            },
        };
    };
}
