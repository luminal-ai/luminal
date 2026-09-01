//! Shared numeric test helpers. The their-pipeline integration tests
//! that lived here (search/compile/custom-op/fuzz) died with that
//! pipeline at M3 Step 4b — differential coverage lives beside the ops
//! (frontend candle harnesses on the native ladder + reference).

use std::fmt::Debug;

use rand::{Rng, rng};

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

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = rng();
    random_vec_rng(n, &mut rng)
}

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}
