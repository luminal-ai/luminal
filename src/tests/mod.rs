use std::fmt::Debug;

use crate::prelude::*;
use candle_core::{Device, Tensor};
use proptest::prelude::*;
use rand::{Rng, rng};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn simple(vals in proptest::collection::vec(-2.0f32..2.0, 3)) {
        prop_assume!(vals.iter().all(|v| v.abs() > 1e-3));
        let mut cx = Graph::new();
        let b = cx.tensor(3);
        let c = cx.tensor(3);
        let g = cx.tensor(3);
        let e = cx.tensor(3);

        let a = (b * c + g).output();
        let d = (b * c / e).sin().output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(b.id, vals.clone());
        rt.set_data(c.id, vals.clone());
        rt.set_data(g.id, vals.clone());
        rt.set_data(e.id, vals.clone());

        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = Tensor::new(vals.clone(), &device).unwrap();
        let ref_c = Tensor::new(vals.clone(), &device).unwrap();
        let ref_g = Tensor::new(vals.clone(), &device).unwrap();
        let ref_e = Tensor::new(vals, &device).unwrap();

        let ref_a = (ref_b.clone() * ref_c.clone() + ref_g).unwrap();
        let ref_d = (ref_b * ref_c / ref_e).unwrap().sin().unwrap();

        assert_close(rt.get_f32(a.id), &ref_a.to_vec1::<f32>().unwrap());
        assert_close(rt.get_f32(d.id), &ref_d.to_vec1::<f32>().unwrap());
    }

    #[test]
    fn test_matmul(m in 1usize..6, k in 1usize..6, n in 1usize..6, lhs in proptest::collection::vec(-2.0f32..2.0, 1..100), rhs in proptest::collection::vec(-2.0f32..2.0, 1..100)) {
        prop_assume!(lhs.len() >= m * k);
        prop_assume!(rhs.len() >= k * n);
        let mut cx = Graph::new();
        let b = cx.tensor((m, k));
        let c = cx.tensor((k, n));

        let a = b.matmul(c).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        let lhs = lhs.into_iter().take(m * k).collect::<Vec<f32>>();
        let rhs = rhs.into_iter().take(k * n).collect::<Vec<f32>>();
        rt.set_data(b.id, lhs.clone());
        rt.set_data(c.id, rhs.clone());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = Tensor::new(lhs, &device).unwrap().reshape((m, k)).unwrap();
        let ref_c = Tensor::new(rhs, &device).unwrap().reshape((k, n)).unwrap();
        let ref_a = ref_b.matmul(&ref_c).unwrap();
        assert_close(
            rt.get_f32(a.id),
            &ref_a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    #[test]
    fn test_shapes(values in proptest::collection::vec(-2.0f32..2.0, 4)) {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 2));
        let b = (a.permute((1, 0)) * 1.0).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        rt.set_data(a.id, values.clone());
        rt.execute(&cx.dyn_map);

        assert_exact(rt.get_f32(b.id), &[values[0], values[2], values[1], values[3]]);
    }

    #[test]
    fn test_top_k_filter(rows in 1usize..6, cols in 3usize..10, k in 1usize..5, values in proptest::collection::vec(-2.0f32..2.0, 1..200)) {
        prop_assume!(k <= cols);
        prop_assume!(values.len() >= rows * cols);
        let mut cx = Graph::new();
        let a = cx.tensor((rows, cols));
        let kth_largest = a.gather(a.topk_indexes(k, 1).slice((.., (k - 1)..k)).squeeze(1));
        let mask = a.ge(kth_largest.expand_dim(1, cols));
        let filtered = (a * mask).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        let values = values.into_iter().take(rows * cols).collect::<Vec<f32>>();
        rt.set_data(a.id, values.clone());
        rt.execute(&cx.dyn_map);

        let mut expected = Vec::with_capacity(values.len());
        for row in values.chunks_exact(cols) {
            let mut indices = (0..cols).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap());
            let kth_index = indices[k - 1];
            let threshold = values[kth_index];
            expected.extend(row.iter().map(|v| if *v >= threshold { *v } else { 0.0 }));
        }
        assert_close(rt.get_f32(filtered.id), &expected);
    }
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
