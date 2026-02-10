use proptest::prelude::*;

use super::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    #[test]
    fn test_add(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a + b, |a, b| (&a + &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a + b, |a, b| (&a + &b).unwrap());
    }

    #[test]
    fn test_mul(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a * b, |a, b| (&a * &b).unwrap());
        test_binary((y, x), (y, x), |a, b| a * b, |a, b| (&a * &b).unwrap());
    }

    #[test]
    fn test_max(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
    }

    #[test]
    fn test_mean(rows in 1usize..8, cols in 1usize..8) {
        test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
    }

    #[test]
    fn test_matmul(m in 1usize..128, n in 1usize..128, k in 1usize..128) {
        // a_shape: (m, k), b_shape: (n, k) - b gets transposed to (k, n) with k-contiguous strides
        test_binary(
            (m, k),
            (n, k),
            |a, b| a.matmul(b.t()),
            |a, b| a.matmul(&b.t().unwrap()).unwrap(),
        );
    }

    // Unary ops tests
    #[test]
    fn test_exp2(x in 1usize..100, y in 1usize..5) {
        // exp2(x) = 2^x, verified by computing 2^x using exp(x * ln(2))
        test_unary(x, |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
        test_unary((y, x), |a| a.exp2(), |a| (a * 2.0f64.ln()).unwrap().exp().unwrap());
    }

    #[test]
    fn test_log2(x in 1usize..100, y in 1usize..5) {
        // log2(x) = ln(x) / ln(2)
        test_unary_positive(x, |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
        test_unary_positive((y, x), |a| a.log2(), |a| (a.log().unwrap() / 2.0f64.ln()).unwrap());
    }

    #[test]
    fn test_sin(x in 1usize..100, y in 1usize..5) {
        test_unary(x, |a| a.sin(), |a| a.sin().unwrap());
        test_unary((y, x), |a| a.sin(), |a| a.sin().unwrap());
    }

    #[test]
    fn test_recip(x in 1usize..100, y in 1usize..5) {
        test_unary_nonzero(x, |a| a.reciprocal(), |a| a.recip().unwrap());
        test_unary_nonzero((y, x), |a| a.reciprocal(), |a| a.recip().unwrap());
    }

    #[test]
    fn test_sqrt(x in 1usize..100, y in 1usize..5) {
        test_unary_positive(x, |a| a.sqrt(), |a| a.sqrt().unwrap());
        test_unary_positive((y, x), |a| a.sqrt(), |a| a.sqrt().unwrap());
    }

    // Binary ops tests
    #[test]
    fn test_mod_op(size in 1usize..100, rows in 1usize..5) {
        test_mod(size, size, |a, b| a % b);
        test_mod((rows, size), (rows, size), |a, b| a % b);
    }

    #[test]
    fn test_less_than(x in 1usize..100, y in 1usize..5) {
        test_binary(x, x, |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        test_binary((y, x), (y, x), |a, b| a.lt(b), |a, b| a.lt(&b).unwrap().to_dtype(candle_core::DType::F32).unwrap());
    }
}
