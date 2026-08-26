use crate::hlir::*;
use crate::prelude::*;

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Sum reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self.graph().add_op(
                SumReduce {
                    dim: axes[dim],
                    input_shape: shape,
                    ..Default::default()
                },
                &[id],
            );
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        let (mut shape, mut id) = (self.shape, self.id);
        // Max reduce each dimension
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            id = self.graph().add_op(
                MaxReduce {
                    dim: axes[dim],
                    input_shape: shape,
                    ..Default::default()
                },
                &[id],
            );
            shape.remove_dim(axes[dim]);
            shape = shape.contiguous();
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Reduce a dimension of the tensor by taking the minimum of all elements along that axis.
    pub fn min(self, axes: impl ToAxes) -> GraphTensor {
        -(-self).max(axes)
    }

    /// Reduce a dimension of the tensor by taking the mean of all elements along that axis.
    pub fn mean(self, axes: impl ToAxes) -> GraphTensor {
        let reduced_elements = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        self.sum(axes) / reduced_elements
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that axis.
    ///
    /// The magnitude comes from `exp(sum(log(|x|)))` and the sign from the
    /// parity of the negative element count. Taking `log` of the signed value
    /// directly is only defined on non-negative inputs, and returned NaN for
    /// any axis containing a negative element.
    pub fn prod(self, axes: impl ToAxes) -> GraphTensor {
        let axes = axes.to_axes();
        let zero = self
            .graph()
            .constant_float(0.0)
            .cast(self.dtype)
            .expand_rhs(self.shape);
        let negatives = self.lt(zero).cast(self.dtype);
        // Even count of negatives -> +1, odd -> -1.
        let sign = 1.0 - (negatives.sum(&axes) % 2.0) * 2.0;
        self.abs().log().sum(&axes).exp() * sign
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::unary::tests::test_unary;
    use crate::prelude::*;
    use crate::tests::assert_close_finite;
    use candle_core::{Device, Tensor};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_sum(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary((rows, cols), |a| a.sum(1), |a| a.sum(1).unwrap());
            test_unary(
                (rows, cols, depth),
                |a| a.sum((0, 2)),
                |a| a.sum((0, 2)).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_max(rows in 1usize..8, cols in 1usize..8) {
            test_unary((rows, cols), |a| a.max(1), |a| a.max(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_min(rows in 1usize..8, cols in 1usize..8) {
            test_unary((rows, cols), |a| a.min(1), |a| a.min(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mean(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary((rows, cols), |a| a.mean(1), |a| a.mean(1).unwrap());
            let denom = (rows * depth) as f32;
            test_unary(
                (rows, cols, depth),
                |a| a.mean((0, 2)),
                |a| {
                    let denom = Tensor::from_vec(vec![denom; cols], cols, a.device()).unwrap();
                    (a.sum(2).unwrap().sum(0).unwrap() / denom).unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_prod(rows in 1usize..8, cols in 1usize..8) {
            test_unary(
                (rows, cols),
                |a| a.prod(1),
                |a| {
                    let v = a.to_vec2::<f32>().unwrap();
                    let out: Vec<f32> = v.iter().map(|row| row.iter().product()).collect();
                    Tensor::from_vec(out, v.len(), &Device::Cpu).unwrap()
                },
            );
        }
    }

    /// `prod` is `exp(sum(log(|x|)))` with the sign recovered separately.
    /// Taking `log` of the signed value returns NaN for any axis holding a
    /// negative element, which `assert_close` cannot catch because every
    /// comparison against NaN is false.
    #[test]
    fn test_prod_signs_and_zeros() {
        for (input, expected) in [
            (vec![-2.0, 3.0], -6.0),
            (vec![-2.0, -3.0], 6.0),
            (vec![2.0, 3.0], 6.0),
            (vec![-1.0, -2.0], 2.0),
            (vec![0.0, 5.0], 0.0),
            (vec![-2.0, 0.0], 0.0),
        ] {
            let cols = input.len();
            let mut cx = Graph::new();
            let a = cx.tensor((1, cols));
            let b = a.prod(1).output();

            cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
            let mut rt = cx.search(
                ReferenceRuntime::default(),
                CompileOptions::default().search_graph_limit(1),
            );
            rt.set_data(a.id, input.clone());
            rt.execute(&cx.dyn_map);

            assert_close_finite(rt.get_f32(b), &[expected]);
        }
    }
}
