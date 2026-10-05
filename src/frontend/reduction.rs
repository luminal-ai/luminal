use crate::prelude::*;

impl GraphTensor {
    /// Reduce a dimension of the tensor by summing all elements along that axis.
    pub fn sum(self, axes: impl ToAxes) -> GraphTensor {
        self.reduce(false, axes)
    }

    /// Reduce a dimension of the tensor by taking the maximum of all elements along that axis.
    pub fn max(self, axes: impl ToAxes) -> GraphTensor {
        self.reduce(true, axes)
    }

    /// One recorded reduce per axis; the operand for the FIRST reduce is
    /// self's value, later reduces consume the previous reduce's result.
    fn reduce(self, is_max: bool, axes: impl ToAxes) -> GraphTensor {
        let (mut dims, mut id) = (self.dims(), self.id);
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            let operand_dims = dims.clone();
            if is_max {
                // The empty max has no value (extent-0 ruling
                // 2026-08-13): the reduced axis contracts to >= 1 —
                // static extents discharge trivially; symbolic ones
                // refuse unless the binding's range excludes 0.
                let extent = operand_dims[axes[dim]];
                self.graph().logical.contract_extent_at_least(&extent, 1);
            }
            let rank = operand_dims.len();
            let axis_from_end = rank - 1 - axes[dim];
            let mut out_dims = operand_dims.clone();
            out_dims.remove(axes[dim]);
            let op = if is_max {
                LogicalOp::ReduceMax { axis_from_end }
            } else {
                LogicalOp::ReduceSum { axis_from_end }
            };
            id = self
                .graph()
                .logical
                .op(op, &[(id, operand_dims)], out_dims.clone(), self.dtype);
            dims = out_dims;
            let axis = axes[dim];
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        GraphTensor::from_id(id, dims, self.graph_ref, self.dtype)
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
            .product::<IntExpr>();
        self.sum(axes) / reduced_elements
    }

    /// Reduce a dimension of the tensor by multiplying all elements along that
    /// axis: the exact product is the last position of an inclusive product scan.
    pub fn prod(self, axes: impl ToAxes) -> GraphTensor {
        // A rank-0 product is the value itself (torch parity): there is no
        // axis to scan.
        if self.dims().is_empty() {
            return self;
        }
        let mut t = self;
        let mut axes = axes.to_axes();
        for dim in 0..axes.len() {
            let axis = axes[dim];
            let dims = t.dims();
            let extent = dims[axis];
            // A product over a statically empty axis is one (torch parity), so
            // no scan is recorded and the axis carries no contract.
            if extent.to_usize() == Some(0) {
                let mut out_dims = dims.clone();
                out_dims.remove(axis);
                // The fill must be built in the tensor's own dtype: a float
                // literal cast to an integer tensor is a refused lossy read.
                let one = match t.dtype {
                    DType::F64 => t.graph().constant_f64(1.0),
                    DType::Int | DType::I64 | DType::I8 | DType::U8 | DType::I16 => {
                        t.graph().constant_i32(1).cast(t.dtype)
                    }
                    _ => t.graph().constant_f32(1.0).cast(t.dtype),
                };
                t = one.expand_rhs(out_dims);
                for ax in &mut axes {
                    if *ax > axis {
                        *ax -= 1;
                    }
                }
                continue;
            }
            // A product over a symbolically empty axis has no value: the axis
            // contracts to >= 1 rather than defaulting to one, as `max` does.
            t.graph().logical.contract_extent_at_least(&extent, 1);
            let rank = dims.len();
            let id = t.graph().logical.op(
                LogicalOp::UnspecifiedOrderScanProd {
                    axis_from_end: rank - 1 - axis,
                },
                &[(t.id, dims.clone())],
                dims.clone(),
                t.dtype,
            );
            t = GraphTensor::from_id(id, dims, t.graph_ref, t.dtype);
            t = t
                .slice_along((extent - IntExpr::from(1))..extent, axis)
                .squeeze(axis);
            for ax in &mut axes {
                if *ax > axis {
                    *ax -= 1;
                }
            }
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::unary::tests::test_unary;
    use crate::tests::assert_exact;
    use candle_core::{Device, Tensor};
    use luminal::prelude::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_sum(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary(vec![rows, cols], |a| a.sum(vec![1]), |a| a.sum(1).unwrap());
            test_unary(
                vec![rows, cols, depth],
                |a| a.sum(vec![0, 2]),
                |a| a.sum((0, 2)).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_max(rows in 1usize..8, cols in 1usize..8) {
            test_unary(vec![rows, cols], |a| a.max(vec![1]), |a| a.max(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_min(rows in 1usize..8, cols in 1usize..8) {
            test_unary(vec![rows, cols], |a| a.min(vec![1]), |a| a.min(1).unwrap());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_mean(rows in 1usize..8, cols in 1usize..8, depth in 1usize..6) {
            test_unary(vec![rows, cols], |a| a.mean(vec![1]), |a| a.mean(1).unwrap());
            let denom = (rows * depth) as f32;
            test_unary(
                vec![rows, cols, depth],
                |a| a.mean(vec![0, 2]),
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
                vec![rows, cols],
                |a| a.prod(vec![1]),
                |a| {
                    let v = a.to_vec2::<f32>().unwrap();
                    let out: Vec<f32> = v.iter().map(|row| row.iter().product()).collect();
                    Tensor::from_vec(out, v.len(), &Device::Cpu).unwrap()
                },
            );
        }
    }

    /// The scan-based product is exact: signs multiply and a zero anywhere
    /// zeroes the result — neither survives an exp/log formulation.
    #[test]
    fn prod_signs_and_zeros_are_exact() {
        let cases: Vec<(Vec<f32>, f32)> = vec![
            (vec![2.0, 3.0], 6.0),
            (vec![-2.0, 3.0], -6.0),
            (vec![-2.0, -3.0], 6.0),
            (vec![-1.0, -2.0, -3.0], -6.0),
            (vec![0.0, 5.0], 0.0),
            (vec![-2.0, 0.0, 3.0], 0.0),
        ];
        for (input, expected) in cases {
            let mut cx = Graph::new();
            let a = cx.tensor(vec![1, input.len()], DType::F32);
            let b = a.prod(vec![1]);
            let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, input.into())]);
            assert_exact(rt.get_f32(b.id).unwrap(), &[expected]);
        }
    }

    /// The empty product is one (torch parity): no scan is recorded, so the
    /// axis needs no non-empty contract.
    #[test]
    fn prod_over_an_empty_axis_is_one() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 0], DType::F32);
        let b = a.prod(vec![1]);
        let rt =
            luminal_reference::harness::run_reference(&cx, &[(a.id, Vec::<f32>::new().into())]);
        assert_exact(rt.get_f32(b.id).unwrap(), &[1.0, 1.0]);
    }

    /// `max` is IEEE 754-2019 `maximum`: a NaN anywhere in the slice is the
    /// slice's maximum.
    #[test]
    fn max_propagates_nan() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![2, 2], DType::F32);
        let b = a.max(vec![1]);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(a.id, vec![1.0, f32::NAN, 3.0, 2.0].into())],
        );
        let out = rt.get_f32(b.id).unwrap();
        assert!(out[0].is_nan(), "row with a NaN: got {}", out[0]);
        assert_eq!(out[1], 3.0);
    }
}
