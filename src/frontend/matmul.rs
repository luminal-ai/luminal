use crate::prelude::*;

impl GraphTensor {
    pub fn matmul(mut self, mut rhs: GraphTensor) -> Self {
        // FP8 matrix multiplication has an F32 product/accumulator/output
        // contract. Express that contract in HLIR with real casts so the
        // fully decomposed Mul + Sum path remains a correct (if expensive)
        // implementation. CUDA backends may absorb these casts and consume
        // the underlying FP8 buffers directly, but they must remain optional
        // alternatives to this semantic reference path.
        let lhs_is_fp8 = matches!(self.dtype, DType::F8E4M3 | DType::F8E5M2);
        let rhs_is_fp8 = matches!(rhs.dtype, DType::F8E4M3 | DType::F8E5M2);
        if lhs_is_fp8 && rhs_is_fp8 {
            self = self.cast(DType::F32);
            rhs = rhs.cast(DType::F32);
        }

        if (self.shape.len() == 1 || self.shape.len() == 2) && rhs.shape.len() == 2 {
            let vec = self.shape.len() == 1;
            if vec {
                self = self.expand_dim(0, 1);
            }
            let (m, _) = self.dims2();
            let (_, n) = rhs.dims2();
            // Broadcasted Multiply
            let mul = self.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);

            // Sum Reduce
            let mut ret = mul.sum(2);
            if vec {
                ret.shape.remove_dim(0);
            }
            ret
        } else if self.shape.len() == 3 {
            let d = *rhs.dims().last().unwrap();
            let (a, b, _) = self.dims3();
            if rhs.shape.len() == 2 {
                // ABCxCD -> ABD
                // Reshape
                let w = rhs.permute((1, 0));

                // Broadcasted Multiply
                let mul = self.expand_dim(2, d) * w.expand_dim(0, a).expand_dim(1, b);

                // Sum Reduce
                mul.sum(3)
            } else if rhs.shape.len() == 3 {
                // Reshape
                let w = rhs.permute((0, 2, 1));

                // Broadcasted Multiply
                let mul = self.expand_dim(2, d) * w.expand_dim(1, b);

                // Sum Reduce
                mul.sum(3)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.shape.len() == 4 {
            let (a, b, c, _) = self.dims4();
            if rhs.shape.len() == 2 {
                // ABCDxDE -> ABCE
                let (_, e) = rhs.dims2();
                // Reshape
                rhs = rhs.permute((1, 0));
                // Broadcasted Multiply
                let mul =
                    self.expand_dim(3, e) * rhs.expand_dim(0, a).expand_dim(1, b).expand_dim(2, c);

                // Sum Reduce
                mul.sum(4)
            } else if rhs.shape.len() == 4 {
                assert_eq!(self.dims()[0], rhs.dims()[0]);
                assert_eq!(self.dims()[1], rhs.dims()[1]);
                // ABCDxABDE -> ABCE
                let (_, _, _, e) = rhs.dims4();
                // Reshape
                rhs = rhs.permute((0, 1, 3, 2));

                // Broadcasted Multiply
                let mul = self.expand_dim(3, e) * rhs.expand_dim(2, c);

                // Sum Reduce
                mul.sum(4)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.shape.len() == 5 && rhs.shape.len() == 5 {
            // ABCDExABCEF -> ABCDF
            let (a, b, c, _, f) = rhs.dims5();
            let (_, _, _, d, _) = self.dims5();
            // Reshape
            let w = rhs.merge_dims(0, 1).merge_dims(0, 1).permute((0, 2, 1));
            let s = self.merge_dims(0, 1).merge_dims(0, 1);

            // Broadcasted Multiply
            let mul = s.expand_dim(2, f) * w.expand_dim(1, d);

            // Sum Reduce
            let mut r = mul.sum(3);
            r.shape = ShapeTracker::new_with_element_bits((a, b, c, d, f), r.dtype.bits());
            r
        } else {
            panic!(
                "Can't matmul lhs {:?} and rhs {:?}",
                self.dims(),
                rhs.dims()
            )
        }
    }

    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor) -> GraphTensor {
        (self * rhs).sum(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::binary::tests::test_binary;
    use crate::prelude::{DType, Graph};
    use proptest::prelude::*;

    #[test]
    fn fp8_matmul_promotes_products_and_accumulation_to_f32() {
        let mut cx = Graph::new();
        let lhs = cx.tensor((2, 4)).as_dtype(DType::F8E4M3);
        let rhs = cx.tensor((4, 3)).as_dtype(DType::F8E4M3);

        let out = lhs.matmul(rhs);

        assert_eq!(out.dtype, DType::F32);
        let promoted_casts = cx
            .graph
            .node_indices()
            .filter(|&node| {
                cx.try_get_op::<crate::hlir::Cast>(node)
                    .is_some_and(|cast| cast.1 == DType::F32)
            })
            .count();
        assert_eq!(
            promoted_casts, 2,
            "both FP8 operands must have explicit F32 semantic fallbacks"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_matrix_vector(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            test_binary(
                (m, k),
                (k, n),
                |a, b| a.matmul(b),
                |a, b| a.matmul(&b).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_matmul(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            test_binary(
                (m, k),
                (k, n),
                |a, b| a.matmul(b),
                |a, b| a.matmul(&b).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_batch_matmul(batch in 1usize..4, m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            test_binary(
                (batch, m, k),
                (k, n),
                |a, b| a.matmul(b),
                |a, b| {
                    a.reshape((batch * m, k))
                        .unwrap()
                        .matmul(&b)
                        .unwrap()
                        .reshape((batch, m, n))
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_batch_batch_matmul(batch in 1usize..4, m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            test_binary(
                (batch, m, k),
                (batch, m, k),
                |a, b| a.matmul(b.permute((0, 2, 1))),
                |a, b| a.matmul(&b.permute((0, 2, 1)).unwrap()).unwrap(),
            );
            test_binary(
                (batch, m, k),
                (batch, k, n),
                |a, b| a.matmul(b),
                |a, b| a.matmul(&b).unwrap(),
            );
        }
    }
}
