use crate::prelude::*;

impl GraphTensor {
    /// Create the `GraphTensor` for the matrix multiplication of these two
    /// interpreted as summing along different commmon sized dimensions
    /// depending on ranks.
    ///
    /// The broadcasted multiplies do not necessarily assert that the dimensions match up.
    /// That would be equality of expressions possibly with dyn dims that have not been resolved yet.
    /// So enforcing that this is a proper matrix multiplication has not happened at this stage.
    /// In this case, there may be unintended behavior despite the construction of `GraphTensor` not causing Panic or Error.
    ///
    /// # Panics
    /// There are some shapes that are not supported by this operation.
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(mut self, mut rhs: GraphTensor) -> Self {
        if (self.shape.len() == 1 || self.shape.len() == 2) && rhs.shape.len() == 2 {
            let vec = self.shape.len() == 1;
            if vec {
                // k * (k,n) -> (1,k)*(k,n)
                self = self.expand_dim(0, 1);
            }
            // The two _ here are supposed to be equal but so far they are just
            // two expressions that are not necessarily identically equal because of
            // not plugging in dyn dims yet.
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
                // but these two C have not been enforced to be strictly equal here

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
            r.shape = ShapeTracker::new((a, b, c, d, f));
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
    /// But also applies for ABC.. dot ABC.. interpreted
    /// as BC... going along for the ride in the dot product in R^A
    pub fn dot(self, rhs: GraphTensor) -> GraphTensor {
        (self * rhs).sum(0)
    }
}

#[cfg(test)]
mod tests {
    use crate::frontend::binary::tests::{test_binary, test_binary_transforms_same_tensor};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_matrix_vector(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            // Vector - Matrix
            test_binary(
                k,
                (k,n),
                |a, b| a.matmul(b),
                |a, b| a.reshape((1,k)).unwrap().matmul(&b).unwrap(),
            );
            // Matrix - Vector
            // The other order where it is (m,k) for LHS and (k,) for RHS
            // does not get implicitly promoted like above. The (k,) doesn't
            // get expanded to (k,1,) within `matmul`.
            // That could be done just like with the `if vec` for the previous case.
            // It is a matter of when to allow implicit expansion vs when to
            // mandate being explicit.
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
            // A -> A^2 with square matrix multiplication
            test_binary_transforms_same_tensor(
                (m, m),
                |a, b| a.matmul(b),
                |a, b| a.matmul(&b).unwrap(),
                |v| v,
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
