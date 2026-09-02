use crate::graph::MapEntry;
use crate::prelude::*;

/// Out-space coordinate reader, from-end (the de Bruijn house
/// convention): `c(n, e)` = "read out coordinate c_n, whose extent is e".
fn c(from_end: usize, extent: IntExpr) -> MapEntry {
    MapEntry::Coord { from_end, extent }
}

/// Loud contract-dim check (the old scaffolding got this for free from
/// the binary-op dims_agree assert on the broadcast intermediates; the
/// direct applies must state it themselves — loud bails, never silent
/// mistranslation).
fn assert_dims_eq(lhs: IntExpr, rhs: IntExpr, what: &str) {
    assert!(
        lhs.simplify().egglog_equal(rhs.simplify()),
        "matmul {what} mismatch: {lhs} vs {rhs}"
    );
}

impl GraphTensor {
    /// Matrix multiply: broadcast + reduce in the operands' shared dtype,
    /// NOTHING else — matmul never inserts a cast and carries no dtype
    /// parameter (ruling 2026-08-27, superseding R-A's matmul_in: "no
    /// casting stuff, to keep things visible and explicit"). Wanting a
    /// different compute precision is the USER's statement:
    /// `lhs.cast(D).matmul(rhs.cast(D))` — the casts are then real,
    /// visible LogicalCast nodes in the recorded model. fp8 operands are
    /// no exception: fp8 x fp8 records as an fp8 broadcast+reduce; the
    /// F32-accumulator contract, where wanted, is spelled with casts.
    /// Mixed dtypes cannot multiply and poison loudly.
    pub fn matmul(self, mut rhs: GraphTensor) -> Self {
        if self.dtype != rhs.dtype {
            self.graph().logical.poison(format!(
                "matmul on {:?} x {:?}: matmul is broadcast+reduce in the \
                 operands' shared dtype and never casts — cast explicitly \
                 (lhs.cast(D).matmul(rhs.cast(D)))",
                self.dtype, rhs.dtype
            ));
            // The graph is already poisoned (nothing records from here on);
            // align the HANDLE dtype only — no cast op, so e.g. an Int x
            // F32 pair can't detonate the float->int cast refusal before
            // the poison surfaces — and let bookkeeping proceed inertly.
            rhs.dtype = self.dtype;
        }
        self.matmul_body(rhs)
    }

    fn matmul_body(mut self, rhs: GraphTensor) -> Self {
        // FOLD-2 REMOVED (Austin's ruling 2026-08-26): matmul records
        // DIRECT broadcast applies — one apply per operand mapping it
        // into the shared mul space P, via record_view_map — instead of
        // permute/expand/merge scaffolding. The recorded chain is exactly
        // the canonical form ReduceSum(Mul(apply(a,...,P), apply(b,...,P))).
        // A user's own permute stays an explicit view node feeding these.
        if (self.rank() == 1 || self.rank() == 2) && rhs.rank() == 2 {
            let vec = self.rank() == 1;
            if vec {
                self = self.expand_dim(0, 1);
            }
            let (m, k) = self.dims2();
            let (rk, n) = rhs.dims2();
            assert_dims_eq(k, rk, "contract dim");
            // P = [m, n, k]: lhs map (c2, c0), rhs map (c0, c1).
            let p = vec![m, n, k];
            let mul = self.record_view_map(vec![c(2, m), c(0, k)], p.clone())
                * rhs.record_view_map(vec![c(0, k), c(1, n)], p);

            // Sum Reduce
            let mut ret = mul.sum(2);
            if vec {
                ret = ret.squeeze(0);
            }
            ret
        } else if self.rank() == 3 {
            let (a, b, k) = self.dims3();
            if rhs.rank() == 2 {
                // ABKxKD -> ABD
                let (rk, d) = rhs.dims2();
                assert_dims_eq(k, rk, "contract dim");
                // P = [a, b, d, k]: lhs (c3, c2, c0), rhs (c0, c1).
                let p = vec![a, b, d, k];
                let mul = self.record_view_map(vec![c(3, a), c(2, b), c(0, k)], p.clone())
                    * rhs.record_view_map(vec![c(0, k), c(1, d)], p);

                // Sum Reduce
                mul.sum(3)
            } else if rhs.rank() == 3 {
                // ABKxAKD -> ABD
                let (ra, rk, d) = rhs.dims3();
                assert_dims_eq(a, ra, "batch dim 0");
                assert_dims_eq(k, rk, "contract dim");
                // P = [a, b, d, k]: lhs (c3, c2, c0), rhs (c3, c0, c1).
                let p = vec![a, b, d, k];
                let mul = self.record_view_map(vec![c(3, a), c(2, b), c(0, k)], p.clone())
                    * rhs.record_view_map(vec![c(3, a), c(0, k), c(1, d)], p);

                // Sum Reduce
                mul.sum(3)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.rank() == 4 {
            let (a, b, cc, k) = self.dims4();
            if rhs.rank() == 2 {
                // ABCKxKE -> ABCE
                let (rk, e) = rhs.dims2();
                assert_dims_eq(k, rk, "contract dim");
                // P = [a, b, c, e, k]: lhs (c4, c3, c2, c0), rhs (c0, c1).
                let p = vec![a, b, cc, e, k];
                let mul = self
                    .record_view_map(vec![c(4, a), c(3, b), c(2, cc), c(0, k)], p.clone())
                    * rhs.record_view_map(vec![c(0, k), c(1, e)], p);

                // Sum Reduce
                mul.sum(4)
            } else if rhs.rank() == 4 {
                // ABCKxABKE -> ABCE
                let (ra, rb, rk, e) = rhs.dims4();
                assert_dims_eq(a, ra, "batch dim 0");
                assert_dims_eq(b, rb, "batch dim 1");
                assert_dims_eq(k, rk, "contract dim");
                // P = [a, b, c, e, k]: lhs (c4, c3, c2, c0), rhs (c4, c3, c0, c1).
                let p = vec![a, b, cc, e, k];
                let mul = self
                    .record_view_map(vec![c(4, a), c(3, b), c(2, cc), c(0, k)], p.clone())
                    * rhs.record_view_map(vec![c(4, a), c(3, b), c(0, k), c(1, e)], p);

                // Sum Reduce
                mul.sum(4)
            } else {
                panic!(
                    "Can't matmul lhs {:?} and rhs {:?}",
                    self.dims(),
                    rhs.dims()
                )
            }
        } else if self.rank() == 5 && rhs.rank() == 5 {
            // ABCDKxABCKF -> ABCDF — fully direct, rank-preserving: the
            // old merge/merge/permute/expand + post-sum split/split
            // scaffolding dies with fold 2.
            let (a, b, cc, d, k) = self.dims5();
            let (ra, rb, rc, rk, f) = rhs.dims5();
            assert_dims_eq(a, ra, "batch dim 0");
            assert_dims_eq(b, rb, "batch dim 1");
            assert_dims_eq(cc, rc, "batch dim 2");
            assert_dims_eq(k, rk, "contract dim");
            // P = [a, b, c, d, f, k]: lhs (c5, c4, c3, c2, c0), rhs (c5, c4, c3, c0, c1).
            let p = vec![a, b, cc, d, f, k];
            let mul = self.record_view_map(
                vec![c(5, a), c(4, b), c(3, cc), c(2, d), c(0, k)],
                p.clone(),
            ) * rhs
                .record_view_map(vec![c(5, a), c(4, b), c(3, cc), c(0, k), c(1, f)], p);

            // Sum Reduce
            mul.sum(5)
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
    use luminal::prelude::{DType, Graph};
    use proptest::prelude::*;

    #[test]
    fn fp8_matmul_with_explicit_casts_records_two_visible_casts() {
        // 2026-08-27 ruling: the F32-accumulator contract is the user's
        // explicit cast spelling — matmul itself never casts.
        let mut cx = Graph::new();
        let lhs = cx.tensor((2, 4), DType::F8E4M3);
        let rhs = cx.tensor((4, 3), DType::F8E4M3);

        let out = lhs.cast(DType::F32).matmul(rhs.cast(DType::F32)).output();

        assert_eq!(out.dtype, DType::F32);
        let model = cx.logical.model_text().expect("recorded model");
        let promoted_casts = model.matches("(LogicalCast").count();
        assert_eq!(
            promoted_casts, 2,
            "both FP8 operands must have explicit F32 casts:\n{model}"
        );
    }

    #[test]
    fn fp8_matmul_records_plainly_in_fp8() {
        // 2026-08-27 ruling: fp8 x fp8 is NOT special — matmul is
        // broadcast+reduce in the shared dtype, no casts, no poison.
        let mut cx = Graph::new();
        let lhs = cx.tensor((2, 4), DType::F8E4M3);
        let rhs = cx.tensor((4, 3), DType::F8E4M3);

        let out = lhs.matmul(rhs).output();

        assert_eq!(out.dtype, DType::F8E4M3);
        let model = cx.logical.model_text().expect("recorded model");
        assert_eq!(
            model.matches("(LogicalCast").count(),
            0,
            "matmul must insert no casts:\n{model}"
        );
    }

    #[test]
    fn mixed_dtype_matmul_without_explicit_precision_is_a_loud_recorder_error() {
        let mut cx = Graph::new();
        let lhs = cx.tensor((2, 4), DType::F32);
        let rhs = cx.tensor((4, 3), DType::F16);

        let _ = lhs.matmul(rhs);

        let reason = cx
            .logical
            .poisoned()
            .expect("mixed-dtype matmul must poison the recorder (matmul never casts)");
        assert!(reason.contains("cast explicitly"), "{reason}");
    }

    #[test]
    fn int_float_matmul_poisons_without_detonating_the_cast_refusal() {
        // The poison path must never record a cast: an Int x F32 pair
        // would otherwise trip the float->int cast refusal assert and
        // panic with a misleading message instead of surfacing the
        // cast-explicitly poison.
        let mut cx = Graph::new();
        let lhs = cx.tensor((2, 4), DType::Int);
        let rhs = cx.tensor((4, 3), DType::F32);

        let _ = lhs.matmul(rhs);

        let reason = cx
            .logical
            .poisoned()
            .expect("Int x F32 matmul must poison, not panic");
        assert!(reason.contains("cast explicitly"), "{reason}");
    }

    #[test]
    fn test_batch3_matmul_5d() {
        // The 5x5 variant's direct-apply maps are hand-derived (fold-2
        // removal killed the merge/merge/permute/expand scaffolding that
        // used to imply them); pin them against candle. Candle has no
        // rank-5 batched matmul, so the reference flattens the three
        // batch dims and unflattens after.
        test_binary(
            vec![2, 2, 3, 4, 5],
            vec![2, 2, 3, 5, 6],
            |a, b| a.matmul(b),
            |a, b| {
                a.reshape((12, 4, 5))
                    .unwrap()
                    .matmul(&b.reshape((12, 5, 6)).unwrap())
                    .unwrap()
                    .reshape((2, 2, 3, 4, 6))
                    .unwrap()
            },
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
