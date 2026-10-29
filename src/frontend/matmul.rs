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
    /// Mixed dtypes cannot multiply and refuse loudly.
    pub fn matmul(self, rhs: GraphTensor) -> Self {
        if self.dtype != rhs.dtype {
            self.graph().logical.refuse(format!(
                "matmul on {:?} x {:?}: matmul is broadcast+reduce in the \
                 operands' shared dtype and never casts — cast explicitly \
                 (lhs.cast(D).matmul(rhs.cast(D)))",
                self.dtype, rhs.dtype
            ));
        }
        self.matmul_body(rhs)
    }

    fn matmul_body(mut self, mut rhs: GraphTensor) -> Self {
        assert!(
            self.rank() > 0 && rhs.rank() > 0,
            "matmul requires non-scalar operands"
        );
        let lhs_vector = self.rank() == 1;
        let rhs_vector = rhs.rank() == 1;
        if lhs_vector {
            self = self.expand_dim(0, 1);
        }
        if rhs_vector {
            rhs = rhs.expand_dim(1, 1);
        }
        let ad = self.dims();
        let bd = rhs.dims();
        let (m, k) = (ad[ad.len() - 2], ad[ad.len() - 1]);
        let (rk, n) = (bd[bd.len() - 2], bd[bd.len() - 1]);
        assert_dims_eq(k, rk, "contract dim");
        let batches = (ad.len() - 2).max(bd.len() - 2);
        let mut p = Vec::with_capacity(batches + 3);
        for axis in 0..batches {
            let a = axis.checked_sub(batches - (ad.len() - 2)).map(|i| ad[i]);
            let b = axis.checked_sub(batches - (bd.len() - 2)).map(|i| bd[i]);
            p.push(match (a, b) {
                (Some(a), Some(b)) if a.to_usize() == Some(1) => b,
                (Some(a), Some(b)) if b.to_usize() == Some(1) => a,
                (Some(a), Some(b)) => {
                    assert_dims_eq(a, b, "batch dim");
                    a
                }
                (Some(d), None) | (None, Some(d)) => d,
                (None, None) => unreachable!(),
            });
        }
        p.extend([m, n, k]);
        // Direct maps into [..., M, N, K] preserve the contraction's canonical
        // form at every rank. A broadcast batch reads coordinate zero.
        let batch_map = |dims: &[IntExpr]| {
            let rank = dims.len() - 2;
            (0..rank)
                .map(|axis| {
                    if dims[axis].to_usize() == Some(1) {
                        MapEntry::Lit(0.into())
                    } else {
                        c(rank - axis + 2, dims[axis])
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut am = batch_map(&ad);
        am.extend([c(2, m), c(0, k)]);
        let mut bm = batch_map(&bd);
        bm.extend([c(0, k), c(1, n)]);
        let mut out = if k.to_usize() == Some(0) {
            // An empty contraction is the additive identity, with no operand reads.
            // Record it before constructing maps over an empty reduction domain.
            p.pop();
            self.graph().constant_i32(0).cast(self.dtype).expand_rhs(p)
        } else {
            let product = self.record_view_map(am, p.clone()) * rhs.record_view_map(bm, p);
            product.sum(vec![batches + 2])
        };
        if rhs_vector {
            out = out.squeeze(batches + 1);
        }
        if lhs_vector {
            out = out.squeeze(batches);
        }
        out
    }

    /// Simple dot product of two vectors
    pub fn dot(self, rhs: GraphTensor) -> GraphTensor {
        (self * rhs).sum(vec![0])
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
        let lhs = cx.tensor(vec![2, 4], DType::F8E4M3FN);
        let rhs = cx.tensor(vec![4, 3], DType::F8E4M3FN);

        let out = lhs.cast(DType::F32).matmul(rhs.cast(DType::F32));

        assert_eq!(out.dtype, DType::F32);
        let model = cx.logical.render_all().expect("recorded model");
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
        let lhs = cx.tensor(vec![2, 4], DType::F8E4M3FN);
        let rhs = cx.tensor(vec![4, 3], DType::F8E4M3FN);

        let out = lhs.matmul(rhs);

        assert_eq!(out.dtype, DType::F8E4M3FN);
        let model = cx.logical.render_all().expect("recorded model");
        assert_eq!(
            model.matches("(LogicalCast").count(),
            0,
            "matmul must insert no casts:\n{model}"
        );
    }

    #[test]
    #[should_panic(expected = "cast explicitly")]
    fn mixed_dtype_matmul_refuses_at_construction() {
        let mut cx = Graph::new();
        let lhs = cx.tensor(vec![2, 4], DType::F32);
        let rhs = cx.tensor(vec![4, 3], DType::F16);
        let _ = lhs.matmul(rhs);
    }

    /// An Int x F32 pair must refuse with the cast-explicitly message,
    /// never by tripping the float->int cast refusal first.
    #[test]
    #[should_panic(expected = "cast explicitly")]
    fn int_float_matmul_refuses_with_the_cast_message() {
        let mut cx = Graph::new();
        let lhs = cx.tensor(vec![2, 4], DType::Int);
        let rhs = cx.tensor(vec![4, 3], DType::F32);
        let _ = lhs.matmul(rhs);
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

    #[test]
    fn empty_contractions_are_zero_without_operand_reads() {
        for (ad, bd, shape) in [
            (vec![2, 0], vec![0, 3], vec![2, 3]),
            (vec![0], vec![0], vec![]),
        ] {
            let mut cx = Graph::new();
            let a = cx.tensor(ad, DType::F32);
            let b = cx.tensor(bd, DType::F32);
            let out = a.matmul(b);
            assert_eq!(
                out.dims()
                    .iter()
                    .map(|d| d.to_usize().unwrap())
                    .collect::<Vec<_>>(),
                shape
            );
            let rt = luminal_reference::harness::run_reference(
                &cx,
                &[
                    (a.id, Vec::<f32>::new().into()),
                    (b.id, Vec::<f32>::new().into()),
                ],
            );
            assert!(rt.get_f32(out.id).unwrap().iter().all(|&v| v == 0.));
        }
    }

    #[test]
    fn vector_batch_reads_every_batch() {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![5], DType::F32);
        let b = cx.tensor(vec![2, 5, 3], DType::F32);
        let out = a.matmul(b);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (a.id, vec![1f32, 2., 3., 4., 5.].into()),
                (b.id, (0..30).map(|i| i as f32).collect::<Vec<_>>().into()),
            ],
        );
        assert_eq!(
            rt.get_f32(out.id).unwrap(),
            &[120., 135., 150., 345., 360., 375.]
        );
    }

    #[test]
    fn vector_and_broadcast_matmuls_match_reference() {
        test_binary(
            vec![5],
            vec![5],
            |a, b| a.matmul(b),
            |a, b| a.mul(&b).unwrap().sum_all().unwrap(),
        );
        test_binary(
            vec![3, 5],
            vec![5],
            |a, b| a.matmul(b),
            |a, b| {
                a.matmul(&b.reshape((5, 1)).unwrap())
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
            },
        );
        test_binary(
            vec![5],
            vec![2, 5, 3],
            |a, b| a.matmul(b),
            |a, b| {
                a.reshape((1, 1, 5))
                    .unwrap()
                    .broadcast_as((2, 1, 5))
                    .unwrap()
                    .contiguous()
                    .unwrap()
                    .matmul(&b)
                    .unwrap()
                    .squeeze(1)
                    .unwrap()
            },
        );
        test_binary(
            vec![2, 1, 3, 5],
            vec![4, 5, 2],
            |a, b| a.matmul(b),
            |a, b| {
                a.broadcast_as((2, 4, 3, 5))
                    .unwrap()
                    .contiguous()
                    .unwrap()
                    .reshape((8, 3, 5))
                    .unwrap()
                    .matmul(
                        &b.broadcast_as((2, 4, 5, 2))
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((8, 5, 2))
                            .unwrap(),
                    )
                    .unwrap()
            },
        );
        test_binary(
            vec![2, 1, 2, 1, 3, 5],
            vec![2, 4, 5, 2],
            |a, b| a.matmul(b),
            |a, b| {
                a.broadcast_as((2, 1, 2, 4, 3, 5))
                    .unwrap()
                    .contiguous()
                    .unwrap()
                    .reshape((16, 3, 5))
                    .unwrap()
                    .matmul(
                        &b.broadcast_as((2, 1, 2, 4, 5, 2))
                            .unwrap()
                            .contiguous()
                            .unwrap()
                            .reshape((16, 5, 2))
                            .unwrap(),
                    )
                    .unwrap()
            },
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_matrix_vector(m in 1usize..6, k in 1usize..6, n in 1usize..6) {
            test_binary(
                vec![m, k],
                vec![k, n],
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
                vec![m, k],
                vec![k, n],
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
                vec![batch, m, k],
                vec![k, n],
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
                vec![batch, m, k],
                vec![batch, m, k],
                |a, b| a.matmul(b.permute(vec![0, 2, 1])),
                |a, b| a.matmul(&b.permute((0, 2, 1)).unwrap()).unwrap(),
            );
            test_binary(
                vec![batch, m, k],
                vec![batch, k, n],
                |a, b| a.matmul(b),
                |a, b| a.matmul(&b).unwrap(),
            );
        }
    }
}
