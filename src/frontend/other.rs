use crate::prelude::*;

impl Graph {
    /// A scalar expression constant
    pub fn constant(&mut self, i: impl Into<IntExpr>) -> GraphTensor {
        let expr = i.into();
        let id = self
            .logical
            .record_iota(&expr, &[])
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, (), self, DType::Int)
    }

    /// A scalar float constant
    pub fn constant_float(&mut self, i: f32) -> GraphTensor {
        let id = self
            .logical
            .op(LogicalOp::Constant(i as f64), &[], Vec::new(), DType::F32)
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, (), self, DType::F32)
    }

    /// Iota as a TRUE COORDINATE FUNCTION (P1 ruling 2026-08-07): the
    /// closure receives one coordinate IntExpr per output axis
    /// (`c[k]` ranges over `0..shape[k]`) and returns the value
    /// expression. Coordinates lower to `CoordVar`; named symbols (any
    /// char — nothing is reserved, `'z'` included) lower to `IntVar` and
    /// resolve at binding time. The old flat-'z' interface is DELETED —
    /// flat-index authoring is a rank-1 iota plus recorded reshapes.
    /// Coordinate Expressions are positional: one that escapes into
    /// another iota's closure means that iota's same-numbered axis —
    /// defined behavior, the expression does what it says; escaping
    /// anywhere else refuses loudly (dim positions reject compound
    /// terms, out-of-range axes poison the recorder).
    pub fn iota(
        &mut self,
        shape: impl ToShape,
        f: impl FnOnce(&[IntExpr]) -> IntExpr,
    ) -> GraphTensor {
        let sh = shape.to_shape();
        let coords: Vec<IntExpr> = (0..sh.len()).map(IntExpr::coord).collect();
        // Frontend simplification restored (Austin's revert ruling
        // 2026-08-27): the recorded value expression is
        // construction-simplified, as pre-R-C.
        let expr = f(&coords).simplify();
        let id = self
            .logical
            .record_iota(&expr, &sh)
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, sh, self, DType::Int)
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<IntExpr>) -> GraphTensor {
        self.iota(to, |c| c[0])
    }

    /// ARange from beginning to end
    pub fn arange_options(
        &mut self,
        start: impl Into<IntExpr>,
        end: impl Into<IntExpr>,
        step: impl Into<IntExpr>,
    ) -> GraphTensor {
        let (start, end, step) = (start.into(), end.into(), step.into());
        self.iota((end - start) / step, move |c| c[0] * step + start)
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<IntExpr>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).cast(DType::F32).expand_dim(0, size);
        let vertical = self.arange(size).cast(DType::F32).expand_dim(1, size);
        (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<IntExpr>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).cast(DType::F32).expand_dim(0, size);
        let vertical = self.arange(size).cast(DType::F32).expand_dim(1, size);
        (horizontal - (diagonal as f32 - 1.)).gt(vertical)
    }

    /// Stack tensors along a new dimension
    pub fn stack(&mut self, tensors: &[GraphTensor], axis: usize) -> GraphTensor {
        assert!(!tensors.is_empty(), "Cannot stack empty tensor list");
        let first = tensors[0].unsqueeze(axis);
        tensors[1..]
            .iter()
            .fold(first, |acc, t| acc.concat_along(t.unsqueeze(axis), axis))
    }
}

impl GraphTensor {
    pub fn cast(self, dtype: DType) -> GraphTensor {
        // Cast policy (2026-08-11): float -> int is a REFUSAL — a
        // rounding/truncation is a lossy read and must be an explicit
        // op, never a cast (the Bool8 projection rule generalized).
        // Refused at authoring time so the author sees it, not the
        // search.
        let float_source = matches!(
            self.dtype,
            DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
        );
        // The narrow integers joined the executable set on 2026-09-02
        // (main #399). Their carve-out is about integer WIDTH — a
        // narrowing int -> int cast truncates — and NOT a licence to
        // make a lossy float read implicit, so they are int targets
        // here like every other integer. (I4/U4/U16 have no storage and
        // no kernel, so a cast to them refuses at the plan instead.)
        let int_target = matches!(
            dtype,
            DType::Int | DType::I64 | DType::I8 | DType::U8 | DType::I16
        );
        assert!(
            !(float_source && int_target),
            "cast {:?} -> {:?} is refused: a float -> int conversion is a \
             lossy read and must appear as an explicit op in the model, \
             never as a cast (typed-buffers ruling 2026-08-11)",
            self.dtype,
            dtype
        );
        if self.dtype == dtype {
            return self;
        }
        if dtype == DType::Bool {
            // The Bool8 ruling: to-Bool is the `!= 0` PROJECTION — an
            // explicit comparison in the model, never a cast. x != 0 iff
            // (0 < x) + (x < 0) is nonzero, and that sum is {0,1}-valued
            // (x cannot be on both sides of 0), so one more `0 <` lands
            // it in Bool exactly — total, and exact at every float.
            let zero = self
                .graph()
                .constant_float(0.0)
                .cast(self.dtype)
                .expand_rhs(self.dims());
            let sum = zero.lt(self).cast(DType::F32) + self.lt(zero).cast(DType::F32);
            let zero_f32 = self.graph().constant_float(0.0).expand_rhs(self.dims());
            return zero_f32.lt(sum);
        }
        let operand = (self.id, self.dims());
        let out_dims = self.dims();
        let id = self
            .graph()
            .logical
            .op(LogicalOp::Cast(dtype), &[operand], out_dims, dtype)
            .unwrap_or_else(crate::graph::unrecorded_value);
        GraphTensor::from_id(id, self.dims(), self.graph_ref, dtype)
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::assert_close;
    use candle_core::{Device, Tensor};
    use luminal::prelude::*;
    use proptest::prelude::*;

    pub fn test_init(
        func: impl Fn(&mut Graph) -> GraphTensor,
        ref_func: impl Fn(&Device) -> Tensor,
    ) {
        let mut cx = Graph::new();
        let b = func(&mut cx).output();

        let rt = luminal_reference::harness::run_reference(&cx, &[]);

        // Reference
        let device = Device::Cpu;
        let ref_b = ref_func(&device).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id).unwrap(), &ref_b.to_vec1::<f32>().unwrap())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_arange(end in 1i32..64) {
            test_init(
                |cx| cx.arange(end).cast(DType::F32) * 1.0,
                |dev| Tensor::arange(0_f32, end as f32, dev).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_arange_options(start in -16i32..16, step in 1i32..6, count in 1i32..20) {
            let end = start + step * count;
            test_init(
                |cx| cx.arange_options(start, end, step).cast(DType::F32) * 1.0,
                |dev| {
                    let values = (0..count)
                        .map(|i| (start + step * i) as f32)
                        .collect::<Vec<f32>>();
                    Tensor::from_vec(values, count as usize, dev).unwrap()
                },
            );
        }
    }

    // test_gather: B-TAIL-GATED (Step 4b). gather1d is the flat gather
    // the recorder still poisons; coordinate-form gather carries the
    // native differential coverage until the B-tail records the sugar.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_triangle_mask(size in 1usize..64) {
            test_init(
                |cx| cx.tril(size as i32, 0).cast(DType::F32),
                |dev| Tensor::tril2(size, candle_core::DType::F32, dev).unwrap(),
            );
            test_init(
                |cx| cx.triu(size as i32, 0).cast(DType::F32),
                |dev| Tensor::triu2(size, candle_core::DType::F32, dev).unwrap(),
            );
        }
    }

    #[test]
    fn test_stack() {
        use crate::tests::random_vec;

        let mut cx = Graph::new();
        let a = cx.tensor((2, 3));
        let b = cx.tensor((2, 3));
        let c = cx.tensor((2, 3));
        let stacked = cx.stack(&[a, b, c], 0).output();

        let a_data = random_vec(6);
        let b_data = random_vec(6);
        let c_data = random_vec(6);
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (a.id, a_data.clone().into()),
                (b.id, b_data.clone().into()),
                (c.id, c_data.clone().into()),
            ],
        );

        let ref_a = Tensor::new(a_data, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        let ref_b = Tensor::new(b_data, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        let ref_c = Tensor::new(c_data, &Device::Cpu)
            .unwrap()
            .reshape((2, 3))
            .unwrap();
        let ref_stacked = Tensor::stack(&[&ref_a, &ref_b, &ref_c], 0).unwrap();

        assert_close(
            rt.get_f32(stacked.id).unwrap(),
            &ref_stacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }
}
