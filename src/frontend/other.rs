use crate::hlir::*;
use crate::prelude::*;

impl Graph {
    /// A scalar expression constant
    pub fn constant(&mut self, i: impl Into<Expression>) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Iota(i.into(), 1.into())).finish(),
            ShapeTracker::new(()),
            self,
            DType::Int,
        )
    }

    /// A scalar float constant
    pub fn constant_float(&mut self, i: f32) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Constant(i)).finish(),
            ShapeTracker::new(()),
            self,
            DType::F32,
        )
    }

    /// Iota expression
    pub fn iota(&mut self, i: impl Into<Expression>, shape: impl ToShape) -> GraphTensor {
        let sh = shape.to_shape();
        GraphTensor::from_id(
            self.add_op(Iota(
                i.into().simplify(),
                sh.iter().copied().product::<Expression>().simplify(),
            ))
            .finish(),
            ShapeTracker::new(sh),
            self,
            DType::Int,
        )
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<Expression>) -> GraphTensor {
        self.iota('z', to)
    }

    /// ARange from beginning to end
    pub fn arange_options(
        &mut self,
        start: impl Into<Expression>,
        end: impl Into<Expression>,
        step: impl Into<Expression>,
    ) -> GraphTensor {
        let (start, end, step) = (start.into(), end.into(), step.into());
        self.iota((Expression::from('z') * step) + start, (end - start) / step)
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size);
        let vertical = self.arange(size).expand_dim(1, size);
        (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size);
        let vertical = self.arange(size).expand_dim(1, size);
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
        if self.dtype == dtype {
            return self;
        }
        let id = self
            .graph()
            .add_op(Cast(self.shape.n_physical_elements(), dtype))
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref, dtype)
    }

    /// Sets this tensor's dtype without doing a cast
    pub fn as_dtype(mut self, dtype: DType) -> GraphTensor {
        self.dtype = dtype;
        if let Some(gmem) = self.graph().try_get_op_mut::<Input>(self.id) {
            gmem.dtype = dtype;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, tests::assert_close};
    use candle_core::{Device, Tensor};
    use proptest::prelude::*;

    pub fn test_init(
        func: impl Fn(&mut Graph) -> GraphTensor,
        ref_func: impl Fn(&Device) -> Tensor,
    ) {
        let mut cx = Graph::new();
        let b = func(&mut cx).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = ref_func(&device).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_gather(base_len in 5usize..64, count in 1usize..16) {
            prop_assume!(base_len >= 2 * count - 1);
            test_init(
                |cx| {
                    cx.arange(base_len as i32)
                        .cast(DType::F32)
                        .gather(cx.iota(Expression::from('z') * 2, count as i32))
                },
                |dev| {
                    let values = (0..count).map(|i| (2 * i) as f32).collect::<Vec<f32>>();
                    Tensor::new(values, dev).unwrap()
                },
            );
        }
    }

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

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        let a_data = random_vec(6);
        let b_data = random_vec(6);
        let c_data = random_vec(6);
        rt.set_data(a.id, a_data.clone());
        rt.set_data(b.id, b_data.clone());
        rt.set_data(c.id, c_data.clone());
        rt.execute(&cx.dyn_map);

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
            rt.get_f32(stacked.id),
            &ref_stacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }
}
