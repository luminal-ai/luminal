use itertools::Itertools;

use crate::{hlir::Gather, prelude::*};

impl GraphTensor {
    /// Swap dimensions of the tensor
    pub fn permute(mut self, axes: impl ToAxes) -> GraphTensor {
        self.shape.permute(&axes.to_axes());
        self
    }

    /// Swap 2 dimensions. This is a view-only operation and does not materialize a new tensor
    pub fn transpose(self, dim0: usize, dim1: usize) -> GraphTensor {
        let num_dims = self.shape.len();
        assert!(
            dim0 < num_dims && dim1 < num_dims,
            "transpose dimensions ({dim0}, {dim1}) out of bounds for tensor with {num_dims} dimensions"
        );
        let mut perm_axes: Vec<usize> = (0..num_dims).collect();
        perm_axes.swap(dim0, dim1);
        self.permute(perm_axes)
    }

    /// Transpose a 2D tensor
    pub fn t(self) -> GraphTensor {
        assert_eq!(self.shape.len(), 2, ".t() supports only 2D tensors");
        self.transpose(0, 1)
    }

    /// Broadcast tensor along a new dimension
    pub fn expand_dim(mut self, axis: usize, size: impl Into<Expression>) -> GraphTensor {
        self.shape.expand_dim(axis, size);
        self
    }

    /// Broadcast tensor along new dimensions on the right-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    pub fn expand_rhs(mut self, shape: impl ToShape) -> GraphTensor {
        let orig_dims = self.shape.len();
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self.shape.expand_dim(orig_dims + i, s);
        }
        self
    }

    /// Broadcast tensor along new dimensions on the left-hand-side. For instance, if the original tensor is [5, 2] and you call .expand([4, 2, 3]), the final  tensor will be [5, 2, 4, 2, 3]
    pub fn expand_lhs(mut self, shape: impl ToShape) -> GraphTensor {
        for (i, s) in shape.to_shape().into_iter().enumerate() {
            self.shape.expand_dim(i, s);
        }
        self
    }

    pub fn expand_to_shape_on_axes(
        mut self,
        shape: impl ToShape,
        axes: impl ToAxes,
    ) -> GraphTensor {
        let shape = shape.to_shape();
        let axes = axes.to_axes();
        assert_eq!(shape.len(), self.shape.len() + axes.len());
        for axis in axes.into_iter().sorted() {
            self = self.expand_dim(axis, shape[axis]);
        }
        self
    }

    /// Merge two dimensions together
    pub fn merge_dims(mut self, axis1: usize, axis2: usize) -> GraphTensor {
        self.shape.merge_dims(axis1, axis2);
        self
    }

    //// Split a dim into 2 dims, new dim is placed directly after original dim
    pub fn split_dims(mut self, axis: usize, new_dim_size: impl Into<Expression>) -> GraphTensor {
        self.shape.split_dims(axis, new_dim_size);
        self
    }

    /// add a new dimension of size 1 at the specified place
    pub fn unsqueeze(mut self, dim: usize) -> GraphTensor {
        assert!(self.shape.len() < 10, "Shape is maxed out at 10 dimensions");
        self.shape.expand_dim(dim, 1);
        self
    }

    /// remove a dimension of size 1
    pub fn squeeze(mut self, axis: usize) -> GraphTensor {
        assert_eq!(
            self.dims()[axis],
            Expression::from(1),
            "Only dimensions of size 1 can be squeezed!"
        );
        self.shape.remove_dim(axis);
        self
    }

    pub fn gather(self, indexes: GraphTensor) -> GraphTensor {
        assert_eq!(
            indexes.dtype,
            DType::Int,
            "Gather indexes must have an integer dtype!"
        );
        let id = self
            .graph()
            .add_op(Gather::default())
            .input(indexes.id, indexes.shape)
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(id, indexes.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Given a tensor of non-repeating indexes along a dimension, generate an inverse permutation
    /// x = [3, 2, 4, 1, 5, 0]
    /// inv_perm(x) = [5, 3, 1, 0, 2, 4]
    #[allow(clippy::needless_range_loop)]
    pub fn inverse_permutation(self, axis: usize) -> GraphTensor {
        // TODO: this is super inefficient because it requires materializing a large (n^2) one-hot tensor
        assert_eq!(self.dtype, DType::Int);
        let dims = self.dims();
        let ax_size = dims[axis];
        let mut dims2 = dims.clone();
        dims2.insert(axis, ax_size);
        // candidate: varies along candidate dim (axis), broadcast elsewhere.
        let mut candidate = self.graph().arange(ax_size);
        for i in 0..axis {
            candidate = candidate.expand_dim(i, dims2[i]);
        }
        for i in axis + 1..dims2.len() {
            candidate = candidate.expand_dim(i, dims2[i]);
        }
        // position: varies along position dim (axis+1), broadcast elsewhere.
        let mut position = self.graph().arange(ax_size);
        for i in 0..(axis + 1) {
            position = position.expand_dim(i, dims2[i]);
        }
        for i in (axis + 2)..dims2.len() {
            position = position.expand_dim(i, dims2[i]);
        }
        // one_hot[candidate, ..., position, ...] = (self[position, ...] == candidate)
        let one_hot = self.expand_dim(axis, ax_size).eq(candidate);
        // inv[candidate, ...] = Σ_pos one_hot * position
        (one_hot * position).sum(axis + 1)
    }

    /// Extracts sliding local windows from an input tensor.
    pub fn unfold(
        self,
        kernel: impl ToShape,
        strides: impl ToShape,
        dilation: impl ToShape,
    ) -> GraphTensor {
        let (kernel, strides, dilation) =
            (kernel.to_shape(), strides.to_shape(), dilation.to_shape());

        assert_eq!(
            self.shape.len(),
            kernel.len(),
            "Kernel must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.shape.len(),
            strides.len(),
            "Strides must be same number of dimensions as tensor!"
        );
        assert_eq!(
            self.shape.len(),
            dilation.len(),
            "Dilation must be same number of dimensions as tensor!"
        );

        // Compute input strides (row-major contiguous)
        let dims = self.dims();
        let n = dims.len();
        let mut in_strides = vec![Expression::from(1); n];
        let mut acc = Expression::from(1);
        for (dim, in_stride) in dims.iter().zip(&mut in_strides).rev() {
            *in_stride = acc;
            acc *= dim;
        }

        // Per-dim window counts
        let mut win = Vec::with_capacity(n);
        for (((dim, k), s), d) in dims.iter().zip(&kernel).zip(&strides).zip(&dilation) {
            let effective_window = *d * (*k - 1) + 1;
            win.push(((*dim - effective_window) / s) + 1);
        }

        // [win..., kernel...]
        let mut final_shape: Vec<Expression> = win.into_iter().map(|e| e.simplify()).collect();
        final_shape.extend(kernel.iter().copied());

        // Axis exprs must match final_shape axis order: first w axes, then k axes.
        // idx = Σ_d (w_d * stride_d + k_d * dilation_d) * in_strides[d]
        let mut axis_exprs = Vec::with_capacity(2 * n);

        // w axes
        for i in 0..n {
            axis_exprs.push(Expression::from('z') * strides[i] * in_strides[i]);
        }
        // k axes
        for i in 0..n {
            axis_exprs.push(Expression::from('z') * dilation[i] * in_strides[i]);
        }

        let index_expression = flatten_z_strides(&final_shape, &axis_exprs).simplify();
        let iota = self.graph().iota(index_expression, final_shape);
        self.gather(iota)
    }

    /// Take a slice of a tensor along multiple dimensions.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
    /// let b = a.slice((2..4, 1..)); // 2x9 tensor
    /// assert_eq!(b.dims(), vec![Expression::from(2), Expression::from(9)]);
    /// ```
    pub fn slice(mut self, slice: impl ToSlice) -> GraphTensor {
        let mut ranges = slice.to_range_vec();
        ranges.extend(
            self.dims()
                .iter()
                .skip(ranges.len())
                .map(|d| (0.into(), *d)),
        ); // Make sure we have a range per dim
        if ranges.iter().any(|(st, _)| *st != 0) {
            // We have a start slice, need to use an iota because tensors don't have offsets
            let mut new_dims = vec![];
            let mut index_expressions = vec![];
            let mut phys_size = Expression::from(1);
            for (dim, (start, end)) in self.dims().into_iter().zip(ranges).rev() {
                index_expressions.push((Expression::from('z') + start) * phys_size);
                phys_size *= dim;
                new_dims.push(dim.min(end) - start);
            }
            new_dims.reverse();
            index_expressions.reverse();
            let index_expression = flatten_z_strides(&new_dims, &index_expressions);
            let iota = self.graph().iota(index_expression, new_dims);
            self.gather(iota)
        } else {
            // No start slices so no iota needed, just reduce the shape down
            for (sh, (_, end)) in self.shape.dims.iter_mut().zip(ranges) {
                *sh = sh.min(end);
            }
            self
        }
    }

    /// Take a slice of a tensor along a dimension.
    ///
    /// ```
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor((5, 10));
    /// let b = a.slice_along(4.., 1); // 5x6 tensor
    /// assert_eq!(b.dims(), vec![Expression::from(5), Expression::from(6)]);
    /// ```
    pub fn slice_along(self, slice: impl SliceRange, axis: usize) -> GraphTensor {
        let mut s = vec![(Expression::from(0), Expression::from(i32::MAX)); axis + 1];
        s[axis] = slice.bounds();
        self.slice(s)
    }

    // /// Cut out 'size' elements every 'spacing' elements on a dimension. 'size' must be smaller than the dimension
    // pub fn excise(mut self, spacing: usize, size: usize) -> GraphTensor {
    //     let n_dims = self.shape.len();
    //     // Pad out to a multiple of spacing + size
    //     let total_size = (self.shape.dims[n_dims - 1] + ((spacing + size) - 1))
    //         / (spacing + size)
    //         * (spacing + size);
    //     let padding = total_size - self.shape.dims[self.shape.indexes[n_dims - 1]];
    //     self.shape.padding[self.shape.indexes[n_dims - 1]].1 = padding;

    //     self = self.contiguous();
    //     // Expand a new dimension to do the slicing on
    //     let n_rows = total_size / (spacing + size);
    //     self.shape.expand_dim(n_dims, spacing + size);
    //     // self = self.contiguous();
    //     self.shape.dims[self.shape.indexes[n_dims - 1]] = n_rows;
    //     self.shape.fake[self.shape.indexes[n_dims]] = false;

    //     // Slice
    //     self.shape.mask[self.shape.indexes[n_dims]].1 = spacing.into();

    //     self = self.contiguous();

    //     self.shape.remove_dim(n_dims);
    //     self
    // }

    /// Pad out dimensions of a tensor with an element
    pub fn pad(self, padding: impl ToPad, elem: f32) -> GraphTensor {
        let mut padding = padding.to_pad_vec();
        padding.extend(vec![(0.into(), 0.into()); self.shape.len() - padding.len()]); // Make sure we have a padding per dim
        let mut index_expressions = vec![];
        let mut phys_size = Expression::from(1);
        let mut new_dims = vec![];
        for (dim, (start, end)) in self.dims().into_iter().zip(&padding).rev() {
            let mut ind = Expression::from('z');
            if *start != 0 {
                ind = (ind - *start).max(0);
            }
            if *end != 0 {
                ind = ind.min(dim - 1);
            }
            index_expressions.push(ind * phys_size);
            phys_size *= dim;
            new_dims.push(dim + *start + *end);
        }
        new_dims.reverse();
        index_expressions.reverse();
        let index_expression = flatten_z_strides(&new_dims, &index_expressions);
        // get indexed tensor
        let new_tensor = self.gather(self.graph().iota(index_expression, new_dims.clone()));
        // mask out padded elements
        let mut mask_expressions = vec![];
        for ((start, end), dim) in padding.into_iter().zip(self.dims()) {
            let mut mask = Expression::from(1);
            if start != 0 {
                mask *= Expression::from('z').gte(start);
            }
            if end != 0 {
                mask *= Expression::from('z').lt(start + dim);
            }
            mask_expressions.push(mask);
        }
        let mask_expression = flatten_z_strides_mask(&new_dims, &mask_expressions);
        let mask = self.graph().iota(mask_expression, new_dims);
        let masked = new_tensor * mask;
        if elem == 0.0 {
            masked
        } else {
            masked + ((1. - mask) * elem)
        }
    }

    /// Pad along an existing dimension
    pub fn pad_along(
        self,
        left: impl Into<Expression>,
        right: impl Into<Expression>,
        axis: usize,
        elem: f32,
    ) -> GraphTensor {
        let mut p = vec![(Expression::from(0), Expression::from(0)); axis + 1];
        p[axis] = (left.into(), right.into());
        self.pad(p, elem)
    }

    /// Concat along an existing dimension
    pub fn concat_along(self, rhs: GraphTensor, axis: usize) -> GraphTensor {
        // Pad and add
        self.pad_along(0, rhs.dims()[axis], axis, 0.)
            + rhs.pad_along(self.dims()[axis], 0, axis, 0.)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        frontend::{binary::tests::test_binary, unary::tests::test_unary},
        op::DType,
        prelude::*,
    };
    use candle_core::{IndexOp, Tensor};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_1d(len in 1usize..64, left in 0usize..6, right in 0usize..6) {
            test_unary(
                len,
                |a| a.pad((left, right), 0.),
                |a| a.pad_with_zeros(0, left, right).unwrap(),
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_pad_2d(rows in 1usize..32, cols in 1usize..32, top in 0usize..6, bottom in 0usize..6, left in 0usize..6, right in 0usize..6) {
            test_unary(
                (rows, cols),
                |a| a.pad(((top, bottom), (left, right)), 0.),
                |a| {
                    a.pad_with_zeros(0, top, bottom)
                        .unwrap()
                        .pad_with_zeros(1, left, right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_slice_pad(
            rows in 3usize..32,
            cols in 3usize..32,
            start_row in 0usize..32,
            end_row in 1usize..32,
            start_col in 0usize..32,
            end_col in 1usize..32,
            pad_top in 0usize..6,
            pad_bottom in 0usize..6,
            pad_left in 0usize..6,
            pad_right in 0usize..6,
        ) {
            prop_assume!(start_row < end_row && end_row <= rows);
            prop_assume!(start_col < end_col && end_col <= cols);
            test_unary(
                (rows, cols),
                |a| a.slice((start_row..end_row, start_col..end_col)).pad(((pad_top, pad_bottom), (pad_left, pad_right)), 0.),
                |a| {
                    a.i((start_row..end_row, start_col..end_col))
                        .unwrap()
                        .pad_with_zeros(0, pad_top, pad_bottom)
                        .unwrap()
                        .pad_with_zeros(1, pad_left, pad_right)
                        .unwrap()
                },
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_transpose(rows in 1usize..32, cols in 1usize..32) {
            test_unary(
                (rows, cols),
                |a| a.transpose(0, 1) * 1.0,
                |a| a.transpose(0, 1).unwrap(),
            );
        }
    }

    #[test]
    fn test_unfold() {
        // Need all this code because candle doesnt do unfold
        #[allow(clippy::too_many_arguments)]
        pub fn unfold_nd_f32(
            x: &[f32],
            shape: &[usize],
            strides: &[usize],
            kernel: &[usize],
            step: &[usize],
            dilation: &[usize],
            pad_before: &[usize],
            pad_after: &[usize],
        ) -> Vec<f32> {
            let n = shape.len();
            assert!(n > 0);
            assert_eq!(strides.len(), n);
            assert_eq!(kernel.len(), n);
            assert_eq!(step.len(), n);
            assert_eq!(dilation.len(), n);
            assert_eq!(pad_before.len(), n);
            assert_eq!(pad_after.len(), n);

            for d in 0..n {
                assert!(kernel[d] > 0);
                assert!(step[d] > 0);
                assert!(dilation[d] > 0);
                assert!(shape[d] > 0);
            }

            // Effective kernel size per dim: (K-1)*d + 1
            let eff_kernel: Vec<usize> =
                (0..n).map(|d| (kernel[d] - 1) * dilation[d] + 1).collect();

            // Output spatial shape (number of windows) per dim
            let mut out_shape = vec![0usize; n];
            for d in 0..n {
                let padded = shape[d] + pad_before[d] + pad_after[d];
                if padded < eff_kernel[d] {
                    return Vec::new();
                }
                out_shape[d] = (padded - eff_kernel[d]) / step[d] + 1;
            }

            let windows = prod(&out_shape);
            let window_elems = prod(kernel);
            let mut out = vec![0.0f32; windows * window_elems];

            // Precompute helpers
            let k_mul = row_major_multipliers(kernel);

            // Current output window position (row-major)
            let mut out_pos = vec![0usize; n];

            for w in 0..windows {
                if w > 0 {
                    incr_row_major(&mut out_pos, &out_shape);
                }

                // Window start in padded coordinates
                let start_padded: Vec<usize> = (0..n).map(|d| out_pos[d] * step[d]).collect();

                let base_out = w * window_elems;

                // Iterate kernel elements (flattened)
                for ke in 0..window_elems {
                    let k_idx = unravel_row_major(ke, kernel, &k_mul);

                    let mut flat: isize = 0;
                    let mut in_bounds = true;

                    for d in 0..n {
                        let p = start_padded[d] + k_idx[d] * dilation[d];
                        let logical = p as isize - pad_before[d] as isize;

                        if logical < 0 || logical >= shape[d] as isize {
                            in_bounds = false;
                            break;
                        }
                        flat += logical * strides[d] as isize;
                    }

                    let out_idx = base_out + ke;
                    out[out_idx] = if in_bounds { x[flat as usize] } else { 0.0 };
                }
            }

            out
        }

        // -------- helpers --------

        fn prod(xs: &[usize]) -> usize {
            xs.iter().copied().product()
        }

        fn row_major_multipliers(shape: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut mul = vec![1usize; n];
            let mut acc = 1usize;
            for d in (0..n).rev() {
                mul[d] = acc;
                acc *= shape[d];
            }
            mul
        }

        fn unravel_row_major(mut idx: usize, shape: &[usize], mul: &[usize]) -> Vec<usize> {
            let n = shape.len();
            let mut coords = vec![0usize; n];
            for d in 0..n {
                coords[d] = idx / mul[d];
                idx %= mul[d];
            }
            coords
        }

        fn incr_row_major(pos: &mut [usize], shape: &[usize]) {
            for d in (0..pos.len()).rev() {
                pos[d] += 1;
                if pos[d] < shape[d] {
                    return;
                }
                pos[d] = 0;
            }
        }

        test_unary(
            5,
            |a| a.unfold(3, 1, 1),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[3],
                        &[1],
                        &[1],
                        &[0],
                        &[0],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
        test_unary(
            (8, 10),
            |a| a.pad(((0, 2), (4, 4)), 0.).unfold((2, 3), (1, 2), (2, 1)),
            |a| {
                Tensor::new(
                    unfold_nd_f32(
                        &a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                        a.dims(),
                        a.stride(),
                        &[2, 3],
                        &[1, 2],
                        &[2, 1],
                        &[0, 4],
                        &[2, 3],
                    ),
                    a.device(),
                )
                .unwrap()
            },
        );
    }

    #[test]
    fn test_unsqueeze() {
        let mut cx = Graph::new();
        let inp = cx.tensor((2, 2, 3));
        let out1 = inp.unsqueeze(1);
        let out2 = inp.unsqueeze(3);
        assert_eq!(out1.dims(), &[2, 1, 2, 3]);
        assert_eq!(out2.dims(), &[2, 2, 3, 1]);
        test_unary(
            (1, 3),
            |a| a.squeeze(0).expand_dim(0, 2) * 1.,
            |a| a.broadcast_as((2, 3)).unwrap(),
        );
        test_unary((2, 1, 3), |a| a.squeeze(1), |a| a.reshape((2, 3)).unwrap());
    }

    #[test]
    fn test_concat() {
        test_binary(
            17,
            32,
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_binary(
            (10, 4),
            (10, 6),
            |a, b| a.concat_along(b, 1),
            |a, b| Tensor::cat(&[a, b], 1).unwrap(),
        );
        test_binary(
            (4, 10),
            (6, 10),
            |a, b| a.concat_along(b, 0),
            |a, b| Tensor::cat(&[a, b], 0).unwrap(),
        );
        test_unary(
            (4, 10),
            |a| a.concat_along(a, 0),
            |a| Tensor::cat(&[a.clone(), a], 0).unwrap(),
        );
    }

    #[test]
    fn test_gather_and_inverse_permutation() {
        let mut cx = Graph::new();
        let data = cx.tensor((2, 3));
        let indexes = cx.tensor(4).as_dtype(DType::Int);
        let gathered = data.gather(indexes).output();
        let perm = cx.tensor(6).as_dtype(DType::Int);
        let inv = perm.inverse_permutation(0).cast(DType::F32).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        rt.set_data(data.id, vec![0., 1., 2., 3., 4., 5.]);
        rt.set_data(indexes.id, vec![5, 0, 3, 2]);
        rt.set_data(perm.id, vec![3, 2, 4, 1, 5, 0]);
        rt.execute(&cx.dyn_map);
        assert_eq!(*rt.get_f32(gathered.id), vec![5., 0., 3., 2.]);
        assert_eq!(*rt.get_f32(inv.id), vec![5., 3., 1., 0., 2., 4.]);
    }

    //     // #[test]
    //     // fn test_cumsum() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.constant(1.).expand_dim(0, 3);
    //     //     let b = a.cumsum_last_dim().retrieve();
    //     //     let c = a
    //     //         .expand_dim(1, 3)
    //     //         .permute((1, 0))
    //     //         .cumsum_last_dim()
    //     //         .permute((1, 0))
    //     //         .retrieve();
    //     //     cx.execute();

    //     //     assert_exact(&b.data(), &[1., 2., 3.]);
    //     //     assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_pool_1d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set([1., 2., 3., 4., 5.]);
    //     //     let inp2 = cx
    //     //         .tensor((2, 5))
    //     //         .set([[15., 14., 13., 12., 11.], [1., 2., 3., 4., 5.]]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(3, 2, 1).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out4 = inp1.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out5 = inp1.pad(((1, 1),)).pool_last_dim(3, 1, 2).retrieve();
    //     //     // Stride 1 Batch 2
    //     //     let out6 = inp2.pool_last_dim(3, 1, 1).retrieve();
    //     //     // Stride 3
    //     //     let out7 = inp2.pool_last_dim(3, 3, 1).retrieve();
    //     //     // Dilation 2
    //     //     let out8 = inp2.pool_last_dim(3, 1, 2).retrieve();
    //     //     // Dilation 2 Padding 1
    //     //     let out9 = inp2.pad(((0, 0), (1, 1))).pool_last_dim(3, 1, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 2., 3.]);
    //     //     assert_exact(&out4.data(), &[1., 3., 5.]);
    //     //     assert_exact(&out5.data(), &[0., 2., 4., 1., 3., 5., 2., 4., 0.]);
    //     //     assert_exact(
    //     //         &out6.data(),
    //     //         &[
    //     //             15., 14., 13., 14., 13., 12., 13., 12., 11., 1., 2., 3., 2., 3., 4., 3., 4., 5.,
    //     //         ],
    //     //     );
    //     //     assert_exact(&out7.data(), &[15., 14., 13., 1., 2., 3.]);
    //     //     assert_exact(&out8.data(), &[15., 13., 11., 1., 3., 5.]);
    //     //     assert_exact(
    //     //         &out9.data(),
    //     //         &[
    //     //             0., 14., 12., 15., 13., 11., 14., 12., 0., 0., 2., 4., 1., 3., 5., 2., 4., 0.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dims() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
    //     //             14., 15., 14., 15., 16.,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_2d() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor((4, 4)).set(vec![
    //     //         1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
    //     //     ]);
    //     //     // 3x3 kernel
    //     //     let out1 = inp1
    //     //         // Pool first dim first by moving it to end
    //     //         .permute((1, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now move other dim to end
    //     //         .permute((1, 2, 0))
    //     //         .pool_last_dim(3, 1, 1)
    //     //         // Now swap middle two dims
    //     //         .permute((0, 2, 1, 3))
    //     //         // Now merge both pooled dimensions
    //     //         .reshape((4, 3, 3))
    //     //         .retrieve();

    //     //     cx.execute();

    //     //     assert_exact(
    //     //         &out1.data(),
    //     //         &[
    //     //             1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
    //     //             7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
    //     //             14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
    //     //         ],
    //     //     );
    //     // }

    //     // #[test]
    //     // fn test_pool_1d_dilation() {
    //     //     let mut cx = Graph::new();

    //     //     let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
    //     //     // Stride 1
    //     //     let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
    //     //     // Stride 2
    //     //     let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
    //     //     // Stride 3
    //     //     let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

    //     //     cx.execute();

    //     //     assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
    //     //     assert_exact(&out2.data(), &[1., 3., 3., 5.]);
    //     //     assert_exact(&out3.data(), &[1., 3.]);
    //     // }

    //     // #[test]
    //     // fn test_rotate_half() {
    //     //     let mut cx = Graph::new();
    //     //     let a = cx.tensor((3, 2));
    //     //     a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
    //     //     let x1 = a.slice((.., ..1)).contiguous();
    //     //     let x2 = a.slice((.., 1..)).contiguous();
    //     //     let c = (-x2).concat_along(x1, 1);
    //     //     c.retrieve();
    //     //     cx.execute();

    //     //     let d_dev = Cpu::default();
    //     //     let d_a = d_dev.tensor_from_vec(
    //     //         vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
    //     //         (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
    //     //     );
    //     //     let d_x1 = d_a.clone().slice((.., ..1));
    //     //     let d_x2 = d_a.slice((.., 1..));
    //     //     let d_c = (-d_x2, d_x1)
    //     //         .concat_along(dfdx::shapes::Axis::<1>)
    //     //         .realize::<Rank2<3, 2>>();

    //     //     assert_close(&c.data(), &d_c.as_vec());
    //     // }
}
