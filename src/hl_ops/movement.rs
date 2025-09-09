use crate::{op, prelude::*};

impl GraphTensor {
    /// Swap dimensions of the tensor
    pub fn permute(mut self, axes: impl ToAxes) -> GraphTensor {
        self.shape.permute(&axes.to_axes());
        self
    }

    pub fn transpose(self, dim0: usize, dim1: usize) -> GraphTensor {
        let num_dims = self.shape.len();
        assert!(
            dim0 < num_dims && dim1 < num_dims,
            "transpose dimensions ({dim0}, {dim1}) out of bounds for tensor with {num_dims} dimensions"
        );

        // Create identity permutation, then swap the two specified dimensions
        let mut perm_axes: Vec<usize> = (0..num_dims).collect();
        perm_axes.swap(dim0, dim1);

        self.permute(perm_axes)
    }

    /// Broadcast tensor along a new dimension
    pub fn expand_dim(mut self, axis: usize, size: impl Into<Expression>) -> GraphTensor {
        self.shape.expand_dim(axis, size);
        self
    }

    /// Broadcast tensor along new dimensions (with explicitly given dest shape)
    pub fn expand(mut self, shape: impl ToShape) -> GraphTensor {
        let s = shape.to_shape();
        for (i, s) in s.into_iter().enumerate() {
            if self.shape.len() <= i || self.shape.dims[self.shape.indexes[i]] != s {
                self.shape.expand_dim(i, s);
            }
        }

        self
    }

    /// Convert tensor to a new shape with an equivalent number of elements
    pub fn reshape(mut self, new_shape: impl ToShape) -> GraphTensor {
        // Insert contiguous call
        self = self.contiguous();
        self.shape = ShapeTracker::new(new_shape);
        self
    }

    /// add a new dimension of size 1 at the specified place
    pub fn unsqueeze(mut self, dim: usize) -> GraphTensor {
        // Insert contiguous call
        self = self.contiguous();
        let last_shape = self.shape.dims();
        assert!(last_shape.len() < 6, "Shape is maxed out at 6 dimensions");
        self.shape.expand_dim(dim, 1);
        self
    }

    pub fn contiguous(mut self) -> GraphTensor {
        if !self.shape.is_reshaped() {
            return self;
        }
        self.id = self
            .graph()
            .add_op(op::Contiguous)
            .input(self.id, 0, self.shape)
            .finish();
        self.shape = self.shape.contiguous();
        self
    }

    /// Take a slice of the original tensor. Any dimension with bounds becomes a dynamic dimension
    pub fn slice(mut self, slice: impl ToSlice) -> GraphTensor {
        let ranges = slice.to_range_vec();
        // This exists because currently padding and slicing on the same dimension (even on opposite sides) is unsupported
        if ranges.iter().zip(self.shape.indexes).any(|(range, ind)| {
            (range.0 != 0 || range.1 != i32::MAX)
                && (self.shape.padding[self.shape.indexes[ind]].0 != 0
                    || self.shape.padding[self.shape.indexes[ind]].1 != 0)
        }) {
            self = self.contiguous();
        }
        self.shape.slice(&ranges);
        self
    }

    pub fn slice_along(self, slice: impl SliceRange, axis: usize) -> GraphTensor {
        let mut s = vec![(Expression::from(0), Expression::from(i32::MAX)); axis + 1];
        s[axis] = slice.bounds();
        self.slice(s)
    }

    /// Cut out 'size' elements every 'spacing' elements in the last dimension. 'size' must be smaller than the last dimension
    pub fn excise(mut self, spacing: usize, size: usize) -> GraphTensor {
        let n_dims = self.shape.len();
        // Pad out to a multiple of spacing + size
        let total_size = (self.shape.dims[self.shape.indexes[n_dims - 1]] + ((spacing + size) - 1))
            / (spacing + size)
            * (spacing + size);
        let padding = total_size - self.shape.dims[self.shape.indexes[n_dims - 1]];
        self.shape.padding[self.shape.indexes[n_dims - 1]].1 = padding;

        self = self.contiguous();
        // Expand a new dimension to do the slicing on
        let n_rows = total_size / (spacing + size);
        self.shape.expand_dim(n_dims, spacing + size);
        // self = self.contiguous();
        self.shape.dims[self.shape.indexes[n_dims - 1]] = n_rows;
        self.shape.fake[self.shape.indexes[n_dims]] = false;

        // Slice
        self.shape.mask[self.shape.indexes[n_dims]].1 = spacing.into();

        self = self.contiguous();

        self.shape.remove_dim(n_dims);
        self
    }

    /// Pool elements along the last dimension, pools are exposed as a new dimension
    pub fn pool_last_dim(
        mut self,
        kernel: impl Into<Expression>,
        stride: impl Into<Expression>,
        dilation: usize,
    ) -> GraphTensor {
        let (kernel, stride) = (kernel.into(), stride.into());
        let n_dims = self.shape.len();
        let full_kernel = kernel + (kernel - 1) * (dilation - 1);
        let dim_size = self.dims().pop().unwrap().simplify();
        let number_of_windows = (((dim_size - full_kernel) / stride) + 1).simplify();
        // Expand new dimension
        self.shape.expand_dim(n_dims - 1, number_of_windows);
        self = self.contiguous();
        if n_dims > 1 {
            // View as single dimension of matrix with wider width
            let mat_size = (dim_size + stride) * number_of_windows;
            let actual_size = (dim_size * number_of_windows).simplify();
            // Reshape into single dimension to pad
            self.shape.remove_dim(n_dims);
            self.shape.dims[self.shape.indexes[n_dims - 1]] = actual_size;
            self.shape.padding[self.shape.indexes[n_dims - 1]].1 =
                (mat_size - actual_size).simplify();
            self = self.contiguous();
            // Reshape back (mats should be full now)
            self.shape.add_dim(n_dims, dim_size + stride);
            self.shape.dims[self.shape.indexes[n_dims - 1]] = number_of_windows;
        } else {
            self.shape.dims[self.shape.indexes[n_dims]] = dim_size + stride;
        }
        // Slice down to kernel size
        self.shape.mask[self.shape.indexes[n_dims]].1 = full_kernel.simplify();
        self.shape.mask[self.shape.indexes[n_dims - 1]].1 = number_of_windows;
        self = self.contiguous();

        if dilation > 1 {
            // Remove dilations
            self.excise(1, dilation - 1)
        } else {
            self
        }
    }

    pub fn pad(mut self, padding: impl ToPad) -> GraphTensor {
        let padding = padding.to_pad_vec();
        // This exists because currently padding and slicing on the same dimension (even on opposite sides) is unsupported
        if padding.iter().zip(self.shape.indexes).any(|(range, ind)| {
            (range.0 != 0 || range.1 != 0)
                && (self.shape.mask[self.shape.indexes[ind]].0 != 0
                    || self.shape.mask[self.shape.indexes[ind]].1 != i32::MAX)
        }) {
            self = self.contiguous();
        }
        self.shape.pad(&padding);
        self
    }

    pub fn pad_along(
        self,
        left: impl Into<Expression>,
        right: impl Into<Expression>,
        axis: usize,
    ) -> GraphTensor {
        let mut p = vec![(Expression::from(0), Expression::from(0)); axis + 1];
        p[axis] = (left.into(), right.into());
        self.pad(p)
    }

    /// Concat along an existing dimension
    pub fn concat_along(self, rhs: GraphTensor, axis: usize) -> GraphTensor {
        // Pad and add
        self.pad_along(0, rhs.shape.dims()[axis], axis)
            + rhs.pad_along(self.shape.dims()[axis], 0, axis)
    }
}

#[cfg(test)]
mod tests {
    use dfdx::{
        shapes::Rank2,
        tensor::{Cpu, TensorFrom, TensorFromVec},
        tensor_ops::{RealizeTo, TryConcatAlong},
    };

    crate::test_imports!();

    #[test]
    fn test_unsqueeze_at_start() {
        let mut cx = Graph::new();

        let inp = cx.tensor((2, 2)).set(vec![1., 2., 3., 4.]);
        let out = inp.unsqueeze(0).retrieve();

        cx.execute();

        assert_eq!(out.dims(), &[1, 2, 2]);
        assert_exact(&out.data(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn test_unsqueeze_in_middle() {
        let mut cx = Graph::new();

        let inp = cx.tensor((2, 2)).set(vec![1., 2., 3., 4.]);
        let out = inp.unsqueeze(1).retrieve();

        cx.execute();

        assert_eq!(out.dims(), &[2, 1, 2]);
        assert_exact(&out.data(), &[1., 2., 3., 4.]);
    }

    #[test]
    fn test_unsqueeze_at_end() {
        let mut cx = Graph::new();

        let inp = cx.tensor((2, 2)).set(vec![1., 2., 3., 4.]);
        let out = inp.unsqueeze(2).retrieve();

        cx.execute();

        assert_eq!(out.dims(), &[2, 2, 1]);
        assert_exact(&out.data(), &[1., 2., 3., 4.]);
    }

    #[test]
    #[should_panic(expected = "Shape is maxed out at 6 dimensions")]
    fn test_unsqueeze_panics_when_shape_exceeds_max() {
        let mut cx = Graph::new();

        let inp = cx.tensor((2, 2, 2, 2, 2, 2)).set(vec![0.; 64]);
        let _out = inp.unsqueeze(6).retrieve();

        cx.execute();
    }

    #[test]
    fn test_transpose_simple_2d() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor((4, 4)).set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        // 3x3 kernel
        let out1 = inp1
            // Pool first dim first by moving it to end
            .transpose(1, 0)
            .retrieve();

        cx.execute();

        assert_exact(
            &out1.data(),
            &[
                1., 5., 9., 13., 2., 6., 10., 14., 3., 7., 11., 15., 4., 8., 12., 16.,
            ],
        );
    }

    #[test]
    #[should_panic(expected = "transpose dimensions")]
    fn test_transpose_out_of_bounds() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor((4, 4)).set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);

        // This should panic because dims 6 and 5 are out of bounds for a 2D tensor
        let _out1 = inp1.transpose(6, 5);
    }

    #[test]
    fn test_concat_1d() {
        let mut cx = Graph::new();
        let a = cx.tensor(4);
        a.set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = cx.tensor(3);
        b.set(vec![2.30434, 2.2343113, 1.4393]);
        let c = a.concat_along(b, 0);
        c.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b = d_dev.tensor([2.30434, 2.2343113, 1.4393]);
        let d_c = (d_a.realize::<(usize,)>(), d_b.realize::<(usize,)>()).concat_along(DAxis::<0>);

        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_concat_self() {
        let mut cx = Graph::new();
        let a = cx.tensor(4).set(vec![1.4325, 2.492428, 3.127365, 3.54865]);
        let b = a.concat_along(a, 0).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([1.4325, 2.492428, 3.127365, 3.54865]);
        let d_b =
            (d_a.clone().realize::<(usize,)>(), d_a.realize::<(usize,)>()).concat_along(DAxis::<0>);

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_concat_2d() {
        let mut cx = Graph::new();
        let a = cx.tensor((3, 2));
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = cx.tensor((3, 2));
        b.set(vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054]);
        let c = a.concat_along(b, 1);
        let d = a.concat_along(b, 0);
        c.retrieve();
        d.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![2.30434, 2.2343113, 1.4393, 482.4312, 8.1234, 54.2054],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_c = (
            d_a.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.clone().realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>);
        let d_d = (
            d_a.realize::<(usize, dfdx::shapes::Const<2>)>(),
            d_b.realize::<(usize, dfdx::shapes::Const<2>)>(),
        )
            .concat_along(dfdx::shapes::Axis::<0>);

        assert_close(&c.data(), &d_c.as_vec());
        assert_close(&d.data(), &d_d.as_vec());
    }

    #[test]
    fn test_pad_2d() {
        let mut cx = Graph::new();
        let a = cx
            .tensor((3, 2))
            .set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.pad(((0, 0), (0, 2))).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_dev.tensor_from_vec(
            vec![0., 0., 0., 0., 0., 0.],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = (
            d_a.realize::<(dfdx::shapes::Const<3>, usize)>(),
            d_b.realize::<(dfdx::shapes::Const<3>, usize)>(),
        )
            .concat_along(dfdx::shapes::Axis::<1>)
            .realize::<Rank2<3, 4>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_slice_2d() {
        let mut cx = Graph::new();
        let a = cx
            .tensor((3, 2))
            .set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let b = a.slice((.., ..1)).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_b = d_a.slice((.., ..1)).realize::<Rank2<3, 1>>();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_cumsum() {
        let mut cx = Graph::new();
        let a = cx.constant(1.).expand_dim(0, 3);
        let b = a.cumsum_last_dim().retrieve();
        let c = a
            .expand_dim(1, 3)
            .permute((1, 0))
            .cumsum_last_dim()
            .permute((1, 0))
            .retrieve();
        cx.execute();

        assert_exact(&b.data(), &[1., 2., 3.]);
        assert_exact(&c.data(), &[1., 1., 1., 2., 2., 2., 3., 3., 3.]);
    }

    #[test]
    fn test_pool_1d() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor(5).set([1., 2., 3., 4., 5.]);
        let inp2 = cx
            .tensor((2, 5))
            .set([[15., 14., 13., 12., 11.], [1., 2., 3., 4., 5.]]);
        // Stride 1
        let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();
        // Stride 2
        let out2 = inp1.pool_last_dim(3, 2, 1).retrieve();
        // Stride 3
        let out3 = inp1.pool_last_dim(3, 3, 1).retrieve();
        // Dilation 2
        let out4 = inp1.pool_last_dim(3, 1, 2).retrieve();
        // Dilation 2 Padding 1
        let out5 = inp1.pad(((1, 1),)).pool_last_dim(3, 1, 2).retrieve();
        // Stride 1 Batch 2
        let out6 = inp2.pool_last_dim(3, 1, 1).retrieve();
        // Stride 3
        let out7 = inp2.pool_last_dim(3, 3, 1).retrieve();
        // Dilation 2
        let out8 = inp2.pool_last_dim(3, 1, 2).retrieve();
        // Dilation 2 Padding 1
        let out9 = inp2.pad(((0, 0), (1, 1))).pool_last_dim(3, 1, 2).retrieve();

        cx.execute();

        assert_exact(&out1.data(), &[1., 2., 3., 2., 3., 4., 3., 4., 5.]);
        assert_exact(&out2.data(), &[1., 2., 3., 3., 4., 5.]);
        assert_exact(&out3.data(), &[1., 2., 3.]);
        assert_exact(&out4.data(), &[1., 3., 5.]);
        assert_exact(&out5.data(), &[0., 2., 4., 1., 3., 5., 2., 4., 0.]);
        assert_exact(
            &out6.data(),
            &[
                15., 14., 13., 14., 13., 12., 13., 12., 11., 1., 2., 3., 2., 3., 4., 3., 4., 5.,
            ],
        );
        assert_exact(&out7.data(), &[15., 14., 13., 1., 2., 3.]);
        assert_exact(&out8.data(), &[15., 13., 11., 1., 3., 5.]);
        assert_exact(
            &out9.data(),
            &[
                0., 14., 12., 15., 13., 11., 14., 12., 0., 0., 2., 4., 1., 3., 5., 2., 4., 0.,
            ],
        );
    }

    #[test]
    fn test_pool_1d_dims() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor((4, 4)).set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        // Stride 1
        let out1 = inp1.pool_last_dim(3, 1, 1).retrieve();

        cx.execute();

        assert_exact(
            &out1.data(),
            &[
                1., 2., 3., 2., 3., 4., 5., 6., 7., 6., 7., 8., 9., 10., 11., 10., 11., 12., 13.,
                14., 15., 14., 15., 16.,
            ],
        );
    }

    #[test]
    fn test_pool_2d() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor((4, 4)).set(vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]);
        // 3x3 kernel
        let out1 = inp1
            // Pool first dim first by moving it to end
            .permute((1, 0))
            .pool_last_dim(3, 1, 1)
            // Now move other dim to end
            .permute((1, 2, 0))
            .pool_last_dim(3, 1, 1)
            // Now swap middle two dims
            .permute((0, 2, 1, 3))
            // Now merge both pooled dimensions
            .reshape((4, 3, 3))
            .retrieve();

        cx.execute();

        assert_exact(
            &out1.data(),
            &[
                1.00, 2.00, 3.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 2.00, 3.00, 4.00, 6.00,
                7.00, 8.00, 10.00, 11.00, 12.00, 5.00, 6.00, 7.00, 9.00, 10.00, 11.00, 13.00,
                14.00, 15.00, 6.00, 7.00, 8.00, 10.00, 11.00, 12.00, 14.00, 15.00, 16.00,
            ],
        );
    }

    #[test]
    fn test_pool_1d_dilation() {
        let mut cx = Graph::new();

        let inp1 = cx.tensor(5).set(vec![1., 2., 3., 4., 5.]);
        // Stride 1
        let out1 = inp1.pool_last_dim(2, 1, 2).retrieve();
        // Stride 2
        let out2 = inp1.pool_last_dim(2, 2, 2).retrieve();
        // Stride 3
        let out3 = inp1.pool_last_dim(2, 3, 2).retrieve();

        cx.execute();

        assert_exact(&out1.data(), &[1., 3., 2., 4., 3., 5.]);
        assert_exact(&out2.data(), &[1., 3., 3., 5.]);
        assert_exact(&out3.data(), &[1., 3.]);
    }

    #[test]
    fn test_rotate_half() {
        let mut cx = Graph::new();
        let a = cx.tensor((3, 2));
        a.set(vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854]);
        let x1 = a.slice((.., ..1)).contiguous();
        let x2 = a.slice((.., 1..)).contiguous();
        let c = (-x2).concat_along(x1, 1);
        c.retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(
            vec![1.4325, 2.492428, 3.127365, 33.2834, 4.18734, 23.854],
            (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<2>),
        );
        let d_x1 = d_a.clone().slice((.., ..1));
        let d_x2 = d_a.slice((.., 1..));
        let d_c = (-d_x2, d_x1)
            .concat_along(dfdx::shapes::Axis::<1>)
            .realize::<Rank2<3, 2>>();

        assert_close(&c.data(), &d_c.as_vec());
    }
}
