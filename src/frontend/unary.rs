use itertools::Itertools;

use crate::prelude::*;
use std::ops::{Add, Mul, Neg};

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::Log2::default())
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::Exp2::default())
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Natural exp
    pub fn exp(self) -> GraphTensor {
        (self * (1.0 / f32::ln(2.))).exp2()
    }

    /// Natural log
    pub fn log(self) -> GraphTensor {
        self.log2() * f32::ln(2.)
    }

    /// Take the reciprocal of each element
    pub fn reciprocal(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::Recip::default())
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::Sin::default())
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// The cos(x) function
    pub fn cos(self) -> GraphTensor {
        ((std::f32::consts::PI / 2.) - self).sin()
    }

    /// Square every element in the tensor
    pub fn square(self) -> GraphTensor {
        self * self
    }

    /// The square root function
    pub fn sqrt(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::Sqrt::default())
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    pub fn graph_break(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(crate::hlir::GraphBreak)
            .input(self.id, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref, self.dtype)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes.to_axes())
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand_to_shape_on_axes(self.shape, axes)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a layer norm along an axis
    pub fn layer_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        self.mean_norm(axes.to_axes()).std_norm(axes, epsilon)
    }

    /// Normalize the tensor along `axes` using an Lp norm.
    pub fn normalize(self, p: f32, axes: impl ToAxes, epsilon: f32) -> GraphTensor {
        let norm = self.abs().pow(p).sum(axes.to_axes()).pow(1.0 / p);
        self / norm
            .maximum_f32(epsilon)
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.shape, axes.to_axes());
        let exp = m.exp();
        exp / exp
            .sum(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self
            - self
                .max(axes.to_axes())
                .expand_to_shape_on_axes(self.shape, axes.to_axes());
        m - m
            .exp()
            .sum(axes.to_axes())
            .log()
            .expand_to_shape_on_axes(m.shape, axes)
    }

    /// Get the indicies of the max elements along an axis
    pub fn argmax(self, axis: usize) -> GraphTensor {
        // Get one-hot along last dimension
        let x_equal = self
            .eq(self.max(axis).expand_dim(axis, self.dims()[axis]))
            .cast(DType::Int);
        // Create index arange for last dimension
        let r = self.graph().arange(self.dims()[axis]);
        let axes = (0..self.shape.len()).filter(|i| *i != axis).collect_vec();
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand_to_shape_on_axes(self.shape, axes)).max(axis)
    }

    /// Get the indices of the min elements along an axis
    pub fn argmin(self, axis: usize) -> GraphTensor {
        (-self).argmax(axis)
    }

    /// Compute the sample variance along axes
    pub fn var(self, axes: impl ToAxes) -> GraphTensor {
        self.var_options(axes, 1)
    }

    /// Compute the sample variance along an axes with options
    pub fn var_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        let axes = axes.to_axes();
        let n = axes
            .to_axes()
            .into_iter()
            .map(|i| self.dims()[i])
            .product::<Expression>();
        let mean = self
            .mean(axes.to_axes())
            .expand_to_shape_on_axes(self.shape, axes.to_axes());
        let centered = self - mean;
        (centered * centered).sum(axes) / (n - correction)
    }

    /// Compute the sample standard deviation along axes
    pub fn std(self, axes: impl ToAxes) -> GraphTensor {
        self.std_options(axes, 1)
    }

    /// Compute the standard deviation along axes with options
    pub fn std_options(self, axes: impl ToAxes, correction: usize) -> GraphTensor {
        self.var_options(axes, correction).sqrt()
    }

    /// Take the absolute value
    pub fn abs(self) -> GraphTensor {
        self.relu() + (-self).relu()
    }

    /// Get the sign of each element, '1' for positive and '-1' for negative
    pub fn sign(self) -> GraphTensor {
        self / (self.abs() + 1e-10)
    }

    /// The Rectified Linear Unit activation function
    pub fn relu(self) -> GraphTensor {
        self.maximum_f32(0.)
    }

    /// The sigmoid activation function
    pub fn sigmoid(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9d142430cbe61121c864c0015f1de83c94a7d2c0/tinygrad/mlops.py#L70
        (1. + (-self).exp()).reciprocal()
    }

    /// The swish (aka silu) activation function
    pub fn swish(self) -> GraphTensor {
        self * self.sigmoid()
    }

    /// The silu (aka swish) activation function
    pub fn silu(self) -> GraphTensor {
        self.swish()
    }

    /// The tanh activation function
    pub fn tanh(self) -> GraphTensor {
        (self * 2.0).sigmoid() * 2.0 - 1.0
    }

    /// The leaky relu activation function
    pub fn leaky_relu(self, neg_slope: f32) -> GraphTensor {
        self.relu() - (self * -neg_slope).relu()
    }

    /// The Gaussian Error Linear Unit activation function
    #[allow(clippy::excessive_precision)]
    pub fn gelu(self) -> GraphTensor {
        // Based on https://github.com/tinygrad/tinygrad/blob/9fc4465557831b614b56dd645eebc940ca0fa1bb/tinygrad/tensor.py#L1162C26-L1162C104
        0.5 * self * (1. + (0.7978845608 * self * (1. + 0.044715 * self * self)).tanh())
    }

    /// Compute the sorted indexes of this tensor along a certian axis
    pub fn argsort(self, axis: usize, descending: bool) -> GraphTensor {
        // Compare all elements with all other elements by making an axis
        let ax_size = self.dims()[axis];
        // Materialize expanded tensors to avoid stride issues with sum reduction
        let a = self.expand_dim(axis + 1, ax_size) + 0.0;
        let b = self.expand_dim(axis, ax_size) + 1e-9; // eps for stable sort
        // lt/gt return Bool, cast to F32 for sum
        let cmp = if descending { a.gt(b) } else { a.lt(b) };
        // Adding 0 forces materialization before sum, avoiding stride issues
        let ind = (cmp.cast(DType::F32) + 0.0).sum(axis).cast(DType::Int);
        ind.inverse_permutation(axis)
    }

    /// Sort the tensor along a certian axis
    pub fn sort(self, axis: usize, descending: bool) -> GraphTensor {
        self.gather(self.argsort(axis, descending))
    }

    /// Sort and retrieve top-k **indexes**
    pub fn topk_indexes(self, k: usize, axis: usize) -> GraphTensor {
        self.argsort(axis, true).slice_along(..k, axis)
    }

    /// Apply a cumulative reduction operation along dimensions
    ///
    /// See `cumsum` or `cummax` for usage examples.
    pub fn cumop(
        mut self,
        axes: impl ToAxes,
        op: impl Fn(GraphTensor, usize) -> GraphTensor,
        pad_elem: f32,
    ) -> Self {
        let n_dims = self.shape.len();
        for axis in axes.to_axes() {
            // Pad out length
            let mut kernel = vec![1.into(); n_dims];
            let mut padding = vec![(Expression::from(0), Expression::from(0)); n_dims];
            let orig_length = self.dims()[axis];
            padding[axis] = (orig_length - 1, 0.into());
            kernel[axis] = orig_length;
            self = self.pad(padding, pad_elem);
            // Unfold
            self = self.unfold(kernel, vec![1; n_dims], vec![1; n_dims]);
            // Remove non-cumulative dimensions
            for i in (0..n_dims).rev() {
                if i != axis {
                    self = self.squeeze(n_dims + i);
                }
            }
            // apply operation along cumulative dimensions
            self = op(self, n_dims);
        }
        self
    }

    /// Apply a cumulative sum along dimensions
    pub fn cumsum(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.sum(axes), 0.)
    }

    /// Apply a cumulative max along dimensions
    pub fn cummax(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.max(axes), f32::MIN)
    }

    /// Apply a cumulative product along dimensions
    pub fn cumprod(self, axes: impl ToAxes) -> Self {
        self.cumop(axes, |t, axes| t.prod(axes), 1.)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::BinaryHeap;

    use crate::{
        prelude::*,
        tests::{assert_close, random_vec},
    };
    use candle_core::{Device, Tensor};
    use candle_nn::ops::softmax;
    use itertools::Itertools;
    use ordered_float::NotNan;
    use proptest::prelude::*;

    fn cummax_ref_2d(a: Tensor) -> Tensor {
        let v = a.to_vec2::<f32>().unwrap();
        let mut out = vec![vec![0.0; v[0].len()]; v.len()];
        for (i, row) in v.iter().enumerate() {
            let mut acc = f32::NEG_INFINITY;
            for (j, val) in row.iter().enumerate() {
                acc = acc.max(*val);
                out[i][j] = acc;
            }
        }
        Tensor::new(out, a.device()).unwrap()
    }

    fn cumprod_ref_2d(a: Tensor) -> Tensor {
        let v = a.to_vec2::<f32>().unwrap();
        let mut out = vec![vec![0.0; v[0].len()]; v.len()];
        for (i, row) in v.iter().enumerate() {
            let mut acc = 1.0;
            for (j, val) in row.iter().enumerate() {
                acc *= val;
                out[i][j] = acc;
            }
        }
        Tensor::new(out, a.device()).unwrap()
    }

    pub fn test_unary(
        shape: impl ToShape,
        func: impl Fn(GraphTensor) -> GraphTensor,
        ref_func: impl Fn(Tensor) -> Tensor,
    ) {
        let shape = shape
            .to_shape()
            .into_iter()
            .map(|e| e.to_usize().unwrap())
            .collect_vec();
        let mut cx = Graph::new();
        let a = cx.tensor(shape.clone());
        let b = func(a).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        let v = random_vec(shape.iter().copied().product());
        rt.set_data(a.id, v.clone());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_a = Tensor::new(v, &device).unwrap().reshape(shape).unwrap();
        let ref_b = ref_func(ref_a).flatten_all().unwrap();

        // need to assert close because some unaries (exp and log) are (good) approximations
        assert_close(rt.get_f32(b.id), &ref_b.to_vec1::<f32>().unwrap())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]

        #[test]
        fn test_exp(size in 1usize..128) {
            test_unary(size, |a| a.exp(), |a| a.exp().unwrap());
        }

        #[test]
        fn test_log(size in 1usize..128) {
            test_unary(size, |a| a.log(), |a| a.log().unwrap());
        }

        #[test]
        fn test_sin(size in 1usize..128) {
            test_unary(size, |a| a.sin(), |a| a.sin().unwrap());
        }

        #[test]
        fn test_cos(size in 1usize..128) {
            test_unary(size, |a| a.cos(), |a| a.cos().unwrap());
        }

        #[test]
        fn test_activations(size in 1usize..128) {
            test_unary(size, |a| a.relu(), |a| a.relu().unwrap());
            test_unary(size, |a| a.gelu(), |a| a.gelu().unwrap());
            test_unary(size, |a| a.swish(), |a| a.silu().unwrap());
            test_unary(size, |a| a.tanh(), |a| a.tanh().unwrap());
        }

        #[test]
        fn test_recip(size in 1usize..128) {
            test_unary(size, |a| a.reciprocal(), |a| a.recip().unwrap());
        }

        #[test]
        fn test_sqrt(size in 1usize..128) {
            test_unary(size, |a| a.sqrt(), |a| a.sqrt().unwrap());
        }

        #[test]
        fn test_square(size in 1usize..128) {
            test_unary(size, |a| a.square(), |a| a.powf(2.0).unwrap());
        }

        #[test]
        fn test_softmax(size in 1usize..128, rows in 1usize..16, cols in 1usize..16) {
            test_unary(size, |a| a.softmax(0), |a| softmax(&a, 0).unwrap());
            test_unary((rows, cols), |a| a.softmax(1), |a| softmax(&a, 1).unwrap());
        }

        #[test]
        fn test_layer_norm(size in 2usize..128) {
            test_unary(
                size,
                |a| a.layer_norm(0, 1e-5),
                |a| {
                    let meaned = (a.clone() - a.mean(0).unwrap().broadcast_as(size)).unwrap();
                    meaned
                        .powf(2.0)
                        .unwrap()
                        .mean(0)
                        .unwrap()
                        .add(&Tensor::new(1e-5_f32, a.device()).unwrap())
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .recip()
                        .unwrap()
                        .broadcast_as(size)
                        .unwrap()
                        .mul(&meaned)
                        .unwrap()
                },
            );
        }

        #[test]
        fn test_cumulative(rows in 1usize..16, cols in 1usize..16) {
            test_unary(rows, |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
            test_unary((rows, cols), |a| a.cumsum(1), |a| a.cumsum(1).unwrap());
            test_unary((rows, cols), |a| a.cumsum(0), |a| a.cumsum(0).unwrap());
            test_unary(
                (rows, cols),
                |a| a.cumsum((0, 1)),
                |a| a.cumsum(0).unwrap().cumsum(1).unwrap(),
            );
            test_unary(
                (rows, cols),
                |a| a.cumsum((1, 0)),
                |a| a.cumsum(1).unwrap().cumsum(0).unwrap(),
            );
            test_unary((rows, cols), |a| a.cummax(1), cummax_ref_2d);
            test_unary((rows, cols), |a| a.cumprod(1), cumprod_ref_2d);
        }

        #[test]
        fn test_argmax(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.argmax(0).cast(DType::F32), |a| a.argmax(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary((rows, cols), |a| a.argmax(1).cast(DType::F32), |a| a.argmax(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_argmin(rows in 1usize..16, cols in 1usize..16) {
            test_unary((rows, cols), |a| a.argmin(0).cast(DType::F32), |a| a.argmin(0).unwrap().to_dtype(candle_core::DType::F32).unwrap());
            test_unary((rows, cols), |a| a.argmin(1).cast(DType::F32), |a| a.argmin(1).unwrap().to_dtype(candle_core::DType::F32).unwrap());
        }

        #[test]
        fn test_var(rows in 2usize..16, cols in 2usize..16) {
            test_unary((rows, cols), |a| a.var(1), |a| a.var(1).unwrap());
            test_unary((rows, cols), |a| a.var(0), |a| a.var(0).unwrap());
        }

        #[test]
        fn test_std(rows in 2usize..16, cols in 2usize..16) {
            test_unary((rows, cols), |a| a.std(1), |a| a.var(1).unwrap().sqrt().unwrap());
        }

        #[test]
        fn test_topk(rows in 1usize..12, cols in 1usize..12, k in 1usize..12) {
            prop_assume!(k <= cols);
            pub fn topk_sorted_indices(x: &[f32], k: usize) -> Vec<usize> {
                if k == 0 {
                    return Vec::new();
                }

                let mut heap: BinaryHeap<std::cmp::Reverse<(NotNan<f32>, usize)>> =
                    BinaryHeap::with_capacity(k);

                for (i, &v) in x.iter().enumerate() {
                    let v = NotNan::new(v).expect("NaN encountered in topk");
                    if heap.len() < k {
                        heap.push(std::cmp::Reverse((v, i)));
                    } else if let Some(&std::cmp::Reverse((min_v, _))) = heap.peek() {
                        if v > min_v {
                            heap.pop();
                            heap.push(std::cmp::Reverse((v, i)));
                        }
                    }
                }

                let mut out: Vec<(NotNan<f32>, usize)> =
                    heap.into_iter().map(|std::cmp::Reverse(t)| t).collect();

                out.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                out.into_iter().map(|(_, i)| i).collect()
            }
            test_unary(
                (rows, cols),
                |a| a.topk_indexes(k, 1).cast(DType::F32) * 1.0,
                |a| {
                    let data = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    let topk = data
                        .chunks_exact(cols)
                        .flat_map(|c| topk_sorted_indices(c, k))
                        .map(|i| i as f32)
                        .collect_vec();
                    Tensor::new(topk, a.device()).unwrap()
                },
            );
        }
    }
}
