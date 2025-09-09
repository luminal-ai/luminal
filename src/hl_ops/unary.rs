use crate::{op, prelude::*};
use std::ops::{Add, Mul, Neg};

impl Neg for GraphTensor {
    type Output = GraphTensor;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl GraphTensor {
    /// Base 2 log
    pub fn log2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Log2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Base 2 exp
    pub fn exp2(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Exp2)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
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
            .add_op(op::Recip)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// The sin(x) function
    pub fn sin(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::Sin)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
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
            .add_op(op::Sqrt)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    pub fn graph_break(self) -> GraphTensor {
        let new_id = self
            .graph()
            .add_op(op::GraphBreak)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(new_id, self.shape.contiguous(), self.graph_ref)
    }

    /// Scale so std is 1.0
    pub fn std_norm<T>(self, axes: impl ToAxes, epsilon: T) -> GraphTensor
    where
        GraphTensor: Add<T, Output = GraphTensor>,
    {
        (self * self)
            .mean(axes)
            .add(epsilon)
            .sqrt()
            .reciprocal()
            .expand(self.shape)
            .mul(self)
    }

    /// Center so mean is 0.0
    pub fn mean_norm(self, axes: impl ToAxes) -> GraphTensor {
        self - self.mean(axes).expand(self.shape)
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
        let norm = self.abs().pow(p).sum(axes).pow(1.0 / p);
        self / norm.maximum_f32(epsilon).expand(self.shape)
    }

    /// Applies a softmax function along an axis
    pub fn softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand(self.shape);
        let exp = m.exp();
        exp / exp.sum(axes).expand(self.shape)
    }

    /// Applies a log softmax function along an axis
    pub fn log_softmax(self, axes: impl ToAxes) -> GraphTensor {
        let m = self - self.max(axes.to_axes()).expand(self.shape);
        m - m.exp().sum(axes.to_axes()).log().expand(m.shape)
    }

    /// Get the indicies of the max elements along the last axis
    pub fn argmax(self) -> GraphTensor {
        // Get one-hot along last dimension
        let x_equal = self.eq(self.max(self.shape.len() - 1).expand(self.shape));
        // Create index arange for last dimension
        let r = self
            .graph()
            .constant(1.)
            .expand(self.shape.dims().last().unwrap())
            .cumsum_last_dim()
            - 1.;
        // Multiply one-hot by expanded index arange
        (x_equal * r.expand(self.shape)).max(self.shape.len() - 1)
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
        1. / (1. + (-self).exp())
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
}

#[cfg(test)]
mod tests {
    crate::test_imports!();

    #[test]
    fn test_exp() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.exp().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.exp();

        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_layer_norm() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.layer_norm(0, 1e-5).retrieve();
        let c = a.layer_norm(1, 1e-5).retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.clone().normalize::<DAxis<0>>(1e-5);
        let d_c = d_a.normalize::<DAxis<1>>(1e-5);

        assert_close(&b.data(), &d_b.as_vec());
        assert_close(&c.data(), &d_c.as_vec());
    }

    #[test]
    fn test_normalize_lp() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.normalize(3.0, 1, 1e-5).retrieve();

        cx.execute();

        let mut expected = vec![0.0; 6];
        for i in 0..2 {
            let row = &a_data[(i * 3)..((i + 1) * 3)];
            let mut norm = row.iter().map(|v| v.abs().powf(3.0)).sum::<f32>();
            norm = norm.powf(1.0 / 3.0).max(1e-5);
            for j in 0..3 {
                expected[i * 3 + j] = row[j] / norm;
            }
        }

        assert_close(&b.data(), &expected);
    }

    #[test]
    fn test_softmax() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.softmax(1).retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.softmax::<DAxis<1>>();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_sin() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.sin().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.sin();

        let r = b.data();
        assert_close(&r, &d_b.as_vec());
    }

    #[test]
    fn test_cos() {
        let mut cx = Graph::new();
        let a_data = random_vec(6);
        let a = cx.tensor((2, 3)).set(a_data.clone());
        let b = a.cos().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<3>));
        let d_b = d_a.cos();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_relu() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.relu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.relu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_gelu() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.gelu().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.fast_gelu();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_sigmoid() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.sigmoid().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_swish() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.swish().retrieve();
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.clone() * d_a.sigmoid();
        assert_close(&b.data(), &d_b.as_vec());
    }

    #[test]
    fn test_tanh() {
        let mut cx = Graph::new();
        let a_data = random_vec(4);
        let a = cx.tensor(('a', 'b')).set_dyn(a_data.clone(), (2, 2));
        let b = a.tanh().retrieve();

        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor_from_vec(a_data, (DConst::<2>, DConst::<2>));
        let d_b = d_a.tanh();
        assert_close(&b.data(), &d_b.as_vec());
    }
}
