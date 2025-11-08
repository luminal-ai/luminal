use luminal::prelude::*;

pub struct Conv1D {
    pub weight: GraphTensor, // ch_out, ch_in * kernel
    pub bias: Option<GraphTensor>,
    padding: usize,
    dilation: usize,
    stride: usize,
    kernel: usize,
    ch_in: usize,
}

impl Conv1D {
    /// Create a new 1D convolution layer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (ch_out, ch_in * kernel)),
            bias: if bias {
                Some(cx.named_tensor("Bias", ch_out))
            } else {
                None
            },
            padding,
            dilation,
            stride,
            kernel,
            ch_in,
        }
    }
}

impl SerializeModule for Conv1D {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
        if let Some(bias) = self.bias {
            s.tensor("bias", bias);
        }
    }
}

impl Module<GraphTensor> for Conv1D {
    type Output = GraphTensor;
    fn forward(&self, input: GraphTensor) -> Self::Output {
        assert_eq!(input.dims()[input.shape.len() - 2], self.ch_in);
        // Input: batch_dims, ch_in, dim_in
        // Reshape to 2 batch dims
        let n_expands = 4 - input.shape.len();
        let mut inp = input;
        for _ in 0..n_expands {
            inp = inp.expand_dim(0, 1);
        }

        let batch1 = inp.dims()[0];
        let batch2 = inp.dims()[1];
        let dim_in = *input.dims().last().unwrap();
        let dim_out = (((dim_in + 2 * self.padding - self.dilation * (self.kernel - 1) - 1)
            / self.stride)
            + 1)
        .simplify();
        let w = self.weight.permute((1, 0));
        let mut out = inp
            // Add padding
            .pad(((0, 0), (0, 0), (0, 0), (self.padding, 0)))
            .contiguous()
            .pad(((0, 0), (0, 0), (0, 0), (0, self.padding)))
            // Pool
            .pool_last_dim(self.kernel, self.stride, self.dilation)
            // Combine channel_in and kernel
            .permute((0, 1, 3, 2, 4))
            .reshape((batch1, batch2, dim_out, self.ch_in * self.kernel))
            .matmul(w)
            .permute((0, 1, 3, 2));
        if let Some(b) = self.bias {
            out += b.expand(out.shape);
        }

        // Reshape back to original shape
        let mut final_shape = out.dims();
        for _ in 0..n_expands {
            final_shape.remove(0);
        }
        out.reshape(final_shape) // Output: batch_dims, ch_out, dim_out
    }
}

pub struct Conv2D {
    pub weight: GraphTensor,       // ch_out, ch_in * kernel_x * kernel_y
    pub bias: Option<GraphTensor>, // ch_out
    kernel: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
    ch_out: usize,
    ch_in: usize,
}

impl Conv2D {
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor("CWeight", (ch_out, ch_in * kernel.0 * kernel.1)),
            bias: if bias {
                Some(cx.named_tensor("CBias", ch_out))
            } else {
                None
            },
            kernel,
            stride,
            dilation,
            ch_out,
            ch_in,
        }
    }
}

impl SerializeModule for Conv2D {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
        if let Some(bias) = self.bias {
            s.tensor("bias", bias);
        }
    }
}

impl Conv2D {
    pub fn forward(&self, mut input: GraphTensor) -> GraphTensor {
        // Input: (batch (optional), ch_in, dimx_in, dimy_in)
        let mut expanded = false;
        if input.shape.len() == 3 {
            // Expand batch
            input = input.expand_dim(0, 1);
            expanded = true;
        }
        let (batch, _, dimx_in, dimy_in) = input.dims4();
        let dimx_out = (((dimx_in - self.dilation.0 * (self.kernel.0 - 1) - 1) / self.stride.0)
            + 1)
        .simplify();
        let dimy_out = (((dimy_in - self.dilation.1 * (self.kernel.1 - 1) - 1) / self.stride.1)
            + 1)
        .simplify();
        let input_pooled = input
            .pool_last_dim(self.kernel.1, self.stride.1, self.dilation.1)
            .permute((0, 1, 3, 4, 2))
            .pool_last_dim(self.kernel.0, self.stride.0, self.dilation.0)
            .permute((0, 1, 5, 3, 4, 2))
            .reshape((
                batch,
                self.ch_in * self.kernel.0 * self.kernel.1,
                dimx_out * dimy_out,
            ));

        let mut o = self
            .weight
            .expand_dim(0, batch)
            .matmul(input_pooled)
            .reshape((
                batch,
                self.ch_out,
                dimx_out.to_usize().unwrap(),
                dimy_out.to_usize().unwrap(),
            ));
        if let Some(b) = self.bias {
            o += b.expand(o.shape);
        }
        if expanded {
            o.reshape((
                self.ch_out,
                dimx_out.to_usize().unwrap(),
                dimy_out.to_usize().unwrap(),
            ))
        } else {
            o
        }
    }
}
pub struct Conv3D {
    pub weight: GraphTensor, // ch_out, ch_in * kernel_x * kernel_y * kernel_z
    pub bias: Option<GraphTensor>, // ch_out
    kernel: (usize, usize, usize),
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    ch_in: usize,
    ch_out: usize,
}

impl Conv3D {
    pub fn new(
        ch_in: usize,
        ch_out: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
        cx: &mut Graph,
    ) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (ch_out, ch_in * kernel.0 * kernel.1 * kernel.2)),
            bias: if bias {
                Some(cx.named_tensor("Bias", ch_out))
            } else {
                None
            },
            kernel,
            stride,
            dilation,
            ch_in,
            ch_out,
        }
    }
}

impl SerializeModule for Conv3D {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
        if let Some(bias) = self.bias {
            s.tensor("bias", bias);
        }
    }
}

impl Conv3D {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        // Input: ch_in, dimx_in, dimy_in, dimz_in
        let dimx_in = input.dims()[1];
        let dimy_in = input.dims()[2];
        let dimz_in = input.dims()[3];
        let dimx_out = (((dimx_in - self.dilation.0 * (self.kernel.0 - 1) - 1) / self.stride.0)
            + 1)
        .simplify();
        let dimy_out = (((dimy_in - self.dilation.1 * (self.kernel.1 - 1) - 1) / self.stride.1)
            + 1)
        .simplify();
        let dimz_out = (((dimz_in - self.dilation.2 * (self.kernel.2 - 1) - 1) / self.stride.2)
            + 1)
        .simplify();

        let input_pooled = input
            .pool_last_dim(self.kernel.1, self.stride.1, self.dilation.1)
            .permute((0, 2, 3, 4, 1))
            .pool_last_dim(self.kernel.0, self.stride.0, self.dilation.0)
            .reshape((
                self.ch_in,
                dimz_out,
                self.kernel.1,
                dimx_out * self.kernel.0,
                dimy_in,
            ));

        let last_pool = input_pooled
            .pool_last_dim(self.kernel.2, self.stride.2, self.dilation.2)
            .permute((0, 2, 5, 3, 1, 4));

        let reshaped = last_pool.reshape((
            self.ch_in * self.kernel.0 * self.kernel.1 * self.kernel.2,
            dimx_out * dimy_out * dimz_out,
        ));

        self.weight
            .matmul(reshaped)
            .reshape((self.ch_out, dimx_out, dimy_out, dimz_out))
    }
}

#[cfg(test)]
mod tests {
    use super::{Conv1D, Conv2D, Conv3D};
    use candle_core::{Device, Tensor};
    use luminal::{
        prelude::*,
        tests::{assert_close, random_vec_rng},
    };
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn test_conv1d_simple() {
        let mut cx = Graph::new();

        const CH_IN: usize = 1;
        const CH_OUT: usize = 1;
        const KERNEL: usize = 2;
        const STRIDE: usize = KERNEL;
        const DIM_IN: usize = 6;
        const DIM_OUT: usize = ((DIM_IN - (KERNEL - 1) - 1) / STRIDE) + 1;

        let model = Conv1D::new(CH_IN, CH_OUT, KERNEL, KERNEL, 1, 0, false, &mut cx);
        model.weight.set([[[0.0316, -0.2057]]]);

        let inp1 = cx.tensor((CH_IN, DIM_IN)).set([[3., 0., 9., 6., 0., 6.]]);

        let out1 = model.forward(inp1).retrieve();
        cx.execute();

        assert_eq!(
            out1.dims(),
            vec![Expression::from(CH_OUT), Expression::from(DIM_OUT)]
        );
        assert_close(&out1.data(), &[0.0948, -0.9498, -1.2342]);
    }

    #[test]
    fn test_conv1d_pad_stride() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);

        const CH_IN: usize = 80;
        const CH_OUT: usize = 384;
        const KERNEL: usize = 3;
        const STRIDE: usize = 1;
        const PADDING: usize = 1;
        const DIM_IN: usize = 10;
        let kernel_data = random_vec_rng(KERNEL * CH_IN * CH_OUT, &mut rng);
        let input_data = random_vec_rng(CH_IN * DIM_IN, &mut rng);

        let model = Conv1D::new(CH_IN, CH_OUT, KERNEL, STRIDE, 1, PADDING, false, &mut cx);
        model.weight.set(kernel_data.clone());

        let inp1 = cx
            .tensor((1, CH_IN, 's'))
            .set_dyn(input_data.clone(), (1, CH_IN, DIM_IN));

        let out1 = model.forward(inp1).retrieve();
        cx.execute();

        let input = Tensor::from_vec(input_data, (1, CH_IN, DIM_IN), &Device::Cpu).unwrap();
        let kernel = Tensor::from_vec(kernel_data, (CH_OUT, CH_IN, KERNEL), &Device::Cpu).unwrap();
        let output = input.conv1d(&kernel, PADDING, STRIDE, 1, 1).unwrap();

        assert_close(
            &out1.data(),
            &output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    #[test]
    fn test_conv1d() {
        let mut cx = Graph::new();

        const CH_IN: usize = 8;
        const CH_OUT: usize = 4;
        const KERNEL: usize = 2;
        const STRIDE: usize = 2;
        const DIM_IN: usize = 12;

        let model = Conv1D::new(CH_IN, CH_OUT, KERNEL, STRIDE, 1, 0, false, &mut cx);
        model.weight.set(vec![
            -0.1700, -0.2000, 0.1000, -0.0200, 0.1000, 0.0200, -0.2100, -0.2300, -0.0600, 0.1500,
            0.1200, 0.1000, 0.1800, 0.0600, -0.1700, -0.0400, 0.1000, -0.0200, -0.1700, 0.1000,
            0.1100, 0.1600, 0.2000, 0.0100, -0.0500, 0.2100, -0.0200, 0.0300, -0.0900, -0.0500,
            0.1600, 0.0400, 0.0400, -0.1700, 0.1100, 0.0600, -0.1200, -0.2300, 0.2300, -0.2100,
            -0.2200, 0.1100, -0.0100, -0.1400, 0.1700, 0.0300, 0.1000, -0.1400, -0.2100, -0.1800,
            0.2000, -0.2300, -0.1600, 0.2200, 0.0900, 0.0700, -0.1000, -0.0400, -0.0500, 0.1400,
            0.0700, -0.1200, 0.1400, 0.2200,
        ]);

        let inp1 = cx.tensor((CH_IN, DIM_IN));
        inp1.set(vec![
            1., 2., 6., 4., 8., 1., 6., 0., 1., 0., 6., 4., 3., 4., 9., 3., 8., 8., 5., 5., 0., 4.,
            2., 7., 6., 4., 2., 2., 8., 0., 7., 3., 0., 0., 7., 2., 3., 3., 1., 9., 5., 4., 5., 5.,
            8., 0., 0., 1., 2., 1., 8., 9., 4., 7., 7., 6., 8., 5., 0., 9., 1., 6., 0., 1., 4., 3.,
            3., 5., 8., 7., 9., 5., 6., 5., 6., 9., 7., 0., 9., 5., 6., 0., 6., 1., 2., 1., 0., 1.,
            3., 6., 8., 0., 6., 6., 3., 2.,
        ]);
        inp1.retrieve();

        let out1 = model.forward(inp1).retrieve();
        cx.execute();

        assert_close(
            &out1.data(),
            &[
                0.7600, -0.4700, 0.0100, -0.1600, -0.1800, 2.2300, 1.7200, 0.6900, 3.5100, 3.7700,
                3.4600, 3.8100, -1.2600, -1.3900, 0.9400, 0.5300, 0.6300, -0.0400, 0.3800, -1.4900,
                -0.8800, -0.3100, 1.7500, -2.7500,
            ],
        );
    }

    #[test]
    fn test_conv2d() {
        let mut cx = Graph::new();

        const CH_IN: usize = 5;
        const CH_OUT: usize = 2;
        const KERNELX: usize = 2;
        const KERNELY: usize = 2;
        const STRIDEX: usize = KERNELX;
        const STRIDEY: usize = KERNELY;
        const DILATIONX: usize = 1;
        const DILATIONY: usize = 1;
        const DIMX_IN: usize = 16;
        const DIMX_OUT: usize = ((DIMX_IN - DILATIONX * (KERNELX - 1) - 1) / STRIDEX) + 1;
        const DIMY_IN: usize = 9;
        const DIMY_OUT: usize = ((DIMY_IN - DILATIONY * (KERNELY - 1) - 1) / STRIDEY) + 1;

        let inp1 = cx.tensor((CH_IN, DIMX_IN, DIMY_IN));
        inp1.set(vec![
            8., 8., 5., 7., 0., 6., 5., 3., 0., 7., 0., 6., 6., 7., 7., 5., 0., 6., 9., 4., 0., 8.,
            8., 5., 7., 6., 2., 8., 9., 5., 0., 3., 1., 1., 8., 4., 1., 1., 5., 6., 9., 3., 2., 9.,
            4., 7., 1., 0., 7., 7., 4., 9., 5., 0., 4., 7., 4., 7., 8., 8., 4., 8., 4., 7., 9., 3.,
            7., 9., 5., 8., 5., 9., 0., 9., 5., 6., 8., 9., 5., 4., 1., 9., 7., 2., 2., 7., 9., 3.,
            1., 2., 8., 4., 0., 8., 0., 5., 6., 7., 7., 4., 3., 4., 6., 8., 3., 7., 8., 8., 7., 1.,
            5., 1., 8., 0., 1., 1., 7., 3., 2., 1., 0., 4., 5., 4., 3., 2., 5., 4., 2., 4., 1., 9.,
            4., 1., 9., 7., 7., 1., 2., 6., 3., 4., 1., 1., 6., 6., 8., 2., 7., 7., 9., 0., 9., 0.,
            1., 4., 2., 4., 9., 6., 8., 6., 1., 6., 3., 8., 3., 4., 5., 0., 2., 1., 8., 2., 2., 8.,
            7., 0., 7., 7., 3., 4., 5., 0., 7., 2., 1., 1., 4., 2., 9., 9., 6., 1., 5., 4., 6., 9.,
            5., 4., 1., 9., 1., 5., 5., 5., 8., 8., 0., 1., 3., 0., 8., 8., 5., 1., 6., 1., 5., 6.,
            4., 4., 4., 0., 1., 1., 5., 1., 7., 2., 3., 5., 5., 4., 9., 1., 3., 7., 6., 7., 1., 5.,
            3., 8., 6., 6., 6., 7., 3., 2., 2., 8., 1., 3., 0., 2., 7., 6., 5., 7., 5., 7., 8., 1.,
            2., 2., 5., 0., 2., 9., 1., 5., 3., 8., 7., 9., 7., 2., 8., 8., 8., 6., 3., 2., 7., 7.,
            0., 3., 7., 8., 3., 7., 2., 3., 2., 7., 5., 5., 6., 0., 9., 0., 9., 9., 1., 8., 7., 9.,
            6., 8., 7., 5., 4., 9., 5., 6., 3., 2., 8., 3., 0., 6., 3., 8., 3., 1., 8., 7., 2., 0.,
            7., 7., 7., 7., 8., 0., 4., 9., 8., 2., 0., 4., 4., 3., 5., 5., 3., 0., 3., 6., 3., 1.,
            2., 9., 9., 6., 8., 1., 2., 6., 8., 6., 0., 0., 2., 8., 8., 5., 0., 5., 9., 0., 8., 1.,
            1., 3., 5., 9., 3., 5., 8., 6., 3., 2., 9., 4., 8., 3., 9., 5., 2., 9., 0., 1., 6., 8.,
            0., 3., 0., 1., 2., 1., 0., 1., 4., 1., 1., 0., 6., 9., 2., 7., 2., 6., 0., 4., 8., 2.,
            6., 7., 2., 2., 7., 4., 5., 8., 1., 4., 7., 5., 9., 7., 2., 5., 9., 1., 6., 1., 7., 9.,
            5., 6., 9., 3., 5., 1., 6., 1., 3., 3., 9., 3., 9., 0., 1., 8., 1., 9., 8., 5., 3., 4.,
            4., 1., 5., 5., 4., 4., 5., 8., 7., 1., 1., 7., 3., 9., 0., 1., 3., 4., 8., 4., 0., 5.,
            6., 2., 0., 7., 8., 2., 6., 2., 9., 6., 2., 0., 3., 7., 5., 7., 1., 8., 5., 5., 9., 1.,
            0., 3., 5., 7., 5., 3., 2., 8., 6., 3., 0., 5., 8., 5., 7., 8., 8., 2., 9., 0., 1., 8.,
            6., 0., 3., 2., 5., 2., 9., 8., 9., 6., 2., 0., 3., 2., 5., 9., 1., 3., 6., 5., 2., 8.,
            2., 2., 1., 8., 6., 4., 1., 6., 0., 7., 3., 0., 9., 6., 5., 5., 5., 2., 4., 2., 8., 3.,
            0., 6., 3., 8., 8., 4., 9., 4., 7., 0., 3., 5., 1., 4., 6., 0., 0., 5., 9., 7., 8., 6.,
            7., 0., 6., 7., 0., 5., 8., 8., 6., 4., 6., 0., 2., 3., 2., 8., 7., 5., 9., 6., 6., 2.,
            0., 4., 4., 4., 4., 2., 7., 5., 3., 2., 6., 3., 7., 0., 7., 2., 5., 1., 4., 4., 5., 1.,
            6., 7., 5., 7., 0., 7., 8., 4., 7., 3., 9., 1., 7., 5., 6., 1., 0., 2., 0., 0., 5., 5.,
            8., 8., 7., 3., 7., 2., 9., 3., 8., 4., 5., 3., 8., 5., 2., 0., 2., 0., 5., 9., 0., 3.,
            8., 0., 4., 1., 8., 4., 8., 9., 1., 1., 4., 5., 0., 2., 0., 9., 4., 2., 3., 9., 0., 7.,
            3., 1., 5., 9., 1., 6., 5., 4., 2., 1., 2., 1., 1., 4., 7., 2.,
        ]);

        let exp_out1 = cx.tensor((CH_OUT, DIMX_OUT, DIMY_OUT));
        exp_out1.set(vec![
            3.9600, -0.3300, -1.7800, 4.0400, 1.5300, 0.2900, 2.8700, 3.0000, 0.9600, -1.8700,
            4.5900, 3.9700, 1.2800, 1.1800, 3.7800, 2.8500, 0.5500, 0.5600, 3.9800, 1.3200,
            -0.7100, -0.6500, 4.3900, 0.4000, 1.0300, 0.9800, 3.1200, 2.7400, 2.5100, 0.1200,
            1.8500, 2.0000, -0.7900, 1.0700, -0.3900, -0.8100, -2.5100, -2.9700, 0.2100, 1.8400,
            -0.7700, -0.3900, 1.2200, 0.1900, 4.1700, -4.3600, -1.8600, 0.4800, -2.4400, 2.6300,
            1.5000, -1.9700, 1.2800, -2.8200, -2.3200, 0.2200, -0.3800, 2.1800, -0.8200, -1.5700,
            1.2000, -3.4200, -1.6700, 0.9000,
        ]);

        exp_out1.retrieve();

        let model = Conv2D::new(
            CH_IN,
            CH_OUT,
            (KERNELX, KERNELY),
            (STRIDEX, STRIDEY),
            (DILATIONX, DILATIONY),
            false,
            &mut cx,
        );
        model.weight.set(vec![
            0.1600, 0.2000, 0.1900, -0.1100, 0.0100, -0.0300, -0.1200, -0.0800, -0.1300, -0.0300,
            0.1600, -0.1700, -0.0000, 0.1900, 0.1300, 0.0300, -0.1500, 0.0900, 0.0100, 0.0200,
            0.1500, 0.0700, -0.0800, 0.1700, 0.1000, -0.0700, 0.1600, -0.1600, -0.1900, -0.0500,
            -0.2100, 0.0100, -0.2000, 0.2100, -0.0400, -0.1400, 0.1500, 0.0500, -0.1700, 0.1400,
        ]);

        let out1 = model.forward(inp1).retrieve();

        cx.execute();

        assert_close(&out1.data(), &exp_out1.data())
    }

    #[test]
    fn test_conv3d() {
        let mut cx = Graph::new();

        const CH_IN: usize = 5;
        const CH_OUT: usize = 2;
        const KERNELX: usize = 2;
        const KERNELY: usize = 2;
        const KERNELZ: usize = 2;
        const STRIDEX: usize = 2;
        const STRIDEY: usize = 2;
        const STRIDEZ: usize = 2;
        const DILATIONX: usize = 1;
        const DILATIONY: usize = 1;
        const DILATIONZ: usize = 1;
        const DIMX_IN: usize = 2;
        const DIMY_IN: usize = 3;
        const DIMZ_IN: usize = 5;
        const DIMX_OUT: usize = ((DIMX_IN - DILATIONX * (KERNELX - 1) - 1) / STRIDEX) + 1;
        const DIMY_OUT: usize = ((DIMY_IN - DILATIONY * (KERNELY - 1) - 1) / STRIDEY) + 1;
        const DIMZ_OUT: usize = ((DIMZ_IN - DILATIONZ * (KERNELZ - 1) - 1) / STRIDEZ) + 1;

        let inp1 = cx.tensor((CH_IN, DIMX_IN, DIMY_IN, DIMZ_IN));
        inp1.set(vec![
            // Example input data (5 channels, 2x3x5 volume)
            8., 8., 5., 7., 0., 6., 5., 3., 0., 7., 0., 6., 6., 7., 7., 5., 0., 6., 9., 4., 0., 8.,
            8., 5., 7., 6., 2., 8., 9., 5., 0., 3., 1., 1., 8., 4., 1., 1., 5., 6., 9., 3., 2., 9.,
            4., 7., 1., 0., 7., 7., 4., 9., 5., 0., 4., 7., 4., 7., 8., 8., 4., 8., 4., 7., 9., 3.,
            7., 9., 5., 8., 5., 9., 0., 9., 5., 6., 8., 9., 5., 4., 1., 9., 7., 2., 2., 7., 9., 3.,
            1., 2., 8., 4., 0., 8., 0., 5., 6., 7., 7., 4., 3., 4., 6., 8., 3., 7., 8., 8., 7., 1.,
            5., 1., 8., 0., 1., 1., 7., 3., 2., 1., 0., 4., 5., 4., 3., 2., 5., 4., 2., 4., 1., 9.,
            4., 1., 9., 7., 7., 1., 2., 6., 3., 4., 1., 1., 6., 6., 8., 2., 7., 7.,
        ]);

        let exp_out1 = cx.tensor((CH_OUT, DIMX_OUT, DIMY_OUT, DIMZ_OUT));
        exp_out1.set(vec![
            // Example expected output data (2 channels, 1x1x2 volume)
            90.6935, 98.7138, 98.8273, 102.6553,
        ]);

        exp_out1.retrieve();

        let model = Conv3D::new(
            CH_IN,
            CH_OUT,
            (KERNELX, KERNELY, KERNELZ),
            (STRIDEX, STRIDEY, STRIDEZ),
            (DILATIONX, DILATIONY, DILATIONZ),
            false,
            &mut cx,
        );
        let weights = vec![
            4.273e-01, 1.388e-01, 3.546e-01, 2.403e-01, 5.572e-01, 2.788e-01, 6.718e-01, 6.935e-01,
            8.410e-01, 1.297e-01, 7.073e-01, 3.455e-01, 4.166e-01, 9.513e-01, 4.682e-01, 4.546e-02,
            5.061e-01, 4.117e-01, 1.667e-01, 5.557e-02, 6.092e-01, 9.675e-01, 7.083e-01, 7.946e-01,
            3.518e-01, 4.697e-01, 6.052e-01, 6.832e-01, 2.312e-02, 6.932e-01, 6.135e-01, 9.216e-01,
            8.011e-01, 1.971e-01, 7.086e-01, 2.394e-01, 3.663e-01, 6.619e-01, 4.211e-01, 1.852e-01,
            8.635e-01, 1.311e-01, 4.206e-01, 5.413e-01, 7.938e-01, 9.604e-01, 7.966e-01, 7.400e-01,
            3.212e-01, 4.644e-01, 3.224e-01, 1.123e-01, 4.000e-01, 7.678e-01, 7.545e-01, 9.423e-01,
            5.605e-02, 2.675e-02, 5.022e-02, 8.632e-01, 9.305e-01, 9.836e-01, 1.635e-01, 2.379e-01,
            9.291e-01, 4.029e-01, 6.675e-01, 4.912e-01, 8.904e-01, 6.938e-01, 9.581e-01, 1.720e-01,
            7.835e-01, 4.658e-04, 2.818e-01, 5.373e-01, 3.437e-01, 1.254e-01, 6.868e-02, 7.546e-01,
        ];
        model.weight.set(weights);

        let out1 = model.forward(inp1).retrieve();

        cx.execute();

        assert_close(&out1.data(), &exp_out1.data());
    }
}
