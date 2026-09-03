use luminal::prelude::*;

/// Generic N-dimensional convolution layer implemented with the GraphTensor `unfold` helper.
///
/// The layer expects inputs shaped like `[batch..., channels, spatial...]` where the number of
/// spatial dimensions is greater than zero. The kernel configuration controls how many spatial
/// axes are convolved (N) and must be shorter than the input rank (K): `K > N` is asserted.
#[derive(Clone, Debug)]
pub struct ConvNdConfig {
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
}

impl ConvNdConfig {
    pub fn new(
        kernel: impl AsRef<[usize]>,
        stride: impl AsRef<[usize]>,
        dilation: impl AsRef<[usize]>,
        padding: impl AsRef<[usize]>,
    ) -> Self {
        let kernel = kernel.as_ref().to_vec();
        let stride = stride.as_ref().to_vec();
        let dilation = dilation.as_ref().to_vec();
        let padding = padding.as_ref().to_vec();
        assert!(
            !kernel.is_empty(),
            "ConvND requires at least one spatial dimension in the kernel",
        );
        let k = kernel.len();
        assert_eq!(
            stride.len(),
            k,
            "Stride dimensions ({}) must match kernel dimensions ({k})",
            stride.len()
        );
        assert_eq!(
            dilation.len(),
            k,
            "Dilation dimensions ({}) must match kernel dimensions ({k})",
            dilation.len()
        );
        assert_eq!(
            padding.len(),
            k,
            "Padding dimensions ({}) must match kernel dimensions ({k})",
            padding.len()
        );

        Self {
            kernel,
            stride,
            dilation,
            padding,
        }
    }

    fn apply(
        &self,
        input: GraphTensor,
        weight: GraphTensor,
        bias: Option<GraphTensor>,
    ) -> GraphTensor {
        let input_dims = input.dims();
        let rank = input_dims.len();
        let spatial = self.kernel.len();

        assert!(
            rank > spatial,
            "ConvND expects input rank ({rank}) to be greater than kernel dims ({spatial})",
        );

        let batch_len = rank - spatial - 1;
        assert_eq!(weight.rank(), 2, "convolution weight must be rank two");
        assert_eq!(
            input.dtype, weight.dtype,
            "convolution input/weight dtype mismatch"
        );
        let kernel_product: usize = self.kernel.iter().product();
        assert_eq!(
            input_dims[batch_len] * kernel_product,
            weight.dims()[1],
            "convolution input channels do not match weight"
        );

        // Pad only the spatial dimensions.
        let mut padding = vec![(IntExpr::from(0), IntExpr::from(0)); rank];
        for (i, pad) in self.padding.iter().enumerate() {
            let axis = batch_len + 1 + i;
            padding[axis] = (IntExpr::from(*pad), IntExpr::from(*pad));
        }
        let padded = input.pad(padding, 0.0);

        // Build unfold parameters with ones for non-spatial axes.
        let mut kernel_shape = vec![1; rank];
        let mut stride_shape = vec![1; rank];
        let mut dilation_shape = vec![1; rank];
        for i in 0..spatial {
            let axis = batch_len + 1 + i;
            kernel_shape[axis] = self.kernel[i];
            stride_shape[axis] = self.stride[i];
            dilation_shape[axis] = self.dilation[i];
        }

        // Keep the unfold and patch reshape in one logical view operation.
        let unfolded = padded.unfold_view(kernel_shape, stride_shape, dilation_shape);
        let unfolded_dims = unfolded.dims();

        // Capture output spatial dimensions from the unfolded view.
        let output_dims: Vec<IntExpr> =
            unfolded_dims[batch_len + 1..batch_len + 1 + spatial].to_vec();

        // Reorder to [batch..., out..., channels, kernel_spatial..., kernel_batch..., kernel_channel].
        let mut order2 = Vec::with_capacity(2 * rank);
        // window batch dims
        order2.extend(0..batch_len);
        // window spatial dims (outputs)
        order2.extend(batch_len + 1..batch_len + 1 + spatial);
        // window channel dim
        order2.push(batch_len);
        // kernel spatial dims
        order2.extend(rank + batch_len + 1..rank + batch_len + 1 + spatial);
        // kernel batch dims and kernel channel dim (to be merged away)
        order2.extend(rank..rank + batch_len + 1);
        let mut patches = unfolded.permute(order2);

        // Drop kernel axes for batch + channel by merging them into the previous dimension.
        for _ in 0..=batch_len {
            let last = patches.rank();
            patches = patches.merge_dims(last - 2, last - 1);
        }

        // Flatten channel and kernel spatial dimensions together.
        for _ in 0..spatial {
            let channel_axis = batch_len + spatial;
            patches = patches.merge_dims(channel_axis, channel_axis + 1);
        }

        // Collapse batch dimensions into one and output dimensions into one for matmul.
        for _ in 1..batch_len {
            patches = patches.merge_dims(0, 1);
        }
        for _ in 1..spatial {
            patches = patches.merge_dims(1, 2);
        }

        let mut out = patches.finish().matmul(weight.permute((1, 0)));

        // Restore batch and spatial dimensions. The collapse loops merged
        // k dims into 1, so restore splits k-1 times: splitting by every dim
        // including the outermost would leave a spurious leading 1-dim.
        let batch_dims = self.input_batch_dims(&input_dims, batch_len);
        let mut out_view = out.view();
        for dim in batch_dims.iter().skip(1).rev() {
            out_view = out_view.split_dims(0, *dim);
        }
        for dim in output_dims.iter().skip(1).rev() {
            out_view = out_view.split_dims(batch_len, *dim);
        }

        // Move channel dimension ahead of the spatial axes: [batch..., ch_out, spatial...]
        let mut final_order: Vec<usize> = (0..batch_len).collect();
        final_order.push(batch_len + spatial);
        final_order.extend(batch_len..batch_len + spatial);
        out = out_view.permute(final_order).finish();

        if let Some(bias) = bias {
            assert_eq!(bias.rank(), 1, "convolution bias must be rank one");
            assert_eq!(
                bias.dtype, out.dtype,
                "convolution output/bias dtype mismatch"
            );
            assert_eq!(bias.dims()[0], weight.dims()[0], "convolution bias width");
            let out_dims = out.dims();
            out += bias
                .expand_lhs(&out_dims[..batch_len])
                .expand_rhs(&out_dims[batch_len + 1..]);
        }

        out
    }

    fn input_batch_dims(&self, input_dims: &[IntExpr], batch_len: usize) -> Vec<IntExpr> {
        input_dims[..batch_len].to_vec()
    }

    pub fn infer_output_shape(&self, input: &[usize], ch_in: usize, ch_out: usize) -> Vec<usize> {
        let rank = input.len();
        let spatial = self.kernel.len();

        assert!(rank > spatial, "expected input rank > spatial dims");
        let batch_len = rank - spatial - 1;
        assert_eq!(
            input[batch_len], ch_in,
            "input channel dimension does not match ch_in",
        );

        let batch_prefix = &input[..batch_len];
        let spatial_dims = &input[batch_len + 1..];
        let out_spatial: Vec<usize> = spatial_dims
            .iter()
            .zip(
                self.kernel
                    .iter()
                    .zip(self.stride.iter())
                    .zip(self.dilation.iter())
                    .zip(self.padding.iter()),
            )
            .map(|(dim, (((k, s), d), p))| (dim + 2 * p - d * (k - 1) - 1) / s + 1)
            .collect();

        let mut shape = batch_prefix.to_vec();
        shape.push(ch_out);
        shape.extend(out_spatial);
        shape
    }
}

/// Apply an N-dimensional convolution with a caller-supplied canonical
/// `(out_channels, in_channels * kernel_elements)` weight.
pub fn conv_nd(
    input: GraphTensor,
    weight: GraphTensor,
    bias: Option<GraphTensor>,
    config: &ConvNdConfig,
) -> GraphTensor {
    config.apply(input, weight, bias)
}

#[cfg(test)]
mod tests {
    use super::ConvNdConfig;
    use candle_core::{Device, Tensor};

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (idx, (lhs, rhs)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            if diff > 1e-4 {
                panic!("values differ at {idx}: {lhs} vs {rhs} (diff {diff})");
            }
        }
    }

    fn candle_conv1d_output(
        conv: &ConvNdConfig,
        ch_in: usize,
        ch_out: usize,
        input: &[f32],
        width: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        let device = Device::Cpu;
        let input = Tensor::from_vec(input.to_vec(), (1, ch_in, width), &device)?;
        let weight = Tensor::from_vec(weight.to_vec(), (ch_out, ch_in, conv.kernel[0]), &device)?;
        let bias = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), ch_out, &device)?),
            None => None,
        };

        let output = input.conv1d(
            &weight,
            conv.padding[0],
            conv.stride[0],
            conv.dilation[0],
            1,
        )?;
        let output = match bias {
            Some(bias) => {
                let bias = bias.reshape((1, ch_out, 1))?;
                output.broadcast_add(&bias)?
            }
            None => output,
        };
        output.flatten_all()?.to_vec1::<f32>()
    }

    fn candle_conv2d_output(
        conv: &ConvNdConfig,
        channels: (usize, usize),
        input: &[f32],
        height: usize,
        width: usize,
        weight: &[f32],
        bias: Option<&[f32]>,
    ) -> candle_core::Result<Vec<f32>> {
        let (ch_in, ch_out) = channels;
        let device = Device::Cpu;
        let input = Tensor::from_vec(input.to_vec(), (1, ch_in, height, width), &device)?;
        let weight = Tensor::from_vec(
            weight.to_vec(),
            (ch_out, ch_in, conv.kernel[0], conv.kernel[1]),
            &device,
        )?;
        let bias = match bias {
            Some(b) => Some(Tensor::from_vec(b.to_vec(), ch_out, &device)?),
            None => None,
        };

        assert_eq!(
            conv.padding[0], conv.padding[1],
            "Candle conv2d only supports equal padding"
        );
        assert_eq!(
            conv.stride[0], conv.stride[1],
            "Candle conv2d only supports equal stride"
        );
        assert_eq!(
            conv.dilation[0], conv.dilation[1],
            "Candle conv2d only supports equal dilation"
        );

        let output = input.conv2d(
            &weight,
            conv.padding[0],
            conv.stride[0],
            conv.dilation[0],
            1,
        )?;
        let output = match bias {
            Some(bias) => {
                let bias = bias.reshape((1, ch_out, 1, 1))?;
                output.broadcast_add(&bias)?
            }
            None => output,
        };
        output.flatten_all()?.to_vec1::<f32>()
    }

    #[test]
    fn conv1d_values_match_expected_window_sums() -> candle_core::Result<()> {
        let conv = ConvNdConfig::new([3], [1], [1], [1]);

        let input = [1., 2., 3., 4., 5.];
        let weight = [1., 1., 1.];
        let bias = [0.5];

        let out = candle_conv1d_output(&conv, 1, 1, &input, input.len(), &weight, Some(&bias))?;

        assert_close(&out, &[3.5, 6.5, 9.5, 12.5, 9.5]);
        Ok(())
    }

    #[test]
    fn conv2d_values_accumulate_across_channels() -> candle_core::Result<()> {
        let conv = ConvNdConfig::new([2, 2], [1, 1], [1, 1], [0, 0]);

        let input = [
            1., 2., 3., 4., 5., 6., 7., 8., 9., // channel 0
            9., 8., 7., 6., 5., 4., 3., 2., 1., // channel 1
        ];
        let weight = [1., 1., 1., 1., 2., 2., 2., 2.];
        let bias = [0.25];

        let out = candle_conv2d_output(&conv, (2, 1), &input, 3, 3, &weight, Some(&bias))?;

        assert_close(&out, &[68.25, 64.25, 56.25, 52.25]);
        Ok(())
    }

    #[test]
    fn conv1d_shapes_follow_stride_and_padding() {
        let conv = ConvNdConfig::new([3], [2], [1], [1]);

        // expected length: floor((padded_len - dilation*(k-1) -1)/stride +1)
        // padded_len = 7 + 2 = 9
        // effective kernel = 3
        // => (9 -3)/2 +1 = 4
        let inferred = conv.infer_output_shape(&[2, 1, 7], 1, 1);
        assert_eq!(inferred, vec![2, 1, 4]);
    }

    #[test]
    fn conv2d_shapes_follow_stride_and_padding() {
        let conv = ConvNdConfig::new([2, 3], [1, 2], [1, 1], [0, 1]);

        // height: (5 - dilation*(2-1) -1 + 0 +0)/1 +1 = 4
        // width: (6 - dilation*(3-1) -1 + 1 +1)/2 +1 = 3
        let inferred = conv.infer_output_shape(&[1, 3, 5, 6], 3, 2);
        assert_eq!(inferred, vec![1, 2, 4, 3]);
    }
}

#[cfg(test)]
mod forward_tests {
    use super::{ConvNdConfig, conv_nd};
    use luminal::prelude::*;

    /// ConvND forward vs a naive host-side convolution, on the reference
    /// runtime. Covers multi-channel input/output and a batch dimension.
    #[test]
    fn convnd_2d_matches_naive() {
        let (b, ci, co, h, w, k) = (2usize, 2usize, 3usize, 4usize, 4usize, 2usize);
        let (oh, ow) = (h - k + 1, w - k + 1);
        let x_data: Vec<f32> = (0..b * ci * h * w)
            .map(|i| (i as f32 * 0.13).sin())
            .collect();
        let w_data: Vec<f32> = (0..co * ci * k * k)
            .map(|i| (i as f32 * 0.29).cos())
            .collect();

        let mut cx = Graph::new();
        let x = cx.tensor((b, ci, h, w), DType::F32);
        let weight = cx.tensor((co, ci * k * k), DType::F32);
        let config = ConvNdConfig::new([k, k], [1, 1], [1, 1], [0, 0]);
        let out = conv_nd(x, weight, None, &config).output();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (x.id, x_data.clone().into()),
                (weight.id, w_data.clone().into()),
            ],
        );
        let got = rt.get_f32(out.id).unwrap().clone();

        // Naive conv; weight layout is (co, ci * k * k), ci-major.
        let mut want = vec![0.0f32; b * co * oh * ow];
        for bi in 0..b {
            for o in 0..co {
                for y in 0..oh {
                    for xx in 0..ow {
                        let mut acc = 0.0;
                        for c in 0..ci {
                            for dy in 0..k {
                                for dx in 0..k {
                                    let xv = x_data[((bi * ci + c) * h + y + dy) * w + xx + dx];
                                    let wv = w_data[(o * ci + c) * k * k + dy * k + dx];
                                    acc += xv * wv;
                                }
                            }
                        }
                        want[((bi * co + o) * oh + y) * ow + xx] = acc;
                    }
                }
            }
        }
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() < 1e-4, "element {i}: got {g}, want {e}");
        }
    }

    #[test]
    fn convnd_applies_caller_supplied_bias() {
        let mut cx = Graph::new();
        let input = cx.tensor((1, 1, 2, 2), DType::F32);
        let weight = cx.tensor((2, 1), DType::F32);
        let bias = cx.tensor(2, DType::F32);
        let config = ConvNdConfig::new([1, 1], [1, 1], [1, 1], [0, 0]);
        let output = conv_nd(input, weight, Some(bias), &config).output();
        let runtime = luminal_reference::harness::run_reference(
            &cx,
            &[
                (input.id, vec![1.0, 2.0, 3.0, 4.0].into()),
                (weight.id, vec![2.0, -1.0].into()),
                (bias.id, vec![0.5, 1.0].into()),
            ],
        );
        assert_eq!(
            runtime.get_f32(output.id).unwrap(),
            &[2.5, 4.5, 6.5, 8.5, 0.0, -1.0, -2.0, -3.0]
        );
    }
}
