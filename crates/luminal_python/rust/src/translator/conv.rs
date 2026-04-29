use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;

use super::Translator;

const CONV_INPUT_ARG: usize = 0;
const CONV_WEIGHT_ARG: usize = 1;
const CONV_BIAS_ARG: usize = 2;
const CONV_STRIDE_ARG: usize = 3;
const CONV_PADDING_ARG: usize = 4;
const CONV_DILATION_ARG: usize = 5;
const CONV_GROUPS_ARG: usize = 6;

const CONVOLUTION_TRANSPOSED_ARG: usize = 6;
const CONVOLUTION_OUTPUT_PADDING_ARG: usize = 7;
const CONVOLUTION_GROUPS_ARG: usize = 8;

impl<'a> Translator<'a> {
    /// Translate aten.conv{1,2,3}d.default and aten.convolution.default.
    ///
    /// The PT2 export may omit defaulted trailing arguments entirely. In practice this means
    /// conv{N}d.default can show up as just `(input, weight)` for the no-bias, stride=1,
    /// padding=0, dilation=1, groups=1 case.
    pub(crate) fn translate_conv(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, CONV_INPUT_ARG)?;
        let weight = self.get_input_tensor(node, CONV_WEIGHT_ARG)?;
        let bias = self.get_input_tensor(node, CONV_BIAS_ARG).ok();

        let x_dims = input.dims();
        let w_dims = weight.dims();
        let rank = x_dims.len();
        let spatial = rank - 2;

        let stride = self
            .get_ints_arg(node, CONV_STRIDE_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let padding = self
            .get_ints_arg(node, CONV_PADDING_ARG)
            .unwrap_or_else(|_| vec![0; spatial]);
        let mut dilation = self
            .get_ints_arg(node, CONV_DILATION_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let groups = if node.target == "torch.ops.aten.convolution.default" {
            let transposed = self
                .get_bool_arg(node, CONVOLUTION_TRANSPOSED_ARG)
                .unwrap_or(false);
            anyhow::ensure!(
                !transposed,
                "conv: ConvTranspose / transposed=true is not supported yet"
            );
            let output_padding = self
                .get_ints_arg(node, CONVOLUTION_OUTPUT_PADDING_ARG)
                .unwrap_or_else(|_| vec![0; spatial]);
            anyhow::ensure!(
                output_padding.iter().all(|&v| v == 0),
                "conv: output_padding is not supported for non-transposed convolution"
            );
            self.get_int_arg(node, CONVOLUTION_GROUPS_ARG).unwrap_or(1) as usize
        } else {
            self.get_int_arg(node, CONV_GROUPS_ARG).unwrap_or(1) as usize
        };
        if dilation.len() != spatial {
            dilation = vec![1; spatial];
        }

        let ch_out = w_dims[0]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("conv: weight C_out must be concrete"))?;
        let ch_in = x_dims[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("conv: input C_in must be concrete"))?;
        anyhow::ensure!(
            stride.len() == spatial && padding.len() == spatial && dilation.len() == spatial,
            "conv: stride/padding/dilation rank must match spatial rank {spatial}"
        );
        anyhow::ensure!(
            groups > 0 && ch_in % groups == 0 && ch_out % groups == 0,
            "conv: invalid group configuration (C_in={ch_in}, C_out={ch_out}, groups={groups})"
        );
        let ch_per_group = ch_in / groups;

        let kernel_shape: Vec<usize> = w_dims[2..]
            .iter()
            .map(|d| {
                d.to_usize()
                    .ok_or_else(|| anyhow::anyhow!("conv: kernel dims must be concrete"))
            })
            .collect::<Result<_>>()?;
        let kernel_product: usize = kernel_shape.iter().product();

        // ATen uses symmetric padding (same begin/end)
        let stride_u: Vec<usize> = stride.iter().map(|&v| v as usize).collect();
        let padding_u: Vec<usize> = padding.iter().map(|&v| v as usize).collect();
        let dilation_u: Vec<usize> = dilation.iter().map(|&v| v as usize).collect();

        let mut out = if groups > 1 {
            let group_out = ch_out / groups;

            if ch_per_group == 1 {
                // Depthwise (including channel multiplier > 1): avoid per-channel slicing.
                depthwise_conv(
                    input,
                    weight,
                    &kernel_shape,
                    &stride_u,
                    &dilation_u,
                    &padding_u,
                    &padding_u,
                    ch_in,
                    group_out,
                    kernel_product,
                    spatial,
                )
            } else {
                // General grouped: pre-pad full input then slice per group
                let padded_input = {
                    let mut pad_spec: Vec<(Expression, Expression)> =
                        vec![(0.into(), 0.into()); 2 + spatial];
                    for i in 0..spatial {
                        pad_spec[2 + i] = (padding_u[i].into(), padding_u[i].into());
                    }
                    input.pad(pad_spec, 0.0)
                };

                let no_pad = vec![0usize; spatial];
                let mut group_outputs = Vec::with_capacity(groups);
                for g in 0..groups {
                    let x_g = slice_channel_group(padded_input, g, ch_per_group, spatial);
                    let w_g =
                        slice_weight_group(weight, g, group_out, ch_per_group * kernel_product);
                    group_outputs.push(conv_unfold(
                        x_g,
                        w_g,
                        &kernel_shape,
                        &stride_u,
                        &dilation_u,
                        &no_pad,
                        &no_pad,
                        ch_per_group,
                        group_out,
                        spatial,
                    ));
                }

                let mut result = group_outputs[0];
                for g_out in &group_outputs[1..] {
                    result = result.concat_along(*g_out, 1);
                }
                result
            }
        } else {
            let mut w_flat = weight;
            w_flat.shape = ShapeTracker::new_with_element_bits(
                vec![ch_out, ch_in * kernel_product],
                weight.dtype.bits(),
            );

            conv_unfold(
                input,
                w_flat,
                &kernel_shape,
                &stride_u,
                &dilation_u,
                &padding_u,
                &padding_u,
                ch_in,
                ch_out,
                spatial,
            )
        };

        if let Some(b) = bias {
            let out_dims = out.dims();
            let mut b_expanded = b.expand_dim(0, 1);
            for i in 0..spatial {
                b_expanded = b_expanded.expand_dim(2 + i, out_dims[2 + i]);
            }
            out += b_expanded;
        }

        Ok(out)
    }
}

/// Slice input channels for one group.
/// Caller must pre-pad `x` so no additional padding is applied to the slice.
fn slice_channel_group(
    x: GraphTensor,
    g: usize,
    ch_per_group: usize,
    spatial: usize,
) -> GraphTensor {
    let start = g * ch_per_group;
    let end = start + ch_per_group;
    let dims = x.dims();
    let rank = 2 + spatial;
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(rank);
    slices.push((0.into(), dims[0]));
    slices.push((start.into(), end.into()));
    for dim in dims.iter().take(rank).skip(2) {
        slices.push((0.into(), *dim));
    }
    x.slice(slices)
}

/// Slice and flatten weight for one group.
fn slice_weight_group(
    w: GraphTensor,
    g: usize,
    group_out: usize,
    flat_inner: usize,
) -> GraphTensor {
    let start = g * group_out;
    let end = start + group_out;
    let w_dims = w.dims();
    let mut slices: Vec<(Expression, Expression)> = Vec::with_capacity(w_dims.len());
    slices.push((start.into(), end.into()));
    for dim in w_dims.iter().skip(1) {
        slices.push((0.into(), *dim));
    }
    // Materialize through Add: binary op outputs are contiguous in Luminal, which makes the
    // following flatten safe for the sliced weight buffer.
    let w_sliced = w.slice(slices) + 0.0;
    let mut w_flat = w_sliced;
    w_flat.shape =
        ShapeTracker::new_with_element_bits(vec![group_out, flat_inner], w_sliced.dtype.bits());
    w_flat
}

/// Core unfold-based convolution for a single group.
///
/// `x`: [batch, ch_in, spatial...]
/// `w_flat`: [ch_out, ch_in * kernel_product] (already reshaped)
/// Returns: [batch, ch_out, out_spatial...]
#[allow(clippy::too_many_arguments)]
fn conv_unfold(
    x: GraphTensor,
    w_flat: GraphTensor,
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads_begin: &[usize],
    pads_end: &[usize],
    _ch_in: usize,
    _ch_out: usize,
    spatial: usize,
) -> GraphTensor {
    let rank = 2 + spatial;

    // Pad spatial dimensions (skip if all padding is zero)
    let needs_pad = pads_begin.iter().any(|&p| p > 0) || pads_end.iter().any(|&p| p > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
        for i in 0..spatial {
            padding[2 + i] = (pads_begin[i].into(), pads_end[i].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Build full-rank unfold parameters (1 for batch/channel, actual for spatial)
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    kernel_full[2..(spatial + 2)].copy_from_slice(&kernel_shape[..spatial]);
    stride_full[2..(spatial + 2)].copy_from_slice(&strides[..spatial]);
    dilation_full[2..(spatial + 2)].copy_from_slice(&dilations[..spatial]);

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
    // Shape: [win_N, win_C, win_spatial..., k_N=1, k_C=1, k_spatial...]

    // Permute to [N, win_spatial..., C_in, k_N, k_C, k_spatial...]
    let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
    perm.push(0);
    perm.extend(2..2 + spatial);
    perm.push(1);
    perm.extend(rank..2 * rank);
    let permuted = unfolded.permute(perm);

    let output_spatial_dims: Vec<Expression> = permuted.dims()[1..1 + spatial].to_vec();

    // Merge all channel+kernel dims into [N, spatial..., ch_in * kernel_product]
    let mut patches = permuted;
    let target = 2 + spatial;
    while patches.dims().len() > target {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }

    // Merge spatial dims into one
    for _ in 1..spatial {
        patches = patches.merge_dims(1, 2);
    }
    // patches: [N, spatial_product, ch_in * kernel_product]

    let mut out = patches.matmul(w_flat.permute((1, 0)));
    // out: [N, spatial_product, ch_out]

    // Restore spatial dimensions
    for i in (1..spatial).rev() {
        out = out.split_dims(1, output_spatial_dims[i]);
    }

    // Move ch_out from last to position 1: [N, ch_out, spatial...]
    let mut final_order: Vec<usize> = Vec::with_capacity(2 + spatial);
    final_order.push(0);
    final_order.push(1 + spatial);
    final_order.extend(1..1 + spatial);
    out.permute(final_order)
}

/// Depthwise convolution: groups == in_channels, ch_per_group == 1.
///
/// Processes all channels simultaneously using element-wise multiply + reduce,
/// avoiding per-channel input slicing which can cause index-expression bugs in luminal.
///
/// out[n, c, oh, ow] = sum_k patches[n, c, oh, ow, k] * weight[c, k]
#[allow(clippy::too_many_arguments)]
fn depthwise_conv(
    x: GraphTensor,
    w: GraphTensor, // [C, 1, *kernel]
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads_begin: &[usize],
    pads_end: &[usize],
    ch: usize,
    group_out: usize,
    kernel_product: usize,
    spatial: usize,
) -> GraphTensor {
    let rank = 2 + spatial;

    let needs_pad = pads_begin.iter().any(|&p| p > 0) || pads_end.iter().any(|&p| p > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); rank];
        for i in 0..spatial {
            padding[2 + i] = (pads_begin[i].into(), pads_end[i].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Unfold the full [N, C, H+2p, W+2p] with kernel [1, 1, kH, kW]
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    kernel_full[2..(spatial + 2)].copy_from_slice(&kernel_shape[..spatial]);
    stride_full[2..(spatial + 2)].copy_from_slice(&strides[..spatial]);
    dilation_full[2..(spatial + 2)].copy_from_slice(&dilations[..spatial]);

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
    // Shape: [N, C, out_H, out_W, 1, 1, kH, kW]

    // Permute to [N, C, out_spatial..., k_all...]
    let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
    perm.push(0); // N
    perm.push(1); // C
    perm.extend(2..2 + spatial); // win_spatial
    perm.extend(rank..2 * rank); // all kernel dims
    let permuted = unfolded.permute(perm);

    let out_spatial_dims: Vec<Expression> = permuted.dims()[2..2 + spatial].to_vec();

    // Merge all kernel dims (including 1-size k_N, k_C) into kernel_product
    let target = 3 + spatial; // [N, C, spatial..., K]
    let mut patches = permuted;
    while patches.dims().len() > target {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    // patches: [N, C, out_H, ..., out_W, kernel_product]

    // Merge spatial into one: [N, C, out_spatial_product, kernel_product]
    for _ in 1..spatial {
        patches = patches.merge_dims(2, 3);
    }

    // Weight [C * group_out, 1, *kernel] -> [C, group_out, kernel_product]
    let mut w_flat = w;
    w_flat.shape =
        ShapeTracker::new_with_element_bits(vec![ch, group_out, kernel_product], w.dtype.bits());

    // patches: [N, C, out_spatial_product, kernel_product]
    // Expand to [N, C, group_out, out_spatial_product, kernel_product]
    let patches = patches.expand_dim(2, group_out);

    // Expand weight for broadcast: [1, C, group_out, out_spatial_product, kernel_product]
    let w_expanded = w_flat.expand_dim(0, 1).expand_dim(3, patches.dims()[3]);

    // Element-wise multiply and sum over kernel dim
    let product = patches * w_expanded;
    let mut out = product.sum(vec![4]).merge_dims(1, 2);
    // out: [N, C * group_out, out_spatial_product]

    // Restore spatial dimensions
    for i in (1..spatial).rev() {
        out = out.split_dims(2, out_spatial_dims[i]);
    }
    // out: [N, C, out_spatial_0, ..., out_spatial_{s-1}]

    out
}
