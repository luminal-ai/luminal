//! Convolution lowerings (port batch 4).

use anyhow::Result;
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, Node};

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

/// View-based reshape with the recorder's `split_dims(axis, inner)`
/// contract: the passed extent is the INNER dim, so splitting off
/// leading target dims means passing the product of the remaining
/// target extents. (The shared `util::reshape_tensor` passes `target[i]`
/// directly, which transposes the result — e.g. `[4,27]` becomes
/// `[27,4]` — so conv reshapes must not rely on it.)
fn reshape_conv(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    let flat = t.flatten();
    let mut chain = flat.view();
    for axis in 0..target.len() - 1 {
        let inner: IntExpr = target[axis + 1..]
            .iter()
            .copied()
            .fold(IntExpr::from(1), |acc, dim| (acc * dim).simplify());
        chain = chain.split_dims(axis, inner);
    }
    chain.finish()
}

impl Translator<'_> {
    /// Translate `aten.conv{1,2,3}d.default` and `aten.convolution.default`.
    ///
    /// The PT2 export may omit defaulted trailing arguments entirely. In practice this means
    /// conv{N}d.default can show up as just `(input, weight)` for the no-bias, stride=1,
    /// padding=0, dilation=1, groups=1 case.
    pub(super) fn translate_conv(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, CONV_INPUT_ARG)?;
        let weight = self.get_input_tensor(node, CONV_WEIGHT_ARG)?;
        let bias = self.get_input_tensor(node, CONV_BIAS_ARG).ok();

        let rank = input.rank();
        anyhow::ensure!(rank >= 3, "conv: input rank {rank} is invalid");
        let spatial = rank - 2;
        let is_convolution = node.target.ends_with("convolution.default");

        if let Some(mode) = string_arg(node, CONV_PADDING_ARG) {
            anyhow::bail!("conv: string padding mode {mode:?} is not ported");
        }
        let stride = self
            .get_ints_arg(node, CONV_STRIDE_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let padding = self
            .get_ints_arg(node, CONV_PADDING_ARG)
            .unwrap_or_else(|_| vec![0; spatial]);
        let mut dilation = self
            .get_ints_arg(node, CONV_DILATION_ARG)
            .unwrap_or_else(|_| vec![1; spatial]);
        let groups = if is_convolution {
            let transposed = self
                .get_bool_arg(node, CONVOLUTION_TRANSPOSED_ARG)
                .unwrap_or(false);
            anyhow::ensure!(
                !transposed,
                "conv: ConvTranspose / transposed=true is not ported"
            );
            let output_padding = self
                .get_ints_arg(node, CONVOLUTION_OUTPUT_PADDING_ARG)
                .unwrap_or_else(|_| vec![0; spatial]);
            anyhow::ensure!(
                output_padding.iter().all(|&value| value == 0),
                "conv: output_padding is not supported for non-transposed convolution"
            );
            self.get_int_arg(node, CONVOLUTION_GROUPS_ARG).unwrap_or(1) as usize
        } else {
            self.get_int_arg(node, CONV_GROUPS_ARG).unwrap_or(1) as usize
        };
        if dilation.len() != spatial {
            dilation = vec![1; spatial];
        }

        let x_dims = input.dims();
        let w_dims = weight.dims();
        anyhow::ensure!(
            w_dims.len() == rank,
            "conv: weight rank {} does not match input rank {rank}",
            w_dims.len()
        );
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
            .map(|dim| {
                dim.to_usize()
                    .ok_or_else(|| anyhow::anyhow!("conv: kernel dims must be concrete"))
            })
            .collect::<Result<_>>()?;
        let kernel_product: usize = kernel_shape.iter().product();

        // ATen uses symmetric padding (same begin/end).
        let stride_u: Vec<usize> = stride.iter().map(|&value| value as usize).collect();
        let padding_u: Vec<usize> = padding.iter().map(|&value| value as usize).collect();
        let dilation_u: Vec<usize> = dilation.iter().map(|&value| value as usize).collect();

        // ATen computes convolutions in the opmath float type for half inputs.
        let in_dtype = input.dtype;
        let compute_dtype = if matches!(in_dtype, DType::F16 | DType::Bf16) {
            DType::F32
        } else {
            in_dtype
        };
        let input = input.cast(compute_dtype);
        let weight = weight.cast(compute_dtype);
        let bias = bias.map(|bias| bias.cast(compute_dtype));

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
                // General grouped: pre-pad the full input, slice per group.
                let padded_input = {
                    let mut pad_spec: Vec<(IntExpr, IntExpr)> =
                        vec![(0.into(), 0.into()); 2 + spatial];
                    for index in 0..spatial {
                        pad_spec[2 + index] = (padding_u[index].into(), padding_u[index].into());
                    }
                    input.pad(pad_spec, 0.0)
                };

                let no_pad = vec![0usize; spatial];
                let mut group_outputs = Vec::with_capacity(groups);
                for group in 0..groups {
                    let x_group = slice_channel_group(padded_input, group, ch_per_group, spatial);
                    let w_group =
                        slice_weight_group(weight, group, group_out, ch_per_group * kernel_product);
                    group_outputs.push(conv_unfold(
                        x_group,
                        w_group,
                        &kernel_shape,
                        &stride_u,
                        &dilation_u,
                        &no_pad,
                        &no_pad,
                        spatial,
                    ));
                }

                let mut result = group_outputs[0];
                for group_out in &group_outputs[1..] {
                    result = result.concat_along(*group_out, 1);
                }
                result
            }
        } else {
            let w_flat = reshape_conv(
                weight,
                &[IntExpr::from(ch_out), IntExpr::from(ch_in * kernel_product)],
            );
            conv_unfold(
                input,
                w_flat,
                &kernel_shape,
                &stride_u,
                &dilation_u,
                &padding_u,
                &padding_u,
                spatial,
            )
        };

        if let Some(bias) = bias {
            let out_dims = out.dims();
            let mut bias_expanded = bias.expand_dim(0, out_dims[0]);
            for index in 0..spatial {
                bias_expanded = bias_expanded.expand_dim(2 + index, out_dims[2 + index]);
            }
            out += bias_expanded;
        }

        Ok(out.cast(in_dtype))
    }
}

/// The padding argument as a string, when PT2 exported `"same"`/`"valid"`.
fn string_arg(node: &Node, index: usize) -> Option<String> {
    let input = node.inputs.get(index)?;
    if let Argument::Other(value) = &input.arg {
        return value
            .get("as_string")
            .and_then(|value| value.as_str())
            .map(str::to_string);
    }
    None
}

/// Slice input channels for one group.
/// Caller must pre-pad `x` so no additional padding is applied to the slice.
fn slice_channel_group(
    x: GraphTensor,
    group: usize,
    ch_per_group: usize,
    spatial: usize,
) -> GraphTensor {
    let start = group * ch_per_group;
    let end = start + ch_per_group;
    let dims = x.dims();
    let rank = 2 + spatial;
    let mut slices: Vec<(IntExpr, IntExpr)> = Vec::with_capacity(rank);
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
    group: usize,
    group_out: usize,
    flat_inner: usize,
) -> GraphTensor {
    let start = group * group_out;
    let end = start + group_out;
    let w_dims = w.dims();
    let mut slices: Vec<(IntExpr, IntExpr)> = Vec::with_capacity(w_dims.len());
    slices.push((start.into(), end.into()));
    for dim in w_dims.iter().skip(1) {
        slices.push((0.into(), *dim));
    }
    // A binary op materializes the sliced weight, which makes the
    // following flatten safe for the sliced buffer.
    let w_sliced = w.slice(slices) + 0.0;
    reshape_conv(
        w_sliced,
        &[IntExpr::from(group_out), IntExpr::from(flat_inner)],
    )
}

/// Core unfold-based convolution for a single group.
///
/// `x`: `[batch, ch_in, spatial...]`
/// `w_flat`: `[ch_out, ch_in * kernel_product]` (already reshaped)
/// Returns: `[batch, ch_out, out_spatial...]`
#[allow(clippy::too_many_arguments)]
fn conv_unfold(
    x: GraphTensor,
    w_flat: GraphTensor,
    kernel_shape: &[usize],
    strides: &[usize],
    dilations: &[usize],
    pads_begin: &[usize],
    pads_end: &[usize],
    spatial: usize,
) -> GraphTensor {
    let rank = 2 + spatial;

    // Pad spatial dimensions (skip if all padding is zero).
    let needs_pad = pads_begin.iter().any(|&pad| pad > 0) || pads_end.iter().any(|&pad| pad > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(IntExpr, IntExpr)> = vec![(0.into(), 0.into()); rank];
        for index in 0..spatial {
            padding[2 + index] = (pads_begin[index].into(), pads_end[index].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Build full-rank unfold parameters (1 for batch/channel, actual for spatial).
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

    let output_spatial_dims: Vec<IntExpr> = permuted.dims()[1..1 + spatial].to_vec();

    // Merge all channel+kernel dims into [N, spatial..., ch_in * kernel_product]
    let mut patches = permuted;
    let target = 2 + spatial;
    while patches.rank() > target {
        let last = patches.rank();
        patches = patches.merge_dims(last - 2, last - 1);
    }

    // Merge the window spatial dims into one.
    for _ in 1..spatial {
        patches = patches.merge_dims(1, 2);
    }
    // patches: [N, spatial_product, ch_in * kernel_product]

    let mut out = patches.matmul(w_flat.permute((1, 0)));
    // out: [N, spatial_product, ch_out]

    // Restore the spatial dimensions.
    for index in (1..spatial).rev() {
        out = out.split_dims(1, output_spatial_dims[index]);
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
/// avoiding per-channel input slicing.
///
/// `out[n, c, oh, ow] = sum_k patches[n, c, oh, ow, k] * weight[c, k]`
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

    let needs_pad = pads_begin.iter().any(|&pad| pad > 0) || pads_end.iter().any(|&pad| pad > 0);
    let padded = if needs_pad {
        let mut padding: Vec<(IntExpr, IntExpr)> = vec![(0.into(), 0.into()); rank];
        for index in 0..spatial {
            padding[2 + index] = (pads_begin[index].into(), pads_end[index].into());
        }
        x.pad(padding, 0.0)
    } else {
        x
    };

    // Unfold the full [N, C, H+2p, W+2p] with kernel [1, 1, kH, kW].
    let mut kernel_full = vec![1usize; rank];
    let mut stride_full = vec![1usize; rank];
    let mut dilation_full = vec![1usize; rank];
    kernel_full[2..(spatial + 2)].copy_from_slice(&kernel_shape[..spatial]);
    stride_full[2..(spatial + 2)].copy_from_slice(&strides[..spatial]);
    dilation_full[2..(spatial + 2)].copy_from_slice(&dilations[..spatial]);

    let unfolded = padded.unfold(kernel_full, stride_full, dilation_full);
    // Shape: [N, C, out_spatial..., 1, 1, k_spatial...]

    // Permute to [N, C, out_spatial..., k_all...]
    let mut perm: Vec<usize> = Vec::with_capacity(2 * rank);
    perm.push(0); // N
    perm.push(1); // C
    perm.extend(2..2 + spatial); // win_spatial
    perm.extend(rank..2 * rank); // all kernel dims
    let permuted = unfolded.permute(perm);

    let out_spatial_dims: Vec<IntExpr> = permuted.dims()[2..2 + spatial].to_vec();

    // Merge all kernel dims (including 1-size k_N, k_C) into kernel_product.
    let target = 3 + spatial; // [N, C, spatial..., K]
    let mut patches = permuted;
    while patches.rank() > target {
        let last = patches.rank();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    // patches: [N, C, out_spatial..., kernel_product]

    // Merge spatial into one: [N, C, out_spatial_product, kernel_product]
    for _ in 1..spatial {
        patches = patches.merge_dims(2, 3);
    }

    // Weight [C * group_out, 1, *kernel] -> [C, group_out, kernel_product]
    let w_flat = reshape_conv(
        w,
        &[
            IntExpr::from(ch),
            IntExpr::from(group_out),
            IntExpr::from(kernel_product),
        ],
    );

    // patches: [N, C, out_spatial_product, kernel_product]
    // Expand to [N, C, group_out, out_spatial_product, kernel_product]
    let patches = patches.expand_dim(2, group_out);

    // Broadcast the weight across the batch and spatial axes so the
    // elementwise multiply sees equal visible shapes.
    let w_expanded = w_flat
        .expand_dim(0, patches.dims()[0])
        .expand_dim(3, patches.dims()[3]);

    // Element-wise multiply and sum over the kernel dim.
    let product = patches * w_expanded;
    let mut out = product.sum(vec![4]).merge_dims(1, 2);
    // out: [N, C * group_out, out_spatial_product]

    // Restore the spatial dimensions.
    for index in (1..spatial).rev() {
        out = out.split_dims(2, out_spatial_dims[index]);
    }
    out
}
