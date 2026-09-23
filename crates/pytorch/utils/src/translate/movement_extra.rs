//! Additional ATen lowerings ported from the parked translator (port batch 6+).
//!
//! This batch covers the remaining movement/resample odds and ends:
//! `permute_copy`, `view_copy`, multi-dim `squeeze`, nearest/bilinear
//! upsample (including the antialiased bilinear variant),
//! `embedding_renorm`, and `slice_scatter`.
//!
//! The recorder has no strides, so the `*_copy` variants (which exist in
//! ATen to force fresh storage) are value-identical to their view
//! counterparts: the reference runtime materializes a fresh output
//! buffer at the boundary, which is where the copy is observable.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, normalize_dim};
use crate::pt2_schema::{Argument, Node, NodeInput};

/// Row-major strides as `IntExpr`s: `stride[i] = prod(dims[i+1..])`.
fn row_major_strides(dims: &[IntExpr]) -> Vec<IntExpr> {
    (0..dims.len())
        .map(|i| {
            dims[i + 1..]
                .iter()
                .fold(IntExpr::from(1), |acc, d| acc * *d)
        })
        .collect()
}

/// Reshape a flat tensor to `target` by splitting from the right, so every
/// intermediate extent is the exact product of the remaining targets. The
/// shared `util::reshape_tensor` splits with `target[i]` as the inner extent,
/// which produces the wrong intermediate dims (and panics) for any target
/// needing more than one split. Recorded reshapes are pure views, so the
/// element order is preserved either way; this form keeps the extents valid.
fn reshape_view(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    let mut flat = t.flatten();
    for axis in 0..target.len().saturating_sub(1) {
        let inner: IntExpr = target[axis + 1..]
            .iter()
            .fold(IntExpr::from(1), |acc, d| acc * *d);
        flat = flat.split_dims(axis, inner);
    }
    flat
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Copy-flavoured movement
    // ---------------------------------------------------------------

    /// `permute_copy.default`: same value as `permute`, but the boundary
    /// copy is observable (see the module doc). `translate.rs` handles
    /// `permute.default` inline; this shares its axis normalization.
    pub(super) fn translate_permute_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let axes = self.ints_arg(&node.inputs[1])?;
        Ok(x.permute(super::normalize_axes(&axes, x.rank())?))
    }

    /// `view_copy.default`: same value as `view` (already in `ops.rs`).
    pub(super) fn translate_view_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_view(node)
    }

    /// `squeeze.dims` / `squeeze.default`.
    ///
    /// `squeeze.dims` removes only the listed axes whose extent is one
    /// (non-one axes are left alone), tracking the axis shift as earlier
    /// axes are dropped. `squeeze.default` has no dims input and removes
    /// every size-1 axis.
    pub(super) fn translate_squeeze_dims(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let Some(dims_input) = node.inputs.get(1) else {
            let mut result = x;
            let mut axis = 0;
            while axis < result.rank() {
                if result.dims()[axis].to_usize() == Some(1) {
                    result = result.squeeze(axis);
                } else {
                    axis += 1;
                }
            }
            return Ok(result);
        };
        let ndim = x.rank();
        let dims = self.ints_arg(dims_input)?;
        let mut sorted: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
        sorted.sort_unstable();
        let mut result = x;
        let mut offset = 0;
        for d in sorted {
            if result.dims()[d - offset].to_usize() == Some(1) {
                result = result.squeeze(d - offset);
                offset += 1;
            }
        }
        Ok(result)
    }

    // ---------------------------------------------------------------
    // Upsample
    // ---------------------------------------------------------------

    /// `upsample_nearest2d.vec`: nearest-neighbour resize of the spatial
    /// axes. The output extents come from PT2 metadata; the optional
    /// scale factors select ATen's `floor(j / s)` indexing branch.
    pub(super) fn translate_upsample_nearest2d(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        if input.rank() != 4 {
            bail!(
                "upsample_nearest2d expects a 4D (N, C, H, W) input, got {}D",
                input.rank()
            );
        }
        let input_height = input.dims()[2]
            .to_usize()
            .context("upsample_nearest2d requires a static input height")?;
        let input_width = input.dims()[3]
            .to_usize()
            .context("upsample_nearest2d requires a static input width")?;

        let output_dimensions = self.output_meta_shape(node)?;
        if output_dimensions.len() != 4 {
            bail!(
                "upsample_nearest2d expects a 4D output, got {}D",
                output_dimensions.len()
            );
        }
        let output_height = output_dimensions[2]
            .to_usize()
            .context("upsample_nearest2d requires a static output height")?;
        let output_width = output_dimensions[3]
            .to_usize()
            .context("upsample_nearest2d requires a static output width")?;

        if input_height == 0 || input_width == 0 || output_height == 0 || output_width == 0 {
            bail!(
                "upsample_nearest2d requires non-zero spatial dims \
                 (in {input_height}x{input_width} -> out {output_height}x{output_width})"
            );
        }

        // `.vec` carries `scales_h`/`scales_w` (either an explicit pair or
        // two scalar optional floats).
        let scales = self.optional_scale_pair(node, 2, 3);
        let result =
            self.upsample_nearest_axis(input, 2, input_height, output_height, scales.map(|s| s.0))?;
        let result =
            self.upsample_nearest_axis(result, 3, input_width, output_width, scales.map(|s| s.1))?;
        Ok(result)
    }

    /// Nearest-neighbour resample of one axis. `out == in` and `out == 2*in`
    /// are ATen kernel fast paths that ignore the scale; integer scales
    /// matching out/in are pure shape movement. Everything else gathers with
    /// `src = min(floor(j * scale_inv), in-1)` where
    /// `scale_inv = 1/s` when scales were provided else `in/out` — the float
    /// chain deliberately mirrors ATen's float32 index math.
    fn upsample_nearest_axis(
        &mut self,
        t: GraphTensor,
        axis: usize,
        in_dim: usize,
        out_dim: usize,
        scale: Option<f64>,
    ) -> Result<GraphTensor> {
        if in_dim != 0 && out_dim.is_multiple_of(in_dim) {
            let k = out_dim / in_dim;
            let scale_matches = scale.is_none_or(|s| (s - k as f64).abs() < 1e-9);
            if k <= 2 || scale_matches {
                if k == 1 {
                    return Ok(t);
                }
                return Ok(t.expand_dim(axis + 1, k).merge_dims(axis, axis + 1));
            }
        }

        let scale_inv = scale.map_or(in_dim as f64 / out_dim as f64, |s| 1.0 / s) as f32;
        let mut out_dims = t.dims();
        out_dims[axis] = IntExpr::from(out_dim);
        let positions = self.axis_positions(&out_dims, axis);
        let index = (positions.cast(DType::F32) * scale_inv)
            .minimum_f32((in_dim - 1) as f32)
            .trunc_cast(DType::Int);
        let mut coords = Vec::with_capacity(t.rank());
        for d in 0..t.rank() {
            if d == axis {
                coords.push(index);
            } else {
                coords.push(self.axis_positions(&out_dims, d));
            }
        }
        Ok(t.gather(&coords))
    }

    /// `upsample_bilinear2d.vec` (non-antialiased) and
    /// `_upsample_bilinear2d_aa.default` (antialiased), selected by
    /// `antialias`.
    pub(super) fn translate_upsample_bilinear2d(
        &mut self,
        node: &Node,
        antialias: bool,
    ) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        if input.rank() != 4 {
            bail!("bilinear2d requires NCHW input");
        }
        let output_shape = self.output_meta_shape(node)?;
        if output_shape.len() != 4 {
            bail!("bilinear2d requires a 4D output");
        }
        let [input_height, input_width, output_height, output_width] =
            Self::static_resize_dimensions(input, &output_shape, "bilinear2d")?;
        let align_corners = self.get_bool_arg(node, 2)?;

        if !antialias {
            let scales = self.optional_scale_pair(node, 3, 4);
            let height = self.bilinear_axis(
                input,
                2,
                input_height,
                output_height,
                align_corners,
                scales.map(|value| value.0),
            );
            return Ok(self.bilinear_axis(
                height,
                3,
                input_width,
                output_width,
                align_corners,
                scales.map(|value| value.1),
            ));
        }

        let scale_height = self.optional_scale_pair(node, 3, 4).map(|s| s.0);
        let scale_width = self.optional_scale_pair(node, 3, 4).map(|s| s.1);
        let quantized_u8 = input.dtype == DType::U8;
        let compute = if quantized_u8 || input.dtype == DType::F64 {
            input
        } else {
            input.cast(DType::F32)
        };
        // The CPU uint8 kernel quantizes after each separable pass, width
        // first.
        let width = self.antialias_bilinear_axis(
            compute,
            3,
            input_width,
            output_width,
            align_corners,
            scale_width,
            quantized_u8,
        );
        let output = self.antialias_bilinear_axis(
            width,
            2,
            input_height,
            output_height,
            align_corners,
            scale_height,
            quantized_u8,
        );
        Ok(if quantized_u8 {
            output
        } else {
            output.cast(input.dtype)
        })
    }

    /// Static `(in_h, in_w, out_h, out_w)` for a 4D resize, or a bail.
    fn static_resize_dimensions(
        input: GraphTensor,
        output_shape: &[IntExpr],
        operation: &str,
    ) -> Result<[usize; 4]> {
        let dimensions = [
            input.dims()[2],
            input.dims()[3],
            output_shape[2],
            output_shape[3],
        ];
        let [
            Some(input_height),
            Some(input_width),
            Some(output_height),
            Some(output_width),
        ] = dimensions.map(|size| size.to_usize())
        else {
            bail!("{operation} dimensions must be static");
        };
        if input_height == 0 || input_width == 0 || output_height == 0 || output_width == 0 {
            bail!("{operation} dimensions must be nonzero");
        }
        Ok([input_height, input_width, output_height, output_width])
    }

    /// Linear interpolation along one spatial axis (`align_corners` and the
    /// optional explicit scale mirror ATen's coordinate transform).
    fn bilinear_axis(
        &mut self,
        input: GraphTensor,
        axis: usize,
        input_size: usize,
        output_size: usize,
        align_corners: bool,
        explicit_scale: Option<f64>,
    ) -> GraphTensor {
        if input_size == output_size {
            return input;
        }
        // Build every coordinate/weight tensor at the full output rank from
        // an axis iota (the same shape-movement form the nearest gather path
        // uses). Doing the float arithmetic on a 1-D arange and expanding
        // afterwards leaves the layout planner with no plan for the expanded
        // arithmetic result.
        let mut out_dims = input.dims();
        out_dims[axis] = IntExpr::from(output_size);
        let positions = self.axis_positions(&out_dims, axis).cast(DType::F32);
        let source = if align_corners {
            let scale = if output_size > 1 {
                (input_size - 1) as f32 / (output_size - 1) as f32
            } else {
                0.0
            };
            positions * scale
        } else {
            let inverse =
                explicit_scale.map_or(input_size as f64 / output_size as f64, |s| 1.0 / s) as f32;
            ((positions + 0.5) * inverse - 0.5).maximum_f32(0.0)
        };
        let lower = source.trunc_cast(DType::Int);
        // Clamp in F32: integer `add`/`maximum` are proof-gated in this
        // runtime (the value-bounds lattice cannot discharge a bounds
        // obligation that flows through a truncating float read), so the
        // whole coordinate chain stays float until its single truncation.
        // `trunc(min(source + 1, in-1)) == min(floor(source) + 1, in-1)` for
        // every `source >= 0`, which both branches guarantee.
        let upper = (source + 1.0)
            .minimum_f32((input_size - 1) as f32)
            .trunc_cast(DType::Int);
        let weight = (source - lower.cast(DType::F32)).cast(input.dtype);
        let left = self.gather_along(input, lower, axis);
        let right = self.gather_along(input, upper, axis);
        left + (right - left) * weight
    }

    /// ATen's antialiased bilinear resize along one axis: an explicit
    /// normalized triangle filter, built as an `[out, in]` weight matrix.
    #[allow(clippy::too_many_arguments)]
    fn antialias_bilinear_axis(
        &mut self,
        input: GraphTensor,
        axis: usize,
        input_size: usize,
        output_size: usize,
        align_corners: bool,
        explicit_scale: Option<f64>,
        quantized_u8: bool,
    ) -> GraphTensor {
        if input_size == output_size {
            return input;
        }
        let scale_value = if align_corners {
            if output_size > 1 {
                (input_size - 1) as f64 / (output_size - 1) as f64
            } else {
                0.0
            }
        } else {
            explicit_scale.map_or(input_size as f64 / output_size as f64, |scale| {
                scale.recip()
            })
        };
        let support_value = scale_value.max(1.0);

        // Both dimensions are static metadata here, and the tensor values
        // never affect extents.
        let weight_dtype = if quantized_u8 {
            DType::F64
        } else {
            input.dtype
        };
        let output_positions = self
            .cx
            .arange(output_size)
            .cast(weight_dtype)
            .expand_dim(1, input_size);
        let input_positions = self
            .cx
            .arange(input_size)
            .cast(weight_dtype)
            .expand_dim(0, output_size);
        let half = self.constant_like(output_positions, 0.5);
        let scale = self.constant_like(output_positions, scale_value);
        let source = (output_positions + half) * scale - half;
        let distance = self.real_abs(input_positions - source);
        let one = self.constant_like(distance, 1.0);
        let support = self.constant_like(distance, support_value);
        let unbounded = one - distance / support;
        let zero = self.constant_like(unbounded, 0.0);
        let positive = unbounded.gt(zero);
        let mut weights = self.select_typed(positive, unbounded, zero);
        let normalization = weights.sum(1).expand_dim(1, input_size);
        weights /= normalization;

        let weights_precision = if quantized_u8 {
            let mut maximum = 0.0_f64;
            for output_index in 0..output_size {
                let center = scale_value * (output_index as f64 + 0.5);
                let row = (0..input_size)
                    .map(|input_index| {
                        (1.0 - ((input_index as f64 + 0.5 - center) / support_value).abs()).max(0.0)
                    })
                    .collect::<Vec<_>>();
                let total = row.iter().sum::<f64>();
                maximum = maximum.max(
                    row.into_iter()
                        .map(|weight| weight / total)
                        .fold(0.0_f64, f64::max),
                );
            }
            let mut precision = 0_u32;
            while precision < 22 {
                let next = (0.5 + maximum * ((1_u64 << (precision + 1)) as f64)) as i64;
                if next >= 1_i64 << 15 {
                    break;
                }
                precision += 1;
            }
            let multiplier = self.constant_like(weights, (1_u64 << precision) as f64);
            let half = self.constant_like(weights, 0.5);
            weights = (weights * multiplier + half).floor().trunc_cast(DType::I64);
            Some(precision)
        } else {
            None
        };

        let mut candidates = if quantized_u8 {
            input.cast(DType::I64).expand_dim(axis, output_size)
        } else {
            input.expand_dim(axis, output_size)
        };
        for dim in 0..axis {
            weights = weights.expand_dim(dim, input.dims()[dim]);
        }
        for dim in axis + 1..input.rank() {
            weights = weights.expand_dim(dim + 1, input.dims()[dim]);
        }
        candidates *= weights;
        let result = candidates.sum(axis + 1);
        if let Some(precision) = weights_precision {
            let result = result.cast(DType::F64);
            let bias = self.constant_like(result, (1_u64 << (precision - 1)) as f64);
            let divisor = self.constant_like(result, (1_u64 << precision) as f64);
            let rounded = ((result + bias) / divisor).floor();
            let zero = self.constant_like(rounded, 0.0);
            let maximum = self.constant_like(rounded, u8::MAX as f64);
            let lower = self.select_typed(rounded.lt(zero), zero, rounded);
            self.select_typed(lower.gt(maximum), maximum, lower)
                .trunc_cast(DType::I64)
                .cast(DType::U8)
        } else {
            result
        }
    }

    // ---------------------------------------------------------------
    // embedding_renorm / slice_scatter
    // ---------------------------------------------------------------

    /// `embedding_renorm.default`: clamp every selected embedding row whose
    /// norm exceeds `max_norm` to that norm (renormalizing in place; SSA
    /// already carries the new value).
    pub(super) fn translate_embedding_renorm(&mut self, node: &Node) -> Result<GraphTensor> {
        let weight = self.operand(&node.inputs[0])?;
        if weight.rank() != 2 {
            bail!("embedding_renorm requires a matrix");
        }
        let indices = self.operand(&node.inputs[1])?.cast(DType::Int).flatten();
        let max_norm = self.get_float_arg(node, 2)?;
        let norm_type = self.get_float_arg(node, 3)?;
        if norm_type <= 0.0 {
            bail!("embedding_renorm requires a positive norm type");
        }

        let rows = weight.dims()[0];
        let columns = weight.dims()[1];
        let row_ids = self.cx.arange(rows).expand_dim(1, indices.dims()[0]);
        let indices = indices.expand_dim(0, rows);
        let selected_count = row_ids.eq(indices).cast(DType::Int).sum(1);
        let zero_count = self.cx.constant_i32(0).expand_rhs(selected_count.dims());
        let selected = selected_count.gt(zero_count);

        let magnitude = self.real_abs(weight);
        let norm = if norm_type == f64::INFINITY {
            magnitude.max(1)
        } else {
            magnitude
                .pow(norm_type as f32)
                .sum(1)
                .pow((1.0 / norm_type) as f32)
        };
        let maximum = self.constant_like(norm, max_norm);
        let exceeds = norm.gt(maximum);
        let apply = self.bool_and(selected, exceeds);
        let scale = maximum / norm;
        let scale = scale.expand_dim(1, columns);
        let apply = apply.expand_dim(1, columns);
        Ok(self.select_typed(apply, weight * scale, weight))
    }

    /// `slice_scatter.default`: copy `source` into `destination[dim,
    /// start:...:step]`. The destination keeps its shape and every
    /// non-slice element.
    pub(super) fn translate_slice_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?.cast(destination.dtype);
        let dim = normalize_dim(self.get_int_arg(node, 2).unwrap_or(0), destination.rank());
        let start = node
            .inputs
            .get(3)
            .and_then(|input| self.resolve_arg_as_expression(&input.arg))
            .unwrap_or_else(|| IntExpr::from(0));
        let start = util::normalize_slice_bound(start, destination.dims()[dim]);
        let step = self.get_int_arg(node, 5).unwrap_or(1);
        if step <= 0 {
            bail!("slice_scatter step must be positive, got {step}");
        }
        if destination.rank() != source.rank() {
            bail!("slice_scatter source and destination ranks must match");
        }

        let destination_dims = destination.dims();
        let strides = row_major_strides(&destination_dims);
        let base = start * strides[dim];
        let step = step as usize;
        let source_dims = source.dims();
        // The destination flat position for each source element: the slice
        // axis advances by `step`, every other axis by one.
        let index = self.cx.iota(source_dims.clone(), |coords| {
            let mut expr = base;
            for (axis, coord) in coords.iter().enumerate() {
                let stride = if axis == dim {
                    strides[axis] * IntExpr::from(step)
                } else {
                    strides[axis]
                };
                expr += *coord * stride;
            }
            expr.simplify()
        });
        let output = destination
            .flatten()
            .scatter(&[index.flatten()], source.flatten());
        Ok(reshape_view(output, &destination_dims))
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    /// Gather `data` along `axis` with `indices` (already shaped to the
    /// output extents) via the recorder's coordinate form.
    fn gather_along(
        &mut self,
        data: GraphTensor,
        indices: GraphTensor,
        axis: usize,
    ) -> GraphTensor {
        let out_dims = indices.dims();
        let mut coords = Vec::with_capacity(data.rank());
        for d in 0..data.rank() {
            if d == axis {
                coords.push(indices);
            } else {
                coords.push(self.axis_positions(&out_dims, d));
            }
        }
        data.gather(&coords)
    }

    /// `where(condition, a, b)` matching `a`'s dtype (the shared
    /// `util::select` blends in F32, which cannot hold an F64 result).
    fn select_typed(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = util::broadcast_binary(a, condition);
        let (a, b) = util::broadcast_binary(a, b);
        let mask = condition.cast(a.dtype);
        let one = self.constant_like(a, 1.0);
        a * mask + b * (one - mask)
    }

    /// The explicit scale pair for an upsample node: a two-element
    /// `as_floats` list at `list_idx`, or the scalar optional floats at
    /// `list_idx`/`second_idx` when PT2 serializes them separately.
    fn optional_scale_pair(
        &self,
        node: &Node,
        list_idx: usize,
        second_idx: usize,
    ) -> Option<(f64, f64)> {
        let input: &NodeInput = node.inputs.get(list_idx)?;
        if let Argument::Other(value) = &input.arg
            && let Some(values) = value.get("as_floats").and_then(|v| v.as_array())
            && values.len() == 2
        {
            return Some((values[0].as_f64()?, values[1].as_f64()?));
        }
        let first = input.arg.as_float()?;
        let second = node
            .inputs
            .get(second_idx)
            .and_then(|input| input.arg.as_float())
            .unwrap_or(first);
        Some((first, second))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides_match_row_major_layout() {
        let dims = vec![
            IntExpr::from(2usize),
            IntExpr::from(3usize),
            IntExpr::from(4usize),
        ];
        let strides = row_major_strides(&dims);
        assert_eq!(strides[0].to_usize(), Some(12));
        assert_eq!(strides[1].to_usize(), Some(4));
        assert_eq!(strides[2].to_usize(), Some(1));
    }
}
