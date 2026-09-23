//! Additional ATen lowerings ported from the parked translator (port batch 6+).
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::Node;

/// Expand a raw spatial argument (empty / one value / exactly `spatial`)
/// into `spatial` concrete extents.
fn pool_more_expand_spatial(values: &[i64], spatial: usize, default: i64) -> Result<Vec<usize>> {
    let values = if values.is_empty() {
        vec![default; spatial]
    } else if values.len() == 1 {
        vec![values[0]; spatial]
    } else {
        values.to_vec()
    };
    anyhow::ensure!(values.len() == spatial, "expected {spatial} spatial values");
    values
        .into_iter()
        .map(|value| usize::try_from(value).context("spatial values must be nonnegative"))
        .collect()
}

/// Row-major strides as symbolic products: `stride[i] = prod(shape[i+1..])`.
fn pool_more_strides(shape: &[IntExpr]) -> Vec<IntExpr> {
    (0..shape.len())
        .map(|i| {
            shape[i + 1..]
                .iter()
                .fold(IntExpr::from(1), |acc, dim| acc * *dim)
        })
        .collect()
}

/// View-reshape a flat tensor to `target`. The shared `util::reshape_tensor`
/// passes `target[i]` as the inner split size, but `split_dims(axis, inner)`
/// yields `[outer, inner]` with `outer = old / inner`, so the inner size at
/// step `i` must be the product of the remaining target extents (otherwise a
/// size-1 axis can be split by a larger extent and panic). Local copy until
/// the shared helper is corrected.
fn pool_more_reshape(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    let mut flat = t.flatten();
    for i in 0..target.len().saturating_sub(1) {
        let inner = target[i + 1..]
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim)
            .simplify();
        flat = flat.split_dims(i, inner);
    }
    flat
}

impl Translator<'_> {
    /// Read a pooling arg by name, falling back to its positional slot.
    fn pool_more_ints(&self, node: &Node, name: &str, index: usize) -> Result<Vec<i64>> {
        let index = node
            .inputs
            .iter()
            .position(|input| input.name == name)
            .unwrap_or(index);
        self.get_ints_arg(node, index)
    }

    /// Max over the trailing axis, selecting via one stable-argsort pass
    /// (a private copy of `pooling::select_pool_max`, which is module-private).
    fn pool_more_select_max(
        &mut self,
        mut candidates: GraphTensor,
        mut indices: GraphTensor,
        output_rank: usize,
    ) -> Result<[GraphTensor; 2]> {
        while candidates.rank() > output_rank + 1 {
            let last = candidates.rank();
            candidates = candidates.merge_dims(last - 2, last - 1);
            indices = indices.merge_dims(last - 2, last - 1);
        }
        let key = match candidates.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::Bf16 | DType::TF32 => candidates,
            DType::Bool => candidates.cast(DType::F32),
            other => bail!(
                "fractional max pooling on {other:?} inputs is not ported: the \
                 stable-argsort selection needs a float key"
            ),
        };
        let axis = output_rank;
        let selected = key
            .stable_argsort(axis, true)
            .slice_along(0..1, axis)
            .squeeze(axis);
        let out_shape = selected.dims();
        let mut coordinates: Vec<GraphTensor> = (0..axis)
            .map(|index| self.axis_positions(&out_shape, index))
            .collect();
        coordinates.push(selected);
        let values = candidates.gather(&coordinates);
        let picked = indices.gather(&coordinates);
        let picked = if picked.dtype == DType::I64 {
            picked
        } else {
            picked.cast(DType::I64)
        };
        Ok([values, picked])
    }

    /// `aten.fractional_max_pool{2,3}d.default`: values + indices.
    pub(super) fn translate_fractional_max_pool(&mut self, node: &Node, dims: usize) -> Result<()> {
        let input = self.get_input_tensor(node, 0)?;
        let random_samples = self.get_input_tensor(node, 3)?;
        let rank = input.rank();
        anyhow::ensure!(
            rank == dims + 1 || rank == dims + 2,
            "fractional max pool input rank is invalid"
        );
        anyhow::ensure!(
            random_samples.rank() == 3 && random_samples.dims()[2].to_usize() == Some(dims),
            "fractional max pool random_samples must have shape [N, C, spatial_rank]"
        );
        let kernel =
            pool_more_expand_spatial(&self.pool_more_ints(node, "kernel_size", 1)?, dims, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&size| size > 0),
            "fractional max pool kernel must be positive"
        );
        let output_shape = self.output_meta_shape(node)?;
        let prefix_rank = rank - dims;
        let input_shape = input.dims();
        let output_spatial = output_shape[prefix_rank..].to_vec();
        let input_spatial = input_shape[prefix_rank..].to_vec();
        let mut window_shape = output_shape.clone();
        window_shape.extend(kernel.iter().copied().map(IntExpr::from));

        let input_strides = pool_more_strides(&input_shape);
        let mut flat_indices = self.axis_positions(&window_shape, 0) * input_strides[0];
        for (prefix, stride) in input_strides
            .iter()
            .copied()
            .enumerate()
            .take(prefix_rank)
            .skip(1)
        {
            flat_indices += self.axis_positions(&window_shape, prefix) * stride;
        }
        let mut logical_spatial_indices = self.cx.constant_i32(0).expand_rhs(window_shape.clone());

        for spatial in 0..dims {
            // The per-sample random lane: 2-D kernels are stored transposed
            // relative to 3-D ones (matching ATen's random_samples layout).
            let sample_lane = if dims == 2 {
                dims - 1 - spatial
            } else {
                spatial
            };
            let mut sample = random_samples
                .slice_along(sample_lane..sample_lane + 1, 2)
                .squeeze(2);
            if prefix_rank == 1 {
                sample = sample.squeeze(0);
            }
            for output_size in output_spatial.iter().copied() {
                sample = sample.expand_dim(sample.rank(), output_size);
            }
            let output_axis = prefix_rank + spatial;
            let output_position = self.axis_positions(&output_shape, output_axis);
            let numerator = self
                .cx
                .constant_i32(input_spatial[spatial] - IntExpr::from(kernel[spatial]))
                .cast(sample.dtype)
                .expand_rhs(sample.dims());
            let denominator = self
                .cx
                .constant_i32(output_spatial[spatial] - IntExpr::from(1usize))
                .cast(sample.dtype)
                .expand_rhs(sample.dims());
            let output_is_one = self.is_zero(denominator);
            let one = self.constant_like(denominator, 1.0);
            let safe_denominator = self.select(output_is_one, one, denominator);
            let alpha = numerator / safe_denominator;
            // start = floor((out_pos + sample) * alpha) - floor(sample * alpha),
            // clamped to the terminal window / the single-output case.
            let ordinary_start = ((output_position.cast(sample.dtype) + sample) * alpha).floor()
                - (sample * alpha).floor();
            let terminal_position = self
                .cx
                .constant_i32(output_spatial[spatial] - IntExpr::from(1usize))
                .cast(DType::Int)
                .expand_rhs(output_position.dims());
            let final_position = output_position.eq(terminal_position);
            let terminal_start = self
                .cx
                .constant_i32(input_spatial[spatial] - IntExpr::from(kernel[spatial]))
                .cast(sample.dtype)
                .expand_rhs(ordinary_start.dims());
            let final_or_ordinary = self.select(final_position, terminal_start, ordinary_start);
            let start = self.select(output_is_one, terminal_start, final_or_ordinary);
            let mut coordinate = start;
            for size in kernel.iter().copied() {
                coordinate = coordinate.expand_dim(coordinate.rank(), size);
            }
            let kernel_axis = rank + spatial;
            coordinate = coordinate
                + self
                    .axis_positions(&window_shape, kernel_axis)
                    .cast(coordinate.dtype);
            // `coordinate` is float (start + a float-cast kernel position);
            // the recorder refuses implicit float -> int casts.
            let coordinate = coordinate.trunc_cast(DType::Int);
            flat_indices += coordinate * input_strides[prefix_rank + spatial];
            let spatial_stride = input_spatial[spatial + 1..]
                .iter()
                .fold(IntExpr::from(1), |acc, dim| acc * *dim);
            logical_spatial_indices += coordinate * spatial_stride;
        }

        let candidates = pool_more_reshape(
            input.flatten().gather(&[flat_indices.flatten()]),
            &window_shape,
        );
        let [values, indices] =
            self.pool_more_select_max(candidates, logical_spatial_indices, rank)?;
        self.bind_outputs(node, vec![values, indices])
    }

    /// `aten.max_pool2d_with_indices_backward.default`: scatter-add the
    /// `updates` into a zero tensor at the max-pool index positions.
    pub(super) fn translate_max_pool2d_with_indices_backward(
        &mut self,
        node: &Node,
    ) -> Result<GraphTensor> {
        let updates = self.get_input_tensor(node, 0)?;
        let input = self.get_input_tensor(node, 1)?;
        let indices = self.get_input_tensor(node, 7)?.cast(DType::Int);
        let input_shape = input.dims();
        let prefix_rank = input_shape.len().saturating_sub(2);
        let strides = pool_more_strides(&input_shape);
        let output_shape = updates.dims();

        // Base flat destination into the flattened input contributed by the
        // batch/channel prefix axes; the spatial axes contribute through the
        // recorded `indices` value below.
        let mut base: Option<GraphTensor> = None;
        for (axis, stride) in strides
            .iter()
            .copied()
            .enumerate()
            .take(prefix_rank.min(output_shape.len()))
        {
            let contribution = self.axis_positions(&output_shape, axis) * stride;
            base = Some(match base {
                Some(acc) => acc + contribution,
                None => contribution,
            });
        }
        let base = match base {
            Some(base) => base,
            None => self.cx.constant_i32(0).expand_rhs(output_shape.clone()),
        };
        let destinations = (base + indices).flatten();

        let flat_updates = updates.flatten();
        let count = updates
            .dims()
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(dim.to_usize()?))
            .context("max_pool2d_with_indices_backward requires a concrete update element count")?;
        let mut output = self
            .full_tensor(input_shape.clone(), input.dtype, 0.0)
            .flatten();
        // The recorder's scatter is overwrite-only, so additive accumulation
        // is one static read/modify/write step per update element.
        for step in 0..count {
            let destination = destinations.slice_along(step..step + 1, 0);
            let update = flat_updates.slice_along(step..step + 1, 0);
            let current = output.gather(&[destination]);
            output = output.scatter(&[destination], current + update);
        }
        Ok(pool_more_reshape(output, &input_shape))
    }
}
