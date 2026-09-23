//! Additional ATen lowerings ported from the parked translator (port batch 6+).
#![allow(dead_code)]

use anyhow::Result;
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::Node;

/// View-reshape a flat tensor to `target`, right-splitting the remaining
/// extent at each step. The shared `util::reshape_tensor` passes `target[i]`
/// as the inner split size, but `GraphTensor::split_dims(axis, inner)`
/// produces `[outer, inner]` (`outer = old / inner`), so the inner size at
/// step `i` must be the product of the *remaining* target extents. With the
/// shared helper, `[12] -> [1,1,3,4]` splits `[12]` by `1` (giving `[12,1]`)
/// and then tries to split a size-1 axis by `3`, panicking. Local copy until
/// the shared helper is corrected.
fn grid_reshape(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
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
    /// IEEE-safe `where(mask, a, b)` through a packed gather, matching the
    /// parked translator's `select`. Arithmetic masking (`a*m + b*(1-m)`)
    /// corrupts the NaN/Inf cases grid sampling must handle: `NaN * 0` and
    /// `inf * 0` are NaN, so a masked-out NaN branch would poison the result,
    /// neutralizing the NaN-replacement logic below. The packed layout puts
    /// `b` at flat `2k` and `a` at flat `2k+1`; the boolean mask is used only
    /// as a gather coordinate, never an arithmetic operand.
    fn grid_select(&mut self, mask: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = util::broadcast_binary(a, b);
        let (a, mask) = util::broadcast_binary(a, mask);
        let shape = a.dims();
        let rank = shape.len();
        let strides: Vec<IntExpr> = (0..rank)
            .map(|d| {
                shape[d + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, size| acc * *size)
            })
            .collect();
        let flat = |c: &[IntExpr]| -> IntExpr {
            (0..rank).fold(IntExpr::from(0), |acc, d| acc + c[d] * strides[d])
        };
        let even = self.cx.iota(shape.clone(), |c| flat(c) * IntExpr::from(2));
        let odd = self.cx.iota(shape.clone(), |c| {
            flat(c) * IntExpr::from(2) + IntExpr::from(1)
        });
        let numel = shape.iter().fold(IntExpr::from(1), |acc, size| acc * *size);
        let zeros = self.full_tensor(vec![numel, IntExpr::from(2)], a.dtype, 0.0);
        let packed = b.scatter1d(even, zeros);
        let packed = a.scatter1d(odd, packed);
        let flat_coord = self.cx.iota(shape, |c| flat(c));
        packed.gather(&[flat_coord, mask.cast(DType::Int)])
    }

    fn clamp_coordinate(
        &mut self,
        value: GraphTensor,
        minimum: GraphTensor,
        maximum: GraphTensor,
    ) -> GraphTensor {
        let lower = self.grid_select(value.lt(minimum), minimum, value);
        self.grid_select(lower.gt(maximum), maximum, lower)
    }

    fn reflect_coordinate(
        &mut self,
        value: GraphTensor,
        size: IntExpr,
        align_corners: bool,
    ) -> GraphTensor {
        let twice_low = if align_corners { 0.0 } else { -1.0 };
        let minimum = self.constant_like(value, twice_low * 0.5);
        let twice_high = size * 2 - if align_corners { 2 } else { 1 };
        let twice_high = self
            .cx
            .constant_i32(twice_high)
            .cast(value.dtype)
            .expand_rhs(value.dims());
        let maximum = twice_high * self.constant_like(value, 0.5);
        let span = maximum - minimum;
        let span_is_zero = self.is_zero(span);
        let one = self.constant_like(span, 1.0);
        let safe_span = self.grid_select(span_is_zero, one, span);
        let distance = self.real_abs(value - minimum);
        let quotient = (distance / safe_span).floor();
        let remainder = distance % safe_span;
        let odd = (quotient.trunc_cast(DType::I64)
            % self
                .cx
                .constant_i32(2)
                .cast(DType::I64)
                .expand_rhs(quotient.dims()))
        .ne(self
            .cx
            .constant_i32(0)
            .cast(DType::I64)
            .expand_rhs(quotient.dims()));
        let reflected = self.grid_select(odd, safe_span - remainder, remainder) + minimum;
        let zero = self.constant_like(reflected, 0.0);
        self.grid_select(span_is_zero, zero, reflected)
    }

    fn grid_source_coordinate(
        &mut self,
        normalized: GraphTensor,
        size: IntExpr,
        padding_mode: i64,
        align_corners: bool,
        nan_replacement: Option<f64>,
    ) -> GraphTensor {
        // Nearest sampling converts NaN through the integer-coordinate path
        // (equivalent to normalized -1). The CPU 3-D border/reflection path
        // clips NaN to the upper boundary; other linear/cubic paths propagate
        // it through the interpolation weights.
        let normalized = if let Some(replacement) = nan_replacement {
            let nan = self.is_nan(normalized);
            let replacement = self.constant_like(normalized, replacement);
            self.grid_select(nan, replacement, normalized)
        } else {
            normalized
        };
        let size_tensor = self
            .cx
            .constant_i32(size)
            .cast(normalized.dtype)
            .expand_rhs(normalized.dims());
        let one = self.constant_like(normalized, 1.0);
        let two = self.constant_like(normalized, 2.0);
        let coordinate = if align_corners {
            (normalized + one) * (size_tensor - one) / two
        } else {
            ((normalized + one) * size_tensor - one) / two
        };
        if padding_mode == 0 {
            coordinate
        } else {
            let padded = if padding_mode == 1 {
                coordinate
            } else {
                self.reflect_coordinate(coordinate, size, align_corners)
            };
            let zero = self.constant_like(padded, 0.0);
            self.clamp_coordinate(padded, zero, size_tensor - one)
        }
    }

    fn bound_grid_index(
        &mut self,
        index: GraphTensor,
        size: IntExpr,
        padding_mode: i64,
        align_corners: bool,
    ) -> (GraphTensor, GraphTensor) {
        let zero = self.constant_like(index, 0.0);
        let size_tensor = self
            .cx
            .constant_i32(size)
            .cast(index.dtype)
            .expand_rhs(index.dims());
        let upper = size_tensor - self.constant_like(index, 1.0);
        let (coordinate, valid) = match padding_mode {
            0 => (index, self.bool_and(index.ge(zero), index.lt(size_tensor))),
            1 => (index, self.full_tensor(index.dims(), DType::Bool, 1.0)),
            2 => (
                self.reflect_coordinate(index, size, align_corners),
                self.full_tensor(index.dims(), DType::Bool, 1.0),
            ),
            _ => unreachable!(),
        };
        (
            self.clamp_coordinate(coordinate, zero, upper)
                .trunc_cast(DType::Int),
            valid,
        )
    }

    fn gather_grid_point(
        &mut self,
        input: GraphTensor,
        indices: &[GraphTensor],
        output_shape: &[IntExpr],
        padding_mode: i64,
        align_corners: bool,
    ) -> GraphTensor {
        let spatial_rank = indices.len();
        let rank = input.rank();
        let shape = input.dims();
        let strides: Vec<IntExpr> = (0..rank)
            .map(|i| {
                shape[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, d| acc * *d)
            })
            .collect();
        let mut flat = self.axis_positions(output_shape, 0) * strides[0]
            + self.axis_positions(output_shape, 1) * strides[1];
        let mut valid = self.full_tensor(indices[0].dims(), DType::Bool, 1.0);
        for (spatial, index) in indices.iter().copied().enumerate() {
            let (bounded, coordinate_valid) = self.bound_grid_index(
                index,
                shape[rank - spatial_rank + spatial],
                padding_mode,
                align_corners,
            );
            valid = self.bool_and(valid, coordinate_valid);
            flat += bounded.expand_dim(1, output_shape[1]) * strides[rank - spatial_rank + spatial];
        }
        let gathered = grid_reshape(input.flatten().gather(&[flat.flatten()]), output_shape);
        let valid = valid.expand_dim(1, output_shape[1]);
        let zero = self.full_tensor(gathered.dims(), input.dtype, 0.0);
        self.grid_select(valid, gathered, zero)
    }

    fn cubic_coefficient(&mut self, distance: GraphTensor) -> GraphTensor {
        let absolute = self.real_abs(distance);
        let one = self.constant_like(absolute, 1.0);
        let two = self.constant_like(absolute, 2.0);
        let alpha = self.constant_like(absolute, -0.75);
        let inner = ((alpha + two) * absolute - (alpha + self.constant_like(absolute, 3.0)))
            * absolute
            * absolute
            + one;
        let outer = ((alpha * absolute - alpha * self.constant_like(absolute, 5.0)) * absolute
            + alpha * self.constant_like(absolute, 8.0))
            * absolute
            - alpha * self.constant_like(absolute, 4.0);
        let inside_one = absolute.le(one);
        let inside_two = absolute.lt(two);
        let zero = self.constant_like(absolute, 0.0);
        let outer_or_zero = self.grid_select(inside_two, outer, zero);
        self.grid_select(inside_one, inner, outer_or_zero)
    }

    pub(super) fn translate_grid_sampler(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let grid = self.get_input_tensor(node, 1)?;
        anyhow::ensure!(
            input.rank() == spatial_rank + 2 && grid.rank() == spatial_rank + 2,
            "grid_sampler_{spatial_rank}d received invalid ranks"
        );
        anyhow::ensure!(
            grid.dims()[spatial_rank + 1].to_usize() == Some(spatial_rank),
            "grid coordinate dimension must equal the spatial rank"
        );
        let interpolation_mode = self.get_int_arg(node, 2)?;
        let padding_mode = self.get_int_arg(node, 3)?;
        let align_corners = self.get_bool_arg(node, 4)?;
        anyhow::ensure!(
            matches!(interpolation_mode, 0 | 1) || (spatial_rank == 2 && interpolation_mode == 2),
            "unsupported grid_sampler interpolation mode {interpolation_mode}"
        );
        anyhow::ensure!(
            matches!(padding_mode, 0..=2),
            "unsupported grid_sampler padding mode {padding_mode}"
        );
        let output_shape = self.output_meta_shape(node)?;
        let mut coordinates = Vec::with_capacity(spatial_rank);
        for spatial in 0..spatial_rank {
            let lane = spatial_rank - 1 - spatial;
            let normalized = grid
                .slice_along(lane..lane + 1, spatial_rank + 1)
                .squeeze(spatial_rank + 1);
            coordinates.push(self.grid_source_coordinate(
                normalized,
                input.dims()[2 + spatial],
                if interpolation_mode == 2 {
                    0
                } else {
                    padding_mode
                },
                align_corners,
                if interpolation_mode == 1 {
                    Some(-1.0)
                } else if spatial_rank == 3 && padding_mode != 0 {
                    Some(1.0)
                } else {
                    None
                },
            ));
        }

        if interpolation_mode == 1 {
            let rounded = coordinates
                .into_iter()
                .map(|coordinate| coordinate.round())
                .collect::<Vec<_>>();
            return Ok(self.gather_grid_point(
                input,
                &rounded,
                &output_shape,
                padding_mode,
                align_corners,
            ));
        }

        if interpolation_mode == 2 {
            let bases = coordinates
                .iter()
                .copied()
                .map(|coordinate| coordinate.floor())
                .collect::<Vec<_>>();
            let mut result = self.full_tensor(output_shape.clone(), input.dtype, 0.0);
            for y_offset in -1..=2 {
                for x_offset in -1..=2 {
                    let y_index = bases[0] + y_offset as f32;
                    let x_index = bases[1] + x_offset as f32;
                    let y_weight = self.cubic_coefficient(coordinates[0] - y_index);
                    let x_weight = self.cubic_coefficient(coordinates[1] - x_index);
                    let value = self.gather_grid_point(
                        input,
                        &[y_index, x_index],
                        &output_shape,
                        padding_mode,
                        align_corners,
                    );
                    result += value * (y_weight * x_weight).expand_dim(1, output_shape[1]);
                }
            }
            return Ok(result);
        }

        let lower = coordinates
            .iter()
            .copied()
            .map(|coordinate| coordinate.floor())
            .collect::<Vec<_>>();
        let fractions = coordinates
            .iter()
            .zip(lower.iter())
            .map(|(&coordinate, &base)| coordinate - base)
            .collect::<Vec<_>>();
        let mut result = self.full_tensor(output_shape.clone(), input.dtype, 0.0);
        for corner in 0..(1usize << spatial_rank) {
            let mut corner_indices = Vec::with_capacity(spatial_rank);
            let mut weight = self.full_tensor(fractions[0].dims(), grid.dtype, 1.0);
            for spatial in 0..spatial_rank {
                if corner & (1 << spatial) == 0 {
                    corner_indices.push(lower[spatial]);
                    weight *= self.constant_like(weight, 1.0) - fractions[spatial];
                } else {
                    corner_indices.push(lower[spatial] + 1.0);
                    weight *= fractions[spatial];
                }
            }
            let value = self.gather_grid_point(
                input,
                &corner_indices,
                &output_shape,
                padding_mode,
                align_corners,
            );
            result += value * weight.cast(input.dtype).expand_dim(1, output_shape[1]);
        }
        Ok(result)
    }
}
