use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::Node;
use crate::pt2_util::reshape_tensor;

use super::Translator;

impl<'a> Translator<'a> {
    fn clamp_coordinate(
        &mut self,
        value: GraphTensor,
        minimum: GraphTensor,
        maximum: GraphTensor,
    ) -> GraphTensor {
        let lower = self.select(value.lt(minimum), minimum, value);
        self.select(lower.gt(maximum), maximum, lower)
    }

    fn reflect_coordinate(
        &mut self,
        value: GraphTensor,
        size: Expression,
        align_corners: bool,
    ) -> GraphTensor {
        let twice_low = if align_corners { 0.0 } else { -1.0 };
        let minimum = self.constant_like(value, twice_low * 0.5);
        let twice_high = self
            .graph
            .constant(size * 2 - if align_corners { 2 } else { 1 })
            .cast(value.dtype)
            .expand_rhs(value.shape);
        let maximum = twice_high * self.constant_like(value, 0.5);
        let span = maximum - minimum;
        let span_is_zero = self.is_zero(span);
        let one = self.constant_like(span, 1.0);
        let safe_span = self.select(span_is_zero, one, span);
        let distance = self.real_abs(value - minimum);
        let quotient = self.floor_tensor(distance / safe_span);
        let remainder = distance % safe_span;
        let odd = (quotient.cast(DType::I64)
            % self
                .graph
                .constant(2)
                .cast(DType::I64)
                .expand_rhs(quotient.shape))
        .ne(self
            .graph
            .constant(0)
            .cast(DType::I64)
            .expand_rhs(quotient.shape));
        let reflected = self.select(odd, safe_span - remainder, remainder) + minimum;
        let zero = self.constant_like(reflected, 0.0);
        self.select(span_is_zero, zero, reflected)
    }

    fn grid_source_coordinate(
        &mut self,
        normalized: GraphTensor,
        size: Expression,
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
            self.select(nan, replacement, normalized)
        } else {
            normalized
        };
        let size_tensor = self
            .graph
            .constant(size)
            .cast(normalized.dtype)
            .expand_rhs(normalized.shape);
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
        size: Expression,
        padding_mode: i64,
        align_corners: bool,
    ) -> (GraphTensor, GraphTensor) {
        let zero = self.constant_like(index, 0.0);
        let size_tensor = self
            .graph
            .constant(size)
            .cast(index.dtype)
            .expand_rhs(index.shape);
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
                .cast(DType::Int),
            valid,
        )
    }

    fn gather_grid_point(
        &mut self,
        input: GraphTensor,
        indices: &[GraphTensor],
        output_shape: &[Expression],
        padding_mode: i64,
        align_corners: bool,
    ) -> GraphTensor {
        let spatial_rank = indices.len();
        let rank = input.shape.len();
        let strides = super::movement_dynamic::row_major_strides(&input.dims());
        let mut flat = self.axis_positions(output_shape, 0) * strides[0]
            + self.axis_positions(output_shape, 1) * strides[1];
        let mut valid = self
            .graph
            .constant(1)
            .cast(DType::Bool)
            .expand_rhs(indices[0].shape);
        for (spatial, index) in indices.iter().copied().enumerate() {
            let (bounded, coordinate_valid) = self.bound_grid_index(
                index,
                input.dims()[rank - spatial_rank + spatial],
                padding_mode,
                align_corners,
            );
            valid = self.bool_and(valid, coordinate_valid);
            flat += bounded.expand_dim(1, output_shape[1]) * strides[rank - spatial_rank + spatial];
        }
        let gathered = reshape_tensor(
            input.flatten().gather(flat.flatten()),
            output_shape.to_vec(),
        );
        let valid = valid.expand_dim(1, output_shape[1]);
        let zero = self.full_tensor(gathered.dims(), input.dtype, 0.0);
        self.select(valid, gathered, zero)
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
        let outer_or_zero = self.select(inside_two, outer, zero);
        self.select(inside_one, inner, outer_or_zero)
    }

    pub(crate) fn translate_grid_sampler(
        &mut self,
        node: &Node,
        spatial_rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let grid = self.get_input_tensor(node, 1)?;
        anyhow::ensure!(
            input.shape.len() == spatial_rank + 2 && grid.shape.len() == spatial_rank + 2,
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
                .map(|coordinate| self.round_to_even(coordinate))
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
                .map(|coordinate| self.floor_tensor(coordinate))
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
            .map(|coordinate| self.floor_tensor(coordinate))
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
