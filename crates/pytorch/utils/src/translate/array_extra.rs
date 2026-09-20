//! Grid sampling, search, triangle indices, slice-scatter, embedding renorm,
//! and the set-grad wrapper (port batch 9).
//!
//! Everything here is expressed with the recorder vocabulary only: coordinate
//! `iota`s, coordinate-form gather/scatter, comparisons, and reductions. Where
//! a float value feeds an integer index the truncating read (`trunc_cast`) is
//! used, and NaN-sensitive selection uses the native `GraphTensor::cond`
//! rather than the arithmetic `util::select` (which leaks NaN through the
//! unselected branch's `NaN * 0`).

use anyhow::{Context, Result, anyhow};
use luminal::prelude::*;

use super::Translator;
use super::util::{ensure_same_dtype, normalize_dim, normalize_slice_bound, same_dim};
use crate::pt2_schema::{Argument, Node};

impl Translator<'_> {
    // -----------------------------------------------------------------
    // grid_sampler_2d / grid_sampler_3d
    // -----------------------------------------------------------------

    /// Clamp `value` into `[minimum, maximum]` while preserving NaN: a NaN
    /// fails both comparisons, so both `cond`s hand it back untouched.
    fn grid_clamp_coordinate(
        &mut self,
        value: GraphTensor,
        minimum: GraphTensor,
        maximum: GraphTensor,
    ) -> GraphTensor {
        let lower = minimum.cond(value.lt(minimum), value);
        maximum.cond(lower.gt(maximum), lower)
    }

    /// Reflect an out-of-range coordinate back into `[0, size-1]` the way
    /// `grid_sampler`'s `padding_mode="reflection"` does.
    fn grid_reflect_coordinate(
        &mut self,
        value: GraphTensor,
        size: IntExpr,
        align_corners: bool,
    ) -> GraphTensor {
        let twice_low = if align_corners { 0.0 } else { -1.0 };
        let minimum = self.constant_like(value, twice_low * 0.5);
        let twice_high = self
            .cx
            .constant_i32(
                size * IntExpr::from(2) - IntExpr::from(if align_corners { 2 } else { 1 }),
            )
            .cast(value.dtype)
            .expand_rhs(value.dims());
        let maximum = twice_high * self.constant_like(value, 0.5);
        let span = maximum - minimum;
        let span_is_zero = self.is_zero(span);
        let one = self.constant_like(span, 1.0);
        let safe_span = one.cond(span_is_zero, span);
        let distance = self.real_abs(value - minimum);
        let quotient = (distance / safe_span).floor();
        let remainder = distance % safe_span;
        let two = self.cx.constant_i64(2).expand_rhs(quotient.dims());
        let zero_i = self.cx.constant_i64(0).expand_rhs(quotient.dims());
        let odd = (quotient.trunc_cast(DType::I64) % two).ne(zero_i);
        let reflected = (safe_span - remainder).cond(odd, remainder) + minimum;
        let zero = self.constant_like(reflected, 0.0);
        zero.cond(span_is_zero, reflected)
    }

    /// Map a normalized `[-1, 1]` grid coordinate to an input-space
    /// coordinate, applying the padding mode. `nan_replacement` mirrors the
    /// old translator's mode-specific NaN handling.
    fn grid_source_coordinate(
        &mut self,
        normalized: GraphTensor,
        size: IntExpr,
        padding_mode: i64,
        align_corners: bool,
        nan_replacement: Option<f64>,
    ) -> GraphTensor {
        let normalized = if let Some(replacement) = nan_replacement {
            let nan = self.is_nan(normalized);
            let replacement = self.constant_like(normalized, replacement);
            replacement.cond(nan, normalized)
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
                self.grid_reflect_coordinate(coordinate, size, align_corners)
            };
            let zero = self.constant_like(padded, 0.0);
            self.grid_clamp_coordinate(padded, zero, size_tensor - one)
        }
    }

    /// Clamp one integer-ish grid coordinate to `[0, size-1]` and report
    /// whether it was in range (padding mode 0 only).
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
                self.grid_reflect_coordinate(index, size, align_corners),
                self.full_tensor(index.dims(), DType::Bool, 1.0),
            ),
            _ => unreachable!(),
        };
        let bounded = self
            .grid_clamp_coordinate(coordinate, zero, upper)
            .trunc_cast(DType::Int);
        (bounded, valid)
    }

    /// Gather one grid point per output position. `indices` holds the
    /// per-spatial-axis coordinates (already interpolated), ordered the same
    /// way as the input's spatial axes.
    fn gather_grid_point(
        &mut self,
        input: GraphTensor,
        indices: &[GraphTensor],
        output_shape: &[IntExpr],
        padding_mode: i64,
        align_corners: bool,
    ) -> GraphTensor {
        let rank = input.rank();
        let spatial_rank = indices.len();
        // Batch and channel coordinates come from iotas; spatial coordinates
        // are the data-dependent indices. Coordinate-form gather keeps all
        // index arithmetic in the recorder's iota form.
        let mut coords = Vec::with_capacity(rank);
        coords.push(self.axis_positions(output_shape, 0));
        coords.push(self.axis_positions(output_shape, 1));
        let mut valid = self.full_tensor(indices[0].dims(), DType::Bool, 1.0);
        let mut spatial_coords = Vec::with_capacity(spatial_rank);
        for (spatial, index) in indices.iter().copied().enumerate() {
            let (bounded, coordinate_valid) = self.bound_grid_index(
                index,
                input.dims()[rank - spatial_rank + spatial],
                padding_mode,
                align_corners,
            );
            valid = self.bool_and(valid, coordinate_valid);
            spatial_coords.push(bounded.expand_dim(1, output_shape[1]));
        }
        coords.extend(spatial_coords);
        let gathered = input.gather(&coords);
        let valid = valid.expand_dim(1, output_shape[1]);
        let zero = self.full_tensor(gathered.dims(), input.dtype, 0.0);
        gathered.cond(valid, zero)
    }

    /// The bicubic interpolation kernel with PyTorch's `alpha = -0.75`.
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
        let outer_or_zero = outer.cond(inside_two, zero);
        inner.cond(inside_one, outer_or_zero)
    }

    pub(super) fn translate_grid_sampler(
        &mut self,
        node: &Node,
        rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        let grid = self.operand(&node.inputs[1])?;
        anyhow::ensure!(
            input.rank() == rank + 2 && grid.rank() == rank + 2,
            "grid_sampler_{rank}d received invalid ranks"
        );
        anyhow::ensure!(
            grid.dims()[rank + 1].to_usize() == Some(rank),
            "grid coordinate dimension must equal the spatial rank"
        );
        let interpolation_mode = self.get_int_arg(node, 2)?;
        let padding_mode = self.get_int_arg(node, 3)?;
        let align_corners = self.get_bool_arg(node, 4)?;
        anyhow::ensure!(
            matches!(interpolation_mode, 0 | 1) || (rank == 2 && interpolation_mode == 2),
            "unsupported grid_sampler interpolation mode {interpolation_mode}"
        );
        anyhow::ensure!(
            matches!(padding_mode, 0..=2),
            "unsupported grid_sampler padding mode {padding_mode}"
        );
        let output_shape = self.output_meta_shape(node)?;
        let mut coordinates = Vec::with_capacity(rank);
        for spatial in 0..rank {
            // The grid's last axis stores (x, y[, z]) — reversed relative to
            // the input's spatial axes.
            let lane = rank - 1 - spatial;
            let normalized = grid.slice_along(lane..lane + 1, rank + 1).squeeze(rank + 1);
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
                } else if rank == 3 && padding_mode != 0 {
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
            for y_offset in -1..=2i64 {
                for x_offset in -1..=2i64 {
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
        for corner in 0..(1usize << rank) {
            let mut corner_indices = Vec::with_capacity(rank);
            let mut weight = self.full_tensor(fractions[0].dims(), grid.dtype, 1.0);
            for spatial in 0..rank {
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

    // -----------------------------------------------------------------
    // searchsorted / bucketize
    // -----------------------------------------------------------------

    pub(super) fn translate_searchsorted(
        &mut self,
        node: &Node,
        scalar: bool,
    ) -> Result<GraphTensor> {
        let sorted = self.operand(&node.inputs[0])?;
        let query = if scalar {
            self.scalar(&node.inputs[1], sorted.dtype)?
        } else {
            self.operand(&node.inputs[1])?
        };
        self.search_lower(node, sorted, query)
    }

    pub(super) fn translate_bucketize(&mut self, node: &Node) -> Result<GraphTensor> {
        // `bucketize(input, boundaries)` is `searchsorted(boundaries, input)`
        // with the operands swapped.
        let query = self.operand(&node.inputs[0])?;
        let sorted = self.operand(&node.inputs[1])?;
        self.search_lower(node, sorted, query)
    }

    /// Lower `searchsorted`/`bucketize` by comparing each query with every
    /// entry of its sorted row and reducing the boolean insertion predicate:
    /// O(N) per query, exact, fixed-shape, and built from broadcast
    /// comparisons and reductions.
    fn search_lower(
        &mut self,
        node: &Node,
        mut sorted: GraphTensor,
        query: GraphTensor,
    ) -> Result<GraphTensor> {
        anyhow::ensure!(
            !sorted.dims().is_empty(),
            "searchsorted requires a sorted sequence with rank at least one"
        );
        let sorted_axis = sorted.rank() - 1;
        // An optional `sorter` permutes the boundaries first; its shape is the
        // gather's output shape.
        if let Some(index) = node
            .inputs
            .iter()
            .position(|input| input.name == "sorter" && input.arg.as_tensor_name().is_some())
        {
            let sorter = self.get_input_tensor(node, index)?.cast(DType::Int);
            anyhow::ensure!(
                sorter.rank() == sorted.rank(),
                "searchsorted sorter rank {} does not match the sequence rank {}",
                sorter.rank(),
                sorted.rank()
            );
            let out_dims = sorter.dims();
            let mut coords = Vec::with_capacity(sorted.rank());
            for axis in 0..sorted.rank() {
                if axis == sorted_axis {
                    coords.push(sorter);
                } else {
                    let position = axis;
                    coords.push(self.cx.iota(out_dims.clone(), move |c| c[position]));
                }
            }
            sorted = sorted.gather(&coords);
        }

        let (sorted, query) = ensure_same_dtype(sorted, query);
        let row_length = sorted.dims()[sorted_axis];
        let query_rank = query.rank();
        let (boundaries, queries) = if sorted.rank() == 1 {
            let mut boundaries = sorted;
            for (axis, size) in query.dims().into_iter().enumerate() {
                boundaries = boundaries.expand_dim(axis, size);
            }
            (boundaries, query.expand_dim(query_rank, row_length))
        } else {
            anyhow::ensure!(
                query_rank == sorted.rank(),
                "batched searchsorted requires query and sequence ranks to match"
            );
            for axis in 0..sorted_axis {
                anyhow::ensure!(
                    same_dim(query.dims()[axis], sorted.dims()[axis]),
                    "batched searchsorted prefix dimensions must match"
                );
            }
            let query_width = query.dims()[sorted_axis];
            (
                sorted.expand_dim(sorted_axis, query_width),
                query.expand_dim(query_rank, row_length),
            )
        };

        let right = self.named_bool_arg(node, "right").unwrap_or(false)
            || node.inputs.iter().any(|input| {
                input.name == "side"
                    && matches!(
                        &input.arg,
                        Argument::Other(value)
                            if value.as_str() == Some("right")
                                || value.get("as_string").and_then(|v| v.as_str()) == Some("right")
                    )
            });
        let before = if right {
            boundaries.le(queries)
        } else {
            boundaries.lt(queries)
        };
        // The insertion count is carried in F32 (exact below 2^24) so the
        // NaN redirect can use the native F32 `cond`; an Int `cond` is refused
        // because it would have to cast the F32 complement back to Int.
        let mut result = before.cast(DType::F32).sum(query_rank);
        if matches!(
            query.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            // NaN sorts past every boundary in both searchsorted and
            // bucketize, so redirect it to the row length.
            let nan = self.is_nan(query);
            let end = self
                .cx
                .constant_i32(row_length)
                .cast(DType::F32)
                .expand_rhs(result.dims());
            result = end.cond(nan, result);
        }
        Ok(result.trunc_cast(self.output_meta_dtype(node)?))
    }

    // -----------------------------------------------------------------
    // tril_indices / triu_indices
    // -----------------------------------------------------------------

    /// Row-major coordinates of the `size` leading true entries of `truth`,
    /// as an I64 `[size, rank]` tensor (the tail is zero-filled). Mirrors the
    /// `nonzero_static` lowering, which needs a data-independent extent.
    fn triangular_coords(&mut self, truth: GraphTensor, size: usize) -> GraphTensor {
        let input_shape = truth.dims();
        let rank = truth.rank();
        let flat_truth = truth.flatten();
        // 1s first and stable: true positions in row-major order.
        let sorted = flat_truth.cast(DType::F32).stable_argsort(0, true);
        let count = flat_truth.cast(DType::F32).sum(0).trunc_cast(DType::Int);
        let numel = input_shape
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim);
        let positions = self.cx.arange(size);
        let last = self
            .cx
            .constant_i32(numel - IntExpr::from(1))
            .cast(DType::F32)
            .expand_rhs(positions.dims());
        let clamped = positions
            .cast(DType::F32)
            .minimum(last)
            .trunc_cast(DType::Int);
        let flat_indices = sorted.gather(&[clamped]);
        let count = count.expand_rhs(positions.dims());
        let numel_const = self.cx.constant_i32(numel).expand_rhs(positions.dims());
        let valid = self.bool_and(positions.lt(count), positions.lt(numel_const));
        let strides: Vec<IntExpr> = (0..rank)
            .map(|axis| {
                input_shape[axis + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, dim| acc * *dim)
            })
            .collect();
        // Per-axis coordinates as flat div/mod in F32 (the plan has no F64
        // binary arms), then one truncation at the end.
        let flat_f = flat_indices.cast(DType::F32);
        let valid_f = valid.cast(DType::F32);
        let one = self.cx.constant_f32(1.0).expand_rhs(valid_f.dims());
        let invalid_f = one - valid_f;
        let fill_f = self.cx.constant_f32(0.0).expand_rhs(valid_f.dims());
        let mut columns = Vec::with_capacity(rank);
        for axis in 0..rank {
            let stride_f = self
                .cx
                .constant_i32(strides[axis])
                .cast(DType::F32)
                .expand_rhs(flat_f.dims());
            let quotient = flat_f / stride_f;
            let dim_f = self
                .cx
                .constant_i32(input_shape[axis])
                .cast(DType::F32)
                .expand_rhs(quotient.dims());
            let coordinate = quotient - (quotient / dim_f).floor() * dim_f;
            columns.push((coordinate * valid_f + fill_f * invalid_f).trunc_cast(DType::I64));
        }
        let rows = self.cx.arange(size);
        let mut result = self
            .cx
            .constant_i64(0)
            .expand_rhs(vec![IntExpr::from(size), IntExpr::from(rank)]);
        for (axis, column) in columns.into_iter().enumerate() {
            let axis_column = self.cx.constant_i32(axis as i64).expand_rhs(rows.dims());
            result = result.scatter(&[rows, axis_column], column);
        }
        result
    }

    pub(super) fn translate_triangular_indices(
        &mut self,
        node: &Node,
        upper: bool,
    ) -> Result<GraphTensor> {
        let rows = self.get_int_arg(node, 0)?;
        let columns = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            rows >= 0 && columns >= 0,
            "triangular index dimensions must be nonnegative"
        );
        let offset = self.get_int_arg(node, 2).unwrap_or(0);
        let rows = IntExpr::from(rows as usize);
        let columns = IntExpr::from(columns as usize);
        let row = self.cx.arange(rows).cast(DType::I64).expand_dim(1, columns);
        let column = self.cx.arange(columns).cast(DType::I64).expand_dim(0, rows);
        let shifted_row = row + IntExpr::from(offset);
        let truth = if upper {
            column.ge(shifted_row)
        } else {
            column.le(shifted_row)
        };
        let output_shape = self.output_meta_shape(node)?;
        anyhow::ensure!(
            output_shape.len() == 2,
            "triangular indices must have rank two"
        );
        let size = output_shape[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("triangular indices need a concrete output width"))?;
        let coordinates = self.triangular_coords(truth, size);
        Ok(coordinates
            .permute(vec![1usize, 0])
            .cast(self.output_meta_dtype(node)?))
    }

    // -----------------------------------------------------------------
    // slice_scatter
    // -----------------------------------------------------------------

    pub(super) fn translate_slice_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?.cast(destination.dtype);
        let rank = destination.rank();
        anyhow::ensure!(rank > 0, "slice_scatter on a rank-0 tensor is not ported");
        anyhow::ensure!(
            source.rank() == rank,
            "slice_scatter source rank {} does not match destination rank {rank}",
            source.rank()
        );
        let raw_dim = self.get_int_arg(node, 2).unwrap_or(0);
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "slice_scatter dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let start = node
            .inputs
            .get(3)
            .and_then(|input| self.resolve_arg_as_expression(&input.arg))
            .unwrap_or_else(|| IntExpr::from(0));
        let start = normalize_slice_bound(start, destination.dims()[dim]);
        let step = self.get_int_arg(node, 5).unwrap_or(1);
        anyhow::ensure!(step > 0, "slice_scatter step must be positive, got {step}");
        // Scatter the source at `start + i * step` along `dim`; every other
        // axis keeps its identity coordinate. Coordinate-form scatter means no
        // flat-index arithmetic enters the graph.
        let src_shape = source.dims();
        let mut coords = Vec::with_capacity(rank);
        for axis in 0..rank {
            if axis == dim {
                coords.push(self.cx.iota(src_shape.clone(), move |c| {
                    c[dim] * IntExpr::from(step) + start
                }));
            } else {
                coords.push(self.cx.iota(src_shape.clone(), move |c| c[axis]));
            }
        }
        Ok(destination.scatter(&coords, source))
    }

    /// `select_scatter(dst, src, dim, index)`: `dst` with the `index`-th
    /// slice along `dim` replaced by `src` (one rank lower). The source's
    /// coordinates map to `dim = index` and the identity elsewhere.
    pub(super) fn translate_select_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?.cast(destination.dtype);
        let rank = destination.rank();
        anyhow::ensure!(rank > 0, "select_scatter on a rank-0 tensor is not ported");
        anyhow::ensure!(
            source.rank() + 1 == rank,
            "select_scatter source rank {} is not one below destination rank {rank}",
            source.rank()
        );
        let raw_dim = self.get_int_arg(node, 2)?;
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "select_scatter dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let index = node
            .inputs
            .get(3)
            .and_then(|input| self.resolve_arg_as_expression(&input.arg))
            .ok_or_else(|| anyhow!("select_scatter has no index"))?;
        let index = normalize_slice_bound(index, destination.dims()[dim]);
        let src_shape = source.dims();
        let mut coords = Vec::with_capacity(rank);
        for axis in 0..rank {
            if axis == dim {
                coords.push(self.cx.iota(src_shape.clone(), move |_| index));
            } else {
                let src_axis = if axis < dim { axis } else { axis - 1 };
                coords.push(self.cx.iota(src_shape.clone(), move |c| c[src_axis]));
            }
        }
        Ok(destination.scatter(&coords, source))
    }

    // -----------------------------------------------------------------
    // embedding_renorm
    // -----------------------------------------------------------------

    pub(super) fn translate_embedding_renorm(&mut self, node: &Node) -> Result<GraphTensor> {
        let weight = self.operand(&node.inputs[0])?;
        anyhow::ensure!(weight.rank() == 2, "embedding_renorm requires a matrix");
        let indices = self.operand(&node.inputs[1])?.cast(DType::Int).flatten();
        let max_norm = self.get_float_arg(node, 2)?;
        let norm_type = self.get_float_arg(node, 3)?;
        anyhow::ensure!(
            norm_type > 0.0,
            "embedding_renorm requires a positive norm type"
        );

        let rows = weight.dims()[0];
        let columns = weight.dims()[1];
        let row_ids = self
            .cx
            .arange(rows)
            .cast(DType::Int)
            .expand_dim(1, indices.dims()[0]);
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
        Ok((weight * scale).cond(apply, weight))
    }

    // -----------------------------------------------------------------
    // higher_order.wrap_with_set_grad_enabled
    // -----------------------------------------------------------------

    /// Translate the inner graph in place and bind its outputs under the
    /// wrapper node's output names. The `set_grad_enabled` flag is a
    /// training-only annotation with no runtime meaning for inference, so the
    /// wrapper is a pure pass-through.
    ///
    /// The embedded subgraph carries its OWN `tensor_values`; PT2 does not
    /// merge them into the top-level map, and `Translator::tensor_meta` reads
    /// that top-level map. There is no `extra_tensor_values` seam on the new
    /// `Translator`, so a merged metadata view is installed on `self.parsed`
    /// for the duration of the subgraph translation. `self.parsed` is a
    /// borrowed `&'a ParsedPT2`, so the merged value is leaked to satisfy the
    /// lifetime (one small program per wrapper node).
    pub(super) fn translate_wrap_set_grad(&mut self, node: &Node) -> Result<()> {
        use crate::pt2_parser::ParsedPT2;
        use crate::pt2_schema::{ExportedProgram, Graph as PtGraph, GraphModule, Signature};

        let subgraph = node.inputs[1]
            .arg
            .as_graph()
            .context("wrap_with_set_grad: missing subgraph")?
            .clone();

        // The wrapper's trailing operands are the subgraph's forwarded
        // inputs, in order.
        for (sg_input, forwarded) in subgraph.graph.inputs.iter().zip(&node.inputs[2..]) {
            if let (Some(sg_name), Some(main_name)) = (
                sg_input
                    .as_tensor
                    .as_ref()
                    .map(|tensor| tensor.name.as_str()),
                forwarded.arg.as_tensor_name(),
            ) {
                let value = *self.values.get(main_name).ok_or_else(|| {
                    anyhow::anyhow!("wrap_with_set_grad: unknown forwarded tensor {main_name:?}")
                })?;
                self.values.insert(sg_name.to_string(), value);
            }
        }

        let main_graph = &self.parsed.program.graph_module.graph;
        let mut tensor_values = main_graph.tensor_values.clone();
        for (name, meta) in &subgraph.graph.tensor_values {
            tensor_values.insert(name.clone(), meta.clone());
        }
        let mut sym_int_values = main_graph.sym_int_values.clone();
        for (name, value) in &subgraph.graph.sym_int_values {
            sym_int_values.insert(name.clone(), value.clone());
        }
        let merged = ParsedPT2 {
            program: ExportedProgram {
                graph_module: GraphModule {
                    graph: PtGraph {
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        nodes: Vec::new(),
                        tensor_values,
                        sym_int_values,
                    },
                    signature: Signature {
                        input_specs: Vec::new(),
                        output_specs: Vec::new(),
                    },
                },
                range_constraints: std::collections::HashMap::new(),
            },
            constants_config: None,
            weights_config: None,
            archive_prefix: String::new(),
            pt2_path: String::new(),
        };
        let leaked: &'static ParsedPT2 = Box::leak(Box::new(merged));
        let previous = std::mem::replace(&mut self.parsed, leaked);
        let mut result = Ok(());
        for (index, sub_node) in subgraph.graph.nodes.iter().enumerate() {
            result = self
                .dispatch(sub_node)
                .with_context(|| format!("Subgraph node {index}: {}", sub_node.target));
            if result.is_err() {
                break;
            }
        }
        self.parsed = previous;
        result?;

        for (main_out, sg_out) in node.outputs.iter().zip(subgraph.graph.outputs.iter()) {
            if let (Some(main_name), Some(sg_name)) =
                (main_out.as_tensor.as_ref(), sg_out.as_tensor.as_ref())
                && main_name.name != sg_name.name
            {
                let value = *self.values.get(&sg_name.name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "wrap_with_set_grad: subgraph output {} was not produced",
                        sg_name.name
                    )
                })?;
                self.values.insert(main_name.name.clone(), value);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, BoolArg, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphArg,
        GraphModule, IntArg, Node, NodeInput, Signature, SubGraph, TensorArg, TensorMeta,
        TensorName, TensorRef,
    };

    fn tensor_ref(name: &str) -> TensorRef {
        TensorRef {
            as_tensor: Some(TensorName {
                name: name.to_string(),
            }),
            as_tensors: None,
            as_sym_int: None,
            as_sym_float: None,
            as_sym_bool: None,
        }
    }

    fn tensor_arg(name: &str) -> Argument {
        Argument::Tensor(TensorArg {
            as_tensor: TensorName {
                name: name.to_string(),
            },
        })
    }

    fn input(name: &str, arg: Argument) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            arg,
            kind: 1,
        }
    }

    fn int(name: &str, value: i64) -> NodeInput {
        input(name, Argument::Int(IntArg { as_int: value }))
    }

    fn float(name: &str, value: f64) -> NodeInput {
        input(name, Argument::Float(FloatArg { as_float: value }))
    }

    fn boolean(name: &str, value: bool) -> NodeInput {
        input(name, Argument::Bool(BoolArg { as_bool: value }))
    }

    fn sizes(values: &[i64]) -> Vec<DimSize> {
        values
            .iter()
            .map(|value| DimSize::Int(DimInt { as_int: *value }))
            .collect()
    }

    fn parsed(
        node: Node,
        metas: &[(&str, u32, &[i64])],
        inputs: &[&str],
        outputs: &[&str],
    ) -> ParsedPT2 {
        let mut tensor_values = HashMap::new();
        for (name, dtype, shape) in metas {
            tensor_values.insert(
                (*name).to_string(),
                TensorMeta {
                    dtype: *dtype,
                    sizes: sizes(shape),
                },
            );
        }
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: Graph {
                    inputs: inputs.iter().map(|name| tensor_ref(name)).collect(),
                    outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
                    nodes: vec![node],
                    tensor_values,
                    sym_int_values: HashMap::new(),
                },
                signature: Signature {
                    input_specs: Vec::new(),
                    output_specs: Vec::new(),
                },
            },
            range_constraints: HashMap::new(),
        };
        ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        }
    }

    fn node(target: &str, inputs: Vec<NodeInput>) -> Node {
        Node {
            target: target.to_string(),
            inputs,
            outputs: vec![tensor_ref("y")],
        }
    }

    fn records(
        node: Node,
        metas: &[(&str, u32, &[i64])],
        inputs: &[&str],
        outputs: &[&str],
    ) -> bool {
        crate::translate::translate(&parsed(node, metas, inputs, outputs)).is_ok()
    }

    #[test]
    fn grid_sampler_2d_modes_record() {
        for (interpolation, padding) in [(0, 0), (1, 2), (2, 2)] {
            assert!(
                records(
                    node(
                        "torch.ops.aten.grid_sampler_2d.default",
                        vec![
                            input("input", tensor_arg("x")),
                            input("grid", tensor_arg("g")),
                            int("interpolation_mode", interpolation),
                            int("padding_mode", padding),
                            boolean("align_corners", false),
                        ],
                    ),
                    &[
                        ("x", 7, &[1, 1, 3, 3]),
                        ("g", 7, &[1, 4, 4, 2]),
                        ("y", 7, &[1, 1, 4, 4]),
                    ],
                    &["x", "g"],
                    &["y"],
                ),
                "grid_sampler_2d mode {interpolation}/{padding}"
            );
        }
    }

    #[test]
    fn grid_sampler_3d_records() {
        assert!(records(
            node(
                "torch.ops.aten.grid_sampler_3d.default",
                vec![
                    input("input", tensor_arg("x")),
                    input("grid", tensor_arg("g")),
                    int("interpolation_mode", 0),
                    int("padding_mode", 1),
                    boolean("align_corners", true),
                ],
            ),
            &[
                ("x", 7, &[1, 1, 2, 2, 2]),
                ("g", 7, &[1, 2, 2, 2, 3]),
                ("y", 7, &[1, 1, 2, 2, 2]),
            ],
            &["x", "g"],
            &["y"],
        ));
    }

    #[test]
    fn searchsorted_tensor_and_scalar_record() {
        assert!(records(
            node(
                "torch.ops.aten.searchsorted.Tensor",
                vec![
                    input("sorted_sequence", tensor_arg("s")),
                    input("values", tensor_arg("v")),
                    boolean("right", false),
                ],
            ),
            &[("s", 7, &[5]), ("v", 7, &[3]), ("y", 5, &[3])],
            &["s", "v"],
            &["y"],
        ));
        assert!(records(
            node(
                "torch.ops.aten.searchsorted.Scalar",
                vec![
                    input("sorted_sequence", tensor_arg("s")),
                    float("values", 2.5),
                ],
            ),
            &[("s", 7, &[5]), ("y", 5, &[])],
            &["s"],
            &["y"],
        ));
    }

    #[test]
    fn bucketize_records() {
        assert!(records(
            node(
                "torch.ops.aten.bucketize.Tensor",
                vec![
                    input("input", tensor_arg("v")),
                    input("boundaries", tensor_arg("b")),
                    boolean("out_int32", false),
                    boolean("right", true),
                ],
            ),
            &[("v", 7, &[3]), ("b", 7, &[5]), ("y", 5, &[3])],
            &["v", "b"],
            &["y"],
        ));
    }

    #[test]
    fn triangular_indices_record() {
        for (target, shape) in [
            ("torch.ops.aten.tril_indices.default", [2, 6]),
            ("torch.ops.aten.triu_indices.default", [2, 6]),
        ] {
            assert!(records(
                node(target, vec![int("row", 3), int("col", 3), int("offset", 0)],),
                &[("y", 5, &shape)],
                &[],
                &["y"],
            ));
        }
    }

    #[test]
    fn slice_scatter_records() {
        assert!(records(
            node(
                "torch.ops.aten.slice_scatter.default",
                vec![
                    input("self", tensor_arg("d")),
                    input("src", tensor_arg("s")),
                    int("dim", 1),
                    int("start", 1),
                    int("end", 3),
                    int("step", 1),
                ],
            ),
            &[("d", 7, &[2, 4]), ("s", 7, &[2, 2]), ("y", 7, &[2, 4])],
            &["d", "s"],
            &["y"],
        ));
    }

    #[test]
    fn embedding_renorm_records() {
        assert!(records(
            node(
                "torch.ops.aten.embedding_renorm.default",
                vec![
                    input("weight", tensor_arg("w")),
                    input("indices", tensor_arg("i")),
                    float("max_norm", 1.0),
                    float("norm_type", 2.0),
                ],
            ),
            &[("w", 7, &[4, 3]), ("i", 5, &[2]), ("y", 7, &[4, 3])],
            &["w", "i"],
            &["y"],
        ));
    }

    #[test]
    fn wrap_with_set_grad_records() {
        let mut inner_values = HashMap::new();
        inner_values.insert(
            "x".to_string(),
            TensorMeta {
                dtype: 7,
                sizes: sizes(&[2, 2]),
            },
        );
        inner_values.insert(
            "inner".to_string(),
            TensorMeta {
                dtype: 7,
                sizes: sizes(&[2, 2]),
            },
        );
        let subgraph = SubGraph {
            graph: Graph {
                inputs: vec![tensor_ref("x")],
                outputs: vec![tensor_ref("inner")],
                nodes: vec![Node {
                    target: "torch.ops.aten.add.Tensor".to_string(),
                    inputs: vec![
                        input("self", tensor_arg("x")),
                        input("other", tensor_arg("x")),
                    ],
                    outputs: vec![tensor_ref("inner")],
                }],
                tensor_values: inner_values,
                sym_int_values: HashMap::new(),
            },
        };
        let wrap_node = Node {
            target: "torch.ops.higher_order.wrap_with_set_grad_enabled".to_string(),
            inputs: vec![
                boolean("enabled", false),
                input("subgraph", Argument::Graph(GraphArg { as_graph: subgraph })),
                input("", tensor_arg("x")),
            ],
            outputs: vec![tensor_ref("y")],
        };
        assert!(records(
            wrap_node,
            &[("x", 7, &[2, 2]), ("y", 7, &[2, 2])],
            &["x"],
            &["y"],
        ));
    }
}
