use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::dim_arith::product_of_dims;
use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Whether `argmax` / `argmin` should pick the largest (descending sort) or
/// smallest (ascending sort) element when scanning the input.
#[derive(Clone, Copy)]
pub(crate) enum ArgExtremum {
    Max,
    Min,
}

impl ArgExtremum {
    fn descending(self) -> bool {
        matches!(self, ArgExtremum::Max)
    }
}

/// Whether a cumulative extremum tracks the running maximum or minimum.
#[derive(Clone, Copy)]
pub(crate) enum CumExtremum {
    Max,
    Min,
}

fn cumulative_axis(dim: i64, rank: usize) -> Result<Option<usize>> {
    if rank == 0 {
        anyhow::ensure!(
            matches!(dim, -1 | 0),
            "Dimension out of range for scalar cumulative op: {dim}"
        );
        return Ok(None);
    }

    let normalized = if dim < 0 { rank as i64 + dim } else { dim };
    anyhow::ensure!(
        (0..rank as i64).contains(&normalized),
        "Dimension out of range for rank-{rank} cumulative op: {dim}"
    );
    Ok(Some(normalized as usize))
}

fn dtype_can_contain_nan(dtype: DType) -> bool {
    !matches!(
        dtype,
        DType::Int
            | DType::I64
            | DType::I4
            | DType::U4
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::Bool
    )
}

fn is_integral_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int
            | DType::I64
            | DType::I4
            | DType::U4
            | DType::I8
            | DType::U8
            | DType::I16
            | DType::U16
            | DType::Bool
    )
}

/// Compute total element count, returning an error if any dimension is symbolic.
fn concrete_numel(a: &GraphTensor) -> Result<usize> {
    a.dims().iter().try_fold(1usize, |acc, d| {
        d.to_usize().map(|v| acc * v).ok_or_else(|| {
            anyhow::anyhow!("Full reduction requires concrete dimensions, got symbolic dim")
        })
    })
}

impl<'a> Translator<'a> {
    fn optional_float_list(node: &Node, name: &str) -> Option<Vec<f64>> {
        let input = node.inputs.iter().find(|input| input.name == name)?;
        match &input.arg {
            Argument::Other(value) => value
                .get("as_floats")?
                .as_array()?
                .iter()
                .map(serde_json::Value::as_f64)
                .collect(),
            _ => None,
        }
    }

    fn histogram_weight(
        &mut self,
        node: &Node,
        sample_count: Expression,
        dtype: DType,
    ) -> Result<GraphTensor> {
        if let Some(weight) = self.named_tensor_arg(node, "weight")? {
            return Ok(reshape_tensor(weight.cast(dtype), vec![sample_count]));
        }
        Ok(self
            .floating_scalar(1.0, dtype)
            .expand_rhs(vec![sample_count]))
    }

    fn histogram_density(node: &Node) -> bool {
        node.inputs
            .iter()
            .find(|input| input.name == "density")
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false)
    }

    fn uniform_histogram_edges(
        &mut self,
        values: GraphTensor,
        bins: usize,
        explicit_range: Option<(f64, f64)>,
    ) -> GraphTensor {
        let (mut lower, mut upper) = if let Some((lower, upper)) = explicit_range {
            (
                self.floating_scalar(lower, values.dtype),
                self.floating_scalar(upper, values.dtype),
            )
        } else {
            (values.min(0), values.max(0))
        };
        let equal = self.is_zero(upper - lower);
        lower = self.select(equal, lower - 0.5, lower);
        upper = self.select(equal, upper + 0.5, upper);
        let position = self.graph.arange(bins + 1).cast(values.dtype);
        let edge_shape = position.dims();
        let lower_expanded = lower.expand_rhs(edge_shape.clone());
        let width = (upper - lower).expand_rhs(edge_shape);
        lower_expanded + position * width / bins as f32
    }

    fn histogram_from_edges(
        &mut self,
        columns: &[GraphTensor],
        edges: &[GraphTensor],
        weight: GraphTensor,
        density: bool,
    ) -> Result<GraphTensor> {
        anyhow::ensure!(
            !columns.is_empty(),
            "histogram must have at least one dimension"
        );
        anyhow::ensure!(columns.len() == edges.len(), "histogram edge rank mismatch");
        let sample_count = columns[0].dims()[0];
        let mut bin_shape = Vec::with_capacity(edges.len());
        for edge in edges {
            let edge_count = edge.dims()[0];
            anyhow::ensure!(
                edge_count.to_usize().is_none_or(|count| count >= 2),
                "histogram bin edges must contain at least two values"
            );
            bin_shape.push(edge_count - 1);
        }

        let mut full_shape = vec![sample_count];
        full_shape.extend_from_slice(&bin_shape);
        let mut membership = self
            .graph
            .constant(1)
            .cast(DType::Bool)
            .expand_rhs(full_shape.clone());
        for (dimension, (column, edge)) in columns.iter().zip(edges).enumerate() {
            let bins = bin_shape[dimension];
            let left = edge.slice_along(Expression::from(0)..bins, 0);
            let right = edge.slice_along(Expression::from(1)..(bins + 1), 0);

            let mut column_shape = vec![Expression::from(1); columns.len() + 1];
            column_shape[0] = sample_count;
            let mut column = reshape_tensor(*column, column_shape);
            column.shape.expand(full_shape.clone());
            let mut edge_shape = vec![Expression::from(1); columns.len() + 1];
            edge_shape[dimension + 1] = bins;
            let mut left = reshape_tensor(left, edge_shape.clone());
            left.shape.expand(full_shape.clone());
            let mut right = reshape_tensor(right, edge_shape.clone());
            right.shape.expand(full_shape.clone());

            let below_right = column.lt(right);
            let equal_right = self.is_zero(column - right);
            let mut last_position = self.graph.arange(bins).cast(DType::Int);
            let last_index = self
                .graph
                .constant(bins - 1)
                .cast(DType::Int)
                .expand_rhs(last_position.dims());
            last_position = last_position.eq(last_index);
            let mut last_position = reshape_tensor(last_position, edge_shape);
            last_position.shape.expand(full_shape.clone());
            let inclusive_last = self.bool_and(equal_right, last_position);
            let upper = self.bool_or(below_right, inclusive_last);
            let in_bin = self.bool_and(column.ge(left), upper);
            let column_nan = self.is_nan(column);
            let in_bin = self.bool_and(in_bin, self.bool_not(column_nan));
            membership = self.bool_and(membership, in_bin);
        }

        let mut weight_shape = vec![Expression::from(1); columns.len() + 1];
        weight_shape[0] = sample_count;
        let mut weight = reshape_tensor(weight, weight_shape);
        weight.shape.expand(full_shape);
        let mut histogram = (membership.cast(weight.dtype) * weight).sum(0);
        if density {
            let axes = (0..bin_shape.len()).collect::<Vec<_>>();
            let total = histogram.sum(axes).expand_rhs(bin_shape.clone());
            let mut volume = self
                .floating_scalar(1.0, histogram.dtype)
                .expand_rhs(bin_shape.clone());
            for (dimension, edge) in edges.iter().enumerate() {
                let bins = bin_shape[dimension];
                let widths = edge.slice_along(Expression::from(1)..(bins + 1), 0)
                    - edge.slice_along(Expression::from(0)..bins, 0);
                let mut width_shape = vec![Expression::from(1); edges.len()];
                width_shape[dimension] = bins;
                let mut widths = reshape_tensor(widths, width_shape);
                widths.shape.expand(bin_shape.clone());
                volume *= widths;
            }
            histogram = histogram / total / volume;
        }
        Ok(histogram)
    }

    fn histogram_columns(
        &mut self,
        input: GraphTensor,
        dimensions: usize,
    ) -> Result<(Vec<GraphTensor>, Expression)> {
        anyhow::ensure!(dimensions > 0, "histogram dimension count must be positive");
        anyhow::ensure!(
            !input.shape.is_empty(),
            "histogramdd input must have a coordinate dimension"
        );
        let input_dims = input.dims();
        anyhow::ensure!(
            input_dims.last().and_then(|dim| dim.to_usize()) == Some(dimensions),
            "histogramdd coordinate dimension does not match bins"
        );
        let sample_count = product_of_dims(input_dims[..input_dims.len() - 1].iter().copied());
        let matrix = reshape_tensor(input, vec![sample_count, Expression::from(dimensions)]);
        let columns = (0..dimensions)
            .map(|dimension| {
                let column = matrix.slice_along(dimension..dimension + 1, 1).squeeze(1);
                super::movement::materialize_tensor(column)
            })
            .collect();
        Ok((columns, sample_count))
    }

    pub(crate) fn translate_histogram(&mut self, node: &Node) -> Result<()> {
        let dtype = self.output_meta_dtype(node)?;
        let input = self.get_input_tensor(node, 0)?.cast(dtype);
        let sample_count = product_of_dims(input.dims());
        let values = reshape_tensor(input, vec![sample_count]);
        let weight = self.histogram_weight(node, sample_count, dtype)?;
        let density = Self::histogram_density(node);
        let edges = if node.target.ends_with("bins_tensor") {
            self.get_input_tensor(node, 1)?.cast(dtype)
        } else {
            let bins = usize::try_from(self.get_int_arg(node, 1)?)
                .context("histogram bins must be nonnegative")?;
            anyhow::ensure!(bins > 0, "histogram bins must be positive");
            let range = Self::optional_float_list(node, "range")
                .filter(|range| !range.is_empty())
                .map(|range| (range[0], range[1]));
            self.uniform_histogram_edges(values, bins, range)
        };
        let histogram = self.histogram_from_edges(&[values], &[edges], weight, density)?;
        self.store_tensor_outputs(node, &[histogram, edges])
    }

    fn histogramdd_uniform_edges(
        &mut self,
        node: &Node,
        columns: &[GraphTensor],
        bins: &[i64],
    ) -> Result<Vec<GraphTensor>> {
        let range = Self::optional_float_list(node, "range").filter(|range| !range.is_empty());
        if let Some(range) = &range {
            anyhow::ensure!(
                range.len() == bins.len() * 2,
                "histogramdd range rank mismatch"
            );
        }
        bins.iter()
            .copied()
            .enumerate()
            .map(|(dimension, bins)| {
                let bins = usize::try_from(bins).context("histogramdd bins must be nonnegative")?;
                anyhow::ensure!(bins > 0, "histogramdd bins must be positive");
                let explicit = range
                    .as_ref()
                    .map(|range| (range[dimension * 2], range[dimension * 2 + 1]));
                Ok(self.uniform_histogram_edges(columns[dimension], bins, explicit))
            })
            .collect()
    }

    pub(crate) fn translate_histogramdd_bin_edges(&mut self, node: &Node) -> Result<()> {
        let bins = self.get_ints_arg(node, 1)?.to_vec();
        let input = self.get_input_tensor(node, 0)?;
        let (columns, _) = self.histogram_columns(input, bins.len())?;
        let edges = self.histogramdd_uniform_edges(node, &columns, &bins)?;
        self.store_tensor_outputs(node, &edges)
    }

    pub(crate) fn translate_histogramdd(
        &mut self,
        node: &Node,
        tensor_edges: bool,
    ) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        let input = self.get_input_tensor(node, 0)?.cast(dtype);
        let (edges, dimensions) = if tensor_edges {
            let names = node.inputs[1]
                .arg
                .as_tensors()
                .context("histogramdd tensor bins must be a tensor list")?;
            let edges = names
                .iter()
                .map(|name| self.get_tensor(&name.name).map(|edge| edge.cast(dtype)))
                .collect::<Result<Vec<_>>>()?;
            let dimensions = edges.len();
            (edges, dimensions)
        } else {
            let bins = self.get_ints_arg(node, 1)?.to_vec();
            let dimensions = bins.len();
            let (columns, _) = self.histogram_columns(input, dimensions)?;
            (
                self.histogramdd_uniform_edges(node, &columns, &bins)?,
                dimensions,
            )
        };
        let (columns, sample_count) = self.histogram_columns(input, dimensions)?;
        let weight = self.histogram_weight(node, sample_count, dtype)?;
        self.histogram_from_edges(&columns, &edges, weight, Self::histogram_density(node))
    }

    pub(crate) fn p_norm(
        &mut self,
        magnitude: GraphTensor,
        p: f64,
        axes: Vec<usize>,
    ) -> GraphTensor {
        if p == 0.0 {
            let zero = self.is_zero(magnitude);
            self.bool_not(zero).cast(magnitude.dtype).sum(axes)
        } else if p == 1.0 {
            magnitude.sum(axes)
        } else if p == 2.0 {
            magnitude.square().sum(axes).sqrt()
        } else if p == f64::INFINITY {
            magnitude.max(axes)
        } else if p == f64::NEG_INFINITY {
            magnitude.min(axes)
        } else {
            magnitude.pow(p as f32).sum(axes).pow((1.0 / p) as f32)
        }
    }

    fn first_along_axis(&mut self, value: GraphTensor, axis: usize) -> GraphTensor {
        (value.slice_along(0..1, axis).squeeze(axis) * 1).cast(value.dtype)
    }

    fn select_along_axis(
        &mut self,
        value: GraphTensor,
        indices_without_axis: GraphTensor,
        axis: usize,
        keepdim: bool,
    ) -> GraphTensor {
        let indices = indices_without_axis.unsqueeze(axis);
        let selected = super::movement_dynamic::pt2_gather_elements(value, indices, axis);
        if keepdim {
            selected
        } else {
            selected.squeeze(axis)
        }
    }

    fn first_nan_index(&mut self, value: GraphTensor, axis: usize) -> (GraphTensor, GraphTensor) {
        let nan = self.is_nan(value);
        let first = self
            .first_along_axis(nan.cast(DType::F32).stable_argsort(axis, true), axis)
            .cast(DType::I64);
        let count = nan.cast(DType::Int).sum(axis);
        let zero = self.graph.constant(0).expand_rhs(count.shape);
        (first, count.gt(zero))
    }

    fn nan_last_argsort(&mut self, value: GraphTensor, axis: usize) -> GraphTensor {
        if !dtype_can_contain_nan(value.dtype) {
            return value.stable_argsort(axis, false);
        }
        let nan = self.is_nan(value);
        let positive_infinity = self.constant_like(value, f64::INFINITY);
        self.select(nan, positive_infinity, value)
            .stable_argsort(axis, false)
    }

    /// Replacing NaNs with +inf makes them sort last except that real +inf
    /// entries share the same key. If a selected slot that belongs to the
    /// non-NaN prefix lands on a NaN, redirect it to the first real +inf.
    fn correct_nan_inf_collision(
        &mut self,
        value: GraphTensor,
        axis: usize,
        selected_index: GraphTensor,
        belongs_to_non_nan_prefix: GraphTensor,
    ) -> GraphTensor {
        if !dtype_can_contain_nan(value.dtype) {
            return selected_index;
        }
        let selected_value = self.select_along_axis(value, selected_index, axis, false);
        let selected_nan = self.is_nan(selected_value);
        let infinite = self.is_inf(value);
        let negative = self.signbit(value);
        let positive = self.bool_not(negative);
        let positive_infinite = self.bool_and(infinite, positive);
        let first_positive_infinite = self
            .first_along_axis(
                positive_infinite
                    .cast(DType::F32)
                    .stable_argsort(axis, true),
                axis,
            )
            .cast(DType::I64);
        let positive_infinite_count = positive_infinite.cast(DType::Int).sum(axis);
        let zero = self
            .graph
            .constant(0)
            .expand_rhs(positive_infinite_count.shape);
        let has_positive_infinity = positive_infinite_count.gt(zero);
        let replace = self.bool_and(
            selected_nan,
            self.bool_and(belongs_to_non_nan_prefix, has_positive_infinity),
        );
        self.select(replace, first_positive_infinite, selected_index)
    }

    pub(crate) fn translate_dim_extremum(&mut self, node: &Node, which: ArgExtremum) -> Result<()> {
        let value = self.get_input_tensor(node, 0)?;
        let raw_axis = self.get_int_arg(node, 1)?;
        let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            names.len() == 2,
            "max/min dim must produce values and indices"
        );

        if value.shape.is_empty() {
            anyhow::ensure!(
                matches!(raw_axis, -1 | 0),
                "dimension out of range for scalar"
            );
            self.tensors.insert(names[0].clone(), value);
            self.tensors
                .insert(names[1].clone(), self.graph.constant(0i64).cast(DType::I64));
            return Ok(());
        }
        let axis = normalize_dim(raw_axis, value.shape.len());
        let sort_key = if value.dtype == DType::Bool {
            value.cast(DType::F32)
        } else {
            value
        };
        let ordered_index = self
            .first_along_axis(sort_key.stable_argsort(axis, which.descending()), axis)
            .cast(DType::I64);
        let index = if dtype_can_contain_nan(value.dtype) {
            let (nan_index, has_nan) = self.first_nan_index(value, axis);
            self.select(has_nan, nan_index, ordered_index)
        } else {
            ordered_index
        };
        let selected = self.select_along_axis(value, index, axis, keepdim);
        let index = if keepdim {
            index.unsqueeze(axis)
        } else {
            index
        };
        self.tensors.insert(names[0].clone(), selected);
        self.tensors
            .insert(names[1].clone(), index.cast(DType::I64));
        Ok(())
    }

    pub(crate) fn translate_median(&mut self, node: &Node) -> Result<()> {
        self.translate_median_impl(node, false)
    }

    pub(crate) fn translate_nanmedian(&mut self, node: &Node) -> Result<()> {
        self.translate_median_impl(node, true)
    }

    fn translate_median_impl(&mut self, node: &Node, ignore_nan: bool) -> Result<()> {
        let value = self.get_input_tensor(node, 0)?;
        let names = Self::tensor_output_names(node);
        let dim_variant = node.target.ends_with(".dim");
        let (base, axis, keepdim) = if dim_variant {
            if value.shape.is_empty() {
                let raw_axis = self.get_int_arg(node, 1)?;
                anyhow::ensure!(
                    matches!(raw_axis, -1 | 0),
                    "dimension out of range for scalar"
                );
                self.tensors.insert(names[0].clone(), value);
                self.tensors
                    .insert(names[1].clone(), self.graph.constant(0i64).cast(DType::I64));
                return Ok(());
            }
            (
                value,
                normalize_dim(self.get_int_arg(node, 1)?, value.shape.len()),
                self.get_bool_arg(node, 2).unwrap_or(false),
            )
        } else {
            if value.shape.is_empty() {
                self.tensors.insert(names[0].clone(), value);
                return Ok(());
            }
            let numel = product_of_dims(value.dims().iter().copied());
            (
                reshape_tensor(super::movement::materialize_tensor(value), vec![numel]),
                0,
                false,
            )
        };

        let index = if ignore_nan && dtype_can_contain_nan(base.dtype) {
            let nan_count = self.is_nan(base).cast(DType::I64).sum(axis);
            let axis_length = self
                .graph
                .constant(base.dims()[axis])
                .cast(DType::I64)
                .expand_rhs(nan_count.shape);
            let valid_count = axis_length - nan_count;
            let zero = self
                .graph
                .constant(0)
                .cast(valid_count.dtype)
                .expand_rhs(valid_count.shape);
            let has_valid = valid_count.gt(zero);
            let kth = (((valid_count - 1).maximum(zero)).cast(DType::F64) * 0.5).cast(DType::I64);
            let sorted = self.nan_last_argsort(base, axis);
            let index = self
                .select_along_axis(sorted, kth, axis, false)
                .cast(DType::I64);
            self.correct_nan_inf_collision(base, axis, index, has_valid)
        } else {
            let axis_length = base.dims()[axis];
            let reduced_shape = base
                .dims()
                .into_iter()
                .enumerate()
                .filter_map(|(dim, size)| (dim != axis).then_some(size))
                .collect::<Vec<_>>();
            let mut kth = self.graph.constant((axis_length - 1) / 2).cast(DType::Int);
            if !reduced_shape.is_empty() {
                kth = kth.expand_rhs(reduced_shape);
            }
            let ordered = self
                .select_along_axis(base.stable_argsort(axis, false), kth, axis, false)
                .cast(DType::I64);
            if dtype_can_contain_nan(base.dtype) {
                let (nan_index, has_nan) = self.first_nan_index(base, axis);
                self.select(has_nan, nan_index, ordered)
            } else {
                ordered
            }
        };
        let selected = self.select_along_axis(base, index, axis, keepdim);
        self.tensors.insert(names[0].clone(), selected);
        if dim_variant {
            let index = if keepdim {
                index.unsqueeze(axis)
            } else {
                index
            };
            self.tensors
                .insert(names[1].clone(), index.cast(DType::I64));
        }
        Ok(())
    }

    pub(crate) fn translate_segment_reduce(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.get_input_tensor(node, 0)?;
        let reduction = node
            .inputs
            .iter()
            .find(|input| input.name == "reduce")
            .and_then(|input| match &input.arg {
                Argument::Other(value) => value
                    .as_str()
                    .or_else(|| value.get("as_string").and_then(|value| value.as_str())),
                _ => None,
            })
            .context("segment_reduce is missing its reduction name")?;
        let axis = self.named_int_arg(node, "axis").unwrap_or(0);
        let axis = normalize_dim(axis, data.shape.len());
        let output_shape = self.output_meta_shape(node)?;
        let segment_count = output_shape[axis];
        let input_count = data.dims()[axis];

        let lengths = self.named_tensor_arg(node, "lengths")?;
        let offsets = self.named_tensor_arg(node, "offsets")?;
        let (starts, ends) = if let Some(lengths) = lengths {
            let lengths = lengths.cast(DType::Int);
            let ends = lengths.cumsum(axis);
            (ends - lengths, ends)
        } else if let Some(offsets) = offsets {
            let offsets = offsets.cast(DType::Int);
            (
                offsets.slice_along(Expression::from(0)..segment_count, axis),
                offsets.slice_along(Expression::from(1)..(segment_count + 1), axis),
            )
        } else {
            anyhow::bail!("segment_reduce requires lengths or offsets");
        };

        let mut pair_shape = output_shape.clone();
        pair_shape.insert(axis + 1, input_count);
        let mut starts = starts.expand_dim(axis + 1, input_count);
        let mut ends = ends.expand_dim(axis + 1, input_count);
        for suffix in data.dims()[axis + 1..].iter().copied() {
            starts = starts.expand_dim(starts.shape.len(), suffix);
            ends = ends.expand_dim(ends.shape.len(), suffix);
        }
        let mut positions = self.graph.arange(input_count).cast(DType::Int);
        for (dimension, size) in pair_shape.iter().copied().enumerate() {
            if dimension != axis + 1 {
                positions = positions.expand_dim(dimension, size);
            }
        }
        let membership = self.bool_and(positions.ge(starts), positions.lt(ends));
        let expanded = data.expand_dim(axis, segment_count);
        let candidate_axis = axis + 1;
        let count = membership.cast(DType::Int).sum(candidate_axis);

        let initial = Self::named_input_index(node, "initial")
            .and_then(|index| self.constructor_scalar_arg(node, index).ok())
            .map(|value| self.typed_scalar_constant(&value, data.dtype))
            .transpose()?
            .map(|value| value.expand_rhs(output_shape.clone()));
        let has_initial = initial.is_some();
        let zero = self.full_tensor(expanded.dims(), data.dtype, 0.0);
        let one = self.full_tensor(expanded.dims(), data.dtype, 1.0);
        match reduction {
            "sum" | "mean" => {
                let mut sum = self.select(membership, expanded, zero).sum(candidate_axis);
                if let Some(initial) = initial {
                    sum += initial;
                }
                if reduction == "sum" {
                    Ok(sum)
                } else {
                    let zero_count = self.graph.constant(0).expand_rhs(count.shape);
                    let nonempty = count.gt(zero_count);
                    let one_count = self.graph.constant(1).expand_rhs(count.shape);
                    let safe_count = self.select(nonempty, count, one_count);
                    let mean = sum / safe_count.cast(data.dtype);
                    if has_initial {
                        Ok(mean)
                    } else {
                        let nan = self
                            .floating_scalar(f64::NAN, data.dtype)
                            .expand_rhs(mean.shape);
                        Ok(self.select(nonempty, mean, nan))
                    }
                }
            }
            "prod" => {
                let selected = self.select(membership, expanded, one);
                let magnitude = self.real_abs(selected).prod(candidate_axis);
                let negative_count = self.signbit(selected).cast(DType::Int).sum(candidate_axis);
                let two = self.graph.constant(2).expand_rhs(negative_count.shape);
                let zero = self.graph.constant(0).expand_rhs(negative_count.shape);
                let odd = (negative_count % two).ne(zero);
                let mut product = self.select(odd, magnitude * -1.0, magnitude);
                if let Some(initial) = initial {
                    product *= initial;
                }
                Ok(product)
            }
            "max" | "min" => {
                let fill = self
                    .floating_scalar(
                        if reduction == "max" {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        },
                        data.dtype,
                    )
                    .expand_rhs(expanded.shape);
                let values = self.select(membership, expanded, fill);
                let mut result = if reduction == "max" {
                    values.max(candidate_axis)
                } else {
                    values.min(candidate_axis)
                };
                let zero_count = self.graph.constant(0).expand_rhs(count.shape);
                let nonempty = count.gt(zero_count);
                if let Some(initial) = initial {
                    let combined = if reduction == "max" {
                        result.maximum(initial)
                    } else {
                        result.minimum(initial)
                    };
                    result = self.select(nonempty, combined, initial);
                } else {
                    let empty = self
                        .floating_scalar(
                            if reduction == "max" {
                                f64::NEG_INFINITY
                            } else {
                                f64::INFINITY
                            },
                            data.dtype,
                        )
                        .expand_rhs(result.shape);
                    result = self.select(nonempty, result, empty);
                }
                Ok(result)
            }
            other => anyhow::bail!("unsupported segment_reduce reduction {other}"),
        }
    }

    /// Normalize an optional ATen reduction-dimension list. Both `None` and
    /// `[]` mean a full reduction for the composed reductions in this file.
    pub(crate) fn composed_reduction_axes(&self, node: &Node, rank: usize) -> Result<Vec<usize>> {
        self.composed_reduction_axes_at(node, rank, 1)
    }

    fn composed_reduction_axes_at(
        &self,
        node: &Node,
        rank: usize,
        dim_arg: usize,
    ) -> Result<Vec<usize>> {
        let dims = self.get_ints_arg(node, dim_arg).ok();
        let raw_dims = match dims {
            Some(dims) if !dims.is_empty() => dims,
            _ => (0..rank).map(|axis| axis as i64).collect(),
        };
        let mut axes = Vec::with_capacity(raw_dims.len());
        for dim in raw_dims {
            if rank == 0 {
                anyhow::ensure!(
                    matches!(dim, -1 | 0),
                    "reduction dimension {dim} is out of range for a scalar"
                );
                continue;
            }
            anyhow::ensure!(
                dim >= -(rank as i64) && dim < rank as i64,
                "reduction dimension {dim} is out of range for rank {rank}"
            );
            let axis = normalize_dim(dim, rank);
            anyhow::ensure!(!axes.contains(&axis), "reduction dimensions must be unique");
            axes.push(axis);
        }
        Ok(axes)
    }

    pub(crate) fn restore_reduced_dims(
        &self,
        mut value: GraphTensor,
        axes: &[usize],
        keepdim: bool,
    ) -> GraphTensor {
        if keepdim {
            let mut axes = axes.to_vec();
            axes.sort_unstable();
            for axis in axes {
                value = value.unsqueeze(axis);
            }
        }
        value
    }

    /// Lower `linalg_vector_norm` after the caller has constructed real-valued
    /// magnitudes. Complex inputs use this same routine after `abs(z)`.
    pub(crate) fn vector_norm_from_magnitude(
        &mut self,
        node: &Node,
        magnitude: GraphTensor,
    ) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let magnitude = magnitude.cast(output_dtype);
        let axes = self.composed_reduction_axes_at(node, magnitude.shape.len(), 2)?;
        let keepdim = self.get_bool_arg(node, 3).unwrap_or(false);
        let ord = self.get_float_arg(node, 1).unwrap_or(2.0);

        if (ord.is_infinite() || ord < 0.0)
            && axes
                .iter()
                .any(|&axis| magnitude.dims()[axis].to_usize() == Some(0))
        {
            anyhow::bail!(
                "linalg_vector_norm order {ord} has no identity for an empty reduction dimension"
            );
        }

        let reduced = if ord == 0.0 {
            let zero = self
                .graph
                .constant_float(0.0)
                .cast(magnitude.dtype)
                .expand_rhs(magnitude.shape);
            let ordered_nonzero = self.bool_or(magnitude.lt(zero), magnitude.gt(zero));
            let nonzero = if dtype_can_contain_nan(magnitude.dtype) {
                let nan = self.is_nan(magnitude);
                self.bool_or(ordered_nonzero, nan)
            } else {
                ordered_nonzero
            };
            nonzero.cast(output_dtype).sum(axes.clone())
        } else if ord == 1.0 {
            magnitude.sum(axes.clone())
        } else if ord == 2.0 {
            (magnitude * magnitude).sum(axes.clone()).sqrt()
        } else if ord == f64::INFINITY {
            magnitude.max(axes.clone())
        } else if ord == f64::NEG_INFINITY {
            magnitude.min(axes.clone())
        } else {
            magnitude
                .pow(ord as f32)
                .sum(axes.clone())
                .pow((1.0 / ord) as f32)
        };
        Ok(self.restore_reduced_dims(reduced, &axes, keepdim))
    }

    pub(crate) fn translate_linalg_vector_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        let magnitude = self.real_abs(value);
        self.vector_norm_from_magnitude(node, magnitude)
    }

    pub(crate) fn translate_dist(&mut self, node: &Node) -> Result<GraphTensor> {
        let lhs = self.get_input_tensor(node, 0)?;
        let rhs = self.get_input_tensor(node, 1)?;
        let (lhs, rhs) = ensure_same_dtype(lhs, rhs);
        let (lhs, rhs) = broadcast_binary(lhs, rhs);
        let magnitude = self.real_abs(lhs - rhs).cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 2).unwrap_or(2.0);
        Ok(self.p_norm(magnitude, p, (0..magnitude.shape.len()).collect()))
    }

    pub(crate) fn translate_cdist(&mut self, node: &Node) -> Result<GraphTensor> {
        let lhs = self.get_input_tensor(node, 0)?;
        let rhs = self.get_input_tensor(node, 1)?;
        anyhow::ensure!(
            lhs.shape.len() >= 2 && rhs.shape.len() >= 2,
            "cdist inputs must be matrices"
        );
        let (mut lhs, mut rhs) = ensure_same_dtype(lhs, rhs);
        let output_shape = self.output_meta_shape(node)?;
        let feature = lhs.dims()[lhs.shape.len() - 1];
        anyhow::ensure!(
            feature == rhs.dims()[rhs.shape.len() - 1],
            "cdist feature dimensions must match"
        );
        let mut pair_shape = output_shape;
        pair_shape.push(feature);
        lhs = lhs.expand_dim(lhs.shape.len() - 1, rhs.dims()[rhs.shape.len() - 2]);
        rhs = rhs.expand_dim(rhs.shape.len() - 2, lhs.dims()[lhs.shape.len() - 3]);
        lhs.shape.expand(pair_shape.clone());
        rhs.shape.expand(pair_shape);
        let magnitude = self.real_abs(lhs - rhs).cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 2)?;
        Ok(self.p_norm(magnitude, p, vec![magnitude.shape.len() - 1]))
    }

    pub(crate) fn translate_pdist(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        anyhow::ensure!(input.shape.len() == 2, "pdist input must be a matrix");
        let rows = input.dims()[0];
        let columns = input.dims()[1];
        let output_shape = self.output_meta_shape(node)?;
        let pairs = output_shape[0];
        let k = self.graph.arange(pairs).cast(DType::F64);
        let rows_f = self
            .graph
            .constant(rows)
            .cast(DType::F64)
            .expand_rhs(k.shape);
        let discriminant = k * -8.0 + rows_f * (rows_f - 1.0) * 4.0 - 7.0;
        let i = (rows_f
            - 2.0
            - (discriminant.sqrt() * 0.5 - 0.5)
                .cast(DType::I64)
                .cast(DType::F64))
        .cast(DType::Int);
        let i_f = i.cast(DType::F64);
        let j = (k + i_f + 1.0 - rows_f * (rows_f - 1.0) * 0.5
            + (rows_f - i_f) * (rows_f - i_f - 1.0) * 0.5)
            .cast(DType::Int);
        let i = i.expand_dim(1, columns);
        let j = j.expand_dim(1, columns);
        let left = super::movement_dynamic::pt2_index_select(
            input,
            i.slice_along(0..1, 1).squeeze(1),
            0,
            &[pairs, columns],
        );
        let right = super::movement_dynamic::pt2_index_select(
            input,
            j.slice_along(0..1, 1).squeeze(1),
            0,
            &[pairs, columns],
        );
        let magnitude = self
            .real_abs(left - right)
            .cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 1).unwrap_or(2.0);
        Ok(self.p_norm(magnitude, p, vec![1]))
    }

    pub(crate) fn translate_log_softmax(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        let rank = value.shape.len();
        let raw_dim = self.get_int_arg(node, 1)?;
        if rank == 0 {
            anyhow::ensure!(
                matches!(raw_dim, -1 | 0),
                "log_softmax dimension {raw_dim} is out of range for a scalar"
            );
            // Preserve PyTorch's IEEE behavior: finite scalars become zero,
            // while +/-inf and NaN become NaN through the ordinary stable
            // log-softmax formula rather than an unconditional zero.
            return Ok(value.unsqueeze(0).log_softmax(0).squeeze(0));
        }
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "log_softmax dimension {raw_dim} is out of range for rank {rank}"
        );
        Ok(value.log_softmax(normalize_dim(raw_dim, rank)))
    }

    pub(crate) fn variance_correction(&self, node: &Node) -> f64 {
        if let Some(correction) = node
            .inputs
            .iter()
            .position(|input| input.name == "correction")
            .and_then(|index| self.get_float_arg(node, index).ok())
        {
            return correction;
        }
        node.inputs
            .iter()
            .position(|input| input.name == "unbiased")
            .and_then(|index| self.get_bool_arg(node, index).ok())
            .map_or(1.0, |unbiased| if unbiased { 1.0 } else { 0.0 })
    }

    pub(crate) fn floating_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        if dtype == DType::F64 {
            self.graph.constant_float64(value)
        } else {
            self.graph.constant_float(value as f32).cast(dtype)
        }
    }

    /// Compute variance and mean for an ordinary real tensor. PyTorch clamps
    /// non-positive degrees of freedom to zero before division, which yields
    /// NaN for a zero numerator and infinity otherwise.
    pub(crate) fn variance_mean_real(
        &mut self,
        node: &Node,
        value: GraphTensor,
    ) -> Result<(GraphTensor, GraphTensor)> {
        let axes = self.composed_reduction_axes(node, value.shape.len())?;
        let keepdim = node
            .inputs
            .iter()
            .position(|input| input.name == "keepdim")
            .and_then(|index| self.get_bool_arg(node, index).ok())
            .unwrap_or(false);
        let correction = self.variance_correction(node);
        let output_dtype = self.output_meta_dtype(node)?;
        let value = value.cast(output_dtype);
        let n = product_of_dims(axes.iter().map(|&axis| value.dims()[axis]));
        let mean = if axes.is_empty() {
            value
        } else {
            value.sum(axes.clone()) / n
        };
        let expanded_mean = mean.expand_to_shape_on_axes(value.shape, axes.clone());
        let centered = value - expanded_mean;
        let numerator = (centered * centered).sum(axes.clone());
        let degrees = self.graph.constant(n).cast(output_dtype)
            - self.floating_scalar(correction, output_dtype);
        let zero = self.floating_scalar(0.0, output_dtype);
        let divisor = degrees.maximum(zero).expand_rhs(numerator.shape);
        let variance = numerator / divisor;
        Ok((
            self.restore_reduced_dims(variance, &axes, keepdim),
            self.restore_reduced_dims(mean, &axes, keepdim),
        ))
    }

    pub(crate) fn translate_var(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        Ok(self.variance_mean_real(node, value)?.0)
    }

    pub(crate) fn translate_var_mean(&mut self, node: &Node) -> Result<()> {
        let value = self.get_input_tensor(node, 0)?;
        let (variance, mean) = self.variance_mean_real(node, value)?;
        let names = node
            .outputs
            .iter()
            .flat_map(|output| {
                output
                    .as_tensors
                    .as_ref()
                    .map(|values| values.iter().map(|value| value.name.clone()).collect())
                    .unwrap_or_else(|| {
                        output
                            .as_tensor
                            .as_ref()
                            .map(|value| vec![value.name.clone()])
                            .unwrap_or_default()
                    })
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(names.len() == 2, "var_mean must have two tensor outputs");
        self.tensors.insert(names[0].clone(), variance);
        self.tensors.insert(names[1].clone(), mean);
        Ok(())
    }

    pub(crate) fn mean_divide(
        &mut self,
        sums: GraphTensor,
        counts: GraphTensor,
        output_dtype: DType,
    ) -> GraphTensor {
        if is_integral_dtype(output_dtype) {
            let quotient = sums.cast(DType::F32) / counts.cast(DType::F32);
            let trunc = quotient.cast(DType::Int).cast(DType::F32);
            let floor = trunc - quotient.lt(trunc).cast(DType::F32);
            floor.cast(output_dtype)
        } else {
            sums / counts.cast(output_dtype)
        }
    }

    /// Reduce PyTorch truth values for the three `aten.any` overloads.
    ///
    /// `truth` is already boolean. `any.dims(dim=[])` is an elementwise bool
    /// cast rather than a full reduction, while a missing dim list reduces
    /// every axis. Keeping that distinction here lets the ordinary and
    /// frontend-only complex paths share exactly the same axis semantics.
    pub(crate) fn translate_any_from_truth(
        &mut self,
        node: &Node,
        truth: GraphTensor,
    ) -> Result<GraphTensor> {
        let rank = truth.shape.len();
        let (axes, keepdim) = match node.target.as_str() {
            "torch.ops.aten.any.default" => ((0..rank).collect::<Vec<_>>(), false),
            "torch.ops.aten.any.dim" => {
                let dim = self.get_int_arg(node, 1)?;
                let axis = if rank == 0 {
                    anyhow::ensure!(
                        matches!(dim, -1 | 0),
                        "any dimension {dim} out of range for a scalar"
                    );
                    0
                } else {
                    anyhow::ensure!(
                        dim >= -(rank as i64) && dim < rank as i64,
                        "any dimension {dim} out of range for rank {rank}"
                    );
                    normalize_dim(dim, rank)
                };
                let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
                (if rank == 0 { vec![] } else { vec![axis] }, keepdim)
            }
            "torch.ops.aten.any.dims" => {
                let axes = match self.get_ints_arg(node, 1) {
                    Ok(dims) => {
                        let mut axes = Vec::with_capacity(dims.len());
                        for dim in dims {
                            anyhow::ensure!(
                                dim >= -(rank as i64) && dim < rank as i64,
                                "any dimension {dim} out of range for rank {rank}"
                            );
                            let axis = normalize_dim(dim, rank);
                            anyhow::ensure!(!axes.contains(&axis), "any dimensions must be unique");
                            axes.push(axis);
                        }
                        axes
                    }
                    Err(_) => (0..rank).collect(),
                };
                let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
                (axes, keepdim)
            }
            other => anyhow::bail!("translate_any_from_truth called for {other}"),
        };

        if axes.is_empty() {
            return Ok(truth.cast(DType::Bool));
        }

        let counts = truth.cast(DType::Int).sum(axes.clone());
        let zero = self.graph.constant(0).expand_rhs(counts.shape);
        let mut result = counts.ne(zero);
        if keepdim {
            let mut sorted_axes = axes;
            sorted_axes.sort_unstable();
            for axis in sorted_axes {
                result = result.unsqueeze(axis);
            }
        }
        Ok(result)
    }

    pub(crate) fn translate_any(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let zero = self
            .graph
            .constant(0)
            .cast(input.dtype)
            .expand_rhs(input.shape);
        self.translate_any_from_truth(node, input.ne(zero))
    }

    /// Build the per-element source indices and validity mask for one
    /// Hillis-Steele inclusive-scan step. A lane at `i` reads `i - offset`;
    /// prefix lanes read zero but are subsequently kept unchanged by `valid`.
    /// This gather-based shift avoids arithmetic padding, which would turn
    /// otherwise inactive `0 * NaN` lanes into NaNs.
    pub(crate) fn scan_shift_indices(
        &mut self,
        shape: &[Expression],
        axis: usize,
        offset: usize,
    ) -> (GraphTensor, GraphTensor) {
        let mut positions = self.graph.arange(shape[axis]);
        for (dim, size) in shape.iter().copied().enumerate() {
            if dim != axis {
                positions = positions.expand_dim(dim, size);
            }
        }

        let offset = self
            .graph
            .constant(offset as i64)
            .expand_rhs(positions.shape);
        let valid = positions.ge(offset);
        let zero = self.graph.constant(0).expand_rhs(positions.shape);
        let shifted = self.select(valid, positions - offset, zero);
        (shifted, valid)
    }

    /// Lower `aten.cumprod.default` as an inclusive multiplication scan.
    /// Unlike Luminal's legacy `GraphTensor::cumprod`, this never rewrites
    /// products through log/exp, so zeros, negatives, integers, and overflow
    /// retain ordinary multiplication semantics.
    pub(crate) fn translate_cumprod(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut values = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        let Some(axis) = cumulative_axis(self.get_int_arg(node, 1)?, values.shape.len())? else {
            return Ok(values);
        };
        let length = values.dims()[axis].to_usize().ok_or_else(|| {
            anyhow::anyhow!("cumprod currently requires a concrete scan dimension")
        })?;

        let mut offset = 1;
        while offset < length {
            let (shifted_indices, valid) = self.scan_shift_indices(&values.dims(), axis, offset);
            let shifted =
                super::movement_dynamic::pt2_gather_elements(values, shifted_indices, axis);
            values = self.select(valid, shifted * values, values);
            offset *= 2;
        }
        Ok(values)
    }

    /// Lower `aten.cummax.default` / `aten.cummin.default`, carrying both the
    /// running value and its source index through the same inclusive scan.
    /// PyTorch selects the later element on equal values and on repeated NaNs;
    /// a prior NaN beats a later ordered value, so NaN propagation is explicit.
    pub(crate) fn translate_cumextremum(&mut self, node: &Node, which: CumExtremum) -> Result<()> {
        let mut values = self.get_input_tensor(node, 0)?;
        let axis = cumulative_axis(self.get_int_arg(node, 1)?, values.shape.len())?;

        let mut indices = match axis {
            None => self.graph.constant(0i64).cast(DType::I64),
            Some(axis) if values.dims()[axis].to_usize() == Some(0) => values.cast(DType::I64),
            Some(axis) => {
                let mut positions = self.graph.arange(values.dims()[axis]).cast(DType::I64);
                for (dim, size) in values.dims().into_iter().enumerate() {
                    if dim != axis {
                        positions = positions.expand_dim(dim, size);
                    }
                }
                positions
            }
        };

        if let Some(axis) = axis {
            let length = values.dims()[axis].to_usize().ok_or_else(|| {
                anyhow::anyhow!("cummax/cummin currently require a concrete scan dimension")
            })?;
            let mut offset = 1;
            while offset < length {
                let (shifted_indices, valid) =
                    self.scan_shift_indices(&values.dims(), axis, offset);
                let left_values =
                    super::movement_dynamic::pt2_gather_elements(values, shifted_indices, axis);
                let left_indices =
                    super::movement_dynamic::pt2_gather_elements(indices, shifted_indices, axis);

                let ordered_left_wins = match which {
                    CumExtremum::Max => values.lt(left_values),
                    CumExtremum::Min => left_values.lt(values),
                };
                let left_wins = if dtype_can_contain_nan(values.dtype) {
                    let left_nan = self.is_nan(left_values);
                    let right_nan = self.is_nan(values);
                    let left_nan_only = self.bool_and(left_nan, self.bool_not(right_nan));
                    self.bool_or(ordered_left_wins, left_nan_only)
                } else {
                    ordered_left_wins
                };
                let selected_values = self.select(left_wins, left_values, values);
                let selected_indices = self.select(left_wins, left_indices, indices);
                values = self.select(valid, selected_values, values);
                indices = self.select(valid, selected_indices, indices);
                offset *= 2;
            }
        }

        let tuple_outputs = node.outputs.first().and_then(|o| o.as_tensors.as_ref());
        let values_name = if let Some(outputs) = tuple_outputs {
            outputs.first().map(|tensor| tensor.name.clone())
        } else {
            node.outputs
                .first()
                .and_then(|output| output.as_tensor.as_ref())
                .map(|tensor| tensor.name.clone())
        };
        let indices_name = if let Some(outputs) = tuple_outputs {
            outputs.get(1).map(|tensor| tensor.name.clone())
        } else {
            node.outputs
                .get(1)
                .and_then(|output| output.as_tensor.as_ref())
                .map(|tensor| tensor.name.clone())
        };
        if let Some(name) = values_name.filter(|name| !name.is_empty()) {
            self.tensors.insert(name, values);
        }
        if let Some(name) = indices_name.filter(|name| !name.is_empty()) {
            self.tensors.insert(name, indices.cast(DType::I64));
        }
        Ok(())
    }

    pub(crate) fn translate_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;

        // Try to get dims arg; if missing or empty, fall back to full reduce
        let dims_result = self.get_ints_arg(node, 1);
        let (axes, keepdim) = match dims_result {
            Ok(ref dims) if !dims.is_empty() => {
                let ndim = a.shape.len();
                let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
                let keepdim = if node.inputs.len() > 2 {
                    self.get_bool_arg(node, 2).unwrap_or(false)
                } else {
                    false
                };
                (axes, keepdim)
            }
            _ => {
                // Full reduce: reduce over every axis, leaving a rank-0 (scalar) tensor.
                // PyTorch eager returns shape () for `x.sum()` etc., and downstream ops
                // (e.g. unsqueeze(0).expand(N)) rely on this rank.
                let ndim = a.shape.len();
                if ndim == 0 {
                    // Already rank-0 — reducing over no axes is a no-op for sum/max/min/prod,
                    // and mean of a scalar is just the scalar.
                    return Ok(a);
                }
                let axes: Vec<usize> = (0..ndim).collect();
                let result = match op {
                    ReductionOp::Sum => a.sum(axes),
                    ReductionOp::Mean => a.mean(axes),
                    ReductionOp::Max => a.max(axes),
                    ReductionOp::Min => a.min(axes),
                    ReductionOp::Prod => a.prod(axes),
                };
                return Ok(result);
            }
        };

        let mut result = match op {
            ReductionOp::Sum => a.sum(axes.clone()),
            ReductionOp::Mean => a.mean(axes.clone()),
            ReductionOp::Max => a.max(axes.clone()),
            ReductionOp::Min => a.min(axes.clone()),
            ReductionOp::Prod => a.prod(axes.clone()),
        };

        if keepdim {
            let mut sorted_axes = axes.clone();
            sorted_axes.sort();
            for &ax in &sorted_axes {
                result = result.unsqueeze(ax);
            }
        }

        Ok(result)
    }

    /// Lower `aten.argmax.default` / `aten.argmin.default` by reusing the
    /// existing `stable_argsort` op and selecting the first index along the
    /// sort axis.
    ///
    /// PyTorch signature: `argmax(self, dim=None, keepdim=False)` (likewise
    /// for argmin). FX export emits the inputs positionally:
    ///   - input 0: tensor
    ///   - input 1: dim (Int) or None (Other) — when `dim=None`
    ///   - input 2: keepdim (Bool, optional)
    ///
    /// When `dim=None`, PyTorch flattens the tensor; we mirror that by
    /// reshaping to a 1-D `[numel]` view (which requires concrete dims).
    /// The result of argsort along the sort axis is sliced at index 0,
    /// then squeezed away — i.e. `select(dim, 0)` — to give the index of
    /// the extremum. With `keepdim=True` we re-insert a size-1 dim at
    /// `dim`.
    ///
    /// The slice + squeeze chain produces a non-contiguous `DType::Int`
    /// view; we materialize it with `* 1` so the resulting node has
    /// contiguous strides matching its visible shape (mirroring the
    /// `topk` lowering in `translate_topk`). Without this, the output
    /// buffer would be sized for the un-sliced argsort tensor while the
    /// shape tracker reports a smaller rank.
    ///
    /// The result is cast to `DType::I64` to match PyTorch's int64
    /// argmax / argmin indices.
    pub(crate) fn translate_argextremum(
        &mut self,
        node: &Node,
        which: ArgExtremum,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;

        // dim is positional input 1. PyTorch encodes `dim=None` as a non-Int
        // argument (typically `Argument::Other(Null)`), so a missing or
        // non-int slot means "reduce over the flattened tensor".
        let dim_opt: Option<i64> = if node.inputs.len() > 1 {
            self.get_int_arg(node, 1).ok()
        } else {
            None
        };
        let keepdim = if node.inputs.len() > 2 {
            self.get_bool_arg(node, 2).unwrap_or(false)
        } else {
            false
        };

        if a.shape.is_empty() {
            match dim_opt {
                None | Some(0) | Some(-1) => {
                    // PyTorch returns scalar index 0 for rank-0 argmax/argmin.
                    // `keepdim=True` does not add a dimension when the input is 0-d.
                    return Ok(self.graph.constant(0i64).cast(DType::I64));
                }
                Some(dim) => {
                    return Err(anyhow::anyhow!(
                        "Dimension out of range (expected to be in range of [-1, 0], but got {dim})"
                    ));
                }
            }
        }

        let descending = which.descending();

        let (sort_axis, base) = match dim_opt {
            None => {
                // Full-reduce: flatten to 1-D, argsort along axis 0.
                let total = concrete_numel(&a)?;
                let flat = reshape_tensor(a, vec![Expression::from(total)]);
                (0usize, flat)
            }
            Some(dim_raw) => {
                let dim = normalize_dim(dim_raw, a.shape.len());
                (dim, a)
            }
        };

        // Pick index 0 along the sort axis. The slice-then-squeeze chain
        // produces a non-contiguous view whose physical buffer is still
        // sized for the un-sliced argsort tensor; the optional `keepdim`
        // unsqueeze adds a stride-0 axis which is also non-contiguous.
        // Materialize at the end with `* 1` so the resulting node has
        // contiguous strides matching its visible shape (matches the
        // pattern used by `translate_topk` for sliced index outputs).
        let sorted = base.stable_argsort(sort_axis, descending);
        let picked = sorted.slice_along(0..1, sort_axis).squeeze(sort_axis);
        let result = if keepdim {
            picked.unsqueeze(sort_axis)
        } else {
            picked
        };
        Ok((result * 1).cast(DType::I64))
    }
}
