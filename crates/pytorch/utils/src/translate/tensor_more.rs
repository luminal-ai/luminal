//! Additional ATen lowerings ported from the parked translator (port batch 6+).
#![allow(dead_code)]

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::{Argument, Node};

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Shared local helpers (the recorder has no constructor-scalar or
    // nonzero-from-truth helpers, so they live here privately).
    // ---------------------------------------------------------------

    /// Bind a node's declared outputs by flattened name order. This mirrors
    /// the old translator's `store_tensor_outputs` and the movement
    /// module's `movement_bind_outputs`: it handles both the tuple-in-one
    /// `as_tensors` form and the one-entry-per-output form.
    fn tm_bind_outputs(&mut self, node: &Node, values: Vec<GraphTensor>) -> Result<()> {
        let names = Self::tensor_output_names(node);
        ensure!(
            names.len() == values.len(),
            "`{}` declared {} outputs but produced {}",
            node.target,
            names.len(),
            values.len()
        );
        for (name, value) in names.into_iter().zip(values) {
            self.values.insert(name, value);
        }
        Ok(())
    }

    /// `beta`/`alpha` scale kwargs of addmv/addbmm. `None` means the kwarg is
    /// absent (treat as 1.0); a literal 0/1 short-circuits. Runtime scalar
    /// references (SymInt/SymFloat/SymBool dataflow) are refused, matching the
    /// old `typed_scalar_constant`.
    fn tm_scale_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: GraphTensor,
    ) -> Result<GraphTensor> {
        let Some(input) = node.inputs.iter().find(|input| input.name == name) else {
            return Ok(value);
        };
        let arg = &input.arg;
        let (is_zero, is_one, scalar) = if let Some(v) = arg.as_float() {
            (v == 0.0, v == 1.0, Some(v))
        } else if let Some(v) = arg.as_int() {
            (v == 0, v == 1, Some(v as f64))
        } else if let Some(v) = arg.as_bool() {
            (!v, v, Some(if v { 1.0 } else { 0.0 }))
        } else {
            if arg.as_tensor_name().is_none()
                && let Some(runtime) = arg.as_value_name()
            {
                bail!("runtime scalar {runtime} is not a literal constant");
            }
            bail!("unsupported scalar argument {arg:?} for {name}");
        };
        if is_one {
            return Ok(value);
        }
        if is_zero {
            return Ok(self.full_tensor(value.dims(), value.dtype, 0.0));
        }
        let v = scalar.unwrap();
        let constant = match value.dtype {
            DType::F64 => self.cx.constant_f64(v),
            DType::I64 | DType::Int | DType::I8 | DType::U8 | DType::I16 => {
                self.cx.constant_i64(v as i64).cast(value.dtype)
            }
            _ => self.cx.constant_f32(v as f32).cast(value.dtype),
        };
        Ok(value * constant.expand_rhs(value.dims()))
    }

    /// `torch.histogram`'s `range` kwarg (an "as_floats" JSON wrapper).
    fn tm_optional_float_list(node: &Node, name: &str) -> Option<Vec<f64>> {
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

    fn tm_histogram_density(node: &Node) -> bool {
        node.inputs
            .iter()
            .find(|input| input.name == "density")
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false)
    }

    fn tm_histogram_weight(
        &mut self,
        node: &Node,
        sample_count: IntExpr,
        dtype: DType,
    ) -> Result<GraphTensor> {
        if let Some(input) = node.inputs.iter().find(|input| input.name == "weight")
            && let Some(weight) = self.optional_tensor_operand(input)?
        {
            return Ok(util::reshape_tensor(weight.cast(dtype), &[sample_count]));
        }
        Ok(self
            .floating_scalar(1.0, dtype)
            .expand_rhs(vec![sample_count]))
    }

    /// Split a `histogramdd` input `[..., D]` into `D` coordinate columns of
    /// shape `[sample_count]`, plus the flattened `sample_count`. Ported from
    /// the old translator's `histogram_columns`.
    fn tm_histogram_columns(
        &mut self,
        input: GraphTensor,
        dimensions: usize,
    ) -> Result<(Vec<GraphTensor>, IntExpr)> {
        ensure!(dimensions > 0, "histogram dimension count must be positive");
        ensure!(
            !input.dims().is_empty(),
            "histogramdd input must have a coordinate dimension"
        );
        let input_dims = input.dims();
        ensure!(
            input_dims.last().and_then(|dim| dim.to_usize()) == Some(dimensions),
            "histogramdd coordinate dimension does not match bins"
        );
        let sample_count = input_dims[..input_dims.len() - 1]
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim);
        let matrix = util::reshape_tensor(input, &[sample_count, IntExpr::from(dimensions)]);
        let columns = (0..dimensions)
            .map(|dimension| {
                let column = matrix.slice_along(dimension..dimension + 1, 1).squeeze(1);
                util::materialize_tensor(column)
            })
            .collect();
        Ok((columns, sample_count))
    }

    /// One uniform edge tensor per `histogramdd` dimension, using `range`
    /// when the caller supplied it. Ported from the old translator's
    /// `histogramdd_uniform_edges`.
    fn tm_histogramdd_uniform_edges(
        &mut self,
        node: &Node,
        columns: &[GraphTensor],
        bins: &[i64],
    ) -> Result<Vec<GraphTensor>> {
        let range = Self::tm_optional_float_list(node, "range").filter(|range| !range.is_empty());
        if let Some(range) = &range {
            ensure!(
                range.len() == bins.len() * 2,
                "histogramdd range rank mismatch"
            );
        }
        bins.iter()
            .copied()
            .enumerate()
            .map(|(dimension, bins)| {
                let bins = usize::try_from(bins).context("histogramdd bins must be nonnegative")?;
                ensure!(bins > 0, "histogramdd bins must be positive");
                let explicit = range
                    .as_ref()
                    .map(|range| (range[dimension * 2], range[dimension * 2 + 1]));
                Ok(self.tm_uniform_histogram_edges(columns[dimension], bins, explicit))
            })
            .collect()
    }

    /// Uniform bin edges in the old translator's `histogram` spelling:
    /// `linspace(lower, upper, bins+1)`, with the degenerate all-equal input
    /// widened to `[x-0.5, x+0.5]`. `explicit_range` is the caller's `range`.
    fn tm_uniform_histogram_edges(
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
            (values.min(vec![0]), values.max(vec![0]))
        };
        let equal = self.is_zero(upper - lower);
        lower = self.select(equal, lower - 0.5, lower);
        upper = self.select(equal, upper + 0.5, upper);
        let position = self.cx.arange(bins + 1).cast(values.dtype);
        let edge_shape = position.dims();
        let lower_expanded = lower.expand_rhs(edge_shape.clone());
        let width = (upper - lower).expand_rhs(edge_shape);
        lower_expanded + position * width / bins as f32
    }

    /// Histogram over `columns` (one per dimension) with the given bin edges.
    /// Bins are left-closed/right-open, with the final bin right-closed, and
    /// NaN never lands in a bin. Shared by `torch.histogram` (1-D) and, if
    /// later wired, `_histogramdd_*`.
    fn tm_histogram_from_edges(
        &mut self,
        columns: &[GraphTensor],
        edges: &[GraphTensor],
        weight: GraphTensor,
        density: bool,
    ) -> Result<GraphTensor> {
        ensure!(
            !columns.is_empty(),
            "histogram must have at least one dimension"
        );
        ensure!(columns.len() == edges.len(), "histogram edge rank mismatch");
        let sample_count = columns[0].dims()[0];
        let mut bin_shape = Vec::with_capacity(edges.len());
        for edge in edges {
            let edge_count = edge.dims()[0];
            ensure!(
                edge_count.to_usize().is_none_or(|count| count >= 2),
                "histogram bin edges must contain at least two values"
            );
            bin_shape.push(edge_count - IntExpr::from(1));
        }

        let mut full_shape = vec![sample_count];
        full_shape.extend_from_slice(&bin_shape);
        let mut membership = self
            .cx
            .constant_i32(1)
            .cast(DType::Bool)
            .expand_rhs(full_shape.clone());
        for (dimension, (column, edge)) in columns.iter().zip(edges).enumerate() {
            let bins = bin_shape[dimension];
            let left = edge.slice_along(IntExpr::from(0)..bins, 0);
            let right = edge.slice_along(IntExpr::from(1)..(bins + IntExpr::from(1)), 0);

            let mut column_shape = vec![IntExpr::from(1); columns.len() + 1];
            column_shape[0] = sample_count;
            let column = util::reshape_tensor(*column, &column_shape).expand(full_shape.clone());
            let mut edge_shape = vec![IntExpr::from(1); columns.len() + 1];
            edge_shape[dimension + 1] = bins;
            let left = util::reshape_tensor(left, &edge_shape).expand(full_shape.clone());
            let right = util::reshape_tensor(right, &edge_shape).expand(full_shape.clone());

            let below_right = column.lt(right);
            let equal_right = self.is_zero(column - right);
            let last_position = self.cx.arange(bins).cast(DType::Int);
            let last_index = self
                .cx
                .constant_i32(bins - IntExpr::from(1))
                .cast(DType::Int)
                .expand_rhs(last_position.dims());
            let last_position = last_position.eq(last_index);
            let last_position =
                util::reshape_tensor(last_position, &edge_shape).expand(full_shape.clone());
            let inclusive_last = self.bool_and(equal_right, last_position);
            let upper = self.bool_or(below_right, inclusive_last);
            let in_bin = self.bool_and(column.ge(left), upper);
            let column_nan = self.is_nan(column);
            let not_nan = self.bool_not(column_nan);
            let in_bin = self.bool_and(in_bin, not_nan);
            membership = self.bool_and(membership, in_bin);
        }

        let mut weight_shape = vec![IntExpr::from(1); columns.len() + 1];
        weight_shape[0] = sample_count;
        let weight = util::reshape_tensor(weight, &weight_shape).expand(full_shape);
        let mut histogram = (membership.cast(weight.dtype) * weight).sum(vec![0]);
        if density {
            let axes = (0..bin_shape.len()).collect::<Vec<_>>();
            let total = histogram.sum(axes).expand_rhs(bin_shape.clone());
            let mut volume = self
                .floating_scalar(1.0, histogram.dtype)
                .expand_rhs(bin_shape.clone());
            for (dimension, edge) in edges.iter().enumerate() {
                let bins = bin_shape[dimension];
                let widths = edge.slice_along(IntExpr::from(1)..(bins + IntExpr::from(1)), 0)
                    - edge.slice_along(IntExpr::from(0)..bins, 0);
                let mut width_shape = vec![IntExpr::from(1); edges.len()];
                width_shape[dimension] = bins;
                let widths = util::reshape_tensor(widths, &width_shape).expand(bin_shape.clone());
                volume *= widths;
            }
            histogram = histogram / total / volume;
        }
        Ok(histogram)
    }

    /// Gather `data` along `axis` using an index tensor of the same rank
    /// (`gather_elements`). Used for searchsorted's `sorter`.
    fn tm_gather_axis(
        &mut self,
        data: GraphTensor,
        indices: GraphTensor,
        axis: usize,
    ) -> Result<GraphTensor> {
        let indices = indices.cast(DType::Int);
        ensure!(
            indices.rank() == data.rank(),
            "gather_elements: index rank {} must match data rank {}",
            indices.rank(),
            data.rank()
        );
        let out_dims = indices.dims();
        let mut coords = Vec::with_capacity(data.rank());
        for a in 0..data.rank() {
            if a == axis {
                coords.push(indices);
            } else {
                coords.push(self.axis_positions(&out_dims, a));
            }
        }
        Ok(data.gather(&coords))
    }

    /// Port of the old translator's `nonzero_static_from_truth`: the row-major
    /// coordinates of the true entries of `truth`, padded to `size` rows with
    /// `fill_value`. Expressed with the recorder's F32 div/mod path (the plan
    /// has no F64 binary arms).
    fn tm_nonzero_static_from_truth(
        &mut self,
        truth: GraphTensor,
        size: IntExpr,
        fill_value: i64,
    ) -> Result<GraphTensor> {
        let input_shape = truth.dims();
        let rank = input_shape.len();
        let numel: IntExpr = input_shape
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim);
        if rank == 0 {
            return Ok(self
                .cx
                .iota(vec![size, IntExpr::from(0usize)], |_| IntExpr::from(0))
                .cast(DType::I64));
        }
        if numel.to_usize() == Some(0) {
            return Ok(self
                .cx
                .constant_i64(fill_value)
                .expand_rhs(vec![size, IntExpr::from(rank)]));
        }

        let flat_truth = truth.flatten();
        let sorted = flat_truth.cast(DType::F32).stable_argsort(0, true);
        let count = flat_truth
            .cast(DType::F32)
            .sum(vec![0])
            .trunc_cast(DType::Int);
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
            .map(|i| {
                input_shape[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, dim| acc * *dim)
            })
            .collect();
        let flat_f = flat_indices.cast(DType::F32);
        let valid_f = valid.cast(DType::F32);
        let one = self.cx.constant_f32(1.0).expand_rhs(valid_f.dims());
        let invalid_f = one - valid_f;
        let fill_f = self
            .cx
            .constant_f32(fill_value as f32)
            .expand_rhs(valid_f.dims());
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
            .expand_rhs(vec![size, IntExpr::from(rank)]);
        for (axis, column) in columns.into_iter().enumerate() {
            let axis_column = self.cx.constant_i32(axis as i64).expand_rhs(rows.dims());
            result = result.scatter(&[rows, axis_column], column);
        }
        Ok(result)
    }

    // ---------------------------------------------------------------
    // Linear algebra
    // ---------------------------------------------------------------

    pub(super) fn translate_trilinear(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut inputs = [
            self.get_input_tensor(node, 0)?,
            self.get_input_tensor(node, 1)?,
            self.get_input_tensor(node, 2)?,
        ];
        // The three per-operand index lists say at which axis each operand's
        // size-1 dimensions are inserted. Negative dims are relative to the
        // post-insertion rank.
        for (input, argument) in inputs.iter_mut().zip(3..6) {
            for raw_dim in self.get_ints_arg(node, argument)? {
                let dim = util::normalize_dim(raw_dim, input.rank() + 1);
                *input = input.unsqueeze(dim);
            }
        }
        let (left, middle) = util::broadcast_binary(inputs[0], inputs[1]);
        let (product, right) = util::broadcast_binary(left * middle, inputs[2]);
        let rank = product.rank();
        let dimensions = self
            .get_ints_arg(node, 6)?
            .into_iter()
            .map(|dim| util::normalize_dim(dim, rank))
            .collect::<Vec<_>>();
        Ok((product * right).sum(dimensions))
    }

    pub(super) fn translate_addmv(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let compute_dtype = match output_dtype {
            DType::F16 | DType::Bf16 => DType::F32,
            dtype => dtype,
        };
        let input = self.get_input_tensor(node, 0)?.cast(compute_dtype);
        let matrix = self.get_input_tensor(node, 1)?.cast(compute_dtype);
        let vector = self.get_input_tensor(node, 2)?.cast(compute_dtype);
        ensure!(matrix.rank() == 2, "addmv matrix must be rank 2");
        ensure!(vector.rank() == 1, "addmv vector must be rank 1");

        let product = matrix.matmul(vector.unsqueeze(1)).squeeze(1);
        let input = self.tm_scale_by_named_scalar(node, "beta", input)?;
        let product = self.tm_scale_by_named_scalar(node, "alpha", product)?;
        let (input, product) = util::broadcast_binary(input, product);
        Ok((input + product).cast(output_dtype))
    }

    pub(super) fn translate_addbmm(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let compute_dtype = match output_dtype {
            DType::F16 | DType::Bf16 => DType::F32,
            dtype => dtype,
        };
        let input = self.get_input_tensor(node, 0)?.cast(compute_dtype);
        let batch1 = self.get_input_tensor(node, 1)?.cast(compute_dtype);
        let batch2 = self.get_input_tensor(node, 2)?.cast(compute_dtype);
        ensure!(batch1.rank() == 3, "addbmm batch1 must be rank 3");
        ensure!(batch2.rank() == 3, "addbmm batch2 must be rank 3");

        // CPU ATen evaluates addbmm as sequential fused addmm updates. A
        // single bmm+sum changes the F32 rounding order, and for BF16 it can
        // move by whole representable values because every update is rounded
        // back to BF16. Preserve the observable order when the batch length is
        // concrete; symbolic batches use the algebraically equivalent fallback.
        if matches!(output_dtype, DType::Bf16 | DType::F32)
            && let Some(batch_count) = batch1.dims()[0].to_usize()
        {
            let mut result = self.tm_scale_by_named_scalar(node, "beta", input)?;
            for batch in 0..batch_count {
                let lhs = batch1.slice_along(batch..batch + 1, 0).squeeze(0);
                let rhs = batch2.slice_along(batch..batch + 1, 0).squeeze(0);
                let product = self.tm_scale_by_named_scalar(node, "alpha", lhs.matmul(rhs))?;
                let (result_broadcast, product) =
                    util::broadcast_binary(result.cast(compute_dtype), product);
                result = (result_broadcast + product).cast(output_dtype);
            }
            return Ok(result.cast(output_dtype));
        }

        let product = batch1.matmul(batch2).sum(vec![0]);
        let input = self.tm_scale_by_named_scalar(node, "beta", input)?;
        let product = self.tm_scale_by_named_scalar(node, "alpha", product)?;
        let (input, product) = util::broadcast_binary(input, product);
        Ok((input + product).cast(output_dtype))
    }

    /// `aten.addmm.default`: `beta*self + alpha*(mat1 @ mat2)`.
    pub(super) fn translate_addmm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let mat1 = self.get_input_tensor(node, 1)?;
        let mat2 = self.get_input_tensor(node, 2)?;
        let beta = self
            .named_float_arg(node, "beta")
            .or_else(|| node.inputs.get(3).and_then(|input| input.arg.as_float()))
            .unwrap_or(1.0) as f32;
        let alpha = self
            .named_float_arg(node, "alpha")
            .or_else(|| node.inputs.get(4).and_then(|input| input.arg.as_float()))
            .unwrap_or(1.0) as f32;
        let (mat1, mat2) = util::ensure_same_dtype(mat1, mat2);
        let mm = mat1.matmul(mat2);
        let (input, mm) = util::ensure_same_dtype(input, mm);
        let (input, mm) = util::broadcast_binary(input, mm);
        Ok(input * beta + mm * alpha)
    }

    // ---------------------------------------------------------------
    // Reduction / search
    // ---------------------------------------------------------------

    /// `aten.histc.default` for the integer-bincount case.
    ///
    /// Qwen3-MoE's expert-balance layer calls
    /// `torch.histc(expert_ids.int(), bins=K, min=0, max=K-1)` to count how
    /// many tokens were routed to each expert. With those args every integer
    /// value `i ∈ [0, K-1]` maps to exactly bin `i`, and the result is
    /// equivalent to `torch.bincount`. We implement that case as a broadcast
    /// equality + sum:
    ///
    ///   counts[b] = sum_i (input[i] == b + min)   for b in [0, bins)
    ///
    /// More general histc bin widths (`bins != max - min + 1`) are not
    /// supported — the equality path would silently drop them, so we bail.
    pub(super) fn translate_histc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let bins_i64 = self
            .get_int_arg(node, 1)
            .context("histc: missing `bins` arg (#1)")?;
        let min = self.get_float_arg(node, 2).unwrap_or(0.0);
        let max = self.get_float_arg(node, 3).unwrap_or(0.0);

        ensure!(
            input.rank() == 1,
            "histc: only 1D input is supported, got {}D",
            input.rank()
        );
        ensure!(bins_i64 > 0, "histc: bins must be positive, got {bins_i64}");
        ensure!(
            (max - min - (bins_i64 - 1) as f64).abs() < 1e-6,
            "histc: only the bincount-equivalent case (bins == max - min + 1) is \
             supported; got bins={bins_i64}, min={min}, max={max}. Other cases \
             would need a general bin-width / right-edge-inclusion implementation.",
        );

        let bins_u = bins_i64 as usize;
        let n = input.dims()[0];

        let mut bins_arange = self.cx.arange(IntExpr::from(bins_u));
        if min != 0.0 {
            let min_i = min as i64;
            // `bins_arange` is Int; build the integer shift directly. A
            // float->int `cast` is refused by the recorder.
            let shift = self
                .cx
                .constant_i32(min_i as i32)
                .expand_rhs(bins_arange.dims());
            bins_arange += shift;
        }
        let bins_expanded = bins_arange.cast(input.dtype).expand_dim(1, n);
        let input_expanded = input.expand_dim(0, IntExpr::from(bins_u));
        let matches = input_expanded.eq(bins_expanded);

        let out_dtype = self.output_meta_dtype(node)?;
        Ok(matches.cast(out_dtype).sum(vec![1]))
    }

    /// `aten.histogram.bin_ct` / `aten.histogram.bins_tensor`: (hist, edges).
    /// Ported from the old translator's `translate_histogram`; the histogramdd
    /// family is intentionally not ported here (see the module report).
    pub(super) fn translate_histogram(&mut self, node: &Node) -> Result<()> {
        let dtype = self.output_meta_dtype(node)?;
        let input = self.get_input_tensor(node, 0)?.cast(dtype);
        let sample_count: IntExpr = input
            .dims()
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim);
        let values = util::reshape_tensor(input, &[sample_count]);
        let weight = self.tm_histogram_weight(node, sample_count, dtype)?;
        let density = Self::tm_histogram_density(node);
        let edges = if node.target.ends_with("bins_tensor") {
            self.get_input_tensor(node, 1)?.cast(dtype)
        } else {
            let bins_i64 = self.get_int_arg(node, 1)?;
            let bins = usize::try_from(bins_i64).context("histogram bins must be nonnegative")?;
            ensure!(bins > 0, "histogram bins must be positive");
            let range = Self::tm_optional_float_list(node, "range")
                .filter(|range| !range.is_empty())
                .map(|range| (range[0], range[1]));
            self.tm_uniform_histogram_edges(values, bins, range)
        };
        let histogram = self.tm_histogram_from_edges(&[values], &[edges], weight, density)?;
        self.tm_bind_outputs(node, vec![histogram, edges])
    }

    /// `aten._histogramdd_from_bin_cts.default`,
    /// `aten._histogramdd_from_bin_tensors.default`, and
    /// `aten._histogramdd_bin_edges.default` (single dispatch entry point;
    /// `node.target` selects the overload). The `from_bin_*` arms bind the
    /// single histogram output; the `_bin_edges` arm binds one edge tensor per
    /// dimension.
    pub(super) fn translate_histogramdd(&mut self, node: &Node) -> Result<()> {
        if node.target.ends_with("histogramdd_bin_edges.default") {
            return self.translate_histogramdd_bin_edges(node);
        }
        let tensor_edges = node.target.ends_with("from_bin_tensors.default");
        let dtype = self.output_meta_dtype(node)?;
        let input = self.get_input_tensor(node, 0)?.cast(dtype);

        let (edges, dimensions) = if tensor_edges {
            let names = node
                .inputs
                .get(1)
                .and_then(|input| input.arg.as_tensors())
                .context("histogramdd tensor bins must be a tensor list")?;
            let edges = names
                .iter()
                .map(|name| {
                    self.values
                        .get(&name.name)
                        .copied()
                        .map(|edge| edge.cast(dtype))
                        .ok_or_else(|| anyhow!("histogramdd: unknown tensor {}", name.name))
                })
                .collect::<Result<Vec<_>>>()?;
            let dimensions = edges.len();
            (edges, dimensions)
        } else {
            let bins = self.get_ints_arg(node, 1)?;
            let dimensions = bins.len();
            let (columns, _) = self.tm_histogram_columns(input, dimensions)?;
            (
                self.tm_histogramdd_uniform_edges(node, &columns, &bins)?,
                dimensions,
            )
        };
        let (columns, sample_count) = self.tm_histogram_columns(input, dimensions)?;
        let weight = self.tm_histogram_weight(node, sample_count, dtype)?;
        let histogram = self.tm_histogram_from_edges(
            &columns,
            &edges,
            weight,
            Self::tm_histogram_density(node),
        )?;
        self.tm_bind_outputs(node, vec![histogram])
    }

    /// `aten._histogramdd_bin_edges.default`: the edge tensor list only.
    pub(super) fn translate_histogramdd_bin_edges(&mut self, node: &Node) -> Result<()> {
        let bins = self.get_ints_arg(node, 1)?;
        let input = self.get_input_tensor(node, 0)?;
        let (columns, _) = self.tm_histogram_columns(input, bins.len())?;
        let edges = self.tm_histogramdd_uniform_edges(node, &columns, &bins)?;
        self.tm_bind_outputs(node, edges)
    }

    /// Lower bucketize/searchsorted by comparing each query with every entry
    /// in its sorted row and reducing the boolean insertion predicate. This is
    /// O(N) per query, but it is exact, fixed-shape, and uses only existing
    /// broadcast/comparison/reduction primitives.
    pub(super) fn translate_searchsorted(&mut self, node: &Node) -> Result<GraphTensor> {
        let bucketize = node.target.ends_with("bucketize.Tensor");
        let mut sorted = self.get_input_tensor(node, usize::from(bucketize))?;
        let query_index = usize::from(!bucketize);
        let query = if node.inputs[query_index].arg.as_tensor_name().is_some() {
            self.operand(&node.inputs[query_index])?
        } else {
            self.scalar(&node.inputs[query_index], sorted.dtype)?
        };

        ensure!(
            !sorted.dims().is_empty(),
            "searchsorted requires a sorted sequence with rank at least one"
        );
        let sorted_axis = sorted.rank() - 1;
        if let Some(sorter_input) = node
            .inputs
            .iter()
            .find(|input| input.name == "sorter")
            .filter(|input| input.arg.as_tensor_name().is_some())
        {
            let sorter = self.operand(sorter_input)?;
            sorted = self.tm_gather_axis(sorted, sorter, sorted_axis)?;
        }

        let (sorted, query) = util::ensure_same_dtype(sorted, query);
        let row_length = sorted.dims()[sorted_axis];
        let query_rank = query.rank();
        let (boundaries, queries) = if sorted.rank() == 1 {
            let mut boundaries = sorted;
            for (axis, size) in query.dims().into_iter().enumerate() {
                boundaries = boundaries.expand_dim(axis, size);
            }
            (boundaries, query.expand_dim(query_rank, row_length))
        } else {
            ensure!(
                query_rank == sorted.rank(),
                "batched searchsorted requires query and sequence ranks to match"
            );
            for axis in 0..sorted_axis {
                ensure!(
                    util::same_dim(query.dims()[axis], sorted.dims()[axis]),
                    "batched searchsorted prefix dimensions must match"
                );
            }
            let query_width = query.dims()[sorted_axis];
            (
                sorted.expand_dim(sorted_axis, query_width),
                query.expand_dim(query_rank, row_length),
            )
        };

        let right = node
            .inputs
            .iter()
            .find(|input| input.name == "right")
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false)
            || node.inputs.iter().any(|input| {
                input.name == "side"
                    && matches!(&input.arg, Argument::Other(value)
                        if value.as_str() == Some("right")
                        || value.get("as_string").and_then(|v| v.as_str()) == Some("right"))
            });
        let before = if right {
            boundaries.le(queries)
        } else {
            boundaries.lt(queries)
        };
        // The reference reducer has no i64 arm; count in i32 and widen.
        let mut result = before
            .cast(DType::Int)
            .sum(vec![query_rank])
            .cast(DType::I64);
        if matches!(
            query.dtype,
            DType::F16 | DType::Bf16 | DType::F32 | DType::F64
        ) {
            let nan = self.is_nan(query);
            let end = self
                .cx
                .constant_i32(row_length)
                .cast(DType::I64)
                .expand_rhs(result.dims());
            result = self.select(nan, end, result);
        }
        Ok(result.cast(self.output_meta_dtype(node)?))
    }

    // ---------------------------------------------------------------
    // Creation
    // ---------------------------------------------------------------

    /// `empty.memory_format`, `empty_permuted`, `empty_strided`,
    /// `new_empty_strided`, and `empty_like`.
    ///
    /// All allocate an uninitialised tensor whose contract is undefined for
    /// any read before a write, so a zero-fill is sound. `dtype`/`layout`/
    /// `device`/`pin_memory` and the memory_format/stride arguments do not
    /// affect the logical values, and the logical shape comes from the PT2
    /// output metadata (which is authoritative for all of these overloads).
    pub(super) fn translate_empty(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.output_meta_shape(node)?;
        let dtype = self.output_meta_dtype(node)?;
        Ok(self.full_tensor(shape, dtype, 0.0))
    }

    // ---------------------------------------------------------------
    // Triangular
    // ---------------------------------------------------------------

    pub(super) fn translate_tril(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, false)
    }

    pub(super) fn translate_triu(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, true)
    }

    fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        const TRIANGULAR_INPUT_ARG: usize = 0;
        const TRIANGULAR_DIAGONAL_ARG: usize = 1;
        let a = self.get_input_tensor(node, TRIANGULAR_INPUT_ARG)?;
        let diagonal = if node.inputs.len() > TRIANGULAR_DIAGONAL_ARG {
            self.get_int_arg(node, TRIANGULAR_DIAGONAL_ARG).unwrap_or(0) as i32
        } else {
            0
        };
        let dims = a.dims();
        ensure!(dims.len() >= 2, "tril/triu requires a matrix input");
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        let (r_val, c_val) = match (rows.to_usize(), cols.to_usize()) {
            (Some(r), Some(c)) => (r, c),
            _ => bail!("tril/triu requires concrete matrix dimensions"),
        };
        let size = r_val.max(c_val);
        let mask = if upper {
            self.cx.triu(size, diagonal)
        } else {
            self.cx.tril(size, diagonal)
        };
        let mask = if rows != cols {
            mask.slice_along(0..r_val, 0).slice_along(0..c_val, 1)
        } else {
            mask
        }
        .cast(a.dtype);
        let mut mask_expanded = mask;
        for i in (0..dims.len() - 2).rev() {
            mask_expanded = mask_expanded.expand_dim(0, dims[i]);
        }
        Ok(a * mask_expanded)
    }

    /// `aten.tril_indices.default` / `aten.triu_indices.default`: the
    /// row-major coordinates of the selected triangle, shape `[2, count]`.
    pub(super) fn translate_tri_indices(&mut self, node: &Node) -> Result<GraphTensor> {
        let rows_i64 = self.get_int_arg(node, 0)?;
        let columns_i64 = self.get_int_arg(node, 1)?;
        ensure!(
            rows_i64 >= 0 && columns_i64 >= 0,
            "triangular index dimensions must be nonnegative"
        );
        let offset = self.get_int_arg(node, 2).unwrap_or(0);
        let rows = IntExpr::from(rows_i64 as usize);
        let columns = IntExpr::from(columns_i64 as usize);
        let row = self.cx.arange(rows).cast(DType::I64).expand_dim(1, columns);
        let column = self.cx.arange(columns).cast(DType::I64).expand_dim(0, rows);
        let shifted_row = row + offset;
        let truth = if node.target.ends_with("tril_indices.default") {
            column.le(shifted_row)
        } else {
            column.ge(shifted_row)
        };
        let output_shape = self.output_meta_shape(node)?;
        ensure!(
            output_shape.len() == 2,
            "triangular indices must have rank two"
        );
        let coordinates = self.tm_nonzero_static_from_truth(truth, output_shape[1], 0)?;
        Ok(coordinates
            .permute(vec![1usize, 0])
            .cast(self.output_meta_dtype(node)?))
    }

    // ---------------------------------------------------------------
    // Grouped matmul (MoE expert dispatch)
    // ---------------------------------------------------------------

    /// Translate `aten._grouped_mm.default(input, weight, offs)` → `[S, N]`.
    ///
    /// `input` is `[S, K]` (tokens sorted by expert), `weight` is `[G, K, N]`
    /// (per-expert weights), `offs` is `[G]` cumulative token counts. Output
    /// `[S, N]` where token m (in group g s.t. `offs[g-1] <= m < offs[g]`) is
    /// multiplied by `weight[g]`.
    ///
    /// For each token we derive its expert id from `offs`, gather only that
    /// expert's `[K, N]` slice, and do a single per-token matmul. The gather
    /// pattern mirrors the rust qwen3_moe example's `gather_experts`, which
    /// the GLUMoE host-op fusion recognises. `offs` stays a runtime tensor, so
    /// one compiled graph handles any routing pattern.
    pub(super) fn translate_grouped_mm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let weight = self.get_input_tensor(node, 1)?;
        let offs = self.get_input_tensor(node, 2)?;
        let out_dtype = self.output_meta_dtype(node)?;

        ensure!(
            input.rank() == 2,
            "_grouped_mm: input must be 2D, got {}D",
            input.rank()
        );
        ensure!(
            weight.rank() == 3,
            "_grouped_mm: weight must be 3D, got {}D",
            weight.rank()
        );
        ensure!(
            offs.rank() == 1,
            "_grouped_mm: offs must be 1D, got {}D",
            offs.rank()
        );

        let s = input.dims()[0];
        let g = weight.dims()[0];
        let k = weight.dims()[1];
        let n = weight.dims()[2];

        // expert_id[m] = number of g s.t. m >= offs[g], clamped to [0, G-1].
        // Staying in Int throughout avoids an F32 round-trip, and the clamp
        // protects search-time profiling (dummy-1 inputs give offs=[1,...,1]).
        let _ = g
            .to_usize()
            .context("_grouped_mm: G (num_experts) must be concrete")?;
        let s_arange = self.cx.arange(s);
        let ge_int = s_arange
            .expand_dim(0, g)
            .ge(offs.expand_dim(1, s))
            .cast(DType::Int);
        let raw = ge_int.sum(vec![0]);
        let cap = self.cx.constant_i32(g - IntExpr::from(1)).expand_dim(0, s);
        let expert_id = raw.minimum(cap);

        // Flat gather index into weight (flattened G*K*N row-major):
        //   flat[m, k_, n_] = expert_id[m] * (K*N) + k_ * N + n_
        let io = k * n;
        let base = expert_id * io;
        let within = self.cx.iota((k, n), |c| c[0] * n + c[1]);
        let exp_base = base.expand_dim(1, k).expand_dim(2, n);
        let exp_within = within.expand_dim(0, s);
        let flat_idx = exp_base + exp_within;

        let weight_gathered = weight.gather1d(flat_idx).cast(out_dtype);
        let input = input.cast(out_dtype);

        // Per-token matmul: [S, 1, K] @ [S, K, N] → [S, 1, N] → [S, N].
        let result = input.unsqueeze(1).matmul(weight_gathered).squeeze(1);
        Ok(result.cast(out_dtype))
    }
}
