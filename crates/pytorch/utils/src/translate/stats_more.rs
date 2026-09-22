//! Distance / norm / histogram / segment-reduce / fractional-pool lowerings
//! (port batch 10).
//!
//! Each method lowers one ATen target (or family) onto the native recorder
//! frontend, following the parked HLIR translator's `reduction.rs`,
//! `tensor.rs`, and `pooling.rs` helpers but re-expressed with `dims()` /
//! `rank()` in place of `legacy_tracker_ref()` and with symbolic iota
//! coordinate functions in place of eager Int tensor arithmetic (plain Int
//! arithmetic is proof-gated on this branch).
//!
//! Multi-output ATen ops (`histogram`, `_histogramdd_bin_edges`,
//! `fractional_max_pool*`) bind every declared tensor output into the SSA
//! map themselves and return their primary result; the wired dispatch then
//! re-binds only the single returned value. That works for a one-output
//! node, and is exactly right if dispatch is changed to `return Ok(())`
//! after these arms (a change owned by `translate.rs`).

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::{Argument, Node};

/// Which distance composite a `dist`-family target is.
#[derive(Clone, Copy)]
pub(super) enum DistVariant {
    Dist,
    Cdist,
    Pdist,
}

/// True for storage dtypes that can carry a NaN.
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

/// Expand `[k]` / `[k, k, ..]` spatial lists to exactly `spatial` entries.
fn expand_spatial(values: &[i64], spatial: usize, default: i64) -> Result<Vec<usize>> {
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

/// Row-major strides as `IntExpr`s: `stride[i] = prod(dims[i+1..])`.
fn row_major_strides(dims: &[IntExpr]) -> Vec<IntExpr> {
    (0..dims.len())
        .map(|i| super::dim_arith::product_of_dims(dims[i + 1..].iter().copied()))
        .collect()
}

/// Extract the `reduce` string of a `segment_reduce` node.
fn segment_reduction_name(node: &Node) -> Result<String> {
    let input = node
        .inputs
        .iter()
        .find(|input| input.name == "reduce")
        .context("segment_reduce is missing its reduction name")?;
    match &input.arg {
        Argument::Other(value) => value
            .as_str()
            .or_else(|| value.get("as_string").and_then(|value| value.as_str()))
            .map(str::to_string)
            .context("segment_reduce reduction name is not a string"),
        _ => bail!("segment_reduce reduction name is not a string"),
    }
}

/// A PT2 float-list argument (arrives as `Argument::Other`).
fn optional_float_list(node: &Node, name: &str) -> Option<Vec<f64>> {
    let input = node.inputs.iter().find(|input| input.name == name)?;
    let Argument::Other(value) = &input.arg else {
        return None;
    };
    value
        .get("as_floats")?
        .as_array()?
        .iter()
        .map(serde_json::Value::as_f64)
        .collect()
}

/// Find the positional index of a named node input.
fn named_index(node: &Node, name: &str) -> Option<usize> {
    node.inputs.iter().position(|input| input.name == name)
}

impl Translator<'_> {
    // -----------------------------------------------------------------
    // Shared arithmetic helpers
    // -----------------------------------------------------------------

    /// `where(cond, a, b)` that keeps `a`'s dtype. Bool blends with the
    /// boolean primitives, Int blends in F32 and truncates back (Int
    /// arithmetic is proof-gated), floats use `cond`.
    ///
    /// Core LUM-804: the float `cond` and Int blend both leak `NaN * 0` /
    /// `inf * 0` from the *unselected* branch. Callers therefore only feed
    /// selection here when at least one branch is finite; where the parked
    /// code selected a non-finite branch away this port keeps the same
    /// selection and documents the limitation at the call site.
    fn blend_select(&mut self, cond: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = util::broadcast_binary(a, b);
        let (a, cond) = util::broadcast_binary(a, cond);
        match a.dtype {
            DType::Bool => {
                let yes = self.bool_and(cond, a);
                let not_cond = self.bool_not(cond);
                let no = self.bool_and(not_cond, b);
                self.bool_or(yes, no)
            }
            DType::Int | DType::I64 => {
                let dtype = a.dtype;
                let a32 = a.cast(DType::F32);
                let b32 = b.cast(DType::F32);
                let mask = cond.cast(DType::F32);
                let one = self.cx.constant_f32(1.0).expand_rhs(mask.dims());
                (a32 * mask + b32 * (one - mask)).trunc_cast(dtype)
            }
            _ => a.cond(cond, b),
        }
    }

    /// A dtype-preserving rank-0 constant scalar.
    fn scalar_value(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::Int | DType::I64 => self.cx.constant_i64(value as i64).cast(dtype),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        }
    }

    /// A dtype-preserving constant scalar expanded to `tensor`'s shape.
    /// Unlike `util::floating_scalar` this also handles integer targets
    /// (the histogram weight is accumulated in the Long output dtype).
    fn scalar_like(&mut self, tensor: GraphTensor, value: f64) -> GraphTensor {
        self.scalar_value(value, tensor.dtype)
            .expand_rhs(tensor.dims())
    }

    /// An explicit float -> int read (`trunc_cast`); every other conversion
    /// is an ordinary `cast`.
    fn lossy_cast(&mut self, tensor: GraphTensor, dtype: DType) -> GraphTensor {
        if tensor.dtype == dtype {
            return tensor;
        }
        if super::is_float(tensor.dtype) && super::is_int(dtype) {
            tensor.trunc_cast(dtype)
        } else {
            tensor.cast(dtype)
        }
    }

    /// A tensor operand by input name (`None` for absent / non-tensor).
    fn named_tensor_operand(&mut self, node: &Node, name: &str) -> Result<Option<GraphTensor>> {
        match named_index(node, name) {
            Some(index) => self.optional_tensor_operand(&node.inputs[index]),
            None => Ok(None),
        }
    }

    /// Re-insert reduced axes as size-1 extents (keepdim).
    fn restore_reduced_dims(
        &self,
        mut value: GraphTensor,
        axes: &[usize],
        keepdim: bool,
    ) -> GraphTensor {
        if keepdim {
            let mut sorted = axes.to_vec();
            sorted.sort_unstable();
            for axis in sorted {
                value = value.expand_dim(axis, 1usize);
            }
        }
        value
    }

    /// An int-list argument (or its symbolic spelling); a missing/unreadable
    /// argument is the empty list.
    fn int_list_arg(&self, node: &Node, index: usize) -> Result<Vec<i64>> {
        match node.inputs.get(index) {
            Some(input) if input.arg.as_ints().is_some() || input.arg.as_sym_ints().is_some() => {
                self.get_ints_arg(node, index)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// The p-norm of already-real-valued `magnitude` over `axes`.
    fn p_norm(&mut self, magnitude: GraphTensor, p: f64, axes: Vec<usize>) -> GraphTensor {
        if p == 0.0 {
            let zero = self.constant_like(magnitude, 0.0);
            let nonzero = self.bool_or(magnitude.lt(zero), magnitude.gt(zero));
            nonzero.cast(magnitude.dtype).sum(axes)
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

    /// `linalg_vector_norm` after the caller constructed real magnitudes.
    fn vector_norm_from_magnitude(
        &mut self,
        node: &Node,
        magnitude: GraphTensor,
    ) -> Result<GraphTensor> {
        let output_dtype = self.compute_dtype(node)?;
        let magnitude = magnitude.cast(output_dtype);
        let rank = magnitude.rank();
        let dims = self.int_list_arg(node, 2)?;
        let axes: Vec<usize> = if dims.is_empty() {
            (0..rank).collect()
        } else {
            dims.iter().map(|&d| util::normalize_dim(d, rank)).collect()
        };
        let keepdim = self
            .named_bool_arg(node, "keepdim")
            .or_else(|| node.inputs.get(3).and_then(|input| input.arg.as_bool()))
            .unwrap_or(false);
        let ord = self.get_float_arg(node, 1).unwrap_or(2.0);

        if (ord.is_infinite() || ord < 0.0)
            && axes
                .iter()
                .any(|&axis| magnitude.dims()[axis].to_usize() == Some(0))
        {
            bail!(
                "linalg_vector_norm order {ord} has no identity for an empty reduction dimension"
            );
        }

        let reduced = if ord == 0.0 {
            let zero = self.constant_like(magnitude, 0.0);
            let ordered = self.bool_or(magnitude.lt(zero), magnitude.gt(zero));
            let nonzero = if dtype_can_contain_nan(magnitude.dtype) {
                let nan = self.is_nan(magnitude);
                self.bool_or(ordered, nan)
            } else {
                ordered
            };
            nonzero.cast(output_dtype).sum(axes.clone())
        } else if ord == 1.0 {
            magnitude.sum(axes.clone())
        } else if ord == 2.0 {
            magnitude.square().sum(axes.clone()).sqrt()
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

    // -----------------------------------------------------------------
    // Distances and norms
    // -----------------------------------------------------------------

    pub(super) fn translate_dist(
        &mut self,
        node: &Node,
        variant: DistVariant,
    ) -> Result<GraphTensor> {
        match variant {
            DistVariant::Dist => {
                let lhs = self.get_input_tensor(node, 0)?;
                let rhs = self.get_input_tensor(node, 1)?;
                let (lhs, rhs) = util::ensure_same_dtype(lhs, rhs);
                let (lhs, rhs) = util::broadcast_binary(lhs, rhs);
                let difference = self.rounded_difference(lhs - rhs);
                let magnitude = self.real_abs(difference).cast(self.compute_dtype(node)?);
                let p = self.get_float_arg(node, 2).unwrap_or(2.0);
                let axes = (0..magnitude.rank()).collect();
                Ok(self.p_norm(magnitude, p, axes))
            }
            DistVariant::Cdist => {
                let lhs = self.get_input_tensor(node, 0)?;
                let rhs = self.get_input_tensor(node, 1)?;
                anyhow::ensure!(
                    lhs.rank() >= 2 && rhs.rank() >= 2,
                    "cdist inputs must be matrices"
                );
                let (mut lhs, mut rhs) = util::ensure_same_dtype(lhs, rhs);
                let output_shape = self.output_meta_shape(node)?;
                let (lhs_rank, rhs_rank) = (lhs.rank(), rhs.rank());
                let feature = lhs.dims()[lhs_rank - 1];
                anyhow::ensure!(
                    feature == rhs.dims()[rhs_rank - 1],
                    "cdist feature dimensions must match"
                );
                // lhs [.., P, M] -> [.., P, R, M]; rhs [.., R, M] -> [.., P, R, M].
                let rows = lhs.dims()[lhs_rank - 2];
                lhs = lhs.expand_dim(lhs_rank - 1, rhs.dims()[rhs_rank - 2]);
                rhs = rhs.expand_dim(rhs_rank - 2, rows);
                let mut pair_shape = output_shape;
                pair_shape.push(feature);
                lhs = lhs.expand(pair_shape.clone());
                rhs = rhs.expand(pair_shape);
                let difference = self.rounded_difference(lhs - rhs);
                let magnitude = self.real_abs(difference).cast(self.compute_dtype(node)?);
                let p = self.get_float_arg(node, 2)?;
                Ok(self.p_norm(magnitude, p, vec![magnitude.rank() - 1]))
            }
            DistVariant::Pdist => {
                let input = self.get_input_tensor(node, 0)?;
                anyhow::ensure!(input.rank() == 2, "pdist input must be a matrix");
                let rows = input.dims()[0];
                let columns = input.dims()[1];
                let output_shape = self.output_meta_shape(node)?;
                let pairs = output_shape[0];
                let k = self.cx.arange(pairs).cast(DType::F32);
                let rows_f = self
                    .cx
                    .constant_i32(rows)
                    .cast(DType::F32)
                    .expand_rhs(k.dims());
                let discriminant = k * -8.0f32 + rows_f * (rows_f - 1.0f32) * 4.0f32 - 7.0f32;
                let i = (rows_f
                    - 2.0f32
                    - (discriminant.sqrt() * 0.5f32 - 0.5f32)
                        .trunc_cast(DType::Int)
                        .cast(DType::F32))
                .trunc_cast(DType::Int);
                let i_f = i.cast(DType::F32);
                let j = (k + i_f + 1.0f32 - rows_f * (rows_f - 1.0f32) * 0.5f32
                    + (rows_f - i_f) * (rows_f - i_f - 1.0f32) * 0.5f32)
                    .trunc_cast(DType::Int);
                let i = i.expand_dim(1, columns);
                let j = j.expand_dim(1, columns);
                let pair_shape = vec![pairs, columns];
                let column_positions = self.axis_positions(&pair_shape, 1);
                let left = input.gather(&[i, column_positions]);
                let right = input.gather(&[j, column_positions]);
                let difference = self.rounded_difference(left - right);
                let magnitude = self.real_abs(difference).cast(self.compute_dtype(node)?);
                let p = self.get_float_arg(node, 1).unwrap_or(2.0);
                Ok(self.p_norm(magnitude, p, vec![1]))
            }
        }
    }

    /// torch forms a distance's pairwise difference in the operand dtype
    /// and accumulates the norm wide: round the difference to the common
    /// dtype before it enters the F32 norm.
    fn rounded_difference(&self, difference: GraphTensor) -> GraphTensor {
        match self.opmath {
            Some(opmath) => {
                super::convert(super::convert(difference, opmath.common), opmath.compute)
            }
            None => difference,
        }
    }

    pub(super) fn translate_trilinear(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut inputs = [
            self.get_input_tensor(node, 0)?,
            self.get_input_tensor(node, 1)?,
            self.get_input_tensor(node, 2)?,
        ];
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

    pub(super) fn translate_linalg_vector_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.get_input_tensor(node, 0)?;
        let magnitude = self.real_abs(value);
        self.vector_norm_from_magnitude(node, magnitude)
    }

    // -----------------------------------------------------------------
    // Histograms
    // -----------------------------------------------------------------

    pub(super) fn translate_histc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let bins_i64 = self
            .get_int_arg(node, 1)
            .context("histc: missing `bins` arg (#1)")?;
        let min = self.get_float_arg(node, 2).unwrap_or(0.0);
        let max = self.get_float_arg(node, 3).unwrap_or(0.0);

        anyhow::ensure!(
            input.rank() == 1,
            "histc: only 1D input is supported, got {}D",
            input.rank()
        );
        anyhow::ensure!(bins_i64 > 0, "histc: bins must be positive, got {bins_i64}");
        // Bincount-equivalent case: one integer value per bin. Wider / offset
        // bins would need a general right-edge-inclusion implementation and
        // would be silently wrong through the equality path.
        anyhow::ensure!(
            (max - min - (bins_i64 - 1) as f64).abs() < 1e-6,
            "histc: only the bincount-equivalent case (bins == max - min + 1) is \
             supported; got bins={bins_i64}, min={min}, max={max}. Other cases \
             would need a general bin-width / right-edge-inclusion implementation."
        );

        let bins = bins_i64 as usize;
        let n = input.dims()[0];

        // arange(bins), optionally shifted by an integer `min`, compared for
        // equality against the broadcast input.
        let min_i = min as i64;
        let bins_arange = self.cx.iota(bins, move |c| c[0] + min_i);
        let bins_expanded = bins_arange.cast(input.dtype).expand_dim(1, n);
        let input_expanded = input.expand_dim(0, IntExpr::from(bins));
        let matches = input_expanded.eq(bins_expanded);

        let out_dtype = self.output_meta_dtype(node)?;
        Ok(matches.cast(out_dtype).sum(1))
    }

    pub(super) fn translate_histogram(
        &mut self,
        node: &Node,
        bins_tensor: bool,
    ) -> Result<GraphTensor> {
        let out_dtype = self.output_meta_dtype(node)?;
        // Columns/edges keep the input's own dtype: the histogram *count* is
        // the Long output, but bin edges are only meaningful in the input's
        // dtype. The parked translator cast the input to the output dtype,
        // which truncates fractional coordinates; this port keeps them.
        let input = self.get_input_tensor(node, 0)?;
        let sample_count = super::dim_arith::product_of_dims(input.dims());
        let values = util::reshape_tensor(input, &[sample_count]);
        let weight = self.histogram_weight(node, sample_count, out_dtype)?;
        let density = Self::histogram_density(node);
        let edges = if bins_tensor {
            self.get_input_tensor(node, 1)?
        } else {
            let bins = usize::try_from(self.get_int_arg(node, 1)?)
                .context("histogram bins must be nonnegative")?;
            anyhow::ensure!(bins > 0, "histogram bins must be positive");
            let range = optional_float_list(node, "range")
                .filter(|range| !range.is_empty())
                .map(|range| (range[0], range[1]));
            self.uniform_histogram_edges(values, bins, range)
        };
        let histogram = self.histogram_from_edges(&[values], &[edges], weight, density)?;
        self.bind_declared_prefix(node, &[histogram, edges])?;
        Ok(histogram)
    }

    pub(super) fn translate_histogramdd(&mut self, node: &Node) -> Result<GraphTensor> {
        if node.target.ends_with("_histogramdd_bin_edges.default") {
            let bins = self.get_ints_arg(node, 1)?.to_vec();
            let input = self.get_input_tensor(node, 0)?;
            let (columns, _) = self.histogram_columns(input, bins.len())?;
            let edges = self.histogramdd_uniform_edges(node, &columns, &bins)?;
            self.bind_declared_prefix(node, &edges)?;
            return edges
                .into_iter()
                .next()
                .context("_histogramdd_bin_edges produced no edges");
        }

        let tensor_edges = node.target.ends_with("_bin_tensors.default");
        let out_dtype = self.output_meta_dtype(node)?;
        let input = self.get_input_tensor(node, 0)?;
        let (edges, dimensions) = if tensor_edges {
            let names = node.inputs[1]
                .arg
                .as_tensors()
                .context("histogramdd tensor bins must be a tensor list")?;
            let mut edges = Vec::with_capacity(names.len());
            for name in names {
                let edge = *self.values.get(&name.name).with_context(|| {
                    format!("histogramdd edge {} was never produced", name.name)
                })?;
                edges.push(edge);
            }
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
        let weight = self.histogram_weight(node, sample_count, out_dtype)?;
        let histogram =
            self.histogram_from_edges(&columns, &edges, weight, Self::histogram_density(node))?;
        self.bind_single_declared(node, histogram)?;
        Ok(histogram)
    }

    /// Bind the first `values.len()` declared tensor outputs (multi-output
    /// ATen ops whose primary result is what the wired dispatch returns).
    fn bind_declared_prefix(&mut self, node: &Node, values: &[GraphTensor]) -> Result<()> {
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
            names.len() <= values.len(),
            "`{}` declares {} tensor outputs but the lowering produced {}",
            node.target,
            names.len(),
            values.len()
        );
        for (name, value) in names.into_iter().zip(values.iter().copied()) {
            self.values.insert(name, value);
        }
        Ok(())
    }

    /// Bind a single declared tensor output (if any).
    fn bind_single_declared(&mut self, node: &Node, value: GraphTensor) -> Result<()> {
        if let Some(name) = Self::tensor_output_names(node).into_iter().next() {
            self.values.insert(name, value);
        }
        Ok(())
    }

    fn histogram_weight(
        &mut self,
        node: &Node,
        sample_count: IntExpr,
        dtype: DType,
    ) -> Result<GraphTensor> {
        if let Some(weight) = self.named_tensor_operand(node, "weight")? {
            let weight = self.lossy_cast(weight, dtype);
            return Ok(util::reshape_tensor(weight, &[sample_count]));
        }
        Ok(self.full_tensor(vec![sample_count], dtype, 1.0))
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
                self.scalar_value(lower, values.dtype),
                self.scalar_value(upper, values.dtype),
            )
        } else {
            (values.min(0), values.max(0))
        };
        // A degenerate range is widened by half a unit, matching torch.
        let equal = self.is_zero(upper - lower);
        lower = self.blend_select(equal, lower - 0.5f32, lower);
        upper = self.blend_select(equal, upper + 0.5f32, upper);
        let position = self.cx.arange(bins + 1).cast(values.dtype);
        let edge_shape = position.dims();
        let lower_expanded = lower.expand_rhs(edge_shape.clone());
        let width = (upper - lower).expand_rhs(edge_shape);
        lower_expanded + position * width / bins as f32
    }

    fn histogramdd_uniform_edges(
        &mut self,
        node: &Node,
        columns: &[GraphTensor],
        bins: &[i64],
    ) -> Result<Vec<GraphTensor>> {
        let range = optional_float_list(node, "range").filter(|range| !range.is_empty());
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

    fn histogram_columns(
        &mut self,
        input: GraphTensor,
        dimensions: usize,
    ) -> Result<(Vec<GraphTensor>, IntExpr)> {
        anyhow::ensure!(dimensions > 0, "histogram dimension count must be positive");
        anyhow::ensure!(
            !input.dims().is_empty(),
            "histogramdd input must have a coordinate dimension"
        );
        let input_dims = input.dims();
        anyhow::ensure!(
            input_dims.last().and_then(|dim| dim.to_usize()) == Some(dimensions),
            "histogramdd coordinate dimension does not match bins"
        );
        let sample_count =
            super::dim_arith::product_of_dims(input_dims[..input_dims.len() - 1].iter().copied());
        let matrix = util::reshape_tensor(input, &[sample_count, IntExpr::from(dimensions)]);
        let columns = (0..dimensions)
            .map(|dimension| matrix.slice_along(dimension..dimension + 1, 1).squeeze(1))
            .collect();
        Ok((columns, sample_count))
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
        // NaN never lands in a bin (torch drops it), and the boolean lattice
        // here is exact, so no non-finite value is selected away.
        let mut membership = self
            .cx
            .constant_i32(1)
            .cast(DType::Bool)
            .expand_rhs(full_shape.clone());
        for (dimension, (column, edge)) in columns.iter().zip(edges).enumerate() {
            let bins = bin_shape[dimension];
            let left = edge.slice_along(IntExpr::from(0usize)..bins, 0);
            let right = edge.slice_along(IntExpr::from(1usize)..(bins + IntExpr::from(1usize)), 0);

            let mut column_shape = vec![IntExpr::from(1); columns.len() + 1];
            column_shape[0] = sample_count;
            let mut column = util::reshape_tensor(*column, &column_shape);
            column = column.expand(full_shape.clone());
            let mut edge_shape = vec![IntExpr::from(1); columns.len() + 1];
            edge_shape[dimension + 1] = bins;
            let left = util::reshape_tensor(left, &edge_shape).expand(full_shape.clone());
            let right = util::reshape_tensor(right, &edge_shape).expand(full_shape.clone());

            let below_right = column.lt(right);
            let equal_right = self.is_zero(column - right);
            let mut last_position = self.cx.arange(bins);
            let last_index = self
                .cx
                .constant_i32(bins - IntExpr::from(1usize))
                .expand_rhs(last_position.dims());
            last_position = last_position.eq(last_index);
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
        let mut histogram = (membership.cast(weight.dtype) * weight).sum(0);
        if density {
            let axes = (0..bin_shape.len()).collect::<Vec<_>>();
            let total = histogram.sum(axes).expand_rhs(bin_shape.clone());
            let mut volume = self
                .scalar_like(histogram, 1.0)
                .expand_rhs(bin_shape.clone());
            for (dimension, edge) in edges.iter().enumerate() {
                let bins = bin_shape[dimension];
                let widths = edge
                    .slice_along(IntExpr::from(1usize)..(bins + IntExpr::from(1usize)), 0)
                    - edge.slice_along(IntExpr::from(0usize)..bins, 0);
                let mut width_shape = vec![IntExpr::from(1); edges.len()];
                width_shape[dimension] = bins;
                let widths = util::reshape_tensor(widths, &width_shape).expand(bin_shape.clone());
                volume *= widths;
            }
            histogram = histogram / total / volume;
        }
        Ok(histogram)
    }

    // -----------------------------------------------------------------
    // Segment reduce
    // -----------------------------------------------------------------

    pub(super) fn translate_segment_reduce(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.get_input_tensor(node, 0)?;
        let reduction = segment_reduction_name(node)?;
        let axis = self.named_int_arg(node, "axis").unwrap_or(0);
        let axis = util::normalize_dim(axis, data.rank());
        let output_shape = self.output_meta_shape(node)?;
        let segment_count = output_shape[axis];
        let input_count = data.dims()[axis];

        let lengths = self.named_tensor_operand(node, "lengths")?;
        let offsets = self.named_tensor_operand(node, "offsets")?;
        let (starts, ends) = if let Some(lengths) = lengths {
            // All the segment bookkeeping runs in F32: the candidate axis is
            // short and this avoids proof-gated Int cumsum/subtract.
            let lengths = lengths.cast(DType::F32);
            let ends = lengths.cumsum(axis);
            (ends - lengths, ends)
        } else if let Some(offsets) = offsets {
            let offsets = offsets.cast(DType::F32);
            (
                offsets.slice_along(IntExpr::from(0usize)..segment_count, axis),
                offsets.slice_along(
                    IntExpr::from(1usize)..(segment_count + IntExpr::from(1usize)),
                    axis,
                ),
            )
        } else {
            bail!("segment_reduce requires lengths or offsets");
        };

        let mut pair_shape = output_shape.clone();
        pair_shape.insert(axis + 1, input_count);
        let mut starts = starts.expand_dim(axis + 1, input_count);
        let mut ends = ends.expand_dim(axis + 1, input_count);
        for suffix in data.dims()[axis + 1..].iter().copied() {
            starts = starts.expand_dim(starts.rank(), suffix);
            ends = ends.expand_dim(ends.rank(), suffix);
        }
        let candidate_axis = axis + 1;
        let positions = self
            .cx
            .iota(pair_shape, |c| c[candidate_axis])
            .cast(DType::F32);
        let membership = self.bool_and(positions.ge(starts), positions.lt(ends));
        let expanded = data.expand_dim(axis, segment_count);
        let count = membership.cast(DType::F32).sum(candidate_axis);

        let initial = named_index(node, "initial")
            .and_then(|index| self.get_number_arg(node, index).ok())
            .map(|value| self.full_tensor(output_shape.clone(), data.dtype, value));
        let has_initial = initial.is_some();
        let zero = self.full_tensor(expanded.dims(), data.dtype, 0.0);
        let one = self.full_tensor(expanded.dims(), data.dtype, 1.0);
        match reduction.as_str() {
            "sum" | "mean" => {
                // LUM-804: the membership mask multiplies the expanded data,
                // so a NaN/inf in an out-of-segment lane leaks through
                // `nonfinite * 0`. Documented limitation of the mask-select
                // form; segment_reduce inputs are finite in the common path.
                let mut sum = self
                    .blend_select(membership, expanded, zero)
                    .sum(candidate_axis);
                if let Some(initial) = initial {
                    sum += initial;
                }
                if reduction == "sum" {
                    return Ok(sum);
                }
                if has_initial {
                    // Empty segments keep the initial value: replace a zero
                    // count by one arithmetically (no branch selection).
                    let zero_count = self.cx.constant_f32(0.0).expand_rhs(count.dims());
                    let nonempty = count.gt(zero_count);
                    let empty = self.bool_not(nonempty).cast(DType::F32);
                    let safe_count = count + empty;
                    Ok(sum / safe_count.cast(data.dtype))
                } else {
                    // An empty segment is 0/0 -> NaN, exactly torch's mean.
                    Ok(sum / count.cast(data.dtype))
                }
            }
            "prod" => {
                let selected = self.blend_select(membership, expanded, one);
                let magnitude = self.real_abs(selected).prod(candidate_axis);
                let negative_count = self.signbit(selected).cast(DType::F32).sum(candidate_axis);
                // odd in {0, 1}; sign multiplication avoids a select (and
                // preserves infinities instead of leaking inf * 0).
                let odd = negative_count - (negative_count * 0.5f32).floor() * 2.0f32;
                let sign = 1.0f32 - 2.0f32 * odd;
                let mut product = magnitude * sign.cast(magnitude.dtype);
                if let Some(initial) = initial {
                    product *= initial;
                }
                Ok(product)
            }
            "max" | "min" => {
                // LUM-804: filling the out-of-segment lanes selects a
                // non-finite identity away from a possibly non-finite data
                // value (same mask-select limitation as the sum path).
                let fill = self.scalar_like(
                    expanded,
                    if reduction == "max" {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    },
                );
                let values = self.blend_select(membership, expanded, fill);
                let mut result = if reduction == "max" {
                    values.max(candidate_axis)
                } else {
                    values.min(candidate_axis)
                };
                if let Some(initial) = initial {
                    // The fill is the identity, so an empty segment lands on
                    // the initial value without a branch select.
                    result = if reduction == "max" {
                        result.maximum(initial)
                    } else {
                        result.minimum(initial)
                    };
                }
                Ok(result)
            }
            other => bail!("unsupported segment_reduce reduction {other}"),
        }
    }

    // -----------------------------------------------------------------
    // Fractional max pool
    // -----------------------------------------------------------------

    /// Read a pooling arg by name, falling back to its positional slot.
    fn stats_pool_ints(&self, node: &Node, name: &str, index: usize) -> Result<Vec<i64>> {
        let index = named_index(node, name).unwrap_or(index);
        self.get_ints_arg(node, index)
    }

    /// Max over the trailing flattened kernel axis, selecting the top-1 entry
    /// with one stable-argsort pass. `candidates`/`logical_indices` carry
    /// `[prefix..., out_spatial..., flatten]`.
    fn stats_select_pool_max(
        &mut self,
        mut candidates: GraphTensor,
        mut logical_indices: GraphTensor,
        output_rank: usize,
    ) -> Result<[GraphTensor; 2]> {
        while candidates.rank() > output_rank + 1 {
            let last = candidates.rank();
            candidates = candidates.merge_dims(last - 2, last - 1);
            logical_indices = logical_indices.merge_dims(last - 2, last - 1);
        }
        let key = match candidates.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::Bf16 | DType::TF32 => candidates,
            DType::Bool => candidates.cast(DType::F32),
            other => bail!(
                "max pooling on {other:?} inputs is not ported: the stable-argsort \
                 selection needs a float key"
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
        let picked = logical_indices.gather(&coordinates);
        let picked = if picked.dtype == DType::I64 {
            picked
        } else {
            picked.cast(DType::I64)
        };
        Ok([values, picked])
    }

    pub(super) fn translate_fractional_max_pool(
        &mut self,
        node: &Node,
        rank: usize,
    ) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let random_samples = self.get_input_tensor(node, 3)?;
        let input_rank = input.rank();
        anyhow::ensure!(
            input_rank == rank + 1 || input_rank == rank + 2,
            "fractional max pool input rank is invalid"
        );
        anyhow::ensure!(
            random_samples.rank() == 3 && random_samples.dims()[2].to_usize() == Some(rank),
            "fractional max pool random_samples must have shape [N, C, spatial_rank]"
        );
        let kernel = expand_spatial(&self.stats_pool_ints(node, "kernel_size", 1)?, rank, 0)?;
        anyhow::ensure!(
            kernel.iter().all(|&size| size > 0),
            "fractional max pool kernel must be positive"
        );
        let output_shape = self.output_meta_shape(node)?;
        let prefix_rank = input_rank - rank;
        let input_shape = input.dims();
        let output_spatial = output_shape[prefix_rank..].to_vec();
        let input_spatial = input_shape[prefix_rank..].to_vec();
        let mut window_shape = output_shape.clone();
        window_shape.extend(kernel.iter().copied().map(IntExpr::from));

        let strides = row_major_strides(&input_shape);
        // Prefix (batch/channel) flat contributions are a pure coordinate
        // function, so no Int tensor arithmetic enters the model.
        let mut flat = self
            .cx
            .iota(window_shape.clone(), |c| {
                (0..prefix_rank).fold(IntExpr::from(0), |acc, axis| acc + c[axis] * strides[axis])
            })
            .cast(DType::F32);
        let mut logical = self
            .cx
            .constant_i32(0)
            .cast(DType::F32)
            .expand_rhs(window_shape.clone());

        for spatial in 0..rank {
            let sample_lane = if rank == 2 {
                rank - 1 - spatial
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
            // Every select below is between finite values, so the LUM-804
            // `nonfinite * 0` leak cannot trigger here.
            let output_is_one = self.is_zero(denominator);
            let one = self.constant_like(denominator, 1.0);
            let safe_denominator = self.blend_select(output_is_one, one, denominator);
            let alpha = numerator / safe_denominator;
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
            let final_or_ordinary =
                self.blend_select(final_position, terminal_start, ordinary_start);
            let start = self.blend_select(output_is_one, terminal_start, final_or_ordinary);

            let mut coordinate = start;
            for size in kernel.iter().copied() {
                coordinate = coordinate.expand_dim(coordinate.rank(), size);
            }
            let kernel_axis = input_rank + spatial;
            coordinate = coordinate
                + self
                    .axis_positions(&window_shape, kernel_axis)
                    .cast(coordinate.dtype);
            let stride = self
                .cx
                .constant_i32(strides[prefix_rank + spatial])
                .cast(DType::F32)
                .expand_rhs(coordinate.dims());
            flat += coordinate * stride;

            let spatial_stride =
                super::dim_arith::product_of_dims(input_spatial[spatial + 1..].iter().copied());
            let stride = self
                .cx
                .constant_i32(spatial_stride)
                .cast(DType::F32)
                .expand_rhs(coordinate.dims());
            logical += coordinate * stride;
        }

        let flat_indices = flat.trunc_cast(DType::Int);
        let logical_indices = logical.trunc_cast(DType::Int);
        let candidates = util::reshape_tensor(
            input.flatten().gather(&[flat_indices.flatten()]),
            &window_shape,
        );
        let [values, indices] =
            self.stats_select_pool_max(candidates, logical_indices, input_rank)?;
        self.bind_declared_prefix(node, &[values, indices])?;
        Ok(values)
    }

    // -----------------------------------------------------------------
    // Max pool backward
    // -----------------------------------------------------------------

    pub(super) fn translate_max_pool_backward(&mut self, node: &Node) -> Result<GraphTensor> {
        let updates = self.get_input_tensor(node, 0)?;
        let input = self.get_input_tensor(node, 1)?;
        let indices = self.get_input_tensor(node, 7)?.cast(DType::Int);
        let input_shape = input.dims();
        let prefix_rank = input_shape.len() - 2;
        let strides = row_major_strides(&input_shape);
        let output_shape = updates.dims();
        // Indices are flattened-within-plane; only the batch/channel prefix
        // needs a stride contribution.
        let base = self
            .cx
            .iota(output_shape.clone(), |c| {
                (0..prefix_rank).fold(IntExpr::from(0), |acc, axis| acc + c[axis] * strides[axis])
            })
            .cast(DType::F32);
        let destinations = (base + indices.cast(DType::F32))
            .trunc_cast(DType::Int)
            .flatten();

        let count = super::dim_arith::product_of_dims(output_shape.iter().copied())
            .to_usize()
            .context("max_pool2d_with_indices_backward requires a concrete output element count")?;
        let flat_updates = updates.flatten();
        let mut output = self
            .full_tensor(input_shape.clone(), input.dtype, 0.0)
            .flatten();
        // Duplicate destinations (overlapping windows selecting the same max)
        // must accumulate; the recorder's `scatter` is overwrite-only, so one
        // read/modify/write step per update element, as in the parked
        // translator's scatter-reduce helper.
        for step in 0..count {
            let destination = destinations.slice_along(step..step + 1, 0);
            let update = flat_updates.slice_along(step..step + 1, 0);
            let current = output.gather(&[destination]);
            output = output.scatter(&[destination], current + update);
        }
        Ok(util::reshape_tensor(output, &input_shape))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::pt2_parser::ParsedPT2;
    use crate::pt2_schema::{
        Argument, BoolArg, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntArg,
        IntsArg, Node, NodeInput, Signature, TensorArg, TensorMeta, TensorName, TensorRef,
        TensorsArg,
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

    fn tensors_arg(names: &[&str]) -> Argument {
        Argument::Tensors(TensorsArg {
            as_tensors: names
                .iter()
                .map(|name| TensorName {
                    name: name.to_string(),
                })
                .collect(),
        })
    }

    fn string_arg(value: &str) -> Argument {
        Argument::Other(serde_json::Value::String(value.to_string()))
    }

    fn input(name: &str, arg: Argument) -> NodeInput {
        NodeInput {
            name: name.to_string(),
            arg,
            kind: 1,
        }
    }

    fn sizes(values: &[i64]) -> Vec<DimSize> {
        values
            .iter()
            .map(|value| DimSize::Int(DimInt { as_int: *value }))
            .collect()
    }

    fn ints(name: &str, values: &[i64]) -> NodeInput {
        input(
            name,
            Argument::Ints(IntsArg {
                as_ints: values.to_vec(),
            }),
        )
    }

    fn scalar_int(name: &str, value: i64) -> NodeInput {
        input(name, Argument::Int(IntArg { as_int: value }))
    }

    fn scalar_float(name: &str, value: f64) -> NodeInput {
        input(name, Argument::Float(FloatArg { as_float: value }))
    }

    fn scalar_bool(name: &str, value: bool) -> NodeInput {
        input(name, Argument::Bool(BoolArg { as_bool: value }))
    }

    fn node(target: &str, inputs: Vec<NodeInput>, outputs: &[&str]) -> Node {
        Node {
            target: target.to_string(),
            inputs,
            outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
        }
    }

    /// Translate the given nodes over the declared tensors, returning whether
    /// translation succeeded. Every tensor operand must be listed in `inputs`
    /// (graph order) and have a `(dtype, shape)` entry in `tensors`.
    fn translates(
        nodes: Vec<Node>,
        tensors: &[(&str, u32, Vec<i64>)],
        inputs: &[&str],
        outputs: &[&str],
    ) -> bool {
        let mut tensor_values = HashMap::new();
        for (name, dtype, shape) in tensors {
            tensor_values.insert(
                name.to_string(),
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
                    nodes,
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
        let parsed = ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        };
        crate::translate::translate(&parsed).is_ok()
    }

    const F32: u32 = 7;
    const I64: u32 = 5;

    #[test]
    fn dist_family_records() {
        // dist.default: full reduction, p defaulted to 2.
        assert!(translates(
            vec![node(
                "torch.ops.aten.dist.default",
                vec![
                    input("self", tensor_arg("x")),
                    input("other", tensor_arg("y")),
                    scalar_float("p", 2.0),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![2, 3]),
                ("y", F32, vec![2, 3]),
                ("out", F32, vec![]),
            ],
            &["x", "y"],
            &["out"],
        ));
        // _cdist_forward.default over [M, F] x [N, F] -> [M, N].
        assert!(translates(
            vec![node(
                "torch.ops.aten._cdist_forward.default",
                vec![
                    input("x1", tensor_arg("x")),
                    input("x2", tensor_arg("y")),
                    scalar_float("p", 2.0),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![2, 3]),
                ("y", F32, vec![4, 3]),
                ("out", F32, vec![2, 4]),
            ],
            &["x", "y"],
            &["out"],
        ));
        // _pdist_forward.default over [N, F] -> [N*(N-1)/2].
        assert!(translates(
            vec![node(
                "torch.ops.aten._pdist_forward.default",
                vec![input("self", tensor_arg("x")), scalar_float("p", 2.0)],
                &["out"],
            )],
            &[("x", F32, vec![3, 4]), ("out", F32, vec![3])],
            &["x"],
            &["out"],
        ));
        // Batched cdist over [B, M, F] x [B, N, F] -> [B, M, N].
        assert!(translates(
            vec![node(
                "torch.ops.aten._cdist_forward.default",
                vec![
                    input("x1", tensor_arg("x")),
                    input("x2", tensor_arg("y")),
                    scalar_float("p", 2.0),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![2, 2, 3]),
                ("y", F32, vec![2, 4, 3]),
                ("out", F32, vec![2, 2, 4]),
            ],
            &["x", "y"],
            &["out"],
        ));
    }

    #[test]
    fn trilinear_and_vector_norm_record() {
        assert!(translates(
            vec![node(
                "torch.ops.aten._trilinear.default",
                vec![
                    input("i1", tensor_arg("a")),
                    input("i2", tensor_arg("b")),
                    input("i3", tensor_arg("c")),
                    ints("expand1", &[1, 2]),
                    ints("expand2", &[0, 2]),
                    ints("expand3", &[0, 1]),
                    ints("sumdim", &[0, 1]),
                ],
                &["out"],
            )],
            &[
                ("a", F32, vec![2]),
                ("b", F32, vec![3]),
                ("c", F32, vec![4]),
                ("out", F32, vec![4]),
            ],
            &["a", "b", "c"],
            &["out"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten.linalg_vector_norm.default",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_float("ord", 2.0),
                    ints("dim", &[1]),
                    scalar_bool("keepdim", false),
                ],
                &["out"],
            )],
            &[("x", F32, vec![2, 3]), ("out", F32, vec![2])],
            &["x"],
            &["out"],
        ));
        // ord=0 with a NaN-including dtype takes the nonzero-count path.
        assert!(translates(
            vec![node(
                "torch.ops.aten.linalg_vector_norm.default",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_float("ord", 0.0),
                    ints("dim", &[1]),
                    scalar_bool("keepdim", true),
                ],
                &["out"],
            )],
            &[("x", F32, vec![2, 3]), ("out", F32, vec![2, 1])],
            &["x"],
            &["out"],
        ));
    }

    #[test]
    fn histc_and_histogram_record() {
        assert!(translates(
            vec![node(
                "torch.ops.aten.histc.default",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_int("bins", 4),
                    scalar_float("min", 0.0),
                    scalar_float("max", 3.0),
                ],
                &["out"],
            )],
            &[("x", F32, vec![6]), ("out", I64, vec![4])],
            &["x"],
            &["out"],
        ));
        // histogram.bin_ct with an explicit range.
        assert!(translates(
            vec![node(
                "torch.ops.aten.histogram.bin_ct",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_int("bins", 4),
                    input(
                        "range",
                        Argument::Other(serde_json::json!({"as_floats": [0.0, 1.0]})),
                    ),
                ],
                &["hist"],
            )],
            &[("x", F32, vec![6]), ("hist", I64, vec![4])],
            &["x"],
            &["hist"],
        ));
        // histogram.bins_tensor with an explicit edge tensor.
        assert!(translates(
            vec![node(
                "torch.ops.aten.histogram.bins_tensor",
                vec![
                    input("self", tensor_arg("x")),
                    input("bins", tensor_arg("edges")),
                ],
                &["hist"],
            )],
            &[
                ("x", F32, vec![6]),
                ("edges", F32, vec![5]),
                ("hist", I64, vec![4]),
            ],
            &["x", "edges"],
            &["hist"],
        ));
    }

    #[test]
    fn histogramdd_record() {
        assert!(translates(
            vec![node(
                "torch.ops.aten._histogramdd_from_bin_cts.default",
                vec![input("self", tensor_arg("x")), ints("bins", &[3, 4])],
                &["hist"],
            )],
            &[("x", F32, vec![8, 2]), ("hist", I64, vec![3, 4])],
            &["x"],
            &["hist"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten._histogramdd_from_bin_tensors.default",
                vec![
                    input("self", tensor_arg("x")),
                    input("bins", tensors_arg(&["e0", "e1"])),
                ],
                &["hist"],
            )],
            &[
                ("x", F32, vec![8, 2]),
                ("e0", F32, vec![4]),
                ("e1", F32, vec![4]),
                ("hist", I64, vec![3, 3]),
            ],
            &["x", "e0", "e1"],
            &["hist"],
        ));
        // bin_edges with one coordinate dimension declares one output.
        assert!(translates(
            vec![node(
                "torch.ops.aten._histogramdd_bin_edges.default",
                vec![input("self", tensor_arg("x")), ints("bins", &[3])],
                &["edge"],
            )],
            &[("x", F32, vec![8, 1]), ("edge", F32, vec![4])],
            &["x"],
            &["edge"],
        ));
    }

    #[test]
    fn segment_reduce_records() {
        assert!(translates(
            vec![node(
                "torch.ops.aten.segment_reduce.default",
                vec![
                    input("data", tensor_arg("x")),
                    input("reduce", string_arg("sum")),
                    scalar_int("axis", 0),
                    input("lengths", tensor_arg("lengths")),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![4, 3]),
                ("lengths", I64, vec![2]),
                ("out", F32, vec![2, 3]),
            ],
            &["x", "lengths"],
            &["out"],
        ));
        // max with an explicit initial scalar.
        assert!(translates(
            vec![node(
                "torch.ops.aten.segment_reduce.default",
                vec![
                    input("data", tensor_arg("x")),
                    input("reduce", string_arg("max")),
                    scalar_int("axis", 0),
                    input("lengths", tensor_arg("lengths")),
                    scalar_float("initial", 0.0),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![4, 3]),
                ("lengths", I64, vec![2]),
                ("out", F32, vec![2, 3]),
            ],
            &["x", "lengths"],
            &["out"],
        ));
        // mean via the offsets form; the offsets tensor has segments + 1 rows.
        assert!(translates(
            vec![node(
                "torch.ops.aten.segment_reduce.default",
                vec![
                    input("data", tensor_arg("x")),
                    input("reduce", string_arg("mean")),
                    scalar_int("axis", 0),
                    input("offsets", tensor_arg("offsets")),
                ],
                &["out"],
            )],
            &[
                ("x", F32, vec![4, 3]),
                ("offsets", I64, vec![3]),
                ("out", F32, vec![2, 3]),
            ],
            &["x", "offsets"],
            &["out"],
        ));
    }

    #[test]
    fn fractional_max_pool_records() {
        assert!(translates(
            vec![node(
                "torch.ops.aten.fractional_max_pool2d.default",
                vec![
                    input("self", tensor_arg("x")),
                    ints("kernel_size", &[2, 2]),
                    ints("output_size", &[3, 3]),
                    input("random_samples", tensor_arg("rand")),
                ],
                &["values"],
            )],
            &[
                ("x", F32, vec![1, 1, 4, 4]),
                ("rand", F32, vec![1, 1, 2]),
                ("values", F32, vec![1, 1, 3, 3]),
            ],
            &["x", "rand"],
            &["values"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten.fractional_max_pool3d.default",
                vec![
                    input("self", tensor_arg("x")),
                    ints("kernel_size", &[2, 2, 2]),
                    ints("output_size", &[3, 3, 3]),
                    input("random_samples", tensor_arg("rand")),
                ],
                &["values"],
            )],
            &[
                ("x", F32, vec![1, 2, 4, 4, 4]),
                ("rand", F32, vec![1, 2, 3]),
                ("values", F32, vec![1, 2, 3, 3, 3]),
            ],
            &["x", "rand"],
            &["values"],
        ));
    }

    #[test]
    fn max_pool_backward_records() {
        assert!(translates(
            vec![node(
                "torch.ops.aten.max_pool2d_with_indices_backward.default",
                vec![
                    input("grad_output", tensor_arg("grad")),
                    input("self", tensor_arg("x")),
                    ints("kernel_size", &[2, 2]),
                    ints("stride", &[2, 2]),
                    ints("padding", &[0, 0]),
                    ints("dilation", &[1, 1]),
                    scalar_bool("ceil_mode", false),
                    input("indices", tensor_arg("indices")),
                ],
                &["grad_input"],
            )],
            &[
                ("grad", F32, vec![1, 1, 3, 3]),
                ("x", F32, vec![1, 1, 4, 4]),
                ("indices", I64, vec![1, 1, 3, 3]),
                ("grad_input", F32, vec![1, 1, 4, 4]),
            ],
            &["grad", "x", "indices"],
            &["grad_input"],
        ));
    }
}
