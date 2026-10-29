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

/// Which of ATen's CPU histogram kernels (`native/cpu/HistogramKernel.cpp`)
/// assigns samples to bins; see [`Translator::histogram_bin_onehot`].
#[derive(Clone, Copy)]
enum HistogramBinSearch {
    /// `BINARY_SEARCH`, used by `histogramdd_stub` for caller-supplied edge
    /// tensors: `upper_bound` over every edge.
    Binary,
    /// `LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH`, used by
    /// `histogramdd_linear_stub` for bin counts: `upper_bound` over the
    /// three edges around the interpolated bin.
    LocalSearch,
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

/// Round a value computed in its (wider) compute dtype to `dtype` and widen
/// it back: the rounding a `scalar_t` operation performs when the
/// translator computes half precision in F32. A no-op when the two agree.
fn round_through(value: GraphTensor, dtype: DType) -> GraphTensor {
    if value.dtype == dtype {
        value
    } else {
        let compute = value.dtype;
        value.cast(dtype).cast(compute)
    }
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
        self.typed_scalar(value, dtype)
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
        super::convert(tensor, dtype)
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

    /// `torch.linalg.vector_norm` of the already-real-valued `magnitude`
    /// over `axes`. `dist(a, b, p)` is this norm of `a - b`.
    ///
    /// Order 0 counts the nonzero entries, torch's sum of `x != 0`. A NaN
    /// entry is nonzero, so it counts once (`inf - inf` included):
    /// `util::is_zero` is torch's `== 0`, which is false for NaN.
    fn vector_norm_reduce(
        &mut self,
        magnitude: GraphTensor,
        ord: f64,
        axes: Vec<usize>,
    ) -> GraphTensor {
        if ord == 0.0 {
            let zero = self.is_zero(magnitude);
            zero.logical_not().cast(magnitude.dtype).sum(axes)
        } else if ord == 1.0 {
            magnitude.sum(axes)
        } else if ord == 2.0 {
            magnitude.square().sum(axes).sqrt()
        } else if ord == f64::INFINITY {
            magnitude.max(axes)
        } else if ord == f64::NEG_INFINITY {
            magnitude.min(axes)
        } else {
            magnitude.pow(ord as f32).sum(axes).pow((1.0 / ord) as f32)
        }
    }

    /// The norm `_cdist_forward` and `_pdist_forward` take over each pair's
    /// `|difference|` row. It is the vector norm, except at order 0.
    ///
    /// At order 0, ATen's CPU kernels (`zdist_calc` in
    /// DistanceOpsKernel.cpp) map each entry `d` to `min(ceil(d), 1)`. That
    /// is 0 at zero, 1 at any other ordered value, and NaN at NaN, so one NaN
    /// difference makes the pair's count NaN. The vector norm and `dist`
    /// count the same entry as 1. Eager disagrees with itself across these
    /// ops, and this lowering follows each op. The CUDA kernel (`dists::zero`
    /// in DistanceKernel.cu) counts `d != 0` instead; the parity oracle here
    /// is CPU eager.
    fn p_norm(&mut self, magnitude: GraphTensor, p: f64, axes: Vec<usize>) -> GraphTensor {
        if p != 0.0 {
            return self.vector_norm_reduce(magnitude, p, axes);
        }
        let zero = self.is_zero(magnitude);
        let indicator = zero.logical_not().cast(magnitude.dtype);
        let nan = magnitude.is_nan();
        self.select(nan, magnitude, indicator).sum(axes)
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

        let reduced = self.vector_norm_reduce(magnitude, ord, axes.clone());
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
                Ok(self.vector_norm_reduce(magnitude, p, axes))
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
                let (i, j) = self.pdist_pair_indices(rows, pairs);
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

    /// The row-major upper-triangle pair `(i, j)`, `i < j`, of every flat
    /// pdist index `k in 0..pairs` (`pairs = n(n-1)/2`, `n = rows`): two
    /// `Int` tensors of shape `[pairs]`, in torch.pdist's output order.
    ///
    /// Invariant: every value below is an INTEGER held in F64 whose
    /// magnitude stays under `8n^2`, which is below `2^53` for every
    /// `n <= 2^25`, so each add, multiply, halving (of an even product) and
    /// compare is exact. That bound covers every pdist this lowering can
    /// express: `k` is an Int (i32) iota, which refuses loudly once
    /// `pairs > i32::MAX` (`n > 65536`). Floating point only SEEDS the row:
    /// the closed-form inverse `n - 2 - floor((sqrt(4n(n-1) - 8k - 7) - 1) / 2)`
    /// can land one row off when the backend's F64 square root or floor is
    /// not correctly rounded (CUDA-lite currently emits `sqrtf`/`floorf`,
    /// i.e. F32 precision, for F64 operands). The seed is then fixed by
    /// exact comparisons: the true row is the unique `r` with
    /// `row_start(r) <= k < row_start(r + 1)`, where
    /// `row_start(r) = r(2n - r - 1)/2` counts the pairs before row `r`, and
    /// one step each way recovers it from any seed within one row of it
    /// (`row_start` is nondecreasing on `-1..=n`, so the seeds `-1` and
    /// `n - 1` correct too). The old lowering ran the whole inverse in F32,
    /// which cannot hold `k` or `4n(n-1)` exactly past `2^24`: its pairs went
    /// wrong (and out of bounds) from `n = 4609`.
    fn pdist_pair_indices(&mut self, rows: IntExpr, pairs: IntExpr) -> (GraphTensor, GraphTensor) {
        let k = self.cx.arange(pairs).cast(DType::F64);
        let n = self
            .cx
            .constant_i32(rows)
            .cast(DType::F64)
            .expand_rhs(k.dims());
        let row_start = |row: GraphTensor| row * (n * 2.0 - row - 1.0) * 0.5;
        let discriminant = n * (n - 1.0) * 4.0 - k * 8.0 - 7.0;
        let seed = n - 2.0 - ((discriminant.sqrt() - 1.0) * 0.5).floor();
        let next = seed + 1.0;
        let row = k
            .lt(row_start(seed))
            .select(seed - 1.0, k.lt(row_start(next)).select(seed, next));
        let column = k - row_start(row) + row + 1.0;
        (row.trunc_cast(DType::Int), column.trunc_cast(DType::Int))
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
        let bins_arange = self.cx.iota(vec![bins], move |c| c[0] + min_i);
        let bins_expanded = bins_arange.cast(input.dtype).expand_dim(1, n);
        let input_expanded = input.expand_dim(0, IntExpr::from(bins));
        let matches = input_expanded.eq(bins_expanded);

        let out_dtype = self.output_meta_dtype(node)?;
        // A narrow-float count is summed in F32 (exact below 2^24) and
        // rounded once to the declared storage dtype.
        let count_dtype = if out_dtype.is_narrow_float() {
            DType::F32
        } else {
            out_dtype
        };
        Ok(super::util::round_to_storage(
            matches.cast(count_dtype).sum(vec![1]),
            out_dtype,
        ))
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
        let search = if bins_tensor {
            HistogramBinSearch::Binary
        } else {
            HistogramBinSearch::LocalSearch
        };
        let histogram = self.histogram_from_edges(&[values], &[edges], weight, density, search)?;
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
        let search = if tensor_edges {
            HistogramBinSearch::Binary
        } else {
            HistogramBinSearch::LocalSearch
        };
        let density = Self::histogram_density(node);
        let histogram = self.histogram_from_edges(&columns, &edges, weight, density, search)?;
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

    /// The `bins + 1` uniform edges of one histogram axis, built the way
    /// ATen builds them.
    ///
    /// `select_outer_bin_edges` picks the range in double: the explicit
    /// `range`, else (0, 1) for an empty input, else the column's min and
    /// max. A range with equal ends is widened by 0.5 on each side. The
    /// edges are then `linspace(lower, upper, bins + 1)` in the values dtype
    /// ([`Self::aten_linspace`]). So `edges[0]` is the lower end and
    /// `edges[bins]` the upper end exactly, and the maximum sample always
    /// lands in the closed last bin.
    fn uniform_histogram_edges(
        &mut self,
        values: GraphTensor,
        bins: usize,
        explicit_range: Option<(f64, f64)>,
    ) -> GraphTensor {
        let dtype = values.dtype;
        let empty = values.dims()[0].to_usize() == Some(0);
        let (lower, upper) = match explicit_range.or(empty.then_some((0.0, 1.0))) {
            // A static range is widened in double, as torch does, and only
            // then rounded to the values dtype.
            Some((lower, upper)) => {
                let (lower, upper) = if lower == upper {
                    (lower - 0.5, upper + 0.5)
                } else {
                    (lower, upper)
                };
                (
                    self.scalar_value(lower, dtype),
                    self.scalar_value(upper, dtype),
                )
            }
            // min/max are values of the dtype, so torch's double `min - 0.5`
            // rounds to the same value as the one rounding of `min - 0.5`
            // in the dtype.
            None => {
                let lower = values.min(vec![0]);
                let upper = values.max(vec![0]);
                let ordered = lower.lt(upper);
                let degenerate = self.bool_not(ordered);
                let half = self.scalar_value(0.5, dtype);
                let widened_lower = self.select(degenerate, lower - half, lower);
                let widened_upper = self.select(degenerate, upper + half, upper);
                (widened_lower, widened_upper)
            }
        };
        self.aten_linspace(lower, upper, bins + 1)
    }

    /// `linspace(start, end, steps)` for `steps >= 2` and rank-0 `start`
    /// and `end`, computed the way ATen's linspace kernel computes it
    /// (`native/cpu/RangeFactoriesKernel.cpp`; the CUDA kernel uses the same
    /// formula). Both ends are in the values dtype `T`.
    /// `step = (end - start) / (steps - 1)` is rounded once in `T`. The
    /// kernel is two-sided:
    ///
    /// ```text
    /// out[i] = start + step * i                  for i < steps / 2
    /// out[i] = end - step * (steps - 1 - i)      otherwise
    /// ```
    ///
    /// So `out[0] == start` and `out[steps - 1] == end` exactly, whatever
    /// `step` rounds to. Each half is non-decreasing because rounding is
    /// monotone. The halves meet in order unless `steps` approaches the
    /// reciprocal of `T`'s epsilon. So the edges are non-decreasing, as
    /// [`Self::histogram_bin_onehot`] requires.
    ///
    /// Rounding invariant: ATen compiles this kernel with floating-point
    /// contraction. Each `base ± step * n` is then a fused multiply-add
    /// that rounds once. This matched eager bit-for-bit on aarch64 for
    /// random F32 and F64 ranges, and nvcc contracts the CUDA kernel the
    /// same way. The IR has no fused multiply-add, so the edge is evaluated
    /// in F64 and rounded to `T` once:
    /// - For an F32 `T`, `step * n` is exact in F64: a 24-bit step times an
    ///   integer below 2^29. The F64 add is the only other rounding. It is
    ///   exact unless `base` and `step` are more than about 28 binades apart.
    ///   So the cast gives the fused result.
    /// - For F64 there is no wider type. The product and the sum each
    ///   round, so an interior edge can sit one ulp from eager's.
    fn aten_linspace(&mut self, start: GraphTensor, end: GraphTensor, steps: usize) -> GraphTensor {
        assert!(steps >= 2, "aten_linspace needs at least two points");
        let dtype = start.dtype;
        let wide = DType::F64;
        let intervals = self.scalar_value((steps - 1) as f64, dtype);
        let step = self.rounded_quotient(end - start, intervals).cast(wide);
        let half = steps / 2;
        // n = i on the rising half, n = steps - 1 - i on the falling half.
        let rising = self.cx.arange(half).cast(wide);
        let falling_len = steps - half;
        let falling = self
            .cx
            .iota(vec![falling_len], move |c| (falling_len - 1) - c[0])
            .cast(wide);
        let rising_dims = rising.dims();
        let falling_dims = falling.dims();
        let below = start.cast(wide).expand_rhs(rising_dims.clone())
            + step.expand_rhs(rising_dims) * rising;
        let above = end.cast(wide).expand_rhs(falling_dims.clone())
            - step.expand_rhs(falling_dims) * falling;
        below.concat_along(above, 0).cast(dtype)
    }

    /// `a / b` rounded once in `a`'s dtype, as IEEE division and ATen do.
    ///
    /// The frontend's F32 `/` is a reciprocal multiply, which rounds twice.
    /// A binary64 quotient holds more than 2p + 2 bits of a p-bit float's
    /// quotient (p = 24 for F32). Rounding it to the narrow dtype therefore
    /// gives the correctly rounded quotient: double rounding is harmless for
    /// division at that width (Figueroa, 1995). F64 `/` already rounds once.
    fn rounded_quotient(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        if a.dtype == DType::F64 {
            return a / b;
        }
        (a.cast(DType::F64) / b.cast(DType::F64)).cast(a.dtype)
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
        search: HistogramBinSearch,
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
        // A sample lands in the bin it hits on every axis. The Boolean
        // lattice is exact, so no non-finite value is selected away.
        let mut membership = None;
        for (dimension, (column, edge)) in columns.iter().zip(edges).enumerate() {
            let hit = self.histogram_bin_onehot(*column, *edge, search);
            let mut hit_shape = vec![IntExpr::from(1); columns.len() + 1];
            hit_shape[0] = sample_count;
            hit_shape[dimension + 1] = bin_shape[dimension];
            let hit = util::reshape_tensor(hit, &hit_shape).expand(full_shape.clone());
            membership = Some(match membership {
                Some(membership) => self.bool_and(membership, hit),
                None => hit,
            });
        }
        let membership = membership.expect("a histogram has at least one dimension");

        let mut weight_shape = vec![IntExpr::from(1); columns.len() + 1];
        weight_shape[0] = sample_count;
        let weight = util::reshape_tensor(weight, &weight_shape).expand(full_shape);
        let mut histogram = (membership.cast(weight.dtype) * weight).sum(vec![0]);
        if density {
            // torch's density divides by the total weight, then by each axis'
            // bin widths in turn. Keeping that order (not one divide by a
            // product volume) lets F64, whose `/` rounds once, round as eager
            // does. An all-zero histogram gives 0/0 = NaN in every bin, as in
            // eager.
            //
            // Shape invariant: `histogram` is `bin_shape`. Each divisor starts
            // below that rank and is expanded to `bin_shape` once: the rank-0
            // total, and each width vector reshaped to `[1, .., bins_d, .., 1]`.
            // Never expand a tensor that is already `bin_shape`: `expand_rhs`
            // appends axes, giving `bin_shape ++ bin_shape`.
            let axes = (0..bin_shape.len()).collect::<Vec<_>>();
            let total = histogram.sum(axes).expand_rhs(bin_shape.clone());
            histogram /= total;
            for (dimension, edge) in edges.iter().enumerate() {
                let bins = bin_shape[dimension];
                let widths = edge
                    .slice_along(IntExpr::from(1usize)..(bins + IntExpr::from(1usize)), 0)
                    - edge.slice_along(IntExpr::from(0usize)..bins, 0);
                let mut width_shape = vec![IntExpr::from(1); edges.len()];
                width_shape[dimension] = bins;
                let widths = util::reshape_tensor(widths, &width_shape).expand(bin_shape.clone());
                histogram /= widths;
            }
        }
        Ok(histogram)
    }

    /// The bin of each sample of `column`, one-hot over the bins of `edges`
    /// (`[samples, bins]`, Bool). This follows ATen's histogram kernel
    /// (`native/cpu/HistogramKernel.cpp`, `histogramdd_cpu_contiguous`).
    ///
    /// A sample is counted iff it is not NaN and
    /// `edges[0] <= x <= edges[bins]`. Its bin is
    /// `upper_bound(window, x) - 1`, and a bin of `bins` becomes `bins - 1`,
    /// which closes the last bin on the right. The window is every edge for
    /// [`HistogramBinSearch::Binary`]. For
    /// [`HistogramBinSearch::LocalSearch`] it is the edges
    /// `[max(0, est - 1), min(est + 2, bins + 1))` around the interpolated
    /// `est = trunc((x - edges[0]) * bins / (edges[bins] - edges[0]))`.
    ///
    /// Invariant: the edges are non-decreasing. torch requires this of edge
    /// tensors, and uniform edges satisfy it (see [`Self::aten_linspace`]).
    /// On a non-decreasing window, `upper_bound(window, x)` equals the
    /// window's start plus the number of its edges `<= x`, and that count is
    /// what this computes. The local window must be reproduced exactly, not
    /// approximated by a full search. When the range spans only a few ulps
    /// per bin, the edges repeat, and eager's bin is the last repeat inside
    /// the window, not the last repeat overall. `est` uses the same rounded
    /// subtract, multiply and divide as eager. Every index is a small
    /// integer, held exactly in the values dtype.
    fn histogram_bin_onehot(
        &mut self,
        column: GraphTensor,
        edges: GraphTensor,
        search: HistogramBinSearch,
    ) -> GraphTensor {
        let dtype = column.dtype;
        let samples = column.dims()[0];
        let edge_count = edges.dims()[0];
        let bins = edge_count - IntExpr::from(1usize);
        let first = edges.slice_along(IntExpr::from(0usize)..IntExpr::from(1usize), 0);
        let first = util::reshape_tensor(first, &[]).expand_rhs(vec![samples]);
        let last = util::reshape_tensor(edges.slice_along(bins..edge_count, 0), &[])
            .expand_rhs(vec![samples]);
        let below = column.lt(first);
        let above = last.lt(column);
        let outside = self.bool_or(below, above);
        let nan = self.is_nan(column);
        let skipped = self.bool_or(outside, nan);
        let counted = self.bool_not(skipped);

        // `edges[k] <= x` for each (sample, edge). Ordered `<` is false at
        // NaN, and `counted` drops NaN samples.
        let sample_grid = column.expand_dim(1, edge_count);
        let edge_grid = edges.expand_dim(0, samples);
        let exceeds = sample_grid.lt(edge_grid);
        let at_or_below = self.bool_not(exceeds);
        let zero = self.constant_like(column, 0.0);
        let one = self.constant_like(column, 1.0);
        let (in_window, window_start) = match search {
            HistogramBinSearch::Binary => (at_or_below, zero),
            HistogramBinSearch::LocalSearch => {
                let bin_count = self
                    .cx
                    .constant_i32(bins)
                    .cast(dtype)
                    .expand_rhs(vec![samples]);
                let scaled = (column - first) * bin_count;
                let estimate = self.rounded_quotient(scaled, last - first).trunc();
                let start = estimate - one;
                let end = estimate + self.constant_like(column, 2.0);
                let position = self
                    .cx
                    .arange(edge_count)
                    .cast(dtype)
                    .expand_dim(0, samples);
                let before_start = position.lt(start.expand_dim(1, edge_count));
                let from_start = self.bool_not(before_start);
                let before_end = position.lt(end.expand_dim(1, edge_count));
                let window = self.bool_and(from_start, before_end);
                (self.bool_and(at_or_below, window), start.maximum(zero))
            }
        };
        let count = in_window.cast(dtype).sum(vec![1]);
        let last_bin = self
            .cx
            .constant_i32(bins - IntExpr::from(1usize))
            .cast(dtype)
            .expand_rhs(vec![samples]);
        let bin = (window_start + count - one).minimum(last_bin);
        // `bin - 1 < index < bin + 1` is the integer `index == bin`, and is
        // false at a NaN `bin`. A NaN only comes from a zero span, where
        // eager's integer cast is undefined; such a sample is dropped
        // rather than counted in every bin.
        let index = self.cx.arange(bins).cast(dtype).expand_dim(0, samples);
        let bin_grid = bin.expand_dim(1, bins);
        let one_grid = one.expand_dim(1, bins);
        let not_below = (bin_grid - one_grid).lt(index);
        let not_above = index.lt(bin_grid + one_grid);
        let hit = self.bool_and(not_below, not_above);
        self.bool_and(hit, counted.expand_dim(1, bins))
    }

    // -----------------------------------------------------------------
    // Segment reduce
    // -----------------------------------------------------------------

    pub(super) fn translate_segment_reduce(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.get_input_tensor(node, 0)?;
        // Narrow storage reduces in F32 and rounds each segment's result
        // once to the storage dtype.
        let storage = data.dtype;
        let reduced = self.segment_reduce_values(node, util::widen_storage(data))?;
        Ok(util::round_to_storage(reduced, storage))
    }

    fn segment_reduce_values(&mut self, node: &Node, data: GraphTensor) -> Result<GraphTensor> {
        let reduction = segment_reduction_name(node)?;
        let axis = self.named_int_arg(node, "axis").unwrap_or(0);
        let axis = util::normalize_dim(axis, data.rank());
        let output_shape = self.output_meta_shape(node)?;
        let segment_count = output_shape[axis];
        let input_count = data.dims()[axis];

        let lengths = self.named_tensor_operand(node, "lengths")?;
        let offsets = self.named_tensor_operand(node, "offsets")?;
        // INVARIANT: segment bounds, input positions, and segment sizes are
        // exact integers compared and counted in Int; no float carries
        // them (F32 rounds past 2^24 and moves a segment boundary). The
        // ends are an Int scan of the lengths (the scan kernels check
        // integer overflow) and the starts are the same scan shifted one
        // segment right with a zero fill (an exclusive scan, so no
        // unproven Int subtraction of caller data); the positions are an
        // Int iota; a segment size counts 0/1 membership indicators.
        let (starts, ends) = if let Some(lengths) = lengths {
            let ends = lengths.cast(DType::Int).cumsum(vec![axis]);
            let starts = if segment_count.to_usize() == Some(0) {
                ends
            } else {
                ends.slice_along(
                    IntExpr::from(0usize)..segment_count - IntExpr::from(1),
                    axis,
                )
                .pad_along(1, 0, axis, 0.0)
            };
            (starts, ends)
        } else if let Some(offsets) = offsets {
            let offsets = offsets.cast(DType::Int);
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
        let positions = self.cx.iota(pair_shape, |c| c[candidate_axis]);
        let membership = self.bool_and(positions.ge(starts), positions.lt(ends));
        let expanded = data.expand_dim(axis, segment_count);
        let count = membership.cast(DType::Int).sum(vec![candidate_axis]);
        // The mean's divisor in the data dtype. A narrow float goes through
        // F32 (exact for any count to 2^24, then one rounding), as the old
        // F32 count did: a direct Int -> F16/BF16 cast is checked-exact and
        // would refuse counts past 2048/256.
        let divisor = |count: GraphTensor| match data.dtype {
            DType::F16 | DType::Bf16 => count.cast(DType::F32).cast(data.dtype),
            dtype => count.cast(dtype),
        };

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
                    .sum(vec![candidate_axis]);
                if let Some(initial) = initial {
                    sum += initial;
                }
                if reduction == "sum" {
                    return Ok(sum);
                }
                if has_initial {
                    // Empty segments keep the initial value: replace a zero
                    // count by one (an Int add of a [0, 1] indicator, in
                    // width by the count's bounds).
                    let zero_count = self.cx.constant_i32(0).expand_rhs(count.dims());
                    let nonempty = count.gt(zero_count);
                    let empty = self.bool_not(nonempty).cast(DType::Int);
                    let safe_count = count + empty;
                    Ok(sum / divisor(safe_count))
                } else {
                    // An empty segment is 0/0 -> NaN, exactly torch's mean.
                    Ok(sum / divisor(count))
                }
            }
            "prod" => {
                let selected = self.blend_select(membership, expanded, one);
                let magnitude = self.real_abs(selected).prod(vec![candidate_axis]);
                // The parity of an exact Int count of negative factors: an
                // F32 count rounds past 2^24 and can flip the sign. The
                // remainder by 2 lies in [0, 1] and the sign in [-1, 1],
                // each in width by the value bounds. Sign multiplication
                // avoids a select (and preserves infinities instead of
                // leaking inf * 0).
                let negative_count = self
                    .signbit(selected)
                    .cast(DType::Int)
                    .sum(vec![candidate_axis]);
                let two = self.cx.constant_i32(2).expand_rhs(negative_count.dims());
                let odd = negative_count.trunc_rem(two);
                let one = self.cx.constant_i32(1).expand_rhs(odd.dims());
                let sign = one - odd * two;
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
                    values.max(vec![candidate_axis])
                } else {
                    values.min(vec![candidate_axis])
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
        // Half-precision keys widen to F32: the widening is exact and keeps
        // the order (NaN included), and the sort's pairwise arithmetic is
        // opmath, which the half formats do not carry.
        let key = match candidates.dtype {
            DType::F64 | DType::F32 | DType::TF32 => candidates,
            DType::F16 | DType::Bf16 | DType::Bool => candidates.cast(DType::F32),
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

    /// `aten.fractional_max_pool{2,3}d.default`: values + int64 indices.
    ///
    /// INDEX INVARIANT: every index is exact, and none is a flat position
    /// carried in a float or built by Int tensor arithmetic.
    /// * The window starts follow ATen's `generate_intervals` in the sample
    ///   dtype (that rounding is the op's semantics). A start is an integer
    ///   below the axis extent, held exactly in the compute dtype (F32 or
    ///   F64), and so is start + kernel offset; the coordinate is truncated
    ///   to Int only after that addition. (The checked Int -> F32 casts of
    ///   positions and extents refuse an axis longer than 2^24 loudly.)
    /// * The input is read with a per-axis coordinate gather (batch/channel
    ///   axis positions plus one coordinate per spatial axis), so no position
    ///   over the whole input is ever formed.
    /// * The returned within-plane index is read from a plane-position iota
    ///   at the same spatial coordinates: a pure coordinate function, so no
    ///   Int add or multiply (each proof-gated) enters the model, and it is
    ///   exact for any plane an Int iota can address.
    ///
    /// A coordinate outside the input (ATen reads out of bounds there, for
    /// `random_samples` far outside `[0, 1)`) is a loud gather refusal.
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
        // ATen's `generate_intervals<scalar_t>` rounds every operation to
        // the sample dtype. Half-precision samples are computed in F32 and
        // rounded back after each operation, which is bit-identical (F32 has
        // more than 2p + 2 bits for both narrow formats).
        let sample_dtype = random_samples.dtype;
        let compute_dtype = match sample_dtype {
            DType::F64 | DType::F32 => sample_dtype,
            DType::F16 | DType::Bf16 => DType::F32,
            other => bail!("fractional max pool random_samples must be floating, got {other:?}"),
        };
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

        // Batch/channel coordinates are the window's own axis positions.
        let mut coordinates: Vec<GraphTensor> = (0..prefix_rank)
            .map(|axis| self.axis_positions(&window_shape, axis))
            .collect();

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
            let sample = sample.cast(compute_dtype);

            let output_axis = prefix_rank + spatial;
            let output_position = self.axis_positions(&output_shape, output_axis);
            // inputSize - poolSize: the last valid window start.
            let last_start = input_spatial[spatial] - IntExpr::from(kernel[spatial]);
            // outputSize - 1: the terminal output position. A single output
            // position is the terminal one, so its start is selected below;
            // dividing by max(outputSize - 1, 1) only keeps the unselected
            // ordinary starts finite.
            let last_position = output_spatial[spatial] - IntExpr::from(1usize);
            let steps = last_position.max(IntExpr::from(1usize));
            // alpha = scalar_t(inputSize - poolSize) / scalar_t(outputSize - 1),
            // one correctly rounded division. The frontend's F32 division is
            // a reciprocal product, so the quotient is taken in F64 and
            // rounded once to the sample dtype: binary64 has more than
            // 2p + 2 bits for every sample format, so that double rounding
            // equals ATen's single one.
            let numerator = round_through(
                self.cx.constant_i32(last_start).cast(compute_dtype),
                sample_dtype,
            );
            let denominator = round_through(
                self.cx.constant_i32(steps).cast(compute_dtype),
                sample_dtype,
            );
            let alpha = (numerator.cast(DType::F64) / denominator.cast(DType::F64))
                .cast(sample_dtype)
                .cast(compute_dtype)
                .expand_rhs(sample.dims());
            // ATen: int64((i + sample) * alpha) - int64(sample * alpha). `i`
            // is rounded to the sample dtype, each operation rounds there,
            // and the int64 conversions truncate toward zero. Both truncated
            // terms are integers of the compute dtype, so their difference
            // is exact whenever the true difference is representable, which
            // covers every in-bounds start.
            let position = round_through(output_position.cast(compute_dtype), sample_dtype);
            let shifted = round_through(position + sample, sample_dtype);
            let upper = round_through(shifted * alpha, sample_dtype).trunc();
            let lower = round_through(sample * alpha, sample_dtype).trunc();
            let ordinary_start = upper - lower;
            let terminal_start = self
                .cx
                .constant_i32(last_start)
                .cast(compute_dtype)
                .expand_rhs(ordinary_start.dims());
            let terminal_position = self
                .cx
                .constant_i32(last_position)
                .expand_rhs(output_position.dims());
            let is_terminal = output_position.eq(terminal_position);
            let mut start = self.select(is_terminal, terminal_start, ordinary_start);

            for size in kernel.iter().copied() {
                start = start.expand_dim(start.rank(), size);
            }
            let kernel_axis = input_rank + spatial;
            let offset = self
                .axis_positions(&window_shape, kernel_axis)
                .cast(compute_dtype);
            coordinates.push((start + offset).trunc_cast(DType::Int));
        }

        let candidates = input.gather(&coordinates);
        let plane_strides = row_major_strides(&input_spatial);
        let plane_positions = self.cx.iota(input_spatial.clone(), |c| {
            (0..rank).fold(IntExpr::from(0), |acc, axis| {
                acc + c[axis] * plane_strides[axis]
            })
        });
        let logical_indices = plane_positions.gather(&coordinates[prefix_rank..]);
        let [values, indices] =
            self.stats_select_pool_max(candidates, logical_indices, input_rank)?;
        self.bind_declared_prefix(node, &[values, indices])?;
        Ok(values)
    }

    // -----------------------------------------------------------------
    // Max pool backward
    // -----------------------------------------------------------------

    /// `aten.max_pool2d_with_indices_backward.default`: add each
    /// `grad_output` element into a zero gradient at the input position its
    /// window's max came from.
    ///
    /// Destination invariant: the forward op's `indices` are flat positions
    /// WITHIN one `H*W` plane, so the destination is addressed in COORDINATE
    /// form on a `[planes, H*W]` view of the gradient. The plane coordinate
    /// (batch and channel) is an iota over the grad shape; the in-plane
    /// coordinate is `indices` itself. No flat `plane_offset + index` sum is
    /// formed. That sum mixes a DATA operand into the arithmetic, so it needs
    /// either Int arithmetic on a value with no proven range (proof-gated) or
    /// float arithmetic (inexact past 2^24 elements). Each coordinate is
    /// checked against its own axis by the gather/scatter contract
    /// (out-of-bounds is UB, refused loudly by the reference), so an index
    /// outside `[0, H*W)` cannot silently land in a neighbouring plane.
    pub(super) fn translate_max_pool_backward(&mut self, node: &Node) -> Result<GraphTensor> {
        let updates = self.get_input_tensor(node, 0)?;
        let input = self.get_input_tensor(node, 1)?;
        let indices = self.get_input_tensor(node, 7)?.cast(DType::Int);
        let input_shape = input.dims();
        anyhow::ensure!(
            matches!(input_shape.len(), 3 | 4),
            "max_pool2d_with_indices_backward expects a 3-D or 4-D input, got rank {}",
            input_shape.len()
        );
        let prefix_rank = input_shape.len() - 2;
        let output_shape = updates.dims();

        let prefix = &input_shape[..prefix_rank];
        let plane_shape = vec![
            super::dim_arith::product_of_dims(prefix.iter().copied()),
            super::dim_arith::product_of_dims(input_shape[prefix_rank..].iter().copied()),
        ];
        // The two Int coordinates of `plane_shape`, each over the grad shape:
        // the plane number is a coordinate function of the grad position
        // (bounded by construction), the in-plane position is `indices`.
        let prefix_strides = row_major_strides(prefix);
        let planes = self.cx.iota(output_shape.clone(), |c| {
            (0..prefix_rank).fold(IntExpr::from(0), |acc, axis| {
                acc + c[axis] * prefix_strides[axis]
            })
        });
        let coordinates = [planes.flatten(), indices.flatten()];

        let count = super::dim_arith::product_of_dims(output_shape.iter().copied())
            .to_usize()
            .context("max_pool2d_with_indices_backward requires a concrete output element count")?;
        let flat_updates = updates.flatten();
        let mut output = self.full_tensor(plane_shape, input.dtype, 0.0);
        // Duplicate destinations (overlapping windows selecting the same max)
        // must accumulate; the recorder's `scatter` is overwrite-only, so one
        // read/modify/write step per update element, in grad row-major order
        // (ATen's CPU kernel order).
        for step in 0..count {
            let destination: Vec<GraphTensor> = coordinates
                .iter()
                .map(|coordinate| coordinate.slice_along(step..step + 1, 0))
                .collect();
            let update = flat_updates.slice_along(step..step + 1, 0);
            let current = output.gather(&destination);
            // Each step rounds to the gradient's dtype, as torch's
            // sequential kernel does; narrow storage adds in F32.
            output = output.scatter(
                &destination,
                util::storage_step(current, update, |a, b| a + b),
            );
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
                    strides: None,
                    storage_offset: None,
                    layout: None,
                    device: None,
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

    /// `density=True` normalises a `bin_shape` histogram by the total and by
    /// each axis' bin widths; every divisor must reach `bin_shape` exactly
    /// (a second `expand_rhs` of an already-`bin_shape` tensor used to panic
    /// in the frontend with "Dims must match to multiply tensors").
    #[test]
    fn histogram_density_records() {
        assert!(translates(
            vec![node(
                "torch.ops.aten.histogram.bin_ct",
                vec![
                    input("self", tensor_arg("x")),
                    scalar_int("bins", 4),
                    scalar_bool("density", true),
                ],
                &["hist", "bin_edges"],
            )],
            &[
                ("x", F32, vec![7]),
                ("hist", F32, vec![4]),
                ("bin_edges", F32, vec![5]),
            ],
            &["x"],
            &["hist", "bin_edges"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten.histogram.bins_tensor",
                vec![
                    input("self", tensor_arg("x")),
                    input("bins", tensor_arg("edges")),
                    input("weight", tensor_arg("w")),
                    scalar_bool("density", true),
                ],
                &["hist"],
            )],
            &[
                ("x", F32, vec![6]),
                ("edges", F32, vec![5]),
                ("w", F32, vec![6]),
                ("hist", F32, vec![4]),
            ],
            &["x", "edges", "w"],
            &["hist"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten._histogramdd_from_bin_cts.default",
                vec![
                    input("self", tensor_arg("x")),
                    ints("bins", &[3, 4]),
                    scalar_bool("density", true),
                ],
                &["hist"],
            )],
            &[("x", F32, vec![8, 2]), ("hist", F32, vec![3, 4])],
            &["x"],
            &["hist"],
        ));
        assert!(translates(
            vec![node(
                "torch.ops.aten._histogramdd_from_bin_tensors.default",
                vec![
                    input("self", tensor_arg("x")),
                    input("bins", tensors_arg(&["e0", "e1"])),
                    input("weight", tensor_arg("w")),
                    scalar_bool("density", true),
                ],
                &["hist"],
            )],
            &[
                ("x", F32, vec![8, 2]),
                ("e0", F32, vec![4]),
                ("e1", F32, vec![5]),
                ("w", F32, vec![8]),
                ("hist", F32, vec![3, 4]),
            ],
            &["x", "e0", "e1", "w"],
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

    /// ATen requires `random_samples` to share the input's dtype, so every
    /// float input reaches the lowering with float samples of its own
    /// width. The window starts round in that dtype (half precision through
    /// F32) and the indices come from coordinate gathers, so no operand pairs
    /// a sample-dtype tensor with an F32 index tensor.
    #[test]
    fn fractional_max_pool_records_every_float_dtype() {
        const F16: u32 = 6;
        const F64: u32 = 8;
        const BF16: u32 = 13;
        for dtype in [F64, F32, F16, BF16] {
            for (target, input_shape, samples_shape, output_shape, kernel, size) in [
                (
                    "torch.ops.aten.fractional_max_pool2d.default",
                    vec![1, 2, 7, 7],
                    vec![1, 2, 2],
                    vec![1, 2, 3, 3],
                    &[2, 2][..],
                    &[3, 3][..],
                ),
                (
                    "torch.ops.aten.fractional_max_pool2d.default",
                    vec![2, 7, 7],
                    vec![1, 2, 2],
                    vec![2, 3, 3],
                    &[2, 2][..],
                    &[3, 3][..],
                ),
                (
                    "torch.ops.aten.fractional_max_pool3d.default",
                    vec![1, 2, 6, 6, 6],
                    vec![1, 2, 3],
                    vec![1, 2, 3, 3, 3],
                    &[2, 2, 2][..],
                    &[3, 3, 3][..],
                ),
            ] {
                assert!(
                    translates(
                        vec![node(
                            target,
                            vec![
                                input("self", tensor_arg("x")),
                                ints("kernel_size", kernel),
                                ints("output_size", size),
                                input("random_samples", tensor_arg("rand")),
                            ],
                            &["values", "indices"],
                        )],
                        &[
                            ("x", dtype, input_shape.clone()),
                            ("rand", dtype, samples_shape),
                            ("values", dtype, output_shape.clone()),
                            ("indices", I64, output_shape),
                        ],
                        &["x", "rand"],
                        &["values", "indices"],
                    ),
                    "{target} on dtype code {dtype} over {input_shape:?} must translate"
                );
            }
        }
    }

    /// The index invariant of `translate_fractional_max_pool` as graph
    /// facts: the input is read with one coordinate per axis rather than
    /// through a flat position, no Int add or multiply (proof-gated, and an
    /// e-graph closure hazard) lies upstream of those coordinates, each
    /// spatial coordinate is truncated to Int only after its float
    /// arithmetic, and the within-plane index is a plane-position iota read
    /// at the same coordinates.
    #[test]
    fn fractional_max_pool_gathers_by_coordinates_without_int_arithmetic() {
        use luminal::prelude::petgraph::Direction;
        use luminal::prelude::petgraph::visit::EdgeRef;
        use luminal::prelude::*;

        for (target, rank, input_shape, samples_shape, output_shape, kernel, size) in [
            (
                "torch.ops.aten.fractional_max_pool2d.default",
                2,
                &[2, 3, 7, 8][..],
                &[2, 3, 2][..],
                &[2, 3, 3, 3][..],
                &[2, 3][..],
                &[3, 3][..],
            ),
            (
                "torch.ops.aten.fractional_max_pool3d.default",
                3,
                &[2, 6, 6, 6][..],
                &[1, 2, 3][..],
                &[2, 3, 3, 3][..],
                &[2, 2, 2][..],
                &[3, 3, 3][..],
            ),
        ] {
            let translation = crate::translate::test_support::translate_one(
                node(
                    target,
                    vec![
                        input("self", tensor_arg("x")),
                        ints("kernel_size", kernel),
                        ints("output_size", size),
                        input("random_samples", tensor_arg("rand")),
                    ],
                    &["values", "indices"],
                ),
                &[
                    ("x", F32, input_shape),
                    ("rand", F32, samples_shape),
                    ("values", F32, output_shape),
                    ("indices", I64, output_shape),
                ],
                &["x", "rand"],
                &["values", "indices"],
            );
            let graph = translation.graph.logical.petgraph();
            let operands = |id: NodeIndex| {
                let mut edges: Vec<_> = graph
                    .edges_directed(id, Direction::Incoming)
                    .map(|edge| (edge.weight().0, edge.source()))
                    .collect();
                edges.sort_by_key(|(port, _)| *port);
                edges
                    .into_iter()
                    .map(|(_, source)| source)
                    .collect::<Vec<_>>()
            };
            let through_views = |mut id: NodeIndex| {
                while matches!(graph[id].op, LogicalOp::IndexMapApply { .. }) {
                    id = operands(id)[0];
                }
                id
            };
            let int_arithmetic_upstream = |root: NodeIndex| {
                let mut stack = vec![root];
                let mut seen = std::collections::HashSet::new();
                while let Some(id) = stack.pop() {
                    if !seen.insert(id) {
                        continue;
                    }
                    let recorded = &graph[id];
                    if matches!(recorded.op, LogicalOp::Add | LogicalOp::Mul)
                        && matches!(recorded.dtype, DType::Int | DType::I64)
                    {
                        return true;
                    }
                    stack.extend(operands(id));
                }
                false
            };

            let input_gathers: Vec<_> = graph
                .node_indices()
                .filter(|id| matches!(graph[*id].op, LogicalOp::Gather))
                .filter(|id| {
                    matches!(
                        &graph[through_views(operands(*id)[0])].op,
                        LogicalOp::Input { label } if label == "x"
                    )
                })
                .collect();
            assert_eq!(input_gathers.len(), 1, "{target} gathers the input once");
            let coordinates = operands(input_gathers[0])[1..].to_vec();
            assert_eq!(
                coordinates.len(),
                input_shape.len(),
                "{target} reads the input with one coordinate per axis"
            );
            for coordinate in &coordinates {
                assert!(
                    !int_arithmetic_upstream(*coordinate),
                    "{target} builds a gather coordinate with Int arithmetic"
                );
            }
            let spatial = &coordinates[coordinates.len() - rank..];
            for coordinate in spatial {
                assert!(
                    matches!(
                        graph[through_views(*coordinate)].op,
                        LogicalOp::TruncCast(DType::Int)
                    ),
                    "{target} spatial coordinates are truncated after their float arithmetic"
                );
            }
            assert!(
                graph.node_indices().any(|id| {
                    matches!(graph[id].op, LogicalOp::Gather)
                        && matches!(
                            graph[through_views(operands(id)[0])].op,
                            LogicalOp::Iota { .. }
                        )
                        && operands(id)[1..] == *spatial
                }),
                "{target} reads the within-plane index from a plane-position iota"
            );
        }
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

    /// The destination of every accumulation step is addressed in
    /// coordinate form on a `[planes, H*W]` view of the gradient: the
    /// in-plane coordinate is the data `indices`, never folded into a flat
    /// sum, so no float detour (inexact past 2^24) and no unproven Int add
    /// is recorded even when one plane holds more than 2^24 elements.
    #[test]
    fn max_pool_backward_scatters_plane_coordinates_without_flat_offsets() {
        use luminal::prelude::{DType, IntExpr, LogicalOp};

        let translation = crate::translate::test_support::translate_one(
            node(
                "torch.ops.aten.max_pool2d_with_indices_backward.default",
                vec![
                    input("grad_output", tensor_arg("grad")),
                    input("self", tensor_arg("x")),
                    ints("kernel_size", &[2050, 2050]),
                    ints("stride", &[2050, 2050]),
                    ints("padding", &[0, 0]),
                    ints("dilation", &[1, 1]),
                    scalar_bool("ceil_mode", false),
                    input("indices", tensor_arg("indices")),
                ],
                &["grad_input"],
            ),
            &[
                ("grad", F32, &[1, 2, 2, 2]),
                ("x", F32, &[1, 2, 4100, 4100]),
                ("indices", I64, &[1, 2, 2, 2]),
                ("grad_input", F32, &[1, 2, 4100, 4100]),
            ],
            &["grad", "x", "indices"],
            &["grad_input"],
        );
        let graph = translation.graph.logical.petgraph();
        for node in graph.node_weights() {
            assert!(
                !matches!(
                    node.op,
                    LogicalOp::Cast(DType::F32) | LogicalOp::TruncCast(_)
                ),
                "no float detour for destinations: found {:?} -> {:?}",
                node.op,
                node.dtype
            );
            assert!(
                !(matches!(node.op, LogicalOp::Add) && node.dtype == DType::Int),
                "no flat Int offset sum over the data indices"
            );
        }
        let plane_view: Vec<IntExpr> = [2usize, 4100 * 4100]
            .into_iter()
            .map(IntExpr::from)
            .collect();
        let scatters: Vec<_> = graph
            .node_weights()
            .filter(|node| matches!(node.op, LogicalOp::Scatter))
            .collect();
        assert_eq!(scatters.len(), 8, "one accumulation step per grad element");
        for scatter in scatters {
            assert_eq!(
                scatter.dims, plane_view,
                "each step writes the [planes, H*W] view"
            );
        }
    }
}
