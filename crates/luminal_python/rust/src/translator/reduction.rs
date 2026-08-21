use anyhow::Result;
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
