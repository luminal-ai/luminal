//! Additional ATen lowerings ported from the parked translator (port batch 6+).
//!
//! This module owns the "composed" reduction family that the parked
//! translator implemented on top of ordinary reductions:
//! `linalg_vector_norm`, `dist`, `_cdist_forward`, `_pdist_forward`,
//! `segment_reduce`, `var_mean`, and the three `any` overloads.
#![allow(dead_code)]

use anyhow::{Context, Result, anyhow, bail};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::{Argument, Node};

/// Float dtypes that can hold NaN. Only these need explicit NaN handling in
/// the order-sensitive paths below.
fn dtype_can_nan(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F16 | DType::Bf16 | DType::F32 | DType::F64 | DType::TF32
    )
}

/// Product of a run of dimension extents, kept symbolic.
fn product_of_dims(dims: impl IntoIterator<Item = IntExpr>) -> IntExpr {
    dims.into_iter().fold(IntExpr::from(1), |acc, d| acc * d)
}

/// Re-insert the reduced axes as size-1 extents (`keepdim`).
fn restore_reduced_dims(mut value: GraphTensor, axes: &[usize], keepdim: bool) -> GraphTensor {
    if keepdim {
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        for axis in axes {
            value = value.unsqueeze(axis);
        }
    }
    value
}

/// Normalize the axis of a cumulative scan. A rank-0 input has no axis to
/// scan; `-1`/`0` are accepted there for torch compatibility and reported as
/// `None`.
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

/// Normalize an optional ATen reduction-dim list. `None` and `[]` both mean
/// a full reduction for the composed reductions in this file.
fn normalize_reduction_axes(
    translator: &Translator<'_>,
    node: &Node,
    rank: usize,
    dim_arg: usize,
) -> Result<Vec<usize>> {
    let dims = translator.get_ints_arg(node, dim_arg).ok();
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
        let axis = util::normalize_dim(dim, rank);
        anyhow::ensure!(!axes.contains(&axis), "reduction dimensions must be unique");
        axes.push(axis);
    }
    Ok(axes)
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Shared private helpers
    // ---------------------------------------------------------------

    /// A rank-0 constant of `dtype`; integer dtypes never route through a
    /// refused float -> int cast.
    fn typed_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::I64 => self.cx.constant_i64(if value.is_nan() {
                0
            } else if value == f64::INFINITY {
                i64::MAX
            } else if value == f64::NEG_INFINITY {
                i64::MIN
            } else {
                value as i64
            }),
            DType::Int => self.cx.constant_i32(if value.is_nan() {
                0
            } else if value == f64::INFINITY {
                i32::MAX as i64
            } else if value == f64::NEG_INFINITY {
                i32::MIN as i64
            } else {
                value as i64
            }),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        }
    }

    /// Gather-based elementwise select. Arithmetic masking (`a*m + b*(1-m)`)
    /// turns excluded IEEE specials into NaN (`0 * inf`), which corrupts the
    /// segment redution fill; this packs and gathers instead so every bit
    /// pattern survives.
    fn ieee_select(&mut self, mask: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
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

    /// The `p`-norm used by `dist`/`cdist`/`pdist`. Its `p == 0` case counts
    /// an element as nonzero only when it is ordered nonzero (NaN is treated
    /// as zero) — deliberately different from `linalg_vector_norm`.
    fn p_norm(&mut self, magnitude: GraphTensor, p: f64, axes: Vec<usize>) -> GraphTensor {
        if p == 0.0 {
            let zero = self.is_zero(magnitude);
            self.bool_not(zero).cast(magnitude.dtype).sum(axes)
        } else if p == 1.0 {
            magnitude.sum(axes)
        } else if p == 2.0 {
            (magnitude * magnitude).sum(axes).sqrt()
        } else if p == f64::INFINITY {
            magnitude.max(axes)
        } else if p == f64::NEG_INFINITY {
            magnitude.min(axes)
        } else {
            magnitude.pow(p as f32).sum(axes).pow((1.0 / p) as f32)
        }
    }

    /// Lower `linalg_vector_norm` after the caller has constructed the real
    /// magnitude (complex inputs call the same routine after `abs(z)`).
    fn vector_norm_from_magnitude(
        &mut self,
        node: &Node,
        magnitude: GraphTensor,
    ) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let magnitude = magnitude.cast(output_dtype);
        let axes = normalize_reduction_axes(self, node, magnitude.rank(), 2)?;
        let keepdim = self.named_bool_arg(node, "keepdim").unwrap_or(false);
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
            let ordered_nonzero = self.bool_or(magnitude.lt(zero), magnitude.gt(zero));
            let nonzero = if dtype_can_nan(magnitude.dtype) {
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
        Ok(restore_reduced_dims(reduced, &axes, keepdim))
    }

    pub(super) fn translate_linalg_vector_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.operand(&node.inputs[0])?;
        let magnitude = self.real_abs(value);
        self.vector_norm_from_magnitude(node, magnitude)
    }

    pub(super) fn translate_dist(&mut self, node: &Node) -> Result<GraphTensor> {
        let lhs = self.operand(&node.inputs[0])?;
        let rhs = self.operand(&node.inputs[1])?;
        let (lhs, rhs) = util::ensure_same_dtype(lhs, rhs);
        let (lhs, rhs) = util::broadcast_binary(lhs, rhs);
        let magnitude = self.real_abs(lhs - rhs).cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 2).unwrap_or(2.0);
        Ok(self.p_norm(magnitude, p, (0..magnitude.rank()).collect()))
    }

    pub(super) fn translate_cdist(&mut self, node: &Node) -> Result<GraphTensor> {
        let lhs = self.operand(&node.inputs[0])?;
        let rhs = self.operand(&node.inputs[1])?;
        let (lhs_rank, rhs_rank) = (lhs.rank(), rhs.rank());
        anyhow::ensure!(
            lhs_rank >= 2 && rhs_rank >= 2,
            "cdist inputs must be matrices"
        );
        let (mut lhs, mut rhs) = util::ensure_same_dtype(lhs, rhs);
        let output_shape = self.output_meta_shape(node)?;
        let feature = lhs.dims()[lhs_rank - 1];
        anyhow::ensure!(
            feature == rhs.dims()[rhs_rank - 1],
            "cdist feature dimensions must match"
        );
        let mut pair_shape = output_shape;
        pair_shape.push(feature);
        lhs = lhs.expand_dim(lhs_rank - 1, rhs.dims()[rhs_rank - 2]);
        rhs = rhs.expand_dim(rhs_rank - 2, lhs.dims()[lhs.rank() - 3]);
        lhs = lhs.expand(pair_shape.clone());
        rhs = rhs.expand(pair_shape);
        let magnitude = self.real_abs(lhs - rhs).cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 2).unwrap_or(2.0);
        let last = magnitude.rank() - 1;
        Ok(self.p_norm(magnitude, p, vec![last]))
    }

    pub(super) fn translate_pdist(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
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
        let discriminant = k * -8.0 + rows_f * (rows_f - 1.0) * 4.0 - 7.0;
        let i = (rows_f
            - 2.0
            - (discriminant.sqrt() * 0.5 - 0.5)
                .trunc_cast(DType::I64)
                .cast(DType::F32))
        .trunc_cast(DType::Int);
        let i_f = i.cast(DType::F32);
        let j = (k + i_f + 1.0 - rows_f * (rows_f - 1.0) * 0.5
            + (rows_f - i_f) * (rows_f - i_f - 1.0) * 0.5)
            .trunc_cast(DType::Int);

        let out_dims = vec![pairs, columns];
        let i = i.expand_dim(1, columns);
        let j = j.expand_dim(1, columns);
        let column_coord = self.axis_positions(&out_dims, 1);
        let left = input.gather(&[i, column_coord]);
        let right = input.gather(&[j, column_coord]);
        let magnitude = self
            .real_abs(left - right)
            .cast(self.output_meta_dtype(node)?);
        let p = self.get_float_arg(node, 1).unwrap_or(2.0);
        Ok(self.p_norm(magnitude, p, vec![1]))
    }

    // ---------------------------------------------------------------
    // cumprod
    // ---------------------------------------------------------------

    /// One Hillis-Steele inclusive-scan step along `axis`: a lane at `i`
    /// reads `i - offset` (prefix lanes read index 0, then `valid` keeps them
    /// unchanged), plus the validity mask for those prefix lanes. The gather
    /// is built from coordinate iotas so no flat Int arithmetic is recorded.
    fn scan_shift(
        &mut self,
        value: GraphTensor,
        axis: usize,
        offset: usize,
    ) -> (GraphTensor, GraphTensor) {
        let dims = value.dims();
        let rank = dims.len();
        let mut positions = self.cx.arange(dims[axis]).cast(DType::Int);
        for (dim, size) in dims.iter().copied().enumerate() {
            if dim != axis {
                positions = positions.expand_dim(dim, size);
            }
        }
        let offset_tensor = self.full_tensor(positions.dims(), DType::Int, offset as f64);
        let valid = positions.ge(offset_tensor);
        let zero = self.full_tensor(positions.dims(), DType::Int, 0.0);
        let shifted = self.ieee_select(valid, positions - offset_tensor, zero);

        let mut coords = Vec::with_capacity(rank);
        for axis_index in 0..rank {
            if axis_index == axis {
                coords.push(shifted);
            } else {
                coords.push(self.axis_positions(&dims, axis_index));
            }
        }
        (value.gather(&coords), valid)
    }

    /// Inclusive multiplication scan along one axis, without log/exp. Shared
    /// by `cumprod.default` and the axis-wise `prod_scan`. Requires a concrete
    /// scan extent and bails otherwise.
    pub(super) fn cumprod_tensor(
        &mut self,
        value: GraphTensor,
        axis: usize,
    ) -> Result<GraphTensor> {
        let length = value.dims()[axis]
            .to_usize()
            .ok_or_else(|| anyhow!("cumprod currently requires a concrete scan dimension"))?;

        let mut values = value;
        let mut offset = 1usize;
        while offset < length {
            let (shifted, valid) = self.scan_shift(values, axis, offset);
            values = self.ieee_select(valid, shifted * values, values);
            offset *= 2;
        }
        Ok(values)
    }

    /// Lower `aten.cumprod.default` as an inclusive multiplication scan.
    /// Core `GraphTensor::cumprod` rewrites products through log/exp, which
    /// turns zeros into NaN and mishandles negatives; accumulating with plain
    /// multiplication keeps ordinary IEEE semantics for zeros, negatives,
    /// integers, and overflow.
    pub(super) fn translate_cumprod(&mut self, node: &Node) -> Result<GraphTensor> {
        let values = self
            .operand(&node.inputs[0])?
            .cast(self.output_meta_dtype(node)?);
        let Some(axis) = cumulative_axis(self.get_int_arg(node, 1)?, values.rank())? else {
            return Ok(values);
        };
        self.cumprod_tensor(values, axis)
    }

    /// Full product over `axes`, computed as one inclusive multiplication
    /// scan per axis followed by the final scan position. Unlike core
    /// `GraphTensor::prod` (log/exp), this keeps zeros and negatives exact.
    ///
    /// Axes are deduplicated and processed in descending order so that
    /// squeezing a reduced axis never shifts the index of a not-yet-reduced
    /// axis. Each scan needs a concrete extent, which `cumprod_tensor`
    /// enforces; a symbolic extent therefore bails cleanly.
    pub(super) fn prod_scan(&mut self, value: GraphTensor, axes: &[usize]) -> Result<GraphTensor> {
        let mut axes: Vec<usize> = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        let mut result = value;
        for &axis in axes.iter().rev() {
            anyhow::ensure!(
                axis < result.rank(),
                "prod_scan axis {axis} out of range for rank {}",
                result.rank()
            );
            // An empty scan axis has no final position; torch's product over
            // an empty axis is the multiplicative identity, 1.
            if result.dims()[axis].to_usize() == Some(0) {
                let out_shape: Vec<IntExpr> = result
                    .dims()
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, size)| (index != axis).then_some(size))
                    .collect();
                result = self.full_tensor(out_shape, result.dtype, 1.0);
                continue;
            }
            let scanned = self.cumprod_tensor(result, axis)?;
            let extent = scanned.dims()[axis];
            let last = scanned.slice_along((extent - IntExpr::from(1))..extent, axis);
            result = last.squeeze(axis);
        }
        Ok(result)
    }

    // ---------------------------------------------------------------
    // segment_reduce
    // ---------------------------------------------------------------

    /// A named tensor operand, or `None` when the slot is absent/None.
    fn named_tensor(&mut self, node: &Node, name: &str) -> Result<Option<GraphTensor>> {
        let Some(input) = node.inputs.iter().find(|input| input.name == name) else {
            return Ok(None);
        };
        let Some(value_name) = input.arg.as_value_name() else {
            return Ok(None);
        };
        self.values
            .get(value_name)
            .copied()
            .map(Some)
            .ok_or_else(|| anyhow!("segment_reduce: unknown tensor {value_name:?}"))
    }

    /// A named string argument (`segment_reduce`'s `reduce`).
    fn named_string(node: &Node, name: &str) -> Option<String> {
        let input = node.inputs.iter().find(|input| input.name == name)?;
        match &input.arg {
            Argument::Other(value) => value.as_str().map(str::to_string).or_else(|| {
                value
                    .get("as_string")
                    .and_then(|value| value.as_str())
                    .map(str::to_string)
            }),
            _ => None,
        }
    }

    /// A named numeric argument, if present.
    fn named_number(node: &Node, name: &str) -> Option<f64> {
        let input = node.inputs.iter().find(|input| input.name == name)?;
        input
            .arg
            .as_float()
            .or_else(|| input.arg.as_int().map(|value| value as f64))
            .or_else(|| {
                input
                    .arg
                    .as_bool()
                    .map(|value| if value { 1.0 } else { 0.0 })
            })
    }

    pub(super) fn translate_segment_reduce(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.operand(&node.inputs[0])?;
        let reduction = Self::named_string(node, "reduce")
            .context("segment_reduce is missing its reduction name")?;
        let axis = util::normalize_dim(self.named_int_arg(node, "axis").unwrap_or(0), data.rank());
        let output_shape = self.output_meta_shape(node)?;
        let segment_count = output_shape[axis];
        let input_count = data.dims()[axis];

        let lengths = self.named_tensor(node, "lengths")?;
        let offsets = self.named_tensor(node, "offsets")?;
        let (starts, ends) = if let Some(lengths) = lengths {
            let lengths = lengths.cast(DType::Int);
            let ends = lengths.cumsum(axis);
            (ends - lengths, ends)
        } else if let Some(offsets) = offsets {
            let offsets = offsets.cast(DType::Int);
            (
                offsets.slice_along(IntExpr::from(0)..segment_count, axis),
                offsets.slice_along(IntExpr::from(1)..(segment_count + IntExpr::from(1)), axis),
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
        let mut positions = self.cx.arange(input_count).cast(DType::Int);
        for (dimension, size) in pair_shape.iter().copied().enumerate() {
            if dimension != axis + 1 {
                positions = positions.expand_dim(dimension, size);
            }
        }
        let membership = self.bool_and(positions.ge(starts), positions.lt(ends));
        let expanded = data.expand_dim(axis, segment_count);
        let candidate_axis = axis + 1;
        let count = membership.cast(DType::Int).sum(candidate_axis);

        let initial = Self::named_number(node, "initial").map(|value| {
            self.full_tensor(vec![], data.dtype, value)
                .expand_rhs(output_shape.clone())
        });
        let has_initial = initial.is_some();
        let zero = self.full_tensor(expanded.dims(), data.dtype, 0.0);
        let one = self.full_tensor(expanded.dims(), data.dtype, 1.0);
        match reduction.as_str() {
            "sum" | "mean" => {
                let mut sum = self
                    .ieee_select(membership, expanded, zero)
                    .sum(candidate_axis);
                if let Some(initial) = initial {
                    sum += initial;
                }
                if reduction == "sum" {
                    Ok(sum)
                } else {
                    let zero_count = self.full_tensor(count.dims(), DType::Int, 0.0);
                    let nonempty = count.gt(zero_count);
                    let one_count = self.full_tensor(count.dims(), DType::Int, 1.0);
                    let safe_count = self.ieee_select(nonempty, count, one_count);
                    let mean = sum / safe_count.cast(data.dtype);
                    if has_initial {
                        Ok(mean)
                    } else {
                        let nan = self
                            .typed_scalar(f64::NAN, data.dtype)
                            .expand_rhs(mean.dims());
                        Ok(self.ieee_select(nonempty, mean, nan))
                    }
                }
            }
            "prod" => {
                let selected = self.ieee_select(membership, expanded, one);
                let magnitude = self.real_abs(selected).prod(candidate_axis);
                let negative_count = self.signbit(selected).cast(DType::Int).sum(candidate_axis);
                let two = self.full_tensor(negative_count.dims(), DType::Int, 2.0);
                let zero_count = self.full_tensor(negative_count.dims(), DType::Int, 0.0);
                let odd = (negative_count % two).ne(zero_count);
                let negative = self.full_tensor(magnitude.dims(), magnitude.dtype, 0.0) - magnitude;
                let mut product = self.ieee_select(odd, negative, magnitude);
                if let Some(initial) = initial {
                    product *= initial;
                }
                Ok(product)
            }
            "max" | "min" => {
                let (fill_value, empty_value) = if reduction == "max" {
                    (f64::NEG_INFINITY, f64::NEG_INFINITY)
                } else {
                    (f64::INFINITY, f64::INFINITY)
                };
                let fill = self
                    .typed_scalar(fill_value, data.dtype)
                    .expand_rhs(expanded.dims());
                let values = self.ieee_select(membership, expanded, fill);
                let mut result = if reduction == "max" {
                    values.max(candidate_axis)
                } else {
                    values.min(candidate_axis)
                };
                let zero_count = self.full_tensor(count.dims(), DType::Int, 0.0);
                let nonempty = count.gt(zero_count);
                if let Some(initial) = initial {
                    let combined = if reduction == "max" {
                        result.maximum(initial)
                    } else {
                        result.minimum(initial)
                    };
                    result = self.ieee_select(nonempty, combined, initial);
                } else {
                    let empty = self
                        .typed_scalar(empty_value, data.dtype)
                        .expand_rhs(result.dims());
                    result = self.ieee_select(nonempty, result, empty);
                }
                Ok(result)
            }
            other => bail!("unsupported segment_reduce reduction {other}"),
        }
    }

    // ---------------------------------------------------------------
    // var_mean
    // ---------------------------------------------------------------

    /// Degrees-of-freedom correction: an explicit named `correction`
    /// overrides a named `unbiased`; the default is Bessel's correction.
    fn var_mean_correction(&self, node: &Node) -> f64 {
        if let Some(correction) = self.named_float_arg(node, "correction") {
            return correction;
        }
        self.named_bool_arg(node, "unbiased")
            .map_or(1.0, |unbiased| if unbiased { 1.0 } else { 0.0 })
    }

    /// Compute `(variance, mean)` for an ordinary real tensor. PyTorch clamps
    /// non-positive degrees of freedom to zero before dividing, yielding NaN
    /// for a zero numerator and infinity otherwise.
    fn variance_mean_real(
        &mut self,
        node: &Node,
        value: GraphTensor,
    ) -> Result<(GraphTensor, GraphTensor)> {
        let axes = normalize_reduction_axes(self, node, value.rank(), 1)?;
        let keepdim = self.named_bool_arg(node, "keepdim").unwrap_or(false);
        let correction = self.var_mean_correction(node);
        let output_dtype = self.output_meta_dtype(node)?;
        let value = value.cast(output_dtype);
        let n = product_of_dims(axes.iter().map(|&axis| value.dims()[axis]));
        let mean = if axes.is_empty() {
            value
        } else {
            value.sum(axes.clone()) / n
        };
        let value_dims = value.dims();
        let expanded_mean = mean.expand_to_shape_on_axes(value_dims, axes.clone());
        let centered = value - expanded_mean;
        let numerator = (centered * centered).sum(axes.clone());
        let degrees = self.cx.constant_i32(n).cast(output_dtype)
            - self.floating_scalar(correction, output_dtype);
        let zero = self.floating_scalar(0.0, output_dtype);
        let divisor = degrees.maximum(zero).expand_rhs(numerator.dims());
        let variance = numerator / divisor;
        Ok((
            restore_reduced_dims(variance, &axes, keepdim),
            restore_reduced_dims(mean, &axes, keepdim),
        ))
    }

    pub(super) fn translate_var_mean(&mut self, node: &Node) -> Result<()> {
        let value = self.operand(&node.inputs[0])?;
        let (variance, mean) = self.variance_mean_real(node, value)?;
        // PyTorch's tuple order is `(variance, mean)`.
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(names.len() == 2, "var_mean must have two tensor outputs");
        self.values.insert(names[0].clone(), variance);
        self.values.insert(names[1].clone(), mean);
        Ok(())
    }

    // ---------------------------------------------------------------
    // any
    // ---------------------------------------------------------------

    pub(super) fn translate_any(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.operand(&node.inputs[0])?;
        let truth = if input.dtype == DType::Bool {
            input
        } else {
            let zero = self.full_tensor(input.dims(), input.dtype, 0.0);
            input.ne(zero)
        };

        let rank = truth.rank();
        let target = node
            .target
            .strip_prefix("torch.ops.aten.")
            .unwrap_or(&node.target);
        let (axes, keepdim) = match target {
            "any.default" => ((0..rank).collect::<Vec<_>>(), false),
            "any.dim" => {
                let dim = self
                    .named_int_arg(node, "dim")
                    .or_else(|| self.get_int_arg(node, 1).ok())
                    .unwrap_or(0);
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
                    util::normalize_dim(dim, rank)
                };
                let keepdim = self.named_bool_arg(node, "keepdim").unwrap_or(false);
                (if rank == 0 { vec![] } else { vec![axis] }, keepdim)
            }
            "any.dims" => {
                let axes = match self.get_ints_arg(node, 1) {
                    Ok(dims) => {
                        let mut axes = Vec::with_capacity(dims.len());
                        for dim in dims {
                            anyhow::ensure!(
                                dim >= -(rank as i64) && dim < rank as i64,
                                "any dimension {dim} out of range for rank {rank}"
                            );
                            let axis = util::normalize_dim(dim, rank);
                            anyhow::ensure!(!axes.contains(&axis), "any dimensions must be unique");
                            axes.push(axis);
                        }
                        axes
                    }
                    Err(_) => (0..rank).collect(),
                };
                let keepdim = self.named_bool_arg(node, "keepdim").unwrap_or(false);
                (axes, keepdim)
            }
            other => bail!("translate_any called for {other}"),
        };

        let result = if axes.is_empty() {
            truth.cast(DType::Bool)
        } else {
            let counts = truth.cast(DType::Int).sum(axes.clone());
            let zero = self.full_tensor(counts.dims(), DType::Int, 0.0);
            let result = counts.ne(zero);
            if keepdim {
                let mut sorted_axes = axes;
                sorted_axes.sort_unstable();
                let mut result = result;
                for axis in sorted_axes {
                    result = result.unsqueeze(axis);
                }
                result
            } else {
                result
            }
        };

        // The parent dispatcher may bind this returned value; binding here as
        // well is idempotent and keeps the `dim`/`dims` overloads correct even
        // if the dispatcher takes the value-returning path.
        if let Some(name) = Self::tensor_output_names(node).first() {
            self.values.insert(name.clone(), result);
        }
        Ok(result)
    }
}
