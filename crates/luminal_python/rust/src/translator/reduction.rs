use anyhow::Result;
use luminal::prelude::*;

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

/// Compute total element count, returning an error if any dimension is symbolic.
fn concrete_numel(a: &GraphTensor) -> Result<usize> {
    a.dims().iter().try_fold(1usize, |acc, d| {
        d.to_usize().map(|v| acc * v).ok_or_else(|| {
            anyhow::anyhow!("Full reduction requires concrete dimensions, got symbolic dim")
        })
    })
}

impl<'a> Translator<'a> {
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
                let total = concrete_numel(&a)?;
                let axes: Vec<usize> = (0..ndim).collect();
                let result = match op {
                    ReductionOp::Sum => a.sum(axes),
                    // Note: the luminal `mean` helper divides by the product of the
                    // axis dims, but we already require concrete dims here so we
                    // divide by the cached `total` to avoid recomputing.
                    ReductionOp::Mean => a.sum(axes) / total as f32,
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
