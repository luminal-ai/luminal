//! Additional common ATen lowerings beyond the parked translator's surface.
#![allow(dead_code)]

use anyhow::{Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, normalize_dim};
use crate::pt2_schema::Node;

/// View-reshape a tensor to `target`. The shared `util::reshape_tensor`
/// passes `target[i]` as the inner split extent, which produces transposed
/// intermediate dims; `split_dims(axis, inner)` expects the product of the
/// remaining target extents. Recorded reshapes are pure views, so element
/// order is preserved either way; this form keeps the extents valid.
fn reshape_view(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    let mut flat = t.flatten();
    for axis in 0..target.len().saturating_sub(1) {
        let inner: IntExpr = target[axis + 1..]
            .iter()
            .fold(IntExpr::from(1), |acc, d| acc * *d);
        flat = flat.split_dims(axis, inner);
    }
    flat
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Multi-output shape ops
    // ---------------------------------------------------------------

    /// `broadcast_tensors.default`: broadcast every tensor input to the
    /// common (right-aligned) shape and bind one output per input.
    pub(super) fn translate_broadcast_tensors(&mut self, node: &Node) -> Result<()> {
        let names: Vec<String> = if let Some(tensors) = node.inputs[0].arg.as_tensors() {
            tensors.iter().map(|t| t.name.clone()).collect()
        } else {
            node.inputs
                .iter()
                .filter_map(|input| input.arg.as_tensor_name().map(str::to_string))
                .collect()
        };
        anyhow::ensure!(
            !names.is_empty(),
            "broadcast_tensors: no tensor inputs found"
        );
        let mut values = Vec::with_capacity(names.len());
        for name in &names {
            values.push(
                *self
                    .values
                    .get(name)
                    .ok_or_else(|| anyhow::anyhow!("broadcast_tensors: unknown tensor {name}"))?,
            );
        }
        // Fold pairwise broadcasts to discover the common shape, then
        // expand each original input to it.
        let mut target = values[0];
        for value in values.iter().skip(1) {
            target = util::broadcast_binary(target, *value).0;
        }
        let target = target.dims();
        let outputs: Vec<GraphTensor> = values
            .into_iter()
            .map(|value| super::broadcast_to(value, &target))
            .collect();
        self.bind_flat_outputs(node, outputs)
    }

    /// `chunk.default` / `unsafe_chunk.default`: split `dim` into `chunks`
    /// near-equal slices; earlier chunks are larger and torch returns
    /// `min(chunks, dim_size)` tensors.
    pub(super) fn translate_chunk(&mut self, node: &Node) -> Result<()> {
        let x = self.get_input_tensor(node, 0)?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "chunk on a rank-0 tensor is not ported");
        let chunks = self
            .named_int_arg(node, "chunks")
            .or_else(|| self.get_int_arg(node, 1).ok())
            .ok_or_else(|| anyhow::anyhow!("chunk: missing chunk count"))?;
        anyhow::ensure!(chunks > 0, "chunk expects at least one chunk, got {chunks}");
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| self.get_int_arg(node, 2).ok())
            .unwrap_or(0);
        anyhow::ensure!(
            raw_dim >= -(rank as i64) && raw_dim < rank as i64,
            "chunk dimension {raw_dim} out of range for rank {rank}"
        );
        let dim = normalize_dim(raw_dim, rank);
        let length = x.dims()[dim]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("chunk requires a concrete chunk dimension"))?;
        let chunks = chunks as usize;
        let sizes: Vec<usize> = if length == 0 {
            // torch returns `chunks` empty tensors for a zero-length axis.
            vec![0; chunks]
        } else {
            let chunk_size = length.div_ceil(chunks);
            let full_chunks = length / chunk_size;
            let tail = length % chunk_size;
            let mut sizes = vec![chunk_size; full_chunks];
            if tail != 0 {
                sizes.push(tail);
            }
            sizes
        };
        let mut outputs = Vec::with_capacity(sizes.len());
        let mut start = 0usize;
        for size in sizes {
            let end = start + size;
            outputs.push(x.slice_along(start..end, dim));
            start = end;
        }
        self.bind_flat_outputs(node, outputs)
    }

    /// `ravel.default`: flatten to 1-D. A rank-0 tensor becomes shape `[1]`.
    pub(super) fn translate_ravel(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        Ok(if x.rank() == 0 {
            x.unsqueeze(0)
        } else {
            x.flatten()
        })
    }

    /// `view_as.default`: `self.view(other.shape)`.
    pub(super) fn translate_view_as(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let other = self.get_input_tensor(node, 1)?;
        let target = other.dims();
        Ok(reshape_view(x, &target))
    }

    /// `reshape_as.default`: `self.reshape(other.shape)` (may copy).
    pub(super) fn translate_reshape_as(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let other = self.get_input_tensor(node, 1)?;
        let target = other.dims();
        Ok(reshape_view(x, &target))
    }

    /// `expand_as.default`: `self.expand(other.shape)`.
    pub(super) fn translate_expand_as(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let other = self.get_input_tensor(node, 1)?;
        Ok(super::broadcast_to(x, &other.dims()))
    }

    /// `sum_to_size.default`: sum the leading extra dims and every target
    /// dim that is 1 where the input is not, then reshape to the target.
    pub(super) fn translate_sum_to_size(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let target: Vec<IntExpr> = self
            .get_ints_arg(node, 1)?
            .into_iter()
            .map(|v| IntExpr::from(v.max(0) as usize))
            .collect();
        let rank = x.rank();
        let target_len = target.len();
        anyhow::ensure!(
            target_len <= rank,
            "sum_to_size target rank {target_len} exceeds input rank {rank}"
        );
        let leading = rank - target_len;
        let mut axes: Vec<usize> = (0..leading).collect();
        for (j, tdim) in target.iter().enumerate() {
            let axis = leading + j;
            if tdim.to_usize() == Some(1) && x.dims()[axis].to_usize() != Some(1) {
                axes.push(axis);
            }
        }
        let reduced = if axes.is_empty() { x } else { x.sum(axes) };
        if target.is_empty() {
            return Ok(reduced);
        }
        Ok(reshape_view(reduced, &target))
    }

    /// `atleast_1d` / `atleast_2d` / `atleast_3d`: prepend and/or append
    /// size-1 dims per torch's rules (atleast_3d appends its final dim).
    pub(super) fn translate_atleast_dims(&mut self, node: &Node, n: usize) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        Ok(match n {
            1 => {
                if x.rank() < 1 {
                    x.unsqueeze(0)
                } else {
                    x
                }
            }
            2 => {
                let mut out = x;
                while out.rank() < 2 {
                    out = out.unsqueeze(0);
                }
                out
            }
            3 => {
                let mut out = x;
                while out.rank() < 2 {
                    out = out.unsqueeze(0);
                }
                if out.rank() < 3 {
                    let at = out.rank();
                    out = out.unsqueeze(at);
                }
                out
            }
            other => bail!("atleast_{other}d is not a supported rank"),
        })
    }

    /// `glu.default`: split `dim` in half, `a * sigmoid(b)`.
    pub(super) fn translate_glu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let rank = x.rank();
        anyhow::ensure!(rank > 0, "glu on a rank-0 tensor is not ported");
        let raw_dim = self
            .named_int_arg(node, "dim")
            .or_else(|| self.get_int_arg(node, 1).ok())
            .unwrap_or(-1);
        let dim = normalize_dim(raw_dim, rank);
        let length = x.dims()[dim]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("glu requires a concrete halved dimension"))?;
        anyhow::ensure!(
            length % 2 == 0,
            "glu: halving dimension must be even, got {length}"
        );
        let half = length / 2;
        let a = x.slice_along(0..half, dim);
        let b = x.slice_along(half..length, dim);
        Ok(a * b.sigmoid())
    }

    // ---------------------------------------------------------------
    // Elementwise activations
    // ---------------------------------------------------------------

    pub(super) fn translate_celu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let alpha = self
            .named_float_arg(node, "alpha")
            .or_else(|| node.inputs.get(1).and_then(|i| i.arg.as_float()))
            .unwrap_or(1.0);
        anyhow::ensure!(alpha != 0.0, "celu: alpha cannot be zero");
        let zero = self.constant_like(x, 0.0);
        let alpha_t = self.constant_like(x, alpha);
        let rhs = alpha_t * self.expm1_tensor(x / alpha_t);
        Ok(self.select(x.gt(zero), x, rhs))
    }

    pub(super) fn translate_selu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let alpha = self.constant_like(x, 1.673_263_242_354_377_2);
        let scale = self.constant_like(x, 1.050_700_987_355_480_5);
        let zero = self.constant_like(x, 0.0);
        let rhs = alpha * self.expm1_tensor(x);
        Ok(scale * self.select(x.gt(zero), x, rhs))
    }

    pub(super) fn translate_hardsigmoid(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        // Eager uses `clamp(x + 3, 0, 6) / 6` (bit-exact vs. `clamp(x/6 +
        // 0.5, 0, 1)` for a large sample).
        let shifted = x + self.constant_like(x, 3.0);
        let clamped = self.clamp_tensor(shifted, 0.0, 6.0);
        Ok(clamped / self.constant_like(clamped, 6.0))
    }

    pub(super) fn translate_log_sigmoid(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let zero = self.constant_like(x, 0.0);
        // min(0, x); NaN flows through the `- log1p(...)` term below.
        let min = self.select(x.lt(zero), x, zero);
        let one = self.constant_like(x, 1.0);
        let z = self.exp_tensor(x.abs() * -1.0);
        Ok(min - (z + one).log())
    }

    pub(super) fn translate_mish(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        // softplus(beta=1, threshold=20), then x * tanh(softplus(x)).
        let one = self.constant_like(x, 1.0);
        let threshold = self.constant_like(x, 20.0);
        let softplus_regular = (self.exp_tensor(x) + one).log();
        let softplus = self.select(x.gt(threshold), x, softplus_regular);
        Ok(x * softplus.tanh())
    }

    pub(super) fn translate_softsign(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let one = self.constant_like(x, 1.0);
        Ok(x / (one + x.abs()))
    }

    /// `prelu.default`: `where(x > 0, x, x * weight)`, with the weight
    /// broadcast along torch's channel axis (dim 1 for rank >= 2, dim 0
    /// for rank 1, scalar/`[1]` for rank 0).
    pub(super) fn translate_prelu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let weight = self.get_input_tensor(node, 1)?;
        let rank = x.rank();
        let weight = if rank == 0 {
            if weight.rank() == 1 {
                anyhow::ensure!(
                    weight.dims()[0].to_usize() == Some(1),
                    "prelu: scalar input needs a size-1 weight"
                );
                weight.squeeze(0)
            } else {
                weight
            }
        } else if weight.rank() == 0 {
            super::broadcast_to(weight, &x.dims())
        } else {
            anyhow::ensure!(
                weight.rank() == 1,
                "prelu: weight must be a scalar or 1-D tensor"
            );
            // A rank-1 weight of size C lines up with the channel axis.
            let axis = if rank == 1 { 0 } else { 1 };
            let mut shaped = weight;
            if axis == 1 {
                shaped = shaped.unsqueeze(0);
            }
            while shaped.rank() < rank {
                let at = shaped.rank();
                shaped = shaped.unsqueeze(at);
            }
            shaped.expand(x.dims())
        };
        anyhow::ensure!(
            weight.dims() == x.dims() || rank == 0,
            "prelu: weight broadcast failed"
        );
        let zero = self.constant_like(x, 0.0);
        let negative = x * weight;
        Ok(self.select(x.gt(zero), x, negative))
    }

    /// `threshold.default`: `where(x <= threshold, value, x)`.
    pub(super) fn translate_threshold(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let threshold = self.get_float_arg(node, 1)?;
        let value = self.get_float_arg(node, 2)?;
        let threshold = self.constant_like(x, threshold);
        let value = self.constant_like(x, value);
        Ok(self.select(x.le(threshold), value, x))
    }

    /// `logit.default` with optional `eps`: clamp to `[eps, 1 - eps]` then
    /// `log(x / (1 - x))`. `eps = None` clamps to `[-1, 2]`, matching eager.
    pub(super) fn translate_logit(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let out_dtype = self.output_meta_dtype(node).unwrap_or(x.dtype);
        let x = if x.dtype == out_dtype {
            x
        } else {
            x.cast(out_dtype)
        };
        let eps = self
            .named_float_arg(node, "eps")
            .or_else(|| node.inputs.get(1).and_then(|i| i.arg.as_float()))
            .unwrap_or(-1.0);
        let lo = self.constant_like(x, eps);
        let hi = self.constant_like(x, 1.0 - eps);
        let upper = self.select(x.gt(hi), hi, x);
        let clamped = self.select(upper.lt(lo), lo, upper);
        let one = self.constant_like(clamped, 1.0);
        Ok((clamped / (one - clamped)).log())
    }

    /// `logaddexp.default`: `where(inf_mask, a, max + log1p(exp(min - max)))`
    /// with `inf_mask = !isfinite(a) && a == b`, exactly as eager.
    pub(super) fn translate_logaddexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let b = self.get_input_tensor(node, 1)?;
        let (a, b) = util::ensure_same_dtype(a, b);
        let (a, b) = util::broadcast_binary(a, b);
        let mask = a.ge(b);
        let max = self.select(mask, a, b);
        let min = self.select(mask, b, a);
        let one = self.constant_like(max, 1.0);
        let log_term = (self.exp_tensor(min - max) + one).log();
        let regular = max + log_term;
        let is_inf = self.is_inf(a);
        let is_nan = self.is_nan(a);
        let not_finite = self.bool_or(is_inf, is_nan);
        let both_equal = a.eq(b);
        let inf_mask = self.bool_and(not_finite, both_equal);
        Ok(self.select(inf_mask, a, regular))
    }

    // ---------------------------------------------------------------
    // Inference dropout and clamp family
    // ---------------------------------------------------------------

    /// Inference-mode dropout variants: identity. A `training=True` flag
    /// would be stochastic and cannot be reproduced, so it bails.
    pub(super) fn translate_dropout(&mut self, node: &Node) -> Result<()> {
        let x = self.get_input_tensor(node, 0)?;
        if let Some(training) = node.inputs.iter().find_map(|input| input.arg.as_bool())
            && training
        {
            bail!("{} in training mode is not reproducible", node.target);
        }
        self.bind_outputs(node, vec![x])
    }

    /// `clamp_min.Tensor` (`minimum = true`) / `clamp_max.Tensor`.
    pub(super) fn translate_clamp_extremum(
        &mut self,
        node: &Node,
        minimum: bool,
    ) -> Result<GraphTensor> {
        if minimum {
            self.binary(&node.inputs, |a, b| a.maximum(b))
        } else {
            self.binary(&node.inputs, |a, b| a.minimum(b))
        }
    }

    // ---------------------------------------------------------------
    // Private helpers
    // ---------------------------------------------------------------

    /// `exp` keeping the tensor's own dtype (matches the parked translator's
    /// `real_exp`; the core `.exp()` hardcodes an f32 log2(e)).
    fn exp_tensor(&mut self, x: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(x, std::f64::consts::LOG2_E);
        (x * log2_e).exp2()
    }

    /// `exp(x) - 1`, matching the parked `translate_expm1` (the runtime has
    /// no `log1p`/`expm1` primitive; precision follows the project's existing
    /// `(1 + x).log()` convention).
    fn expm1_tensor(&mut self, x: GraphTensor) -> GraphTensor {
        let one = self.constant_like(x, 1.0);
        self.exp_tensor(x) - one
    }

    /// `clamp(x, lo, hi)` as a structural select so NaN propagates.
    fn clamp_tensor(&mut self, x: GraphTensor, lo: f64, hi: f64) -> GraphTensor {
        let lo_t = self.constant_like(x, lo);
        let hi_t = self.constant_like(x, hi);
        let lower = self.select(x.lt(lo_t), lo_t, x);
        self.select(lower.gt(hi_t), hi_t, lower)
    }

    /// Bind a node's outputs by flattened name (handles the `as_tensors`
    /// form used by list-valued outputs such as `chunk`/`broadcast_tensors`).
    fn bind_flat_outputs(&mut self, node: &Node, values: Vec<GraphTensor>) -> Result<()> {
        let names = Self::tensor_output_names(node);
        anyhow::ensure!(
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(values: &[usize]) -> Vec<IntExpr> {
        values.iter().map(|v| IntExpr::from(*v)).collect()
    }

    /// `reshape_view` must produce the exact target extents for any target
    /// (the shared `util::reshape_tensor` transposes multi-split targets).
    #[test]
    fn reshape_view_matches_target() {
        for target in [
            vec![6, 4],
            vec![4, 6],
            vec![24],
            vec![1, 24],
            vec![24, 1],
            vec![2, 12],
            vec![2, 3, 4],
            vec![1, 2, 12],
        ] {
            let mut cx = Graph::new();
            let x = cx.tensor((2, 3, 4), DType::F32);
            let reshaped = reshape_view(x, &dims(&target));
            assert_eq!(reshaped.dims(), dims(&target), "target {target:?}");
        }
    }
}
