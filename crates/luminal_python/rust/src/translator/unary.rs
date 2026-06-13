use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::{broadcast_binary, torch_dtype_int_to_luminal};

use super::Translator;

const ARGSORT_INPUT_ARG: usize = 0;
const ARGSORT_DIM_ARG: usize = 1;
const ARGSORT_DESCENDING_ARG: usize = 2;

const MASKED_FILL_INPUT_ARG: usize = 0;
const MASKED_FILL_MASK_ARG: usize = 1;
const MASKED_FILL_VALUE_ARG: usize = 2;

const FLOOR_DIVIDE_INPUT_ARG: usize = 0;
const FLOOR_DIVIDE_OTHER_ARG: usize = 1;

const DIV_MODE_INPUT_ARG: usize = 0;
const DIV_MODE_OTHER_ARG: usize = 1;

impl<'a> Translator<'a> {
    pub(crate) fn translate_argsort(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, ARGSORT_INPUT_ARG)?;
        let dim = if node.inputs.len() > ARGSORT_DIM_ARG {
            self.get_int_arg(node, ARGSORT_DIM_ARG).unwrap_or(-1)
        } else {
            -1
        };
        let descending = if node.inputs.len() > ARGSORT_DESCENDING_ARG {
            self.get_bool_arg(node, ARGSORT_DESCENDING_ARG)
                .unwrap_or(false)
        } else {
            false
        };
        let dim = crate::pt2_util::normalize_dim(dim, a.shape.len());
        // PyTorch's `torch.argsort` returns int64 unconditionally;
        // luminal's frontend `stable_argsort` returns i32 (storage-
        // efficient default for native Rust callers). Cast at the
        // PT2↔luminal boundary so the strict output-read path sees
        // an I64 buffer.
        Ok(a.stable_argsort(dim, descending).cast(DType::I64))
    }

    pub(crate) fn translate_unary_op(
        &mut self,
        node: &Node,
        f: impl Fn(GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        Ok(f(a))
    }

    pub(crate) fn translate_to_copy(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        for input in &node.inputs {
            if input.name == "dtype" {
                let dtype_int = input
                    .arg
                    .as_int()
                    .map(|i| i as u32)
                    .or_else(|| input.arg.as_scalar_type());
                if let Some(d) = dtype_int {
                    let dtype = torch_dtype_int_to_luminal(d);
                    // Skip emitting a Cast op when the dtype already matches —
                    // PT2 graphs frequently emit `_to_copy` purely as a clone hint
                    // (e.g. dtype=float32 on a tensor that is already F32), and
                    // every redundant Cast inflates the graph and survives until
                    // optimization passes can prove it as a no-op.
                    return Ok(if a.dtype == dtype { a } else { a.cast(dtype) });
                }
            }
        }
        Ok(a)
    }

    pub(crate) fn translate_layer_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let normalized_shape = self.get_ints_arg(node, 1)?;

        // Axes to normalize over = last N dims where N = len(normalized_shape)
        let ndim = input.shape.len();
        let num_norm_dims = normalized_shape.len();
        let axes: Vec<usize> = ((ndim - num_norm_dims)..ndim).collect();

        // eps is arg 4 (after input, normalized_shape, weight, bias), default 1e-5
        let eps = self.get_float_arg(node, 4).unwrap_or(1e-5) as f32;

        let mut result = input.layer_norm(axes, eps);

        // Apply weight (arg 2) if present and not None
        if let Some(weight_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?;
            let (r, w) = broadcast_binary(result, w);
            result = r * w;
        }

        // Apply bias (arg 3) if present and not None
        if let Some(bias_name) = node.inputs.get(3).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?;
            let (r, b) = broadcast_binary(result, b);
            result = r + b;
        }

        Ok(result)
    }

    /// Translate `aten.native_group_norm.default`.
    ///
    /// Schema: `native_group_norm(input, weight?, bias?, N, C, HxW, num_groups, eps)
    /// -> (out, mean, rstd)`. We only produce the normalized `out`; the `mean`/`rstd`
    /// outputs exist solely for the backward pass and are never consumed by inference
    /// graphs, so (like `translate_layer_norm`) we return a single tensor and let the
    /// dispatcher assign it to output[0] while the unused outputs are DCE'd.
    ///
    /// GroupNorm splits the `C` channels into `num_groups` groups and normalizes each
    /// `(batch, group)` slice jointly over its `group_size * spatial` elements, then
    /// applies a per-channel affine. We compose this from existing primitives (no new
    /// op): reshape so each group's volume is a single contiguous axis, `layer_norm`
    /// over that one axis, reshape back, then the affine.
    ///
    /// The per-group volume is flattened into ONE axis before normalizing rather than
    /// reducing over multiple axes: the multi-axis reduction form is dropped by the
    /// e-graph during cleanup when composed into deep conv chains (see the note in
    /// `examples/flux2/src/vae.rs`). Reshapes use `Expression` extents throughout, so
    /// dynamic batch and dynamic spatial dims are preserved.
    pub(crate) fn translate_group_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let num_groups = self.get_int_arg(node, 6)? as usize;
        let eps = self.get_float_arg(node, 7).unwrap_or(1e-5) as f32;

        let orig_dims = input.dims();
        let ndim = orig_dims.len();
        anyhow::ensure!(
            ndim >= 2,
            "group_norm expects input rank >= 2 (N, C, ...), got {ndim}"
        );

        // Channel count must be static to size the groups (it always is — channel
        // count is a model-config constant).
        let c = orig_dims[1]
            .to_usize()
            .ok_or_else(|| anyhow::anyhow!("group_norm requires a static channel dim"))?;
        anyhow::ensure!(
            num_groups != 0 && c % num_groups == 0,
            "group_norm: num_channels ({c}) must be a positive multiple of num_groups ({num_groups})"
        );
        let group_size = c / num_groups;

        // Per-group volume V = group_size * (product of spatial dims). Spatial extents
        // stay symbolic so dynamic spatial dims flow through.
        let spatial: Expression = orig_dims[2..].iter().cloned().product();
        let group_volume = spatial * Expression::from(group_size);

        // Flatten everything after the batch dim into one axis: (N, C, ...) -> (N, M),
        // where M = C * spatial. Group volumes are contiguous in this layout.
        let mut t = input;
        while t.shape.len() > 2 {
            t = t.merge_dims(1, 2);
        }
        // (N, M) -> (N, num_groups, group_volume): M / group_volume == num_groups.
        t = t.split_dims(1, group_volume);

        // Normalize over the single per-group axis (matches PyTorch: biased variance,
        // eps inside the sqrt).
        t = t.layer_norm(2, eps);

        // Reshape back to the original (N, C, ...spatial).
        t = t.merge_dims(1, 2); // (N, num_groups, V) -> (N, M)
        // Peel the trailing (non-batch) dims back off one at a time, left to right.
        let trailing = &orig_dims[1..];
        for i in 0..trailing.len().saturating_sub(1) {
            let suffix: Expression = trailing[i + 1..].iter().cloned().product();
            t = t.split_dims(1 + i, suffix);
        }

        // Per-channel affine on the channel axis (axis 1). weight/bias are shape (C,);
        // broadcast them onto every axis except the channel axis.
        let non_channel_axes: Vec<usize> = (0..ndim).filter(|&a| a != 1).collect();
        if let Some(weight_name) = node.inputs.get(1).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?;
            let w = w.expand_to_shape_on_axes(t.shape, non_channel_axes.clone());
            let (r, w) = broadcast_binary(t, w);
            t = r * w;
        }
        if let Some(bias_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?;
            let b = b.expand_to_shape_on_axes(t.shape, non_channel_axes);
            let (r, b) = broadcast_binary(t, b);
            t = r + b;
        }

        Ok(t)
    }

    pub(crate) fn translate_sign(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let zero = self
            .graph
            .constant_float(0.0)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        let pos = a.gt(zero).cast(DType::Int);
        let neg = a.lt(zero).cast(DType::Int);
        let signed = pos - neg;
        Ok(if a.dtype == DType::Int {
            signed
        } else {
            signed.cast(a.dtype)
        })
    }

    pub(crate) fn translate_bitwise_not(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        Ok(match a.dtype {
            DType::Bool => {
                let one = self
                    .graph
                    .constant_float(1.0)
                    .cast(DType::Int)
                    .expand_rhs(a.shape);
                (one - a.cast(DType::Int)).cast(DType::Bool)
            }
            DType::Int => (a + 1) * -1.0,
            other => {
                anyhow::bail!("bitwise_not only supports Bool/Int routing tensors, got {other:?}")
            }
        })
    }

    pub(crate) fn translate_masked_fill_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        // `masked_fill(input, mask, fill)` = `where(mask, fill, input)`.
        // Routes through the shared `where_formula` helper so we exercise
        // the exact same code path as `aten.where.self`, which is verified
        // to handle the bf16 cast-back correctly. Hand-rolling the same
        // formula directly here used to drift (egglog made different
        // rewrite choices on the rebuilt-locally graph), so we deliberately
        // re-use the helper.
        // `aten.masked_fill.Scalar(input, mask, fill)` ≡
        // `aten.where.self(mask, full_like(input, fill), input)`. The
        // `full_like + where` sequence is the verified-working path
        // (test: `where(mask, torch.zeros_like(x), x)` round-trips with
        // max_diff = 0); we reproduce its exact graph-build order here.
        // Hand-rolling the formula in any other shape (single-mul, F32
        // throughout, alternative constant-cast orderings) routes egglog
        // through a rewrite that returns an F32 buffer downstream-read as
        // bf16 — the every-other-element-zero pattern.
        let input = self.get_input_tensor(node, MASKED_FILL_INPUT_ARG)?;
        let mask = self.get_input_tensor(node, MASKED_FILL_MASK_ARG)?;
        let fill = self.get_float_arg(node, MASKED_FILL_VALUE_ARG)? as f32;
        let out_dtype = input.dtype;
        // Build fill_t exactly like translate_full_like does:
        //   constant_float(val).cast(dtype).expand_rhs(reference.shape)
        let fill_t = self
            .graph
            .constant_float(fill)
            .cast(out_dtype)
            .expand_rhs(input.shape);
        Ok(self.where_formula(mask, fill_t, input, out_dtype))
    }

    pub(crate) fn translate_floor_divide(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, FLOOR_DIVIDE_INPUT_ARG)?;
        let b = if let Some(name) = node
            .inputs
            .get(FLOOR_DIVIDE_OTHER_ARG)
            .and_then(|i| i.arg.as_tensor_name())
        {
            self.get_tensor(name)?
        } else {
            let scalar = self.get_float_arg(node, FLOOR_DIVIDE_OTHER_ARG)? as f32;
            self.graph
                .constant_float(scalar)
                .cast(a.dtype)
                .expand_rhs(a.shape)
        };
        let (a, b) = crate::pt2_util::ensure_same_dtype(a, b);
        let (a, b) = broadcast_binary(a, b);
        let quotient = a.cast(DType::F32) / b.cast(DType::F32);
        let trunc = quotient.cast(DType::Int).cast(DType::F32);
        let adjust = quotient.lt(trunc).cast(DType::F32);
        let floored = trunc - adjust;
        Ok(if a.dtype == DType::Int {
            floored.cast(DType::Int)
        } else {
            floored.cast(a.dtype)
        })
    }

    pub(crate) fn translate_div_tensor_mode(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, DIV_MODE_INPUT_ARG)?;
        let b = if let Some(name) = node
            .inputs
            .get(DIV_MODE_OTHER_ARG)
            .and_then(|i| i.arg.as_tensor_name())
        {
            self.get_tensor(name)?
        } else {
            let scalar = self.get_float_arg(node, DIV_MODE_OTHER_ARG)? as f32;
            self.graph
                .constant_float(scalar)
                .cast(a.dtype)
                .expand_rhs(a.shape)
        };
        let (a, b) = crate::pt2_util::ensure_same_dtype(a, b);
        let (a, b) = broadcast_binary(a, b);

        // Check rounding_mode kwarg. PT2 serializes string args as
        // {"as_string": "<value>"}, so we have to drill into the JSON.
        let rounding_mode = node.inputs.iter().find_map(|input| {
            if input.name == "rounding_mode"
                && let Argument::Other(val) = &input.arg
            {
                if let Some(s) = val.as_str() {
                    return Some(s.to_string());
                }
                if let Some(s) = val.get("as_string").and_then(|v| v.as_str()) {
                    return Some(s.to_string());
                }
            }
            None
        });

        let quotient = a.cast(DType::F32) / b.cast(DType::F32);
        match rounding_mode.as_deref() {
            Some("floor") => {
                let trunc = quotient.cast(DType::Int).cast(DType::F32);
                let adjust = quotient.lt(trunc).cast(DType::F32);
                let floored = trunc - adjust;
                Ok(if a.dtype == DType::Int {
                    floored.cast(DType::Int)
                } else {
                    floored.cast(a.dtype)
                })
            }
            Some("trunc") => Ok(if a.dtype == DType::Int {
                quotient.cast(DType::Int)
            } else {
                quotient.cast(DType::Int).cast(a.dtype)
            }),
            _ => {
                // No rounding mode — regular division
                Ok(quotient.cast(a.dtype))
            }
        }
    }

    pub(crate) fn translate_clamp(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let min_val = if node.inputs.len() > 1 {
            self.get_float_arg(node, 1).ok().map(|f| f as f32)
        } else {
            None
        };
        let max_val = if node.inputs.len() > 2 {
            self.get_float_arg(node, 2).ok().map(|f| f as f32)
        } else {
            None
        };

        let mut result = a;
        if let Some(min) = min_val {
            result = result.maximum_f32(min);
        }
        if let Some(max) = max_val {
            result = result.minimum_f32(max);
        }
        Ok(result)
    }

    /// `aten.clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None)`
    ///
    /// Unlike `clamp.default` (which takes Python scalar bounds), the `.Tensor`
    /// overload takes tensor bounds that appear as separate input nodes in the
    /// FX graph. PyTorch supports any NumPy-broadcastable bound shape:
    ///
    ///   - rank-0 (scalar wrapped in a tensor) — most common
    ///   - same shape as self (per-element clamp, e.g. learned bounds)
    ///   - any shape that broadcasts to self via right-align + size-1 expand
    ///     (e.g. `(3, 1)` against `(3, 4)` for per-row clamp; `(4,)` against
    ///     `(3, 4)` for per-column clamp; `(3, 4)` against `(2, 3, 4)`)
    ///
    /// We use `broadcast_binary` to right-align and expand both operands to a
    /// common shape before the elementwise max/min, matching PyTorch semantics
    /// across all three modes.
    ///
    /// Either bound may be absent (FX represents this as a non-tensor argument
    /// at the corresponding input slot), in which case we clamp to one side
    /// only.
    pub(crate) fn translate_clamp_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let min_tensor = node
            .inputs
            .get(1)
            .and_then(|i| i.arg.as_tensor_name())
            .map(|n| self.get_tensor(n))
            .transpose()?;
        let max_tensor = node
            .inputs
            .get(2)
            .and_then(|i| i.arg.as_tensor_name())
            .map(|n| self.get_tensor(n))
            .transpose()?;

        let mut result = a;
        if let Some(lo) = min_tensor {
            let lo = lo.cast(result.dtype);
            let (r, lo) = broadcast_binary(result, lo);
            result = r.maximum(lo);
        }
        if let Some(hi) = max_tensor {
            let hi = hi.cast(result.dtype);
            let (r, hi) = broadcast_binary(result, hi);
            result = r.minimum(hi);
        }
        Ok(result)
    }
}
