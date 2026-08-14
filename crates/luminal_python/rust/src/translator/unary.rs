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
        let a = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(f(a))
    }

    /// Translate `aten.acos.default` into existing elementwise HLIR primitives.
    ///
    /// For `x >= 0`, approximate `acos(x)` as `sqrt(1 - x) * P(x)`, where
    /// `P` is a degree-8 Chebyshev approximation of
    /// `acos(x) / sqrt(1 - x)` on `[0, 1]`. Extend it to negative inputs with
    /// `acos(-x) = pi - acos(x)`. Factoring out the square-root endpoint
    /// behavior keeps the polynomial smooth and also makes out-of-domain real
    /// inputs produce NaN through `sqrt(1 - abs(x))`, matching PyTorch.
    ///
    /// PyTorch promotes integral and bool inputs to its default floating dtype.
    #[allow(clippy::excessive_precision)]
    pub(crate) fn translate_acos(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(self.real_acos(input))
    }

    /// Elementwise real acos used by both real ATen dispatch and compound
    /// complex inverse functions.
    #[allow(clippy::excessive_precision)]
    pub(crate) fn real_acos(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();

        // Horner form, highest-order coefficient first. The maximum absolute
        // approximation error in F32 is below 3e-7 over the real acos domain.
        let polynomial =
            self.constant_like(x, 0.000_684_531_8) * x - self.constant_like(x, 0.003_974_577_8);
        let polynomial = polynomial * x + self.constant_like(x, 0.011_028_381);
        let polynomial = polynomial * x - self.constant_like(x, 0.020_727_666);
        let polynomial = polynomial * x + self.constant_like(x, 0.032_571_17);
        let polynomial = polynomial * x - self.constant_like(x, 0.050_593_574);
        let polynomial = polynomial * x + self.constant_like(x, 0.089_030_14);
        let polynomial = polynomial * x - self.constant_like(x, 0.214_601_16);
        let half_pi = self.constant_like(x, std::f64::consts::FRAC_PI_2);
        let polynomial = polynomial * x + half_pi;
        let one = self.constant_like(x, 1.0);
        let positive = polynomial * (one - x).sqrt();

        let zero = self.constant_like(input, 0.0);
        let negative = input.lt(zero).cast(input.dtype);
        let pi = self.constant_like(input, std::f64::consts::PI);
        let two = self.constant_like(input, 2.0);
        positive + negative * (pi - two * positive)
    }

    /// Translate `aten.acosh.default` into existing elementwise HLIR primitives.
    ///
    /// The textbook `log(x + sqrt(x * x - 1))` form overflows before the log
    /// for large finite inputs, especially in F16. Use the equivalent form
    ///
    /// `log(x) + log(1 + sqrt(1 - 1 / x^2))`
    ///
    /// instead. For real inputs below one, either the square root or `log(x)`
    /// naturally produces NaN, matching PyTorch's real-domain behavior.
    /// PyTorch promotes integral and bool inputs to its default floating dtype,
    /// while floating inputs retain their dtype.
    pub(crate) fn translate_acosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(self.real_acosh(input))
    }

    /// Elementwise real acosh used by both real ATen dispatch and compound
    /// complex inverse functions.
    pub(crate) fn real_acosh(&mut self, input: GraphTensor) -> GraphTensor {
        let reciprocal_squared = input.reciprocal().square();
        let one = self.constant_like(input, 1.0);
        input.log() + (one + (one - reciprocal_squared).sqrt()).log()
    }

    fn unary_input(&mut self, node: &Node) -> Result<GraphTensor> {
        Ok(self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?))
    }

    pub(crate) fn translate_exp(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_exp(input))
    }

    /// Keep log2(e) in the tensor's actual dtype. `GraphTensor::exp` uses its
    /// historical F32 scalar API, which is not precise enough for F64 PT2.
    pub(crate) fn real_exp(&mut self, input: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(input, std::f64::consts::LOG2_E);
        (input * log2_e).exp2()
    }

    pub(crate) fn translate_cos(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_cos(input))
    }

    /// Keep pi/2 in the tensor's actual dtype rather than narrowing F64.
    pub(crate) fn real_cos(&mut self, input: GraphTensor) -> GraphTensor {
        (self.constant_like(input, std::f64::consts::FRAC_PI_2) - input).sin()
    }

    pub(crate) fn translate_asin(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_asin(input))
    }

    /// `asin(x) = atan(x / sqrt(1 - x^2))` is accurate around zero and reaches
    /// the correct signed infinities at both endpoints. The square root also
    /// supplies PyTorch's NaN for real inputs outside [-1, 1].
    pub(crate) fn real_asin(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let denominator = (one - input.square()).sqrt();
        let ratio = input / denominator;
        self.real_atan(ratio)
    }

    pub(crate) fn translate_asinh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_asinh(input))
    }

    /// Stable real asinh. A short odd series preserves tiny values that would
    /// be rounded away by `log(|x| + hypot(x, 1))`; the logarithmic form covers
    /// the rest, with `log(|x|) + log(2)` preventing finite overflow near the
    /// largest representable value.
    pub(crate) fn real_asinh(&mut self, input: GraphTensor) -> GraphTensor {
        let x = self.real_abs(input);
        let x2 = x.square();
        let mut series = self.constant_like(x, 35.0 / 1152.0);
        series = series * x2 - self.constant_like(x, 5.0 / 112.0);
        series = series * x2 + self.constant_like(x, 3.0 / 40.0);
        series = series * x2 - self.constant_like(x, 1.0 / 6.0);
        let series = x + x * x2 * series;

        let one = self.constant_like(x, 1.0);
        let magnitude = x + self.real_hypot(x, one);
        let regular = magnitude.log();
        let log_two = self.constant_like(x, std::f64::consts::LN_2);
        let large = x.log() + log_two;
        let large_threshold = self.constant_like(
            x,
            match x.dtype {
                DType::F16 => 32_752.0,
                DType::F32 => (f32::MAX / 2.0) as f64,
                DType::F64 => f64::MAX / 2.0,
                DType::Bf16 => (f32::MAX / 2.0) as f64,
                other => unreachable!("asinh has non-floating dtype {other:?}"),
            },
        );
        let nonsmall = self.select(x.gt(large_threshold), large, regular);
        let threshold = self.constant_like(x, 0.125);
        let result = self.select(x.le(threshold), series, nonsmall);
        self.copy_sign(result, input)
    }

    pub(crate) fn translate_atan(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_atan(input))
    }

    /// Range-reduced odd Taylor series for atan. Reciprocal reduction maps the
    /// full real line to [0, 1], then a pi/4 transform bounds the polynomial
    /// argument by sqrt(2)-1. Degree 27 keeps F64 error below 1e-12 while using
    /// only ordinary real HLIR primitives.
    pub(crate) fn real_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = self.real_abs(input);
        let one = self.constant_like(x, 1.0);
        let reciprocal_branch = x.gt(one);
        let reduced = self.select(reciprocal_branch, x.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.select(reciprocal_branch, half_pi - base, base);
        self.copy_sign(angle, input)
    }

    pub(crate) fn translate_atanh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        Ok(self.real_atanh(input))
    }

    /// Stable real atanh. The series avoids cancellation near zero; the log
    /// difference supplies infinities at +/-1 and NaN outside the real domain.
    pub(crate) fn real_atanh(&mut self, input: GraphTensor) -> GraphTensor {
        let x2 = input.square();
        let mut series = self.constant_like(input, 1.0 / 11.0);
        series = series * x2 + self.constant_like(input, 1.0 / 9.0);
        series = series * x2 + self.constant_like(input, 1.0 / 7.0);
        series = series * x2 + self.constant_like(input, 1.0 / 5.0);
        series = series * x2 + self.constant_like(input, 1.0 / 3.0);
        let series = input + input * x2 * series;

        let one = self.constant_like(input, 1.0);
        let half = self.constant_like(input, 0.5);
        let regular = half * ((one + input).log() - (one - input).log());
        let small = self.real_abs(input).le(self.constant_like(input, 0.125));
        self.select(small, series, regular)
    }

    pub(crate) fn translate_cosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let output_dtype = input.dtype;
        let opmath = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            input.cast(DType::F32)
        } else {
            input
        };
        Ok(self.real_cosh(opmath).cast(output_dtype))
    }

    pub(crate) fn real_cosh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.real_exp(input) + self.real_exp(input * -1.0))
    }

    pub(crate) fn real_sinh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        let result = half * (self.real_exp(input) - self.real_exp(input * -1.0));
        let zero = self.is_zero(input);
        self.select(zero, input, result)
    }

    fn real_hypot(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let a = self.real_abs(a);
        let b = self.real_abs(b);
        let a_is_large = a.ge(b);
        let large = self.select(a_is_large, a, b);
        let small = self.select(a_is_large, b, a);
        let one = self.constant_like(large, 1.0);
        let large_is_zero = self.is_zero(large);
        let safe_large = self.select(large_is_zero, one, large);
        let ratio = small / safe_large;
        let finite = large * (one + ratio.square()).sqrt();
        let a_inf = self.is_inf(a);
        let b_inf = self.is_inf(b);
        let any_inf = self.bool_or(a_inf, b_inf);
        let infinity = self.constant_like(finite, f64::INFINITY);
        self.select(any_inf, infinity, finite)
    }

    pub(crate) fn translate_trunc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let integer_dtype = if input.dtype == DType::F64 {
            DType::I64
        } else {
            DType::Int
        };
        let truncated = input.cast(integer_dtype).cast(input.dtype);
        let threshold = self.constant_like(
            input,
            if input.dtype == DType::F64 {
                9_223_372_036_854_775_808.0
            } else {
                2_147_483_648.0
            },
        );
        // Beyond the integer range every representable float is already
        // integral. Preserve those values, non-finites, and signed zero.
        let at_limit = self.real_abs(input).ge(threshold);
        let is_inf = self.is_inf(input);
        let is_nan = self.is_nan(input);
        let nonfinite = self.bool_or(is_inf, is_nan);
        let preserve = self.bool_or(at_limit, nonfinite);
        let is_zero = self.is_zero(input);
        let preserve = self.bool_or(preserve, is_zero);
        Ok(self.select(preserve, input, truncated))
    }

    /// Translate `aten.gelu`, honoring the `approximate` kwarg. PyTorch's default
    /// (`approximate="none"`) is the exact erf form; `"tanh"` selects the tanh
    /// approximation. Mapping both to a single `gelu()` (as before) silently used the
    /// tanh approximation even when the model asked for exact, which accumulates
    /// visible error in deep GELU-heavy stacks (ViT, Whisper).
    pub(crate) fn translate_gelu(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        // PT2 serializes string args as {"as_string": "<value>"}; drill into the JSON.
        let approximate = node.inputs.iter().find_map(|input| {
            if input.name == "approximate"
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
        Ok(match approximate.as_deref() {
            Some("tanh") => a.gelu_fast_tanh_approximation(),
            _ => a.gelu(),
        })
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

        // torch computes LN statistics in fp32 (opmath) even for fp16/bf16
        // inputs — "For FP16 or BFloat16 inputs, ops should perform internal
        // math in FP32" (aten/src/ATen/OpMathType.h; used by
        // layer_norm_kernel.cpp as `opmath_t`). fp16 statistics overflow on
        // outlier activations (x^2 > 65504 at |x| > ~256 — the OPT-family
        // residual-stream profile).
        // Mirror translate_fused_rms_norm: normalize + affine in F32, cast
        // the result back to the input dtype.
        let out_dtype = input.dtype;
        let mut result = input.cast(DType::F32).layer_norm(axes, eps);

        // Apply weight (arg 2) if present and not None
        if let Some(weight_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let (r, w) = broadcast_binary(result, w);
            result = r * w;
        }

        // Apply bias (arg 3) if present and not None
        if let Some(bias_name) = node.inputs.get(3).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?.cast(DType::F32);
            let (r, b) = broadcast_binary(result, b);
            result = r + b;
        }

        Ok(result.cast(out_dtype))
    }

    /// `aten._fused_rms_norm` (F.rms_norm on CUDA): frontend `std_norm` +
    /// optional affine. Only `out` is consumed; `rstd` is DCE'd.
    pub(crate) fn translate_fused_rms_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let normalized_shape = self.get_ints_arg(node, 1)?;

        let ndim = input.shape.len();
        let num_norm_dims = normalized_shape.len();
        anyhow::ensure!(
            num_norm_dims <= ndim,
            "rms_norm normalized_shape rank {num_norm_dims} exceeds input rank {ndim}"
        );
        let axes: Vec<usize> = ((ndim - num_norm_dims)..ndim).collect();

        // eps (arg 3): eager resolves None to the fp32 machine epsilon
        // regardless of input dtype.
        let eps = self.get_float_arg(node, 3).unwrap_or(f32::EPSILON as f64) as f32;

        // Eager's fused kernel computes entirely in fp32 and casts the result
        // to the input dtype; matching it also handles mixed-dtype weights.
        let out_dtype = input.dtype;
        let mut result = input.cast(DType::F32).std_norm(axes, eps);

        // Apply weight (arg 2) if present and not None.
        if let Some(weight_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let (r, w) = broadcast_binary(result, w);
            result = r * w;
        }

        Ok(result.cast(out_dtype))
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

        // torch computes group-norm statistics in fp32 (opmath); fp16 stats
        // overflow on outlier activations. Normalize + affine in F32 and
        // cast back at the end (mirrors translate_fused_rms_norm).
        let out_dtype = input.dtype;
        // Flatten everything after the batch dim into one axis: (N, C, ...) -> (N, M),
        // where M = C * spatial. Group volumes are contiguous in this layout.
        let mut t = input.cast(DType::F32);
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
            let w = self.get_tensor(weight_name)?.cast(DType::F32);
            let w = w.expand_to_shape_on_axes(t.shape, non_channel_axes.clone());
            let (r, w) = broadcast_binary(t, w);
            t = r * w;
        }
        if let Some(bias_name) = node.inputs.get(2).and_then(|i| i.arg.as_tensor_name()) {
            let b = self.get_tensor(bias_name)?.cast(DType::F32);
            let b = b.expand_to_shape_on_axes(t.shape, non_channel_axes);
            let (r, b) = broadcast_binary(t, b);
            t = r + b;
        }

        Ok(t.cast(out_dtype))
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
                // No rounding mode is true division, which ATen builds with
                // build_borrowing_binary_float_op — so an integral input comes
                // back float, not cast back to `a.dtype`.
                Ok(match self.recorded_output_dtype(node) {
                    Some(dtype) => quotient.cast(dtype),
                    None => quotient,
                })
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
