//! Small ATen compositions shared by all runtimes. No data-dependent unrolling.
use super::{Translator, util};
use crate::pt2_schema::Node;
use anyhow::{Result, ensure};
use luminal::prelude::*;

impl Translator<'_> {
    pub(super) fn translate_mvlgamma(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let p = self
            .named_int_arg(node, "p")
            .ok_or_else(|| anyhow::anyhow!("mvlgamma requires integer p"))?;
        ensure!(p >= 1, "mvlgamma requires positive p");
        let mut out = self.constant_like(
            x,
            (p as f64 * (p - 1) as f64 / 4.0) * std::f64::consts::PI.ln(),
        );
        for i in 0..p {
            let offset = self.constant_like(x, -(i as f64) / 2.0);
            let shifted = x + offset;
            let name = format!("__mvlgamma_{i}");
            self.values.insert(name.clone(), shifted);
            let mut arg = node.clone();
            arg.inputs.truncate(1);
            if let crate::pt2_schema::Argument::Tensor(input) = &mut arg.inputs[0].arg {
                input.as_tensor.name = name;
            }
            out += self.translate_lgamma(&arg)?;
        }
        Ok(out)
    }
    pub(super) fn translate_rrelu_inference(&mut self, node: &Node) -> Result<()> {
        ensure!(
            !self.named_bool_arg(node, "training").unwrap_or(false),
            "training RReLU requires native RNG dispatch"
        );
        let x = self.operand(&node.inputs[0])?;
        let noise = self.raw_operand(&node.inputs[1])?;
        let lower = self.named_float_arg(node, "lower").unwrap_or(0.125);
        let upper = self.named_float_arg(node, "upper").unwrap_or(1.0 / 3.0);
        let alpha = self.constant_like(x, (lower + upper) / 2.0);
        let zero = self.constant_like(x, 0.0);
        let out = self.select(x.le(zero), x * alpha, x);
        self.bind_outputs(node, vec![out, noise])
    }
    pub(super) fn translate_nextafter(&mut self, node: &Node) -> Result<GraphTensor> {
        // Work at the storage dtype, not opmath: one representable step in
        // half/bfloat16 must not become one step in float32 followed by rounding.
        let dtype = self.output_meta_dtype(node)?;
        let a = self.raw_operand(&node.inputs[0])?.cast(dtype);
        let b = self.raw_operand(&node.inputs[1])?.cast(dtype);
        let (a, b) = util::broadcast_binary(a, b);
        let int = match dtype {
            DType::F16 | DType::Bf16 => DType::I16,
            DType::F32 => DType::Int,
            DType::F64 => DType::I64,
            _ => anyhow::bail!("nextafter requires IEEE floating operands"),
        };
        let a_bits = a.bits_to_bools();
        let b_bits = b.bits_to_bools();
        let width = dtype.logical_bits();
        let axis = a.rank();
        let bits = a_bits.bools_to_bits(int).squeeze(axis).cast(DType::U64);
        let other = b_bits.bools_to_bits(int).squeeze(axis).cast(DType::U64);
        let negative = a_bits.slice_along(width - 1..width, axis).squeeze(axis);
        let b_negative = b_bits.slice_along(width - 1..width, axis).squeeze(axis);
        let sign_diff = negative
            .logical_and(b_negative.logical_not())
            .logical_or(negative.logical_not().logical_and(b_negative));
        let same_sign_increase = self.select(negative, other.lt(bits), bits.lt(other));
        let increasing = self.select(sign_diff, negative, same_sign_increase);
        let increment = negative
            .logical_and(increasing.logical_not())
            .logical_or(negative.logical_not().logical_and(increasing));
        let one = self
            .cx
            .constant_i64(1)
            .cast(DType::U64)
            .expand_rhs(a.dims());
        // UInt64 deliberately supplies modulo arithmetic. Signed Int/Int64
        // arithmetic is non-wrapping in core and cannot step arbitrary codes.
        let step = self.select(increment, bits + one, bits - one);
        let stepped = step
            .bits_to_bools()
            .slice_along(0..width, axis)
            .bools_to_bits(dtype)
            .squeeze(axis);
        let tiny = one
            .bits_to_bools()
            .slice_along(0..width, axis)
            .bools_to_bits(dtype)
            .squeeze(axis);
        let negative_code = ((1u64 << (width - 1)) | 1) as i64;
        let tiny_negative = self
            .cx
            .constant_i64(negative_code)
            .expand_rhs(a.dims())
            .bits_to_bools()
            .slice_along(0..width, axis)
            .bools_to_bits(dtype)
            .squeeze(axis);
        let zero = self.cx.constant_i32(0).expand_rhs(a.dims());
        let a_zero = a_bits
            .slice_along(0..width - 1, axis)
            .cast(DType::Int)
            .sum(vec![axis])
            .eq(zero);
        let b_zero = b_bits
            .slice_along(0..width - 1, axis)
            .cast(DType::Int)
            .sum(vec![axis])
            .eq(zero);
        let from_zero = self.select(b_negative, tiny_negative, tiny);
        let out = self.select(a_zero, from_zero, stepped);
        let both_zero = self.bool_and(a_zero, b_zero);
        let same = self.bool_or(bits.eq(other), both_zero);
        let out = self.select(same, b, out);
        let a_nan = self.is_nan(a);
        let b_nan = self.is_nan(b);
        let any_nan = self.bool_or(a_nan, b_nan);
        let nan = self.select(a_nan, a, b);
        Ok(self.select(any_nan, nan, out))
    }

    pub(super) fn translate_rms_norm(&mut self, node: &Node) -> Result<GraphTensor> {
        Ok(self.rms_norm_parts(node)?.0)
    }

    pub(super) fn translate_fused_rms_norm(&mut self, node: &Node) -> Result<()> {
        let (value, rstd) = self.rms_norm_parts(node)?;
        let names = Self::tensor_output_names(node);
        ensure!(
            !names.is_empty() && names.len() <= 2,
            "fused RMSNorm expects output and optional rstd"
        );
        for (name, tensor) in names.into_iter().zip([value, rstd]) {
            let shape = self.tensor_meta_to_shape(self.tensor_meta(&name)?)?;
            self.bind_value(name, super::util::reshape_tensor(tensor, &shape));
        }
        Ok(())
    }

    fn rms_norm_parts(&mut self, node: &Node) -> Result<(GraphTensor, GraphTensor)> {
        let x = self.operand(&node.inputs[0])?;
        let normalized = self.get_ints_arg(node, 1)?;
        ensure!(
            !normalized.is_empty() && normalized.len() <= x.rank(),
            "invalid RMS normalization dimensions"
        );
        for (actual, expected) in x.dims()[x.rank() - normalized.len()..]
            .iter()
            .zip(&normalized)
        {
            ensure!(
                util::same_dim(*actual, IntExpr::from(*expected)),
                "RMS normalization shape mismatch"
            );
        }
        let eps = node
            .inputs
            .iter()
            .find(|i| i.name == "eps")
            .and_then(|i| i.arg.as_float())
            // ATen chooses the omitted epsilon at the arithmetic dtype:
            // half and bfloat16 are widened to float32 before this choice.
            .unwrap_or(match x.dtype {
                DType::F64 => f64::EPSILON,
                _ => f32::EPSILON as f64,
            });
        let axes = (x.rank() - normalized.len()..x.rank()).collect::<Vec<_>>();
        let square_mean = (x * x).mean(axes.clone());
        let epsilon = self.constant_like(square_mean, eps);
        let mut scale = (square_mean + epsilon).sqrt().reciprocal();
        for axis in axes {
            scale = scale.unsqueeze(axis);
        }
        let mut result = x * scale.expand(x.dims());
        if let Some(weight) = node.inputs.iter().find(|i| i.name == "weight")
            && let Some(weight) = self.optional_tensor_operand(weight)?
        {
            result *= super::broadcast_to(weight, &x.dims());
        }
        Ok((result, scale))
    }

    pub(super) fn translate_small_math(
        &mut self,
        node: &Node,
        target: &str,
    ) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let zero = self.constant_like(x, 0.0);
        let one = self.constant_like(x, 1.0);
        let result = match target {
            "logit.default" => {
                let mut x = x;
                if let Some(eps) = self.named_float_arg(node, "eps") {
                    // ATen clamps as `x < lo ? lo : (x > hi ? hi : x)` with
                    // `lo = eps` and `hi = 1 - eps`, both in the arithmetic
                    // dtype. The order of the two tests is the semantics: when
                    // `eps > 0.5` the bounds cross (`lo > hi`), and inputs
                    // below `lo` take `lo` while every other input takes `hi`.
                    // `min(max(x, lo), hi)` collapses that to `hi` everywhere,
                    // so the clamp must be this nested select. Only the ordered
                    // `lt`/`gt` predicates are used: both are false for NaN, so
                    // a NaN input passes through, and a NaN `eps` clamps
                    // nothing. A negative `eps` (the kernel's "no clamp") gives
                    // the same values either way: `x < lo < 0` and
                    // `x > hi > 1` are already NaN in `log(x / (1 - x))`.
                    let lo = self.constant_like(x, eps);
                    let hi = one - lo;
                    let upper = self.select(x.gt(hi), hi, x);
                    x = self.select(x.lt(lo), lo, upper);
                }
                (x / (one - x)).log()
            }
            "logaddexp.default" | "logaddexp2.default" => {
                let y = self.operand(&node.inputs[1])?;
                let (x, y) = util::broadcast_binary(x, y);
                let max = x.maximum(y);
                let delta = -(x - y).abs();
                let out = if target == "logaddexp2.default" {
                    let one = self.constant_like(delta, 1.0);
                    max + (one + delta.exp2()).log2()
                } else {
                    let one = self.constant_like(delta, 1.0);
                    max + (one + delta.exp()).log()
                };
                // equal infinities have NaN differences but a defined sum.
                let extra = self.constant_like(
                    x,
                    if target == "logaddexp2.default" {
                        1.0
                    } else {
                        std::f64::consts::LN_2
                    },
                );
                let result = self.select(x.eq(y), x + extra, out);
                let nan = self.is_nan(y);
                self.select(nan, y, result)
            }
            "xlogy.Tensor" | "special_xlog1py.default" => {
                let y = self.operand(&node.inputs[1])?;
                let (x, y) = util::broadcast_binary(x, y);
                let z = self.constant_like(x, 0.0);
                let log = if target == "special_xlog1py.default" {
                    (y + 1.0).log()
                } else {
                    y.log()
                };
                let normal = self.select(x.eq(z), z, x * log);
                let nan = self.is_nan(y);
                self.select(nan, y, normal)
            }
            "sinc.default" => {
                let pi = self.constant_like(x, std::f64::consts::PI);
                let y = x * pi;
                self.select(x.eq(zero), one, y.sin() / y)
            }
            "softplus.default" | "mish.default" => {
                let beta = self.named_float_arg(node, "beta").unwrap_or(1.0);
                let threshold = self.named_float_arg(node, "threshold").unwrap_or(20.0);
                let beta = self.constant_like(x, beta);
                let threshold = self.constant_like(x, threshold);
                let z = x * beta;
                let soft = self.select(z.gt(threshold), x, (one + z.exp()).log() / beta);
                if target == "mish.default" {
                    x * soft.tanh()
                } else {
                    soft
                }
            }
            "celu.default" => {
                let alpha = self.named_float_arg(node, "alpha").unwrap_or(1.0);
                ensure!(alpha != 0.0, "CELU alpha must be nonzero");
                let alpha = self.constant_like(x, alpha);
                self.select(x.gt(zero), x, alpha * ((x / alpha).exp() - one))
            }
            "clamp_min.Tensor" | "clamp_max.Tensor" => {
                let other = self.operand(&node.inputs[1])?;
                let (x, other) = util::broadcast_binary(x, other);
                if target == "clamp_min.Tensor" {
                    x.maximum(other)
                } else {
                    x.minimum(other)
                }
            }
            _ => anyhow::bail!("unknown small composition {target}"),
        };
        let nan = self.is_nan(x);
        Ok(self.select(nan, x, result))
    }

    pub(super) fn translate_simple_loss(
        &mut self,
        node: &Node,
        target: &str,
    ) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let y = self.operand(&node.inputs[1])?;
        let (x, y) = util::broadcast_binary(x, y);
        let one = self.constant_like(x, 1.0);
        let loss = match target {
            "smooth_l1_loss.default" => {
                let beta = self.named_float_arg(node, "beta").unwrap_or(1.0);
                let d = (x - y).abs();
                if beta == 0.0 {
                    d
                } else {
                    let b = self.constant_like(d, beta);
                    self.select(d.lt(b), d * d / (b * 2.0), d - b * 0.5)
                }
            }
            "soft_margin_loss.default" => (one + (-x * y).exp()).log(),
            "binary_cross_entropy_with_logits.default" => {
                let zero = self.constant_like(x, 0.0);
                let log_sigmoid = x.minimum(zero) - (one + (-x.abs()).exp()).log();
                let mut weight = one;
                if let Some(p) = node.inputs.iter().find(|i| i.name == "pos_weight")
                    && let Some(p) = self.optional_tensor_operand(p)?
                {
                    weight = one + (super::broadcast_to(p, &x.dims()) - one) * y;
                }
                let mut out = (one - y) * x - weight * log_sigmoid;
                if let Some(w) = node.inputs.iter().find(|i| i.name == "weight")
                    && let Some(w) = self.optional_tensor_operand(w)?
                {
                    out *= super::broadcast_to(w, &x.dims());
                }
                out
            }
            _ => anyhow::bail!("unknown loss {target}"),
        };
        let axes = (0..loss.rank()).collect::<Vec<_>>();
        Ok(match self.named_int_arg(node, "reduction").unwrap_or(1) {
            0 => loss,
            1 => loss.mean(axes),
            2 => loss.sum(axes),
            r => anyhow::bail!("invalid loss reduction {r}"),
        })
    }

    pub(super) fn translate_glu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let axis = util::normalize_dim(self.named_int_arg(node, "dim").unwrap_or(-1), x.rank());
        let len = x.dims()[axis];
        ensure!(
            len.to_usize().is_none_or(|n| n % 2 == 0),
            "GLU requires an even split dimension"
        );
        let half = len / 2;
        Ok(x.slice_along(IntExpr::from(0)..half, axis) * x.slice_along(half..len, axis).sigmoid())
    }

    pub(super) fn translate_nansum(&mut self, node: &Node) -> Result<GraphTensor> {
        // Remove NaNs in the source dtype before reduction promotion. Casting
        // to an integral result dtype first would lose the NaN classification.
        let x = self.raw_operand(&node.inputs[0])?;
        let nan = self.is_nan(x);
        let zero = self.constant_like(x, 0.0);
        let clean = self.select(nan, zero, x);
        let mut inner = node.clone();
        let name = "__nansum_clean".to_string();
        self.values.insert(name.clone(), clean);
        if let crate::pt2_schema::Argument::Tensor(arg) = &mut inner.inputs[0].arg {
            arg.as_tensor.name = name;
        }
        self.translate_reduction(&inner, super::ops::ReductionOp::Sum)
    }
}
