//! Unary/special-function lowerings (port batch 3). Real-domain only —
//! complex is skipped by decision; the composite helpers (`select`,
//! `constant_like`, predicates) live in `util`.

use anyhow::Result;
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, Node};

impl Translator<'_> {
    fn unary_input(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.output_meta_dtype(node).unwrap_or(x.dtype);
        Ok(if x.dtype == dtype { x } else { x.cast(dtype) })
    }

    /// Keep log2(e) in the tensor's own dtype.
    fn real_exp(&mut self, input: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(input, std::f64::consts::LOG2_E);
        (input * log2_e).exp2()
    }

    pub(super) fn translate_exp(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        Ok(self.real_exp(x))
    }

    pub(super) fn translate_expm1(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        Ok(self.real_exp(x) - one)
    }

    pub(super) fn translate_log1p(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        Ok((x + one).log())
    }

    pub(super) fn translate_log10(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let ln_ten = self.constant_like(x, std::f64::consts::LN_10);
        Ok(x.log() / ln_ten)
    }

    pub(super) fn translate_rsqrt(&mut self, node: &Node) -> Result<GraphTensor> {
        Ok(self.unary_input(node)?.sqrt().reciprocal())
    }

    pub(super) fn translate_sinh(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let half = self.constant_like(x, 0.5);
        Ok(half * (self.real_exp(x) - self.real_exp(x * -1.0)))
    }

    pub(super) fn translate_cosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let half = self.constant_like(x, 0.5);
        Ok(half * (self.real_exp(x) + self.real_exp(x * -1.0)))
    }

    pub(super) fn translate_tan(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let cosine = self.real_cos(x);
        Ok(x.sin() / cosine)
    }

    pub(super) fn translate_cos(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        Ok(self.real_cos(x))
    }

    /// `cos(x) = sin(pi/2 - x)`, keeping pi/2 in the tensor's dtype.
    fn real_cos(&mut self, input: GraphTensor) -> GraphTensor {
        let half_pi = self.constant_like(input, std::f64::consts::FRAC_PI_2);
        (half_pi - input).sin()
    }

    pub(super) fn translate_asin(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        let denominator = (one - x.square()).sqrt();
        Ok(self.real_atan(x / denominator))
    }

    pub(super) fn translate_acos(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        let denominator = (one - x.square()).sqrt();
        let asin = self.real_atan(x / denominator);
        let half_pi = self.constant_like(x, std::f64::consts::FRAC_PI_2);
        Ok(half_pi - asin)
    }

    pub(super) fn translate_atan(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        Ok(self.real_atan(x))
    }

    /// Range-reduced odd Taylor series for atan (parked port).
    fn real_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
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

    pub(super) fn translate_asinh(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        let magnitude = x.abs() + (x.square() + one).sqrt();
        let regular = magnitude.log();
        let log_two = self.constant_like(x, std::f64::consts::LN_2);
        let huge = x.abs().log() + log_two;
        let threshold = self.constant_like(x, (f32::MAX as f64) / 2.0);
        let result = self.select(x.abs().gt(threshold), huge, regular);
        Ok(self.copy_sign(result, x))
    }

    pub(super) fn translate_atanh(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        let half = self.constant_like(x, 0.5);
        let regular = half * ((one + x).log() - (one - x).log());
        Ok(regular)
    }

    pub(super) fn translate_acosh(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        let reciprocal_squared = x.reciprocal().square();
        Ok(x.log() + (one + (one - reciprocal_squared).sqrt()).log())
    }

    pub(super) fn translate_hardtanh(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.operand(&node.inputs[0])?;
        let minimum = self.get_float_arg(node, 1).unwrap_or(-1.0);
        let maximum = self.get_float_arg(node, 2).unwrap_or(1.0);
        let minimum = self.constant_like(value, minimum);
        let maximum = self.constant_like(value, maximum);
        let lower = self.select(value.lt(minimum), minimum, value);
        Ok(self.select(lower.gt(maximum), maximum, lower))
    }

    pub(super) fn translate_elu(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.unary_input(node)?;
        let alpha = self.get_float_arg(node, 1).unwrap_or(1.0);
        let scale = self.get_float_arg(node, 2).unwrap_or(1.0);
        let input_scale = self.get_float_arg(node, 3).unwrap_or(1.0);
        let zero = self.constant_like(value, 0.0);
        let one = self.constant_like(value, 1.0);
        let scaled = {
            let input_scaled = self.constant_like(value, input_scale);
            self.real_exp(value * input_scaled)
        };
        let negative = (scaled - one) * self.constant_like(value, alpha);
        let selected = self.select(value.gt(zero), value, negative);
        Ok(selected * self.constant_like(value, scale))
    }

    pub(super) fn translate_erf(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        Ok(self.real_erf(x))
    }

    pub(super) fn translate_erfc(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.unary_input(node)?;
        let one = self.constant_like(x, 1.0);
        Ok(one - self.real_erf(x))
    }

    /// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    fn real_erf(&mut self, u: GraphTensor) -> GraphTensor {
        const P: f64 = 0.3275911;
        const A1: f64 = 0.254829592;
        const A2: f64 = -0.284496736;
        const A3: f64 = 1.421413741;
        const A4: f64 = -1.453152027;
        const A5: f64 = 1.061405429;
        let t = (self.constant_like(u, 1.0) + self.constant_like(u, P) * u.abs()).reciprocal();
        let poly = ((((self.constant_like(t, A5) * t + self.constant_like(t, A4)) * t
            + self.constant_like(t, A3))
            * t
            + self.constant_like(t, A2))
            * t
            + self.constant_like(t, A1))
            * t;
        let one = self.constant_like(u, 1.0);
        u.sign() * (one - poly * (u.square() * -1.0).exp())
    }

    pub(super) fn translate_sign(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let zero = self.constant_like(a, 0.0);
        let pos = a.gt(zero).cast(DType::Int);
        let neg = a.lt(zero).cast(DType::Int);
        let signed = pos - neg;
        Ok(if a.dtype == DType::Int {
            signed
        } else {
            signed.cast(a.dtype)
        })
    }

    pub(super) fn translate_signbit(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        if is_float_dtype(a.dtype) {
            Ok(self.signbit(a))
        } else {
            let zero = self.constant_like(a, 0.0);
            Ok(a.lt(zero))
        }
    }

    pub(super) fn translate_isinf(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        if is_float_dtype(a.dtype) {
            Ok(self.is_inf(a))
        } else {
            let zero = self.constant_like(a, 0.0);
            let false_ = zero.gt(zero);
            Ok(false_.expand_rhs(a.dims()))
        }
    }

    pub(super) fn translate_bitwise_not(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        Ok(match a.dtype {
            DType::Bool => {
                let one = self.constant_like(a, 1.0).cast(DType::Int);
                (one - a.cast(DType::Int)).cast(DType::Bool)
            }
            DType::Int => (a + 1) * -1.0,
            other => anyhow::bail!("bitwise_not supports Bool/Int, got {other:?}"),
        })
    }

    pub(super) fn translate_ldexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.unary_input(node)?;
        let exponent = self.operand(&node.inputs[1])?.cast(input.dtype);
        let (input, exponent) = super::util::broadcast_binary(input, exponent);
        Ok(input * exponent.exp2())
    }

    pub(super) fn translate_floor_divide(&mut self, node: &Node) -> Result<GraphTensor> {
        self.div_with_mode(node, Some("floor"))
    }

    pub(super) fn translate_div_tensor_mode(&mut self, node: &Node) -> Result<GraphTensor> {
        let mode = named_string(node, "rounding_mode");
        self.div_with_mode(node, mode.as_deref())
    }

    fn div_with_mode(&mut self, node: &Node, mode: Option<&str>) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = if let Some(t) = self.optional_tensor_operand(&node.inputs[1])? {
            t
        } else {
            self.scalar(&node.inputs[1], a.dtype)?
        };
        let (a, b) = super::util::ensure_same_dtype(a, b);
        let (a, b) = super::util::broadcast_binary(a, b);
        let integer = a.dtype == DType::Int || a.dtype == DType::I64;
        let float_dtype = if a.dtype == DType::F64 {
            DType::F64
        } else {
            DType::F32
        };
        let quotient = a.cast(float_dtype) / b.cast(float_dtype);
        let out = match mode {
            Some("trunc") if integer => a.trunc_div(b),
            Some("floor") if integer => {
                let trunc = a.trunc_div(b);
                let rem = a.trunc_rem(b);
                let zero = self.constant_like(rem, 0.0);
                let signs_differ = a.lt(zero).ne(b.lt(zero));
                let adjust = rem.ne(zero).cast(a.dtype) * signs_differ.cast(a.dtype);
                trunc - adjust
            }
            Some("trunc") => quotient.trunc().cast(a.dtype),
            Some("floor") => quotient.floor().cast(a.dtype),
            _ => quotient.cast(a.dtype),
        };
        Ok(out)
    }
}

fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64)
}

fn named_string(node: &Node, name: &str) -> Option<String> {
    node.inputs.iter().find_map(|input| {
        if input.name != name {
            return None;
        }
        if let Argument::Other(value) = &input.arg {
            if let Some(s) = value.as_str() {
                return Some(s.to_string());
            }
            if let Some(s) = value.get("as_string").and_then(|v| v.as_str()) {
                return Some(s.to_string());
            }
        }
        None
    })
}
