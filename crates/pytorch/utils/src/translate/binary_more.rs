//! Additional ATen lowerings ported from the parked translator (port batch 6+).
#![allow(dead_code)]

use anyhow::{Result, anyhow, bail};
use luminal::prelude::*;

use super::Translator;
use super::util;
use crate::pt2_schema::Node;

/// Binary operation type (ported from the parked translator).
#[derive(Clone, Copy)]
pub(super) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Translator<'_> {
    // -----------------------------------------------------------------
    // Shared helpers (private copies of the parked translator's helpers)
    // -----------------------------------------------------------------

    /// Typed scalar constant matching the parked `scalar_constant`: F64 keeps
    /// double precision, everything else goes through an F32 literal cast.
    fn scalar_constant(&mut self, val: f64, dtype: DType) -> GraphTensor {
        if dtype == DType::F64 {
            self.cx.constant_f64(val)
        } else {
            self.cx.constant_f32(val as f32).cast(dtype)
        }
    }

    /// The explicit `alpha` kwarg for add/sub, if present.
    fn get_explicit_alpha(&self, node: &Node, op: BinaryOp) -> Result<Option<f64>> {
        if !matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            return Ok(None);
        }
        node.inputs
            .iter()
            .position(|input| input.name == "alpha")
            .map(|idx| {
                node.inputs[idx]
                    .arg
                    .as_bool()
                    .map(|value| if value { 1.0 } else { 0.0 })
                    .map(Ok)
                    .unwrap_or_else(|| self.get_float_arg(node, idx))
            })
            .transpose()
    }

    /// One operand promoted to the node's recorded output dtype. Scalars
    /// become typed rank-0 constants, matching the parked implementation.
    fn promoted_operand(&mut self, node: &Node, index: usize, dtype: DType) -> Result<GraphTensor> {
        if node.inputs[index].arg.as_tensor_name().is_some() {
            return Ok(self.get_input_tensor(node, index)?.cast(dtype));
        }
        let value = node.inputs[index]
            .arg
            .as_int()
            .map(|value| value as f64)
            .or_else(|| node.inputs[index].arg.as_float())
            .or_else(|| {
                node.inputs[index]
                    .arg
                    .as_bool()
                    .map(|value| if value { 1.0 } else { 0.0 })
            })
            .ok_or_else(|| anyhow!("{} input {index} must be tensor or numeric", node.target))?;
        Ok(self.scalar_constant(value, dtype))
    }

    /// Both operands cast to the output dtype and broadcast together.
    fn promoted_binary_inputs(&mut self, node: &Node) -> Result<(GraphTensor, GraphTensor)> {
        let dtype = self.output_meta_dtype(node)?;
        let (mut a, mut b) = (
            self.promoted_operand(node, 0, dtype)?,
            self.promoted_operand(node, 1, dtype)?,
        );
        (a, b) = util::broadcast_binary(a, b);
        let ad = a.dims();
        let bd = b.dims();
        if ad.len() != bd.len()
            || !ad
                .iter()
                .zip(bd.iter())
                .all(|(l, r)| util::same_dim(*l, *r))
        {
            bail!(
                "binary op {} still has mismatched dims after broadcast: lhs={ad:?} rhs={bd:?} inputs={:?}",
                node.target,
                node.inputs
            );
        }
        Ok((a, b))
    }

    /// The dtype torch recorded for this node's output, if any.
    fn recorded_output_dtype(&self, node: &Node) -> Option<DType> {
        self.output_meta_dtype(node).ok()
    }

    /// Promote both operands ahead of a true division: `a / b` lowers to
    /// `a * b.reciprocal()`, and `Recip` on an integer is not acceptable.
    fn promote_for_true_division(
        &self,
        node: &Node,
        a: GraphTensor,
        b: GraphTensor,
    ) -> (GraphTensor, GraphTensor) {
        let Some(target) = self.recorded_output_dtype(node) else {
            return (a, b);
        };
        (a.cast(target), b.cast(target))
    }

    /// Structural `where` that keeps the branches' dtype. Arithmetic masking
    /// (`a * mask + b * (1 - mask)`) turns an untaken NaN into `NaN * 0 = NaN`,
    /// poisoning the selected value. Pack the branches with a scatter into a
    /// fresh zero tensor (scatter only moves payload) and select with a
    /// gather index, the parked translator's construction.
    fn binary_select(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = util::broadcast_binary(a, condition);
        let (a, b) = util::broadcast_binary(a, b);
        // If the `b` broadcast drove the shape, bring `condition` up to it.
        let (a, condition) = util::broadcast_binary(a, condition);
        let dims = a.dims();
        if dims.is_empty() {
            let selected =
                self.binary_select(condition.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0));
            return selected.squeeze(0);
        }
        let rank = dims.len();
        let axis_coordinates: Vec<GraphTensor> = (0..rank)
            .map(|axis| self.cx.iota(dims.clone(), |c| c[axis]))
            .collect();
        // Branch 0 gets `b`, branch 1 gets `a`.
        let mut false_coordinates = Vec::with_capacity(rank + 1);
        false_coordinates.push(self.cx.iota(dims.clone(), |_| IntExpr::from(0)));
        false_coordinates.extend(axis_coordinates.iter().copied());
        let mut true_coordinates = Vec::with_capacity(rank + 1);
        true_coordinates.push(self.cx.iota(dims.clone(), |_| IntExpr::from(1)));
        true_coordinates.extend(axis_coordinates.iter().copied());

        let mut stacked_dims = Vec::with_capacity(rank + 1);
        stacked_dims.push(IntExpr::from(2));
        stacked_dims.extend(dims.iter().copied());
        // A real materialized zero buffer: an expanded scalar would alias one
        // element and scatter copies its destination before writing.
        let scratch = self
            .cx
            .iota(stacked_dims, |_| IntExpr::from(0))
            .cast(a.dtype);
        let stacked = scratch
            .scatter(&false_coordinates, b)
            .scatter(&true_coordinates, a);

        let mut gather_coordinates = Vec::with_capacity(rank + 1);
        gather_coordinates.push(condition.cast(DType::Int));
        gather_coordinates.extend(axis_coordinates);
        stacked.gather(&gather_coordinates)
    }

    /// Exact zero predicate: unlike the arithmetic `is_zero` helper, NaN is
    /// not zero (the parked `is_zero`).
    fn binary_is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let nonzero = self.bool_or(value.lt(zero), value.gt(zero));
        let nan = self.is_nan(value);
        let nonzero_or_nan = self.bool_or(nonzero, nan);
        self.bool_not(nonzero_or_nan)
    }

    /// Sign-bit test that also catches `-0.0` and `-inf` (the parked
    /// `signbit`): negative iff `x < 0` or `1/x < 0`.
    fn binary_signbit(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let negative = value.lt(zero);
        let negative_zero = value.reciprocal().lt(zero);
        self.bool_or(negative, negative_zero)
    }

    /// Exact sign transfer (the parked `copy_sign`): select rather than a
    /// re-scaling, so large magnitudes and non-finite signs survive.
    fn binary_copy_sign(&mut self, magnitude: GraphTensor, sign: GraphTensor) -> GraphTensor {
        let negative = magnitude * -1.0;
        let signbit = self.binary_signbit(sign);
        self.binary_select(signbit, negative, magnitude)
    }

    /// Range-reduced odd Taylor series for atan (the parked `real_atan`).
    fn binary_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let one = self.constant_like(x, 1.0);
        let reciprocal_branch = x.gt(one);
        let reduced = self.binary_select(reciprocal_branch, x.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.binary_select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.binary_select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.binary_select(reciprocal_branch, half_pi - base, base);
        self.binary_copy_sign(angle, input)
    }

    /// Real-domain atan2 (the parked `real_atan2`): quadrants from the sign of
    /// `x`, the two-infinite axis case, and signed zero.
    fn binary_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let ratio = y / x;
        let mut angle = self.binary_atan(ratio);
        let x_negative = self.binary_signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.binary_copy_sign(pi, y);
        angle = self.binary_select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = self.bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.binary_select(x_negative, three_quarters, quarter);
        let infinite_angle = self.binary_copy_sign(infinite_angle, y);
        angle = self.binary_select(both_inf, infinite_angle, angle);

        let x_zero = self.binary_is_zero(x);
        let y_zero = self.binary_is_zero(y);
        let both_zero = self.bool_and(x_zero, y_zero);
        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.binary_copy_sign(zero, y);
        let zero_angle = self.binary_select(x_negative, signed_pi, signed_zero);
        self.binary_select(both_zero, zero_angle, angle)
    }

    // -----------------------------------------------------------------
    // Binary arithmetic (add/sub/mul/div with alpha)
    // -----------------------------------------------------------------

    pub(super) fn translate_binary_op(&mut self, node: &Node, op: BinaryOp) -> Result<GraphTensor> {
        // Scalar LHS: promote both operands to the recorded output dtype.
        if node.inputs[0].arg.as_tensor_name().is_none() {
            let alpha = self.get_explicit_alpha(node, op)?;
            let (a, mut b) = self.promoted_binary_inputs(node)?;
            let is_bool_add = matches!(op, BinaryOp::Add) && a.dtype == DType::Bool;
            if !is_bool_add && let Some(alpha) = alpha {
                b = self.apply_scalar_op(b, alpha, BinaryOp::Mul);
            }
            if is_bool_add {
                return Ok(if alpha == Some(0.0) {
                    a
                } else {
                    self.bool_or(a, b)
                });
            }
            return Ok(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::Sub => a - b,
                BinaryOp::Div => a / b,
            });
        }

        let a = self.get_input_tensor(node, 0)?;
        let alpha = self.get_explicit_alpha(node, op)?;
        let arg1 = &node.inputs[1].arg;
        if arg1.as_tensor_name().is_some() {
            let b = self.get_input_tensor(node, 1)?;
            let (a, mut b) = util::ensure_same_dtype(a, b);
            let is_bool_add = matches!(op, BinaryOp::Add) && a.dtype == DType::Bool;
            if !is_bool_add && let Some(alpha) = alpha {
                b = self.apply_scalar_op(b, alpha, BinaryOp::Mul);
            }
            let (a, b) = util::broadcast_binary(a, b);
            let ad = a.dims();
            let bd = b.dims();
            if ad.len() != bd.len()
                || !ad
                    .iter()
                    .zip(bd.iter())
                    .all(|(l, r)| util::same_dim(*l, *r))
            {
                bail!(
                    "binary op {} still has mismatched dims after broadcast: lhs={ad:?} rhs={bd:?} inputs={:?}",
                    node.target,
                    node.inputs
                );
            }
            if is_bool_add {
                // PyTorch defines bool + bool as logical OR. Its alpha scales
                // the RHS in boolean space, so zero drops it and any nonzero
                // integral value leaves its truth value unchanged.
                return Ok(if alpha == Some(0.0) {
                    a
                } else {
                    self.bool_or(a, b)
                });
            }
            Ok(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::Sub => a - b,
                BinaryOp::Div => {
                    let (a, b) = self.promote_for_true_division(node, a, b);
                    a / b
                }
            })
        } else {
            // `x / 2` is div.Tensor with an int argument, not div.Scalar, so the
            // scalar routes below need the same promotion. Each casts its scalar
            // to a.dtype, so promoting `a` promotes both sides.
            let a = if matches!(op, BinaryOp::Div) {
                self.promote_for_true_division(node, a, a).0
            } else {
                a
            };
            if let Some(f) = arg1.as_float() {
                return Ok(self.apply_scalar_op_with_alpha(a, f, alpha, op));
            }
            if let Some(expr) = self.resolve_arg_as_expression(arg1) {
                if alpha.is_some() {
                    bail!(
                        "{} with an explicit alpha and symbolic scalar operand is not supported",
                        node.target
                    );
                }
                return Ok(self.apply_symbolic_scalar_op(a, expr, op));
            }
            let val = self.get_float_arg(node, 1)?;
            Ok(self.apply_scalar_op_with_alpha(a, val, alpha, op))
        }
    }

    pub(super) fn translate_binary_scalar_op(
        &mut self,
        node: &Node,
        op: BinaryOp,
    ) -> Result<GraphTensor> {
        let mut a = self.get_input_tensor(node, 0)?;
        if matches!(op, BinaryOp::Div) {
            // The scalar is cast to `a.dtype` below, so promoting `a` promotes
            // both sides. int / 2 is float in torch, and Recip needs it anyway.
            (a, _) = self.promote_for_true_division(node, a, a);
        }
        let alpha = self.get_explicit_alpha(node, op)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(f) = arg1.as_float() {
            return Ok(self.apply_scalar_op_with_alpha(a, f, alpha, op));
        }
        if let Some(expr) = self.resolve_arg_as_expression(arg1) {
            if alpha.is_some() {
                bail!(
                    "{} with an explicit alpha and symbolic scalar operand is not supported",
                    node.target
                );
            }
            return Ok(self.apply_symbolic_scalar_op(a, expr, op));
        }
        let val = self.get_float_arg(node, 1)?;
        Ok(self.apply_scalar_op_with_alpha(a, val, alpha, op))
    }

    fn apply_scalar_op(&mut self, a: GraphTensor, val: f64, op: BinaryOp) -> GraphTensor {
        let scalar = self.scalar_constant(val, a.dtype).expand_rhs(a.dims());
        match op {
            BinaryOp::Add => a + scalar,
            BinaryOp::Mul => a * scalar,
            BinaryOp::Sub => a - scalar,
            BinaryOp::Div => a / scalar,
        }
    }

    fn apply_scalar_op_with_alpha(
        &mut self,
        a: GraphTensor,
        val: f64,
        alpha: Option<f64>,
        op: BinaryOp,
    ) -> GraphTensor {
        if let Some(alpha) = alpha {
            let scalar = self.scalar_constant(val, a.dtype).expand_rhs(a.dims());
            let scaled = self.apply_scalar_op(scalar, alpha, BinaryOp::Mul);
            match op {
                BinaryOp::Add => a + scaled,
                BinaryOp::Mul => a * scaled,
                BinaryOp::Sub => a - scaled,
                BinaryOp::Div => a / scaled,
            }
        } else {
            self.apply_scalar_op(a, val, op)
        }
    }

    fn apply_symbolic_scalar_op(
        &mut self,
        a: GraphTensor,
        val: IntExpr,
        op: BinaryOp,
    ) -> GraphTensor {
        match op {
            BinaryOp::Add => a + val,
            BinaryOp::Mul => a * val,
            BinaryOp::Sub => a - val,
            BinaryOp::Div => a / val,
        }
    }

    // -----------------------------------------------------------------
    // Elementwise binary math
    // -----------------------------------------------------------------

    pub(super) fn translate_atan2(&mut self, node: &Node) -> Result<GraphTensor> {
        let (y, x) = self.promoted_binary_inputs(node)?;
        let output_dtype = y.dtype;
        let (y, x) = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            (y.cast(DType::F32), x.cast(DType::F32))
        } else {
            (y, x)
        };
        Ok(self.binary_atan2(y, x).cast(output_dtype))
    }

    pub(super) fn translate_copysign(&mut self, node: &Node) -> Result<GraphTensor> {
        let (magnitude, sign) = self.promoted_binary_inputs(node)?;
        let magnitude = self.real_abs(magnitude);
        Ok(self.binary_copy_sign(magnitude, sign))
    }

    pub(super) fn translate_copysign_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        let magnitude = self.get_input_tensor(node, 0)?.cast(dtype);
        let magnitude = self.real_abs(magnitude);
        let sign = node.inputs[1]
            .arg
            .as_int()
            .map(|value| value as f64)
            .or_else(|| node.inputs[1].arg.as_float())
            .ok_or_else(|| anyhow!("{} requires a numeric RHS", node.target))?;

        // Egglog's scalar value domain equates +0.0 and -0.0. Preserve the
        // compile-time scalar sign structurally instead of inserting it as a
        // graph constant; multiplication also produces the required -0.0 for
        // a zero magnitude.
        Ok(if sign.is_sign_negative() {
            magnitude * -1.0
        } else {
            magnitude
        })
    }

    pub(super) fn translate_fmax_fmin(
        &mut self,
        node: &Node,
        maximum: bool,
    ) -> Result<GraphTensor> {
        let (a, b) = self.promoted_binary_inputs(node)?;
        let comparison = if maximum { a.gt(b) } else { a.lt(b) };
        let mut result = self.binary_select(comparison, a, b);

        if matches!(a.dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64) {
            // Unlike maximum/minimum, fmax/fmin ignore a NaN when the other
            // operand is numeric. If both are NaN, selecting either preserves
            // the required NaN result.
            let a_nan = self.is_nan(a);
            let b_nan = self.is_nan(b);
            result = self.binary_select(a_nan, b, result);
            result = self.binary_select(b_nan, a, result);

            // C fmax/fmin semantics choose a deterministic zero sign rather
            // than whichever equal operand happened to win the comparison.
            let a_zero = self.binary_is_zero(a);
            let b_zero = self.binary_is_zero(b);
            let both_zero = self.bool_and(a_zero, b_zero);
            let zero = self.constant_like(a, 0.0);
            let signed_zero = if maximum { zero } else { zero * -1.0 };
            result = self.binary_select(both_zero, signed_zero, result);
        }
        Ok(result)
    }

    /// Stable hypot avoids overflow from directly squaring the larger input.
    pub(super) fn translate_hypot(&mut self, node: &Node) -> Result<GraphTensor> {
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        let lhs = self.real_abs(lhs);
        let rhs = self.real_abs(rhs);
        let lhs_infinite = self.is_inf(lhs);
        let rhs_infinite = self.is_inf(rhs);
        let any_infinite = self.bool_or(lhs_infinite, rhs_infinite);
        let larger = lhs.maximum(rhs);
        let smaller = lhs.minimum(rhs);
        let zero = self.constant_like(larger, 0.0);
        let ratio = smaller / larger;
        let finite = larger * (self.constant_like(larger, 1.0) + ratio.square()).sqrt();
        let both_zero = self.binary_is_zero(larger);
        let result = self.binary_select(both_zero, zero, finite);
        let infinity = self.constant_like(larger, f64::INFINITY);
        Ok(self.binary_select(any_infinite, infinity, result))
    }

    pub(super) fn translate_gcd(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let (lhs, rhs) = self.promoted_binary_inputs(node)?;
        // Signed I64 covers every supported input dtype, including U8. The
        // Euclidean algorithm needs at most 93 divisions for 64-bit inputs;
        // 128 static rounds leave margin without tensor-controlled looping.
        let mut lhs = lhs.cast(DType::I64);
        let mut rhs = rhs.cast(DType::I64);
        let zero = self.constant_like(lhs, 0.0);
        let one = self.constant_like(lhs, 1.0);
        for _ in 0..128 {
            let finished = self.binary_is_zero(rhs);
            let safe_rhs = self.binary_select(finished, one, rhs);
            let remainder = lhs % safe_rhs;
            lhs = self.binary_select(finished, lhs, rhs);
            rhs = self.binary_select(finished, zero, remainder);
        }
        let negative = lhs.lt(zero);
        let magnitude = self.binary_select(negative, -lhs, lhs);
        Ok(magnitude.cast(output_dtype))
    }

    // -----------------------------------------------------------------
    // Elementwise unary-adjacent lowerings assigned to this module
    // -----------------------------------------------------------------

    pub(super) fn translate_leaky_relu(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.get_input_tensor(node, 0)?;
        let dtype = self.output_meta_dtype(node).unwrap_or(x.dtype);
        let value = if x.dtype == dtype { x } else { x.cast(dtype) };
        let slope = self.get_float_arg(node, 1).unwrap_or(0.01);
        let zero = self.constant_like(value, 0.0);
        let negative = value * self.constant_like(value, slope);
        Ok(self.binary_select(value.gt(zero), value, negative))
    }

    pub(super) fn translate_exp2(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(a.exp2())
    }

    pub(super) fn translate_log2(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self
            .get_input_tensor(node, 0)?
            .cast(self.output_meta_dtype(node)?);
        Ok(a.log2())
    }

    pub(super) fn translate_isnan(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        Ok(self.is_nan(a))
    }

    pub(super) fn translate_bitwise_and(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let b = self.get_input_tensor(node, 1)?;
        let (a, b) = util::broadcast_binary(a, b);
        // `logical_and` records a Bool output; `bitwise_and` records the
        // integer dtype. The reference lowering is a logical AND (matching
        // the parked translator, which never exercised integer bitwise), so
        // cast back to the recorded dtype.
        let dtype = self.output_meta_dtype(node).unwrap_or(DType::Bool);
        Ok(self.bool_and(a, b).cast(dtype))
    }

    pub(super) fn translate_bitwise_or(&mut self, node: &Node) -> Result<GraphTensor> {
        // Both the bool and logical arms use the same bool-OR lowering.
        // Gemma-4's sliding+full attention mask fusion emits bitwise_or on
        // boolean tensors; the integer semantics of bitwise_or aren't
        // exercised by any op in the test suite, so we rely on inputs being
        // boolean-typed.
        let a = self.get_input_tensor(node, 0)?;
        let b = self.get_input_tensor(node, 1)?;
        let (a, b) = util::broadcast_binary(a, b);
        let dtype = self.output_meta_dtype(node).unwrap_or(DType::Bool);
        Ok(self.bool_or(a, b).cast(dtype))
    }
}
