use anyhow::Result;
use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_expr::{ExprBounds, canonical_equal_expr, same_expr_with_ranges, sym_char_ranges};
use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

fn normalize_equal_dims(
    a: &mut GraphTensor,
    b: &mut GraphTensor,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) {
    for i in 0..a.shape.len() {
        let lhs = a.shape.dims[i];
        let rhs = b.shape.dims[i];
        if let Some(canonical) = canonical_equal_expr(lhs, rhs, sym_ranges) {
            a.shape.dims[i] = canonical;
            b.shape.dims[i] = canonical;
        }
    }
}

fn same_dims(
    lhs: &[Expression],
    rhs: &[Expression],
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| same_expr_with_ranges(*lhs, *rhs, sym_ranges))
}

impl<'a> Translator<'a> {
    fn scalar_constant(&mut self, val: f64, dtype: DType) -> GraphTensor {
        if dtype == DType::F64 {
            self.graph.constant_float64(val)
        } else {
            self.graph.constant_float(val as f32).cast(dtype)
        }
    }

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

    fn promoted_binary_inputs(&mut self, node: &Node) -> Result<(GraphTensor, GraphTensor)> {
        let dtype = self.output_meta_dtype(node)?;
        let mut a = self.get_input_tensor(node, 0)?.cast(dtype);
        let mut b = if let Some(name) = node.inputs[1].arg.as_tensor_name() {
            self.get_tensor(name)?.cast(dtype)
        } else {
            let value = node.inputs[1]
                .arg
                .as_int()
                .map(|value| value as f64)
                .or_else(|| node.inputs[1].arg.as_float())
                .ok_or_else(|| anyhow::anyhow!("{} requires a numeric RHS", node.target))?;
            self.scalar_constant(value, dtype)
        };
        (a, b) = broadcast_binary(a, b);

        let sym_ranges = sym_char_ranges(&self.sym_map);
        normalize_equal_dims(&mut a, &mut b, &sym_ranges);
        let lhs_dims = a.dims();
        let rhs_dims = b.dims();
        anyhow::ensure!(
            same_dims(&lhs_dims, &rhs_dims, &sym_ranges),
            "binary op {} still has mismatched dims after broadcast: lhs={lhs_dims:?} rhs={rhs_dims:?} inputs={:?}",
            node.target,
            node.inputs
        );
        Ok((a, b))
    }

    pub(crate) fn translate_atan2(&mut self, node: &Node) -> Result<GraphTensor> {
        let (y, x) = self.promoted_binary_inputs(node)?;
        let output_dtype = y.dtype;
        let (y, x) = if matches!(output_dtype, DType::F16 | DType::Bf16) {
            (y.cast(DType::F32), x.cast(DType::F32))
        } else {
            (y, x)
        };
        Ok(self.real_atan2(y, x).cast(output_dtype))
    }

    pub(crate) fn translate_copysign(&mut self, node: &Node) -> Result<GraphTensor> {
        let (magnitude, sign) = self.promoted_binary_inputs(node)?;
        let magnitude = self.real_abs(magnitude);
        Ok(self.copy_sign(magnitude, sign))
    }

    pub(crate) fn translate_copysign_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        let magnitude = self.get_input_tensor(node, 0)?.cast(dtype);
        let magnitude = self.real_abs(magnitude);
        let sign = node.inputs[1]
            .arg
            .as_int()
            .map(|value| value as f64)
            .or_else(|| node.inputs[1].arg.as_float())
            .ok_or_else(|| anyhow::anyhow!("{} requires a numeric RHS", node.target))?;

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

    pub(crate) fn translate_fmax_fmin(
        &mut self,
        node: &Node,
        maximum: bool,
    ) -> Result<GraphTensor> {
        let (a, b) = self.promoted_binary_inputs(node)?;
        let comparison = if maximum { a.gt(b) } else { a.lt(b) };
        let mut result = self.select(comparison, a, b);

        if matches!(a.dtype, DType::F16 | DType::Bf16 | DType::F32 | DType::F64) {
            // Unlike maximum/minimum, fmax/fmin ignore a NaN when the other
            // operand is numeric. If both are NaN, selecting either preserves
            // the required NaN result.
            let a_nan = self.is_nan(a);
            let b_nan = self.is_nan(b);
            result = self.select(a_nan, b, result);
            result = self.select(b_nan, a, result);

            // C fmax/fmin semantics choose a deterministic zero sign rather
            // than whichever equal operand happened to win the comparison.
            let a_zero = self.is_zero(a);
            let b_zero = self.is_zero(b);
            let both_zero = self.bool_and(a_zero, b_zero);
            let zero = self.constant_like(a, 0.0);
            let signed_zero = if maximum { zero } else { zero * -1.0 };
            result = self.select(both_zero, signed_zero, result);
        }
        Ok(result)
    }

    /// Lower boolean OR through numeric HLIR ops until HLIR has native logical ops.
    pub(crate) fn apply_bool_or(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let a = a.cast(DType::F32);
        let b = b.cast(DType::F32);
        (a + b - a * b).cast(DType::Bool)
    }

    /// The dtype torch recorded for this node's output — division's result
    /// type depends on the operand types and the rounding mode, and export
    /// already ran that rule and wrote the answer down, so read it.
    pub(crate) fn recorded_output_dtype(&self, node: &Node) -> Option<DType> {
        let name = node
            .outputs
            .first()?
            .as_tensor
            .as_ref()
            .map(|t| t.name.clone())?;
        self.tensor_meta(&name)
            .map(|meta| torch_dtype_int_to_luminal(meta.dtype))
    }

    /// Promote both operands ahead of a true division.
    ///
    /// Must happen before the divide, not after: `a / b` lowers to
    /// `a * b.reciprocal()`, so an integral `b` emits `Recip` on an integer,
    /// which no backend region contract accepts.
    pub(crate) fn promote_for_true_division(
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

    pub(crate) fn translate_binary_op(&mut self, node: &Node, op: BinaryOp) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let alpha = self.get_explicit_alpha(node, op)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(name) = arg1.as_tensor_name() {
            let b = self.get_tensor(name)?;
            let (a, mut b) = ensure_same_dtype(a, b);
            let is_bool_add = matches!(op, BinaryOp::Add) && a.dtype == DType::Bool;
            if !is_bool_add && let Some(alpha) = alpha {
                b = self.apply_scalar_op(b, alpha, BinaryOp::Mul);
            }
            let (mut a, mut b) = broadcast_binary(a, b);
            let sym_ranges = sym_char_ranges(&self.sym_map);
            normalize_equal_dims(&mut a, &mut b, &sym_ranges);
            let lhs_dims = a.dims();
            let rhs_dims = b.dims();
            if !same_dims(&lhs_dims, &rhs_dims, &sym_ranges) {
                anyhow::bail!(
                    "binary op {} still has mismatched dims after broadcast: lhs={lhs_dims:?} rhs={rhs_dims:?} inputs={:?}",
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
                    self.apply_bool_or(a, b)
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
                anyhow::ensure!(
                    alpha.is_none(),
                    "{} with an explicit alpha and symbolic scalar operand is not supported",
                    node.target
                );
                return Ok(self.apply_symbolic_scalar_op(a, expr, op));
            }
            let val = self.get_float_arg(node, 1)?;
            Ok(self.apply_scalar_op_with_alpha(a, val, alpha, op))
        }
    }

    pub(crate) fn translate_binary_scalar_op(
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
            anyhow::ensure!(
                alpha.is_none(),
                "{} with an explicit alpha and symbolic scalar operand is not supported",
                node.target
            );
            return Ok(self.apply_symbolic_scalar_op(a, expr, op));
        }
        let val = self.get_float_arg(node, 1)?;
        Ok(self.apply_scalar_op_with_alpha(a, val, alpha, op))
    }

    pub(crate) fn apply_scalar_op(
        &mut self,
        a: GraphTensor,
        val: f64,
        op: BinaryOp,
    ) -> GraphTensor {
        let scalar = self.scalar_constant(val, a.dtype).expand_rhs(a.shape);
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
            let scalar = self.scalar_constant(val, a.dtype).expand_rhs(a.shape);
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

    pub(crate) fn apply_symbolic_scalar_op(
        &mut self,
        a: GraphTensor,
        val: Expression,
        op: BinaryOp,
    ) -> GraphTensor {
        match op {
            BinaryOp::Add => a + val,
            BinaryOp::Mul => a * val,
            BinaryOp::Sub => a - val,
            BinaryOp::Div => a / val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pt2_expr::simplify_expr_with_ranges;

    #[test]
    fn simplifies_mark_dynamic_slice_shapes_using_lower_bound() {
        let a = Expression::from('a');
        let lhs = (a.min(1) + a).min(a + 1) - 1;
        let rhs = (a.min(1) + a).min(a);
        let sym_ranges = [(
            Symbol::from('a'),
            ExprBounds {
                min: Some(2),
                max: None,
            },
        )]
        .into_iter()
        .collect::<FxHashMap<_, _>>();

        let lhs_simplified = simplify_expr_with_ranges(lhs, &sym_ranges);
        let rhs_simplified = simplify_expr_with_ranges(rhs, &sym_ranges);

        assert_eq!(lhs_simplified, Expression::from('a'));
        assert_eq!(rhs_simplified, Expression::from('a'));
        assert!(same_expr_with_ranges(lhs, rhs, &sym_ranges));
    }
}
