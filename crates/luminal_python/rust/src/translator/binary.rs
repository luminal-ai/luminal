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
    sym_ranges: &FxHashMap<char, ExprBounds>,
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
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| same_expr_with_ranges(*lhs, *rhs, sym_ranges))
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_bitwise_and_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let b = self.get_input_tensor(node, 1)?;
        if a.dtype != DType::Bool || b.dtype != DType::Bool {
            anyhow::bail!(
                "aten.bitwise_and.Tensor currently only supports Bool tensors; got {:?} and {:?}",
                a.dtype,
                b.dtype
            );
        }
        let (a, b) = broadcast_binary(a, b);
        Ok((a.cast(DType::F32) * b.cast(DType::F32)).cast(DType::Bool))
    }

    pub(crate) fn translate_bitwise_and_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let mask = self.get_int_arg(node, 1)?;
        if mask != 15 {
            anyhow::bail!(
                "aten.bitwise_and.Scalar currently only supports the low-nibble mask 15; got {mask}"
            );
        }
        if !matches!(
            a.dtype,
            DType::Int
                | DType::I64
                | DType::I4
                | DType::U4
                | DType::I8
                | DType::U8
                | DType::I16
                | DType::U16
        ) {
            anyhow::bail!(
                "aten.bitwise_and.Scalar mask 15 requires an integer tensor; got {:?}",
                a.dtype
            );
        }

        // For two's-complement integers, `x & 15` is floor-modulo 16. HLIR's
        // Mod is signed remainder, so normalize it into [0, 15]. Widen first
        // so the intermediate `remainder + 16` cannot overflow a narrow dtype.
        let output_dtype = a.dtype;
        let a = a.cast(DType::Int);
        let modulus = self.graph.constant(16).cast(DType::Int).expand_rhs(a.shape);
        let remainder = a % modulus;
        Ok(((remainder + modulus) % modulus).cast(output_dtype))
    }

    pub(crate) fn translate_bitwise_right_shift(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let b = self.get_input_tensor(node, 1)?;
        let (a, b) = ensure_same_dtype(a, b);
        let (a, b) = broadcast_binary(a, b);
        Ok(a.bitwise_right_shift(b))
    }

    pub(crate) fn translate_binary_op(&mut self, node: &Node, op: BinaryOp) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(name) = arg1.as_tensor_name() {
            let b = self.get_tensor(name)?;
            let (a, b) = ensure_same_dtype(a, b);
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
            Ok(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::Sub => a - b,
                BinaryOp::Div => a / b,
            })
        } else {
            if let Some(f) = arg1.as_float() {
                return Ok(self.apply_scalar_op(a, f as f32, op));
            }
            if let Some(expr) = self.resolve_arg_as_expression(arg1) {
                return Ok(self.apply_symbolic_scalar_op(a, expr, op));
            }
            let val = self.get_float_arg(node, 1)? as f32;
            Ok(self.apply_scalar_op(a, val, op))
        }
    }

    pub(crate) fn translate_binary_scalar_op(
        &mut self,
        node: &Node,
        op: BinaryOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(f) = arg1.as_float() {
            return Ok(self.apply_scalar_op(a, f as f32, op));
        }
        if let Some(expr) = self.resolve_arg_as_expression(arg1) {
            return Ok(self.apply_symbolic_scalar_op(a, expr, op));
        }
        let val = self.get_float_arg(node, 1)? as f32;
        Ok(self.apply_scalar_op(a, val, op))
    }

    pub(crate) fn apply_scalar_op(
        &mut self,
        a: GraphTensor,
        val: f32,
        op: BinaryOp,
    ) -> GraphTensor {
        let scalar = self
            .graph
            .constant_float(val)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        match op {
            BinaryOp::Add => a + scalar,
            BinaryOp::Mul => a * scalar,
            BinaryOp::Sub => a - scalar,
            BinaryOp::Div => a / scalar,
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
            'a',
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
