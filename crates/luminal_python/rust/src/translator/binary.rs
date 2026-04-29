use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

impl<'a> Translator<'a> {
    pub(crate) fn translate_binary_op(&mut self, node: &Node, op: BinaryOp) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(name) = arg1.as_tensor_name() {
            let b = self.get_tensor(name)?;
            let (a, b) = ensure_same_dtype(a, b);
            let (a, b) = broadcast_binary(a, b);
            Ok(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::Sub => a - b,
                BinaryOp::Div => a / b,
            })
        } else {
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
}
