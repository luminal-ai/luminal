use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::{broadcast_binary, torch_dtype_int_to_luminal};

use super::Translator;

impl<'a> Translator<'a> {
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
            if input.name == "dtype"
                && let Some(dtype_int) = input.arg.as_int()
            {
                let dtype = torch_dtype_int_to_luminal(dtype_int as u32);
                return Ok(a.cast(dtype));
            }
        }
        Ok(a)
    }

    pub(crate) fn translate_to_dtype(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        if let Some(dtype_int) = node.inputs.get(1).and_then(|i| i.arg.as_scalar_type()) {
            let dtype = torch_dtype_int_to_luminal(dtype_int);
            Ok(a.cast(dtype))
        } else if let Some(dtype_int) = node.inputs.get(1).and_then(|i| i.arg.as_int()) {
            let dtype = torch_dtype_int_to_luminal(dtype_int as u32);
            Ok(a.cast(dtype))
        } else {
            Ok(a)
        }
    }

    pub(crate) fn translate_to_dtype_layout(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        for input in &node.inputs {
            if input.name == "dtype" {
                if let Some(dtype_int) = input.arg.as_scalar_type() {
                    let dtype = torch_dtype_int_to_luminal(dtype_int);
                    return Ok(a.cast(dtype));
                }
                if let Some(dtype_int) = input.arg.as_int() {
                    let dtype = torch_dtype_int_to_luminal(dtype_int as u32);
                    return Ok(a.cast(dtype));
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
}
