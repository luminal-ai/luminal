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
        Ok(a.stable_argsort(dim, descending))
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
                if let Some(dtype_int) = input.arg.as_int() {
                    let dtype = torch_dtype_int_to_luminal(dtype_int as u32);
                    return Ok(a.cast(dtype));
                }
                if let Some(dtype_int) = input.arg.as_scalar_type() {
                    let dtype = torch_dtype_int_to_luminal(dtype_int);
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
        let input = self.get_input_tensor(node, MASKED_FILL_INPUT_ARG)?;
        let mask = self.get_input_tensor(node, MASKED_FILL_MASK_ARG)?;
        let fill = self.get_float_arg(node, MASKED_FILL_VALUE_ARG)? as f32;
        let (input, mask) = broadcast_binary(input, mask);
        let work_dtype = if input.dtype == DType::Bool {
            DType::Int
        } else {
            input.dtype
        };
        let input_work = if input.dtype == DType::Bool {
            input.cast(DType::Int)
        } else {
            input
        };
        let mask_work = mask.cast(work_dtype);
        let fill_work = self
            .graph
            .constant_float(fill)
            .cast(work_dtype)
            .expand_rhs(input_work.shape);
        let one = self
            .graph
            .constant_float(1.0)
            .cast(work_dtype)
            .expand_rhs(input_work.shape);
        let result = mask_work * fill_work + (one - mask_work) * input_work;
        Ok(if input.dtype == DType::Bool {
            result.cast(DType::Bool)
        } else {
            result
        })
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

        // Check rounding_mode kwarg
        let rounding_mode = node.inputs.iter().find_map(|input| {
            if input.name == "rounding_mode"
                && let Argument::Other(val) = &input.arg
            {
                return val.as_str().map(|s| s.to_string());
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
}
