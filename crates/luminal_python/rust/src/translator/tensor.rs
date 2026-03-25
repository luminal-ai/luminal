use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

impl<'a> Translator<'a> {
    pub(crate) fn translate_arange(&mut self, node: &Node) -> Result<GraphTensor> {
        let positional_args: Vec<Expression> = node
            .inputs
            .iter()
            .filter(|i| i.kind <= 1)
            .filter_map(|i| self.resolve_arg_as_expression(&i.arg))
            .collect();

        match positional_args.len() {
            0 => anyhow::bail!("arange: no positional args found"),
            1 => Ok(self.graph.arange(positional_args[0])),
            _ => Ok(self
                .graph
                .arange_options(positional_args[0], positional_args[1], 1)),
        }
    }

    pub(crate) fn translate_full(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.get_exprs_arg(node, 0)?;
        let val = self.get_float_arg(node, 1)? as f32;
        Ok(self.graph.constant_float(val).expand_rhs(shape))
    }

    pub(crate) fn translate_zeros(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_constant_fill(node, 0.0)
    }

    pub(crate) fn translate_ones(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_constant_fill(node, 1.0)
    }

    pub(crate) fn translate_new_ones(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_constant_fill(node, 1.0)
    }

    fn translate_constant_fill(&mut self, node: &Node, val: f32) -> Result<GraphTensor> {
        let output_name = node
            .outputs
            .first()
            .and_then(|o| o.as_tensor.as_ref())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&output_name)
            .context("Missing tensor meta for constant fill output")?;
        let shape = self.tensor_meta_to_shape(meta)?;
        if shape.is_empty() {
            Ok(self.graph.constant_float(val))
        } else {
            Ok(self.graph.constant_float(val).expand_rhs(shape))
        }
    }

    pub(crate) fn translate_where(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let y = self.get_input_tensor(node, 2)?;
        // Broadcast all three tensors to a common shape first
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let (cond_bc, y_b) = broadcast_binary(cond_b, y);
        let (x_bc, y_bc) = broadcast_binary(x_b, y_b);
        let c = cond_bc.cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(c.shape);
        Ok(c * x_bc + (one - c) * y_bc)
    }

    pub(crate) fn translate_where_scalar_other(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let other_val = self.get_float_arg(node, 2)? as f32;
        // Broadcast cond and x to a common shape
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let c = cond_b.cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(c.shape);
        let other = self.graph.constant_float(other_val).expand_rhs(c.shape);
        Ok(c * x_b + (one - c) * other)
    }

    pub(crate) fn translate_diff(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(-1)
        } else {
            -1
        };
        let dim = normalize_dim(dim, input.shape.len());

        let prepend = if node.inputs.len() > 3 {
            self.get_input_tensor(node, 3).ok()
        } else {
            None
        };

        let x = if let Some(prep) = prepend {
            prep.concat_along(input, dim)
        } else {
            input
        };

        let dim_size = x.shape.dims[dim];
        let front = x.slice_along(Expression::from(1)..dim_size, dim);
        let back = x.slice_along(Expression::from(0)..dim_size - 1, dim);
        Ok(front - back)
    }

    pub(crate) fn translate_tril(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, false)
    }

    pub(crate) fn translate_triu(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, true)
    }

    fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let diagonal = if node.inputs.len() > 1 {
            self.get_int_arg(node, 1).unwrap_or(0) as i32
        } else {
            0
        };
        let dims = a.shape.dims;
        let rows = dims[dims.len() - 2];
        let cols = dims[dims.len() - 1];
        let (r_val, c_val) = match (rows.to_usize(), cols.to_usize()) {
            (Some(r), Some(c)) => (r, c),
            _ => anyhow::bail!("tril/triu requires concrete matrix dimensions"),
        };
        let size = r_val.max(c_val);
        let mask = if upper {
            self.graph.triu(size, diagonal)
        } else {
            self.graph.tril(size, diagonal)
        }
        .cast(DType::F32);
        let mask = if rows != cols {
            mask.slice_along(0..r_val, 0).slice_along(0..c_val, 1)
        } else {
            mask
        };
        let mut mask_expanded = mask;
        for i in (0..dims.len() - 2).rev() {
            mask_expanded = mask_expanded.expand_dim(0, dims[i]);
        }
        Ok(a * mask_expanded)
    }

    pub(crate) fn translate_topk(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        let k = self.get_int_arg(node, 1)? as usize;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(-1)
        } else {
            -1
        };
        let dim = normalize_dim(dim, a.shape.len());

        // Determine output names
        let values_name = node
            .outputs
            .first()
            .and_then(|o| o.as_tensor.as_ref().map(|t| t.name.clone()));
        let indices_name =
            if let Some(ts) = node.outputs.first().and_then(|o| o.as_tensors.as_ref()) {
                ts.get(1).map(|t| t.name.clone())
            } else if node.outputs.len() > 1 {
                node.outputs[1].as_tensor.as_ref().map(|t| t.name.clone())
            } else {
                None
            };

        // Use full argsort then slice, rather than topk_indexes/topk_values directly.
        // This avoids a CUDA gather kernel bug when data and index shapes differ
        // along the gather axis (topk_indexes returns a sliced tensor).
        let full_argsort = a.argsort(dim, true);

        // Only build each branch when its output is consumed.
        // Dead nodes in the graph can confuse the CUDA optimizer.
        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = a.gather_elements(full_argsort, dim).slice_along(..k, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            // Materialize Int indices as F32 with `* 1.0` to force a contiguous copy.
            // Without this, CUDA can't correctly read the sliced Int view.
            let indices = full_argsort.slice_along(..k, dim) * 1.0;
            self.tensors.insert(idx_name, indices);
        }

        Ok(())
    }

    pub(crate) fn translate_one_hot(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let num_classes = self.get_int_arg(node, 1)? as usize;
        // one_hot: output[..., i] = 1 if input[...] == i else 0
        let a_int = a.cast(DType::Int);
        let classes = self.graph.arange(num_classes);
        // Expand a to [..., 1] and classes to [..., num_classes]
        let a_expanded = a_int.expand_dim(a.shape.len(), num_classes);
        let mut classes_expanded = classes;
        for d in a.shape.dims.iter().rev() {
            classes_expanded = classes_expanded.expand_dim(0, *d);
        }
        Ok(a_expanded.eq(classes_expanded).cast(DType::Int))
    }

    pub(crate) fn translate_wrap_set_grad(&mut self, node: &Node) -> Result<()> {
        let subgraph = node.inputs[1]
            .arg
            .as_graph()
            .context("wrap_with_set_grad: missing subgraph")?
            .clone();

        let sg_inputs = &subgraph.graph.inputs;
        let forwarded_args = &node.inputs[2..];
        for (sg_input, fwd_arg) in sg_inputs.iter().zip(forwarded_args) {
            if let Some(sg_name) = sg_input.as_tensor.as_ref()
                && let Some(main_name) = fwd_arg.arg.as_tensor_name()
            {
                let tensor = self.get_tensor(main_name)?;
                self.tensors.insert(sg_name.name.clone(), tensor);
            }
        }

        for (k, v) in &subgraph.graph.tensor_values {
            self.extra_tensor_values.insert(k.clone(), v.clone());
        }

        let sg_nodes = subgraph.graph.nodes.clone();
        for (i, sg_node) in sg_nodes.iter().enumerate() {
            self.translate_node(sg_node)
                .with_context(|| format!("Subgraph node {i}: {}", sg_node.target))?;
        }

        for (main_out, sg_out) in node.outputs.iter().zip(subgraph.graph.outputs.iter()) {
            if let (Some(main_name), Some(sg_name)) =
                (main_out.as_tensor.as_ref(), sg_out.as_tensor.as_ref())
                && main_name.name != sg_name.name
            {
                let tensor = self.get_tensor(&sg_name.name)?;
                self.tensors.insert(main_name.name.clone(), tensor);
            }
        }

        Ok(())
    }
}
