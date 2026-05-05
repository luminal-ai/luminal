use anyhow::{Context, Result};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

const FULL_SHAPE_ARG: usize = 0;
const FULL_VALUE_ARG: usize = 1;

const FULL_LIKE_INPUT_ARG: usize = 0;
const FULL_LIKE_VALUE_ARG: usize = 1;

const TOPK_INPUT_ARG: usize = 0;
const TOPK_K_ARG: usize = 1;
const TOPK_DIM_ARG: usize = 2;

const SORT_INPUT_ARG: usize = 0;
const SORT_DIM_ARG: usize = 1;
const SORT_DESCENDING_ARG: usize = 2;

const WHERE_COND_ARG: usize = 0;
const WHERE_X_ARG: usize = 1;
const WHERE_OTHER_ARG: usize = 2;

const TRIANGULAR_INPUT_ARG: usize = 0;
const TRIANGULAR_DIAGONAL_ARG: usize = 1;

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
            2 => Ok(self
                .graph
                .arange_options(positional_args[0], positional_args[1], 1)),
            _ => Ok(self.graph.arange_options(
                positional_args[0],
                positional_args[1],
                positional_args[2],
            )),
        }
    }

    pub(crate) fn translate_full(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.get_exprs_arg(node, FULL_SHAPE_ARG)?;
        // fill_value can be float, int, or bool after decomposition
        let val = if let Ok(f) = self.get_float_arg(node, FULL_VALUE_ARG) {
            f as f32
        } else if let Ok(b) = self.get_bool_arg(node, FULL_VALUE_ARG) {
            if b { 1.0 } else { 0.0 }
        } else {
            anyhow::bail!(
                "full: unsupported fill value type: {:?}",
                node.inputs.get(FULL_VALUE_ARG)
            );
        };
        let dtype = self.output_meta_dtype(node)?;
        let value = self.graph.constant_float(val).cast(dtype);
        Ok(if shape.is_empty() {
            value
        } else {
            value.expand_rhs(shape)
        })
    }

    pub(crate) fn translate_full_like(&mut self, node: &Node) -> Result<GraphTensor> {
        let reference = self.get_input_tensor(node, FULL_LIKE_INPUT_ARG)?;
        let val = if let Ok(f) = self.get_float_arg(node, FULL_LIKE_VALUE_ARG) {
            f as f32
        } else if let Ok(b) = self.get_bool_arg(node, FULL_LIKE_VALUE_ARG) {
            if b { 1.0 } else { 0.0 }
        } else {
            anyhow::bail!(
                "full_like: unsupported fill value type: {:?}",
                node.inputs.get(FULL_LIKE_VALUE_ARG)
            );
        };
        let dtype = self.output_meta_dtype(node)?;
        let value = self.graph.constant_float(val).cast(dtype);
        Ok(value.expand_rhs(reference.shape))
    }

    fn output_meta_dtype(&self, node: &Node) -> Result<DType> {
        let output_name = node
            .outputs
            .first()
            .and_then(|o| o.as_tensor.as_ref())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&output_name)
            .context("Missing tensor meta for output dtype")?;
        Ok(torch_dtype_int_to_luminal(meta.dtype))
    }

    /// Translate `aten._grouped_mm.default(input, weight, offs)` → `Tensor[S, N]`.
    ///
    /// Grouped matmul: `input` is `[S, K]` (tokens sorted by expert), `weight` is
    /// `[G, K, N]` (per-expert weights), `offs` is `[G]` cumulative token counts.
    /// Output `[S, N]` where token m (in group g s.t. `offs[g-1] <= m < offs[g]`)
    /// is multiplied by `weight[g]`.
    ///
    /// Implementation:
    ///   1. Batched matmul across every expert: `[G, S, K] @ [G, K, N] → [G, S, N]`
    ///      (input broadcast along the G batch dim — matches luminal's 3D@3D pattern
    ///      so the CUDA optimizer can fuse it into a batched GEMM).
    ///   2. Build a `[G, S]` group-membership mask from `offs`:
    ///      `expert_id[m] = Σ_g (offs[g] <= m)`, then `mask[g, m] = (g == expert_id[m])`.
    ///   3. Multiply `[G, S, N]` result by the broadcast mask and sum over `G`.
    ///
    /// `offs` flows through as a runtime tensor — the routing decision is computed
    /// at execution time by the gate network and the same compiled graph handles
    /// any routing pattern without recompilation.
    pub(crate) fn translate_grouped_mm(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let weight = self.get_input_tensor(node, 1)?;
        let offs = self.get_input_tensor(node, 2)?;

        anyhow::ensure!(
            input.shape.len() == 2,
            "_grouped_mm: input must be 2D, got {}D",
            input.shape.len()
        );
        anyhow::ensure!(
            weight.shape.len() == 3,
            "_grouped_mm: weight must be 3D, got {}D",
            weight.shape.len()
        );
        anyhow::ensure!(
            offs.shape.len() == 1,
            "_grouped_mm: offs must be 1D, got {}D",
            offs.shape.len()
        );

        let s = input.shape.dims[0];
        let g = weight.shape.dims[0];
        let n = weight.shape.dims[2];

        let input_f = input.cast(DType::F32);
        let weight_f = weight.cast(DType::F32);
        let offs_f = offs.cast(DType::F32);

        // Batched matmul over every expert: [G, S, K] @ [G, K, N] → [G, S, N].
        let input_batched = input_f.expand_dim(0, g);
        let all_out = input_batched.matmul(weight_f);

        // Group mask [G, S].
        let s_arange = self.graph.arange(s).cast(DType::F32);
        let g_arange = self.graph.arange(g).cast(DType::F32);
        let ge_boundary = s_arange
            .expand_dim(0, g)
            .ge(offs_f.expand_dim(1, s))
            .cast(DType::F32);
        let expert_id = ge_boundary.sum(0);
        let mask = g_arange
            .expand_dim(1, s)
            .eq(expert_id.expand_dim(0, g))
            .cast(DType::F32);

        // Apply mask and sum over experts.
        let out = (all_out * mask.expand_dim(2, n)).sum(0);

        Ok(out.cast(input.dtype))
    }

    pub(crate) fn translate_where(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let y = self.get_input_tensor(node, 2)?;
        // Ensure x and y have the same dtype
        let (x, y) = ensure_same_dtype(x, y);
        // Broadcast all three tensors to a common shape first
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let (cond_bc, y_b) = broadcast_binary(cond_b, y);
        let (x_bc, y_bc) = broadcast_binary(x_b, y_b);
        let c = cond_bc.cast(DType::F32);
        let x_f = x_bc.cast(DType::F32);
        let y_f = y_bc.cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(c.shape);
        Ok(c * x_f + (one - c) * y_f)
    }

    pub(crate) fn translate_where_scalar_other(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, WHERE_COND_ARG)?;
        let x = self.get_input_tensor(node, WHERE_X_ARG)?;
        let other_val = self.get_float_arg(node, WHERE_OTHER_ARG)? as f32;
        // Broadcast cond and x to a common shape
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let c = cond_b.cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(c.shape);
        let other = self.graph.constant_float(other_val).expand_rhs(c.shape);
        Ok(c * x_b + (one - c) * other)
    }

    pub(crate) fn translate_tril(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, false)
    }

    pub(crate) fn translate_triu(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_triangular(node, true)
    }

    fn translate_triangular(&mut self, node: &Node, upper: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, TRIANGULAR_INPUT_ARG)?;
        let diagonal = if node.inputs.len() > TRIANGULAR_DIAGONAL_ARG {
            self.get_int_arg(node, TRIANGULAR_DIAGONAL_ARG).unwrap_or(0) as i32
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
        let a = self.get_input_tensor(node, TOPK_INPUT_ARG)?;
        let k = self.get_int_arg(node, TOPK_K_ARG)? as usize;
        let dim = if node.inputs.len() > TOPK_DIM_ARG {
            self.get_int_arg(node, TOPK_DIM_ARG).unwrap_or(-1)
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

        // Build top-k outputs from a full stable argsort, then slice to k.
        let full_argsort = a.stable_argsort(dim, true);

        // Only build the outputs that are consumed.
        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = a.gather_elements(full_argsort, dim).slice_along(..k, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            // Materialize the sliced indices through a copy before storing them.
            let indices = full_argsort.slice_along(..k, dim) * 1.0;
            self.tensors.insert(idx_name, indices);
        }

        Ok(())
    }

    pub(crate) fn translate_sort(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, SORT_INPUT_ARG)?;
        let dim = if node.inputs.len() > SORT_DIM_ARG {
            self.get_int_arg(node, SORT_DIM_ARG).unwrap_or(-1)
        } else {
            -1
        };
        let descending = if node.inputs.len() > SORT_DESCENDING_ARG {
            self.get_bool_arg(node, SORT_DESCENDING_ARG)
                .unwrap_or(false)
        } else {
            false
        };
        let dim = normalize_dim(dim, a.shape.len());

        // Determine output names (sort returns (values, indices))
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

        let full_argsort = a.stable_argsort(dim, descending);

        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = a.gather_elements(full_argsort, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            let indices = full_argsort * 1.0;
            self.tensors.insert(idx_name, indices);
        }

        Ok(())
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
