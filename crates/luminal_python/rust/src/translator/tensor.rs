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
    /// Implementation: for each token m we (a) compute its expert id from offs,
    /// (b) gather only that expert's `[K, N]` slice from weight, and (c) do a
    /// single per-token matmul. The gather pattern mirrors the rust qwen3_moe
    /// example's `gather_experts`, which the GLUMoE host-op fusion in
    /// `luminal_cuda_lite` is designed to recognise.
    ///
    /// Why not the straightforward `[G, S, K] @ [G, K, N] → [G, S, N]` + mask:
    /// it forces a full F32 cast of the entire `[G, K, N]` weight tensor as
    /// search-time intermediate, which OOMs on real MoE checkpoints
    /// (Qwen3-30B-A3B: 1.5 GB / layer × 48 layers for gate-up alone). Gathering
    /// first keeps the F32 cast on `[S, K, N]` instead — for prefill (S = top_k)
    /// that is a 16× shrink (G=128, top_k=8).
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
        let k = weight.shape.dims[1];
        let n = weight.shape.dims[2];

        // expert_id[m] = number of g s.t. m >= offs[g], clamped to [0, G-1].
        // Same value as HF MoE's `expert_ids.clamp(0, num_experts-1)` for
        // invalid expert IDs from EP, AND protects search-time profiling:
        // dummy-1 input bytes give offs=[1,…,1], which pushes the raw count
        // to G for any token with index ≥ 1 and would OOB the weight gather.
        //
        // Stay in Int throughout — arange / offs are already Int, ge → Bool
        // → cast(Int), sum stays Int, and the binary `minimum` handles the
        // clamp without an F32 round-trip.
        let _ = g
            .to_usize()
            .context("_grouped_mm: G (num_experts) must be concrete")?;
        let s_arange = self.graph.arange(s); // Int [S]
        let ge_int = s_arange
            .expand_dim(0, g)
            .ge(offs.expand_dim(1, s)) // Bool [G, S]
            .cast(DType::Int); // Int [G, S]
        let raw = ge_int.sum(0); // Int [S], values in [0, G]
        let cap = self.graph.constant(g - 1).expand_dim(0, s); // Int [S], all G-1
        let expert_id = raw.minimum(cap); // Int [S]

        // Flat gather index into weight (treated as a length-G*K*N 1D buffer):
        //   flat[m, k_, n_] = expert_id[m] * (K*N) + k_ * N + n_
        // Encoded as `Mul(expert_id, Iota(io_const)) + Iota(MIter, K*N)` so the
        // resulting Gather matches the GLUMoE / gather-experts egglog patterns.
        let io = k * n;
        let base = expert_id * io;
        let within = self.graph.iota(Expression::from('z'), (k, n));
        let exp_base = base.expand_dim(1, k).expand_dim(2, n);
        let exp_within = within.expand_dim(0, s);
        let flat_idx = exp_base + exp_within;

        // Gather → [S, K, N], preserves weight's native dtype (bf16 stays bf16).
        let weight_gathered = weight.gather(flat_idx);

        // Per-token matmul: [S, 1, K] @ [S, K, N] → [S, 1, N] → [S, N].
        // Operands stay in their native dtype — no F32 cast on the gathered
        // weight or the input. The earlier cast(F32) was a holdover from the
        // broadcast-and-mask version (which had to use F32 because of the
        // cast(F32) on the mask). Gather-then-matmul has no such requirement,
        // and casting `[S, K, N]` to F32 doubled the gather scratch (~100 MB
        // to ~200 MB per layer for Qwen3-30B-A3B prefill). Matmul rewrites
        // (cuBLASLt etc.) handle bf16 input with F32 accumulator internally.
        let result = input.unsqueeze(1).matmul(weight_gathered).squeeze(1);

        Ok(result.cast(input.dtype))
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
