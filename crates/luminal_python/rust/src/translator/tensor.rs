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

    /// Lower `aten.histc.default` for the integer-bincount case.
    ///
    /// Qwen3-MoE's expert-balance layer calls
    /// `torch.histc(expert_ids.int(), bins=K, min=0, max=K-1)` to count how
    /// many tokens were routed to each expert. With those args every
    /// integer value `i ∈ [0, K-1]` maps to exactly bin `i`, and the result
    /// is equivalent to `torch.bincount`. We implement that case as a
    /// broadcast equality + sum:
    ///
    ///   counts[b] = sum_i (input[i] == b + min)   for b in [0, bins)
    ///
    /// More general histc bin widths (`bins != max - min + 1`, or
    /// non-integer values that span fractional bins) are not supported
    /// today — the equality path would silently drop them. We bail rather
    /// than produce wrong counts.
    pub(crate) fn translate_histc(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let bins_i64: i64 = self
            .get_int_arg(node, 1)
            .context("histc: missing `bins` arg (#1)")?;
        // `min`/`max` are float kwargs (default 0.0 each, which means
        // "auto-pick from input"); for the qwen3-moe call they're always
        // integers passed as floats.
        let min = self.get_float_arg(node, 2).unwrap_or(0.0);
        let max = self.get_float_arg(node, 3).unwrap_or(0.0);

        anyhow::ensure!(
            input.shape.len() == 1,
            "histc: only 1D input is supported, got {}D",
            input.shape.len()
        );
        anyhow::ensure!(
            bins_i64 > 0,
            "histc: bins must be positive, got {}",
            bins_i64
        );
        // Bincount-equivalent case: one integer value per bin.
        anyhow::ensure!(
            (max - min - (bins_i64 - 1) as f64).abs() < 1e-6,
            "histc: only the bincount-equivalent case (bins == max - min + 1) is \
             supported; got bins={}, min={}, max={}. Other cases would need a \
             general bin-width / right-edge-inclusion implementation.",
            bins_i64,
            min,
            max,
        );

        let bins_u = bins_i64 as usize;
        let n = input.shape.dims[0];

        // arange(bins) [bins] → cast to input dtype, optionally shift by min,
        // broadcast to [bins, N], compare for equality with input broadcast.
        let mut bins_arange = self.graph.arange(Expression::from(bins_u));
        if min != 0.0 {
            // `min` is non-zero (uncommon in the qwen3-moe path but legal)
            // — shift the comparison values to start at min.
            let min_i = min as i64;
            let shift = self
                .graph
                .constant_float(min_i as f32)
                .cast(bins_arange.dtype)
                .expand_rhs(bins_arange.shape);
            bins_arange += shift;
        }
        let bins_expanded = bins_arange.cast(input.dtype).expand_dim(1, n);
        let input_expanded = input.expand_dim(0, Expression::from(bins_u));
        let matches = input_expanded.eq(bins_expanded); // Bool [bins, N]

        let out_dtype = self.output_meta_dtype(node)?;
        Ok(matches.cast(out_dtype).sum(1))
    }

    /// Lower `aten.empty.memory_format` and `aten.empty_permuted.default`.
    ///
    /// Both allocate an uninitialised tensor; the caller is responsible for
    /// writing into it. We materialise zeros instead — luminal has no
    /// "uninitialised" notion, and PyTorch's contract on `empty` outputs is
    /// undefined for any read prior to a write, so a zero-fill is sound.
    /// `aten.empty_permuted` additionally takes a `physical_layout` arg
    /// (the storage permutation); for a zero-filled tensor that's a no-op.
    pub(crate) fn translate_empty(&mut self, node: &Node) -> Result<GraphTensor> {
        let shape = self.get_exprs_arg(node, FULL_SHAPE_ARG)?;
        let dtype = self.output_meta_dtype(node)?;
        let zero = self.graph.constant_float(0.0).cast(dtype);
        Ok(if shape.is_empty() {
            zero
        } else {
            zero.expand_rhs(shape)
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
        let out_dtype = self.output_meta_dtype(node)?;

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

        // Gather → [S, K, N], then normalize both operands to the op's declared
        // output dtype before matmul. On real Qwen3-MoE bf16 checkpoints the FX
        // graph inserts casts on the activation path, and relying on the input
        // tensor's translated dtype can leave us with mixed F32/Bf16 operands
        // by the time matmul expands into elementwise Mul. Using the PT2 output
        // metadata keeps the matmul dtype aligned with the exported contract
        // without upcasting the full expert weight bank.
        let weight_gathered = weight.gather(flat_idx).cast(out_dtype);
        let input = input.cast(out_dtype);

        // Per-token matmul: [S, 1, K] @ [S, K, N] → [S, 1, N] → [S, N].
        // Operands stay in their native dtype — no F32 cast on the gathered
        // weight or the input. The earlier cast(F32) was a holdover from the
        // broadcast-and-mask version (which had to use F32 because of the
        // cast(F32) on the mask). Gather-then-matmul has no such requirement,
        // and casting `[S, K, N]` to F32 doubled the gather scratch (~100 MB
        // to ~200 MB per layer for Qwen3-30B-A3B prefill). Matmul rewrites
        // (cuBLASLt etc.) handle bf16 input with F32 accumulator internally.
        let result = input.unsqueeze(1).matmul(weight_gathered).squeeze(1);

        Ok(result.cast(out_dtype))
    }

    /// Build the where-formula graph: `cond * x + (1 - cond) * y`, computed
    /// in F32, cast back to `out_dtype`. Shared between `translate_where`,
    /// `translate_where_scalar_other`, and `translate_masked_fill_scalar` so
    /// they all go through one well-tested code path.
    pub(crate) fn where_formula(
        &mut self,
        cond: GraphTensor,
        x: GraphTensor,
        y: GraphTensor,
        out_dtype: DType,
    ) -> GraphTensor {
        let (cond_b, x_b) = broadcast_binary(cond, x);
        let (cond_bc, y_b) = broadcast_binary(cond_b, y);
        let (x_bc, y_bc) = broadcast_binary(x_b, y_b);
        // Lower as `y + c*(x - y)` rather than `c*x + (1-c)*y`: 3 ops vs 4 ops
        // plus the explicit `1.0` constant. Mathematically identical for
        // c ∈ {0, 1} and produces the same F32 output type.
        let c = cond_bc.cast(DType::F32);
        let x_f = x_bc.cast(DType::F32);
        let y_f = y_bc.cast(DType::F32);
        // Cast back: an F32 result downstream-interpreted as bf16 walks the
        // buffer at half-stride, returning every-other-element zeros.
        (y_f + c * (x_f - y_f)).cast(out_dtype)
    }

    pub(crate) fn translate_where(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, 0)?;
        let x = self.get_input_tensor(node, 1)?;
        let y = self.get_input_tensor(node, 2)?;
        let (x, y) = ensure_same_dtype(x, y);
        let out_dtype = x.dtype;
        Ok(self.where_formula(cond, x, y, out_dtype))
    }

    pub(crate) fn translate_where_scalar_other(&mut self, node: &Node) -> Result<GraphTensor> {
        let cond = self.get_input_tensor(node, WHERE_COND_ARG)?;
        let x = self.get_input_tensor(node, WHERE_X_ARG)?;
        let other_val = self.get_float_arg(node, WHERE_OTHER_ARG)? as f32;
        let out_dtype = x.dtype;
        // Build a tensor for the scalar `other` matching `x`'s shape so we
        // can route through the shared where_formula helper.
        let other = self.graph.constant_float(other_val).expand_rhs(x.shape);
        Ok(self.where_formula(cond, x, other, out_dtype))
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
        let tuple_outputs = node.outputs.first().and_then(|o| o.as_tensors.as_ref());
        let values_name = if let Some(ts) = tuple_outputs {
            ts.first().map(|t| t.name.clone())
        } else {
            node.outputs
                .first()
                .and_then(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
        };
        let indices_name = if let Some(ts) = tuple_outputs {
            ts.get(1).map(|t| t.name.clone())
        } else if node.outputs.len() > 1 {
            node.outputs[1].as_tensor.as_ref().map(|t| t.name.clone())
        } else {
            None
        };

        // Build top-k outputs from a full stable argsort. Slice the indices
        // before gathering values so the gather shape matches the requested
        // top-k output rather than the full sort width.
        let full_argsort = a.stable_argsort(dim, true);
        let topk_indices = full_argsort.slice_along(..k, dim) * 1.0;

        // Only build the outputs that are consumed.
        if let Some(val_name) = values_name
            && !val_name.is_empty()
        {
            let values = a.gather_elements(topk_indices, dim);
            self.tensors.insert(val_name, values);
        }
        if let Some(idx_name) = indices_name {
            self.tensors.insert(idx_name, topk_indices);
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
