use anyhow::{Context, Result, bail};
use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_expr::{ExprBounds, canonical_equal_expr, sym_char_ranges};
use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

const SCATTER_INPUT_ARG: usize = 0;
const SCATTER_DIM_ARG: usize = 1;
const SCATTER_INDEX_ARG: usize = 2;
const SCATTER_VALUE_ARG: usize = 3;

fn normalize_concat_dims(
    lhs: &mut GraphTensor,
    rhs: &mut GraphTensor,
    skip_dim: Option<usize>,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) {
    for i in 0..lhs.shape.len() {
        if Some(i) == skip_dim {
            continue;
        }
        let lhs_dim = lhs.shape.dims[i];
        let rhs_dim = rhs.shape.dims[i];
        if let Some(canonical) = canonical_equal_expr(lhs_dim, rhs_dim, sym_ranges) {
            lhs.shape.dims[i] = canonical;
            rhs.shape.dims[i] = canonical;
        }
    }
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_reshape(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;

        let shape = if let Ok(target_shape) = self.get_ints_arg(node, 1) {
            resolve_neg1_dim(&target_shape, &a.shape.dims)
        } else {
            let exprs = self.get_exprs_arg(node, 1)?;
            resolve_neg1_dim_exprs(&exprs, &a.shape.dims)
        };

        let has_broadcast = a
            .shape
            .dims
            .iter()
            .zip(a.shape.strides.iter())
            .any(|(d, s)| s.to_usize() == Some(0) && d.to_usize() != Some(1));

        let a = if has_broadcast || !a.shape.is_contiguous() {
            a + 0.0
        } else {
            a
        };

        let new_shape = ShapeTracker::new(shape);
        Ok(GraphTensor {
            id: a.id,
            graph_ref: a.graph_ref,
            shape: new_shape,
            dtype: a.dtype,
        })
    }

    pub(crate) fn translate_permute(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dims = self.get_ints_arg(node, 1)?;
        let axes: Vec<usize> = dims
            .iter()
            .map(|&d| normalize_dim(d, a.shape.len()))
            .collect();
        Ok(a.permute(axes))
    }

    pub(crate) fn translate_expand(&mut self, node: &Node) -> Result<GraphTensor> {
        let mut a = self.get_input_tensor(node, 0)?;
        let neg1_expr = Expression::from(-1i32);
        let target_shape: Vec<Expression> = if let Ok(sizes) = self.get_ints_arg(node, 1) {
            sizes
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    if s == -1 {
                        a.shape.dims[i]
                    } else {
                        Expression::from(s as usize)
                    }
                })
                .collect()
        } else {
            self.get_exprs_arg(node, 1)?
                .into_iter()
                .enumerate()
                .map(|(i, e)| if e == neg1_expr { a.shape.dims[i] } else { e })
                .collect()
        };
        a.shape.expand(target_shape);
        Ok(a)
    }

    pub(crate) fn translate_slice(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1).unwrap_or(0);
        let dim = normalize_dim(dim, a.shape.len());

        let start: Expression = if node.inputs.len() > 2 {
            self.get_expr_arg(node, 2)
                .unwrap_or_else(|_| Expression::from(0usize))
        } else {
            Expression::from(0usize)
        };

        if node.inputs.len() <= 3 {
            return Ok(a);
        }

        let end_is_sentinel = self
            .get_int_arg(node, 3)
            .map(|e| e == i64::MAX)
            .unwrap_or(false);

        if end_is_sentinel {
            return Ok(if start.to_usize() == Some(0) {
                a
            } else {
                a.slice_along(start.., dim)
            });
        }

        let end: Expression = self.get_expr_arg(node, 3)?;

        if let Some(s) = start.to_usize()
            && let Some(e) = end.to_usize()
        {
            return Ok(a.slice_along(s..e, dim));
        }

        Ok(a.slice_along(start..end, dim))
    }

    /// `aten.select.int(self, dim, index)` — select element `index` along
    /// `dim`, dropping that dim. Output rank = input rank − 1, so a 1-D input
    /// produces a rank-0 scalar. Both `dim` and `index` may be negative and
    /// are normalized against the input shape.
    ///
    /// Lowered as `slice_along(index..index+1, dim).squeeze(dim)`. We use the
    /// slice + squeeze decomposition (rather than `gather`) because the
    /// composition is a pure shape manipulation with a single iota, which the
    /// luminal compiler can fold into surrounding ops.
    pub(crate) fn translate_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let index_raw = self.get_int_arg(node, 2)?;

        // Normalize a possibly-negative index. PyTorch accepts indices in
        // [-size, size); negative wraps from the end.
        let index = if index_raw < 0 {
            let axis_size = a.shape.dims[dim].to_usize().ok_or_else(|| {
                anyhow::anyhow!(
                    "select.int: dim {} must be concrete to normalize a negative index",
                    dim
                )
            })?;
            let normalized = axis_size as i64 + index_raw;
            if normalized < 0 {
                bail!(
                    "select.int: index {} out of range for dim {} of size {}",
                    index_raw,
                    dim,
                    axis_size
                );
            }
            normalized as usize
        } else {
            index_raw as usize
        };

        Ok(a.slice_along(index..index + 1, dim).squeeze(dim))
    }

    pub(crate) fn translate_cat(&mut self, node: &Node) -> Result<GraphTensor> {
        let tensors: Vec<GraphTensor> = if let Some(names) = node.inputs[0].arg.as_tensors() {
            names
                .iter()
                .map(|n| self.get_tensor(&n.name))
                .collect::<Result<_>>()?
        } else {
            let mut ts = Vec::new();
            for input in &node.inputs {
                if let Some(name) = input.arg.as_tensor_name()
                    && let Ok(t) = self.get_tensor(name)
                {
                    ts.push(t);
                }
            }
            ts
        };

        if tensors.is_empty() {
            bail!("cat: no tensor inputs found");
        }

        let dim = node
            .inputs
            .iter()
            .find(|i| i.arg.as_int().is_some() && i.name != "tensors")
            .and_then(|i| i.arg.as_int())
            .unwrap_or(0);

        let tensors: Vec<GraphTensor> = tensors
            .into_iter()
            .filter(|t| !t.shape.dims.iter().any(|d| d.to_usize() == Some(0)))
            .collect();

        if tensors.is_empty() {
            bail!("cat: all tensor inputs are empty");
        }

        let dim = normalize_dim(dim, tensors[0].shape.len());
        let mut result = tensors[0];
        let sym_ranges = sym_char_ranges(&self.sym_map);
        for t in &tensors[1..] {
            let mut next = *t;
            normalize_concat_dims(&mut result, &mut next, Some(dim), &sym_ranges);

            let lhs_axis = result.dims()[dim];
            let rhs_axis = next.dims()[dim];
            let mut lhs_padded = result.pad_along(0, rhs_axis, dim, 0.);
            let mut rhs_padded = next.pad_along(lhs_axis, 0, dim, 0.);
            normalize_concat_dims(&mut lhs_padded, &mut rhs_padded, None, &sym_ranges);
            result = lhs_padded + rhs_padded;
        }
        Ok(result)
    }

    pub(crate) fn translate_embedding(&mut self, node: &Node) -> Result<GraphTensor> {
        let weight = self.get_input_tensor(node, 0)?;
        let indices = self.get_input_tensor(node, 1)?;

        let hidden_dim = weight.shape.dims[1];
        let seq_shape = indices.shape.dims;

        let indices_int = indices.cast(DType::Int);
        let ids_expanded = (indices_int * hidden_dim).expand_dim(seq_shape.len(), hidden_dim);

        let arange = self.graph.arange(hidden_dim);
        let mut arange_expanded = arange;
        for d in seq_shape.iter().rev() {
            arange_expanded = arange_expanded.expand_dim(0, *d);
        }

        Ok(weight.gather(ids_expanded + arange_expanded))
    }

    pub(crate) fn translate_index_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let source = self.get_input_tensor(node, 0)?;

        // Handle indices as_tensors (all non-None) or as individual args with None entries
        let index_names: Vec<crate::pt2_schema::TensorName>;
        let mut first_non_none_dim = 0usize;

        if let Some(names) = node.inputs[1].arg.as_tensors() {
            index_names = names.to_vec();
        } else {
            let indices_arg = &node.inputs[1].arg;

            // Check if it's a single tensor (1D indexing)
            if let Some(name) = indices_arg.as_tensor_name() {
                index_names = vec![crate::pt2_schema::TensorName {
                    name: name.to_string(),
                }];
            } else if let Some(opt_tensors) = indices_arg.as_optional_tensors() {
                // Optional tensors list: [None, tensor, None, ...] for selective dim indexing
                use crate::pt2_schema::OptionalTensorEntry;
                let mut found_tensors: Vec<crate::pt2_schema::TensorName> = Vec::new();
                for (i, entry) in opt_tensors.iter().enumerate() {
                    if let OptionalTensorEntry::Tensor(t) = entry {
                        if found_tensors.is_empty() {
                            first_non_none_dim = i;
                        }
                        found_tensors.push(t.as_tensor.clone());
                    }
                }
                if found_tensors.is_empty() {
                    bail!("index.Tensor: no index tensors in optional_tensors list");
                }
                index_names = found_tensors;
                // Simple case: single non-None index on a specific dim → gather_elements
                if first_non_none_dim > 0 && index_names.len() == 1 {
                    let idx = self.get_tensor(&index_names[0].name)?.cast(DType::Int);
                    // gather_elements requires indices to have the same rank as data.
                    // PyTorch fancy indexing gives 1D indices that broadcast across other dims.
                    // Add unit leading dims to match rank, then broadcast to output shape.
                    let src_dims = source.shape.dims;
                    let src_rank = src_dims.len();
                    let mut expanded = idx;
                    for _ in 0..(src_rank - expanded.shape.len()) {
                        expanded = expanded.expand_dim(0, Expression::from(1usize));
                    }
                    // Build target shape: source dims everywhere except the indexed dim
                    let idx_dim_size = expanded.shape.dims[first_non_none_dim];
                    let mut target: Vec<Expression> = src_dims.to_vec();
                    target[first_non_none_dim] = idx_dim_size;
                    expanded.shape.expand(target);
                    return Ok(source.gather_elements(expanded, first_non_none_dim));
                }
            } else {
                bail!(
                    "index.Tensor: unsupported indices format: {:?}",
                    indices_arg
                );
            }
        }

        let index_names = &index_names;

        let src_shape = source.shape.dims;
        let n_indexed = index_names.len();

        let mut strides: Vec<Expression> = vec![Expression::from(1usize); n_indexed];
        for i in (0..n_indexed - 1).rev() {
            strides[i] = strides[i + 1] * src_shape[i + 1];
        }

        let mut flat_idx: Option<GraphTensor> = None;
        for (dim_idx, idx_name) in index_names.iter().enumerate() {
            let idx_tensor = self.get_tensor(&idx_name.name)?;

            // Normalize negative indices for this dimension. Stay in Int —
            // multiplying an Int tensor by an Expression broadcasts the axis
            // size, so we avoid three Cast nodes (Int→F32 for indices, F32→Int
            // for the result, Bool→F32 for the negative mask) per indexed dim.
            let axis_size = src_shape[dim_idx];
            let idx_int = idx_tensor.cast(DType::Int);
            let zero = self.graph.constant(0).expand_rhs(idx_int.shape);
            let is_negative = idx_int.lt(zero).cast(DType::Int);
            let idx_int = idx_int + is_negative * axis_size;

            let stride = &strides[dim_idx];
            let weighted = if stride.to_usize() == Some(1) {
                idx_int
            } else {
                idx_int * *stride
            };

            flat_idx = Some(match flat_idx {
                Some(acc) => {
                    let (acc_b, w_b) = broadcast_binary(acc, weighted);
                    acc_b + w_b
                }
                None => weighted,
            });
        }

        let mut indexed_size = Expression::from(1usize);
        for i in 0..n_indexed {
            indexed_size *= src_shape[i];
        }
        let remaining_dims: Vec<Expression> = src_shape[n_indexed..].to_vec();

        let mut flat_shape = vec![indexed_size];
        flat_shape.extend_from_slice(&remaining_dims);
        let flat_source = reshape_tensor(source, flat_shape);

        let flat_idx = flat_idx.context("index.Tensor: no indices")?;

        if remaining_dims.is_empty() {
            Ok(flat_source.gather(flat_idx))
        } else {
            let mut remaining_size = Expression::from(1usize);
            for d in &remaining_dims {
                remaining_size *= *d;
            }

            let idx_shape = flat_idx.shape.dims;
            let mut expanded_idx = flat_idx * remaining_size;

            expanded_idx = expanded_idx.expand_dim(idx_shape.len(), remaining_size);

            let arange = self.graph.arange(remaining_size);
            let mut arange_expanded = arange;
            for d in idx_shape.iter().rev() {
                arange_expanded = arange_expanded.expand_dim(0, *d);
            }

            let final_idx = expanded_idx + arange_expanded;
            let total_elements = indexed_size * remaining_size;
            let fully_flat = reshape_tensor(flat_source, vec![total_elements]);
            let gathered = fully_flat.gather(final_idx);

            let mut result_shape: Vec<Expression> = idx_shape.to_vec();
            result_shape.extend_from_slice(&remaining_dims);
            Ok(reshape_tensor(gathered, result_shape))
        }
    }

    pub(crate) fn translate_gather(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?;

        // PyTorch eager allows torch.gather(rank-1, 0, rank-0) and returns
        // a rank-0 scalar — the only rank-mismatch case eager permits. Our
        // gather_elements requires the index rank to match the source rank,
        // so unsqueeze the rank-0 index to (1,), gather, then squeeze back.
        let promoted_rank0 = indices.shape.is_empty() && a.shape.len() == 1;
        let indices = if promoted_rank0 {
            indices.unsqueeze(0)
        } else {
            indices
        };

        // Normalize negative indices: -1 → last, -2 → second-to-last, etc.
        // Stay in Int the whole way — multiplying an Int tensor by an
        // Expression broadcasts the axis size and avoids three Cast nodes
        // (Int→F32 for indices, F32→Int for the result, plus a Bool→F32 for
        // the negative mask) that the previous F32-routed path emitted.
        let axis_dim = a.shape.dims[dim];
        let indices_int = indices.cast(DType::Int);
        let zero = self.graph.constant(0).expand_rhs(indices_int.shape);
        let is_negative = indices_int.lt(zero).cast(DType::Int);
        let normalized = indices_int + is_negative * axis_dim;

        let result = a.gather_elements(normalized, dim);
        Ok(if promoted_rank0 {
            result.squeeze(0)
        } else {
            result
        })
    }

    pub(crate) fn translate_scatter_src(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?;
        let src = self.get_input_tensor(node, 3)?;
        Ok(a.scatter_elements(indices.cast(DType::Int), src, dim))
    }

    pub(crate) fn translate_scatter_value(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, SCATTER_INPUT_ARG)?;
        let dim = self.get_int_arg(node, SCATTER_DIM_ARG)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, SCATTER_INDEX_ARG)?;
        let value_arg = &node
            .inputs
            .get(SCATTER_VALUE_ARG)
            .context("scatter.value missing value input")?
            .arg;
        let value = if let Some(b) = value_arg.as_bool() {
            self.graph.constant(if b { 1 } else { 0 }).cast(a.dtype)
        } else if let Some(i) = value_arg.as_int() {
            self.graph.constant(i).cast(a.dtype)
        } else if let Some(f) = value_arg.as_float() {
            self.graph.constant_float(f as f32).cast(a.dtype)
        } else {
            bail!("scatter.value: unsupported scalar argument {:?}", value_arg);
        }
        .expand_rhs(indices.shape);
        Ok(a.scatter_elements(indices.cast(DType::Int), value, dim))
    }

    pub(crate) fn translate_index_put(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let index_names = node.inputs[1]
            .arg
            .as_tensors()
            .context("index_put: indices not as_tensors")?;
        let values = self.get_input_tensor(node, 2)?;

        if index_names.len() == 1 {
            let idx_tensor = self.get_tensor(&index_names[0].name)?;

            // Boolean-mask index_put: when the only index is a Bool tensor whose
            // shape matches the data tensor, PyTorch semantics are
            //   data[mask] = value   ↔   where(mask, value, data)
            // NOT a scatter into positions. Casting the Bool mask to Int and
            // feeding it to scatter_nd would reinterpret True/False as row
            // indices 1/0 and silently corrupt the data. Reproducer:
            //   x = arange(16).reshape(4, 4); mask = zeros(4, 4, dtype=bool)
            //   y = x.clone(); y[mask] = 99   # eager: y == x (no-op)
            // Pre-fix the compiled graph wrote 99 to row 0; this branch
            // ensures the bool-mask path lowers to a where-blend instead.
            if idx_tensor.dtype == DType::Bool && idx_tensor.shape.dims == a.shape.dims {
                // Broadcast the (often scalar) value tensor to match data shape,
                // then blend by mask. Cast mask to data's dtype for the
                // arithmetic so this works for both integer and float data.
                let mask_f = idx_tensor.cast(a.dtype);
                let values_b = values.cast(a.dtype).expand_rhs(a.shape);
                // where(mask, value, a) as `a + mask*(value - a)`. Saves a mul
                // and the `1.0` constant compared to the `a*(1 - m) + v*m`
                // form; works for any numeric dtype without a dedicated cond.
                return Ok(a + mask_f * (values_b - a));
            }

            // Integer-index scatter: index_put with indices=[idx_tensor] writes
            // into dim 0 of `a` at every position named in idx_tensor (flattened),
            // broadcasting values across the trailing dims of `a`. idx_tensor can
            // be ANY shape — its whole shape is "batch dims" in scatter_nd terms,
            // and K is always 1 (number of dims we're indexing into). Always pad
            // a trailing size-1 dim so the rank-1 and rank-N cases share a path.
            let indices = idx_tensor.cast(DType::Int);
            let new_last = indices.shape.len();
            let indices = indices.expand_dim(new_last, Expression::from(1usize));
            Ok(a.scatter_nd(indices, values))
        } else {
            bail!("index_put with multiple index tensors not yet supported");
        }
    }

    pub(crate) fn translate_split_with_sizes(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let sizes = self.get_ints_arg(node, 1)?;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(0)
        } else {
            0
        };
        let dim = normalize_dim(dim, a.shape.len());

        let output_names: Vec<String> = node
            .outputs
            .first()
            .and_then(|o| o.as_tensors.as_ref())
            .map(|ts| ts.iter().map(|t| t.name.clone()).collect())
            .unwrap_or_else(|| {
                node.outputs
                    .iter()
                    .filter_map(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
                    .collect()
            });

        let mut offset = 0usize;
        let mut first_chunk = None;
        for (i, &size) in sizes.iter().enumerate() {
            let size = size as usize;
            let chunk = a.slice_along(offset..offset + size, dim);
            if let Some(name) = output_names.get(i) {
                self.tensors.insert(name.clone(), chunk);
            }
            if i == 0 {
                first_chunk = Some(chunk);
            }
            offset += size;
        }

        first_chunk.ok_or_else(|| anyhow::anyhow!("split_with_sizes: empty sizes list"))
    }
}
