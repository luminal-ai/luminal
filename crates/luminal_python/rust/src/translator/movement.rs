use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

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

    pub(crate) fn translate_transpose(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim0 = self.get_int_arg(node, 1)?;
        let dim1 = self.get_int_arg(node, 2)?;
        let dim0 = normalize_dim(dim0, a.shape.len());
        let dim1 = normalize_dim(dim1, a.shape.len());
        Ok(a.transpose(dim0, dim1))
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

    pub(crate) fn translate_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let index = self.get_int_arg(node, 2)?;
        let index = if index < 0 {
            bail!("Negative select index not yet supported");
        } else {
            index as usize
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
        for t in &tensors[1..] {
            result = result.concat_along(*t, dim);
        }
        Ok(result)
    }

    pub(crate) fn translate_index_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?.cast(DType::Int);
        let src_dims = a.shape.dims;
        let idx_len = indices.shape.dims[0];

        // Reshape 1D indices [K] → [1,..,K,..,1] with K at position `dim`
        let mut idx = indices;
        for _ in 0..dim {
            idx = idx.unsqueeze(0);
        }
        for _ in (dim + 1)..src_dims.len() {
            idx = idx.expand_dim(idx.shape.len(), Expression::from(1usize));
        }

        // Expand to output shape: src_dims with dim replaced by idx_len
        let mut target: Vec<Expression> = src_dims.to_vec();
        target[dim] = idx_len;
        idx.shape.expand(target);

        Ok(a.gather_elements(idx, dim))
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

            // Normalize negative indices for this dimension
            let axis_size = src_shape[dim_idx].to_usize().ok_or_else(|| {
                anyhow::anyhow!(
                    "index.Tensor: dim {} must be concrete for negative index normalization",
                    dim_idx
                )
            })?;
            let idx_f32 = idx_tensor.cast(DType::F32);
            let zero = self.graph.constant_float(0.0).expand_rhs(idx_f32.shape);
            let adjustment = self
                .graph
                .constant_float(axis_size as f32)
                .expand_rhs(idx_f32.shape);
            let is_negative = idx_f32.lt(zero).cast(DType::F32);
            let idx_int = (idx_f32 + is_negative * adjustment).cast(DType::Int);

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

        // Normalize negative indices: -1 → last, -2 → second-to-last, etc.
        let axis_dim = a.shape.dims[dim].to_usize().ok_or_else(|| {
            anyhow::anyhow!("Gather: axis dim must be concrete for negative index normalization")
        })?;
        let indices_f32 = indices.cast(DType::F32);
        let zero = self.graph.constant_float(0.0).expand_rhs(indices_f32.shape);
        let adjustment = self
            .graph
            .constant_float(axis_dim as f32)
            .expand_rhs(indices_f32.shape);
        let is_negative = indices_f32.lt(zero).cast(DType::F32);
        let normalized = (indices_f32 + is_negative * adjustment).cast(DType::Int);

        Ok(a.gather_elements(normalized, dim))
    }

    pub(crate) fn translate_scatter_src(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let indices = self.get_input_tensor(node, 2)?;
        let src = self.get_input_tensor(node, 3)?;
        Ok(a.scatter_elements(indices.cast(DType::Int), src, dim))
    }

    pub(crate) fn translate_index_put(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let index_names = node.inputs[1]
            .arg
            .as_tensors()
            .context("index_put: indices not as_tensors")?;
        let values = self.get_input_tensor(node, 2)?;

        if index_names.len() == 1 {
            let indices = self.get_tensor(&index_names[0].name)?.cast(DType::Int);
            // scatter_nd expects indices of shape [batch, K] where K = number of index dims.
            // PT2's index_put gives 1D indices [batch]; reshape to [batch, 1].
            let indices = if indices.shape.len() == 1 {
                indices.expand_dim(1, Expression::from(1usize))
            } else {
                indices
            };
            Ok(a.scatter_nd(indices, values))
        } else {
            bail!("index_put with multiple index tensors not yet supported");
        }
    }

    pub(crate) fn translate_split(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let split_size = self.get_int_arg(node, 1)? as usize;
        let dim = if node.inputs.len() > 2 {
            self.get_int_arg(node, 2).unwrap_or(0)
        } else {
            0
        };
        let dim = normalize_dim(dim, a.shape.len());

        let dim_size = a.shape.dims[dim];
        if let Some(total) = dim_size.to_usize() {
            // Collect output names from as_tensors (multi-output) or as_tensor (single)
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

            // Store each chunk under its output name
            for (i, out_name) in output_names.iter().enumerate() {
                let start = i * split_size;
                let end = ((i + 1) * split_size).min(total);
                if start < total {
                    let chunk = a.slice_along(start..end, dim);
                    self.tensors.insert(out_name.clone(), chunk);
                }
            }

            // Return the first chunk
            Ok(a.slice_along(0..split_size.min(total), dim))
        } else {
            Ok(a.slice_along(0..split_size, dim))
        }
    }
}
