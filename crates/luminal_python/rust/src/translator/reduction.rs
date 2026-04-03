use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

/// Re-insert reduced axes as size-1 dims (keepdim behavior).
fn keepdim_unsqueeze(mut t: GraphTensor, axes: &[usize]) -> GraphTensor {
    let mut sorted = axes.to_vec();
    sorted.sort();
    for &ax in &sorted {
        t = t.unsqueeze(ax);
    }
    t
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dims = self.get_ints_arg(node, 1)?;
        let keepdim = if node.inputs.len() > 2 {
            self.get_bool_arg(node, 2).unwrap_or(false)
        } else {
            false
        };

        let ndim = a.shape.len();
        let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();

        let mut result = match op {
            ReductionOp::Sum => a.sum(axes.clone()),
            ReductionOp::Mean => a.mean(axes.clone()),
            ReductionOp::Max => a.max(axes.clone()),
            ReductionOp::Min => a.min(axes.clone()),
        };

        if keepdim {
            result = keepdim_unsqueeze(result, &axes);
        }

        Ok(result)
    }

    // --- Single-output reduction ops ---

    pub(crate) fn translate_argmax(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_argmax_min(node, true)
    }

    pub(crate) fn translate_argmin(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_argmax_min(node, false)
    }

    fn translate_argmax_min(&mut self, node: &Node, is_max: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut result = if is_max { a.argmax(dim) } else { a.argmin(dim) };
            if keepdim {
                result = result.unsqueeze(dim);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&a)?;
            let mut flat = a;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(if is_max { flat.argmax(1) } else { flat.argmin(1) })
        }
    }

    pub(crate) fn translate_prod(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut result = a.prod(vec![dim]);
            if keepdim {
                result = result.unsqueeze(dim);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&a)?;
            let mut flat = a;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(flat.prod(vec![1]))
        }
    }

    pub(crate) fn translate_argsort_op(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1).unwrap_or(-1);
        let dim = normalize_dim(dim, a.shape.len());
        let descending = self.get_bool_arg(node, 2).unwrap_or(false);
        Ok(a.argsort(dim, descending))
    }

    pub(crate) fn translate_log_softmax(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        Ok(a.log_softmax(dim))
    }

    pub(crate) fn translate_all(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_all_any(node, true)
    }

    pub(crate) fn translate_any(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_all_any(node, false)
    }

    /// Shared impl for `all` (use_min=true) and `any` (use_min=false).
    fn translate_all_any(&mut self, node: &Node, use_min: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let zero = self
            .graph
            .constant_float(0.0)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        let nonzero = a.cast(DType::F32).ne(zero.cast(DType::F32)).cast(DType::F32);
        let reduce = |t: GraphTensor, axes: Vec<usize>| {
            if use_min { t.min(axes) } else { t.max(axes) }
        };
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut result = reduce(nonzero, vec![dim]);
            if keepdim {
                result = result.unsqueeze(dim);
            }
            Ok(result.cast(DType::Bool))
        } else {
            let total = concrete_numel(&nonzero)?;
            let mut flat = nonzero;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(reduce(flat, vec![1]).cast(DType::Bool))
        }
    }

    pub(crate) fn translate_count_nonzero(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let zero = self
            .graph
            .constant_float(0.0)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        let nonzero = a.cast(DType::F32).ne(zero.cast(DType::F32)).cast(DType::F32);
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            Ok(nonzero.sum(vec![dim]).cast(DType::Int))
        } else if node.inputs.len() > 1 && node.inputs[1].arg.as_ints().is_some() {
            let dims = self.get_ints_arg(node, 1)?;
            let ndim = a.shape.len();
            let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
            Ok(nonzero.sum(axes).cast(DType::Int))
        } else {
            let total = concrete_numel(&nonzero)?;
            let mut flat = nonzero;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(flat.sum(vec![1]).cast(DType::Int))
        }
    }

    pub(crate) fn translate_nansum(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let clean = self.nan_to_zero(a);
        if node.inputs.len() > 1 && node.inputs[1].arg.as_ints().is_some() {
            let dims = self.get_ints_arg(node, 1)?;
            let ndim = a.shape.len();
            let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut result = clean.sum(axes.clone());
            if keepdim {
                result = keepdim_unsqueeze(result, &axes);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&clean)?;
            let mut flat = clean;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(flat.sum(vec![1]))
        }
    }

    pub(crate) fn translate_nanmean(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let is_nan = a.ne(a).cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(a.shape);
        let clean = a * (one - is_nan);
        let count = one - is_nan;
        if node.inputs.len() > 1 && node.inputs[1].arg.as_ints().is_some() {
            let dims = self.get_ints_arg(node, 1)?;
            let ndim = a.shape.len();
            let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut result = clean.sum(axes.clone()) / count.sum(axes.clone());
            if keepdim {
                result = keepdim_unsqueeze(result, &axes);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&clean)?;
            let mut flat_clean = clean;
            flat_clean.shape = ShapeTracker::new(vec![1, total]);
            let mut flat_count = count;
            flat_count.shape = ShapeTracker::new(vec![1, total]);
            Ok(flat_clean.sum(vec![1]) / flat_count.sum(vec![1]))
        }
    }

    pub(crate) fn translate_logsumexp(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dims = self.get_ints_arg(node, 1)?;
        let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
        let ndim = a.shape.len();
        let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();

        // Numerically stable: max + log(sum(exp(x - max)))
        let m = a.max(axes.clone());
        let m_expanded = keepdim_unsqueeze(m, &axes);
        let (a_bc, m_bc) = broadcast_binary(a, m_expanded);
        let shifted = a_bc - m_bc;
        let mut result = shifted.exp().sum(axes.clone()).log() + m;
        if keepdim {
            result = keepdim_unsqueeze(result, &axes);
        }
        Ok(result)
    }

    pub(crate) fn translate_sum_to_size(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let target = self.get_ints_arg(node, 1)?;
        let mut axes = Vec::new();
        for (i, &t) in target.iter().enumerate() {
            if t == 1 {
                if let Some(s) = a.shape.dims[i].to_usize() {
                    if s > 1 {
                        axes.push(i);
                    }
                }
            }
        }
        if axes.is_empty() {
            Ok(a)
        } else {
            let result = a.sum(axes.clone());
            Ok(keepdim_unsqueeze(result, &axes))
        }
    }

    pub(crate) fn translate_std_op(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_std_or_var(node, true)
    }

    pub(crate) fn translate_var_op(&mut self, node: &Node) -> Result<GraphTensor> {
        self.translate_std_or_var(node, false)
    }

    /// Shared impl for `std` (is_std=true) and `var` (is_std=false).
    fn translate_std_or_var(&mut self, node: &Node, is_std: bool) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let compute = |t: GraphTensor, axes: Vec<usize>, correction: usize| {
            if is_std {
                t.std_options(axes, correction)
            } else {
                t.var_options(axes, correction)
            }
        };
        if node.inputs.len() > 1 && node.inputs[1].arg.as_ints().is_some() {
            let dims = self.get_ints_arg(node, 1)?;
            let ndim = a.shape.len();
            let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
            let correction = self.get_int_arg(node, 2).unwrap_or(1) as usize;
            let keepdim = self.get_bool_arg(node, 3).unwrap_or(false);
            let mut result = compute(a, axes.clone(), correction);
            if keepdim {
                result = keepdim_unsqueeze(result, &axes);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&a)?;
            let mut flat = a;
            flat.shape = ShapeTracker::new(vec![1, total]);
            Ok(compute(flat, vec![1], 1))
        }
    }

    // --- Multi-output reduction ops ---

    pub(crate) fn translate_sort_op(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1).unwrap_or(-1);
        let dim = normalize_dim(dim, a.shape.len());
        let descending = self.get_bool_arg(node, 2).unwrap_or(false);

        let values = a.sort(dim, descending);
        let indices = a.argsort(dim, descending);
        self.store_multi_outputs(node, values, indices);
        Ok(())
    }

    pub(crate) fn translate_max_min_dim(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        let dim = self.get_int_arg(node, 1)?;
        let dim = normalize_dim(dim, a.shape.len());
        let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);

        let is_max = node.target.contains("max");
        let mut values = if is_max { a.max(vec![dim]) } else { a.min(vec![dim]) };
        let mut indices = if is_max { a.argmax(dim) } else { a.argmin(dim) };

        if keepdim {
            values = values.unsqueeze(dim);
            indices = indices.unsqueeze(dim);
        }

        self.store_multi_outputs(node, values, indices);
        Ok(())
    }

    pub(crate) fn translate_std_mean(&mut self, node: &Node) -> Result<()> {
        self.translate_stat_mean(node, true)
    }

    pub(crate) fn translate_var_mean(&mut self, node: &Node) -> Result<()> {
        self.translate_stat_mean(node, false)
    }

    /// Shared impl for `std_mean` (is_std=true) and `var_mean` (is_std=false).
    fn translate_stat_mean(&mut self, node: &Node, is_std: bool) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        let (axes, correction, keepdim) = self.parse_reduction_with_correction(node, &a)?;

        let mean_val = a.mean(axes.clone());
        let stat_val = if is_std {
            a.std_options(axes.clone(), correction)
        } else {
            a.var_options(axes.clone(), correction)
        };

        let (mut stat_out, mut mean_out) = (stat_val, mean_val);
        if keepdim {
            stat_out = keepdim_unsqueeze(stat_out, &axes);
            mean_out = keepdim_unsqueeze(mean_out, &axes);
        }

        self.store_multi_outputs(node, stat_out, mean_out);
        Ok(())
    }

    pub(crate) fn translate_aminmax(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let mut min_val = a.min(vec![dim]);
            let mut max_val = a.max(vec![dim]);
            if keepdim {
                min_val = min_val.unsqueeze(dim);
                max_val = max_val.unsqueeze(dim);
            }
            self.store_multi_outputs(node, min_val, max_val);
        } else {
            let total = concrete_numel(&a)?;
            let mut flat = a;
            flat.shape = ShapeTracker::new(vec![1, total]);
            self.store_multi_outputs(node, flat.min(vec![1]), flat.max(vec![1]));
        }
        Ok(())
    }

    pub(crate) fn translate_kthvalue(&mut self, node: &Node) -> Result<()> {
        let a = self.get_input_tensor(node, 0)?;
        let k = self.get_int_arg(node, 1)? as usize;
        let dim = self.get_int_arg(node, 2).unwrap_or(-1);
        let dim = normalize_dim(dim, a.shape.len());
        let keepdim = self.get_bool_arg(node, 3).unwrap_or(false);

        let sorted = a.sort(dim, false);
        let indices = a.argsort(dim, false);
        let mut value = sorted.slice_along((k - 1)..k, dim);
        let mut index = indices.slice_along((k - 1)..k, dim);
        if !keepdim {
            value = value.squeeze(dim);
            index = index.squeeze(dim);
        }

        self.store_multi_outputs(node, value, index);
        Ok(())
    }

    pub(crate) fn translate_median(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        if node.inputs.len() > 1 && node.inputs[1].arg.as_int().is_some() {
            let dim = self.get_int_arg(node, 1)?;
            let dim = normalize_dim(dim, a.shape.len());
            let keepdim = self.get_bool_arg(node, 2).unwrap_or(false);
            let dim_size = a.dims()[dim]
                .to_usize()
                .ok_or_else(|| anyhow::anyhow!("median requires concrete dim size"))?;
            let mid = dim_size / 2;
            let sorted = a.sort(dim, false);
            let mut result = sorted.slice_along(mid..mid + 1, dim);
            if !keepdim {
                result = result.squeeze(dim);
            }
            Ok(result)
        } else {
            let total = concrete_numel(&a)?;
            let mut flat = a;
            flat.shape = ShapeTracker::new(vec![1, total]);
            let sorted = flat.sort(1, false);
            let mid = total / 2;
            Ok(sorted.slice_along(mid..mid + 1, 1).squeeze(1))
        }
    }

    // --- Helpers ---

    /// Replace NaN values with 0.0.
    fn nan_to_zero(&mut self, a: GraphTensor) -> GraphTensor {
        let is_nan = a.ne(a).cast(DType::F32);
        let one = self.graph.constant_float(1.0).expand_rhs(a.shape);
        a * (one - is_nan)
    }

    /// Parse axes, correction, and keepdim for std_mean/var_mean.
    fn parse_reduction_with_correction(
        &self,
        node: &Node,
        a: &GraphTensor,
    ) -> Result<(Vec<usize>, usize, bool)> {
        let ndim = a.shape.len();
        if node.inputs.len() > 1 && node.inputs[1].arg.as_ints().is_some() {
            let dims = self.get_ints_arg(node, 1)?;
            let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
            let correction = self.get_int_arg(node, 2).unwrap_or(1) as usize;
            let keepdim = self.get_bool_arg(node, 3).unwrap_or(false);
            Ok((axes, correction, keepdim))
        } else {
            let axes: Vec<usize> = (0..ndim).collect();
            let correction = self.get_int_arg(node, 1).unwrap_or(1) as usize;
            Ok((axes, correction, false))
        }
    }

    /// Store two outputs for multi-output ops.
    fn store_multi_outputs(&mut self, node: &Node, first: GraphTensor, second: GraphTensor) {
        if let Some(tensors) = node.outputs[0].as_tensors.as_ref() {
            if tensors.len() >= 2 {
                self.tensors.insert(tensors[0].name.clone(), first);
                self.tensors.insert(tensors[1].name.clone(), second);
            }
        }
    }
}

/// Compute total element count, returning an error if any dimension is symbolic.
pub(crate) fn concrete_numel(a: &GraphTensor) -> Result<usize> {
    a.dims().iter().try_fold(1usize, |acc, d| {
        d.to_usize().map(|v| acc * v).ok_or_else(|| {
            anyhow::anyhow!("Full reduction requires concrete dimensions, got symbolic dim")
        })
    })
}
