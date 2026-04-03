use anyhow::{Result, bail};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;
use super::reduction::concrete_numel;

impl<'a> Translator<'a> {
    pub(crate) fn translate_node(&mut self, node: &Node) -> Result<()> {
        let target = &node.target;
        let output_name = node
            .outputs
            .first()
            .and_then(|o| {
                o.as_tensor.as_ref().map(|t| t.name.clone()).or_else(|| {
                    o.as_tensors
                        .as_ref()
                        .and_then(|ts| ts.first().map(|t| t.name.clone()))
                })
            })
            .unwrap_or_default();

        // No-output ops
        match target.as_str() {
            "torch.ops.aten._assert_tensor_metadata.default"
            | "torch.ops.aten._assert_scalar.default" => return Ok(()),
            "torch.ops.higher_order.wrap_with_set_grad_enabled" => {
                return self.translate_wrap_set_grad(node);
            }
            _ => {}
        }

        let has_tensor_output = node
            .outputs
            .iter()
            .any(|o| o.as_tensor.is_some() || o.as_tensors.is_some());
        if !has_tensor_output {
            return Ok(());
        }

        let result = match target.as_str() {
            // Binary ops
            // Note: rsub/rdiv are not handled here because torch.export decomposes them
            // into sub/div with swapped operands before emission.
            "torch.ops.aten.add.Tensor" => self.translate_binary_op(node, BinaryOp::Add)?,
            "torch.ops.aten.add.Scalar" => self.translate_binary_scalar_op(node, BinaryOp::Add)?,
            "torch.ops.aten.mul.Tensor" => self.translate_binary_op(node, BinaryOp::Mul)?,
            "torch.ops.aten.mul.Scalar" => self.translate_binary_scalar_op(node, BinaryOp::Mul)?,
            "torch.ops.aten.sub.Tensor" => self.translate_binary_op(node, BinaryOp::Sub)?,
            "torch.ops.aten.sub.Scalar" => self.translate_binary_scalar_op(node, BinaryOp::Sub)?,
            "torch.ops.aten.div.Tensor" => self.translate_binary_op(node, BinaryOp::Div)?,
            "torch.ops.aten.div.Scalar" => self.translate_binary_scalar_op(node, BinaryOp::Div)?,

            // Unary ops
            "torch.ops.aten.neg.default" => self.translate_unary_op(node, |a| a * (-1.0))?,
            "torch.ops.aten.exp.default" => self.translate_unary_op(node, |a| a.exp())?,
            "torch.ops.aten.sin.default" => self.translate_unary_op(node, |a| a.sin())?,
            "torch.ops.aten.cos.default" => self.translate_unary_op(node, |a| a.cos())?,
            "torch.ops.aten.sqrt.default" => self.translate_unary_op(node, |a| a.sqrt())?,
            "torch.ops.aten.rsqrt.default" => {
                self.translate_unary_op(node, |a| a.sqrt().reciprocal())?
            }
            "torch.ops.aten.reciprocal.default" => {
                self.translate_unary_op(node, |a| a.reciprocal())?
            }
            "torch.ops.aten.sigmoid.default" => self.translate_unary_op(node, |a| a.sigmoid())?,
            "torch.ops.aten.relu.default" => self.translate_unary_op(node, |a| a.relu())?,
            "torch.ops.aten.silu.default" => self.translate_unary_op(node, |a| a.swish())?,
            "torch.ops.aten.tanh.default" => self.translate_unary_op(node, |a| a.tanh())?,
            "torch.ops.aten.abs.default" => self.translate_unary_op(node, |a| a.abs())?,
            "torch.ops.aten.log.default" => self.translate_unary_op(node, |a| a.log())?,

            // Cast
            "torch.ops.aten._to_copy.default" => self.translate_to_copy(node)?,
            "torch.ops.aten.to.dtype" => self.translate_to_dtype(node)?,
            "torch.ops.aten.to.dtype_layout" => self.translate_to_dtype_layout(node)?,

            // No-op pass-throughs
            "torch.ops.aten.alias.default"
            | "torch.ops.aten.detach_.default"
            | "torch.ops.aten.lift_fresh_copy.default" => self.get_input_tensor(node, 0)?,
            "torch.ops.aten.dropout.default" => self.get_input_tensor(node, 0)?,

            // Shape ops
            "torch.ops.aten.view.default"
            | "torch.ops.aten.reshape.default"
            | "torch.ops.aten._unsafe_view.default" => self.translate_reshape(node)?,
            "torch.ops.aten.permute.default" => self.translate_permute(node)?,
            "torch.ops.aten.transpose.int" => self.translate_transpose(node)?,
            "torch.ops.aten.t.default" => {
                let a = self.get_input_tensor(node, 0)?;
                a.t()
            }
            "torch.ops.aten.unsqueeze.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len() + 1);
                a.unsqueeze(dim)
            }
            "torch.ops.aten.squeeze.dim" | "torch.ops.aten.squeeze.default" => {
                let a = self.get_input_tensor(node, 0)?;
                if node.inputs.len() > 1 {
                    let dim = self.get_int_arg(node, 1)?;
                    let dim = normalize_dim(dim, a.shape.len());
                    a.squeeze(dim)
                } else {
                    let mut result = a;
                    let dims = a.shape.dims;
                    let mut offset = 0;
                    for (i, d) in dims.iter().enumerate() {
                        if d.to_usize() == Some(1) {
                            result = result.squeeze(i - offset);
                            offset += 1;
                        }
                    }
                    result
                }
            }
            "torch.ops.aten.expand.default" => self.translate_expand(node)?,
            "torch.ops.aten.contiguous.default" | "torch.ops.aten.clone.default" => {
                let a = self.get_input_tensor(node, 0)?;
                if !a.shape.is_contiguous() { a + 0.0 } else { a }
            }

            // Matmul
            "torch.ops.aten.mm.default"
            | "torch.ops.aten.bmm.default"
            | "torch.ops.aten.matmul.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                a.matmul(b)
            }

            // Linear
            "torch.ops.aten.linear.default" => self.translate_linear(node)?,

            // Reduction ops
            "torch.ops.aten.sum.dim_IntList" => self.translate_reduction(node, ReductionOp::Sum)?,
            "torch.ops.aten.mean.dim" => self.translate_reduction(node, ReductionOp::Mean)?,
            "torch.ops.aten.amax.default" => self.translate_reduction(node, ReductionOp::Max)?,

            // Slice/index ops
            "torch.ops.aten.slice.Tensor" => self.translate_slice(node)?,
            "torch.ops.aten.select.int" => self.translate_select(node)?,
            "torch.ops.aten.cat.default" => self.translate_cat(node)?,
            "torch.ops.aten.index_select.default" => self.translate_index_select(node)?,
            "torch.ops.aten.index.Tensor" => self.translate_index_tensor(node)?,

            // Embedding
            "torch.ops.aten.embedding.default" => self.translate_embedding(node)?,

            // Softmax
            "torch.ops.aten._softmax.default" | "torch.ops.aten.softmax.int" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len());
                a.softmax(dim)
            }

            // LayerNorm
            "torch.ops.aten.layer_norm.default" => self.translate_layer_norm(node)?,

            // Where
            "torch.ops.aten.where.self" => self.translate_where(node)?,
            "torch.ops.aten.where.ScalarOther" => self.translate_where_scalar_other(node)?,

            // Pow
            "torch.ops.aten.pow.Tensor_Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let exp = self.get_float_arg(node, 1)?;
                a.pow(exp as f32)
            }
            "torch.ops.aten.pow.Tensor_Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                (b * a.log2()).exp2()
            }

            // Creation ops
            "torch.ops.aten.arange.default" | "torch.ops.aten.arange.start" => {
                self.translate_arange(node)?
            }
            "torch.ops.aten.full.default" => self.translate_full(node)?,
            "torch.ops.aten.zeros.default" | "torch.ops.aten.zeros_like.default" => {
                self.translate_zeros(node)?
            }
            "torch.ops.aten.ones.default" | "torch.ops.aten.ones_like.default" => {
                self.translate_ones(node)?
            }
            "torch.ops.aten.new_ones.default" => self.translate_new_ones(node)?,

            // Scalar comparisons
            "torch.ops.aten.gt.Scalar" => self.translate_scalar_comparison(node, |a, s| a.gt(s))?,
            "torch.ops.aten.lt.Scalar" => self.translate_scalar_comparison(node, |a, s| a.lt(s))?,
            "torch.ops.aten.ge.Scalar" => self.translate_scalar_comparison(node, |a, s| a.ge(s))?,
            "torch.ops.aten.le.Scalar" => self.translate_scalar_comparison(node, |a, s| a.le(s))?,

            // Tensor comparisons
            "torch.ops.aten.ne.Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let val = self.get_float_arg(node, 1)? as f32;
                let scalar = self
                    .graph
                    .constant_float(val)
                    .cast(a.dtype)
                    .expand_rhs(a.shape);
                a.ne(scalar)
            }
            "torch.ops.aten.eq.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.eq(b)
            }
            "torch.ops.aten.le.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.le(b)
            }
            "torch.ops.aten.__and__.Tensor" | "torch.ops.aten.logical_and.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                let a = a.cast(DType::F32);
                let b = b.cast(DType::F32);
                (a * b).cast(DType::Bool)
            }
            "torch.ops.aten.logical_or.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                let a = a.cast(DType::F32);
                let b = b.cast(DType::F32);
                (a + b - a * b).cast(DType::Bool)
            }
            "torch.ops.aten.logical_xor.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                let a = a.cast(DType::F32);
                let b = b.cast(DType::F32);
                a.ne(b)
            }

            // Clamp
            "torch.ops.aten.clamp.default" | "torch.ops.aten.clamp_min.default" => {
                self.translate_clamp(node)?
            }

            // Cumsum
            "torch.ops.aten.cumsum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len());
                let a = if a.dtype == DType::Bool {
                    a.cast(DType::Int)
                } else {
                    a
                };
                a.cumsum(dim)
            }

            // Diff
            "torch.ops.aten.diff.default" => self.translate_diff(node)?,

            // Floor / Ceil / Erf (approximations)
            "torch.ops.aten.floor.default" => {
                let a = self.get_input_tensor(node, 0)?;
                // floor(x) = trunc(x) - (x < trunc(x))
                let trunc = a.cast(DType::Int).cast(DType::F32);
                let adjust = a.lt(trunc).cast(DType::F32);
                trunc - adjust
            }
            "torch.ops.aten.ceil.default" => {
                let a = self.get_input_tensor(node, 0)?;
                // ceil(x) = -floor(-x)
                let neg_a = a * (-1.0);
                let trunc = neg_a.cast(DType::Int).cast(DType::F32);
                let adjust = neg_a.lt(trunc).cast(DType::F32);
                let floor_neg = trunc - adjust;
                floor_neg * (-1.0)
            }
            "torch.ops.aten.erf.default" => {
                let a = self.get_input_tensor(node, 0)?;
                // Abramowitz & Stegun approximation 7.1.28 (max error ~1.5e-7)
                // erf(x) = sign(x) * (1 - poly(t) * exp(-x^2))
                // where t = 1/(1 + 0.3275911*|x|), poly in Horner form
                let ax = a.abs();
                let x2 = a * a;
                let t = (ax * 0.3275911_f32 + 1.0).reciprocal();
                // Horner: t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
                let poly = t
                    * (t * (t
                        * (t * (t * 1.061_405_4_f32 + (-1.453_152_1_f32)) + 1.421_413_8_f32)
                        + (-0.284_496_72_f32))
                        + 0.254_829_6_f32);
                let result_abs =
                    self.graph.constant_float(1.0).expand_rhs(a.shape) - poly * (x2 * (-1.0)).exp();
                // sign(x) = 2*(x >= 0) - 1
                let zero = self.graph.constant_float(0.0).expand_rhs(a.shape);
                let sign = a.ge(zero).cast(DType::F32) * 2.0 - 1.0;
                result_abs * sign
            }
            "torch.ops.aten.isnan.default" => {
                let a = self.get_input_tensor(node, 0)?;
                a.ne(a)
            }
            "torch.ops.aten.logical_not.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let one = self.graph.constant_float(1.0).expand_rhs(a.shape);
                (one - a.cast(DType::F32)).cast(DType::Bool)
            }

            // Element-wise min/max (tensor-tensor)
            "torch.ops.aten.maximum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                a.maximum(b)
            }
            "torch.ops.aten.minimum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                a.minimum(b)
            }

            // Tensor comparisons (additional)
            "torch.ops.aten.ge.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.ge(b)
            }
            "torch.ops.aten.lt.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.lt(b)
            }
            "torch.ops.aten.gt.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.gt(b)
            }
            "torch.ops.aten.ne.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.ne(b)
            }

            // Reductions without dim arg (full reduce)
            // Flatten to [1, N] and reduce axis 1 to avoid multi-step HLIR
            // that CUDA can't schedule (grid (0,1,1) invalid launch).
            "torch.ops.aten.sum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let total = concrete_numel(&a)?;
                let mut flat = a;
                flat.shape = ShapeTracker::new(vec![1, total]);
                flat.sum(vec![1])
            }
            "torch.ops.aten.mean.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let total = concrete_numel(&a)?;
                let mut flat = a;
                flat.shape = ShapeTracker::new(vec![1, total]);
                flat.sum(vec![1]) / total as f32
            }
            "torch.ops.aten.max.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let total = concrete_numel(&a)?;
                let mut flat = a;
                flat.shape = ShapeTracker::new(vec![1, total]);
                flat.max(vec![1])
            }
            "torch.ops.aten.min.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let total = concrete_numel(&a)?;
                let mut flat = a;
                flat.shape = ShapeTracker::new(vec![1, total]);
                flat.min(vec![1])
            }
            "torch.ops.aten.amin.default" => self.translate_reduction(node, ReductionOp::Min)?,

            // Single-output reduction ops
            "torch.ops.aten.argmax.default" => self.translate_argmax(node)?,
            "torch.ops.aten.argmin.default" => self.translate_argmin(node)?,
            "torch.ops.aten.prod.default" | "torch.ops.aten.prod.dim_int" => {
                self.translate_prod(node)?
            }
            "torch.ops.aten.argsort.default" | "torch.ops.aten.argsort.stable" => {
                self.translate_argsort_op(node)?
            }
            "torch.ops.aten.log_softmax.int" | "torch.ops.aten._log_softmax.default" => {
                self.translate_log_softmax(node)?
            }
            "torch.ops.aten.all.default" | "torch.ops.aten.all.dim" => {
                self.translate_all(node)?
            }
            "torch.ops.aten.any.default" | "torch.ops.aten.any.dim" => {
                self.translate_any(node)?
            }
            "torch.ops.aten.count_nonzero.default"
            | "torch.ops.aten.count_nonzero.dim_IntList" => self.translate_count_nonzero(node)?,
            "torch.ops.aten.nansum.default" => self.translate_nansum(node)?,
            "torch.ops.aten.nanmean.default" => self.translate_nanmean(node)?,
            "torch.ops.aten.logsumexp.default" => self.translate_logsumexp(node)?,
            "torch.ops.aten.sum_to_size.default" => self.translate_sum_to_size(node)?,
            "torch.ops.aten.std.correction" | "torch.ops.aten.std.default" => {
                self.translate_std_op(node)?
            }
            "torch.ops.aten.var.correction" | "torch.ops.aten.var.default" => {
                self.translate_var_op(node)?
            }
            "torch.ops.aten.median.default" | "torch.ops.aten.median.dim" => {
                self.translate_median(node)?
            }
            "torch.ops.aten.msort.default" => {
                let a = self.get_input_tensor(node, 0)?;
                a.sort(0, false)
            }

            // Multi-output reduction ops
            "torch.ops.aten.sort.default" | "torch.ops.aten.sort.stable" => {
                self.translate_sort_op(node)?;
                return Ok(());
            }
            "torch.ops.aten.max.dim" | "torch.ops.aten.min.dim" => {
                self.translate_max_min_dim(node)?;
                return Ok(());
            }
            "torch.ops.aten.std_mean.correction" | "torch.ops.aten.std_mean.default" => {
                self.translate_std_mean(node)?;
                return Ok(());
            }
            "torch.ops.aten.var_mean.correction" | "torch.ops.aten.var_mean.default" => {
                self.translate_var_mean(node)?;
                return Ok(());
            }
            "torch.ops.aten.aminmax.default" => {
                self.translate_aminmax(node)?;
                return Ok(());
            }
            "torch.ops.aten.kthvalue.default" => {
                self.translate_kthvalue(node)?;
                return Ok(());
            }

            // Gather (axis-aware)
            "torch.ops.aten.gather.default" => self.translate_gather(node)?,

            // Scatter ops
            "torch.ops.aten.scatter.src" => self.translate_scatter_src(node)?,
            "torch.ops.aten.index_put_.default" => self.translate_index_put(node)?,

            // Triangular
            "torch.ops.aten.tril.default" => self.translate_tril(node)?,
            "torch.ops.aten.triu.default" => self.translate_triu(node)?,

            // TopK — handles its own output storage, returns early
            "torch.ops.aten.topk.default" => {
                self.translate_topk(node)?;
                return Ok(());
            }

            // Split
            "torch.ops.aten.split.Tensor"
            | "torch.ops.aten.split_with_sizes.default"
            | "torch.ops.aten.unsafe_split.Tensor"
            | "torch.ops.aten.split_with_sizes_copy.default" => self.translate_split(node)?,

            // Chunk
            "torch.ops.aten.chunk.default" | "torch.ops.aten.unsafe_chunk.default" => {
                self.translate_chunk(node)?;
                return Ok(());
            }

            // Unbind
            "torch.ops.aten.unbind.int" | "torch.ops.aten.unbind_copy.int" => {
                self.translate_unbind(node)?;
                return Ok(());
            }

            // Directional splits
            "torch.ops.aten.dsplit.int" | "torch.ops.aten.dsplit.array" => {
                self.translate_dsplit(node)?;
                return Ok(());
            }
            "torch.ops.aten.hsplit.int" => {
                self.translate_hsplit(node)?;
                return Ok(());
            }
            "torch.ops.aten.vsplit.int" => {
                self.translate_vsplit(node)?;
                return Ok(());
            }

            // One-hot
            "torch.ops.aten.one_hot.default" => self.translate_one_hot(node)?,

            // Fmod
            "torch.ops.aten.fmod.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                a % b
            }
            "torch.ops.aten.fmod.Scalar" | "torch.ops.aten.remainder.Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let val = self.get_float_arg(node, 1)? as f32;
                let b = self.graph.constant_float(val).expand_rhs(a.shape);
                a % b
            }

            // Shape / View / Movement ops
            "torch.ops.aten.flatten.using_ints" => self.translate_flatten(node)?,
            "torch.ops.aten.unflatten.int" => self.translate_unflatten(node)?,
            "torch.ops.aten.ravel.default" => {
                let a = self.get_input_tensor(node, 0)?;
                a.flatten()
            }
            "torch.ops.aten.view_as.default" | "torch.ops.aten.reshape_as.default" => {
                self.translate_reshape_as(node)?
            }
            "torch.ops.aten.movedim.intlist" | "torch.ops.aten.movedim.int" => {
                self.translate_movedim(node)?
            }
            "torch.ops.aten.atleast_1d.default" => {
                let a = self.get_input_tensor(node, 0)?;
                if a.shape.len() == 0 {
                    a.unsqueeze(0)
                } else {
                    a
                }
            }
            "torch.ops.aten.atleast_2d.default" => {
                let a = self.get_input_tensor(node, 0)?;
                match a.shape.len() {
                    0 => a.unsqueeze(0).unsqueeze(0),
                    1 => a.unsqueeze(0),
                    _ => a,
                }
            }
            "torch.ops.aten.atleast_3d.default" => {
                let a = self.get_input_tensor(node, 0)?;
                match a.shape.len() {
                    0 => a.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    1 => a.unsqueeze(0).unsqueeze(2),
                    2 => a.unsqueeze(2),
                    _ => a,
                }
            }
            "torch.ops.aten.narrow.default"
            | "torch.ops.aten.narrow_copy.default"
            | "torch.ops.aten.narrow.Tensor" => self.translate_narrow(node)?,
            "torch.ops.aten.constant_pad_nd.default" => self.translate_constant_pad_nd(node)?,
            "torch.ops.aten.repeat.default" | "torch.ops.aten.tile.default" => {
                self.translate_repeat(node)?
            }
            "torch.ops.aten.fill.Scalar" | "torch.ops.aten.fill.Tensor" => {
                self.translate_fill(node)?
            }
            "torch.ops.aten.broadcast_to.default" => self.translate_broadcast_to(node)?,
            "torch.ops.aten.broadcast_tensors.default" => {
                self.translate_broadcast_tensors(node)?;
                return Ok(());
            }
            "torch.ops.aten.mT.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let n = a.shape.len();
                a.transpose(n - 2, n - 1)
            }

            other => {
                bail!("Unsupported ATen op: {other}");
            }
        };

        if !output_name.is_empty() {
            self.tensors.insert(output_name, result);
        }
        Ok(())
    }
}

impl<'a> Translator<'a> {
    fn translate_scalar_comparison(
        &mut self,
        node: &Node,
        cmp: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let val = self.get_float_arg(node, 1)? as f32;
        let scalar = self
            .graph
            .constant_float(val)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        Ok(cmp(a, scalar))
    }
}
