use anyhow::{Result, bail};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

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
            "torch.ops.aten.div.Tensor_mode" => self.translate_div_tensor_mode(node)?,

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
            "torch.ops.aten.tanh.default" => self.translate_unary_op(node, |a| a.tanh())?,
            "torch.ops.aten.abs.default" => self.translate_unary_op(node, |a| a.abs())?,
            "torch.ops.aten.log.default" => self.translate_unary_op(node, |a| a.log())?,
            "torch.ops.aten.log2.default" => self.translate_unary_op(node, |a| a.log2())?,
            "torch.ops.aten.exp2.default" => self.translate_unary_op(node, |a| a.exp2())?,
            "torch.ops.aten.sign.default" => self.translate_sign(node)?,
            "torch.ops.aten.bitwise_not.default" => self.translate_bitwise_not(node)?,

            // Cast
            "torch.ops.aten._to_copy.default" => self.translate_to_copy(node)?,

            // No-op
            "torch.ops.aten.alias.default" => self.get_input_tensor(node, 0)?,

            // Shape ops
            "torch.ops.aten.view.default" => self.translate_reshape(node)?,
            "torch.ops.aten.permute.default" => self.translate_permute(node)?,
            "torch.ops.aten.unsqueeze.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len() + 1);
                a.unsqueeze(dim)
            }
            "torch.ops.aten.squeeze.dims" => {
                let a = self.get_input_tensor(node, 0)?;
                let dims = self.get_ints_arg(node, 1)?;
                let ndim = a.shape.len();
                let mut sorted_dims: Vec<usize> =
                    dims.iter().map(|&d| normalize_dim(d, ndim)).collect();
                sorted_dims.sort();
                let mut result = a;
                let mut offset = 0;
                for d in sorted_dims {
                    if result.shape.dims[d - offset].to_usize() == Some(1) {
                        result = result.squeeze(d - offset);
                        offset += 1;
                    }
                }
                result
            }
            "torch.ops.aten.expand.default" => self.translate_expand(node)?,
            "torch.ops.aten.clone.default" => {
                let a = self.get_input_tensor(node, 0)?;
                if !a.shape.is_contiguous() { a + 0.0 } else { a }
            }
            "torch.ops.aten.argsort.default" => self.translate_argsort(node)?,

            // Matmul
            "torch.ops.aten.mm.default" | "torch.ops.aten.bmm.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                a.matmul(b)
            }

            // addmm: beta*input + alpha*(mat1 @ mat2)
            "torch.ops.aten.addmm.default" => {
                let input = self.get_input_tensor(node, 0)?;
                let mat1 = self.get_input_tensor(node, 1)?;
                let mat2 = self.get_input_tensor(node, 2)?;
                let beta = self.get_float_arg(node, 3).unwrap_or(1.0) as f32;
                let alpha = self.get_float_arg(node, 4).unwrap_or(1.0) as f32;
                let mm = mat1.matmul(mat2);
                let (input, mm) = broadcast_binary(input, mm);
                input * beta + mm * alpha
            }

            // Convolution
            "torch.ops.aten.convolution.default" => self.translate_conv(node)?,

            // Reduction ops
            "torch.ops.aten.sum.dim_IntList" => self.translate_reduction(node, ReductionOp::Sum)?,
            "torch.ops.aten.mean.dim" => self.translate_reduction(node, ReductionOp::Mean)?,
            "torch.ops.aten.amax.default" => self.translate_reduction(node, ReductionOp::Max)?,

            // Slice/index ops
            "torch.ops.aten.slice.Tensor" => self.translate_slice(node)?,
            "torch.ops.aten.cat.default" => self.translate_cat(node)?,
            "torch.ops.aten.index.Tensor" => self.translate_index_tensor(node)?,

            // Embedding
            "torch.ops.aten.embedding.default" => self.translate_embedding(node)?,

            // Softmax
            "torch.ops.aten._softmax.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len());
                a.softmax(dim)
            }

            // LayerNorm
            "torch.ops.aten.native_layer_norm.default" => self.translate_layer_norm(node)?,

            // Where
            "torch.ops.aten.where.self" => self.translate_where(node)?,
            "torch.ops.aten.where.ScalarOther" => self.translate_where_scalar_other(node)?,
            "torch.ops.aten.masked_fill.Scalar" => self.translate_masked_fill_scalar(node)?,

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
            "torch.ops.aten.arange.start_step" => self.translate_arange(node)?,
            "torch.ops.aten.full.default" => self.translate_full(node)?,
            "torch.ops.aten.full_like.default" => self.translate_full_like(node)?,
            "torch.ops.aten.scalar_tensor.default" => {
                let val = self.get_float_arg(node, 0)? as f32;
                self.graph.constant_float(val)
            }
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
            "torch.ops.aten.bitwise_and.Tensor" | "torch.ops.aten.logical_and.default" => {
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
            "torch.ops.aten.clamp.default" => self.translate_clamp(node)?,

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

            // Full-reduce variants (no dim arg) — handled by translate_reduction fallback
            "torch.ops.aten.sum.default" => self.translate_reduction(node, ReductionOp::Sum)?,
            "torch.ops.aten.mean.default" => self.translate_reduction(node, ReductionOp::Mean)?,
            "torch.ops.aten.max.default" => self.translate_reduction(node, ReductionOp::Max)?,
            "torch.ops.aten.min.default" => self.translate_reduction(node, ReductionOp::Min)?,
            "torch.ops.aten.amin.default" => self.translate_reduction(node, ReductionOp::Min)?,

            // Gather (axis-aware)
            "torch.ops.aten.gather.default" => self.translate_gather(node)?,

            // Scatter ops
            "torch.ops.aten.scatter.src" => self.translate_scatter_src(node)?,
            "torch.ops.aten.scatter.value" => self.translate_scatter_value(node)?,
            "torch.ops.aten.index_put_.default" | "torch.ops.aten.index_put.default" => {
                self.translate_index_put(node)?
            }

            // Integer routing math
            "torch.ops.aten.floor_divide.default" => self.translate_floor_divide(node)?,

            // Triangular
            "torch.ops.aten.tril.default" => self.translate_tril(node)?,
            "torch.ops.aten.triu.default" => self.translate_triu(node)?,

            // TopK — handles its own output storage, returns early
            "torch.ops.aten.topk.default" => {
                self.translate_topk(node)?;
                return Ok(());
            }

            // Sort — handles its own output storage, returns early
            "torch.ops.aten.sort.default" => {
                self.translate_sort(node)?;
                return Ok(());
            }

            // Split
            "torch.ops.aten.split_with_sizes.default" => self.translate_split_with_sizes(node)?,

            // Fmod
            "torch.ops.aten.fmod.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                a % b
            }
            // Prod reduction
            "torch.ops.aten.prod.dim_int" => self.translate_reduction(node, ReductionOp::Prod)?,

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
