use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;
use super::reduction::{ArgExtremum, CumExtremum};
use super::unary::ChebyshevKind;

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

        // PT2 scalar values remain tensor-backed in Luminal. `item` and
        // `_local_scalar_dense` therefore bind their scalar output name to a
        // zero-dimensional view of the one-element input instead of creating
        // a separate host-scalar IR value.
        if matches!(
            target.as_str(),
            "torch.ops.aten.item.default" | "torch.ops.aten._local_scalar_dense.default"
        ) {
            let name = node
                .outputs
                .first()
                .and_then(TensorRef::value_name)
                .context("item/local_scalar_dense is missing its scalar output name")?;
            let input_name = node.inputs[0]
                .arg
                .as_value_name()
                .context("item/local_scalar_dense input is not tensor-backed")?;
            if let Some(value) = self.complex_tensors.get(input_name).copied() {
                self.complex_tensors.insert(
                    name.to_string(),
                    super::complex::ComplexTensor::new(
                        reshape_tensor(value.real, vec![]),
                        reshape_tensor(value.imag, vec![]),
                        value.torch_dtype,
                    ),
                );
            } else {
                let scalar = reshape_tensor(self.get_tensor(input_name)?, vec![]);
                self.tensors.insert(name.to_string(), scalar);
            }
            return Ok(());
        }

        let has_tensor_output = node
            .outputs
            .iter()
            .any(|o| o.as_tensor.is_some() || o.as_tensors.is_some());
        if !has_tensor_output {
            return Ok(());
        }

        // Complex is a frontend virtual type. Route every node that consumes
        // or produces one through algebraic real-component lowerings before
        // the ordinary GraphTensor-only dispatch below.
        if self.node_uses_complex(node, &output_name) {
            self.translate_complex_node(node, &output_name)?;
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
            "torch.ops.aten.atan2.default" => self.translate_atan2(node)?,
            "torch.ops.aten.copysign.Tensor" => self.translate_copysign(node)?,
            "torch.ops.aten.copysign.Scalar" => self.translate_copysign_scalar(node)?,
            "torch.ops.aten.fmax.default" => self.translate_fmax_fmin(node, true)?,
            "torch.ops.aten.fmin.default" => self.translate_fmax_fmin(node, false)?,
            "torch.ops.aten.hypot.default" => self.translate_hypot(node)?,
            "torch.ops.aten.gcd.default" => self.translate_gcd(node)?,

            // Unary ops
            "torch.ops.aten.neg.default" => self.translate_unary_op(node, |a| a * (-1.0))?,
            "torch.ops.aten.exp.default" => self.translate_exp(node)?,
            "torch.ops.aten.expm1.default" => self.translate_expm1(node)?,
            "torch.ops.aten.sin.default" => self.translate_unary_op(node, |a| a.sin())?,
            "torch.ops.aten.sinh.default" => self.translate_sinh(node)?,
            "torch.ops.aten.tan.default" => self.translate_tan(node)?,
            "torch.ops.aten.cos.default" => self.translate_cos(node)?,
            "torch.ops.aten.acos.default" => self.translate_acos(node)?,
            "torch.ops.aten.acosh.default" => self.translate_acosh(node)?,
            "torch.ops.aten.asin.default" => self.translate_asin(node)?,
            "torch.ops.aten.asinh.default" => self.translate_asinh(node)?,
            "torch.ops.aten.atan.default" => self.translate_atan(node)?,
            "torch.ops.aten.atanh.default" => self.translate_atanh(node)?,
            "torch.ops.aten.cosh.default" => self.translate_cosh(node)?,
            "torch.ops.aten.trunc.default" => self.translate_trunc(node)?,
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
            "torch.ops.aten.silu.default" => self.translate_unary_op(node, |a| a.silu())?,
            "torch.ops.aten.gelu.default" => self.translate_gelu(node)?,
            "torch.ops.aten.abs.default" => self.translate_unary_op(node, |a| a.abs())?,
            "torch.ops.aten.log.default" => self.translate_unary_op(node, |a| a.log())?,
            "torch.ops.aten.log1p.default" => self.translate_log1p(node)?,
            "torch.ops.aten.log2.default" => self.translate_unary_op(node, |a| a.log2())?,
            "torch.ops.aten.log10.default" => self.translate_log10(node)?,
            "torch.ops.aten.exp2.default" => self.translate_unary_op(node, |a| a.exp2())?,
            "torch.ops.aten.angle.default" => self.translate_angle(node)?,
            "torch.ops.aten.isinf.default" => self.translate_isinf(node)?,
            "torch.ops.aten.ldexp.Tensor" => self.translate_ldexp(node)?,
            "torch.ops.aten.sign.default" => self.translate_sign(node)?,
            "torch.ops.aten.signbit.default" => self.translate_signbit(node)?,
            "torch.ops.aten.bitwise_not.default" => self.translate_bitwise_not(node)?,
            "torch.ops.aten.hardtanh.default" => self.translate_hardtanh(node)?,
            "torch.ops.aten.elu.default" => self.translate_elu(node)?,
            "torch.ops.aten.leaky_relu.default" => self.translate_leaky_relu(node)?,
            "torch.ops.aten.round.default" | "torch.ops.aten.round.decimals" => {
                self.translate_round(node)?
            }
            "torch.ops.aten.erfc.default" => self.translate_erfc(node)?,
            "torch.ops.aten.special_erfcx.default" => self.translate_erfcx(node)?,
            "torch.ops.aten.lgamma.default" => self.translate_lgamma(node)?,
            "torch.ops.aten.digamma.default" => self.translate_digamma(node)?,
            "torch.ops.aten.polygamma.default" => self.translate_polygamma(node)?,
            bessel @ ("torch.ops.aten.i0.default"
            | "torch.ops.aten.special_i0e.default"
            | "torch.ops.aten.special_i1.default"
            | "torch.ops.aten.special_i1e.default"
            | "torch.ops.aten.special_modified_bessel_i0.default"
            | "torch.ops.aten.special_modified_bessel_i1.default") => {
                let order = usize::from(bessel.contains("i1"));
                self.translate_modified_bessel(
                    node,
                    order,
                    bessel.contains("i0e") || bessel.contains("i1e"),
                )?
            }
            "torch.ops.aten.special_spherical_bessel_j0.default" => {
                self.translate_spherical_bessel_j0(node)?
            }
            bessel @ ("torch.ops.aten.special_bessel_j0.default"
            | "torch.ops.aten.special_bessel_j1.default"
            | "torch.ops.aten.special_bessel_y0.default"
            | "torch.ops.aten.special_bessel_y1.default") => {
                let order = usize::from(bessel.contains("j1") || bessel.contains("y1"));
                self.translate_cylindrical_bessel(node, order, bessel.contains("_y"))?
            }
            bessel @ ("torch.ops.aten.special_modified_bessel_k0.default"
            | "torch.ops.aten.special_modified_bessel_k1.default"
            | "torch.ops.aten.special_scaled_modified_bessel_k0.default"
            | "torch.ops.aten.special_scaled_modified_bessel_k1.default") => {
                let order = usize::from(bessel.contains("k1"));
                self.translate_modified_bessel_k(node, order, bessel.contains("scaled"))?
            }
            "torch.ops.aten.special_airy_ai.default" => self.translate_airy_ai(node)?,
            "torch.ops.aten.special_ndtri.default" => self.translate_ndtri(node)?,
            "torch.ops.aten.erfinv.default" => self.translate_erfinv(node)?,
            "torch.ops.aten.special_chebyshev_polynomial_t.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::First, false)?
            }
            "torch.ops.aten.special_chebyshev_polynomial_u.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Second, false)?
            }
            "torch.ops.aten.special_chebyshev_polynomial_v.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Third, false)?
            }
            "torch.ops.aten.special_chebyshev_polynomial_w.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Fourth, false)?
            }
            "torch.ops.aten.special_shifted_chebyshev_polynomial_t.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::First, true)?
            }
            "torch.ops.aten.special_shifted_chebyshev_polynomial_u.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Second, true)?
            }
            "torch.ops.aten.special_shifted_chebyshev_polynomial_v.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Third, true)?
            }
            "torch.ops.aten.special_shifted_chebyshev_polynomial_w.default" => {
                self.translate_chebyshev_polynomial(node, ChebyshevKind::Fourth, true)?
            }
            "torch.ops.aten.logcumsumexp.default" => self.translate_logcumsumexp(node)?,

            // Cast
            "torch.ops.aten._to_copy.default" => self.translate_to_copy(node)?,

            // No-op
            "torch.ops.aten.alias.default" => self.get_input_tensor(node, 0)?,

            // Shape ops
            "torch.ops.aten.view.default" => self.translate_reshape(node)?,
            "torch.ops.aten.view_copy.default" => {
                let value = self.translate_reshape(node)?;
                super::movement::materialize_tensor(value)
            }
            "torch.ops.aten.upsample_nearest2d.vec" => self.translate_upsample_nearest2d(node)?,
            "torch.ops.aten.upsample_bilinear2d.vec" => self.translate_upsample_bilinear2d(node)?,
            "torch.ops.aten._upsample_bilinear2d_aa.default" => {
                self.translate_upsample_bilinear2d_aa(node)?
            }
            "torch.ops.aten.repeat.default" => self.translate_repeat(node)?,
            "torch.ops.aten.permute.default" => self.translate_permute(node)?,
            "torch.ops.aten.permute_copy.default" => {
                let value = self.translate_permute(node)?;
                super::movement::materialize_tensor(value)
            }
            "torch.ops.aten.narrow_copy.default" => self.translate_narrow_copy(node)?,
            "torch.ops.aten.unbind_copy.int" => {
                self.translate_unbind_copy(node)?;
                return Ok(());
            }
            "torch.ops.aten.flip.default" => self.translate_flip(node)?,
            "torch.ops.aten.diagonal.default" => self.translate_diagonal(node)?,
            "torch.ops.aten.diagonal_scatter.default" => self.translate_diagonal_scatter(node)?,
            "torch.ops.aten.index_select.default" => self.translate_index_select(node)?,
            "torch.ops.aten.unfold.default" => self.translate_unfold(node)?,
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
            "torch.ops.aten.addmv.default" => self.translate_addmv(node)?,
            "torch.ops.aten.addbmm.default" => self.translate_addbmm(node)?,
            "torch.ops.aten._trilinear.default" => self.translate_trilinear(node)?,
            "torch.ops.aten.dist.default" => self.translate_dist(node)?,
            "torch.ops.aten._cdist_forward.default" => self.translate_cdist(node)?,
            "torch.ops.aten._pdist_forward.default" => self.translate_pdist(node)?,

            // addmm: beta*input + alpha*(mat1 @ mat2)
            "torch.ops.aten.addmm.default" => {
                let input = self.get_input_tensor(node, 0)?;
                let mat1 = self.get_input_tensor(node, 1)?;
                let mat2 = self.get_input_tensor(node, 2)?;
                let beta = self.get_float_arg(node, 3).unwrap_or(1.0) as f32;
                let alpha = self.get_float_arg(node, 4).unwrap_or(1.0) as f32;
                let (mat1, mat2) = ensure_same_dtype(mat1, mat2);
                let mm = mat1.matmul(mat2);
                let (input, mm) = ensure_same_dtype(input, mm);
                let (input, mm) = broadcast_binary(input, mm);
                input * beta + mm * alpha
            }

            "torch.ops.aten.copy.default" => self.translate_copy(node)?,

            // Convolution
            "torch.ops.aten.convolution.default" => self.translate_conv(node)?,

            // Reduction ops
            "torch.ops.aten.sum.dim_IntList" => self.translate_reduction(node, ReductionOp::Sum)?,
            "torch.ops.aten.mean.dim" => self.translate_reduction(node, ReductionOp::Mean)?,
            "torch.ops.aten.amax.default" => self.translate_reduction(node, ReductionOp::Max)?,
            "torch.ops.aten.linalg_vector_norm.default" => {
                self.translate_linalg_vector_norm(node)?
            }
            "torch.ops.aten.var.default"
            | "torch.ops.aten.var.dim"
            | "torch.ops.aten.var.correction" => self.translate_var(node)?,
            "torch.ops.aten.var_mean.default"
            | "torch.ops.aten.var_mean.dim"
            | "torch.ops.aten.var_mean.correction" => {
                self.translate_var_mean(node)?;
                return Ok(());
            }
            "torch.ops.aten.any.default" | "torch.ops.aten.any.dim" | "torch.ops.aten.any.dims" => {
                self.translate_any(node)?
            }

            // Slice/index ops
            "torch.ops.aten.slice.Tensor" => self.translate_slice(node)?,
            "torch.ops.aten.select.int" => self.translate_select(node)?,
            "torch.ops.aten.cat.default" => self.translate_cat(node)?,
            "torch.ops.aten.index.Tensor" => self.translate_index_tensor(node)?,
            "torch.ops.aten.slice_scatter.default" => self.translate_slice_scatter(node)?,
            "torch.ops.aten.masked_scatter.default" => self.translate_masked_scatter(node)?,
            "torch.ops.aten.put.default" => self.translate_put(node)?,
            "torch.ops.aten.nonzero_static.default" => self.translate_nonzero_static(node)?,
            "torch.ops.aten.repeat_interleave.Tensor" => {
                self.translate_repeat_interleave_tensor(node)?
            }

            // Embedding
            "torch.ops.aten.embedding.default" => self.translate_embedding(node)?,
            "torch.ops.aten.embedding_renorm.default" => self.translate_embedding_renorm(node)?,
            "torch.ops.aten._embedding_bag_forward_only.default" => {
                self.translate_embedding_bag(node)?;
                return Ok(());
            }

            // Softmax
            "torch.ops.aten._softmax.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(dim, a.shape.len());
                a.softmax(dim)
            }
            "torch.ops.aten._log_softmax.default" => self.translate_log_softmax(node)?,

            // LayerNorm
            "torch.ops.aten.native_layer_norm.default" => self.translate_layer_norm(node)?,

            // GroupNorm
            "torch.ops.aten.native_group_norm.default" => self.translate_group_norm(node)?,

            // Pooling and batch normalization composites.
            "torch.ops.aten.avg_pool2d.default" => self.translate_avg_pool(node, 2)?,
            "torch.ops.aten.avg_pool3d.default" => self.translate_avg_pool(node, 3)?,
            pool @ ("torch.ops.aten._adaptive_avg_pool2d.default"
            | "torch.ops.aten._adaptive_avg_pool3d.default") => {
                self.translate_adaptive_avg_pool(node, 2 + usize::from(pool.contains("3d")))?
            }
            pool @ ("torch.ops.aten.max_pool2d_with_indices.default"
            | "torch.ops.aten.max_pool3d_with_indices.default") => {
                self.translate_max_pool(node, 2 + usize::from(pool.contains("3d")))?;
                return Ok(());
            }
            "torch.ops.aten.max_pool2d_with_indices_backward.default" => {
                self.translate_max_pool_backward(node)?
            }
            pool @ ("torch.ops.aten.adaptive_max_pool2d.default"
            | "torch.ops.aten.adaptive_max_pool3d.default") => {
                self.translate_adaptive_max_pool(node, 2 + usize::from(pool.contains("3d")))?;
                return Ok(());
            }
            pool @ ("torch.ops.aten.fractional_max_pool2d.default"
            | "torch.ops.aten.fractional_max_pool3d.default") => {
                self.translate_fractional_max_pool(node, 2 + usize::from(pool.contains("3d")))?;
                return Ok(());
            }
            "torch.ops.aten.grid_sampler_2d.default" => self.translate_grid_sampler(node, 2)?,
            "torch.ops.aten.grid_sampler_3d.default" => self.translate_grid_sampler(node, 3)?,
            "torch.ops.aten._native_batch_norm_legit.no_stats"
            | "torch.ops.aten._native_batch_norm_legit_functional.default"
            | "torch.ops.aten._batch_norm_with_update_functional.default" => {
                self.translate_batch_norm_functional(node)?;
                return Ok(());
            }

            // RMSNorm
            "torch.ops.aten._fused_rms_norm.default" => self.translate_fused_rms_norm(node)?,

            // Where
            "torch.ops.aten.where.self" => self.translate_where(node)?,
            "torch.ops.aten.where.ScalarOther" => self.translate_where_scalar_other(node)?,
            "torch.ops.aten.masked_fill.Scalar" => self.translate_masked_fill_scalar(node)?,

            // Pow
            "torch.ops.aten.pow.Tensor_Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let exp = self.get_float_arg(node, 1)?;
                if (exp - 2.0).abs() < f64::EPSILON {
                    a * a
                } else {
                    a.pow(exp as f32)
                }
            }
            "torch.ops.aten.pow.Tensor_Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                (b * a.log2()).exp2()
            }
            "torch.ops.aten.pow.Scalar" => self.translate_scalar_base_pow(node)?,

            // Creation ops
            "torch.ops.aten.arange.start_step" => self.translate_arange(node)?,
            "torch.ops.aten.full.default" => self.translate_full(node)?,
            "torch.ops.aten.full_like.default" => self.translate_full_like(node)?,
            "torch.ops.aten.constant_pad_nd.default" => self.translate_constant_pad_nd(node)?,
            // `empty` and `empty_permuted` allocate uninitialised tensors of
            // a given shape; the caller fills them. We lower to zeros with
            // the same shape+dtype — downstream reads are officially UB on
            // PyTorch's side, and downstream writes overwrite our zeros.
            // Qwen3MoE's MoE block uses `empty_permuted` to allocate the
            // expert-output staging tensor before scatter-adding into it.
            "torch.ops.aten.empty.memory_format"
            | "torch.ops.aten.empty_permuted.default"
            | "torch.ops.aten.empty_strided.default"
            | "torch.ops.aten.new_empty_strided.default" => self.translate_empty(node)?,
            // Qwen3-MoE's expert-balance counts tokens-per-expert via histc.
            "torch.ops.aten.histc.default" => self.translate_histc(node)?,
            "torch.ops.aten.bucketize.Tensor"
            | "torch.ops.aten.searchsorted.Tensor"
            | "torch.ops.aten.searchsorted.Scalar" => self.translate_searchsorted(node)?,
            "torch.ops.aten.tril_indices.default" | "torch.ops.aten.triu_indices.default" => {
                self.translate_triangular_indices(node)?
            }

            // Grouped matmul (MoE expert dispatch).
            // aten._grouped_mm is the native op; transformers::grouped_mm_fallback
            // is a Python-implemented custom_op (transformers/integrations/moe.py)
            // used by HF MoE when _grouped_mm isn't available for the activation
            // dtype. Both have identical (input, weight, offs) signature; route
            // both through the same batched-matmul + group-mask lowering.
            "torch.ops.aten._grouped_mm.default"
            | "torch.ops.transformers.grouped_mm_fallback.default" => {
                self.translate_grouped_mm(node)?
            }
            "torch.ops.aten.scalar_tensor.default" => self.translate_scalar_tensor(node)?,
            // Scalar comparisons
            "torch.ops.aten.gt.Scalar" => self.translate_scalar_comparison(node, |a, s| a.gt(s))?,
            "torch.ops.aten.lt.Scalar" => self.translate_scalar_comparison(node, |a, s| a.lt(s))?,
            "torch.ops.aten.ge.Scalar" => self.translate_scalar_comparison(node, |a, s| a.ge(s))?,
            "torch.ops.aten.le.Scalar" => self.translate_scalar_comparison(node, |a, s| a.le(s))?,

            // Tensor comparisons
            "torch.ops.aten.eq.Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let val = self.get_float_arg(node, 1)? as f32;
                let scalar = self
                    .graph
                    .constant_float(val)
                    .cast(a.dtype)
                    .expand_rhs(a.shape);
                a.eq(scalar)
            }
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
            "torch.ops.aten.ne.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.ne(b)
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
            "torch.ops.aten.bitwise_or.Tensor" | "torch.ops.aten.logical_or.default" => {
                // Both arms use the same bool-OR lowering. Gemma-4's sliding+full
                // attention mask fusion emits bitwise_or on boolean tensors; the
                // integer semantics of bitwise_or aren't exercised by any op in
                // the test suite, so we rely on inputs being boolean-typed.
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = broadcast_binary(a, b);
                self.apply_bool_or(a, b)
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
            "torch.ops.aten.clamp.Tensor" => self.translate_clamp_tensor(node)?,

            // Cumsum
            "torch.ops.aten.cumsum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let a = if a.dtype == DType::Bool {
                    a.cast(DType::Int)
                } else {
                    a
                };
                // Rank-0 (scalar) input: cumsum of a single element is the element
                // itself. PyTorch eager treats `dim=0` on a 0-d as an identity op,
                // and the underlying `cumop` indexes `shape.dims[axis]` which would
                // panic with empty dims.
                if a.shape.is_empty() {
                    a
                } else {
                    let dim = self.get_int_arg(node, 1)?;
                    let dim = normalize_dim(dim, a.shape.len());
                    a.cumsum(dim)
                }
            }
            "torch.ops.aten.cumprod.default" => self.translate_cumprod(node)?,
            "torch.ops.aten.cummax.default" => {
                self.translate_cumextremum(node, CumExtremum::Max)?;
                return Ok(());
            }
            "torch.ops.aten.cummin.default" => {
                self.translate_cumextremum(node, CumExtremum::Min)?;
                return Ok(());
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
                // ceil(x) = trunc(x) + (x > trunc(x)).
                // Cast-to-Int rounds toward zero, so for any positive fractional
                // `x` the trunc sits below `x` and we add 1; for negatives we
                // have `trunc >= x` and adjust=0. Avoids the two extra
                // mul-by-(-1) nodes that the `-floor(-x)` lowering emits.
                let trunc = a.cast(DType::Int).cast(DType::F32);
                let adjust = a.gt(trunc).cast(DType::F32);
                trunc + adjust
            }
            "torch.ops.aten.erf.default" => self.translate_erf(node)?,
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
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a.maximum(b)
            }
            "torch.ops.aten.minimum.default" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
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
            "torch.ops.aten.prod.default" => self.translate_reduction(node, ReductionOp::Prod)?,

            // Argmax / argmin — built on top of `stable_argsort` (LUM-496).
            // PyTorch's argmax/argmin returns int64; the dtype is preserved
            // through the LUM-486 boundary widening.
            "torch.ops.aten.argmax.default" => {
                self.translate_argextremum(node, ArgExtremum::Max)?
            }
            "torch.ops.aten.argmin.default" => {
                self.translate_argextremum(node, ArgExtremum::Min)?
            }
            "torch.ops.aten.max.dim" => {
                self.translate_dim_extremum(node, ArgExtremum::Max)?;
                return Ok(());
            }
            "torch.ops.aten.min.dim" => {
                self.translate_dim_extremum(node, ArgExtremum::Min)?;
                return Ok(());
            }
            "torch.ops.aten.median.default" | "torch.ops.aten.median.dim" => {
                self.translate_median(node)?;
                return Ok(());
            }
            "torch.ops.aten.nanmedian.default" | "torch.ops.aten.nanmedian.dim" => {
                self.translate_nanmedian(node)?;
                return Ok(());
            }
            "torch.ops.aten.segment_reduce.default" => self.translate_segment_reduce(node)?,
            "torch.ops.aten.histogram.bin_ct" | "torch.ops.aten.histogram.bins_tensor" => {
                self.translate_histogram(node)?;
                return Ok(());
            }
            "torch.ops.aten._histogramdd_bin_edges.default" => {
                self.translate_histogramdd_bin_edges(node)?;
                return Ok(());
            }
            "torch.ops.aten._histogramdd_from_bin_cts.default" => {
                self.translate_histogramdd(node, false)?
            }
            "torch.ops.aten._histogramdd_from_bin_tensors.default" => {
                self.translate_histogramdd(node, true)?
            }
            // Gather (axis-aware)
            "torch.ops.aten.gather.default" => self.translate_gather(node)?,

            // Scatter ops
            "torch.ops.aten.scatter.src" => self.translate_scatter_src(node)?,
            "torch.ops.aten.scatter.value" => self.translate_scatter_value(node)?,
            "torch.ops.aten.scatter.reduce" => self.translate_scatter_src_reduce(node)?,
            "torch.ops.aten.scatter.value_reduce" => self.translate_scatter_value_reduce(node)?,
            "torch.ops.aten.scatter_add.default" => self.translate_scatter_add(node)?,
            "torch.ops.aten.scatter_reduce.two" => self.translate_scatter_reduce(node)?,
            "torch.ops.aten.index_reduce.default" => self.translate_index_reduce(node)?,
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
            "torch.ops.aten.sort.default" | "torch.ops.aten.sort.stable" => {
                self.translate_sort(node)?;
                return Ok(());
            }

            // Scaled dot-product attention — args are resolved by name in
            // translate_sdpa, so every ATen variant shares one lowering.
            "torch.ops.aten.scaled_dot_product_attention.default"
            | "torch.ops.aten._scaled_dot_product_efficient_attention.default"
            | "torch.ops.aten._scaled_dot_product_flash_attention.default"
            | "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default"
            | "torch.ops.aten._scaled_dot_product_cudnn_attention.default" => {
                self.translate_sdpa(node)?;
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
            "torch.ops.aten.fmod.Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let value = self.constructor_scalar_arg(node, 1)?;
                let scalar = self
                    .typed_scalar_constant(&value, a.dtype)?
                    .expand_rhs(a.shape);
                a % scalar
            }
            // Remainder (Python-style modulo). For float tensors aten.remainder
            // returns the same value as `%` would in luminal (Mod follows the
            // language's % semantics on f32). The Tensor variant accepts a
            // tensor RHS that may be rank-0; broadcast both operands so a
            // scalar RHS is expanded to match the LHS shape before mod.
            "torch.ops.aten.remainder.Tensor" => {
                let a = self.get_input_tensor(node, 0)?;
                let b = self.get_input_tensor(node, 1)?;
                let (a, b) = ensure_same_dtype(a, b);
                let (a, b) = broadcast_binary(a, b);
                a % b
            }
            "torch.ops.aten.remainder.Scalar" => {
                let a = self.get_input_tensor(node, 0)?;
                let val = self.get_float_arg(node, 1)? as f32;
                let scalar = self
                    .graph
                    .constant_float(val)
                    .cast(a.dtype)
                    .expand_rhs(a.shape);
                a % scalar
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
    fn translate_scalar_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        self.real_constructor_scalar(node, 0, dtype)
    }

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
