//! Frontend-only complex tensor support.
//!
//! HLIR intentionally has no complex dtype. A complex PT2 value is carried as
//! two ordinary real `GraphTensor`s, and every supported complex ATen op is
//! lowered algebraically before it reaches HLIR. PyTorch's interleaved complex
//! storage is preserved only at graph inputs and outputs.

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use crate::dim_arith::product_of_dims;
use crate::pt2_schema::{Argument, Node, OptionalTensorEntry};
use crate::pt2_util::{
    BinaryOp, ReductionOp, broadcast_binary, normalize_dim, normalize_slice_bound, reshape_tensor,
};
use crate::torch_dtype::TorchDType;

use super::Translator;
use super::movement::{
    diagonal_indices, diagonal_scatter_tensor, flip_indices, index_select_tensor,
    masked_scatter_tensor, materialize_tensor, narrow_copy_tensor, nonzero_static_from_truth,
    normalize_diagonal_dims, normalize_flip_dims, put_tensor, slice_scatter_tensor, unfold_tensor,
};
use super::movement_dynamic::ScatterReduction;
use super::tensor::{ConstructorScalar, copy_tensor};

#[derive(Clone, Copy, Debug)]
pub(crate) struct ComplexTensor {
    pub(crate) real: GraphTensor,
    pub(crate) imag: GraphTensor,
    pub(crate) torch_dtype: TorchDType,
}

impl ComplexTensor {
    pub(crate) fn new(real: GraphTensor, imag: GraphTensor, torch_dtype: TorchDType) -> Self {
        debug_assert_eq!(Some(real.dtype), torch_dtype.complex_component_dtype());
        debug_assert_eq!(real.dims(), imag.dims());
        Self {
            real,
            imag,
            torch_dtype,
        }
    }

    pub(crate) fn from_interleaved(
        graph: &mut Graph,
        backing: GraphTensor,
        torch_dtype: TorchDType,
    ) -> Result<Self> {
        let axis = backing
            .shape
            .len()
            .checked_sub(1)
            .context("complex interleaved storage must have a component dimension")?;
        anyhow::ensure!(
            backing.shape.dims[axis].to_usize() == Some(2),
            "complex interleaved storage must end in dimension 2, got {:?}",
            backing.dims()
        );
        let mut shape = backing.dims();
        shape.pop();
        let real = backing.gather(graph.iota(Expression::from('z') * 2, shape.clone()));
        let imag = backing.gather(graph.iota(Expression::from('z') * 2 + 1, shape));
        Ok(Self::new(real, imag, torch_dtype))
    }

    pub(crate) fn pack(self, graph: &mut Graph) -> GraphTensor {
        interleave(graph, self.real, self.imag)
    }

    fn cast(self, torch_dtype: TorchDType) -> Self {
        let dtype = torch_dtype
            .complex_component_dtype()
            .expect("complex cast target must have a real component dtype");
        Self::new(self.real.cast(dtype), self.imag.cast(dtype), torch_dtype)
    }

    fn map(self, mut f: impl FnMut(GraphTensor) -> GraphTensor) -> Self {
        Self {
            real: f(self.real),
            imag: f(self.imag),
            ..self
        }
    }

    fn try_map(self, mut f: impl FnMut(GraphTensor) -> Result<GraphTensor>) -> Result<Self> {
        Ok(Self {
            real: f(self.real)?,
            imag: f(self.imag)?,
            ..self
        })
    }
}

/// Store two equally-shaped tensors in a contiguous final dimension. Scatter
/// is structural, so inactive lanes cannot contaminate infinities with NaNs.
fn interleave(graph: &mut Graph, first: GraphTensor, second: GraphTensor) -> GraphTensor {
    let shape = first.dims();
    let mut packed_shape = shape.clone();
    packed_shape.push(2usize.into());
    let even = graph.iota(Expression::from('z') * 2, shape.clone());
    let odd = graph.iota(Expression::from('z') * 2 + 1, shape);
    let zero = graph.iota(0, packed_shape).cast(first.dtype);
    second.scatter(odd, first.scatter(even, zero))
}

fn squeeze_dims(mut tensor: GraphTensor, dims: &[usize]) -> GraphTensor {
    let mut removed = 0;
    for &original_dim in dims {
        let dim = original_dim - removed;
        if tensor.shape.dims[dim].to_usize() == Some(1) {
            tensor = tensor.squeeze(dim);
            removed += 1;
        }
    }
    tensor
}

fn float_max(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => 65_504.0,
        DType::Bf16 => 3.389_531_389_251_535_5e38,
        DType::F32 => f32::MAX as f64,
        DType::F64 => f64::MAX,
        _ => unreachable!("complex component has non-float dtype {dtype:?}"),
    }
}

impl<'a> Translator<'a> {
    pub(crate) fn node_uses_complex(&self, node: &Node, output_name: &str) -> bool {
        if self
            .tensor_meta(output_name)
            .and_then(|m| TorchDType::from_code(m.dtype).ok())
            .is_some_and(TorchDType::is_complex)
        {
            return true;
        }
        node.inputs.iter().any(|input| {
            input
                .arg
                .as_value_name()
                .is_some_and(|name| self.complex_tensors.contains_key(name))
                || input.arg.as_tensors().is_some_and(|names| {
                    names
                        .iter()
                        .any(|name| self.complex_tensors.contains_key(&name.name))
                })
        })
    }

    pub(crate) fn translate_complex_node(&mut self, node: &Node, output_name: &str) -> Result<()> {
        let target = node.target.as_str();

        match target {
            "torch.ops.aten.add.Tensor"
            | "torch.ops.aten.sub.Tensor"
            | "torch.ops.aten.mul.Tensor"
            | "torch.ops.aten.div.Tensor" => {
                let op = match target {
                    "torch.ops.aten.add.Tensor" => BinaryOp::Add,
                    "torch.ops.aten.sub.Tensor" => BinaryOp::Sub,
                    "torch.ops.aten.mul.Tensor" => BinaryOp::Mul,
                    _ => BinaryOp::Div,
                };
                let value = if node.inputs[1].arg.as_value_name().is_some() {
                    self.translate_complex_binary(node, op, output_name)?
                } else {
                    self.translate_complex_scalar_binary(node, op, output_name)?
                };
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.add.Scalar"
            | "torch.ops.aten.sub.Scalar"
            | "torch.ops.aten.mul.Scalar"
            | "torch.ops.aten.div.Scalar" => {
                let op = match target {
                    "torch.ops.aten.add.Scalar" => BinaryOp::Add,
                    "torch.ops.aten.sub.Scalar" => BinaryOp::Sub,
                    "torch.ops.aten.mul.Scalar" => BinaryOp::Mul,
                    _ => BinaryOp::Div,
                };
                let value = self.translate_complex_scalar_binary(node, op, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.neg.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.store_complex(output_name, value.map(|component| component * -1.0));
            }
            "torch.ops.aten.reciprocal.default" => {
                let value = self.get_complex_input(node, 0)?;
                let one = self.complex_constant_like(value.real, 1.0, 0.0, value.torch_dtype);
                let result = self.stable_complex_div(one, value)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.sqrt.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sqrt(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.rsqrt.default" => {
                let value = self.get_complex_input(node, 0)?;
                let root = self.complex_sqrt(value);
                let one = self.complex_constant_like(root.real, 1.0, 0.0, root.torch_dtype);
                let result = self.stable_complex_div(one, root)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.sigmoid.default" => {
                let value = self.get_complex_input(node, 0)?;
                let negative = value.map(|component| component * -1.0);
                let exponential = self.complex_exp(negative);
                let one = self.complex_constant_like(value.real, 1.0, 0.0, value.torch_dtype);
                let denominator = self.add_complex(one, exponential);
                let result = self.stable_complex_div(one, denominator)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.ldexp.Tensor" => {
                let value = self.get_complex_input(node, 0)?;
                let exponent = self.get_input_tensor(node, 1)?.cast(value.real.dtype);
                let (real, real_exponent) = broadcast_binary(value.real, exponent);
                let (imag, imag_exponent) = broadcast_binary(value.imag, exponent);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        real * real_exponent.exp2(),
                        imag * imag_exponent.exp2(),
                        value.torch_dtype,
                    ),
                );
            }
            "torch.ops.aten._conj.default"
            | "torch.ops.aten._conj_physical.default"
            | "torch.ops.aten.conj_physical.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.store_complex(
                    output_name,
                    ComplexTensor::new(value.real, value.imag * -1.0, value.torch_dtype),
                );
            }
            "torch.ops.aten.resolve_conj.default" | "torch.ops.aten.alias.default" => {
                self.store_complex(output_name, self.get_complex_input(node, 0)?);
            }
            "torch.ops.aten._fft_c2c.default" => {
                let value = self.translate_fft_c2c(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten._fft_r2c.default" => {
                let value = self.translate_fft_r2c(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten._fft_c2r.default" => {
                let value = self.translate_fft_c2r(node)?;
                self.tensors.insert(output_name.to_string(), value);
            }
            "torch.ops.aten.index_put.default" | "torch.ops.aten.index_put_.default" => {
                let value = self.translate_complex_index_put(node)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.clone.default" => {
                let value = self.get_complex_input(node, 0)?;
                let value = value.map(|component| self.materialize(component));
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.abs.default" => {
                let value = self.get_complex_input(node, 0)?;
                let out = self.complex_abs(value);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.linalg_vector_norm.default" => {
                let value = self.get_complex_input(node, 0)?;
                // A complex dtype= override changes the precision of the
                // magnitude computation even though the recorded output is
                // its real component dtype.
                let component_dtype = self.output_meta_dtype(node)?;
                let compute_dtype = match component_dtype {
                    DType::F16 => TorchDType::ComplexHalf,
                    DType::F32 => TorchDType::ComplexFloat,
                    DType::F64 => TorchDType::ComplexDouble,
                    other => anyhow::bail!(
                        "complex linalg_vector_norm requires a floating component dtype, got {other:?}"
                    ),
                };
                let value = ComplexTensor::new(
                    value.real.cast(component_dtype),
                    value.imag.cast(component_dtype),
                    compute_dtype,
                );
                let magnitude = self.complex_abs(value);
                let out = self.vector_norm_from_magnitude(node, magnitude)?;
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.dist.default" => {
                let lhs_name = self.input_value_name(node, 0)?;
                let rhs_name = self.input_value_name(node, 1)?;
                let dtype = self
                    .complex_tensors
                    .get(lhs_name)
                    .or_else(|| self.complex_tensors.get(rhs_name))
                    .context("complex dist has no complex operand")?
                    .torch_dtype;
                let lhs = self.value_as_complex(lhs_name, dtype)?;
                let rhs = self.value_as_complex(rhs_name, dtype)?;
                let (lhs_real, rhs_real) = broadcast_binary(lhs.real, rhs.real);
                let (lhs_imag, rhs_imag) = broadcast_binary(lhs.imag, rhs.imag);
                let real = lhs_real - rhs_real;
                let imag = lhs_imag - rhs_imag;
                let magnitude = (real.square() + imag.square()).sqrt();
                let p = self.get_float_arg(node, 2).unwrap_or(2.0);
                let axes = (0..magnitude.shape.len()).collect();
                let out = self.p_norm(magnitude, p, axes);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.angle.default" => {
                let value = self.get_complex_input(node, 0)?;
                let out = self.real_atan2(value.imag, value.real);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.isinf.default" => {
                let value = self.get_complex_input(node, 0)?;
                let real = self.is_inf(value.real);
                let imag = self.is_inf(value.imag);
                let out = self.bool_or(real, imag);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.isnan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let real = self.is_nan(value.real);
                let imag = self.is_nan(value.imag);
                let out = self.bool_or(real, imag);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.any.default" | "torch.ops.aten.any.dim" | "torch.ops.aten.any.dims" => {
                let value = self.get_complex_input(node, 0)?;
                let real_zero = self.is_zero(value.real);
                let imag_zero = self.is_zero(value.imag);
                let both_zero = self.bool_and(real_zero, imag_zero);
                let truth = self.bool_not(both_zero);
                let out = self.translate_any_from_truth(node, truth)?;
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.logical_not.default" => {
                let name = self.input_value_name(node, 0)?;
                let truth = self.truth_of_value(name)?;
                let out = self.bool_not(truth);
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.logical_and.default"
            | "torch.ops.aten.logical_or.default"
            | "torch.ops.aten.logical_xor.default" => {
                let lhs_name = self.input_value_name(node, 0)?;
                let rhs_name = self.input_value_name(node, 1)?;
                let lhs = self.truth_of_value(lhs_name)?;
                let rhs = self.truth_of_value(rhs_name)?;
                let (lhs, rhs) = broadcast_binary(lhs, rhs);
                let out = match target {
                    "torch.ops.aten.logical_and.default" => self.bool_and(lhs, rhs),
                    "torch.ops.aten.logical_or.default" => self.bool_or(lhs, rhs),
                    _ => lhs.ne(rhs),
                };
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.acos.default" | "torch.ops.aten.acosh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let (acos, acosh, _) = self.complex_acos_acosh(value);
                self.store_complex(
                    output_name,
                    if target == "torch.ops.aten.acos.default" {
                        acos
                    } else {
                        acosh
                    },
                );
            }
            "torch.ops.aten.asin.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_asin(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.asinh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_asinh(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.atan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_atan(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.atanh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_atanh(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.exp.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_exp(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.expm1.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_exp(value);
                let one = self.constant_like(result.real, 1.0);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(result.real - one, result.imag, result.torch_dtype),
                );
            }
            "torch.ops.aten.exp2.default" => {
                let value = self.get_complex_input(node, 0)?;
                let ln_two = self.constant_like(value.real, std::f64::consts::LN_2);
                let scaled =
                    ComplexTensor::new(value.real * ln_two, value.imag * ln_two, value.torch_dtype);
                let result = self.complex_exp(scaled);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.log.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_log(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.log1p.default" => {
                let value = self.get_complex_input(node, 0)?;
                let one = self.constant_like(value.real, 1.0);
                let shifted = ComplexTensor::new(value.real + one, value.imag, value.torch_dtype);
                let result = self.complex_log(shifted);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.log2.default" | "torch.ops.aten.log10.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_log(value);
                let denominator = if target == "torch.ops.aten.log2.default" {
                    std::f64::consts::LN_2
                } else {
                    std::f64::consts::LN_10
                };
                let denominator = self.constant_like(result.real, denominator);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        result.real / denominator,
                        result.imag / denominator,
                        result.torch_dtype,
                    ),
                );
            }
            "torch.ops.aten.sin.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sin(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.sinh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sinh(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.tan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let numerator = self.complex_sin(value);
                let denominator = self.complex_cos(value);
                let result = self.stable_complex_div(numerator, denominator)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.tanh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let numerator = self.complex_sinh(value);
                let denominator = self.complex_cosh(value);
                let result = self.stable_complex_div(numerator, denominator)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.cos.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_cos(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.cosh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_cosh(value);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.real.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.tensors.insert(output_name.to_string(), value.real);
            }
            "torch.ops.aten.imag.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.tensors.insert(output_name.to_string(), value.imag);
            }
            "torch.ops.aten.view_as_real.default" => {
                let value = self.get_complex_input(node, 0)?;
                let packed = value.pack(&mut self.graph);
                self.tensors.insert(output_name.to_string(), packed);
            }
            "torch.ops.aten.full.default" | "torch.ops.aten.full_like.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let shape = self.output_meta_shape(node)?;
                let value = self.complex_constructor_scalar_arg(node, 1, dtype)?;
                let value = if shape.is_empty() {
                    value
                } else {
                    value.map(|component| component.expand_rhs(shape.clone()))
                };
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.empty.memory_format"
            | "torch.ops.aten.empty_permuted.default"
            | "torch.ops.aten.empty_strided.default"
            | "torch.ops.aten.new_empty_strided.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let shape = self.output_meta_shape(node)?;
                let zero = self.graph.constant_float(0.0).cast(component_dtype);
                let real = if shape.is_empty() {
                    zero
                } else {
                    zero.expand_rhs(shape)
                };
                let imag = self.constant_like(real, 0.0);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "torch.ops.aten.scalar_tensor.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let value = self.complex_constructor_scalar_arg(node, 0, dtype)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.constant_pad_nd.default" => {
                let value = self.get_complex_input(node, 0)?;
                let padding = self.constant_pad_spec(node, value.real.shape.len())?;
                let fill = match node.inputs.iter().position(|input| input.name == "value") {
                    Some(index) => {
                        self.complex_constructor_scalar_arg(node, index, value.torch_dtype)?
                    }
                    None => self
                        .complex_constructor_scalar(ConstructorScalar::Int(0), value.torch_dtype)?,
                };
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        value.real.pad_with(padding.clone(), fill.real),
                        value.imag.pad_with(padding, fill.imag),
                        value.torch_dtype,
                    ),
                );
            }
            "torch.ops.aten.view_as_complex.default" => {
                let backing = self.get_input_tensor(node, 0)?;
                let dtype = self.output_complex_dtype(output_name)?;
                let value = ComplexTensor::from_interleaved(&mut self.graph, backing, dtype)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.complex.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let real = self.get_input_tensor(node, 0)?.cast(component_dtype);
                let imag = self.get_input_tensor(node, 1)?.cast(component_dtype);
                let (real, imag) = broadcast_binary(real, imag);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "torch.ops.aten._to_copy.default" => {
                let output_dtype = self
                    .tensor_meta(output_name)
                    .and_then(|m| TorchDType::from_code(m.dtype).ok())
                    .context("complex _to_copy output is missing dtype metadata")?;
                let input_name = self.input_value_name(node, 0)?;
                if output_dtype.is_complex() {
                    let value = self.value_as_complex(input_name, output_dtype)?;
                    self.store_complex(output_name, value);
                } else {
                    let value = self.get_complex(input_name)?;
                    let dtype = DType::try_from(output_dtype).map_err(|t| {
                        anyhow::anyhow!("unsupported real cast target {}", t.name())
                    })?;
                    let out = if dtype == DType::Bool {
                        let real_zero = self.is_zero(value.real);
                        let imag_zero = self.is_zero(value.imag);
                        let both_zero = self.bool_and(real_zero, imag_zero);
                        self.bool_not(both_zero)
                    } else {
                        value.real.cast(dtype)
                    };
                    self.tensors.insert(output_name.to_string(), out);
                }
            }
            "torch.ops.aten.copy.default" => {
                let output_dtype = self
                    .tensor_meta(output_name)
                    .and_then(|meta| TorchDType::from_code(meta.dtype).ok())
                    .context("complex copy output is missing dtype metadata")?;
                let destination_name = self.input_value_name(node, 0)?;
                let source_name = self.input_value_name(node, 1)?;
                if output_dtype.is_complex() {
                    let destination = self.value_as_complex(destination_name, output_dtype)?;
                    let source = self.value_as_complex(source_name, output_dtype)?;
                    let component_dtype = output_dtype.complex_component_dtype().unwrap();
                    self.store_complex(
                        output_name,
                        ComplexTensor::new(
                            copy_tensor(destination.real, source.real, component_dtype)?,
                            copy_tensor(destination.imag, source.imag, component_dtype)?,
                            output_dtype,
                        ),
                    );
                } else {
                    let destination = self.get_tensor(destination_name)?;
                    let source = self.get_complex(source_name)?;
                    let dtype = DType::try_from(output_dtype).map_err(|dtype| {
                        anyhow::anyhow!("unsupported real copy target {}", dtype.name())
                    })?;
                    let out = if dtype == DType::Bool {
                        let real = copy_tensor(destination, source.real, source.real.dtype)?;
                        let imag = copy_tensor(destination, source.imag, source.imag.dtype)?;
                        let real_zero = self.is_zero(real);
                        let imag_zero = self.is_zero(imag);
                        let both_zero = self.bool_and(real_zero, imag_zero);
                        self.bool_not(both_zero)
                    } else {
                        copy_tensor(destination, source.real, dtype)?
                    };
                    self.tensors.insert(output_name.to_string(), out);
                }
            }
            "torch.ops.aten.view.default" | "torch.ops.aten.view_copy.default" => {
                let value = self.get_complex_input(node, 0)?;
                let shape = self.output_meta_shape(node)?;
                let mut value =
                    value.map(|component| self.reshape_complex_component(component, shape.clone()));
                if target == "torch.ops.aten.view_copy.default" {
                    value = value.map(materialize_tensor);
                }
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.permute.default" | "torch.ops.aten.permute_copy.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dims = self.get_ints_arg(node, 1)?;
                let axes: Vec<usize> = dims
                    .iter()
                    .map(|&dim| normalize_dim(dim, value.real.shape.len()))
                    .collect();
                let copy = target == "torch.ops.aten.permute_copy.default";
                let value = value.map(|component| {
                    let component = component.permute(axes.clone());
                    if copy {
                        materialize_tensor(component)
                    } else {
                        component
                    }
                });
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.narrow_copy.default" => {
                let value = self.get_complex_input(node, 0)?;
                let raw_dim = self.get_int_arg(node, 1)?;
                anyhow::ensure!(
                    raw_dim >= -(value.real.shape.len() as i64)
                        && raw_dim < value.real.shape.len() as i64,
                    "complex narrow_copy dimension {raw_dim} out of range for rank {}",
                    value.real.shape.len()
                );
                let dim = normalize_dim(raw_dim, value.real.shape.len());
                let start = self.get_expr_arg(node, 2)?;
                let length = self.get_expr_arg(node, 3)?;
                let real = narrow_copy_tensor(value.real, dim, start, length)?;
                let imag = narrow_copy_tensor(value.imag, dim, start, length)?;
                self.store_complex(
                    output_name,
                    ComplexTensor::new(real, imag, value.torch_dtype),
                );
            }
            "torch.ops.aten.unbind_copy.int" => {
                self.translate_complex_unbind_copy(node)?;
            }
            "torch.ops.aten.flip.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dims =
                    normalize_flip_dims(&self.get_ints_arg(node, 1)?, value.real.shape.len())?;
                let indices = flip_indices(value.real, &dims);
                self.store_complex(
                    output_name,
                    value.map(|component| component.gather(indices)),
                );
            }
            "torch.ops.aten.diagonal.default" => {
                let value = self.get_complex_input(node, 0)?;
                let offset = self.get_int_arg(node, 1).unwrap_or(0);
                let (dim1, dim2) = normalize_diagonal_dims(
                    self.get_int_arg(node, 2).unwrap_or(0),
                    self.get_int_arg(node, 3).unwrap_or(1),
                    value.real.shape.len(),
                )?;
                let output_shape = self.output_meta_shape(node)?;
                let indices = diagonal_indices(value.real, &output_shape, offset, dim1, dim2)?;
                self.store_complex(
                    output_name,
                    value.map(|component| component.gather(indices)),
                );
            }
            "torch.ops.aten.diagonal_scatter.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let destination = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let source = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
                let offset = self.get_int_arg(node, 2).unwrap_or(0);
                let (dim1, dim2) = normalize_diagonal_dims(
                    self.get_int_arg(node, 3).unwrap_or(0),
                    self.get_int_arg(node, 4).unwrap_or(1),
                    destination.real.shape.len(),
                )?;
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        diagonal_scatter_tensor(destination.real, source.real, offset, dim1, dim2)?,
                        diagonal_scatter_tensor(destination.imag, source.imag, offset, dim1, dim2)?,
                        dtype,
                    ),
                );
            }
            "torch.ops.aten.index_select.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let value = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let raw_dim = self.get_int_arg(node, 1)?;
                let dim = if value.real.shape.is_empty() {
                    anyhow::ensure!(
                        raw_dim == 0 || raw_dim == -1,
                        "index_select dimension {raw_dim} out of range for a scalar"
                    );
                    0
                } else {
                    anyhow::ensure!(
                        raw_dim >= -(value.real.shape.len() as i64)
                            && raw_dim < value.real.shape.len() as i64,
                        "index_select dimension {raw_dim} out of range for rank {}",
                        value.real.shape.len()
                    );
                    normalize_dim(raw_dim, value.real.shape.len())
                };
                let index = self.get_input_tensor(node, 2)?;
                let output_shape = self.output_meta_shape(node)?;
                self.store_complex(
                    output_name,
                    value.try_map(|component| {
                        index_select_tensor(component, index, dim, &output_shape)
                    })?,
                );
            }
            "torch.ops.aten.unfold.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let value = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let raw_dim = self.get_int_arg(node, 1)?;
                let dim = if value.real.shape.is_empty() {
                    anyhow::ensure!(
                        raw_dim == 0 || raw_dim == -1,
                        "unfold dimension {raw_dim} out of range for a scalar"
                    );
                    0
                } else {
                    anyhow::ensure!(
                        raw_dim >= -(value.real.shape.len() as i64)
                            && raw_dim < value.real.shape.len() as i64,
                        "unfold dimension {raw_dim} out of range for rank {}",
                        value.real.shape.len()
                    );
                    normalize_dim(raw_dim, value.real.shape.len())
                };
                let size = self.get_int_arg(node, 2)?;
                let step = self.get_int_arg(node, 3)?;
                let output_shape = self.output_meta_shape(node)?;
                self.store_complex(
                    output_name,
                    value.try_map(|component| {
                        unfold_tensor(component, dim, size, step, &output_shape)
                    })?,
                );
            }
            "torch.ops.aten.unsqueeze.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dim = normalize_dim(self.get_int_arg(node, 1)?, value.real.shape.len() + 1);
                self.store_complex(output_name, value.map(|component| component.unsqueeze(dim)));
            }
            "torch.ops.aten.squeeze.dims" | "torch.ops.aten.squeeze.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dims = if target.ends_with("dims") {
                    self.get_ints_arg(node, 1)?
                } else {
                    value
                        .real
                        .dims()
                        .iter()
                        .enumerate()
                        .filter_map(|(axis, dim)| {
                            (dim.to_usize() == Some(1)).then_some(axis as i64)
                        })
                        .collect()
                };
                let rank = value.real.shape.len();
                let mut dims: Vec<usize> = dims
                    .into_iter()
                    .map(|dim| normalize_dim(dim, rank))
                    .collect();
                dims.sort();
                self.store_complex(
                    output_name,
                    value.map(|component| squeeze_dims(component, &dims)),
                );
            }
            "torch.ops.aten.expand.default" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.expand_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.repeat.default" => {
                let value = self.get_complex_input(node, 0)?;
                let repeats = self.get_ints_arg(node, 1)?;
                let value = value
                    .try_map(|component| self.repeat_complex_component(component, &repeats))?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.slice.Tensor" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.slice_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.index.Tensor" => {
                let value = self.get_complex_input(node, 0)?;
                let real = self.translate_index_tensor_from_source(node, value.real)?;
                let imag = self.translate_index_tensor_from_source(node, value.imag)?;
                self.store_complex(
                    output_name,
                    ComplexTensor::new(real, imag, value.torch_dtype),
                );
            }
            "torch.ops.aten.gather.default" => {
                let value = self.get_complex_input(node, 0)?;
                let raw_dim = self.get_int_arg(node, 1)?;
                let dim = normalize_dim(raw_dim, value.real.shape.len());
                let indices = self.get_input_tensor(node, 2)?;
                let real = super::movement_dynamic::pt2_gather_elements(value.real, indices, dim);
                let imag = super::movement_dynamic::pt2_gather_elements(value.imag, indices, dim);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(real, imag, value.torch_dtype),
                );
            }
            "torch.ops.aten.slice_scatter.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let destination = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let source = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
                let dim = normalize_dim(
                    self.get_int_arg(node, 2).unwrap_or(0),
                    destination.real.shape.len(),
                );
                let start = normalize_slice_bound(
                    self.get_expr_arg(node, 3)
                        .unwrap_or_else(|_| Expression::from(0)),
                    destination.real.dims()[dim],
                );
                let step = self.get_int_arg(node, 5).unwrap_or(1);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        slice_scatter_tensor(destination.real, source.real, dim, start, step)?,
                        slice_scatter_tensor(destination.imag, source.imag, dim, start, step)?,
                        dtype,
                    ),
                );
            }
            "torch.ops.aten.masked_scatter.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let destination = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let mask = self.get_input_tensor(node, 1)?;
                let source = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
                let real = masked_scatter_tensor(self, destination.real, mask, source.real)?;
                let imag = masked_scatter_tensor(self, destination.imag, mask, source.imag)?;
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "torch.ops.aten.put.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let destination = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let indices = self.get_input_tensor(node, 1)?;
                let source = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
                let accumulate = self.get_bool_arg(node, 3).unwrap_or(false);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(
                        put_tensor(destination.real, indices, source.real, accumulate)?,
                        put_tensor(destination.imag, indices, source.imag, accumulate)?,
                        dtype,
                    ),
                );
            }
            "torch.ops.aten.nonzero_static.default" => {
                let value = self.get_complex_input(node, 0)?;
                let real_zero = self.is_zero(value.real);
                let imag_zero = self.is_zero(value.imag);
                let truth = self.bool_not(self.bool_and(real_zero, imag_zero));
                let size_index = node
                    .inputs
                    .iter()
                    .position(|input| input.name == "size")
                    .unwrap_or(1);
                let size = self.get_expr_arg(node, size_index)?;
                let fill_value = node
                    .inputs
                    .iter()
                    .position(|input| input.name == "fill_value")
                    .and_then(|index| self.get_int_arg(node, index).ok())
                    .unwrap_or(-1);
                let result = nonzero_static_from_truth(self, truth, size, fill_value);
                self.tensors.insert(output_name.to_string(), result);
            }
            "torch.ops.aten.split_with_sizes.default" => {
                self.translate_complex_split_with_sizes(node)?;
            }
            "torch.ops.aten.select.int" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.select_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.cat.default" => {
                self.translate_complex_cat(node, output_name)?;
            }
            "torch.ops.aten.where.self" => {
                let condition = self.get_input_tensor(node, 0)?;
                let dtype = self.output_complex_dtype(output_name)?;
                let lhs = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
                let rhs = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
                let (lhs_real, rhs_real) = broadcast_binary(lhs.real, rhs.real);
                let (lhs_imag, rhs_imag) = broadcast_binary(lhs.imag, rhs.imag);
                let (condition, lhs_real) = broadcast_binary(condition, lhs_real);
                let (rhs_real, _) = broadcast_binary(rhs_real, lhs_real);
                let (lhs_imag, _) = broadcast_binary(lhs_imag, lhs_real);
                let (rhs_imag, _) = broadcast_binary(rhs_imag, lhs_real);
                let real = self.select(condition, lhs_real, rhs_real);
                let imag = self.select(condition, lhs_imag, rhs_imag);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "torch.ops.aten.polar.default" => {
                let magnitude = self.get_input_tensor(node, 0)?;
                let angle = self.get_input_tensor(node, 1)?;
                let (magnitude, angle) = broadcast_binary(magnitude, angle);
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let magnitude = magnitude.cast(component_dtype);
                let angle = angle.cast(component_dtype);
                let real = magnitude * self.real_cos(angle);
                let imag = magnitude * angle.sin();
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "torch.ops.aten.pow.Tensor_Scalar" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let base = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let (real, imag) = self.complex_scalar_arg(node, 1)?;
                let exponent = self.complex_constant_like(base.real, real, imag, dtype);
                let result = self.complex_pow(base, exponent);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.pow.Tensor_Tensor" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let mut base = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let mut exponent = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
                let (base_real, exponent_real) = broadcast_binary(base.real, exponent.real);
                let (base_imag, exponent_imag) = broadcast_binary(base.imag, exponent.imag);
                base.real = base_real;
                base.imag = base_imag;
                exponent.real = exponent_real;
                exponent.imag = exponent_imag;
                let result = self.complex_pow(base, exponent);
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.scatter.src"
            | "torch.ops.aten.scatter.reduce"
            | "torch.ops.aten.scatter_add.default" => {
                let result = self.translate_complex_scatter_src(node, output_name)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.scatter.value" | "torch.ops.aten.scatter.value_reduce" => {
                let result = self.translate_complex_scatter_value(node, output_name)?;
                self.store_complex(output_name, result);
            }
            "torch.ops.aten.sum.dim_IntList" | "torch.ops.aten.sum.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Sum)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.mean.dim" | "torch.ops.aten.mean.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Mean)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.var.default"
            | "torch.ops.aten.var.dim"
            | "torch.ops.aten.var.correction" => {
                let (variance, _) = self.translate_complex_var_mean(node)?;
                self.tensors.insert(output_name.to_string(), variance);
            }
            "torch.ops.aten.var_mean.default"
            | "torch.ops.aten.var_mean.dim"
            | "torch.ops.aten.var_mean.correction" => {
                let (variance, mean) = self.translate_complex_var_mean(node)?;
                let names = Self::tensor_output_names(node);
                anyhow::ensure!(names.len() == 2, "complex var_mean must have two outputs");
                self.tensors.insert(names[0].clone(), variance);
                self.store_complex(&names[1], mean);
            }
            "torch.ops.aten.prod.dim_int" | "torch.ops.aten.prod.default" => {
                let value = self.translate_complex_product(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.cumsum.default" => {
                let value = self.get_complex_input(node, 0)?;
                if value.real.shape.is_empty() {
                    self.store_complex(output_name, value);
                } else {
                    let dim = normalize_dim(self.get_int_arg(node, 1)?, value.real.shape.len());
                    self.store_complex(output_name, value.map(|component| component.cumsum(dim)));
                }
            }
            "torch.ops.aten.logcumsumexp.default" => {
                let value = self.get_complex_input(node, 0)?;
                let raw_dim = self.get_int_arg(node, 1)?;
                if value.real.shape.is_empty() {
                    anyhow::ensure!(
                        matches!(raw_dim, -1 | 0),
                        "dimension out of range for scalar"
                    );
                    self.store_complex(output_name, value);
                } else {
                    let dim = normalize_dim(raw_dim, value.real.shape.len());
                    let rank = value.real.shape.len();
                    let length = value.real.dims()[dim];
                    let mut padding = vec![(Expression::from(0), Expression::from(0)); rank];
                    padding[dim] = (length - 1, Expression::from(0));
                    let negative_infinity =
                        self.floating_scalar(f64::NEG_INFINITY, value.real.dtype);
                    let zero = self.floating_scalar(0.0, value.imag.dtype);
                    let mut real_windows = value.real.pad_with(padding.clone(), negative_infinity);
                    let mut imag_windows = value.imag.pad_with(padding, zero);
                    let mut kernel = vec![Expression::from(1); rank];
                    kernel[dim] = length;
                    real_windows =
                        real_windows.unfold(kernel.clone(), vec![1usize; rank], vec![1usize; rank]);
                    imag_windows =
                        imag_windows.unfold(kernel, vec![1usize; rank], vec![1usize; rank]);
                    for kernel_axis in (0..rank).rev() {
                        if kernel_axis != dim {
                            real_windows = real_windows.squeeze(rank + kernel_axis);
                            imag_windows = imag_windows.squeeze(rank + kernel_axis);
                        }
                    }

                    // Factor out the largest real component in each prefix.
                    // This is the complex analogue of the stable real
                    // log-sum-exp identity and prevents exp(real(z)) overflow.
                    let reduction_axis = rank;
                    let maximum = real_windows.max(reduction_axis);
                    let expanded =
                        maximum.expand_dim(reduction_axis, real_windows.dims()[reduction_axis]);
                    let magnitude = self.real_exp(real_windows - expanded);
                    let terms = ComplexTensor::new(
                        magnitude * self.real_cos(imag_windows),
                        magnitude * imag_windows.sin(),
                        value.torch_dtype,
                    );
                    let summed = ComplexTensor::new(
                        terms.real.sum(reduction_axis),
                        terms.imag.sum(reduction_axis),
                        value.torch_dtype,
                    );
                    let logarithm = self.complex_log(summed);
                    self.store_complex(
                        output_name,
                        ComplexTensor::new(
                            logarithm.real + maximum,
                            logarithm.imag,
                            value.torch_dtype,
                        ),
                    );
                }
            }
            "torch.ops.aten.cumprod.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let mut value = self.get_complex_input(node, 0)?.cast(dtype);
                let dim = self.get_int_arg(node, 1)?;
                if value.real.shape.is_empty() {
                    anyhow::ensure!(
                        matches!(dim, -1 | 0),
                        "Dimension out of range for scalar cumprod: {dim}"
                    );
                } else {
                    let dim = normalize_dim(dim, value.real.shape.len());
                    anyhow::ensure!(
                        dim < value.real.shape.len(),
                        "Dimension out of range for complex cumprod: {dim}"
                    );
                    let length = value.real.dims()[dim]
                        .to_usize()
                        .context("complex cumprod currently requires a concrete scan dimension")?;
                    let mut offset = 1;
                    while offset < length {
                        let (shifted_indices, valid) =
                            self.scan_shift_indices(&value.real.dims(), dim, offset);
                        let left = ComplexTensor::new(
                            super::movement_dynamic::pt2_gather_elements(
                                value.real,
                                shifted_indices,
                                dim,
                            ),
                            super::movement_dynamic::pt2_gather_elements(
                                value.imag,
                                shifted_indices,
                                dim,
                            ),
                            dtype,
                        );
                        let product = self.complex_mul(left, value);
                        value = ComplexTensor::new(
                            self.select(valid, product.real, value.real),
                            self.select(valid, product.imag, value.imag),
                            dtype,
                        );
                        offset *= 2;
                    }
                }
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.mm.default" | "torch.ops.aten.bmm.default" => {
                let value = self.translate_complex_matmul(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.addmv.default" => {
                let value = self.translate_complex_addmv(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.addbmm.default" => {
                let value = self.translate_complex_addbmm(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.addmm.default" => {
                let value = self.translate_complex_addmm(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.eq.Tensor" | "torch.ops.aten.ne.Tensor" => {
                let is_eq = target == "torch.ops.aten.eq.Tensor";
                let out = self.translate_complex_comparison(node, is_eq)?;
                self.tensors.insert(output_name.to_string(), out);
            }
            "torch.ops.aten.eq.Scalar" | "torch.ops.aten.ne.Scalar" => {
                let value = self.get_complex_input(node, 0)?;
                let (real, imag) = self.complex_scalar_arg(node, 1)?;
                let scalar = self.complex_constant_like(value.real, real, imag, value.torch_dtype);
                let value_real_nan = self.is_nan(value.real);
                let scalar_real_nan = self.is_nan(scalar.real);
                let real_nan = self.bool_or(value_real_nan, scalar_real_nan);
                let value_imag_nan = self.is_nan(value.imag);
                let scalar_imag_nan = self.is_nan(scalar.imag);
                let imag_nan = self.bool_or(value_imag_nan, scalar_imag_nan);
                let real_eq = self.bool_and(value.real.eq(scalar.real), self.bool_not(real_nan));
                let imag_eq = self.bool_and(value.imag.eq(scalar.imag), self.bool_not(imag_nan));
                let equal = self.bool_and(real_eq, imag_eq);
                let out = if target == "torch.ops.aten.eq.Scalar" {
                    equal
                } else {
                    self.bool_not(equal)
                };
                self.tensors.insert(output_name.to_string(), out);
            }
            other => bail!(
                "Unsupported complex ATen op: {other}. Complex values must be lowered into real components before HLIR"
            ),
        }
        Ok(())
    }

    fn store_complex(&mut self, name: &str, value: ComplexTensor) {
        self.complex_tensors.insert(name.to_string(), value);
    }

    fn get_complex(&self, name: &str) -> Result<ComplexTensor> {
        self.complex_tensors
            .get(name)
            .copied()
            .with_context(|| format!("Unknown complex tensor: {name}"))
    }

    fn get_complex_input(&self, node: &Node, idx: usize) -> Result<ComplexTensor> {
        self.get_complex(self.input_value_name(node, idx)?)
    }

    fn input_value_name<'n>(&self, node: &'n Node, idx: usize) -> Result<&'n str> {
        node.inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg
            .as_value_name()
            .with_context(|| format!("Input {idx} of {} is not tensor-backed", node.target))
    }

    fn truth_of_value(&mut self, name: &str) -> Result<GraphTensor> {
        if let Some(value) = self.complex_tensors.get(name).copied() {
            let real_zero = self.is_zero(value.real);
            let imag_zero = self.is_zero(value.imag);
            let both_zero = self.bool_and(real_zero, imag_zero);
            Ok(self.bool_not(both_zero))
        } else {
            let value = self.get_tensor(name)?;
            let zero = self.constant_like(value, 0.0);
            Ok(value.ne(zero))
        }
    }

    fn complex_scatter_reduction(&self, node: &Node) -> Result<ScatterReduction> {
        if node.target == "torch.ops.aten.scatter_add.default" {
            return Ok(ScatterReduction::Add);
        }
        let reduce = node
            .inputs
            .iter()
            .find(|input| input.name == "reduce")
            .and_then(|input| match &input.arg {
                Argument::Other(value) => value
                    .as_str()
                    .or_else(|| value.get("as_string").and_then(|value| value.as_str())),
                _ => None,
            })
            .with_context(|| format!("{} is missing its reduce argument", node.target))?;
        match reduce {
            "add" | "sum" => Ok(ScatterReduction::Add),
            "multiply" | "prod" => Ok(ScatterReduction::Multiply),
            other => bail!("Unsupported complex scatter reduction: {other}"),
        }
    }

    fn reduce_complex_scatter(
        &mut self,
        data: ComplexTensor,
        indices: GraphTensor,
        updates: ComplexTensor,
        axis: usize,
        reduction: ScatterReduction,
    ) -> Result<ComplexTensor> {
        anyhow::ensure!(
            indices.shape.len() == updates.real.shape.len(),
            "complex scatter reduction requires index/update ranks to match"
        );
        let index_shape = indices.dims();
        let update_shape = updates.real.dims();
        anyhow::ensure!(
            index_shape
                .iter()
                .zip(&update_shape)
                .all(|(index, update)| index == update || index.egglog_equal(*update)),
            "complex scatter reduction currently requires index and update shapes to match"
        );
        let update_count = product_of_dims(index_shape.iter().copied())
            .to_usize()
            .context("complex scatter reduction requires a concrete update element count")?;
        let destinations =
            super::movement_dynamic::pt2_scatter_element_indices(data.real, indices, axis);
        let output_shape = data.real.dims();
        let update_real = updates.real.flatten();
        let update_imag = updates.imag.flatten();
        let mut output_real = data.real.flatten();
        let mut output_imag = data.imag.flatten();

        for index in 0..update_count {
            let destination = destinations.slice_along(index..index + 1, 0);
            let incoming = ComplexTensor::new(
                update_real.slice_along(index..index + 1, 0),
                update_imag.slice_along(index..index + 1, 0),
                data.torch_dtype,
            );
            let current = ComplexTensor::new(
                output_real.gather(destination),
                output_imag.gather(destination),
                data.torch_dtype,
            );
            let combined = match reduction {
                ScatterReduction::Add => ComplexTensor::new(
                    current.real + incoming.real,
                    current.imag + incoming.imag,
                    data.torch_dtype,
                ),
                ScatterReduction::Multiply => self.complex_mul(current, incoming),
            };
            output_real = combined.real.scatter(destination, output_real);
            output_imag = combined.imag.scatter(destination, output_imag);
        }

        output_real.shape = ShapeTracker::new(output_shape.clone());
        output_imag.shape = ShapeTracker::new(output_shape);
        Ok(ComplexTensor::new(
            output_real,
            output_imag,
            data.torch_dtype,
        ))
    }

    fn translate_complex_scatter_src(
        &mut self,
        node: &Node,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let data = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let raw_dim = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            raw_dim >= -(data.real.shape.len() as i64) && raw_dim < data.real.shape.len() as i64,
            "complex scatter dimension {raw_dim} out of range for rank {}",
            data.real.shape.len()
        );
        let dim = normalize_dim(raw_dim, data.real.shape.len());
        let indices = self.get_input_tensor(node, 2)?.cast(DType::Int);
        let updates = self.value_as_complex(self.input_value_name(node, 3)?, dtype)?;
        if node.target == "torch.ops.aten.scatter.src" {
            Ok(ComplexTensor::new(
                super::movement_dynamic::pt2_scatter_elements(
                    data.real,
                    indices,
                    updates.real,
                    dim,
                ),
                super::movement_dynamic::pt2_scatter_elements(
                    data.imag,
                    indices,
                    updates.imag,
                    dim,
                ),
                dtype,
            ))
        } else {
            let reduction = self.complex_scatter_reduction(node)?;
            self.reduce_complex_scatter(data, indices, updates, dim, reduction)
        }
    }

    fn translate_complex_scatter_value(
        &mut self,
        node: &Node,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let data = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let raw_dim = self.get_int_arg(node, 1)?;
        anyhow::ensure!(
            raw_dim >= -(data.real.shape.len() as i64) && raw_dim < data.real.shape.len() as i64,
            "complex scatter dimension {raw_dim} out of range for rank {}",
            data.real.shape.len()
        );
        let dim = normalize_dim(raw_dim, data.real.shape.len());
        let indices = self.get_input_tensor(node, 2)?.cast(DType::Int);
        let scalar = self.complex_constructor_scalar_arg(node, 3, dtype)?;
        let updates = ComplexTensor::new(
            scalar.real.expand_rhs(indices.shape),
            scalar.imag.expand_rhs(indices.shape),
            dtype,
        );
        if node.target == "torch.ops.aten.scatter.value" {
            Ok(ComplexTensor::new(
                super::movement_dynamic::pt2_scatter_elements(
                    data.real,
                    indices,
                    updates.real,
                    dim,
                ),
                super::movement_dynamic::pt2_scatter_elements(
                    data.imag,
                    indices,
                    updates.imag,
                    dim,
                ),
                dtype,
            ))
        } else {
            let reduction = self.complex_scatter_reduction(node)?;
            self.reduce_complex_scatter(data, indices, updates, dim, reduction)
        }
    }

    fn translate_complex_unbind_copy(&mut self, node: &Node) -> Result<()> {
        let value = self.get_complex_input(node, 0)?;
        let raw_dim = self.get_int_arg(node, 1).unwrap_or(0);
        anyhow::ensure!(
            raw_dim >= -(value.real.shape.len() as i64) && raw_dim < value.real.shape.len() as i64,
            "complex unbind_copy dimension {raw_dim} out of range for rank {}",
            value.real.shape.len()
        );
        let dim = normalize_dim(raw_dim, value.real.shape.len());
        let output_names: Vec<String> = node
            .outputs
            .iter()
            .flat_map(|output| {
                output
                    .as_tensors
                    .as_ref()
                    .map(|tensors| {
                        tensors
                            .iter()
                            .map(|tensor| tensor.name.clone())
                            .collect::<Vec<_>>()
                    })
                    .or_else(|| {
                        output
                            .as_tensor
                            .as_ref()
                            .map(|tensor| vec![tensor.name.clone()])
                    })
                    .unwrap_or_default()
            })
            .collect();
        let axis_size = value.real.shape.dims[dim]
            .to_usize()
            .context("complex unbind_copy requires a concrete unbound dimension")?;
        anyhow::ensure!(
            output_names.len() == axis_size,
            "complex unbind_copy produced {} outputs for an axis of size {axis_size}",
            output_names.len()
        );
        for (index, name) in output_names.into_iter().enumerate() {
            let real = value.real.slice_along(index..index + 1, dim).squeeze(dim);
            let imag = value.imag.slice_along(index..index + 1, dim).squeeze(dim);
            let real = materialize_tensor(real);
            let imag = materialize_tensor(imag);
            self.store_complex(&name, ComplexTensor::new(real, imag, value.torch_dtype));
        }
        Ok(())
    }

    fn output_complex_dtype(&self, output_name: &str) -> Result<TorchDType> {
        let dtype = self
            .tensor_meta(output_name)
            .with_context(|| format!("Missing tensor metadata for {output_name}"))?
            .dtype;
        let dtype = TorchDType::from_code(dtype)
            .map_err(|code| anyhow::anyhow!("Unknown PT2 dtype code {code}"))?;
        anyhow::ensure!(dtype.is_complex(), "Output {output_name} is not complex");
        Ok(dtype)
    }

    fn complex_scalar_arg(&self, node: &Node, index: usize) -> Result<(f64, f64)> {
        Ok(match node.inputs[index].arg.as_complex() {
            Some(value) => value,
            None => (self.get_float_arg(node, index)?, 0.0),
        })
    }

    fn complex_constructor_scalar_arg(
        &mut self,
        node: &Node,
        index: usize,
        dtype: TorchDType,
    ) -> Result<ComplexTensor> {
        let value = self.constructor_scalar_arg(node, index)?;
        self.complex_constructor_scalar(value, dtype)
    }

    fn complex_constructor_scalar(
        &mut self,
        value: ConstructorScalar,
        dtype: TorchDType,
    ) -> Result<ComplexTensor> {
        let (real, imag) = match value {
            ConstructorScalar::Value(name) => {
                let value = self.value_as_complex(&name, dtype)?;
                return Ok(ComplexTensor::new(
                    reshape_tensor(value.real, vec![]),
                    reshape_tensor(value.imag, vec![]),
                    dtype,
                ));
            }
            ConstructorScalar::Complex(real, imag) => (
                ConstructorScalar::Float(real),
                ConstructorScalar::Float(imag),
            ),
            real => (real, ConstructorScalar::Int(0)),
        };
        let component_dtype = dtype.complex_component_dtype().unwrap();
        Ok(ComplexTensor::new(
            self.typed_scalar_constant(&real, component_dtype)?,
            self.typed_scalar_constant(&imag, component_dtype)?,
            dtype,
        ))
    }

    fn value_as_complex(&mut self, name: &str, dtype: TorchDType) -> Result<ComplexTensor> {
        if let Some(value) = self.complex_tensors.get(name).copied() {
            return Ok(value.cast(dtype));
        }
        let component_dtype = dtype.complex_component_dtype().unwrap();
        let real = self.get_tensor(name)?.cast(component_dtype);
        let imag = self.constant_like(real, 0.0);
        Ok(ComplexTensor::new(real, imag, dtype))
    }

    pub(crate) fn constant_like(&mut self, tensor: GraphTensor, value: f64) -> GraphTensor {
        let scalar = if tensor.dtype == DType::F64 {
            self.graph.constant_float64(value)
        } else {
            self.graph.constant_float(value as f32).cast(tensor.dtype)
        };
        scalar.expand_rhs(tensor.shape)
    }

    fn complex_constant_like(
        &mut self,
        tensor: GraphTensor,
        real: f64,
        imag: f64,
        dtype: TorchDType,
    ) -> ComplexTensor {
        ComplexTensor::new(
            self.constant_like(tensor, real),
            self.constant_like(tensor, imag),
            dtype,
        )
    }

    fn complex_mul(&self, a: ComplexTensor, b: ComplexTensor) -> ComplexTensor {
        ComplexTensor::new(
            a.real * b.real - a.imag * b.imag,
            a.real * b.imag + a.imag * b.real,
            a.torch_dtype,
        )
    }

    fn complex_matmul(&self, a: ComplexTensor, b: ComplexTensor) -> ComplexTensor {
        ComplexTensor::new(
            a.real.matmul(b.real) - a.imag.matmul(b.imag),
            a.real.matmul(b.imag) + a.imag.matmul(b.real),
            a.torch_dtype,
        )
    }

    fn scale_complex_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: ComplexTensor,
    ) -> Result<ComplexTensor> {
        let Some(index) = node.inputs.iter().position(|input| input.name == name) else {
            return Ok(value);
        };
        let scalar_arg = self.constructor_scalar_arg(node, index)?;
        if scalar_arg.is_literal_one() {
            return Ok(value);
        }
        if scalar_arg.is_literal_zero() {
            return Ok(ComplexTensor::new(
                self.constant_like(value.real, 0.0),
                self.constant_like(value.imag, 0.0),
                value.torch_dtype,
            ));
        }
        let scalar = self.complex_constructor_scalar(scalar_arg, value.torch_dtype)?;
        let scalar = ComplexTensor::new(
            scalar.real.expand_rhs(value.real.shape),
            scalar.imag.expand_rhs(value.imag.shape),
            value.torch_dtype,
        );
        Ok(self.complex_mul(value, scalar))
    }

    fn add_complex(&self, lhs: ComplexTensor, rhs: ComplexTensor) -> ComplexTensor {
        let (lhs_real, rhs_real) = broadcast_binary(lhs.real, rhs.real);
        let (lhs_imag, rhs_imag) = broadcast_binary(lhs.imag, rhs.imag);
        ComplexTensor::new(lhs_real + rhs_real, lhs_imag + rhs_imag, lhs.torch_dtype)
    }

    fn scale_addend(&mut self, node: &Node, value: ComplexTensor) -> Result<ComplexTensor> {
        let (real, imag) = match node.inputs.iter().position(|input| input.name == "alpha") {
            Some(index) => self.complex_scalar_arg(node, index)?,
            None => (1.0, 0.0),
        };
        let alpha = self.complex_constant_like(value.real, real, imag, value.torch_dtype);
        Ok(self.complex_mul(value, alpha))
    }

    fn apply_complex_binary(
        &mut self,
        op: BinaryOp,
        a: ComplexTensor,
        b: ComplexTensor,
    ) -> Result<ComplexTensor> {
        let dtype = a.torch_dtype;
        match op {
            BinaryOp::Add => Ok(ComplexTensor::new(a.real + b.real, a.imag + b.imag, dtype)),
            BinaryOp::Sub => Ok(ComplexTensor::new(a.real - b.real, a.imag - b.imag, dtype)),
            BinaryOp::Mul => Ok(self.complex_mul(a, b)),
            BinaryOp::Div => self.stable_complex_div(a, b),
        }
    }

    /// Elementwise selection through gather, not arithmetic masking. This is
    /// essential for IEEE values because `0 * inf` and `0 * NaN` are NaN.
    pub(crate) fn select(
        &mut self,
        condition: GraphTensor,
        if_true: GraphTensor,
        if_false: GraphTensor,
    ) -> GraphTensor {
        let shape = if_true.dims();
        let packed = interleave(&mut self.graph, if_false, if_true);
        let base = self.graph.iota(Expression::from('z') * 2, shape);
        packed.gather(base + condition.cast(DType::Int))
    }

    pub(crate) fn real_abs(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let magnitude = self.select(value.lt(zero), value * -1.0, value);
        // `abs(-0)` is +0; the comparison above deliberately treats both
        // zeros alike, so canonicalize zero without touching NaN or infinity.
        let is_zero = self.is_zero(value);
        self.select(is_zero, zero, magnitude)
    }

    pub(crate) fn bool_or(&self, lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor {
        let lhs = lhs.cast(DType::F32);
        let rhs = rhs.cast(DType::F32);
        (lhs + rhs - lhs * rhs).cast(DType::Bool)
    }

    pub(crate) fn bool_and(&self, lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor {
        (lhs.cast(DType::F32) * rhs.cast(DType::F32)).cast(DType::Bool)
    }

    pub(crate) fn bool_not(&self, value: GraphTensor) -> GraphTensor {
        (1.0 - value.cast(DType::F32)).cast(DType::Bool)
    }

    pub(crate) fn is_inf(&mut self, value: GraphTensor) -> GraphTensor {
        let largest = float_max(value.dtype);
        let positive = value.gt(self.constant_like(value, largest));
        let negative = value.lt(self.constant_like(value, -largest));
        self.bool_or(positive, negative)
    }

    pub(crate) fn is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let nonzero = self.bool_or(value.lt(zero), value.gt(zero));
        let nan = self.is_nan(value);
        let nonzero_or_nan = self.bool_or(nonzero, nan);
        self.bool_not(nonzero_or_nan)
    }

    pub(crate) fn is_nan(&mut self, value: GraphTensor) -> GraphTensor {
        let largest = self.constant_like(value, f32::MAX as f64);
        let ordered = self.bool_or(value.lt(largest), value.gt(largest * -1.0));
        self.bool_not(ordered)
    }

    pub(crate) fn signbit(&mut self, value: GraphTensor) -> GraphTensor {
        let zero = self.constant_like(value, 0.0);
        let negative = value.lt(zero);
        let negative_zero = value.reciprocal().lt(zero);
        self.bool_or(negative, negative_zero)
    }

    fn signed_constant_like(&mut self, sign: GraphTensor, magnitude: f64) -> GraphTensor {
        let positive = self.constant_like(sign, magnitude);
        let negative = self.constant_like(sign, -magnitude);
        let signbit = self.signbit(sign);
        self.select(signbit, negative, positive)
    }

    pub(crate) fn copy_sign(&mut self, magnitude: GraphTensor, sign: GraphTensor) -> GraphTensor {
        let negative = magnitude * -1.0;
        let signbit = self.signbit(sign);
        self.select(signbit, negative, magnitude)
    }

    fn signed_indicator(&mut self, value: GraphTensor, condition: GraphTensor) -> GraphTensor {
        let one = self.constant_like(value, 1.0);
        let zero = self.constant_like(value, 0.0);
        let magnitude = self.select(condition, one, zero);
        let negative = magnitude * -1.0;
        let signbit = self.signbit(value);
        self.select(signbit, negative, magnitude)
    }

    fn materialize(&mut self, value: GraphTensor) -> GraphTensor {
        if value.shape.is_contiguous() {
            value
        } else {
            let indexes = self.graph.iota(Expression::from('z'), value.dims());
            value.gather(indexes)
        }
    }

    fn reciprocal_overflows(&mut self, value: GraphTensor) -> GraphTensor {
        let reciprocal_inf = self.is_inf(value.reciprocal());
        let value_zero = self.is_zero(value);
        self.bool_and(reciprocal_inf, self.bool_not(value_zero))
    }

    /// HLIR division uses reciprocal, which overflows for subnormal divisors.
    /// Scale both operands into the normal range before dividing in that case.
    fn safe_div(&mut self, numerator: GraphTensor, denominator: GraphTensor) -> GraphTensor {
        let should_scale = self.reciprocal_overflows(denominator);
        let largest = self.constant_like(denominator, float_max(denominator.dtype));
        let one = self.constant_like(denominator, 1.0);
        let scale = self.select(should_scale, largest, one);
        (numerator * scale) / (denominator * scale)
    }

    fn recover_underflow(
        &mut self,
        trigger: GraphTensor,
        current: GraphTensor,
        alternate: GraphTensor,
    ) -> GraphTensor {
        let current_zero = self.is_zero(current);
        let alternate_zero = self.is_zero(alternate);
        let alternate_nan = self.is_nan(alternate);
        let alternate_nonzero = self.bool_not(self.bool_or(alternate_zero, alternate_nan));
        let use_alternate = self.bool_and(trigger, self.bool_and(current_zero, alternate_nonzero));
        self.select(use_alternate, alternate, current)
    }

    fn translate_complex_binary(
        &mut self,
        node: &Node,
        op: BinaryOp,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let mut a = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let mut b = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;

        let (ar, br) = broadcast_binary(a.real, b.real);
        let (ai, bi) = broadcast_binary(a.imag, b.imag);
        a.real = ar;
        a.imag = ai;
        b.real = br;
        b.imag = bi;

        if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            // ATen implements complex add/sub as `a +/- alpha*b`, including
            // the implicit alpha=1. Going through complex multiplication is
            // observable for IEEE values (`0*inf`), so preserve it here.
            b = self.scale_addend(node, b)?;
        }
        self.apply_complex_binary(op, a, b)
    }

    fn translate_complex_scalar_binary(
        &mut self,
        node: &Node,
        op: BinaryOp,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let a = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let (scalar_real, scalar_imag) = self.complex_scalar_arg(node, 1)?;
        let mut scalar = self.complex_constant_like(a.real, scalar_real, scalar_imag, dtype);
        if matches!(op, BinaryOp::Add | BinaryOp::Sub) {
            scalar = self.scale_addend(node, scalar)?;
        }
        self.apply_complex_binary(op, a, scalar)
    }

    fn scale_component(&mut self, value: GraphTensor, scale: f64) -> GraphTensor {
        value * self.constant_like(value, scale)
    }

    /// Smith's finite complex division plus the C99 recovery cases for zero
    /// and infinite operands. Selection is structural so unused NaNs cannot
    /// contaminate the chosen branch.
    fn stable_complex_div(&mut self, a: ComplexTensor, b: ComplexTensor) -> Result<ComplexTensor> {
        let dtype = a.torch_dtype;
        let b_real_abs = self.real_abs(b.real);
        let b_imag_abs = self.real_abs(b.imag);
        let choose_real = b_real_abs.ge(b_imag_abs);
        let choose_imag = self.bool_not(choose_real);
        let one = self.constant_like(b.real, 1.0);
        let zero = self.constant_like(b.real, 0.0);
        let branch_real = self.select(choose_real, b.real, one);
        let branch_imag = self.select(choose_real, b.imag, zero);
        let branch2_imag = self.select(choose_imag, b.imag, one);
        let branch2_real = self.select(choose_imag, b.real, zero);

        let ratio_real = self.safe_div(branch_imag, branch_real);
        let denom_real = branch_real + branch_imag * ratio_real;
        let scale_real = denom_real.reciprocal();
        let out_real_a = (a.real + a.imag * ratio_real) * scale_real;
        let out_imag_a = (a.imag - a.real * ratio_real) * scale_real;

        let ratio_imag = self.safe_div(branch2_real, branch2_imag);
        let denom_imag = branch2_imag + branch2_real * ratio_imag;
        let scale_imag = denom_imag.reciprocal();
        let out_real_b = (a.real * ratio_imag + a.imag) * scale_imag;
        let out_imag_b = (a.imag * ratio_imag - a.real) * scale_imag;

        let mut result = ComplexTensor::new(
            self.select(choose_real, out_real_a, out_real_b),
            self.select(choose_real, out_imag_a, out_imag_b),
            dtype,
        );

        // The Smith form can round a subnormal numerator to zero even when
        // the direct form retains one representable unit. Use the direct form
        // only to recover such a lost nonzero component.
        let real_subnormal = self.reciprocal_overflows(a.real);
        let imag_subnormal = self.reciprocal_overflows(a.imag);
        let numerator_subnormal = self.bool_or(real_subnormal, imag_subnormal);
        let direct_denom = b.real * b.real + b.imag * b.imag;
        let direct_real = (a.real * b.real + a.imag * b.imag) / direct_denom;
        let direct_imag = (a.imag * b.real - a.real * b.imag) / direct_denom;
        result.real = self.recover_underflow(numerator_subnormal, result.real, direct_real);
        result.imag = self.recover_underflow(numerator_subnormal, result.imag, direct_imag);

        let b_real_inf = self.is_inf(b.real);
        let b_imag_inf = self.is_inf(b.imag);
        let denominator_inf = self.bool_or(b_real_inf, b_imag_inf);
        let denominator_one_inf = b_real_inf.ne(b_imag_inf);
        let a_real_inf = self.is_inf(a.real);
        let a_imag_inf = self.is_inf(a.imag);
        let numerator_inf = self.bool_or(a_real_inf, a_imag_inf);

        // Finite / infinite: normalize the infinite denominator direction,
        // then multiply by +0. This preserves the signed-zero quadrant.
        let c = self.signed_indicator(b.real, b_real_inf);
        let d = self.signed_indicator(b.imag, b_imag_inf);
        let zero = self.constant_like(a.real, 0.0);
        let inf_den_real = zero * (a.real * c + a.imag * d);
        let inf_den_imag = zero * (a.imag * c - a.real * d);
        let denominator_inf_only = {
            let numerator_not_inf = self.bool_not(numerator_inf);
            self.bool_and(denominator_one_inf, numerator_not_inf)
        };
        result.real = self.select(denominator_inf_only, inf_den_real, result.real);
        result.imag = self.select(denominator_inf_only, inf_den_imag, result.imag);

        // Infinite / finite: normalize the numerator direction and restore
        // the infinite scale after the finite multiply.
        let ar = self.signed_indicator(a.real, a_real_inf);
        let ai = self.signed_indicator(a.imag, a_imag_inf);
        let infinity = self.constant_like(a.real, f64::INFINITY);
        let inf_num_real = infinity * (ar * b.real + ai * b.imag);
        let inf_num_imag = infinity * (ai * b.real - ar * b.imag);
        let numerator_inf_only = {
            let denominator_not_inf = self.bool_not(denominator_inf);
            let both_nan = {
                let real_nan = self.is_nan(result.real);
                let imag_nan = self.is_nan(result.imag);
                self.bool_and(real_nan, imag_nan)
            };
            let inf_over_finite = self.bool_and(numerator_inf, denominator_not_inf);
            self.bool_and(inf_over_finite, both_nan)
        };
        result.real = self.select(numerator_inf_only, inf_num_real, result.real);
        result.imag = self.select(numerator_inf_only, inf_num_imag, result.imag);

        // Division by signed complex zero follows libc/compiler-rt: the sign
        // of each denominator component controls the matching output lane.
        let b_real_zero = self.is_zero(b.real);
        let b_imag_zero = self.is_zero(b.imag);
        let denominator_zero = self.bool_and(b_real_zero, b_imag_zero);
        let real_inf = self.signed_constant_like(b.real, f64::INFINITY);
        let imag_inf = self.signed_constant_like(b.imag, f64::INFINITY);
        result.real = self.select(denominator_zero, real_inf * a.real, result.real);
        result.imag = self.select(denominator_zero, imag_inf * a.imag, result.imag);
        Ok(result)
    }

    /// Scaled hypot avoids finite overflow/underflow and explicitly gives
    /// infinity precedence over NaN, matching PyTorch/libc hypot semantics.
    fn complex_abs(&mut self, value: ComplexTensor) -> GraphTensor {
        let real = self.real_abs(value.real);
        let imag = self.real_abs(value.imag);
        let real_is_large = real.ge(imag);
        let large = self.select(real_is_large, real, imag);
        let small = self.select(real_is_large, imag, real);
        let one = self.constant_like(large, 1.0);
        let large_is_zero = self.is_zero(large);
        let safe_large = self.select(large_is_zero, one, large);
        let ratio = self.safe_div(small, safe_large);
        let finite = large * (ratio * ratio + self.constant_like(ratio, 1.0)).sqrt();
        let real_inf = self.is_inf(value.real);
        let imag_inf = self.is_inf(value.imag);
        let any_inf = self.bool_or(real_inf, imag_inf);
        let infinity = self.constant_like(finite, f64::INFINITY);
        self.select(any_inf, infinity, finite)
    }

    /// Compute complex acos and acosh without introducing a complex HLIR type.
    ///
    /// With `r = hypot(x + 1, y)` and `s = hypot(x - 1, y)`, define
    /// `alpha = (r + s) / 2` and `beta = (r - s) / 2 = x / alpha`.
    /// Then
    ///
    /// `acos(x + iy)  = acos(beta) - i*copysign(acosh(alpha), y)`
    /// `acosh(x + iy) = acosh(alpha) + i*copysign(acos(beta), y)`.
    ///
    /// This form uses only real HLIR primitives, preserves PyTorch's branch
    /// choice through the sign bit of `y`, and avoids the cancellation and
    /// overflow of the textbook complex log/sqrt identities.
    fn complex_acos_acosh(
        &mut self,
        value: ComplexTensor,
    ) -> (ComplexTensor, ComplexTensor, ComplexTensor) {
        let one = self.constant_like(value.real, 1.0);
        let plus_one = ComplexTensor::new(value.real + one, value.imag, value.torch_dtype);
        let minus_one = ComplexTensor::new(value.real - one, value.imag, value.torch_dtype);
        let r = self.complex_abs(plus_one);
        let s = self.complex_abs(minus_one);

        // Halve before adding so finite values near the dtype maximum do not
        // overflow merely while forming r + s.
        let mut alpha = r * 0.5 + s * 0.5;
        let one = self.constant_like(alpha, 1.0);
        alpha = self.select(alpha.lt(one), one, alpha);

        // x / alpha is algebraically (r - s) / 2 but remains accurate when
        // r and s are close. Restore the limiting direction for infinities,
        // where the quotient would otherwise be inf / inf.
        let mut beta = value.real / alpha;
        let real_inf = self.is_inf(value.real);
        let imag_inf = self.is_inf(value.imag);
        let real_nan = self.is_nan(value.real);
        let imag_nan = self.is_nan(value.imag);
        let both_inf = self.bool_and(real_inf, imag_inf);
        let real_inf_only =
            self.bool_and(real_inf, self.bool_not(self.bool_or(imag_inf, imag_nan)));
        let imag_inf_only =
            self.bool_and(imag_inf, self.bool_not(self.bool_or(real_inf, real_nan)));
        let diagonal = self.signed_constant_like(value.real, std::f64::consts::FRAC_1_SQRT_2);
        beta = self.select(both_inf, diagonal, beta);
        let real_axis = self.signed_constant_like(value.real, 1.0);
        beta = self.select(real_inf_only, real_axis, beta);
        let zero = self.constant_like(beta, 0.0);
        beta = self.select(imag_inf_only, zero, beta);

        // Roundoff can move finite alpha/beta a few ulps outside their exact
        // domains. Clamp only ordered values; NaNs continue to propagate.
        let negative_one = self.constant_like(beta, -1.0);
        beta = self.select(beta.lt(negative_one), negative_one, beta);
        beta = self.select(beta.gt(one), one, beta);

        let acos_beta = self.real_acos(beta);
        let acosh_alpha = self.real_acosh(alpha);
        let signed_acosh = self.copy_sign(acosh_alpha, value.imag);
        // When y is tiny, alpha rounds to exactly one before acosh and loses
        // the imaginary component. Since beta = sin(real(result)), recover
        // sinh(imag(result)) = y / sqrt(1 - beta^2). Keep the alpha/acosh form
        // at beta=+/-1, where it carries the real-axis branch cut.
        let cosine = (one - beta.square()).sqrt();
        let stable_signed_acosh = self.real_asinh(value.imag / cosine);
        let well_conditioned = self.constant_like(cosine, 0.25);
        let signed_acosh = self.select(
            cosine.gt(well_conditioned),
            stable_signed_acosh,
            signed_acosh,
        );
        let signed_acos = self.copy_sign(acos_beta, value.imag);

        let mut acos_real = acos_beta;
        let mut acos_imag = signed_acosh * -1.0;

        // C99/PyTorch special cases not defined by the limiting beta ratio.
        // acos(+-0 + iNaN) keeps the known real part pi/2, while an infinite
        // real component with NaN imaginary component takes the infinity sign
        // from the real axis.
        let real_zero = self.is_zero(value.real);
        let zero_with_nan_imag = self.bool_and(real_zero, imag_nan);
        let half_pi = self.constant_like(acos_real, std::f64::consts::FRAC_PI_2);
        acos_real = self.select(zero_with_nan_imag, half_pi, acos_real);
        let real_inf_with_nan_imag = self.bool_and(real_inf, imag_nan);
        let signed_infinity = self.signed_constant_like(value.real, f64::INFINITY);
        acos_imag = self.select(real_inf_with_nan_imag, signed_infinity, acos_imag);

        // Use the direct real asin approximation rather than pi/2 - acos.
        // The subtraction amplifies the existing acos approximation error for
        // small real components. Derive the imaginary lane from the finalized
        // acos lane so its C99 infinity/NaN special cases remain identical.
        let mut asin_real = self.real_asin(beta);
        asin_real = self.select(zero_with_nan_imag, value.real, asin_real);
        let asin_imag = acos_imag * -1.0;

        (
            ComplexTensor::new(acos_real, acos_imag, value.torch_dtype),
            ComplexTensor::new(acosh_alpha, signed_acos, value.torch_dtype),
            ComplexTensor::new(asin_real, asin_imag, value.torch_dtype),
        )
    }

    fn complex_asin(&mut self, value: ComplexTensor) -> ComplexTensor {
        self.complex_acos_acosh(value).2
    }

    fn complex_asinh(&mut self, value: ComplexTensor) -> ComplexTensor {
        // asinh(z) = -i asin(i z)
        let rotated = ComplexTensor::new(value.imag * -1.0, value.real, value.torch_dtype);
        let asin = self.complex_asin(rotated);
        ComplexTensor::new(asin.imag, asin.real * -1.0, value.torch_dtype)
    }

    fn complex_exp(&mut self, value: ComplexTensor) -> ComplexTensor {
        let scale = self.real_exp(value.real);
        ComplexTensor::new(
            scale * self.real_cos(value.imag),
            scale * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_sin(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.real_cosh(value.imag);
        let sinh = self.real_sinh(value.imag);
        ComplexTensor::new(
            value.real.sin() * cosh,
            self.real_cos(value.real) * sinh,
            value.torch_dtype,
        )
    }

    fn complex_sinh(&mut self, value: ComplexTensor) -> ComplexTensor {
        let sinh = self.real_sinh(value.real);
        let cosh = self.real_cosh(value.real);
        ComplexTensor::new(
            sinh * self.real_cos(value.imag),
            cosh * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_sqrt(&mut self, value: ComplexTensor) -> ComplexTensor {
        // Stable principal square root. Choosing the formula by the real
        // component avoids cancellation in `hypot(z) +/- real(z)`.
        let magnitude = self.complex_abs(value);
        let half = self.constant_like(magnitude, 0.5);
        let real_negative = self.signbit(value.real);

        let positive_real = (magnitude * half + value.real * half).sqrt();
        let positive_denom = positive_real * 2.0;
        let positive_imag = self.safe_div(value.imag, positive_denom);

        let negative_imag_magnitude = (magnitude * half - value.real * half).sqrt();
        let negative_imag = self.copy_sign(negative_imag_magnitude, value.imag);
        let negative_denom = negative_imag_magnitude * 2.0;
        let imag_magnitude = self.real_abs(value.imag);
        let negative_real = self.safe_div(imag_magnitude, negative_denom);

        let mut real = self.select(real_negative, negative_real, positive_real);
        let mut imag = self.select(real_negative, negative_imag, positive_imag);

        // Both signed complex zeros need an explicit branch because the
        // quotient formulas above contain 0/0. Preserve the input imaginary
        // sign, as required by the principal branch.
        let magnitude_zero = self.is_zero(magnitude);
        let zero = self.constant_like(real, 0.0);
        let signed_zero = self.copy_sign(zero, value.imag);
        real = self.select(magnitude_zero, zero, real);
        imag = self.select(magnitude_zero, signed_zero, imag);

        // An infinite imaginary component has infinite magnitude in both
        // output lanes. The quotient branch would otherwise form inf/inf.
        let imag_inf = self.is_inf(value.imag);
        let infinity = self.constant_like(real, f64::INFINITY);
        let signed_infinity = self.copy_sign(infinity, value.imag);
        real = self.select(imag_inf, infinity, real);
        imag = self.select(imag_inf, signed_infinity, imag);

        ComplexTensor::new(real, imag, value.torch_dtype)
    }

    fn complex_pow(&mut self, base: ComplexTensor, exponent: ComplexTensor) -> ComplexTensor {
        let logarithm = self.complex_log(base);
        let exponent_imag_zero = self.is_zero(exponent.imag);
        let general = self.complex_mul(exponent, logarithm);
        let real_exponent = ComplexTensor::new(
            exponent.real * logarithm.real,
            exponent.real * logarithm.imag,
            base.torch_dtype,
        );
        let product = ComplexTensor::new(
            self.select(exponent_imag_zero, real_exponent.real, general.real),
            self.select(exponent_imag_zero, real_exponent.imag, general.imag),
            base.torch_dtype,
        );
        let result = self.complex_exp(product);

        // ATen defines z**0 as exactly 1+0j for every z, including zero,
        // infinities, and NaNs. Select structurally so the unused logarithm
        // branch cannot contaminate that result.
        let real_zero = self.is_zero(exponent.real);
        let imag_zero = self.is_zero(exponent.imag);
        let exponent_zero = self.bool_and(real_zero, imag_zero);
        let one = self.constant_like(result.real, 1.0);
        let zero = self.constant_like(result.imag, 0.0);
        ComplexTensor::new(
            self.select(exponent_zero, one, result.real),
            self.select(exponent_zero, zero, result.imag),
            base.torch_dtype,
        )
    }

    fn complex_cos(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.real_cosh(value.imag);
        let sinh = self.real_sinh(value.imag);
        ComplexTensor::new(
            self.real_cos(value.real) * cosh,
            value.real.sin() * sinh * -1.0,
            value.torch_dtype,
        )
    }

    fn complex_cosh(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.real_cosh(value.real);
        let sinh = self.real_sinh(value.real);
        ComplexTensor::new(
            cosh * self.real_cos(value.imag),
            sinh * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_log(&mut self, value: ComplexTensor) -> ComplexTensor {
        let magnitude = self.complex_abs(value);
        let angle = self.real_atan2(value.imag, value.real);
        ComplexTensor::new(magnitude.log(), angle, value.torch_dtype)
    }

    pub(crate) fn real_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let ratio = y / x;
        let mut angle = self.real_atan(ratio);
        let x_negative = self.signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.copy_sign(pi, y);
        angle = self.select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = self.bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.select(x_negative, three_quarters, quarter);
        let infinite_angle = self.copy_sign(infinite_angle, y);
        angle = self.select(both_inf, infinite_angle, angle);

        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);
        let both_zero = self.bool_and(x_zero, y_zero);
        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.copy_sign(zero, y);
        let zero_angle = self.select(x_negative, signed_pi, signed_zero);
        self.select(both_zero, zero_angle, angle)
    }

    fn complex_atan(&mut self, value: ComplexTensor) -> ComplexTensor {
        // atan(z) = i/2 * (log(1 - i z) - log(1 + i z))
        let one = self.constant_like(value.real, 1.0);
        let minus_iz = ComplexTensor::new(one + value.imag, value.real * -1.0, value.torch_dtype);
        let plus_iz = ComplexTensor::new(one - value.imag, value.real, value.torch_dtype);
        let a = self.complex_log(minus_iz);
        let b = self.complex_log(plus_iz);
        ComplexTensor::new(
            (a.imag - b.imag) * -0.5,
            (a.real - b.real) * 0.5,
            value.torch_dtype,
        )
    }

    fn complex_atanh(&mut self, value: ComplexTensor) -> ComplexTensor {
        // atanh(z) = 1/2 * (log(1 + z) - log(1 - z))
        let one = self.constant_like(value.real, 1.0);
        let plus = ComplexTensor::new(one + value.real, value.imag, value.torch_dtype);
        let minus = ComplexTensor::new(one - value.real, value.imag * -1.0, value.torch_dtype);
        let plus = self.complex_log(plus);
        let minus = self.complex_log(minus);
        ComplexTensor::new(
            (plus.real - minus.real) * 0.5,
            (plus.imag - minus.imag) * 0.5,
            value.torch_dtype,
        )
    }

    fn reshape_complex_component(
        &mut self,
        value: GraphTensor,
        shape: Vec<Expression>,
    ) -> GraphTensor {
        reshape_tensor(self.materialize(value), shape)
    }

    fn expand_complex_component(&self, mut value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let raw: Vec<Expression> = if let Ok(sizes) = self.get_ints_arg(node, 1) {
            sizes.into_iter().map(Expression::from).collect()
        } else {
            self.get_exprs_arg(node, 1)?
        };
        anyhow::ensure!(
            raw.len() >= value.shape.len(),
            "complex expand rank mismatch"
        );
        let offset = raw.len() - value.shape.len();
        for _ in 0..offset {
            value = value.unsqueeze(0);
        }
        let neg_one = Expression::from(-1i32);
        let target: Vec<Expression> = raw
            .into_iter()
            .enumerate()
            .map(|(axis, dim)| {
                if dim == neg_one {
                    value.shape.dims[axis]
                } else {
                    dim
                }
            })
            .collect();
        value.shape.expand(target);
        Ok(value)
    }

    fn repeat_complex_component(
        &self,
        mut value: GraphTensor,
        repeats: &[i64],
    ) -> Result<GraphTensor> {
        anyhow::ensure!(
            repeats.len() >= value.shape.len(),
            "complex repeat rank mismatch"
        );
        anyhow::ensure!(
            repeats.iter().all(|&r| r >= 1),
            "repeat counts must be >= 1"
        );
        for _ in 0..(repeats.len() - value.shape.len()) {
            value = value.unsqueeze(0);
        }
        Ok(value.repeat(repeats.iter().map(|&r| r as usize).collect::<Vec<_>>()))
    }

    fn slice_complex_component(&self, value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let dim = normalize_dim(self.get_int_arg(node, 1).unwrap_or(0), value.shape.len());
        let start = self
            .get_expr_arg(node, 2)
            .unwrap_or_else(|_| Expression::from(0usize));
        let start = normalize_slice_bound(start, value.shape.dims[dim]);
        if self.get_int_arg(node, 3).is_ok_and(|end| end == i64::MAX) {
            Ok(if start.to_usize() == Some(0) {
                value
            } else {
                value.slice_along(start.., dim)
            })
        } else {
            let end = normalize_slice_bound(self.get_expr_arg(node, 3)?, value.shape.dims[dim]);
            Ok(value.slice_along(start..end, dim))
        }
    }

    fn translate_complex_split_with_sizes(&mut self, node: &Node) -> Result<()> {
        let value = self.get_complex_input(node, 0)?;
        let sizes = self.get_ints_arg(node, 1)?;
        let dim = normalize_dim(
            if node.inputs.len() > 2 {
                self.get_int_arg(node, 2).unwrap_or(0)
            } else {
                0
            },
            value.real.shape.len(),
        );
        let output_names: Vec<String> = node
            .outputs
            .iter()
            .flat_map(|output| {
                if let Some(tensor) = output.as_tensor.as_ref() {
                    vec![tensor.name.clone()]
                } else if let Some(tensors) = output.as_tensors.as_ref() {
                    tensors.iter().map(|tensor| tensor.name.clone()).collect()
                } else {
                    vec![]
                }
            })
            .collect();

        anyhow::ensure!(
            sizes.len() == output_names.len(),
            "complex split_with_sizes produced {} outputs for {} sizes",
            output_names.len(),
            sizes.len()
        );

        let mut offset = 0usize;
        for (name, size) in output_names.into_iter().zip(sizes) {
            let size = usize::try_from(size)
                .context("complex split_with_sizes sizes must be nonnegative")?;
            let end = offset
                .checked_add(size)
                .context("complex split_with_sizes offset overflow")?;
            self.store_complex(
                &name,
                ComplexTensor::new(
                    value.real.slice_along(offset..end, dim),
                    value.imag.slice_along(offset..end, dim),
                    value.torch_dtype,
                ),
            );
            offset = end;
        }
        Ok(())
    }

    fn translate_complex_index_put(&mut self, node: &Node) -> Result<ComplexTensor> {
        let data = self.get_complex_input(node, 0)?;
        let values = self.get_complex_input(node, 2)?.cast(data.torch_dtype);
        let entries = node.inputs[1]
            .arg
            .as_optional_tensors()
            .context("complex index_put requires optional tensor indices")?;

        let mut axis_and_name = None;
        for (axis, entry) in entries.iter().enumerate() {
            if let OptionalTensorEntry::Tensor(tensor) = entry {
                anyhow::ensure!(
                    axis_and_name.is_none(),
                    "complex index_put supports one tensor index"
                );
                axis_and_name = Some((axis, tensor.as_tensor.name.as_str()));
            }
        }
        let (axis, index_name) =
            axis_and_name.context("complex index_put requires one tensor index")?;
        let index = self.get_tensor(index_name)?.cast(DType::Int);
        anyhow::ensure!(
            index.shape.len() == 1,
            "complex index_put requires a 1-D tensor index"
        );
        anyhow::ensure!(
            values.real.shape.len() == data.real.shape.len(),
            "complex index_put does not support value broadcasting"
        );
        anyhow::ensure!(
            axis < values.real.shape.len(),
            "complex index_put axis is outside the value rank"
        );

        let mut expanded_index = index;
        for (dim, &size) in values.real.dims().iter().enumerate().take(axis) {
            expanded_index = expanded_index.expand_dim(dim, size);
        }
        for (dim, &size) in values.real.dims().iter().enumerate().skip(axis + 1) {
            expanded_index = expanded_index.expand_dim(dim, size);
        }

        let accumulate = node
            .inputs
            .get(3)
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false);
        let update = |component_data, component_values| {
            if accumulate {
                super::movement_dynamic::pt2_scatter_elements_reduce(
                    component_data,
                    expanded_index,
                    component_values,
                    axis,
                    ScatterReduction::Add,
                )
            } else {
                Ok(super::movement_dynamic::pt2_scatter_elements(
                    component_data,
                    expanded_index,
                    component_values,
                    axis,
                ))
            }
        };
        Ok(ComplexTensor::new(
            update(data.real, values.real)?,
            update(data.imag, values.imag)?,
            data.torch_dtype,
        ))
    }

    fn select_complex_component(&self, value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let dim = normalize_dim(self.get_int_arg(node, 1)?, value.shape.len());
        let raw_index = self.get_int_arg(node, 2)?;
        let index = if raw_index < 0 {
            let size = value.shape.dims[dim]
                .to_usize()
                .context("negative complex select index requires a concrete dimension")?;
            (size as i64 + raw_index) as usize
        } else {
            raw_index as usize
        };
        Ok(value.slice_along(index..index + 1, dim).squeeze(dim))
    }

    fn translate_complex_cat(&mut self, node: &Node, output_name: &str) -> Result<()> {
        let names = node.inputs[0]
            .arg
            .as_tensors()
            .context("complex cat is missing its tensor list")?;
        anyhow::ensure!(
            !names.is_empty(),
            "complex cat requires at least one tensor"
        );
        let dtype = self.output_complex_dtype(output_name)?;
        let values = names
            .iter()
            .map(|name| self.value_as_complex(&name.name, dtype))
            .collect::<Result<Vec<_>>>()?;
        let dim = node
            .inputs
            .iter()
            .find(|input| input.name != "tensors")
            .and_then(|input| input.arg.as_int())
            .unwrap_or(0);
        let dim = normalize_dim(dim, values[0].real.shape.len());
        let real = values[1..].iter().fold(values[0].real, |acc, value| {
            acc.concat_along(value.real, dim)
        });
        let imag = values[1..].iter().fold(values[0].imag, |acc, value| {
            acc.concat_along(value.imag, dim)
        });
        self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
        Ok(())
    }

    fn fft_dims(&self, node: &Node, rank: usize) -> Result<Vec<usize>> {
        anyhow::ensure!(rank > 0, "{} requires a non-scalar input", node.target);
        let mut axes = Vec::new();
        for dim in self.get_ints_arg(node, 1)? {
            anyhow::ensure!(
                dim >= -(rank as i64) && dim < rank as i64,
                "FFT dimension {dim} is out of range for rank {rank}"
            );
            let axis = normalize_dim(dim, rank);
            anyhow::ensure!(!axes.contains(&axis), "FFT dimensions must be unique");
            axes.push(axis);
        }
        Ok(axes)
    }

    /// Apply one ordinary complex DFT along `axis` using only existing HLIR
    /// arithmetic, trigonometric, movement, matmul, and reduction primitives.
    /// The Fourier matrix has expression-shaped `[N, N]` extents, so dynamic
    /// lengths do not change graph topology.
    fn complex_dft_axis(
        &mut self,
        value: ComplexTensor,
        axis: usize,
        forward: bool,
    ) -> ComplexTensor {
        let component_dtype = value.real.dtype;
        let original_shape = value.real.dims();
        let rank = original_shape.len();
        let length = original_shape[axis];

        let mut permutation = (0..rank).filter(|&dim| dim != axis).collect::<Vec<_>>();
        permutation.push(axis);
        let mut inverse_permutation = vec![0; rank];
        for (position, &original_axis) in permutation.iter().enumerate() {
            inverse_permutation[original_axis] = position;
        }

        // A direct F32 Fourier matrix followed by an F32 reduction accumulates
        // noticeably more error than ATen's FFT kernels. Compose the DFT in
        // F64 and round once at the axis boundary. Cast, trig, matmul, and
        // reduction are all existing HLIR primitives.
        let real = self.materialize(value.real.cast(DType::F64).permute(permutation.clone()));
        let imag = self.materialize(value.imag.cast(DType::F64).permute(permutation));
        let permuted_shape = real.dims();
        let batch_shape = &permuted_shape[..rank - 1];
        let batch = product_of_dims(batch_shape.iter().copied());
        let real = reshape_tensor(real, vec![batch, length]);
        let imag = reshape_tensor(imag, vec![batch, length]);

        // Row-major matrix coordinates: input frequency/time index `j` is
        // z/N and output index `k` is z%N. Reducing j*k modulo N bounds the
        // trigonometric arguments to [0, 2pi), improving large-N accuracy.
        let z = Expression::from('z');
        let input_index = z / length;
        let output_index = z % length;
        let phase_index = (input_index * output_index) % length;
        let phase = self
            .graph
            .iota(phase_index, vec![length, length])
            .cast(real.dtype);
        let denominator = self
            .graph
            .constant(length)
            .cast(real.dtype)
            .expand_rhs(phase.shape);
        let tau = self.constant_like(phase, std::f64::consts::TAU);
        let angle = phase * tau / denominator;
        let cosine = self.real_cos(angle);
        let sine = angle.sin();

        let real_cosine = real.matmul(cosine);
        let imag_cosine = imag.matmul(cosine);
        let real_sine = real.matmul(sine);
        let imag_sine = imag.matmul(sine);
        let (out_real, out_imag) = if forward {
            (real_cosine + imag_sine, imag_cosine - real_sine)
        } else {
            (real_cosine - imag_sine, imag_cosine + real_sine)
        };

        let mut permuted_output_shape = batch_shape.to_vec();
        permuted_output_shape.push(length);
        let out_real = reshape_tensor(out_real, permuted_output_shape.clone())
            .permute(inverse_permutation.clone())
            .cast(component_dtype);
        let out_imag = reshape_tensor(out_imag, permuted_output_shape)
            .permute(inverse_permutation)
            .cast(component_dtype);
        ComplexTensor::new(out_real, out_imag, value.torch_dtype)
    }

    /// Finish the final one-sided dimension of a complex-to-real inverse DFT.
    /// Endpoint imaginary values are explicitly ignored, matching ATen for DC
    /// and (when N is even) Nyquist. Interior bins contribute their conjugate
    /// partner through a factor of two.
    fn complex_to_real_dft_axis(
        &mut self,
        value: ComplexTensor,
        axis: usize,
        output_length: Expression,
        output_dtype: DType,
    ) -> GraphTensor {
        let input_shape = value.real.dims();
        let rank = input_shape.len();
        let input_length = input_shape[axis];

        let mut permutation = (0..rank).filter(|&dim| dim != axis).collect::<Vec<_>>();
        permutation.push(axis);
        let mut inverse_permutation = vec![0; rank];
        for (position, &original_axis) in permutation.iter().enumerate() {
            inverse_permutation[original_axis] = position;
        }

        let real = self.materialize(value.real.cast(DType::F64).permute(permutation.clone()));
        let imag = self.materialize(value.imag.cast(DType::F64).permute(permutation));
        let permuted_shape = real.dims();
        let batch_shape = &permuted_shape[..rank - 1];
        let batch = product_of_dims(batch_shape.iter().copied());
        let real = reshape_tensor(real, vec![batch, input_length]);
        let imag = reshape_tensor(imag, vec![batch, input_length]);

        let z = Expression::from('z');
        let frequency = z / output_length;
        let time = z % output_length;
        let phase_index = (frequency * time) % output_length;
        let matrix_shape = vec![input_length, output_length];
        let phase = self
            .graph
            .iota(phase_index, matrix_shape.clone())
            .cast(DType::F64);
        let denominator = self
            .graph
            .constant(output_length)
            .cast(DType::F64)
            .expand_rhs(phase.shape);
        let tau = self.constant_like(phase, std::f64::consts::TAU);
        let angle = phase * tau / denominator;
        let cosine = self.real_cos(angle);
        let sine = angle.sin();

        let frequency = Expression::from('z') / output_length;
        let interior = frequency.gte(1) * (frequency * 2).lt(output_length);
        let real_coefficient = self
            .graph
            .iota(1 + interior, matrix_shape.clone())
            .cast(DType::F64);
        let imag_coefficient = self.graph.iota(interior * 2, matrix_shape).cast(DType::F64);
        let out = real.matmul(cosine * real_coefficient) - imag.matmul(sine * imag_coefficient);

        let mut permuted_output_shape = batch_shape.to_vec();
        permuted_output_shape.push(output_length);
        reshape_tensor(out, permuted_output_shape)
            .permute(inverse_permutation)
            .cast(output_dtype)
    }

    fn fft_normalization(&self, node: &Node) -> Result<i64> {
        let normalization = self.get_int_arg(node, 2)?;
        anyhow::ensure!(
            (0..=2).contains(&normalization),
            "Unsupported FFT normalization code {normalization}"
        );
        Ok(normalization)
    }

    fn normalize_complex_fft(
        &mut self,
        value: ComplexTensor,
        transformed_elements: Expression,
        normalization: i64,
    ) -> ComplexTensor {
        if normalization == 0 {
            return value;
        }
        value.map(|component| {
            let mut divisor = self
                .graph
                .constant(transformed_elements)
                .cast(component.dtype);
            if normalization == 1 {
                divisor = divisor.sqrt();
            }
            component / divisor.expand_rhs(component.shape)
        })
    }

    fn normalize_real_fft(
        &mut self,
        value: GraphTensor,
        transformed_elements: Expression,
        normalization: i64,
    ) -> GraphTensor {
        if normalization == 0 {
            return value;
        }
        let mut divisor = self.graph.constant(transformed_elements).cast(value.dtype);
        if normalization == 1 {
            divisor = divisor.sqrt();
        }
        value / divisor.expand_rhs(value.shape)
    }

    fn translate_fft_c2c(&mut self, node: &Node, output_name: &str) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let mut value = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let axes = self.fft_dims(node, value.real.shape.len())?;
        let transformed_elements =
            product_of_dims(axes.iter().map(|&axis| value.real.dims()[axis]));
        let forward = self.get_bool_arg(node, 3)?;
        for axis in axes {
            value = self.complex_dft_axis(value, axis, forward);
        }
        let normalization = self.fft_normalization(node)?;
        Ok(self.normalize_complex_fft(value, transformed_elements, normalization))
    }

    fn translate_fft_r2c(&mut self, node: &Node, output_name: &str) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let mut value = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let axes = self.fft_dims(node, value.real.shape.len())?;
        let transformed_elements =
            product_of_dims(axes.iter().map(|&axis| value.real.dims()[axis]));
        for &axis in &axes {
            value = self.complex_dft_axis(value, axis, true);
        }
        if self.get_bool_arg(node, 3)? {
            let last_axis = *axes
                .last()
                .context("one-sided r2c FFT requires a dimension")?;
            let one_sided_length = (value.real.dims()[last_axis] / 2 + 1).simplify();
            value = value.map(|component| {
                let mut component = component.slice_along(..one_sided_length, last_axis);
                component.shape.dims[last_axis] = one_sided_length;
                component
            });
        }
        let normalization = self.fft_normalization(node)?;
        Ok(self.normalize_complex_fft(value, transformed_elements, normalization))
    }

    fn translate_fft_c2r(&mut self, node: &Node) -> Result<GraphTensor> {
        let output_dtype = self.output_meta_dtype(node)?;
        let compute_dtype = match output_dtype {
            DType::F32 => TorchDType::ComplexFloat,
            DType::F64 => TorchDType::ComplexDouble,
            DType::F16 => TorchDType::ComplexHalf,
            other => anyhow::bail!("c2r FFT requires a floating output dtype, got {other:?}"),
        };
        let mut value = self.value_as_complex(self.input_value_name(node, 0)?, compute_dtype)?;
        let axes = self.fft_dims(node, value.real.shape.len())?;
        let last_axis = *axes.last().context("c2r FFT requires a dimension")?;
        let output_length = self.get_expr_arg(node, 3)?;
        let transformed_elements = product_of_dims(axes.iter().map(|&axis| {
            if axis == last_axis {
                output_length
            } else {
                value.real.dims()[axis]
            }
        }));

        for &axis in &axes[..axes.len() - 1] {
            value = self.complex_dft_axis(value, axis, false);
        }
        let output = self.complex_to_real_dft_axis(value, last_axis, output_length, output_dtype);
        let normalization = self.fft_normalization(node)?;
        Ok(self.normalize_real_fft(output, transformed_elements, normalization))
    }

    fn translate_complex_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<ComplexTensor> {
        let value = self.get_complex_input(node, 0)?;
        let rank = value.real.shape.len();
        if rank == 0 {
            return Ok(value);
        }
        let dims = self.get_ints_arg(node, 1).ok();
        let axes: Vec<usize> = match dims {
            Some(dims) if !dims.is_empty() => dims
                .into_iter()
                .map(|dim| normalize_dim(dim, rank))
                .collect(),
            _ => (0..rank).collect(),
        };
        let keepdim = node.inputs.len() > 2 && self.get_bool_arg(node, 2).unwrap_or(false);
        let mut result = value.map(|tensor| match op {
            ReductionOp::Sum => tensor.sum(axes.clone()),
            ReductionOp::Mean => tensor.mean(axes.clone()),
            _ => unreachable!("only sum and mean are complex componentwise reductions"),
        });
        if keepdim {
            let mut sorted = axes;
            sorted.sort();
            for axis in sorted {
                result = result.map(|component| component.unsqueeze(axis));
            }
        }
        Ok(result)
    }

    fn translate_complex_var_mean(&mut self, node: &Node) -> Result<(GraphTensor, ComplexTensor)> {
        let value = self.get_complex_input(node, 0)?;
        let axes = self.composed_reduction_axes(node, value.real.shape.len())?;
        let keepdim = node
            .inputs
            .iter()
            .position(|input| input.name == "keepdim")
            .and_then(|index| self.get_bool_arg(node, index).ok())
            .unwrap_or(false);
        let correction = self.variance_correction(node);
        let component_dtype = value.torch_dtype.complex_component_dtype().unwrap();
        let real = value.real.cast(component_dtype);
        let imag = value.imag.cast(component_dtype);
        let n = product_of_dims(axes.iter().map(|&axis| real.dims()[axis]));
        let mean = if axes.is_empty() {
            ComplexTensor::new(real, imag, value.torch_dtype)
        } else {
            ComplexTensor::new(
                real.sum(axes.clone()) / n,
                imag.sum(axes.clone()) / n,
                value.torch_dtype,
            )
        };
        let centered_real = real - mean.real.expand_to_shape_on_axes(real.shape, axes.clone());
        let centered_imag = imag - mean.imag.expand_to_shape_on_axes(imag.shape, axes.clone());
        let numerator =
            (centered_real * centered_real + centered_imag * centered_imag).sum(axes.clone());
        let degrees = self.graph.constant(n).cast(component_dtype)
            - self.floating_scalar(correction, component_dtype);
        let zero = self.floating_scalar(0.0, component_dtype);
        let divisor = degrees.maximum(zero).expand_rhs(numerator.shape);
        let variance = numerator / divisor;
        let variance = self.restore_reduced_dims(variance, &axes, keepdim);
        let mean = ComplexTensor::new(
            self.restore_reduced_dims(mean.real, &axes, keepdim),
            self.restore_reduced_dims(mean.imag, &axes, keepdim),
            mean.torch_dtype,
        );
        Ok((variance, mean))
    }

    fn translate_complex_product(
        &mut self,
        node: &Node,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let mut value = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        if value.real.shape.is_empty() {
            if node.target == "torch.ops.aten.prod.dim_int" {
                let dim = self.get_int_arg(node, 1)?;
                anyhow::ensure!(
                    matches!(dim, -1 | 0),
                    "complex scalar product dimension must be -1 or 0, got {dim}"
                );
            }
            return Ok(value);
        }
        let (axis, keepdim) = if node.target == "torch.ops.aten.prod.dim_int" {
            let axis = normalize_dim(self.get_int_arg(node, 1)?, value.real.shape.len());
            let keepdim = node.inputs.len() > 2 && self.get_bool_arg(node, 2).unwrap_or(false);
            (axis, keepdim)
        } else {
            let dims = value.real.dims();
            let numel = dims.iter().try_fold(1usize, |acc, dim| {
                dim.to_usize()
                    .map(|size| acc * size)
                    .context("full complex product requires concrete dimensions")
            })?;
            value.real = self.reshape_complex_component(value.real, vec![numel.into()]);
            value.imag = self.reshape_complex_component(value.imag, vec![numel.into()]);
            (0, false)
        };

        let axis_size = value.real.shape.dims[axis]
            .to_usize()
            .context("complex product requires a concrete reduction dimension")?;
        if axis_size == 0 {
            let shape = self.output_meta_shape(node)?;
            let shape_tracker = ShapeTracker::new(shape);
            let component_dtype = dtype.complex_component_dtype().unwrap();
            let real_scalar = if component_dtype == DType::F64 {
                self.graph.constant_float64(1.0)
            } else {
                self.graph.constant_float(1.0).cast(component_dtype)
            };
            let imag_scalar = if component_dtype == DType::F64 {
                self.graph.constant_float64(0.0)
            } else {
                self.graph.constant_float(0.0).cast(component_dtype)
            };
            return Ok(ComplexTensor::new(
                real_scalar.expand_rhs(shape_tracker),
                imag_scalar.expand_rhs(shape_tracker),
                dtype,
            ));
        }

        let select = |tensor: GraphTensor, index: usize| {
            tensor.slice_along(index..index + 1, axis).squeeze(axis)
        };
        let mut result = ComplexTensor::new(select(value.real, 0), select(value.imag, 0), dtype);
        for index in 1..axis_size {
            let next_real = select(value.real, index);
            let next_imag = select(value.imag, index);
            result = ComplexTensor::new(
                result.real * next_real - result.imag * next_imag,
                result.real * next_imag + result.imag * next_real,
                dtype,
            );
        }
        if keepdim {
            result.real = result.real.unsqueeze(axis);
            result.imag = result.imag.unsqueeze(axis);
        }
        Ok(result)
    }

    fn translate_complex_matmul(
        &mut self,
        node: &Node,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let a = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let b = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
        Ok(self.complex_matmul(a, b))
    }

    fn translate_complex_addmv(&mut self, node: &Node, output_name: &str) -> Result<ComplexTensor> {
        let output_dtype = self.output_complex_dtype(output_name)?;
        let compute_dtype = if output_dtype == TorchDType::ComplexHalf {
            TorchDType::ComplexFloat
        } else {
            output_dtype
        };
        let input = self.value_as_complex(self.input_value_name(node, 0)?, compute_dtype)?;
        let matrix = self.value_as_complex(self.input_value_name(node, 1)?, compute_dtype)?;
        let vector = self.value_as_complex(self.input_value_name(node, 2)?, compute_dtype)?;
        anyhow::ensure!(matrix.real.shape.len() == 2, "addmv matrix must be rank 2");
        anyhow::ensure!(vector.real.shape.len() == 1, "addmv vector must be rank 1");

        let vector = vector.map(|component| component.unsqueeze(1));
        let product = self
            .complex_matmul(matrix, vector)
            .map(|component| component.squeeze(1));
        let input = self.scale_complex_by_named_scalar(node, "beta", input)?;
        let product = self.scale_complex_by_named_scalar(node, "alpha", product)?;
        Ok(self.add_complex(input, product).cast(output_dtype))
    }

    fn translate_complex_addbmm(
        &mut self,
        node: &Node,
        output_name: &str,
    ) -> Result<ComplexTensor> {
        let output_dtype = self.output_complex_dtype(output_name)?;
        let compute_dtype = if output_dtype == TorchDType::ComplexHalf {
            TorchDType::ComplexFloat
        } else {
            output_dtype
        };
        let input = self.value_as_complex(self.input_value_name(node, 0)?, compute_dtype)?;
        let batch1 = self.value_as_complex(self.input_value_name(node, 1)?, compute_dtype)?;
        let batch2 = self.value_as_complex(self.input_value_name(node, 2)?, compute_dtype)?;
        anyhow::ensure!(batch1.real.shape.len() == 3, "addbmm batch1 must be rank 3");
        anyhow::ensure!(batch2.real.shape.len() == 3, "addbmm batch2 must be rank 3");

        let product = self
            .complex_matmul(batch1, batch2)
            .map(|component| component.sum(0));
        let input = self.scale_complex_by_named_scalar(node, "beta", input)?;
        let product = self.scale_complex_by_named_scalar(node, "alpha", product)?;
        Ok(self.add_complex(input, product).cast(output_dtype))
    }

    fn translate_complex_addmm(&mut self, node: &Node, output_name: &str) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let input = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let a = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
        let b = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
        let beta = node
            .inputs
            .iter()
            .position(|input| input.name == "beta")
            .map(|idx| self.get_float_arg(node, idx))
            .transpose()?
            .unwrap_or(1.0);
        let alpha = node
            .inputs
            .iter()
            .position(|input| input.name == "alpha")
            .map(|idx| self.get_float_arg(node, idx))
            .transpose()?
            .unwrap_or(1.0);
        let mm = self.complex_matmul(a, b);
        let input_real = self.scale_component(input.real, beta);
        let input_imag = self.scale_component(input.imag, beta);
        let mm_real = self.scale_component(mm.real, alpha);
        let mm_imag = self.scale_component(mm.imag, alpha);
        let (input_real, mm_real) = broadcast_binary(input_real, mm_real);
        let (input_imag, mm_imag) = broadcast_binary(input_imag, mm_imag);
        Ok(ComplexTensor::new(
            input_real + mm_real,
            input_imag + mm_imag,
            dtype,
        ))
    }

    fn translate_complex_comparison(&mut self, node: &Node, is_eq: bool) -> Result<GraphTensor> {
        let lhs_name = self.input_value_name(node, 0)?;
        let rhs_name = self.input_value_name(node, 1)?;
        let dtype = self
            .complex_tensors
            .get(lhs_name)
            .or_else(|| self.complex_tensors.get(rhs_name))
            .context("complex comparison has no complex operand")?
            .torch_dtype;
        let lhs = self.value_as_complex(lhs_name, dtype)?;
        let rhs = self.value_as_complex(rhs_name, dtype)?;
        let (lr, rr) = broadcast_binary(lhs.real, rhs.real);
        let (li, ri) = broadcast_binary(lhs.imag, rhs.imag);
        let lr_nan = self.is_nan(lr);
        let rr_nan = self.is_nan(rr);
        let li_nan = self.is_nan(li);
        let ri_nan = self.is_nan(ri);
        let real_nan = self.bool_or(lr_nan, rr_nan);
        let imag_nan = self.bool_or(li_nan, ri_nan);
        let real_eq = self.bool_and(lr.eq(rr), self.bool_not(real_nan));
        let imag_eq = self.bool_and(li.eq(ri), self.bool_not(imag_nan));
        let equal = self.bool_and(real_eq, imag_eq);
        Ok(if is_eq { equal } else { self.bool_not(equal) })
    }
}
