//! Frontend-only complex tensor support.
//!
//! HLIR intentionally has no complex dtype. A complex PT2 value is carried as
//! two ordinary real `GraphTensor`s, and every supported complex ATen op is
//! lowered algebraically before it reaches HLIR. PyTorch's interleaved complex
//! storage is preserved only at graph inputs and outputs.

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use crate::pt2_schema::Node;
use crate::pt2_util::{
    BinaryOp, ReductionOp, broadcast_binary, normalize_dim, normalize_slice_bound, reshape_tensor,
};
use crate::torch_dtype::TorchDType;

use super::Translator;
use super::movement::{
    diagonal_indices, diagonal_scatter_tensor, flip_indices, index_select_tensor,
    normalize_diagonal_dims, normalize_flip_dims, unfold_tensor,
};
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
            "torch.ops.aten.view.default" => {
                let value = self.get_complex_input(node, 0)?;
                let shape = self.output_meta_shape(node)?;
                let value =
                    value.map(|component| self.reshape_complex_component(component, shape.clone()));
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.permute.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dims = self.get_ints_arg(node, 1)?;
                let axes: Vec<usize> = dims
                    .iter()
                    .map(|&dim| normalize_dim(dim, value.real.shape.len()))
                    .collect();
                self.store_complex(
                    output_name,
                    value.map(|component| component.permute(axes.clone())),
                );
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
            "torch.ops.aten.select.int" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.select_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.cat.default" => {
                self.translate_complex_cat(node, output_name)?;
            }
            "torch.ops.aten.sum.dim_IntList" | "torch.ops.aten.sum.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Sum)?;
                self.store_complex(output_name, value);
            }
            "torch.ops.aten.mean.dim" | "torch.ops.aten.mean.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Mean)?;
                self.store_complex(output_name, value);
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
