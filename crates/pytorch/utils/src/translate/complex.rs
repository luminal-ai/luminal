//! Frontend-only complex tensor support.
//!
//! HLIR intentionally has no complex dtype. A complex PT2 value is carried as
//! two ordinary real `GraphTensor`s, and every supported complex ATen op is
//! lowered algebraically before it reaches the recorder. PyTorch's
//! interleaved complex storage is preserved only at graph inputs and outputs.
//!
//! This is the recorder-frontend port of the parked HLIR translator's
//! `translator/complex.rs`. The structural `ComplexTensor`, the target
//! routing, and the arithmetic / elementary / movement / reduction surface
//! are ported here; the remaining target families (FFT/DFT, index/scatter,
//! cumulative scan, variance/product, `linalg_vector_norm`/`dist`,
//! `constant_pad_nd`) still bail explicitly with the ATen target name.

use anyhow::{Context, Result, anyhow, bail};
use luminal::prelude::*;

use crate::dtype::TorchDType;
use crate::pt2_schema::Node;

use super::Translator;
use super::ops::ReductionOp;
use super::util::{
    broadcast_binary, materialize_tensor, normalize_dim, normalize_slice_bound, reshape_tensor,
    resolve_neg1_dim_exprs,
};

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

    /// Split PyTorch's interleaved `[..., 2]` real storage into components.
    pub(crate) fn from_interleaved(
        cx: &mut Graph,
        backing: GraphTensor,
        torch_dtype: TorchDType,
    ) -> Result<Self> {
        let rank = backing.rank();
        let axis = rank
            .checked_sub(1)
            .context("complex interleaved storage must have a component dimension")?;
        anyhow::ensure!(
            backing.dims()[axis].to_usize() == Some(2),
            "complex interleaved storage must end in dimension 2, got {:?}",
            backing.dims()
        );
        let mut shape = backing.dims();
        shape.pop();
        let real = gather_component(cx, backing, &shape, 0);
        let imag = gather_component(cx, backing, &shape, 1);
        Ok(Self::new(real, imag, torch_dtype))
    }

    /// Repack two components into interleaved `[..., 2]` storage.
    pub(crate) fn pack(self, cx: &mut Graph) -> GraphTensor {
        interleave(cx, self.real, self.imag)
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

/// Gather one component (`offset` 0 = real, 1 = imag) out of interleaved
/// `[..., 2]` storage. The output shape is `shape`; the trailing component
/// axis is fixed at `offset` by a constant coordinate.
fn gather_component(
    cx: &mut Graph,
    backing: GraphTensor,
    shape: &[IntExpr],
    offset: i32,
) -> GraphTensor {
    let mut coords: Vec<GraphTensor> = (0..shape.len())
        .map(|axis| cx.iota(shape.to_vec(), |coords| coords[axis]))
        .collect();
    coords.push(cx.iota(shape.to_vec(), |_| IntExpr::from(offset)));
    backing.gather(&coords)
}

/// Store two equally-shaped tensors in a contiguous final extent-2 axis.
/// Scatter is structural, so inactive lanes cannot contaminate the other
/// component with `0 * inf` or `0 * NaN`.
fn interleave(cx: &mut Graph, first: GraphTensor, second: GraphTensor) -> GraphTensor {
    let shape = first.dims();
    let mut packed_shape = shape.clone();
    packed_shape.push(2usize.into());
    let zero = cx
        .iota(packed_shape, |_| IntExpr::from(0))
        .cast(first.dtype);
    let mut even: Vec<GraphTensor> = (0..shape.len())
        .map(|axis| cx.iota(shape.clone(), |coords| coords[axis]))
        .collect();
    even.push(cx.iota(shape.clone(), |_| IntExpr::from(0)));
    let mut odd = even.clone();
    odd[shape.len()] = cx.iota(shape.clone(), |_| IntExpr::from(1));
    let with_real = zero.scatter(&even, first);
    with_real.scatter(&odd, second)
}

fn squeeze_dims(mut tensor: GraphTensor, dims: &[usize]) -> GraphTensor {
    let mut removed = 0;
    for &original_dim in dims {
        let dim = original_dim - removed;
        if tensor.dims()[dim].to_usize() == Some(1) {
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

/// Boolean helpers as free functions so they can be nested without holding a
/// mutable borrow of the translator across the inner call.
fn cx_bool_or(lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    let (lhs, rhs) = (lhs.cast(DType::F32), rhs.cast(DType::F32));
    (lhs + rhs - lhs * rhs).cast(DType::Bool)
}

fn cx_bool_and(lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor {
    let (lhs, rhs) = broadcast_binary(lhs, rhs);
    (lhs.cast(DType::F32) * rhs.cast(DType::F32)).cast(DType::Bool)
}

fn cx_bool_not(value: GraphTensor) -> GraphTensor {
    (1.0f32 - value.cast(DType::F32)).cast(DType::Bool)
}

impl Translator<'_> {
    pub(crate) fn node_uses_complex(&self, node: &Node, output_name: &str) -> bool {
        if self
            .tensor_meta(output_name)
            .ok()
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
        let target = node
            .target
            .strip_prefix("torch.ops.aten.")
            .or_else(|| node.target.strip_prefix("torch.ops."))
            .unwrap_or(&node.target);

        match target {
            // ---- arithmetic ----
            "add.Tensor" | "sub.Tensor" | "mul.Tensor" | "div.Tensor" => {
                let op = match target {
                    "add.Tensor" => BinaryOp::Add,
                    "sub.Tensor" => BinaryOp::Sub,
                    "mul.Tensor" => BinaryOp::Mul,
                    _ => BinaryOp::Div,
                };
                let value = if node.inputs[1].arg.as_value_name().is_some() {
                    self.translate_complex_binary(node, op, output_name)?
                } else {
                    self.translate_complex_scalar_binary(node, op, output_name)?
                };
                self.store_complex(output_name, value);
            }
            "add.Scalar" | "sub.Scalar" | "mul.Scalar" | "div.Scalar" => {
                let op = match target {
                    "add.Scalar" => BinaryOp::Add,
                    "sub.Scalar" => BinaryOp::Sub,
                    "mul.Scalar" => BinaryOp::Mul,
                    _ => BinaryOp::Div,
                };
                let value = self.translate_complex_scalar_binary(node, op, output_name)?;
                self.store_complex(output_name, value);
            }
            "neg.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.store_complex(output_name, value.map(|component| component * -1.0));
            }
            "reciprocal.default" => {
                let value = self.get_complex_input(node, 0)?;
                let one = self.complex_constant_like(value.real, 1.0, 0.0, value.torch_dtype);
                let result = self.stable_complex_div(one, value)?;
                self.store_complex(output_name, result);
            }
            "sqrt.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sqrt(value);
                self.store_complex(output_name, result);
            }
            "rsqrt.default" => {
                let value = self.get_complex_input(node, 0)?;
                let root = self.complex_sqrt(value);
                let one = self.complex_constant_like(root.real, 1.0, 0.0, root.torch_dtype);
                let result = self.stable_complex_div(one, root)?;
                self.store_complex(output_name, result);
            }
            "sigmoid.default" => {
                let value = self.get_complex_input(node, 0)?;
                let negative = value.map(|component| component * -1.0);
                let exponential = self.complex_exp(negative);
                let one = self.complex_constant_like(value.real, 1.0, 0.0, value.torch_dtype);
                let denominator = self.add_complex(one, exponential);
                let result = self.stable_complex_div(one, denominator)?;
                self.store_complex(output_name, result);
            }
            "pow.Tensor_Scalar" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let base = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
                let (real, imag) = self.complex_scalar_arg(node, 1)?;
                let exponent = self.complex_constant_like(base.real, real, imag, dtype);
                let result = self.complex_pow(base, exponent);
                self.store_complex(output_name, result);
            }
            "pow.Tensor_Tensor" => {
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

            // ---- conjugation and aliasing ----
            "_conj.default" | "_conj_physical.default" | "conj_physical.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.store_complex(
                    output_name,
                    ComplexTensor::new(value.real, value.imag * -1.0, value.torch_dtype),
                );
            }
            "resolve_conj.default" | "alias.default" | "clone.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.store_complex(output_name, value.map(materialize_tensor));
            }

            // ---- magnitude / components ----
            "abs.default" => {
                let value = self.get_complex_input(node, 0)?;
                let out = self.complex_abs(value);
                self.values.insert(output_name.to_string(), out);
            }
            "real.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.values.insert(output_name.to_string(), value.real);
            }
            "imag.default" => {
                let value = self.get_complex_input(node, 0)?;
                self.values.insert(output_name.to_string(), value.imag);
            }
            "view_as_real.default" => {
                let value = self.get_complex_input(node, 0)?;
                let packed = value.pack(&mut self.cx);
                self.values.insert(output_name.to_string(), packed);
            }
            "view_as_complex.default" => {
                let backing = self.get_input_tensor(node, 0)?;
                let dtype = self.output_complex_dtype(output_name)?;
                let value = ComplexTensor::from_interleaved(&mut self.cx, backing, dtype)?;
                self.store_complex(output_name, value);
            }
            "complex.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let real = self.get_input_tensor(node, 0)?.cast(component_dtype);
                let imag = self.get_input_tensor(node, 1)?.cast(component_dtype);
                let (real, imag) = broadcast_binary(real, imag);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }

            // ---- constructors / casts ----
            "full.default" | "full_like.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let shape = if target == "full_like.default" {
                    self.get_complex_input(node, 0)?.real.dims()
                } else {
                    self.get_ints_arg(node, 0)?
                        .into_iter()
                        .map(|v| IntExpr::from(v as usize))
                        .collect()
                };
                let value = self.complex_constructor_scalar_arg(node, 1, dtype)?;
                let value = if shape.is_empty() {
                    value
                } else {
                    value.map(|component| component.expand_rhs(shape.clone()))
                };
                self.store_complex(output_name, value);
            }
            "empty.memory_format"
            | "empty_permuted.default"
            | "empty_strided.default"
            | "new_empty_strided.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let shape = self.output_meta_shape(node)?;
                let real = self
                    .cx
                    .iota(shape.clone(), |_| IntExpr::from(0))
                    .cast(component_dtype);
                let imag = self.constant_like(real, 0.0);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }
            "scalar_tensor.default" => {
                let dtype = self.output_complex_dtype(output_name)?;
                let value = self.complex_constructor_scalar_arg(node, 0, dtype)?;
                self.store_complex(output_name, value);
            }
            "_to_copy.default" => {
                let output_dtype = self.output_torch_dtype(output_name)?;
                let input_name = self.input_value_name(node, 0)?;
                if output_dtype.is_complex() {
                    let value = self.value_as_complex(input_name, output_dtype)?;
                    self.store_complex(output_name, value);
                } else {
                    let value = self.get_complex(input_name)?;
                    let dtype = DType::try_from(output_dtype)
                        .map_err(|t| anyhow!("unsupported real cast target {}", t.name()))?;
                    let out = if dtype == DType::Bool {
                        let real_zero = self.is_zero(value.real);
                        let imag_zero = self.is_zero(value.imag);
                        let both_zero = cx_bool_and(real_zero, imag_zero);
                        cx_bool_not(both_zero)
                    } else {
                        value.real.cast(dtype)
                    };
                    self.values.insert(output_name.to_string(), out);
                }
            }

            // ---- elementwise elementary functions ----
            "exp.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_exp(value);
                self.store_complex(output_name, result);
            }
            "expm1.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_exp(value);
                let one = self.constant_like(result.real, 1.0);
                self.store_complex(
                    output_name,
                    ComplexTensor::new(result.real - one, result.imag, result.torch_dtype),
                );
            }
            "exp2.default" => {
                let value = self.get_complex_input(node, 0)?;
                let ln_two = self.constant_like(value.real, std::f64::consts::LN_2);
                let scaled =
                    ComplexTensor::new(value.real * ln_two, value.imag * ln_two, value.torch_dtype);
                let result = self.complex_exp(scaled);
                self.store_complex(output_name, result);
            }
            "log.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_log(value);
                self.store_complex(output_name, result);
            }
            "log1p.default" => {
                let value = self.get_complex_input(node, 0)?;
                let one = self.constant_like(value.real, 1.0);
                let shifted = ComplexTensor::new(value.real + one, value.imag, value.torch_dtype);
                let result = self.complex_log(shifted);
                self.store_complex(output_name, result);
            }
            "log2.default" | "log10.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_log(value);
                let denominator = if target == "log2.default" {
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
            "sin.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sin(value);
                self.store_complex(output_name, result);
            }
            "sinh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_sinh(value);
                self.store_complex(output_name, result);
            }
            "tan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let numerator = self.complex_sin(value);
                let denominator = self.complex_cos(value);
                let result = self.stable_complex_div(numerator, denominator)?;
                self.store_complex(output_name, result);
            }
            "tanh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let numerator = self.complex_sinh(value);
                let denominator = self.complex_cosh(value);
                let result = self.stable_complex_div(numerator, denominator)?;
                self.store_complex(output_name, result);
            }
            "cos.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_cos(value);
                self.store_complex(output_name, result);
            }
            "cosh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_cosh(value);
                self.store_complex(output_name, result);
            }
            "acos.default" | "acosh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let (acos, acosh, _) = self.complex_acos_acosh(value);
                self.store_complex(
                    output_name,
                    if target == "acos.default" {
                        acos
                    } else {
                        acosh
                    },
                );
            }
            "asin.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_asin(value);
                self.store_complex(output_name, result);
            }
            "asinh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_asinh(value);
                self.store_complex(output_name, result);
            }
            "atan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_atan(value);
                self.store_complex(output_name, result);
            }
            "atanh.default" => {
                let value = self.get_complex_input(node, 0)?;
                let result = self.complex_atanh(value);
                self.store_complex(output_name, result);
            }
            "angle.default" => {
                let value = self.get_complex_input(node, 0)?;
                let out = self.cx_atan2(value.imag, value.real);
                self.values.insert(output_name.to_string(), out);
            }
            "isinf.default" => {
                let value = self.get_complex_input(node, 0)?;
                let real = self.is_inf(value.real);
                let imag = self.is_inf(value.imag);
                self.values
                    .insert(output_name.to_string(), cx_bool_or(real, imag));
            }
            "isnan.default" => {
                let value = self.get_complex_input(node, 0)?;
                let real = self.is_nan(value.real);
                let imag = self.is_nan(value.imag);
                self.values
                    .insert(output_name.to_string(), cx_bool_or(real, imag));
            }
            "polar.default" => {
                let magnitude = self.get_input_tensor(node, 0)?;
                let angle = self.get_input_tensor(node, 1)?;
                let (magnitude, angle) = broadcast_binary(magnitude, angle);
                let dtype = self.output_complex_dtype(output_name)?;
                let component_dtype = dtype.complex_component_dtype().unwrap();
                let magnitude = magnitude.cast(component_dtype);
                let angle = angle.cast(component_dtype);
                let real = magnitude * self.cx_cos(angle);
                let imag = magnitude * angle.sin();
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }

            // ---- logic ----
            "any.default" | "any.dim" | "any.dims" => {
                let value = self.get_complex_input(node, 0)?;
                let real_zero = self.is_zero(value.real);
                let imag_zero = self.is_zero(value.imag);
                let truth = cx_bool_not(cx_bool_and(real_zero, imag_zero));
                let out = self.translate_complex_any(node, truth)?;
                self.values.insert(output_name.to_string(), out);
            }
            "logical_not.default" => {
                let name = self.input_value_name(node, 0)?;
                let truth = self.truth_of_value(name)?;
                let out = cx_bool_not(truth);
                self.values.insert(output_name.to_string(), out);
            }
            "logical_and.default" | "logical_or.default" | "logical_xor.default" => {
                let lhs_name = self.input_value_name(node, 0)?;
                let rhs_name = self.input_value_name(node, 1)?;
                let lhs = self.truth_of_value(lhs_name)?;
                let rhs = self.truth_of_value(rhs_name)?;
                let (lhs, rhs) = broadcast_binary(lhs, rhs);
                let out = match target {
                    "logical_and.default" => cx_bool_and(lhs, rhs),
                    "logical_or.default" => cx_bool_or(lhs, rhs),
                    _ => lhs.ne(rhs),
                };
                self.values.insert(output_name.to_string(), out);
            }

            // ---- comparisons ----
            "eq.Tensor" | "ne.Tensor" => {
                let is_eq = target == "eq.Tensor";
                let out = self.translate_complex_comparison(node, is_eq)?;
                self.values.insert(output_name.to_string(), out);
            }
            "eq.Scalar" | "ne.Scalar" => {
                let value = self.get_complex_input(node, 0)?;
                let (real, imag) = self.complex_scalar_arg(node, 1)?;
                let scalar = self.complex_constant_like(value.real, real, imag, value.torch_dtype);
                let value_real_nan = self.is_nan(value.real);
                let scalar_real_nan = self.is_nan(scalar.real);
                let real_nan = cx_bool_or(value_real_nan, scalar_real_nan);
                let value_imag_nan = self.is_nan(value.imag);
                let scalar_imag_nan = self.is_nan(scalar.imag);
                let imag_nan = cx_bool_or(value_imag_nan, scalar_imag_nan);
                let real_eq = cx_bool_and(value.real.eq(scalar.real), cx_bool_not(real_nan));
                let imag_eq = cx_bool_and(value.imag.eq(scalar.imag), cx_bool_not(imag_nan));
                let equal = cx_bool_and(real_eq, imag_eq);
                let out = if target == "eq.Scalar" {
                    equal
                } else {
                    cx_bool_not(equal)
                };
                self.values.insert(output_name.to_string(), out);
            }

            // ---- reductions ----
            "sum.dim_IntList" | "sum.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Sum)?;
                self.store_complex(output_name, value);
            }
            "mean.dim" | "mean.default" => {
                let value = self.translate_complex_reduction(node, ReductionOp::Mean)?;
                self.store_complex(output_name, value);
            }
            "cumsum.default" => {
                let value = self.get_complex_input(node, 0)?;
                if value.real.rank() == 0 {
                    self.store_complex(output_name, value);
                } else {
                    let dim = normalize_dim(self.get_int_arg(node, 1)?, value.real.rank());
                    self.store_complex(output_name, value.map(|c| c.cumsum(dim)));
                }
            }

            // ---- matmul family ----
            "mm.default" | "bmm.default" => {
                let value = self.translate_complex_matmul(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "addmv.default" => {
                let value = self.translate_complex_addmv(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "addbmm.default" => {
                let value = self.translate_complex_addbmm(node, output_name)?;
                self.store_complex(output_name, value);
            }
            "addmm.default" => {
                let value = self.translate_complex_addmm(node, output_name)?;
                self.store_complex(output_name, value);
            }

            // ---- movement ----
            "view.default" | "view_copy.default" | "reshape.default" | "_unsafe_view.default" => {
                let value = self.get_complex_input(node, 0)?;
                let target_shape = match self.resolve_shape_arg(node, 1) {
                    Some(target) => target,
                    None => self
                        .get_ints_arg(node, 1)?
                        .into_iter()
                        .map(IntExpr::from)
                        .collect(),
                };
                let dims = resolve_neg1_dim_exprs(&target_shape, &value.real.dims());
                let value = value.map(|component| reshape_tensor(component, &dims));
                let value = if target == "view_copy.default" {
                    value.map(materialize_tensor)
                } else {
                    value
                };
                self.store_complex(output_name, value);
            }
            "permute.default" | "permute_copy.default" => {
                let value = self.get_complex_input(node, 0)?;
                let rank = value.real.rank();
                let axes: Vec<usize> = self
                    .get_ints_arg(node, 1)?
                    .iter()
                    .map(|&dim| normalize_dim(dim, rank))
                    .collect();
                let copy = target == "permute_copy.default";
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
            "unsqueeze.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dim = normalize_dim(self.get_int_arg(node, 1)?, value.real.rank() + 1);
                self.store_complex(output_name, value.map(|component| component.unsqueeze(dim)));
            }
            "squeeze.dims" | "squeeze.default" => {
                let value = self.get_complex_input(node, 0)?;
                let dims = if target == "squeeze.dims" {
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
                let rank = value.real.rank();
                let mut dims: Vec<usize> = dims
                    .into_iter()
                    .map(|dim| normalize_dim(dim, rank))
                    .collect();
                dims.sort_unstable();
                self.store_complex(
                    output_name,
                    value.map(|component| squeeze_dims(component, &dims)),
                );
            }
            "expand.default" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.expand_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "repeat.default" => {
                let value = self.get_complex_input(node, 0)?;
                let repeats = self.get_ints_arg(node, 1)?;
                let value = value.try_map(|component| {
                    anyhow::ensure!(
                        repeats.len() >= component.rank(),
                        "complex repeat rank mismatch"
                    );
                    anyhow::ensure!(
                        repeats.iter().all(|&r| r >= 1),
                        "repeat counts must be >= 1"
                    );
                    let mut padded = component;
                    for _ in 0..(repeats.len() - padded.rank()) {
                        padded = padded.unsqueeze(0);
                    }
                    let counts: Vec<usize> = repeats.iter().map(|&r| r as usize).collect();
                    Ok(padded.repeat(counts.as_slice()))
                })?;
                self.store_complex(output_name, value);
            }
            "slice.Tensor" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.slice_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "select.int" => {
                let value = self.get_complex_input(node, 0)?;
                let value =
                    value.try_map(|component| self.select_complex_component(component, node))?;
                self.store_complex(output_name, value);
            }
            "cat.default" => {
                self.translate_complex_cat(node, output_name)?;
            }
            "narrow_copy.default" | "narrow.default" => {
                let value = self.get_complex_input(node, 0)?;
                let rank = value.real.rank();
                let dim = normalize_dim(self.get_int_arg(node, 1)?, rank);
                let start = self
                    .resolve_arg_as_expression(&node.inputs[2].arg)
                    .map(|e| normalize_slice_bound(e, value.real.dims()[dim]))
                    .unwrap_or_else(|| IntExpr::from(0));
                let length = self.get_int_arg(node, 3)?;
                let end = start + IntExpr::from(length.max(0) as usize);
                self.store_complex(
                    output_name,
                    value.map(|component| component.slice_along(start..end, dim)),
                );
            }
            "where.self" => {
                let condition = self.get_input_tensor(node, 0)?;
                let dtype = self.output_complex_dtype(output_name)?;
                let lhs = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
                let rhs = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
                let (lhs_real, rhs_real) = broadcast_binary(lhs.real, rhs.real);
                let (lhs_imag, rhs_imag) = broadcast_binary(lhs.imag, rhs.imag);
                let (lhs_real, condition) = broadcast_binary(lhs_real, condition);
                let (lhs_real, rhs_real) = broadcast_binary(lhs_real, rhs_real);
                let (lhs_imag, rhs_imag) = broadcast_binary(lhs_imag, rhs_imag);
                let real = self.cx_select(condition, lhs_real, rhs_real);
                let imag = self.cx_select(condition, lhs_imag, rhs_imag);
                self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
            }

            other => bail!(
                "Unsupported complex ATen op: {other}. Complex values must be lowered into real components before HLIR"
            ),
        }
        Ok(())
    }

    // ---------------------------------------------------------------
    // Component access and storage
    // ---------------------------------------------------------------

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
            let both_zero = cx_bool_and(real_zero, imag_zero);
            Ok(cx_bool_not(both_zero))
        } else {
            let value = *self
                .values
                .get(name)
                .ok_or_else(|| anyhow!("Unknown value {name}"))?;
            let zero = self.constant_like(value, 0.0);
            Ok(value.ne(zero))
        }
    }

    fn output_complex_dtype(&self, output_name: &str) -> Result<TorchDType> {
        let dtype = self
            .tensor_meta(output_name)
            .with_context(|| format!("Missing tensor metadata for {output_name}"))?
            .dtype;
        let dtype = TorchDType::from_code(dtype)
            .map_err(|code| anyhow!("Unknown PT2 dtype code {code}"))?;
        anyhow::ensure!(dtype.is_complex(), "Output {output_name} is not complex");
        Ok(dtype)
    }

    fn output_torch_dtype(&self, output_name: &str) -> Result<TorchDType> {
        let dtype = self
            .tensor_meta(output_name)
            .with_context(|| format!("Missing tensor metadata for {output_name}"))?
            .dtype;
        TorchDType::from_code(dtype).map_err(|code| anyhow!("Unknown PT2 dtype code {code}"))
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
        let arg = &node
            .inputs
            .get(index)
            .with_context(|| format!("{} missing input {index}", node.target))?
            .arg;
        if let Some((real, imag)) = arg.as_complex() {
            return self.complex_constant_scalars(real, imag, dtype);
        }
        if let Some(value) = arg.as_float() {
            return self.complex_constant_scalars(value, 0.0, dtype);
        }
        if let Some(value) = arg.as_int() {
            return self.complex_constant_scalars(value as f64, 0.0, dtype);
        }
        bail!(
            "input {index} of {} is not a complex-usable scalar: {arg:?}",
            node.target
        )
    }

    fn complex_constant_scalars(
        &mut self,
        real: f64,
        imag: f64,
        dtype: TorchDType,
    ) -> Result<ComplexTensor> {
        let component_dtype = dtype
            .complex_component_dtype()
            .context("complex constructor target is not complex")?;
        Ok(ComplexTensor::new(
            self.floating_scalar(real, component_dtype),
            self.floating_scalar(imag, component_dtype),
            dtype,
        ))
    }

    fn value_as_complex(&mut self, name: &str, dtype: TorchDType) -> Result<ComplexTensor> {
        if let Some(value) = self.complex_tensors.get(name).copied() {
            return Ok(value.cast(dtype));
        }
        let component_dtype = dtype
            .complex_component_dtype()
            .context("complex view target is not complex")?;
        let real = self
            .values
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("Unknown value {name}"))?
            .cast(component_dtype);
        let imag = self.constant_like(real, 0.0);
        Ok(ComplexTensor::new(real, imag, dtype))
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

    // ---------------------------------------------------------------
    // Complex algebra
    // ---------------------------------------------------------------

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

    fn add_complex(&self, lhs: ComplexTensor, rhs: ComplexTensor) -> ComplexTensor {
        let (lhs_real, rhs_real) = broadcast_binary(lhs.real, rhs.real);
        let (lhs_imag, rhs_imag) = broadcast_binary(lhs.imag, rhs.imag);
        ComplexTensor::new(lhs_real + rhs_real, lhs_imag + rhs_imag, lhs.torch_dtype)
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
    pub(crate) fn cx_select(
        &mut self,
        condition: GraphTensor,
        if_true: GraphTensor,
        if_false: GraphTensor,
    ) -> GraphTensor {
        let (if_true, condition) = broadcast_binary(if_true, condition);
        let (if_true, if_false) = broadcast_binary(if_true, if_false);
        let shape = if_true.dims();
        let packed = interleave(&mut self.cx, if_false, if_true);
        let mut coords: Vec<GraphTensor> = (0..shape.len())
            .map(|axis| self.cx.iota(shape.clone(), |coords| coords[axis]))
            .collect();
        coords.push(condition.cast(DType::Int));
        packed.gather(&coords)
    }

    fn safe_div(&mut self, numerator: GraphTensor, denominator: GraphTensor) -> GraphTensor {
        let should_scale = self.reciprocal_overflows(denominator);
        let largest = self.constant_like(denominator, float_max(denominator.dtype));
        let one = self.constant_like(denominator, 1.0);
        let scale = self.cx_select(should_scale, largest, one);
        (numerator * scale) / (denominator * scale)
    }

    fn reciprocal_overflows(&mut self, value: GraphTensor) -> GraphTensor {
        let reciprocal_inf = self.is_inf(value.reciprocal());
        let value_zero = self.is_zero(value);
        cx_bool_and(reciprocal_inf, cx_bool_not(value_zero))
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
        let alternate_nonzero = cx_bool_not(cx_bool_or(alternate_zero, alternate_nan));
        let use_alternate = cx_bool_and(trigger, cx_bool_and(current_zero, alternate_nonzero));
        self.cx_select(use_alternate, alternate, current)
    }

    fn signed_constant_like(&mut self, sign: GraphTensor, magnitude: f64) -> GraphTensor {
        let positive = self.constant_like(sign, magnitude);
        let negative = self.constant_like(sign, -magnitude);
        let signbit = self.signbit(sign);
        self.cx_select(signbit, negative, positive)
    }

    fn signed_indicator(&mut self, value: GraphTensor, condition: GraphTensor) -> GraphTensor {
        let one = self.constant_like(value, 1.0);
        let zero = self.constant_like(value, 0.0);
        let magnitude = self.cx_select(condition, one, zero);
        let negative = magnitude * -1.0;
        let signbit = self.signbit(value);
        self.cx_select(signbit, negative, magnitude)
    }

    /// Smith's finite complex division plus the C99 recovery cases for zero
    /// and infinite operands. Selection is structural so unused NaNs cannot
    /// contaminate the chosen branch.
    fn stable_complex_div(&mut self, a: ComplexTensor, b: ComplexTensor) -> Result<ComplexTensor> {
        let dtype = a.torch_dtype;
        let b_real_abs = self.real_abs(b.real);
        let b_imag_abs = self.real_abs(b.imag);
        let choose_real = b_real_abs.ge(b_imag_abs);
        let choose_imag = cx_bool_not(choose_real);
        let one = self.constant_like(b.real, 1.0);
        let zero = self.constant_like(b.real, 0.0);
        let branch_real = self.cx_select(choose_real, b.real, one);
        let branch_imag = self.cx_select(choose_real, b.imag, zero);
        let branch2_imag = self.cx_select(choose_imag, b.imag, one);
        let branch2_real = self.cx_select(choose_imag, b.real, zero);

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
            self.cx_select(choose_real, out_real_a, out_real_b),
            self.cx_select(choose_real, out_imag_a, out_imag_b),
            dtype,
        );

        let real_subnormal = self.reciprocal_overflows(a.real);
        let imag_subnormal = self.reciprocal_overflows(a.imag);
        let numerator_subnormal = cx_bool_or(real_subnormal, imag_subnormal);
        let direct_denom = b.real * b.real + b.imag * b.imag;
        let direct_real = (a.real * b.real + a.imag * b.imag) / direct_denom;
        let direct_imag = (a.imag * b.real - a.real * b.imag) / direct_denom;
        result.real = self.recover_underflow(numerator_subnormal, result.real, direct_real);
        result.imag = self.recover_underflow(numerator_subnormal, result.imag, direct_imag);

        let b_real_inf = self.is_inf(b.real);
        let b_imag_inf = self.is_inf(b.imag);
        let denominator_inf = cx_bool_or(b_real_inf, b_imag_inf);
        let denominator_one_inf = b_real_inf.ne(b_imag_inf);
        let a_real_inf = self.is_inf(a.real);
        let a_imag_inf = self.is_inf(a.imag);
        let numerator_inf = cx_bool_or(a_real_inf, a_imag_inf);

        let c = self.signed_indicator(b.real, b_real_inf);
        let d = self.signed_indicator(b.imag, b_imag_inf);
        let zero = self.constant_like(a.real, 0.0);
        let inf_den_real = zero * (a.real * c + a.imag * d);
        let inf_den_imag = zero * (a.imag * c - a.real * d);
        let denominator_inf_only = {
            let numerator_not_inf = cx_bool_not(numerator_inf);
            cx_bool_and(denominator_one_inf, numerator_not_inf)
        };
        result.real = self.cx_select(denominator_inf_only, inf_den_real, result.real);
        result.imag = self.cx_select(denominator_inf_only, inf_den_imag, result.imag);

        let ar = self.signed_indicator(a.real, a_real_inf);
        let ai = self.signed_indicator(a.imag, a_imag_inf);
        let infinity = self.constant_like(a.real, f64::INFINITY);
        let inf_num_real = infinity * (ar * b.real + ai * b.imag);
        let inf_num_imag = infinity * (ai * b.real - ar * b.imag);
        let numerator_inf_only = {
            let denominator_not_inf = cx_bool_not(denominator_inf);
            let both_nan = {
                let real_nan = self.is_nan(result.real);
                let imag_nan = self.is_nan(result.imag);
                cx_bool_and(real_nan, imag_nan)
            };
            let inf_over_finite = cx_bool_and(numerator_inf, denominator_not_inf);
            cx_bool_and(inf_over_finite, both_nan)
        };
        result.real = self.cx_select(numerator_inf_only, inf_num_real, result.real);
        result.imag = self.cx_select(numerator_inf_only, inf_num_imag, result.imag);

        let b_real_zero = self.is_zero(b.real);
        let b_imag_zero = self.is_zero(b.imag);
        let denominator_zero = cx_bool_and(b_real_zero, b_imag_zero);
        let real_inf = self.signed_constant_like(b.real, f64::INFINITY);
        let imag_inf = self.signed_constant_like(b.imag, f64::INFINITY);
        result.real = self.cx_select(denominator_zero, real_inf * a.real, result.real);
        result.imag = self.cx_select(denominator_zero, imag_inf * a.imag, result.imag);
        Ok(result)
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

    fn scale_addend(&mut self, node: &Node, value: ComplexTensor) -> Result<ComplexTensor> {
        let (real, imag) = match node.inputs.iter().position(|input| input.name == "alpha") {
            Some(index) => self.complex_scalar_arg(node, index)?,
            None => (1.0, 0.0),
        };
        let alpha = self.complex_constant_like(value.real, real, imag, value.torch_dtype);
        Ok(self.complex_mul(value, alpha))
    }

    // ---------------------------------------------------------------
    // Elementary functions (real HLIR primitives only)
    // ---------------------------------------------------------------

    /// `exp(x)` via the base-2 exponential, keeping log2(e) in the dtype.
    fn cx_exp(&mut self, input: GraphTensor) -> GraphTensor {
        let log2_e = self.constant_like(input, std::f64::consts::LOG2_E);
        (input * log2_e).exp2()
    }

    /// `cos(x) = sin(pi/2 - x)`, keeping pi/2 in the tensor's dtype.
    fn cx_cos(&mut self, input: GraphTensor) -> GraphTensor {
        let half_pi = self.constant_like(input, std::f64::consts::FRAC_PI_2);
        (half_pi - input).sin()
    }

    fn cx_sinh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.cx_exp(input) - self.cx_exp(input * -1.0))
    }

    fn cx_cosh(&mut self, input: GraphTensor) -> GraphTensor {
        let half = self.constant_like(input, 0.5);
        half * (self.cx_exp(input) + self.cx_exp(input * -1.0))
    }

    /// Range-reduced odd Taylor series for atan (recorder port).
    fn cx_atan(&mut self, input: GraphTensor) -> GraphTensor {
        let x = input.abs();
        let one = self.constant_like(x, 1.0);
        let reciprocal_branch = x.gt(one);
        let reduced = self.cx_select(reciprocal_branch, x.reciprocal(), x);

        let threshold = self.constant_like(reduced, std::f64::consts::SQRT_2 - 1.0);
        let quarter_turn_branch = reduced.gt(threshold);
        let transformed = (reduced - one) / (reduced + one);
        let z = self.cx_select(quarter_turn_branch, transformed, reduced);
        let z2 = z.square();

        let mut polynomial = self.constant_like(z, -1.0 / 27.0);
        for degree in (0..13).rev() {
            let coefficient = if degree % 2 == 0 { 1.0 } else { -1.0 } / (2 * degree + 1) as f64;
            polynomial = polynomial * z2 + self.constant_like(z, coefficient);
        }
        let base = z * polynomial;
        let quarter_pi = self.constant_like(z, std::f64::consts::FRAC_PI_4);
        let base = self.cx_select(quarter_turn_branch, quarter_pi + base, base);
        let half_pi = self.constant_like(z, std::f64::consts::FRAC_PI_2);
        let angle = self.cx_select(reciprocal_branch, half_pi - base, base);
        self.copy_sign(angle, input)
    }

    fn cx_asin(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let denominator = (one - input.square()).sqrt();
        self.cx_atan(input / denominator)
    }

    fn cx_acos(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let denominator = (one - input.square()).sqrt();
        let asin = self.cx_atan(input / denominator);
        let half_pi = self.constant_like(input, std::f64::consts::FRAC_PI_2);
        half_pi - asin
    }

    fn cx_asinh(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let magnitude = input.abs() + (input.square() + one).sqrt();
        self.copy_sign(magnitude.log(), input)
    }

    fn cx_acosh(&mut self, input: GraphTensor) -> GraphTensor {
        let one = self.constant_like(input, 1.0);
        let reciprocal_squared = input.reciprocal().square();
        input.log() + (one + (one - reciprocal_squared).sqrt()).log()
    }

    /// `atan2(y, x)` with the C99/PyTorch quadrant and special-case rules.
    fn cx_atan2(&mut self, y: GraphTensor, x: GraphTensor) -> GraphTensor {
        let ratio = y / x;
        let mut angle = self.cx_atan(ratio);
        let x_negative = self.signbit(x);
        let pi = self.constant_like(y, std::f64::consts::PI);
        let signed_pi = self.copy_sign(pi, y);
        angle = self.cx_select(x_negative, angle + signed_pi, angle);

        let x_inf = self.is_inf(x);
        let y_inf = self.is_inf(y);
        let both_inf = cx_bool_and(x_inf, y_inf);
        let quarter = self.constant_like(y, std::f64::consts::FRAC_PI_4);
        let three_quarters = self.constant_like(y, 3.0 * std::f64::consts::FRAC_PI_4);
        let infinite_angle = self.cx_select(x_negative, three_quarters, quarter);
        let infinite_angle = self.copy_sign(infinite_angle, y);
        angle = self.cx_select(both_inf, infinite_angle, angle);

        let x_zero = self.is_zero(x);
        let y_zero = self.is_zero(y);
        let both_zero = cx_bool_and(x_zero, y_zero);
        let zero = self.constant_like(y, 0.0);
        let signed_zero = self.copy_sign(zero, y);
        let zero_angle = self.cx_select(x_negative, signed_pi, signed_zero);
        self.cx_select(both_zero, zero_angle, angle)
    }

    /// Scaled hypot avoids finite overflow/underflow and explicitly gives
    /// infinity precedence over NaN, matching PyTorch/libc hypot semantics.
    fn complex_abs(&mut self, value: ComplexTensor) -> GraphTensor {
        let real = self.real_abs(value.real);
        let imag = self.real_abs(value.imag);
        let real_is_large = real.ge(imag);
        let large = self.cx_select(real_is_large, real, imag);
        let small = self.cx_select(real_is_large, imag, real);
        let one = self.constant_like(large, 1.0);
        let large_is_zero = self.is_zero(large);
        let safe_large = self.cx_select(large_is_zero, one, large);
        let ratio = self.safe_div(small, safe_large);
        let finite = large * (ratio * ratio + self.constant_like(ratio, 1.0)).sqrt();
        let real_inf = self.is_inf(value.real);
        let imag_inf = self.is_inf(value.imag);
        let any_inf = cx_bool_or(real_inf, imag_inf);
        let infinity = self.constant_like(finite, f64::INFINITY);
        self.cx_select(any_inf, infinity, finite)
    }

    fn complex_acos_acosh(
        &mut self,
        value: ComplexTensor,
    ) -> (ComplexTensor, ComplexTensor, ComplexTensor) {
        let one = self.constant_like(value.real, 1.0);
        let plus_one = ComplexTensor::new(value.real + one, value.imag, value.torch_dtype);
        let minus_one = ComplexTensor::new(value.real - one, value.imag, value.torch_dtype);
        let r = self.complex_abs(plus_one);
        let s = self.complex_abs(minus_one);

        let mut alpha = r * 0.5 + s * 0.5;
        let one = self.constant_like(alpha, 1.0);
        alpha = self.cx_select(alpha.lt(one), one, alpha);

        let mut beta = value.real / alpha;
        let real_inf = self.is_inf(value.real);
        let imag_inf = self.is_inf(value.imag);
        let real_nan = self.is_nan(value.real);
        let imag_nan = self.is_nan(value.imag);
        let both_inf = cx_bool_and(real_inf, imag_inf);
        let real_inf_only = cx_bool_and(real_inf, cx_bool_not(cx_bool_or(imag_inf, imag_nan)));
        let imag_inf_only = cx_bool_and(imag_inf, cx_bool_not(cx_bool_or(real_inf, real_nan)));
        let diagonal = self.signed_constant_like(value.real, std::f64::consts::FRAC_1_SQRT_2);
        beta = self.cx_select(both_inf, diagonal, beta);
        let real_axis = self.signed_constant_like(value.real, 1.0);
        beta = self.cx_select(real_inf_only, real_axis, beta);
        let zero = self.constant_like(beta, 0.0);
        beta = self.cx_select(imag_inf_only, zero, beta);

        let negative_one = self.constant_like(beta, -1.0);
        beta = self.cx_select(beta.lt(negative_one), negative_one, beta);
        beta = self.cx_select(beta.gt(one), one, beta);

        let acos_beta = self.cx_acos(beta);
        let acosh_alpha = self.cx_acosh(alpha);
        let signed_acosh = self.copy_sign(acosh_alpha, value.imag);
        let cosine = (one - beta.square()).sqrt();
        let stable_signed_acosh = self.cx_asinh(value.imag / cosine);
        let well_conditioned = self.constant_like(cosine, 0.25);
        let signed_acosh = self.cx_select(
            cosine.gt(well_conditioned),
            stable_signed_acosh,
            signed_acosh,
        );
        let signed_acos = self.copy_sign(acos_beta, value.imag);

        let mut acos_real = acos_beta;
        let mut acos_imag = signed_acosh * -1.0;

        let real_zero = self.is_zero(value.real);
        let zero_with_nan_imag = cx_bool_and(real_zero, imag_nan);
        let half_pi = self.constant_like(acos_real, std::f64::consts::FRAC_PI_2);
        acos_real = self.cx_select(zero_with_nan_imag, half_pi, acos_real);
        let real_inf_with_nan_imag = cx_bool_and(real_inf, imag_nan);
        let signed_infinity = self.signed_constant_like(value.real, f64::INFINITY);
        acos_imag = self.cx_select(real_inf_with_nan_imag, signed_infinity, acos_imag);

        let mut asin_real = self.cx_asin(beta);
        asin_real = self.cx_select(zero_with_nan_imag, value.real, asin_real);
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
        let scale = self.cx_exp(value.real);
        ComplexTensor::new(
            scale * self.cx_cos(value.imag),
            scale * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_sin(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.cx_cosh(value.imag);
        let sinh = self.cx_sinh(value.imag);
        ComplexTensor::new(
            value.real.sin() * cosh,
            self.cx_cos(value.real) * sinh,
            value.torch_dtype,
        )
    }

    fn complex_sinh(&mut self, value: ComplexTensor) -> ComplexTensor {
        let sinh = self.cx_sinh(value.real);
        let cosh = self.cx_cosh(value.real);
        ComplexTensor::new(
            sinh * self.cx_cos(value.imag),
            cosh * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_sqrt(&mut self, value: ComplexTensor) -> ComplexTensor {
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

        let mut real = self.cx_select(real_negative, negative_real, positive_real);
        let mut imag = self.cx_select(real_negative, negative_imag, positive_imag);

        let magnitude_zero = self.is_zero(magnitude);
        let zero = self.constant_like(real, 0.0);
        let signed_zero = self.copy_sign(zero, value.imag);
        real = self.cx_select(magnitude_zero, zero, real);
        imag = self.cx_select(magnitude_zero, signed_zero, imag);

        let imag_inf = self.is_inf(value.imag);
        let infinity = self.constant_like(real, f64::INFINITY);
        let signed_infinity = self.copy_sign(infinity, value.imag);
        real = self.cx_select(imag_inf, infinity, real);
        imag = self.cx_select(imag_inf, signed_infinity, imag);

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
            self.cx_select(exponent_imag_zero, real_exponent.real, general.real),
            self.cx_select(exponent_imag_zero, real_exponent.imag, general.imag),
            base.torch_dtype,
        );
        let result = self.complex_exp(product);

        let real_zero = self.is_zero(exponent.real);
        let imag_zero = self.is_zero(exponent.imag);
        let exponent_zero = cx_bool_and(real_zero, imag_zero);
        let one = self.constant_like(result.real, 1.0);
        let zero = self.constant_like(result.imag, 0.0);
        ComplexTensor::new(
            self.cx_select(exponent_zero, one, result.real),
            self.cx_select(exponent_zero, zero, result.imag),
            base.torch_dtype,
        )
    }

    fn complex_cos(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.cx_cosh(value.imag);
        let sinh = self.cx_sinh(value.imag);
        ComplexTensor::new(
            self.cx_cos(value.real) * cosh,
            value.real.sin() * sinh * -1.0,
            value.torch_dtype,
        )
    }

    fn complex_cosh(&mut self, value: ComplexTensor) -> ComplexTensor {
        let cosh = self.cx_cosh(value.real);
        let sinh = self.cx_sinh(value.real);
        ComplexTensor::new(
            cosh * self.cx_cos(value.imag),
            sinh * value.imag.sin(),
            value.torch_dtype,
        )
    }

    fn complex_log(&mut self, value: ComplexTensor) -> ComplexTensor {
        let magnitude = self.complex_abs(value);
        let angle = self.cx_atan2(value.imag, value.real);
        ComplexTensor::new(magnitude.log(), angle, value.torch_dtype)
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

    // ---------------------------------------------------------------
    // Movement / reductions / matmul helpers
    // ---------------------------------------------------------------

    fn expand_complex_component(&self, mut value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let raw: Vec<IntExpr> = match self.resolve_shape_arg(node, 1) {
            Some(target) => target,
            None => self
                .get_ints_arg(node, 1)?
                .into_iter()
                .map(IntExpr::from)
                .collect(),
        };
        anyhow::ensure!(raw.len() >= value.rank(), "complex expand rank mismatch");
        for _ in 0..(raw.len() - value.rank()) {
            value = value.unsqueeze(0);
        }
        let neg_one = IntExpr::from(-1i32);
        let target: Vec<IntExpr> = raw
            .into_iter()
            .enumerate()
            .map(|(axis, dim)| {
                if dim == neg_one {
                    value.dims()[axis]
                } else {
                    dim
                }
            })
            .collect();
        Ok(value.expand(target))
    }

    fn slice_complex_component(&self, value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let rank = value.rank();
        let dim = normalize_dim(self.get_int_arg(node, 1).unwrap_or(0), rank);
        let start = match node.inputs.get(2) {
            Some(input) => {
                let expr = self
                    .resolve_arg_as_expression(&input.arg)
                    .ok_or_else(|| anyhow!("slice start is not an expression"))?;
                normalize_slice_bound(expr, value.dims()[dim])
            }
            None => IntExpr::from(0),
        };
        let end = match node.inputs.get(3) {
            Some(input) => {
                let expr = self
                    .resolve_arg_as_expression(&input.arg)
                    .ok_or_else(|| anyhow!("slice end is not an expression"))?;
                match expr.as_num() {
                    Some(v) if v < 0 => {
                        normalize_slice_bound(IntExpr::from(-1i32), value.dims()[dim]) + 1
                    }
                    _ => normalize_slice_bound(expr, value.dims()[dim]),
                }
            }
            None => value.dims()[dim],
        };
        let step = node.inputs.get(4).and_then(|i| i.arg.as_int()).unwrap_or(1);
        anyhow::ensure!(step == 1, "complex slice step {step} != 1 is not ported");
        let mut ranges: Vec<(IntExpr, IntExpr)> =
            value.dims().iter().map(|d| (0.into(), *d)).collect();
        ranges[dim] = (start, end);
        Ok(value.slice(ranges))
    }

    fn select_complex_component(&self, value: GraphTensor, node: &Node) -> Result<GraphTensor> {
        let dim = normalize_dim(self.get_int_arg(node, 1)?, value.rank());
        let index = self.get_int_arg(node, 2)?;
        let index = if index < 0 {
            let size = value.dims()[dim]
                .to_usize()
                .context("negative complex select index requires a concrete dimension")?;
            index + size as i64
        } else {
            index
        };
        let mut ranges: Vec<(IntExpr, IntExpr)> =
            value.dims().iter().map(|d| (0.into(), *d)).collect();
        ranges[dim] = (IntExpr::from(index), IntExpr::from(index + 1));
        Ok(value.slice(ranges).squeeze(dim))
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
        let dim = normalize_dim(dim, values[0].real.rank());
        let real = values[1..].iter().fold(values[0].real, |acc, value| {
            acc.concat_along(value.real, dim)
        });
        let imag = values[1..].iter().fold(values[0].imag, |acc, value| {
            acc.concat_along(value.imag, dim)
        });
        self.store_complex(output_name, ComplexTensor::new(real, imag, dtype));
        Ok(())
    }

    fn translate_complex_any(&mut self, node: &Node, truth: GraphTensor) -> Result<GraphTensor> {
        let rank = truth.rank();
        let (axes, keepdim) = if node.target.ends_with("any.default") {
            ((0..rank).collect::<Vec<_>>(), false)
        } else if node.target.ends_with("any.dim") {
            let dim = self.get_int_arg(node, 1)?;
            let keepdim = self
                .named_bool_arg(node, "keepdim")
                .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
                .unwrap_or(false);
            if rank == 0 {
                anyhow::ensure!(
                    matches!(dim, -1 | 0),
                    "any dimension {dim} out of range for a scalar"
                );
                (Vec::new(), keepdim)
            } else {
                anyhow::ensure!(
                    dim >= -(rank as i64) && dim < rank as i64,
                    "any dimension {dim} out of range for rank {rank}"
                );
                (vec![normalize_dim(dim, rank)], keepdim)
            }
        } else {
            let keepdim = self
                .named_bool_arg(node, "keepdim")
                .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
                .unwrap_or(false);
            let axes = match self.get_ints_arg(node, 1) {
                Ok(dims) => {
                    let mut axes = Vec::with_capacity(dims.len());
                    for dim in dims {
                        anyhow::ensure!(
                            dim >= -(rank as i64) && dim < rank as i64,
                            "any dimension {dim} out of range for rank {rank}"
                        );
                        axes.push(normalize_dim(dim, rank));
                    }
                    axes
                }
                Err(_) => (0..rank).collect(),
            };
            (axes, keepdim)
        };
        if axes.is_empty() {
            return Ok(truth);
        }
        let counts = truth.cast(DType::F32).sum(&axes);
        let zero = self.cx.constant_f32(0.0).expand_rhs(counts.dims());
        let result = counts.gt(zero);
        Ok(if keepdim {
            let mut sorted = axes;
            sorted.sort_unstable();
            let mut result = result;
            for axis in sorted {
                result = result.expand_dim(axis, 1usize);
            }
            result
        } else {
            result
        })
    }

    fn translate_complex_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<ComplexTensor> {
        let value = self.get_complex_input(node, 0)?;
        let rank = value.real.rank();
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
            ReductionOp::Sum => tensor.sum(&axes),
            ReductionOp::Mean => tensor.mean(&axes),
            _ => unreachable!("only sum and mean are complex componentwise reductions"),
        });
        if keepdim {
            let mut sorted = axes;
            sorted.sort_unstable();
            for axis in sorted {
                result = result.map(|component| component.expand_dim(axis, 1usize));
            }
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

    fn scale_complex_by_named_scalar(
        &mut self,
        node: &Node,
        name: &str,
        value: ComplexTensor,
    ) -> Result<ComplexTensor> {
        let Some(index) = node.inputs.iter().position(|input| input.name == name) else {
            return Ok(value);
        };
        let (real, imag) = self.complex_scalar_arg(node, index)?;
        if real == 1.0 && imag == 0.0 {
            return Ok(value);
        }
        let scalar = self.complex_constant_like(value.real, real, imag, value.torch_dtype);
        Ok(self.complex_mul(value, scalar))
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
        anyhow::ensure!(matrix.real.rank() == 2, "addmv matrix must be rank 2");
        anyhow::ensure!(vector.real.rank() == 1, "addmv vector must be rank 1");

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
        anyhow::ensure!(batch1.real.rank() == 3, "addbmm batch1 must be rank 3");
        anyhow::ensure!(batch2.real.rank() == 3, "addbmm batch2 must be rank 3");

        let product = self
            .complex_matmul(batch1, batch2)
            .map(|component| component.sum(&[0]));
        let input = self.scale_complex_by_named_scalar(node, "beta", input)?;
        let product = self.scale_complex_by_named_scalar(node, "alpha", product)?;
        Ok(self.add_complex(input, product).cast(output_dtype))
    }

    fn translate_complex_addmm(&mut self, node: &Node, output_name: &str) -> Result<ComplexTensor> {
        let dtype = self.output_complex_dtype(output_name)?;
        let input = self.value_as_complex(self.input_value_name(node, 0)?, dtype)?;
        let a = self.value_as_complex(self.input_value_name(node, 1)?, dtype)?;
        let b = self.value_as_complex(self.input_value_name(node, 2)?, dtype)?;
        let beta = self.named_float_arg(node, "beta").unwrap_or(1.0);
        let alpha = self.named_float_arg(node, "alpha").unwrap_or(1.0);
        let mm = self.complex_matmul(a, b);
        let beta = self.constant_like(input.real, beta);
        let alpha = self.constant_like(mm.real, alpha);
        let input_real = input.real * beta;
        let input_imag = input.imag * beta;
        let mm_real = mm.real * alpha;
        let mm_imag = mm.imag * alpha;
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
        let real_nan = cx_bool_or(lr_nan, rr_nan);
        let imag_nan = cx_bool_or(li_nan, ri_nan);
        let real_eq = cx_bool_and(lr.eq(rr), cx_bool_not(real_nan));
        let imag_eq = cx_bool_and(li.eq(ri), cx_bool_not(imag_nan));
        let equal = cx_bool_and(real_eq, imag_eq);
        Ok(if is_eq { equal } else { cx_bool_not(equal) })
    }
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[cfg(test)]
mod tests {
    use crate::pt2_parser::ParsedPT2;
    use crate::translate::translate;

    /// Two complex inputs and a node chain exercising add, mul,
    /// view_as_real, view_as_complex, real and imag.
    fn program() -> ParsedPT2 {
        let json = r#"{
          "graph_module": {
            "graph": {
              "inputs": [
                {"as_tensor": {"name": "x"}},
                {"as_tensor": {"name": "y"}}
              ],
              "outputs": [
                {"as_tensor": {"name": "z"}},
                {"as_tensor": {"name": "zr"}},
                {"as_tensor": {"name": "zc"}},
                {"as_tensor": {"name": "r"}},
                {"as_tensor": {"name": "i"}}
              ],
              "nodes": [
                {"target": "torch.ops.aten.add.Tensor",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "x"}}},
                   {"name": "other", "arg": {"as_tensor": {"name": "y"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "z"}}]},
                {"target": "torch.ops.aten.mul.Tensor",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "z"}}},
                   {"name": "other", "arg": {"as_tensor": {"name": "x"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "z2"}}]},
                {"target": "torch.ops.aten.view_as_real.default",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "z2"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "zr"}}]},
                {"target": "torch.ops.aten.view_as_complex.default",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "zr"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "zc"}}]},
                {"target": "torch.ops.aten.real.default",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "zc"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "r"}}]},
                {"target": "torch.ops.aten.imag.default",
                 "inputs": [
                   {"name": "self", "arg": {"as_tensor": {"name": "zc"}}}
                 ],
                 "outputs": [{"as_tensor": {"name": "i"}}]}
              ],
              "tensor_values": {
                "x": {"dtype": 10, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "y": {"dtype": 10, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "z": {"dtype": 10, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "z2": {"dtype": 10, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "zr": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}, {"as_int": 2}]},
                "zc": {"dtype": 10, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "r": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}]},
                "i": {"dtype": 7, "sizes": [{"as_int": 2}, {"as_int": 3}]}
              }
            },
            "signature": {"input_specs": []}
          }
        }"#;
        ParsedPT2 {
            program: serde_json::from_str(json).expect("complex fixture must parse"),
            constants_config: None,
            weights_config: None,
            archive_prefix: String::new(),
            pt2_path: String::new(),
        }
    }

    #[test]
    fn complex_core_ops_record() {
        let parsed = program();
        let t = translate(&parsed).expect("complex graph must translate");
        // Five outputs: packed complex add/mul result, view_as_real,
        // view_as_complex, real, imag.
        assert_eq!(t.outputs.len(), 5);
        for output in &t.outputs {
            match output.graph_name.as_str() {
                "zr" => assert_eq!(output.shape, vec![2, 3, 2], "{}", output.graph_name),
                _ => assert_eq!(output.shape, vec![2, 3], "{}", output.graph_name),
            }
            assert_eq!(output.dtype, luminal::prelude::DType::F32);
        }
        // The two complex inputs carry a trailing interleaved component axis.
        for input in &t.inputs {
            assert_eq!(input.shape, vec![2, 3, 2], "{}", input.graph_name);
            assert_eq!(input.dtype, luminal::prelude::DType::F32);
        }
    }
}
