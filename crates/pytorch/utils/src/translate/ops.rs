//! Category lowerings for the common ATen surface (port batch 1).
//!
//! Each method lowers one ATen target (or family) onto the recorder
//! frontend. The parked translator's ShapeTracker-era helpers are replaced
//! by the recorder's explicit views; `util` carries the shared helpers.

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, normalize_dim};
use crate::pt2_schema::Node;

/// Reduction family for the keepdim-aware `sum`/`mean`/`max`/`min`/`prod`.
#[derive(Clone, Copy)]
pub(super) enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Elementwise
    // ---------------------------------------------------------------

    pub(super) fn comparison(
        &mut self,
        node: &Node,
        cmp: impl FnOnce(GraphTensor, GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = if let Some(t) = self.optional_tensor_operand(&node.inputs[1])? {
            t
        } else {
            // Unlike arithmetic, torch rounds a comparison's scalar to the
            // operands' common dtype before comparing.
            match self.opmath {
                Some(opmath) => {
                    let literal = self.scalar(&node.inputs[1], opmath.common)?;
                    super::convert(literal, opmath.compute)
                }
                None => self.scalar(&node.inputs[1], a.dtype)?,
            }
        };
        let (a, b) = util::broadcast_binary(a, b);
        Ok(cmp(a, b))
    }

    pub(super) fn logical_binary(
        &mut self,
        node: &Node,
        or: bool,
        xor: bool,
    ) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = self.operand(&node.inputs[1])?;
        let (a, b) = util::broadcast_binary(a, b);
        let (af, bf) = (a.cast(DType::F32), b.cast(DType::F32));
        Ok(if or {
            (af + bf).cast(DType::Bool)
        } else if xor {
            af.ne(bf)
        } else {
            (af * bf).cast(DType::Bool)
        })
    }

    pub(super) fn pow_tensor_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let exponent = self.get_float_arg(node, 1)? as f32;
        Ok(a.pow(exponent))
    }

    pub(super) fn pow_tensor_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = self.operand(&node.inputs[1])?;
        let (a, b) = util::broadcast_binary(a, b);
        Ok((b * a.log2()).exp2())
    }

    pub(super) fn pow_scalar_base(&mut self, node: &Node) -> Result<GraphTensor> {
        let base = self.get_float_arg(node, 0)?;
        let b = self.operand(&node.inputs[1])?;
        let log_base = self
            .floating_scalar(base.ln(), b.dtype)
            .expand_rhs(b.dims());
        Ok((b * log_base).exp())
    }

    pub(super) fn fmod_remainder(&mut self, node: &Node, fmod: bool) -> Result<GraphTensor> {
        if fmod {
            return self.binary(&node.inputs, |a, b| a % b);
        }
        // Python-style remainder: a - floor(a/b)*b.
        let a = self.operand(&node.inputs[0])?;
        let b = if let Some(t) = self.optional_tensor_operand(&node.inputs[1])? {
            t
        } else {
            self.scalar(&node.inputs[1], a.dtype)?
        };
        let (a, b) = util::broadcast_binary(a, b);
        Ok(a - (a / b).floor() * b)
    }

    // ---------------------------------------------------------------
    // Movement
    // ---------------------------------------------------------------

    pub(super) fn translate_view(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        // Prefer the symbolic target (sym-int entries and compound sympy
        // expressions resolve through the parser); fall back to the sym-int
        // list reader, which keeps symbolic entries symbolic.
        let target = match self.resolve_shape_arg(node, 1) {
            Some(target) => target,
            None => self.get_int_exprs_arg(node, 1)?,
        };
        // Export already solved the inferred dimension under its shape
        // guards. Preserve that expression instead of introducing a quotient
        // such as (s * 8) / s which obscures exact divisibility in the recorder.
        let dims = if target.contains(&IntExpr::from(-1i32)) {
            let name = Self::tensor_output_names(node)
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("view has no output metadata"))?;
            let meta = self.tensor_meta(&name)?.clone();
            self.boundary_shape(&meta, &name)?
        } else {
            target
        };
        Ok(util::reshape_tensor(x, &dims))
    }

    pub(super) fn translate_flatten(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let start = normalize_dim(
            node.inputs.get(1).and_then(|i| i.arg.as_int()).unwrap_or(0),
            x.rank(),
        );
        let end = normalize_dim(
            node.inputs
                .get(2)
                .and_then(|i| i.arg.as_int())
                .unwrap_or(-1),
            x.rank(),
        );
        let mut out = x;
        for _ in start..end {
            out = out.merge_dims(start, start + 1);
        }
        Ok(out)
    }

    pub(super) fn translate_slice(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let rank = x.rank();
        let dim = normalize_dim(self.get_int_arg(node, 1)?, rank);
        let start = match node.inputs.get(2) {
            Some(input) if !matches!(&input.arg, crate::pt2_schema::Argument::Other(value) if value.get("as_none").is_some()) =>
            {
                let expr = self
                    .resolve_arg_as_expression(&input.arg)
                    .ok_or_else(|| anyhow::anyhow!("slice start is not an expression"))?;
                util::normalize_slice_bound(expr, x.dims()[dim])
            }
            _ => IntExpr::from(0),
        };
        let end = match node.inputs.get(3) {
            Some(input) if !matches!(&input.arg, crate::pt2_schema::Argument::Other(value) if value.get("as_none").is_some()) =>
            {
                let expr = self
                    .resolve_arg_as_expression(&input.arg)
                    .ok_or_else(|| anyhow::anyhow!("slice end is not an expression"))?;
                match expr.as_num() {
                    Some(v) if v < 0 => {
                        util::normalize_slice_bound(IntExpr::from(-1i32), x.dims()[dim]) + 1
                    }
                    _ => util::normalize_slice_bound(expr, x.dims()[dim]),
                }
            }
            _ => x.dims()[dim],
        };
        let step = node.inputs.get(4).and_then(|i| i.arg.as_int()).unwrap_or(1);
        if step != 1 {
            bail!("slice step {step} != 1 is not ported");
        }
        let mut ranges: Vec<(IntExpr, IntExpr)> = x.dims().iter().map(|d| (0.into(), *d)).collect();
        ranges[dim] = (start, end);
        Ok(x.slice(ranges))
    }

    pub(super) fn translate_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        let index = self.get_int_arg(node, 2)?;
        // Only a NEGATIVE index needs the axis extent to wrap; a
        // non-negative select on a dynamic axis is fine.
        let index = if index < 0 {
            let extent = x.dims()[dim].to_usize().ok_or_else(|| {
                anyhow::anyhow!("select with a negative index on a dynamic axis is not ported")
            })?;
            index + extent as i64
        } else {
            index
        };
        let mut ranges: Vec<(IntExpr, IntExpr)> = x.dims().iter().map(|d| (0.into(), *d)).collect();
        ranges[dim] = (IntExpr::from(index), IntExpr::from(index + 1));
        Ok(x.slice(ranges).squeeze(dim))
    }

    pub(super) fn translate_expand(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let target = match self.resolve_shape_arg(node, 1) {
            Some(target) => target,
            None => self
                .get_ints_arg(node, 1)?
                .into_iter()
                .map(IntExpr::from)
                .collect(),
        };
        let dims = x.dims();
        let rank = target.len();
        let neg1 = IntExpr::from(-1i32);
        let mut aligned = vec![IntExpr::from(1); rank];
        for (i, size) in target.iter().enumerate() {
            let from_end = rank - i;
            aligned[i] = if *size == neg1 {
                if from_end <= dims.len() {
                    dims[dims.len() - from_end]
                } else {
                    IntExpr::from(1)
                }
            } else {
                *size
            };
        }
        let mut out = x;
        while out.rank() < rank {
            out = out.expand_dim(0, 1usize);
        }
        Ok(out.expand(aligned))
    }

    pub(super) fn translate_repeat(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let repeats = self.get_ints_arg(node, 1)?;
        let repeats: Vec<usize> = repeats.iter().map(|r| (*r).max(0) as usize).collect();
        let x = (0..repeats.len().saturating_sub(x.rank())).fold(x, |v, _| v.expand_dim(0, 1usize));
        Ok(x.repeat(repeats.as_slice()))
    }

    pub(super) fn translate_stack(&mut self, node: &Node) -> Result<GraphTensor> {
        let tensors = node.inputs[0]
            .arg
            .as_tensors()
            .ok_or_else(|| anyhow::anyhow!("stack: first operand is not a tensor list"))?;
        let axis = self.named_int_arg(node, "dim").unwrap_or(0).max(0) as usize;
        let mut values = Vec::with_capacity(tensors.len());
        for t in tensors {
            values.push(
                *self
                    .values
                    .get(&t.name)
                    .ok_or_else(|| anyhow::anyhow!("stack: unknown tensor {}", t.name))?,
            );
        }
        Ok(self.cx.stack(&values, axis))
    }

    // ---------------------------------------------------------------
    // Reductions
    // ---------------------------------------------------------------

    pub(super) fn translate_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dims = self.optional_int_list(node, 1)?;
        let rank = x.rank();
        // A rank-0 value has no axis: the only dims torch accepts there name
        // nothing, so the reduction and its keepdim are both the identity.
        if rank == 0
            && let Some(&dim) = dims.iter().find(|&&d| d != 0 && d != -1)
        {
            bail!("dim {dim} is out of range for a rank-0 value");
        }
        let axes: Vec<usize> = if rank == 0 || dims.is_empty() {
            (0..rank).collect()
        } else {
            dims.iter().map(|&d| normalize_dim(d, rank)).collect()
        };
        let keepdim = self
            .inputs_bool(node, "keepdim")
            .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false);
        let result = match op {
            ReductionOp::Sum => x.sum(axes.clone()),
            ReductionOp::Mean => x.mean(axes.clone()),
            ReductionOp::Max => x.max(axes.clone()),
            ReductionOp::Min => x.min(axes.clone()),
            ReductionOp::Prod => x.prod(axes.clone()),
        };
        Ok(if keepdim {
            keep_axes(result, &axes)
        } else {
            result
        })
    }

    pub(super) fn translate_argextremum(&mut self, node: &Node, max: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let keepdim = self
            .inputs_bool(node, "keepdim")
            .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false);
        let dim = node.inputs.get(1).and_then(|i| i.arg.as_int());
        if x.rank() == 0 {
            if let Some(d) = dim
                && d != 0
                && d != -1
            {
                bail!("dim {d} is out of range for a rank-0 value");
            }
            return Ok(self.cx.constant_i64(0));
        }
        let (result, axis) = match dim {
            Some(d) => {
                let axis = normalize_dim(d, x.rank());
                let r = if max { x.argmax(axis) } else { x.argmin(axis) };
                (r, Some(axis))
            }
            None => {
                let flattened = x.flatten();
                (
                    if max {
                        flattened.argmax(0)
                    } else {
                        flattened.argmin(0)
                    },
                    None,
                )
            }
        };
        let dtype = self.output_meta_dtype(node).unwrap_or(DType::I64);
        let result = if result.dtype != dtype {
            result.cast(dtype)
        } else {
            result
        };
        Ok(match (keepdim, axis) {
            (true, Some(axis)) => result.expand_dim(axis, 1usize),
            (true, None) => (0..x.rank()).fold(result, |r, axis| r.expand_dim(axis, 1usize)),
            _ => result,
        })
    }

    pub(super) fn translate_var(&mut self, node: &Node, std: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dims = self.optional_int_list(node, 1)?;
        let rank = x.rank();
        let axes: Vec<usize> = if dims.is_empty() {
            (0..rank).collect()
        } else {
            dims.iter().map(|&d| normalize_dim(d, rank)).collect()
        };
        let correction = self.variance_correction(node);
        let variance = self.variance_with_correction(x, &axes, correction);
        let result = if std { variance.sqrt() } else { variance };
        let keepdim = self
            .inputs_bool(node, "keepdim")
            .or_else(|| node.inputs.get(3).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false);
        Ok(if keepdim {
            keep_axes(result, &axes)
        } else {
            result
        })
    }

    pub(super) fn translate_cumulative(&mut self, node: &Node, prod: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        // ATen accumulates in the declared output dtype: bool and narrow
        // integers promote to i64 by default, and dtype= overrides that.
        // Casting only the scan result would lose carries or overflow first.
        let x = super::convert(x, self.compute_dtype(node)?);
        let dim = self.get_int_arg(node, 1)?;
        // A rank-0 scan is the identity, and only dims torch accepts there
        // reach it.
        if x.rank() == 0 {
            if dim != 0 && dim != -1 {
                bail!("dim {dim} is out of range for a rank-0 value");
            }
            return Ok(x);
        }
        let dim = normalize_dim(dim, x.rank());
        Ok(if prod { x.cumprod(dim) } else { x.cumsum(dim) })
    }

    /// `max.dim` / `min.dim`: values and indices into one graph.
    pub(super) fn translate_dim_extremum(&mut self, node: &Node, max: bool) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        let keepdim = self
            .inputs_bool(node, "keepdim")
            .or_else(|| node.inputs.get(2).and_then(|i| i.arg.as_bool()))
            .unwrap_or(false);
        let (values, indices) = if max {
            (x.max(dim), x.argmax(dim))
        } else {
            (x.min(dim), x.argmin(dim))
        };
        let indices = if indices.dtype != DType::I64 {
            indices.cast(DType::I64)
        } else {
            indices
        };
        let (values, indices) = if keepdim {
            (
                values.expand_dim(dim, 1usize),
                indices.expand_dim(dim, 1usize),
            )
        } else {
            (values, indices)
        };
        self.bind_outputs(node, vec![values, indices])
    }

    // ---------------------------------------------------------------
    // Creation and selection
    // ---------------------------------------------------------------

    pub(super) fn translate_fill(&mut self, node: &Node, value: f64) -> Result<GraphTensor> {
        let shape = self.output_meta_shape(node)?;
        let dtype = self.output_meta_dtype(node)?;
        Ok(self.full_tensor(shape, dtype, value))
    }

    pub(super) fn translate_full(&mut self, node: &Node, like: bool) -> Result<GraphTensor> {
        let (shape, dtype) = if like {
            let x = self.operand(&node.inputs[0])?;
            (x.dims(), self.output_meta_dtype(node).unwrap_or(x.dtype))
        } else {
            (
                self.get_int_exprs_arg(node, 0)?,
                self.output_meta_dtype(node)?,
            )
        };
        if let Some(name) = node.inputs[1].arg.as_value_name()
            && let Some(value) = self.values.get(name).copied()
        {
            return Ok(value.cast(dtype).expand_rhs(shape));
        }
        let value = self.get_number_arg(node, 1)?;
        Ok(self.full_tensor(shape, dtype, value))
    }

    pub(super) fn translate_arange(&mut self, node: &Node, _kind: u8) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        // PT2 omits default arguments. Read by schema name, and preserve
        // symbolic scalar operands rather than interpreting them as literals.
        let literal =
            |name: &str, default: f64| match node.inputs.iter().find(|input| input.name == name) {
                None => Some(default),
                Some(input) => input
                    .arg
                    .as_float()
                    .or_else(|| input.arg.as_int().map(|v| v as f64))
                    .or_else(|| input.arg.as_bool().map(f64::from)),
            };
        if let (Some(start), Some(step)) = (literal("start", 0.0), literal("step", 1.0)) {
            anyhow::ensure!(step != 0.0, "arange step must be nonzero");
            let name = Self::tensor_output_names(node)
                .into_iter()
                .next()
                .context("arange has no output")?;
            let meta = self.tensor_meta(&name)?.clone();
            let shape = self.boundary_shape(&meta, &name)?;
            let integer = matches!(
                dtype,
                DType::Int | DType::I64 | DType::I8 | DType::U8 | DType::I16 | DType::U16
            );
            if integer {
                let start = IntExpr::from(start as i64);
                let step = IntExpr::from(step as i64);
                return Ok(self
                    .cx
                    .iota(shape, |coordinates| coordinates[0] * step + start)
                    .cast(dtype));
            }
            let arithmetic_dtype = if dtype == DType::F64 {
                DType::F64
            } else {
                DType::F32
            };
            let positions = self
                .cx
                .iota(shape.clone(), |coordinates| coordinates[0])
                .cast(arithmetic_dtype);
            let start = self.full_tensor(shape.clone(), arithmetic_dtype, start);
            let step = self.full_tensor(shape, arithmetic_dtype, step);
            return Ok((positions * step + start).cast(dtype));
        }
        let argument = |name: &str, default: i64| -> Result<IntExpr> {
            let Some(input) = node.inputs.iter().find(|input| input.name == name) else {
                return Ok(IntExpr::from(default));
            };
            self.resolve_arg_as_expression(&input.arg)
                .or_else(|| {
                    input
                        .arg
                        .as_float()
                        .filter(|v| {
                            v.is_finite()
                                && v.fract() == 0.0
                                && *v >= i64::MIN as f64
                                && *v < -(i64::MIN as f64)
                        })
                        .map(|v| IntExpr::from(v as i64))
                })
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{}: {name} must be an integer or supported symbolic integer",
                        node.target
                    )
                })
        };
        let start = argument("start", 0)?;
        let step = argument("step", 1)?;
        if step.as_num() == Some(0) {
            bail!("arange step must be nonzero");
        }
        // Export provides the exact extent, including ceil division for
        // non-divisible steps and descending ranges. No hint is substituted.
        let name = Self::tensor_output_names(node)
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("arange has no output metadata"))?;
        let meta = self.tensor_meta(&name)?.clone();
        let shape = self.boundary_shape(&meta, &name)?;
        let arange = self
            .cx
            .iota(shape, move |coordinates| coordinates[0] * step + start);
        Ok(arange.cast(dtype))
    }

    /// `zeros_like`/`ones_like`: a full tensor at the operand's own dims.
    pub(super) fn translate_like_fill(&mut self, node: &Node, value: f64) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        Ok(self.full_tensor(x.dims(), x.dtype, value))
    }

    pub(super) fn translate_scalar_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        if let Some(name) = node.inputs[0].arg.as_value_name()
            && let Some(value) = self.values.get(name).copied()
        {
            return Ok(value.cast(dtype));
        }
        let value = self.get_float_arg(node, 0)?;
        let scalar = match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::Int => self.cx.constant_i64(value as i64).cast(DType::Int),
            DType::I64 => self.cx.constant_i64(value as i64),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        };
        Ok(scalar)
    }

    pub(super) fn translate_where(
        &mut self,
        node: &Node,
        scalar_other: bool,
    ) -> Result<GraphTensor> {
        let condition = self.operand(&node.inputs[0])?;
        let a = self.operand(&node.inputs[1])?;
        // torch promotes the two branches to the result dtype; the
        // condition is a mask and is never promoted with them.
        let dtype = self.output_meta_dtype(node).unwrap_or(a.dtype);
        let b = if scalar_other {
            self.scalar(&node.inputs[2], dtype)?
        } else {
            self.operand(&node.inputs[2])?
        };
        let a = super::convert(a, dtype);
        let b = super::convert(b, dtype);
        Ok(self.select(condition, a, b))
    }

    pub(super) fn translate_masked_fill_scalar(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let mask = self.operand(&node.inputs[1])?;
        let value = self.get_float_arg(node, 2)? as f32;
        let fill = self
            .cx
            .constant_f32(value)
            .cast(x.dtype)
            .expand_rhs(x.dims());
        // `masked_fill(mask, value)` = `where(mask, value, x)`.
        Ok(self.select(mask, fill, x))
    }

    pub(super) fn translate_clamp(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let min = node.inputs.get(1).and_then(|i| i.arg.as_float());
        let max = node.inputs.get(2).and_then(|i| i.arg.as_float());
        let mut out = x;
        if let Some(min) = min {
            let bound = self
                .cx
                .constant_f32(min as f32)
                .cast(x.dtype)
                .expand_rhs(x.dims());
            out = out.maximum(bound);
        }
        if let Some(max) = max {
            let bound = self
                .cx
                .constant_f32(max as f32)
                .cast(x.dtype)
                .expand_rhs(x.dims());
            out = out.minimum(bound);
        }
        Ok(out)
    }

    pub(super) fn translate_clamp_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let mut out = x;
        if let Some(min) = self.optional_tensor_operand(&node.inputs[1])? {
            let (out_b, min) = util::broadcast_binary(out, min);
            out = out_b.maximum(min);
        }
        if let Some(max) = node
            .inputs
            .get(2)
            .map(|input| self.optional_tensor_operand(input))
            .transpose()?
            .flatten()
        {
            let (out_b, max) = util::broadcast_binary(out, max);
            out = out_b.minimum(max);
        }
        Ok(out)
    }

    pub(super) fn translate_softmax(&mut self, node: &Node, log: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        if x.rank() == 0 {
            // Preserve NaN for infinite inputs, as ATen scalar softmax does.
            #[allow(clippy::eq_op)]
            let zero = x - x;
            return Ok(if log { zero } else { zero.exp() });
        }
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        Ok(if log {
            x.log_softmax(dim)
        } else {
            x.softmax(dim)
        })
    }

    /// `embedding(weight, indices)`: coordinate gather over `[rows, cols]`.
    /// No flat-index arithmetic: the coordinate form needs no value-range
    /// proof (plain Int math is proof-gated on this branch).
    pub(super) fn translate_embedding(&mut self, node: &Node) -> Result<GraphTensor> {
        let weight = self.operand(&node.inputs[0])?;
        let indices = self.operand(&node.inputs[1])?;
        let d = weight.dims()[1];
        let idx = indices.cast(DType::Int);
        let mut out_dims = idx.dims();
        out_dims.push(d);
        let rows = idx.expand_dim(idx.rank(), d);
        let cols = self.cx.iota(out_dims.clone(), |c| c[c.len() - 1]);
        Ok(weight.gather(&[rows, cols]))
    }

    pub(super) fn translate_item(&mut self, node: &Node) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let scalar = util::reshape_tensor(x, &[]);
        let name = node
            .outputs
            .iter()
            .find_map(|output| output.value_name())
            .context("item is missing its scalar output name")?
            .to_owned();
        self.values.insert(name, scalar);
        Ok(())
    }

    /// `native_layer_norm(input, normalized_shape, weight?, bias?, eps)`
    /// -> (out, mean, rstd).
    pub(super) fn translate_layer_norm(&mut self, node: &Node) -> Result<()> {
        let x = self.operand(&node.inputs[0])?;
        let normalized: Vec<i64> = self
            .get_ints_arg(node, 1)
            .unwrap_or_else(|_| vec![x.rank() as i64]);
        let weight = self.optional_operand_at_compute(&node.inputs[2])?;
        let bias = self.optional_operand_at_compute(&node.inputs[3])?;
        let eps = self.get_float_arg(node, 4).unwrap_or(1e-5) as f32;
        let rank = x.rank();
        let n = normalized.len().min(rank);
        let axes: Vec<usize> = (rank - n..rank).collect();
        let mut out = x.layer_norm(axes.clone(), eps);
        if let Some(weight) = weight {
            let (o, w) = util::broadcast_binary(out, weight);
            out = o * w;
        }
        if let Some(bias) = bias {
            let (o, b) = util::broadcast_binary(out, bias);
            out = o + b;
        }
        let mean = x.mean(axes.clone());
        let var = x.var_options(axes.clone(), 0);
        let eps_const = self.cx.constant_f32(eps).expand_rhs(var.dims());
        let rstd = (var + eps_const).sqrt().reciprocal();
        // `native_layer_norm` returns (out, mean, rstd) with the normalized
        // axes kept as size-1 extents; `layer_norm` returns just `out`.
        if node.outputs.len() > 1 {
            let mean = keep_axes(mean, &axes);
            let rstd = keep_axes(rstd, &axes);
            self.bind_outputs(node, vec![out, mean, rstd])
        } else {
            self.bind_outputs(node, vec![out])
        }
    }

    /// `inputs` sugar: a named boolean arg.
    fn inputs_bool(&self, node: &Node, name: &str) -> Option<bool> {
        self.named_bool_arg(node, name)
    }

    /// An optional `int[1]?` dim list: absent/None is the empty list. A bare
    /// scalar int named `dim` is the one-element list (`prod.dim_int` spells
    /// `dim` that way); the name is load-bearing because PT2 drops omitted
    /// defaults, so at this index another argument (`correction`) can sit.
    fn optional_int_list(&self, node: &Node, idx: usize) -> Result<Vec<i64>> {
        match node.inputs.get(idx) {
            Some(input) if input.arg.as_ints().is_some() || input.arg.as_sym_ints().is_some() => {
                self.get_ints_arg(node, idx)
            }
            Some(input) if input.name == "dim" => {
                Ok(input.arg.as_int().map_or_else(Vec::new, |dim| vec![dim]))
            }
            _ => Ok(Vec::new()),
        }
    }
}

/// Re-insert the reduced axes as size-1 extents (keepdim).
fn keep_axes(mut t: GraphTensor, axes: &[usize]) -> GraphTensor {
    let mut sorted = axes.to_vec();
    sorted.sort_unstable();
    for axis in sorted {
        t = t.expand_dim(axis, 1usize);
    }
    t
}
