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
        let (a, b) = (self.truth(a), self.truth(b));
        Ok(if or {
            a.logical_or(b)
        } else if xor {
            a.logical_or(b).logical_and(a.logical_and(b).logical_not())
        } else {
            a.logical_and(b)
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
        let result = (b * a.log2()).exp2();
        // The logarithmic form yields NaN for zero exponents at zero or nonfinite bases.
        let zero = self.constant_like(b, 0.0);
        let one = self.constant_like(a, 1.0);
        Ok(b.eq(zero).select(one, result))
    }

    pub(super) fn pow_scalar_base(&mut self, node: &Node) -> Result<GraphTensor> {
        let base = self.get_float_arg(node, 0)?;
        let b = self.operand(&node.inputs[1])?;
        let log_base = self
            .floating_scalar(base.ln(), b.dtype)
            .expand_rhs(b.dims());
        let result = (b * log_base).exp();
        let zero = self.constant_like(b, 0.0);
        let one = self.constant_like(b, 1.0);
        Ok(b.eq(zero).select(one, result))
    }

    pub(super) fn fmod_remainder(&mut self, node: &Node, fmod: bool) -> Result<GraphTensor> {
        let a = self.operand(&node.inputs[0])?;
        let b = if let Some(t) = self.optional_tensor_operand(&node.inputs[1])? {
            t
        } else {
            self.scalar(&node.inputs[1], a.dtype)?
        };
        let (a, b) = util::broadcast_binary(a, b);
        if matches!(
            a.dtype,
            DType::Int | DType::I64 | DType::I8 | DType::I16 | DType::U8
        ) {
            let dtype = a.dtype;
            // Small integers compute in i32. Never pass integer payloads
            // through floating division: even f64 cannot represent all i64s.
            let (a, b) = if matches!(dtype, DType::Int | DType::I64) {
                (a, b)
            } else {
                (a.cast(DType::Int), b.cast(DType::Int))
            };
            let r = a.trunc_rem(b);
            if fmod {
                return Ok(r.cast(dtype));
            }
            let adjust = self.remainder_sign_disagreement(r, b);
            return Ok((r + b * adjust).cast(dtype));
        }
        if fmod {
            return Ok(a % b);
        }
        // Floating Python-style remainder: a - floor(a/b)*b.
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
        anyhow::ensure!(step > 0, "slice step must be positive, got {step}");
        if step != 1 {
            let name = Self::tensor_output_names(node)
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("slice has no output metadata"))?;
            let shape = self.tensor_meta_to_shape(self.tensor_meta(&name)?)?;
            let start = start.max(0).min(x.dims()[dim]);
            let coords = (0..rank)
                .map(|axis| {
                    self.cx.iota(shape.clone(), |c| {
                        if axis == dim {
                            start + c[axis] * IntExpr::from(step)
                        } else {
                            c[axis]
                        }
                    })
                })
                .collect::<Vec<_>>();
            return Ok(x.gather(&coords));
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
        Ok(x.repeat(repeats.as_slice()))
    }

    pub(super) fn translate_stack(&mut self, node: &Node) -> Result<GraphTensor> {
        let tensors = node.inputs[0]
            .arg
            .as_tensors()
            .ok_or_else(|| anyhow::anyhow!("stack: first operand is not a tensor list"))?;
        let raw_axis = self.named_int_arg(node, "dim").unwrap_or(0);
        let mut values = Vec::with_capacity(tensors.len());
        for t in tensors {
            values.push(
                *self
                    .values
                    .get(&t.name)
                    .ok_or_else(|| anyhow::anyhow!("stack: unknown tensor {}", t.name))?,
            );
        }
        let first = values.first().context("stack: empty tensor list")?;
        let output_rank = first.rank() + 1;
        let axis = if raw_axis < 0 {
            raw_axis + output_rank as i64
        } else {
            raw_axis
        };
        let axis = usize::try_from(axis)
            .ok()
            .filter(|axis| *axis < output_rank)
            .with_context(|| {
                format!("stack: axis {raw_axis} out of range for output rank {output_rank}")
            })?;
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
        let rank = x.rank();
        let axes = match dim {
            Some(d) if rank == 0 => {
                anyhow::ensure!(d == 0 || d == -1, "argextremum: scalar dim must be 0 or -1");
                Vec::new()
            }
            Some(d) => vec![normalize_dim(d, rank)],
            None => (0..rank).collect(),
        };
        for &axis in &axes {
            anyhow::ensure!(
                x.dims()[axis].to_usize() != Some(0),
                "argextremum: reduction axis {axis} must be nonempty"
            );
        }
        let result = if rank == 0 {
            self.cx.constant_i32(0)
        } else if dim.is_none() {
            let flat = util::reshape_tensor(x, &[x.dims().into_iter().product()]);
            if max { flat.argmax(0) } else { flat.argmin(0) }
        } else if max {
            x.argmax(axes[0])
        } else {
            x.argmin(axes[0])
        };
        let dtype = self.output_meta_dtype(node).unwrap_or(DType::I64);
        let result = if result.dtype != dtype {
            result.cast(dtype)
        } else {
            result
        };
        Ok(if keepdim {
            keep_axes(result, &axes)
        } else {
            result
        })
    }

    pub(super) fn translate_cumulative(&mut self, node: &Node, prod: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        // ATen casts BEFORE scanning: absent dtype promotes bool/integer
        // inputs to int64; an explicit dtype overrides that default. PT2's
        // output metadata has already resolved both cases, even when the
        // exported node has no dtype argument or explicit cast. Luminal's
        // scans themselves preserve their operand dtype.
        let result = self.output_meta_dtype(node)?;
        let x = super::convert(x, result);
        let dim = self.get_int_arg(node, 1)?;
        // A rank-0 scan is the identity AFTER the dtype conversion.
        if x.rank() == 0 {
            if dim != 0 && dim != -1 {
                bail!("dim {dim} is out of range for a rank-0 value");
            }
            return Ok(x);
        }
        let dim = normalize_dim(dim, x.rank());
        // The running total accumulates in the result dtype's opmath
        // (ATen's `acc_type`: F32 for F16/BF16) and each partial sum is
        // rounded to the result dtype once. A half-precision scan that
        // rounded every step would drift by whole ulps of the total.
        let compute = super::opmath_compute(result);
        let x = super::convert(x, compute);
        let scanned = if prod {
            x.cumprod(vec![dim])
        } else {
            x.cumsum(vec![dim])
        };
        Ok(super::convert(scanned, result))
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
            (x.max(vec![dim]), x.argmax(dim))
        } else {
            (x.min(vec![dim]), x.argmin(dim))
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

    pub(super) fn translate_full(&mut self, node: &Node, like: bool) -> Result<GraphTensor> {
        let (shape, dtype, value) = if like {
            let x = self.operand(&node.inputs[0])?;
            let meta = self.output_meta_dtype(node).unwrap_or(x.dtype);
            (x.dims(), meta, self.get_number_arg(node, 1)?)
        } else {
            let shape = self.get_int_exprs_arg(node, 0)?;
            (
                shape,
                self.output_meta_dtype(node)?,
                self.get_number_arg(node, 1)?,
            )
        };
        Ok(self.full_tensor(shape, dtype, value))
    }

    pub(super) fn translate_arange(&mut self, node: &Node, _kind: u8) -> Result<GraphTensor> {
        let dtype = self.output_meta_dtype(node)?;
        // PT2 omits default arguments. Read by schema name, and preserve
        // symbolic scalar operands rather than interpreting them as literals.
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
        self.scalar(&node.inputs[0], dtype)
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
        let fill = self.scalar(&node.inputs[2], x.dtype)?.expand_rhs(x.dims());
        // `masked_fill(mask, value)` = `where(mask, value, x)`.
        Ok(self.select(mask, fill, x))
    }

    pub(super) fn translate_clamp(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dtype = self.output_meta_dtype(node)?;
        let mut out = super::convert(x, dtype);
        for name in ["min", "max"] {
            if let Some(input) = node.inputs.iter().find(|input| input.name == name)
                && (input.arg.as_int().is_some() || input.arg.as_float().is_some())
            {
                let bound = self.scalar(input, dtype)?.expand_rhs(x.dims());
                out = if name == "min" {
                    out.maximum(bound)
                } else {
                    out.minimum(bound)
                };
            }
        }
        Ok(out)
    }

    pub(super) fn translate_clamp_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let mut out = x;
        if let Some(min) = self.named_tensor_operand(node, "min")? {
            let (out_b, min) = util::broadcast_binary(out, min);
            out = out_b.maximum(min);
        }
        if let Some(max) = self.named_tensor_operand(node, "max")? {
            let (out_b, max) = util::broadcast_binary(out, max);
            out = out_b.minimum(max);
        }
        Ok(out)
    }

    pub(super) fn translate_softmax(&mut self, node: &Node, log: bool) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        Ok(if log {
            x.log_softmax(vec![dim])
        } else {
            x.softmax(vec![dim])
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
        let names = Self::tensor_output_names(node);
        let name = names
            .first()
            .context("item is missing its scalar output name")?
            .clone();
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

#[cfg(test)]
mod stack_tests {
    use super::super::test_support::*;
    use crate::pt2_schema::{Argument, TensorName, TensorsArg};
    use crate::translate::translate;

    #[test]
    fn stack_axes_use_the_output_rank() {
        for input_shape in [vec![], vec![5], vec![2, 3]] {
            let output_rank = input_shape.len() + 1;
            for axis in 0..output_rank {
                for raw_axis in [axis as i64, axis as i64 - output_rank as i64] {
                    let mut shape = input_shape.clone();
                    shape.insert(axis, 2);
                    let parsed = parsed_nodes(
                        vec![node(
                            "torch.ops.aten.stack.default",
                            vec![
                                input(
                                    "tensors",
                                    Argument::Tensors(TensorsArg {
                                        as_tensors: vec![
                                            TensorName { name: "a".into() },
                                            TensorName { name: "b".into() },
                                        ],
                                    }),
                                ),
                                scalar_int("dim", raw_axis),
                            ],
                            &["out"],
                        )],
                        &[
                            ("a", FLOAT, &input_shape),
                            ("b", FLOAT, &input_shape),
                            ("out", FLOAT, &shape),
                        ],
                        &["a", "b"],
                        &["out"],
                    );
                    let translated = translate(&parsed).unwrap();
                    let actual = translated
                        .graph
                        .logical
                        .value_dims(translated.outputs[0].tensor);
                    assert_eq!(
                        actual
                            .iter()
                            .map(|d| d.to_usize().unwrap())
                            .collect::<Vec<_>>(),
                        shape.iter().map(|d| *d as usize).collect::<Vec<_>>(),
                        "axis {raw_axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn stack_rejects_out_of_range_axes() {
        for axis in [-3, 2] {
            let parsed = parsed_nodes(
                vec![node(
                    "torch.ops.aten.stack.default",
                    vec![
                        input(
                            "tensors",
                            Argument::Tensors(TensorsArg {
                                as_tensors: vec![
                                    TensorName { name: "a".into() },
                                    TensorName { name: "b".into() },
                                ],
                            }),
                        ),
                        scalar_int("dim", axis),
                    ],
                    &["out"],
                )],
                &[
                    ("a", FLOAT, &[5]),
                    ("b", FLOAT, &[5]),
                    ("out", FLOAT, &[5, 2]),
                ],
                &["a", "b"],
                &["out"],
            );
            let error = translate(&parsed).err().expect("invalid axis must fail");
            assert!(format!("{error:#}").contains("out of range"));
        }
    }
}
