//! Shared translator helpers, ported from the parked HLIR translator's
//! `pt2_util`/`mod` helper surface and re-expressed on the recorder
//! frontend (dims-based, no ShapeTracker).
//!
//! The surface is deliberately complete for the remaining category
//! ports, so some helpers are not yet called by this crate.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{DimSize, Node, SymIntEntry, TensorMeta};

pub(super) fn same_dim(lhs: IntExpr, rhs: IntExpr) -> bool {
    lhs == rhs || lhs.simplify() == rhs.simplify() || lhs.egglog_equal(rhs)
}

/// Normalize a potentially negative dimension index.
pub(super) fn normalize_dim(dim: i64, ndim: usize) -> usize {
    if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    }
}

/// Normalize a possibly-negative slice bound against the axis extent.
pub(super) fn normalize_slice_bound(bound: IntExpr, dim_size: IntExpr) -> IntExpr {
    match bound.as_num() {
        Some(n) if n < 0 => (dim_size + IntExpr::from(n as i32)).simplify(),
        _ => bound,
    }
}

/// Right-align two tensors and expand size-1 axes (NumPy/torch broadcast).
pub(super) fn broadcast_binary(
    mut a: GraphTensor,
    mut b: GraphTensor,
) -> (GraphTensor, GraphTensor) {
    let (a_rank, b_rank) = (a.rank(), b.rank());
    for _ in 0..b_rank.saturating_sub(a_rank) {
        a = a.expand_dim(0, 1usize);
    }
    for _ in 0..a_rank.saturating_sub(b_rank) {
        b = b.expand_dim(0, 1usize);
    }
    let rank = a.rank();
    let ad = a.dims();
    let bd = b.dims();
    let mut a_target = Vec::with_capacity(rank);
    let mut b_target = Vec::with_capacity(rank);
    for i in 0..rank {
        if same_dim(ad[i], bd[i]) {
            a_target.push(ad[i]);
            b_target.push(bd[i]);
        } else if ad[i].to_usize() == Some(1) {
            a_target.push(bd[i]);
            b_target.push(bd[i]);
        } else if bd[i].to_usize() == Some(1) {
            a_target.push(ad[i]);
            b_target.push(ad[i]);
        } else {
            panic!("incompatible broadcast dims {i}: {} vs {}", ad[i], bd[i]);
        }
    }
    (a.expand(a_target), b.expand(b_target))
}

/// Promote two tensors to a common dtype following torch's promotion
/// lattice (wider wins; floats beat ints; f16/bf16 mixed promotes to f32).
pub(super) fn ensure_same_dtype(a: GraphTensor, b: GraphTensor) -> (GraphTensor, GraphTensor) {
    if a.dtype == b.dtype {
        return (a, b);
    }
    let target = match (a.dtype, b.dtype) {
        (DType::F64, _) | (_, DType::F64) => DType::F64,
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        (DType::F16, DType::Bf16) | (DType::Bf16, DType::F16) => DType::F32,
        (DType::F16, _) | (_, DType::F16) => DType::F16,
        (DType::Bf16, _) | (_, DType::Bf16) => DType::Bf16,
        (DType::I64, _) | (_, DType::I64) => DType::I64,
        (DType::Int, _) | (_, DType::Int) => DType::Int,
        _ => DType::F32,
    };
    (
        if a.dtype == target { a } else { a.cast(target) },
        if b.dtype == target { b } else { b.cast(target) },
    )
}

/// Reshape by view: flatten then split each leading target extent off the
/// remaining axis. The total element count must agree (torch view rules).
pub(super) fn reshape_tensor(t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    if t.dims() == target {
        return t;
    }
    if target.is_empty() {
        return t.flatten().squeeze(0);
    }
    let mut flat = t.flatten();
    for (i, dim) in target.iter().enumerate() {
        if i + 1 == target.len() {
            break;
        }
        flat = flat.split_dims(i, *dim);
    }
    flat
}

/// Value-identity: the recorder has no strides, so a copy of a view is
/// only semantically distinct at the boundary, which the binding and
/// search handle. Kept as a named seam for the parked `*_copy` ops.
pub(super) fn materialize_tensor(t: GraphTensor) -> GraphTensor {
    t
}

/// Resolve `-1` in a reshape target against the input's logical dims.
pub(super) fn resolve_neg1_dim(target: &[i64], current_dims: &[IntExpr]) -> Vec<IntExpr> {
    let mut neg1_idx = None;
    let mut known_product: i64 = 1;
    let mut result = Vec::with_capacity(target.len());
    for (i, &s) in target.iter().enumerate() {
        if s == -1 {
            neg1_idx = Some(i);
            result.push(IntExpr::from(0usize));
        } else {
            known_product *= s;
            result.push(IntExpr::from(s as usize));
        }
    }
    if let Some(idx) = neg1_idx {
        let product: IntExpr = current_dims
            .iter()
            .fold(IntExpr::from(1), |acc, d| acc * *d);
        result[idx] = product / IntExpr::from(known_product as usize);
    }
    result
}

/// `-1` resolution for targets already carrying `IntExpr` extents.
pub(super) fn resolve_neg1_dim_exprs(target: &[IntExpr], current_dims: &[IntExpr]) -> Vec<IntExpr> {
    let neg1 = IntExpr::from(-1i32);
    let Some(idx) = target.iter().position(|e| *e == neg1) else {
        return target.to_vec();
    };
    let input_product: IntExpr = current_dims
        .iter()
        .fold(IntExpr::from(1), |acc, d| acc * *d);
    let known_product: IntExpr = target
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != idx)
        .fold(IntExpr::from(1), |acc, (_, d)| acc * *d);
    let mut result = target.to_vec();
    result[idx] = (input_product / known_product).simplify();
    result
}

impl Translator<'_> {
    /// A typed scalar constant matching `tensor`'s dtype and dims.
    pub(super) fn constant_like(&mut self, tensor: GraphTensor, value: f64) -> GraphTensor {
        let scalar = match tensor.dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::I64 => self.cx.constant_i64(value as i64),
            DType::Int => self.cx.constant_i64(value as i64).cast(DType::Int),
            _ => self.cx.constant_f32(value as f32).cast(tensor.dtype),
        };
        scalar.expand_rhs(tensor.dims())
    }

    /// `where(condition, a, b)` over already-broadcast operands.
    pub(super) fn select(
        &mut self,
        condition: GraphTensor,
        a: GraphTensor,
        b: GraphTensor,
    ) -> GraphTensor {
        let (a, condition) = broadcast_binary(a, condition);
        let (a, b) = broadcast_binary(a, b);
        let mask = condition.cast(DType::F32);
        let one = self.cx.constant_f32(1.0).expand_rhs(mask.dims());
        a * mask + b * (one - mask)
    }

    pub(super) fn bool_or(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = broadcast_binary(a, b);
        let (af, bf) = (a.cast(DType::F32), b.cast(DType::F32));
        (af + bf - af * bf).cast(DType::Bool)
    }

    pub(super) fn bool_and(&mut self, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = broadcast_binary(a, b);
        (a.cast(DType::F32) * b.cast(DType::F32)).cast(DType::Bool)
    }

    pub(super) fn bool_not(&mut self, value: GraphTensor) -> GraphTensor {
        let one = self.cx.constant_f32(1.0).expand_rhs(value.dims());
        (one - value.cast(DType::F32)).cast(DType::Bool)
    }

    pub(super) fn real_abs(&mut self, value: GraphTensor) -> GraphTensor {
        value.abs()
    }

    pub(super) fn is_nan(&mut self, value: GraphTensor) -> GraphTensor {
        value.ne(value)
    }

    pub(super) fn is_inf(&mut self, value: GraphTensor) -> GraphTensor {
        let largest = match value.dtype {
            DType::F64 => f64::MAX,
            _ => f32::MAX as f64,
        };
        let positive = value.gt(self.constant_like(value, largest));
        let negative = value.lt(self.constant_like(value, -largest));
        self.bool_or(positive, negative)
    }

    pub(super) fn is_zero(&mut self, value: GraphTensor) -> GraphTensor {
        let x = value.cast(DType::F32);
        let zero = self.cx.constant_f32(0.0).expand_rhs(x.dims());
        let nonzero = self.bool_or(x.lt(zero), x.gt(zero));
        self.bool_not(nonzero)
    }

    pub(super) fn signbit(&mut self, value: GraphTensor) -> GraphTensor {
        // Negative iff x < 0, or x is a negative zero / NaN: the
        // reciprocal flips the sign bit for every finite value and keeps
        // -0.0 negative.
        let zero = self.constant_like(value, 0.0);
        value.reciprocal().lt(zero)
    }

    pub(super) fn copy_sign(&mut self, magnitude: GraphTensor, sign: GraphTensor) -> GraphTensor {
        let (magnitude, sign) = broadcast_binary(magnitude, sign);
        magnitude.abs() * sign.sign().cast(magnitude.dtype)
    }

    /// A floating scalar of the tensor's own dtype (F64 keeps doubles).
    pub(super) fn floating_scalar(&mut self, value: f64, dtype: DType) -> GraphTensor {
        match dtype {
            DType::F64 => self.cx.constant_f64(value),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        }
    }

    pub(super) fn get_input_tensor(&mut self, node: &Node, idx: usize) -> Result<GraphTensor> {
        let input = node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?;
        self.operand(input)
    }

    pub(super) fn get_int_arg(&self, node: &Node, idx: usize) -> Result<i64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(v) = arg.as_int() {
            return Ok(v);
        }
        if let Some(expr) = self.resolve_arg_as_expression(arg)
            && let Some(v) = expr.to_usize()
        {
            return Ok(v as i64);
        }
        bail!("input {idx} of {} is not an int: {arg:?}", node.target)
    }

    pub(super) fn get_float_arg(&self, node: &Node, idx: usize) -> Result<f64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(f) = arg.as_float() {
            return Ok(f);
        }
        if let Some(i) = arg.as_int() {
            return Ok(i as f64);
        }
        bail!("input {idx} of {} is not a float: {arg:?}", node.target)
    }

    pub(super) fn get_ints_arg(&self, node: &Node, idx: usize) -> Result<Vec<i64>> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        if let Some(entries) = arg.as_sym_ints() {
            let mut out = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    SymIntEntry::Int(i) => out.push(i.as_int),
                    SymIntEntry::Name(s) => {
                        let v = self
                            .resolve_sym_int(&s.as_name)
                            .and_then(|e| e.to_usize())
                            .map(|u| u as i64)
                            .with_context(|| {
                                format!("input {idx} of {} has an unresolved sym_int", node.target)
                            })?;
                        out.push(v);
                    }
                }
            }
            return Ok(out);
        }
        arg.as_ints()
            .map(|v| v.to_vec())
            .with_context(|| format!("input {idx} of {} is not an int list", node.target))
    }

    pub(super) fn get_bool_arg(&self, node: &Node, idx: usize) -> Result<bool> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("{} missing input {idx}", node.target))?
            .arg;
        arg.as_bool()
            .with_context(|| format!("input {idx} of {} is not a bool", node.target))
    }

    pub(super) fn named_int_arg(&self, node: &Node, name: &str) -> Option<i64> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_int_arg(node, idx).ok()
    }

    pub(super) fn named_float_arg(&self, node: &Node, name: &str) -> Option<f64> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_float_arg(node, idx).ok()
    }

    pub(super) fn named_bool_arg(&self, node: &Node, name: &str) -> Option<bool> {
        let idx = node.inputs.iter().position(|input| input.name == name)?;
        self.get_bool_arg(node, idx).ok()
    }

    /// All tensor output names of a node, multi-output ops included.
    pub(super) fn tensor_output_names(node: &Node) -> Vec<String> {
        node.outputs
            .iter()
            .flat_map(|output| {
                output
                    .as_tensors
                    .as_ref()
                    .map(|values| values.iter().map(|value| value.name.clone()).collect())
                    .unwrap_or_else(|| {
                        output
                            .as_tensor
                            .as_ref()
                            .map(|value| vec![value.name.clone()])
                            .unwrap_or_default()
                    })
            })
            .collect()
    }

    /// Shape of the node's first output from its PT2 metadata.
    pub(super) fn output_meta_shape(&self, node: &Node) -> Result<Vec<IntExpr>> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&name)
            .with_context(|| format!("missing tensor meta for output {name}"))?;
        self.tensor_meta_to_shape(meta)
    }

    /// Dtype of the node's first tensor output from its PT2 metadata.
    pub(super) fn output_meta_dtype(&self, node: &Node) -> Result<DType> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();
        let meta = self
            .tensor_meta(&name)
            .with_context(|| format!("missing tensor meta for output {name}"))?;
        super::dtype_of(meta.dtype)
    }

    pub(super) fn tensor_meta_to_shape(&self, meta: &TensorMeta) -> Result<Vec<IntExpr>> {
        meta.sizes
            .iter()
            .map(|s| self.dim_size_to_expr(s))
            .collect()
    }

    pub(super) fn dim_size_to_expr(&self, dim: &DimSize) -> Result<IntExpr> {
        match dim {
            DimSize::Int(i) => Ok(IntExpr::from(i.as_int)),
            DimSize::Expr(e) => self.resolve_expr_value(&e.as_expr).with_context(|| {
                format!("cannot resolve dimension expression {}", e.as_expr.expr_str)
            }),
        }
    }

    /// A full tensor of `value` at `shape`, using the exact-constant path
    /// per dtype (F64 keeps double precision).
    pub(super) fn full_tensor(
        &mut self,
        shape: Vec<IntExpr>,
        dtype: DType,
        value: f64,
    ) -> GraphTensor {
        let scalar = match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::I64 => self.cx.constant_i64(value as i64),
            DType::Int => self.cx.constant_i64(value as i64).cast(DType::Int),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        };
        scalar.expand_rhs(shape)
    }

    /// Coordinate arange along `axis`, expanded across the other axes.
    pub(super) fn axis_positions(&mut self, full_shape: &[IntExpr], axis: usize) -> GraphTensor {
        let mut positions = self.cx.arange(full_shape[axis]);
        for (dim, size) in full_shape.iter().copied().enumerate() {
            if dim != axis {
                positions = positions.expand_dim(dim, size);
            }
        }
        positions
    }
}
