//! Indexing, gather/scatter, and embedding-bag lowerings (port batch 2).
//!
//! Gather/scatter are built from the recorder's coordinate forms so no
//! caller-data flat-index arithmetic (proof-gated Int add/mul) enters the
//! graph. Index normalization, where-selection, and the reductions that
//! would need Int arithmetic run in F32 and `trunc_cast` back (the plan
//! has no F64 binary arms); anything that cannot be done faithfully bails
//! with its op name.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, normalize_dim, reshape_tensor};
use crate::pt2_schema::{Argument, Node, OptionalTensorEntry};

/// Reduction flavours shared by the scatter-reduce / index-reduce family.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IdxReduce {
    Sum,
    Prod,
    Mean,
    Max,
    Min,
}

fn idx_is_float(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
    )
}

fn idx_is_int(dtype: DType) -> bool {
    matches!(dtype, DType::Int | DType::I64)
}

/// Row-major element count when every extent is concrete.
fn idx_numel(dims: &[IntExpr]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(dim.to_usize()?))
}

/// Whether `from` broadcasts to `to` right-aligned.
fn idx_can_broadcast_to(from: &[IntExpr], to: &[IntExpr]) -> bool {
    if from.len() > to.len() {
        return false;
    }
    let offset = to.len() - from.len();
    from.iter()
        .enumerate()
        .all(|(i, dim)| dim.to_usize() == Some(1) || util::same_dim(*dim, to[offset + i]))
}

/// PyTorch advanced-indexing layout for `index.Tensor`/`index_put`:
/// adjacent advanced indices insert their broadcast block at the first
/// indexed axis; indices separated by basic axes put the block first.
/// Returns the output shape, the block's start axis, and
/// `(source_axis, output_axis)` for every non-indexed source axis.
fn idx_advanced_layout(
    source_dims: &[IntExpr],
    indexed: &[usize],
    bshape: &[IntExpr],
) -> (Vec<IntExpr>, usize, Vec<(usize, usize)>) {
    let first = indexed[0];
    let last = indexed[indexed.len() - 1];
    let adjacent = indexed.windows(2).all(|w| w[1] == w[0] + 1);
    let mut out_dims = Vec::new();
    let mut basic = Vec::new();
    let block_start;
    if adjacent {
        out_dims.extend_from_slice(&source_dims[..first]);
        block_start = out_dims.len();
        out_dims.extend_from_slice(bshape);
        let after = out_dims.len();
        out_dims.extend_from_slice(&source_dims[last + 1..]);
        for axis in 0..first {
            basic.push((axis, axis));
        }
        for axis in last + 1..source_dims.len() {
            basic.push((axis, after + axis - last - 1));
        }
    } else {
        block_start = 0;
        out_dims.extend_from_slice(bshape);
        for (axis, dim) in source_dims.iter().enumerate() {
            if !indexed.contains(&axis) {
                basic.push((axis, out_dims.len()));
                out_dims.push(*dim);
            }
        }
    }
    (out_dims, block_start, basic)
}

impl Translator<'_> {
    // ---------------------------------------------------------------
    // Shared helpers
    // ---------------------------------------------------------------

    /// `where(cond, a, b)` that works for index dtypes: Int blends in F32
    /// and truncates back (Int arithmetic is proof-gated, and the plan has
    /// no F64 binary arms), Bool uses the boolean primitives, floats use
    /// `cond`.
    fn idx_cond(&mut self, cond: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        let (a, b) = util::broadcast_binary(a, b);
        let (a, cond) = util::broadcast_binary(a, cond);
        match a.dtype {
            DType::Bool => {
                let yes = self.bool_and(cond, a);
                let not_cond = self.bool_not(cond);
                let no = self.bool_and(not_cond, b);
                self.bool_or(yes, no)
            }
            DType::Int | DType::I64 => {
                let dtype = a.dtype;
                let a32 = a.cast(DType::F32);
                let b32 = b.cast(DType::F32);
                let mask = cond.cast(DType::F32);
                let one = self.cx.constant_f32(1.0).expand_rhs(mask.dims());
                (a32 * mask + b32 * (one - mask)).trunc_cast(dtype)
            }
            _ => a.cond(cond, b),
        }
    }

    /// Wrap negative index values into `[0, dim)` without proof-gated Int
    /// arithmetic (the adjustment runs in F32).
    fn idx_normalize(&mut self, indices: GraphTensor, dim: IntExpr) -> GraphTensor {
        let zero = self.cx.constant_i32(0).expand_rhs(indices.dims());
        let negative = indices.lt(zero);
        let dim_f = self
            .cx
            .constant_i32(dim)
            .cast(DType::F32)
            .expand_rhs(indices.dims());
        let shifted = (indices.cast(DType::F32) + dim_f).trunc_cast(DType::Int);
        self.idx_cond(negative, shifted, indices)
    }

    /// Cast to a storage dtype, using the explicit truncating read for
    /// float -> int.
    fn idx_cast_like(&mut self, t: GraphTensor, dtype: DType) -> GraphTensor {
        if t.dtype == dtype {
            return t;
        }
        if idx_is_float(t.dtype) && idx_is_int(dtype) {
            return t.trunc_cast(dtype);
        }
        t.cast(dtype)
    }

    /// Right-aligned broadcast to `target` (prepends size-1 dims, expands).
    fn idx_broadcast_to(&mut self, mut t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
        while t.rank() < target.len() {
            t = t.expand_dim(0, 1usize);
        }
        t.expand(target.to_vec())
    }

    /// Slice `updates` down to `shape` along every axis whose extent differs.
    fn idx_crop_to(&mut self, mut updates: GraphTensor, shape: &[IntExpr]) -> GraphTensor {
        for (axis, want) in shape.iter().enumerate() {
            if axis >= updates.rank() {
                break;
            }
            if util::same_dim(updates.dims()[axis], *want) {
                continue;
            }
            updates = updates.slice_along(IntExpr::from(0)..*want, axis);
        }
        updates
    }

    /// Broadcast every present index tensor to their common shape and
    /// return that shape.
    fn idx_broadcast_indices(&mut self, tensors: &mut [Option<GraphTensor>]) -> Vec<IntExpr> {
        let present: Vec<usize> = (0..tensors.len())
            .filter(|d| tensors[*d].is_some())
            .collect();
        if present.is_empty() {
            return Vec::new();
        }
        let mut current = tensors[present[0]].unwrap();
        for &axis in &present[1..] {
            let other = tensors[axis].unwrap();
            let (a, b) = util::broadcast_binary(current, other);
            current = a;
            tensors[axis] = Some(b);
        }
        let shape = current.dims();
        for &axis in &present {
            let t = tensors[axis].unwrap();
            let aligned = t.rank() == shape.len()
                && t.dims()
                    .iter()
                    .zip(&shape)
                    .all(|(a, b)| util::same_dim(*a, *b));
            if !aligned {
                tensors[axis] = Some(self.idx_broadcast_to(t, &shape));
            }
        }
        shape
    }

    /// Coordinate tensor per source axis for an advanced-index layout.
    fn idx_advanced_coords(
        &mut self,
        out_dims: &[IntExpr],
        block_start: usize,
        basic: &[(usize, usize)],
        index_for_dim: &[Option<GraphTensor>],
    ) -> Vec<GraphTensor> {
        let block_len = out_dims.len() - basic.len();
        let non_block: Vec<usize> = (0..out_dims.len())
            .filter(|p| !(block_start..block_start + block_len).contains(p))
            .collect();
        let mut coords: Vec<Option<GraphTensor>> = vec![None; index_for_dim.len()];
        for (axis, t) in index_for_dim.iter().enumerate() {
            if let Some(t) = t {
                coords[axis] =
                    Some(t.expand_to_shape_on_axes(out_dims.to_vec(), non_block.clone()));
            }
        }
        for (axis, out_axis) in basic {
            coords[*axis] = Some(self.axis_positions(out_dims, *out_axis));
        }
        coords
            .into_iter()
            .map(|c| c.expect("advanced index: every source axis has a coordinate"))
            .collect()
    }

    /// Parse the `indices` input of `index.Tensor`/`index_put` into one
    /// optional entry per indexed axis (None = a basic slice axis).
    fn idx_entries(node: &Node) -> Result<Vec<Option<String>>> {
        let input = node
            .inputs
            .get(1)
            .with_context(|| format!("{} is missing its indices input", node.target))?;
        if let Some(names) = input.arg.as_tensors() {
            return Ok(names.iter().map(|name| Some(name.name.clone())).collect());
        }
        if let Some(entries) = input.arg.as_optional_tensors() {
            return Ok(entries
                .iter()
                .map(|entry| match entry {
                    OptionalTensorEntry::Tensor(t) => Some(t.as_tensor.name.clone()),
                    OptionalTensorEntry::None(_) => None,
                })
                .collect());
        }
        if let Some(name) = input.arg.as_tensor_name() {
            return Ok(vec![Some(name.to_string())]);
        }
        bail!(
            "{}: unsupported indices argument {:?}",
            node.target,
            input.arg
        )
    }

    /// Boolean-mask `x[mask]`: gather the flattened true positions in
    /// row-major order (stable argsort puts the 1s first).
    fn idx_bool_mask_gather(
        &mut self,
        source: GraphTensor,
        mask: GraphTensor,
        node: &Node,
    ) -> Result<GraphTensor> {
        if mask.rank() != source.rank()
            || !mask
                .dims()
                .iter()
                .zip(&source.dims())
                .all(|(a, b)| util::same_dim(*a, *b))
        {
            bail!(
                "{}: a boolean index must match the input shape",
                node.target
            );
        }
        let out_dims = self.output_meta_shape(node)?;
        if out_dims.len() != 1 {
            bail!("{}: a boolean index must produce a 1-D output", node.target);
        }
        let sorted = mask.cast(DType::F32).flatten().stable_argsort(0, true);
        let positions = self.cx.arange(out_dims[0]);
        let indices = sorted.gather(&[positions]);
        Ok(source.flatten().gather(&[indices]))
    }

    /// Gather along `axis` with an index tensor of the output shape (ONNX
    /// GatherElements), built from coordinates.
    fn idx_gather_axis(
        &mut self,
        data: GraphTensor,
        indices: GraphTensor,
        axis: usize,
    ) -> Result<GraphTensor> {
        if data.rank() == 0 || indices.rank() != data.rank() {
            bail!("gather_elements: index rank must match data rank");
        }
        let out_dims = indices.dims();
        let mut coords = Vec::with_capacity(data.rank());
        for d in 0..data.rank() {
            if d == axis {
                coords.push(indices);
            } else {
                coords.push(self.axis_positions(&out_dims, d));
            }
        }
        Ok(data.gather(&coords))
    }

    /// NaN-propagating max/min selection (ATen's amax/amin).
    fn idx_extremum(
        &mut self,
        current: GraphTensor,
        incoming: GraphTensor,
        maximum: bool,
    ) -> GraphTensor {
        let ordered = if maximum {
            self.idx_cond(current.ge(incoming), current, incoming)
        } else {
            self.idx_cond(current.le(incoming), current, incoming)
        };
        if idx_is_float(current.dtype) {
            let current_nan = self.is_nan(current);
            let incoming_nan = self.is_nan(incoming);
            let with_current = self.idx_cond(current_nan, current, ordered);
            self.idx_cond(incoming_nan, incoming, with_current)
        } else {
            ordered
        }
    }

    /// Sequential read/modify/write scatter-reduce: duplicate destinations
    /// must accumulate in update order and the recorder's scatter is
    /// overwrite-only, so one static graph step per update element.
    fn idx_scatter_reduce(
        &mut self,
        data: GraphTensor,
        coords: Vec<GraphTensor>,
        updates: GraphTensor,
        reduction: IdxReduce,
        include_self: bool,
    ) -> Result<GraphTensor> {
        if coords.len() != data.rank() {
            bail!(
                "scatter reduction: {} coordinate tensors for rank {}",
                coords.len(),
                data.rank()
            );
        }
        let count = idx_numel(&updates.dims())
            .context("scatter reduction requires a concrete update element count")?;
        let bool_data = data.dtype == DType::Bool;
        if bool_data && matches!(reduction, IdxReduce::Mean) {
            bail!("scatter reduction `mean` on Bool data is not ported");
        }
        let int_data = idx_is_int(data.dtype);
        let work_dtype = if int_data { DType::F32 } else { data.dtype };
        let original = data.flatten();
        let original_work = if int_data {
            original.cast(DType::F32)
        } else {
            original
        };
        let mut output = original_work;
        let flat_updates = if int_data {
            updates.flatten().cast(DType::F32)
        } else {
            updates.flatten()
        };
        // Collapse the per-axis coordinates into one flat destination index
        // (F32 multiply/add then a single truncation): the gather/scatter
        // coordinate forms take exactly one Int coordinate per axis, so a
        // multi-axis destination cannot address flat storage directly.
        let strides: Vec<IntExpr> = (0..data.rank())
            .map(|i| {
                data.dims()[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, d| acc * *d)
            })
            .collect();
        let mut flat_dest: Option<GraphTensor> = None;
        for (coord, stride) in coords.iter().zip(strides) {
            let stride_f = self
                .cx
                .constant_i32(stride)
                .cast(DType::F32)
                .expand_rhs(coord.dims());
            let contribution = coord.cast(DType::F32) * stride_f;
            flat_dest = Some(match flat_dest {
                Some(acc) => acc + contribution,
                None => contribution,
            });
        }
        let flat_dest = flat_dest
            .context("scatter reduction requires at least one coordinate axis")?
            .flatten()
            .trunc_cast(DType::Int);
        let track_counts = !include_self || matches!(reduction, IdxReduce::Mean);
        let mut counts = self
            .full_tensor(
                data.dims(),
                DType::F32,
                if include_self { 1.0 } else { 0.0 },
            )
            .flatten();
        for step in 0..count {
            let dest = flat_dest.slice_along(step..step + 1, 0);
            let update = flat_updates.slice_along(step..step + 1, 0);
            let current = output.gather(&[dest]);
            let mut combined = match reduction {
                IdxReduce::Sum | IdxReduce::Mean => {
                    if bool_data {
                        self.bool_or(current, update)
                    } else {
                        current + update
                    }
                }
                IdxReduce::Prod => {
                    if bool_data {
                        self.bool_and(current, update)
                    } else {
                        current * update
                    }
                }
                IdxReduce::Max => {
                    if bool_data {
                        self.bool_or(current, update)
                    } else {
                        self.idx_extremum(current, update, true)
                    }
                }
                IdxReduce::Min => {
                    if bool_data {
                        self.bool_and(current, update)
                    } else {
                        self.idx_extremum(current, update, false)
                    }
                }
            };
            if track_counts {
                let count_here = counts.gather(&[dest]);
                if !include_self {
                    let zero = self.cx.constant_f32(0.0).expand_rhs(count_here.dims());
                    let first_update = count_here.eq(zero);
                    combined = self.idx_cond(first_update, update, combined);
                }
                let one = self.cx.constant_f32(1.0).expand_rhs(count_here.dims());
                counts = counts.scatter(&[dest], count_here + one);
            }
            output = output.scatter(&[dest], combined);
        }
        if matches!(reduction, IdxReduce::Mean) {
            let zero = self.cx.constant_f32(0.0).expand_rhs(counts.dims());
            let has_values = counts.gt(zero);
            let divisor = counts.cast(work_dtype);
            let means = output / divisor;
            output = self.idx_cond(has_values, means, original_work);
        }
        let output = if int_data {
            output.trunc_cast(data.dtype)
        } else {
            output
        };
        Ok(reshape_tensor(output, &data.dims()))
    }

    /// Scatter coordinates for an update at `[.., index[c], ..]` (ONNX
    /// ScatterElements): one coordinate tensor per data axis over the
    /// update shape.
    fn idx_scatter_coords(
        &mut self,
        update_dims: &[IntExpr],
        axis: usize,
        index: GraphTensor,
    ) -> Vec<GraphTensor> {
        let mut coords = Vec::with_capacity(update_dims.len());
        for d in 0..update_dims.len() {
            if d == axis {
                coords.push(index);
            } else {
                coords.push(self.axis_positions(update_dims, d));
            }
        }
        coords
    }

    /// A scalar `value` argument of the requested dtype.
    fn idx_scalar_value(&mut self, node: &Node, index: usize, dtype: DType) -> Result<GraphTensor> {
        let arg = &node
            .inputs
            .get(index)
            .with_context(|| format!("{} is missing scalar input {index}", node.target))?
            .arg;
        let value = if let Some(v) = arg.as_bool() {
            if v { 1.0 } else { 0.0 }
        } else if let Some(v) = arg.as_int() {
            v as f64
        } else if let Some(v) = arg.as_float() {
            v
        } else {
            bail!("{}: unsupported scalar value {arg:?}", node.target);
        };
        Ok(match dtype {
            DType::F64 => self.cx.constant_f64(value),
            DType::I64 => self.cx.constant_i64(value as i64),
            DType::Int => self.cx.constant_i32(value as i64),
            DType::Bool => self.cx.constant_f32(value as f32).cast(DType::Bool),
            _ => self.cx.constant_f32(value as f32).cast(dtype),
        })
    }

    fn idx_reduce_arg(&self, node: &Node) -> Result<String> {
        let input = node
            .inputs
            .iter()
            .find(|input| input.name == "reduce")
            .or_else(|| node.inputs.get(4))
            .with_context(|| format!("{} is missing its reduce argument", node.target))?;
        if let Argument::Other(value) = &input.arg {
            if let Some(s) = value.as_str() {
                return Ok(s.to_string());
            }
            if let Some(s) = value.get("as_string").and_then(|v| v.as_str()) {
                return Ok(s.to_string());
            }
        }
        bail!(
            "{}: unsupported reduce argument {:?}",
            node.target,
            input.arg
        )
    }

    /// Legacy `scatter.reduce`/`scatter.value_reduce` names.
    fn idx_legacy_reduction(&self, node: &Node) -> Result<IdxReduce> {
        match self.idx_reduce_arg(node)?.as_str() {
            "add" | "sum" => Ok(IdxReduce::Sum),
            "multiply" | "prod" => Ok(IdxReduce::Prod),
            other => bail!("unsupported {} reduction: {other}", node.target),
        }
    }

    /// Modern `scatter_reduce`/`index_reduce` names.
    fn idx_modern_reduction(&self, node: &Node) -> Result<IdxReduce> {
        match self.idx_reduce_arg(node)?.as_str() {
            "sum" => Ok(IdxReduce::Sum),
            "prod" => Ok(IdxReduce::Prod),
            "mean" => Ok(IdxReduce::Mean),
            "amax" => Ok(IdxReduce::Max),
            "amin" => Ok(IdxReduce::Min),
            other => bail!("unsupported {} reduction: {other}", node.target),
        }
    }

    fn idx_include_self(&self, node: &Node) -> bool {
        self.named_bool_arg(node, "include_self").unwrap_or(true)
    }

    /// `data[mask] = source` with `source` a flat list of the true slots.
    fn idx_masked_scatter(
        &mut self,
        destination: GraphTensor,
        mask: GraphTensor,
        source: GraphTensor,
    ) -> Result<GraphTensor> {
        let output_shape = destination.dims();
        let (mask, destination) = util::broadcast_binary(mask, destination);
        let mask = mask.cast(DType::Bool).flatten();
        let destination_flat = destination.flatten();
        let source_flat = self.idx_cast_like(source, destination.dtype).flatten();
        if source_flat.dims()[0].to_usize() == Some(0) {
            return Ok(reshape_tensor(destination_flat, &output_shape));
        }
        // Prefix count of trues; the source element for true slot i is at
        // rank i, so gather source at max(prefix - 1, 0) only where true.
        let prefix = mask.cast(DType::F32).cumsum(0);
        let one = self.cx.constant_f32(1.0).expand_rhs(prefix.dims());
        let zero = self.cx.constant_f32(0.0).expand_rhs(prefix.dims());
        let positions = (prefix - one).maximum(zero).trunc_cast(DType::Int);
        let updates = source_flat.gather(&[positions]);
        let output = self.idx_cond(mask, updates, destination_flat);
        Ok(reshape_tensor(output, &output_shape))
    }

    // ---------------------------------------------------------------
    // Public lowerings
    // ---------------------------------------------------------------

    /// `aten.index.Tensor`: advanced indexing with one optional tensor per
    /// axis. Coordinates are built per axis (basic axes read their iota),
    /// so no flat-index arithmetic is needed.
    pub(super) fn translate_index_tensor(&mut self, node: &Node) -> Result<GraphTensor> {
        let source = self.operand(&node.inputs[0])?;
        if source.rank() == 0 {
            bail!("index.Tensor on a scalar is not ported");
        }
        let entries = Self::idx_entries(node)?;
        let present: Vec<&String> = entries.iter().flatten().collect();
        if present.is_empty() {
            bail!("index.Tensor: no index tensors");
        }
        if present.len() == 1 {
            let mask =
                self.values.get(present[0]).copied().with_context(|| {
                    format!("index.Tensor: unknown index tensor {}", present[0])
                })?;
            if mask.dtype == DType::Bool {
                return self.idx_bool_mask_gather(source, mask, node);
            }
        }
        let mut index_for_dim: Vec<Option<GraphTensor>> = vec![None; source.rank()];
        let mut indexed = Vec::new();
        for (dim, name) in entries.iter().enumerate() {
            let Some(name) = name else { continue };
            if dim >= source.rank() {
                bail!(
                    "index.Tensor: index dim {dim} out of range for rank {}",
                    source.rank()
                );
            }
            let t = self
                .values
                .get(name)
                .copied()
                .with_context(|| format!("index.Tensor: unknown index tensor {name}"))?;
            if t.dtype == DType::Bool {
                bail!("index.Tensor: a boolean index cannot be mixed with tensor indices");
            }
            index_for_dim[dim] = Some(self.idx_normalize(t.cast(DType::Int), source.dims()[dim]));
            indexed.push(dim);
        }
        let bshape = self.idx_broadcast_indices(&mut index_for_dim);
        let (out_dims, block_start, basic) = idx_advanced_layout(&source.dims(), &indexed, &bshape);
        let coords = self.idx_advanced_coords(&out_dims, block_start, &basic, &index_for_dim);
        Ok(source.gather(&coords))
    }

    /// `aten.index_select.default`: gather `index` along `dim`. The index
    /// is rank 0 or 1; the output replaces `dim` with the index extent.
    pub(super) fn translate_index_select(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        if x.rank() == 0 {
            if !self.output_meta_shape(node)?.is_empty() {
                bail!("index_select on a scalar must produce a scalar");
            }
            return Ok(x);
        }
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        let index = self.operand(&node.inputs[2])?.cast(DType::Int);
        let index = if index.rank() == 0 {
            index.unsqueeze(0)
        } else {
            index
        };
        if index.rank() != 1 {
            bail!(
                "index_select index must be rank 0 or 1, got rank {}",
                index.rank()
            );
        }
        let index = self.idx_normalize(index, x.dims()[dim]);
        let out_dims = self.output_meta_shape(node)?;
        let inserted: Vec<usize> = (0..x.rank()).filter(|axis| *axis != dim).collect();
        let axis_coord = index.expand_to_shape_on_axes(out_dims.clone(), inserted);
        let mut coords = Vec::with_capacity(x.rank());
        for axis in 0..x.rank() {
            if axis == dim {
                coords.push(axis_coord);
            } else {
                coords.push(self.axis_positions(&out_dims, axis));
            }
        }
        Ok(x.gather(&coords))
    }

    /// `aten.gather.default`: `out[i...] = x[i..., index[i...], ...]` with
    /// the index shaping the output.
    pub(super) fn translate_gather(&mut self, node: &Node) -> Result<GraphTensor> {
        let x = self.operand(&node.inputs[0])?;
        if x.rank() == 0 {
            bail!("gather on a scalar is not ported");
        }
        let dim = normalize_dim(self.get_int_arg(node, 1)?, x.rank());
        let indices = self.operand(&node.inputs[2])?.cast(DType::Int);
        // Eager permits rank-1 data with a rank-0 index, producing a scalar.
        if indices.rank() == 0 && x.rank() == 1 {
            let index = self.idx_normalize(indices, x.dims()[dim]);
            return Ok(x.gather(&[index]));
        }
        if indices.rank() != x.rank() {
            bail!(
                "gather index rank {} != input rank {}",
                indices.rank(),
                x.rank()
            );
        }
        let index = self.idx_normalize(indices, x.dims()[dim]);
        let out_dims = index.dims();
        let mut coords = Vec::with_capacity(x.rank());
        for axis in 0..x.rank() {
            if axis == dim {
                coords.push(index);
            } else {
                coords.push(self.axis_positions(&out_dims, axis));
            }
        }
        Ok(x.gather(&coords))
    }

    /// The `scatter.*` family. `kind` is 0 src, 1 value, 2 reduce,
    /// 3 value_reduce, 4 scatter_add, 5 scatter_reduce.
    pub(super) fn translate_scatter(&mut self, node: &Node, kind: u8) -> Result<GraphTensor> {
        let data = self.operand(&node.inputs[0])?;
        let rank = data.rank();
        let raw_dim = self.get_int_arg(node, 1)?;
        let dim = if rank == 0 {
            if !matches!(raw_dim, -1 | 0) {
                bail!("scatter dim {raw_dim} out of range for a scalar");
            }
            0
        } else {
            normalize_dim(raw_dim, rank)
        };
        let indices = self.operand(&node.inputs[2])?.cast(DType::Int);
        let (updates, reduction) = match kind {
            0 => (self.operand(&node.inputs[3])?, None),
            1 => (
                self.idx_scalar_value(node, 3, data.dtype)?
                    .expand_rhs(indices.dims()),
                None,
            ),
            2 => (
                self.operand(&node.inputs[3])?,
                Some(self.idx_legacy_reduction(node)?),
            ),
            3 => (
                self.idx_scalar_value(node, 3, data.dtype)?
                    .expand_rhs(indices.dims()),
                Some(self.idx_legacy_reduction(node)?),
            ),
            4 => (self.operand(&node.inputs[3])?, Some(IdxReduce::Sum)),
            5 => (
                self.operand(&node.inputs[3])?,
                Some(self.idx_modern_reduction(node)?),
            ),
            other => bail!("scatter variant {other} is not ported"),
        };
        let include_self = if kind == 5 {
            self.idx_include_self(node)
        } else {
            true
        };
        // A rank-0 scatter is expressible as a rank-1 one; squeeze back.
        let (data, indices, updates, squeeze) = if rank == 0 {
            (
                data.unsqueeze(0),
                indices.unsqueeze(0),
                updates.unsqueeze(0),
                true,
            )
        } else {
            (data, indices, updates, false)
        };
        let index = self.idx_normalize(indices, data.dims()[dim]);
        let update_dims = index.dims();
        let updates = self.idx_crop_to(updates, &update_dims);
        let updates = self.idx_cast_like(updates, data.dtype);
        let coords = self.idx_scatter_coords(&update_dims, dim, index);
        let result = match reduction {
            None => data.scatter(&coords, updates),
            Some(reduction) => {
                self.idx_scatter_reduce(data, coords, updates, reduction, include_self)?
            }
        };
        Ok(if squeeze { result.squeeze(0) } else { result })
    }

    /// `index_put_`/`index_put`: `data[indices...] = values` (functional
    /// SSA; the boundary handles the in-place writeback).
    pub(super) fn translate_index_put(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.operand(&node.inputs[0])?;
        let values = self.operand(&node.inputs[2])?;
        let accumulate = node
            .inputs
            .get(3)
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false);
        let entries = Self::idx_entries(node)?;
        let present: Vec<&String> = entries.iter().flatten().collect();
        if present.is_empty() {
            bail!("index_put: no index tensors");
        }
        // A lone boolean mask writes through `where` (or a masked scatter
        // when the values are a flat list of the true slots); treating the
        // mask as integer indices would silently corrupt the data.
        if present.len() == 1 {
            let mask = self
                .values
                .get(present[0])
                .copied()
                .with_context(|| format!("index_put: unknown index tensor {}", present[0]))?;
            if mask.dtype == DType::Bool {
                if accumulate {
                    bail!("index_put with a Bool mask and accumulate=true is not ported");
                }
                let values = self.idx_cast_like(values, data.dtype);
                if idx_can_broadcast_to(&values.dims(), &data.dims()) {
                    return Ok(self.idx_cond(mask, values, data));
                }
                return self.idx_masked_scatter(data, mask, values);
            }
        }
        let mut index_for_dim: Vec<Option<GraphTensor>> = vec![None; data.rank()];
        let mut indexed = Vec::new();
        for (dim, name) in entries.iter().enumerate() {
            let Some(name) = name else { continue };
            if dim >= data.rank() {
                bail!(
                    "index_put: index dim {dim} out of range for rank {}",
                    data.rank()
                );
            }
            let t = self
                .values
                .get(name)
                .copied()
                .with_context(|| format!("index_put: unknown index tensor {name}"))?;
            if t.dtype == DType::Bool {
                bail!("index_put: a boolean mask cannot be mixed with tensor indices");
            }
            index_for_dim[dim] = Some(self.idx_normalize(t.cast(DType::Int), data.dims()[dim]));
            indexed.push(dim);
        }
        let bshape = self.idx_broadcast_indices(&mut index_for_dim);
        let (out_dims, block_start, basic) = idx_advanced_layout(&data.dims(), &indexed, &bshape);
        let coords = self.idx_advanced_coords(&out_dims, block_start, &basic, &index_for_dim);
        let updates = self.idx_cast_like(values, data.dtype);
        let updates = self.idx_broadcast_to(updates, &out_dims);
        Ok(if accumulate {
            self.idx_scatter_reduce(data, coords, updates, IdxReduce::Sum, true)?
        } else {
            data.scatter(&coords, updates)
        })
    }

    /// `index_reduce.default`: reduce `source` into `data` along `dim`
    /// using the 1-D `index` (no duplicates ordering issues beyond
    /// scatter-reduce's sequential semantics).
    pub(super) fn translate_index_reduce(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.operand(&node.inputs[0])?;
        let raw_dim = self.get_int_arg(node, 1)?;
        let index = self.operand(&node.inputs[2])?.cast(DType::Int);
        let source = self.operand(&node.inputs[3])?;
        let reduction = self.idx_modern_reduction(node)?;
        if matches!(reduction, IdxReduce::Sum) {
            bail!("index_reduce does not support the sum reduction");
        }
        let include_self = self.idx_include_self(node);
        if data.rank() == 0 {
            if !matches!(raw_dim, -1 | 0) {
                bail!("index_reduce dim {raw_dim} out of range for a scalar");
            }
            if index.rank() != 1 || index.dims()[0].to_usize() != Some(1) {
                bail!("scalar index_reduce requires a one-element index");
            }
            if source.rank() != 0 {
                bail!("scalar index_reduce requires a scalar source");
            }
            let data = data.unsqueeze(0);
            let source = source.unsqueeze(0);
            let index = self.idx_normalize(index, data.dims()[0]);
            let updates = self.idx_cast_like(source, data.dtype);
            let result =
                self.idx_scatter_reduce(data, vec![index], updates, reduction, include_self)?;
            return Ok(result.squeeze(0));
        }
        let dim = normalize_dim(raw_dim, data.rank());
        if index.rank() != 1 {
            bail!("index_reduce index must be one-dimensional");
        }
        if source.rank() != data.rank() {
            bail!("index_reduce source rank must match self");
        }
        if !util::same_dim(source.dims()[dim], index.dims()[0]) {
            bail!("index_reduce index length must equal source size along the reduced dimension");
        }
        for axis in 0..data.rank() {
            if axis != dim && !util::same_dim(source.dims()[axis], data.dims()[axis]) {
                bail!("index_reduce source/self sizes must match outside the reduced dimension");
            }
        }
        let inserted: Vec<usize> = (0..data.rank()).filter(|axis| *axis != dim).collect();
        let index = self.idx_normalize(index, data.dims()[dim]);
        let expanded = index.expand_to_shape_on_axes(source.dims(), inserted);
        let mut coords = Vec::with_capacity(data.rank());
        for axis in 0..data.rank() {
            if axis == dim {
                coords.push(expanded);
            } else {
                coords.push(self.axis_positions(&source.dims(), axis));
            }
        }
        let updates = self.idx_cast_like(source, data.dtype);
        self.idx_scatter_reduce(data, coords, updates, reduction, include_self)
    }

    /// `masked_scatter.default`: copy `source` into the true slots of
    /// `destination` in row-major order.
    pub(super) fn translate_masked_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let destination = self.operand(&node.inputs[0])?;
        let mask = self.operand(&node.inputs[1])?;
        let source = self.operand(&node.inputs[2])?;
        self.idx_masked_scatter(destination, mask, source)
    }

    /// `put.default`: write `source` (flattened) into `destination` at the
    /// flattened `indices`, optionally accumulating duplicates.
    pub(super) fn translate_put(&mut self, node: &Node) -> Result<GraphTensor> {
        let data = self.operand(&node.inputs[0])?;
        let indices = self.operand(&node.inputs[1])?;
        let source = self.operand(&node.inputs[2])?;
        let accumulate = node
            .inputs
            .get(3)
            .and_then(|input| input.arg.as_bool())
            .unwrap_or(false);
        if data.dtype == DType::Bool && accumulate {
            let int_data = data.cast(DType::Int);
            let int_source = self.idx_cast_like(source, DType::Int);
            let output = self.idx_put(int_data, indices, int_source, true)?;
            return Ok(output.cast(DType::Bool));
        }
        self.idx_put(data, indices, source, accumulate)
    }

    fn idx_put(
        &mut self,
        data: GraphTensor,
        indices: GraphTensor,
        source: GraphTensor,
        accumulate: bool,
    ) -> Result<GraphTensor> {
        let output_shape = data.dims();
        let flat_size = data
            .dims()
            .iter()
            .fold(IntExpr::from(1), |acc, dim| acc * *dim);
        let index = indices.cast(DType::Int).flatten();
        let zero = self.cx.constant_i32(0).expand_rhs(index.dims());
        let negative = index.lt(zero);
        let size_f = self
            .cx
            .constant_i32(flat_size)
            .cast(DType::F32)
            .expand_rhs(index.dims());
        let shifted = (index.cast(DType::F32) + size_f).trunc_cast(DType::Int);
        let index = self.idx_cond(negative, shifted, index);
        let destination = data.flatten();
        let source = self.idx_cast_like(source, data.dtype).flatten();
        let source = self.idx_crop_to(source, &index.dims());
        let output = if accumulate {
            self.idx_scatter_reduce(destination, vec![index], source, IdxReduce::Sum, true)?
        } else {
            destination.scatter(&[index], source)
        };
        Ok(reshape_tensor(output, &output_shape))
    }

    /// `nonzero_static.default`: the coordinates of the nonzero elements
    /// (row-major), padded to `size` rows with `fill_value`.
    pub(super) fn translate_nonzero_static(&mut self, node: &Node) -> Result<GraphTensor> {
        let value = self.operand(&node.inputs[0])?;
        let size = node
            .inputs
            .iter()
            .position(|input| input.name == "size")
            .or_else(|| (node.inputs.len() > 1).then_some(1))
            .map(|index| self.get_int_arg(node, index))
            .transpose()?
            .context("nonzero_static is missing its size argument")?;
        if size < 0 {
            bail!("nonzero_static size {size} is negative");
        }
        let size = size as usize;
        let fill_value = self
            .named_int_arg(node, "fill_value")
            .or_else(|| node.inputs.get(2).and_then(|input| input.arg.as_int()))
            .unwrap_or(0);
        let input_shape = value.dims();
        let rank = value.rank();
        let truth = if idx_is_float(value.dtype) {
            let zero = self.is_zero(value);
            self.bool_not(zero)
        } else {
            value.ne(self.constant_like(value, 0.0))
        };
        if rank == 0 {
            return Ok(self
                .cx
                .iota(vec![IntExpr::from(size), IntExpr::from(0usize)], |_| {
                    IntExpr::from(0)
                })
                .cast(DType::I64));
        }
        if idx_numel(&input_shape) == Some(0) {
            return Ok(self
                .cx
                .constant_i64(fill_value)
                .expand_rhs(vec![IntExpr::from(size), IntExpr::from(rank)]));
        }
        let flat_truth = truth.flatten();
        // 1s first, stable: the true positions in row-major order.
        let sorted = flat_truth.cast(DType::F32).stable_argsort(0, true);
        let count = flat_truth.cast(DType::F32).sum(0).trunc_cast(DType::Int);
        let numel = input_shape.iter().fold(IntExpr::from(1), |acc, d| acc * *d);
        let positions = self.cx.arange(size);
        let last = self
            .cx
            .constant_i32(numel - IntExpr::from(1))
            .cast(DType::F32)
            .expand_rhs(positions.dims());
        let clamped = positions
            .cast(DType::F32)
            .minimum(last)
            .trunc_cast(DType::Int);
        let flat_indices = sorted.gather(&[clamped]);
        let count = count.expand_rhs(positions.dims());
        let numel_const = self.cx.constant_i32(numel).expand_rhs(positions.dims());
        let valid = self.bool_and(positions.lt(count), positions.lt(numel_const));
        // Per-axis coordinates as flat div/mod in F32 (the plan has no F64
        // binary arms), blended with the fill value and truncated once at
        // the end.
        let strides: Vec<IntExpr> = (0..rank)
            .map(|i| {
                input_shape[i + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, d| acc * *d)
            })
            .collect();
        let flat_f = flat_indices.cast(DType::F32);
        let valid_f = valid.cast(DType::F32);
        let one = self.cx.constant_f32(1.0).expand_rhs(valid_f.dims());
        let invalid_f = one - valid_f;
        let fill_f = self
            .cx
            .constant_f32(fill_value as f32)
            .expand_rhs(valid_f.dims());
        let mut columns = Vec::with_capacity(rank);
        for axis in 0..rank {
            let stride_f = self
                .cx
                .constant_i32(strides[axis])
                .cast(DType::F32)
                .expand_rhs(flat_f.dims());
            let quotient = flat_f / stride_f;
            let dim_f = self
                .cx
                .constant_i32(input_shape[axis])
                .cast(DType::F32)
                .expand_rhs(quotient.dims());
            let coordinate = quotient - (quotient / dim_f).floor() * dim_f;
            columns.push((coordinate * valid_f + fill_f * invalid_f).trunc_cast(DType::I64));
        }
        let rows = self.cx.arange(size);
        let mut result = self
            .cx
            .constant_i64(0)
            .expand_rhs(vec![IntExpr::from(size), IntExpr::from(rank)]);
        for (axis, column) in columns.into_iter().enumerate() {
            let axis_column = self.cx.constant_i32(axis as i64).expand_rhs(rows.dims());
            result = result.scatter(&[rows, axis_column], column);
        }
        Ok(result)
    }

    /// `embedding_bag.default`: sum/mean/max over the bags of `indices`
    /// delimited by `offsets`; binds (output, offset2bag, bag_size,
    /// max_indices).
    pub(super) fn translate_embedding_bag(&mut self, node: &Node) -> Result<()> {
        let weight = self.operand(&node.inputs[0])?;
        let indices = self.operand(&node.inputs[1])?.cast(DType::Int);
        let offsets = self.operand(&node.inputs[2])?.cast(DType::Int);
        if weight.rank() != 2 || indices.rank() != 1 || offsets.rank() != 1 {
            bail!("embedding_bag requires matrix weights and one-dimensional indices/offsets");
        }
        let mode = self.named_int_arg(node, "mode").unwrap_or(0);
        if !(0..=2).contains(&mode) {
            bail!("unsupported embedding_bag mode {mode}");
        }
        let include_last_offset = self
            .named_bool_arg(node, "include_last_offset")
            .unwrap_or(false);
        let padding_idx = self.named_int_arg(node, "padding_idx").unwrap_or(-1);
        let index_count = indices.dims()[0];
        let embedding_size = weight.dims()[1];
        let offset_count = offsets.dims()[0];
        let bag_count = offset_count - IntExpr::from(include_last_offset as usize);
        // ends[b] = offsets[b + 1], with the terminal bag end pinned to
        // index_count; built by scatter so no Int pad arithmetic is needed.
        let starts_full = offsets.slice_along(IntExpr::from(0)..bag_count, 0);
        let shifted = offsets.slice_along(1.., 0);
        let ends_full = self
            .cx
            .constant_i32(index_count)
            .expand_rhs(vec![offset_count]);
        let ends_full = shifted.scatter(
            &[self.cx.arange(offset_count - IntExpr::from(1))],
            ends_full,
        );
        let positions = self.cx.arange(index_count).expand_dim(0, bag_count);
        let starts = starts_full.expand_dim(1, index_count);
        let ends = ends_full
            .slice_along(IntExpr::from(0)..bag_count, 0)
            .expand_dim(1, index_count);
        let bag_membership = self.bool_and(positions.ge(starts), positions.lt(ends));
        let expanded_indices = indices.expand_dim(0, bag_count);
        let mut membership = bag_membership;
        if padding_idx >= 0 {
            let padding = self
                .cx
                .constant_i32(padding_idx)
                .expand_rhs(expanded_indices.dims());
            membership = self.bool_and(membership, expanded_indices.ne(padding));
        }
        // Coordinate gather: rows are the bag index ids, cols carry the
        // embedding coordinate.
        let rows = expanded_indices.expand_dim(2, embedding_size);
        let cols = self
            .cx
            .iota(vec![bag_count, index_count, embedding_size], |c| c[2]);
        let gathered = weight.gather(&[rows, cols]);
        let membership_values = membership.expand_dim(2, embedding_size);
        let zero_values = self.full_tensor(gathered.dims(), weight.dtype, 0.0);
        let mut selected = self.idx_cond(membership_values, gathered, zero_values);
        if let Some(index) = node
            .inputs
            .iter()
            .position(|input| input.name == "per_sample_weights")
            && let Some(per_sample) = self.optional_tensor_operand(&node.inputs[index])?
        {
            if mode != 0 {
                bail!("per-sample weights require embedding_bag sum mode");
            }
            let scale = per_sample
                .cast(weight.dtype)
                .expand_dim(0, bag_count)
                .expand_dim(2, embedding_size);
            selected *= scale;
        }
        let counts = membership.cast(DType::F32).sum(1).trunc_cast(DType::I64);
        let nonzero = self.cx.constant_i64(0).expand_rhs(counts.dims());
        let nonempty = counts.gt(nonzero);
        let sum = selected.sum(1);
        let (output, max_indices) = match mode {
            0 => (sum, self.cx.constant_i64(0).expand_rhs(vec![bag_count])),
            1 => {
                let one = self.full_tensor(vec![bag_count], DType::I64, 1.0);
                let safe_counts = self.idx_cond(nonempty, counts, one);
                let divisor = safe_counts.cast(weight.dtype).expand_dim(1, embedding_size);
                (sum / divisor, counts)
            }
            _ => {
                let lowest = self.full_tensor(gathered.dims(), weight.dtype, f64::NEG_INFINITY);
                let candidates = self.idx_cond(membership_values, gathered, lowest);
                let selected_positions = candidates.stable_argsort(1, true).slice_along(0..1, 1);
                let values = self
                    .idx_gather_axis(candidates, selected_positions, 1)?
                    .squeeze(1);
                let candidate_indices = expanded_indices.expand_dim(2, embedding_size);
                let selected_indices = self
                    .idx_gather_axis(candidate_indices, selected_positions, 1)?
                    .squeeze(1)
                    .cast(DType::I64);
                let output_nonempty = nonempty.expand_dim(1, embedding_size);
                let zero_output = self.full_tensor(values.dims(), weight.dtype, 0.0);
                let zero_indices = self.cx.constant_i64(0).expand_rhs(selected_indices.dims());
                (
                    self.idx_cond(output_nonempty, values, zero_output),
                    self.idx_cond(output_nonempty, selected_indices, zero_indices),
                )
            }
        };
        let offset_to_bag = if mode == 0 && padding_idx < 0 {
            self.cx
                .constant_i64(0)
                .expand_rhs(vec![IntExpr::from(0usize)])
        } else {
            let bag_ids = self
                .cx
                .arange(bag_count)
                .cast(DType::F32)
                .expand_dim(1, index_count);
            (bag_membership.cast(DType::F32) * bag_ids)
                .sum(0)
                .trunc_cast(DType::I64)
        };
        let bag_size = if mode == 0 {
            self.cx.constant_i64(0).expand_rhs(vec![bag_count])
        } else {
            counts
        };
        self.bind_outputs(node, vec![output, offset_to_bag, bag_size, max_indices])
    }
}
