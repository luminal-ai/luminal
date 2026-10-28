//! Indexing, gather/scatter, and embedding-bag lowerings (port batch 2).
//!
//! Gather/scatter are built from the recorder's coordinate forms so no
//! caller-data flat-index arithmetic (proof-gated Int add/mul) enters the
//! graph. INVARIANT: an index, coordinate, offset, or count is an exact
//! integer and is never carried in a float (F32 holds integers exactly
//! only to 2^24, so a longer axis would saturate or round silently).
//! Counts are Bool -> Int/I64 indicator sums or Int scans; structural
//! positions are `IntExpr` iotas; caller indices are normalized with
//! bounded Int remainders; per-axis destinations use the coordinate form
//! instead of a flat index. Anything that cannot be done faithfully bails
//! with its op name.
#![allow(dead_code)]

use anyhow::{Context, Result, bail};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, normalize_dim, reshape_tensor, storage_step};
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

/// One destination coordinate per data axis of an index write. A lowering
/// builds one per data axis and consumes them at translation time, so the
/// variant size difference is immaterial.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
enum IdxCoord {
    /// Caller index data over the update shape: exact Int, already wrapped
    /// into the axis.
    Data(GraphTensor),
    /// The update element's own position along this update axis
    /// (structural: an iota).
    Position(usize),
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

    /// Index updates select storage values without arithmetic or dtype loss.
    fn idx_cond(&mut self, cond: GraphTensor, a: GraphTensor, b: GraphTensor) -> GraphTensor {
        self.select(cond, a, b)
    }

    /// Wrap negative index values into `[0, dim)`, PyTorch's rule for an
    /// index in `[-dim, dim)`, in exact Int: `((idx rem dim) + dim) rem dim`
    /// with truncated remainders. Int arithmetic is proof-gated and the
    /// index is caller data with no attested range, so the spelling is
    /// chosen to discharge its own obligations: a truncated remainder is
    /// bounded by its divisor alone (`trunc_rem/value_bounds`), the add of
    /// two bounded tensors stays in width, and the second remainder of a
    /// non-negative dividend lands in `[0, dim)`, which is also the tight
    /// bound every consumer's index arithmetic wants. The former float32
    /// detour (cast, add, blend, truncating cast back) was exact only below
    /// 2^24. An out-of-range index, which PyTorch rejects, wraps silently
    /// here as it did before.
    fn idx_normalize(&mut self, indices: GraphTensor, dim: IntExpr) -> GraphTensor {
        let dim = self.cx.constant_i32(dim).expand_rhs(indices.dims());
        (indices.trunc_rem(dim) + dim).trunc_rem(dim)
    }

    /// Cast to a storage dtype, using the explicit truncating read for
    /// float -> int.
    fn idx_cast_like(&mut self, t: GraphTensor, dtype: DType) -> GraphTensor {
        super::convert(t, dtype)
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

    /// Coordinate per source axis for an advanced-index layout: indexed
    /// axes read the broadcast index data, basic axes their output position.
    fn idx_advanced_plan(
        &mut self,
        out_dims: &[IntExpr],
        block_start: usize,
        basic: &[(usize, usize)],
        index_for_dim: &[Option<GraphTensor>],
    ) -> Vec<IdxCoord> {
        let block_len = out_dims.len() - basic.len();
        let non_block: Vec<usize> = (0..out_dims.len())
            .filter(|p| !(block_start..block_start + block_len).contains(p))
            .collect();
        let mut coords: Vec<Option<IdxCoord>> = vec![None; index_for_dim.len()];
        for (axis, t) in index_for_dim.iter().enumerate() {
            if let Some(t) = t {
                coords[axis] = Some(IdxCoord::Data(
                    t.expand_to_shape_on_axes(out_dims.to_vec(), non_block.clone()),
                ));
            }
        }
        for (axis, out_axis) in basic {
            coords[*axis] = Some(IdxCoord::Position(*out_axis));
        }
        coords
            .into_iter()
            .map(|c| c.expect("advanced index: every source axis has a coordinate"))
            .collect()
    }

    /// Coordinate tensor per source axis for an advanced-index layout.
    fn idx_advanced_coords(
        &mut self,
        out_dims: &[IntExpr],
        block_start: usize,
        basic: &[(usize, usize)],
        index_for_dim: &[Option<GraphTensor>],
    ) -> Vec<GraphTensor> {
        let plan = self.idx_advanced_plan(out_dims, block_start, basic, index_for_dim);
        self.idx_coordinate_tensors(&plan, out_dims)
    }

    /// The coordinate tensors of a plan over the update shape.
    fn idx_coordinate_tensors(
        &mut self,
        plan: &[IdxCoord],
        update_dims: &[IntExpr],
    ) -> Vec<GraphTensor> {
        plan.iter()
            .map(|coord| match *coord {
                IdxCoord::Data(index) => index,
                IdxCoord::Position(axis) => self.axis_positions(update_dims, axis),
            })
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
        coords: Vec<IdxCoord>,
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
        let data_dims = data.dims();
        if count == 0 || idx_numel(&data_dims) == Some(0) {
            return Ok(data);
        }
        let update_dims = updates.dims();
        // INVARIANT: every destination is addressed in the coordinate form,
        // so no flat index is ever formed: an F32 flat index rounds past
        // 2^24 elements and silently retargets the update, and an Int flat
        // sum of caller indices is unproven. The structural axes (the update
        // element's own position) are permuted to the front of a view of the
        // data and merged into one plane axis, addressed by one exact
        // `IntExpr` iota over the update positions; each caller-data axis
        // keeps its own coordinate. Fewer coordinates keep the unrolled
        // graph small: one coordinate per data axis measured up to 1.7x the
        // old flat lowering's search time at rank 4, the plane view about
        // 1.2-1.45x. A symbolic structural extent keeps one coordinate per
        // axis (the plane strides must be concrete to split the view back).
        // Step `k` reads and writes the data at the k-th element of every
        // flattened coordinate tensor.
        let structural: Vec<(usize, usize)> = coords
            .iter()
            .enumerate()
            .filter_map(|(axis, coord)| match coord {
                IdxCoord::Position(update_axis) => Some((axis, *update_axis)),
                IdxCoord::Data(_) => None,
            })
            .collect();
        let plane = !structural.is_empty()
            && structural
                .iter()
                .all(|(axis, _)| data_dims[*axis].to_usize().is_some());
        // `order[k]` is the data axis at position `k` of the working view.
        let order: Vec<usize> = if plane {
            structural
                .iter()
                .map(|(axis, _)| *axis)
                .chain((0..coords.len()).filter(|axis| matches!(coords[*axis], IdxCoord::Data(_))))
                .collect()
        } else {
            (0..coords.len()).collect()
        };
        let mut flat_coords = Vec::with_capacity(coords.len());
        if plane {
            let terms: Vec<(usize, IntExpr)> = structural
                .iter()
                .enumerate()
                .map(|(k, (_, update_axis))| {
                    let stride = structural[k + 1..]
                        .iter()
                        .fold(IntExpr::from(1), |acc, (axis, _)| acc * data_dims[*axis]);
                    (*update_axis, stride)
                })
                .collect();
            let plane_coordinate = self.cx.iota(update_dims.clone(), move |c| {
                terms.iter().fold(IntExpr::from(0), |acc, (axis, stride)| {
                    acc + c[*axis] * *stride
                })
            });
            flat_coords.push(plane_coordinate.flatten());
            for coord in &coords {
                if let IdxCoord::Data(index) = coord {
                    flat_coords.push(index.flatten());
                }
            }
        } else {
            for coord in self.idx_coordinate_tensors(&coords, &update_dims) {
                flat_coords.push(coord.flatten());
            }
        }
        let planes = structural.len();
        let to_working = |tensor: GraphTensor| {
            if !plane {
                return tensor;
            }
            let mut view = tensor.view().permute(order.clone());
            for _ in 1..planes {
                view = view.merge_dims(0, 1);
            }
            view.finish()
        };
        let int_data = idx_is_int(data.dtype);
        let work_dtype = if int_data { DType::F32 } else { data.dtype };
        let original_work = to_working(if int_data {
            data.cast(DType::F32)
        } else {
            data
        });
        let mut output = original_work;
        let flat_updates = if int_data {
            updates.flatten().cast(DType::F32)
        } else {
            updates.flatten()
        };
        let track_counts = !include_self || matches!(reduction, IdxReduce::Mean);
        let mut counts = self.full_tensor(
            original_work.dims(),
            DType::F32,
            if include_self { 1.0 } else { 0.0 },
        );
        for step in 0..count {
            let dest: Vec<GraphTensor> = flat_coords
                .iter()
                .map(|coord| coord.slice_along(step..step + 1, 0))
                .collect();
            let update = flat_updates.slice_along(step..step + 1, 0);
            let current = output.gather(&dest);
            // Each step rounds to the data dtype, as torch's sequential
            // kernels do; narrow storage computes the step in F32.
            let mut combined = match reduction {
                IdxReduce::Sum | IdxReduce::Mean => {
                    if bool_data {
                        self.bool_or(current, update)
                    } else {
                        storage_step(current, update, |a, b| a + b)
                    }
                }
                IdxReduce::Prod => {
                    if bool_data {
                        self.bool_and(current, update)
                    } else {
                        storage_step(current, update, |a, b| a * b)
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
                // The per-destination update count is bounded by the
                // number of unrolled steps, far below 2^24, so its F32
                // carrier (also the mean's divisor dtype) is exact.
                let count_here = counts.gather(&dest);
                if !include_self {
                    let zero = self.cx.constant_f32(0.0).expand_rhs(count_here.dims());
                    let first_update = count_here.eq(zero);
                    combined = self.idx_cond(first_update, update, combined);
                }
                let one = self.cx.constant_f32(1.0).expand_rhs(count_here.dims());
                counts = counts.scatter(&dest, count_here + one);
            }
            output = output.scatter(&dest, combined);
        }
        if matches!(reduction, IdxReduce::Mean) {
            let zero = self.cx.constant_f32(0.0).expand_rhs(counts.dims());
            let has_values = counts.gt(zero);
            let divisor = counts.cast(work_dtype);
            let means = storage_step(output, divisor, |a, b| a / b);
            output = self.idx_cond(has_values, means, original_work);
        }
        // Split the plane axis back into the structural axes and restore
        // the data axis order.
        let output = if plane {
            let mut view = output.view();
            for k in 0..planes - 1 {
                let inner = structural[k + 1..]
                    .iter()
                    .fold(IntExpr::from(1), |acc, (axis, _)| acc * data_dims[*axis]);
                view = view.split_dims(k, inner);
            }
            let mut inverse = vec![0; order.len()];
            for (position, axis) in order.iter().enumerate() {
                inverse[*axis] = position;
            }
            view.permute(inverse).finish()
        } else {
            output
        };
        Ok(if int_data {
            super::convert(output, data.dtype)
        } else {
            output
        })
    }

    /// Scatter coordinates for an update at `[.., index[c], ..]` (ONNX
    /// ScatterElements): one coordinate tensor per data axis over the
    /// update shape.
    fn idx_scatter_plan(rank: usize, axis: usize, index: GraphTensor) -> Vec<IdxCoord> {
        (0..rank)
            .map(|d| {
                if d == axis {
                    IdxCoord::Data(index)
                } else {
                    IdxCoord::Position(d)
                }
            })
            .collect()
    }

    /// A scalar `value` argument of the requested dtype.
    fn idx_scalar_value(&mut self, node: &Node, index: usize, dtype: DType) -> Result<GraphTensor> {
        let input = node
            .inputs
            .get(index)
            .with_context(|| format!("{} is missing scalar input {index}", node.target))?;
        self.scalar(input, dtype)
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
        let slots = mask.dims()[0];
        if source_flat.dims()[0].to_usize() == Some(0) || slots.to_usize() == Some(0) {
            return Ok(reshape_tensor(destination_flat, &output_shape));
        }
        // INVARIANT: the source element for a true slot is at that slot's
        // rank among the true slots, an exact integer count. It is the
        // EXCLUSIVE prefix count of the mask: an Int scan of the 0/1
        // indicators (the scan kernels check integer overflow) shifted one
        // slot right with a zero fill, so no float carries it (F32
        // rounds past 2^24 slots) and no unproven Int subtraction forms
        // it. False slots read source element 0, which exists because the
        // source is non-empty, and are discarded by the select.
        let inclusive = mask.cast(DType::Int).cumsum(vec![0]);
        let exclusive = inclusive
            .slice_along(IntExpr::from(0)..slots - IntExpr::from(1), 0)
            .pad_along(1, 0, 0, 0.0);
        let first = self.cx.constant_i32(0).expand_rhs(exclusive.dims());
        let positions = mask.select(exclusive, first);
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
        let plan = Self::idx_scatter_plan(update_dims.len(), dim, index);
        let result = match reduction {
            None => {
                let coords = self.idx_coordinate_tensors(&plan, &update_dims);
                data.scatter(&coords, updates)
            }
            Some(reduction) => {
                self.idx_scatter_reduce(data, plan, updates, reduction, include_self)?
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
                let axis = entries.iter().position(Option::is_some).unwrap();
                let mask_rank = mask.rank();
                let dims = data.dims();
                anyhow::ensure!(
                    entries.len() - 1 + mask.rank() <= data.rank()
                        && mask.dims().iter().enumerate().all(|(i, dim)| {
                            dims.get(axis + i)
                                .is_some_and(|other| util::same_dim(*dim, *other))
                        }),
                    "index_put: boolean mask must match the indexed input axes"
                );
                // x[basic..., mask, basic...] has indexed shape
                // [before..., count_true(mask), after...]. Values broadcast
                // against that shape, not the original input shape.
                let mut selected = dims[..axis].to_vec();
                selected.push(IntExpr::from(1));
                selected.extend_from_slice(&dims[axis + mask.rank()..]);
                let value_dims = values.dims();
                anyhow::ensure!(
                    value_dims.len() <= selected.len(),
                    "index_put: values rank exceeds the boolean-indexed result rank"
                );
                let mut aligned = vec![IntExpr::from(1); selected.len() - value_dims.len()];
                aligned.extend_from_slice(&value_dims);
                selected[axis] = aligned[axis];
                anyhow::ensure!(
                    idx_can_broadcast_to(&aligned, &selected),
                    "index_put: values do not broadcast to the boolean-indexed result"
                );
                if idx_numel(&value_dims) == Some(0) {
                    return Ok(data);
                }
                anyhow::ensure!(
                    mask_rank != 0 || aligned[axis].to_usize() == Some(1),
                    "index_put: a scalar boolean index has at most one selected slot"
                );
                let mask = mask
                    .expand_lhs(vec![IntExpr::from(1); axis])
                    .expand_rhs(vec![IntExpr::from(1); data.rank() - axis - mask_rank])
                    .expand(dims.clone());
                let values = values
                    .expand_lhs(vec![IntExpr::from(1); aligned.len() - value_dims.len()])
                    .expand(selected);
                if aligned[axis].to_usize() == Some(1) {
                    // One update per unindexed slice: broadcast it over all
                    // mask axes and select, preserving integer and IEEE bits.
                    let mut shape = dims[..axis].to_vec();
                    shape.extend(vec![IntExpr::from(1); mask_rank]);
                    shape.extend_from_slice(&dims[axis + mask_rank..]);
                    return Ok(self.idx_cond(mask, reshape_tensor(values, &shape), data));
                }
                // Expand broadcasts in the selected result before assigning
                // flattened true slots in their stable row-major order.
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
        let plan = self.idx_advanced_plan(&out_dims, block_start, &basic, &index_for_dim);
        let updates = self.idx_cast_like(values, data.dtype);
        let updates = self.idx_broadcast_to(updates, &out_dims);
        Ok(if accumulate {
            self.idx_scatter_reduce(data, plan, updates, IdxReduce::Sum, true)?
        } else {
            let coords = self.idx_coordinate_tensors(&plan, &out_dims);
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
            let result = self.idx_scatter_reduce(
                data,
                vec![IdxCoord::Data(index)],
                updates,
                reduction,
                include_self,
            )?;
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
        let plan = Self::idx_scatter_plan(data.rank(), dim, expanded);
        let updates = self.idx_cast_like(source, data.dtype);
        self.idx_scatter_reduce(data, plan, updates, reduction, include_self)
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
        // The flat index is caller data: wrap it into [0, numel) in exact
        // Int (an F32 detour rounds past 2^24 and silently retargets the
        // write).
        let index = indices.cast(DType::Int).flatten();
        let index = self.idx_normalize(index, flat_size);
        let destination = data.flatten();
        let source = self.idx_cast_like(source, data.dtype).flatten();
        let source = self.idx_crop_to(source, &index.dims());
        let output = if accumulate {
            self.idx_scatter_reduce(
                destination,
                vec![IdxCoord::Data(index)],
                source,
                IdxReduce::Sum,
                true,
            )?
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
        Ok(self.true_coordinates(truth, size, fill_value))
    }

    /// Row-major coordinates of the first `size` true entries of `truth`,
    /// as an I64 `[size, rank]` tensor; rows past the true count (or past
    /// the element count) hold `fill`. Shared by `nonzero_static` and
    /// `tril_indices`/`triu_indices`.
    ///
    /// INVARIANT: every index here is an exact integer that never passes
    /// through a float. The stable descending argsort of the 0/1 truth
    /// key (an exact F32 key; its ranks and output indices are Int)
    /// lists the true flat positions first, in row-major order. Slot `k`
    /// reads position `min(k, numel - 1)` (an `IntExpr` iota), and is
    /// valid exactly when it is a real slot (`k < numel`) whose element
    /// is true: the sort puts every true element ahead of every false
    /// one, so no count is needed. Each coordinate column gathers that
    /// axis's coordinate iota over the flattened shape, so no flat
    /// div/mod is formed (F32 div/mod rounds past 2^24 elements).
    pub(super) fn true_coordinates(
        &mut self,
        truth: GraphTensor,
        size: usize,
        fill: i64,
    ) -> GraphTensor {
        let shape = truth.dims();
        let rank = truth.rank();
        let fill = self.cx.constant_i64(fill);
        if idx_numel(&shape) == Some(0) {
            return fill.expand_rhs(vec![IntExpr::from(size), IntExpr::from(rank)]);
        }
        let flat_truth = truth.flatten();
        let sorted = flat_truth.cast(DType::F32).stable_argsort(0, true);
        let numel = shape.iter().fold(IntExpr::from(1), |acc, d| acc * *d);
        let slots = self.cx.arange(size);
        let read = self.cx.iota(vec![IntExpr::from(size)], move |c| {
            c[0].min(numel - IntExpr::from(1))
        });
        let flat_indices = sorted.gather(&[read]);
        let numel_const = self.cx.constant_i32(numel).expand_rhs(slots.dims());
        let read_true = flat_truth.gather(&[flat_indices]);
        let valid = self.bool_and(slots.lt(numel_const), read_true);
        let fill = fill.expand_rhs(slots.dims());
        let rows = self.cx.arange(size);
        let mut result = self
            .cx
            .constant_i64(0)
            .expand_rhs(vec![IntExpr::from(size), IntExpr::from(rank)]);
        for axis in 0..rank {
            let coordinate = self
                .cx
                .iota(shape.clone(), move |c| c[axis])
                .flatten()
                .gather(&[flat_indices])
                .cast(DType::I64);
            let column = valid.select(coordinate, fill);
            let axis_column = self.cx.constant_i32(axis as i64).expand_rhs(rows.dims());
            result = result.scatter(&[rows, axis_column], column);
        }
        result
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
        // The pinned row is the scatter's destination and the shifted
        // offsets are its source (one coordinate per source element).
        let starts_full = offsets.slice_along(IntExpr::from(0)..bag_count, 0);
        let shifted = offsets.slice_along(1.., 0);
        let ends_full = self
            .cx
            .constant_i32(index_count)
            .expand_rhs(vec![offset_count]);
        let ends_full =
            ends_full.scatter(&[self.cx.arange(offset_count - IntExpr::from(1))], shifted);
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
        // Bags accumulate at torch's opmath dtype and round once at the store.
        let accumulate = super::opmath_compute(weight.dtype);
        let gathered = weight.gather(&[rows, cols]).cast(accumulate);
        let membership_values = membership.expand_dim(2, embedding_size);
        let zero_values = self.full_tensor(gathered.dims(), accumulate, 0.0);
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
                .cast(accumulate)
                .expand_dim(0, bag_count)
                .expand_dim(2, embedding_size);
            selected *= scale;
        }
        // INVARIANT: a bag size is an exact count of 0/1 membership
        // indicators, summed in I64 (proved in width from the indicator
        // bounds scaled by the index count); an F32 sum saturates at 2^24.
        let counts = membership.cast(DType::I64).sum(vec![1]);
        let nonzero = self.cx.constant_i64(0).expand_rhs(counts.dims());
        let nonempty = counts.gt(nonzero);
        let sum = selected.sum(vec![1]);
        let (output, max_indices) = match mode {
            0 => (sum, self.cx.constant_i64(0).expand_rhs(vec![bag_count])),
            1 => {
                let one = self.full_tensor(vec![bag_count], DType::I64, 1.0);
                let safe_counts = self.idx_cond(nonempty, counts, one);
                let divisor = safe_counts.cast(accumulate).expand_dim(1, embedding_size);
                (sum / divisor, counts)
            }
            _ => {
                let lowest = self.full_tensor(gathered.dims(), accumulate, f64::NEG_INFINITY);
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
                let zero_output = self.full_tensor(values.dims(), accumulate, 0.0);
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
            // Each index belongs to at most one bag, so summing the bag id
            // where it is a member names that bag (0 for none). The ids
            // are an exact Int iota selected natively and summed in I64,
            // proved in width from the iota bounds; F32 ids round past 2^24.
            let bag_ids = self
                .cx
                .arange(bag_count)
                .cast(DType::I64)
                .expand_dim(1, index_count);
            let no_bag = self.cx.constant_i64(0).expand_rhs(bag_ids.dims());
            bag_membership.select(bag_ids, no_bag).sum(vec![0])
        };
        let bag_size = if mode == 0 {
            self.cx.constant_i64(0).expand_rhs(vec![bag_count])
        } else {
            counts
        };
        let output = super::convert(output, weight.dtype);
        self.bind_outputs(node, vec![output, offset_to_bag, bag_size, max_indices])
    }
}
