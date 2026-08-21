//! Symbolic-dim-safe `gather_elements` / `scatter_elements` / `scatter_nd`
//! lowerings for the PT2 translator.
//!
//! The luminal-core versions in `luminal::frontend::movement` require
//! concrete shape dims — they call `d.to_usize().expect(...)` on every
//! input dim and panic at translate-time when `torch.compile` hands us a
//! batch dim, sequence-length dim, or any other dynamic dim. PT2's whole
//! point is dynamic shapes, so we re-implement the same three ops here
//! using `Expression`-typed shape arithmetic and only call luminal-core
//! primitives that already accept `Expression`s (`Graph::constant`,
//! `Graph::iota`, `flatten_strides`, `ShapeTracker::new(Vec<Expression>)`,
//! `expand_dim`, `expand_rhs`, `flatten`, `slice_along`, `squeeze`,
//! `cast`, `scatter`, `gather`).
//!
//! Every shape product flows through `crate::dim_arith::product_of_dims`
//! so the `Expression`s we build are canonical: two callers that produce
//! the same logical dim via differently-ordered multiplications end up
//! with byte-identical `Expression`s. Without this, downstream dim-equality
//! asserts in luminal-core's `Add` / `Sub` (see `src/frontend/binary.rs`)
//! panic on `a*8` ≠ `8*a` after these helpers feed into broadcast paths.

use luminal::prelude::*;

use crate::dim_arith::product_of_dims;

#[derive(Clone, Copy, Debug)]
pub(super) enum ScatterReduction {
    Add,
    Multiply,
}

/// Row-major strides as `Expression`s. `stride[i] = prod(dims[i+1..])`.
pub(super) fn row_major_strides(dims: &[Expression]) -> Vec<Expression> {
    let rank = dims.len();
    (0..rank)
        .map(|i| product_of_dims(dims[i + 1..].iter().copied()))
        .collect()
}

/// Build a tensor of logical flat indices from one contribution expression
/// per output axis. `Gather` applies the data tensor's ShapeTracker after this
/// mapping, so callers deliberately use logical row-major strides here rather
/// than the data tensor's physical strides.
pub(super) fn logical_flat_indices(
    graph: &mut Graph,
    output_shape: &[Expression],
    axis_contributions: &[Expression],
    base: Expression,
) -> GraphTensor {
    let index = base + flatten_strides(output_shape, axis_contributions);
    graph.iota(index, output_shape.to_vec())
}

/// Build the additive non-axis contribution to a flat index over a
/// rank-`rank` output of shape `out_shape`. The axis dim contributes
/// 0; every other dim `d` contributes `iota_d * strides[d]`. Materialised
/// via one `Graph::iota` call with `flatten_strides(out_shape, axis_exprs)`
/// — same pattern luminal core uses, just with `Expression` throughout.
fn non_axis_flat(
    graph: &mut Graph,
    out_shape: &[Expression],
    strides: &[Expression],
    axis: usize,
) -> GraphTensor {
    let rank = out_shape.len();
    let axis_exprs: Vec<Expression> = (0..rank)
        .map(|d| {
            if d == axis {
                Expression::from(0)
            } else {
                Expression::from('z') * strides[d]
            }
        })
        .collect();
    graph.iota(flatten_strides(out_shape, &axis_exprs), out_shape.to_vec())
}

/// Wrap negative axis indices into `[0, axis_dim)`. Equivalent to
/// `if idx < 0 { idx + axis_dim } else { idx }` in tensor form.
fn normalize_negative_index(indices: GraphTensor, axis_dim: Expression) -> GraphTensor {
    let idx_f32 = indices.cast(DType::F32);
    let zero = idx_f32
        .graph()
        .constant_float(0.0)
        .expand_rhs(idx_f32.shape);
    let adj = idx_f32
        .graph()
        .constant(axis_dim)
        .cast(DType::F32)
        .expand_rhs(idx_f32.shape);
    let is_neg = idx_f32.lt(zero).cast(DType::F32);
    (idx_f32 + (is_neg * adj)).cast(DType::Int)
}

/// Translator-local `gather_elements` that accepts symbolic shape dims.
/// Mirrors `GraphTensor::gather_elements` semantics but uses
/// `Expression`-typed shape arithmetic and only calls symbol-safe
/// luminal-core primitives.
///
/// `output[i0,..,ik] = self[i0,..,i_{axis-1}, indices[i0,..,ik], i_{axis+1},..,ik]`
pub fn pt2_gather_elements(data: GraphTensor, indexes: GraphTensor, axis: usize) -> GraphTensor {
    let dims = data.dims();
    let out_shape: Vec<Expression> = indexes.dims();
    let strides = row_major_strides(&dims);

    let idx_normalized = normalize_negative_index(indexes, dims[axis]);
    let non_axis_flat = non_axis_flat(data.graph(), &out_shape, &strides, axis);

    let stride_tensor = data
        .graph()
        .constant(strides[axis])
        .expand_rhs(idx_normalized.shape);
    let flat_idx = non_axis_flat + idx_normalized * stride_tensor;

    data.gather(flat_idx)
}

/// `index_select` with a rank-0 or rank-1 index tensor. Unlike
/// `gather_elements`, ATen does not wrap negative indices here; invalid index
/// values therefore flow to Gather and fail instead of being normalized.
pub fn pt2_index_select(
    data: GraphTensor,
    indices: GraphTensor,
    axis: usize,
    output_shape: &[Expression],
) -> GraphTensor {
    let rank = data.shape.len();
    let indices = if indices.shape.is_empty() {
        indices.unsqueeze(0)
    } else {
        indices
    };
    assert_eq!(
        indices.shape.len(),
        1,
        "index_select index must be rank 0 or 1"
    );
    assert_eq!(output_shape.len(), rank);

    let inserted_axes = (0..rank).filter(|&dim| dim != axis).collect::<Vec<_>>();
    let indices = indices
        .expand_to_shape_on_axes(output_shape.to_vec(), inserted_axes)
        .cast(DType::Int);
    let strides = row_major_strides(&data.dims());
    let non_axis_flat = non_axis_flat(data.graph(), output_shape, &strides, axis);
    let axis_stride = data
        .graph()
        .constant(strides[axis])
        .expand_rhs(indices.shape);
    data.gather(non_axis_flat + indices * axis_stride)
}

/// Translator-local `scatter_elements` that accepts symbolic shape dims.
/// Same semantics as `GraphTensor::scatter_elements`.
pub(super) fn pt2_scatter_element_indices(
    data: GraphTensor,
    indices: GraphTensor,
    axis: usize,
) -> GraphTensor {
    let data_dims = data.dims();
    let idx_shape: Vec<Expression> = indices.dims();
    let strides = row_major_strides(&data_dims);

    let idx_normalized = normalize_negative_index(indices, data_dims[axis]);
    let non_axis_flat = non_axis_flat(data.graph(), &idx_shape, &strides, axis);

    let stride_tensor = data
        .graph()
        .constant(strides[axis])
        .expand_rhs(idx_normalized.shape);
    let flat_dest = non_axis_flat + idx_normalized * stride_tensor;

    flat_dest.flatten()
}

pub fn pt2_scatter_elements(
    data: GraphTensor,
    indices: GraphTensor,
    updates: GraphTensor,
    axis: usize,
) -> GraphTensor {
    let data_dims = data.dims();
    let flat_dest_1d = pt2_scatter_element_indices(data, indices, axis);
    let flat_updates = updates.flatten();
    let flat_data = data.flatten();

    let output_flat = flat_updates.scatter(flat_dest_1d, flat_data);

    // View-only reshape back to data shape; the buffer is already laid
    // out row-major from the scatter, so swapping the tracker is safe.
    let mut result = output_flat;
    result.shape = ShapeTracker::new(data_dims);
    result
}

/// Scatter with accumulation for ATen's legacy `reduce=` overloads and
/// `scatter_add`. Core Scatter intentionally implements overwrite/last-write
/// semantics, so duplicate destinations must be combined before each write.
///
/// The update extent controls how many dependent read-modify-write steps the
/// graph contains and therefore must be concrete at translation time. This is
/// an explicit graph-topology restriction, not a layout assumption; every
/// rank, axis, negative index, and duplicate pattern is otherwise handled.
pub(super) fn pt2_scatter_elements_reduce(
    data: GraphTensor,
    indices: GraphTensor,
    updates: GraphTensor,
    axis: usize,
    reduction: ScatterReduction,
) -> anyhow::Result<GraphTensor> {
    anyhow::ensure!(
        indices.shape.len() == updates.shape.len(),
        "scatter reduction requires index/update ranks to match, got {} and {}",
        indices.shape.len(),
        updates.shape.len()
    );
    let index_shape = indices.dims();
    let update_shape = updates.dims();
    anyhow::ensure!(
        index_shape
            .iter()
            .zip(&update_shape)
            .all(|(index, update)| index == update || index.egglog_equal(*update)),
        "scatter reduction currently requires index and update shapes to match"
    );
    let update_count = product_of_dims(index_shape.iter().copied())
        .to_usize()
        .ok_or_else(|| {
            anyhow::anyhow!("scatter reduction requires a concrete update element count")
        })?;

    let flat_destinations = pt2_scatter_element_indices(data, indices, axis);
    let flat_updates = updates.flatten();
    let output_shape = data.dims();
    let mut output = data.flatten();

    for index in 0..update_count {
        let destination = flat_destinations.slice_along(index..index + 1, 0);
        let update = flat_updates.slice_along(index..index + 1, 0);
        let current = output.gather(destination);
        let combined = match reduction {
            ScatterReduction::Add => current + update,
            ScatterReduction::Multiply => current * update,
        };
        output = combined.scatter(destination, output);
    }

    output.shape = ShapeTracker::new(output_shape);
    Ok(output)
}

/// Translator-local `scatter_nd` that accepts symbolic shape dims.
/// Mirrors `GraphTensor::scatter_nd` semantics.
pub fn pt2_scatter_nd(
    data: GraphTensor,
    indices: GraphTensor,
    updates: GraphTensor,
) -> GraphTensor {
    let indices = indices.cast(DType::Int);
    let data_dims = data.dims();
    let data_rank = data_dims.len();
    let idx_dims = indices.dims();
    let idx_rank = idx_dims.len();

    // The last dim of indices is the index width K — it must be
    // concrete at translate-time because it controls how many
    // contribution terms we build statically. HuggingFace's MoE
    // accumulator (the path that brought us here via `index_put`)
    // always passes a literal; non-HF callers with a SymInt K would
    // need a different lowering.
    let k = idx_dims[idx_rank - 1]
        .to_usize()
        .expect("scatter_nd: indices innermost dim (K) must be concrete");
    assert!(k <= data_rank, "scatter_nd: K must be <= data rank");

    // Batch shape = indices shape without last dim.
    let batch_shape: Vec<Expression> = idx_dims[..idx_rank - 1].to_vec();
    let batch_numel = product_of_dims(batch_shape.iter().copied());

    // Trailing shape = data_shape[K..]
    let trailing_shape: Vec<Expression> = data_dims[k..].to_vec();
    let trailing_numel = product_of_dims(trailing_shape.iter().copied());

    let data_strides = row_major_strides(&data_dims);

    // Flatten batch dims of indices to [batch_numel, K] via view reshape.
    let mut indices_flat = indices;
    if idx_rank > 2 {
        indices_flat.shape = ShapeTracker::new(vec![batch_numel, Expression::from(k)]);
    }

    let mut flat_base: Option<GraphTensor> = None;
    for (k_dim, stride) in data_strides.iter().copied().enumerate().take(k) {
        let idx_k = indices_flat.slice_along(k_dim..k_dim + 1, indices_flat.dims().len() - 1);
        let idx_k = idx_k.squeeze(idx_k.dims().len() - 1);

        let stride_tensor = data.graph().constant(stride).expand_rhs(idx_k.shape);
        let contribution = idx_k * stride_tensor;

        flat_base = Some(match flat_base {
            Some(fb) => fb + contribution,
            None => contribution,
        });
    }
    let flat_base = flat_base.unwrap();

    // Trailing-numel concreteness drives whether we need the expand-and-fold
    // path. If trailing_shape is empty OR its numel collapses to 1, the flat
    // base is already the full destination index.
    let trailing_is_unit = trailing_shape.is_empty() || trailing_numel.to_usize() == Some(1);
    let mut full_flat_dest = if trailing_is_unit {
        flat_base
    } else {
        // The trailing offset of flat position `t` is just `t`.
        //
        // `data_strides` is row-major over `data_dims`, so `data_strides[k..]`
        // are exactly the row-major strides of `trailing_shape` — walking the
        // trailing block in flat order walks memory in step. There is nothing
        // to weight.
        //
        // This replaced a per-dim loop that built an `arange` for each trailing
        // dim, gave it EXPANDED (0-stride) dims to broadcast, and then
        // overwrote its ShapeTracker with a contiguous `[trailing_numel]` view.
        // Overwriting a tracker is a view-only reshape, which is valid only if
        // the buffer really is laid out that way — an expanded dim is virtual,
        // so it is not. It went unnoticed because a rank-2 target has
        // trailing_rank 1 and never introduces an expanded dim; from rank 3 up
        // it does, and the scatter then wrote one element per row instead of
        // the whole block.
        let base_expanded = flat_base.expand_dim(1, trailing_numel);
        let offsets = data
            .graph()
            .arange(trailing_numel)
            .expand_dim(0, batch_numel);
        base_expanded + offsets
    };

    full_flat_dest = full_flat_dest.flatten();

    let flat_updates = updates.flatten();
    let flat_data = data.flatten();

    let output_flat = flat_updates.scatter(full_flat_dest, flat_data);

    let mut result = output_flat;
    result.shape = ShapeTracker::new(data_dims);
    result
}
