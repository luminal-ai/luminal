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

/// Row-major strides as `Expression`s. `stride[i] = prod(dims[i+1..])`.
fn row_major_strides(dims: &[Expression]) -> Vec<Expression> {
    let rank = dims.len();
    (0..rank)
        .map(|i| product_of_dims(dims[i + 1..].iter().copied()))
        .collect()
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

/// Translator-local `scatter_elements` that accepts symbolic shape dims.
/// Same semantics as `GraphTensor::scatter_elements`.
pub fn pt2_scatter_elements(
    data: GraphTensor,
    indices: GraphTensor,
    updates: GraphTensor,
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

    let flat_dest_1d = flat_dest.flatten();
    let flat_updates = updates.flatten();
    let flat_data = data.flatten();

    let output_flat = flat_updates.scatter(flat_dest_1d, flat_data);

    // View-only reshape back to data shape; the buffer is already laid
    // out row-major from the scatter, so swapping the tracker is safe.
    let mut result = output_flat;
    result.shape = ShapeTracker::new(data_dims);
    result
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
        let mut base_expanded = flat_base.expand_dim(1, trailing_numel);

        let trailing_rank = trailing_shape.len();
        for (ti, d) in (k..data_rank).enumerate() {
            let ar = data.graph().arange(data_dims[d]);
            let mut ar_shaped = ar;
            for _ in ti + 1..trailing_rank {
                let n = ar_shaped.dims().len();
                ar_shaped = ar_shaped.expand_dim(n, 1);
            }
            for _ in 0..ti {
                ar_shaped = ar_shaped.expand_dim(0, 1);
            }
            ar_shaped.shape.expand(trailing_shape.clone());
            let mut ar_flat = ar_shaped;
            ar_flat.shape = ShapeTracker::new(vec![trailing_numel]);
            ar_flat = ar_flat.expand_dim(0, batch_numel);

            let stride_tensor = data
                .graph()
                .constant(data_strides[d])
                .expand_rhs(ar_flat.shape);
            base_expanded += ar_flat * stride_tensor;
        }
        base_expanded
    };

    full_flat_dest = full_flat_dest.flatten();

    let flat_updates = updates.flatten();
    let flat_data = data.flatten();

    let output_flat = flat_updates.scatter(full_flat_dest, flat_data);

    let mut result = output_flat;
    result.shape = ShapeTracker::new(data_dims);
    result
}
