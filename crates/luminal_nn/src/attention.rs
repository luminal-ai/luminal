use luminal::prelude::*;
use luminal::shape::IntExpr;

use crate::{CacheAccess, KvCache};

const MASKED_SCORE: f32 = -1e10;

fn flattened_rows(tensor: GraphTensor) -> (GraphTensor, Vec<IntExpr>) {
    assert!(tensor.rank() >= 2, "row operations require rank >= 2");
    let payload_dims = tensor.dims()[1..].to_vec();
    let mut flattened = tensor.view();
    while flattened.rank() > 2 {
        flattened = flattened.merge_dims(1, 2);
    }
    (flattened.finish(), payload_dims)
}

fn restore_row_payload(tensor: GraphTensor, payload_dims: &[IntExpr]) -> GraphTensor {
    let mut tensor = tensor.view();
    for axis in 0..payload_dims.len().saturating_sub(1) {
        let inner = payload_dims[axis + 1..]
            .iter()
            .copied()
            .fold(IntExpr::from(1), |acc, dim| acc * dim)
            .simplify();
        tensor = tensor.split_dims(axis + 1, inner);
    }
    tensor.finish()
}

/// Gather entries along the first axis while preserving all trailing axes.
///
/// `data` has shape `[rows, ...payload]`, `indices` has shape `[n]`, and
/// the result has shape `[n, ...payload]`.
pub fn gather_rows(data: GraphTensor, indices: GraphTensor) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int, "row indices must be Int");
    assert_eq!(indices.rank(), 1, "row indices must be rank 1");
    let (data, payload_dims) = flattened_rows(data);
    let n = indices.dims1();
    let width = data.dims()[1];
    let rows = indices.expand_dim(1, width);
    let cols = data.graph().iota((n, width), |coordinate| coordinate[1]);
    restore_row_payload(data.gather(&[rows, cols]), &payload_dims)
}

/// Overwrite entries along the first axis while preserving trailing axes.
///
/// `src` has shape `[n, ...payload]`, `indices` has shape `[n]`, and
/// `dest` has shape `[rows, ...payload]`.
pub fn scatter_rows(src: GraphTensor, indices: GraphTensor, dest: GraphTensor) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int, "row indices must be Int");
    assert_eq!(indices.rank(), 1, "row indices must be rank 1");
    assert_eq!(
        src.dims()[0],
        indices.dims1(),
        "source/index length mismatch"
    );
    assert_eq!(
        &src.dims()[1..],
        &dest.dims()[1..],
        "source and destination row shapes must match"
    );
    let (src, payload_dims) = flattened_rows(src);
    let (dest, _) = flattened_rows(dest);
    let n = indices.dims1();
    let width = src.dims()[1];
    let rows = indices.expand_dim(1, width);
    let cols = src.graph().iota((n, width), |coordinate| coordinate[1]);
    restore_row_payload(dest.scatter(&[rows, cols], src), &payload_dims)
}

/// The head layout and score scale for grouped-query attention.
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl AttentionGeometry {
    /// Create a head layout with the standard inverse-square-root score scale.
    pub fn new(query_heads: usize, kv_heads: usize, head_dim: usize) -> Self {
        Self::with_scale(
            query_heads,
            kv_heads,
            head_dim,
            1.0 / (head_dim as f32).sqrt(),
        )
    }

    /// Create a head layout with an explicit score scale.
    pub fn with_scale(query_heads: usize, kv_heads: usize, head_dim: usize, scale: f32) -> Self {
        let geometry = Self {
            query_heads,
            kv_heads,
            head_dim,
            scale,
        };
        geometry.validate();
        geometry
    }

    pub fn groups(self) -> usize {
        self.validate();
        self.query_heads / self.kv_heads
    }

    fn validate(self) {
        assert!(self.query_heads > 0, "query_heads must be nonzero");
        assert!(self.kv_heads > 0, "kv_heads must be nonzero");
        assert!(self.head_dim > 0, "head_dim must be nonzero");
        assert_eq!(
            self.query_heads % self.kv_heads,
            0,
            "query_heads must be divisible by kv_heads"
        );
        assert!(self.scale.is_finite(), "attention scale must be finite");
    }
}

/// Output of a paged-attention operation, including the updated cache value.
#[derive(Clone, Copy)]
pub struct PagedAttentionOutput {
    pub output: GraphTensor,
    pub cache: KvCache,
}

fn assert_vector(tensor: GraphTensor, name: &str) {
    assert_eq!(tensor.rank(), 1, "{name} must be rank 1");
    assert_eq!(tensor.dtype, DType::Int, "{name} must be Int");
}

/// Build an additive causal bias from explicit logical token positions.
///
/// Query and key positions need not match their physical cache slots or their
/// order in the gathered context.
pub fn causal_bias(query_positions: GraphTensor, key_positions: GraphTensor) -> GraphTensor {
    assert_vector(query_positions, "query_positions");
    assert_vector(key_positions, "key_positions");
    let query_count = query_positions.dims1();
    let key_count = key_positions.dims1();
    query_positions
        .expand_dim(1, key_count)
        .lt(key_positions.expand_dim(0, query_count))
        .cast(DType::F32)
        * MASKED_SCORE
}

/// Build an additive bias that isolates queries and keys by sequence ID.
///
/// This can be added to [`causal_bias`] or [`sliding_window_bias`] when a
/// gathered context interleaves multiple sequences in an arbitrary order.
pub fn sequence_isolation_bias(
    query_sequence_ids: GraphTensor,
    key_sequence_ids: GraphTensor,
) -> GraphTensor {
    assert_vector(query_sequence_ids, "query_sequence_ids");
    assert_vector(key_sequence_ids, "key_sequence_ids");
    let query_count = query_sequence_ids.dims1();
    let key_count = key_sequence_ids.dims1();
    let same_sequence = query_sequence_ids
        .expand_dim(1, key_count)
        .eq(key_sequence_ids.expand_dim(0, query_count))
        .cast(DType::F32);
    same_sequence * -MASKED_SCORE + MASKED_SCORE
}

/// Build a causal additive bias restricted to a trailing logical window.
pub fn sliding_window_bias(
    query_positions: GraphTensor,
    key_positions: GraphTensor,
    window: usize,
) -> GraphTensor {
    assert!(window > 0, "attention window must be nonzero");
    assert_vector(query_positions, "query_positions");
    assert_vector(key_positions, "key_positions");
    let query_count = query_positions.dims1();
    let key_count = key_positions.dims1();
    let queries = query_positions.expand_dim(1, key_count);
    let keys = key_positions.expand_dim(0, query_count);
    let future = queries.lt(keys).cast(DType::F32);
    let before_window = keys
        .cast(DType::F32)
        .lt(queries.cast(DType::F32) - (window as f32 - 1.0))
        .cast(DType::F32);
    (future + before_window) * MASKED_SCORE
}

/// Map each item in a packed tensor to the sequence containing it.
///
/// `indptr` contains the cumulative sequence boundaries and has shape
/// `[batch + 1]`; the result has shape `[item_count]`.
pub fn packed_sequence_ids(indptr: GraphTensor, item_count: IntExpr) -> GraphTensor {
    assert_vector(indptr, "indptr");
    let boundary_count = indptr.dims1();
    let items = indptr
        .graph()
        .arange(item_count)
        .expand_dim(1, boundary_count);
    let boundaries = indptr.expand_dim(0, item_count);
    boundaries
        .le(items)
        .cast(DType::Int)
        .sum(1)
        .cast(DType::Int)
        - 1
}

fn packed_position_bias(
    query_positions: GraphTensor,
    query_indptr: GraphTensor,
    context_indptr: GraphTensor,
    context_count: IntExpr,
    window: Option<usize>,
) -> GraphTensor {
    assert_vector(query_positions, "query_positions");
    assert_vector(query_indptr, "query_indptr");
    assert_vector(context_indptr, "context_indptr");
    assert_eq!(
        query_indptr.dims1(),
        context_indptr.dims1(),
        "query and context indptrs must describe the same batch"
    );
    if let Some(window) = window {
        assert!(window > 0, "attention window must be nonzero");
    }

    let query_count = query_positions.dims1();
    let query_sequence = packed_sequence_ids(query_indptr, query_count);
    let context_sequence = packed_sequence_ids(context_indptr, context_count);
    let context_range = query_positions.graph().arange(context_count);
    let context_start = context_indptr.gather(&[context_sequence]);
    let context_positions = context_range - context_start;

    let same_sequence = query_sequence
        .expand_dim(1, context_count)
        .eq(context_sequence.expand_dim(0, query_count));
    let queries = query_positions.expand_dim(1, context_count);
    let keys = context_positions.expand_dim(0, query_count);
    let mut allowed = same_sequence.cast(DType::F32) * keys.le(queries).cast(DType::F32);
    if let Some(window) = window {
        allowed *= (queries.cast(DType::F32) - (window as f32 - 1.0))
            .le(keys.cast(DType::F32))
            .cast(DType::F32);
    }
    allowed * -MASKED_SCORE + MASKED_SCORE
}

/// Build an additive causal bias for packed variable-length sequences.
///
/// Context rows for each sequence must be ordered by logical position within
/// that sequence. The physical cache slots may have any layout. Runtime
/// indptr inputs participate in integer arithmetic and therefore need valid
/// value-range attestations when the selected runtime requires them.
pub fn packed_causal_bias(
    query_positions: GraphTensor,
    query_indptr: GraphTensor,
    context_indptr: GraphTensor,
    context_count: IntExpr,
) -> GraphTensor {
    packed_position_bias(
        query_positions,
        query_indptr,
        context_indptr,
        context_count,
        None,
    )
}

/// Packed causal bias with a trailing logical attention window.
pub fn packed_sliding_window_bias(
    query_positions: GraphTensor,
    query_indptr: GraphTensor,
    context_indptr: GraphTensor,
    context_count: IntExpr,
    window: usize,
) -> GraphTensor {
    packed_position_bias(
        query_positions,
        query_indptr,
        context_indptr,
        context_count,
        Some(window),
    )
}

/// Convert an additive score bias into grouped-query score layout.
///
/// Rank-2 input is shared by every head. Rank-3 input supplies one bias per
/// query head. Rank-4 input is already in canonical grouped-query layout.
pub fn grouped_query_score_bias(bias: GraphTensor, geometry: AttentionGeometry) -> GraphTensor {
    let groups = geometry.groups();
    match bias.rank() {
        2 => bias
            .view()
            .expand_dim(0, geometry.kv_heads)
            .expand_dim(1, groups)
            .finish(),
        3 => {
            assert_eq!(
                bias.dims()[0],
                IntExpr::from(geometry.query_heads),
                "rank-3 attention bias must have one row per query head"
            );
            bias.split_dims(0, groups)
        }
        4 => {
            assert_eq!(
                &bias.dims()[..2],
                &[IntExpr::from(geometry.kv_heads), IntExpr::from(groups)],
                "rank-4 attention bias must use grouped-query head layout"
            );
            bias
        }
        rank => panic!("attention bias must have rank 2, 3, or 4, got {rank}"),
    }
}

/// Compute scaled grouped-query scores in
/// `[kv_heads, groups, queries, context]` layout.
pub fn grouped_query_scores(
    query: GraphTensor,
    keys: GraphTensor,
    geometry: AttentionGeometry,
) -> GraphTensor {
    let groups = geometry.groups();
    assert_eq!(query.rank(), 2, "query must have shape [queries, width]");
    assert_eq!(keys.rank(), 2, "keys must have shape [context, width]");
    assert_eq!(
        query.dims()[1],
        IntExpr::from(geometry.query_heads * geometry.head_dim),
        "query width does not match attention geometry"
    );
    assert_eq!(
        keys.dims()[1],
        IntExpr::from(geometry.kv_heads * geometry.head_dim),
        "key width does not match attention geometry"
    );

    let query = query
        .view()
        .split_dims(1, geometry.head_dim)
        .split_dims(1, groups)
        .permute((1, 2, 0, 3))
        .finish();
    let keys = keys
        .view()
        .split_dims(1, geometry.head_dim)
        .permute((1, 2, 0))
        .expand_dim(1, groups)
        .finish();
    query.matmul(keys) * geometry.scale
}

/// Apply grouped-query attention weights to values and merge the query heads.
pub fn grouped_query_apply(
    weights: GraphTensor,
    values: GraphTensor,
    geometry: AttentionGeometry,
) -> GraphTensor {
    let groups = geometry.groups();
    assert_eq!(weights.rank(), 4, "attention weights must be rank 4");
    assert_eq!(values.rank(), 2, "values must have shape [context, width]");
    assert_eq!(
        weights.dims()[0],
        IntExpr::from(geometry.kv_heads),
        "attention weight KV-head count does not match geometry"
    );
    assert_eq!(
        weights.dims()[1],
        IntExpr::from(groups),
        "attention weight group count does not match geometry"
    );
    assert_eq!(
        weights.dims()[3],
        values.dims()[0],
        "attention context length does not match values"
    );
    assert_eq!(
        values.dims()[1],
        IntExpr::from(geometry.kv_heads * geometry.head_dim),
        "value width does not match attention geometry"
    );

    let values = values
        .view()
        .split_dims(1, geometry.head_dim)
        .permute((1, 0, 2))
        .expand_dim(1, groups)
        .finish();
    weights
        .matmul(values)
        .view()
        .permute((2, 0, 1, 3))
        .merge_dims(1, 2)
        .merge_dims(1, 2)
        .finish()
}

/// Grouped-query attention over already materialized keys and values.
///
/// `score_bias` may have shape `[queries, context]`,
/// `[query_heads, queries, context]`, or the canonical grouped-query score
/// shape `[kv_heads, groups, queries, context]`.
pub fn grouped_query_attention(
    query: GraphTensor,
    keys: GraphTensor,
    values: GraphTensor,
    score_bias: GraphTensor,
    geometry: AttentionGeometry,
) -> GraphTensor {
    let scores = grouped_query_scores(query, keys, geometry);
    let bias = grouped_query_score_bias(score_bias, geometry).cast(scores.dtype);
    assert_eq!(
        bias.dims(),
        scores.dims(),
        "attention bias dimensions do not match scores"
    );
    grouped_query_apply((scores + bias).softmax(3), values, geometry)
}

/// Write new keys and values, read the requested context, and evaluate GQA.
///
/// Cache allocation and page-table policy stay outside this function. The
/// caller supplies physical write/read slots and an additive score bias built
/// from logical sequence metadata.
pub fn paged_attention(
    query: GraphTensor,
    new_keys: GraphTensor,
    new_values: GraphTensor,
    cache: KvCache,
    access: CacheAccess,
    score_bias: GraphTensor,
    geometry: AttentionGeometry,
) -> PagedAttentionOutput {
    assert_eq!(
        query.dims()[0],
        access.write_slots.dims1(),
        "one cache write slot is required per query"
    );
    let cache = cache.write(access.write_slots, new_keys, new_values);
    let context = cache.read(access.read_slots);
    PagedAttentionOutput {
        output: grouped_query_attention(query, context.keys, context.values, score_bias, geometry),
        cache,
    }
}

/// Rotary embedding applied through a PAIRING MATRIX — the concat-free
/// spelling (workaround for the rejoin saturation divergence,
/// 2026-08-10): rope(x) = x ⊙ cos + (x @ R) ⊙ sin, where R is the
/// constant signed pair-permutation encoding the family's rotation
/// (split-half or interleaved — see [`rope_pairing_matrix`]) and the
/// full-width per-position cos/sin tables are host-precomputed (see
/// [`rope_tables_split_half`]; the flux2 example's own convention).
/// Mathematically identical to the slice/neg/concat rotation; the
/// matmul and muls are COMPUTE ops, so no pure-view pad∘slice stack —
/// the divergence precondition — ever forms.
///
/// x (s, n·head_dim); cos/sin (s, head_dim); rot (head_dim, head_dim).
pub fn rotary_apply(
    x: GraphTensor,
    head_dim: usize,
    cos: GraphTensor,
    sin: GraphTensor,
    rot: GraphTensor,
) -> GraphTensor {
    let heads = x.split_dims(1, head_dim); // (s, n, head_dim)
    let dims = heads.dims();
    let rotated = heads.matmul(rot); // (s, n, head_dim)
    let cos = cos.view().unsqueeze(1).expand(dims.clone()).finish();
    let sin = sin.view().unsqueeze(1).expand(dims).finish();
    (heads * cos + rotated * sin).merge_dims(1, 2)
}

/// The signed pair-permutation R with x@R = rot(x) for the SPLIT-HALF
/// rotation rot(x) = [−x₂ ‖ x₁] (halves pair (j, j+h/2)): column i<h/2
/// reads −x[i+h/2], column i≥h/2 reads +x[i−h/2]. Host-side constant,
/// fed as an input like the tables. `interleaved` gives the flux2
/// adjacent-pair form rot(x) = [−x₁, x₀, −x₃, x₂, …] instead.
pub fn rope_pairing_matrix(head_dim: usize, interleaved: bool) -> Vec<f32> {
    let mut rot = vec![0f32; head_dim * head_dim];
    for column in 0..head_dim {
        // rot(x)[column] = ±x[source]  ⇒  R[source, column] = ±1
        let (source, sign) = if interleaved {
            if column % 2 == 0 {
                (column + 1, -1.0)
            } else {
                (column - 1, 1.0)
            }
        } else {
            let half = head_dim / 2;
            if column < half {
                (column + half, -1.0)
            } else {
                (column - half, 1.0)
            }
        };
        rot[source * head_dim + column] = sign;
    }
    rot
}

/// Host-side split-half rope tables for decode positions: row p gets
/// `cos/sin(pos[p] * scale * theta^(-2j/head_dim))` at both slots (j and
/// j+h/2) of frequency j — the full-width tables [`rotary_apply`]
/// consumes. Dual-theta families (gemma) call this once per layer role.
pub fn rope_tables_split_half(
    positions: &[f32],
    head_dim: usize,
    theta: f32,
    pos_scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &position in positions {
        let (mut cos_row, mut sin_row) = (vec![0f32; head_dim], vec![0f32; head_dim]);
        for j in 0..half {
            let freq = theta.powf(-2.0 * j as f32 / head_dim as f32);
            let arg = position * pos_scale * freq;
            cos_row[j] = arg.cos();
            cos_row[j + half] = arg.cos();
            sin_row[j] = arg.sin();
            sin_row[j + half] = arg.sin();
        }
        cos.extend(cos_row);
        sin.extend(sin_row);
    }
    (cos, sin)
}

/// Host-side split-half tables for PARTIAL rotary (gemma-4's full
/// layers: only `rotary_fraction` of the head rotates): frequency
/// pairs at or beyond floor(fraction·head_dim/2) get angle 0 — under
/// the pairing form cos=1/sin=0 lanes pass through untouched, so
/// partial rotary needs no construct change, only these tables.
pub fn rope_tables_partial(
    positions: &[f32],
    head_dim: usize,
    theta: f32,
    rotary_fraction: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let rotated = ((rotary_fraction * head_dim as f32) as usize) / 2;
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &position in positions {
        let (mut cos_row, mut sin_row) = (vec![1f32; head_dim], vec![0f32; head_dim]);
        for j in 0..rotated.min(half) {
            let freq = theta.powf(-2.0 * j as f32 / (2.0 * rotated as f32));
            let arg = position * freq;
            cos_row[j] = arg.cos();
            cos_row[j + half] = arg.cos();
            sin_row[j] = arg.sin();
            sin_row[j + half] = arg.sin();
        }
        cos.extend(cos_row);
        sin.extend(sin_row);
    }
    (cos, sin)
}

/// Per-head RMS norm over a flat (s, n_heads·head_dim) projection with a
/// learned (head_dim,) weight — the QK-norm primitive Qwen3 and the DiT
/// family apply to q and k before position encoding. No shift, F32.
/// [`rms_norm_heads`] without a learned weight — gemma-4's VALUE norm
/// (per-head RMS on V, weightless, no rope).
pub fn rms_norm_heads_unweighted(x: GraphTensor, head_dim: usize, epsilon: f32) -> GraphTensor {
    let heads = x.split_dims(1, head_dim);
    let dims = heads.dims();
    let inv = ((heads * heads).mean(2) + epsilon).sqrt().reciprocal();
    (heads * inv.view().unsqueeze(2).expand(dims).finish()).merge_dims(1, 2)
}

pub fn rms_norm_heads(
    x: GraphTensor,
    head_dim: usize,
    weight: GraphTensor,
    epsilon: f32,
) -> GraphTensor {
    let heads = x.split_dims(1, head_dim); // (s, n_heads, head_dim)
    let dims = heads.dims();
    let inv = ((heads * heads).mean(2) + epsilon).sqrt().reciprocal(); // (s, n_heads)
    let scaled = heads
        * inv.view().unsqueeze(2).expand(dims.clone()).finish()
        * weight
            .view()
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(dims)
            .finish();
    scaled.merge_dims(1, 2)
}

/// Plain multi-head attention, no mask, no cache: q (sq, d) attends over
/// k/v (sk, d) — the encoder/cross-attention primitive.
pub fn attention(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    n_heads: usize,
    head_dim: usize,
) -> GraphTensor {
    let scale = 1.0 / (head_dim as f32).sqrt();
    assert_eq!(
        q.dims()[1],
        n_heads * head_dim,
        "query width must equal n_heads * head_dim"
    );
    let sq = q.dims()[0];
    let sk = k.dims()[0];
    let q = q.view().split_dims(1, head_dim).permute((1, 0, 2)).finish(); // (nh, sq, hd)
    let k = k.view().split_dims(1, head_dim).permute((1, 2, 0)).finish(); // (nh, hd, sk)
    let v = v.view().split_dims(1, head_dim).permute((1, 0, 2)).finish(); // (nh, sk, hd)
    let scores = q.matmul(k) * scale; // (nh, sq, sk)
    let weights = scores.softmax(2);
    let out = weights.matmul(v); // (nh, sq, hd)
    let _ = (sq, sk);
    out.view().permute((1, 0, 2)).merge_dims(1, 2).finish() // (sq, nh·hd)
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionGeometry, causal_bias, gather_rows, grouped_query_attention, packed_causal_bias,
        paged_attention, scatter_rows, sequence_isolation_bias, sliding_window_bias,
    };
    use crate::{CacheAccess, KvCache};
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::shape::IntExpr;
    use luminal_reference::ReferenceRuntime;
    use rustc_hash::FxHashMap;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// Row gather through the M3 ladder: out[i] = data[indices[i]].
    #[test]
    fn gather_rows_selects_rows() {
        let mut cx = Graph::new();
        let data = cx.tensor((4, 3), DType::F32);
        let idx = cx.tensor(2, DType::Int);
        let out = gather_rows(data, idx).output();

        let data_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let idx_vals = vec![2i32, 0];
        let expected = vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0];

        let mut inputs = FxHashMap::default();
        inputs.insert(data.id, data_vals.clone().into());
        inputs.insert(idx.id, idx_vals.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&inputs, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(data.id, data_vals);
        rt.set_data(idx.id, idx_vals);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    #[test]
    fn gather_rows_preserves_trailing_axes() {
        let mut cx = Graph::new();
        let data = cx.tensor((3, 2, 2), DType::F32);
        let indices = cx.tensor(2, DType::Int);
        let output = gather_rows(data, indices).output();
        assert_eq!(
            output.dims(),
            vec![IntExpr::from(2), IntExpr::from(2), IntExpr::from(2)]
        );

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (
                    data.id,
                    (0..12).map(|value| value as f32).collect::<Vec<_>>().into(),
                ),
                (indices.id, vec![2i32, 0].into()),
            ],
        );
        assert_close(
            rt.get_f32(output.id).expect("gathered rows"),
            &[8.0, 9.0, 10.0, 11.0, 0.0, 1.0, 2.0, 3.0],
        );
    }

    /// Row scatter through the M3 ladder: copy dest, replace the indexed
    /// rows with src.
    #[test]
    fn scatter_rows_replaces_rows() {
        let mut cx = Graph::new();
        let src = cx.tensor((2, 3), DType::F32);
        let idx = cx.tensor(2, DType::Int);
        let dest = cx.tensor((4, 3), DType::F32);
        let out = scatter_rows(src, idx, dest).output();
        assert_eq!(out.dims(), dest.dims());

        let src_vals = vec![100.0f32, 101.0, 102.0, 200.0, 201.0, 202.0];
        let idx_vals = vec![1i32, 3];
        let dest_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let expected = vec![
            0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 6.0, 7.0, 8.0, 200.0, 201.0, 202.0,
        ];

        let mut inputs = FxHashMap::default();
        inputs.insert(src.id, src_vals.clone().into());
        inputs.insert(idx.id, idx_vals.clone().into());
        inputs.insert(dest.id, dest_vals.clone().into());
        let mut rt = ReferenceRuntime::load(&cx).expect("native load");
        rt.search(&inputs, &ImplementationSearchOptions::default())
            .expect("search finds a plan");
        rt.set_data(src.id, src_vals);
        rt.set_data(idx.id, idx_vals);
        rt.set_data(dest.id, dest_vals);
        rt.execute().expect("winner executes");
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }

    /// Full paged attention, decode step: one new token (s=1) after one
    /// cached token, 2 KV heads with no grouping (kv_groups=1), head_dim 2.
    /// Reference computed by scalar loops below; run_reference drives the full
    /// search ladder (rediagnosis 2026-08-12: it is not a plain
    /// extraction shortcut).
    #[test]
    fn paged_attention_decode_step_matches_scalar_reference() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM; // == kv_dim here
        const SLOTS: usize = 4;
        const CTX: usize = 2;
        let prev_seq = 1usize;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN), DType::F32);
        let k_new = cx.tensor((1, HIDDEN), DType::F32);
        let v_new = cx.tensor((1, HIDDEN), DType::F32);
        let k_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let v_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let gather_idx = cx.tensor(CTX, DType::Int);
        let scatter_idx = cx.tensor(1, DType::Int);
        let query_positions = cx.iota(1, |c| c[0] + IntExpr::from(prev_seq));
        let key_positions = cx.arange(CTX);
        let result = paged_attention(
            q,
            k_new,
            v_new,
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(query_positions, key_positions),
            AttentionGeometry::new(N_HEADS, N_KV_HEADS, HEAD_DIM),
        );
        let attn = result.output.output();
        let k_cache_new = result.cache.keys.output();
        let v_cache_new = result.cache.values.output();

        let q_vals = vec![0.5f32, -0.3, 0.8, 0.1];
        let k_new_vals = vec![0.2f32, 0.4, -0.1, 0.3];
        let v_new_vals = vec![1.0f32, -1.0, 0.5, 2.0];
        let k_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.1).collect();
        let v_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.2 + 1.0).collect();
        let gather_vals = vec![0i32, 1]; // context = slots 0, 1
        let scatter_vals = vec![1i32]; // new KV lands in slot 1

        // Scalar reference. Cache update: row 1 replaced by k_new/v_new.
        let mut k_cache_ref = k_cache_vals.clone();
        let mut v_cache_ref = v_cache_vals.clone();
        k_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&k_new_vals);
        v_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&v_new_vals);
        // Attention per head over the gathered context rows (slots 0, 1).
        // The query is token index prev_seq = 1, so both context positions
        // are causally visible.
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_ref = vec![0.0f32; HIDDEN];
        for h in 0..N_HEADS {
            let q_h = &q_vals[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let mut scores = [0.0f32; CTX];
            for (j, score) in scores.iter_mut().enumerate() {
                let slot = gather_vals[j] as usize;
                let k_row = &k_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                *score = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                let slot = gather_vals[j] as usize;
                let v_row = &v_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                for (d, v) in v_row.iter().enumerate() {
                    attn_ref[h * HEAD_DIM + d] += e / denom * v;
                }
            }
        }

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (q.id, q_vals.into()),
                (k_new.id, k_new_vals.into()),
                (v_new.id, v_new_vals.into()),
                (k_cache.id, k_cache_vals.into()),
                (v_cache.id, v_cache_vals.into()),
                (gather_idx.id, gather_vals.into()),
                (scatter_idx.id, scatter_vals.into()),
            ],
        );
        assert_close(rt.get_f32(attn.id).expect("attn out"), &attn_ref);
        assert_close(rt.get_f32(k_cache_new.id).expect("k cache"), &k_cache_ref);
        assert_close(rt.get_f32(v_cache_new.id).expect("v cache"), &v_cache_ref);
    }

    #[test]
    fn grouped_query_attention_shares_each_kv_head_across_its_query_group() {
        const QUERY_HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const CONTEXT: usize = 2;

        let mut cx = Graph::new();
        let query = cx.tensor((1, QUERY_HEADS * HEAD_DIM), DType::F32);
        let keys = cx.tensor((CONTEXT, KV_HEADS * HEAD_DIM), DType::F32);
        let values = cx.tensor((CONTEXT, KV_HEADS * HEAD_DIM), DType::F32);
        let bias = cx.tensor((1, CONTEXT), DType::F32);
        let output = grouped_query_attention(
            query,
            keys,
            values,
            bias,
            AttentionGeometry::new(QUERY_HEADS, KV_HEADS, HEAD_DIM),
        )
        .output();

        let query_values = vec![0.5, -0.3, 0.8, 0.1, -0.2, 0.7, 0.4, -0.6];
        let key_values = vec![0.2, 0.4, -0.1, 0.3, 0.6, -0.5, 0.9, 0.2];
        let value_values = vec![1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 1.5, -0.75];
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let groups = QUERY_HEADS / KV_HEADS;
        let mut expected = vec![0.0; QUERY_HEADS * HEAD_DIM];
        for query_head in 0..QUERY_HEADS {
            let kv_head = query_head / groups;
            let query_head_values =
                &query_values[query_head * HEAD_DIM..(query_head + 1) * HEAD_DIM];
            let mut scores = [0.0; CONTEXT];
            for (context, score) in scores.iter_mut().enumerate() {
                let key_offset = context * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                *score = query_head_values
                    .iter()
                    .zip(&key_values[key_offset..key_offset + HEAD_DIM])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials = scores.map(|score| (score - max).exp());
            let denominator: f32 = exponentials.iter().sum();
            for (context, exponential) in exponentials.iter().enumerate() {
                let value_offset = context * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                for lane in 0..HEAD_DIM {
                    expected[query_head * HEAD_DIM + lane] +=
                        exponential / denominator * value_values[value_offset + lane];
                }
            }
        }

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (query.id, query_values.into()),
                (keys.id, key_values.into()),
                (values.id, value_values.into()),
                (bias.id, vec![0.0f32; CONTEXT].into()),
            ],
        );
        assert_close(rt.get_f32(output.id).expect("GQA output"), &expected);
    }

    /// Runtime positions and graph-derived positions build the same causal
    /// score bias and both exclude stale cache rows beyond the write frontier.
    #[test]
    fn paged_attention_runtime_positions_mask_future_context() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM;
        const SLOTS: usize = 4;
        const CTX: usize = 3;
        let q_position = 1usize;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN), DType::F32);
        let k_new = cx.tensor((1, HIDDEN), DType::F32);
        let v_new = cx.tensor((1, HIDDEN), DType::F32);
        let k_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let v_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let gather_idx = cx.tensor(CTX, DType::Int);
        let scatter_idx = cx.tensor(1, DType::Int);
        let q_pos = cx.tensor(1, DType::Int);
        let key_positions = cx.arange(CTX);
        let result_pos = paged_attention(
            q,
            k_new,
            v_new,
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(q_pos, key_positions),
            AttentionGeometry::new(N_HEADS, N_KV_HEADS, HEAD_DIM),
        );
        let expr_positions = cx.iota(1, |c| c[0] + IntExpr::from(q_position));
        let result_expr = paged_attention(
            q,
            k_new,
            v_new,
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            causal_bias(expr_positions, key_positions),
            AttentionGeometry::new(N_HEADS, N_KV_HEADS, HEAD_DIM),
        );
        let attn_pos = result_pos.output.output();
        let attn_expr = result_expr.output.output();
        let k_pos_out = result_pos.cache.keys.output();
        let v_pos_out = result_pos.cache.values.output();

        let q_vals = vec![0.5f32, -0.3, 0.8, 0.1];
        let k_new_vals = vec![0.2f32, 0.4, -0.1, 0.3];
        let v_new_vals = vec![1.0f32, -1.0, 0.5, 2.0];
        let k_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.1).collect();
        let v_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.2 + 1.0).collect();
        let gather_vals = vec![0i32, 1, 2]; // slot 2 is beyond position 1
        let scatter_vals = vec![1i32];
        let q_pos_vals = vec![q_position as i32];

        let mut k_cache_ref = k_cache_vals.clone();
        let mut v_cache_ref = v_cache_vals.clone();
        k_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&k_new_vals);
        v_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&v_new_vals);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut attn_ref = vec![0.0f32; HIDDEN];
        for h in 0..N_HEADS {
            let q_h = &q_vals[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            // Only context columns 0 and 1 are visible (col 2 > position 1).
            let visible = 2usize;
            let mut scores = vec![0.0f32; visible];
            for (j, score) in scores.iter_mut().enumerate() {
                let slot = gather_vals[j] as usize;
                let k_row = &k_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                *score = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (j, e) in exps.iter().enumerate() {
                let slot = gather_vals[j] as usize;
                let v_row = &v_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                for (d, v) in v_row.iter().enumerate() {
                    attn_ref[h * HEAD_DIM + d] += e / denom * v;
                }
            }
        }

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (q.id, q_vals.into()),
                (k_new.id, k_new_vals.into()),
                (v_new.id, v_new_vals.into()),
                (k_cache.id, k_cache_vals.into()),
                (v_cache.id, v_cache_vals.into()),
                (gather_idx.id, gather_vals.into()),
                (scatter_idx.id, scatter_vals.into()),
                (q_pos.id, q_pos_vals.into()),
            ],
        );
        assert_close(rt.get_f32(attn_pos.id).expect("positional attn"), &attn_ref);
        assert_close(
            rt.get_f32(attn_expr.id).expect("expression attn"),
            &attn_ref,
        );
        assert_close(rt.get_f32(k_pos_out.id).expect("k cache"), &k_cache_ref);
        assert_close(rt.get_f32(v_pos_out.id).expect("v cache"), &v_cache_ref);
    }

    #[test]
    fn packed_causal_bias_isolates_variable_length_sequences() {
        let mut cx = Graph::new();
        let query_positions = cx.tensor(3, DType::Int);
        let query_indptr = cx.tensor(3, DType::Int);
        let context_indptr = cx.tensor(3, DType::Int);
        let bias = packed_causal_bias(
            query_positions,
            query_indptr,
            context_indptr,
            IntExpr::from(5),
        )
        .output();

        let rt = luminal_reference::harness::run_reference_with_ranges(
            &cx,
            &[
                (query_positions.id, vec![1i32, 2, 0].into()),
                (query_indptr.id, vec![0i32, 2, 3].into()),
                (context_indptr.id, vec![0i32, 3, 5].into()),
            ],
            &[
                (query_positions.id, 0, 2),
                (query_indptr.id, 0, 3),
                (context_indptr.id, 0, 5),
            ],
        );
        assert_close(
            rt.get_f32(bias.id).expect("packed bias"),
            &[
                0.0, 0.0, -1e10, -1e10, -1e10, 0.0, 0.0, 0.0, -1e10, -1e10, -1e10, -1e10, -1e10,
                0.0, -1e10,
            ],
        );
    }

    #[test]
    fn sliding_window_bias_masks_future_and_old_context() {
        let mut cx = Graph::new();
        let query_positions = cx.tensor(1, DType::Int);
        let key_positions = cx.tensor(5, DType::Int);
        let bias = sliding_window_bias(query_positions, key_positions, 2).output();

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (query_positions.id, vec![3i32].into()),
                (key_positions.id, vec![0i32, 1, 2, 3, 4].into()),
            ],
        );
        assert_close(
            rt.get_f32(bias.id).expect("sliding-window bias"),
            &[-1e10, -1e10, 0.0, 0.0, -1e10],
        );
    }

    #[test]
    fn sequence_isolation_bias_supports_interleaved_context() {
        let mut cx = Graph::new();
        let query_sequences = cx.tensor(2, DType::Int);
        let key_sequences = cx.tensor(4, DType::Int);
        let bias = sequence_isolation_bias(query_sequences, key_sequences).output();

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (query_sequences.id, vec![7i32, 9].into()),
                (key_sequences.id, vec![9i32, 7, 9, 7].into()),
            ],
        );
        assert_close(
            rt.get_f32(bias.id).expect("sequence-isolation bias"),
            &[-1e10, 0.0, -1e10, 0.0, 0.0, -1e10, 0.0, -1e10],
        );
    }
}

#[cfg(test)]
mod score_bias_tests {
    use super::{AttentionGeometry, paged_attention};
    use crate::{CacheAccess, KvCache};
    use luminal::prelude::*;

    /// A caller-provided score bias can hide an otherwise visible context row.
    #[test]
    fn paged_attention_accepts_arbitrary_score_bias() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM;
        const SLOTS: usize = 4;
        const CTX: usize = 3;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN), DType::F32);
        let k_new = cx.tensor((1, HIDDEN), DType::F32);
        let v_new = cx.tensor((1, HIDDEN), DType::F32);
        let k_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let v_cache = cx.tensor((SLOTS, HIDDEN), DType::F32);
        let gather_idx = cx.tensor(CTX, DType::Int);
        let scatter_idx = cx.tensor(1, DType::Int);
        let mask = cx.tensor((1, CTX), DType::F32);
        let result = paged_attention(
            q,
            k_new,
            v_new,
            KvCache::new(k_cache, v_cache),
            CacheAccess::new(scatter_idx, gather_idx),
            mask,
            AttentionGeometry::new(N_HEADS, N_KV_HEADS, HEAD_DIM),
        );
        let attn = result.output.output();

        let q_vals = vec![0.5f32, -0.3, 0.8, 0.1];
        let k_new_vals = vec![0.2f32, 0.4, -0.1, 0.3];
        let v_new_vals = vec![1.0f32, -1.0, 0.5, 2.0];
        let k_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.1).collect();
        let v_cache_vals: Vec<f32> = (0..SLOTS * HIDDEN).map(|v| v as f32 * 0.2 + 1.0).collect();
        let gather_vals = vec![0i32, 1, 2];
        let scatter_vals = vec![1i32];
        // Column 1 HIDDEN by the mask even though position-wise it
        // would be visible; column 2 visible.
        let mask_vals = vec![0.0f32, -1e9, 0.0];

        let mut k_cache_ref = k_cache_vals.clone();
        let mut v_cache_ref = v_cache_vals.clone();
        k_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&k_new_vals);
        v_cache_ref[HIDDEN..2 * HIDDEN].copy_from_slice(&v_new_vals);
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let visible: Vec<usize> = vec![0, 2];
        let mut attn_ref = vec![0.0f32; HIDDEN];
        for h in 0..N_HEADS {
            let q_h = &q_vals[h * HEAD_DIM..(h + 1) * HEAD_DIM];
            let mut scores = Vec::new();
            for j in &visible {
                let slot = gather_vals[*j] as usize;
                let k_row = &k_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                scores.push(q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale);
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (index, j) in visible.iter().enumerate() {
                let slot = gather_vals[*j] as usize;
                let v_row = &v_cache_ref[slot * HIDDEN + h * HEAD_DIM..][..HEAD_DIM];
                for (d, v) in v_row.iter().enumerate() {
                    attn_ref[h * HEAD_DIM + d] += exps[index] / denom * v;
                }
            }
        }

        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (q.id, q_vals.into()),
                (k_new.id, k_new_vals.into()),
                (v_new.id, v_new_vals.into()),
                (k_cache.id, k_cache_vals.into()),
                (v_cache.id, v_cache_vals.into()),
                (gather_idx.id, gather_vals.into()),
                (scatter_idx.id, scatter_vals.into()),
                (mask.id, mask_vals.into()),
            ],
        );
        let ours = rt.get_f32(attn.id).expect("masked attn");
        for (index, (a, b)) in ours.iter().zip(&attn_ref).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }
}
