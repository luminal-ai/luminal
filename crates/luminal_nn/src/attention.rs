use luminal::prelude::*;
use luminal::shape::IntExpr;

/// Gather entire rows from a 2D tensor using row indices.
///
/// - `data`: (R, D) tensor
/// - `indices`: (N,) Int tensor of row indices
/// - `d`: the number of columns (D), must match data's second dimension
///
/// Returns: (N, D) tensor where output[i] = data[indices[i]]
///
/// COORDINATE form, not flat sugar — the 2026-07-31 "coordinate form
/// is THE primary" ruling. (Historically this also dodged the
/// Int-in-f32 smuggling's 2^24 cliff; typed buffers now make flat
/// index arithmetic exact too, but per-axis coordinates remain the
/// primary spelling and keep every component bounded by its axis
/// extent.)
pub fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();
    let rows = indices.expand_dim(1, d); // (N, D)
    let cols = data.graph().arange(d as i32).expand_dim(0, n); // (N, D)
    data.gather(&[rows, cols])
}

/// Scatter entire rows into a 2D tensor using row indices.
///
/// - `src`: (N, D) tensor of values to write
/// - `indices`: (N,) Int tensor of destination row indices
/// - `dest`: (R, D) tensor to write into (copied first, then overwritten at index positions)
/// - `d`: the number of columns (D)
///
/// Returns: (R, D) tensor where output = copy(dest); output[indices[i]] = src[i]
///
/// Coordinate form for the same exactness reason as [`gather_rows`].
pub fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    assert_eq!(indices.dtype, DType::Int);
    let n = indices.dims1();
    let rows = indices.expand_dim(1, d); // (N, D) over src's shape
    let cols = src.graph().arange(d as i32).expand_dim(0, n); // (N, D)
    dest.scatter(&[rows, cols], src)
}

/// Pure HLIR paged attention for one layer with causal masking.
///
/// Inputs:
/// - `q`:           (s, hidden)         f32 — query vectors
/// - `k_new`:       (s, kv_dim)         f32 — new key vectors
/// - `v_new`:       (s, kv_dim)         f32 — new value vectors
/// - `k_cache`:     (num_slots, kv_dim) f32 — key cache (preallocated)
/// - `v_cache`:     (num_slots, kv_dim) f32 — value cache (preallocated)
/// - `gather_idx`:  (ctx_len,)          Int — which cache slots to read
/// - `scatter_idx`: (s,)                Int — which cache slots to write new KV into
/// - `prev_seq`:    number of previously cached tokens (for causal mask offset)
/// - `n_heads`:     number of query heads
/// - `n_kv_heads`:  number of KV heads (for GQA)
/// - `head_dim`:    dimension per head
///
/// Returns: (attn_out, k_cache_new, v_cache_new)
///   - `attn_out`:     (s, hidden)         f32
///   - `k_cache_new`:  (num_slots, kv_dim) f32
///   - `v_cache_new`:  (num_slots, kv_dim) f32
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: IntExpr,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    paged_attention_windowed(
        q,
        k_new,
        v_new,
        k_cache,
        v_cache,
        gather_idx,
        scatter_idx,
        prev_seq,
        n_heads,
        n_kv_heads,
        head_dim,
        None,
        1.0 / (head_dim as f32).sqrt(),
    )
}

/// [`paged_attention`] with an optional SLIDING WINDOW (gemma's local
/// layers): position j is additionally masked when j < i − (window − 1),
/// i.e. a query attends at most `window` trailing positions (itself
/// included), in the same position vocabulary the causal mask uses.
/// `score_scale` makes the scale's HOME explicit: 1/√head_dim here for
/// the scale-on-scores spelling, or 1.0 when the caller folds the scale
/// into Q (gemma).
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_windowed(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    prev_seq: IntExpr,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    score_scale: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let s = q.dims()[0];
    let ctx = gather_idx.dims()[0];
    // Query positions synthesized in-graph: row i sits at prev_seq + i.
    let row_vals = q
        .graph()
        .iota(s, |c| c[0] + prev_seq)
        .expand_dim(1, ctx)
        .cast(DType::F32); // (s, ctx)
    paged_attention_core(
        q,
        k_new,
        v_new,
        k_cache,
        v_cache,
        gather_idx,
        scatter_idx,
        row_vals,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        score_scale,
    )
}

/// [`paged_attention_windowed`] with the query positions fed as DATA —
/// a `q_pos` (s,) Int input tensor — instead of a `prev_seq`
/// IntExpr. An IntExpr is baked concrete into the searched plan
/// (bucketed search pins the representative), so a decode loop that
/// wants ONE search and N executes needs the position to arrive through
/// a buffer like any other per-step input. Same cache/gather/mask
/// semantics: context slot j is visible to row i iff j ≤ q_pos[i], so
/// unwritten cache slots (j beyond the write frontier) are masked out
/// by the same comparison. The full-size example decode loops are built
/// on this form.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_positional(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    q_pos: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    score_scale: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    assert_eq!(q_pos.dtype, DType::Int);
    let ctx = gather_idx.dims()[0];
    let row_vals = q_pos.expand_dim(1, ctx).cast(DType::F32); // (s, ctx)
    paged_attention_core(
        q,
        k_new,
        v_new,
        k_cache,
        v_cache,
        gather_idx,
        scatter_idx,
        row_vals,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        score_scale,
    )
}

/// [`paged_attention_positional`] with the ADDITIVE MASK arriving as
/// DATA — an (s, ctx) F32 input of 0.0 (visible) / large-negative
/// (hidden) — instead of deriving from positions in-graph. This is the
/// PAGE-TABLE cache form's contract (see [`crate::PageTable`]): one
/// slot pool serves several sequences per tick, and only the host's
/// table knows which context columns belong to which query row, so
/// causality AND cross-sequence isolation ride the mask together.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_masked(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    mask: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    score_scale: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    let s = q.dims()[0];
    let ctx = gather_idx.dims()[0];
    assert_eq!(mask.dims(), vec![s, ctx], "mask is (s, ctx)");

    let k_cache = scatter_rows(k_new, scatter_idx, k_cache, kv_dim);
    let v_cache = scatter_rows(v_new, scatter_idx, v_cache, kv_dim);
    let k = gather_rows(k_cache, gather_idx, kv_dim);
    let v = gather_rows(v_cache, gather_idx, kv_dim);

    let q = q
        .split_dims(1, head_dim)
        .split_dims(1, kv_groups)
        .permute((1, 2, 0, 3));
    let k = k
        .split_dims(1, head_dim)
        .permute((1, 2, 0))
        .expand_dim(1, kv_groups);
    let v = v
        .split_dims(1, head_dim)
        .permute((1, 0, 2))
        .expand_dim(1, kv_groups);

    let scores = q.matmul(k) * score_scale;
    let mask = mask.expand_dim(0, n_kv_heads).expand_dim(1, kv_groups);
    let weights = (scores + mask).softmax(3);
    let out = weights.matmul(v);
    let out = out.permute((2, 0, 1, 3)).merge_dims(1, 2).merge_dims(1, 2);
    (out, k_cache, v_cache)
}

/// Shared body: `row_vals` (s, ctx) F32 carries each query row's
/// absolute position, however the caller sourced it.
#[allow(clippy::too_many_arguments)]
fn paged_attention_core(
    q: GraphTensor,
    k_new: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    gather_idx: GraphTensor,
    scatter_idx: GraphTensor,
    row_vals: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: Option<usize>,
    score_scale: f32,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    let scale = score_scale;
    let s = q.dims()[0];
    let ctx = gather_idx.dims()[0];
    let cx = q.graph();

    // ── Phase 1: Write new KV into cache ──
    let k_cache = scatter_rows(k_new, scatter_idx, k_cache, kv_dim);
    let v_cache = scatter_rows(v_new, scatter_idx, v_cache, kv_dim);

    // ── Phase 2: Gather context KV from cache ──
    let k = gather_rows(k_cache, gather_idx, kv_dim); // (ctx, kv_dim)
    let v = gather_rows(v_cache, gather_idx, kv_dim); // (ctx, kv_dim)

    // ── Phase 3: Reshape for multi-head attention ──
    // Q: (s, hidden) → (s, n_heads, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    //                 → (n_kv_heads, kv_groups, s, head_dim)
    let q = q
        .split_dims(1, head_dim) // (s, n_heads, head_dim)
        .split_dims(1, kv_groups) // (s, n_kv_heads, kv_groups, head_dim)
        .permute((1, 2, 0, 3)); // (n_kv_heads, kv_groups, s, head_dim)

    // K: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, head_dim, ctx)
    let k = k
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 2, 0)); // (n_kv_heads, head_dim, ctx)

    // V: (ctx, kv_dim) → (ctx, n_kv_heads, head_dim) → (n_kv_heads, ctx, head_dim)
    let v = v
        .split_dims(1, head_dim) // (ctx, n_kv_heads, head_dim)
        .permute((1, 0, 2)); // (n_kv_heads, ctx, head_dim)

    // ── Phase 4: Attention ──
    // Broadcast K, V over kv_groups dimension
    let k = k.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, head_dim, ctx)
    let v = v.expand_dim(1, kv_groups); // (n_kv_heads, kv_groups, ctx, head_dim)

    // QK^T: (n_kv_heads, kv_groups, s, head_dim) @ (n_kv_heads, kv_groups, head_dim, ctx)
    //     → (n_kv_heads, kv_groups, s, ctx)
    let scores = q.matmul(k) * scale;

    // Build causal mask: the query at absolute position row_vals[i] can
    // attend to context j iff j <= row_vals[i].
    // mask[i,j] = -1e9 where row_vals[i] < col_vals[j], else 0
    let col_vals = cx.arange(ctx).expand_dim(0, s).cast(DType::F32); // (s, ctx)
    let mut mask = row_vals.lt(col_vals).cast(DType::F32) * -1e9;
    if let Some(window) = window {
        // masked where col < row − (window − 1)
        let outside = col_vals
            .lt(row_vals - (window as f32 - 1.0))
            .cast(DType::F32)
            * -1e9;
        mask += outside;
    }

    // Broadcast (s, ctx) → (n_kv_heads, kv_groups, s, ctx)
    let mask = mask.expand_dim(0, n_kv_heads).expand_dim(1, kv_groups);
    let scores = scores + mask;

    // Softmax over context dimension (axis 3)
    let weights = scores.softmax(3);

    // Weighted sum: (n_kv_heads, kv_groups, s, ctx) @ (n_kv_heads, kv_groups, ctx, head_dim)
    //            → (n_kv_heads, kv_groups, s, head_dim)
    let out = weights.matmul(v);

    // ── Phase 5: Reshape output ──
    // (n_kv_heads, kv_groups, s, head_dim) → (s, n_kv_heads, kv_groups, head_dim)
    // Head merge as an EXPLICIT view (A2: no tracker reassignment —
    // the old code silently reinterpreted the permuted view as fresh
    // contiguous storage): (s, n_kv, groups, hd) -> (s, n_heads*hd).
    let out = out.permute((2, 0, 1, 3)).merge_dims(1, 2).merge_dims(1, 2);

    (out, k_cache, v_cache)
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
    let cos = cos.unsqueeze(1).expand(dims.clone());
    let sin = sin.unsqueeze(1).expand(dims);
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
/// cos/sin(pos[p]·scale · theta^(−2j/head_dim)) at BOTH slots (j and
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

/// Host-side split-half rope tables with the LLAMA-3.1 frequency ramp
/// (rope_scaling type "llama3"): frequencies whose wavelength exceeds
/// `original_max / low_freq_factor` divide by `factor`; wavelengths
/// below `original_max / high_freq_factor` keep their frequency; the
/// band between interpolates smoothly. Full-width rows (both half
/// slots), same consumption contract as [`rope_tables_split_half`].
#[allow(clippy::too_many_arguments)]
pub fn rope_tables_llama3_scaled(
    positions: &[f32],
    head_dim: usize,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_positions: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let low_freq_wavelen = original_max_positions / low_freq_factor;
    let high_freq_wavelen = original_max_positions / high_freq_factor;
    let scaled_freq = |freq: f32| -> f32 {
        let wavelen = 2.0 * std::f32::consts::PI / freq;
        if wavelen > low_freq_wavelen {
            freq / factor
        } else if wavelen < high_freq_wavelen {
            freq
        } else {
            let smooth = (original_max_positions / wavelen - low_freq_factor)
                / (high_freq_factor - low_freq_factor);
            (1.0 - smooth) * freq / factor + smooth * freq
        }
    };
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &position in positions {
        let (mut cos_row, mut sin_row) = (vec![0f32; head_dim], vec![0f32; head_dim]);
        for j in 0..half {
            let base = theta.powf(-2.0 * j as f32 / head_dim as f32);
            let arg = position * scaled_freq(base);
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
    (heads * inv.unsqueeze(2).expand(dims)).merge_dims(1, 2)
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
        * inv.unsqueeze(2).expand(dims.clone())
        * weight.unsqueeze(0).unsqueeze(0).expand(dims);
    scaled.merge_dims(1, 2)
}

/// Plain multi-head attention, no mask, no cache: q (sq, d) attends over
/// k/v (sk, d) — the encoder/cross-attention primitive.
pub fn attention(
    q: GraphTensor,
    k: GraphTensor,
    v: GraphTensor,
    _n_heads: usize,
    head_dim: usize,
) -> GraphTensor {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let sq = q.dims()[0];
    let sk = k.dims()[0];
    let q = q.split_dims(1, head_dim).permute((1, 0, 2)); // (nh, sq, hd)
    let k = k.split_dims(1, head_dim).permute((1, 2, 0)); // (nh, hd, sk)
    let v = v.split_dims(1, head_dim).permute((1, 0, 2)); // (nh, sk, hd)
    let scores = q.matmul(k) * scale; // (nh, sq, sk)
    let weights = scores.softmax(2);
    let out = weights.matmul(v); // (nh, sq, hd)
    let _ = (sq, sk);
    out.permute((1, 0, 2)).merge_dims(1, 2) // (sq, nh·hd)
}

#[cfg(test)]
mod tests {
    use super::{gather_rows, paged_attention, paged_attention_positional, scatter_rows};
    use luminal::implementation_search::ImplementationSearchOptions;
    use luminal::prelude::*;
    use luminal::reference::ReferenceRuntime;
    use luminal::shape::IntExpr;
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
        let data = cx.tensor((4, 3));
        let idx = cx.tensor_dtyped(2, DType::Int);
        let out = gather_rows(data, idx, 3).output();

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

    /// Row scatter through the M3 ladder: copy dest, replace the indexed
    /// rows with src.
    #[test]
    fn scatter_rows_replaces_rows() {
        let mut cx = Graph::new();
        let src = cx.tensor((2, 3));
        let idx = cx.tensor_dtyped(2, DType::Int);
        let dest = cx.tensor((4, 3));
        let out = scatter_rows(src, idx, dest, 3).output();
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
        let q = cx.tensor((1, HIDDEN));
        let k_new = cx.tensor((1, HIDDEN));
        let v_new = cx.tensor((1, HIDDEN));
        let k_cache = cx.tensor((SLOTS, HIDDEN));
        let v_cache = cx.tensor((SLOTS, HIDDEN));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let (attn, k_cache_new, v_cache_new) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(prev_seq),
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
        );
        let attn = attn.output();
        let k_cache_new = k_cache_new.output();
        let v_cache_new = v_cache_new.output();

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

        let rt = luminal::test_support::run_reference(
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

    /// The positional (data-masked) form against a scalar reference
    /// where the mask BINDS: the query sits at position 1 while the
    /// gather covers slots 0..3, so slot 2 must be masked out (it is
    /// beyond the write frontier and holds stale cache rows). Also
    /// pins positional ≡ the IntExpr form: same fixture, both
    /// attention spellings in one graph, outputs asserted against the
    /// same reference.
    #[test]
    fn paged_attention_positional_masks_beyond_position() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM;
        const SLOTS: usize = 4;
        const CTX: usize = 3;
        let q_position = 1usize;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN));
        let k_new = cx.tensor((1, HIDDEN));
        let v_new = cx.tensor((1, HIDDEN));
        let k_cache = cx.tensor((SLOTS, HIDDEN));
        let v_cache = cx.tensor((SLOTS, HIDDEN));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let q_pos = cx.tensor_dtyped(1, DType::Int);
        let (attn_pos, k_pos_out, v_pos_out) = paged_attention_positional(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            q_pos,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            None,
            1.0 / (HEAD_DIM as f32).sqrt(),
        );
        let (attn_expr, _, _) = paged_attention(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            IntExpr::from(q_position),
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
        );
        let attn_pos = attn_pos.output();
        let attn_expr = attn_expr.output();
        let k_pos_out = k_pos_out.output();
        let v_pos_out = v_pos_out.output();

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

        let rt = luminal::test_support::run_reference(
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
}

#[cfg(test)]
mod masked_tests {
    use super::paged_attention_masked;
    use luminal::prelude::*;

    /// The data-mask form against a scalar reference that honors the
    /// mask array literally — including a hidden slot the positional
    /// form would have shown (proving the mask, not the positions, is
    /// in charge).
    #[test]
    fn paged_attention_masked_matches_scalar_reference() {
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const HIDDEN: usize = N_HEADS * HEAD_DIM;
        const SLOTS: usize = 4;
        const CTX: usize = 3;

        let mut cx = Graph::new();
        let q = cx.tensor((1, HIDDEN));
        let k_new = cx.tensor((1, HIDDEN));
        let v_new = cx.tensor((1, HIDDEN));
        let k_cache = cx.tensor((SLOTS, HIDDEN));
        let v_cache = cx.tensor((SLOTS, HIDDEN));
        let gather_idx = cx.tensor_dtyped(CTX, DType::Int);
        let scatter_idx = cx.tensor_dtyped(1, DType::Int);
        let mask = cx.tensor((1, CTX));
        let (attn, _, _) = paged_attention_masked(
            q,
            k_new,
            v_new,
            k_cache,
            v_cache,
            gather_idx,
            scatter_idx,
            mask,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
        );
        let attn = attn.output();

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

        let rt = luminal::test_support::run_reference(
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
