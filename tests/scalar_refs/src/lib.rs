//! The TRUSTED SCALAR VALIDATORS — plain-Rust reference kernels that
//! fidelity tests compare runtime plans against (validator = trusted
//! kernel, minimal-example-repo doctrine). Rulings 2026-08-13: what a
//! runtime compares to is outside luminal_nn's scope, and validators
//! do not live in the main tree at all — validation of the reference
//! runtime is a TESTS concern, so this is a tests-only crate.
#![allow(dead_code)]

pub fn assert_close(ours: &[f32], expected: &[f32]) {
    assert_eq!(ours.len(), expected.len(), "length mismatch");
    for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 1e-4 * b.abs().max(1.0),
            "element {index}: ours {a} vs expected {b}"
        );
    }
}

/// Deterministic pseudo-random weights (no RNG dependency; values in
/// roughly [-0.6, 0.6] so activations stay in a well-conditioned
/// range).
pub fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// y = x · W, x (1, in), W (in, out).
pub fn ref_matmul(x: &[f32], w: &[f32], in_w: usize, out_w: usize) -> Vec<f32> {
    (0..out_w)
        .map(|c| (0..in_w).map(|k| x[k] * w[k * out_w + c]).sum())
        .collect()
}

/// One decode step of paged attention for a single new token at
/// position `prev_seq`, all context causally visible.
#[allow(clippy::too_many_arguments)]
pub fn ref_paged_step(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    gather: &[usize],
    scatter_slot: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_dim = n_heads * head_dim; // kv_groups = 1 in every model test
    k_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(k_new);
    v_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(v_new);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; kv_dim];
    for h in 0..n_heads {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let scores: Vec<f32> = gather
            .iter()
            .map(|slot| {
                let k_row = &k_cache[slot * kv_dim + h * head_dim..][..head_dim];
                q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for (j, slot) in gather.iter().enumerate() {
            let v_row = &v_cache[slot * kv_dim + h * head_dim..][..head_dim];
            for (d, v) in v_row.iter().enumerate() {
                out[h * head_dim + d] += exps[j] / denom * v;
            }
        }
    }
    out
}

/// LayerNorm as the nn module computes it: mean-subtract, then
/// x / sqrt(mean(x²) + eps), no affine.
pub fn ref_layer_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let centered: Vec<f32> = x.iter().map(|v| v - mean).collect();
    let ms: f32 = centered.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + epsilon).sqrt();
    centered.iter().map(|v| v * inv).collect()
}

/// MoE forward for one row, k = 1: softmax routing, best expert's
/// matmul scaled by its routing weight.
pub fn ref_moe_k1(
    x: &[f32],
    router: &[f32],
    experts: &[f32],
    d: usize,
    e_count: usize,
) -> Vec<f32> {
    let logits: Vec<f32> = (0..e_count)
        .map(|e| (0..d).map(|i| x[i] * router[i * e_count + e]).sum())
        .collect();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let denom: f32 = exps.iter().sum();
    let best = (0..e_count)
        .max_by(|a, b| logits[*a].partial_cmp(&logits[*b]).unwrap())
        .unwrap();
    let weight = exps[best] / denom;
    let w = &experts[best * d * d..(best + 1) * d * d];
    ref_matmul(x, w, d, d).iter().map(|v| v * weight).collect()
}

/// The whole decoder-block reference: x' = x + Wo·attn; x'' = x' + ff(x').
#[allow(clippy::too_many_arguments)]
pub fn ref_block_step(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    ff: &dyn Fn(&[f32]) -> Vec<f32>,
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    gather: &[usize],
    scatter_slot: usize,
    n_heads: usize,
    head_dim: usize,
    d: usize,
) -> Vec<f32> {
    let q = ref_matmul(x, wq, d, d);
    let k = ref_matmul(x, wk, d, d);
    let v = ref_matmul(x, wv, d, d);
    let attn = ref_paged_step(
        &q,
        &k,
        &v,
        k_cache,
        v_cache,
        gather,
        scatter_slot,
        n_heads,
        head_dim,
    );
    let attn_proj = ref_matmul(&attn, wo, d, d);
    let x1: Vec<f32> = x.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
    let ff_out = ff(&x1);
    x1.iter().zip(&ff_out).map(|(a, b)| a + b).collect()
}

/// GQA paged-attention reference, one new token: query head h reads
/// KV head h / (n_heads / n_kv_heads).
#[allow(clippy::too_many_arguments)]
pub fn ref_paged_step_gqa(
    q: &[f32],
    k_new: &[f32],
    v_new: &[f32],
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    gather: &[usize],
    scatter_slot: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    q_pos: usize,
    window: Option<usize>,
    score_scale: f32,
) -> Vec<f32> {
    let kv_dim = n_kv_heads * head_dim;
    let kv_groups = n_heads / n_kv_heads;
    k_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(k_new);
    v_cache[scatter_slot * kv_dim..(scatter_slot + 1) * kv_dim].copy_from_slice(v_new);
    let scale = score_scale;
    let mut out = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_head = h / kv_groups;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let scores: Vec<f32> = gather
            .iter()
            .enumerate()
            .map(|(position, slot)| {
                // Sliding window (gemma local layers): gathered position
                // j is outside when j < q_pos − (window − 1) — mirror of
                // the graph-side mask in paged_attention_windowed.
                if let Some(window) = window {
                    if (position as i64) < q_pos as i64 - (window as i64 - 1) {
                        return f32::NEG_INFINITY;
                    }
                }
                let k_row = &k_cache[slot * kv_dim + kv_head * head_dim..][..head_dim];
                q_h.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for (j, slot) in gather.iter().enumerate() {
            let v_row = &v_cache[slot * kv_dim + kv_head * head_dim..][..head_dim];
            for (dim, v) in v_row.iter().enumerate() {
                out[h * head_dim + dim] += exps[j] / denom * v;
            }
        }
    }
    out
}

/// RMSNorm: x / sqrt(mean(x²) + eps) — no mean subtraction.
pub fn ref_rms_norm(x: &[f32], epsilon: f32) -> Vec<f32> {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + epsilon).sqrt();
    x.iter().map(|v| v * inv).collect()
}

pub fn ref_silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

pub fn ref_gelu_tanh(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|v| {
            let scaled = 1.595_769_2 * v * (1.0 + 0.044715 * v * v);
            v / (1.0 + (-scaled).exp())
        })
        .collect()
}

/// Plain multi-head attention reference: q rows attend over all k/v
/// rows, per head.
pub fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    sq: usize,
    sk: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; sq * d];
    for row in 0..sq {
        for h in 0..n_heads {
            let q_h = &q[row * d + h * head_dim..][..head_dim];
            let scores: Vec<f32> = (0..sk)
                .map(|col| {
                    let k_h = &k[col * d + h * head_dim..][..head_dim];
                    q_h.iter().zip(k_h).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for (col, e) in exps.iter().enumerate() {
                let v_h = &v[col * d + h * head_dim..][..head_dim];
                for (dim, value) in v_h.iter().enumerate() {
                    out[row * d + h * head_dim + dim] += e / denom * value;
                }
            }
        }
    }
    out
}

/// Per-head RMS norm over a flat (groups·head_dim) row with a learned
/// (head_dim,) weight — the QK-norm reference (eps 1e-6, matching
/// `rms_norm_heads`).
pub fn ref_rms_head_norm(x: &[f32], head_dim: usize, w: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for head in x.chunks(head_dim) {
        let ms: f32 = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        out.extend(head.iter().zip(w).map(|(v, weight)| v * inv * weight));
    }
    out
}

/// One llama-block reference step (rms → GQA attention → residual →
/// rms → gated FFN → residual), gate activation supplied; optional
/// Qwen3-style QK-norm weight seeds.
#[allow(clippy::too_many_arguments)]
pub fn ref_llama_block(
    x: &[f32],
    seeds: (usize, usize, usize, usize, usize, usize, usize),
    qk_seeds: Option<(usize, usize)>,
    d: usize,
    ff: usize,
    kv_dim: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    k_cache: &mut [f32],
    v_cache: &mut [f32],
    gather: &[usize],
    scatter_slot: usize,
    gate_act: &dyn Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    let (wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s) = seeds;
    let normed = ref_rms_norm(x, 1e-5);
    let mut q = ref_matmul(&normed, &weights(d * d, wq_s), d, d);
    let mut k_new = ref_matmul(&normed, &weights(d * kv_dim, wk_s), d, kv_dim);
    if let Some((q_seed, k_seed)) = qk_seeds {
        q = ref_rms_head_norm(&q, head_dim, &weights(head_dim, q_seed));
        k_new = ref_rms_head_norm(&k_new, head_dim, &weights(head_dim, k_seed));
    }
    let v_new = ref_matmul(&normed, &weights(d * kv_dim, wv_s), d, kv_dim);
    let attn = ref_paged_step_gqa(
        &q,
        &k_new,
        &v_new,
        k_cache,
        v_cache,
        gather,
        scatter_slot,
        n_heads,
        n_kv,
        head_dim,
        scatter_slot,
        None,
        1.0 / (head_dim as f32).sqrt(),
    );
    let attn_proj = ref_matmul(&attn, &weights(d * d, wo_s), d, d);
    let x1: Vec<f32> = x.iter().zip(&attn_proj).map(|(a, b)| a + b).collect();
    let ff_in = ref_rms_norm(&x1, 1e-5);
    let gate = gate_act(&ref_matmul(&ff_in, &weights(d * ff, gate_s), d, ff));
    let up = ref_matmul(&ff_in, &weights(d * ff, up_s), d, ff);
    let hidden: Vec<f32> = gate.iter().zip(&up).map(|(a, b)| a * b).collect();
    let ffo = ref_matmul(&hidden, &weights(ff * d, down_s), ff, d);
    x1.iter().zip(&ffo).map(|(a, b)| a + b).collect()
}

/// Table-and-matrix rope reference — mirror of `rotary_apply`:
/// out = x ⊙ cos + (x @ R) ⊙ sin, per head, single row.
pub fn ref_rotary_apply(
    x: &[f32],
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
    rot: &[f32],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for head in x.chunks(head_dim) {
        for i in 0..head_dim {
            let rotated: f32 = (0..head_dim)
                .map(|source| head[source] * rot[source * head_dim + i])
                .sum();
            out.push(head[i] * cos[i] + rotated * sin[i]);
        }
    }
    out
}
