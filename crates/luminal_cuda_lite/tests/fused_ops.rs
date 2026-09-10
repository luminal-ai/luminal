//! THE FUSED SERVING OPS (`device` feature only): the extern logical ops
//! this runtime implements — paged attention with sinks/window, and the
//! two MXFP4 MoE halves — saturate, match, extract and execute through
//! the ordinary ladder, and their kernels agree with scalar host
//! references. Plus the serving graph's shape trick: fixed-capacity
//! per-tick inputs shrunk to a bucketed `s`.
#![cfg(feature = "device")]
#![allow(clippy::needless_range_loop)]

use luminal::dtype::{DType, PlanDtype};
use luminal::graph::{DimBucket, Graph};
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::fused::{
    DownSpec, GateUpSpec, Mxfp4Experts, PagedAttentionInputs, PagedAttentionSpec, argmax_rows,
    moe_down_mxfp4, moe_gate_up_mxfp4, moe_topk_ids, paged_attention, rms_norm, rope_split_half,
    take_rows,
};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn run(cx: &Graph, inputs: Vec<(NodeIndex, HostBuffer)>, out: NodeIndex) -> Vec<f32> {
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs.iter().cloned().collect();
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let outcome = rt
        .search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("cuda search: {e:#}"));
    let b = &outcome.refusal_breakdown;
    assert_eq!(
        (
            b.extract_refusals,
            b.plan_build_refusals,
            b.execute_refusals
        ),
        (0, 0, 0),
        "refusals: {}",
        b.summary()
    );
    for (id, v) in inputs {
        rt.set_data(id, v);
    }
    rt.execute().expect("device execute");
    let (data, binding) = rt.fetch(out).expect("fetch");
    luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap()
}

fn assert_close(want: &[f32], got: &[f32], what: &str, tol: f32) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let t = tol.max(w.abs() * tol);
        assert!(
            (w - g).abs() <= t,
            "{what}: element {i} diverges — reference {w} vs device {g}"
        );
    }
}

/// Scalar paged attention: the op's contract, spelled the slow way.
#[allow(clippy::too_many_arguments)]
fn reference_attention(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    slot_table: &[i32],
    qo: &[i32],
    kv: &[i32],
    q_pos: &[i32],
    sinks: &[f32],
    spec: PagedAttentionSpec,
    s: usize,
) -> Vec<f32> {
    let group = spec.heads / spec.kv_heads;
    let d = spec.head_dim;
    let q_dim = spec.heads * d;
    let kv_dim = spec.kv_heads * d;
    let mut out = vec![0f32; s * q_dim];
    for i in 0..s {
        let r = (0..qo.len() - 1)
            .find(|&r| qo[r] <= i as i32 && (i as i32) < qo[r + 1])
            .expect("every query has a request");
        let (start, end) = (kv[r] as usize, kv[r + 1] as usize);
        for h in 0..spec.heads {
            let g = h / group;
            let mut scores = Vec::new();
            for j in start..end {
                let p = (j - start) as i32;
                let ok = p <= q_pos[i] && (spec.window == 0 || p > q_pos[i] - spec.window as i32);
                if !ok {
                    continue;
                }
                let slot = slot_table[j] as usize;
                let mut dot = 0f32;
                for dd in 0..d {
                    dot += q[i * q_dim + h * d + dd] * kc[slot * kv_dim + g * d + dd];
                }
                scores.push((slot, dot * spec.scale as f32));
            }
            let m = scores.iter().map(|(_, sc)| *sc).fold(sinks[h], f32::max);
            let mut denom = (sinks[h] - m).exp();
            let mut acc = vec![0f32; d];
            for (slot, sc) in &scores {
                let p = (sc - m).exp();
                denom += p;
                for dd in 0..d {
                    acc[dd] += p * vc[slot * kv_dim + g * d + dd];
                }
            }
            for dd in 0..d {
                out[i * q_dim + h * d + dd] = acc[dd] / denom;
            }
        }
    }
    out
}

fn attention_case(window: usize) {
    let spec = PagedAttentionSpec {
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        window,
        scale: 1.0 / (32f32).sqrt() as f64,
    };
    const SLOTS: usize = 16;
    // Two requests: request 0 has 5 context rows with 2 queries at
    // positions 3,4; request 1 has 3 context rows with 1 query at 2.
    let s = 3;
    let slot_table: Vec<i32> = vec![7, 2, 9, 0, 4, 11, 3, 8];
    let qo = vec![0i32, 2, 3];
    let kv = vec![0i32, 5, 8];
    let q_pos = vec![3i32, 4, 2];
    let mut seed = 7u64;
    let q: Vec<f32> = (0..s * spec.heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let kc: Vec<f32> = (0..SLOTS * spec.kv_heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let vc: Vec<f32> = (0..SLOTS * spec.kv_heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let sinks: Vec<f32> = (0..spec.heads).map(|_| lcg(&mut seed) * 2.0).collect();
    let want = reference_attention(&q, &kc, &vc, &slot_table, &qo, &kv, &q_pos, &sinks, spec, s);

    let mut cx = Graph::new();
    let q_t = cx.tensor((s, spec.heads * spec.head_dim), DType::F32);
    let kc_t = cx.tensor((SLOTS, spec.kv_heads * spec.head_dim), DType::F32);
    let vc_t = cx.tensor((SLOTS, spec.kv_heads * spec.head_dim), DType::F32);
    let st_t = cx.tensor(slot_table.len(), DType::Int);
    let qo_t = cx.tensor(qo.len(), DType::Int);
    let kv_t = cx.tensor(kv.len(), DType::Int);
    let qp_t = cx.tensor(s, DType::Int);
    let sk_t = cx.tensor(spec.heads, DType::F32);
    let out = paged_attention(
        PagedAttentionInputs {
            q: q_t,
            k_cache: kc_t,
            v_cache: vc_t,
            slot_table: st_t,
            qo_indptr: qo_t,
            kv_indptr: kv_t,
            q_pos: qp_t,
            sinks: sk_t,
        },
        spec,
    )
    .output();
    let got = run(
        &cx,
        vec![
            (q_t.id, q.into()),
            (kc_t.id, kc.into()),
            (vc_t.id, vc.into()),
            (st_t.id, slot_table.into()),
            (qo_t.id, qo.into()),
            (kv_t.id, kv.into()),
            (qp_t.id, q_pos.into()),
            (sk_t.id, sinks.into()),
        ],
        out.id,
    );
    assert_close(
        &want,
        &got,
        &format!("paged attention (window {window})"),
        1e-4,
    );
}

#[test]
fn paged_attention_full_matches_reference() {
    attention_case(0);
}

#[test]
fn paged_attention_sliding_window_matches_reference() {
    attention_case(2);
}

const FP4: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Dequantize one packed row: `blocks[k/2]` nibble pairs, `scales[k/32]`.
fn dequant_row(blocks: &[u8], scales: &[u8], k: usize) -> Vec<f32> {
    (0..k)
        .map(|c| {
            let byte = blocks[c / 2];
            let nibble = if c % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let scale = f32::from_bits(u32::from(scales[c / 32]) << 23);
            FP4[nibble as usize] * scale
        })
        .collect()
}

fn bf16_bits(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// Warp-per-route GEMV path (few routes, tile-unfriendly dims).
#[test]
fn moe_mxfp4_matches_reference() {
    moe_reference_case(64, 32, 3, 2, 3);
}

/// Expert-major tensor-core path: 128+ routes over tile-aligned dims
/// (n % 160 == 0, k % 64 == 0), with more routes to one expert than an
/// M tile holds and an expert nobody routes to.
#[test]
fn moe_mxfp4_tensor_core_matches_reference() {
    moe_reference_case(320, 320, 3, 2, 64);
    moe_reference_case(320, 320, 3, 2, 100);
}

fn moe_reference_case(hidden: usize, inter: usize, experts: usize, top_k: usize, s: usize) {
    #![allow(non_snake_case)]
    let (HIDDEN, INTER, EXPERTS, TOP_K, S) = (hidden, inter, experts, top_k, s);
    const ALPHA: f32 = 1.702;
    const LIMIT: f32 = 7.0;
    let mut seed = 11u64 + s as u64;
    let x: Vec<f32> = (0..S * HIDDEN).map(|_| lcg(&mut seed)).collect();
    // Routes: expert 2 is never chosen once s > 3 (a zero-route expert
    // for the grouped kernel); expert 0 takes most of the rest.
    let ids: Vec<i32> = (0..S * TOP_K)
        .map(|p| {
            if S <= 3 {
                [0, 2, 1, 1, 2, 0][p]
            } else if p % 5 == 0 {
                1
            } else {
                0
            }
        })
        .collect();
    // The router's raw logits; the down half softmaxes each token's
    // SELECTED ones into its route weights.
    let logits: Vec<f32> = (0..S * EXPERTS).map(|_| lcg(&mut seed) * 3.0).collect();
    let mut weights = vec![0f32; S * TOP_K];
    for t in 0..S {
        let picked: Vec<f32> = (0..TOP_K)
            .map(|kk| logits[t * EXPERTS + ids[t * TOP_K + kk] as usize])
            .collect();
        let m = picked.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let denom: f32 = picked.iter().map(|v| (v - m).exp()).sum();
        for kk in 0..TOP_K {
            weights[t * TOP_K + kk] = (picked[kk] - m).exp() / denom;
        }
    }
    let byte = |s: &mut u64| ((lcg(s) + 1.0) * 127.5) as u8;
    let gu_blocks: Vec<u8> = (0..EXPERTS * 2 * INTER * HIDDEN / 2)
        .map(|_| byte(&mut seed))
        .collect();
    // e8m0 scales around 2^-3 .. 2^2 so products stay O(1).
    let gu_scales: Vec<u8> = (0..EXPERTS * 2 * INTER * HIDDEN / 32)
        .map(|_| 124 + (byte(&mut seed) % 6))
        .collect();
    let gu_bias_f: Vec<f32> = (0..EXPERTS * 2 * INTER).map(|_| lcg(&mut seed)).collect();
    let dn_blocks: Vec<u8> = (0..EXPERTS * HIDDEN * INTER / 2)
        .map(|_| byte(&mut seed))
        .collect();
    let dn_scales: Vec<u8> = (0..EXPERTS * HIDDEN * INTER / 32)
        .map(|_| 124 + (byte(&mut seed) % 6))
        .collect();
    let dn_bias_f: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| lcg(&mut seed)).collect();
    let gu_bias: Vec<u8> = gu_bias_f
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect();
    let dn_bias: Vec<u8> = dn_bias_f
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect();
    let bf = |v: f32| half::bf16::from_f32(v).to_f32();
    // The tensor-core path (64+ routes over tile-aligned dims) feeds the
    // MMA bf16 activations; the reference rounds at the same two points.
    let tensor_path = S * TOP_K >= 128
        && (2 * INTER).is_multiple_of(160)
        && HIDDEN.is_multiple_of(160)
        && HIDDEN.is_multiple_of(64)
        && INTER.is_multiple_of(64);
    let act = |v: f32| if tensor_path { bf(v) } else { v };

    // Reference.
    let mut want = vec![0f32; S * HIDDEN];
    for t in 0..S {
        for kk in 0..TOP_K {
            let e = ids[t * TOP_K + kk] as usize;
            let mut hidden = vec![0f32; INTER];
            for j in 0..INTER {
                let mut gu = [0f32; 2];
                for (which, g) in gu.iter_mut().enumerate() {
                    let row = 2 * j + which;
                    let w = dequant_row(
                        &gu_blocks[(e * 2 * INTER + row) * HIDDEN / 2..][..HIDDEN / 2],
                        &gu_scales[(e * 2 * INTER + row) * HIDDEN / 32..][..HIDDEN / 32],
                        HIDDEN,
                    );
                    *g = (0..HIDDEN)
                        .map(|c| w[c] * act(x[t * HIDDEN + c]))
                        .sum::<f32>()
                        + bf(gu_bias_f[e * 2 * INTER + row]);
                }
                let gate = gu[0].min(LIMIT);
                let up = gu[1].clamp(-LIMIT, LIMIT);
                let sig = 1.0 / (1.0 + (-ALPHA * gate).exp());
                hidden[j] = (up + 1.0) * gate * sig;
            }
            for r in 0..HIDDEN {
                let w = dequant_row(
                    &dn_blocks[(e * HIDDEN + r) * INTER / 2..][..INTER / 2],
                    &dn_scales[(e * HIDDEN + r) * INTER / 32..][..INTER / 32],
                    INTER,
                );
                let dot = (0..INTER).map(|c| w[c] * act(hidden[c])).sum::<f32>()
                    + bf(dn_bias_f[e * HIDDEN + r]);
                want[t * HIDDEN + r] += weights[t * TOP_K + kk] * dot;
            }
        }
    }

    let mut cx = Graph::new();
    let x_t = cx.tensor((S, HIDDEN), DType::F32);
    let ids_t = cx.tensor((S, TOP_K), DType::Int);
    let w_t = cx.tensor((S, EXPERTS), DType::F32);
    let gu = Mxfp4Experts {
        blocks: cx.tensor((EXPERTS, 2 * INTER, HIDDEN / 2), DType::U8),
        scales: cx.tensor((EXPERTS, 2 * INTER, HIDDEN / 32), DType::F8UE8M0),
        bias: cx.tensor((EXPERTS, 2 * INTER), DType::Bf16),
    };
    let dn = Mxfp4Experts {
        blocks: cx.tensor((EXPERTS, HIDDEN, INTER / 2), DType::U8),
        scales: cx.tensor((EXPERTS, HIDDEN, INTER / 32), DType::F8UE8M0),
        bias: cx.tensor((EXPERTS, HIDDEN), DType::Bf16),
    };
    let hidden = moe_gate_up_mxfp4(
        x_t,
        ids_t,
        gu,
        GateUpSpec {
            inter: INTER,
            top_k: TOP_K,
            alpha: ALPHA as f64,
            limit: LIMIT as f64,
        },
    );
    let out = moe_down_mxfp4(
        hidden,
        ids_t,
        w_t,
        dn,
        DownSpec {
            hidden: HIDDEN,
            top_k: TOP_K,
        },
    )
    .output();
    let got = run(
        &cx,
        vec![
            (x_t.id, x.into()),
            (ids_t.id, ids.into()),
            (w_t.id, logits.into()),
            (
                gu.blocks.id,
                HostBuffer::new(PlanDtype::U8, gu_blocks).unwrap(),
            ),
            (
                gu.scales.id,
                HostBuffer::new(PlanDtype::F8UE8M0, gu_scales).unwrap(),
            ),
            (
                gu.bias.id,
                HostBuffer::new(PlanDtype::Bf16, gu_bias).unwrap(),
            ),
            (
                dn.blocks.id,
                HostBuffer::new(PlanDtype::U8, dn_blocks).unwrap(),
            ),
            (
                dn.scales.id,
                HostBuffer::new(PlanDtype::F8UE8M0, dn_scales).unwrap(),
            ),
            (
                dn.bias.id,
                HostBuffer::new(PlanDtype::Bf16, dn_bias).unwrap(),
            ),
        ],
        out.id,
    );
    assert_close(&want, &got, "moe mxfp4", 2e-3);
}

/// THE SERVING SHAPE TRICK: per-tick inputs are declared at a fixed
/// CAPACITY and shrunk to the bucketed `s`, so ONE staged data map fits
/// every bucket's plan; `set_dim` picks the plan and the executor runs
/// it at the representative.
#[test]
fn fixed_capacity_inputs_shrink_to_bucketed_rows() {
    const CAP: usize = 8;
    let mut cx = Graph::new();
    let tokens = cx.tensor((CAP, 2), DType::F32);
    let weight = cx.tensor(2, DType::F32);
    let rows = take_rows(tokens, 's');
    let out = (rows * weight.expand_lhs([luminal::shape::IntExpr::from('s')])).output();

    let data: Vec<f32> = (0..CAP * 2).map(|i| i as f32).collect();
    let w = vec![2.0f32, 3.0];
    let staged: FxHashMap<NodeIndex, HostBuffer> = [
        (tokens.id, HostBuffer::from(data.clone())),
        (weight.id, HostBuffer::from(w.clone())),
    ]
    .into_iter()
    .collect();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    rt.bind_dim_buckets(
        's',
        vec![
            DimBucket::new(1, 4).representative(4),
            DimBucket::new(5, 8).representative(8),
        ],
    )
    .expect("buckets bind");
    let options = luminal_cuda_lite::CompileOptions {
        profile_on_device: true,
        ..luminal_cuda_lite::harness_search_options()
    };
    rt.search(&staged, &options)
        .unwrap_or_else(|e| panic!("bucketed search: {e:#}"));
    assert_eq!(rt.bucket_plans().len(), 2);
    rt.set_data(tokens.id, data.clone());
    rt.set_data(weight.id, w.clone());
    for s in [4usize, 8] {
        rt.set_dim('s', s);
        rt.execute().expect("execute at the representative");
        let got = rt.get_f32(out.id).expect("out");
        let want: Vec<f32> = (0..s * 2).map(|i| data[i] * w[i % 2]).collect();
        assert_close(&want, &got, &format!("bucket s={s}"), 1e-6);
    }
}

fn run_i32(cx: &Graph, inputs: Vec<(NodeIndex, HostBuffer)>, out: NodeIndex) -> Vec<i32> {
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs.iter().cloned().collect();
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("cuda search: {e:#}"));
    for (id, v) in inputs {
        rt.set_data(id, v);
    }
    rt.execute().expect("device execute");
    let (data, binding) = rt.fetch(out).expect("fetch");
    assert!(
        binding
            .layout
            .has::<luminal::layouts::RightMajorContiguousElementLayout>(),
        "int output must come back dense"
    );
    data.as_i32().unwrap()
}

/// The decode launch diet's norm: one launch, `luminal_nn::rms_norm`
/// semantics.
#[test]
fn rms_norm_matches_reference() {
    const S: usize = 5;
    const W: usize = 96;
    const EPS: f32 = 1e-5;
    let mut seed = 31u64;
    let x: Vec<f32> = (0..S * W).map(|_| lcg(&mut seed) * 4.0).collect();
    let w: Vec<f32> = (0..W).map(|_| lcg(&mut seed) + 1.5).collect();
    let mut want = vec![0f32; S * W];
    for t in 0..S {
        let mean: f32 = x[t * W..][..W].iter().map(|v| v * v).sum::<f32>() / W as f32;
        let inv = 1.0 / (mean + EPS).sqrt();
        for i in 0..W {
            want[t * W + i] = x[t * W + i] * inv * w[i];
        }
    }
    let mut cx = Graph::new();
    let x_t = cx.tensor((S, W), DType::F32);
    let w_t = cx.tensor(W, DType::F32);
    let out = rms_norm(x_t, w_t, EPS).output();
    let got = run(&cx, vec![(x_t.id, x.into()), (w_t.id, w.into())], out.id);
    assert_close(&want, &got, "rms_norm", 1e-5);
}

/// The decode launch diet's rotary embedding: `rotary_apply` with the
/// split-half pairing, f32 out and bf16 out.
#[test]
fn rope_split_half_matches_reference() {
    const S: usize = 3;
    const HEADS: usize = 2;
    const HD: usize = 8;
    let mut seed = 41u64;
    let x: Vec<f32> = (0..S * HEADS * HD).map(|_| lcg(&mut seed)).collect();
    let cos: Vec<f32> = (0..S * HD).map(|_| lcg(&mut seed)).collect();
    let sin: Vec<f32> = (0..S * HD).map(|_| lcg(&mut seed)).collect();
    let mut want = vec![0f32; S * HEADS * HD];
    for t in 0..S {
        for h in 0..HEADS {
            for d in 0..HD {
                let base = t * HEADS * HD + h * HD;
                let rot = if d < HD / 2 {
                    -x[base + d + HD / 2]
                } else {
                    x[base + d - HD / 2]
                };
                want[base + d] = x[base + d] * cos[t * HD + d] + rot * sin[t * HD + d];
            }
        }
    }
    for out_dtype in [DType::F32, DType::Bf16] {
        let mut cx = Graph::new();
        let x_t = cx.tensor((S, HEADS * HD), DType::F32);
        let c_t = cx.tensor((S, HD), DType::F32);
        let s_t = cx.tensor((S, HD), DType::F32);
        let rope = rope_split_half(x_t, c_t, s_t, HD, out_dtype);
        // Read back through f32 either way (a bf16 result widens).
        let out = rope.cast(DType::F32).output();
        let got = run(
            &cx,
            vec![
                (x_t.id, x.clone().into()),
                (c_t.id, cos.clone().into()),
                (s_t.id, sin.clone().into()),
            ],
            out.id,
        );
        let want_here: Vec<f32> = if out_dtype == DType::Bf16 {
            want.iter()
                .map(|v| half::bf16::from_f32(*v).to_f32())
                .collect()
        } else {
            want.clone()
        };
        assert_close(&want_here, &got, &format!("rope {out_dtype:?}"), 1e-6);
    }
}

/// The decode launch diet's router: the stable descending argsort's
/// first k, ties to the lower index — including exact ties.
#[test]
fn moe_topk_ids_match_stable_argsort() {
    const S: usize = 6;
    const E: usize = 128;
    const K: usize = 4;
    let mut seed = 51u64;
    let mut logits: Vec<f32> = (0..S * E).map(|_| (lcg(&mut seed) * 8.0).round()).collect();
    // Row 5: a fully tied row selects 0, 1, 2, 3.
    for e in 0..E {
        logits[5 * E + e] = 2.0;
    }
    let mut want = vec![0i32; S * K];
    for t in 0..S {
        let mut order: Vec<usize> = (0..E).collect();
        order.sort_by(|&a, &b| {
            logits[t * E + b]
                .partial_cmp(&logits[t * E + a])
                .unwrap()
                .then(a.cmp(&b))
        });
        for kk in 0..K {
            want[t * K + kk] = order[kk] as i32;
        }
    }
    let mut cx = Graph::new();
    let l_t = cx.tensor((S, E), DType::F32);
    let out = moe_topk_ids(l_t, K).output();
    let got = run_i32(&cx, vec![(l_t.id, logits.clone().into())], out.id);
    assert_eq!(want, got, "top-k ids");
    // And the same rows through the decomposed spelling agree.
    let mut cx2 = Graph::new();
    let l2 = cx2.tensor((S, E), DType::F32);
    let out2 = l2.topk_indexes(K, 1).output();
    let spelled = run_i32(&cx2, vec![(l2.id, logits.into())], out2.id);
    assert_eq!(want, spelled, "topk_indexes spelling");
}

/// The decode launch diet's greedy sampler: `argmax(1)` semantics over a
/// wide row, ties to the higher index.
#[test]
fn argmax_rows_matches_reference() {
    const N: usize = 4;
    const W: usize = 3000;
    let mut seed = 61u64;
    let mut x: Vec<f32> = (0..N * W).map(|_| lcg(&mut seed)).collect();
    // Row 1: the maximum appears twice; the higher index wins.
    x[W + 17] = 5.0;
    x[W + 2900] = 5.0;
    // Row 3: the maximum is the last element.
    x[3 * W + W - 1] = 9.0;
    let mut want = vec![0i32; N];
    for t in 0..N {
        let row = &x[t * W..][..W];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        want[t] = row.iter().rposition(|v| *v == m).unwrap() as i32;
    }
    let mut cx = Graph::new();
    let x_t = cx.tensor((N, W), DType::F32);
    let out = argmax_rows(x_t).output();
    let got = run_i32(&cx, vec![(x_t.id, x.into())], out.id);
    assert_eq!(want, got, "argmax rows");
}

/// `set_data_prefix`: a capacity-sized input whose first bytes changed
/// is refreshed on the device in that prefix only; the graph reads just
/// the bucket's rows, so the stale tail is never observed.
#[test]
fn prefix_uploads_refresh_only_the_bucket_rows() {
    const CAP: usize = 64;
    let mut cx = Graph::new();
    let tokens = cx.tensor(CAP, DType::F32);
    let rows = take_rows(tokens, 's');
    let out = (rows * 3.0).output();
    let first: Vec<f32> = (0..CAP).map(|v| v as f32).collect();
    let data: FxHashMap<NodeIndex, HostBuffer> =
        std::iter::once((tokens.id, HostBuffer::from(first.clone()))).collect();
    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let s_bucket = DimBucket::new(1, 8).representative(8);
    rt.bind_dim_buckets('s', vec![s_bucket]).expect("bucket");
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("cuda search: {e:#}"));
    rt.set_dim('s', 8);
    rt.set_data(tokens.id, first);
    rt.execute().expect("execute");
    let (got, _) = rt.fetch(out.id).expect("fetch");
    assert_eq!(
        got.as_f32().unwrap(),
        (0..8).map(|v| v as f32 * 3.0).collect::<Vec<_>>()
    );
    // Only the first 8 entries are declared changed; the tail is garbage
    // the device never receives.
    let mut second: Vec<f32> = vec![-1000.0; CAP];
    for (i, v) in second.iter_mut().enumerate().take(8) {
        *v = 100.0 + i as f32;
    }
    rt.set_data_prefix(tokens.id, second, 8 * std::mem::size_of::<f32>());
    rt.execute().expect("execute");
    let (got, _) = rt.fetch(out.id).expect("fetch");
    assert_eq!(
        got.as_f32().unwrap(),
        (0..8).map(|v| (100.0 + v as f32) * 3.0).collect::<Vec<_>>()
    );
}

/// The bf16 dense linear: the GEMV path (few rows) and the tensor-core
/// path (many rows, both column tiles) against a host reference with
/// bf16-rounded activations where the MMA rounds them.
#[test]
fn linear_bf16_matches_reference() {
    for (s, n, k) in [
        (3usize, 96usize, 64usize),
        (40, 192, 192),
        (100, 256, 128),
        (8, 130, 24),
    ] {
        let mut seed = 71u64 + s as u64;
        let x: Vec<f32> = (0..s * k).map(|_| lcg(&mut seed)).collect();
        let w_f: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
        let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        let bf = |v: f32| half::bf16::from_f32(v).to_f32();
        let tensor_path = s > 8;
        let act = |v: f32| if tensor_path { bf(v) } else { v };
        let mut want = vec![0f32; s * n];
        for t in 0..s {
            for j in 0..n {
                let dot: f32 = (0..k).map(|c| act(x[t * k + c]) * bf(w_f[j * k + c])).sum();
                want[t * n + j] = dot + bias[j];
            }
        }
        let mut cx = Graph::new();
        let x_t = cx.tensor((s, k), DType::F32);
        let w_t = cx.tensor((n, k), DType::Bf16);
        let b_t = cx.tensor(n, DType::F32);
        let out = luminal_cuda_lite::fused::linear_bf16(x_t, w_t, b_t).output();
        let w_bytes: Vec<u8> = w_f
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        let got = run(
            &cx,
            vec![
                (x_t.id, x.into()),
                (w_t.id, HostBuffer::new(PlanDtype::Bf16, w_bytes).unwrap()),
                (b_t.id, bias.into()),
            ],
            out.id,
        );
        assert_close(&want, &got, &format!("linear_bf16 s{s} n{n} k{k}"), 1e-3);
    }
}
