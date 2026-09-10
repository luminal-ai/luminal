//! BF16 STORAGE (the serving landing, 2026-09-10): bf16 is a storage
//! dtype this runtime's codegen reads and writes through conversions —
//! never arithmetic on the bits. A KV pool kept in bf16 halves the
//! serving cache's footprint and traffic; these pin the round trip, the
//! bf16 scatter (a pure 16-bit move) and the fused attention over a bf16
//! cache against host references computed with `half`.
#![cfg(feature = "device")]
#![allow(clippy::needless_range_loop)]

use luminal::dtype::{DType, PlanDtype};
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::fused::{PagedAttentionInputs, PagedAttentionSpec, paged_attention};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn run(cx: &Graph, inputs: Vec<(NodeIndex, HostBuffer)>, out: NodeIndex) -> Vec<f32> {
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs.iter().cloned().collect();
    let mut rt = CudaRuntime::load(cx).expect("load");
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .unwrap_or_else(|e| panic!("search: {e:#}"));
    for (id, v) in inputs {
        rt.set_data(id, v);
    }
    rt.execute().expect("execute");
    let (data, binding) = rt.fetch(out).expect("fetch");
    luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout).unwrap()
}

#[test]
fn f32_to_bf16_round_trip_rounds_to_nearest_even() {
    let mut cx = Graph::new();
    let x = cx.tensor(64, DType::F32);
    let out = x.cast(DType::Bf16).cast(DType::F32).output();
    let mut seed = 3u64;
    let values: Vec<f32> = (0..64).map(|_| lcg(&mut seed) * 1000.0).collect();
    let want: Vec<f32> = values
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_f32())
        .collect();
    let got = run(&cx, vec![(x.id, values.into())], out.id);
    assert_eq!(want, got, "bf16 round trip");
}

#[test]
fn bf16_rows_scatter_into_a_bf16_pool_and_sum_in_f32() {
    const SLOTS: usize = 8;
    const WIDTH: usize = 32;
    let mut cx = Graph::new();
    let pool = cx.tensor((SLOTS, WIDTH), DType::Bf16);
    let rows = cx.tensor((2, WIDTH), DType::F32);
    let slots = cx.tensor(2, DType::Int);
    let written = luminal_nn::scatter_rows(rows.cast(DType::Bf16), slots, pool);
    let out = written.cast(DType::F32).sum(0).output();
    let mut seed = 9u64;
    let pool_values: Vec<f32> = (0..SLOTS * WIDTH).map(|_| lcg(&mut seed)).collect();
    let row_values: Vec<f32> = (0..2 * WIDTH).map(|_| lcg(&mut seed) * 4.0).collect();
    let mut expected_pool: Vec<f32> = pool_values
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_f32())
        .collect();
    for (r, slot) in [3usize, 6].iter().enumerate() {
        for c in 0..WIDTH {
            expected_pool[slot * WIDTH + c] =
                half::bf16::from_f32(row_values[r * WIDTH + c]).to_f32();
        }
    }
    let want: Vec<f32> = (0..WIDTH)
        .map(|c| (0..SLOTS).map(|s| expected_pool[s * WIDTH + c]).sum())
        .collect();
    let got = run(
        &cx,
        vec![
            (
                pool.id,
                HostBuffer::new(PlanDtype::Bf16, bf16_bytes(&pool_values)).unwrap(),
            ),
            (rows.id, row_values.into()),
            (slots.id, vec![3i32, 6].into()),
        ],
        out.id,
    );
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(
            (w - g).abs() <= 1e-4 * w.abs().max(1.0),
            "column {i}: {w} vs {g}"
        );
    }
}

#[test]
fn paged_attention_reads_a_bf16_cache() {
    let spec = PagedAttentionSpec {
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        window: 0,
        scale: 0.125,
    };
    const SLOTS: usize = 16;
    let s = 3;
    let slot_table: Vec<i32> = vec![7, 2, 9, 0, 4, 11, 3, 8];
    let qo = vec![0i32, 2, 3];
    let kv = vec![0i32, 5, 8];
    let q_pos = vec![3i32, 4, 2];
    let mut seed = 21u64;
    let q: Vec<f32> = (0..s * spec.heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let kc: Vec<f32> = (0..SLOTS * spec.kv_heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let vc: Vec<f32> = (0..SLOTS * spec.kv_heads * spec.head_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let sinks: Vec<f32> = (0..spec.heads).map(|_| lcg(&mut seed)).collect();
    // The reference sees the cache as the device does: bf16-rounded.
    let round = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .map(|x| half::bf16::from_f32(*x).to_f32())
            .collect()
    };
    let (kc_r, vc_r) = (round(&kc), round(&vc));
    let d = spec.head_dim;
    let (q_dim, kv_dim) = (spec.heads * d, spec.kv_heads * d);
    let mut want = vec![0f32; s * q_dim];
    for i in 0..s {
        let r = (0..qo.len() - 1)
            .find(|&r| qo[r] <= i as i32 && (i as i32) < qo[r + 1])
            .unwrap();
        for h in 0..spec.heads {
            let g = h / (spec.heads / spec.kv_heads);
            let mut scores = Vec::new();
            for j in kv[r] as usize..kv[r + 1] as usize {
                let p = (j - kv[r] as usize) as i32;
                if p > q_pos[i] {
                    continue;
                }
                let slot = slot_table[j] as usize;
                let dot: f32 = (0..d)
                    .map(|dd| q[i * q_dim + h * d + dd] * kc_r[slot * kv_dim + g * d + dd])
                    .sum();
                scores.push((slot, dot * spec.scale as f32));
            }
            let m = scores.iter().map(|(_, sc)| *sc).fold(sinks[h], f32::max);
            let mut denom = (sinks[h] - m).exp();
            let mut acc = vec![0f32; d];
            for (slot, sc) in &scores {
                let p = (sc - m).exp();
                denom += p;
                for dd in 0..d {
                    acc[dd] += p * vc_r[slot * kv_dim + g * d + dd];
                }
            }
            for dd in 0..d {
                want[i * q_dim + h * d + dd] = acc[dd] / denom;
            }
        }
    }
    let mut cx = Graph::new();
    let q_t = cx.tensor((s, q_dim), DType::F32);
    let kc_t = cx.tensor((SLOTS, kv_dim), DType::Bf16);
    let vc_t = cx.tensor((SLOTS, kv_dim), DType::Bf16);
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
            (
                kc_t.id,
                HostBuffer::new(PlanDtype::Bf16, bf16_bytes(&kc)).unwrap(),
            ),
            (
                vc_t.id,
                HostBuffer::new(PlanDtype::Bf16, bf16_bytes(&vc)).unwrap(),
            ),
            (st_t.id, slot_table.into()),
            (qo_t.id, qo.into()),
            (kv_t.id, kv.into()),
            (qp_t.id, q_pos.into()),
            (sk_t.id, sinks.into()),
        ],
        out.id,
    );
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(
            (w - g).abs() <= 1e-4 * w.abs().max(1.0),
            "element {i}: {w} vs {g}"
        );
    }
}
