//! MXFP4 MoE kernel timing at gpt-oss geometry (ignored by default —
//! a development probe, not a check):
//!
//!   cargo test --release -p luminal_cuda_lite --features device \
//!       --test moe_bench -- --ignored --nocapture
#![cfg(feature = "device")]

use luminal::dtype::{DType, PlanDtype};
use luminal::graph::Graph;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::fused::{
    DownSpec, GateUpSpec, Mxfp4Experts, moe_down_mxfp4, moe_gate_up_mxfp4, moe_topk_ids,
};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

#[test]
#[ignore]
fn moe_gpt_oss_geometry_timing() {
    const HIDDEN: usize = 2880;
    const INTER: usize = 2880;
    const EXPERTS: usize = 128;
    const TOP_K: usize = 4;
    let mut seed = 7u64;
    let byte = |s: &mut u64| ((lcg(s) + 1.0) * 127.5) as u8;
    let gu_blocks: Vec<u8> = (0..EXPERTS * 2 * INTER * HIDDEN / 2)
        .map(|_| byte(&mut seed))
        .collect();
    let gu_scales: Vec<u8> = (0..EXPERTS * 2 * INTER * HIDDEN / 32)
        .map(|_| 120 + (byte(&mut seed) % 6))
        .collect();
    let gu_bias: Vec<u8> = vec![0u8; EXPERTS * 2 * INTER * 2];
    let dn_blocks: Vec<u8> = (0..EXPERTS * HIDDEN * INTER / 2)
        .map(|_| byte(&mut seed))
        .collect();
    let dn_scales: Vec<u8> = (0..EXPERTS * HIDDEN * INTER / 32)
        .map(|_| 120 + (byte(&mut seed) % 6))
        .collect();
    let dn_bias: Vec<u8> = vec![0u8; EXPERTS * HIDDEN * 2];
    for s in [1usize, 4, 16, 64, 256, 1024] {
        let x: Vec<f32> = (0..s * HIDDEN).map(|_| lcg(&mut seed)).collect();
        let logits: Vec<f32> = (0..s * EXPERTS).map(|_| lcg(&mut seed) * 3.0).collect();
        let mut cx = Graph::new();
        let x_t = cx.tensor((s, HIDDEN), DType::F32);
        let l_t = cx.tensor((s, EXPERTS), DType::F32);
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
        let ids = moe_topk_ids(l_t, TOP_K);
        let hidden = moe_gate_up_mxfp4(
            x_t,
            ids,
            gu,
            GateUpSpec {
                inter: INTER,
                top_k: TOP_K,
                alpha: 1.702,
                limit: 7.0,
            },
        );
        let out = moe_down_mxfp4(
            hidden,
            ids,
            l_t,
            dn,
            DownSpec {
                hidden: HIDDEN,
                top_k: TOP_K,
            },
        )
        .output();
        let inputs: Vec<(_, HostBuffer)> = vec![
            (x_t.id, x.into()),
            (l_t.id, logits.into()),
            (
                gu.blocks.id,
                HostBuffer::new(PlanDtype::U8, gu_blocks.clone()).unwrap(),
            ),
            (
                gu.scales.id,
                HostBuffer::new(PlanDtype::F8UE8M0, gu_scales.clone()).unwrap(),
            ),
            (
                gu.bias.id,
                HostBuffer::new(PlanDtype::Bf16, gu_bias.clone()).unwrap(),
            ),
            (
                dn.blocks.id,
                HostBuffer::new(PlanDtype::U8, dn_blocks.clone()).unwrap(),
            ),
            (
                dn.scales.id,
                HostBuffer::new(PlanDtype::F8UE8M0, dn_scales.clone()).unwrap(),
            ),
            (
                dn.bias.id,
                HostBuffer::new(PlanDtype::Bf16, dn_bias.clone()).unwrap(),
            ),
        ];
        let data: FxHashMap<_, _> = inputs.iter().cloned().collect();
        let mut rt = CudaRuntime::load(&cx).expect("load");
        rt.search(&data, &luminal_cuda_lite::harness_search_options())
            .unwrap_or_else(|e| panic!("search: {e:#}"));
        for (id, v) in inputs {
            rt.set_data(id, v);
        }
        for _ in 0..3 {
            rt.execute().expect("warmup");
        }
        let _ = rt.fetch(out.id);
        let iters = 10;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rt.execute().expect("execute");
        }
        let _ = rt.fetch(out.id).expect("fetch");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        // Expert traffic if every routed expert were read once per token.
        let bytes_per_pair = (2 * INTER * HIDDEN / 2 + HIDDEN * INTER / 2) as f64;
        println!(
            "s {s:>5}: {ms:>9.3} ms per tick (topk + gate_up + down + sum); \
             pair-major traffic {:.1} GB, per-tick effective {:.2} TB/s",
            s as f64 * TOP_K as f64 * bytes_per_pair / 1e9,
            s as f64 * TOP_K as f64 * bytes_per_pair / (ms * 1e-3) / 1e12
        );
    }
}
