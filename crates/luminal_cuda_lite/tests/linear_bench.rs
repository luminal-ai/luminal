//! bf16 dense linear timing at gpt-oss shapes (ignored — a probe):
//!
//!   cargo test --release -p luminal_cuda_lite --features device \
//!       --test linear_bench -- --ignored --nocapture
#![cfg(feature = "device")]

use luminal::dtype::{DType, PlanDtype};
use luminal::graph::Graph;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::fused::linear_bf16;
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

#[test]
#[ignore]
fn linear_bf16_gpt_oss_shapes() {
    let mut seed = 3u64;
    for (name, n, k) in [
        ("q_proj", 4096usize, 2880usize),
        ("o_proj", 2880, 4096),
        ("lm_head", 201088, 2880),
    ] {
        let w: Vec<u8> = (0..n * k)
            .flat_map(|_| half::bf16::from_f32(lcg(&mut seed)).to_bits().to_le_bytes())
            .collect();
        let bias: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
        for s in [1usize, 4, 8, 16, 64, 256, 1024] {
            if name == "lm_head" && s > 64 {
                continue;
            }
            let x: Vec<f32> = (0..s * k).map(|_| lcg(&mut seed)).collect();
            let mut cx = Graph::new();
            let x_t = cx.tensor((s, k), DType::F32);
            let w_t = cx.tensor((n, k), DType::Bf16);
            let b_t = cx.tensor(n, DType::F32);
            let out = linear_bf16(x_t, w_t, b_t).output();
            let inputs: Vec<(_, HostBuffer)> = vec![
                (x_t.id, x.into()),
                (w_t.id, HostBuffer::new(PlanDtype::Bf16, w.clone()).unwrap()),
                (b_t.id, bias.clone().into()),
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
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                rt.execute().expect("execute");
                let _ = rt.fetch(out.id).expect("fetch");
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let gb = (n * k * 2) as f64 / 1e9;
            println!(
                "{name:>8} s {s:>5}: {ms:>8.3} ms (weights {gb:.2} GB -> {:.2} TB/s incl. fetch)",
                gb / (ms * 1e-3) / 1e3
            );
        }
    }
}
