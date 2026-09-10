//! Paged attention timing at gpt-oss geometry over long decode contexts
//! (ignored by default — a development probe):
//!
//!   cargo test --release -p luminal_cuda_lite --features device \
//!       --test attention_bench -- --ignored --nocapture
#![cfg(feature = "device")]

use luminal::dtype::{DType, PlanDtype};
use luminal::graph::Graph;
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::fused::{PagedAttentionInputs, PagedAttentionSpec, paged_attention};
use luminal_cuda_lite::{CudaRuntime, HostBuffer};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

#[test]
#[ignore]
fn paged_attention_decode_context_timing() {
    let spec = PagedAttentionSpec {
        heads: 64,
        kv_heads: 8,
        head_dim: 64,
        window: 0,
        scale: 0.125,
    };
    const SLOTS: usize = 8192;
    let kv_dim = spec.kv_heads * spec.head_dim;
    let mut seed = 5u64;
    let pool: Vec<u8> = (0..SLOTS * kv_dim)
        .flat_map(|_| half::bf16::from_f32(lcg(&mut seed)).to_bits().to_le_bytes())
        .collect();
    // (s, context rows, visible position, window): the last cases are the
    // serving warmup's shape — one query at position 0 over a 1024-row
    // context — for the full and the windowed kernel.
    for (s, context, qpos, window) in [
        (1usize, 128usize, 127i32, 0usize),
        (1, 1024, 1023, 0),
        (1, 2048, 2047, 0),
        (8, 1536, 1535, 0),
        (64, 1536, 1535, 0),
        (1, 1024, 0, 0),
        (1, 1024, 0, 128),
        (1, 1024, 1023, 128),
    ] {
        let spec = PagedAttentionSpec { window, ..spec };
        let mut cx = Graph::new();
        let q_t = cx.tensor((s, spec.heads * spec.head_dim), DType::F32);
        let kc_t = cx.tensor((SLOTS, kv_dim), DType::Bf16);
        let vc_t = cx.tensor((SLOTS, kv_dim), DType::Bf16);
        let st_t = cx.tensor(s * context, DType::Int);
        let qo_t = cx.tensor(s + 1, DType::Int);
        let kv_t = cx.tensor(s + 1, DType::Int);
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
        let q: Vec<f32> = (0..s * spec.heads * spec.head_dim)
            .map(|_| lcg(&mut seed))
            .collect();
        // Each request owns `context` slots, strided through the pool.
        let slot_table: Vec<i32> = (0..s * context).map(|j| ((j * 7) % SLOTS) as i32).collect();
        let qo: Vec<i32> = (0..=s as i32).collect();
        let kv: Vec<i32> = (0..=s as i32).map(|r| r * context as i32).collect();
        let q_pos = vec![qpos; s];
        let sinks: Vec<f32> = (0..spec.heads).map(|_| lcg(&mut seed)).collect();
        let inputs: Vec<(_, HostBuffer)> = vec![
            (q_t.id, q.into()),
            (
                kc_t.id,
                HostBuffer::new(PlanDtype::Bf16, pool.clone()).unwrap(),
            ),
            (
                vc_t.id,
                HostBuffer::new(PlanDtype::Bf16, pool.clone()).unwrap(),
            ),
            (st_t.id, slot_table.into()),
            (qo_t.id, qo.into()),
            (kv_t.id, kv.into()),
            (qp_t.id, q_pos.into()),
            (sk_t.id, sinks.into()),
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
        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rt.execute().expect("execute");
        }
        let _ = rt.fetch(out.id).expect("fetch");
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!(
            "s {s:>3} context {context:>5} qpos {qpos:>5} window {window:>3}: {ms:>8.3} ms per tick (attention)"
        );
    }
}
