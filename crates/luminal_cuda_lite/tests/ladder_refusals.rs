//! M4 PHASE 5 GATE (b): the REFUSAL-ACCOUNTING LADDER on CUDA-lite with
//! views admitted.
//!
//! This is the cycle-anatomy measurement discipline (the L1-L8 ladder,
//! 2026-08-07) re-run against THIS runtime: the same llama-block
//! configs as `luminal_nn::models::tests::measure_scaling_curves` —
//! depth 1 at d ∈ {8,16,32}, then depth 2/4/8 at d=8 — searched through
//! `CudaRuntime::search` (CL matcher set + CL allow list, two-phase
//! sampler, static byte-move profiler). The original harness runs the
//! ReferenceRuntime; the graphs, budgets, and refusal accounting are
//! identical, so this is the nearest CL-direct equivalent.
//!
//! ACCEPTANCE: refusals MUST be zero on every rung — extraction
//! (choice-cycles, dead-ends), bufferize, and execute alike. With the
//! view op electable, every class gains zero-byte view candidates; a
//! nonzero refusal count here would be the sampler-vs-view-2-cycles
//! regression the ladder exists to catch.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{DType, FxHashMap, NodeIndex};
use luminal::shape::IntExpr;
use luminal_cuda_lite::CudaRuntime;
use mini_llama3::{model_support::Namespace, MiniLlama3Layer};

/// Deterministic pseudo-random weights (the nn harness's `weights`
/// discipline: value content is irrelevant, only geometry matters).
fn weights(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}

fn run_rung(layers: usize, d: usize, default_budget: bool) -> (usize, usize, usize, usize) {
    let n_heads = 4;
    let n_kv = 2;
    let ff = d + d / 2;
    let kv_dim = n_kv * (d / n_heads);
    const SLOTS: usize = 4;
    const CTX: usize = 2;

    let mut cx = Graph::new();
    let blocks: Vec<MiniLlama3Layer> = (0..layers)
        .map(|l| {
            MiniLlama3Layer::new(
                d,
                ff,
                n_heads,
                n_kv,
                &Namespace::root().child("layers").index(l),
                &mut cx,
            )
        })
        .collect();
    let x = cx.tensor((1, d), DType::F32);
    let caches: Vec<_> = (0..layers)
        .map(|_| {
            (
                cx.tensor((SLOTS, kv_dim), DType::F32),
                cx.tensor((SLOTS, kv_dim), DType::F32),
            )
        })
        .collect();
    let gather_idx = cx.tensor(CTX, luminal::dtype::DType::Int);
    let scatter_idx = cx.tensor(1, luminal::dtype::DType::Int);
    let mut h = x;
    for (layer, block) in blocks.iter().enumerate() {
        let (next, kc, vc) = block.forward(
            h,
            caches[layer].0,
            caches[layer].1,
            gather_idx,
            scatter_idx,
            IntExpr::from(1usize),
        );
        h = next;
        kc.output();
        vc.output();
    }
    let _ = h.output();

    let mut pairs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (x.id, weights(d, 90).into()),
        (gather_idx.id, vec![0i32, 1].into()),
        (scatter_idx.id, vec![1i32].into()),
    ];
    for (layer, block) in blocks.iter().enumerate() {
        pairs.push((block.wq.weight.id, weights(d * d, 91 + layer as u64).into()));
        pairs.push((
            block.wk.weight.id,
            weights(d * kv_dim, 92 + layer as u64).into(),
        ));
        pairs.push((
            block.wv.weight.id,
            weights(d * kv_dim, 93 + layer as u64).into(),
        ));
        pairs.push((block.wo.weight.id, weights(d * d, 94 + layer as u64).into()));
        pairs.push((
            block.gate.weight.id,
            weights(d * ff, 95 + layer as u64).into(),
        ));
        pairs.push((
            block.up.weight.id,
            weights(d * ff, 96 + layer as u64).into(),
        ));
        pairs.push((
            block.down.weight.id,
            weights(ff * d, 97 + layer as u64).into(),
        ));
        pairs.push((
            caches[layer].0.id,
            weights(SLOTS * kv_dim, 98 + layer as u64).into(),
        ));
        pairs.push((
            caches[layer].1.id,
            weights(SLOTS * kv_dim, 99 + layer as u64).into(),
        ));
    }
    let data: FxHashMap<NodeIndex, TypedBuffer> = pairs.into_iter().collect();

    let mut rt = CudaRuntime::load(&cx).expect("cuda load");
    let budget = if default_budget {
        ImplementationSearchOptions::default()
    } else {
        // The fixed 8-genome budget of the original ladder's depth-1
        // rungs: comparable refusal RATES across d.
        ImplementationSearchOptions {
            generations: 2,
            generation_size: 4,
            mutations: 2,
            trials: 1,
            seed: 0,
            search_log: false,
        }
    };
    let start = std::time::Instant::now();
    let outcome = rt
        .search(&data, &budget)
        .unwrap_or_else(|e| panic!("L{layers} d{d}: SEARCH REFUSED: {e:#}"));
    let b = &outcome.refusal_breakdown;
    eprintln!(
        "L{layers} d{d} | {:.1}s | plans {} | {}",
        start.elapsed().as_secs_f64(),
        outcome.plans_profiled,
        b.summary(),
    );
    assert!(
        outcome.plans_profiled > 0,
        "L{layers} d{d}: no plans profiled"
    );
    (
        b.extract_refusals,
        b.with_choice_cycles,
        b.plan_build_refusals,
        b.execute_refusals,
    )
}

fn assert_zero(rung: &str, counts: (usize, usize, usize, usize)) {
    let (extract, cycles, bufferize, execute) = counts;
    assert_eq!(
        (extract, cycles, bufferize, execute),
        (0, 0, 0, 0),
        "{rung}: refusals must be ZERO with views admitted \
         (extract {extract} / choice-cycles {cycles} / bufferize {bufferize} / execute {execute})"
    );
}

#[test]
fn ladder_l1_d8() {
    assert_zero("L1 d8", run_rung(1, 8, false));
}

#[test]
fn ladder_l1_d16() {
    assert_zero("L1 d16", run_rung(1, 16, false));
}

#[test]
fn ladder_l1_d32() {
    assert_zero("L1 d32", run_rung(1, 32, false));
}

#[test]
#[ignore = "deep rung (default budget, ~10-60s release) — run explicitly by name, \
            mirroring the original ladder harness discipline"]
fn ladder_l2_d8() {
    assert_zero("L2 d8", run_rung(2, 8, true));
}

#[test]
#[ignore = "deep rung (default budget, ~10-60s release) — run explicitly by name, \
            mirroring the original ladder harness discipline"]
fn ladder_l4_d8() {
    assert_zero("L4 d8", run_rung(4, 8, true));
}

#[test]
#[ignore = "deep rung (default budget, ~10-60s release) — run explicitly by name, \
            mirroring the original ladder harness discipline"]
fn ladder_l8_d8() {
    assert_zero("L8 d8", run_rung(8, 8, true));
}
