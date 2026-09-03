//! Synthetic watch-item blocks at sizes where execute cost is visible:
//! cumsum/unfold-family (conv-shaped windowed reduce), top-k (MoE routing
//! scaffold), and a rank-4 scalar-broadcast chain. Identical source in
//! the baseline and transparent worktrees; measurement-only.

use luminal::prelude::*;
use std::time::Instant;

fn random_vec(n: usize) -> Vec<f32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn measure_plan(
    name: &str,
    cx: &Graph,
    pairs: &[(petgraph::graph::NodeIndex, TypedBuffer)],
    t0: Instant,
) {
    let rec_us = t0.elapsed().as_micros();
    let model = match cx.logical.model_text() {
        Ok(m) => m,
        Err(e) => {
            println!("{name}: RECORD-POISONED: {e}");
            return;
        }
    };
    let rows = model.lines().filter(|l| !l.trim().is_empty()).count();
    let applies = model.matches("(LogicalIndexMapApply").count();
    let mut depth = std::collections::HashMap::new();
    let mut max_chain = 0usize;
    let mut hist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for line in model.lines() {
        if let Some(rest) = line.strip_prefix("(let v") {
            let id: usize = rest.split_whitespace().next().unwrap().parse().unwrap();
            let d = if rest.contains("(LogicalIndexMapApply v") {
                let op: usize = rest
                    .split("(LogicalIndexMapApply v")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap();
                depth.get(&op).copied().unwrap_or(0) + 1
            } else {
                0
            };
            if d > 0 {
                *hist.entry(d).or_insert(0) += 1;
            }
            max_chain = max_chain.max(d);
            depth.insert(id, d);
        }
    }
    println!("{name}: MODEL rows={rows} applies={applies} max_apply_chain={max_chain} record_us={rec_us}");
    println!("{name}: CHAIN_DEPTH_HIST {hist:?}");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rt = luminal_reference::ReferenceRuntime::load(cx).expect("native load");
        let data = pairs.iter().cloned().collect();
        let t = Instant::now();
        let outcome = rt
            .search(&data, &luminal::test_support::harness_search_options())
            .expect("search finds a plan");
        let search_ms = t.elapsed().as_millis();
        println!(
            "{name}: SEARCH wall_ms={search_ms} [{}]",
            outcome.timings.summary()
        );
        let summary = outcome.best_plan.summary();
        let mut in_ops = false;
        let mut total = 0usize;
        let mut mats = 0usize;
        let mut allocs = 0usize;
        for line in summary.lines() {
            if line.starts_with("ops (") {
                in_ops = true;
                continue;
            }
            if line.starts_with("anti (") {
                in_ops = false;
            }
            if in_ops {
                if let Some(rest) = line.strip_prefix("  ") {
                    let label = rest.split(':').next().unwrap_or("");
                    if label.contains("Materialize") {
                        mats += 1;
                    }
                    if label == "BufferAlloc" {
                        allocs += 1;
                    }
                    total += 1;
                }
            }
        }
        println!("{name}: PLAN total_ops={total} materialize={mats} buffer_alloc={allocs}");
        for (id, v) in pairs {
            rt.set_data(*id, v.clone());
        }
        let t = Instant::now();
        rt.execute().expect("winner executes");
        println!("{name}: EXECUTE wall_ms={}", t.elapsed().as_millis());
    }));
    if let Err(e) = result {
        let msg = e
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic>");
        println!("{name}: LADDER PANICKED: {msg}");
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();

    // Watch item 1 stand-in: cumsum/unfold family at visible size.
    // (4,4,64,64) cumsum over the last axis — the windowed-expand
    // scaffold materializes O(n^2) on that axis.
    if which.is_empty() || which == "cumsum_big" {
        let t0 = Instant::now();
        let mut cx = Graph::new();
        let x = cx.tensor((4, 4, 64, 64), DType::F32);
        let _ = x.cumsum(3).output();
        let pairs = vec![(x.id, TypedBuffer::from(random_vec(4 * 4 * 64 * 64)))];
        measure_plan("cumsum_big", &cx, &pairs, t0);
    }

    // Watch item 2 stand-in: MoE-router-shaped top-k at visible size.
    if which.is_empty() || which == "topk_big" {
        let t0 = Instant::now();
        let mut cx = Graph::new();
        let x = cx.tensor((8, 256), DType::F32);
        let _ = x.topk_indexes(4, 1).output();
        let pairs = vec![(x.id, TypedBuffer::from(random_vec(8 * 256)))];
        measure_plan("topk_big", &cx, &pairs, t0);
    }

    // The expand_rhs scalar-broadcast chain at rank 4, visible size.
    if which.is_empty() || which == "scalar_broadcast_big" {
        let t0 = Instant::now();
        let mut cx = Graph::new();
        let x = cx.tensor((16, 16, 64, 64), DType::F32);
        let _ = (x * 2.0f32).output();
        let pairs = vec![(x.id, TypedBuffer::from(random_vec(16 * 16 * 64 * 64)))];
        measure_plan("scalar_broadcast_big", &cx, &pairs, t0);
    }
}
