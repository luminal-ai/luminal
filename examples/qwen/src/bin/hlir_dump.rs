//! Dump the raw (pre-search) HLIR op histogram of the hand-coded Qwen3 graph,
//! to compare against the luminal_python translator output op-by-op.
//!
//! Usage: cargo run -p qwen --bin hlir_dump -- [layers]   (default 1)
//!
//! Builds exactly the graph the example builds (Qwen::init().forward()) but
//! stops before build_search_space, so cx.graph is the frontend HLIR — the
//! same level at which we dump the python translator's output.
use luminal::prelude::*;
use qwen::model::{KVCache, Qwen};
use std::collections::BTreeMap;

fn main() {
    let layers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let max_seq = 4096;

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq, layers);
    let (logits, cache_outputs) =
        Qwen::init(&mut cx, layers).forward(input, token_ids, &kv_cache);
    let _ = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for n in cx.graph.node_indices() {
        let disp = format!("{}", cx.graph[n]);
        let bare = disp.split('(').next().unwrap_or(&disp).trim().to_string();
        *counts.entry(bare).or_insert(0) += 1;
    }
    let total: usize = counts.values().sum();
    println!("=== RUST QWEN HLIR HISTOGRAM (layers={layers}) total_nodes={total} ===");
    for (k, v) in &counts {
        println!("    {k:<16} {v}");
    }
    println!("=== END RUST QWEN HLIR HISTOGRAM ===");
}
