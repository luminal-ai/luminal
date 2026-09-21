//! Temporary full-checkpoint validation of the default chat search.
use anyhow::{Result, ensure};
use llm_chat::{
    backend::cuda::CudaBackend,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType},
};
use luminal_cuda_lite::CompileOptions;
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let path = Path::new("/dev/shm/llm-chat-bench-checkpoints/llama3-8b");
    let config = checkpoint::read_json(&path.join("config.json"))?;
    let graph = LlmGraph::build(
        ModelConfig::from_checkpoint(ModelType::Llama3, &config)?,
        2048,
        128,
    )?;
    eprintln!("Loading Llama3-8B checkpoint");
    let weights = checkpoint::load(path, &graph.parameters)?;
    eprintln!("Starting full 100+100 candidate search at default context capacity 2048");
    let start = Instant::now();
    let mut backend = CudaBackend::compile(
        &graph,
        weights,
        &CompileOptions {
            generations: llm_chat::search::DEFAULT_SEARCH_GENERATIONS,
            generation_size: llm_chat::search::DEFAULT_SEARCH_POPULATION,
            ..Default::default()
        },
    )?;
    let search_seconds = start.elapsed().as_secs_f64();
    let bound = llm_chat::backend::cuda::bindings(&graph)
        .bind(&graph.graph.logical)
        .map_err(anyhow::Error::msg)?;
    let mut full_plans = Vec::new();
    let buckets: Vec<_> = backend
        .bucket_plans()
        .iter()
        .map(|bucket| {
            let o = &bucket.outcome;
            let mut bounds = bucket.ranges.clone();
            bounds.insert('c'.into(), (1, graph.capacity));
            full_plans.push((bucket.plan.clone(), bounds.clone()));
            let allocation = luminal_cuda_lite::resident::allocate(
                vec![(bucket.plan.clone(), bounds)],
                bound.residents(),
            )
            .unwrap();
            let attempts = o.plans_profiled
                + o.fingerprint_hits
                + o.refusal_breakdown.extract_refusals
                + o.refusal_breakdown.plan_build_refusals
                + o.refusal_breakdown.execute_refusals
                + o.refusal_breakdown.timed_out;
            assert_eq!(attempts, 100);
            json!({"q":bucket.representative[&'q'.into()],"c":bucket.representative[&'c'.into()],
            "attempts":attempts,"profiles":o.plans_profiled,"cache_hits":o.fingerprint_hits,
            "transient_bytes":bucket.slab_bytes,"arena_bytes":allocation.bytes,
            "best_ms":o.best_nanos as f64 / 1e6,
            "refusals":o.refusal_breakdown.summary(),
            "oversized_tensors_pruned":o.memory_pruning.oversized_tensors,
            "producer_classes_pruned":o.memory_pruning.producer_classes,
            "nodes_removed":o.memory_pruning.removed_nodes})
        })
        .collect();
    let installed_arena_bytes =
        luminal_cuda_lite::resident::allocate(full_plans, bound.residents())?.bytes;
    let tokens: Vec<u32> = (0..128).map(|i| i % 101 + 1).collect();
    let start = Instant::now();
    let first = backend.step(graph.step_inputs(&tokens, 0)?, 128, 128)?;
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        first.len() == graph.vocab && first.iter().all(|x| x.is_finite()),
        "invalid prefill logits"
    );
    let next = first
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32;
    let start = Instant::now();
    let decode = backend.step(graph.step_inputs(&[next], 128)?, 1, 129)?;
    let decode_ms = start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        decode.len() == graph.vocab && decode.iter().all(|x| x.is_finite()),
        "invalid decode logits"
    );
    backend.reset()?;
    let reset = backend.step(graph.step_inputs(&tokens, 0)?, 128, 128)?;
    ensure!(reset == first, "reset changed logits");
    let report = json!({"model":"Llama3-8B", "max_context":2048, "prefill_chunk":128,
        "search_seconds":search_seconds,"buckets":buckets,"cold_prefill_ms":prefill_ms,
        "installed_arena_bytes":installed_arena_bytes,
        "decode_ms":decode_ms,"finite_logits":true,"reset_matches":true});
    std::fs::write(
        "/tmp/prefill-run-result.json",
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
