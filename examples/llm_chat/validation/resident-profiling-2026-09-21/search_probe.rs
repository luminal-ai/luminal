//! Temporary measurement harness; build as a CUDA llm_chat example.
use anyhow::Result;
use llm_chat::{TensorData, backend::cuda::bindings, checkpoint, graph::{LlmGraph, ModelConfig, ModelType}};
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer, cuda_registry};
use serde_json::json;
use std::{path::Path, time::Instant};

fn main() -> Result<()> {
    let checkpoint_path = Path::new("/dev/shm/llm-chat-bench-checkpoints/llama3-8b");
    let config = checkpoint::read_json(&checkpoint_path.join("config.json"))?;
    let graph = LlmGraph::build(ModelConfig::from_checkpoint(ModelType::Llama3, &config)?, 2048, 8)?;
    let mut runtime = CudaRuntime::load_with(&graph.graph, bindings(&graph), cuda_registry())?;
    runtime.bind_dyn_range('q', 1, 8)?;
    runtime.bind_dyn_range('c', 1, 2048)?;
    runtime.set_dim('q', 1);
    runtime.set_dim('c', 1);
    eprintln!("Loading checkpoint");
    let mut weights = checkpoint::load(checkpoint_path, &graph.parameters)?;
    weights.extend(graph.initial_inputs());
    weights.extend(graph.step_inputs(&[0], 0)?);
    let data: FxHashMap<_, HostBuffer> = weights.into_iter().map(|(id, value)| (id, match value {
        TensorData::F32(v) => v.into(), TensorData::I32(v) => v.into(),
    })).collect();
    let mut options = CompileOptions { generations: 2, generation_size: 4, trials: 3, seed: 0, search_log: true, ..Default::default() };
    eprintln!("Starting resident-aware search");
    let start = Instant::now();
    let outcome = runtime.search(&data, &options)?;
    let search_seconds = start.elapsed().as_secs_f64();
    let search_stats = runtime.graph_stats().unwrap();
    eprintln!("Search finished: {search_seconds:.3}s, winner {:.3}ms", outcome.best_nanos as f64 / 1e6);

    // Independently profile the selected plan with a new device, so its upload
    // counters and staging capacity are independent of search/cache-hit counts.
    let staged = data.iter().map(|(id, data)| Ok((runtime.input_buffer(*id)?, data))).collect::<Result<FxHashMap<_, _>>>()?;
    let residents = luminal::resident::ResidentBindings { inputs: runtime.residents().clone(), externals: Default::default() };
    options.shapes.values.insert('q'.into(), 1);
    options.shapes.values.insert('c'.into(), 1);
    options.shapes.bounds.insert('q'.into(), (1, 8));
    options.shapes.bounds.insert('c'.into(), (1, 2048));
    let mut device = luminal_cuda_lite::device::CudaDevice::new(0)?;
    let result = luminal_cuda_lite::profile::profile_candidate_at(&mut device, runtime.plan().unwrap(), &staged, &residents, 3, None, None, &options.shapes).map_err(|e| anyhow::anyhow!("{e}"))?;
    let stats = device.stats();
    let luminal_cuda_lite::profile::Measurement::Timed { mean_nanos, completed_trials } = result else { unreachable!() };
    let transient_input_bytes: usize = staged.iter().filter(|(lit, _)| !residents.inputs.contains(lit)).map(|(_, d)| d.bytes.len()).sum();
    let resident_input_bytes: usize = staged.iter().filter(|(lit, _)| residents.inputs.contains(lit)).map(|(_, d)| d.bytes.len()).sum();
    device.release_slab();
    for (id, value) in data { runtime.set_data(id, value)?; }
    runtime.execute()?;
    let logits = runtime.get_f32(graph.logits)?;
    anyhow::ensure!(logits.iter().all(|v| v.is_finite()), "nonfinite logits");
    let report = json!({
        "checkpoint": checkpoint_path, "model": "Llama3-8B FP32", "max_context": 2048, "prefill_chunk": 8,
        "profile_q": 1, "profile_c": 1, "generations": 2, "population": 4, "trials": 3, "seed": 0,
        "search_seconds": search_seconds, "best_trial_mean_seconds": outcome.best_nanos as f64 / 1e9,
        "plans_profiled": outcome.plans_profiled, "fingerprint_hits": outcome.fingerprint_hits,
        "search_launches": search_stats.launches, "search_resident_upload_bytes": search_stats.resident_upload_bytes,
        "selected_plan_fresh_profile": {
            "mean_trial_seconds": mean_nanos as f64 / 1e9, "completed_trials": completed_trials,
            "launches": stats.launches, "resident_upload_bytes": stats.resident_upload_bytes,
            "staging_bytes": stats.staging_bytes, "arena_bytes": stats.arena_bytes,
            "resident_input_bytes": resident_input_bytes, "transient_input_bytes": transient_input_bytes,
            "kernel_compilations": stats.kernel_compilations,
        }, "finite_serving_logits": true, "logits_len": logits.len(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    std::fs::write("/tmp/resident-profiling-llama.json", serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}
