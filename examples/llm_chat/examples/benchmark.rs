//! Fixed-length latency measurement using the chat application's real session.
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use llm_chat::{
    backend::cuda::CudaBackend,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType, checkpoint_dtype},
    sampling::Sampler,
    session::Session,
    tokenizer::{ChatTokenizer, Message},
};
use luminal::bufferize::BufferNode;
use luminal_cuda_lite::CompileOptions;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    model: ModelType,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    report: PathBuf,
    /// Save the selected operation inventory for comparisons.
    #[arg(long)]
    plan_report: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    input_tokens: usize,
    #[arg(long, default_value_t = 128)]
    output_tokens: usize,
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_PREFILL_CHUNK)]
    prefill_chunk: usize,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    /// Search generations per bucket.
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_SEARCH_GENERATIONS)]
    search_generations: usize,
    /// Candidate attempts per generation, per bucket.
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_SEARCH_POPULATION)]
    search_population: usize,
}

// Keep the complete checkpoint chat template, including its generation suffix.
// Padding is inside the user message, never raw token IDs after that suffix.
fn prompt(tokenizer: &ChatTokenizer, count: usize) -> Result<String> {
    let base = "Write a detailed, continuous story of at least five hundred words about a traveler who discovers an old teapot in a quiet mountain village. Describe the landscape, the people, and the surprising history of the teapot. Start the story immediately and keep writing without a conclusion or a summary. Include these details in your story:";
    let mut text = base.to_string();
    for _ in 0..count {
        let tokens = tokenizer.encode_chat(&[Message::new("user", &text)], false)?;
        if tokens.len() == count {
            return Ok(text);
        }
        ensure!(tokens.len() < count, "prompt exceeds requested token count");
        text.push_str(" quiet");
    }
    Err(anyhow!("could not construct exact-length chat prompt"))
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.
    } else {
        values[mid]
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(cfg!(feature = "cuda_lite"), "requires --features cuda");
    ensure!(args.repetitions > 0, "repetitions must be positive");
    ensure!(args.output_tokens > 1, "need at least two output tokens");
    ensure!(
        args.search_generations > 0 && args.search_population > 0,
        "search budget must be positive"
    );
    let setup = Instant::now();
    let config = checkpoint::read_json(&args.checkpoint.join("config.json"))?;
    let tokenizer = ChatTokenizer::load(&args.checkpoint, &config)?;
    let user_prompt = prompt(&tokenizer, args.input_tokens)?;
    let messages = vec![Message::new("user", &user_prompt)];
    let prompt_ids = tokenizer.encode_chat(&messages, false)?;
    ensure!(
        prompt_ids.len() == args.input_tokens,
        "input length differs"
    );
    let capacity = args.input_tokens + args.output_tokens;
    let graph = LlmGraph::build_with_parameter_dtype(
        ModelConfig::from_checkpoint(args.model, &config)?,
        checkpoint_dtype(&config)?,
        capacity,
        args.prefill_chunk,
    )?;
    let weight_bytes: usize = graph
        .parameters
        .iter()
        .map(|p| p.shape.iter().product::<usize>() * p.dtype.bits().div_ceil(8))
        .sum();
    eprintln!(
        "Loading {} tensors ({weight_bytes} native-precision bytes)",
        graph.parameters.len()
    );
    let weights = checkpoint::load(&args.checkpoint, &graph.parameters)?;
    let load_seconds = setup.elapsed().as_secs_f64();
    eprintln!(
        "Compiling {:?}, chunk={}, context={capacity}",
        args.model, args.prefill_chunk
    );
    let compile = Instant::now();
    let backend = CudaBackend::compile(
        &graph,
        weights,
        &CompileOptions {
            generations: args.search_generations,
            generation_size: args.search_population,
            seed: 0,
            search_log: false,

            ..Default::default()
        },
    )?;
    let compile_seconds = compile.elapsed().as_secs_f64();
    let search_buckets: Vec<_> = backend.bucket_plans().iter().map(|bucket| {
        let mut counts = BTreeMap::<String, usize>::new();
        for node in bucket.plan.dag.node_weights() {
            if let BufferNode::Compute { op, .. } = node {
                *counts.entry(op.label().to_owned()).or_default() += 1;
            }
        }
        json!({"q_range":bucket.ranges[&'q'.into()], "representative_q":bucket.representative[&'q'.into()],
            "representative_c":bucket.representative[&'c'.into()], "plans_profiled":bucket.outcome.plans_profiled,
            "fingerprint_hits":bucket.outcome.fingerprint_hits, "best_nanos":bucket.outcome.best_nanos.to_string(),
            "selected_counts":counts, "memory_pruning":{
                "oversized_tensors":bucket.outcome.memory_pruning.oversized_tensors,
                "producer_classes":bucket.outcome.memory_pruning.producer_classes,
                "removed_nodes":bucket.outcome.memory_pruning.removed_nodes,
                "largest_tensor_bytes":bucket.outcome.memory_pruning.largest_tensor_bytes}})
    }).collect();
    let plan = backend.plan().context("missing selected plan")?;
    let mut selected_counts = BTreeMap::<String, usize>::new();
    let mut nodes = Vec::new();
    for id in plan.dag.node_indices() {
        let label = match &plan.dag[id] {
            BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                if args.plan_report.is_some() {
                    nodes.push(json!({
                        "id": id.index(), "op": op.label(), "details": format!("{op:?}"),
                        "reads": reads.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>(),
                        "writes": writes.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>(),
                        "operand_shapes": operand_info.iter().map(|s| format!("{:?}", s.layout.shape())).collect::<Vec<_>>(),
                        "result_shapes": result_info.iter().map(|s| format!("{:?}", s.layout.shape())).collect::<Vec<_>>()
                    }));
                }
                op.label()
            }
            BufferNode::BufferInput { .. } => "BufferInput",
            BufferNode::BufferOutput { .. } => "BufferOutput",
            BufferNode::BufferCopy { .. } => "BufferCopy",
        };
        *selected_counts.entry(label.to_owned()).or_default() += 1;
    }
    if let Some(path) = &args.plan_report {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&json!({
                "model": format!("{:?}", args.model), "ranking": "device_time",
                "selected_phase": "decode", "search_buckets": search_buckets,
                "selected_counts": selected_counts, "nodes": nodes
            }))? + "\n",
        )?;
    }
    let mut session = Session::new(graph, backend);
    let mut runs = Vec::new();
    let mut baseline = None;
    // First request records cold/lazy setup and warms every required shape.
    // Later requests reset KV state, so no prompt-prefix reuse enters timings.
    for run in 0..=args.repetitions {
        session.reset()?;
        let mut sampler = Sampler::new(0., 1., 0)?;
        let mut emitted_ms = Vec::with_capacity(args.output_tokens);
        let mut stream = tokenizer.tokenizer.decode_stream(true);
        let start = Instant::now();
        let tokens = tokenizer.encode_chat(&messages, false)?;
        // Fixed-length benchmarking deliberately disables early EOS termination.
        let generated = session.generate(
            &tokens,
            args.output_tokens,
            &BTreeSet::new(),
            &mut sampler,
            |token| {
                let _ = stream
                    .step(token)
                    .map_err(|e| anyhow!("stream decode: {e}"))?;
                emitted_ms.push(start.elapsed().as_secs_f64() * 1000.);
                Ok(())
            },
        )?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.;
        ensure!(
            generated.len() == args.output_tokens,
            "output length differs"
        );
        ensure!(
            emitted_ms.len() == args.output_tokens,
            "emission count differs"
        );
        ensure!(session.cached().len() == capacity, "cache length differs");
        if let Some(expected) = &baseline {
            ensure!(*expected == generated, "tokens differ after KV reset");
        } else {
            baseline = Some(generated.clone());
        }
        let ttft_ms = emitted_ms[0];
        let tpot_ms =
            (emitted_ms[args.output_tokens - 1] - ttft_ms) / (args.output_tokens - 1) as f64;
        let first_eos = generated
            .iter()
            .position(|t| tokenizer.stop_tokens.contains(t));
        eprintln!(
            "run={run} warm={} input={} output={} TTFT={ttft_ms:.3}ms TPOT={tpot_ms:.3}ms total={total_ms:.3}ms first_eos={first_eos:?}",
            run > 0,
            tokens.len(),
            generated.len()
        );
        runs.push(json!({
            "run": run, "warm": run > 0, "input_tokens": tokens.len(),
            "output_tokens": generated.len(), "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "decode_tokens_per_second": 1000. / tpot_ms, "total_ms": total_ms,
            "first_eos_index": first_eos, "emission_timestamps_ms": emitted_ms,
            "generated_tokens": generated, "text": tokenizer.decode(&generated)?,
        }));
    }
    let warm: Vec<_> = runs.iter().filter(|r| r["warm"] == true).collect();
    let med = |key: &str| median(warm.iter().map(|r| r[key].as_f64().unwrap()));
    let source: Option<Value> =
        checkpoint::read_json(&args.checkpoint.join("BENCH_SOURCE.json")).ok();
    let report = json!({
        "model": format!("{:?}", args.model), "checkpoint": args.checkpoint,
        "source": source, "backend": "cuda_lite",
        "parameter_dtype": format!("{:?}", checkpoint_dtype(&config)?), "compute_dtype": "F32",
        "batch_size": 1,
        "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
        "max_context": capacity, "prefill_chunk": args.prefill_chunk,
        "search_generations": args.search_generations, "search_population": args.search_population,
        "ranking": "device_time", "selected_phase": "decode", "selected_counts": selected_counts, "search_buckets": search_buckets,
        "seed": 0, "temperature": 0, "thinking": false, "ignore_eos": true,
        "warmups": 1, "repetitions": args.repetitions,
        "load_seconds": load_seconds, "compile_seconds": compile_seconds,
        "native_weight_bytes": weight_bytes, "user_prompt": user_prompt, "prompt_tokens": prompt_ids,
        "measurement": "Host wall clock through the real Session::generate callback, including chat tokenization, input staging, synchronized logits readback, sampling and stream decoding. Excludes initial loading/compiler search, reset allocation and terminal I/O. Session ingests each sampled token before emitting it, so TTFT includes one decode step. Warm requests reset KV state; resident-cache reset uploads remain inside the timed execution. TPOT=(last emission-first emission)/(output tokens-1).",
        "median_ttft_ms": med("ttft_ms"), "median_tpot_ms": med("tpot_ms"),
        "median_decode_tokens_per_second": med("decode_tokens_per_second"),
        "median_total_ms": med("total_ms"), "runs": runs,
    });
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
