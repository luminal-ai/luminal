mod model;

use luminal::prelude::*;
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing::{span, Level};

// This example compiles and runs Llama 3 8B on CUDA. On an H100, this should hit >75% MBU

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 10;
    let search_graphs = 5; // the number of graphs we want to search during compilation
    let prompt = "Hello, how are you";

    // Set up tracing to perfetto
    let trace_session = luminal_tracing::subscriber()
        .perfetto("trace.pftrace")
        .env_filter(format!(
            "{}=trace,luminal=trace,luminal_cuda=trace",
            env!("CARGO_PKG_NAME")
        ))
        .init();

    // Set up cuda context and stream
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Tokenize prompt
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();
    let mut sentence = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    // Allocate kv cache
    let mut kv_cache = KVCache::new(&stream, max_seq_len);

    // Create compute graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let model = model::Llama::init(&mut cx);
    let logits = model.forward(input, token_ids, &kv_cache).output();

    // Build search space
    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    // Load model weights from safetensors file
    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, "setup/model_combined.safetensors");

    // Run search process
    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1]);
    runtime.set_data(token_ids, vec![0]);
    runtime = cx.search(runtime, search_graphs);
    kv_cache.reset();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    // Decode loop
    let mut prev_seq = 0;
    let mut fwd_durations = vec![];
    for i in 0..gen_tokens {
        let start = std::time::Instant::now();
        let _span = if i == 0 {
            span!(Level::INFO, "prefill")
        } else {
            span!(Level::INFO, "decode")
        }
        .entered();

        // Set runtime dimensions
        let seq_len = sentence.len();
        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        // Set input data
        runtime.set_data(
            input,
            sentence.iter().map(|i| *i as i32).collect::<Vec<_>>(),
        );
        runtime.set_data(
            token_ids,
            (prev_seq as i32..(seq_len + prev_seq) as i32).collect::<Vec<_>>(),
        );

        // Execute forward pass
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Sample next token
        let _sample_span = span!(Level::INFO, "sample_full").entered();
        sentence = vec![*sample(&logits_data, VOCAB_SIZE).last().unwrap()];
        prev_seq += seq_len;
        print!("{}", tokenizer.decode(&sentence, true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Report benchmarks
    println!("  TTFT: {:.2} ms", fwd_durations[0].as_secs_f64() * 1e3);
    println!(
        "  TPOT: {:.2} ms",
        // Don't record prefill or first decode
        (fwd_durations.iter().skip(2).sum::<Duration>() / (fwd_durations.len() - 2) as u32)
            .as_secs_f64()
            * 1_000.
    );
    runtime.print_execution_stats();
    // Dump cuda trace to timeline
    trace_session.stop();
    if let Some(path) = trace_session.perfetto_path {
        runtime.record_cuda_perfetto_trace(path);
    }
}

#[tracing::instrument(skip_all)]
fn sample(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as u32
        })
        .collect()
}
