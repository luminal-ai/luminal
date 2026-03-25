mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use rustc_hash::FxHashSet;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

const REPO_ID: &str = "Qwen/Qwen3-30B-A3B";

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 30;
    let search_graphs = 50;
    let prompt = "The capital of France is";

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt_tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let (logits, cache_outputs) = Qwen3MoE::init(&mut cx).forward(input, pos_ids, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let cache_bytes = N_KV_HEADS * max_seq_len * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 1);
    runtime.set_data(input, vec![1]);
    runtime.set_data(pos_ids, vec![1]);
    runtime = cx.search(runtime, search_graphs);

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut prev_seq = 0usize;
    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();
    let repetition_penalty: f32 = 1.05;

    const EOS_TOKEN: u32 = 151645;
    const STOP_TOKEN: u32 = 151643;

    // Prefill: process prompt tokens one at a time
    let prefill_start = std::time::Instant::now();
    for &token in &prompt_tokens {
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);

        // Round-trip KV cache
        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }

        prev_seq += 1;
    }
    let prefill_duration = prefill_start.elapsed();

    // Get logits from last prefill step and sample first new token
    let logits_data = runtime.get_f32(logits);
    let last_row = &logits_data[..VOCAB_SIZE];
    let mut next_token = last_row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32;
    print!("{}", tokenizer.decode(&[next_token], true).unwrap());
    std::io::stdout().flush().unwrap();
    seen_tokens.insert(next_token);

    // Decode loop
    for _ in 1..gen_tokens {
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);

        // Round-trip KV cache
        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }

        prev_seq += 1;

        let logits_data = runtime.get_f32(logits);
        let mut last_row = logits_data[..VOCAB_SIZE].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
        next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        seen_tokens.insert(next_token);

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Report benchmarks
    println!(
        "  TTFT: {:.2} ms ({} prompt tokens)",
        prefill_duration.as_secs_f64() * 1e3,
        prompt_tokens.len()
    );
    if fwd_durations.len() > 1 {
        println!(
            "  TPOT: {:.2} ms",
            (fwd_durations.iter().skip(1).sum::<Duration>() / (fwd_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1_000.
        );
    }
}
