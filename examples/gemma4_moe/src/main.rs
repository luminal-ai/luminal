mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use rustc_hash::FxHashSet;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

const REPO_ID: &str = "google/gemma-4-26B-A4B";

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn main() {
    let max_seq_len = env_usize("MAX_SEQ_LEN", 4096);
    let gen_tokens = env_usize("GEN_TOKENS", 30);
    let search_graphs = env_usize("SEARCH_GRAPHS", 50);
    let prompt = std::env::var("PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());
    let print_token_ids = env_bool("PRINT_TOKEN_IDS");

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt_tokens = tokenizer
        .encode(prompt.as_str(), true)
        .unwrap()
        .get_ids()
        .to_vec();

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let (logits, cache_outputs) = Gemma4MoE::init(&mut cx).forward(input, pos_ids, &kv_cache);
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

    for layer in 0..LAYERS {
        let cache_bytes = cache_bytes_for_layer(layer, max_seq_len);
        runtime.set_zeros(kv_cache.k_caches[layer], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[layer], cache_bytes);
    }

    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 1);
    runtime.set_data(input, vec![1]);
    runtime.set_data(pos_ids, vec![1]);
    runtime = cx.search(runtime, search_graphs);

    for layer in 0..LAYERS {
        let cache_bytes = cache_bytes_for_layer(layer, max_seq_len);
        runtime.set_zeros(kv_cache.k_caches[layer], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[layer], cache_bytes);
    }

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut prev_seq = 0usize;
    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();
    let mut generated_token_ids = vec![];
    let repetition_penalty: f32 = 1.05;

    const EOS_TOKEN: u32 = 1;

    let prefill_start = std::time::Instant::now();
    for &token in &prompt_tokens {
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);

        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }

        prev_seq += 1;
    }
    let prefill_duration = prefill_start.elapsed();

    let logits_data = runtime.get_f32(logits);
    let last_row = &logits_data[..VOCAB_SIZE];
    let mut next_token = last_row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32;
    generated_token_ids.push(next_token);
    print!("{}", tokenizer.decode(&[next_token], true).unwrap());
    std::io::stdout().flush().unwrap();
    seen_tokens.insert(next_token);

    for _ in 1..gen_tokens {
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);

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
        generated_token_ids.push(next_token);
        seen_tokens.insert(next_token);

        if next_token == EOS_TOKEN {
            break;
        }

        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();
    if print_token_ids {
        println!("Generated token ids: {generated_token_ids:?}");
    }

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
