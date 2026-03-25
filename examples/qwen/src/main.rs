mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use rustc_hash::FxHashSet;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "Qwen/Qwen3-4B";

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 500;
    let search_graphs = 500;
    let prompt = "Explain what a neural network is in a paragraph.";

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt_tokens = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let (logits, cache_outputs) = Qwen::init(&mut cx).forward(input, token_ids, &kv_cache);
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
    runtime.set_data(token_ids, vec![1]);
    runtime = cx.search(runtime, search_graphs);

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    let mut prev_seq = 1usize;
    let mut sentence = vec![prompt_tokens[0]];
    let total_steps = prompt_tokens.len() - 1 + gen_tokens;
    let prompt_len = prompt_tokens.len();
    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();
    let repetition_penalty: f32 = 1.05;

    const EOS_TOKEN: u32 = 151645; // <|endoftext|>
    const STOP_TOKEN: u32 = 151643; // <|end|>

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, gen_tokens
    );

    for i in 0..total_steps {
        let start = std::time::Instant::now();
        let is_prefill = i < prompt_len - 1;
        let seq_len = sentence.len();

        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        runtime.set_data(
            input,
            sentence.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        );
        runtime.set_data(
            token_ids,
            (prev_seq as i32..(seq_len + prev_seq) as i32).collect::<Vec<_>>(),
        );

        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Round-trip KV cache
        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
        }

        prev_seq += seq_len;
        fwd_durations.push(start.elapsed());

        if is_prefill {
            sentence = vec![prompt_tokens[i + 1]];
            continue;
        }

        // Greedy decode with repetition penalty
        let mut last_row = logits_data[logits_data.len() - VOCAB_SIZE..].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= repetition_penalty;
            } else {
                *logit *= repetition_penalty;
            }
        }
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        sentence = vec![next_token];
        seen_tokens.insert(next_token);

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        let decoded = tokenizer.decode(&[next_token], true).unwrap();
        print!("{}", decoded);
        std::io::stdout().flush().unwrap();
    }
    println!();

    // Benchmarks
    let decode_durations: Vec<_> = fwd_durations.iter().skip(prompt_len).collect();
    if decode_durations.len() > 2 {
        println!(
            "  TTFT: {:.2} ms",
            fwd_durations[..prompt_len]
                .iter()
                .sum::<Duration>()
                .as_secs_f64()
                * 1e3
        );
        println!(
            "  TPOT: {:.2} ms",
            (decode_durations.iter().skip(1).copied().sum::<Duration>()
                / (decode_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1_000.
        );
    }
}
