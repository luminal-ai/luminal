// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after heavy search heap churn (hundreds of
// thousands of compiles). jemalloc built with unprefixed symbols
// interposes malloc for the whole process, including dlopened CUDA
// libraries like libnvrtc — a Rust-only global allocator would not.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use rand::{SeedableRng, rngs::SmallRng};
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "unsloth/gemma-3-4b-it";
const SEARCH_SEED: u64 = 0;

fn gemma3_chat_prompt(user_prompt: &str) -> String {
    format!("<bos><start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n")
}

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 500;
    let search_graphs = 500;
    let prompt = "Explain what a neural network is in simple terms:";

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let chat_prompt = gemma3_chat_prompt(prompt);
    let prompt_tokens = tokenizer
        .encode(chat_prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let prompt_len = prompt_tokens.len();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let seen_mask_t = cx.named_tensor("seen_mask", VOCAB_SIZE);
    let new_token_t = cx.named_tensor("new_token", 1).as_dtype(DType::Int);
    let repetition_penalty: f32 = 1.05;
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let (token_ids, seen_out, cache_outputs) = Gemma::init(&mut cx).forward_with_sampling(
        input,
        pos_ids,
        scatter_idx_t,
        gather_idx_t,
        &kv_cache,
        seen_mask_t,
        new_token_t,
        repetition_penalty,
    );
    let token_ids = token_ids.output();
    seen_out.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    let max_prefill = (prompt_len + 16).next_power_of_two().min(max_seq_len);
    let search_s = 16.min(max_prefill).max(2);
    let build_options = CompileOptions::default()
        .dim_buckets(
            's',
            &[
                DimBucket::new(1, 1),
                DimBucket::new(2, max_prefill).representative(search_s),
            ],
        )
        .dim_buckets(
            'c',
            &[DimBucket::new(1, max_seq_len).representative(search_s)],
        );

    println!("Building E-Graph...");
    let phase = std::time::Instant::now();
    cx.build_search_space::<CudaRuntime>(build_options);
    println!("  e-graph build: {:.1}s", phase.elapsed().as_secs_f64());

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream).with_max_memory_mib(2048);
    let weights_path = model_dir.join("model_combined_bf16_v1.safetensors");
    let phase = std::time::Instant::now();
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    println!("  weight load: {:.1}s", phase.elapsed().as_secs_f64());

    let cache_bytes = cache_bytes(max_seq_len);
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }
    runtime.set_zeros(seen_mask_t, VOCAB_SIZE * std::mem::size_of::<f32>());

    println!("Compiling...");
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_s);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(pos_ids, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(new_token_t, vec![-1i32]);
    let mut rng = SmallRng::seed_from_u64(SEARCH_SEED);
    let search_options = CompileOptions::default().search_graph_limit(search_graphs);
    runtime = cx.search_with_rng(runtime, search_options, &mut rng);

    // Reclaim memory left in the async allocator pool by search profiling
    // before the first real execute.
    runtime.release_pooled_memory();

    // Pre-size the gather index buffer to its maximum so per-step set_data
    // reuses the same device pointer — growth reallocation would invalidate
    // the FlashInfer capture signatures and force per-step recaptures.
    runtime.set_data_with_capacity(
        gather_idx_t,
        Vec::<i32>::new(),
        max_seq_len * std::mem::size_of::<i32>(),
    );

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }
    runtime.set_zeros(seen_mask_t, VOCAB_SIZE * std::mem::size_of::<f32>());

    const EOS_TOKEN: u32 = 1; // <eos>
    const STOP_TOKEN: u32 = 106; // <end_of_turn>

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, gen_tokens
    );

    let mut prev_seq: usize;
    let mut fwd_durations = vec![];

    // Prefill: process the whole prompt in one tick; sampling runs on-device.
    let prefill_start = std::time::Instant::now();
    cx.set_dim('s', prompt_len);
    cx.set_dim('c', prompt_len);
    runtime.set_data(
        input,
        prompt_tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
    );
    runtime.set_data(pos_ids, (0..prompt_len as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..prompt_len as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..prompt_len as i32).collect::<Vec<_>>());
    runtime.set_data(new_token_t, vec![-1i32]);
    runtime.execute(&cx.dyn_map);
    prev_seq = prompt_len;

    let ids = runtime.get_i32(token_ids);
    let mut next_token = ids[prompt_len - 1] as u32;
    let prefill_duration = prefill_start.elapsed();
    let mut generated = 1usize;

    if next_token != EOS_TOKEN && next_token != STOP_TOKEN {
        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
    }

    while generated < gen_tokens {
        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('c', prev_seq + 1);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.set_data(scatter_idx_t, vec![prev_seq as i32]);
        runtime.set_data(gather_idx_t, (0..=prev_seq as i32).collect::<Vec<_>>());
        runtime.set_data(new_token_t, vec![next_token as i32]);
        runtime.execute(&cx.dyn_map);

        prev_seq += 1;
        let ids = runtime.get_i32(token_ids);
        next_token = ids[0] as u32;
        generated += 1;

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Benchmarks
    println!(
        "  TTFT: {:.2} ms ({} prompt tokens)",
        prefill_duration.as_secs_f64() * 1e3,
        prompt_len
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
