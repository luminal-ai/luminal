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
use model::*;
use rand::{SeedableRng, rngs::SmallRng};
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;

const REPO_ID: &str = "Qwen/Qwen3-30B-A3B";
const SEARCH_SEED: u64 = 0;

fn qwen3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 500;
    let search_graphs = 200;
    let prompt = "What is the capital of France?";

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let chat_prompt = qwen3_chat_prompt(prompt);
    let prompt_tokens = tokenizer
        .encode(chat_prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let seen_mask_t = cx.named_tensor("seen_mask", VOCAB_SIZE);
    let new_token_t = cx.named_tensor("new_token", 1).as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let repetition_penalty: f32 = 1.05;
    let (token_ids, seen_out, cache_outputs) = Qwen3MoE::init(&mut cx).forward_with_sampling(
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
    let max_prefill = (prompt_tokens.len() + 16)
        .next_power_of_two()
        .min(max_seq_len);
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
    let mut runtime = CudaRuntime::initialize(stream).with_max_memory_gib(20);
    let weights_path = model_dir.join("model_combined_bf16_v1.safetensors");
    let phase = std::time::Instant::now();
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    println!("  weight load: {:.1}s", phase.elapsed().as_secs_f64());

    let cache_bytes = max_seq_len * KV_DIM * 2; // bf16
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_s);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(pos_ids, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(new_token_t, vec![-1i32]);
    runtime.set_zeros(seen_mask_t, VOCAB_SIZE * std::mem::size_of::<f32>());
    let search_seed = std::env::var("LUMINAL_SEARCH_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(SEARCH_SEED);
    println!("Search seed: {search_seed}");
    let mut rng = SmallRng::seed_from_u64(search_seed);
    let search_options = CompileOptions::default().search_graph_limit(search_graphs);
    runtime = cx.search_with_rng(runtime, search_options, &mut rng);

    // Reclaim memory the async allocator pool retains from search profiling so
    // the stitched-graph arena fits alongside the weights (see gemma4_moe).
    runtime.release_pooled_memory();

    // Pre-size the gather index buffer to its maximum so per-step set_data
    // reuses the same device pointer — otherwise every growth reallocates,
    // the pointer change invalidates the FlashInfer capture signatures, and
    // all attention islands recapture every step (~6.8 ms/step measured).
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

    println!("Prompt: {prompt}");
    print!("Response: ");
    std::io::stdout().flush().unwrap();

    let mut prev_seq: usize;
    let mut fwd_durations = vec![];

    const EOS_TOKEN: u32 = 151645; // <|im_end|>
    const STOP_TOKEN: u32 = 151643; // <|endoftext|>

    // KV caches update in place (the cache scatters write through to the
    // cache input buffers via the in-place alias machinery) and sampling
    // runs on-device — per-step host I/O is one token id each way.
    let prefill_start = std::time::Instant::now();
    let plen = prompt_tokens.len();
    cx.set_dim('s', plen);
    cx.set_dim('c', plen);
    runtime.set_data(
        input,
        prompt_tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
    );
    runtime.set_data(pos_ids, (0..plen as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..plen as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..plen as i32).collect::<Vec<_>>());
    runtime.set_data(new_token_t, vec![-1i32]);
    runtime.execute(&cx.dyn_map);
    prev_seq = plen;

    // One sampled id per row; index from the START of the (possibly larger)
    // buffer.
    let ids = runtime.get_i32(token_ids);
    let mut next_token = ids[prompt_tokens.len() - 1] as u32;
    let prefill_duration = prefill_start.elapsed();
    print!("{}", tokenizer.decode(&[next_token], true).unwrap());
    std::io::stdout().flush().unwrap();

    // Decode loop
    let mut t_set = Duration::ZERO;
    let mut t_exec = Duration::ZERO;
    let mut t_get = Duration::ZERO;
    #[allow(clippy::explicit_counter_loop)]
    for _ in 1..gen_tokens {
        let start = std::time::Instant::now();
        cx.set_dim('s', 1);
        cx.set_dim('c', prev_seq + 1);
        runtime.set_data(input, vec![next_token as i32]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);
        runtime.set_data(scatter_idx_t, vec![prev_seq as i32]);
        runtime.set_data(gather_idx_t, (0..=prev_seq as i32).collect::<Vec<_>>());
        runtime.set_data(new_token_t, vec![next_token as i32]);
        t_set += start.elapsed();
        let e = std::time::Instant::now();
        runtime.execute(&cx.dyn_map);
        t_exec += e.elapsed();

        prev_seq += 1;
        let g = std::time::Instant::now();
        let ids = runtime.get_i32(token_ids);
        t_get += g.elapsed();
        next_token = ids[0] as u32;

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        print!("{}", tokenizer.decode(&[next_token], true).unwrap());
        std::io::stdout().flush().unwrap();
        fwd_durations.push(start.elapsed());
    }
    println!();

    let n = fwd_durations.len().max(1) as u32;
    eprintln!(
        "  step breakdown: set={:.2}ms exec={:.2}ms get={:.2}ms",
        (t_set / n).as_secs_f64() * 1e3,
        (t_exec / n).as_secs_f64() * 1e3,
        (t_get / n).as_secs_f64() * 1e3
    );

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
