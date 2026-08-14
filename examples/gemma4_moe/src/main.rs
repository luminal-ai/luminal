// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after hours of search heap churn (hundreds of
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

const REPO_ID: &str = "google/gemma-4-26B-A4B";
const SEARCH_SEED: u64 = 0;

fn env_bool(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

/// Allocate user-owned buffers for all persistent state and register each
/// as BOTH the input and the output buffer (in-place aliasing). A step is
/// then exactly one execute: in-place candidates write through the alias,
/// materializing candidates pay a graph-visible copy priced by the search.
fn alias_persistent_state(
    runtime: &mut CudaRuntime,
    stream: &std::sync::Arc<luminal_cuda_lite::cudarc::driver::CudaStream>,
    seen_out: GraphTensor,
    seen_mask: GraphTensor,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    kv_cache: &KVCache,
    max_seq_len: usize,
) -> Vec<luminal_cuda_lite::cudarc::driver::CudaSlice<u8>> {
    use luminal_cuda_lite::cudarc::driver::DevicePtr;
    let mut owned = Vec::new();
    let mut alias = |runtime: &mut CudaRuntime, t_in: GraphTensor, t_out: GraphTensor, bytes| {
        let buf = stream.alloc_zeros::<u8>(bytes).unwrap();
        let ptr = buf.device_ptr(stream).0;
        unsafe {
            runtime.set_device_ptr(t_in, ptr, bytes);
            runtime.set_output_device_ptr(t_out, ptr, bytes);
        }
        owned.push(buf);
    };
    alias(
        runtime,
        seen_mask,
        seen_out,
        VOCAB_SIZE * std::mem::size_of::<f32>(),
    );
    for (layer, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let bytes = cache_bytes_for_layer(layer, max_seq_len);
        alias(runtime, kv_cache.k_caches[layer], *k_out, bytes);
        alias(runtime, kv_cache.v_caches[layer], *v_out, bytes);
    }
    owned
}

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 500;
    // 500 candidates leaves the decode search short of full fused-op
    // adoption on some runs (the full-adoption genome first appears past
    // candidate ~120 on average); 1000 closes the TPOT gap at ~2x compile
    // time, still well inside the CI window.
    let search_graphs = 500;
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
    let seen_mask_t = cx.named_tensor("seen_mask", VOCAB_SIZE);
    let new_token_t = cx.named_tensor("new_token", 1).as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let repetition_penalty: f32 = 1.05;
    let kv_cache = KVCache::new(&mut cx, max_seq_len);
    let (token_ids, seen_out, cache_outputs) = Gemma4MoE::init(&mut cx).forward_with_sampling(
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
    // Profile the context dim at a representative size, not the bucket
    // minimum: at c=16 attention cost differences (FlashInfer vs mask+softmax
    // kernels, fused vs unfused chains) are below profiling noise, so the
    // search cannot rank them. 512 makes size-dependent costs visible while
    // keeping per-candidate profiling cheap.
    let search_c = 512.min(max_seq_len);
    let compile_options = CompileOptions::default()
        .dim_buckets(
            's',
            &[
                DimBucket::new(1, 1),
                DimBucket::new(2, max_prefill).representative(search_s),
            ],
        )
        .search_dim('c', search_c)
        .search_graph_limit(search_graphs);

    println!("Loading weights...");
    // ~52 GB of weights leave room for the arena on an 80 GB A100 only after
    // release_pooled_memory() (below) reclaims what search profiling leaves in
    // the async allocator pool — without that trim even a 10 GiB arena OOMs.
    let mut runtime = CudaRuntime::initialize(stream.clone()).with_max_memory_gib(12);
    let weights_path = model_dir.join("model_combined_bf16_v1.safetensors");
    let phase = std::time::Instant::now();
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    println!("  weight load: {:.1}s", phase.elapsed().as_secs_f64());

    // Persistent state is user-owned and registered as both input and
    // output before compile: the search profiles candidates writing into
    // the real buffers, and deployment steps are a single execute.
    let persistent_buffers = alias_persistent_state(
        &mut runtime,
        &stream,
        seen_out,
        seen_mask_t,
        &cache_outputs,
        &kv_cache,
        max_seq_len,
    );

    println!("Compiling...");
    cx.set_dim('s', search_s);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(pos_ids, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_c as i32).collect::<Vec<_>>());
    runtime.set_data(new_token_t, vec![-1i32]);
    let mut rng = SmallRng::seed_from_u64(SEARCH_SEED);
    // Profiling timeouts use the CompileOptions defaults (60s candidate / 1s execution).
    runtime = cx.compile_with_rng(runtime, compile_options, &mut rng);

    // Search profiling leaves several GB cached in the async allocator pool;
    // reclaim it before the first real execute so the stitched-graph arena
    // allocation has room alongside the 52 GB of weights on an 80 GB A100.
    runtime.release_pooled_memory();

    // Pre-size the gather index buffer to its maximum so per-step set_data
    // reuses the same device pointer — growth reallocation would invalidate
    // the FlashInfer capture signatures and force per-step recaptures.
    runtime.set_data_with_capacity(
        gather_idx_t,
        Vec::<i32>::new(),
        max_seq_len * std::mem::size_of::<i32>(),
    );

    // Search profiling wrote into the persistent buffers — zero them
    // before real inference.
    for buf in &persistent_buffers {
        let mut view = unsafe {
            stream.upgrade_device_ptr::<u8>(
                luminal_cuda_lite::cudarc::driver::DevicePtr::device_ptr(buf, &stream).0,
                buf.len(),
            )
        };
        stream.memset_zeros(&mut view).unwrap();
        std::mem::forget(view);
    }

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut prev_seq: usize;
    let mut fwd_durations = vec![];
    let mut generated_token_ids = vec![];

    const EOS_TOKEN: u32 = 1;

    // Sampling runs on-device, and each execute's selected cache/seen outputs
    // are promoted back into the persistent input slots before the next step.
    // Per-step host I/O is one token id each way.
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
    prev_seq = prompt_tokens.len();

    let ids = runtime.get_i32(token_ids);
    let mut next_token = ids[prompt_tokens.len() - 1] as u32;
    let prefill_duration = prefill_start.elapsed();
    generated_token_ids.push(next_token);
    print!("{}", tokenizer.decode(&[next_token], true).unwrap());
    std::io::stdout().flush().unwrap();

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
        runtime.execute(&cx.dyn_map);

        prev_seq += 1;
        let ids = runtime.get_i32(token_ids);
        next_token = ids[0] as u32;
        generated_token_ids.push(next_token);

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
