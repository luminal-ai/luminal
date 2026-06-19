// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after hours of search heap churn (hundreds of
// thousands of compiles). jemalloc built with unprefixed symbols
// interposes malloc for the whole process, including dlopened CUDA
// libraries like libnvrtc — a Rust-only global allocator would not.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod hf;
mod model;

use hf::{WeightFormat, prepare_hf_model};
use luminal::{dtype::DType, prelude::*};
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use rand::{SeedableRng, rngs::StdRng};
use std::{env, io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const BF16_REPO_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
const FP8_REPO_ID: &str = "nvidia/Llama-3.1-8B-Instruct-FP8";
const MAX_SEQ_LEN: usize = 4096;
const GEN_TOKENS: usize = 500;
const SEARCH_GRAPHS: usize = 500;
const SEARCH_TRIALS: usize = 10;
const SEARCH_KEEP_BEST: usize = 4;
const SEARCH_MEMORY_MIB: usize = 2048;
const SEARCH_SEED: u64 = 0;
const PROMPT: &str = "Explain what a neural network is in a paragraph.";
const REPETITION_PENALTY: f32 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaWeightMode {
    Bf16,
    Fp8,
}

impl LlamaWeightMode {
    fn repo_id(self) -> &'static str {
        match self {
            Self::Bf16 => BF16_REPO_ID,
            Self::Fp8 => FP8_REPO_ID,
        }
    }

    fn weight_format(self) -> WeightFormat {
        match self {
            Self::Bf16 => WeightFormat::Bf16,
            Self::Fp8 => WeightFormat::Fp8,
        }
    }

    fn kv_dtype(self) -> luminal::dtype::DType {
        luminal::dtype::DType::Bf16
    }

    fn kv_element_bytes(self) -> usize {
        2
    }
}

fn print_usage(program: &str) {
    println!("Usage: {program} [--bf16|--fp8]");
    println!();
    println!("  --bf16    Native bf16 weights, activations and KV cache (default)");
    println!("  --fp8     FP8 linear weights with bf16 activations and KV cache");
    println!("  -h,--help Show this help");
}

fn parse_args() -> LlamaWeightMode {
    let mut mode = LlamaWeightMode::Bf16;
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "llama".to_string());
    for arg in args {
        match arg.as_str() {
            "--fp8" => mode = LlamaWeightMode::Fp8,
            "--bf16" => mode = LlamaWeightMode::Bf16,
            "-h" | "--help" => {
                print_usage(&program);
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {arg}");
                print_usage(&program);
                std::process::exit(2);
            }
        }
    }
    mode
}

fn llama3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

#[derive(Default, Clone)]
struct StepProfile {
    total: Duration,
    set_inputs: Duration,
    execute: Duration,
    get_logits: Duration,
    sample: Duration,
}

fn sum_profiles<'a>(profiles: impl Iterator<Item = &'a StepProfile>) -> StepProfile {
    profiles.fold(StepProfile::default(), |mut acc, p| {
        acc.total += p.total;
        acc.set_inputs += p.set_inputs;
        acc.execute += p.execute;
        acc.get_logits += p.get_logits;
        acc.sample += p.sample;
        acc
    })
}

fn avg_ms(duration: Duration, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1e3 / n as f64
    }
}

fn print_profile(label: &str, profile: &StepProfile, n: usize) {
    println!(
        "  {label}: n={n}, avg={:.2} ms [set={:.2}, exec={:.2}, logits_dtoh={:.2}, sample={:.2}]",
        avg_ms(profile.total, n),
        avg_ms(profile.set_inputs, n),
        avg_ms(profile.execute, n),
        avg_ms(profile.get_logits, n),
        avg_ms(profile.sample, n),
    );
}

fn print_host_op_summary(runtime: &CudaRuntime, label: &str) {
    let host_ops = runtime.host_ops();
    let debug_ops = host_ops
        .iter()
        .map(|op| format!("{op:?}"))
        .collect::<Vec<_>>();
    let cublaslt = debug_ops
        .iter()
        .filter(|op| op.contains("CuBlasLt"))
        .count();
    let fp8_cublaslt = debug_ops
        .iter()
        .filter(|op| {
            op.contains("CuBlasLt") && (op.contains("a_dtype: F8") || op.contains("b_dtype: F8"))
        })
        .count();
    let scaled_fp8_cublaslt = debug_ops
        .iter()
        .filter(|op| {
            op.contains("CuBlasLt")
                && (op.contains("a_dtype: F8") || op.contains("b_dtype: F8"))
                && op.contains("a_scale_input: true")
                && op.contains("b_scale_input: true")
        })
        .count();
    let flashinfer = debug_ops
        .iter()
        .filter(|op| op.contains("FlashInferAttention"))
        .count();
    println!(
        "Host op summary ({label}): total={}, cublasLt={}, fp8_cublasLt={}, scaled_fp8_cublasLt={}, flashinfer={}",
        debug_ops.len(),
        cublaslt,
        fp8_cublaslt,
        scaled_fp8_cublaslt,
        flashinfer
    );
}

fn print_cuda_graph_summary(runtime: &CudaRuntime, label: &str) {
    if std::env::var_os("LUMINAL_LLAMA_CUDA_GRAPH_SUMMARY").is_none() {
        return;
    }

    let summaries = runtime.debug_cuda_graph_summaries();
    let total_kernels: usize = summaries.iter().map(|s| s.n_kernels).sum();
    let total_cublaslt: usize = summaries.iter().map(|s| s.n_cublaslt).sum();
    let total_flashinfer: usize = summaries.iter().map(|s| s.n_flashinfer).sum();
    let total_steps: usize = summaries.iter().map(|s| s.n_steps).sum();
    println!(
        "CUDA graph summary ({label}): graphs={}, kernels={}, cublasLt={}, flashinfer={}, steps={}",
        summaries.len(),
        total_kernels,
        total_cublaslt,
        total_flashinfer,
        total_steps
    );
    for (idx, summary) in summaries.iter().enumerate() {
        let dep_sum: usize = summary.step_dependency_counts.iter().sum();
        let dep_max = summary
            .step_dependency_counts
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        println!(
            "  graph[{idx}]: kernels={}, cublasLt={}, flashinfer={}, prepared_cublasLt={}, steps={}, absorbed_host_nodes={}, dep_sum={}, dep_max={}, flashinfer_inputs={:?}, flashinfer_recaptures={:?}",
            summary.n_kernels,
            summary.n_cublaslt,
            summary.n_flashinfer,
            summary.n_cublaslt_prepared,
            summary.n_steps,
            summary.absorbed_host_nodes.len(),
            dep_sum,
            dep_max,
            summary.flashinfer_input_counts,
            summary.flashinfer_recapture_counts,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_model_step(
    cx: &mut Graph,
    runtime: &mut CudaRuntime,
    input: GraphTensor,
    q_pos_t: GraphTensor,
    scatter_idx_t: GraphTensor,
    gather_idx_t: GraphTensor,
    new_token_t: GraphTensor,
    token_ids: GraphTensor,
    tokens: &[u32],
    q_pos: &[i32],
    scatter_idx: &[i32],
    gather_idx: &[i32],
    prev_sampled: i32,
) -> (i32, StepProfile) {
    let start = std::time::Instant::now();
    let seq_len = tokens.len();
    let mut profile = StepProfile::default();

    cx.set_dim('s', seq_len);
    cx.set_dim('c', gather_idx.len());

    let set_start = std::time::Instant::now();
    runtime.set_data(input, tokens.iter().map(|t| *t as i32).collect::<Vec<_>>());
    runtime.set_data(q_pos_t, q_pos.to_vec());
    runtime.set_data(scatter_idx_t, scatter_idx.to_vec());
    runtime.set_data(gather_idx_t, gather_idx.to_vec());
    runtime.set_data(new_token_t, vec![prev_sampled]);
    profile.set_inputs = set_start.elapsed();

    let execute_start = std::time::Instant::now();
    runtime.execute(&cx.dyn_map);
    profile.execute = execute_start.elapsed();

    let logits_start = std::time::Instant::now();
    // Sampling runs on-device (penalty + argmax); fetch one id per row and
    // index from the START of the buffer (it may exceed the current s).
    let ids = runtime.get_i32(token_ids);
    let sampled = ids[seq_len - 1];
    profile.get_logits = logits_start.elapsed();

    profile.total = start.elapsed();

    (sampled, profile)
}

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let weight_mode = parse_args();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let prepared = prepare_hf_model(weight_mode.repo_id(), weight_mode.weight_format())
        .expect("Failed to prepare model");
    println!("Using model: {}", weight_mode.repo_id());
    println!("Using model directory: {}", prepared.model_dir.display());

    let tokenizer = Tokenizer::from_file(prepared.model_dir.join("tokenizer.json")).unwrap();
    let chat_prompt = llama3_chat_prompt(PROMPT);
    let prompt_tokens = tokenizer
        .encode(chat_prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let prompt_len = prompt_tokens.len();

    // Build graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos_t = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let seen_mask_t = cx.named_tensor("seen_mask", VOCAB_SIZE);
    let new_token_t = cx.named_tensor("new_token", 1).as_dtype(DType::Int);
    let kv_cache = KVCache::new_dtype(&mut cx, MAX_SEQ_LEN, weight_mode.kv_dtype());
    let llama = match weight_mode {
        LlamaWeightMode::Bf16 => Llama::init_bf16(&mut cx),
        LlamaWeightMode::Fp8 => Llama::init_fp8(&mut cx),
    };
    let (token_ids, seen_out, cache_outputs) = llama.forward_with_sampling(
        input,
        q_pos_t,
        scatter_idx_t,
        gather_idx_t,
        &kv_cache,
        seen_mask_t,
        new_token_t,
        REPETITION_PENALTY,
    );
    let token_ids = token_ids.output();
    seen_out.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    cx.set_dim('s', 1);
    cx.set_dim('c', 1);
    let max_prefill = (prompt_len + 16).next_power_of_two().min(MAX_SEQ_LEN);
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
            &[DimBucket::new(1, MAX_SEQ_LEN).representative(search_s)],
        );

    println!("Building E-Graph...");
    let egraph_start = std::time::Instant::now();
    cx.build_search_space::<CudaRuntime>(build_options);
    println!(
        "  E-Graph build: {:.2} s",
        egraph_start.elapsed().as_secs_f64()
    );

    println!("Loading weights...");
    let load_start = std::time::Instant::now();
    let mut runtime = CudaRuntime::initialize(stream).with_max_memory_mib(SEARCH_MEMORY_MIB);
    for weights_path in &prepared.weight_files {
        println!("  Loading {}", weights_path.display());
        runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    }
    println!("  Weight load: {:.2} s", load_start.elapsed().as_secs_f64());

    let cache_bytes = MAX_SEQ_LEN * KV_DIM * weight_mode.kv_element_bytes();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }
    runtime.set_zeros(seen_mask_t, VOCAB_SIZE * std::mem::size_of::<f32>());

    println!("Compiling...");
    let compile_start = std::time::Instant::now();
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_s);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(new_token_t, vec![-1i32]);
    runtime.set_data(q_pos_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    let search_seed = std::env::var("LUMINAL_LLAMA_SEARCH_SEED")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(SEARCH_SEED);
    println!("  Search seed: {search_seed}");
    println!("  Search trials: {SEARCH_TRIALS}");
    println!("  Search keep-best: {SEARCH_KEEP_BEST}");
    let mut rng = StdRng::seed_from_u64(search_seed);
    let search_options = CompileOptions::default()
        .search_graph_limit(SEARCH_GRAPHS)
        .trials(SEARCH_TRIALS)
        .keep_best(SEARCH_KEEP_BEST);
    runtime = cx.search_with_rng(runtime, search_options, &mut rng);
    println!(
        "  Search/compile: {:.2} s",
        compile_start.elapsed().as_secs_f64()
    );
    print_host_op_summary(&runtime, "post-compile active bucket");
    print_cuda_graph_summary(&runtime, "post-compile active bucket");

    runtime.set_data_with_capacity(
        gather_idx_t,
        Vec::<i32>::new(),
        MAX_SEQ_LEN * std::mem::size_of::<i32>(),
    );

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }
    runtime.set_zeros(seen_mask_t, VOCAB_SIZE * std::mem::size_of::<f32>());

    let mut context_len = 0usize;
    let mut fwd_durations = vec![];
    let mut step_profiles = vec![];
    const EOS_TOKEN: u32 = 128009;
    const STOP_TOKEN: u32 = 128001;

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, GEN_TOKENS
    );

    let mut generated = 0usize;
    let mut next_token = None;
    if GEN_TOKENS > 0 && prompt_len > 0 {
        let positions: Vec<usize> = (0..prompt_len).collect();
        let q_pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let scatter_idx = q_pos.clone();
        let gather_idx = q_pos.clone();
        let (sampled, mut profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            new_token_t,
            token_ids,
            &prompt_tokens,
            &q_pos,
            &scatter_idx,
            &gather_idx,
            -1, // nothing sampled yet — the seen-mask scatter skips -1
        );
        print_host_op_summary(&runtime, "after prefill");
        print_cuda_graph_summary(&runtime, "after prefill");
        context_len = prompt_len;

        let sample_start = std::time::Instant::now();
        let token = sampled as u32;
        profile.sample = sample_start.elapsed();
        profile.total += profile.sample;
        next_token = Some(token);
        generated = 1;

        fwd_durations.push(profile.total);
        step_profiles.push(profile);

        if token != EOS_TOKEN && token != STOP_TOKEN {
            let decoded = tokenizer.decode(&[token], true).unwrap();
            print!("{}", decoded);
            std::io::stdout().flush().unwrap();
        }
    }

    while generated < GEN_TOKENS {
        let current_token = match next_token {
            Some(token) if token != EOS_TOKEN && token != STOP_TOKEN => token,
            _ => break,
        };

        // Optional nsys capture window over steady-state decode:
        // `nsys profile -c cudaProfilerApi` + LUMINAL_NSYS_DECODE=1.
        if std::env::var_os("LUMINAL_NSYS_DECODE").is_some() {
            if generated == 5 {
                luminal_cuda_lite::cudarc::driver::safe::profiler_start().unwrap();
            } else if generated == 55 {
                luminal_cuda_lite::cudarc::driver::safe::profiler_stop().unwrap();
            }
        }
        let (sampled, mut profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            new_token_t,
            token_ids,
            &[current_token],
            &[context_len as i32],
            &[context_len as i32],
            &(0..=context_len as i32).collect::<Vec<_>>(),
            current_token as i32,
        );
        if generated == 1 {
            print_host_op_summary(&runtime, "after first decode");
            print_cuda_graph_summary(&runtime, "after first decode");
        }
        context_len += 1;

        let sample_start = std::time::Instant::now();
        let token = sampled as u32;
        profile.sample = sample_start.elapsed();
        profile.total += profile.sample;
        next_token = Some(token);
        generated += 1;

        fwd_durations.push(profile.total);
        step_profiles.push(profile);

        if token == EOS_TOKEN || token == STOP_TOKEN {
            break;
        }

        let decoded = tokenizer.decode(&[token], true).unwrap();
        print!("{}", decoded);
        std::io::stdout().flush().unwrap();
    }
    println!();

    // Benchmarks
    let prefill_steps = usize::from(!step_profiles.is_empty());
    let ttft_steps = prefill_steps;
    let decode_durations: Vec<_> = fwd_durations.iter().skip(ttft_steps).collect();
    if decode_durations.len() > 2 {
        println!(
            "  TTFT: {:.2} ms",
            fwd_durations
                .iter()
                .take(ttft_steps)
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
    println!("\nProfile breakdown:");
    let decode_steps = step_profiles.len().saturating_sub(ttft_steps);
    let prefill = sum_profiles(step_profiles.iter().take(prefill_steps));
    let decode = sum_profiles(step_profiles.iter().skip(ttft_steps));
    print_profile("batched prefill", &prefill, prefill_steps);
    print_profile("steady decode", &decode, decode_steps);
    if ttft_steps > 0 {
        let ttft = sum_profiles(step_profiles.iter().take(ttft_steps));
        println!(
            "  TTFT attribution: {:.2} ms [batched prefill {:.2}]",
            ttft.total.as_secs_f64() * 1e3,
            prefill.total.as_secs_f64() * 1e3,
        );
    }
}
