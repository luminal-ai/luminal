mod hf;
mod model;

use hf::{WeightFormat, prepare_hf_model};
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use rand::{SeedableRng, rngs::StdRng};
use rustc_hash::FxHashSet;
use std::{env, io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const FP32_REPO_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";
const FP8_REPO_ID: &str = "nvidia/Llama-3.1-8B-Instruct-FP8";
const MAX_SEQ_LEN: usize = 4096;
const GEN_TOKENS: usize = 500;
const SEARCH_GRAPHS: usize = 500;
const SEARCH_TRIALS: usize = 1;
const SEARCH_KEEP_BEST: usize = 4;
const SEARCH_MEMORY_MIB: usize = 2048;
const SEARCH_SEED: u64 = 0;
const PROMPT: &str = "Explain what a neural network is in a paragraph.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LlamaWeightMode {
    Fp32,
    Fp8,
}

impl LlamaWeightMode {
    fn repo_id(self) -> &'static str {
        match self {
            Self::Fp32 => FP32_REPO_ID,
            Self::Fp8 => FP8_REPO_ID,
        }
    }

    fn weight_format(self) -> WeightFormat {
        match self {
            Self::Fp32 => WeightFormat::Fp32,
            Self::Fp8 => WeightFormat::Fp8,
        }
    }
}

fn print_usage(program: &str) {
    println!("Usage: {program} [--fp8]");
    println!();
    println!("  --fp8     Use {FP8_REPO_ID} with FP8 projection weights");
    println!("  -h,--help Show this help");
}

fn parse_args() -> LlamaWeightMode {
    let mut mode = LlamaWeightMode::Fp32;
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "llama".to_string());
    for arg in args {
        match arg.as_str() {
            "--fp8" => mode = LlamaWeightMode::Fp8,
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
    cache_roundtrip: Duration,
    sample: Duration,
}

fn sum_profiles<'a>(profiles: impl Iterator<Item = &'a StepProfile>) -> StepProfile {
    profiles.fold(StepProfile::default(), |mut acc, p| {
        acc.total += p.total;
        acc.set_inputs += p.set_inputs;
        acc.execute += p.execute;
        acc.get_logits += p.get_logits;
        acc.cache_roundtrip += p.cache_roundtrip;
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
        "  {label}: n={n}, avg={:.2} ms [set={:.2}, exec={:.2}, logits_dtoh={:.2}, cache={:.2}, sample={:.2}]",
        avg_ms(profile.total, n),
        avg_ms(profile.set_inputs, n),
        avg_ms(profile.execute, n),
        avg_ms(profile.get_logits, n),
        avg_ms(profile.cache_roundtrip, n),
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
    println!(
        "Host op summary ({label}): total={}, cublasLt={}, fp8_cublasLt={}, scaled_fp8_cublasLt={}",
        debug_ops.len(),
        cublaslt,
        fp8_cublaslt,
        scaled_fp8_cublaslt
    );
}

fn sample_greedy(logits_row: &[f32], seen: &FxHashSet<u32>, repetition_penalty: f32) -> u32 {
    let mut row = logits_row.to_vec();
    for &tok in seen {
        let logit = &mut row[tok as usize];
        if *logit > 0.0 {
            *logit /= repetition_penalty;
        } else {
            *logit *= repetition_penalty;
        }
    }
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

#[allow(clippy::too_many_arguments)]
fn run_model_step(
    cx: &mut Graph,
    runtime: &mut CudaRuntime,
    input: GraphTensor,
    q_pos_t: GraphTensor,
    scatter_idx_t: GraphTensor,
    gather_idx_t: GraphTensor,
    attn_mask_t: GraphTensor,
    logits: GraphTensor,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    tokens: &[u32],
    q_pos: &[i32],
    scatter_idx: &[i32],
    gather_idx: &[i32],
    attn_mask: &[f32],
) -> (Vec<f32>, StepProfile) {
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
    runtime.set_data(attn_mask_t, attn_mask.to_vec());
    profile.set_inputs = set_start.elapsed();

    let execute_start = std::time::Instant::now();
    runtime.execute(&cx.dyn_map);
    profile.execute = execute_start.elapsed();

    let logits_start = std::time::Instant::now();
    let logits_data = runtime.get_f32(logits);
    profile.get_logits = logits_start.elapsed();

    let cache_start = std::time::Instant::now();
    for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let k_buf = runtime.remove_buffer(*k_out);
        let v_buf = runtime.remove_buffer(*v_out);
        runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
        runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
    }
    profile.cache_roundtrip = cache_start.elapsed();
    profile.total = start.elapsed();

    (logits_data, profile)
}

fn causal_mask(q_pos: &[usize], context_len: usize) -> Vec<f32> {
    let mut mask = vec![-1e10f32; q_pos.len() * context_len];
    for (qi, &pos) in q_pos.iter().enumerate() {
        for ci in 0..context_len {
            if ci <= pos {
                mask[qi * context_len + ci] = 0.0;
            }
        }
    }
    mask
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
    let attn_mask_t = cx.named_tensor("attn_mask", ('s', 'c'));
    let kv_cache = KVCache::new(&mut cx, MAX_SEQ_LEN);
    let llama = match weight_mode {
        LlamaWeightMode::Fp32 => Llama::init(&mut cx),
        LlamaWeightMode::Fp8 => Llama::init_fp8(&mut cx),
    };
    let (logits, cache_outputs) = llama.forward(
        input,
        q_pos_t,
        scatter_idx_t,
        gather_idx_t,
        attn_mask_t,
        &kv_cache,
    );
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    cx.set_dim('s', 1);
    cx.set_dim('c', 1);

    println!("Building E-Graph...");
    let egraph_start = std::time::Instant::now();
    cx.build_search_space_with_options::<CudaRuntime>(
        BuildSearchSpaceOptions::new().max_memory_mib(SEARCH_MEMORY_MIB),
    );
    println!(
        "  E-Graph build: {:.2} s",
        egraph_start.elapsed().as_secs_f64()
    );

    println!("Loading weights...");
    let load_start = std::time::Instant::now();
    let mut runtime = CudaRuntime::initialize(stream);
    for weights_path in &prepared.weight_files {
        println!("  Loading {}", weights_path.display());
        runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    }
    println!("  Weight load: {:.2} s", load_start.elapsed().as_secs_f64());

    let cache_bytes = MAX_SEQ_LEN * KV_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    let compile_start = std::time::Instant::now();
    let max_prefill = (prompt_len + 16).next_power_of_two().min(MAX_SEQ_LEN);
    let search_s = 16.min(max_prefill).max(2);
    cx.set_dim_buckets(
        's',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_prefill).representative(search_s),
        ],
    );
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_s);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(q_pos_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(attn_mask_t, vec![0.0f32; search_s * search_s]);
    println!("  Search seed: {SEARCH_SEED}");
    println!("  Search trials: {SEARCH_TRIALS}");
    println!("  Search keep-best: {SEARCH_KEEP_BEST}");
    let mut rng = StdRng::seed_from_u64(SEARCH_SEED);
    runtime = cx.search_options(
        runtime,
        SearchOptions::new(SEARCH_GRAPHS)
            .trials(SEARCH_TRIALS)
            .keep_best(SEARCH_KEEP_BEST),
        &mut rng,
    );
    println!(
        "  Search/compile: {:.2} s",
        compile_start.elapsed().as_secs_f64()
    );
    print_host_op_summary(&runtime, "post-compile active bucket");

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    let mut context_len = 0usize;
    let mut fwd_durations = vec![];
    let mut step_profiles = vec![];
    let mut seen_tokens = FxHashSet::default();
    let repetition_penalty: f32 = 1.05;

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
        let mask = causal_mask(&positions, prompt_len);
        let (logits_data, mut profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            attn_mask_t,
            logits,
            &kv_cache,
            &cache_outputs,
            &prompt_tokens,
            &q_pos,
            &scatter_idx,
            &gather_idx,
            &mask,
        );
        print_host_op_summary(&runtime, "after prefill");
        context_len = prompt_len;

        let sample_start = std::time::Instant::now();
        let token = sample_greedy(
            &logits_data[logits_data.len() - VOCAB_SIZE..],
            &seen_tokens,
            repetition_penalty,
        );
        profile.sample = sample_start.elapsed();
        profile.total += profile.sample;
        seen_tokens.insert(token);
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

        let (logits_data, mut profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            attn_mask_t,
            logits,
            &kv_cache,
            &cache_outputs,
            &[current_token],
            &[context_len as i32],
            &[context_len as i32],
            &(0..=context_len as i32).collect::<Vec<_>>(),
            &causal_mask(&[context_len], context_len + 1),
        );
        if generated == 1 {
            print_host_op_summary(&runtime, "after first decode");
        }
        context_len += 1;

        let sample_start = std::time::Instant::now();
        let token = sample_greedy(
            &logits_data[logits_data.len() - VOCAB_SIZE..],
            &seen_tokens,
            repetition_penalty,
        );
        profile.sample = sample_start.elapsed();
        profile.total += profile.sample;
        seen_tokens.insert(token);
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
