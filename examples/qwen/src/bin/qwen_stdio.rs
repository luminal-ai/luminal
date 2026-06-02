//! Apples-to-apples bench harness for Qwen3-4B, mirroring
//! luminal-benchmarks `rust/crates/qwen3_4b/src/main.rs`.
//!
//! Reuses the upstream qwen lib (model.rs/hf.rs byte-identical) and the same
//! graph-build / weight-load / search-compile sequence as run_qwen, then a
//! `serve` loop reads one prompt per stdin line and emits each generated token
//! as `TOK\t<text>` to stdout (no internal timing — the Python driver times
//! token *arrivals*, exactly like the benchmark harness). EOQ closes a prompt.
//!
//! Build: cargo run --release -p qwen --bin qwen_stdio --features cuda
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use qwen::hf::prepare_hf_model;
use qwen::model::*;
use std::io::{BufRead, Write};
use tokenizers::Tokenizer;

const REPO_ID: &str = "Qwen/Qwen3-4B";
const MAX_SEQ_LEN: usize = 4096;
const EOS_TOKEN: u32 = 151645;
const STOP_TOKEN: u32 = 151643;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i as u32).unwrap_or(0)
}

fn qwen3_chat_prompt(p: &str) -> String {
    format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn emit_tok(text: &str) {
    let mut out = std::io::stdout().lock();
    // escape newlines/tabs so one token == one line
    let esc: String = text.chars().flat_map(|c| match c {
        '\\' => "\\\\".chars().collect::<Vec<_>>(),
        '\t' => "\\t".chars().collect(),
        '\n' => "\\n".chars().collect(),
        '\r' => "\\r".chars().collect(),
        _ => vec![c],
    }).collect();
    let _ = writeln!(out, "TOK\t{esc}");
    let _ = out.flush();
}

fn main() {
    let gen_tokens = env_usize("GEN_TOKENS", 100);
    let search_graphs = env_usize("SEARCH_GRAPHS", 500);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let model_dir = prepare_hf_model(REPO_ID).expect("prepare model");
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let max_prefill = 512usize.min(MAX_SEQ_LEN);
    let search_s = 16.min(max_prefill).max(2);

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, MAX_SEQ_LEN, LAYERS);
    let (logits, cache_outputs) = Qwen::init(&mut cx, LAYERS).forward(input, token_ids, &kv_cache);
    let logits = logits.output();
    for (k, v) in &cache_outputs {
        k.output();
        v.output();
    }

    let mut compile_options = CompileOptions::default().dim_buckets(
        's',
        &[DimBucket::new(1, 1), DimBucket::new(2, max_prefill).representative(search_s)],
    );
    let max_decode_p = MAX_SEQ_LEN.saturating_sub(1);
    compile_options = compile_options.dim_buckets(
        'p',
        &[DimBucket::new(0, 0), DimBucket::new(1, max_decode_p).representative(64)],
    );

    eprintln!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>(compile_options);

    eprintln!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let cache_bytes = N_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    eprintln!("Compiling...");
    cx.set_dim('s', search_s);
    cx.set_dim('p', 0);
    runtime.set_data(input.id, vec![1; search_s]);
    runtime.set_data(token_ids.id, (0..search_s as i32).collect::<Vec<_>>());
    let search_options = CompileOptions::default().search_graph_limit(search_graphs);
    runtime = cx.search(runtime, search_options);

    // Reset KV cache after compile (search writes profiling data into them).
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    // READY signals the driver that init/compile is done.
    {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "READY");
        let _ = out.flush();
    }

    let stdin = std::io::stdin();
    let mut handle = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        if handle.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let prompt = line.trim_end().to_string();
        if prompt.is_empty() {
            continue;
        }
        run_prompt(&mut cx, &mut runtime, &tokenizer, input, token_ids, logits,
                   &kv_cache, &cache_outputs, cache_bytes, &prompt, gen_tokens);
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "EOQ");
        let _ = out.flush();
    }
}

#[allow(clippy::too_many_arguments)]
fn run_prompt(
    cx: &mut Graph,
    runtime: &mut CudaRuntime,
    tokenizer: &Tokenizer,
    input: GraphTensor,
    token_ids: GraphTensor,
    logits: GraphTensor,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    cache_bytes: usize,
    prompt: &str,
    gen_tokens: usize,
) {
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(qwen3_chat_prompt(prompt).as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    if prompt_tokens.is_empty() || gen_tokens == 0 {
        return;
    }
    let prompt_len = prompt_tokens.len().min(MAX_SEQ_LEN);

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    // Prefill
    cx.set_dim('s', prompt_len);
    cx.set_dim('p', 0);
    runtime.set_data(input.id, prompt_tokens[..prompt_len].iter().map(|t| *t as i32).collect::<Vec<_>>());
    runtime.set_data(token_ids.id, (0..prompt_len as i32).collect::<Vec<_>>());
    runtime.execute(&cx.dyn_map);
    let logits_data = runtime.get_f32(logits.id);
    for (li, (k, v)) in cache_outputs.iter().enumerate() {
        let kb = runtime.remove_buffer(k.id);
        let vb = runtime.remove_buffer(v.id);
        runtime.set_buffer(kv_cache.k_caches[li].id, kb);
        runtime.set_buffer(kv_cache.v_caches[li].id, vb);
    }
    let row = (prompt_len - 1) * VOCAB_SIZE;
    let mut next = argmax(&logits_data[row..row + VOCAB_SIZE]);
    let mut prev_seq = prompt_len;
    let mut emitted = 0usize;

    while emitted < gen_tokens {
        if next != EOS_TOKEN && next != STOP_TOKEN {
            emit_tok(&tokenizer.decode(&[next], true).unwrap());
        }
        emitted += 1;
        if next == EOS_TOKEN || next == STOP_TOKEN || prev_seq >= MAX_SEQ_LEN {
            break;
        }
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);
        runtime.set_data(input.id, vec![next as i32]);
        runtime.set_data(token_ids.id, vec![prev_seq as i32]);
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits.id);
        for (li, (k, v)) in cache_outputs.iter().enumerate() {
            let kb = runtime.remove_buffer(k.id);
            let vb = runtime.remove_buffer(v.id);
            runtime.set_buffer(kv_cache.k_caches[li].id, kb);
            runtime.set_buffer(kv_cache.v_caches[li].id, vb);
        }
        prev_seq += 1;
        next = argmax(&logits_data[logits_data.len() - VOCAB_SIZE..]);
    }
}
