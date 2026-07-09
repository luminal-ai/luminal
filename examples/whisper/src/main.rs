// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after heavy search heap churn (hundreds of
// thousands of compiles). jemalloc built with unprefixed symbols
// interposes malloc for the whole process, including dlopened CUDA
// libraries like libnvrtc — a Rust-only global allocator would not.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod audio;
mod hf;
mod model;

use audio::{
    N_FRAMES, N_MELS, N_SAMPLES, load_wav, load_wav_bytes, log_mel_spectrogram, pad_or_trim,
};
use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use model::*;
use std::{io::Write, time::Instant};
use tokenizers::Tokenizer;

const REPO_ID: &str = "openai/whisper-tiny.en";

/// Bundled JFK sample (16 kHz mono PCM WAV, ~11 seconds) so the example runs out of the box
/// without needing a local audio file.
const DEFAULT_AUDIO_BYTES: &[u8] = include_bytes!("../assets/jfk.wav");

fn main() {
    let max_target_pos = N_TEXT_CTX; // 448
    let gen_tokens = 200;
    let search_graphs = 50;
    let audio_path = std::env::args().nth(1);

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let audio = match audio_path.as_deref() {
        Some(path) => {
            println!("Loading audio: {path}");
            load_wav(path).expect("failed to load audio")
        }
        None => {
            println!("Using bundled JFK sample audio");
            load_wav_bytes(DEFAULT_AUDIO_BYTES).expect("failed to decode bundled audio")
        }
    };
    let audio = pad_or_trim(&audio, N_SAMPLES);
    println!("Computing log-mel spectrogram...");
    let mel_data = log_mel_spectrogram(&audio, N_MELS);
    assert_eq!(mel_data.len(), N_MELS * N_FRAMES);

    // Build graph
    let mut cx = Graph::default();
    let mel_tensor = cx.named_tensor("mel", (N_MELS, N_FRAMES)).persist();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, max_target_pos);

    let whisper = Whisper::init(&mut cx);
    let xa = whisper.encoder.forward(mel_tensor);
    let (logits, cache_outputs) = whisper.decoder.forward(input, pos_ids, xa, &kv_cache);

    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }
    let prompt: Vec<u32> = vec![TOKEN_SOT, TOKEN_NO_TIMESTAMPS];
    let max_prefill = prompt.len().max(2);
    let build_options = CompileOptions::default().dim_buckets(
        's',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_prefill).representative(max_prefill),
        ],
    );

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>(build_options);

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let cache_bytes_per_layer =
        N_TEXT_HEAD * max_target_pos * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..N_TEXT_LAYER {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes_per_layer);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes_per_layer);
    }

    // Set the mel spectrogram once.
    runtime.set_data(mel_tensor, mel_data.clone());

    println!("Compiling...");
    cx.set_dim('s', max_prefill);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1i32; max_prefill]);
    runtime.set_data(pos_ids, (0..max_prefill as i32).collect::<Vec<_>>());
    let search_options = CompileOptions::default().search_graph_limit(search_graphs);
    runtime = cx.search(runtime, search_options);

    // Reset the KV caches and re-set the mel after search (which executes test runs).
    for i in 0..N_TEXT_LAYER {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes_per_layer);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes_per_layer);
    }
    runtime.set_data(mel_tensor, mel_data);

    // -- Decoding loop --
    // For tiny.en, decoder starts with [<|startoftranscript|>, <|notimestamps|>] as a batched
    // causal prefill, then samples greedily token-by-token.
    let mut prev_seq = 0usize;
    let mut next_input = None;
    let mut generated: Vec<u32> = Vec::new();
    let mut step = 0usize;

    print!("Transcription:");
    std::io::stdout().flush().unwrap();

    let start = Instant::now();
    if gen_tokens > 0 {
        cx.set_dim('s', prompt.len());
        cx.set_dim('p', 0);

        runtime.set_data(
            input,
            prompt.iter().map(|token| *token as i32).collect::<Vec<_>>(),
        );
        runtime.set_data(pos_ids, (0..prompt.len() as i32).collect::<Vec<_>>());

        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Round-trip the KV caches
        for (i, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[i], k_buf);
            runtime.set_buffer(kv_cache.v_caches[i], v_buf);
        }

        prev_seq = prompt.len();

        let last_row = logits_data[logits_data.len() - N_VOCAB..].to_vec();
        let next_token = greedy_decode(&last_row, true);

        if next_token != TOKEN_EOT {
            if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
                print!("{decoded}");
                std::io::stdout().flush().unwrap();
            }

            generated.push(next_token);
            next_input = Some(next_token as i32);
            step = 1;
        }
    }

    while step < gen_tokens && prev_seq < max_target_pos - 1 {
        let Some(current_input) = next_input else {
            break;
        };
        cx.set_dim('s', 1);
        cx.set_dim('p', prev_seq);

        runtime.set_data(input, vec![current_input]);
        runtime.set_data(pos_ids, vec![prev_seq as i32]);

        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Round-trip the KV caches
        for (i, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(*k_out);
            let v_buf = runtime.remove_buffer(*v_out);
            runtime.set_buffer(kv_cache.k_caches[i], k_buf);
            runtime.set_buffer(kv_cache.v_caches[i], v_buf);
        }

        prev_seq += 1;

        let last_row = logits_data[logits_data.len() - N_VOCAB..].to_vec();
        let next_token = greedy_decode(&last_row, false);

        if next_token == TOKEN_EOT {
            break;
        }

        if let Ok(decoded) = tokenizer.decode(&[next_token], false) {
            print!("{decoded}");
            std::io::stdout().flush().unwrap();
        }

        generated.push(next_token);
        next_input = Some(next_token as i32);
        step += 1;

        if step >= gen_tokens {
            break;
        }
        if prev_seq >= max_target_pos - 1 {
            break;
        }
    }
    let elapsed = start.elapsed();
    println!();
    println!(
        "Decoded {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        elapsed.as_secs_f64(),
        generated.len() as f64 / elapsed.as_secs_f64().max(1e-6),
    );
}

/// Greedy argmax with whisper-style suppression of special tokens.
fn greedy_decode(logits: &[f32], is_first_step: bool) -> u32 {
    debug_assert_eq!(logits.len(), N_VOCAB);
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        // Suppress all special / language / timestamp tokens (except <|endoftext|>).
        // For tiny.en these live in the >= 50257 range. We allow only TOKEN_EOT.
        if i as u32 != TOKEN_EOT && i >= TOKEN_SOT as usize {
            continue;
        }
        // Suppress <|endoftext|> on the very first generated step to avoid empty output.
        if is_first_step && i as u32 == TOKEN_EOT {
            continue;
        }
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}
