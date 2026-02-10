mod hf;
mod model;

use hf::{load_expert_weights, load_sinks, prepare_hf_model};
use luminal::prelude::*;
use luminal_cuda::{
    cudarc::driver::{sys::CUdevice_attribute, CudaContext},
    runtime::CudaRuntime,
};
use luminal_tracing::*;
use model::*;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing::{span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "openai/GPT-OSS-120B";

// This example compiles and runs GPT-OSS 120B with MXFP4 quantized MoE weights on CUDA.

fn main() {
    let max_seq_len = 4096;
    let gen_tokens = 200;
    let search_graphs = 200;
    let prompt = "Hello, how are you";

    // Tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    // Set up cuda context and stream
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Download model if needed and prepare weights
    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    // Tokenize prompt with harmony chat template
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // Apply harmony template: <|start|>user<|message|>{prompt}<|end|>\n<|start|>assistant
    let harmony_prompt = format!("<|start|>user<|message|>{prompt}<|end|>\n<|start|>assistant");
    let mut sentence = tokenizer
        .encode(harmony_prompt.as_str(), false)
        .unwrap()
        .get_ids()
        .to_vec();

    // Allocate KV cache, expert weight buffers, MoE scratchpad, and sink buffers
    let mut kv_cache = KVCache::new(&stream, max_seq_len);
    let mut expert_weights = ExpertWeightBuffers::new(&stream);
    let scratchpad = MoeScratchpad::new(&stream);
    let mut sink_buffers = SinkBuffers::new(&stream);

    // Load expert weights directly into GPU buffers from HF shards
    println!("Loading expert weights...");
    load_expert_weights(&model_dir, &mut expert_weights, &stream)
        .expect("Failed to load expert weights");

    // Get SM count for MoE kernel parallelism
    let sm_count = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .expect("Failed to get SM count") as u32;
    println!("GPU SMs: {sm_count}");

    // Create compute graph
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let model = GptOss::init(&mut cx);
    let logits = model
        .forward(
            input,
            &kv_cache,
            &expert_weights,
            &scratchpad,
            &sink_buffers,
            sm_count,
        )
        .output();

    // Build search space
    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    // Load non-quantized model weights from combined safetensors
    println!("Loading non-quantized weights...");
    let mut runtime = CudaRuntime::initialize(stream.clone());
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    // Load sink values into GPU buffers
    println!("Loading sink values...");
    load_sinks(&model_dir, &mut sink_buffers, &stream).expect("Failed to load sink values");

    // Run search process
    println!("Compiling...");
    cx.set_dim('s', 1);
    cx.set_dim('p', 0);
    runtime.set_data(input, vec![1]);
    runtime = cx.search(runtime, search_graphs);
    kv_cache.reset();

    println!("> {prompt}");
    std::io::stdout().flush().unwrap();

    // Feed prompt tokens one at a time (MoeExperts only supports single-token)
    let prompt_tokens = sentence.clone();
    let total_tokens = prompt_tokens.len() + gen_tokens;
    let mut fwd_durations = vec![];

    // Harmony template state: track whether we're in the "final" channel
    // Generated tokens go through: <|channel|>analysis<|message|>...<|end|>\n<|start|>assistant<|channel|>final<|message|>...<|end|>
    let mut in_final_channel = false;
    let mut recent_tokens: Vec<u32> = Vec::new(); // Track recent tokens to detect final channel marker
    const CHANNEL_TOKEN: u32 = 200005; // <|channel|>
    const MESSAGE_TOKEN: u32 = 200008; // <|message|>
    const END_TOKEN: u32 = 200007; // <|end|>
    const EOS_TOKEN: u32 = 200002; // <|return|>

    for i in 0..total_tokens {
        let start = std::time::Instant::now();
        let _span = if i < prompt_tokens.len() {
            span!(Level::INFO, "prefill")
        } else {
            span!(Level::INFO, "decode")
        }
        .entered();

        // Always feed one token at a time
        let token = if i < prompt_tokens.len() {
            prompt_tokens[i]
        } else {
            sentence[0]
        };

        cx.set_dim('s', 1);
        cx.set_dim('p', i);

        runtime.set_data(input, vec![token as i32]);

        // Execute forward pass
        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits);

        // Sample next token (greedy)
        let _sample_span = span!(Level::INFO, "sample_full").entered();
        sentence = vec![*sample(&logits_data, VOCAB_SIZE).last().unwrap()];

        // Only process generated tokens (not prompt tokens)
        if i >= prompt_tokens.len() {
            let tok = sentence[0];

            // Stop on true EOS (<|return|>)
            if tok == EOS_TOKEN {
                break;
            }

            // Stop on <|end|> only if we're in the final channel
            if tok == END_TOKEN && in_final_channel {
                break;
            }

            // Track recent tokens to detect <|channel|>final<|message|> sequence
            recent_tokens.push(tok);
            if recent_tokens.len() > 3 {
                recent_tokens.remove(0);
            }

            // Detect final channel marker: <|channel|>(17196=final)<|message|>
            if recent_tokens.len() >= 3 {
                let n = recent_tokens.len();
                if recent_tokens[n - 3] == CHANNEL_TOKEN
                    && recent_tokens[n - 2] == 17196 // "final"
                    && recent_tokens[n - 1] == MESSAGE_TOKEN
                {
                    in_final_channel = true;
                }
            }

            // Only print tokens in the final channel
            if in_final_channel && tok != MESSAGE_TOKEN {
                print!("{}", tokenizer.decode(&sentence, true).unwrap());
                std::io::stdout().flush().unwrap();
            }
        }
        fwd_durations.push(start.elapsed());
    }
    println!();

    // Report benchmarks
    let prompt_len = prompt_tokens.len();
    println!(
        "  TTFT: {:.2} ms ({}tok prefill)",
        fwd_durations[..prompt_len]
            .iter()
            .sum::<Duration>()
            .as_secs_f64()
            * 1e3,
        prompt_len,
    );
    if fwd_durations.len() > prompt_len + 1 {
        let decode_durs = &fwd_durations[prompt_len + 1..];
        println!(
            "  TPOT: {:.2} ms",
            (decode_durs.iter().sum::<Duration>() / decode_durs.len() as u32).as_secs_f64()
                * 1_000.
        );
    }
    runtime.print_execution_stats();
}

#[tracing::instrument(skip_all)]
fn sample(logits: &[f32], vocab_size: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab_size)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0 as u32
        })
        .collect()
}
