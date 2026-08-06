#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod hf;
mod model;

use hf::{WeightFormat, prepare_hf_model};
use luminal::{dtype::DType, prelude::*};
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use std::env;
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "google-bert/bert-base-uncased";
const MAX_SEQ_LEN: usize = 128;
const MASK_TOKEN: u32 = 103;
const TOP_K: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BertWeightMode {
    F32,
    Bf16,
}

impl BertWeightMode {
    fn weight_format(self) -> WeightFormat {
        match self {
            Self::F32 => WeightFormat::F32,
            Self::Bf16 => WeightFormat::Bf16,
        }
    }
}

fn print_usage(program: &str) {
    println!("Usage: {program} [--bf16|--f32]");
    println!();
    println!("  --f32     Native f32 weights and activations (default)");
    println!("  --bf16    Bf16 weights and activations (norms stay F32)");
    println!("  -h,--help Show this help");
}

fn parse_args() -> BertWeightMode {
    let mut mode = BertWeightMode::F32;
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "bert".to_string());
    for arg in args {
        match arg.as_str() {
            "--bf16" => mode = BertWeightMode::Bf16,
            "--f32" => mode = BertWeightMode::F32,
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

fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let weight_mode = parse_args();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let prepared =
        prepare_hf_model(REPO_ID, weight_mode.weight_format()).expect("Failed to prepare model");
    println!("Using model: {REPO_ID}");
    println!("Using model directory: {}", prepared.model_dir.display());

    let tokenizer = Tokenizer::from_file(prepared.model_dir.join("tokenizer.json")).unwrap();

    // Build graph
    let mut cx = Graph::default();
    let input_ids = cx.named_tensor("input_ids", 's').as_dtype(DType::Int);
    let token_type_ids = cx.named_tensor("token_type_ids", 's').as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("pos_ids", 's').as_dtype(DType::Int);
    let mask = cx.named_tensor("mask", (1, 's', 's'));

    let bert = match weight_mode {
        BertWeightMode::F32 => BertForMaskedLM::init_f32(&mut cx),
        BertWeightMode::Bf16 => BertForMaskedLM::init_bf16(&mut cx),
    };
    let logits = bert
        .forward(input_ids, token_type_ids, pos_ids, mask)
        .output();

    cx.set_dim('s', 1);

    println!("Loading weights...");
    let load_start = std::time::Instant::now();
    let mut runtime = CudaRuntime::initialize(stream.clone());
    for weights_path in &prepared.weight_files {
        println!("  Loading {}", weights_path.display());
        runtime.load_safetensors(&cx, weights_path.to_str().unwrap());
    }
    println!("  Weight load: {:.2} s", load_start.elapsed().as_secs_f64());

    println!("Compiling...");
    let compile_start = std::time::Instant::now();
    cx.set_dim('s', MAX_SEQ_LEN);
    runtime = cx.compile(runtime, CompileOptions::default());
    println!("  Compile: {:.2} s", compile_start.elapsed().as_secs_f64());

    // Example: "The capital of France is [MASK]."
    let sentence = "The capital of France is [MASK].";
    let encoding = tokenizer.encode(sentence, true).unwrap();
    let tokens = encoding.get_ids();
    let seq_len = tokens.len().min(MAX_SEQ_LEN);
    let tokens: Vec<u32> = tokens[..seq_len].to_vec();

    // Find mask positions
    let mask_positions: Vec<usize> = tokens
        .iter()
        .enumerate()
        .filter(|&(_, t)| *t == MASK_TOKEN)
        .map(|(i, _)| i)
        .collect();

    if mask_positions.is_empty() {
        eprintln!("No [MASK] token found in input. Use the token [MASK] in your sentence.");
        std::process::exit(1);
    }

    println!(
        "Input: \"{sentence}\" ({} tokens, mask at positions {:?})",
        seq_len, mask_positions
    );

    // Build position ids and token type ids
    let pos_ids_data: Vec<i32> = (0..seq_len as i32).collect();
    let token_type_ids_data: Vec<i32> = vec![0; seq_len];

    // Build attention mask (causal not needed for BERT, just 1s for valid tokens)
    let mut mask_data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            mask_data[i * seq_len + j] = 0.0; // 0 = not masked (BERT uses 0/1, not -inf/0)
        }
    }

    cx.set_dim('s', seq_len);
    runtime.set_data(
        input_ids,
        tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
    );
    runtime.set_data(token_type_ids, token_type_ids_data);
    runtime.set_data(pos_ids, pos_ids_data);
    runtime.set_data(mask, mask_data);

    runtime.execute(&cx.dyn_map);

    let logits_data = runtime.get_f32(logits);

    // For each mask position, find top-k predictions
    for &mask_pos in &mask_positions {
        println!("\nTop {TOP_K} predictions at position {mask_pos}:");
        let start = mask_pos * VOCAB_SIZE;
        let end = start + VOCAB_SIZE;
        let scores = &logits_data[start..end];

        let mut indices: Vec<usize> = (0..VOCAB_SIZE).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());

        for (rank, &idx) in indices.iter().take(TOP_K).enumerate() {
            let token_str = tokenizer
                .decode(&[idx as u32], true)
                .unwrap_or_else(|_| format!("<id {idx}>"));
            println!(
                "  {}. {} (id={}, score={:.4})",
                rank + 1,
                token_str.trim(),
                idx,
                scores[idx]
            );
        }
    }
}
