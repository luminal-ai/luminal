#![allow(unreachable_code)]

use std::{
    io::{self, Write},
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use model::{HEAD_DIM, N_KV_HEADS};
use tokenizers::Tokenizer;

mod gguf;
mod loader;
mod model;

use crate::model::KVCache;
use luminal::prelude::*;

// Command args parser
#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
pub struct CLIArgs {
    /// Number of tokens to generate
    #[clap(short = 't', long = "gen_tokens", default_value = "256")]
    gen_tokens: i32,

    /// Prompt for the model
    #[clap(short = 'p', long = "prompt", default_value = include_str!("../prompts/merge_sort.txt"))]
    prompt: String,
}

fn main() {
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    panic!("Either metal or cuda feature must be used for this example!");

    let cli_args = CLIArgs::parse();
    let tokenizer = Tokenizer::from_file("setup/tokenizer.json").unwrap();

    print!("Defining graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();

    // Set up graph
    let mut cx = Graph::new();
    let mut input = cx.named_tensor("Input", (1, 's'));
    let mut cache_src: Vec<KVCache> = (0..model::NUM_LAYERS)
        .map(|_| {
            (
                cx.named_tensor("Key Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (1, N_KV_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();
    cache_src.set_dyn(vec![], (1, model::N_KV_HEADS, 0, model::HEAD_DIM));
    let model = model::Qwen::new(&mut cx);
    let mut model_weights = params(&model);
    cx.keep_tensors(&model_weights);
    let (logits, mut cache_dest) = model.forward((input, &cache_src));
    let mut logits = logits
        .slice((.., Expression::from('s') - 1.., ..))
        .retrieve();
    cache_dest.keep();
    println!("\t\t - {}ms", now.elapsed().as_millis());

    print!("Compiling graph");
    io::stdout().flush().unwrap();
    let now = Instant::now();

    // Set up model loading
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let q_weights = loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    loader::q8_load("setup/qwen3-4b.gguf", &model, &mut cx);

    cx.compile(
        (
            GenericCompiler::default(),
            #[cfg(feature = "metal")]
            (
                luminal_metal::MetalCompilerPreBuffer::<f32>::default(),
                luminal_metal::quantized::MetalQuantizedCompiler::<f32>::new(q_weights),
                luminal_metal::BufferCompilers::default(),
            ),
            #[cfg(feature = "cuda")]
            (
                luminal_cuda::CudaCompiler::<f32>::default(),
                luminal_cuda::CudaQuantizedCompiler::<f32>::new(q_weights),
            ),
        ),
        (
            &mut input,
            &mut logits,
            &mut cache_src,
            &mut cache_dest,
            &mut model_weights,
        ),
    );
    let cache_src = downstream(&cache_src, &cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    // Initial forward pass to load weights
    print!("Loading model");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    input.set_dyn(vec![1.], (1, 1));
    cx.set_dyn_dim('t', 1);
    cx.execute();
    logits.drop();
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    // Now that weights are loaded, delete the loading nodes so they don't run again
    delete_inputs(&cache_src, &mut cx);
    delete_inputs(downstream(model_weights, &cx), &mut cx);

    // Run prompt processing pass
    let input_ids = tokenizer
        .encode(&cli_args.prompt as &str, false)
        .unwrap()
        .get_ids()
        .to_vec();
    input.set_dyn(
        input_ids.iter().map(|i| *i as f32).collect::<Vec<_>>(),
        (1, input_ids.len()),
    );
    cx.set_dyn_dim('t', input_ids.len());
    print!("Processing Prompt");
    io::stdout().flush().unwrap();
    let now = Instant::now();
    cx.execute();
    let elapsed_ms = now.elapsed().as_millis();
    println!(
        "\t - {elapsed_ms}ms ({:.2} tok/s, {} prompt tokens)",
        1000.0 * (input_ids.len() as f64) / (elapsed_ms as f64),
        input_ids.len()
    );
    let mut output_ids = vec![argmax(&logits.data())];
    logits.drop();

    // Decode token
    print!("{}", cli_args.prompt.white().bold());
    let initial = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    print!("{initial}",);
    io::stdout().flush().unwrap();

    // Swap caches
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

    // Decode loop
    let start_decode = std::time::Instant::now();
    let mut prev_output_len = initial.len();
    for _ in 0..cli_args.gen_tokens {
        input.set_dyn(vec![*output_ids.last().unwrap() as f32], (1, 1));
        cx.set_dyn_dim('p', input_ids.len() + output_ids.len() - 1);
        cx.execute();

        // Sample tokens
        let output_id = argmax(&logits.data());
        logits.drop();
        output_ids.push(output_id);

        // Get the current decoded output
        let current_output = tokenizer.decode(&output_ids, false).unwrap();

        // Print the new substring added to the decoded output
        let legal_byte_num = utf8_legal_byte_num(current_output.as_bytes()[prev_output_len]);
        if let Some(byte_num) = legal_byte_num {
            if current_output.len() > byte_num + prev_output_len {
                print!(
                    "{}",
                    current_output[prev_output_len..prev_output_len + byte_num].bright_green()
                );
                io::stdout().flush().unwrap();

                // Update the previous output
                prev_output_len += byte_num
            }
        }

        // Swap caches
        transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);
    }

    println!();
    let avg_token_time =
        start_decode.elapsed().as_micros() as f32 / (output_ids.len() - 1) as f32 / 1000.0;
    println!(
        "\nAverage token generated in {:.2}ms\t - ({:.2} tok/s)",
        avg_token_time,
        1000.0 / avg_token_time
    );
}

// Currently just an argmax, do actual sampling here
fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}

/// return utf8 char num corresponding to start byte, if it's not a start byte return [`None`]
fn utf8_legal_byte_num(byte: u8) -> Option<usize> {
    match byte {
        // ASCII  (0xxxxxxx)
        0x00..=0x7F => Some(1),
        // char of 2 bytes (110xxxxx)
        0xC0..=0xDF => Some(2),
        // char of 3 bytes (1110xxxx)
        0xE0..=0xEF => Some(3),
        // char of 4 bytes (11110xxx)
        0xF0..=0xF7 => Some(4),
        // not a start byte
        _ => None,
    }
}
