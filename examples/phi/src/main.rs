#![allow(unreachable_code)]

use std::{
    io::{self, Write},
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use model::{Phi, HEAD_DIM, N_HEADS};
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
    #[clap(short = 't', long = "gen_tokens", default_value = "512")]
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
                cx.named_tensor("Key Cache", (1, N_HEADS, 'p', HEAD_DIM)),
                cx.named_tensor("Value Cache", (1, N_HEADS, 'p', HEAD_DIM)),
            )
        })
        .collect();
    cache_src.set_dyn(vec![], (1, N_HEADS, 0, HEAD_DIM));
    let model = Phi::new(&mut cx);
    let mut model_weights = params(&model);
    cx.keep_tensors(&model_weights);
    let (logits, mut cache_dest) = model.forward((input, &cache_src));
    let mut logits = logits
        .slice((.., (Expression::from('s') - 1).., ..))
        .retrieve();
    cache_dest.keep();

    // Set up model loading
    #[cfg(any(feature = "metal", feature = "cuda"))]
    let q_weights = loader::q8_load("setup/phi3.gguf", &model, &mut cx);
    #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
    loader::q8_load("setup/phi3.gguf", &model, &mut cx);
    println!("\t\t - {}ms", now.elapsed().as_millis());

    print!("Compiling graph");
    io::stdout().flush().unwrap();
    if debug() {
        println!();
    }
    let now = Instant::now();
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
                luminal_cuda::CudaCompiler::<f16>::default(),
                luminal_cuda::CudaQuantizedCompiler::<f16>::new(q_weights),
            ),
            #[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
            luminal_cpu::CPUCompiler::default(),
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
    let out = tokenizer.decode(&output_ids, false).unwrap().bright_green();
    print!("{out}");
    io::stdout().flush().unwrap();

    // Swap caches
    transfer_data_same_graph(&cache_dest, &cache_src, &mut cx);

    // Decode loop
    let start_decode = std::time::Instant::now();
    let mut prev_output_len = out.len();
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
        print!("{}", current_output[prev_output_len..].bright_green());
        io::stdout().flush().unwrap();

        // Update the previous output
        prev_output_len = current_output.len();

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

fn argmax(dist: &[f32]) -> u32 {
    dist.iter()
        .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as u32
}
