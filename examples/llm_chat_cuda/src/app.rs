//! This example's chat CLI.
use crate::{
    backend::CudaBackend,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType},
    sampling::Sampler,
    session::Session,
    tokenizer::{ChatTokenizer, Message},
};
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use luminal_cuda_lite::CompileOptions;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "Chat with a model-zoo LLM on the CUDA-lite runtime.")]
struct Args {
    #[arg(long, value_enum)]
    model: ModelType,
    /// Local Hugging Face checkpoint directory (config, tokenizer, safetensors).
    #[arg(long)]
    checkpoint: PathBuf,
    /// JSON safetensors-name -> model-namespace overrides.
    #[arg(long)]
    tensor_map: Option<PathBuf>,
    #[arg(long, default_value_t = 2048)]
    max_context: usize,
    #[arg(long, default_value_t = 8)]
    prefill_chunk: usize,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    system: Option<String>,
    /// Run one turn and exit; omit to start the interactive chat.
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    enable_thinking: bool,
    /// Rank compiler candidates on the selected device using device measurements.
    #[arg(long, default_value_t = CompileOptions::default().profile_on_device,
        action = clap::ArgAction::Set, num_args = 0..=1,
        default_missing_value = "true", require_equals = true)]
    profile: bool,
    #[arg(long, default_value_t = 2)]
    search_generations: usize,
    #[arg(long, default_value_t = 4)]
    search_population: usize,
}

pub fn main() -> Result<()> {
    ensure!(
        cfg!(feature = "device"),
        "build with --features device to execute on a CUDA GPU"
    );
    run(Args::parse())
}
fn run(args: Args) -> Result<()> {
    ensure!(args.max_new_tokens > 0, "max-new-tokens must be positive");
    ensure!(
        args.search_generations > 0 && args.search_population > 0,
        "search generations/population must be positive"
    );
    let mut sampler = Sampler::new(args.temperature, args.top_p, args.seed)?;
    let config = checkpoint::read_json(&args.checkpoint.join("config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    if let Some(max) = text["max_position_embeddings"].as_u64() {
        ensure!(
            args.max_context as u64 <= max,
            "max-context exceeds checkpoint max_position_embeddings={max}"
        );
    }
    let model = ModelConfig::from_checkpoint(args.model, &config)?;
    let tokenizer = ChatTokenizer::load(&args.checkpoint, &config)?;
    let mut graph = LlmGraph::build(model, args.max_context, args.prefill_chunk)?;
    ensure!(
        tokenizer
            .tokenizer
            .get_vocab(true)
            .values()
            .all(|&id| (id as usize) < graph.vocab),
        "tokenizer vocabulary exceeds model vocabulary"
    );
    ensure!(
        tokenizer
            .stop_tokens
            .iter()
            .all(|&id| (id as usize) < graph.vocab),
        "EOS token outside model vocabulary"
    );
    if let Some(path) = &args.tensor_map {
        let map: BTreeMap<String, String> = serde_json::from_value(checkpoint::read_json(path)?)?;
        checkpoint::apply_name_map(&mut graph.parameters, &map)?;
    }
    eprintln!(
        "Loading {} tensors from {}",
        graph.parameters.len(),
        args.checkpoint.display()
    );
    let weights = checkpoint::load(&args.checkpoint, &graph.parameters)?;
    eprintln!("Compiling {:?} for CUDA Lite...", args.model);
    let options = CompileOptions {
        profile_on_device: args.profile,
        // The heuristic is a byte estimate, not the duration printed by the
        // runtime progress display. Show timings only with device profiling.
        search_log: args.profile,
        generations: args.search_generations,
        generation_size: args.search_population,
        seed: args.seed,
        ..Default::default()
    };
    let backend = CudaBackend::compile(&graph, weights, &options).context("compile chat graph")?;
    let mut session = Session::new(graph, backend);
    let mut history = vec![];
    if let Some(system) = &args.system {
        history.push(Message::new("system", system));
    }
    let initial_history = history.clone();
    if let Some(prompt) = &args.prompt {
        turn(
            &args,
            prompt,
            &tokenizer,
            &mut history,
            &mut session,
            &mut sampler,
        )?;
        return Ok(());
    }
    eprintln!("Ready. /reset clears the conversation; /quit exits.");
    loop {
        print!("You: ");
        io::stdout().flush()?;
        let mut prompt = String::new();
        if io::stdin().read_line(&mut prompt)? == 0 {
            break;
        }
        match prompt.trim() {
            "/quit" | "/exit" => break,
            "/reset" => {
                session.reset()?;
                history = initial_history.clone();
                continue;
            }
            "" => continue,
            _ => {}
        }
        if let Err(error) = turn(
            &args,
            prompt.trim_end(),
            &tokenizer,
            &mut history,
            &mut session,
            &mut sampler,
        ) {
            eprintln!("{error:#}");
            session.reset()?;
        }
    }
    Ok(())
}
fn turn(
    args: &Args,
    prompt: &str,
    tokenizer: &ChatTokenizer,
    history: &mut Vec<Message>,
    session: &mut Session,
    sampler: &mut Sampler,
) -> Result<()> {
    let mut messages = history.clone();
    messages.push(Message::new("user", prompt));
    let tokens = tokenizer.encode_chat(&messages, args.enable_thinking)?;
    let mut stream = tokenizer.tokenizer.decode_stream(true);
    let mut streamed = String::new();
    print!("Assistant: ");
    io::stdout().flush()?;
    let generated = session.generate(
        &tokens,
        args.max_new_tokens,
        &tokenizer.stop_tokens,
        sampler,
        |token| {
            if let Some(text) = stream
                .step(token)
                .map_err(|e| anyhow!("stream decode: {e}"))?
            {
                print!("{text}");
                streamed.push_str(&text);
                io::stdout().flush()?;
            }
            Ok(())
        },
    )?;
    let decoded = tokenizer.decode(&generated)?;
    if let Some(tail) = decoded.strip_prefix(&streamed) {
        print!("{tail}");
    }
    println!();
    messages.push(Message::new("assistant", decoded));
    *history = messages;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiling_cli_defaults_on_and_accepts_explicit_opt_out() {
        let base = [
            "llm_chat_cuda",
            "--model",
            "qwen3",
            "--checkpoint",
            "/tmp/checkpoint",
        ];
        assert!(Args::try_parse_from(base).unwrap().profile);
        for (flag, expected) in [
            ("--profile", true),
            ("--profile=true", true),
            ("--profile=false", false),
        ] {
            let args = Args::try_parse_from(base.into_iter().chain([flag])).unwrap();
            assert_eq!(args.profile, expected);
        }
    }
}
