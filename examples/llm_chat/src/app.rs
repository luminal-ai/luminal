//! This example's chat CLI.
use crate::{
    Inputs,
    backend::Backend,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType, WeightDtype, checkpoint_dtype},
    hf,
    sampling::Sampler,
    session::{GenerationTimings, Session},
    tokenizer::{ChatTokenizer, Message},
};
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use std::{
    collections::BTreeMap,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Chat with a model-zoo LLM.")]
struct Args {
    #[arg(long, value_enum)]
    model: ModelType,
    /// Parameter storage dtype. `checkpoint` follows config.json's torch_dtype.
    #[arg(long, value_enum, default_value_t = WeightDtype::Checkpoint)]
    weight_dtype: WeightDtype,
    /// Local Hugging Face checkpoint directory (config, tokenizer, safetensors).
    /// When omitted, the checkpoint is downloaded from the Hugging Face Hub.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Hugging Face repo id to download when --checkpoint is omitted. Defaults
    /// to the repository validated for --model.
    #[arg(long)]
    repo: Option<String>,
    /// Hugging Face revision (branch, tag, or commit) to download.
    #[arg(long)]
    revision: Option<String>,
    /// JSON safetensors-name -> model-namespace overrides.
    #[arg(long)]
    tensor_map: Option<PathBuf>,
    #[arg(long, default_value_t = 2048)]
    max_context: usize,
    #[arg(long, default_value_t = crate::search::DEFAULT_PREFILL_CHUNK)]
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
    /// Search generations per bucket.
    #[arg(long, default_value_t = crate::search::DEFAULT_SEARCH_GENERATIONS)]
    search_generations: usize,
    /// Candidate attempts per generation, per bucket.
    #[arg(long, default_value_t = crate::search::DEFAULT_SEARCH_POPULATION)]
    search_population: usize,
}

pub struct SearchOptions {
    pub generations: usize,
    pub population: usize,
    pub seed: u64,
}

pub fn main<B: Backend>(
    device: &str,
    compile: impl FnOnce(&LlmGraph, Inputs, SearchOptions) -> Result<B>,
) -> Result<()> {
    run(Args::parse(), device, compile)
}
fn run<B: Backend>(
    args: Args,
    device: &str,
    compile: impl FnOnce(&LlmGraph, Inputs, SearchOptions) -> Result<B>,
) -> Result<()> {
    ensure!(args.max_new_tokens > 0, "max-new-tokens must be positive");
    ensure!(
        args.search_generations > 0 && args.search_population > 0,
        "search generations/population must be positive"
    );
    let mut sampler = Sampler::new(args.temperature, args.top_p, args.seed)?;
    let checkpoint_dir = resolve_checkpoint(&args)?;
    let config = checkpoint::read_json(&checkpoint_dir.join("config.json"))?;
    let text = config.get("text_config").unwrap_or(&config);
    if let Some(max) = text["max_position_embeddings"].as_u64() {
        ensure!(
            args.max_context as u64 <= max,
            "max-context exceeds checkpoint max_position_embeddings={max}"
        );
    }
    let model = ModelConfig::from_checkpoint(args.model, &config)?;
    let dtype = args.weight_dtype.resolve(checkpoint_dtype(&config)?);
    let tokenizer = ChatTokenizer::load(&checkpoint_dir, &config)?;
    let mut graph =
        LlmGraph::build_with_parameter_dtype(model, dtype, args.max_context, args.prefill_chunk)?;
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
        checkpoint_dir.display()
    );
    let load_started = Instant::now();
    let weights = checkpoint::load(&checkpoint_dir, &graph.parameters)?;
    eprintln!(
        "Loaded {} tensors in {:.1} s",
        graph.parameters.len(),
        load_started.elapsed().as_secs_f64()
    );
    eprintln!("Compiling {:?} for {device}...", args.model);
    let options = SearchOptions {
        generations: args.search_generations,
        population: args.search_population,
        seed: args.seed,
    };
    let compile_started = Instant::now();
    let backend = compile(&graph, weights, options).context("compile chat graph")?;
    eprintln!(
        "Compiled graph in {:.1} s",
        compile_started.elapsed().as_secs_f64()
    );
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
/// The directory the loader reads: the user-provided path, or a Hub download
/// for the selected model.
fn resolve_checkpoint(args: &Args) -> Result<PathBuf> {
    if let Some(path) = &args.checkpoint {
        ensure!(
            args.repo.is_none() && args.revision.is_none(),
            "--repo/--revision select a download and cannot be combined with --checkpoint"
        );
        return Ok(path.clone());
    }
    let repo = args
        .repo
        .clone()
        .unwrap_or_else(|| args.model.default_hf_repo().to_owned());
    eprintln!(
        "Downloading {repo}{} from the Hugging Face Hub...",
        args.revision
            .as_deref()
            .map(|revision| format!("@{revision}"))
            .unwrap_or_default()
    );
    let directory = hf::download_checkpoint(&repo, args.revision.as_deref())
        .with_context(|| format!("download Hugging Face checkpoint {repo}"))?;
    eprintln!("Using checkpoint {}", directory.display());
    Ok(directory)
}
fn turn<B: Backend>(
    args: &Args,
    prompt: &str,
    tokenizer: &ChatTokenizer,
    history: &mut Vec<Message>,
    session: &mut Session<B>,
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
    if let Some(timing) = session.timings() {
        print_timings(&timing);
    }
    let decoded = tokenizer.decode(&generated)?;
    if let Some(tail) = decoded.strip_prefix(&streamed) {
        print!("{tail}");
    }
    println!();
    messages.push(Message::new("assistant", decoded));
    *history = messages;
    Ok(())
}
fn print_timings(timing: &GenerationTimings) {
    let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1e3;
    let tpot = timing
        .tpot()
        .map(|duration| format!("{:.2} ms/token", ms(duration)))
        .unwrap_or_else(|| "n/a".to_owned());
    let decode_rate = if timing.generated_tokens > 1 && timing.decode.as_secs_f64() > 0. {
        format!(
            "{:.1} tok/s",
            (timing.generated_tokens - 1) as f64 / timing.decode.as_secs_f64()
        )
    } else {
        "n/a".to_owned()
    };
    eprintln!(
        "Latency: TTFT {:.1} ms | TPOT {tpot} ({decode_rate}) | prefill {} tok in {:.1} ms | decode {:.1} ms | total {:.1} ms | {} new tok",
        ms(timing.ttft),
        timing.prompted_tokens,
        ms(timing.prefill),
        ms(timing.decode),
        ms(timing.total),
        timing.generated_tokens
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_accepts_bf16_for_every_model() {
        for model in ["llama3", "qwen3", "gemma3", "qwen3-moe"] {
            let args = Args::try_parse_from([
                "llm_chat",
                "--model",
                model,
                "--weight-dtype",
                "bf16",
                "--checkpoint",
                "/tmp/checkpoint",
            ])
            .unwrap();
            assert_eq!(args.weight_dtype, WeightDtype::Bf16);
        }
    }

    #[test]
    fn profiling_toggle_is_not_an_option() {
        let base = [
            "llm_chat",
            "--model",
            "qwen3",
            "--checkpoint",
            "/tmp/checkpoint",
        ];
        let args = Args::try_parse_from(base).unwrap();
        assert_eq!(args.search_generations * args.search_population, 100);
        assert_eq!(args.prefill_chunk, 128);
        for flag in ["--profile", "--profile=true", "--profile=false"] {
            assert!(Args::try_parse_from(base.into_iter().chain([flag])).is_err());
        }
    }
}
