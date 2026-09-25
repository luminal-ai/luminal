//! Real-checkpoint differential validation, separate from the product CLI.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use llm_chat::{
    backend::cuda::CudaBackend,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType, checkpoint_dtype},
    sampling::Sampler,
    tokenizer::{ChatTokenizer, Message},
};
use luminal_cuda_lite::CompileOptions;
use memmap2::{Mmap, MmapOptions};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs::File, path::PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    model: ModelType,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    suite: Option<PathBuf>,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_PREFILL_CHUNK)]
    prefill_chunk: usize,
    /// Search generations per bucket.
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_SEARCH_GENERATIONS)]
    search_generations: usize,
    /// Candidate attempts per generation, per bucket.
    #[arg(long, default_value_t = llm_chat::search::DEFAULT_SEARCH_POPULATION)]
    search_population: usize,
    /// Check configuration and parameter names without loading weights.
    #[arg(long)]
    inspect: bool,
    #[arg(long, default_value_t = 0.005)]
    atol: f32,
    #[arg(long, default_value_t = 0.0005)]
    rtol: f32,
}
#[derive(Deserialize)]
struct Suite {
    model: String,
    checkpoint_revision: Option<String>,
    max_context: usize,
    max_new_tokens: usize,
    stop_tokens: Vec<u32>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    reset: bool,
    messages: Vec<Message>,
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    text: String,
    logits_file: String,
}
/// The validator drives its own prefill/decode loop, so that every single
/// execution's logits can be compared against the reference row for that
/// context length — the chat CLI's session reports only what it decodes.
struct Checked {
    graph: LlmGraph,
    backend: CudaBackend,
    cached: Vec<u32>,
    last_logits: Vec<f32>,
    expected: Option<Mmap>,
    atol: f32,
    rtol: f32,
    steps: usize,
    resets: usize,
    max_abs: f32,
    max_ratio: f32,
}
impl Checked {
    fn reset(&mut self) -> Result<()> {
        self.resets += 1;
        self.cached.clear();
        self.last_logits.clear();
        self.backend.reset()
    }
    /// One execution, and its comparison against the reference.
    fn step(&mut self, tokens: &[u32]) -> Result<()> {
        let inputs = self.graph.step_inputs(tokens, self.cached.len())?;
        let (query, context) = (tokens.len(), self.cached.len() + tokens.len());
        let actual = self.backend.step(inputs, query, context)?;
        let tensors = SafeTensors::deserialize(self.expected.as_ref().unwrap())?;
        let tensor = tensors.tensor("logits")?;
        ensure!(
            tensor.dtype() == Dtype::F32 && tensor.shape().len() == 2,
            "expected F32 logit matrix"
        );
        let vocab = tensor.shape()[1];
        ensure!(
            actual.len() == vocab && context > 0 && context <= tensor.shape()[0],
            "reference shape/context mismatch"
        );
        let row = &tensor.data()[(context - 1) * vocab * 4..context * vocab * 4];
        let mut max_abs = 0f32;
        let mut max_ratio = 0f32;
        for (&got, bytes) in actual.iter().zip(row.as_chunks::<4>().0) {
            let expected = f32::from_le_bytes(*bytes);
            ensure!(
                got.is_finite() && expected.is_finite(),
                "non-finite logit at context {context}"
            );
            let error = (got - expected).abs();
            max_abs = max_abs.max(error);
            max_ratio = max_ratio.max(error / (self.atol + self.rtol * expected.abs()));
        }
        self.steps += 1;
        self.max_abs = self.max_abs.max(max_abs);
        self.max_ratio = self.max_ratio.max(max_ratio);
        ensure!(
            max_ratio <= 1.,
            "logits differ at query={query}, context={context}: max_abs={max_abs}, max_tolerance_ratio={max_ratio}"
        );
        self.cached.extend_from_slice(tokens);
        self.last_logits = actual;
        Ok(())
    }
    fn ingest(&mut self, tokens: &[u32]) -> Result<()> {
        for chunk in tokens.chunks(self.graph.chunk_size) {
            if let Err(error) = self.step(chunk) {
                self.reset()?;
                return Err(error);
            }
        }
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stops: &BTreeSet<u32>,
        sampler: &mut Sampler,
    ) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(
            prompt.len() < self.graph.capacity,
            "prompt fills the context"
        );
        // Token prefixes, not rendered strings, decide cache validity.
        if !prompt.starts_with(&self.cached) {
            self.reset()?;
        }
        let consumed = self.cached.len();
        self.ingest(&prompt[consumed..])?;
        let mut generated = vec![];
        for _ in 0..max_new_tokens.min(self.graph.capacity - self.cached.len()) {
            let token = sampler.sample(&self.last_logits)?;
            self.ingest(&[token])?;
            generated.push(token);
            if stops.contains(&token) {
                break;
            }
        }
        Ok(generated)
    }
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        cfg!(feature = "cuda_lite"),
        "build with --features cuda: this validator executes on a CUDA GPU"
    );
    ensure!(
        args.atol.is_finite() && args.atol > 0. && args.rtol.is_finite() && args.rtol >= 0.,
        "invalid tolerances"
    );
    ensure!(
        args.search_generations > 0 && args.search_population > 0,
        "search generations and population must be positive"
    );
    let config = checkpoint::read_json(&args.checkpoint.join("config.json"))?;
    let model = ModelConfig::from_checkpoint(args.model, &config)?;
    let dtype = checkpoint_dtype(&config)?;
    let suite: Option<Suite> = args
        .suite
        .as_ref()
        .map(|p| -> Result<_> { Ok(serde_json::from_value(checkpoint::read_json(p)?)?) })
        .transpose()?;
    let capacity = suite.as_ref().map_or(128, |s| s.max_context);
    // A reference suite can intentionally use a smaller context.
    let prefill_chunk = args.prefill_chunk.min(capacity);
    let graph = LlmGraph::build_with_parameter_dtype(model, dtype, capacity, prefill_chunk)?;
    if args.inspect {
        let index = checkpoint::read_json(&args.checkpoint.join("model.safetensors.index.json"))?;
        let names = index["weight_map"]
            .as_object()
            .context("missing weight_map")?;
        let missing: Vec<_> = graph
            .parameters
            .iter()
            .filter(|p| !names.contains_key(&p.checkpoint_name))
            .map(|p| &p.checkpoint_name)
            .collect();
        println!(
            "{}",
            json!({"parameters": graph.parameters.len(), "f32_weight_bytes": graph.parameters.iter().map(|p| p.shape.iter().product::<usize>() * 4).sum::<usize>(), "missing_names": missing})
        );
        ensure!(
            missing.is_empty(),
            "checkpoint is missing {} parameters",
            missing.len()
        );
        return Ok(());
    }
    let suite = suite.context("--suite is required for execution")?;
    ensure!(!suite.cases.is_empty(), "reference suite has no cases");
    ensure!(
        suite.model
            == format!("{:?}", args.model)
                .to_lowercase()
                .replace("qwen3moe", "qwen3-moe"),
        "suite model differs from selected adapter"
    );
    if let Some(revision) = &suite.checkpoint_revision {
        ensure!(
            std::fs::read_to_string(args.checkpoint.join("REVISION"))?.trim() == revision,
            "checkpoint revision differs from reference"
        );
    }
    let tokenizer = ChatTokenizer::load(&args.checkpoint, &config)?;
    ensure!(
        tokenizer.stop_tokens == suite.stop_tokens.iter().copied().collect(),
        "EOS tokens differ from Transformers"
    );
    eprintln!("Loading {} parameters", graph.parameters.len());
    let weights = checkpoint::load(&args.checkpoint, &graph.parameters)?;
    eprintln!(
        "Compiling with prefill chunk {} and context {capacity}",
        prefill_chunk
    );
    let backend = CudaBackend::compile(
        &graph,
        weights,
        &CompileOptions {
            generations: args.search_generations,
            generation_size: args.search_population,
            seed: 0,
            search_log: false,
            ..Default::default()
        },
    )?;
    let mut checked = Checked {
        graph,
        backend,
        cached: vec![],
        last_logits: vec![],
        expected: None,
        atol: args.atol,
        rtol: args.rtol,
        steps: 0,
        resets: 0,
        max_abs: 0.,
        max_ratio: 0.,
    };
    let mut history = Vec::new();
    let mut sampler = Sampler::new(0., 1., 0)?;
    let mut results: Vec<Value> = vec![];
    let reference_dir = args
        .suite
        .as_ref()
        .unwrap()
        .parent()
        .context("suite parent")?;
    let mut failure = None;
    for case in suite.cases {
        checked.steps = 0;
        checked.resets = 0;
        checked.max_abs = 0.;
        checked.max_ratio = 0.;
        let result = (|| -> Result<_> {
            if case.reset {
                checked.reset()?;
                history.clear();
            }
            let user = case.messages.last().context("missing user message")?;
            ensure!(user.role == "user", "fixture must end in a user message");
            history.push(user.clone());
            ensure!(
                serde_json::to_value(&history)? == serde_json::to_value(&case.messages)?,
                "conversation history differs"
            );
            let prompt = tokenizer.encode_chat(&history, false)?;
            ensure!(
                prompt == case.prompt_tokens,
                "chat-template token IDs differ from Transformers"
            );
            let file = File::open(reference_dir.join(&case.logits_file))?;
            // Fixtures are immutable while this validator executes.
            checked.expected = Some(unsafe { MmapOptions::new().map(&file)? });
            let tokens = checked.generate(
                &prompt,
                suite.max_new_tokens,
                &tokenizer.stop_tokens,
                &mut sampler,
            )?;
            let text = tokenizer.decode(&tokens)?;
            ensure!(
                tokens == case.generated_tokens,
                "greedy tokens differ: actual={tokens:?}, expected={:?}; text={text:?}",
                case.generated_tokens
            );
            ensure!(text == case.text, "decoded text differs");
            history.push(Message::new("assistant", &text));
            Ok(text)
        })();
        let error = result.as_ref().err().map(|e| format!("{e:#}"));
        eprintln!(
            "{}: {} ({} steps, max_abs={:.6}, resets={})",
            case.name,
            result.as_deref().unwrap_or("FAILED"),
            checked.steps,
            checked.max_abs,
            checked.resets
        );
        results.push(
            json!({"case": case.name, "passed": result.is_ok(), "text": result.ok(), "error": error,
            "steps": checked.steps, "resets": checked.resets,
            "max_abs_error": checked.max_abs, "max_tolerance_ratio": checked.max_ratio}),
        );
        if error.is_some() {
            failure = error;
            break;
        }
    }
    let report = json!({"model": suite.model, "checkpoint_revision": suite.checkpoint_revision,
        "backend": "cuda_lite",
        "prefill_chunk": prefill_chunk, "max_context": capacity, "atol": args.atol, "rtol": args.rtol,
        "search_generations": args.search_generations, "search_population": args.search_population, "seed": 0,
        "passed": failure.is_none(), "cases": results});
    if let Some(path) = args.report {
        std::fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(error) = failure {
        anyhow::bail!(error);
    }
    Ok(())
}
