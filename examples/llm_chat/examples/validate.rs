//! Real-checkpoint differential validation, separate from the product CLI.
#[cfg(any(feature = "cuda_lite", feature = "metal"))]
mod enabled {
    use anyhow::{Context, Result, ensure};
    use clap::Parser;
    use llm_chat::{
        Inputs,
        backend::{Backend, CompileOptions, GpuBackend},
        checkpoint,
        graph::{LlmGraph, ModelConfig, ModelType},
        sampling::Sampler,
        session::Session,
        tokenizer::{ChatTokenizer, Message},
    };
    use memmap2::{Mmap, MmapOptions};
    use safetensors::{Dtype, SafeTensors};
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::{fs::File, path::PathBuf};

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
        #[arg(long, default_value_t = 8)]
        prefill_chunk: usize,
        #[arg(long, default_value_t = 2)]
        search_generations: usize,
        #[arg(long, default_value_t = 4)]
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
    struct CheckedBackend {
        inner: GpuBackend,
        expected: Option<Mmap>,
        atol: f32,
        rtol: f32,
        steps: usize,
        resets: usize,
        max_abs: f32,
        max_ratio: f32,
    }
    impl Backend for CheckedBackend {
        fn step(&mut self, inputs: Inputs, query: usize, context: usize) -> Result<Vec<f32>> {
            let actual = self.inner.step(inputs, query, context)?;
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
            Ok(actual)
        }
        fn reset(&mut self) -> Result<()> {
            self.resets += 1;
            self.inner.reset()
        }
    }
    pub fn main() -> Result<()> {
        let args = Args::parse();
        ensure!(
            cfg!(feature = "cuda_lite") ^ cfg!(feature = "metal"),
            "select exactly one backend"
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
        let suite: Option<Suite> = args
            .suite
            .as_ref()
            .map(|p| -> Result<_> { Ok(serde_json::from_value(checkpoint::read_json(p)?)?) })
            .transpose()?;
        let capacity = suite.as_ref().map_or(16, |s| s.max_context);
        let graph = LlmGraph::build(model, capacity, args.prefill_chunk)?;
        if args.inspect {
            let index =
                checkpoint::read_json(&args.checkpoint.join("model.safetensors.index.json"))?;
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
            args.prefill_chunk
        );
        let inner = GpuBackend::compile(
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
        let mut session = Session::new(
            graph,
            CheckedBackend {
                inner,
                expected: None,
                atol: args.atol,
                rtol: args.rtol,
                steps: 0,
                resets: 0,
                max_abs: 0.,
                max_ratio: 0.,
            },
        );
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
            session.backend.steps = 0;
            session.backend.resets = 0;
            session.backend.max_abs = 0.;
            session.backend.max_ratio = 0.;
            let result = (|| -> Result<_> {
                if case.reset {
                    session.reset()?;
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
                session.backend.expected = Some(unsafe { MmapOptions::new().map(&file)? });
                let tokens = session.generate(
                    &prompt,
                    suite.max_new_tokens,
                    &tokenizer.stop_tokens,
                    &mut sampler,
                    |_| Ok(()),
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
                session.backend.steps,
                session.backend.max_abs,
                session.backend.resets
            );
            results.push(json!({"case": case.name, "passed": result.is_ok(), "text": result.ok(), "error": error,
                "steps": session.backend.steps, "resets": session.backend.resets,
                "max_abs_error": session.backend.max_abs, "max_tolerance_ratio": session.backend.max_ratio}));
            if error.is_some() {
                failure = error;
                break;
            }
        }
        let report = json!({"model": suite.model, "checkpoint_revision": suite.checkpoint_revision,
            "backend": if cfg!(feature = "metal") { "metal" } else { "cuda_lite" },
            "prefill_chunk": args.prefill_chunk, "max_context": capacity, "atol": args.atol, "rtol": args.rtol,
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
}
fn main() -> anyhow::Result<()> {
    #[cfg(any(feature = "cuda_lite", feature = "metal"))]
    {
        enabled::main()
    }
    #[cfg(not(any(feature = "cuda_lite", feature = "metal")))]
    anyhow::bail!("select --features cuda_lite or --features metal")
}
