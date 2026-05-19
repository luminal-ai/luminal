use std::{
    collections::HashMap,
    env,
    io::{self, Read, Write},
    process::{Command, Stdio},
    time::Instant,
};

const DEFAULT_EXAMPLES: &[&str] = &[
    "llama",
    "gemma",
    "qwen",
    "qwen3_moe",
    "gemma4_moe",
    "whisper",
];

#[derive(Clone)]
struct ExampleSpec {
    name: &'static str,
    cargo_args: &'static [&'static str],
    validation: Validation,
}

#[derive(Clone)]
enum Validation {
    Concepts(&'static [&'static [&'static str]]),
    Phrases(&'static [&'static str]),
}

#[derive(Default)]
struct Metrics {
    ttft_ms: Option<f64>,
    tpot_ms: Option<f64>,
    tps: Option<f64>,
}

struct ExampleResult {
    name: String,
    status: Result<(), String>,
    metrics: Metrics,
    elapsed_s: f64,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).filter(|arg| arg != "--").collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_help();
        return;
    }
    if args.iter().any(|arg| arg == "--list") {
        for example in DEFAULT_EXAMPLES {
            println!("{example}");
        }
        return;
    }

    let examples = if args.is_empty() {
        DEFAULT_EXAMPLES.iter().map(|s| s.to_string()).collect()
    } else {
        args
    };

    let specs = specs_by_name();
    let mut results = Vec::new();
    for example in examples {
        let Some(spec) = specs.get(example.as_str()) else {
            eprintln!("unknown validated example: {example}");
            eprintln!("known examples: {}", DEFAULT_EXAMPLES.join(", "));
            results.push(ExampleResult {
                name: example,
                status: Err("unknown example".to_string()),
                metrics: Metrics::default(),
                elapsed_s: 0.0,
            });
            continue;
        };
        results.push(run_example(spec));
    }

    print_table(&results);

    if results.iter().any(|result| result.status.is_err()) {
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "Run validated Luminal examples, validate textual output, and summarize perf.\n\
\n\
Usage:\n\
  cargo examples-perf\n\
  cargo examples-perf llama qwen whisper\n\
\n\
Options:\n\
  --list    Print the default validated examples\n\
  -h, --help\n\
\n\
The default set matches the Modal examples CI: {}.",
        DEFAULT_EXAMPLES.join(", ")
    );
}

fn specs_by_name() -> HashMap<&'static str, ExampleSpec> {
    let specs = [
        ExampleSpec {
            name: "llama",
            cargo_args: &["run", "--release", "-p", "llama"],
            validation: Validation::Concepts(&[
                &["layers"],
                &["neurons", "nodes"],
                &["learn", "learning", "adapt"],
                &["data", "patterns", "features"],
            ]),
        },
        ExampleSpec {
            name: "gemma",
            cargo_args: &["run", "--release", "-p", "gemma"],
            validation: Validation::Concepts(&[
                &["neural network", "neural networks"],
                &["nodes", "neurons"],
                &["layers"],
                &["weights"],
                &["training", "learn", "learns"],
            ]),
        },
        ExampleSpec {
            name: "qwen",
            cargo_args: &["run", "--release", "-p", "qwen", "--features", "cuda"],
            validation: Validation::Concepts(&[
                &["neural network", "neural networks"],
                &["computational model", "computational system"],
                &["brain"],
                &["layers"],
                &["neurons", "nodes"],
                &["learn", "learning", "training"],
            ]),
        },
        ExampleSpec {
            name: "qwen3_moe",
            cargo_args: &["run", "--release", "-p", "qwen3_moe"],
            validation: Validation::Concepts(&[&["capital"], &["france"], &["paris"]]),
        },
        ExampleSpec {
            name: "gemma4_moe",
            cargo_args: &["run", "--release", "-p", "gemma4_moe"],
            validation: Validation::Phrases(&["city of romance, art and culture"]),
        },
        ExampleSpec {
            name: "whisper",
            cargo_args: &["run", "--release", "-p", "whisper"],
            validation: Validation::Phrases(&["ask not what your country can do for you"]),
        },
    ];
    specs.into_iter().map(|spec| (spec.name, spec)).collect()
}

fn run_example(spec: &ExampleSpec) -> ExampleResult {
    println!("\n=== Running {} ===", spec.name);
    println!("$ cargo {}", spec.cargo_args.join(" "));
    let started = Instant::now();

    let mut command = Command::new("cargo");
    command.args(spec.cargo_args);
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    command.env(
        "CUDARC_CUDA_VERSION",
        env::var("CUDARC_CUDA_VERSION").unwrap_or_else(|_| "12080".to_string()),
    );

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            return ExampleResult {
                name: spec.name.to_string(),
                status: Err(format!("failed to start cargo: {err}")),
                metrics: Metrics::default(),
                elapsed_s: started.elapsed().as_secs_f64(),
            };
        }
    };

    let mut stdout = child.stdout.take().expect("child stdout must be piped");
    let mut stderr = child.stderr.take().expect("child stderr must be piped");
    let stdout_thread = std::thread::spawn(move || read_stream(&mut stdout, false));
    let stderr_thread = std::thread::spawn(move || read_stream(&mut stderr, true));

    let wait_status = child.wait();
    let mut output = stdout_thread.join().unwrap_or_default();
    output.push_str(&stderr_thread.join().unwrap_or_default());

    let metrics = parse_metrics(&output);
    let elapsed_s = started.elapsed().as_secs_f64();

    let status = match wait_status {
        Ok(status) if status.success() => validate_output(spec, &output),
        Ok(status) => Err(format!("process exited with {status}")),
        Err(err) => Err(format!("failed to wait for process: {err}")),
    };

    match &status {
        Ok(()) => println!("Output check passed for {:?}", spec.name),
        Err(err) => println!("Output check failed for {:?}: {err}", spec.name),
    }

    ExampleResult {
        name: spec.name.to_string(),
        status,
        metrics,
        elapsed_s,
    }
}

fn read_stream(stream: &mut dyn Read, is_stderr: bool) -> String {
    let mut collected = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                collected.extend_from_slice(&buf[..n]);
                if is_stderr {
                    let _ = io::stderr().write_all(&buf[..n]);
                    let _ = io::stderr().flush();
                } else {
                    let _ = io::stdout().write_all(&buf[..n]);
                    let _ = io::stdout().flush();
                }
            }
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&collected).into_owned()
}

fn validate_output(spec: &ExampleSpec, output: &str) -> Result<(), String> {
    let normalized = normalize_output(output);
    match &spec.validation {
        Validation::Concepts(groups) => {
            let missing: Vec<_> = groups
                .iter()
                .filter(|group| {
                    !group
                        .iter()
                        .any(|term| normalized.contains(&normalize_output(term)))
                })
                .collect();
            if missing.is_empty() {
                Ok(())
            } else {
                let missing = missing
                    .iter()
                    .map(|group| group.join(" / "))
                    .collect::<Vec<_>>()
                    .join("; ");
                Err(format!("missing concept groups: {missing}"))
            }
        }
        Validation::Phrases(phrases) => {
            if phrases
                .iter()
                .any(|phrase| normalized.contains(&normalize_output(phrase)))
            {
                Ok(())
            } else {
                Err(format!("missing expected phrase: {}", phrases.join(" / ")))
            }
        }
    }
}

fn normalize_output(output: &str) -> String {
    let mut stripped = String::with_capacity(output.len());
    let mut chars = output.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for seq_ch in chars.by_ref() {
                if ('@'..='~').contains(&seq_ch) {
                    break;
                }
            }
        } else if ch.is_whitespace() {
            stripped.push(' ');
        } else {
            stripped.extend(ch.to_lowercase());
        }
    }
    stripped
}

fn parse_metrics(output: &str) -> Metrics {
    let mut metrics = Metrics::default();
    for line in output.lines() {
        if let Some(value) = parse_number_after(line, "TTFT:") {
            metrics.ttft_ms = Some(value);
        }
        if let Some(value) = parse_number_after(line, "TPOT:") {
            metrics.tpot_ms = Some(value);
        }
        if let Some(value) = parse_tok_per_sec(line) {
            metrics.tps = Some(value);
        }
    }
    if metrics.tps.is_none() {
        metrics.tps = metrics.tpot_ms.map(|tpot_ms| 1000.0 / tpot_ms);
    }
    metrics
}

fn parse_number_after(line: &str, marker: &str) -> Option<f64> {
    let rest = line.split_once(marker)?.1.trim_start();
    parse_leading_number(rest)
}

fn parse_tok_per_sec(line: &str) -> Option<f64> {
    let tok_pos = line.find("tok/s")?;
    parse_trailing_number(&line[..tok_pos])
}

fn parse_leading_number(input: &str) -> Option<f64> {
    let number: String = input
        .chars()
        .take_while(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect();
    if number.is_empty() {
        None
    } else {
        number.parse().ok()
    }
}

fn parse_trailing_number(input: &str) -> Option<f64> {
    let trimmed = input.trim_end_matches(|ch: char| ch.is_whitespace() || ch == '(');
    let start = trimmed
        .rfind(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .map_or(0, |idx| idx + 1);
    trimmed[start..].parse().ok()
}

fn print_table(results: &[ExampleResult]) {
    println!("\nSummary");
    println!(
        "{:<14} {:<8} {:>10} {:>10} {:>10} {:>10}",
        "example", "status", "TTFT ms", "TPOT ms", "tok/s", "wall s"
    );
    println!("{}", "-".repeat(68));
    for result in results {
        let status = if result.status.is_ok() {
            "ok"
        } else {
            "failed"
        };
        println!(
            "{:<14} {:<8} {:>10} {:>10} {:>10} {:>10.1}",
            result.name,
            status,
            format_metric(result.metrics.ttft_ms),
            format_metric(result.metrics.tpot_ms),
            format_metric(result.metrics.tps),
            result.elapsed_s,
        );
    }
}

fn format_metric(metric: Option<f64>) -> String {
    metric
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llm_timing_metrics_and_derives_tps() {
        let metrics = parse_metrics(
            "\
  TTFT: 36.34 ms
  TPOT: 14.87 ms
",
        );
        assert_eq!(metrics.ttft_ms, Some(36.34));
        assert_eq!(metrics.tpot_ms, Some(14.87));
        assert!((metrics.tps.unwrap() - 67.2495).abs() < 1e-3);
    }

    #[test]
    fn parses_whisper_tokens_per_second() {
        let metrics = parse_metrics("Decoded 25 tokens in 2.00s (12.5 tok/s)");
        assert_eq!(metrics.ttft_ms, None);
        assert_eq!(metrics.tpot_ms, None);
        assert_eq!(metrics.tps, Some(12.5));
    }

    #[test]
    fn validates_concept_groups() {
        let specs = specs_by_name();
        let spec = specs.get("llama").unwrap();
        let output = "A neural network has layers of neurons that learn patterns in data.";
        assert!(validate_output(spec, output).is_ok());
    }
}
