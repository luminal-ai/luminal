#![allow(unused)]

//! Debug script to locate which HLIR op(s) fail to lower to a backend dialect,
//! leading to `No valid graphs present in the e-graph!`.
//!
//! This tool is backend-agnostic. The specific backend is selected via feature flags.
//! All core analysis logic lives in `luminal_bench::egglog_debug` module.
//!
//! Usage examples: see `crates/luminal_bench/README.md`.

use luminal::op::IntoEgglogOp;
use luminal::prelude::*;
use luminal::{egglog_utils::hlir_to_egglog, hlir::HLIROps};
use luminal_bench::egglog_debug::{
    DebugReport, FactQuery, analyze_hlir_dtype_chain, analyze_hlir_function_chain,
    analyze_lowering, analyze_with_ops, inspect_var_hlir, print_dtype_chain, print_function_chain,
    print_lowering_analysis, print_var_inspection, summarize_egglog_ops, summarize_hlir_ops,
};

// ============================================================================
// Backend Configuration
// ============================================================================

/// Backend-specific configuration trait.
trait BackendConfig {
    type Runtime: luminal::op::Runtime;
    const NAME: &'static str;

    fn build_search_space(cx: &mut Graph);
}

#[cfg(feature = "metal")]
mod metal_backend {
    use super::*;
    use luminal_metal::runtime::MetalRuntime;

    pub struct MetalConfig;

    impl BackendConfig for MetalConfig {
        type Runtime = MetalRuntime;
        const NAME: &'static str = "Metal";

        fn build_search_space(cx: &mut Graph) {
            cx.build_search_space::<MetalRuntime>();
        }
    }
}

#[cfg(feature = "metal")]
use metal_backend::MetalConfig as ActiveBackend;

// Future: Add CUDA backend
// #[cfg(feature = "cuda")]
// mod cuda_backend { ... }

// ============================================================================
// Test Cases
// ============================================================================

#[derive(Clone, Copy, Debug)]
enum Case {
    Mul,
    Sigmoid,
    Tanh,
    GeluInner,
    Gelu,
    LayerNorm,
}

impl Case {
    fn all() -> &'static [Case] {
        &[
            Case::Mul,
            Case::Sigmoid,
            Case::Tanh,
            Case::GeluInner,
            Case::Gelu,
            Case::LayerNorm,
        ]
    }

    fn from_str(s: &str) -> Option<Case> {
        match s {
            "mul" => Some(Case::Mul),
            "sigmoid" => Some(Case::Sigmoid),
            "tanh" => Some(Case::Tanh),
            "gelu-inner" => Some(Case::GeluInner),
            "gelu" => Some(Case::Gelu),
            "layer-norm" | "layer_norm" => Some(Case::LayerNorm),
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Case::Mul => "Mul",
            Case::Sigmoid => "Sigmoid",
            Case::Tanh => "Tanh",
            Case::GeluInner => "GeluInner",
            Case::Gelu => "Gelu",
            Case::LayerNorm => "LayerNorm",
        }
    }

    fn build(&self, cx: &mut Graph, size: usize) {
        let out = match self {
            Case::Mul => {
                let x = cx.tensor(size);
                x.clone() * x
            }
            Case::Sigmoid => cx.tensor(size).sigmoid(),
            Case::Tanh => cx.tensor(size).tanh(),
            Case::GeluInner => {
                let x = cx.tensor(size);
                (0.797_884_560_8_f32 * x.clone() * (1. + 0.044_715_f32 * x.clone() * x)).tanh()
            }
            Case::Gelu => cx.tensor(size).gelu(),
            Case::LayerNorm => {
                // Mirror `crates/luminal_bench/src/patterns.rs`: normalize along last axis.
                let hidden_dim = 128usize;
                let batch_seq = (size / hidden_dim).max(1);
                cx.tensor((batch_seq, hidden_dim)).layer_norm(1, 1e-5)
            }
        };
        let _ = out.output();
    }
}

// ============================================================================
// CLI Argument Parsing
// ============================================================================

struct Args {
    case: Case,
    size: usize,
    dump_egglog: Option<std::path::PathBuf>,
    print_egglog: bool,
    analyze: bool,
    inspect_vars: Vec<String>,
    inspect_ops: Vec<(String, String)>,
    trace_facts: Vec<(String, String)>,
    trace_first_missing_facts: Vec<TraceFirstMissingFact>,
    checks: Vec<Check>,
    json_out: Option<std::path::PathBuf>,
    all: bool,
}

#[derive(Clone, Debug)]
struct TraceFirstMissingFact {
    fn_name: String,
    within_op: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
    MissingBackend,
    DType,
    Function,
    All,
}

fn parse_args() -> Args {
    let mut args = Args {
        case: Case::Gelu,
        size: 262_144,
        dump_egglog: None,
        print_egglog: false,
        analyze: false,
        inspect_vars: Vec::new(),
        inspect_ops: Vec::new(),
        trace_facts: Vec::new(),
        trace_first_missing_facts: Vec::new(),
        checks: Vec::new(),
        json_out: None,
        all: false,
    };

    // If the user writes: --trace-first-missing-fact dtype --within-op Add
    // we attach the next --within-op to the last pending request.
    let mut pending_within_op_for: Option<usize> = None;

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--case" => {
                let val = iter.next().expect("Missing value for --case");
                args.case = Case::from_str(&val).unwrap_or_else(|| {
                    panic!(
                        "Unknown case: {}. Use: mul|sigmoid|tanh|gelu-inner|gelu",
                        val
                    )
                });
            }
            "--size" => {
                let val = iter.next().expect("Missing value for --size");
                args.size = val.parse().expect("Invalid --size value");
            }
            "--dump-egglog" => {
                let val = iter.next().expect("Missing value for --dump-egglog");
                args.dump_egglog = Some(val.into());
            }
            "--print-egglog" => args.print_egglog = true,
            "--analyze" => args.analyze = true,
            "--trace-fact" => {
                let fn_name = iter.next().expect("Missing function name for --trace-fact");
                let var = iter.next().expect("Missing variable for --trace-fact");
                args.trace_facts.push((fn_name, var));
            }
            "--trace-first-missing-fact" => {
                let fn_name = iter
                    .next()
                    .expect("Missing function name for --trace-first-missing-fact");
                args.trace_first_missing_facts.push(TraceFirstMissingFact {
                    fn_name,
                    within_op: String::new(),
                });
                pending_within_op_for = Some(args.trace_first_missing_facts.len() - 1);
            }
            "--within-op" => {
                let op = iter.next().expect("Missing op head for --within-op");
                let Some(idx) = pending_within_op_for.take() else {
                    eprintln!("--within-op must follow a --trace-first-missing-fact");
                    std::process::exit(2);
                };
                args.trace_first_missing_facts[idx].within_op = op;
            }
            "--inspect-var" => {
                let val = iter.next().expect("Missing value for --inspect-var");
                args.inspect_vars.push(val);
            }
            "--inspect-op" => {
                let val = iter.next().expect("Missing value for --inspect-op");
                let mut parts = val.split(':');
                let hlir = parts.next().unwrap_or("").to_string();
                let backend = parts.next().unwrap_or("").to_string();
                if hlir.is_empty() || backend.is_empty() || parts.next().is_some() {
                    eprintln!("Invalid --inspect-op format. Expected HLIR:Backend, got {val}");
                    std::process::exit(2);
                }
                args.inspect_ops.push((hlir, backend));
            }
            "--check" => {
                let val = iter.next().expect("Missing value for --check");
                let check = match val.as_str() {
                    "missing-backend" => Check::MissingBackend,
                    "dtype" => Check::DType,
                    "fn" | "function" => Check::Function,
                    "all" => Check::All,
                    _ => {
                        eprintln!("Unknown --check {val}. Use: missing-backend|dtype|fn|all");
                        std::process::exit(2);
                    }
                };
                args.checks.push(check);
            }
            "--json" => {
                let val = iter.next().expect("Missing value for --json");
                args.json_out = Some(val.into());
            }
            "--all" => args.all = true,
            "--help" | "-h" => {
                println!(
                    "Usage: debug_ops [OPTIONS]\n\n\
                    Options:\n  \
                      --case <CASE>       Test case: mul|sigmoid|tanh|gelu-inner|gelu (default: gelu)\n  \
                                         (also: layer-norm)\n  \
                      --size <N>          Tensor size (default: 262144)\n  \
                      --all               Run all test cases\n  \
                      --analyze           Run lowering analysis\n  \
                      --trace-fact FN VAR            Trace fact FN for VAR (HLIR-only), e.g. dtype t24\n  \
                      --trace-first-missing-fact FN  Find first missing FN within an op-head, then trace it (HLIR-only)\n  \
                      --within-op OPHEAD             Used with --trace-first-missing-fact (e.g. Add)\n  \
                      --inspect-var VAR   Print detailed eclass + dtype info for VAR (HLIR-only)\n  \
                      --inspect-op HLIR:Backend  Check backend coverage for an op mapping\n  \
                      --check KIND         Run checks: missing-backend|dtype|fn|all\n  \
                      --json PATH          Write JSON report (use '-' for stdout)\n  \
                      --dump-egglog PATH  Write egglog program to file\n  \
                      --print-egglog      Print egglog program to stdout\n  \
                      --help              Show this help"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}. Use --help for usage.", other);
                std::process::exit(2);
            }
        }
    }

    // Expand checks into concrete actions and validate requirements.
    if args.checks.contains(&Check::All) {
        args.checks = vec![Check::MissingBackend, Check::DType, Check::Function];
    }
    if args.checks.contains(&Check::DType) {
        // Preserve the previous semantics: scan Add for missing dtype, then trace.
        let already_has_add_dtype = args
            .trace_first_missing_facts
            .iter()
            .any(|r| r.fn_name == "dtype" && r.within_op == "Add");
        if !already_has_add_dtype {
            args.trace_first_missing_facts.push(TraceFirstMissingFact {
                fn_name: "dtype".to_string(),
                within_op: "Add".to_string(),
            });
        }
    }
    if args.checks.contains(&Check::MissingBackend) && args.inspect_ops.is_empty() {
        eprintln!("--check missing-backend requires at least one --inspect-op HLIR:Backend");
        std::process::exit(2);
    }
    if args.checks.contains(&Check::Function) && args.trace_facts.is_empty() {
        eprintln!("--check fn requires at least one --trace-fact FN VAR");
        std::process::exit(2);
    }

    args
}

// ============================================================================
// Main Logic
// ============================================================================

fn run_case<B: BackendConfig>(case: Case, size: usize, args: &Args)
where
    B::Runtime: luminal::op::Runtime,
    <B::Runtime as luminal::op::Runtime>::Ops: luminal::op::IntoEgglogOp,
{
    println!(
        "\n=== Case: {} (size={}) [{}] ===",
        case.name(),
        size,
        B::NAME
    );

    // Build graph
    let mut cx = Graph::default();
    case.build(&mut cx, size);

    // Summarize HLIR
    let hlir_counts = summarize_hlir_ops(&cx);
    println!("-- HLIR node types --");
    for (k, v) in &hlir_counts {
        println!("  {}: {}", k, v);
    }

    // Get egglog program
    let (program, root) = hlir_to_egglog(&cx);

    // Summarize egglog ops
    let egglog_counts = summarize_egglog_ops(&program);
    println!("-- Egglog op heads --");
    for (k, v) in &egglog_counts {
        println!("  {}: {}", k, v);
    }
    println!("-- Egglog root: {} --", root);

    // Dump egglog if requested
    if let Some(ref base_path) = args.dump_egglog {
        let path = if args.all {
            let parent = base_path.parent().unwrap_or(std::path::Path::new("."));
            let stem = base_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("debug");
            parent.join(format!("{}-{}.egg", stem, case.name()))
        } else {
            base_path.clone()
        };

        let content = format!("; hlir_to_egglog dump\n; root: {root}\n{program}");
        std::fs::write(&path, content).expect("Failed to write egglog file");
        println!("Wrote egglog program to {}", path.display());
    }

    if args.print_egglog {
        println!("-- Egglog program --\n{program}");
    }

    let find_vars_by_head = |head: &str| -> Vec<String> {
        let mut vars = Vec::new();
        for line in program.lines() {
            let line = line.trim();
            if !line.starts_with("(let ") {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() >= 3 && tokens[0] == "(let" {
                let var = tokens[1].to_string();
                let op = tokens[2].trim_start_matches('(');
                if op == head {
                    vars.push(var);
                }
            }
        }
        vars
    };

    // Validate any pending --within-op pairing.
    for req in &args.trace_first_missing_facts {
        if req.within_op.is_empty() {
            eprintln!(
                "--trace-first-missing-fact {} requires --within-op OPHEAD",
                req.fn_name
            );
            std::process::exit(2);
        }
    }

    // Prepare fact queries needed for scan-first-missing-fact.
    let mut hlir_analysis = None;
    let mut backend_analysis = None;
    let mut fact_queries: Vec<FactQuery> = Vec::new();
    for req in &args.trace_first_missing_facts {
        fact_queries.push(FactQuery {
            fn_name: req.fn_name.clone(),
            vars: find_vars_by_head(&req.within_op),
        });
    }

    // Only compute backend analysis if requested; compute HLIR analysis if needed
    // for either --analyze or scan-first-missing-fact.
    let need_backend_analysis = args.analyze || !args.inspect_ops.is_empty();
    let need_hlir_analysis = args.analyze || !fact_queries.is_empty();

    if need_backend_analysis {
        let (hlir, backend) =
            analyze_lowering::<B::Runtime>(&program, &root, &fact_queries, &args.inspect_ops);
        hlir_analysis = Some(hlir);
        backend_analysis = Some(backend);
    } else if need_hlir_analysis {
        let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
        hlir_analysis = Some(analyze_with_ops(
            &program,
            &root,
            hlir_ops,
            "HLIR",
            &fact_queries,
            &[],
        ));
    }

    if args.analyze {
        println!("-- Lowering analysis --");
        if let Some(ref hlir) = hlir_analysis {
            print_lowering_analysis(hlir);
        }
        if let Some(ref backend) = backend_analysis {
            print_lowering_analysis(backend);
        }
    } else if !args.inspect_ops.is_empty() {
        if let Some(ref backend) = backend_analysis {
            print_lowering_analysis(backend);
        }
    }

    // Trace facts for explicit variables.
    let mut function_traces = Vec::new();
    for (fn_name, var) in &args.trace_facts {
        if fn_name == "dtype" {
            println!("-- Trace dtype chain for {} (HLIR-only) --", var);
            let chain = analyze_hlir_dtype_chain(&program, var);
            print_dtype_chain(&chain);
            // Also record a function-trace entry for JSON output.
            function_traces.push(analyze_hlir_function_chain(&program, fn_name, var));
        } else {
            let trace = analyze_hlir_function_chain(&program, fn_name, var);
            print_function_chain(&trace);
            function_traces.push(trace);
        }
    }

    // Scan for first missing fact within an op-head, then trace.
    for req in &args.trace_first_missing_facts {
        let Some(ref hlir) = hlir_analysis else {
            println!(
                "-- Trace first missing fact (fn={}) within op={} --",
                req.fn_name, req.within_op
            );
            println!("  error  Skipped: HLIR analysis did not run");
            continue;
        };

        let vars = find_vars_by_head(&req.within_op);
        if vars.is_empty() {
            println!(
                "-- Trace first missing fact (fn={}) within op={} --",
                req.fn_name, req.within_op
            );
            println!("  √ No matching vars found (op head not present)");
            continue;
        }

        let table = hlir.facts.get(&req.fn_name);
        let first_missing = table.and_then(|t| {
            vars.iter()
                .find_map(|v| t.get(v).and_then(|s| s.is_missing().then(|| v.clone())))
        });

        if let Some(var) = first_missing {
            println!(
                "-- Trace first missing fact (fn={}) within op={} --",
                req.fn_name, req.within_op
            );
            println!("  ❌ first missing at: {}", var);
            if req.fn_name == "dtype" {
                let chain = analyze_hlir_dtype_chain(&program, &var);
                print_dtype_chain(&chain);
                function_traces.push(analyze_hlir_function_chain(&program, "dtype", &var));
            } else {
                let trace = analyze_hlir_function_chain(&program, &req.fn_name, &var);
                print_function_chain(&trace);
                function_traces.push(trace);
            }
        } else {
            println!(
                "-- Trace first missing fact (fn={}) within op={} --",
                req.fn_name, req.within_op
            );
            println!("  √ No missing values found");
        }
    }

    let mut var_inspections = Vec::new();
    if !args.inspect_vars.is_empty() {
        for var in &args.inspect_vars {
            let inspection = inspect_var_hlir(&program, var);
            print_var_inspection(&inspection);
            var_inspections.push(inspection);
        }
    }

    // Try to build search space
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        B::build_search_space(&mut cx);
    }));
    std::panic::set_hook(prev_hook);

    let build_succeeded = result.is_ok();
    match result {
        Ok(()) => println!("√ build_search_space succeeded"),
        Err(_) => println!("❌ build_search_space failed"),
    }

    if let Some(ref path) = args.json_out {
        let report = DebugReport {
            case_name: case.name().to_string(),
            size,
            hlir_counts,
            egglog_counts,
            hlir_analysis,
            backend_analysis,
            var_inspections,
            function_traces,
            build_succeeded,
        };
        let json = serde_json::to_string_pretty(&report).expect("failed to serialize report");
        if path.as_os_str() == "-" {
            println!("{}", json);
        } else {
            std::fs::write(path, json).expect("failed to write json report");
            println!("Wrote JSON report to {}", path.display());
        }
    }
}

#[cfg(feature = "metal")]
fn main() {
    let args = parse_args();

    println!("=== debug_ops ({}) ===", ActiveBackend::NAME);
    println!("Backend: {}", ActiveBackend::NAME);
    println!("Tip: Use --analyze for detailed lowering analysis.\n");

    if args.all {
        for case in Case::all() {
            run_case::<ActiveBackend>(*case, args.size, &args);
        }
    } else {
        run_case::<ActiveBackend>(args.case, args.size, &args);
    }
}

#[cfg(not(feature = "metal"))]
fn main() {}
