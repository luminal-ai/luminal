use std::{collections::BTreeMap, sync::Arc, time::Instant};

use itertools::Itertools;
use luminal::prelude::egglog::{ast::Span, prelude::RustSpan};
use luminal::{
    dtype::DType,
    egglog_utils::{
        base::{base_cleanup_egglog, base_expression_egglog},
        hlir_to_egglog,
    },
    hlir::HLIROps,
    op::{EgglogOp, IntoEgglogOp, Runtime},
    prelude::*,
    shape::Expression,
};
use luminal_cuda_lite::runtime::CudaRuntime;

const DEFAULT_PASSES: usize = 256;
const EGGLOG_RULESETS: &[&str] = &[
    "matmul_flatten",
    "kernel_lower",
    "direct_kernel",
    "kernel_specialize",
    "buffer_reuse",
    "matmul_backend",
    "glumoe",
    "fusion_pair",
    "fusion_grow",
    "fusion_merge",
];
const MOE_SEQ: usize = 2;
const MOE_HIDDEN: usize = 16;
const MOE_NUM_EXPERTS: usize = 8;
const MOE_TOP_K: usize = 2;
const MOE_INTERMEDIATE: usize = 6;
const GEMMA_RMS_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
enum Backend {
    Native,
    Cuda,
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Current,
    Steps,
    FullDefault,
    FullCycle,
}

#[derive(Debug, Clone, Copy)]
enum Case {
    Mul,
    UnaryChain(usize),
    Gelu,
    Softmax,
    LayerNorm,
    Matmul,
    Attention,
    QwenMoe,
    GemmaMoe,
}

#[derive(Debug)]
struct Args {
    backend: Backend,
    mode: Mode,
    case: Case,
    passes: usize,
    cleanup: bool,
    skip_roll: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        backend: Backend::Cuda,
        mode: Mode::Current,
        case: Case::Gelu,
        passes: DEFAULT_PASSES,
        cleanup: true,
        skip_roll: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--backend" => {
                args.backend = match iter.next().as_deref() {
                    Some("native") => Backend::Native,
                    Some("cuda") => Backend::Cuda,
                    other => panic!("invalid --backend {other:?}; use native|cuda"),
                };
            }
            "--mode" => {
                args.mode = match iter.next().as_deref() {
                    Some("current") => Mode::Current,
                    Some("steps") => Mode::Steps,
                    Some("full-default") => Mode::FullDefault,
                    Some("full-cycle") => Mode::FullCycle,
                    other => panic!(
                        "invalid --mode {other:?}; use current|steps|full-default|full-cycle"
                    ),
                };
            }
            "--case" => {
                args.case = parse_case(&iter.next().expect("missing --case value"));
            }
            "--passes" => {
                args.passes = iter
                    .next()
                    .expect("missing --passes value")
                    .parse()
                    .expect("invalid --passes value");
            }
            "--no-cleanup" => args.cleanup = false,
            "--skip-roll" => args.skip_roll = true,
            "--help" | "-h" => {
                println!(
                    "Usage: egglog_saturation [OPTIONS]\n\
                     \n\
                     Options:\n\
                       --backend native|cuda          default: cuda\n\
                       --mode current|steps|full-default|full-cycle\n\
                       --case mul|unary-chain:N|gelu|softmax|layer-norm|matmul|attention|qwen-moe|gemma-moe\n\
                       --passes N                    default: 256\n\
                       --no-cleanup                  omit backend/HLIR cleanup rules\n\
                       --skip-roll                   skip auto loop rolling prepass"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}; use --help"),
        }
    }

    args
}

fn parse_case(s: &str) -> Case {
    if let Some(n) = s.strip_prefix("unary-chain:") {
        return Case::UnaryChain(n.parse().expect("invalid unary-chain length"));
    }
    match s {
        "mul" => Case::Mul,
        "gelu" => Case::Gelu,
        "softmax" => Case::Softmax,
        "layer-norm" | "layer_norm" => Case::LayerNorm,
        "matmul" => Case::Matmul,
        "attention" => Case::Attention,
        "qwen-moe" | "qwen_moe" => Case::QwenMoe,
        "gemma-moe" | "gemma_moe" => Case::GemmaMoe,
        other => panic!("unknown case {other}"),
    }
}

fn build_case(case: Case) -> Graph {
    let mut cx = Graph::new();
    let out = match case {
        Case::Mul => {
            let x = cx.tensor((64, 64));
            x * x
        }
        Case::UnaryChain(n) => {
            let mut x = cx.tensor((64, 64));
            for i in 0..n {
                x = match i % 6 {
                    0 => x.sin(),
                    1 => x.sqrt(),
                    2 => x.reciprocal(),
                    3 => x.exp2(),
                    4 => x.log2(),
                    _ => x * 1.125,
                };
            }
            x
        }
        Case::Gelu => cx.tensor((64, 64)).gelu(),
        Case::Softmax => cx.tensor((128, 128)).softmax(1),
        Case::LayerNorm => cx.tensor((128, 128)).layer_norm(1, 1e-5),
        Case::Matmul => {
            let a = cx.tensor((32, 64));
            let b = cx.tensor((64, 32));
            a.matmul(b)
        }
        Case::Attention => {
            let q = cx.tensor((64, 32));
            let k = cx.tensor((64, 32));
            let v = cx.tensor((64, 32));
            let scores = q.matmul(k.permute((1, 0))) * (1.0 / 32.0_f32.sqrt());
            scores.softmax(1).matmul(v)
        }
        Case::QwenMoe => build_qwen_moe(&mut cx),
        Case::GemmaMoe => build_gemma_moe(&mut cx),
    };
    let _ = out.output();
    cx
}

fn build_qwen_moe(cx: &mut Graph) -> GraphTensor {
    cx.set_dim('s', MOE_SEQ);
    let x = cx.tensor(('s', MOE_HIDDEN));
    let router = cx.tensor((MOE_NUM_EXPERTS, MOE_HIDDEN));
    let gate_up_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_INTERMEDIATE * 2, MOE_HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = x.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(MOE_TOP_K);

    let routing_weights = x.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(MOE_TOP_K, n - 1);
    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);

    let gate_up_gathered = gather_experts(x, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = x.expand_dim(n - 1, MOE_TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(x, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

fn build_gemma_moe(cx: &mut Graph) -> GraphTensor {
    cx.set_dim('s', MOE_SEQ);
    let router_input = cx.tensor(('s', MOE_HIDDEN));
    let expert_input = cx.tensor(('s', MOE_HIDDEN));
    let router_scale = cx.tensor(MOE_HIDDEN);
    let router_proj = cx.tensor((MOE_NUM_EXPERTS, MOE_HIDDEN));
    let per_expert_scale = cx.tensor(MOE_NUM_EXPERTS);
    let gate_up_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_INTERMEDIATE * 2, MOE_HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = router_input.dims().len();
    let e_dim = *router_proj.dims().first().unwrap();
    let k_expr = Expression::from(MOE_TOP_K);

    let router_hidden = router_input.std_norm(n - 1, GEMMA_RMS_NORM_EPS)
        * router_scale.expand_lhs(&router_input.dims()[..n - 1])
        * (MOE_HIDDEN as f32).sqrt().recip();
    let routing_weights = router_hidden.matmul(router_proj.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(MOE_TOP_K, n - 1);
    let row_offsets = router_input
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_norm = top_k_values.sum(n - 1).expand_dim(n - 1, MOE_TOP_K);
    let top_k_weights = (top_k_values / top_k_norm) * per_expert_scale.gather(top_k_indices);

    let gate_up_gathered =
        gather_experts(expert_input, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = expert_input.expand_dim(n - 1, MOE_TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gemma_gelu(gate) * up;

    let down_gathered = gather_experts(expert_input, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_weights.unsqueeze(top_k_weights.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (axis, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(axis, *dim);
    }
    weights.gather(exp_base + exp_within)
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    let mut ir_variants = Vec::new();
    let mut opkind_variants = Vec::new();
    for op in ops {
        let sort = op.sort();
        let variant = format!(
            "({} {})",
            sort.name,
            sort.fields.iter().map(|field| &field.sort).join(" ")
        );
        match sort.class.as_str() {
            "IR" => ir_variants.push(variant),
            "OpKind" => opkind_variants.push(variant),
            other => panic!("unknown sort class {other} for {}", sort.name),
        }
    }
    let extra_ir = ops.iter().flat_map(|op| op.ir_defs()).unique().join("\n");
    format!(
        "
(datatype*
    (IR
        (OutputJoin IR IR)
        (Op OpKind IList)
        {extra_ir}
        {}
    )
    (OpKind
        {}
    )
    (IList
        (ICons IR IList)
        (INil)
    )
)
(function dtype (IR) DType :merge new)
",
        ir_variants.join("\n"),
        opkind_variants.join("\n")
    )
}

fn op_cleanups_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    ops.iter()
        .filter(|op| op.cleanup())
        .map(|op| {
            let sort = op.sort();
            let fields = (0..sort.fields.len())
                .map(|i| (b'a' + i as u8) as char)
                .join(" ");
            if sort.class == "OpKind" {
                format!(
                    "(rule
                       ((= ?m (Op ({} {fields}) ?__cleanup_inputs)))
                       ((delete (Op ({} {fields}) ?__cleanup_inputs)))
                       :ruleset cleanup)",
                    sort.name, sort.name
                )
            } else {
                format!(
                    "(rule
                       ((= ?m ({} {fields})))
                       ((delete ({} {fields})))
                       :ruleset cleanup)",
                    sort.name, sort.name
                )
            }
        })
        .join("\n")
}

fn setup_program(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    let rewrites = ops
        .iter()
        .flat_map(|op| op.rewrites())
        .map(|rule| rule.to_egglog_string())
        .join("\n");
    [
        EGGLOG_RULESETS
            .iter()
            .map(|ruleset| format!("(ruleset {ruleset})"))
            .join("\n"),
        base_expression_egglog(),
        op_defs_string(ops),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            String::new()
        },
        base_cleanup_egglog(),
        rewrites,
        program.to_string(),
    ]
    .join("\n")
}

fn producer_schedule() -> String {
    "(seq
        (saturate expr)
        (saturate dtype_prop)
        (run matmul_flatten)
        (run kernel_lower)
        (run direct_kernel)
        (run kernel_specialize)
        (run buffer_reuse)
        (run matmul_backend)
        (run glumoe)
        (run fusion_pair)
    )"
    .to_string()
}

fn fusion_schedule() -> String {
    "(seq
        (saturate expr)
        (saturate dtype_prop)
        (run fusion_grow)
        (run fusion_merge)
    )"
    .to_string()
}

fn split_cycle() -> Vec<(&'static str, String)> {
    vec![
        ("producers", format!("(saturate {})", producer_schedule())),
        ("fusion", format!("(saturate {})", fusion_schedule())),
    ]
}

fn split_cycle_schedule() -> String {
    format!(
        "(seq
            (saturate {})
            (saturate {})
        )",
        producer_schedule(),
        fusion_schedule()
    )
}

fn phase(egraph: &mut egglog::EGraph, name: &str, schedule: &str) -> bool {
    let before = egraph.num_tuples();
    let start = Instant::now();
    let command = format!("(run-schedule {schedule})");
    let outputs = egraph
        .parse_and_run_program(None, &command)
        .unwrap_or_else(|err| panic!("failed phase {name} schedule {schedule}: {err}"));
    let elapsed = start.elapsed();
    let after = egraph.num_tuples();
    let report = outputs
        .into_iter()
        .find_map(|output| match output {
            egglog::CommandOutput::RunSchedule(report) => Some(report),
            _ => None,
        })
        .expect("run-schedule did not return a report");
    let mut rules = report
        .search_and_apply_time_per_rule
        .iter()
        .map(|(rule, time)| {
            (
                rule.to_string(),
                *time,
                report
                    .num_matches_per_rule
                    .get(rule)
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect_vec();
    rules.sort_by_key(|(_, time, matches)| (std::cmp::Reverse(*time), std::cmp::Reverse(*matches)));
    let matches = report.num_matches_per_rule.values().sum::<usize>();
    println!(
        "phase {name:<18} {elapsed_ms:>8.2} ms | tuples {before} -> {after} ({delta:+}) | updated={updated} | iters={iters} | matches={matches}",
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        delta = after as isize - before as isize,
        updated = report.updated,
        iters = report.iterations.len(),
    );
    for (rule, time, matches) in rules
        .into_iter()
        .filter(|(_, time, matches)| !time.is_zero() || *matches > 0)
        .take(8)
    {
        println!(
            "  rule {rule:<82} {ms:>8.2} ms | matches {matches}",
            ms = time.as_secs_f64() * 1000.0,
        );
    }
    report.updated
}

fn serialize_summary(egraph: &mut egglog::EGraph, root: &str) {
    let (sort, value) = egraph.eval_expr(&egglog::var!(root.to_string())).unwrap();
    let output = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        max_functions: None,
        include_temporary_functions: false,
        max_calls_per_function: None,
    });
    let mut classes = std::collections::BTreeSet::new();
    let mut top_ops = BTreeMap::<String, usize>::new();
    let mut nodes = 0usize;
    for node in output.egraph.nodes.values().filter(|node| !node.subsumed) {
        nodes += 1;
        classes.insert(node.eclass.clone());
        *top_ops.entry(node.op.clone()).or_default() += 1;
    }
    let top_ops = top_ops
        .into_iter()
        .sorted_by_key(|(_, count)| std::cmp::Reverse(*count))
        .take(12)
        .map(|(op, count)| format!("{op}={count}"))
        .join(", ");
    println!(
        "serialize nodes={nodes} classes={} roots={} top_ops={top_ops}",
        classes.len(),
        output.egraph.root_eclasses.len()
    );
}

fn run(args: Args) {
    let mut graph = build_case(args.case);
    let rolled = if args.skip_roll {
        0
    } else {
        graph.auto_roll_loops_prepass()
    };
    let (program, root) = hlir_to_egglog(&graph);

    let mut ops = match args.backend {
        Backend::Native => <NativeRuntime as Runtime>::Ops::into_vec(),
        Backend::Cuda => <CudaRuntime as Runtime>::Ops::into_vec(),
    };
    ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
    let cleanup = args.cleanup && matches!(args.backend, Backend::Cuda);
    let setup = setup_program(&program, &ops, cleanup);

    println!(
        "case={:?} backend={:?} mode={:?} passes={} cleanup={} rolled={} hlir_nodes={} setup_lines={} setup_bytes={} root={root}",
        args.case,
        args.backend,
        args.mode,
        args.passes,
        cleanup,
        rolled,
        graph.graph.node_count(),
        setup.lines().count(),
        setup.len(),
    );

    let mut egraph = egglog::EGraph::default();
    let before = egraph.num_tuples();
    let start = Instant::now();
    let commands = egraph.parser.get_program_from_string(None, &setup).unwrap();
    egraph.run_program(commands).unwrap();
    println!(
        "setup {:>8.2} ms | tuples {before} -> {} ({:+})",
        start.elapsed().as_secs_f64() * 1000.0,
        egraph.num_tuples(),
        egraph.num_tuples() as isize - before as isize,
    );

    match args.mode {
        Mode::Current | Mode::Steps => {
            for pass in 1..=args.passes {
                let mut updated = false;
                for (name, schedule) in split_cycle() {
                    updated |= phase(&mut egraph, &format!("{pass:03} {name}"), &schedule);
                }
                if matches!(args.mode, Mode::Current) && !updated {
                    break;
                }
            }
        }
        Mode::FullDefault => {
            phase(&mut egraph, "expr", "(saturate expr)");
            phase(&mut egraph, "dtype", "(saturate dtype_prop)");
            phase(&mut egraph, "default-full", "(saturate (run))");
        }
        Mode::FullCycle => {
            phase(
                &mut egraph,
                "cycle-full",
                &format!("(saturate {})", split_cycle_schedule()),
            );
        }
    }

    phase(&mut egraph, "final expr", "(saturate expr)");
    if cleanup {
        phase(&mut egraph, "cleanup", "(saturate cleanup)");
    }
    phase(&mut egraph, "base cleanup", "(saturate base_cleanup)");
    serialize_summary(&mut egraph, &root);
}

fn main() {
    run(parse_args());
}
