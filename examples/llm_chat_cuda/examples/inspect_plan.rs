//! Reproduce chat's heuristic search and inventory its selected execution plan.
//! No weights or GPU execution are needed by the heuristic evaluator.
use anyhow::{Context, Result};
use clap::Parser;
use llm_chat_cuda::{
    backend::bindings,
    checkpoint,
    graph::{LlmGraph, ModelConfig, ModelType},
};
use luminal::{bufferize::BufferNode, prelude::FxHashMap};
use luminal_cuda_lite::{CompileOptions, CudaRuntime, cuda_registry};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, value_enum)]
    model: ModelType,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    report: PathBuf,
    #[arg(long, default_value_t = 2)]
    search_generations: usize,
    #[arg(long, default_value_t = 4)]
    search_population: usize,
    /// Inspect viable alternatives to each selected reduction.
    #[arg(long)]
    audit_candidates: bool,
}

fn candidate_audit(
    egraph: &luminal::prelude::egraph_serialize::EGraph,
    plan: &luminal_cuda_lite::layouts::CudaPlan,
) -> Result<serde_json::Value> {
    use luminal::extraction::ExtractionSession;
    let matchers = luminal_cuda_lite::ops::cuda_matchers();
    let session = ExtractionSession::new_with_matcher_set(egraph, None, &matchers);
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    let mut rows = Vec::new();
    for id in plan.dag.node_indices() {
        let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = &plan.dag[id]
        else {
            continue;
        };
        if op.label() != "ReduceSumGeneric" {
            continue;
        }
        let value = &result_info[0].value;
        let mut queue = VecDeque::from([(value.clone(), Vec::<serde_json::Value>::new())]);
        let mut seen = BTreeSet::new();
        let mut found = None;
        while let Some((class, path)) = queue.pop_front() {
            if !seen.insert(class.clone()) {
                continue;
            }
            let Some(entries) = index.get(&class) else {
                continue;
            };
            if let Some((label, choice)) = entries
                .iter()
                .find(|(label, _)| label.starts_with("LayoutTensorOpCublasLt"))
            {
                let mut route = path.clone();
                route.push(
                    json!({"value":class.to_string(), "op":label,"enode":choice.enode.to_string()}),
                );
                found = Some(route);
                break;
            }
            for (i, (label, choice)) in entries.iter().enumerate() {
                if !matches!(
                    label.as_str(),
                    "LayoutTensorOpCopyGeneric"
                        | "LayoutTensorOpIndexMapApplyViewGeneric"
                        | "LayoutTensorOpIndexMapApplyMaterialize"
                ) {
                    continue;
                }
                for source in &space.candidate_inputs[&class][i] {
                    let mut next = path.clone();
                    next.push(json!({"value":class.to_string(),"op":label,"enode":choice.enode.to_string()}));
                    queue.push_back((source.clone(), next));
                }
            }
        }
        rows.push(json!({"node":id.index(), "value":value.to_string(),
            "input_shape":format!("{:?}", operand_info[0].layout.shape()),
            "output_shape":format!("{:?}", result_info[0].layout.shape()),
            "has_viable_producer_row":index.contains_key(value),
            "cublas_route":found}));
    }
    Ok(
        json!({"reductions":rows,"method":"Search the runtime-viable producer index for a cuBLASLt candidate, allowing only copy and index-map view/materialization steps from each selected reduction's result. A route establishes a matched alternative, not its globally optimal cost or measured speed."}),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = checkpoint::read_json(&args.checkpoint.join("config.json"))?;
    let graph = LlmGraph::build(ModelConfig::from_checkpoint(args.model, &config)?, 256, 8)?;
    let registry = cuda_registry();
    let registered: Vec<_> = registry.iter().map(|op| op.label().to_owned()).collect();
    let mut runtime = CudaRuntime::load_with(&graph.graph, bindings(&graph), registry)?;
    runtime.bind_dyn_range('q', 1, graph.chunk_size as u64)?;
    runtime.bind_dyn_range('c', 1, graph.capacity as u64)?;
    runtime.set_dim('q', 1);
    runtime.set_dim('c', 1);
    eprintln!(
        "Searching {:?}: q=1..8, c=1..256, seed=0, generations={}, population={}",
        args.model, args.search_generations, args.search_population
    );
    let start = Instant::now();
    let mut options = CompileOptions {
        generations: args.search_generations,
        generation_size: args.search_population,
        seed: 0,
        search_log: false,
        profile_on_device: false,
        ..Default::default()
    };
    // Keep selection and audit on ONE e-graph: e-class identities are local
    // to a saturation run and must not be joined across fresh assemblies.
    let egraph = args
        .audit_candidates
        .then(|| runtime.saturated_egraph())
        .transpose()?;
    let outcome = if let Some(egraph) = &egraph {
        use luminal_cuda_lite::search::{Evaluator, SearchProgram, search_implementations};
        let bound = bindings(&graph)
            .bind(&graph.graph.logical)
            .map_err(anyhow::Error::msg)?;
        let program = SearchProgram {
            text: String::new(),
            inputs: bound.inputs,
            outputs: bound.outputs,
        };
        options.shapes.bounds.insert('q'.into(), (1, 8));
        options.shapes.bounds.insert('c'.into(), (1, 256));
        options.shapes.values.insert('q'.into(), 1);
        options.shapes.values.insert('c'.into(), 1);
        search_implementations(
            egraph,
            &program,
            &options,
            Some(CudaRuntime::allow_list()),
            &luminal_cuda_lite::ops::cuda_matchers(),
            Evaluator::Heuristic,
        )?
    } else {
        runtime.search(&FxHashMap::default(), &options)?
    };
    let plan = if args.audit_candidates {
        &outcome.best_plan
    } else {
        runtime.plan().context("missing selected plan")?
    };
    let mut counts = BTreeMap::<String, usize>::new();
    let mut nodes = Vec::new();
    for id in plan.dag.node_indices() {
        let node = &plan.dag[id];
        let label = match node {
            BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                nodes.push(json!({
                    "id": id.index(), "op": op.label(), "details": format!("{op:?}"),
                    "reads": reads.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>(),
                    "writes": writes.iter().map(|b| format!("{b:?}")).collect::<Vec<_>>(),
                    "operand_values": operand_info.iter().map(|s| s.value.to_string()).collect::<Vec<_>>(),
                    "result_values": result_info.iter().map(|s| s.value.to_string()).collect::<Vec<_>>(),
                    "operand_shapes": operand_info.iter().map(|s| format!("{:?}", s.layout.shape())).collect::<Vec<_>>(),
                    "result_shapes": result_info.iter().map(|s| format!("{:?}", s.layout.shape())).collect::<Vec<_>>()
                }));
                op.label()
            }
            BufferNode::BufferInput { .. } => "BufferInput",
            BufferNode::BufferOutput { .. } => "BufferOutput",
            BufferNode::BufferCopy { .. } => "BufferCopy",
        };
        *counts.entry(label.to_owned()).or_default() += 1;
    }
    let text_config = config.get("text_config").unwrap_or(&config);
    let search_seconds = start.elapsed().as_secs_f64();
    let audit = if args.audit_candidates {
        eprintln!("Auditing matched alternatives to selected reductions...");
        Some(candidate_audit(egraph.as_ref().unwrap(), plan)?)
    } else {
        None
    };
    let report = json!({
        "model": format!("{:?}", args.model), "checkpoint": args.checkpoint,
        "layers": text_config["num_hidden_layers"], "hidden_size": text_config["hidden_size"],
        "registered_ops": registered, "selected_counts": counts, "nodes": nodes,
        "search_seconds": search_seconds, "candidate_audit":audit,
        "search_settings": {"q_range": [1,8], "c_range": [1,256], "initial_q":1, "initial_c":1,
            "generations":args.search_generations, "population":args.search_population, "seed":0, "profile_on_device":false},
        "heuristic_cost": outcome.best_heuristic_cost.to_string(),
        "method": "Reproduced the chat backend's graph, bindings, dynamic bounds, initial dimensions, registry and heuristic search options. The heuristic does not read checkpoint values; empty host data is sufficient. One selected plan serves both prefill and decode."
    });
    std::fs::write(&args.report, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&counts)?);
    Ok(())
}
