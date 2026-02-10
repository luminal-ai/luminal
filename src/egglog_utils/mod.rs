use colored::Colorize;
use egglog::{ast::Span, prelude::RustSpan, var};
use itertools::Itertools;
use petgraph::{Direction, graph::NodeIndex, visit::EdgeRef};
use rand::{Rng, rngs::ThreadRng};
use rustc_hash::FxHashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{str, sync::Arc};
use tracing::trace;

pub const BASE: &str = include_str!("base.egg");
pub const BASE_CLEANUP: &str = include_str!("base_cleanup.egg");
pub const RUN_SCHEDULE: &str = include_str!("run_schedule.egg");

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    let ops_str = ops
        .iter()
        .map(|o| {
            let (name, body) = o.term();
            format!(
                "({name} {})",
                body.into_iter().map(|j| format!("{j:?}")).join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "
    (datatype*
        (IR
            (OutputJoin IR IR)
            {ops_str}
        )
        (IList
            (ICons IR IList)
            (INil)
        )
    )
    (function dtype (IR) DType :merge new)
    "
    )
}

fn op_cleanups_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    format!(
        "
    {}
    ",
        ops.iter()
            .filter(|op| op.cleanup())
            .map(|o| {
                let (name, body) = o.term();
                let body_terms = (0..body.len()).map(|i| (b'a' + i as u8) as char).join(" ");
                format!(
                    "(rule
                ((= ?m ({name} {body_terms})))
                ((delete ({name} {body_terms})))
                :ruleset cleanup
            )"
                )
            })
            .join("\n")
    )
}

pub fn early_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> String {
    [
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.early_rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        format!(
            "(run-schedule
                (saturate expr)
                (run)
                (saturate base_cleanup)
            )
            (extract {root})"
        ),
    ]
    .join("\n")
}

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    [
        BASE.to_string(),
        op_defs_string(ops),
        ops.iter().flat_map(|o| o.rewrites()).join("\n"),
        if cleanup {
            op_cleanups_string(ops)
        } else {
            "".to_string()
        },
        BASE_CLEANUP.to_string(),
        program.to_string(),
        RUN_SCHEDULE.to_string(),
    ]
    .join("\n")
}

use crate::{
    graph::{Graph, LLIRGraph, SubgraphDescriptor},
    hlir::{Input, Output},
    op::{CustomOp, DType, EgglogOp},
    prelude::FxHashMap,
    shape::Expression,
};
use egglog::{ArcSort, CommandOutput, EGraph, Value};
use egraph_serialize::{ClassId, NodeId};

#[derive(Debug)]
///  This is snapshot of an EGraph with Rust native hash maps and sets for enabling more native traversal / algorithm writing.
///  The name comes from the serialize egraph crates, which returns a ETermDAG, which caused issues, so this is a homebrew semi-static egraph
pub struct SerializedEGraph {
    pub enodes: FxHashMap<NodeId, (String, Vec<ClassId>)>,
    pub eclasses: FxHashMap<ClassId, (String, Vec<NodeId>)>,
    pub node_to_class: FxHashMap<NodeId, ClassId>,
    pub roots: Vec<ClassId>,
}

impl SerializedEGraph {
    /// This is an opinionated function which does more than strictly take the state of the egglog object.
    /// It also filters out "[...]" nodes and then changes the structure from the e-termDAG that egraph-serialize
    /// produces to a strict egraph, where the children of e-classes are e-nodes.
    pub fn new(egraph: &EGraph, root_eclasses: Vec<(ArcSort, Value)>) -> Self {
        let s = egraph.serialize(egglog::SerializeConfig {
            root_eclasses,
            max_functions: None,
            include_temporary_functions: false,
            max_calls_per_function: None,
        });
        // Convert to SerializedEGraph
        let mut classes = FxHashMap::default();
        for (node_id, node) in &s.egraph.nodes {
            classes
                .entry(node.eclass.clone())
                .or_insert(vec![])
                .push(node_id.clone())
        }
        let mut s_egraph = SerializedEGraph {
            roots: s.egraph.root_eclasses,
            node_to_class: s
                .egraph
                .nodes
                .iter()
                .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
                .collect(),
            enodes: s
                .egraph
                .nodes
                .iter()
                .map(|(n, enode)| {
                    (
                        n.clone(),
                        (
                            enode.op.clone(),
                            enode
                                .children
                                .iter()
                                .map(|n| s.egraph.nodes[n].eclass.clone())
                                .collect(),
                        ),
                    )
                })
                .collect(),
            eclasses: s
                .egraph
                .class_data
                .iter()
                .map(|(c, eclass)| (c.clone(), (eclass.typ.clone().unwrap(), classes[c].clone())))
                .collect(),
        };
        // Strip out all [...] enodes
        s_egraph.enodes.retain(|_, (label, _)| label != "[...]");
        loop {
            let mut to_remove = vec![];
            for (id, (_, children)) in &s_egraph.enodes {
                if children.iter().any(|c| {
                    !s_egraph.eclasses[c]
                        .1
                        .iter()
                        .any(|n| s_egraph.enodes.contains_key(n))
                }) {
                    to_remove.push(id.clone());
                }
            }
            for n in &to_remove {
                s_egraph.enodes.remove(n);
            }
            if to_remove.is_empty() {
                break;
            }
        }
        // Correct the eclass mapping
        for (_, enodes) in s_egraph.eclasses.values_mut() {
            enodes.retain(|n| s_egraph.enodes.contains_key(n));
        }
        s_egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
        s_egraph
            .node_to_class
            .retain(|n, _| s_egraph.enodes.contains_key(n));
        s_egraph
    }
}

/// Hash a SerializedEGraph by its structural content for dedup comparison.
/// Only considers IR/IList eclasses and enodes (not primitives like i64, String, DType
/// which contain per-chunk-specific values like node indices and weight labels).
pub fn hash_serialized_egraph(egraph: &SerializedEGraph) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Only count IR/IList eclasses (computation nodes, not primitives)
    let ir_eclasses: Vec<_> = egraph
        .eclasses
        .values()
        .filter(|(label, _)| label.contains("IR") || label.contains("IList"))
        .collect();
    ir_eclasses.len().hash(&mut hasher);
    let mut eclass_info: Vec<_> = ir_eclasses
        .iter()
        .map(|(label, enodes)| (label.clone(), enodes.len()))
        .collect();
    eclass_info.sort();
    eclass_info.hash(&mut hasher);
    // Only hash IR/IList enodes by op name and child count
    let mut enode_info: Vec<_> = egraph
        .enodes
        .iter()
        .filter(|(node_id, _)| {
            let eclass = &egraph.node_to_class[*node_id];
            if let Some((label, _)) = egraph.eclasses.get(eclass) {
                label.contains("IR") || label.contains("IList")
            } else {
                false
            }
        })
        .map(|(_, (op, children))| (op.clone(), children.len()))
        .collect();
    enode_info.sort();
    enode_info.hash(&mut hasher);
    hasher.finish()
}

/// Hash egglog text with normalization for structural dedup.
///
/// Structurally identical chunks (e.g. transformer layers) produce identical
/// egglog text except for:
/// - Input node indices and labels (differ per layer)
/// - Output node indices (differ per layer)
/// - CustomOpHLIR integer IDs (global custom_ops index, differs per layer)
///
/// This function hashes the text while normalizing those chunk-specific values:
/// - Input lines: only the dtype is hashed (not node index or label)
/// - Output lines: only the "OUTPUT" marker is hashed (not the node index)
/// - CustomOpHLIR lines: the integer ID is replaced with a constant
/// - All other lines (ops, shapes, strides): hashed verbatim
pub fn hash_egglog_normalized(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for line in text.lines() {
        if line.contains("(Input ") {
            // Format: (let tN (Input NODE "LABEL" (DTYPE)))
            // Strip the node index and label, keep only the dtype.
            // The dtype is the last parenthesized token, e.g. "(F32)".
            if let Some(dtype_start) = line.rfind(" (") {
                let dtype = &line[dtype_start + 1..];
                ("INPUT", dtype).hash(&mut hasher);
            } else {
                line.hash(&mut hasher);
            }
        } else if line.contains("(Output ") && !line.contains("(OutputJoin ") {
            "OUTPUT".hash(&mut hasher);
        } else if line.contains("(CustomOpHLIR ") {
            // Format: (let tN (CustomOpHLIR (ICons ... (INil)) ID (DTYPE)))
            // The integer ID varies per layer. Replace it with a constant.
            normalize_custom_op_id(line).hash(&mut hasher);
        } else {
            line.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Replace the integer ID in a CustomOpHLIR egglog line with a constant "0".
/// Input format: `(let tN (CustomOpHLIR (ICons ... (INil))) ID (DTYPE)))`
/// The ID is the integer between the closing of IList and the opening of DType.
fn normalize_custom_op_id(line: &str) -> String {
    if let Some(custom_start) = line.find("(CustomOpHLIR ") {
        let after = &line[custom_start + "(CustomOpHLIR ".len()..];
        // Find the dtype opening paren (last " (" in the line)
        if let Some(last_open) = after.rfind(" (") {
            let before_dtype = &after[..last_open];
            // Find the space before the integer ID
            if let Some(space_before_id) = before_dtype.rfind(' ') {
                let id_str = &before_dtype[space_before_id + 1..];
                if id_str.chars().all(|c| c.is_ascii_digit()) {
                    return format!(
                        "{}0{}",
                        &line[..custom_start + "(CustomOpHLIR ".len() + space_before_id + 1],
                        &line[custom_start + "(CustomOpHLIR ".len() + last_open..]
                    );
                }
            }
        }
    }
    line.to_string()
}

pub fn hlir_to_egglog(graph: &Graph) -> (String, String) {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    // 1. Topo-order with tie-break: lower NodeIndex first
    let mut indeg: HashMap<NodeIndex, usize> = graph
        .node_indices()
        .map(|n| (n, graph.neighbors_directed(n, Direction::Incoming).count()))
        .collect();

    let mut ready: BinaryHeap<(Reverse<usize>, NodeIndex)> = BinaryHeap::new();
    for (n, &d) in &indeg {
        if d == 0 {
            ready.push((Reverse(n.index()), *n));
        }
    }

    let mut topo_order: Vec<NodeIndex> = Vec::with_capacity(indeg.len());
    while let Some((_, n)) = ready.pop() {
        topo_order.push(n);
        for succ in graph.neighbors_directed(n, Direction::Outgoing) {
            let e = indeg.get_mut(&succ).unwrap();
            *e -= 1;
            if *e == 0 {
                ready.push((Reverse(succ.index()), succ));
            }
        }
    }

    // 2. Map <node-id> → <egglog var name>
    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut out = String::new();

    let mut curr_id = 0;
    for n in topo_order {
        let sources = graph
            .get_sources(n)
            .into_iter()
            .zip(
                graph
                    .edges_directed(n, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| names[&e.source()].clone()),
            )
            .map(|((n, sh), name)| (n, name, sh))
            .collect_vec();
        let code = graph[n].to_egglog(&sources);
        out.push_str(&format!("(let t{curr_id} {code})\n"));
        names.insert(n, format!("t{curr_id}"));
        curr_id += 1;
    }

    // Join outputs using dummy op
    let names = graph
        .externals(Direction::Outgoing)
        .map(|n| names.remove(&n).unwrap())
        .collect_vec();
    let mut root = names[0].clone();
    for node in names.into_iter().skip(1) {
        curr_id += 1;
        out.push_str(&format!("(let t{curr_id} (OutputJoin {root} {node}))\n"));
        root = format!("t{curr_id}");
    }
    (out.replace("(MVar \"z\")", "(MIter)"), root)
}

/// Convert a subgraph of the HLIR to egglog, injecting synthetic Input/Output
/// nodes at graph break boundaries.
pub fn hlir_subgraph_to_egglog(graph: &Graph, subgraph: &SubgraphDescriptor) -> (String, String) {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    let mut names: HashMap<NodeIndex, String> = HashMap::new();
    let mut out = String::new();
    let mut curr_id = 0;

    // Emit synthetic Input nodes for boundary inputs
    for boundary in &subgraph.boundary_inputs {
        let var_name = format!("t{curr_id}");
        let code = format!(
            "(Input {} \"boundary\" ({:?}))",
            boundary.break_node.index(),
            boundary.dtype
        );
        out.push_str(&format!("(let {var_name} {code})\n"));
        // Map the GraphBreak node to this synthetic Input variable.
        // When downstream nodes reference the GraphBreak as a source, they'll use this.
        names.insert(boundary.break_node, var_name);
        curr_id += 1;
    }

    // Topo-order only the nodes in this subgraph
    // Build sub-indeg map restricted to subgraph nodes
    let mut indeg: HashMap<NodeIndex, usize> = HashMap::new();
    for &n in &subgraph.nodes {
        let count = graph
            .graph
            .neighbors_directed(n, Direction::Incoming)
            .filter(|pred| subgraph.nodes.contains(pred))
            .count();
        indeg.insert(n, count);
    }

    let mut ready: BinaryHeap<(Reverse<usize>, NodeIndex)> = BinaryHeap::new();
    for (&n, &d) in &indeg {
        if d == 0 {
            ready.push((Reverse(n.index()), n));
        }
    }

    let mut topo_order: Vec<NodeIndex> = Vec::with_capacity(indeg.len());
    while let Some((_, n)) = ready.pop() {
        topo_order.push(n);
        for succ in graph.graph.neighbors_directed(n, Direction::Outgoing) {
            if let Some(e) = indeg.get_mut(&succ) {
                *e -= 1;
                if *e == 0 {
                    ready.push((Reverse(succ.index()), succ));
                }
            }
        }
    }

    // Convert each node in topological order to egglog
    for n in topo_order {
        let sources = graph
            .get_sources(n)
            .into_iter()
            .zip(
                graph
                    .graph
                    .edges_directed(n, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .map(|e| {
                        names.get(&e.source()).cloned().unwrap_or_else(|| {
                            panic!("Missing egglog name for node {:?}", e.source())
                        })
                    }),
            )
            .map(|((n, sh), name)| (n, name, sh))
            .collect_vec();
        let code = graph.graph[n].to_egglog(&sources);
        out.push_str(&format!("(let t{curr_id} {code})\n"));
        names.insert(n, format!("t{curr_id}"));
        curr_id += 1;
    }

    // Emit synthetic Output nodes for boundary outputs
    for &brk in &subgraph.boundary_outputs {
        // The predecessor of the GraphBreak is the actual producer
        let pred = graph
            .graph
            .neighbors_directed(brk, Direction::Incoming)
            .next()
            .expect("GraphBreak must have exactly one input");
        let pred_name = names.get(&pred).cloned().unwrap_or_else(|| {
            panic!(
                "Missing egglog name for boundary output predecessor {:?}",
                pred
            )
        });
        let code = format!("(Output {} {})", pred_name, brk.index());
        out.push_str(&format!("(let t{curr_id} {code})\n"));
        names.insert(brk, format!("t{curr_id}"));
        curr_id += 1;
    }

    // Join outputs: real outputs (nodes with no outgoing edges within the subgraph)
    // plus boundary outputs
    let mut output_names: Vec<String> = vec![];

    // Boundary outputs
    for &brk in &subgraph.boundary_outputs {
        if let Some(name) = names.get(&brk) {
            output_names.push(name.clone());
        }
    }

    // Real outputs: only actual Output HLIR ops that exist in this subgraph
    // (not arbitrary nodes that happen to have no subgraph successors)
    for &n in &subgraph.nodes {
        if graph.try_get_op::<Output>(n).is_some() {
            if let Some(name) = names.get(&n) {
                output_names.push(name.clone());
            }
        }
    }

    if output_names.is_empty() {
        // Fallback: use the last node added
        output_names.push(format!("t{}", curr_id - 1));
    }

    // Join with OutputJoin
    let mut root = output_names[0].clone();
    for node in output_names.into_iter().skip(1) {
        curr_id += 1;
        out.push_str(&format!("(let t{curr_id} (OutputJoin {root} {node}))\n"));
        root = format!("t{curr_id}");
    }

    (out.replace("(MVar \"z\")", "(MIter)"), root)
}

pub fn elist_to_egglog(shape: &[Expression]) -> String {
    list_to_egglog(
        &shape.iter().map(|e| e.to_egglog()).collect_vec(),
        "ECons",
        "ENil",
    )
}

pub fn list_to_egglog(list: &[impl ToString], cons: &str, nil: &str) -> String {
    if list.is_empty() {
        format!("({nil})")
    } else {
        format!(
            "({cons} {} {})",
            list[0].to_string(),
            list_to_egglog(&list[1..], cons, nil)
        )
    }
}

fn termdag_to_egglog(td: &egglog::TermDag, root: egglog::TermId) -> (String, String) {
    let mut out = String::new();
    for id in 0..td.size() {
        let code = match td.get(id) {
            egglog::Term::Lit(lit) => format!("{lit}"),
            egglog::Term::Var(v) => v.clone(),
            egglog::Term::App(head, args) => format!(
                "({head} {})",
                args.iter().map(|s| format!("t{s}")).join(" ")
            ),
        };
        out.push_str(&format!("(let t{id} {code})\n"));
    }
    (out.replace("(MVar \"z\")", "(MIter)"), format!("t{root}"))
}

#[tracing::instrument(skip_all)]
pub fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    let start = std::time::Instant::now();
    let code = early_egglog(program, root, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    let outputs = egraph.run_program(commands)?;
    let CommandOutput::ExtractBest(termdag, _cost, term) = outputs.last().unwrap() else {
        panic!();
    };
    let (program, root) = termdag_to_egglog(termdag, termdag.lookup(term));
    let code = full_egglog(&program, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    trace!("{}", "Egglog running...".green());
    let _outputs = egraph.run_program(commands)?;
    trace!("{}", "---- Egglog Rule Matches ----".green());
    let run_report = egraph.get_overall_run_report();
    trace!(
        "{}",
        run_report
            .num_matches_per_rule
            .iter()
            .filter(|(k, _)| !k.contains("("))
            .map(|(k, v)| format!(
                "{k}: {v} ({})",
                pretty_duration::pretty_duration(
                    &run_report.search_and_apply_time_per_rule[k],
                    None
                )
            ))
            .join("\n")
            .green()
    );
    trace!(
        "{}",
        format!(
            "---- Egglog Took {} ----",
            pretty_duration::pretty_duration(&start.elapsed(), None).bold()
        )
        .green()
    );
    // if enabled!(Level::TRACE) {
    //     let log_dir = Path::new("egraph");
    //     if log_dir.exists() {
    //         fs::remove_dir_all(log_dir).unwrap();
    //     }
    //     fs::create_dir(log_dir).unwrap();
    //     fs::write(log_dir.join("egraph.dot"), egraph.to_dot().unwrap()).unwrap();
    //     fs::write(log_dir.join("egraph.html"), egraph.to_html().unwrap()).unwrap();
    // }
    let (sort, value) = egraph.eval_expr(&var!(root))?;
    let s = egraph.serialize(egglog::SerializeConfig {
        root_eclasses: vec![(sort, value)],
        max_functions: None,
        include_temporary_functions: false,
        max_calls_per_function: None,
    });
    // Convert to SerializedEGraph
    let mut classes = FxHashMap::default();
    for (node_id, node) in &s.egraph.nodes {
        classes
            .entry(node.eclass.clone())
            .or_insert(vec![])
            .push(node_id.clone())
    }
    let mut egraph = SerializedEGraph {
        roots: s.egraph.root_eclasses,
        node_to_class: s
            .egraph
            .nodes
            .iter()
            .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
            .collect(),
        enodes: s
            .egraph
            .nodes
            .iter()
            .map(|(n, enode)| {
                (
                    n.clone(),
                    (
                        enode.op.clone(),
                        enode
                            .children
                            .iter()
                            .map(|n| s.egraph.nodes[n].eclass.clone())
                            .collect(),
                    ),
                )
            })
            .collect(),
        eclasses: s
            .egraph
            .class_data
            .iter()
            .map(|(c, eclass)| (c.clone(), (eclass.typ.clone().unwrap(), classes[c].clone())))
            .collect(),
    };
    // Strip out all [...] enodes
    egraph.enodes.retain(|_, (label, _)| label != "[...]");
    loop {
        let mut to_remove = vec![];
        for (id, (_, children)) in &egraph.enodes {
            if children.iter().any(|c| {
                !egraph.eclasses[c]
                    .1
                    .iter()
                    .any(|n| egraph.enodes.contains_key(n))
            }) {
                to_remove.push(id.clone());
            }
        }
        for n in &to_remove {
            egraph.enodes.remove(n);
        }
        if to_remove.is_empty() {
            break;
        }
    }
    // Correct the eclass mapping
    for (_, enodes) in egraph.eclasses.values_mut() {
        enodes.retain(|n| egraph.enodes.contains_key(n));
    }
    egraph.eclasses.retain(|_, (_, c)| !c.is_empty());
    egraph
        .node_to_class
        .retain(|n, _| egraph.enodes.contains_key(n));
    assert!(
        egraph.roots.iter().all(|c| egraph.eclasses.contains_key(c)),
        "No valid graphs present in the e-graph!"
    );

    Ok(egraph)
}

pub fn extract_expr_list<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> Option<Vec<Expression>> {
    if let Some(l) = list_cache.get(node) {
        return Some(l.clone());
    }
    if egraph.enodes[node].0 == "ENil" {
        return Some(vec![]);
    }
    let eclass = &egraph.enodes[node].1[0];
    let expr = extract_expr(egraph, &egraph.eclasses[eclass].1[0], expr_cache)?;
    match egraph.enodes[&egraph.eclasses[&egraph.enodes[node].1[1]].1[0]]
        .0
        .as_str()
    {
        "ENil" => Some(vec![expr]),
        "ECons" => {
            let mut rest = extract_expr_list(
                egraph,
                &egraph.eclasses[&egraph.enodes[node].1[1]].1[0],
                list_cache,
                expr_cache,
            )?;
            rest.insert(0, expr);
            list_cache.insert(node, rest.clone());
            Some(rest)
        }
        _ => unreachable!(),
    }
}

pub fn extract_dtype<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> DType {
    match egraph.enodes[node].0.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "Int" => DType::Int,
        "Bool" => DType::Bool,
        "NvFp4" => DType::NvFp4,
        "Mxfp4" => DType::Mxfp4,
        other => panic!("unknown dtype {other}"),
    }
}

pub fn extract_expr<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> Option<Expression> {
    if let Some(e) = expr_cache.get(node) {
        return Some(*e);
    }

    fn extract_shortest<'a>(
        egraph: &'a SerializedEGraph,
        class: &'a ClassId,
        seen: &mut FxHashMap<&'a NodeId, usize>,
        cache: &mut FxHashMap<&'a NodeId, Option<Vec<&'a NodeId>>>,
    ) -> Option<Vec<&'a NodeId>> {
        const MAX_CYCLES: usize = 1;
        egraph.eclasses[class]
            .1
            .iter()
            .filter_map(|en| {
                if *seen.get(en).unwrap_or(&0) >= MAX_CYCLES || egraph.enodes[en].0 == "[...]" {
                    return None;
                }
                if let Some(c) = cache.get(en) {
                    return c.clone();
                }
                *seen.entry(en).or_insert(0) += 1;
                let out = if egraph.enodes[en].1.is_empty() {
                    Some(vec![en])
                } else {
                    egraph.enodes[en]
                        .1
                        .iter()
                        .try_fold(vec![en], |mut acc, ch| {
                            extract_shortest(egraph, ch, seen, cache).map(|p| {
                                acc.extend(p);
                                acc
                            })
                        })
                };
                *seen.get_mut(en).unwrap() -= 1;
                cache.insert(en, out.clone());
                out
            })
            .min_by_key(|p| p.len())
    }

    let traj = extract_shortest(
        egraph,
        &egraph.node_to_class[node],
        &mut FxHashMap::default(),
        &mut FxHashMap::default(),
    )?;
    fn build_expression(
        egraph: &SerializedEGraph,
        trajectory: &[&NodeId],
        current: &mut usize,
    ) -> Expression {
        let nid = trajectory[*current];
        let op = egraph.enodes[nid].0.as_str();
        match op {
            // unary math
            "MNeg" | "MRecip" => {
                *current += 1;
                let c0 = build_expression(egraph, trajectory, current);
                match op {
                    "MNeg" => c0 * -1,
                    "MRecip" => 1 / c0,
                    _ => unreachable!(),
                }
            }
            // binary math
            "MAdd" | "MSub" | "MMul" | "MDiv" | "MMod" | "MMin" | "MMax" | "MAnd" | "MOr"
            | "MGte" | "MLt" | "MFloorTo" | "MCeilDiv" => {
                *current += 1;
                let lhs = build_expression(egraph, trajectory, current);
                *current += 1;
                let rhs = build_expression(egraph, trajectory, current);
                match op {
                    "MAdd" => lhs + rhs,
                    "MSub" => lhs - rhs,
                    "MMul" => lhs * rhs,
                    "MDiv" => lhs / rhs,
                    "MMod" => lhs % rhs,
                    "MMin" => lhs.min(rhs),
                    "MMax" => lhs.max(rhs),
                    "MAnd" => lhs & rhs,
                    "MOr" => lhs | rhs,
                    "MGte" => lhs.gte(rhs),
                    "MLt" => lhs.lt(rhs),
                    "MCeilDiv" => lhs.ceil_div(rhs),
                    "MFloorTo" => lhs / rhs * rhs, // TODO: real floorto in Expression
                    _ => unreachable!(),
                }
            }
            // wrappers around a literal/var child
            "MNum" | "MVar" => {
                *current += 1;
                build_expression(egraph, trajectory, current)
            }
            "MIter" => Expression::from('z'),
            op if op.starts_with("Boxed(\"") => {
                let name = op.replace("Boxed(\"", "").replace("\")", "");
                Expression::from(name.chars().next().unwrap())
            }
            op => op
                .parse::<i32>()
                .map(Expression::from)
                .or_else(|_| op.replace('"', "").parse::<char>().map(Expression::from))
                .unwrap_or_else(|_| panic!("unsupported expression op '{op}'")),
        }
    }
    let e = build_expression(egraph, &traj, &mut 0);
    expr_cache.insert(node, e);
    Some(e)
}

pub type EGraphChoiceSet<'a> = FxHashMap<&'a ClassId, &'a NodeId>;

pub fn random_initial_choice<'a>(
    egraph: &'a SerializedEGraph,
    rng: &mut ThreadRng,
) -> EGraphChoiceSet<'a> {
    let mut choices = FxHashMap::default();
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") && !label.contains("IList") {
            continue;
        }
        choices.insert(eclass, &enodes[rng.random_range(0..enodes.len())]);
    }
    choices
}

/// Validate that a choice set is complete and consistent.
/// Returns Ok(()) if valid, Err with description if invalid.
pub fn validate_choice_set<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
    ops: &[Arc<Box<dyn EgglogOp>>],
) -> Result<(), String> {
    // Check all IR/IList eclasses have a choice
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") && !label.contains("IList") {
            continue;
        }
        let Some(chosen) = choices.get(eclass) else {
            return Err(format!("Missing choice for eclass {}", eclass.as_ref()));
        };
        // Check chosen enode exists in the eclass
        if !enodes.contains(chosen) {
            return Err(format!(
                "Chosen enode {} not in eclass {}",
                chosen.as_ref(),
                eclass.as_ref()
            ));
        }
    }

    // Verify reachability from root
    let mut reachable = FxHashSet::default();
    let root_choice = choices
        .get(&egraph.roots[0])
        .ok_or_else(|| format!("No choice for root eclass {}", egraph.roots[0].as_ref()))?;
    reachable.insert(*root_choice);
    let mut stack = vec![*root_choice];
    while let Some(r) = stack.pop() {
        let (_, children) = egraph
            .enodes
            .get(r)
            .ok_or_else(|| format!("Enode {} not found in egraph", r.as_ref()))?;
        for ch in children {
            let (label, _) = egraph
                .eclasses
                .get(ch)
                .ok_or_else(|| format!("Eclass {} not found", ch.as_ref()))?;
            if label.contains("IR") || label.contains("IList") {
                let n = choices
                    .get(ch)
                    .ok_or_else(|| format!("No choice for reachable eclass {}", ch.as_ref()))?;
                if !reachable.contains(n) {
                    stack.push(n);
                    reachable.insert(n);
                }
            }
        }
    }

    // Check all reachable IR nodes have corresponding ops
    for node in &reachable {
        let (op_name, _) = &egraph.enodes[*node];
        let eclass = &egraph.node_to_class[*node];
        let (label, _) = &egraph.eclasses[eclass];
        if label != "IR" {
            continue; // Skip IList nodes
        }
        if op_name == "OutputJoin" || op_name == "CustomOpHLIR" {
            continue;
        }
        if !ops.iter().any(|op| op.term().0 == *op_name) {
            return Err(format!("No extractor for op {}", op_name));
        }
    }

    Ok(())
}

/// Hash a choice set for uniqueness checking
pub fn hash_choice_set(choices: &EGraphChoiceSet) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Sort by ClassId for deterministic hashing
    let mut sorted: Vec<_> = choices.iter().collect();
    sorted.sort_by(|(k1, _), (k2, _)| k1.as_ref().cmp(k2.as_ref()));
    for (class_id, node_id) in sorted {
        class_id.hash(&mut hasher);
        node_id.hash(&mut hasher);
    }
    hasher.finish()
}

/// Extract a generation of mutated offspring from a base genome.
///
/// Takes a base `EGraphChoiceSet` and produces up to `generation_size` mutated offspring,
/// each with `mutations_per_generation` random mutations. Offspring are deduplicated
/// against `prev_selected` (which is updated with new hashes).
///
/// If the search space is exhausted, returns as many unique offspring as possible.
pub fn extract_generation<'a>(
    egraph: &'a SerializedEGraph,
    base: &EGraphChoiceSet<'a>,
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut ThreadRng,
) -> Vec<EGraphChoiceSet<'a>> {
    // Get list of mutable eclasses (those with more than one enode option)
    let mutable_classes: Vec<&ClassId> = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .map(|(class_id, _)| class_id)
        .collect();

    // If there are no mutable classes, we can only return the base if it's unseen
    if mutable_classes.is_empty() {
        let h = hash_choice_set(base);
        if !prev_selected.contains(&h) {
            prev_selected.insert(h);
            return vec![base.clone()];
        }
        return vec![];
    }

    let mut offspring = Vec::with_capacity(generation_size);
    // Limit attempts to avoid infinite loops when search space is exhausted
    let max_attempts = generation_size * 100;
    let mut attempts = 0;

    while offspring.len() < generation_size && attempts < max_attempts {
        attempts += 1;

        // Create a mutated offspring from base
        let mut child = base.clone();

        for _ in 0..rng.random_range(1..=mutations_per_generation) {
            // Pick a random mutable eclass
            let class_id = mutable_classes[rng.random_range(0..mutable_classes.len())];
            let (_, enodes) = &egraph.eclasses[class_id];
            // Pick a random enode for this class
            child.insert(class_id, &enodes[rng.random_range(0..enodes.len())]);
        }

        // Hash and check if seen before
        let h = hash_choice_set(&child);
        if !prev_selected.contains(&h) {
            prev_selected.insert(h);
            offspring.push(child);
        }
    }
    offspring
}

#[tracing::instrument(skip_all)]
pub fn egglog_to_llir<'a>(
    egraph: &'a SerializedEGraph,
    choices: EGraphChoiceSet<'a>,
    ops: &'a Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    custom_op_id_remap: Option<&FxHashMap<usize, usize>>,
) -> LLIRGraph {
    // Get maps for all e-classes to e-node options
    // if enabled!(Level::DEBUG) {
    //     let log_dir = Path::new("llir_graphs");

    //     if log_dir.exists() {
    //         fs::remove_dir_all(log_dir).unwrap();
    //     }
    //     fs::create_dir(log_dir).unwrap();
    // }

    // Make reachability set from root
    let mut reachable = FxHashSet::default();
    reachable.insert(choices[&egraph.roots[0]]);
    let mut reachability_stack = vec![choices[&egraph.roots[0]]];
    while let Some(r) = reachability_stack.pop() {
        for ch in &egraph.enodes[r].1 {
            if egraph.eclasses[ch].0.contains("IR") || egraph.eclasses[ch].0.contains("IList") {
                let n = choices[ch];
                if !reachable.contains(n) {
                    reachability_stack.push(n);
                    reachable.insert(n);
                }
            }
        }
    }
    let mut graph = LLIRGraph::default();
    let mut edges_to_place = vec![];
    let mut enode_to_node = FxHashMap::default();
    for &node in choices.values() {
        if !reachable.contains(node) {
            continue;
        }
        if egraph.eclasses[&egraph.node_to_class[node]].0 != "IR" {
            // Skip IList
            continue;
        }
        let ch = egraph.enodes[node]
            .1
            .iter()
            .map(|c| {
                if egraph.eclasses[c].0.contains("IR") || egraph.eclasses[c].0.contains("IList") {
                    choices[c]
                } else {
                    &egraph.eclasses[c].1[0]
                }
            })
            .collect_vec();
        if egraph.enodes[node].0.as_str() == "CustomOpHLIR" {
            // Extract custom op inputs and id
            let mut inputs = vec![];
            // Walk through the IList to get inputs - use choices[] for IR/IList eclasses
            let ilist_eclass = &egraph.enodes[node].1[0];
            let mut ch = choices[ilist_eclass];
            loop {
                if egraph.enodes[ch].0 == "INil" {
                    break;
                } else {
                    // The first child of ICons is an IR node - use choices[] to get the chosen enode
                    let input_eclass = &egraph.enodes[ch].1[0];
                    inputs.push(choices[input_eclass]);
                    // The second child of ICons is the rest of the IList - use choices[] for the tail
                    ch = choices[&egraph.enodes[ch].1[1]];
                }
            }
            let id: usize = egraph.enodes[&egraph.eclasses[&egraph.enodes[node].1[1]].1[0]]
                .0
                .parse()
                .unwrap();
            let remapped_id = custom_op_id_remap
                .and_then(|m| m.get(&id).copied())
                .unwrap_or(id);
            let r = graph.add_node(custom_ops[remapped_id].to_llir_op());
            enode_to_node.insert(node, r);
            for source in inputs {
                edges_to_place.push((source, node));
            }
        } else if egraph.enodes[node].0.as_str() != "OutputJoin" {
            let Some(op) = ops
                .iter()
                .find(|op| egraph.enodes[node].0.as_str() == op.term().0)
            else {
                todo!("{} extraction not implemented!", egraph.enodes[node].0);
            };
            // Extract this op
            let (op_instance, sources) = op.extract(egraph, &ch, list_cache, expr_cache);
            let r = graph.add_node(op_instance);
            enode_to_node.insert(node, r);
            for source in sources {
                edges_to_place.push((source, node));
            }
        }
    }
    for (src, dest) in edges_to_place {
        let src_node_id = *enode_to_node.get(&src).unwrap_or_else(|| {
            panic!("Source enode {src:?} not found in enode_to_node map during edge placement")
        });
        let dest_node_id = *enode_to_node.get(&dest).unwrap_or_else(|| {
            panic!(
                "Destination enode {dest:?} not found in enode_to_node map during edge placement",
            )
        });

        graph.add_edge(src_node_id, dest_node_id, ());
    }
    // if enabled!(Level::TRACE) {
    //     fs::write(
    //         format!("llir_graphs/llir_{}.dot", i),
    //         graph.clone().to_dot().unwrap(),
    //     )
    //     .unwrap();
    // }
    graph
}

/// Merge multiple per-chunk LLIR graphs into a single LLIR graph,
/// resolving boundary Input/Output nodes at graph break boundaries.
pub fn stitch_llir_graphs(
    chunk_llirs: &[LLIRGraph],
    descriptors: &[SubgraphDescriptor],
) -> LLIRGraph {
    use petgraph::stable_graph::NodeIndex;

    let mut merged = LLIRGraph::default();

    // Collect the set of boundary break_node indices for matching
    let mut boundary_output_set: FxHashSet<usize> = FxHashSet::default();
    let mut boundary_input_set: FxHashSet<usize> = FxHashSet::default();
    for desc in descriptors {
        for brk in &desc.boundary_outputs {
            boundary_output_set.insert(brk.index());
        }
        for bi in &desc.boundary_inputs {
            boundary_input_set.insert(bi.break_node.index());
        }
    }

    // Per-chunk node mapping: old NodeIndex -> new NodeIndex in merged graph
    let mut node_maps: Vec<FxHashMap<NodeIndex, NodeIndex>> = Vec::with_capacity(chunk_llirs.len());

    // Track boundary producers: break_node_index -> new NodeIndex of the actual producer
    let mut boundary_producers: FxHashMap<usize, NodeIndex> = FxHashMap::default();

    // Track real Input node deduplication: Input.node -> new NodeIndex
    let mut real_inputs: FxHashMap<usize, NodeIndex> = FxHashMap::default();

    for (_chunk_idx, chunk_graph) in chunk_llirs.iter().enumerate() {
        let mut this_map: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();

        // Pass 1: Add all non-boundary nodes
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];

            // Check if this is a boundary Output
            if let Some(output_op) = op.to_op::<Output>() {
                if boundary_output_set.contains(&output_op.node) {
                    // Skip — will resolve in pass 2
                    continue;
                }
            }

            // Check if this is a boundary Input
            if let Some(input_op) = op.to_op::<Input>() {
                if boundary_input_set.contains(&input_op.node) {
                    // Skip — will resolve in pass 2
                    continue;
                }

                // Check if this is a real Input that was already added (dedup)
                if let Some(&existing) = real_inputs.get(&input_op.node) {
                    this_map.insert(old_node, existing);
                    continue;
                }
            }

            let new_node = merged.add_node(op.clone());
            this_map.insert(old_node, new_node);

            // Track real inputs for deduplication
            if let Some(input_op) = op.to_op::<Input>() {
                real_inputs.insert(input_op.node, new_node);
            }
        }

        // Pass 2: Resolve boundary Output nodes (record the producer)
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];
            if let Some(output_op) = op.to_op::<Output>() {
                if boundary_output_set.contains(&output_op.node) {
                    // Find the predecessor (the actual producer)
                    let pred = chunk_graph
                        .neighbors_directed(old_node, petgraph::Direction::Incoming)
                        .next()
                        .expect("Boundary Output must have exactly one input");
                    if let Some(&producer_new) = this_map.get(&pred) {
                        boundary_producers.insert(output_op.node, producer_new);
                    } else {
                        eprintln!(
                            "[stitch] WARNING: chunk {}: boundary Output node={} predecessor {:?} not in this_map!",
                            _chunk_idx,
                            output_op.node,
                            pred.index()
                        );
                    }
                }
            }
        }

        // Pass 2b: Resolve boundary Input nodes (map to producer from prior chunk)
        for old_node in chunk_graph.node_indices() {
            let op = &chunk_graph[old_node];
            if let Some(input_op) = op.to_op::<Input>() {
                if boundary_input_set.contains(&input_op.node) {
                    if let Some(&producer) = boundary_producers.get(&input_op.node) {
                        this_map.insert(old_node, producer);
                    } else {
                        eprintln!(
                            "[stitch] WARNING: chunk {}: boundary Input node={} has no producer in boundary_producers!",
                            _chunk_idx, input_op.node
                        );
                        eprintln!(
                            "[stitch]   available producers: {:?}",
                            boundary_producers.keys().collect::<Vec<_>>()
                        );
                    }
                }
            }
        }

        // Pass 3: Add edges (preserving duplicate edges for ops like x*x)
        for edge in chunk_graph.edge_indices() {
            let (src, dst) = chunk_graph.edge_endpoints(edge).unwrap();
            if let (Some(&new_src), Some(&new_dst)) = (this_map.get(&src), this_map.get(&dst)) {
                if new_src != new_dst {
                    merged.add_edge(new_src, new_dst, ());
                }
            }
        }

        node_maps.push(this_map);
    }

    merged
}
