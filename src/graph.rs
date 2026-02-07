use crate::{egglog_utils, hlir::CustomOpHLIR, op::*, prelude::*};
use std::{
    any::TypeId,
    fmt::Debug,
    io::Write,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use colored::Colorize;
use egglog::{CommandOutput, prelude::RustSpan, var};
use egglog_ast::span::Span;
use egraph_serialize::{ClassId, NodeId};
use itertools::Itertools;
use petgraph::{Direction, stable_graph::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::info;

pub type LLIRGraph = StableGraph<LLIROp, ()>;
pub type HLIRGraph = StableGraph<Box<dyn HLIROp>, ShapeTracker>;

/// A Luminal compute graph.
///
/// All computation is represented as a directed acyclic graph.
#[derive(Debug, Default)]
pub struct Graph {
    /// A map of dynamic dimensions to concrete dimension sizes
    pub dyn_map: FxHashMap<char, usize>,
    /// Edge weights: (Input index, Output index, Input shape)
    pub graph: HLIRGraph,
    /// E-Graph search space
    egraph: Option<SerializedEGraph>,
    /// Available ops
    pub ops: Option<Vec<Arc<Box<dyn EgglogOp>>>>,
    /// Custom ops
    pub custom_ops: Vec<Box<dyn CustomOp>>,
}

impl Graph {
    /// Create a new graph
    pub fn new() -> Graph {
        Graph::default()
    }

    /// Set a runtime dimension
    pub fn set_dim(&mut self, dimension: char, val: usize) {
        self.dyn_map.insert(dimension, val);
    }

    /// Create a new tensor with shape S
    pub fn tensor(&mut self, shape: impl ToShape) -> GraphTensor {
        self.named_tensor("Tensor", shape)
    }

    /// Create a new tensor with shape S and a name. This name will show up on the graph when displayed
    pub fn named_tensor(&mut self, name: impl ToString, shape: impl ToShape) -> GraphTensor {
        let id = self.graph.add_node(Box::new(crate::hlir::Input {
            node: 0,
            label: name.to_string(),
            dtype: DType::default(),
        }));
        self.get_op_mut::<crate::hlir::Input>(id).node = id.index();
        GraphTensor {
            id,
            graph_ref: self,
            shape: ShapeTracker::new(shape),
            dtype: DType::default(),
        }
    }

    /// Get the sources of a node given it's id
    pub fn get_sources(&self, node_id: NodeIndex) -> Vec<(NodeIndex, ShapeTracker)> {
        self.graph
            .edges_directed(node_id, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.source(), *e.weight()))
            .collect()
    }

    /// Get the dests of a node given it's id
    #[allow(clippy::borrowed_box)]
    pub fn get_dests(&self, node_id: NodeIndex) -> Vec<(NodeIndex, &Box<dyn HLIROp>)> {
        self.graph
            .edges_directed(node_id, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| (e.target(), &self.graph[e.target()]))
            .collect()
    }

    /// Add op on the graph, and get back a NewOp
    ///
    /// ```rust
    /// # use luminal::prelude::*;
    /// # let mut cx = Graph::new();
    /// let a = cx.tensor(3);
    /// let b_id = cx
    ///     .add_op(luminal::hlir::Mul::default())
    ///     .input(a.id, a.shape)
    ///     .finish();
    /// let b = GraphTensor::from_id(b_id, a.shape, a.graph(), a.dtype);
    /// ```
    pub fn add_op<O: HLIROp + 'static>(&mut self, op: O) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(Box::new(op)),
            graph_ref: self,
            num_srcs: 0,
        }
    }
    /// Add op on the graph, and get back a NewOp. Just like add_op, except a boxed op is expected.
    pub fn add_boxed_op(&mut self, op: Box<dyn HLIROp + 'static>) -> NewOp<'_> {
        NewOp {
            new_op_id: self.graph.add_node(op),
            graph_ref: self,
            num_srcs: 0,
        }
    }

    pub fn try_get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> Option<&T> {
        self.node_weight(node).unwrap().as_any().downcast_ref::<T>()
    }
    pub fn get_op<T: HLIROp + 'static>(&self, node: NodeIndex) -> &T {
        self.try_get_op(node).unwrap()
    }
    pub fn try_get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> Option<&mut T> {
        self.node_weight_mut(node)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<T>()
    }
    pub fn get_op_mut<T: HLIROp + 'static>(&mut self, node: NodeIndex) -> &mut T {
        self.try_get_op_mut(node).unwrap()
    }

    pub fn custom_op(
        &mut self,
        op: impl CustomOp + 'static,
        inputs: impl ToIds,
        shape: impl ToShape,
        dtype: DType,
    ) -> GraphTensor {
        self.custom_ops.push(Box::new(op));
        let mut add = self.add_op(CustomOpHLIR {
            id: self.custom_ops.len() - 1,
            dtype,
        });
        for input in inputs.to_ids() {
            add = add.input(input, ShapeTracker::new(()));
        }
        GraphTensor::from_id(add.finish(), ShapeTracker::new(shape), self, dtype)
    }

    #[tracing::instrument(skip_all)]
    pub fn build_search_space<Rt: Runtime + 'static>(&mut self) {
        let mut ops = Rt::Ops::into_vec();
        ops.extend(<crate::hlir::HLIROps as IntoEgglogOp>::into_vec());
        let (program, root) = hlir_to_egglog(self);
        self.egraph = Some(
            run_egglog(
                &program,
                &root,
                &ops,
                TypeId::of::<Rt>() != TypeId::of::<NativeRuntime>(), // need to ignore hlir op cleanups if we're on native runtime
            )
            .unwrap(),
        );
        self.ops = Some(ops);
    }

    #[tracing::instrument(skip_all)]
    pub fn search<R: Runtime>(&mut self, mut runtime: R, limit: usize) -> R {
        let llir_graphs = egglog_to_llir(
            self.egraph.as_ref().unwrap(),
            self.ops.as_ref().unwrap(),
            &self.custom_ops,
            limit,
        );
        let n_graphs = llir_graphs.len();
        let start = std::time::Instant::now();
        let mut best_graph = StableGraph::default();
        let mut best_metric: Option<R::ProfileMetric> = None;
        let total = llir_graphs.len();
        let bar_width = 24;

        let progress_bar = |i| {
            let head = ((i as f32 / total as f32) * bar_width as f32)
                .clamp(0.0, bar_width as f32)
                .floor() as usize;
            let bar = if head == 0 {
                format!("[>{}]", " ".repeat(bar_width - 1))
            } else if head >= bar_width {
                format!("[{}>]", "=".repeat(bar_width))
            } else {
                format!(
                    "[{}>{}]",
                    "=".repeat(head),
                    " ".repeat(bar_width - head - 1)
                )
            };
            print!(
                "\r\x1b[2K  {:>6}  {bar} {i}/{total}",
                "Searching".cyan().bold(),
            );
            std::io::stdout().flush().unwrap();
        };

        // Search loop
        for (i, llir_graph) in llir_graphs.into_iter().enumerate() {
            progress_bar(i + 1);
            let (new_metric, display_metric) = runtime.profile(&llir_graph, &self.dyn_map);
            let mut new_best = false;
            if let Some(old_metric) = &best_metric {
                if old_metric.gt(&new_metric) {
                    best_metric = Some(new_metric);
                    best_graph = llir_graph;
                    new_best = true;
                }
            } else {
                best_metric = Some(new_metric);
                best_graph = llir_graph;
                new_best = true;
            }
            print!("\r\x1b[2K"); // clear line
            std::io::stdout().flush().unwrap();
            println!(
                "   {:>6}  Graph {}: {}",
                "Searched".green().bold(),
                i + 1,
                if new_best {
                    display_metric.bold().green().to_string()
                } else {
                    display_metric
                }
            );
        }

        info!(
            target: "luminal::search",
            graphs = n_graphs,
            limit,
            limit_reached = n_graphs >= limit,
            duration_ms = start.elapsed().as_millis() as u64,
            "search completed"
        );
        println!(
            "   {:>6}  {n_graphs} graphs in {}",
            "Searched".green().bold(),
            pretty_duration::pretty_duration(&start.elapsed(), None)
        );
        runtime.load_llir(&best_graph);
        runtime
    }
}

impl Deref for Graph {
    type Target = HLIRGraph;
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Graph {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

pub struct NewOp<'a> {
    new_op_id: NodeIndex,
    graph_ref: &'a mut Graph,
    num_srcs: u8,
}

impl NewOp<'_> {
    pub fn finish(self) -> NodeIndex {
        self.new_op_id
    }

    pub fn input(mut self, id: NodeIndex, shape: ShapeTracker) -> Self {
        self.graph_ref.graph.add_edge(id, self.new_op_id, shape);
        self.num_srcs += 1;
        self
    }
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
fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    let start = std::time::Instant::now();
    let code = egglog_utils::early_egglog(program, root, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    let outputs = egraph.run_program(commands)?;
    let CommandOutput::ExtractBest(termdag, _cost, term) = outputs.last().unwrap() else {
        panic!();
    };
    let (program, root) = termdag_to_egglog(termdag, termdag.lookup(term));
    let code = egglog_utils::full_egglog(&program, ops, cleanup);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code)?;
    println!("{}", "Egglog running...".green());
    let _outputs = egraph.run_program(commands)?;
    println!("{}", "---- Egglog Rule Matches ----".green());
    let run_report = egraph.get_overall_run_report();
    println!(
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
    println!(
        "{}",
        format!(
            "---- Egglog Took {} ----",
            pretty_duration::pretty_duration(&start.elapsed(), None).bold()
        )
        .green()
    );

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
    match egraph.enodes[node].0.as_str() {
        "ENil" => return Some(vec![]),
        "Concat" => {
            let children = &egraph.enodes[node].1;
            if children.len() < 2 {
                return None;
            }
            let a_rep = &egraph.eclasses[&children[0]].1[0];
            let b_rep = &egraph.eclasses[&children[1]].1[0];
            let mut a_list = extract_expr_list(egraph, a_rep, list_cache, expr_cache)?;
            let b_list = extract_expr_list(egraph, b_rep, list_cache, expr_cache)?;
            a_list.extend(b_list);
            list_cache.insert(node, a_list.clone());
            return Some(a_list);
        }
        "Drop" => {
            let children = &egraph.enodes[node].1;
            if children.len() < 2 {
                return None;
            }
            let list_rep = &egraph.eclasses[&children[0]].1[0];
            let n_rep = &egraph.eclasses[&children[1]].1[0];
            let list = extract_expr_list(egraph, list_rep, list_cache, expr_cache)?;
            let n = extract_drop_count(egraph, n_rep, list_cache, expr_cache)?;
            let dropped: Vec<Expression> = list.into_iter().skip(n).collect();
            list_cache.insert(node, dropped.clone());
            return Some(dropped);
        }
        "Take" => {
            let children = &egraph.enodes[node].1;
            if children.len() < 2 {
                return None;
            }
            let list_rep = &egraph.eclasses[&children[0]].1[0];
            let n_rep = &egraph.eclasses[&children[1]].1[0];
            let list = extract_expr_list(egraph, list_rep, list_cache, expr_cache)?;
            let n = extract_drop_count(egraph, n_rep, list_cache, expr_cache)?;
            let taken: Vec<Expression> = list.into_iter().take(n).collect();
            list_cache.insert(node, taken.clone());
            return Some(taken);
        }
        _ => {}
    }
    // ECons
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
        "Concat" | "Drop" | "Take" => {
            let rest_rep = &egraph.eclasses[&egraph.enodes[node].1[1]].1[0];
            let mut rest = extract_expr_list(egraph, rest_rep, list_cache, expr_cache)?;
            rest.insert(0, expr);
            list_cache.insert(node, rest.clone());
            Some(rest)
        }
        _ => unreachable!(),
    }
}

fn extract_drop_count<'a>(
    egraph: &'a SerializedEGraph,
    node: &'a NodeId,
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
) -> Option<usize> {
    let label = &egraph.enodes[node].0;
    if let Ok(n) = label.parse::<i64>() {
        return Some(n.max(0) as usize);
    }
    if label == "len" && egraph.enodes[node].1.len() == 1 {
        let list_eclass = &egraph.enodes[node].1[0];
        let list_rep = &egraph.eclasses[list_eclass].1[0];
        let list = extract_expr_list(egraph, list_rep, list_cache, expr_cache)?;
        return Some(list.len());
    }
    None
}

pub fn extract_dtype<'a>(egraph: &'a SerializedEGraph, node: &'a NodeId) -> DType {
    match egraph.enodes[node].0.as_str() {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "Int" => DType::Int,
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

fn add_ilist_inputs_to_reachable<'a>(
    egraph: &'a SerializedEGraph,
    ilist_rep: &'a NodeId,
    choice: &'a FxHashMap<&'a ClassId, &'a NodeId>,
    reachable: &mut FxHashSet<&'a NodeId>,
    stack: &mut Vec<&'a NodeId>,
) {
    match egraph.enodes[ilist_rep].0.as_str() {
        "INil" => {}
        "ICons" => {
            let input_eclass = &egraph.enodes[ilist_rep].1[0];
            let chosen = choice
                .iter()
                .find(|(k, _)| ***k == *input_eclass)
                .map(|(_, v)| *v)
                .unwrap();
            if !reachable.contains(chosen) {
                reachable.insert(chosen);
                stack.push(chosen);
            }
            let rest_eclass = &egraph.enodes[ilist_rep].1[1];
            let rest_rep = &egraph.eclasses[rest_eclass].1[0];
            add_ilist_inputs_to_reachable(egraph, rest_rep, choice, reachable, stack);
        }
        "IConcat" => {
            let a_eclass = &egraph.enodes[ilist_rep].1[0];
            let b_eclass = &egraph.enodes[ilist_rep].1[1];
            let a_rep = &egraph.eclasses[a_eclass].1[0];
            let b_rep = &egraph.eclasses[b_eclass].1[0];
            add_ilist_inputs_to_reachable(egraph, a_rep, choice, reachable, stack);
            add_ilist_inputs_to_reachable(egraph, b_rep, choice, reachable, stack);
        }
        _ => {}
    }
}

fn collect_ilist_sources<'a>(
    egraph: &'a SerializedEGraph,
    ilist_rep: &'a NodeId,
    choice: &'a FxHashMap<&'a ClassId, &'a NodeId>,
    sources: &mut Vec<&'a NodeId>,
) {
    match egraph.enodes[ilist_rep].0.as_str() {
        "INil" => {}
        "ICons" => {
            let input_eclass = &egraph.enodes[ilist_rep].1[0];
            let chosen = choice
                .iter()
                .find(|(k, _)| ***k == *input_eclass)
                .map(|(_, v)| *v)
                .unwrap();
            sources.push(chosen);
            let rest_eclass = &egraph.enodes[ilist_rep].1[1];
            let rest_rep = &egraph.eclasses[rest_eclass].1[0];
            collect_ilist_sources(egraph, rest_rep, choice, sources);
        }
        "IConcat" => {
            let a_eclass = &egraph.enodes[ilist_rep].1[0];
            let b_eclass = &egraph.enodes[ilist_rep].1[1];
            let a_rep = &egraph.eclasses[a_eclass].1[0];
            let b_rep = &egraph.eclasses[b_eclass].1[0];
            collect_ilist_sources(egraph, a_rep, choice, sources);
            collect_ilist_sources(egraph, b_rep, choice, sources);
        }
        _ => {}
    }
}

#[tracing::instrument(skip_all)]
pub fn egglog_to_llir(
    egraph: &SerializedEGraph,
    ops: &Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    limit: usize,
) -> Vec<LLIRGraph> {
    // Get maps for all e-classes to e-node options
    let mut choices = vec![FxHashMap::default()];
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !label.contains("IR") && !label.contains("IList") {
            continue;
        }
        choices = enodes
            .iter()
            .flat_map(|enode| {
                choices.iter().cloned().map(move |mut choice_map| {
                    choice_map.insert(eclass, enode);
                    choice_map
                })
            })
            .rev()
            .take(limit)
            .collect();
    }

    // Create IR graphs
    let mut graphs = vec![];
    let mut c = FxHashMap::default();
    let mut lc = FxHashMap::default();
    for choice in choices {
        // Make reachability set from root
        let mut reachable = FxHashSet::default();
        reachable.insert(choice[&egraph.roots[0]]);
        let mut reachability_stack = vec![choice[&egraph.roots[0]]];
        while let Some(r) = reachability_stack.pop() {
            if egraph.eclasses[&egraph.node_to_class[r]]
                .0
                .contains("IList")
            {
                add_ilist_inputs_to_reachable(
                    egraph,
                    r,
                    &choice,
                    &mut reachable,
                    &mut reachability_stack,
                );
            }
            for ch in &egraph.enodes[r].1 {
                if egraph.eclasses[ch].0.contains("IR") || egraph.eclasses[ch].0.contains("IList") {
                    let n = choice[ch];
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
        for &node in choice.values() {
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
                    if egraph.eclasses[c].0.contains("IR") || egraph.eclasses[c].0.contains("IList")
                    {
                        choice[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();
            if egraph.enodes[node].0.as_str() == "CustomOpHLIR" {
                // Extract custom op inputs and id
                let mut inputs = vec![];
                // Walk through the IList to get inputs - use choice[] for IR eclasses to get the chosen enode
                let ilist_eclass = &egraph.enodes[node].1[0];
                let mut ch = &egraph.eclasses[ilist_eclass].1[0];
                loop {
                    if egraph.enodes[ch].0 == "INil" {
                        break;
                    } else {
                        // The first child of ICons is an IR node - use choice[] to get the chosen enode
                        let input_eclass = &egraph.enodes[ch].1[0];
                        inputs.push(choice[input_eclass]);
                        // The second child of ICons is the rest of the IList
                        ch = &egraph.eclasses[&egraph.enodes[ch].1[1]].1[0];
                    }
                }
                let id: usize = egraph.enodes[&egraph.eclasses[&egraph.enodes[node].1[1]].1[0]]
                    .0
                    .parse()
                    .unwrap();
                let r = graph.add_node(custom_ops[id].to_llir_op());
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
                let (op_instance, mut sources) = op.extract(egraph, &ch, &mut lc, &mut c);
                // Generic: walk any IList parameters to find IR input sources
                for (idx, param) in op.term().1.iter().enumerate() {
                    if matches!(param, OpParam::IList) {
                        let ilist_eclass = &egraph.enodes[node].1[idx];
                        let ilist_ch = &egraph.eclasses[ilist_eclass].1[0];
                        collect_ilist_sources(egraph, ilist_ch, &choice, &mut sources);
                    }
                }
                let r = graph.add_node(op_instance);
                enode_to_node.insert(node, r);
                for source in sources {
                    edges_to_place.push((source, node));
                }
            }
        }
        for (src, dest) in edges_to_place {
            graph.add_edge(enode_to_node[src], enode_to_node[dest], ());
        }

        graphs.push(graph);
    }
    graphs
}
