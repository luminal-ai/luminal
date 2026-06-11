use colored::Colorize;
use egglog::{ast::Span, prelude::RustSpan, var};
use itertools::Itertools;
use petgraph::{Direction, graph::NodeIndex};
use rand::Rng;
use rustc_hash::FxHashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::{str, sync::Arc, time::Duration};
use tracing::trace;

pub use egraph_serialize::{ClassId, NodeId};

pub mod api;
pub mod base;

const MAIN_SCHEDULE_MAX_CYCLES: usize = 256;
const MAIN_SCHEDULE_MAX_TUPLES: usize = 10_000_000;
const SLOW_PHASE_TIME: Duration = Duration::from_secs(1);
const BIG_TUPLE_DELTA: isize = 5_000;

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

#[derive(Debug, Clone)]
struct EgglogSchedulePhase {
    name: String,
    schedule: String,
}

pub type EGraphPostprocess = Arc<dyn Fn(&mut SerializedEGraph) + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub struct LateEgglogPass {
    /// Egglog declarations and rules for a backend-provided late pass.
    ///
    /// These fragments are appended after the core/backend rewrite declarations
    /// and before the graph program itself, so they can refer to the full IR and
    /// OpKind datatypes.
    pub program: String,
    /// Schedule to run after the normal full-egraph rewrite + cleanup schedule.
    ///
    /// Backends can use this for analysis-only layers or for analysis followed
    /// by backend-specific cleanup rules.
    pub schedule: String,
    /// Optional Rust post-processing hook that runs on the serialized e-graph
    /// after egglog has finished and before the final empty-eclass cascade.
    pub postprocess: Option<EGraphPostprocess>,
}

impl std::fmt::Debug for LateEgglogPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LateEgglogPass")
            .field("program", &self.program)
            .field("schedule", &self.schedule)
            .field("has_postprocess", &self.postprocess.is_some())
            .finish()
    }
}

impl LateEgglogPass {
    pub fn new(program: impl Into<String>, schedule: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            schedule: schedule.into(),
            postprocess: None,
        }
    }

    pub fn with_postprocess(
        mut self,
        postprocess: impl Fn(&mut SerializedEGraph) + Send + Sync + 'static,
    ) -> Self {
        self.postprocess = Some(Arc::new(postprocess));
        self
    }
}

fn op_defs_string(ops: &[Arc<Box<dyn EgglogOp>>]) -> String {
    // Partition ops by sort class: IR-class (Input, Output) vs OpKind-class (everything else)
    let mut ir_variants = Vec::new();
    let mut opkind_variants = Vec::new();
    for o in ops {
        let s = o.sort();
        let variant_str = format!(
            "({} {})",
            s.name,
            s.fields.iter().map(|f| &f.sort).join(" ")
        );
        if s.class == "IR" {
            ir_variants.push(variant_str);
        } else if s.class == "OpKind" {
            opkind_variants.push(variant_str);
        } else {
            panic!("Unknown sort class '{}' for op '{}'", s.class, s.name);
        }
    }
    let ir_str = ir_variants.join("\n");
    let opkind_str = opkind_variants.join("\n");
    let extra_ir: FxHashSet<String> = ops.iter().flat_map(|o| o.ir_defs()).collect();
    let extra_ir_str = extra_ir.into_iter().join("\n");
    format!(
        "
    (datatype*
        (IR
            (OutputJoin IR IR)
            (Op OpKind IList)
            {extra_ir_str}
            {ir_str}
        )
        (OpKind
            {opkind_str}
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

pub fn full_egglog(program: &str, ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> String {
    let parts = OpTextParts::new(ops, cleanup);
    full_egglog_with(program, &parts)
}

/// Pre-computed per-op text fragments. Materialising all op-derived strings
/// once up front means callers that want to drive multiple egglog runs in
/// parallel only need to share `&str` references and never touch the non-Send
/// trait objects in `ops`.
pub struct OpTextParts {
    op_defs: String,
    cleanups: String,
    /// Names of op kinds that are eligible for cleanup (cleanup() == true).
    /// Used by the Rust post-processing pass to safely strip HLIR ops only
    /// when an alternative survives in the same eclass.
    pub(crate) cleanable_op_names: FxHashSet<String>,
    late_program: String,
    rewrites: String,
    late_phases: Vec<EgglogSchedulePhase>,
    late_postprocesses: Vec<EGraphPostprocess>,
}

impl OpTextParts {
    pub fn new(ops: &[Arc<Box<dyn EgglogOp>>], cleanup: bool) -> Self {
        Self::new_with_late_passes(ops, cleanup, &[])
    }

    pub fn new_with_late_passes(
        ops: &[Arc<Box<dyn EgglogOp>>],
        cleanup: bool,
        late_passes: &[LateEgglogPass],
    ) -> Self {
        let cleanable_op_names: FxHashSet<String> = ops
            .iter()
            .filter(|op| op.cleanup())
            .map(|op| op.sort().name.to_string())
            .collect();
        Self {
            op_defs: op_defs_string(ops),
            // The egglog `cleanup` ruleset deletes HLIR ops unconditionally,
            // even when no kernel rewrite fired in their eclass. On large
            // graphs (e.g. YOLO v11) that produces empty eclasses and the
            // post-processing cascade panics with "No valid graphs present".
            // We always emit an empty cleanup ruleset and instead do
            // conditional cleanup in Rust after egglog finishes.
            cleanups: String::new(),
            rewrites: ops
                .iter()
                .flat_map(|o| o.rewrites())
                .map(|r| r.to_egglog_string())
                .join("\n"),
            cleanable_op_names: if cleanup {
                cleanable_op_names
            } else {
                FxHashSet::default()
            },
            late_program: late_passes.iter().map(|p| p.program.as_str()).join("\n"),
            late_phases: late_passes
                .iter()
                .enumerate()
                .filter_map(|(i, pass)| {
                    let schedule = normalize_late_schedule(&pass.schedule);
                    (!schedule.is_empty()).then(|| EgglogSchedulePhase {
                        name: format!("late pass {:02}", i + 1),
                        schedule,
                    })
                })
                .collect(),
            late_postprocesses: late_passes
                .iter()
                .filter_map(|pass| pass.postprocess.clone())
                .collect(),
        }
    }
}

fn full_egglog_with(program: &str, parts: &OpTextParts) -> String {
    let mut chunks = vec![egglog_setup_with(program, parts), egglog_schedule_program()];
    chunks.extend(
        parts
            .late_phases
            .iter()
            .map(|phase| format!("(run-schedule {})", phase.schedule)),
    );
    chunks.join("\n")
}

fn normalize_late_schedule(schedule: &str) -> String {
    let schedule = schedule.trim();
    schedule
        .strip_prefix("(run-schedule ")
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or(schedule)
        .trim()
        .to_string()
}

fn egglog_ruleset_declarations() -> String {
    EGGLOG_RULESETS
        .iter()
        .map(|ruleset| format!("(ruleset {ruleset})"))
        .join("\n")
}

fn expr_schedule(use_interval_analysis: bool) -> &'static str {
    if use_interval_analysis {
        "(saturate (seq expr interval_expr))"
    } else {
        "(saturate expr)"
    }
}

fn egglog_main_cycle_phases(cycle: usize, use_interval_analysis: bool) -> Vec<EgglogSchedulePhase> {
    vec![EgglogSchedulePhase {
        name: format!("cycle {cycle:03} main"),
        schedule: egglog_main_schedule(use_interval_analysis),
    }]
}

fn egglog_final_phases(use_interval_analysis: bool) -> Vec<EgglogSchedulePhase> {
    vec![
        EgglogSchedulePhase {
            name: "final expr".to_string(),
            schedule: expr_schedule(use_interval_analysis).to_string(),
        },
        EgglogSchedulePhase {
            name: "cleanup".to_string(),
            schedule: "(saturate cleanup)".to_string(),
        },
        EgglogSchedulePhase {
            name: "post cleanup".to_string(),
            schedule: "(saturate post_cleanup)".to_string(),
        },
        EgglogSchedulePhase {
            name: "base cleanup".to_string(),
            schedule: "(saturate base_cleanup)".to_string(),
        },
    ]
}

fn egglog_main_schedule(use_interval_analysis: bool) -> String {
    let expr = expr_schedule(use_interval_analysis);
    // Producer rules create raw alternatives that downstream fusion consumes.
    // Fusion grow/merge only consumes Kernel*/FusionEnd alternatives, so keeping
    // producer discovery saturated before fusion reaches the same fixed point
    // while avoiding repeated expensive pair-discovery scans during growth.
    format!(
        "(saturate (seq
        (saturate (seq
            {expr}
            (saturate dtype_prop)
            (run matmul_flatten)
            (run kernel_lower)
            (run direct_kernel)
            (run kernel_specialize)
            (run buffer_reuse)
            (run matmul_backend)
            (run glumoe)
            (run fusion_pair)
        ))
        (saturate (seq
            {expr}
            (saturate dtype_prop)
            (run fusion_grow)
            (run fusion_merge)
        ))
    ))"
    )
}

fn egglog_schedule_program() -> String {
    let mut schedules = vec![format!("(run-schedule {})", egglog_main_schedule(false))];
    schedules.extend(
        egglog_final_phases(false)
            .into_iter()
            .map(|phase| format!("(run-schedule {})", phase.schedule)),
    );
    schedules.join("\n")
}

fn egglog_setup_with(program: &str, parts: &OpTextParts) -> String {
    egglog_setup_with_options(program, parts, false)
}

fn egglog_setup_with_options(
    program: &str,
    parts: &OpTextParts,
    use_interval_analysis: bool,
) -> String {
    let base_program = if use_interval_analysis {
        base::base_expression_egglog_with_intervals()
    } else {
        base::base_expression_egglog()
    };
    [
        egglog_ruleset_declarations(),
        base_program,
        parts.op_defs.clone(),
        parts.cleanups.clone(),
        base::base_cleanup_egglog(),
        parts.rewrites.clone(),
        parts.late_program.clone(),
        program.to_string(),
    ]
    .join("\n")
}

use crate::{
    dtype::DType,
    graph::{Graph, LLIRGraph},
    op::{CustomOp, EgglogOp},
    prelude::FxHashMap,
    shape::Expression,
};
use egglog::{ArcSort, CommandOutput, EGraph, Value};
use egglog_reports::ReportLevel;

#[derive(Debug, Clone)]
///  This is snapshot of an EGraph with Rust native hash maps and sets for enabling more native traversal / algorithm writing.
///  The name comes from the serialize egraph crates, which returns a ETermDAG, which caused issues, so this is a homebrew semi-static egraph
pub struct SerializedEGraph {
    pub enodes: FxHashMap<NodeId, (String, Vec<ClassId>)>,
    pub eclasses: FxHashMap<ClassId, (String, Vec<NodeId>)>,
    pub node_to_class: FxHashMap<NodeId, ClassId>,
    pub roots: Vec<ClassId>,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogStageReport {
    pub num_matches_per_rule: FxHashMap<String, usize>,
    pub search_and_apply_time_per_rule: FxHashMap<String, Duration>,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogRunReport {
    pub full: EgglogStageReport,
    pub phases: Vec<EgglogPhaseReport>,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct EgglogPhaseReport {
    pub name: String,
    pub schedule: String,
    pub updated: bool,
    pub iterations: usize,
    pub tuples_before: usize,
    pub tuples_after: usize,
    pub total_time: Duration,
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
        for (node_id, node) in s.egraph.nodes.iter().filter(|(_, node)| !node.subsumed) {
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
                .filter(|(_, enode)| !enode.subsumed)
                .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
                .collect(),
            enodes: s
                .egraph
                .nodes
                .iter()
                .filter(|(_, enode)| !enode.subsumed)
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
                .filter_map(|(c, eclass)| {
                    classes
                        .get(c)
                        .map(|nodes| (c.clone(), (eclass.typ.clone().unwrap(), nodes.clone())))
                })
                .collect(),
        };
        // Strip out all [...] enodes
        s_egraph.enodes.retain(|_, (label, _)| label != "[...]");
        loop {
            let mut to_remove = vec![];
            for (id, (_, children)) in &s_egraph.enodes {
                if children.iter().any(|c| {
                    s_egraph.eclasses.get(c).is_none_or(|(_, nodes)| {
                        !nodes.iter().any(|n| s_egraph.enodes.contains_key(n))
                    })
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
/// - CustomOpKind integer IDs (global custom_ops index, differs per layer)
///
/// This function hashes the text while normalizing those chunk-specific values:
/// - Input lines: only the dtype is hashed (not node index or label)
/// - Output lines: only the "OUTPUT" marker is hashed (not the node index)
/// - CustomOpKind lines: the integer ID is replaced with a constant
/// - All other lines (ops, shapes, strides): hashed verbatim
pub fn hash_egglog_normalized(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    for line in text.lines() {
        if line.contains("(Input ") {
            // Format: (let tN (Input NODE "LABEL" (DTYPE)))
            // Strip the node index and label identity, but preserve whether this
            // is a synthetic boundary input or a real graph input.
            // The dtype is the last parenthesized token, e.g. "(F32)".
            if let Some(dtype_start) = line.rfind(" (") {
                let dtype = &line[dtype_start + 1..];
                let kind = if line.contains("\"boundary\"") {
                    "BOUNDARY_INPUT"
                } else {
                    "REAL_INPUT"
                };
                (kind, dtype).hash(&mut hasher);
            } else {
                line.hash(&mut hasher);
            }
        } else if line.contains("(Output ") && !line.contains("(OutputJoin ") {
            "OUTPUT".hash(&mut hasher);
        } else if line.contains("(CustomOpKind ") {
            // Format: (let tN (Op (CustomOpKind ID (DTYPE)) (ICons ...)))
            // The integer ID varies per layer. Replace it with a constant.
            normalize_custom_op_id(line).hash(&mut hasher);
        } else {
            line.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Replace the integer ID in a CustomOpKind egglog line with a constant "0".
fn normalize_custom_op_id(line: &str) -> String {
    if let Some(custom_start) = line.find("(CustomOpKind ") {
        let after = &line[custom_start + "(CustomOpKind ".len()..];
        // The ID is the first token (integer) after "CustomOpKind "
        if let Some(space_after_id) = after.find(' ') {
            let id_str = &after[..space_after_id];
            if id_str.chars().all(|c| c.is_ascii_digit()) {
                return format!(
                    "{}0{}",
                    &line[..custom_start + "(CustomOpKind ".len()],
                    &line[custom_start + "(CustomOpKind ".len() + space_after_id..]
                );
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
    // Pre-size output to avoid growth reallocations; ops emit ~100-200 chars each.
    let mut out = String::with_capacity(topo_order.len() * 160);

    use std::fmt::Write;
    let mut curr_id = 0;
    for n in topo_order {
        let sources: Vec<(NodeIndex, String)> = graph
            .get_sources(n)
            .into_iter()
            .map(|src| (src, names[&src].clone()))
            .collect_vec();
        let code = graph[n].to_egglog(&sources);
        // write!() into the existing buffer skips the intermediate String
        // that format! would otherwise allocate for each node.
        let _ = writeln!(out, "(let t{curr_id} {code})");
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
        let _ = writeln!(out, "(let t{curr_id} (OutputJoin {root} {node}))");
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

fn stage_report(egraph: &egglog::EGraph, total_time: Duration) -> EgglogStageReport {
    let run_report = egraph.get_overall_run_report();
    EgglogStageReport {
        num_matches_per_rule: run_report
            .num_matches_per_rule
            .iter()
            .map(|(name, matches)| (name.to_string(), *matches))
            .collect(),
        search_and_apply_time_per_rule: run_report
            .search_and_apply_time_per_rule
            .iter()
            .map(|(name, elapsed)| (name.to_string(), *elapsed))
            .collect(),
        total_time,
    }
}

fn trace_stage_report(header: &str, report: &EgglogStageReport) {
    trace!("{}", header.green());
    trace!(
        "{}",
        report
            .num_matches_per_rule
            .iter()
            .filter(|(k, _)| !k.contains("("))
            .map(|(k, v)| format!(
                "{k}: {v} ({})",
                pretty_duration::pretty_duration(&report.search_and_apply_time_per_rule[k], None)
            ))
            .join("\n")
            .green()
    );
    trace!(
        "{}",
        format!(
            "---- {} Took {} ----",
            header,
            pretty_duration::pretty_duration(&report.total_time, None).bold()
        )
        .green()
    );
}

fn metric_duration(duration: Duration) -> String {
    pretty_duration::pretty_duration(&duration, None)
}

/// Verbose per-rule / per-ruleset printing. Off by default — the cycle
/// header, summary header, and final timing line stay; the breakdown
/// tables and `print_serialized_shape` ("Egglog extract root shape …")
/// are suppressed unless `EGGLOG_DEBUG=1`.
fn egglog_debug() -> bool {
    std::env::var("EGGLOG_DEBUG").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn metric_name(name: &str) -> String {
    let mut name = name.split_whitespace().join(" ");
    if name.len() > 96 {
        name.truncate(93);
        name.push_str("...");
    }
    name
}

fn sorted_rule_metrics(report: &egglog_reports::RunReport) -> Vec<(String, Duration, usize)> {
    let mut rules = report
        .search_and_apply_time_per_rule
        .iter()
        .map(|(rule, elapsed)| {
            (
                rule.to_string(),
                *elapsed,
                report
                    .num_matches_per_rule
                    .get(rule)
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect_vec();
    rules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    rules
}

fn print_rule_plan_hotspots(
    report: &egglog_reports::RunReport,
    rules: &[(String, Duration, usize)],
) {
    for (rule, _, _) in rules.iter().take(3) {
        let mut max_stage = None;
        let mut max_shape = None;
        for iteration in &report.iterations {
            if let Some(rule_reports) = iteration.rule_reports().get(rule.as_str()) {
                for rule_report in rule_reports {
                    if let Some(plan) = &rule_report.plan {
                        let scans = plan
                            .stages
                            .iter()
                            .map(|(stage, _, _)| match stage {
                                egglog_reports::Stage::Intersect { scans } => scans.len(),
                                egglog_reports::Stage::FusedIntersect { to_intersect, .. } => {
                                    to_intersect.len() + 1
                                }
                            })
                            .sum::<usize>();
                        max_shape = Some(
                            max_shape
                                .map(|(stages, shape_scans)| {
                                    if plan.stages.len() > stages {
                                        (plan.stages.len(), scans)
                                    } else {
                                        (stages, shape_scans)
                                    }
                                })
                                .unwrap_or((plan.stages.len(), scans)),
                        );
                        for (_, stats, _) in &plan.stages {
                            if let Some(stats) = stats {
                                if max_stage
                                    .map(|(candidates, _)| stats.num_candidates > candidates)
                                    .unwrap_or(true)
                                {
                                    max_stage = Some((stats.num_candidates, stats.num_succeeded));
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some((stages, scans)) = max_shape {
            eprintln!(
                "      plan    {:<96} stages {:>2} scans {:>2} | max candidates {}",
                metric_name(rule),
                stages,
                scans,
                max_stage
                    .map(|(candidates, succeeded)| format!("{candidates} -> {succeeded}"))
                    .unwrap_or_else(|| "n/a".to_string())
            );
        }
    }
}

fn print_slow_phase_detail(
    phase: &EgglogSchedulePhase,
    report: &egglog_reports::RunReport,
    tuple_delta: isize,
    elapsed: Duration,
    rules: &[(String, Duration, usize)],
) {
    eprintln!("      detail  schedule {}", metric_name(&phase.schedule));
    if tuple_delta > 0 && elapsed > Duration::ZERO {
        eprintln!(
            "      detail  growth {:.0} tuples/s | {:.3} ms/new tuple",
            tuple_delta as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000.0 / tuple_delta as f64
        );
    }
    for (rule, elapsed, _) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches == 0)
        .take(5)
    {
        eprintln!(
            "      zero    {:<96} {:>10}",
            metric_name(rule),
            metric_duration(*elapsed)
        );
    }
    let mut per_match = rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches > 0)
        .map(|(rule, elapsed, matches)| {
            (
                rule,
                elapsed.as_secs_f64() * 1_000.0 / *matches as f64,
                *matches,
            )
        })
        .collect_vec();
    per_match.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rule, ms_per_match, matches) in per_match.into_iter().take(3) {
        eprintln!(
            "      cost    {:<96} {:.3} ms/match | matches {}",
            metric_name(rule),
            ms_per_match,
            matches
        );
    }
    print_rule_plan_hotspots(report, rules);

    let iteration_count = report.iterations.len();
    for (index, iteration) in report.iterations.iter().enumerate() {
        if iteration_count > 12 && (8..iteration_count.saturating_sub(3)).contains(&index) {
            if index == 8 {
                eprintln!("      iter   ...");
            }
            continue;
        }
        let (rule, elapsed, matches) = iteration
            .rule_reports()
            .iter()
            .map(|(rule, reports)| {
                (
                    rule.to_string(),
                    reports
                        .iter()
                        .map(|report| report.search_and_apply_time)
                        .sum::<Duration>(),
                    reports
                        .iter()
                        .map(|report| report.num_matches)
                        .sum::<usize>(),
                )
            })
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)))
            .unwrap_or_else(|| ("-".to_string(), Duration::ZERO, 0));
        eprintln!(
            "      iter   {:>2} changed={} | search {:>10} | merge {:>10} | rebuild {:>10} | top {:<64} {:>10} matches {}",
            index + 1,
            iteration.changed(),
            metric_duration(iteration.search_and_apply_time()),
            metric_duration(iteration.rule_set_report.merge_time),
            metric_duration(iteration.rebuild_time),
            metric_name(&rule),
            metric_duration(elapsed),
            matches
        );
    }
}

fn print_run_summary(run_report: &EgglogRunReport) {
    eprintln!(
        "{}",
        format!(
            "   Egglog summary total {} | phases {}",
            metric_duration(run_report.total_time),
            run_report.phases.len()
        )
        .cyan()
    );
    if !egglog_debug() {
        return;
    }
    let mut phases = run_report.phases.iter().collect_vec();
    phases.sort_by_key(|phase| std::cmp::Reverse(phase.total_time));
    for phase in phases.into_iter().take(5) {
        eprintln!(
            "      phase   {:<28} {:>10} | tuples {:+} | iterations {}",
            metric_name(&phase.name),
            metric_duration(phase.total_time),
            phase.tuples_after as isize - phase.tuples_before as isize,
            phase.iterations
        );
    }
    let mut growth = run_report
        .phases
        .iter()
        .map(|phase| {
            (
                phase,
                phase.tuples_after as isize - phase.tuples_before as isize,
            )
        })
        .filter(|(_, delta)| *delta > 0)
        .collect_vec();
    growth.sort_by_key(|(_, delta)| std::cmp::Reverse(*delta));
    for (phase, delta) in growth.into_iter().take(3) {
        eprintln!(
            "      growth  {:<28} tuples {:+} | {}",
            metric_name(&phase.name),
            delta,
            metric_duration(phase.total_time)
        );
    }
    let mut rules = run_report
        .full
        .search_and_apply_time_per_rule
        .iter()
        .map(|(rule, elapsed)| {
            (
                rule,
                *elapsed,
                run_report
                    .full
                    .num_matches_per_rule
                    .get(rule)
                    .copied()
                    .unwrap_or_default(),
            )
        })
        .collect_vec();
    rules.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
    for (rule, elapsed, matches) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO || *matches > 0)
        .take(8)
    {
        eprintln!(
            "      slow    {:<96} {:>10} | matches {}",
            metric_name(rule),
            metric_duration(*elapsed),
            matches
        );
    }
    for (rule, elapsed, _) in rules
        .iter()
        .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO && *matches == 0)
        .take(8)
    {
        eprintln!(
            "      zero    {:<96} {:>10}",
            metric_name(rule),
            metric_duration(*elapsed)
        );
    }
}

fn print_serialized_shape(s: &egglog::SerializeOutput) {
    if !egglog_debug() {
        return;
    }
    let mut classes = FxHashSet::default();
    let mut labels: FxHashMap<String, usize> = FxHashMap::default();
    let mut nodes = 0;
    for node in s.egraph.nodes.values().filter(|node| !node.subsumed) {
        nodes += 1;
        classes.insert(node.eclass.clone());
        *labels.entry(node.op.clone()).or_default() += 1;
    }
    let mut labels = labels.into_iter().collect_vec();
    labels.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    eprintln!(
        "{}",
        format!(
            "   Egglog extract root shape nodes={} classes={} roots={} top_ops={}",
            nodes,
            classes.len(),
            s.egraph.root_eclasses.len(),
            labels
                .into_iter()
                .take(8)
                .map(|(label, count)| format!("{}={}", metric_name(&label), count))
                .join(", ")
        )
        .cyan()
    );
}

fn run_schedule_phase(
    egraph: &mut egglog::EGraph,
    phases: &mut Vec<EgglogPhaseReport>,
    phase: &EgglogSchedulePhase,
) -> Result<bool, egglog::Error> {
    let command = format!("(run-schedule {})", phase.schedule);
    let tuples_before = egraph.num_tuples();
    let start = std::time::Instant::now();
    let outputs = egraph.parse_and_run_program(None, &command)?;
    let elapsed = start.elapsed();
    let tuples_after = egraph.num_tuples();

    let report = outputs
        .into_iter()
        .find_map(|output| match output {
            CommandOutput::RunSchedule(report) => Some(report),
            _ => None,
        })
        .expect("run-schedule did not return a report");

    let updated = report.updated;
    let iterations = report.iterations.len();
    let tuple_delta = tuples_after as isize - tuples_before as isize;
    eprintln!(
        "{}",
        format!(
            "   Egglog {:<28} {:>10} | tuples {} -> {} ({:+}) | updated={} | iterations={}",
            phase.name,
            metric_duration(elapsed),
            tuples_before,
            tuples_after,
            tuple_delta,
            updated,
            iterations,
        )
        .cyan()
    );

    if egglog_debug() {
        let mut rulesets = report
            .search_and_apply_time_per_ruleset
            .keys()
            .chain(report.merge_time_per_ruleset.keys())
            .chain(report.rebuild_time_per_ruleset.keys())
            .map(|ruleset| ruleset.to_string())
            .unique()
            .collect_vec();
        let ruleset_total = |ruleset: &str| {
            report
                .search_and_apply_time_per_ruleset
                .get(ruleset)
                .copied()
                .unwrap_or(Duration::ZERO)
                + report
                    .merge_time_per_ruleset
                    .get(ruleset)
                    .copied()
                    .unwrap_or(Duration::ZERO)
                + report
                    .rebuild_time_per_ruleset
                    .get(ruleset)
                    .copied()
                    .unwrap_or(Duration::ZERO)
        };
        rulesets.sort_by_key(|ruleset| std::cmp::Reverse(ruleset_total(ruleset)));
        for ruleset in rulesets.into_iter().take(4) {
            let search = report
                .search_and_apply_time_per_ruleset
                .get(ruleset.as_str())
                .copied()
                .unwrap_or(Duration::ZERO);
            let merge = report
                .merge_time_per_ruleset
                .get(ruleset.as_str())
                .copied()
                .unwrap_or(Duration::ZERO);
            let rebuild = report
                .rebuild_time_per_ruleset
                .get(ruleset.as_str())
                .copied()
                .unwrap_or(Duration::ZERO);
            eprintln!(
                "      ruleset {:<18} search {:>10} | merge {:>10} | rebuild {:>10}",
                metric_name(&ruleset),
                metric_duration(search),
                metric_duration(merge),
                metric_duration(rebuild)
            );
        }

        let rules = sorted_rule_metrics(&report);
        for (rule, elapsed, matches) in rules
            .iter()
            .filter(|(_, elapsed, matches)| *elapsed > Duration::ZERO || *matches > 0)
            .take(5)
        {
            eprintln!(
                "      rule    {:<96} {:>10} | matches {}",
                metric_name(rule),
                metric_duration(*elapsed),
                matches
            );
        }
        if elapsed >= SLOW_PHASE_TIME || tuple_delta.abs() >= BIG_TUPLE_DELTA {
            print_slow_phase_detail(phase, &report, tuple_delta, elapsed, &rules);
        }
    }

    phases.push(EgglogPhaseReport {
        name: phase.name.clone(),
        schedule: phase.schedule.clone(),
        updated,
        iterations,
        tuples_before,
        tuples_after,
        total_time: elapsed,
    });

    Ok(updated)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_report(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let op_parts = OpTextParts::new(ops, cleanup);
    run_egglog_with_report_parts(program, root, &op_parts)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_report_and_late_passes(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
    late_passes: &[LateEgglogPass],
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let op_parts = OpTextParts::new_with_late_passes(ops, cleanup, late_passes);
    run_egglog_with_report_parts(program, root, &op_parts)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_report_late_passes_and_interval_analysis(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
    late_passes: &[LateEgglogPass],
    use_interval_analysis: bool,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    let op_parts = OpTextParts::new_with_late_passes(ops, cleanup, late_passes);
    run_egglog_with_report_parts_impl(program, root, &op_parts, use_interval_analysis)
}

/// Same as [`run_egglog_with_report`], but takes pre-computed [`OpTextParts`].
/// Useful when a caller runs many egglog invocations with the same op set
/// and wants to factor the op-derived text work out of a parallel loop.
/// Takes only `&str` / `&OpTextParts` inputs so the whole function is `Send`.
#[tracing::instrument(skip_all)]
pub fn run_egglog_with_report_parts(
    program: &str,
    root: &str,
    op_parts: &OpTextParts,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    run_egglog_with_report_parts_impl(program, root, op_parts, false)
}

fn run_egglog_with_report_parts_impl(
    program: &str,
    root: &str,
    op_parts: &OpTextParts,
    use_interval_analysis: bool,
) -> Result<(SerializedEGraph, EgglogRunReport), egglog::Error> {
    #[cfg(debug_assertions)]
    {
        use std::sync::atomic::{AtomicBool, Ordering};

        static WARNED_DEBUG_EGGLOG: AtomicBool = AtomicBool::new(false);
        if !WARNED_DEBUG_EGGLOG.swap(true, Ordering::Relaxed) {
            eprintln!(
                "{}",
                "   Egglog warning: running in a debug build; model-sized saturation can be very slow. Use `cargo run --release ...` for CUDA model examples."
                    .yellow()
            );
        }
    }

    let total_start = std::time::Instant::now();

    let full_start = std::time::Instant::now();
    let setup_text_start = std::time::Instant::now();
    let setup_code = egglog_setup_with_options(program, op_parts, use_interval_analysis);
    let setup_text_elapsed = setup_text_start.elapsed();
    let setup_lines = setup_code.lines().count();
    let mut egraph = egglog::EGraph::default();
    egraph.set_report_level(ReportLevel::WithPlan);
    let setup_start = std::time::Instant::now();
    let setup_tuples_before = egraph.num_tuples();
    let parse_start = std::time::Instant::now();
    let commands = egraph.parser.get_program_from_string(None, &setup_code)?;
    let parse_elapsed = parse_start.elapsed();
    trace!("{}", "Egglog setup running...".green());
    let setup_run_start = std::time::Instant::now();
    let _outputs = egraph.run_program(commands)?;
    let setup_run_elapsed = setup_run_start.elapsed();
    let setup_tuples_after = egraph.num_tuples();

    eprintln!(
        "{}",
        format!(
            "   Egglog {:<28} {:>10} | text {} parse {} run {} | lines {} bytes {} | tuples {} -> {} ({:+})",
            "setup",
            metric_duration(setup_start.elapsed()),
            metric_duration(setup_text_elapsed),
            metric_duration(parse_elapsed),
            metric_duration(setup_run_elapsed),
            setup_lines,
            setup_code.len(),
            setup_tuples_before,
            setup_tuples_after,
            setup_tuples_after as isize - setup_tuples_before as isize,
        )
        .cyan()
    );

    trace!("{}", "Egglog running...".green());
    let mut phases = Vec::new();
    let mut reached_fixed_point = false;
    for cycle in 1..=MAIN_SCHEDULE_MAX_CYCLES {
        let mut cycle_updated = false;
        for phase in egglog_main_cycle_phases(cycle, use_interval_analysis) {
            cycle_updated |= run_schedule_phase(&mut egraph, &mut phases, &phase)?;
        }
        if egraph.num_tuples() > MAIN_SCHEDULE_MAX_TUPLES {
            return Err(egglog::Error::BackendError(format!(
                "egglog saturation exceeded tuple budget: {} > {}",
                egraph.num_tuples(),
                MAIN_SCHEDULE_MAX_TUPLES
            )));
        }
        if !cycle_updated {
            reached_fixed_point = true;
            break;
        }
    }
    if !reached_fixed_point {
        return Err(egglog::Error::BackendError(format!(
            "egglog saturation did not reach a fixed point within {MAIN_SCHEDULE_MAX_CYCLES} cycles"
        )));
    }
    for phase in egglog_final_phases(use_interval_analysis) {
        run_schedule_phase(&mut egraph, &mut phases, &phase)?;
    }
    for phase in &op_parts.late_phases {
        run_schedule_phase(&mut egraph, &mut phases, phase)?;
    }
    let full_report = stage_report(&egraph, full_start.elapsed());
    trace_stage_report("---- Egglog Rule Matches ----", &full_report);

    let run_report = EgglogRunReport {
        full: full_report,
        phases,
        total_time: total_start.elapsed(),
    };
    print_run_summary(&run_report);
    trace!(
        "{}",
        format!(
            "---- Egglog Total Took {} ----",
            pretty_duration::pretty_duration(&run_report.total_time, None).bold()
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
    print_serialized_shape(&s);
    // Convert to SerializedEGraph
    let mut classes = FxHashMap::default();
    for (node_id, node) in s.egraph.nodes.iter().filter(|(_, node)| !node.subsumed) {
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
            .filter(|(_, enode)| !enode.subsumed)
            .map(|(n, enode)| (n.clone(), enode.eclass.clone()))
            .collect(),
        enodes: s
            .egraph
            .nodes
            .iter()
            .filter(|(_, enode)| !enode.subsumed)
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
            .filter_map(|(c, eclass)| {
                classes
                    .get(c)
                    .map(|nodes| (c.clone(), (eclass.typ.clone().unwrap(), nodes.clone())))
            })
            .collect(),
    };
    // Strip out all [...] enodes
    egraph.enodes.retain(|_, (label, _)| label != "[...]");

    // Conditional cleanup: an `Op` enode in our IR has the shape
    // `Op (OpKind ...) (IList ...)`. The first child of `Op` is an OpKind
    // eclass; the OpKind enode's label tells us whether it's an HLIR op
    // (e.g. "MulKind") or a backend op (e.g. "FusionEndKind"). The egglog
    // `cleanup` ruleset can over-delete by removing HLIR variants that have
    // no kernel alternative (e.g. when dtype propagation didn't reach the
    // sub-expression). On conv-heavy graphs that drops Op eclasses to empty
    // and cascades all the way to the root with "No valid graphs present in
    // the e-graph!".
    //
    // We now perform cleanup here in Rust where it's safe: for every Op
    // eclass we look at the OpKind eclasses each Op points at, and strip
    // Op enodes whose kind is cleanable only if a non-cleanable kind exists
    // somewhere in the same Op eclass.
    let cleanable = &op_parts.cleanable_op_names;
    if !cleanable.is_empty() {
        // For each OpKind eclass, find the unique kind label (if any).
        // OpKind eclasses contain enodes like (MulKind ...) / (FusionEndKind ...).
        let mut opkind_class_kinds: FxHashMap<egraph_serialize::ClassId, FxHashSet<String>> =
            FxHashMap::default();
        for (nid, (label, _)) in &egraph.enodes {
            let cid = &egraph.node_to_class[nid];
            // The opkind labels come straight from the serializer's `op` field.
            opkind_class_kinds
                .entry(cid.clone())
                .or_default()
                .insert(label.clone());
        }

        let mut to_strip = Vec::new();
        // Walk Op enodes and decide which to drop.
        // For each Op eclass, group its Op enodes by (cleanable | survivor)
        // based on the OpKind in the first child eclass.
        let mut op_eclass_status: FxHashMap<
            egraph_serialize::ClassId,
            (Vec<egraph_serialize::NodeId>, bool),
        > = FxHashMap::default();
        for (nid, (label, children)) in &egraph.enodes {
            if label != "Op" {
                continue;
            }
            // Identify OpKind via first child eclass.
            let kind_class = match children.first() {
                Some(c) => c,
                None => continue,
            };
            let kinds = match opkind_class_kinds.get(kind_class) {
                Some(k) => k,
                None => continue,
            };
            let cleanable_kind = kinds.iter().all(|k| cleanable.contains(k));
            let op_class = &egraph.node_to_class[nid];
            let entry = op_eclass_status
                .entry(op_class.clone())
                .or_insert_with(|| (Vec::new(), false));
            if cleanable_kind {
                entry.0.push(nid.clone());
            } else {
                entry.1 = true;
            }
        }
        for (_op_class, (cleanable_nodes, has_survivor)) in op_eclass_status {
            if has_survivor {
                to_strip.extend(cleanable_nodes);
            }
        }
        for nid in to_strip {
            egraph.enodes.remove(&nid);
        }
    }

    for postprocess in &op_parts.late_postprocesses {
        postprocess(&mut egraph);
    }

    // Cascade: remove enodes whose children reference empty eclasses
    loop {
        let mut to_remove = vec![];
        for (id, (_, children)) in &egraph.enodes {
            if children.iter().any(|c| {
                egraph
                    .eclasses
                    .get(c)
                    .is_none_or(|(_, nodes)| !nodes.iter().any(|n| egraph.enodes.contains_key(n)))
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

    Ok((egraph, run_report))
}

#[tracing::instrument(skip_all)]
pub fn run_egglog(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    run_egglog_with_report(program, root, ops, cleanup).map(|(egraph, _)| egraph)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_late_passes(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
    late_passes: &[LateEgglogPass],
) -> Result<SerializedEGraph, egglog::Error> {
    run_egglog_with_report_and_late_passes(program, root, ops, cleanup, late_passes)
        .map(|(egraph, _)| egraph)
}

#[tracing::instrument(skip_all)]
pub fn run_egglog_with_late_passes_and_interval_analysis(
    program: &str,
    root: &str,
    ops: &[Arc<Box<dyn EgglogOp>>],
    cleanup: bool,
    late_passes: &[LateEgglogPass],
    use_interval_analysis: bool,
) -> Result<SerializedEGraph, egglog::Error> {
    run_egglog_with_report_late_passes_and_interval_analysis(
        program,
        root,
        ops,
        cleanup,
        late_passes,
        use_interval_analysis,
    )
    .map(|(egraph, _)| egraph)
}

/// Same as [`run_egglog`] but takes pre-computed [`OpTextParts`], so the
/// whole function is `Send`. Used by the parallel grouped-egraphs build.
#[tracing::instrument(skip_all)]
pub fn run_egglog_with(
    program: &str,
    root: &str,
    op_parts: &OpTextParts,
) -> Result<SerializedEGraph, egglog::Error> {
    run_egglog_with_report_parts(program, root, op_parts).map(|(egraph, _)| egraph)
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
        "F64" => DType::F64,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "Int" => DType::Int,
        // `"Int64"` rather than `"I64"` to avoid colliding with egglog's
        // built-in I64 primitive (see `DType::I64` docstring).
        "Int64" => DType::I64,
        "Bool" => DType::Bool,
        "F4E2M1" => DType::F4E2M1,
        "F6E2M3" => DType::F6E2M3,
        "F6E3M2" => DType::F6E3M2,
        "F8E4M3" => DType::F8E4M3,
        "F8E5M2" => DType::F8E5M2,
        "F8UE8M0" => DType::F8UE8M0,
        "I4" => DType::I4,
        "U4" => DType::U4,
        "I8" => DType::I8,
        "U8" => DType::U8,
        "I16" => DType::I16,
        "U16" => DType::U16,
        "TF32" => DType::TF32,
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
                .parse::<i64>()
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

fn is_search_choice_eclass(label: &str) -> bool {
    label.contains("IR") || label.contains("IList") || label.contains("OpKind")
}

fn extractor_list_len(egraph: &SerializedEGraph, eclass_id: &ClassId) -> Option<usize> {
    let mut len = 0usize;
    let mut cur_eclass: ClassId = eclass_id.clone();
    let mut visited: FxHashSet<ClassId> = FxHashSet::default();
    loop {
        if !visited.insert(cur_eclass.clone()) {
            return None;
        }
        let (label, enodes) = egraph.eclasses.get(&cur_eclass)?;
        if !label.contains("List") {
            return Some(len);
        }
        let head_enode = enodes.first()?;
        let head_label = &egraph.enodes[head_enode].0;
        if head_label == "ENil" || head_label == "INil" {
            return Some(len);
        }
        if head_label != "ECons" && head_label != "ICons" {
            return Some(len);
        }
        len += 1;
        let children = &egraph.enodes[head_enode].1;
        if children.len() < 2 {
            return Some(len);
        }
        cur_eclass = children[1].clone();
    }
}

fn opkind_metadata_consistent(egraph: &SerializedEGraph, node: &NodeId) -> bool {
    let lens: Vec<usize> = egraph.enodes[node]
        .1
        .iter()
        .filter_map(|c| {
            let lbl = &egraph.eclasses[c].0;
            if lbl.contains("List") {
                extractor_list_len(egraph, c)
            } else {
                None
            }
        })
        .collect();
    lens.is_empty() || lens.iter().all(|l| *l == lens[0])
}

/// Count the total number of possible searchable choice sets, capped at `limit`.
///
/// Search deduplicates candidates by `EGraphChoiceSet`, so this gives the exact
/// number of candidates when it is below `limit` without risking overflow on
/// large search spaces.
pub fn count_choice_sets_up_to(egraph: &SerializedEGraph, limit: usize) -> usize {
    if limit == 0 {
        return 0;
    }

    let mut count = 1usize;
    for (label, enodes) in egraph.eclasses.values() {
        if !is_search_choice_eclass(label) {
            continue;
        }

        count = count.saturating_mul(enodes.len());
        if count >= limit {
            return limit;
        }
    }
    count
}

pub fn random_initial_choice<'a>(
    egraph: &'a SerializedEGraph,
    rng: &mut impl Rng,
) -> EGraphChoiceSet<'a> {
    let mut choices = FxHashMap::default();
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !is_search_choice_eclass(label) {
            continue;
        }
        // Use synth-injected enodes when available — they point at
        // deterministic single-variant kind eclasses produced by the
        // deep-clone fallback in `inject_kernel_alternatives`, so the
        // extractor's first-enode walk is guaranteed length-consistent.
        // Without this bias, the chooser frequently lands on an egglog-
        // rewritten KernelOp whose ELIST children inherit a multi-variant
        // eclass (caused by length-changing rewrites such as merge_dims /
        // RemoveNthFromEnd), which produces a flatten_strides assertion
        // mismatch downstream.
        let synth_indices: Vec<usize> = enodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().starts_with("synth_").then_some(i))
            .collect();
        let consistent_opkind_indices: Vec<usize> = if label == "OpKind" {
            enodes
                .iter()
                .enumerate()
                .filter_map(|(i, n)| opkind_metadata_consistent(egraph, n).then_some(i))
                .collect()
        } else {
            Vec::new()
        };
        let pick_idx = if !consistent_opkind_indices.is_empty() {
            consistent_opkind_indices[rng.random_range(0..consistent_opkind_indices.len())]
        } else if !synth_indices.is_empty() {
            synth_indices[rng.random_range(0..synth_indices.len())]
        } else {
            rng.random_range(0..enodes.len())
        };
        choices.insert(eclass, &enodes[pick_idx]);
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
    // Check all searchable eclasses have a choice.
    for (eclass, (label, enodes)) in &egraph.eclasses {
        if !is_search_choice_eclass(label) {
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
            if is_search_choice_eclass(label) {
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
        let (op_name, children) = &egraph.enodes[*node];
        let eclass = &egraph.node_to_class[*node];
        let (label, _) = &egraph.eclasses[eclass];
        if label != "IR" {
            continue; // Skip IList / OpKind nodes
        }
        if op_name == "OutputJoin" {
            continue;
        }
        if op_name == "Op" {
            // Normalized op — check OpKind child
            if let Some(kind_eclass) = children.first() {
                if let Some(kn) = choices.get(kind_eclass) {
                    let kind_name = &egraph.enodes[kn].0;
                    if kind_name != "CustomOpKind"
                        && !ops.iter().any(|op| op.sort().name == *kind_name)
                    {
                        return Err(format!("No extractor for OpKind {kind_name}"));
                    }
                }
            }
            continue;
        }
        // Direct IR variant (Input, Output)
        if !ops.iter().any(|op| op.sort().name == *op_name) {
            return Err(format!("No extractor for op {op_name}"));
        }
    }

    Ok(())
}

/// Hash a single (class_id, node_id) entry. Used both for the full
/// choice-set hash and for the incremental updates in
/// `extract_generation`.
fn hash_choice_entry(class_id: &ClassId, node_id: &NodeId) -> u64 {
    let mut hasher = DefaultHasher::new();
    class_id.hash(&mut hasher);
    node_id.hash(&mut hasher);
    hasher.finish()
}

/// Hash a choice set for uniqueness checking. Order-independent XOR
/// of per-entry hashes. The XOR design lets `extract_generation`
/// update the hash incrementally on each `insert(k, new)` by XORing
/// out `hash_choice_entry(k, old)` and XORing in
/// `hash_choice_entry(k, new)`, dropping the per-attempt cost from
/// O(N log N) over the full choice set to O(M) where M = mutations
/// applied. On large e-graphs (e.g. Gemma's ~3.5M-entry choice set)
/// that's the difference between ~135 seconds and a few milliseconds
/// per generation.
pub fn hash_choice_set(choices: &EGraphChoiceSet) -> u64 {
    let mut h = 0u64;
    for (k, v) in choices {
        h ^= hash_choice_entry(k, v);
    }
    h
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
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'a>> {
    let mutable_classes: Vec<&ClassId> = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| is_search_choice_eclass(label) && enodes.len() > 1)
        .map(|(class_id, _)| class_id)
        .collect();
    extract_generation_from_classes(
        egraph,
        base,
        &mutable_classes,
        generation_size,
        mutations_per_generation,
        prev_selected,
        rng,
    )
}

pub fn extract_reachable_generation<'a>(
    egraph: &'a SerializedEGraph,
    base: &EGraphChoiceSet<'a>,
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'a>> {
    let mutable_classes = reachable_mutable_choice_classes(egraph, base);
    extract_generation_from_classes(
        egraph,
        base,
        &mutable_classes,
        generation_size,
        mutations_per_generation,
        prev_selected,
        rng,
    )
}

fn reachable_mutable_choice_classes<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
) -> Vec<&'a ClassId> {
    let mut reachable = FxHashSet::default();
    let mut class_seen = FxHashSet::default();
    let mut mutable_classes = Vec::new();
    let Some(root) = egraph.roots.first() else {
        return mutable_classes;
    };
    let Some(root_choice) = choices.get(root) else {
        return mutable_classes;
    };
    if let Some((label, enodes)) = egraph.eclasses.get(root) {
        if is_search_choice_eclass(label) && enodes.len() > 1 && class_seen.insert(root) {
            mutable_classes.push(root);
        }
    }

    let mut stack = vec![*root_choice];
    while let Some(node) = stack.pop() {
        if !reachable.insert(node) {
            continue;
        }
        let Some((_, children)) = egraph.enodes.get(node) else {
            continue;
        };
        for child_class in children {
            let Some((label, enodes)) = egraph.eclasses.get(child_class) else {
                continue;
            };
            if is_search_choice_eclass(label) {
                if enodes.len() > 1 && class_seen.insert(child_class) {
                    mutable_classes.push(child_class);
                }
                if let Some(chosen_node) = choices.get(child_class) {
                    stack.push(*chosen_node);
                }
            }
        }
    }

    mutable_classes
}

fn extract_generation_from_classes<'a>(
    egraph: &'a SerializedEGraph,
    base: &EGraphChoiceSet<'a>,
    mutable_classes: &[&'a ClassId],
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'a>> {
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
    // Compute the base's full hash exactly once. Each attempt starts from
    // this and applies XOR diffs for its mutations — no per-attempt
    // O(N log N) sort+hash over the full choice set.
    let base_hash = hash_choice_set(base);

    while offspring.len() < generation_size && attempts < max_attempts {
        attempts += 1;

        // Create a mutated offspring from base
        let mut child = base.clone();
        let mut child_hash = base_hash;

        for _ in 0..rng.random_range(1..=mutations_per_generation) {
            // Pick a random mutable eclass
            let class_id = mutable_classes[rng.random_range(0..mutable_classes.len())];
            let (label, enodes) = &egraph.eclasses[class_id];
            // Pick a random enode for this class
            let consistent_opkind_nodes: Vec<&NodeId> = if label == "OpKind" {
                enodes
                    .iter()
                    .filter(|n| opkind_metadata_consistent(egraph, n))
                    .collect()
            } else {
                Vec::new()
            };
            let new_node = if !consistent_opkind_nodes.is_empty() {
                consistent_opkind_nodes[rng.random_range(0..consistent_opkind_nodes.len())]
            } else {
                &enodes[rng.random_range(0..enodes.len())]
            };
            // Insert returns the previous binding (if any); fold the diff
            // into the running hash. If the new pick equals the old one,
            // the two XORs cancel and `child_hash` is unchanged — exactly
            // the right behaviour.
            let old_node = child.insert(class_id, new_node);
            if let Some(old_node) = old_node {
                child_hash ^= hash_choice_entry(class_id, old_node);
            }
            child_hash ^= hash_choice_entry(class_id, new_node);
        }

        // Hash and check if seen before
        if !prev_selected.contains(&child_hash) {
            prev_selected.insert(child_hash);
            offspring.push(child);
        }
    }
    offspring
}

/// Walk an IList in the egraph, returning the chosen IR enodes in order.
fn walk_ilist<'a>(
    egraph: &'a SerializedEGraph,
    ilist_eclass: &'a ClassId,
    choices: &EGraphChoiceSet<'a>,
) -> Vec<&'a NodeId> {
    let mut inputs = Vec::new();
    let mut current = choices[ilist_eclass];
    loop {
        if egraph.enodes[current].0 == "INil" {
            break;
        }
        // ICons: child[0] = IR eclass, child[1] = IList tail eclass
        let input_eclass = &egraph.enodes[current].1[0];
        let input_node = choices[input_eclass];
        inputs.push(input_node);
        let tail_eclass = &egraph.enodes[current].1[1];
        current = choices[tail_eclass];
    }
    inputs
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
    egglog_to_llir_from_root(
        egraph,
        choices,
        ops,
        custom_ops,
        list_cache,
        expr_cache,
        custom_op_id_remap,
        &egraph.roots[0],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn egglog_to_llir_from_root<'a>(
    egraph: &'a SerializedEGraph,
    choices: EGraphChoiceSet<'a>,
    ops: &'a Vec<Arc<Box<dyn EgglogOp>>>,
    custom_ops: &[Box<dyn CustomOp>],
    list_cache: &mut FxHashMap<&'a NodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a NodeId, Expression>,
    custom_op_id_remap: Option<&FxHashMap<usize, usize>>,
    root_class: &ClassId,
) -> LLIRGraph {
    // Make reachability set from root
    let mut reachable = FxHashSet::default();
    reachable.insert(choices[root_class]);
    let mut reachability_stack = vec![choices[root_class]];
    while let Some(r) = reachability_stack.pop() {
        for ch in &egraph.enodes[r].1 {
            if is_search_choice_eclass(&egraph.eclasses[ch].0) {
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
    // Iterate the small reachable set rather than the full choice set.
    // On large e-graphs (e.g., Gemma's ~3.48M-entry choice set produced
    // by the binary-fusion grow rules cascading through super-block
    // chains), `reachable` is ~3K nodes and the choice set is ~1000×
    // larger. Filtering the choice set against `reachable` was
    // dominating per-candidate `egglog_to_llir` time.
    for &node in &reachable {
        if egraph.eclasses[&egraph.node_to_class[node]].0 != "IR" {
            // Skip IList enodes — `reachable` includes them because the
            // reachability walk follows IList children, but only IR
            // enodes become LLIR nodes.
            continue;
        }
        let enode_label = egraph.enodes[node].0.as_str();
        if enode_label == "Op" {
            // Normalized op: (Op OpKind IList)
            // child[0] = OpKind eclass, child[1] = IList eclass
            let kind_eclass = &egraph.enodes[node].1[0];
            let ilist_eclass = &egraph.enodes[node].1[1];

            // Resolve OpKind enode. The kind eclass may contain multiple
            // structurally-equivalent kind enodes whose ELIST children
            // were unioned but resolve (under the extractor's first-enode
            // walk) to inconsistent lengths — picking such an enode causes
            // a downstream `flatten_strides` length mismatch. Candidate
            // generation filters these out where possible; this fallback is
            // structural only and does not rank backend implementations.
            let kind_enodes = &egraph.eclasses[kind_eclass].1;
            let kind_enode = choices
                .get(kind_eclass)
                .copied()
                .filter(|n| opkind_metadata_consistent(egraph, n))
                .or_else(|| {
                    kind_enodes
                        .iter()
                        .find(|n| opkind_metadata_consistent(egraph, n))
                })
                .unwrap_or(&kind_enodes[0]);
            let kind_label = &egraph.enodes[kind_enode].0;

            // Resolve kind's metadata children (shapes, strides, etc.)
            let kind_children: Vec<&NodeId> = egraph.enodes[kind_enode]
                .1
                .iter()
                .map(|c| {
                    if is_search_choice_eclass(&egraph.eclasses[c].0) {
                        choices[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();

            // Walk IList to get IR inputs
            let input_enodes = walk_ilist(egraph, ilist_eclass, &choices);

            // Check for CustomOpKind first
            if kind_label == "CustomOpKind" {
                // kind_children: [id, dtype]
                let id: usize = egraph.enodes[kind_children[0]].0.parse().unwrap();
                let remapped_id = custom_op_id_remap
                    .and_then(|m| m.get(&id).copied())
                    .unwrap_or(id);
                let r = graph.add_node(custom_ops[remapped_id].to_llir_op());
                enode_to_node.insert(node, r);
                for source in input_enodes {
                    edges_to_place.push((source, node));
                }
            } else {
                // Find matching op by OpKind name
                let Some(op) = ops.iter().find(|op| kind_label.as_str() == op.sort().name) else {
                    todo!("{kind_label} extraction not implemented!");
                };
                let (op_instance, sources) =
                    op.extract(egraph, &kind_children, input_enodes, list_cache, expr_cache);
                let r = graph.add_node(op_instance);
                enode_to_node.insert(node, r);
                for source in sources {
                    edges_to_place.push((source, node));
                }
            }
        } else if enode_label != "OutputJoin" {
            // Direct IR variant (Input, Output) — skip unknown labels (backend IR wrappers)
            let Some(op) = ops.iter().find(|op| enode_label == op.sort().name) else {
                continue;
            };
            let ch = egraph.enodes[node]
                .1
                .iter()
                .map(|c| {
                    if is_search_choice_eclass(&egraph.eclasses[c].0) {
                        choices[c]
                    } else {
                        &egraph.eclasses[c].1[0]
                    }
                })
                .collect_vec();
            // Direct IR ops pass children as kind_children, empty input_enodes
            let (op_instance, sources) = op.extract(egraph, &ch, vec![], list_cache, expr_cache);
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
    // Loop markers (LoopStart/End/Input/InputStatic/Output) are intentionally
    // preserved here — `crate::graph::collapse_loops_to_first_iter` produces
    // a single-iteration LLIR for fast per-candidate profiling, and the full
    // `crate::graph::unroll_loops_in_llir` runs once on the chosen best LLIR
    // before it is loaded into the runtime.
    graph
}

#[cfg(test)]
mod tests {
    use super::{
        LateEgglogPass, SerializedEGraph, count_choice_sets_up_to, run_egglog_with_late_passes,
    };
    use crate::prelude::FxHashMap;
    use crate::{hlir::HLIROps, op::IntoEgglogOp};
    use egraph_serialize::{ClassId, NodeId};

    fn eclass(id: &str, label: &str, n_nodes: usize) -> (ClassId, (String, Vec<NodeId>)) {
        (
            ClassId::from(id),
            (
                label.to_string(),
                (0..n_nodes)
                    .map(|i| NodeId::from(format!("{id}_{i}")))
                    .collect(),
            ),
        )
    }

    fn egraph(eclasses: Vec<(ClassId, (String, Vec<NodeId>))>) -> SerializedEGraph {
        SerializedEGraph {
            enodes: FxHashMap::default(),
            eclasses: eclasses.into_iter().collect(),
            node_to_class: FxHashMap::default(),
            roots: Vec::new(),
        }
    }

    #[test]
    fn counts_ir_and_ilist_choice_sets() {
        let egraph = egraph(vec![
            eclass("a", "IR", 2),
            eclass("b", "IList", 3),
            eclass("op", "OpKind", 5),
            eclass("c", "Shape", 99),
        ]);

        assert_eq!(count_choice_sets_up_to(&egraph, 100), 30);
    }

    #[test]
    fn caps_count_at_limit() {
        let egraph = egraph(vec![eclass("a", "IR", 1_000), eclass("b", "IList", 1_000)]);

        assert_eq!(count_choice_sets_up_to(&egraph, 10), 10);
    }

    #[test]
    fn runs_late_pass_after_full_cleanup() {
        let ops = <HLIROps as IntoEgglogOp>::into_vec();
        let program = r#"
            (let t0 (Input 0 "" (F32)))
            (let t1 (Output t0 0))
        "#;
        let late_pass = LateEgglogPass::new(
            r#"
            (ruleset late_test)
            (rule ((= ?out (Output ?inp ?id)))
                  ((union ?out ?inp))
                  :ruleset late_test
                  :name "late-output-to-input")
            "#,
            "(run-schedule (saturate late_test))",
        );

        let egraph = run_egglog_with_late_passes(program, "t1", &ops, false, &[late_pass])
            .expect("late pass should run");
        let root = egraph.roots.first().expect("root eclass");
        let root_labels: Vec<_> = egraph.eclasses[root]
            .1
            .iter()
            .map(|node| egraph.enodes[node].0.as_str())
            .collect();

        assert!(
            root_labels.contains(&"Input"),
            "late union should add Input to root eclass, got {root_labels:?}"
        );
    }
}
