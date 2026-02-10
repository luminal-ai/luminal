//! Core analysis functions for egglog debugging.

use super::{
    DTypeChainAnalysis, DTypeStatus, DependencyGraph, FactStatus, FunctionChainAnalysis,
    FunctionTraceEntry,
};
use egraph_serialize::ClassId;
use luminal::egglog_utils;
use luminal::hlir::HLIROps;
use luminal::op::{EgglogOp, IntoEgglogOp, Runtime};
use luminal::prelude::egglog;
use luminal::prelude::egglog::prelude::exprs;
use luminal::prelude::egglog_ast::{RustSpan, Span};
use luminal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Analysis result for lowering.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LoweringAnalysis {
    pub label: String,
    pub root_labels: Vec<String>,
    pub output_input_labels: Vec<String>,
    /// Optional op-coverage reports (only filled when explicitly requested).
    pub op_reports: Vec<OpLoweringReport>,
    /// Optional evaluated facts, keyed by function name then variable name.
    pub facts: BTreeMap<String, BTreeMap<String, FactStatus>>,
}

/// Query for evaluating a function on a set of variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQuery {
    pub fn_name: String,
    pub vars: Vec<String>,
}

/// Missing backend equivalent for a specific HLIR op instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpMissing {
    pub class_id: String,
    pub op: String,
    pub children: Vec<ChildInspection>,
}

/// Report for op lowering coverage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpLoweringReport {
    pub label: String,
    pub hlir_op: String,
    pub backend_op: String,
    pub total_classes: usize,
    pub missing: Vec<OpMissing>,
}

/// Inspection of a specific variable's eclass and dtype facts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VarInspection {
    pub label: String,
    pub var: String,
    pub let_line: Option<String>,
    pub eval_error: Option<String>,
    pub class_id: Option<String>,
    pub class_type: Option<String>,
    pub class_labels: Vec<String>,
    pub dtype: Option<String>,
    pub enodes: Vec<EnodeInspection>,
}

/// Inspection of an enode within an eclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnodeInspection {
    pub label: String,
    pub children: Vec<ChildInspection>,
}

/// Inspection of a child eclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChildInspection {
    pub class_id: String,
    pub class_type: String,
    pub class_labels: Vec<String>,
    pub dtype: Option<String>,
}

fn find_let_line(program: &str, var: &str) -> Option<String> {
    for line in program.lines() {
        let line = line.trim();
        if line.starts_with("(let ") && line.split_whitespace().nth(1) == Some(var) {
            return Some(line.to_string());
        }
    }
    None
}

fn run_egraph(program: &str, ops: Vec<Arc<Box<dyn EgglogOp>>>) -> egglog::EGraph {
    let code = egglog_utils::full_egglog(program, &ops, false);
    let mut egraph = egglog::EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code).unwrap();
    let _outputs = egraph.run_program(commands).unwrap();
    egraph
}

fn annotate_dtypes(graph: &mut DependencyGraph, egraph: &mut egglog::EGraph) {
    for node in graph.nodes.values_mut() {
        node.dtype = Some(eval_dtype(egraph, &node.var));
    }
}

fn class_labels(serialized: &egglog_utils::SerializedEGraph, class_id: &ClassId) -> Vec<String> {
    let Some((_, nodes)) = serialized.eclasses.get(class_id) else {
        return vec!["<missing>".to_string()];
    };
    let mut labels: Vec<String> = nodes
        .iter()
        .filter_map(|node_id| {
            serialized
                .enodes
                .get(node_id)
                .map(|(label, _)| label.clone())
        })
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn class_type(serialized: &egglog_utils::SerializedEGraph, class_id: &ClassId) -> String {
    serialized
        .eclasses
        .get(class_id)
        .map(|(typ, _)| typ.clone())
        .unwrap_or_else(|| "<missing>".to_string())
}

fn collect_dtype_facts(serialized: &egglog_utils::SerializedEGraph) -> FxHashMap<ClassId, String> {
    let mut map: FxHashMap<ClassId, String> = FxHashMap::default();
    for (node_id, (label, children)) in &serialized.enodes {
        if !label.starts_with("dtype") {
            continue;
        }
        if children.is_empty() {
            continue;
        }
        let input_class = children[0].clone();
        let dtype_class = serialized.node_to_class[node_id].clone();
        let dtype_labels = class_labels(serialized, &dtype_class);
        let dtype_label = dtype_labels
            .first()
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());
        map.insert(input_class, dtype_label);
    }
    map
}

fn eval_function(egraph: &mut egglog::EGraph, fn_name: &str, var: &str) -> FactStatus {
    let expr = exprs::call(fn_name, vec![exprs::var(var)]);
    match egraph.eval_expr(&expr) {
        Ok((sort, value)) => match egraph.extract_value_to_string(&sort, value) {
            Ok((s, _)) => FactStatus::Resolved(s),
            Err(_) => FactStatus::Missing("extract-error".to_string()),
        },
        Err(err) => FactStatus::Missing(format!("{err}")),
    }
}

/// Inspect a specific var with a given set of ops.
pub fn inspect_var_with_ops(
    program: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    var: &str,
    label: &str,
) -> VarInspection {
    let mut egraph = run_egraph(program, ops);
    let let_line = find_let_line(program, var);

    let mut inspection = VarInspection {
        label: label.to_string(),
        var: var.to_string(),
        let_line,
        ..Default::default()
    };

    let var_expr = egglog::var!(var.to_string());
    let (sort, value) = match egraph.eval_expr(&var_expr) {
        Ok(res) => res,
        Err(err) => {
            inspection.eval_error = Some(format!("{err}"));
            return inspection;
        }
    };

    let serialized = egglog_utils::SerializedEGraph::new(&egraph, vec![(sort, value)]);
    let dtype_facts = collect_dtype_facts(&serialized);

    let class_id = serialized.roots.first().cloned();
    if let Some(class_id) = class_id {
        inspection.class_id = Some(format!("{:?}", class_id));
        inspection.class_type = Some(class_type(&serialized, &class_id));
        inspection.class_labels = class_labels(&serialized, &class_id);
        inspection.dtype = dtype_facts.get(&class_id).cloned();

        if let Some((_, nodes)) = serialized.eclasses.get(&class_id) {
            for node_id in nodes {
                let Some((label, children)) = serialized.enodes.get(node_id) else {
                    continue;
                };
                let mut enode = EnodeInspection {
                    label: label.clone(),
                    ..Default::default()
                };
                for child in children {
                    let child_labels = class_labels(&serialized, child);
                    let child_type = class_type(&serialized, child);
                    let dtype = dtype_facts.get(child).cloned();
                    enode.children.push(ChildInspection {
                        class_id: format!("{:?}", child),
                        class_type: child_type,
                        class_labels: child_labels,
                        dtype,
                    });
                }
                inspection.enodes.push(enode);
            }
        }
    }

    inspection
}

/// Inspect a specific var using HLIR-only ops.
pub fn inspect_var_hlir(program: &str, var: &str) -> VarInspection {
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    inspect_var_with_ops(program, hlir_ops, var, "HLIR")
}

/// Evaluate dtype for a variable in an egraph.
pub fn eval_dtype(egraph: &mut egglog::EGraph, var: &str) -> DTypeStatus {
    let expr = egglog::call!("dtype", vec![egglog::var!(var.to_string())]);
    match egraph.eval_expr(&expr) {
        Ok((sort, value)) => match egraph.extract_value_to_string(&sort, value) {
            Ok((s, _)) => DTypeStatus::Resolved(s),
            Err(_) => DTypeStatus::Missing("extract-error".to_string()),
        },
        Err(err) => DTypeStatus::Missing(format!("{err}")),
    }
}

/// Analyze backend lowering for a specific HLIR op -> backend op mapping.
pub fn analyze_op_lowering_with_ops(
    program: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    hlir_op: &str,
    backend_op: &str,
    label: &str,
) -> OpLoweringReport {
    let mut egraph = run_egraph(program, ops);
    let (sort, value) = egraph
        .eval_expr(&egglog::var!("t0"))
        .or_else(|_| egraph.eval_expr(&egglog::var!("t1")))
        .unwrap_or_else(|_| {
            panic!("failed to eval any root variable (t0/t1) for op inspection");
        });
    let serialized = egglog_utils::SerializedEGraph::new(&egraph, vec![(sort, value)]);
    let dtype_facts = collect_dtype_facts(&serialized);

    let mut eclass_has_backend: FxHashSet<ClassId> = FxHashSet::default();
    for (node_id, (lbl, _)) in &serialized.enodes {
        if lbl == backend_op {
            eclass_has_backend.insert(serialized.node_to_class[node_id].clone());
        }
    }

    let mut seen_classes: FxHashSet<ClassId> = FxHashSet::default();
    let mut missing: Vec<OpMissing> = Vec::new();
    for (node_id, (lbl, children)) in &serialized.enodes {
        if lbl != hlir_op {
            continue;
        }
        let class_id = &serialized.node_to_class[node_id];
        if seen_classes.contains(class_id) {
            continue;
        }
        seen_classes.insert(class_id.clone());
        if eclass_has_backend.contains(class_id) {
            continue;
        }
        let mut child_summaries = Vec::new();
        for child in children {
            let labels = class_labels(&serialized, child);
            let typ = class_type(&serialized, child);
            let dtype = dtype_facts.get(child).cloned();
            child_summaries.push(ChildInspection {
                class_id: format!("{:?}", child),
                class_type: typ,
                class_labels: labels,
                dtype,
            });
        }
        missing.push(OpMissing {
            class_id: format!("{:?}", class_id),
            op: hlir_op.to_string(),
            children: child_summaries,
        });
    }

    OpLoweringReport {
        label: label.to_string(),
        hlir_op: hlir_op.to_string(),
        backend_op: backend_op.to_string(),
        total_classes: seen_classes.len(),
        missing,
    }
}

/// Analyze backend lowering for a specific HLIR op using a runtime's ops.
pub fn analyze_backend_op_lowering<R: Runtime>(
    program: &str,
    hlir_op: &str,
    backend_op: &str,
) -> OpLoweringReport
where
    R::Ops: IntoEgglogOp,
{
    let mut backend_ops = R::Ops::into_vec();
    backend_ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
    let label = format!(
        "{}+HLIR",
        std::any::type_name::<R>()
            .split("::")
            .last()
            .unwrap_or("Backend")
    );
    analyze_op_lowering_with_ops(program, backend_ops, hlir_op, backend_op, &label)
}

/// Analyze lowering with a specific set of ops.
///
/// This is the core analysis function that works with any set of ops.
/// Use `analyze_lowering` for convenience when working with a specific backend.
pub fn analyze_with_ops(
    program: &str,
    root: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    label: &str,
    fact_queries: &[FactQuery],
    op_mappings: &[(String, String)],
) -> LoweringAnalysis {
    let mut egraph = run_egraph(program, ops);

    let (sort, value) = egraph.eval_expr(&egglog::var!(root)).unwrap();
    let serialized = egglog_utils::SerializedEGraph::new(&egraph, vec![(sort, value)]);
    let dtype_facts = collect_dtype_facts(&serialized);

    let root_class_id = serialized.roots.first().unwrap();
    let root_labels = class_labels(&serialized, root_class_id);

    let mut analysis = LoweringAnalysis {
        label: label.to_string(),
        root_labels,
        output_input_labels: Vec::new(),
        op_reports: Vec::new(),
        facts: BTreeMap::new(),
    };

    // Output input labels (if any Output exists under this root).
    for (lbl, children) in serialized.enodes.values() {
        if lbl != "Output" || children.is_empty() {
            continue;
        }
        analysis.output_input_labels = class_labels(&serialized, &children[0]);
        break;
    }

    // Op coverage reports (only when explicitly requested).
    for (hlir_op, backend_op) in op_mappings {
        // Determine which eclasses contain the backend op.
        let mut eclass_has_backend: FxHashSet<ClassId> = FxHashSet::default();
        for (node_id, (lbl, _)) in &serialized.enodes {
            if lbl == backend_op {
                eclass_has_backend.insert(serialized.node_to_class[node_id].clone());
            }
        }

        let mut seen_classes: FxHashSet<ClassId> = FxHashSet::default();
        let mut missing: Vec<OpMissing> = Vec::new();
        for (node_id, (lbl, children)) in &serialized.enodes {
            if lbl != hlir_op {
                continue;
            }
            let class_id = &serialized.node_to_class[node_id];
            if !seen_classes.insert(class_id.clone()) {
                continue;
            }
            if eclass_has_backend.contains(class_id) {
                continue;
            }
            let mut child_summaries = Vec::new();
            for child in children {
                child_summaries.push(ChildInspection {
                    class_id: format!("{:?}", child),
                    class_type: class_type(&serialized, child),
                    class_labels: class_labels(&serialized, child),
                    dtype: dtype_facts.get(child).cloned(),
                });
            }
            missing.push(OpMissing {
                class_id: format!("{:?}", class_id),
                op: hlir_op.clone(),
                children: child_summaries,
            });
        }

        analysis.op_reports.push(OpLoweringReport {
            label: label.to_string(),
            hlir_op: hlir_op.clone(),
            backend_op: backend_op.clone(),
            total_classes: seen_classes.len(),
            missing,
        });
    }

    // Evaluate requested facts.
    for q in fact_queries {
        let mut table: BTreeMap<String, FactStatus> = BTreeMap::new();
        for var in &q.vars {
            table.insert(var.clone(), eval_function(&mut egraph, &q.fn_name, var));
        }
        analysis.facts.insert(q.fn_name.clone(), table);
    }

    analysis
}

/// Analyze dtype propagation chain for a specific variable with a given set of ops.
pub fn analyze_dtype_chain_with_ops(
    program: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    target: &str,
) -> DTypeChainAnalysis {
    let mut egraph = run_egraph(program, ops);
    let mut graph = DependencyGraph::from_program(program);
    annotate_dtypes(&mut graph, &mut egraph);
    DTypeChainAnalysis::analyze(&graph, target)
}

/// Analyze dtype propagation chain for a specific variable using HLIR-only ops.
pub fn analyze_hlir_dtype_chain(program: &str, target: &str) -> DTypeChainAnalysis {
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    analyze_dtype_chain_with_ops(program, hlir_ops, target)
}

/// Analyze function propagation chain for a specific variable with a given set of ops.
pub fn analyze_function_chain_with_ops(
    program: &str,
    ops: Vec<Arc<Box<dyn EgglogOp>>>,
    fn_name: &str,
    target: &str,
) -> FunctionChainAnalysis {
    let mut egraph = run_egraph(program, ops);
    let graph = DependencyGraph::from_program(program);
    let trace = graph.trace_back(target, 20);
    let mut chain = Vec::new();
    let mut first_missing = None;

    for entry in trace {
        let status = eval_function(&mut egraph, fn_name, &entry.var);
        if first_missing.is_none() && status.is_missing() {
            first_missing = Some(entry.var.clone());
        }
        chain.push(FunctionTraceEntry {
            depth: entry.depth,
            var: entry.var,
            op_type: entry.op_type,
            status,
        });
    }

    let all_resolved = first_missing.is_none();
    FunctionChainAnalysis {
        target: target.to_string(),
        fn_name: fn_name.to_string(),
        chain,
        first_missing,
        all_resolved,
    }
}

/// Analyze function propagation chain for a specific variable using HLIR-only ops.
pub fn analyze_hlir_function_chain(
    program: &str,
    fn_name: &str,
    target: &str,
) -> FunctionChainAnalysis {
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    analyze_function_chain_with_ops(program, hlir_ops, fn_name, target)
}

/// Run full lowering analysis comparing HLIR-only vs Backend+HLIR.
///
/// This is a generic function that works with any backend implementing `Runtime`.
///
/// # Type Parameters
/// * `R` - The runtime type (e.g., `MetalRuntime`, `CudaRuntime`)
///
/// # Arguments
/// * `program` - The egglog program string
/// * `root` - The root variable name
/// * `backend_add_name` - The name of the backend's Add operation (e.g., "MetalAdd")
pub fn analyze_lowering<R: Runtime>(
    program: &str,
    root: &str,
    fact_queries: &[FactQuery],
    op_mappings: &[(String, String)],
) -> (LoweringAnalysis, LoweringAnalysis)
where
    R::Ops: IntoEgglogOp,
{
    // HLIR-only analysis
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    let hlir_analysis = analyze_with_ops(program, root, hlir_ops, "HLIR", fact_queries, &[]);

    // Backend+HLIR analysis
    let mut backend_ops = R::Ops::into_vec();
    backend_ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
    let backend_label = format!(
        "{}+HLIR",
        std::any::type_name::<R>()
            .split("::")
            .last()
            .unwrap_or("Backend")
    );
    let backend_analysis = analyze_with_ops(
        program,
        root,
        backend_ops,
        &backend_label,
        fact_queries,
        op_mappings,
    );

    (hlir_analysis, backend_analysis)
}

/// Convenience function for HLIR-only analysis (no backend).
pub fn analyze_hlir_only(program: &str, root: &str) -> LoweringAnalysis {
    let hlir_ops = <HLIROps as IntoEgglogOp>::into_vec();
    analyze_with_ops(program, root, hlir_ops, "HLIR", &[], &[])
}
