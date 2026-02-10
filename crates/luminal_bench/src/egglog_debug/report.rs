//! Formatted output and reporting for egglog debug analysis.

use super::{
    DTypeChainAnalysis, DTypeStatus, EnodeInspection, FunctionChainAnalysis, LoweringAnalysis,
    OpLoweringReport, TraceEntry, VarInspection,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Print HLIR node type summary.
pub fn print_hlir_summary(counts: &BTreeMap<String, usize>) {
    println!("-- HLIR node types --");
    for (k, v) in counts {
        println!("  {}: {}", k, v);
    }
}

/// Print egglog op head summary.
pub fn print_egglog_summary(counts: &BTreeMap<String, usize>) {
    println!("-- Egglog op heads --");
    for (k, v) in counts {
        println!("  {}: {}", k, v);
    }
}

/// Print lowering analysis results.
pub fn print_lowering_analysis(analysis: &LoweringAnalysis) {
    println!("-- {} Analysis --", analysis.label);
    println!("  Root eclass labels: {}", analysis.root_labels.join("|"));

    if !analysis.output_input_labels.is_empty() {
        println!(
            "  Output input labels: {}",
            analysis.output_input_labels.join("|")
        );
    }

    if !analysis.facts.is_empty() {
        println!("  Facts:");
        for (fn_name, table) in &analysis.facts {
            println!("    {}:", fn_name);
            for (var, status) in table {
                let prefix = if status.is_missing() { "❌" } else { "√" };
                println!("      {} {}: {}", prefix, var, status);
            }
        }
    }

    if !analysis.op_reports.is_empty() {
        for report in &analysis.op_reports {
            print_op_lowering_report(report);
        }
    }
}

/// Print dependency trace as a tree.
pub fn print_trace_tree(trace: &[TraceEntry]) {
    println!("-- Dependency trace --");
    for entry in trace {
        println!("{}", entry.format_tree());
    }
}

/// Print dtype chain analysis.
pub fn print_dtype_chain(analysis: &DTypeChainAnalysis) {
    println!("-- DType chain analysis for {} --", analysis.target);

    if analysis.all_resolved {
        println!("  √ All nodes in chain have resolved dtype");
    } else if let Some(ref first) = analysis.first_missing {
        println!("  ❌ First missing dtype at: {}", first);
    }

    println!("  Chain:");
    for entry in &analysis.chain {
        let dtype_str = match &entry.dtype {
            Some(DTypeStatus::Resolved(s)) => format!("√ {}", s),
            Some(DTypeStatus::Missing(_)) => "❌ missing".to_string(),
            None => "? unknown".to_string(),
        };
        let indent = "    ".repeat(entry.depth + 1);
        println!("{}{} ({}) {}", indent, entry.var, entry.op_type, dtype_str);
    }
}

/// Print function chain analysis.
pub fn print_function_chain(analysis: &FunctionChainAnalysis) {
    println!(
        "-- Function chain analysis for {} (fn={}) --",
        analysis.target, analysis.fn_name
    );

    if analysis.all_resolved {
        println!("  √ All nodes in chain have resolved value");
    } else if let Some(ref first) = analysis.first_missing {
        println!("  ❌ First missing at: {}", first);
    }

    println!("  Chain:");
    for entry in &analysis.chain {
        println!("{}", entry.format_tree());
    }
}

/// Print op lowering report.
pub fn print_op_lowering_report(report: &OpLoweringReport) {
    println!(
        "-- Op lowering [{}] {} -> {} --",
        report.label, report.hlir_op, report.backend_op
    );
    println!("  total eclasses: {}", report.total_classes);
    if report.missing.is_empty() {
        println!("  √ All eclasses have backend equivalent");
    } else {
        println!("  ❌ Missing backend in {} eclasses:", report.missing.len());
        for miss in &report.missing {
            println!("    - class={} op={}", miss.class_id, miss.op);
            for (idx, child) in miss.children.iter().enumerate() {
                let labels = if child.class_labels.is_empty() {
                    "<none>".to_string()
                } else {
                    child.class_labels.join("|")
                };
                let dtype = child
                    .dtype
                    .clone()
                    .unwrap_or_else(|| "<missing>".to_string());
                println!(
                    "      [{}] class={} type={} labels={} dtype={}",
                    idx, child.class_id, child.class_type, labels, dtype
                );
            }
        }
    }
}

fn print_enode(enode: &EnodeInspection) {
    println!("    - {}", enode.label);
    for (idx, child) in enode.children.iter().enumerate() {
        let dtype = child
            .dtype
            .clone()
            .unwrap_or_else(|| "<missing>".to_string());
        let labels = if child.class_labels.is_empty() {
            "<none>".to_string()
        } else {
            child.class_labels.join("|")
        };
        println!(
            "      [{}] class={} type={} labels={} dtype={}",
            idx, child.class_id, child.class_type, labels, dtype
        );
    }
}

/// Print inspection results for a specific variable.
pub fn print_var_inspection(inspection: &VarInspection) {
    println!(
        "-- Var inspection [{}] {} --",
        inspection.label, inspection.var
    );

    if let Some(ref line) = inspection.let_line {
        println!("  let: {}", line);
    }
    if let Some(ref err) = inspection.eval_error {
        println!("  eval error: {}", err);
        return;
    }

    let class_id = inspection.class_id.as_deref().unwrap_or("<unknown>");
    let class_type = inspection.class_type.as_deref().unwrap_or("<unknown>");
    let dtype = inspection.dtype.as_deref().unwrap_or("<missing>");
    let labels = if inspection.class_labels.is_empty() {
        "<none>".to_string()
    } else {
        inspection.class_labels.join("|")
    };

    println!(
        "  class: {} type={} labels={}",
        class_id, class_type, labels
    );
    println!("  dtype: {}", dtype);
    println!("  enodes:");
    for enode in &inspection.enodes {
        print_enode(enode);
    }
}

/// Summary report for a debug session.
#[derive(Debug, Serialize, Deserialize)]
pub struct DebugReport {
    pub case_name: String,
    pub size: usize,
    pub hlir_counts: BTreeMap<String, usize>,
    pub egglog_counts: BTreeMap<String, usize>,
    pub hlir_analysis: Option<LoweringAnalysis>,
    pub backend_analysis: Option<LoweringAnalysis>,
    pub var_inspections: Vec<VarInspection>,
    pub function_traces: Vec<FunctionChainAnalysis>,
    pub build_succeeded: bool,
}

impl DebugReport {
    /// Print full report to stdout.
    pub fn print(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Case: {} (size={})", self.case_name, self.size);
        println!("{}", "=".repeat(60));

        print_hlir_summary(&self.hlir_counts);
        println!();
        print_egglog_summary(&self.egglog_counts);

        if let Some(ref analysis) = self.hlir_analysis {
            println!();
            print_lowering_analysis(analysis);
        }

        if let Some(ref analysis) = self.backend_analysis {
            println!();
            print_lowering_analysis(analysis);
        }

        if !self.function_traces.is_empty() {
            for trace in &self.function_traces {
                println!();
                print_function_chain(trace);
            }
        }

        if !self.var_inspections.is_empty() {
            for inspection in &self.var_inspections {
                println!();
                print_var_inspection(inspection);
            }
        }

        println!();
        if self.build_succeeded {
            println!("√ build_search_space succeeded");
        } else {
            println!("❌ build_search_space failed");
        }
    }
}
