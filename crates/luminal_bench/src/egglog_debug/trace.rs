//! Dependency chain tracing for dtype propagation analysis.

use super::{DTypeStatus, FactStatus};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A node in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepNode {
    pub var: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub dtype: Option<DTypeStatus>,
}

/// Dependency graph built from egglog program.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub nodes: HashMap<String, DepNode>,
    pub roots: Vec<String>,
}

impl DependencyGraph {
    /// Build dependency graph from egglog program.
    pub fn from_program(program: &str) -> Self {
        let mut graph = DependencyGraph::default();

        for line in program.lines() {
            let line = line.trim();
            if !line.starts_with("(let ") {
                continue;
            }

            // Parse: (let t1 (OpName args...))
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 3 || tokens[0] != "(let" {
                continue;
            }

            let var = tokens[1].to_string();
            let op_type = tokens[2].trim_start_matches('(').to_string();

            // Extract input variables (t followed by digits)
            let mut inputs = Vec::new();
            let bytes = line.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                if bytes[i] == b't' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    let found_var = String::from_utf8_lossy(&bytes[start..i]).to_string();
                    // Don't include self
                    if found_var != var {
                        inputs.push(found_var);
                    }
                } else {
                    i += 1;
                }
            }

            graph.nodes.insert(
                var.clone(),
                DepNode {
                    var,
                    op_type,
                    inputs,
                    dtype: None,
                },
            );
        }

        // Find roots (nodes that are not inputs to any other node)
        let all_inputs: HashSet<String> = graph
            .nodes
            .values()
            .flat_map(|n| n.inputs.iter().cloned())
            .collect();

        graph.roots = graph
            .nodes
            .keys()
            .filter(|k| !all_inputs.contains(*k))
            .cloned()
            .collect();

        graph
    }

    /// Trace the dependency chain from a target variable back to inputs.
    pub fn trace_back(&self, target: &str, max_depth: usize) -> Vec<TraceEntry> {
        let mut result = Vec::new();
        self.trace_back_recursive(target, 0, max_depth, &mut result, &mut HashSet::new());
        result
    }

    fn trace_back_recursive(
        &self,
        var: &str,
        depth: usize,
        max_depth: usize,
        result: &mut Vec<TraceEntry>,
        visited: &mut HashSet<String>,
    ) {
        if depth > max_depth || visited.contains(var) {
            return;
        }
        visited.insert(var.to_string());

        let node = match self.nodes.get(var) {
            Some(n) => n,
            None => {
                result.push(TraceEntry {
                    depth,
                    var: var.to_string(),
                    op_type: "<unknown>".to_string(),
                    dtype: None,
                });
                return;
            }
        };

        result.push(TraceEntry {
            depth,
            var: node.var.clone(),
            op_type: node.op_type.clone(),
            dtype: node.dtype.clone(),
        });

        for input in &node.inputs {
            self.trace_back_recursive(input, depth + 1, max_depth, result, visited);
        }
    }

    /// Find the first node in a chain that has missing dtype.
    pub fn find_dtype_break(&self, target: &str) -> Option<String> {
        let trace = self.trace_back(target, 20);
        for entry in trace {
            if let Some(DTypeStatus::Missing(_)) = entry.dtype {
                return Some(entry.var);
            }
        }
        None
    }
}

/// Entry in a trace result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub depth: usize,
    pub var: String,
    pub op_type: String,
    pub dtype: Option<DTypeStatus>,
}

impl TraceEntry {
    /// Format as indented tree line.
    pub fn format_tree(&self) -> String {
        let indent = "  ".repeat(self.depth);
        let prefix = if self.depth == 0 { "" } else { "├── " };
        let dtype_str = match &self.dtype {
            Some(d) => format!(" dtype={}", d),
            None => String::new(),
        };
        format!(
            "{}{}{} ({}){}",
            indent, prefix, self.var, self.op_type, dtype_str
        )
    }
}

/// Entry in a function trace result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTraceEntry {
    pub depth: usize,
    pub var: String,
    pub op_type: String,
    pub status: FactStatus,
}

impl FunctionTraceEntry {
    pub fn format_tree(&self) -> String {
        let indent = "  ".repeat(self.depth);
        let prefix = if self.depth == 0 { "" } else { "├── " };
        format!(
            "{}{}{} ({}) {}",
            indent, prefix, self.var, self.op_type, self.status
        )
    }
}

/// Result of dtype chain analysis.
#[derive(Debug, Serialize, Deserialize)]
pub struct DTypeChainAnalysis {
    pub target: String,
    pub chain: Vec<TraceEntry>,
    pub first_missing: Option<String>,
    pub all_resolved: bool,
}

impl DTypeChainAnalysis {
    /// Create from dependency graph and target variable.
    pub fn analyze(graph: &DependencyGraph, target: &str) -> Self {
        let chain = graph.trace_back(target, 20);
        let first_missing = chain
            .iter()
            .find(|e| matches!(&e.dtype, Some(DTypeStatus::Missing(_))))
            .map(|e| e.var.clone());
        let all_resolved = chain
            .iter()
            .all(|e| matches!(&e.dtype, Some(DTypeStatus::Resolved(_)) | None));

        DTypeChainAnalysis {
            target: target.to_string(),
            chain,
            first_missing,
            all_resolved,
        }
    }
}

/// Result of function chain analysis.
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionChainAnalysis {
    pub target: String,
    pub fn_name: String,
    pub chain: Vec<FunctionTraceEntry>,
    pub first_missing: Option<String>,
    pub all_resolved: bool,
}
