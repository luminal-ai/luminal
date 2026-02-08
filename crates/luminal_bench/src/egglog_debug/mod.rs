//! Egglog debugging and analysis utilities.
//!
//! This module provides tools for diagnosing egglog lowering issues,
//! particularly when HLIR operations fail to convert to backend implementations.

mod analysis;
mod report;
mod trace;

pub use analysis::*;
pub use report::*;
pub use trace::*;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Extract the operation head from an egglog expression.
///
/// Example: `(Add t1 t2 ...)` -> `Some("Add")`
pub fn egglog_op_head(code: &str) -> Option<&str> {
    let code = code.trim();
    code.strip_prefix('(')
        .and_then(|s| s.split_whitespace().next())
}

/// Summarize HLIR node types in a graph.
pub fn summarize_hlir_ops(cx: &luminal::prelude::Graph) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in cx.graph.node_indices() {
        let name = cx.graph[node].type_name().to_string();
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

/// Summarize egglog operation heads from a program string.
pub fn summarize_egglog_ops(program: &str) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for line in program.lines() {
        // Parse lines like: (let t1 (Add ...))
        let Some(code) = line.splitn(3, ' ').nth(2) else {
            continue;
        };
        let Some(head) = egglog_op_head(code) else {
            continue;
        };
        *counts.entry(head.to_string()).or_insert(0) += 1;
    }
    counts
}

/// Result of dtype analysis for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DTypeStatus {
    /// dtype was successfully resolved
    Resolved(String),
    /// dtype lookup failed
    Missing(String),
}

impl DTypeStatus {
    pub fn is_missing(&self) -> bool {
        matches!(self, DTypeStatus::Missing(_))
    }

    pub fn is_resolved(&self) -> bool {
        matches!(self, DTypeStatus::Resolved(_))
    }
}

impl std::fmt::Display for DTypeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypeStatus::Resolved(s) => write!(f, "{}", s),
            DTypeStatus::Missing(err) => write!(f, "<missing:{}>", err),
        }
    }
}

/// Result of evaluating an arbitrary egglog function for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactStatus {
    /// function value was successfully resolved
    Resolved(String),
    /// function lookup failed
    Missing(String),
}

impl FactStatus {
    pub fn is_missing(&self) -> bool {
        matches!(self, FactStatus::Missing(_))
    }
}

impl std::fmt::Display for FactStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactStatus::Resolved(s) => write!(f, "{}", s),
            FactStatus::Missing(err) => write!(f, "<missing:{}>", err),
        }
    }
}
