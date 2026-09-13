//! Symbolic-dimension resolution for PT2 expressions.
//!
//! Bare symbols map to recorder `Symbol` dims so dynamic dims survive as
//! symbols instead of freezing at the export hint. Compound sympy
//! expressions are a later port (`pt2_expr`); anything unresolved falls
//! back to the exported hint, which keeps static programs exact.
#![allow(dead_code)]

use std::collections::HashMap;

use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, DimSize, ExprValue, Node};

/// Collect the bare symbols PT2 uses in its tensor metadata, mapped to
/// recorder dim symbols. A name the recorder rejects is remapped (never
/// dropped): a dropped symbol would freeze the dim at its hint.
pub(super) fn build_sym_map(parsed: &crate::pt2_parser::ParsedPT2) -> HashMap<String, Symbol> {
    let mut names: Vec<String> = Vec::new();
    for meta in parsed.program.graph_module.graph.tensor_values.values() {
        for size in &meta.sizes {
            if let DimSize::Expr(expr) = size
                && let Some(name) =
                    crate::pt2_parser::extract_symbol_name_pub(&expr.as_expr.expr_str)
                && !names.contains(&name)
            {
                names.push(name);
            }
        }
    }
    let mut map = HashMap::new();
    let mut minted = 0usize;
    for name in names {
        let symbol = Symbol::try_new_dim(&name).unwrap_or_else(|_| {
            let replacement = loop {
                let candidate = format!("pt2_dim_{minted}");
                minted += 1;
                if !map.contains_key(&candidate) {
                    break candidate;
                }
            };
            Symbol::try_new_dim(&replacement).expect("minted names are well-formed")
        });
        map.insert(name, symbol);
    }
    map
}

impl Translator<'_> {
    /// Resolve a PT2 sym_int value by name to a dimension expression.
    pub(super) fn resolve_sym_int(&self, name: &str) -> Option<IntExpr> {
        let values = &self.parsed.program.graph_module.graph.sym_int_values;
        let value = values.get(name)?;
        if let Some(expr_str) = value
            .get("as_expr")
            .and_then(|e| e.get("expr_str"))
            .and_then(|s| s.as_str())
            && let Some(expr) = self.resolve_expr_str(expr_str)
        {
            return Some(expr);
        }
        value
            .get("as_expr")
            .and_then(|e| e.get("hint"))
            .and_then(|h| h.get("as_int"))
            .and_then(|v| v.as_i64())
            .map(IntExpr::from)
    }

    pub(super) fn resolve_arg_as_expression(&self, arg: &Argument) -> Option<IntExpr> {
        if let Some(v) = arg.as_int() {
            return Some(IntExpr::from(v));
        }
        if let Some(name) = arg.as_sym_int_name() {
            return self.resolve_sym_int(name);
        }
        if let Argument::Expr(e) = arg {
            return self.resolve_expr_value(&e.as_expr);
        }
        None
    }

    /// A bare `Symbol('s77', ...)` becomes its recorder dim symbol; a
    /// compound expression is not parsed yet.
    pub(super) fn resolve_expr_str(&self, expr_str: &str) -> Option<IntExpr> {
        let sym = crate::pt2_parser::extract_symbol_name_pub(expr_str)?;
        self.symbols.get(&sym).copied().map(IntExpr::from)
    }

    pub(super) fn resolve_expr_value(&self, expr: &ExprValue) -> Option<IntExpr> {
        self.resolve_expr_str(&expr.expr_str).or_else(|| {
            expr.hint
                .as_ref()
                .and_then(|h| h.as_int())
                .map(IntExpr::from)
        })
    }

    /// Shape of a node operand from PT2 metadata, for ops whose output
    /// extent is not derivable from the args (e.g. reductions on `dim`).
    pub(super) fn operand_meta_shape(&self, node: &Node, idx: usize) -> Option<Vec<IntExpr>> {
        let name = node.inputs.get(idx)?.arg.as_value_name()?.to_string();
        let meta = self.tensor_meta(&name).ok()?;
        self.tensor_meta_to_shape(meta).ok()
    }
}
