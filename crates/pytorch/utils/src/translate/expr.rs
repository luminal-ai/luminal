//! Symbolic-dimension resolution for PT2 expressions.
//!
//! Bare symbols and compound sympy expressions (`Add`, `Mul`, `FloorDiv`,
//! `Mod`, `Min`, `Max`) map to recorder `IntExpr`s so dynamic dims survive
//! as expressions instead of freezing at the export hint. Torch's exported
//! range constraints feed the bounds simplifier; anything unresolved falls
//! back to the exported hint, which keeps static programs exact.
#![allow(dead_code)]

use luminal::prelude::*;

use super::Translator;
use crate::pt2_schema::{Argument, ExprValue, Node, SymIntEntry};

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

    /// Resolve a shape-like argument into recorder dimension expressions.
    /// Handles an int list, a sym-int list (each entry a literal or a
    /// named sym-int that may itself be a compound expression), and a
    /// single expression. Returns `None` when the argument is absent or
    /// not shape-like, or when any entry cannot be resolved.
    pub(super) fn resolve_shape_arg(&self, node: &Node, idx: usize) -> Option<Vec<IntExpr>> {
        match &node.inputs.get(idx)?.arg {
            Argument::Ints(list) => Some(list.as_ints.iter().map(|v| IntExpr::from(*v)).collect()),
            Argument::SymInts(list) => list
                .as_sym_ints
                .iter()
                .map(|entry| match entry {
                    SymIntEntry::Int(i) => Some(IntExpr::from(i.as_int)),
                    SymIntEntry::Name(name) => self.resolve_sym_int(&name.as_name),
                })
                .collect(),
            Argument::Expr(expr) => self.resolve_expr_value(&expr.as_expr).map(|e| vec![e]),
            _ => None,
        }
    }

    /// Resolve a PT2 sympy `srepr` expression string into a recorder
    /// dimension. Bare `Symbol('s77', ...)` and compound expressions
    /// (`Add`, `Mul`, `FloorDiv`, `Mod`, `Min`, `Max`) are parsed; torch's
    /// exported range constraints feed the bounds simplifier.
    pub(super) fn resolve_expr_str(&self, expr_str: &str) -> Option<IntExpr> {
        super::sympy::parse_sympy_expr_with_ranges(expr_str, &self.symbols, &self.ranges)
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
