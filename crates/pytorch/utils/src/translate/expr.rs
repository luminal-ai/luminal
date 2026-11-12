//! Symbolic-value resolution for PT2 expressions.
//!
//! Bare symbols and compound sympy expressions (`Add`, `Mul`, `Pow`,
//! `FloorDiv`, `CleanDiv`, `Mod`, `PythonMod`, `Min`, `Max`) map to recorder
//! `IntExpr`s so dynamic dims survive as expressions instead of freezing at
//! the export hint. A symbolic bool (a comparison of symbolic integers)
//! maps to the expression of its integer reading, 1 or 0. Torch's exported
//! range constraints feed the bounds simplifier; anything unresolved
//! fails translation; a sample hint must never freeze a dynamic dimension.
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
        {
            return self.resolve_expr_str(expr_str);
        }
        value
            .get("as_int")
            .and_then(|v| v.as_i64())
            .map(IntExpr::from)
    }

    /// Resolve a PT2 sym_bool value by name to the expression of its
    /// integer reading: 1 where the comparison holds, 0 where it does not.
    pub(super) fn resolve_sym_bool(&self, name: &str) -> Option<IntExpr> {
        let values = &self.parsed.program.graph_module.graph.sym_bool_values;
        let value = values.get(name)?;
        if let Some(expr_str) = value
            .get("as_expr")
            .and_then(|e| e.get("expr_str"))
            .and_then(|s| s.as_str())
        {
            return self.resolve_expr_str(expr_str);
        }
        value
            .get("as_bool")
            .and_then(|v| v.as_bool())
            .map(|v| IntExpr::from(v as i64))
    }

    /// A symbolic scalar argument as the expression of its value: a
    /// symbolic integer, or a symbolic bool read as 1 or 0.
    pub(super) fn resolve_symbolic_scalar(&self, arg: &Argument) -> Option<IntExpr> {
        match arg {
            Argument::SymBool(value) => self.resolve_sym_bool(&value.as_sym_bool.as_name),
            _ => self.resolve_arg_as_expression(arg),
        }
    }

    /// The sympy expression torch recorded for a named symbolic value, for
    /// a refusal that has to say what it could not use.
    pub(super) fn symbolic_value_source(&self, name: &str) -> Option<&str> {
        let graph = &self.parsed.program.graph_module.graph;
        graph
            .sym_int_values
            .get(name)
            .or_else(|| graph.sym_bool_values.get(name))
            .or_else(|| graph.sym_float_values.get(name))?
            .get("as_expr")?
            .get("expr_str")?
            .as_str()
    }

    /// ` (`name` = <sympy expression>)` for a named symbolic argument, and
    /// the argument's debug form for anything else.
    pub(super) fn describe_symbolic_arg(&self, arg: &Argument) -> String {
        let name = match arg {
            Argument::SymInt(value) => Some(value.as_sym_int.as_name.as_str()),
            Argument::SymBool(value) => Some(value.as_sym_bool.as_name.as_str()),
            Argument::SymFloat(value) => Some(value.as_sym_float.as_name.as_str()),
            _ => None,
        };
        match name {
            Some(name) => match self.symbolic_value_source(name) {
                Some(source) => format!(" (`{name}` = {source})"),
                None => format!(" (`{name}`)"),
            },
            None => format!(" (arg: {arg:?})"),
        }
    }

    /// Whether torch's range constraints prove a dimension expression
    /// nonnegative for every call the program admits.
    pub(super) fn provably_nonnegative(&self, expr: IntExpr) -> bool {
        if let Some(value) = expr.as_num() {
            return value >= 0;
        }
        let symbols = super::sympy::symbol_bounds(&self.symbols, &self.ranges);
        super::sympy::bounds_of_expr(expr, &symbols)
            .min
            .is_some_and(|min| min >= 0)
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
    /// dimension. Bare `Symbol('s77', ...)` and the compound expressions
    /// `sympy::parse_sympy_expr` reads are parsed; torch's exported range
    /// constraints feed the bounds simplifier and decide how a division or
    /// remainder that may see a negative operand is spelled.
    pub(super) fn resolve_expr_str(&self, expr_str: &str) -> Option<IntExpr> {
        super::sympy::parse_sympy_expr_with_ranges(expr_str, &self.symbols, &self.ranges)
    }

    pub(super) fn resolve_expr_value(&self, expr: &ExprValue) -> Option<IntExpr> {
        self.resolve_expr_str(&expr.expr_str)
    }

    /// Shape of a node operand from PT2 metadata, for ops whose output
    /// extent is not derivable from the args (e.g. reductions on `dim`).
    pub(super) fn operand_meta_shape(&self, node: &Node, idx: usize) -> Option<Vec<IntExpr>> {
        let name = node.inputs.get(idx)?.arg.as_value_name()?.to_string();
        let meta = self.tensor_meta(&name).ok()?;
        self.tensor_meta_to_shape(meta).ok()
    }
}
