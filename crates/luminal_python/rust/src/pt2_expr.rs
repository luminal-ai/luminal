use std::collections::HashMap;

use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_parser::SymDimMap;
use crate::pt2_schema::RangeConstraint;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExprBounds {
    pub(crate) min: Option<i64>,
    pub(crate) max: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
struct ParsedExpr {
    expr: Expression,
    bounds: ExprBounds,
}

impl ParsedExpr {
    fn exact(expr: Expression, value: i64) -> Self {
        Self {
            expr,
            bounds: ExprBounds {
                min: Some(value),
                max: Some(value),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BoundedExpr {
    expr: Expression,
    bounds: ExprBounds,
}

/// Parse a sympy `srepr`-style expression string into a luminal `Expression`.
///
/// Supports the subset of sympy heads PT2 emits for symbolic shape metadata.
pub(crate) fn parse_sympy_expr(
    expr: &str,
    sym_to_char: &HashMap<String, char>,
) -> Option<Expression> {
    parse_sympy_expr_with_ranges(expr, sym_to_char, &HashMap::new())
}

pub(crate) fn parse_sympy_expr_with_ranges(
    expr: &str,
    sym_to_char: &HashMap<String, char>,
    ranges: &HashMap<String, RangeConstraint>,
) -> Option<Expression> {
    parse_sympy_expr_inner(expr, sym_to_char, ranges).map(|parsed| parsed.expr)
}

pub(crate) fn sym_char_ranges(sym_map: &SymDimMap) -> FxHashMap<char, ExprBounds> {
    sym_map
        .sym_to_char
        .iter()
        .map(|(sym_name, sym_char)| {
            let range = sym_map.ranges.get(sym_name);
            let min = range
                .and_then(|range| range.min_val)
                .map(|min| min.max(0))
                .or(Some(0));
            let max = range
                .and_then(|range| range.max_val)
                .filter(|max| *max >= 0);
            (*sym_char, ExprBounds { min, max })
        })
        .collect()
}

pub(crate) fn simplify_expr_with_ranges(
    expr: Expression,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> Expression {
    simplify_bound_expr(expr, sym_ranges).expr
}

pub(crate) fn same_expr_with_ranges(
    lhs: Expression,
    rhs: Expression,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> bool {
    let lhs = simplify_bound_expr(lhs, sym_ranges);
    let rhs = simplify_bound_expr(rhs, sym_ranges);
    lhs.expr == rhs.expr
        || lhs.expr.egglog_equal(rhs.expr)
        || (exact_value(lhs) == exact_value(rhs) && exact_value(lhs).is_some())
}

pub(crate) fn canonical_equal_expr(
    lhs: Expression,
    rhs: Expression,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> Option<Expression> {
    if !same_expr_with_ranges(lhs, rhs, sym_ranges) {
        return None;
    }
    let lhs_simplified = simplify_expr_with_ranges(lhs, sym_ranges);
    let rhs_simplified = simplify_expr_with_ranges(rhs, sym_ranges);
    Some(if lhs_simplified.len() <= rhs_simplified.len() {
        lhs_simplified
    } else {
        rhs_simplified
    })
}

fn parse_sympy_expr_inner(
    expr: &str,
    sym_to_char: &HashMap<String, char>,
    ranges: &HashMap<String, RangeConstraint>,
) -> Option<ParsedExpr> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    if let Ok(value) = expr.parse::<i64>() {
        return Some(ParsedExpr::exact(Expression::from(value), value));
    }

    let (head, body) = split_head(expr)?;
    match head {
        "Symbol" => {
            let name = extract_first_quoted(body)?;
            let bounds = infer_symbol_bounds(body, ranges.get(&name));
            sym_to_char.get(&name).map(|c| ParsedExpr {
                expr: Expression::from(*c),
                bounds,
            })
        }
        "Integer" | "Number" => {
            let value = body.trim().parse::<i64>().ok()?;
            Some(ParsedExpr::exact(Expression::from(value), value))
        }
        "NegativeOne" => Some(ParsedExpr::exact(Expression::from(-1i64), -1)),
        "Zero" => Some(ParsedExpr::exact(Expression::from(0i64), 0)),
        "One" => Some(ParsedExpr::exact(Expression::from(1i64), 1)),
        "Mul" | "Add" | "Min" | "Max" => {
            let parts = split_top_level_args(body);
            if parts.is_empty() {
                return None;
            }
            let mut iter = parts.into_iter();
            let mut acc = parse_sympy_expr_inner(iter.next()?, sym_to_char, ranges)?;
            for part in iter {
                let rhs = parse_sympy_expr_inner(part, sym_to_char, ranges)?;
                acc = match head {
                    "Mul" => ParsedExpr {
                        expr: normalize_mul_expr(acc.expr, rhs.expr),
                        bounds: mul_bounds(acc.bounds, rhs.bounds),
                    },
                    "Add" => ParsedExpr {
                        expr: normalize_add_expr(acc.expr, rhs.expr),
                        bounds: add_bounds(acc.bounds, rhs.bounds),
                    },
                    "Min" => reduce_min(acc, rhs),
                    "Max" => reduce_max(acc, rhs),
                    _ => unreachable!(),
                };
            }
            Some(acc)
        }
        "FloorDiv" => {
            let mut parts = split_top_level_args(body).into_iter();
            let lhs = parse_sympy_expr_inner(parts.next()?, sym_to_char, ranges)?;
            let rhs = parse_sympy_expr_inner(parts.next()?, sym_to_char, ranges)?;
            if parts.next().is_some() {
                return None;
            }
            Some(ParsedExpr {
                expr: lhs.expr / rhs.expr,
                bounds: div_bounds(lhs.bounds, rhs.bounds),
            })
        }
        "Mod" => {
            let mut parts = split_top_level_args(body).into_iter();
            let lhs = parse_sympy_expr_inner(parts.next()?, sym_to_char, ranges)?;
            let rhs = parse_sympy_expr_inner(parts.next()?, sym_to_char, ranges)?;
            if parts.next().is_some() {
                return None;
            }
            Some(ParsedExpr {
                expr: lhs.expr % rhs.expr,
                bounds: mod_bounds(lhs.bounds, rhs.bounds),
            })
        }
        _ => None,
    }
}

fn infer_symbol_bounds(body: &str, range: Option<&RangeConstraint>) -> ExprBounds {
    let mut bounds = ExprBounds::default();
    if body.contains("positive=True") {
        bounds.min = Some(1);
    } else if body.contains("nonnegative=True") {
        bounds.min = Some(0);
    }
    if let Some(range) = range {
        bounds.min = match (bounds.min, range.min_val) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            (None, Some(rhs)) => Some(rhs),
            (lhs, None) => lhs,
        };
        bounds.max = range.max_val;
    }
    bounds
}

fn exact_expr(value: i64) -> BoundedExpr {
    BoundedExpr {
        expr: Expression::from(value),
        bounds: ExprBounds {
            min: Some(value),
            max: Some(value),
        },
    }
}

fn exact_value(expr: BoundedExpr) -> Option<i64> {
    expr.expr.as_num().or({
        (expr.bounds.min == expr.bounds.max)
            .then_some(expr.bounds.min)
            .flatten()
    })
}

fn exact_bound_value(bounds: ExprBounds) -> Option<i64> {
    (bounds.min == bounds.max).then_some(bounds.min).flatten()
}

fn with_bounds(expr: Expression, bounds: ExprBounds) -> BoundedExpr {
    BoundedExpr { expr, bounds }
}

fn bool_bounds() -> ExprBounds {
    ExprBounds {
        min: Some(0),
        max: Some(1),
    }
}

fn normalize_expr(expr: Expression) -> Expression {
    if expr.len() <= 16 {
        expr.simplify()
    } else {
        expr
    }
}

fn normalize_add_expr(lhs: Expression, rhs: Expression) -> Expression {
    normalize_expr(crate::dim_arith::add_dims(lhs, rhs))
}

fn normalize_mul_expr(lhs: Expression, rhs: Expression) -> Expression {
    normalize_expr(crate::dim_arith::mul_dims(lhs, rhs))
}

fn checked_add_opt(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    lhs.zip(rhs).and_then(|(lhs, rhs)| lhs.checked_add(rhs))
}

fn checked_sub_opt(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    lhs.zip(rhs).and_then(|(lhs, rhs)| lhs.checked_sub(rhs))
}

fn checked_mul_opt(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    lhs.zip(rhs).and_then(|(lhs, rhs)| lhs.checked_mul(rhs))
}

fn add_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: checked_add_opt(lhs.min, rhs.min),
        max: checked_add_opt(lhs.max, rhs.max),
    }
}

fn mul_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    if lhs.min.unwrap_or(i64::MIN) >= 0 && rhs.min.unwrap_or(i64::MIN) >= 0 {
        return ExprBounds {
            min: checked_mul_opt(lhs.min, rhs.min),
            max: checked_mul_opt(lhs.max, rhs.max),
        };
    }
    ExprBounds::default()
}

fn sub_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: checked_sub_opt(lhs.min, rhs.max),
        max: checked_sub_opt(lhs.max, rhs.min),
    }
}

fn div_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    let (Some(rhs_min), Some(rhs_max)) = (rhs.min, rhs.max) else {
        return ExprBounds::default();
    };
    if rhs_min <= 0 || rhs_max <= 0 {
        return ExprBounds::default();
    }
    ExprBounds {
        min: lhs.min.and_then(|lhs_min| lhs_min.checked_div(rhs_max)),
        max: lhs.max.and_then(|lhs_max| lhs_max.checked_div(rhs_min)),
    }
}

fn mod_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    if lhs.min.unwrap_or(i64::MIN) < 0 {
        return ExprBounds::default();
    }
    match exact_bound_value(rhs) {
        Some(rhs_exact) if rhs_exact > 0 => ExprBounds {
            min: Some(0),
            max: rhs_exact.checked_sub(1),
        },
        _ => ExprBounds::default(),
    }
}

fn reduce_min(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if lhs.expr == rhs.expr || lhs.expr.egglog_equal(rhs.expr) {
        return ParsedExpr {
            expr: lhs.expr,
            bounds: min_bounds(lhs.bounds, rhs.bounds),
        };
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return lhs;
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return rhs;
    }
    if expr_is_offset_by_small_const(lhs.expr, rhs.expr) {
        return rhs;
    }
    if expr_is_offset_by_small_const(rhs.expr, lhs.expr) {
        return lhs;
    }
    ParsedExpr {
        expr: lhs.expr.min(rhs.expr),
        bounds: min_bounds(lhs.bounds, rhs.bounds),
    }
}

fn reduce_max(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if lhs.expr == rhs.expr || lhs.expr.egglog_equal(rhs.expr) {
        return ParsedExpr {
            expr: lhs.expr,
            bounds: max_bounds(lhs.bounds, rhs.bounds),
        };
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return rhs;
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return lhs;
    }
    if expr_is_offset_by_small_const(lhs.expr, rhs.expr) {
        return lhs;
    }
    if expr_is_offset_by_small_const(rhs.expr, lhs.expr) {
        return rhs;
    }
    ParsedExpr {
        expr: lhs.expr.max(rhs.expr),
        bounds: max_bounds(lhs.bounds, rhs.bounds),
    }
}

fn min_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: match (lhs.min, rhs.min) {
            (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
            _ => None,
        },
        max: match (lhs.max, rhs.max) {
            (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
            _ => None,
        },
    }
}

fn max_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: match (lhs.min, rhs.min) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            _ => None,
        },
        max: match (lhs.max, rhs.max) {
            (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
            _ => None,
        },
    }
}

fn expr_is_offset_by_small_const(lhs: Expression, rhs: Expression) -> bool {
    (1..=8).any(|delta| lhs.egglog_equal(rhs + delta))
}

fn split_add_const(expr: Expression) -> Option<(i64, Expression)> {
    let terms = expr.terms.read();
    if terms.len() >= 3 && terms.last() == Some(&Term::Add) {
        if let Some(Term::Num(n)) = terms.first() {
            return Some((*n, Expression::new(terms[1..terms.len() - 1].to_vec())));
        }
        if let Some(Term::Num(n)) = terms.get(terms.len() - 2) {
            return Some((*n, Expression::new(terms[..terms.len() - 2].to_vec())));
        }
    }
    None
}

fn simplify_add(lhs: BoundedExpr, rhs: BoundedExpr) -> BoundedExpr {
    let expr = match (exact_value(lhs), exact_value(rhs)) {
        (Some(0), _) => rhs.expr,
        (_, Some(0)) => lhs.expr,
        (Some(lhs), Some(rhs)) => Expression::from(lhs + rhs),
        (_, Some(rhs)) => normalize_add_expr(lhs.expr, Expression::from(rhs)),
        (Some(lhs), _) => normalize_add_expr(Expression::from(lhs), rhs.expr),
        _ => normalize_add_expr(lhs.expr, rhs.expr),
    };
    with_bounds(expr, add_bounds(lhs.bounds, rhs.bounds))
}

fn simplify_sub(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    if same_expr_with_ranges(lhs.expr, rhs.expr, sym_ranges) {
        return exact_expr(0);
    }
    let expr = match exact_value(rhs) {
        Some(0) => lhs.expr,
        Some(rhs_const) => {
            if let Some((lhs_const, lhs_base)) = split_add_const(lhs.expr) {
                normalize_expr(lhs_base + (lhs_const - rhs_const))
            } else {
                normalize_expr(lhs.expr - rhs_const)
            }
        }
        None => normalize_expr(lhs.expr - rhs.expr),
    };
    with_bounds(expr, sub_bounds(lhs.bounds, rhs.bounds))
}

fn simplify_min(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    let bounds = min_bounds(lhs.bounds, rhs.bounds);
    if same_expr_with_ranges(lhs.expr, rhs.expr, sym_ranges) {
        return with_bounds(lhs.expr, bounds);
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return with_bounds(lhs.expr, bounds);
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return with_bounds(rhs.expr, bounds);
    }
    if let Some((lhs_const, lhs_base)) = split_add_const(lhs.expr)
        && lhs_const >= 0
        && same_expr_with_ranges(lhs_base, rhs.expr, sym_ranges)
    {
        return with_bounds(rhs.expr, bounds);
    }
    if let Some((rhs_const, rhs_base)) = split_add_const(rhs.expr)
        && rhs_const >= 0
        && same_expr_with_ranges(rhs_base, lhs.expr, sym_ranges)
    {
        return with_bounds(lhs.expr, bounds);
    }
    with_bounds(normalize_expr(lhs.expr.min(rhs.expr)), bounds)
}

fn simplify_max(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    let bounds = max_bounds(lhs.bounds, rhs.bounds);
    if same_expr_with_ranges(lhs.expr, rhs.expr, sym_ranges) {
        return with_bounds(lhs.expr, bounds);
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return with_bounds(rhs.expr, bounds);
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return with_bounds(lhs.expr, bounds);
    }
    if let Some((lhs_const, lhs_base)) = split_add_const(lhs.expr)
        && lhs_const >= 0
        && same_expr_with_ranges(lhs_base, rhs.expr, sym_ranges)
    {
        return with_bounds(lhs.expr, bounds);
    }
    if let Some((rhs_const, rhs_base)) = split_add_const(rhs.expr)
        && rhs_const >= 0
        && same_expr_with_ranges(rhs_base, lhs.expr, sym_ranges)
    {
        return with_bounds(rhs.expr, bounds);
    }
    with_bounds(normalize_expr(lhs.expr.max(rhs.expr)), bounds)
}

fn simplify_bound_expr(expr: Expression, sym_ranges: &FxHashMap<char, ExprBounds>) -> BoundedExpr {
    let mut stack: Vec<BoundedExpr> = Vec::new();
    let terms = expr.terms.read().clone();
    for term in terms {
        match term {
            Term::Num(n) => stack.push(exact_expr(n)),
            Term::Var(c) => stack.push(with_bounds(
                Expression::from(c),
                sym_ranges.get(&c).copied().unwrap_or_default(),
            )),
            Term::Add => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(simplify_add(lhs, rhs));
            }
            Term::Sub => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(simplify_sub(lhs, rhs, sym_ranges));
            }
            Term::Mul => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (exact_value(lhs), exact_value(rhs)) {
                    (Some(0), _) | (_, Some(0)) => Expression::from(0),
                    (Some(1), _) => rhs.expr,
                    (_, Some(1)) => lhs.expr,
                    (Some(lhs), Some(rhs)) => Expression::from(lhs * rhs),
                    _ => normalize_mul_expr(lhs.expr, rhs.expr),
                };
                stack.push(with_bounds(expr, mul_bounds(lhs.bounds, rhs.bounds)));
            }
            Term::Div | Term::CeilDiv => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (term, exact_value(lhs), exact_value(rhs)) {
                    (_, Some(0), _) => Expression::from(0),
                    (_, _, Some(1)) => lhs.expr,
                    (Term::Div, Some(lhs), Some(rhs)) if rhs != 0 => Expression::from(lhs / rhs),
                    (Term::CeilDiv, Some(lhs), Some(rhs)) if rhs > 0 => {
                        Expression::from(if lhs % rhs != 0 {
                            lhs / rhs + 1
                        } else {
                            lhs / rhs
                        })
                    }
                    (Term::Div, _, _) => normalize_expr(lhs.expr / rhs.expr),
                    (Term::CeilDiv, _, _) => normalize_expr(lhs.expr.ceil_div(rhs.expr)),
                    _ => unreachable!(),
                };
                stack.push(with_bounds(expr, div_bounds(lhs.bounds, rhs.bounds)));
            }
            Term::Mod => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (exact_value(lhs), exact_value(rhs)) {
                    (Some(0), _) | (_, Some(1)) => Expression::from(0),
                    (Some(lhs), Some(rhs)) if rhs != 0 => Expression::from(lhs % rhs),
                    _ => normalize_expr(lhs.expr % rhs.expr),
                };
                stack.push(with_bounds(expr, mod_bounds(lhs.bounds, rhs.bounds)));
            }
            Term::Min => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(simplify_min(lhs, rhs, sym_ranges));
            }
            Term::Max => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                stack.push(simplify_max(lhs, rhs, sym_ranges));
            }
            term @ (Term::And | Term::Or | Term::Gte | Term::Lt) => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (term, exact_value(lhs), exact_value(rhs)) {
                    (Term::And, Some(lhs), Some(rhs)) => {
                        Expression::from((lhs != 0 && rhs != 0) as i64)
                    }
                    (Term::And, _, _) => normalize_expr(lhs.expr & rhs.expr),
                    (Term::Or, Some(lhs), Some(rhs)) => {
                        Expression::from((lhs != 0 || rhs != 0) as i64)
                    }
                    (Term::Or, _, _) => normalize_expr(lhs.expr | rhs.expr),
                    (Term::Gte, Some(lhs), Some(rhs)) => Expression::from((lhs >= rhs) as i64),
                    (Term::Gte, _, _) => normalize_expr(lhs.expr.gte(rhs.expr)),
                    (Term::Lt, Some(lhs), Some(rhs)) => Expression::from((lhs < rhs) as i64),
                    (Term::Lt, _, _) => normalize_expr(lhs.expr.lt(rhs.expr)),
                    _ => unreachable!(),
                };
                stack.push(with_bounds(expr, bool_bounds()));
            }
        }
    }
    stack
        .pop()
        .unwrap_or(with_bounds(expr, ExprBounds::default()))
}

/// Split `Head(body)` into `(head, body)`.
fn split_head(expr: &str) -> Option<(&str, &str)> {
    let open = expr.find('(')?;
    if !expr.ends_with(')') {
        return None;
    }
    Some((&expr[..open], &expr[open + 1..expr.len() - 1]))
}

/// Pull out the first single- or double-quoted token from a sympy arg list.
fn extract_first_quoted(expr: &str) -> Option<String> {
    let bytes = expr.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '\'' || c == '"' {
            let quote = c;
            let start = i + 1;
            i += 1;
            while i < bytes.len() && bytes[i] as char != quote {
                i += 1;
            }
            return Some(expr[start..i].to_string());
        }
        i += 1;
    }
    None
}

/// Split a sympy-style argument list at top-level commas, respecting nested
/// parens and quoted strings. Drops `key=value` kwargs.
fn split_top_level_args(expr: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = expr.as_bytes();
    let mut depth = 0;
    let mut in_quote: Option<char> = None;
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        match in_quote {
            Some(q) => {
                if c == q {
                    in_quote = None;
                }
            }
            None => match c {
                '\'' | '"' => in_quote = Some(c),
                '(' | '[' => depth += 1,
                ')' | ']' => depth -= 1,
                ',' if depth == 0 => {
                    let part = expr[start..i].trim();
                    if !part.is_empty() && !looks_like_kwarg(part) {
                        out.push(part);
                    }
                    start = i + 1;
                }
                _ => {}
            },
        }
    }
    let part = expr[start..].trim();
    if !part.is_empty() && !looks_like_kwarg(part) {
        out.push(part);
    }
    out
}

fn looks_like_kwarg(part: &str) -> bool {
    if let Some(eq) = part.find('=') {
        let key = part[..eq].trim();
        return !key.is_empty() && key.chars().all(|c| c == '_' || c.is_ascii_alphanumeric());
    }
    false
}
