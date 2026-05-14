use std::collections::HashMap;

use luminal::prelude::*;

use crate::pt2_schema::RangeConstraint;

#[derive(Clone, Copy, Debug, Default)]
struct ExprBounds {
    min: Option<i64>,
    max: Option<i64>,
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
                        expr: acc.expr * rhs.expr,
                        bounds: mul_bounds(acc.bounds, rhs.bounds),
                    },
                    "Add" => ParsedExpr {
                        expr: acc.expr + rhs.expr,
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
                bounds: floordiv_bounds(lhs.bounds, rhs.bounds),
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

fn checked_add_opt(lhs: Option<i64>, rhs: Option<i64>) -> Option<i64> {
    lhs.zip(rhs).and_then(|(lhs, rhs)| lhs.checked_add(rhs))
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
    if lhs.min.unwrap_or(0) >= 0 && rhs.min.unwrap_or(0) >= 0 {
        return ExprBounds {
            min: checked_mul_opt(lhs.min, rhs.min),
            max: checked_mul_opt(lhs.max, rhs.max),
        };
    }
    ExprBounds::default()
}

fn floordiv_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
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

fn mod_bounds(_lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    match (rhs.min, rhs.max) {
        (Some(rhs_min), Some(rhs_max)) if rhs_min == rhs_max && rhs_max > 0 => ExprBounds {
            min: Some(0),
            max: rhs_max.checked_sub(1),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sym_map() -> HashMap<String, char> {
        HashMap::from([("s40".to_string(), 'a'), ("s77".to_string(), 'b')])
    }

    #[test]
    fn parses_nested_add_and_mul() {
        let expr = parse_sympy_expr(
            "Add(Symbol('s40', positive=True, integer=True), Mul(Integer(2), Symbol('s77', positive=True, integer=True)))",
            &sym_map(),
        )
        .unwrap();
        assert_eq!(
            expr,
            Expression::from('a') + (Expression::from(2i64) * Expression::from('b'))
        );
    }

    #[test]
    fn parses_negative_and_variadic_heads() {
        assert_eq!(
            parse_sympy_expr("Integer(-1)", &sym_map()).unwrap(),
            Expression::from(-1i64)
        );
        assert_eq!(
            parse_sympy_expr(
                "Max(Symbol('s40', positive=True, integer=True), Integer(16))",
                &sym_map()
            )
            .unwrap(),
            Expression::from('a').max(Expression::from(16i64))
        );
    }

    #[test]
    fn ignores_symbol_kwargs() {
        assert_eq!(
            parse_sympy_expr("Symbol('s40', positive=True, integer=True)", &sym_map()).unwrap(),
            Expression::from('a')
        );
    }

    #[test]
    fn folds_min_with_positive_symbol_to_constant() {
        assert_eq!(
            parse_sympy_expr(
                "Min(Symbol('s40', positive=True, integer=True), Integer(1))",
                &sym_map()
            )
            .unwrap(),
            Expression::from(1i64)
        );
    }

    #[test]
    fn folds_min_of_symbol_and_symbol_plus_one() {
        assert_eq!(
            parse_sympy_expr(
                "Min(Add(Symbol('s40', positive=True, integer=True), Integer(1)), Symbol('s40', positive=True, integer=True))",
                &sym_map()
            )
            .unwrap(),
            Expression::from('a')
        );
    }
}
