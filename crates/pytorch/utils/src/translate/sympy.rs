#![allow(dead_code)] // wired incrementally with the symbolic translator port
use std::collections::HashMap;

use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_parser::SymDimMap;
use crate::pt2_schema::RangeConstraint;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ExprBounds {
    pub(super) min: Option<i64>,
    pub(super) max: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
struct ParsedExpr {
    expr: IntExpr,
    bounds: ExprBounds,
}

impl ParsedExpr {
    fn exact(expr: IntExpr, value: i64) -> Self {
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
    expr: IntExpr,
    bounds: ExprBounds,
}

/// Parse a sympy `srepr`-style expression string into a luminal `IntExpr`.
///
/// Supports the subset of sympy heads PT2 emits for symbolic shape metadata
/// and symbolic scalar values: integer arithmetic (`Add`, `Mul`, `Pow` by a
/// literal natural, `Min`, `Max`), torch's division family with torch's
/// rounding (`FloorDiv`, `CleanDiv`, `Mod`, `PythonMod`), and comparisons
/// and boolean connectives, which read as the integer 1 or 0.
pub(super) fn parse_sympy_expr(
    expr: &str,
    sym_to_symbol: &HashMap<String, Symbol>,
) -> Option<IntExpr> {
    parse_sympy_expr_with_ranges(expr, sym_to_symbol, &HashMap::new())
}

pub(super) fn parse_sympy_expr_with_ranges(
    expr: &str,
    sym_to_symbol: &HashMap<String, Symbol>,
    ranges: &HashMap<String, RangeConstraint>,
) -> Option<IntExpr> {
    parse_sympy_expr_inner(expr, sym_to_symbol, ranges).map(|parsed| parsed.expr)
}

pub(super) fn sym_char_ranges(sym_map: &SymDimMap) -> FxHashMap<Symbol, ExprBounds> {
    symbol_bounds(&sym_map.sym_to_symbol, &sym_map.ranges)
}

/// The bounds of every dimension symbol: torch's exported range, clamped
/// to the nonnegative values a recorder dimension symbol can hold.
pub(super) fn symbol_bounds(
    symbols: &HashMap<String, Symbol>,
    ranges: &HashMap<String, RangeConstraint>,
) -> FxHashMap<Symbol, ExprBounds> {
    symbols
        .iter()
        .map(|(sym_name, sym_char)| {
            let range = ranges.get(sym_name);
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

pub(super) fn simplify_expr_with_ranges(
    expr: IntExpr,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) -> IntExpr {
    simplify_bound_expr(expr, sym_ranges).expr
}

pub(super) fn bounds_of_expr(
    expr: IntExpr,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) -> ExprBounds {
    simplify_bound_expr(expr, sym_ranges).bounds
}

pub(super) fn same_expr_with_ranges(
    lhs: IntExpr,
    rhs: IntExpr,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) -> bool {
    let lhs = simplify_bound_expr(lhs, sym_ranges);
    let rhs = simplify_bound_expr(rhs, sym_ranges);
    lhs.expr == rhs.expr
        || lhs.expr.egglog_equal(rhs.expr)
        || (exact_value(lhs) == exact_value(rhs) && exact_value(lhs).is_some())
}

pub(super) fn canonical_equal_expr(
    lhs: IntExpr,
    rhs: IntExpr,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
) -> Option<IntExpr> {
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
    sym_to_symbol: &HashMap<String, Symbol>,
    ranges: &HashMap<String, RangeConstraint>,
) -> Option<ParsedExpr> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    if let Ok(value) = expr.parse::<i64>() {
        return Some(ParsedExpr::exact(IntExpr::from(value), value));
    }
    // sympy's boolean constants print bare.
    match expr {
        "true" => return Some(truth(true)),
        "false" => return Some(truth(false)),
        _ => {}
    }

    let (head, body) = split_head(expr)?;
    match head {
        "Symbol" => {
            let name = extract_first_quoted(body)?;
            let bounds = infer_symbol_bounds(body, ranges.get(&name));
            sym_to_symbol.get(&name).map(|c| ParsedExpr {
                expr: IntExpr::from(*c),
                bounds,
            })
        }
        "Integer" | "Number" => {
            let value = body.trim().parse::<i64>().ok()?;
            Some(ParsedExpr::exact(IntExpr::from(value), value))
        }
        "NegativeOne" => Some(ParsedExpr::exact(IntExpr::from(-1i64), -1)),
        "Zero" => Some(ParsedExpr::exact(IntExpr::from(0i64), 0)),
        "One" => Some(ParsedExpr::exact(IntExpr::from(1i64), 1)),
        "Mul" | "Add" | "Min" | "Max" => {
            let parts = split_top_level_args(body);
            if parts.is_empty() {
                return None;
            }
            let mut iter = parts.into_iter();
            let mut acc = parse_sympy_expr_inner(iter.next()?, sym_to_symbol, ranges)?;
            for part in iter {
                let rhs = parse_sympy_expr_inner(part, sym_to_symbol, ranges)?;
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
        // sympy spells `n * n` as `Pow(n, 2)`. A literal natural exponent is
        // repeated multiplication; any other power (`Pow(n, -1)` is sympy's
        // `1 / n`) is not an integer expression and stays unread.
        "Pow" | "PowByNatural" => {
            let (base, exponent) = parse_two_args(body, sym_to_symbol, ranges)?;
            let exponent = exponent.expr.as_num()?;
            if !(0..=MAX_NATURAL_POWER).contains(&exponent) {
                return None;
            }
            if exponent == 0 {
                return Some(ParsedExpr::exact(IntExpr::from(1i64), 1));
            }
            let mut acc = base;
            for _ in 1..exponent {
                acc = ParsedExpr {
                    expr: normalize_mul_expr(acc.expr, base.expr),
                    bounds: mul_bounds(acc.bounds, base.bounds),
                };
            }
            Some(acc)
        }
        "FloorDiv" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(floor_div(lhs, rhs))
        }
        // torch's `CleanDiv` is a `FloorDiv` it has proved exact, and an
        // exact quotient is the same under every rounding rule.
        "CleanDiv" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(ParsedExpr {
                expr: lhs.expr / rhs.expr,
                bounds: div_bounds(lhs.bounds, rhs.bounds),
            })
        }
        // torch's `Mod` is defined on nonnegative operands only (it prints
        // `PythonMod` for anything else), where it is the recorder's `%`.
        "Mod" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(ParsedExpr {
                expr: lhs.expr % rhs.expr,
                bounds: mod_bounds(lhs.bounds, rhs.bounds),
            })
        }
        "PythonMod" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(python_mod(lhs, rhs))
        }
        // A symbolic bool is the integer 1 or 0, the value torch gives it
        // wherever a program uses one as a number.
        "StrictLessThan" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(less_than(lhs, rhs))
        }
        "StrictGreaterThan" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(less_than(rhs, lhs))
        }
        "GreaterThan" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(at_least(lhs, rhs))
        }
        "LessThan" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(at_least(rhs, lhs))
        }
        "Equality" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(both(at_least(lhs, rhs), at_least(rhs, lhs)))
        }
        "Unequality" => {
            let (lhs, rhs) = parse_two_args(body, sym_to_symbol, ranges)?;
            Some(either(less_than(lhs, rhs), less_than(rhs, lhs)))
        }
        "And" | "Or" => {
            let mut parts = split_top_level_args(body).into_iter();
            let mut acc = parse_sympy_expr_inner(parts.next()?, sym_to_symbol, ranges)?;
            for part in parts {
                let rhs = parse_sympy_expr_inner(part, sym_to_symbol, ranges)?;
                acc = if head == "And" {
                    both(acc, rhs)
                } else {
                    either(acc, rhs)
                };
            }
            Some(acc)
        }
        "Not" => {
            let mut parts = split_top_level_args(body).into_iter();
            let operand = parse_sympy_expr_inner(parts.next()?, sym_to_symbol, ranges)?;
            if parts.next().is_some() {
                return None;
            }
            Some(negation(operand))
        }
        _ => None,
    }
}

/// The largest literal exponent expanded into repeated multiplication.
const MAX_NATURAL_POWER: i64 = 16;

/// The two operands of a binary sympy head; any other arity is unread.
fn parse_two_args(
    body: &str,
    sym_to_symbol: &HashMap<String, Symbol>,
    ranges: &HashMap<String, RangeConstraint>,
) -> Option<(ParsedExpr, ParsedExpr)> {
    let mut parts = split_top_level_args(body).into_iter();
    let lhs = parse_sympy_expr_inner(parts.next()?, sym_to_symbol, ranges)?;
    let rhs = parse_sympy_expr_inner(parts.next()?, sym_to_symbol, ranges)?;
    if parts.next().is_some() {
        return None;
    }
    Some((lhs, rhs))
}

fn is_nonnegative(bounds: ExprBounds) -> bool {
    bounds.min.is_some_and(|min| min >= 0)
}

fn is_positive(bounds: ExprBounds) -> bool {
    bounds.min.is_some_and(|min| min >= 1)
}

/// The largest shift `shifted_nonnegative` adds to a dividend. A shift is a
/// distance below zero that a dimension-sized value can reach; a lower
/// bound further out than this is no bound to shift by, and adding its
/// magnitude would only move the overflow to the other end.
const MAX_SHIFT: i64 = i32::MAX as i64;

/// `lhs + k * rhs` for the least `k` that makes it nonnegative, when the
/// bounds give one: `lhs` has a finite lower bound and `rhs` is one positive
/// number. Adding a multiple of the divisor moves a floor quotient by `k`
/// and leaves the Python remainder alone, so both can then be read with the
/// recorder's truncating `/` and `%`, whose operands stay nonnegative as
/// the recorder's expression algebra assumes.
fn shifted_nonnegative(lhs: ParsedExpr, rhs: ParsedExpr) -> Option<(ParsedExpr, i64)> {
    let divisor = exact_bound_value(rhs.bounds).filter(|divisor| *divisor > 0)?;
    let deficit = lhs.bounds.min?.checked_neg()?;
    let k = deficit.checked_add(divisor - 1)? / divisor;
    let shift = k.checked_mul(divisor).filter(|shift| *shift <= MAX_SHIFT)?;
    let shifted = ParsedExpr {
        expr: normalize_add_expr(lhs.expr, IntExpr::from(shift)),
        bounds: add_bounds(
            lhs.bounds,
            ExprBounds {
                min: Some(shift),
                max: Some(shift),
            },
        ),
    };
    Some((shifted, k))
}

/// Python's `%` for operands of any sign, in the recorder's truncating
/// `%`: `((a % b) + b) % b`. The inner remainder lies strictly between
/// `-|b|` and `|b|`; adding `b` gives it `b`'s sign or zero.
fn python_mod_expr(lhs: IntExpr, rhs: IntExpr) -> IntExpr {
    ((lhs % rhs) + rhs) % rhs
}

/// torch's `PythonMod`: the remainder takes the divisor's sign. The
/// recorder's `%` truncates, which agrees only on a nonnegative dividend
/// over a positive divisor.
fn python_mod(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if is_nonnegative(lhs.bounds) && is_positive(rhs.bounds) {
        return ParsedExpr {
            expr: lhs.expr % rhs.expr,
            bounds: mod_bounds(lhs.bounds, rhs.bounds),
        };
    }
    if let Some((shifted, _)) = shifted_nonnegative(lhs, rhs) {
        return ParsedExpr {
            expr: shifted.expr % rhs.expr,
            bounds: mod_bounds(shifted.bounds, rhs.bounds),
        };
    }
    let bounds = match exact_bound_value(rhs.bounds) {
        Some(divisor) if divisor > 0 => ExprBounds {
            min: Some(0),
            max: Some(divisor - 1),
        },
        _ => ExprBounds::default(),
    };
    ParsedExpr {
        expr: python_mod_expr(lhs.expr, rhs.expr),
        bounds,
    }
}

/// torch's `FloorDiv` rounds toward negative infinity. The recorder's `/`
/// truncates, which agrees only on a nonnegative dividend over a positive
/// divisor; that case keeps the plain spelling. Otherwise the quotient is
/// read off a dividend shifted nonnegative, or, with no bound to shift by,
/// as the exact division of `a - PythonMod(a, b)`.
fn floor_div(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if is_nonnegative(lhs.bounds) && is_positive(rhs.bounds) {
        return ParsedExpr {
            expr: lhs.expr / rhs.expr,
            bounds: div_bounds(lhs.bounds, rhs.bounds),
        };
    }
    if let Some((shifted, k)) = shifted_nonnegative(lhs, rhs) {
        let quotient = div_bounds(shifted.bounds, rhs.bounds);
        return ParsedExpr {
            expr: normalize_expr(shifted.expr / rhs.expr - k),
            bounds: sub_bounds(
                quotient,
                ExprBounds {
                    min: Some(k),
                    max: Some(k),
                },
            ),
        };
    }
    ParsedExpr {
        expr: (lhs.expr - python_mod_expr(lhs.expr, rhs.expr)) / rhs.expr,
        bounds: ExprBounds::default(),
    }
}

fn truth(value: bool) -> ParsedExpr {
    ParsedExpr::exact(IntExpr::from(value as i64), value as i64)
}

/// `lhs < rhs` as 1 or 0, decided here when the bounds decide it.
fn less_than(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max < rhs_min
    {
        return truth(true);
    }
    if let (Some(lhs_min), Some(rhs_max)) = (lhs.bounds.min, rhs.bounds.max)
        && lhs_min >= rhs_max
    {
        return truth(false);
    }
    ParsedExpr {
        expr: lhs.expr.lt(rhs.expr),
        bounds: bool_bounds(),
    }
}

/// `lhs >= rhs` as 1 or 0, decided here when the bounds decide it.
fn at_least(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    if let (Some(lhs_min), Some(rhs_max)) = (lhs.bounds.min, rhs.bounds.max)
        && lhs_min >= rhs_max
    {
        return truth(true);
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max < rhs_min
    {
        return truth(false);
    }
    ParsedExpr {
        expr: lhs.expr.gte(rhs.expr),
        bounds: bool_bounds(),
    }
}

/// Conjunction of two 1-or-0 values: their product.
fn both(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    match (exact_bound_value(lhs.bounds), exact_bound_value(rhs.bounds)) {
        (Some(0), _) | (_, Some(0)) => truth(false),
        (Some(_), _) => rhs,
        (_, Some(_)) => lhs,
        _ => ParsedExpr {
            expr: lhs.expr * rhs.expr,
            bounds: bool_bounds(),
        },
    }
}

/// Disjunction of two 1-or-0 values: their maximum.
fn either(lhs: ParsedExpr, rhs: ParsedExpr) -> ParsedExpr {
    match (exact_bound_value(lhs.bounds), exact_bound_value(rhs.bounds)) {
        (Some(0), _) => rhs,
        (_, Some(0)) => lhs,
        (Some(_), _) | (_, Some(_)) => truth(true),
        _ => ParsedExpr {
            expr: lhs.expr.max(rhs.expr),
            bounds: bool_bounds(),
        },
    }
}

/// Negation of a 1-or-0 value.
fn negation(operand: ParsedExpr) -> ParsedExpr {
    match exact_bound_value(operand.bounds) {
        Some(value) => truth(value == 0),
        None => ParsedExpr {
            expr: IntExpr::from(1i64) - operand.expr,
            bounds: bool_bounds(),
        },
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
        expr: IntExpr::from(value),
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

fn with_bounds(expr: IntExpr, bounds: ExprBounds) -> BoundedExpr {
    BoundedExpr { expr, bounds }
}

fn bool_bounds() -> ExprBounds {
    ExprBounds {
        min: Some(0),
        max: Some(1),
    }
}

fn normalize_expr(expr: IntExpr) -> IntExpr {
    if expr.len() <= 16 {
        expr.simplify()
    } else {
        expr
    }
}

fn normalize_add_expr(lhs: IntExpr, rhs: IntExpr) -> IntExpr {
    normalize_expr(super::dim_arith::add_dims(lhs, rhs))
}

fn normalize_mul_expr(lhs: IntExpr, rhs: IntExpr) -> IntExpr {
    normalize_expr(super::dim_arith::mul_dims(lhs, rhs))
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

fn expr_is_offset_by_small_const(lhs: IntExpr, rhs: IntExpr) -> bool {
    (1..=8).any(|delta| lhs.egglog_equal(rhs + delta))
}

fn split_add_const(expr: IntExpr) -> Option<(i64, IntExpr)> {
    let terms = expr.terms.read();
    if terms.len() >= 3 && terms.last() == Some(&Term::Add) {
        if let Some(Term::Num(n)) = terms.first() {
            return Some((*n, IntExpr::new(terms[1..terms.len() - 1].to_vec())));
        }
        if let Some(Term::Num(n)) = terms.get(terms.len() - 2) {
            return Some((*n, IntExpr::new(terms[..terms.len() - 2].to_vec())));
        }
    }
    None
}

fn simplify_add(lhs: BoundedExpr, rhs: BoundedExpr) -> BoundedExpr {
    let expr = match (exact_value(lhs), exact_value(rhs)) {
        (Some(0), _) => rhs.expr,
        (_, Some(0)) => lhs.expr,
        (Some(lhs), Some(rhs)) => IntExpr::from(lhs + rhs),
        (_, Some(rhs)) => normalize_add_expr(lhs.expr, IntExpr::from(rhs)),
        (Some(lhs), _) => normalize_add_expr(IntExpr::from(lhs), rhs.expr),
        _ => normalize_add_expr(lhs.expr, rhs.expr),
    };
    with_bounds(expr, add_bounds(lhs.bounds, rhs.bounds))
}

fn simplify_sub(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
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
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
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
    sym_ranges: &FxHashMap<Symbol, ExprBounds>,
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

fn simplify_bound_expr(expr: IntExpr, sym_ranges: &FxHashMap<Symbol, ExprBounds>) -> BoundedExpr {
    let mut stack: Vec<BoundedExpr> = Vec::new();
    let terms = expr.terms.read().clone();
    for term in terms {
        match term {
            Term::Num(n) => stack.push(exact_expr(n)),
            Term::Var(c) => stack.push(with_bounds(
                IntExpr::from(c),
                sym_ranges.get(&c).copied().unwrap_or_default(),
            )),
            // Structural iota coordinates are only meaningful inside an
            // `iota` value expression; treat one appearing in a dim as an
            // opaque atom with unknown bounds.
            Term::Coord(c) => stack.push(with_bounds(
                IntExpr::new(vec![Term::Coord(c)]),
                ExprBounds::default(),
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
                    (Some(0), _) | (_, Some(0)) => IntExpr::from(0),
                    (Some(1), _) => rhs.expr,
                    (_, Some(1)) => lhs.expr,
                    (Some(lhs), Some(rhs)) => IntExpr::from(lhs * rhs),
                    _ => normalize_mul_expr(lhs.expr, rhs.expr),
                };
                stack.push(with_bounds(expr, mul_bounds(lhs.bounds, rhs.bounds)));
            }
            Term::Div | Term::CeilDiv => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (term, exact_value(lhs), exact_value(rhs)) {
                    (_, Some(0), _) => IntExpr::from(0),
                    (_, _, Some(1)) => lhs.expr,
                    (Term::Div, Some(lhs), Some(rhs)) if rhs != 0 => IntExpr::from(lhs / rhs),
                    (Term::CeilDiv, Some(lhs), Some(rhs)) if rhs > 0 => {
                        IntExpr::from(if lhs % rhs != 0 {
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
                    (Some(0), _) | (_, Some(1)) => IntExpr::from(0),
                    (Some(lhs), Some(rhs)) if rhs != 0 => IntExpr::from(lhs % rhs),
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
                        IntExpr::from((lhs != 0 && rhs != 0) as i64)
                    }
                    (Term::And, _, _) => normalize_expr(lhs.expr & rhs.expr),
                    (Term::Or, Some(lhs), Some(rhs)) => {
                        IntExpr::from((lhs != 0 || rhs != 0) as i64)
                    }
                    (Term::Or, _, _) => normalize_expr(lhs.expr | rhs.expr),
                    (Term::Gte, Some(lhs), Some(rhs)) => IntExpr::from((lhs >= rhs) as i64),
                    (Term::Gte, _, _) => normalize_expr(lhs.expr.gte(rhs.expr)),
                    (Term::Lt, Some(lhs), Some(rhs)) => IntExpr::from((lhs < rhs) as i64),
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
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn sym_map() -> HashMap<String, Symbol> {
        let mut map = HashMap::new();
        map.insert("s0".to_string(), Symbol::try_new_dim("a").unwrap());
        map.insert("s1".to_string(), Symbol::try_new_dim("b").unwrap());
        map
    }

    #[test]
    fn compound_add_parses_to_two_vars() {
        let expr = parse_sympy_expr_with_ranges(
            "Add(Symbol('s0', integer=True), Symbol('s1', integer=True))",
            &sym_map(),
            &HashMap::new(),
        )
        .expect("compound add should parse");
        let a = IntExpr::from(Symbol::try_new_dim("a").unwrap());
        let b = IntExpr::from(Symbol::try_new_dim("b").unwrap());
        assert_eq!(expr, a + b);
    }

    #[test]
    fn mul_by_integer_parses() {
        let expr = parse_sympy_expr_with_ranges(
            "Mul(Integer(8), Symbol('s0', integer=True))",
            &sym_map(),
            &HashMap::new(),
        )
        .expect("scaled symbol should parse");
        let a = IntExpr::from(Symbol::try_new_dim("a").unwrap());
        assert_eq!(expr, IntExpr::from(8i64) * a);
    }

    fn parse(expr: &str) -> IntExpr {
        parse_sympy_expr_with_ranges(expr, &sym_map(), &HashMap::new())
            .unwrap_or_else(|| panic!("{expr} should parse"))
    }

    /// The expression's value with `s0` bound to `value`.
    fn eval(expr: IntExpr, value: usize) -> i64 {
        let mut dims = FxHashMap::default();
        dims.insert(Symbol::try_new_dim("a").unwrap(), value);
        expr.exec(&dims).expect("expression should evaluate") as i64
    }

    const S0: &str = "Symbol('s0', positive=True, integer=True)";

    #[test]
    fn floordiv_of_a_nonnegative_dividend_keeps_the_plain_spelling() {
        let expr = parse(&format!("FloorDiv({S0}, Integer(4))"));
        let a = IntExpr::from(Symbol::try_new_dim("a").unwrap());
        assert_eq!(expr, a / IntExpr::from(4i64));
    }

    #[test]
    fn floordiv_rounds_toward_negative_infinity() {
        // Bounded below (`s0 - 3 >= -2`), unbounded below (`-s0`), and a
        // symbol with no stated sign: each must floor, not truncate.
        for (text, dividend) in [
            (
                format!("FloorDiv(Add({S0}, Integer(-3)), Integer(2))"),
                (|a| a - 3) as fn(i64) -> i64,
            ),
            (
                format!("FloorDiv(Mul(Integer(-1), {S0}), Integer(2))"),
                |a| -a,
            ),
            (
                "FloorDiv(Add(Symbol('s0', integer=True), Integer(-5)), Integer(2))".to_string(),
                |a| a - 5,
            ),
        ] {
            let expr = parse(&text);
            for a in 1..=9i64 {
                assert_eq!(
                    eval(expr, a as usize),
                    dividend(a).div_euclid(2),
                    "{text} at s0 = {a}"
                );
            }
        }
    }

    #[test]
    fn a_lower_bound_too_far_out_is_not_shifted_by() {
        // A symbol whose declared range reaches far below zero: shifting the
        // dividend up by that distance would overflow on ordinary values,
        // so the quotient and remainder are corrected in place instead.
        let ranges = HashMap::from([(
            "s0".to_string(),
            RangeConstraint {
                min_val: Some(-(1i64 << 62)),
                max_val: None,
            },
        )]);
        let parse = |text: &str| {
            parse_sympy_expr_with_ranges(text, &sym_map(), &ranges).expect("should parse")
        };
        let quotient = parse("FloorDiv(Add(Symbol('s0', integer=True), Integer(-5)), Integer(2))");
        let remainder =
            parse("PythonMod(Add(Symbol('s0', integer=True), Integer(-5)), Integer(4))");
        for a in 0..=9i64 {
            assert_eq!(eval(quotient, a as usize), (a - 5).div_euclid(2));
            assert_eq!(eval(remainder, a as usize), (a - 5).rem_euclid(4));
        }
    }

    #[test]
    fn clean_div_is_exact_division() {
        let expr = parse(&format!("CleanDiv(Mul(Integer(6), {S0}), Integer(3))"));
        for a in 1..=5 {
            assert_eq!(eval(expr, a), 2 * a as i64);
        }
    }

    #[test]
    fn python_mod_takes_the_sign_of_the_divisor() {
        for (text, dividend) in [
            (
                format!("PythonMod(Add({S0}, Integer(-7)), Integer(4))"),
                (|a| a - 7) as fn(i64) -> i64,
            ),
            (
                format!("PythonMod(Add(Integer(4), Mul(Integer(-1), {S0})), Integer(4))"),
                |a| 4 - a,
            ),
            (format!("PythonMod({S0}, Integer(4))"), |a| a),
        ] {
            let expr = parse(&text);
            for a in 1..=12i64 {
                assert_eq!(
                    eval(expr, a as usize),
                    dividend(a).rem_euclid(4),
                    "{text} at s0 = {a}"
                );
            }
        }
        // A nonnegative dividend needs no correction.
        let a = IntExpr::from(Symbol::try_new_dim("a").unwrap());
        assert_eq!(
            parse(&format!("PythonMod({S0}, Integer(4))")),
            a % IntExpr::from(4i64)
        );
    }

    #[test]
    fn natural_power_is_repeated_multiplication() {
        let expr = parse(&format!("Pow({S0}, Integer(3))"));
        for a in 1..=5 {
            assert_eq!(eval(expr, a), (a * a * a) as i64);
        }
        assert_eq!(
            parse(&format!("Pow({S0}, Integer(0))")),
            IntExpr::from(1i64)
        );
        // `1 / n` is not an integer expression.
        assert!(
            parse_sympy_expr_with_ranges(
                &format!("Pow({S0}, Integer(-1))"),
                &sym_map(),
                &HashMap::new()
            )
            .is_none()
        );
    }

    #[test]
    fn comparisons_read_as_one_or_zero() {
        for (head, holds) in [
            ("StrictGreaterThan", (|a| a > 3) as fn(i64) -> bool),
            ("GreaterThan", |a| a >= 3),
            ("StrictLessThan", |a| a < 3),
            ("LessThan", |a| a <= 3),
            ("Equality", |a| a == 3),
            ("Unequality", |a| a != 3),
        ] {
            let expr = parse(&format!("{head}({S0}, Integer(3))"));
            for a in 1..=6i64 {
                assert_eq!(eval(expr, a as usize), holds(a) as i64, "{head} at {a}");
            }
        }
    }

    #[test]
    fn boolean_connectives_combine_comparisons() {
        let inside = parse(&format!(
            "And(StrictGreaterThan({S0}, Integer(2)), StrictLessThan({S0}, Integer(5)))"
        ));
        let outside = parse(&format!(
            "Or(LessThan({S0}, Integer(2)), GreaterThan({S0}, Integer(5)))"
        ));
        let not_three = parse(&format!("Not(Equality({S0}, Integer(3)))"));
        for a in 1..=6i64 {
            assert_eq!(eval(inside, a as usize), (a > 2 && a < 5) as i64);
            assert_eq!(eval(outside, a as usize), (a <= 2 || a >= 5) as i64);
            assert_eq!(eval(not_three, a as usize), (a != 3) as i64);
        }
        assert_eq!(parse("true"), IntExpr::from(1i64));
        assert_eq!(parse("false"), IntExpr::from(0i64));
    }

    #[test]
    fn a_comparison_the_range_decides_is_a_constant() {
        let ranges = HashMap::from([(
            "s0".to_string(),
            RangeConstraint {
                min_val: Some(4),
                max_val: Some(8),
            },
        )]);
        let parse = |text: &str| {
            parse_sympy_expr_with_ranges(text, &sym_map(), &ranges).expect("should parse")
        };
        assert_eq!(
            parse("StrictGreaterThan(Symbol('s0', integer=True), Integer(3))"),
            IntExpr::from(1i64)
        );
        assert_eq!(
            parse("StrictLessThan(Symbol('s0', integer=True), Integer(4))"),
            IntExpr::from(0i64)
        );
    }

    #[test]
    fn range_constraints_feed_the_bounds() {
        // A bounded symbol still parses; the bounds are used by the
        // simplifier (parsing must not drop the expression).
        let ranges = HashMap::from([(
            "s0".to_string(),
            RangeConstraint {
                min_val: Some(1),
                max_val: Some(16),
            },
        )]);
        let expr = parse_sympy_expr_with_ranges("Symbol('s0', integer=True)", &sym_map(), &ranges)
            .expect("bounded symbol should parse");
        assert_eq!(expr, IntExpr::from(Symbol::try_new_dim("a").unwrap()));
    }
}
