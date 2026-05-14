use anyhow::Result;
use luminal::prelude::*;
use rustc_hash::FxHashMap;

use crate::pt2_parser::SymDimMap;
use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

#[derive(Clone, Copy, Debug, Default)]
struct ExprBounds {
    min: Option<i64>,
    max: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
struct BoundedExpr {
    expr: Expression,
    bounds: ExprBounds,
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
    if expr.len() <= 16 { expr.simplify() } else { expr }
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

fn add_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: checked_add_opt(lhs.min, rhs.min),
        max: checked_add_opt(lhs.max, rhs.max),
    }
}

fn sub_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    ExprBounds {
        min: checked_sub_opt(lhs.min, rhs.max),
        max: checked_sub_opt(lhs.max, rhs.min),
    }
}

fn mul_bounds(lhs: ExprBounds, rhs: ExprBounds) -> ExprBounds {
    if lhs.min.unwrap_or(i64::MIN) >= 0 && rhs.min.unwrap_or(i64::MIN) >= 0 {
        ExprBounds {
            min: checked_mul_opt(lhs.min, rhs.min),
            max: checked_mul_opt(lhs.max, rhs.max),
        }
    } else {
        ExprBounds::default()
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

fn sym_char_ranges(sym_map: &SymDimMap) -> FxHashMap<char, ExprBounds> {
    sym_map
        .sym_to_char
        .iter()
        .map(|(sym_name, sym_char)| {
            let range = sym_map.ranges.get(sym_name);
            let min = range
                .and_then(|range| range.min_val)
                .map(|min| min.max(0))
                .or(Some(0));
            let max = range.and_then(|range| range.max_val).filter(|max| *max >= 0);
            (*sym_char, ExprBounds { min, max })
        })
        .collect()
}

fn split_add_const(expr: Expression) -> Option<(i64, Expression)> {
    let terms = expr.terms.read();
    if terms.len() >= 3
        && terms.last() == Some(&Term::Add)
        && let Some(Term::Num(n)) = terms.first()
    {
        return Some((*n, Expression::new(terms[1..terms.len() - 1].to_vec())));
    }
    None
}

fn same_expr(lhs: Expression, rhs: Expression, sym_ranges: &FxHashMap<char, ExprBounds>) -> bool {
    let lhs = simplify_expr_with_bounds(lhs, sym_ranges);
    let rhs = simplify_expr_with_bounds(rhs, sym_ranges);
    lhs.expr == rhs.expr
        || lhs.expr.egglog_equal(rhs.expr)
        || (exact_value(lhs) == exact_value(rhs) && exact_value(lhs).is_some())
}

fn simplify_add(lhs: BoundedExpr, rhs: BoundedExpr) -> BoundedExpr {
    let expr = match (exact_value(lhs), exact_value(rhs)) {
        (Some(0), _) => rhs.expr,
        (_, Some(0)) => lhs.expr,
        (Some(lhs), Some(rhs)) => Expression::from(lhs + rhs),
        (_, Some(rhs)) => normalize_expr(lhs.expr + rhs),
        (Some(lhs), _) => normalize_expr(rhs.expr + lhs),
        _ => normalize_expr(lhs.expr + rhs.expr),
    };
    BoundedExpr {
        expr,
        bounds: add_bounds(lhs.bounds, rhs.bounds),
    }
}

fn simplify_sub(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    if same_expr(lhs.expr, rhs.expr, sym_ranges) {
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
    BoundedExpr {
        expr,
        bounds: sub_bounds(lhs.bounds, rhs.bounds),
    }
}

fn simplify_min(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    let bounds = min_bounds(lhs.bounds, rhs.bounds);
    if same_expr(lhs.expr, rhs.expr, sym_ranges) {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return BoundedExpr {
            expr: rhs.expr,
            bounds,
        };
    }
    if let Some((lhs_const, lhs_base)) = split_add_const(lhs.expr)
        && lhs_const >= 0
        && same_expr(lhs_base, rhs.expr, sym_ranges)
    {
        return BoundedExpr {
            expr: rhs.expr,
            bounds,
        };
    }
    if let Some((rhs_const, rhs_base)) = split_add_const(rhs.expr)
        && rhs_const >= 0
        && same_expr(rhs_base, lhs.expr, sym_ranges)
    {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    BoundedExpr {
        expr: normalize_expr(lhs.expr.min(rhs.expr)),
        bounds,
    }
}

fn simplify_max(
    lhs: BoundedExpr,
    rhs: BoundedExpr,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
    let bounds = max_bounds(lhs.bounds, rhs.bounds);
    if same_expr(lhs.expr, rhs.expr, sym_ranges) {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    if let (Some(lhs_max), Some(rhs_min)) = (lhs.bounds.max, rhs.bounds.min)
        && lhs_max <= rhs_min
    {
        return BoundedExpr {
            expr: rhs.expr,
            bounds,
        };
    }
    if let (Some(rhs_max), Some(lhs_min)) = (rhs.bounds.max, lhs.bounds.min)
        && rhs_max <= lhs_min
    {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    if let Some((lhs_const, lhs_base)) = split_add_const(lhs.expr)
        && lhs_const >= 0
        && same_expr(lhs_base, rhs.expr, sym_ranges)
    {
        return BoundedExpr {
            expr: lhs.expr,
            bounds,
        };
    }
    if let Some((rhs_const, rhs_base)) = split_add_const(rhs.expr)
        && rhs_const >= 0
        && same_expr(rhs_base, lhs.expr, sym_ranges)
    {
        return BoundedExpr {
            expr: rhs.expr,
            bounds,
        };
    }
    BoundedExpr {
        expr: normalize_expr(lhs.expr.max(rhs.expr)),
        bounds,
    }
}

fn simplify_expr_with_bounds(
    expr: Expression,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> BoundedExpr {
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
                    _ => normalize_expr(lhs.expr * rhs.expr),
                };
                stack.push(BoundedExpr {
                    expr,
                    bounds: mul_bounds(lhs.bounds, rhs.bounds),
                });
            }
            Term::Div => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (exact_value(lhs), exact_value(rhs)) {
                    (Some(0), _) => Expression::from(0),
                    (_, Some(1)) => lhs.expr,
                    (Some(lhs), Some(rhs)) if rhs != 0 => Expression::from(lhs / rhs),
                    _ => normalize_expr(lhs.expr / rhs.expr),
                };
                stack.push(BoundedExpr {
                    expr,
                    bounds: div_bounds(lhs.bounds, rhs.bounds),
                });
            }
            Term::CeilDiv => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (exact_value(lhs), exact_value(rhs)) {
                    (Some(0), _) => Expression::from(0),
                    (_, Some(1)) => lhs.expr,
                    (Some(lhs), Some(rhs)) if rhs > 0 => {
                        Expression::from(if lhs % rhs != 0 { lhs / rhs + 1 } else { lhs / rhs })
                    }
                    _ => normalize_expr(lhs.expr.ceil_div(rhs.expr)),
                };
                stack.push(BoundedExpr {
                    expr,
                    bounds: div_bounds(lhs.bounds, rhs.bounds),
                });
            }
            Term::Mod => {
                let lhs = stack.pop().unwrap();
                let rhs = stack.pop().unwrap();
                let expr = match (exact_value(lhs), exact_value(rhs)) {
                    (Some(0), _) => Expression::from(0),
                    (_, Some(1)) => Expression::from(0),
                    (Some(lhs), Some(rhs)) if rhs != 0 => Expression::from(lhs % rhs),
                    _ => normalize_expr(lhs.expr % rhs.expr),
                };
                stack.push(BoundedExpr {
                    expr,
                    bounds: mod_bounds(lhs.bounds, rhs.bounds),
                });
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
    stack.pop().unwrap_or(with_bounds(expr, ExprBounds::default()))
}

fn canonical_dim(
    lhs: Expression,
    rhs: Expression,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> Expression {
    let lhs_simplified = simplify_expr_with_bounds(lhs, sym_ranges).expr;
    let rhs_simplified = simplify_expr_with_bounds(rhs, sym_ranges).expr;
    if lhs_simplified.len() <= rhs_simplified.len() {
        lhs_simplified
    } else {
        rhs_simplified
    }
}

fn normalize_equal_dims(
    a: &mut GraphTensor,
    b: &mut GraphTensor,
    sym_ranges: &FxHashMap<char, ExprBounds>,
) {
    for i in 0..a.shape.len() {
        let lhs = a.shape.dims[i];
        let rhs = b.shape.dims[i];
        if same_expr(lhs, rhs, sym_ranges) {
            let canonical = canonical_dim(lhs, rhs, sym_ranges);
            a.shape.dims[i] = canonical;
            b.shape.dims[i] = canonical;
        }
    }
}

fn same_dims(
    lhs: &[Expression],
    rhs: &[Expression],
    sym_ranges: &FxHashMap<char, ExprBounds>,
) -> bool {
    lhs.len() == rhs.len()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(lhs, rhs)| same_expr(*lhs, *rhs, sym_ranges))
}

impl<'a> Translator<'a> {
    pub(crate) fn translate_binary_op(&mut self, node: &Node, op: BinaryOp) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(name) = arg1.as_tensor_name() {
            let b = self.get_tensor(name)?;
            let (a, b) = ensure_same_dtype(a, b);
            let (mut a, mut b) = broadcast_binary(a, b);
            let sym_ranges = sym_char_ranges(&self.sym_map);
            normalize_equal_dims(&mut a, &mut b, &sym_ranges);
            let lhs_dims = a.dims();
            let rhs_dims = b.dims();
            if !same_dims(&lhs_dims, &rhs_dims, &sym_ranges) {
                anyhow::bail!(
                    "binary op {} still has mismatched dims after broadcast: lhs={lhs_dims:?} rhs={rhs_dims:?} inputs={:?}",
                    node.target,
                    node.inputs
                );
            }
            Ok(match op {
                BinaryOp::Add => a + b,
                BinaryOp::Mul => a * b,
                BinaryOp::Sub => a - b,
                BinaryOp::Div => a / b,
            })
        } else {
            if let Some(f) = arg1.as_float() {
                return Ok(self.apply_scalar_op(a, f as f32, op));
            }
            if let Some(expr) = self.resolve_arg_as_expression(arg1) {
                return Ok(self.apply_symbolic_scalar_op(a, expr, op));
            }
            let val = self.get_float_arg(node, 1)? as f32;
            Ok(self.apply_scalar_op(a, val, op))
        }
    }

    pub(crate) fn translate_binary_scalar_op(
        &mut self,
        node: &Node,
        op: BinaryOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let arg1 = &node.inputs[1].arg;
        if let Some(f) = arg1.as_float() {
            return Ok(self.apply_scalar_op(a, f as f32, op));
        }
        if let Some(expr) = self.resolve_arg_as_expression(arg1) {
            return Ok(self.apply_symbolic_scalar_op(a, expr, op));
        }
        let val = self.get_float_arg(node, 1)? as f32;
        Ok(self.apply_scalar_op(a, val, op))
    }

    pub(crate) fn apply_scalar_op(
        &mut self,
        a: GraphTensor,
        val: f32,
        op: BinaryOp,
    ) -> GraphTensor {
        let scalar = self
            .graph
            .constant_float(val)
            .cast(a.dtype)
            .expand_rhs(a.shape);
        match op {
            BinaryOp::Add => a + scalar,
            BinaryOp::Mul => a * scalar,
            BinaryOp::Sub => a - scalar,
            BinaryOp::Div => a / scalar,
        }
    }

    pub(crate) fn apply_symbolic_scalar_op(
        &mut self,
        a: GraphTensor,
        val: Expression,
        op: BinaryOp,
    ) -> GraphTensor {
        match op {
            BinaryOp::Add => a + val,
            BinaryOp::Mul => a * val,
            BinaryOp::Sub => a - val,
            BinaryOp::Div => a / val,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplifies_mark_dynamic_slice_shapes_using_lower_bound() {
        let a = Expression::from('a');
        let lhs = (a.min(1) + a).min(a + 1) - 1;
        let rhs = (a.min(1) + a).min(a);
        let sym_ranges = [('a', ExprBounds {
            min: Some(2),
            max: None,
        })]
        .into_iter()
        .collect::<FxHashMap<_, _>>();

        let lhs_simplified = simplify_expr_with_bounds(lhs, &sym_ranges).expr;
        let rhs_simplified = simplify_expr_with_bounds(rhs, &sym_ranges).expr;

        assert_eq!(lhs_simplified, Expression::from('a'));
        assert_eq!(rhs_simplified, Expression::from('a'));
        assert!(same_expr(lhs, rhs, &sym_ranges));
    }
}
