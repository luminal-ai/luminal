//! Canonical-form helpers for dimension `Expression` arithmetic — used
//! by the translator to keep shape arithmetic syntactically consistent
//! across code paths.
//!
//! `Expression` equality is syntactic; `a * 8` and `8 * a` are distinct
//! objects despite being mathematically equal. When two translator code
//! paths build the same logical dim via differently-ordered
//! multiplications, downstream `assert_eq!(self.dims(), rhs.dims())`
//! checks in `GraphTensor::Add` / `Sub` / `Mul` / `Rem` panic. These
//! helpers solve that at the construction site: every shape product
//! goes through `product_of_dims`, which sorts the operand list before
//! folding, so two callers passing the operands in different orders
//! produce identical `Expression`s.
//!
//! Lives in `luminal_python` (rather than upstream `luminal::shape`) so
//! the change is contained to the translator. luminal-core callers of
//! `gather_elements` / `scatter_elements` / `scatter_nd` historically
//! pass concrete dims, so they don't need this; the translator-local
//! lowerings in `translator::movement_dynamic` do.
//!
//! The ordering matches what `pt2_expr.rs::normalize_mul_expr` was
//! using locally before being promoted here — see that file for the
//! original canonical-sort logic.

use luminal::prelude::Expression;

/// Sort key for the canonical commutative ordering. Sorts by RPN-term
/// count first so single-term operands (variables, literals) sort
/// before compound subexpressions; ties broken by debug repr so two
/// single-term operands have a stable alphabetic order.
///
/// O(n) string alloc per compare — only call on shape products, never
/// per-element in a kernel.
#[inline]
pub(crate) fn commutative_key(expr: &Expression) -> (usize, String) {
    (expr.len(), format!("{expr:?}"))
}

/// Order `(a, b)` so the canonically-smaller expression is first.
#[inline]
pub(crate) fn sort_pair(a: Expression, b: Expression) -> (Expression, Expression) {
    if commutative_key(&a) <= commutative_key(&b) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Multiply two dim expressions with canonical operand ordering.
#[inline]
pub(crate) fn mul_dims(a: Expression, b: Expression) -> Expression {
    let (a, b) = sort_pair(a, b);
    a * b
}

/// Add two dim expressions with canonical operand ordering.
#[inline]
pub(crate) fn add_dims(a: Expression, b: Expression) -> Expression {
    let (a, b) = sort_pair(a, b);
    a + b
}

/// Product of a sequence of dim expressions. Operands are sorted
/// canonically before folding so callers passing the same logical
/// dim set in different orders produce identical `Expression`s.
/// Empty sequence → `Expression::from(1usize)`.
pub(crate) fn product_of_dims<I>(dims: I) -> Expression
where
    I: IntoIterator<Item = Expression>,
{
    let mut v: Vec<Expression> = dims.into_iter().collect();
    v.sort_by_key(commutative_key);
    v.into_iter()
        .fold(Expression::from(1usize), |acc, d| acc * d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul_dims_canonicalises_commutative_order() {
        let a = Expression::from('a');
        let n = Expression::from(8i64);
        assert_eq!(mul_dims(a, n), mul_dims(n, a));
    }

    #[test]
    fn product_of_dims_independent_of_input_order() {
        let a = Expression::from('a');
        let b = Expression::from('b');
        let n = Expression::from(8i64);
        let p1 = product_of_dims([a, n, b]);
        let p2 = product_of_dims([n, b, a]);
        let p3 = product_of_dims([b, a, n]);
        assert_eq!(p1, p2);
        assert_eq!(p1, p3);
    }

    #[test]
    fn empty_product_is_one() {
        let empty: Vec<Expression> = vec![];
        assert_eq!(product_of_dims(empty), Expression::from(1usize));
    }

    #[test]
    fn mixed_numeric_types_canonicalise_together() {
        // `pt2_util` builds with `Expression::from(usize)` while tests /
        // direct callers reach for `i64`. The two literal paths must
        // produce identical reprs or `product_of_dims` will sort them
        // into different positions and we lose the canonical-form
        // guarantee across call sites.
        assert_eq!(Expression::from(8usize), Expression::from(8i64));
        let a = Expression::from('a');
        assert_eq!(
            product_of_dims([Expression::from(8usize), a]),
            product_of_dims([Expression::from(8i64), a]),
        );
    }
}
