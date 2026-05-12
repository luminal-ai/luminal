use std::collections::HashMap;

use luminal::prelude::*;

/// Parse a sympy `srepr`-style expression string into a luminal `Expression`.
///
/// Supports the subset of sympy heads PT2 emits for symbolic shape metadata.
pub(crate) fn parse_sympy_expr(
    expr: &str,
    sym_to_char: &HashMap<String, char>,
) -> Option<Expression> {
    let expr = expr.trim();
    if expr.is_empty() {
        return None;
    }

    if let Ok(value) = expr.parse::<i64>() {
        return Some(Expression::from(value));
    }

    let (head, body) = split_head(expr)?;
    match head {
        "Symbol" => {
            let name = extract_first_quoted(body)?;
            sym_to_char.get(&name).map(|c| Expression::from(*c))
        }
        "Integer" | "Number" => {
            let value = body.trim().parse::<i64>().ok()?;
            Some(Expression::from(value))
        }
        "NegativeOne" => Some(Expression::from(-1i64)),
        "Zero" => Some(Expression::from(0i64)),
        "One" => Some(Expression::from(1i64)),
        "Mul" | "Add" | "Min" | "Max" => {
            let parts = split_top_level_args(body);
            if parts.is_empty() {
                return None;
            }
            let mut iter = parts.into_iter();
            let mut acc = parse_sympy_expr(iter.next()?, sym_to_char)?;
            for part in iter {
                let rhs = parse_sympy_expr(part, sym_to_char)?;
                acc = match head {
                    "Mul" => acc * rhs,
                    "Add" => acc + rhs,
                    "Min" => acc.min(rhs),
                    "Max" => acc.max(rhs),
                    _ => unreachable!(),
                };
            }
            Some(acc)
        }
        "FloorDiv" => {
            let mut parts = split_top_level_args(body).into_iter();
            let lhs = parse_sympy_expr(parts.next()?, sym_to_char)?;
            let rhs = parse_sympy_expr(parts.next()?, sym_to_char)?;
            if parts.next().is_some() {
                return None;
            }
            Some(lhs / rhs)
        }
        "Mod" => {
            let mut parts = split_top_level_args(body).into_iter();
            let lhs = parse_sympy_expr(parts.next()?, sym_to_char)?;
            let rhs = parse_sympy_expr(parts.next()?, sym_to_char)?;
            if parts.next().is_some() {
                return None;
            }
            Some(lhs % rhs)
        }
        _ => None,
    }
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
}
