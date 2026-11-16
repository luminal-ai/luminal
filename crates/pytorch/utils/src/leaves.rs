//! The output leaves of a compiled graph, as the shared frontend states
//! them and as a code object carries them.
//!
//! The frontend writes one row per output leaf (`LocalGraph.leaf_rows` in
//! `luminal_torch_common`). [`parse`] reads those rows into the leaf table
//! every backend's code object holds (`luminal_torch_code::Leaf`): the same
//! facts with every expression parsed.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::prelude::Symbol;
use luminal_torch_code::{Leaf, Literal, Source};

/// One output leaf as the frontend writes it.
#[derive(serde::Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum LeafRow {
    Row {
        row: usize,
        scalar: bool,
    },
    Argument {
        slot: usize,
    },
    Expression {
        kind: String,
        expr: String,
        /// Where a call states each dimension the expression names:
        /// `(name, argument slot, axis or null)`.
        sources: Vec<(String, usize, Option<usize>)>,
    },
    Literal {
        value: serde_json::Value,
    },
}

/// Read the frontend's leaf rows (JSON text). A symbolic float is refused by
/// name: a code object carries symbolic integers and bools only (LUM-1490).
pub fn parse(leaves: &str) -> Result<Vec<Leaf>> {
    let rows: Vec<LeafRow> = serde_json::from_str(leaves).context("reading the output leaves")?;
    rows.into_iter()
        .enumerate()
        .map(|(position, row)| {
            Ok(match row {
                LeafRow::Row { row, scalar } => Leaf::Row { row, scalar },
                LeafRow::Argument { slot } => Leaf::Argument { slot },
                LeafRow::Expression {
                    kind,
                    expr,
                    sources,
                } => {
                    ensure!(
                        kind == "int" || kind == "bool",
                        "output leaf {position} is a symbolic {kind} ({expr}); a code object \
                         carries symbolic integers and bools only (LUM-1490)"
                    );
                    // The expression is evaluated from the call's own
                    // arguments, so its symbols are the leaf's alone.
                    let symbols: HashMap<String, Symbol> = sources
                        .iter()
                        .map(|(name, _, _)| (name.clone(), Symbol::new(name.as_str())))
                        .collect();
                    let parsed =
                        crate::translate::parse_expr(&symbols, &expr).ok_or_else(|| {
                            anyhow!(
                                "output leaf {position} is {expr}, which names a dimension no \
                             argument of the call states, or a form the parser does not read"
                            )
                        })?;
                    let sources = sources
                        .into_iter()
                        .map(|(name, slot, axis)| Source {
                            symbol: symbols[&name],
                            slot,
                            axis,
                        })
                        .collect();
                    if kind == "bool" {
                        Leaf::Bool {
                            expr: parsed,
                            sources,
                        }
                    } else {
                        Leaf::Integer {
                            expr: parsed,
                            sources,
                        }
                    }
                }
                LeafRow::Literal { value } => Leaf::Literal(match value {
                    serde_json::Value::Null => Literal::None,
                    serde_json::Value::Bool(value) => Literal::Bool(value),
                    serde_json::Value::String(value) => Literal::Str(value),
                    serde_json::Value::Number(number) => match number.as_i64() {
                        Some(value) => Literal::Int(value),
                        None => Literal::Float(number.as_f64().ok_or_else(|| {
                            anyhow!("output leaf {position} is the number {number}")
                        })?),
                    },
                    other => bail!("output leaf {position} is the literal {other}"),
                }),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frontends_rows_are_read_with_their_expressions_parsed() {
        let leaves = parse(
            r#"[
                {"source": "row", "row": 1, "scalar": true},
                {"source": "argument", "slot": 2},
                {"source": "expression", "kind": "int",
                 "expr": "Mul(Integer(2), Symbol('s0', integer=True, positive=True))",
                 "sources": [["s0", 0, 1]]},
                {"source": "expression", "kind": "bool",
                 "expr": "Symbol('s0', integer=True, positive=True)",
                 "sources": [["s0", 3, null]]},
                {"source": "literal", "value": null},
                {"source": "literal", "value": 1.5},
                {"source": "literal", "value": "text"}
            ]"#,
        )
        .unwrap();
        let s0 = Symbol::new("s0");
        assert!(matches!(
            leaves[0],
            Leaf::Row {
                row: 1,
                scalar: true
            }
        ));
        assert!(matches!(leaves[1], Leaf::Argument { slot: 2 }));
        let Leaf::Integer { expr, sources } = &leaves[2] else {
            panic!("leaf 2 read as {:?}", leaves[2]);
        };
        assert_eq!(
            sources,
            &[Source {
                symbol: s0,
                slot: 0,
                axis: Some(1)
            }]
        );
        let stated = [(s0, 5)].into_iter().collect();
        assert_eq!(expr.exec(&stated), Some(10));
        let Leaf::Bool { sources, .. } = &leaves[3] else {
            panic!("leaf 3 read as {:?}", leaves[3]);
        };
        assert_eq!(
            sources,
            &[Source {
                symbol: s0,
                slot: 3,
                axis: None
            }]
        );
        assert!(matches!(&leaves[4], Leaf::Literal(Literal::None)));
        assert!(matches!(&leaves[5], Leaf::Literal(Literal::Float(value)) if *value == 1.5));
        assert!(matches!(&leaves[6], Leaf::Literal(Literal::Str(text)) if text == "text"));
    }

    #[test]
    fn a_symbolic_float_is_refused_by_name() {
        let error = parse(
            r#"[{"source": "expression", "kind": "float", "expr": "Symbol('zf0')",
                 "sources": []}]"#,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("symbolic float"), "{error}");
        assert!(error.contains("LUM-1490"), "{error}");
    }
}
