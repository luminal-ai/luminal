//! TEMPORARY (LUM-1580): the compile-time refusal of data-dependent sizes.
//!
//! Every dimension symbol of a compiled program gets its value from the
//! arguments of a call: from the shape of an input tensor, or from a
//! symbolic scalar argument. A size that a node computes from tensor data
//! (the row count of `nonzero`, the length of `x[mask]`, a size read with
//! `.item()`) has no value until part of the program has run, and nothing
//! downstream can state one: the search needs a value for every symbol,
//! memory is planned from the symbols' ranges, each output is allocated
//! before the program starts, and the CUDA runtime returns from a call
//! before the device has run it.
//!
//! This pass names the first node whose result has such a size and refuses
//! the program, so the failure says what is unsupported instead of
//! surfacing later as a dimension nothing supplies. It is one function
//! over the parsed graph and keeps no state. Remove this module and its
//! one call in `boundary::plan` when data-dependent sizes are supported.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::pt2_parser::{ParsedPT2, symbol_names_in};
use crate::pt2_schema::{Graph, TensorRef};

/// Refuse a program in which the shape of a node's result uses a symbol
/// that no argument states.
pub fn refuse(parsed: &ParsedPT2) -> Result<()> {
    let graph = &parsed.program.graph_module.graph;

    // What the arguments state: every symbol in the shape of an input
    // tensor or in the expression of a symbolic scalar argument.
    let mut stated: HashSet<String> = HashSet::new();
    for input in &graph.inputs {
        stated.extend(
            tensor_symbols(graph, input)
                .into_iter()
                .map(|(_, symbol)| symbol),
        );
        stated.extend(scalar_symbols(graph, input));
    }

    // A symbol a node creates as a scalar (`.item()`), and the node that
    // created it: a later size that uses it names where it came from.
    let mut computed: HashMap<String, &str> = HashMap::new();
    for node in &graph.nodes {
        for output in &node.outputs {
            for symbol in scalar_symbols(graph, output) {
                if !stated.contains(&symbol) {
                    computed.entry(symbol).or_insert(&node.target);
                }
            }
            for (tensor, symbol) in tensor_symbols(graph, output) {
                if stated.contains(&symbol) {
                    continue;
                }
                let source = match computed.get(&symbol) {
                    Some(target) if *target != node.target => {
                        format!(", which `{target}` computes from tensor data")
                    }
                    _ => ", a size computed from tensor data".to_string(),
                };
                bail!(
                    "`{}`: the shape of its result `{tensor}` uses `{symbol}`{source}. Every \
                     size of a compiled program must be known from the arguments of a call, \
                     before the program runs; data-dependent sizes are not supported yet \
                     (LUM-1580). By default (`fullgraph=False`) torch runs such an op in \
                     eager instead of sending it to a backend.",
                    node.target
                );
            }
        }
    }
    Ok(())
}

/// Each symbol in the shape of each tensor a reference names, with that
/// tensor's name.
fn tensor_symbols<'a>(graph: &'a Graph, reference: &'a TensorRef) -> Vec<(&'a str, String)> {
    let names = reference
        .as_tensor
        .iter()
        .chain(reference.as_tensors.iter().flatten())
        .map(|tensor| tensor.name.as_str());
    let mut symbols = Vec::new();
    for name in names {
        let Some(meta) = graph.tensor_values.get(name) else {
            continue;
        };
        for size in &meta.sizes {
            if let Some(expr) = size.symbol_name() {
                symbols.extend(
                    symbol_names_in(expr)
                        .into_iter()
                        .map(|symbol| (name, symbol)),
                );
            }
        }
    }
    symbols
}

/// Each symbol in the expression of the symbolic scalar a reference names.
fn scalar_symbols(graph: &Graph, reference: &TensorRef) -> Vec<String> {
    let scalars = [
        (&reference.as_sym_int, &graph.sym_int_values),
        (&reference.as_sym_float, &graph.sym_float_values),
        (&reference.as_sym_bool, &graph.sym_bool_values),
    ];
    let mut symbols = Vec::new();
    for (scalar, values) in scalars {
        let expr = scalar
            .as_ref()
            .and_then(|scalar| values.get(&scalar.as_name))
            .and_then(|value| value.get("as_expr"))
            .and_then(|expr| expr.get("expr_str"))
            .and_then(|expr| expr.as_str());
        if let Some(expr) = expr {
            symbols.extend(symbol_names_in(expr));
        }
    }
    symbols
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pt2_schema::{DimExpr, DimSize, ExprValue, ScalarName};
    use crate::translate::test_support::{node, parsed_nodes, tensor_input};

    const FLOAT32: u32 = 7;

    fn symbol(name: &str) -> DimSize {
        DimSize::Expr(DimExpr {
            as_expr: ExprValue {
                expr_str: format!("Symbol('{name}', integer=True)"),
                hint: None,
            },
        })
    }

    fn set_sizes(parsed: &mut ParsedPT2, tensor: &str, sizes: Vec<DimSize>) {
        parsed
            .program
            .graph_module
            .graph
            .tensor_values
            .get_mut(tensor)
            .expect("declared tensor")
            .sizes = sizes;
    }

    #[test]
    fn a_size_a_node_computes_is_refused_by_name() {
        let index = node(
            "torch.ops.aten.index.Tensor",
            vec![tensor_input("self", "x"), tensor_input("indices", "mask")],
            &["index"],
        );
        let mut parsed = parsed_nodes(
            vec![index],
            &[
                ("x", FLOAT32, &[4, 6]),
                ("mask", FLOAT32, &[4, 6]),
                ("index", FLOAT32, &[1]),
            ],
            &["x", "mask"],
            &["index"],
        );
        set_sizes(&mut parsed, "index", vec![symbol("u0")]);

        let refusal = refuse(&parsed)
            .expect_err("a size no argument states")
            .to_string();
        assert!(
            refusal.contains("`torch.ops.aten.index.Tensor`"),
            "{refusal}"
        );
        assert!(refusal.contains("result `index` uses `u0`"), "{refusal}");
        assert!(
            refusal.contains("a size computed from tensor data"),
            "{refusal}"
        );
        assert!(refusal.contains("LUM-1580"), "{refusal}");
    }

    #[test]
    fn a_size_an_argument_states_passes() {
        let relu = node(
            "torch.ops.aten.relu.default",
            vec![tensor_input("self", "x")],
            &["relu"],
        );
        let mut parsed = parsed_nodes(
            vec![relu],
            &[("x", FLOAT32, &[4, 6]), ("relu", FLOAT32, &[4, 6])],
            &["x"],
            &["relu"],
        );
        // An expression over a symbol the input states is not a new size.
        set_sizes(
            &mut parsed,
            "x",
            vec![
                symbol("s0"),
                DimSize::Int(crate::pt2_schema::DimInt { as_int: 6 }),
            ],
        );
        set_sizes(
            &mut parsed,
            "relu",
            vec![
                DimSize::Expr(DimExpr {
                    as_expr: ExprValue {
                        expr_str: "Mul(Integer(2), Symbol('s0', integer=True))".to_string(),
                        hint: None,
                    },
                }),
                DimSize::Int(crate::pt2_schema::DimInt { as_int: 6 }),
            ],
        );
        refuse(&parsed).expect("every size is stated by the argument");
    }

    #[test]
    fn a_size_read_with_item_names_the_node_that_read_it() {
        let mut item = node(
            "torch.ops.aten._local_scalar_dense.default",
            vec![tensor_input("self", "count")],
            &[],
        );
        item.outputs = vec![TensorRef {
            as_tensor: None,
            as_tensors: None,
            as_sym_int: Some(ScalarName {
                as_name: "item".to_string(),
            }),
            as_sym_float: None,
            as_sym_bool: None,
        }];
        let slice = node(
            "torch.ops.aten.slice.Tensor",
            vec![tensor_input("self", "x")],
            &["slice"],
        );
        let mut parsed = parsed_nodes(
            vec![item, slice],
            &[
                ("x", FLOAT32, &[24]),
                ("count", FLOAT32, &[]),
                ("slice", FLOAT32, &[1]),
            ],
            &["x", "count"],
            &["slice"],
        );
        parsed.program.graph_module.graph.sym_int_values.insert(
            "item".to_string(),
            serde_json::json!({"as_expr": {"expr_str": "Symbol('u0', integer=True)"}}),
        );
        set_sizes(&mut parsed, "slice", vec![symbol("u0")]);

        let refusal = refuse(&parsed)
            .expect_err("a size read from a tensor")
            .to_string();
        assert!(
            refusal.contains("`torch.ops.aten.slice.Tensor`"),
            "{refusal}"
        );
        assert!(
            refusal.contains(
                "uses `u0`, which `torch.ops.aten._local_scalar_dense.default` computes from \
                 tensor data"
            ),
            "{refusal}"
        );
    }
}
