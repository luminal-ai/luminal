//! PT2 program JSON parser.
//!
//! Reads the program a frontend hands over in process (the `model.json`
//! document of a PT2 archive, without the archive) and extracts the graph
//! structure, input classification, and symbolic shape info.

use std::collections::HashMap;

use anyhow::{Context, Result};

use luminal::prelude::Symbol;
use luminal::prelude::tracing::warn;

use crate::pt2_schema::*;

/// A parsed program — everything needed for graph translation.
#[derive(Debug)]
pub struct ParsedPT2 {
    /// The exported program (graph, signature, etc.)
    pub program: ExportedProgram,
}

/// Classification of a graph input.
#[derive(Debug, Clone)]
pub enum InputKind {
    /// A tensor the program holds: a lifted tensor constant, whose bytes the
    /// caller stages. `original_name` is the constant's own name. Reported
    /// to Python as kind `"buffer"`.
    Buffer {
        graph_name: String,
        original_name: String,
    },
    /// A user-provided input tensor (e.g., "x")
    UserInput { graph_name: String },
}

/// Symbolic dimension mapping: PT2 symbol name -> luminal dim symbol.
///
/// Usually the same string — torch's `s77` stays `s77` — but a name luminal
/// cannot use is mapped to a minted one.
#[derive(Debug, Clone)]
pub struct SymDimMap {
    /// Maps PT2 symbol names (e.g. "s77") to the dim symbol of the same name.
    pub sym_to_symbol: HashMap<String, Symbol>,
    /// Range constraints for each symbol
    pub ranges: HashMap<String, RangeConstraint>,
}

impl ParsedPT2 {
    /// Classify the graph inputs the signature names: held tensor constants
    /// and user input tensors. A user input that is not a tensor (a
    /// symbolic or constant scalar) has no storage and is not listed.
    pub fn classify_inputs(&self) -> Vec<InputKind> {
        self.program
            .graph_module
            .signature
            .input_specs
            .iter()
            .filter_map(|spec| match spec {
                InputSpec::TensorConstant(tc) => Some(InputKind::Buffer {
                    graph_name: tc.arg.name.clone(),
                    original_name: tc.tensor_constant_name.clone(),
                }),
                InputSpec::UserInput(u) => {
                    u.arg.as_tensor_name().map(|name| InputKind::UserInput {
                        graph_name: name.to_string(),
                    })
                }
            })
            .collect()
    }

    /// Every value name a node takes as an operand.
    pub fn operand_names(&self) -> std::collections::HashSet<&str> {
        let mut operands = std::collections::HashSet::new();
        for node in &self.program.graph_module.graph.nodes {
            for input in &node.inputs {
                input.arg.for_each_value_name(&mut |name| {
                    operands.insert(name);
                });
            }
        }
        operands
    }

    /// The integer arguments that define a symbol, as `argument name ->
    /// symbol`.
    ///
    /// torch binds a symbol from an integer argument that IS that symbol.
    /// Here the argument defines it when the program uses either one: a
    /// node reads the argument, or another value's size or expression
    /// names the symbol. An argument nothing reads, whose symbol nothing
    /// else names, is never touched and defines nothing (a symbol torch
    /// mints for a stride alone is one).
    pub fn integer_symbol_definitions(&self) -> HashMap<String, String> {
        let graph = &self.program.graph_module.graph;
        let operands = self.operand_names();
        let candidates: Vec<(&str, String)> = graph
            .inputs
            .iter()
            .filter_map(|input| {
                let name = input.as_sym_int.as_ref()?.as_name.as_str();
                let symbol = graph
                    .sym_int_values
                    .get(name)
                    .and_then(bare_symbol_of_value)?;
                Some((name, symbol))
            })
            .collect();

        // Every symbol another value names: a tensor's size, or the
        // expression of a symbolic scalar. An argument no node reads does
        // not vouch for its own symbol.
        let mut named: std::collections::HashSet<String> = std::collections::HashSet::new();
        for meta in graph.tensor_values.values() {
            for size in &meta.sizes {
                if let Some(expr) = size.symbol_name() {
                    named.extend(symbol_names_in(expr));
                }
            }
        }
        let silent =
            |name: &str| !operands.contains(name) && candidates.iter().any(|(own, _)| *own == name);
        for values in [
            &graph.sym_int_values,
            &graph.sym_bool_values,
            &graph.sym_float_values,
        ] {
            for (name, value) in values {
                if silent(name) {
                    continue;
                }
                if let Some(expr) = value
                    .get("as_expr")
                    .and_then(|expr| expr.get("expr_str"))
                    .and_then(|expr| expr.as_str())
                {
                    named.extend(symbol_names_in(expr));
                }
            }
        }

        candidates
            .into_iter()
            .filter(|(name, symbol)| operands.contains(name) || named.contains(symbol))
            .map(|(name, symbol)| (name.to_string(), symbol))
            .collect()
    }

    /// Get tensor metadata by name.
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.program.graph_module.graph.tensor_values.get(name)
    }

    /// Build the symbolic dimension mapping.
    pub fn build_sym_dim_map(&self) -> SymDimMap {
        let mut sym_to_symbol = HashMap::new();

        // Collect all symbolic dimension names from tensor_values
        let mut sym_set = std::collections::HashSet::new();
        for meta in self.program.graph_module.graph.tensor_values.values() {
            for size in &meta.sizes {
                if let Some(sym_str) = size.symbol_name()
                    && let Some(name) = extract_symbol_name(sym_str)
                {
                    sym_set.insert(name);
                }
            }
        }

        // An integer argument that is a symbol states that symbol too, also
        // when no tensor axis carries it (`t * n`, or a backward graph that
        // is handed only a saved size).
        sym_set.extend(self.integer_symbol_definitions().into_values());

        let mut sym_names: Vec<String> = sym_set.into_iter().collect();
        sym_names.sort();

        // Every spelling torch uses is kept verbatim. The one name the
        // recorder refuses (the empty string) is remapped, not dropped: a
        // symbol absent from this map never gets a value, so its dim would
        // freeze at the export hint while torch, told it was dynamic,
        // declines to recompile. Counted rather than derived, and looping
        // because `Dim("pt2_dim_0")` is legal input.
        let mut minted = 0usize;
        for name in &sym_names {
            let symbol = Symbol::try_new_dim(name).unwrap_or_else(|e| {
                let replacement = loop {
                    let candidate = format!("pt2_dim_{minted}");
                    minted += 1;
                    if !sym_names.contains(&candidate) {
                        break candidate;
                    }
                };
                warn!(
                    "PT2 symbol {name:?} is not a usable luminal dimension name \
                     ({e}); using {replacement:?} internally. The dim stays \
                     dynamic and is still addressed as {name:?}."
                );
                Symbol::try_new_dim(&replacement)
                    .expect("minted names are well-formed by construction")
            });
            sym_to_symbol.insert(name.clone(), symbol);
        }

        SymDimMap {
            sym_to_symbol,
            ranges: self.program.range_constraints.clone(),
        }
    }
}

/// Extract the symbol name from a string like "Symbol('s77', positive=True, integer=True)".
/// Public alias for use by translator.
pub fn extract_symbol_name_pub(expr_str: &str) -> Option<String> {
    extract_symbol_name(expr_str)
}

/// The symbol a serialized symbolic integer IS, when its expression is one
/// bare symbol (`Symbol('u0', integer=True)`) and nothing else.
pub fn bare_symbol_of_value(value: &serde_json::Value) -> Option<String> {
    let expr = value.get("as_expr")?.get("expr_str")?.as_str()?.trim();
    // A compound expression starts with its function (`Mul(`, `Add(`), and
    // a bare symbol's `srepr` has no nested call.
    if !expr.starts_with("Symbol(") || expr.matches('(').count() != 1 {
        return None;
    }
    extract_symbol_name(expr)
}

/// The name of every `Symbol('...')` a sympy `srepr` string mentions.
fn symbol_names_in(expr_str: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = expr_str;
    while let Some(at) = rest.find("Symbol(") {
        rest = &rest[at..];
        if let Some(name) = extract_symbol_name(rest) {
            names.push(name);
        }
        rest = &rest["Symbol(".len()..];
    }
    names
}

fn extract_symbol_name(expr_str: &str) -> Option<String> {
    // Look for Symbol('name' or Symbol("name"
    let start = expr_str.find("Symbol(")? + 7;
    let rest = &expr_str[start..];
    // Skip the opening quote
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &rest[1..];
    let end = rest.find(quote)?;
    Some(rest[..end].to_string())
}

/// Parse a program handed over in process: the `model.json` document of a
/// PT2 archive, without the archive. Constant data does not travel with it;
/// the caller stages every held tensor itself. A signature naming an input
/// or output kind the in-process hand-off never writes is refused here, by
/// that kind's name.
pub fn parse_program(json: &[u8]) -> Result<ParsedPT2> {
    let program: ExportedProgram =
        serde_json::from_slice(json).context("Failed to parse the serialized program")?;
    Ok(ParsedPT2 { program })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_symbol_name() {
        assert_eq!(
            extract_symbol_name("Symbol('s77', positive=True, integer=True)"),
            Some("s77".to_string())
        );
        assert_eq!(
            extract_symbol_name("Symbol(\"batch\", positive=True)"),
            Some("batch".to_string())
        );
        assert_eq!(extract_symbol_name("not_a_symbol"), None);
    }

    /// Builds a program whose only tensor has one dim per given symbol name.
    fn program_with_dim_symbols(names: &[&str]) -> ParsedPT2 {
        let sizes = names
            .iter()
            .map(|n| {
                format!(
                    r#"{{"as_expr":{{"expr_str":"Symbol('{n}', positive=True, integer=True)"}}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{"graph_module":{{"graph":{{"inputs":[],"outputs":[],"nodes":[],
               "tensor_values":{{"x":{{"dtype":6,"sizes":[{sizes}]}}}}}},
               "signature":{{"input_specs":[]}}}}}}"#
        );
        ParsedPT2 {
            program: serde_json::from_str(&json).expect("fixture must parse"),
        }
    }

    /// A program whose signature carries the given specs, read the way the
    /// in-process hand-off reads it.
    fn program_with_signature(input_specs: &str, output_specs: &str) -> Result<ParsedPT2> {
        let json = format!(
            r#"{{"graph_module":{{"graph":{{"inputs":[],"outputs":[],
               "nodes":[],"tensor_values":{{}}}},
               "signature":{{"input_specs":[{input_specs}],
               "output_specs":[{output_specs}]}}}}}}"#
        );
        parse_program(json.as_bytes())
    }

    /// The signature kinds the in-process hand-off writes are read; a lifted
    /// tensor constant is the one input the program holds, classified as a
    /// buffer under its constant name.
    #[test]
    fn the_in_process_signature_kinds_are_read() {
        let parsed = program_with_signature(
            r#"{"user_input":{"arg":{"as_tensor":{"name":"x"}}}},
               {"user_input":{"arg":{"as_sym_int":{"as_name":"n"}}}},
               {"tensor_constant":{"arg":{"name":"c_lifted_tensor_0"},
                                   "tensor_constant_name":"lifted_tensor_0"}}"#,
            r#"{"user_input_mutation":{"arg":{"name":"add"},"user_input_name":"x"}},
               {"user_output":{"arg":{"as_tensor":{"name":"add"}}}},
               {"user_output":{"arg":{"as_none":true}}}"#,
        )
        .expect("every kind here is one the hand-off writes");
        let kinds: Vec<_> = parsed
            .classify_inputs()
            .into_iter()
            .map(|kind| match kind {
                InputKind::Buffer {
                    graph_name,
                    original_name,
                } => format!("buffer {graph_name} {original_name}"),
                InputKind::UserInput { graph_name } => format!("user_input {graph_name}"),
            })
            .collect();
        assert_eq!(
            kinds,
            vec!["user_input x", "buffer c_lifted_tensor_0 lifted_tensor_0"],
            "a scalar user input has no storage and is not classified"
        );
    }

    /// An input kind the in-process hand-off never writes is refused by
    /// name, never read as a user input: a `torch.export` parameter or
    /// buffer is held state this path has no storage for.
    #[test]
    fn an_unknown_input_spec_kind_is_refused_by_name() {
        for (kind, spec) in [
            (
                "parameter",
                r#"{"parameter":{"arg":{"name":"p_w"},"parameter_name":"w"}}"#,
            ),
            (
                "buffer",
                r#"{"buffer":{"arg":{"name":"b_cache"},"buffer_name":"cache"}}"#,
            ),
            ("custom_obj", r#"{"custom_obj":{"arg":{"name":"obj"}}}"#),
        ] {
            let error = program_with_signature(spec, "").expect_err(kind);
            let text = format!("{error:#}");
            assert!(text.contains(&format!("`{kind}`")), "{kind}: {text}");
        }
    }

    /// An output kind the in-process hand-off never writes is refused by
    /// name, never read as a user output: a mutation of some other kind
    /// would otherwise be returned instead of written back.
    #[test]
    fn an_unknown_output_spec_kind_is_refused_by_name() {
        for (kind, spec) in [
            (
                "buffer_mutation",
                r#"{"buffer_mutation":{"arg":{"name":"add"},"buffer_name":"cache"}}"#,
            ),
            ("loss_output", r#"{"loss_output":{"arg":{"name":"loss"}}}"#),
        ] {
            let error = program_with_signature("", spec).expect_err(kind);
            let text = format!("{error:#}");
            assert!(text.contains(&format!("`{kind}`")), "{kind}: {text}");
        }
    }

    /// torch lets a user write `Dim("_batch")`, `Dim("a__b")` or any other
    /// string, and torch's own symbols are `s77`/`u0`. Every spelling is
    /// kept verbatim and stays a *dynamic* dim: dropping or renaming the
    /// symbol would freeze it at the export-time hint with nothing to
    /// correct it later, and torch will not recompile. (`z` is an ordinary
    /// named symbol on this branch — the HLIR loop index that reserved it
    /// is gone.)
    #[test]
    fn build_sym_dim_map_keeps_every_torch_spelling() {
        let names = [
            "s77", "u0", "batch", "_batch", "a__b", "a.b", "seq len", "z",
        ];
        let map = program_with_dim_symbols(&names).build_sym_dim_map();

        assert_eq!(
            map.sym_to_symbol.len(),
            names.len(),
            "no dim may be dropped"
        );
        for name in names {
            assert_eq!(
                map.sym_to_symbol
                    .get(name)
                    .map(|s| s.to_string())
                    .as_deref(),
                Some(name),
                "{name} is addressed by its own spelling"
            );
        }

        let distinct: std::collections::HashSet<_> =
            map.sym_to_symbol.values().map(|s| s.to_string()).collect();
        assert_eq!(
            distinct.len(),
            names.len(),
            "two dims must never share a symbol"
        );
    }

    /// A symbol only an integer argument states (`t * n`: no tensor axis
    /// carries `n`) is a symbol of the program when the program uses it: a
    /// node reads the argument, or another value's expression names the
    /// symbol. A compound integer argument states no symbol of its own,
    /// and neither does an argument nothing reads whose symbol nothing
    /// else names.
    #[test]
    fn build_sym_dim_map_reads_bare_integer_arguments() {
        let json = r#"{"graph_module":{"graph":{
            "inputs":[{"as_tensor":{"name":"x"}},{"as_sym_int":{"as_name":"n"}},
                      {"as_sym_int":{"as_name":"twice"}},
                      {"as_sym_int":{"as_name":"unread"}},
                      {"as_sym_int":{"as_name":"named"}}],
            "outputs":[],
            "nodes":[{"target":"torch.ops.aten.mul.Tensor",
                      "inputs":[{"name":"self","arg":{"as_tensor":{"name":"x"}}},
                                {"name":"other","arg":{"as_sym_int":{"as_name":"n"}}},
                                {"name":"more","arg":{"as_sym_int":{"as_name":"twice"}}}],
                      "outputs":[{"as_tensor":{"name":"y"}}]}],
            "tensor_values":{"x":{"dtype":6,"sizes":[{"as_int":4}]}},
            "sym_int_values":{
              "n":{"as_expr":{"expr_str":"Symbol('u0', integer=True)","hint":null}},
              "twice":{"as_expr":{"expr_str":"Mul(Integer(2), Symbol('u1', integer=True))"}},
              "unread":{"as_expr":{"expr_str":"Symbol('u2', integer=True)"}},
              "named":{"as_expr":{"expr_str":"Symbol('u3', integer=True)"}}},
            "sym_bool_values":{
              "big":{"as_expr":{"expr_str":"StrictGreaterThan(Symbol('u3', integer=True), Integer(3))"}}}},
            "signature":{"input_specs":[]}}}"#;
        let parsed = ParsedPT2 {
            program: serde_json::from_str(json).expect("fixture must parse"),
        };
        let map = parsed.build_sym_dim_map();
        let mut symbols: Vec<_> = map.sym_to_symbol.keys().cloned().collect();
        symbols.sort();
        assert_eq!(
            symbols,
            vec!["u0", "u3"],
            "the argument a node reads, and the one another expression names"
        );
        assert_eq!(
            parsed
                .integer_symbol_definitions()
                .get("named")
                .map(String::as_str),
            Some("u3")
        );
    }

    /// The empty string is the one name the recorder refuses; it is minted
    /// a replacement, and minting must not collide with a name the user
    /// actually wrote.
    #[test]
    fn build_sym_dim_map_mints_around_a_name_the_user_already_used() {
        let map = program_with_dim_symbols(&["pt2_dim_0", ""]).build_sym_dim_map();

        assert_eq!(map.sym_to_symbol["pt2_dim_0"].to_string(), "pt2_dim_0");
        assert!(map.sym_to_symbol.contains_key(""), "the dim is not dropped");
        assert_ne!(
            map.sym_to_symbol[""].to_string(),
            "pt2_dim_0",
            "minted name must not steal the one the user declared"
        );
    }

    /// The ceiling this branch removed. `build_sym_dim_map` used to hand each
    /// symbol a char from an `a..y` + `A..Z` pool and panic at 52 — and symbol
    /// #26 landed on `z`, the reserved index, so it vanished from the buffer
    /// planner. A 32-layer llama under sdpa + DynamicCache mints more than 26.
    #[test]
    fn build_sym_dim_map_has_no_symbol_ceiling() {
        let names: Vec<String> = (0..200).map(|n| format!("s{n}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let map = program_with_dim_symbols(&refs).build_sym_dim_map();
        assert_eq!(map.sym_to_symbol.len(), 200);
        let distinct: std::collections::HashSet<_> =
            map.sym_to_symbol.values().map(|s| s.to_string()).collect();
        assert_eq!(distinct.len(), 200, "every dim must stay distinct");
    }
}
