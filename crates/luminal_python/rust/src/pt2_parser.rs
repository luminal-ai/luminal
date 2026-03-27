//! PT2 ZIP + JSON parser.
//!
//! Opens a .pt2 file (ZIP archive), reads the model JSON, and extracts
//! the graph structure, weight mapping, and symbolic shape info.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use anyhow::{Context, Result};
use zip::ZipArchive;

use crate::pt2_schema::*;

/// Parsed PT2 file contents — everything needed for graph translation.
#[derive(Debug)]
pub struct ParsedPT2 {
    /// The exported program (graph, signature, etc.)
    pub program: ExportedProgram,
    /// Constants config: tensor constant name -> (file path in zip, tensor metadata)
    pub constants_config: Option<WeightsConfig>,
    /// Archive name prefix (e.g., "luminal_mlp")
    pub archive_prefix: String,
    /// Path to the original .pt2 file (for re-reading constants)
    pub pt2_path: String,
}

/// Classification of a graph input.
#[derive(Debug, Clone)]
pub enum InputKind {
    /// A model parameter (e.g., "fc1.weight")
    Parameter {
        graph_name: String,
        original_name: String,
    },
    /// A model buffer (e.g., "running_mean")
    Buffer {
        graph_name: String,
        original_name: String,
    },
    /// A user-provided input tensor (e.g., "x")
    UserInput { graph_name: String },
}

/// Symbolic dimension mapping: PT2 symbol name -> luminal char variable.
#[derive(Debug, Clone)]
pub struct SymDimMap {
    /// Maps PT2 symbol names (e.g., "s77") to luminal char variables ('a', 'b', ...)
    pub sym_to_char: HashMap<String, char>,
    /// Range constraints for each symbol
    pub ranges: HashMap<String, RangeConstraint>,
}

impl ParsedPT2 {
    /// Classify all graph inputs into parameters, buffers, and user inputs.
    pub fn classify_inputs(&self) -> Vec<InputKind> {
        self.program
            .graph_module
            .signature
            .input_specs
            .iter()
            .filter_map(|spec| match spec {
                InputSpec::Parameter(p) => Some(InputKind::Parameter {
                    graph_name: p.parameter.arg.name.clone(),
                    original_name: p.parameter.parameter_name.clone(),
                }),
                InputSpec::Buffer(b) => Some(InputKind::Buffer {
                    graph_name: b.buffer.arg.name.clone(),
                    original_name: b.buffer.buffer_name.clone(),
                }),
                InputSpec::TensorConstant(tc) => Some(InputKind::Buffer {
                    graph_name: tc.tensor_constant.arg.name.clone(),
                    original_name: tc.tensor_constant.tensor_constant_name.clone(),
                }),
                InputSpec::UserInput(u) => {
                    u.user_input
                        .arg
                        .as_tensor_name()
                        .map(|name| InputKind::UserInput {
                            graph_name: name.to_string(),
                        })
                }
                InputSpec::Other(_) => None,
            })
            .collect()
    }

    /// Get the output tensor names.
    pub fn output_names(&self) -> Vec<String> {
        self.program
            .graph_module
            .graph
            .outputs
            .iter()
            .filter_map(|o| o.as_tensor.as_ref().map(|t| t.name.clone()))
            .collect()
    }

    /// Get tensor metadata by name.
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.program.graph_module.graph.tensor_values.get(name)
    }

    /// Build the symbolic dimension mapping.
    pub fn build_sym_dim_map(&self) -> SymDimMap {
        let mut sym_to_char = HashMap::new();
        let mut next_char = b'a';

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

        let mut sym_names: Vec<String> = sym_set.into_iter().collect();
        sym_names.sort();

        for name in &sym_names {
            if next_char <= b'z' {
                sym_to_char.insert(name.clone(), next_char as char);
                next_char += 1;
            }
        }

        SymDimMap {
            sym_to_char,
            ranges: self.program.range_constraints.clone(),
        }
    }
}

/// Extract the symbol name from a string like "Symbol('s77', positive=True, integer=True)".
/// Public alias for use by translator.
pub fn extract_symbol_name_pub(expr_str: &str) -> Option<String> {
    extract_symbol_name(expr_str)
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

/// Parse a .pt2 file from disk.
pub fn parse_pt2(path: &str) -> Result<ParsedPT2> {
    let file = File::open(path).with_context(|| format!("Failed to open PT2 file: {path}"))?;
    let mut archive = ZipArchive::new(file).context("Failed to read PT2 ZIP archive")?;

    // Determine archive prefix from the first entry
    let archive_prefix = {
        let first = archive
            .file_names()
            .next()
            .context("Empty PT2 archive")?
            .to_string();
        first.split('/').next().unwrap_or(&first).to_string()
    };

    // Read model.json
    let model_json_path = format!("{archive_prefix}/models/model.json");
    let program: ExportedProgram = {
        let mut entry = archive
            .by_name(&model_json_path)
            .with_context(|| format!("Missing {model_json_path} in PT2 archive"))?;
        let mut buf = String::new();
        entry.read_to_string(&mut buf)?;
        serde_json::from_str(&buf).with_context(|| "Failed to parse model.json")?
    };

    // Read constants config (optional — not all models have constants)
    let constants_config_path =
        format!("{archive_prefix}/data/constants/model_constants_config.json");
    let constants_config: Option<WeightsConfig> = archive
        .by_name(&constants_config_path)
        .ok()
        .and_then(|mut entry| {
            let mut buf = String::new();
            entry.read_to_string(&mut buf).ok()?;
            serde_json::from_str(&buf).ok()
        });

    Ok(ParsedPT2 {
        program,
        constants_config,
        archive_prefix,
        pt2_path: path.to_string(),
    })
}

/// Read raw constant bytes from the PT2 archive for a given constant entry.
pub fn read_constant_bytes(
    pt2_path: &str,
    archive_prefix: &str,
    entry: &WeightEntry,
) -> Result<Vec<u8>> {
    let file = File::open(pt2_path)?;
    let mut archive = ZipArchive::new(file)?;
    let path = format!("{archive_prefix}/data/constants/{}", entry.path_name);
    let mut zip_entry = archive
        .by_name(&path)
        .with_context(|| format!("Missing constant file: {path}"))?;
    let mut buf = Vec::new();
    zip_entry.read_to_end(&mut buf)?;
    Ok(buf)
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

    #[test]
    fn test_parse_addone_pt2() {
        let path = "/tmp/luminal_addone.pt2";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let parsed = parse_pt2(path).unwrap();
        assert_eq!(parsed.program.graph_module.graph.nodes.len(), 1);
        assert_eq!(
            parsed.program.graph_module.graph.nodes[0].target,
            "torch.ops.aten.add.Tensor"
        );
        let inputs = parsed.classify_inputs();
        assert_eq!(inputs.len(), 1);
        assert!(matches!(&inputs[0], InputKind::UserInput { graph_name } if graph_name == "x"));
        let outputs = parsed.output_names();
        assert_eq!(outputs, vec!["add"]);
    }

    #[test]
    fn test_parse_mlp_pt2() {
        let path = "/tmp/luminal_mlp.pt2";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let parsed = parse_pt2(path).unwrap();
        assert_eq!(parsed.program.graph_module.graph.nodes.len(), 3);

        let inputs = parsed.classify_inputs();
        let params: Vec<_> = inputs
            .iter()
            .filter(|i| matches!(i, InputKind::Parameter { .. }))
            .collect();
        let user_inputs: Vec<_> = inputs
            .iter()
            .filter(|i| matches!(i, InputKind::UserInput { .. }))
            .collect();
        assert_eq!(params.len(), 3); // fc1.weight, fc2.weight, fc2.bias
        assert_eq!(user_inputs.len(), 1);
    }

    #[test]
    fn test_parse_dynamic_pt2() {
        let path = "/tmp/luminal_dyn.pt2";
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping: {path} not found");
            return;
        }
        let parsed = parse_pt2(path).unwrap();
        let sym_map = parsed.build_sym_dim_map();
        // Should have one symbolic dim (s77)
        assert_eq!(sym_map.sym_to_char.len(), 1);
        assert!(sym_map.sym_to_char.contains_key("s77"));
        assert_eq!(sym_map.sym_to_char["s77"], 'a');
    }
}
