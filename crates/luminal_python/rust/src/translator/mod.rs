//! PT2 graph nodes -> Luminal Graph translation.
//!
//! Walks the parsed PT2 graph and constructs an equivalent Luminal computation graph.

mod binary;
mod conv;
mod dispatch;
mod movement;
mod reduction;
mod tensor;
mod unary;

use std::collections::HashMap;

use anyhow::{Context, Result};
use luminal::graph::Graph;
use luminal::prelude::*;

use crate::pt2_parser::{InputKind, ParsedPT2, SymDimMap};
use crate::pt2_schema::*;
use crate::pt2_util;

/// Result of translating a PT2 graph to a Luminal graph.
pub struct TranslatedGraph {
    /// The luminal computation graph.
    pub graph: Graph,
    /// Node IDs for user inputs (in order).
    pub user_input_ids: Vec<(String, NodeIndex)>,
    /// Node IDs for outputs (in order).
    pub output_ids: Vec<(String, NodeIndex)>,
    /// Symbolic dimension mapping.
    pub sym_map: SymDimMap,
}

/// Main translation entry point.
pub fn translate(parsed: &ParsedPT2) -> Result<TranslatedGraph> {
    let mut translator = Translator::new(parsed)?;
    translator.translate_graph()?;
    Ok(translator.finish())
}

pub(crate) struct Translator<'a> {
    pub(crate) parsed: &'a ParsedPT2,
    pub(crate) graph: Graph,
    /// Maps tensor name -> GraphTensor
    pub(crate) tensors: HashMap<String, GraphTensor>,
    pub(crate) sym_map: SymDimMap,
    pub(crate) user_input_ids: Vec<(String, NodeIndex)>,
    pub(crate) output_ids: Vec<(String, NodeIndex)>,
    /// Extra tensor metadata from inlined subgraphs.
    pub(crate) extra_tensor_values: HashMap<String, TensorMeta>,
}

impl<'a> Translator<'a> {
    fn new(parsed: &'a ParsedPT2) -> Result<Self> {
        let sym_map = parsed.build_sym_dim_map();
        Ok(Self {
            parsed,
            graph: Graph::new(),
            tensors: HashMap::new(),
            sym_map,
            user_input_ids: Vec::new(),
            output_ids: Vec::new(),
            extra_tensor_values: HashMap::new(),
        })
    }

    fn translate_graph(&mut self) -> Result<()> {
        self.create_inputs()?;

        let nodes = &self.parsed.program.graph_module.graph.nodes;
        for (i, node) in nodes.iter().enumerate() {
            self.translate_node(node)
                .with_context(|| format!("Failed to translate node {i}: {}", node.target))?;
        }

        let output_names = self.parsed.output_names();
        for name in &output_names {
            let tensor = self.get_tensor(name)?;
            // Cast non-float outputs (Bool, Int) to F32 for the runtime.
            // Preserve F16/BF16/F32 as-is to avoid corrupting half-precision models.
            let tensor = match tensor.dtype {
                DType::Bool | DType::Int => tensor.cast(DType::F32) + 0.0,
                _ => tensor + 0.0,
            };
            tensor.output();
            self.output_ids.push((name.clone(), tensor.id));
        }

        Ok(())
    }

    fn create_inputs(&mut self) -> Result<()> {
        let inputs = self.parsed.classify_inputs();
        for input in &inputs {
            match input {
                InputKind::Parameter {
                    graph_name,
                    original_name,
                } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for param {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self
                        .graph
                        .named_tensor(original_name, shape)
                        .as_dtype(dtype);
                    tensor.persist();
                    self.tensors.insert(graph_name.clone(), tensor);
                }
                InputKind::Buffer {
                    graph_name,
                    original_name,
                } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for buffer {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self
                        .graph
                        .named_tensor(original_name, shape)
                        .as_dtype(dtype);
                    tensor.persist();
                    self.tensors.insert(graph_name.clone(), tensor);
                }
                InputKind::UserInput { graph_name } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for input {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    let dtype = pt2_util::torch_dtype_int_to_luminal(meta.dtype);
                    let tensor = self.graph.named_tensor(graph_name, shape).as_dtype(dtype);
                    self.user_input_ids.push((graph_name.clone(), tensor.id));
                    self.tensors.insert(graph_name.clone(), tensor);
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> TranslatedGraph {
        TranslatedGraph {
            graph: self.graph,
            user_input_ids: self.user_input_ids,
            output_ids: self.output_ids,
            sym_map: self.sym_map,
        }
    }

    // --- Helper methods ---

    pub(crate) fn get_tensor(&self, name: &str) -> Result<GraphTensor> {
        self.tensors
            .get(name)
            .copied()
            .with_context(|| format!("Unknown tensor: {name}"))
    }

    pub(crate) fn get_input_tensor(&self, node: &Node, idx: usize) -> Result<GraphTensor> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        let name = arg.as_tensor_name().with_context(|| {
            format!("Input {idx} of {} is not a tensor: {:?}", node.target, arg)
        })?;
        self.get_tensor(name)
    }

    pub(crate) fn get_int_arg(&self, node: &Node, idx: usize) -> Result<i64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        arg.as_int()
            .with_context(|| format!("Input {idx} of {} is not an int: {:?}", node.target, arg))
    }

    pub(crate) fn get_float_arg(&self, node: &Node, idx: usize) -> Result<f64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(f) = arg.as_float() {
            return Ok(f);
        }
        if let Some(i) = arg.as_int() {
            return Ok(i as f64);
        }
        anyhow::bail!("Input {idx} of {} is not a float: {:?}", node.target, arg)
    }

    pub(crate) fn get_ints_arg(&self, node: &Node, idx: usize) -> Result<Vec<i64>> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        arg.as_ints()
            .map(|v| v.to_vec())
            .with_context(|| format!("Input {idx} of {} is not int list: {:?}", node.target, arg))
    }

    pub(crate) fn get_expr_arg(&self, node: &Node, idx: usize) -> Result<Expression> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        self.resolve_arg_as_expression(arg).with_context(|| {
            format!(
                "Input {idx} of {} cannot be resolved to Expression: {:?}",
                node.target, arg
            )
        })
    }

    pub(crate) fn get_exprs_arg(&self, node: &Node, idx: usize) -> Result<Vec<Expression>> {
        use crate::pt2_schema::SymIntEntry;
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(ints) = arg.as_ints() {
            return Ok(ints.iter().map(|&v| Expression::from(v as usize)).collect());
        }
        if let Some(entries) = arg.as_sym_ints() {
            return entries
                .iter()
                .map(|entry| match entry {
                    SymIntEntry::Int(i) => Ok(Expression::from(i.as_int as usize)),
                    SymIntEntry::Name(s) => self
                        .resolve_sym_int(&s.as_name)
                        .with_context(|| format!("Cannot resolve sym_int: {}", s.as_name)),
                })
                .collect();
        }
        anyhow::bail!(
            "Input {idx} of {} is not int list or sym_int list: {:?}",
            node.target,
            arg
        )
    }

    pub(crate) fn get_bool_arg(&self, node: &Node, idx: usize) -> Result<bool> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        arg.as_bool()
            .with_context(|| format!("Input {idx} of {} is not a bool: {:?}", node.target, arg))
    }

    pub(crate) fn tensor_meta_to_shape(&self, meta: &TensorMeta) -> Result<Vec<Expression>> {
        meta.sizes
            .iter()
            .map(|s| self.dim_size_to_expr(s))
            .collect()
    }

    pub(crate) fn dim_size_to_expr(&self, dim: &DimSize) -> Result<Expression> {
        match dim {
            DimSize::Int(i) => Ok(Expression::from(i.as_int as usize)),
            DimSize::Expr(e) => {
                let sym_name = crate::pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str)
                    .with_context(|| format!("Cannot parse symbol: {}", e.as_expr.expr_str))?;
                let c = self
                    .sym_map
                    .sym_to_char
                    .get(&sym_name)
                    .with_context(|| format!("Unknown symbol: {sym_name}"))?;
                Ok(Expression::from(*c))
            }
        }
    }

    pub(crate) fn resolve_sym_int(&self, name: &str) -> Option<Expression> {
        let sym_int_values = &self.parsed.program.graph_module.graph.sym_int_values;
        if let Some(val) = sym_int_values.get(name) {
            if let Some(expr_str) = val
                .get("as_expr")
                .and_then(|e| e.get("expr_str"))
                .and_then(|s| s.as_str())
                && let Some(sym) = crate::pt2_parser::extract_symbol_name_pub(expr_str)
                && let Some(&c) = self.sym_map.sym_to_char.get(&sym)
            {
                return Some(Expression::from(c));
            }
            if let Some(hint) = val
                .get("as_expr")
                .and_then(|e| e.get("hint"))
                .and_then(|h| h.get("as_int"))
                .and_then(|v| v.as_i64())
            {
                return Some(Expression::from(hint as usize));
            }
        }
        None
    }

    pub(crate) fn resolve_arg_as_expression(&self, arg: &Argument) -> Option<Expression> {
        if let Some(v) = arg.as_int() {
            return Some(Expression::from(v as usize));
        }
        if let Some(name) = arg.as_sym_int_name() {
            return self.resolve_sym_int(name);
        }
        if let Argument::Expr(e) = arg {
            if let Some(sym) = crate::pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str)
                && let Some(&c) = self.sym_map.sym_to_char.get(&sym)
            {
                return Some(Expression::from(c));
            }
            if let Some(hint) = e.as_expr.hint.as_ref().and_then(|h| h.as_int()) {
                return Some(Expression::from(hint as usize));
            }
        }
        None
    }
}
