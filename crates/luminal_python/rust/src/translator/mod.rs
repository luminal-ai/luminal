//! PT2 graph nodes -> Luminal Graph translation.
//!
//! Walks the parsed PT2 graph and constructs an equivalent Luminal computation graph.

mod attention;
mod binary;
mod complex;
mod conv;
mod dispatch;
mod movement;
mod movement_dynamic;
mod reduction;
mod tensor;
mod unary;

use std::collections::HashMap;

use anyhow::{Context, Result};
use luminal::graph::Graph;
use luminal::prelude::*;

use crate::pt2_expr::parse_sympy_expr_with_ranges;
use crate::pt2_parser::{InputKind, ParsedPT2, SymDimMap};
use crate::pt2_schema::*;
use crate::pt2_util;
use crate::torch_dtype::TorchDType;

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
    /// Maps PT2 tensor and scalar value names to GraphTensors. Scalars are
    /// represented as typed zero-dimensional tensors.
    pub(crate) tensors: HashMap<String, GraphTensor>,
    /// Complex PT2 values represented as pairs of ordinary real tensors.
    /// Complex never becomes an HLIR dtype.
    pub(crate) complex_tensors: HashMap<String, complex::ComplexTensor>,
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
            complex_tensors: HashMap::new(),
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
            let tensor = if let Some(complex) = self.complex_tensors.get(name).copied() {
                complex.pack(&mut self.graph)
            } else {
                self.get_tensor(name)?
            };
            let tensor = if tensor.dtype == DType::Bool {
                tensor.cast(DType::Int).cast(DType::Bool)
            } else if tensor.dtype == DType::Int || tensor.shape.is_contiguous() {
                tensor
            } else {
                tensor + 0.0
            };
            // `output()` may materialize a non-contiguous view with a Gather.
            // Record that returned node, not the pre-materialization producer.
            let output = tensor.output();
            self.output_ids.push((name.clone(), output.id));
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
                    self.create_input_value(graph_name, original_name, shape, meta.dtype, false)?;
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
                    self.create_input_value(graph_name, original_name, shape, meta.dtype, false)?;
                }
                InputKind::UserInput { graph_name } => {
                    let meta = self
                        .parsed
                        .tensor_meta(graph_name)
                        .with_context(|| format!("Missing tensor meta for input {graph_name}"))?;
                    let shape = self.tensor_meta_to_shape(meta)?;
                    self.create_input_value(graph_name, graph_name, shape, meta.dtype, true)?;
                }
            }
        }
        Ok(())
    }

    fn create_input_value(
        &mut self,
        graph_name: &str,
        label: &str,
        mut logical_shape: Vec<Expression>,
        dtype_code: u32,
        user_input: bool,
    ) -> Result<()> {
        let torch_dtype = TorchDType::from_code(dtype_code)
            .map_err(|code| anyhow::anyhow!("Unknown PT2 dtype code {code} for {graph_name}"))?;
        if let Some(component_dtype) = torch_dtype.complex_component_dtype() {
            // PyTorch complex storage is interleaved real/imaginary values.
            // Preserve that layout in one real-valued input and split it into
            // the two frontend-only components before translating operators.
            logical_shape.push(Expression::from(2usize));
            let backing = self
                .graph
                .named_tensor(label, logical_shape)
                .as_dtype(component_dtype);
            backing.persist();
            let complex =
                complex::ComplexTensor::from_interleaved(&mut self.graph, backing, torch_dtype)?;
            if user_input {
                self.user_input_ids
                    .push((graph_name.to_string(), backing.id));
            }
            self.complex_tensors.insert(graph_name.to_string(), complex);
        } else {
            let dtype = pt2_util::torch_dtype_int_to_luminal(dtype_code);
            let tensor = self
                .graph
                .named_tensor(label, logical_shape)
                .as_dtype(dtype);
            if !user_input {
                tensor.persist();
            }
            if user_input {
                self.user_input_ids
                    .push((graph_name.to_string(), tensor.id));
            }
            self.tensors.insert(graph_name.to_string(), tensor);
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

    pub(crate) fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.extra_tensor_values
            .get(name)
            .or_else(|| self.parsed.tensor_meta(name))
    }

    pub(crate) fn get_tensor(&self, name: &str) -> Result<GraphTensor> {
        if self.complex_tensors.contains_key(name) {
            anyhow::bail!(
                "Complex tensor {name} reached a real-only lowering; add a frontend complex lowering"
            );
        }
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
        let name = arg.as_value_name().with_context(|| {
            format!(
                "Input {idx} of {} is not a tensor-backed value: {:?}",
                node.target, arg
            )
        })?;
        self.get_tensor(name)
    }

    pub(crate) fn get_int_arg(&self, node: &Node, idx: usize) -> Result<i64> {
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        if let Some(v) = arg.as_int() {
            return Ok(v);
        }
        // Fall through to symbolic-aware resolution. Op-arg slots like `dim`
        // and `axis` are always concrete in practice, but with dynamic shapes
        // PT2 occasionally hands us a SymInt that is fully bound at export
        // time (e.g. an `unsqueeze` whose dim was derived from `len(shape)`);
        // accept those when they reduce to a concrete int rather than failing
        // with the misleading "not an int" diagnostic.
        if let Some(expr) = self.resolve_arg_as_expression(arg)
            && let Some(v) = expr.to_usize()
        {
            return Ok(v as i64);
        }
        anyhow::bail!("Input {idx} of {} is not an int: {:?}", node.target, arg)
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
        use crate::pt2_schema::SymIntEntry;
        let arg = &node
            .inputs
            .get(idx)
            .with_context(|| format!("Node {} missing input {idx}", node.target))?
            .arg;
        // Symbolic int lists: tolerate them as long as every entry is a
        // bound concrete value. Prevents false "not an int list" failures on
        // graphs where torch.export emits sym_ints for what is dimensionally
        // a static parameter (kernel sizes, etc. with dynamic batch).
        if let Some(entries) = arg.as_sym_ints() {
            let mut out = Vec::with_capacity(entries.len());
            for entry in entries {
                let v = match entry {
                    SymIntEntry::Int(i) => Some(i.as_int),
                    SymIntEntry::Name(s) => self
                        .resolve_sym_int(&s.as_name)
                        .and_then(|e| e.to_usize().map(|u| u as i64)),
                };
                match v {
                    Some(n) => out.push(n),
                    None => {
                        anyhow::bail!(
                            "Input {idx} of {} contains an unresolved sym_int entry",
                            node.target
                        )
                    }
                }
            }
            return Ok(out);
        }
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
            return Ok(ints.iter().map(|&v| Expression::from(v)).collect());
        }
        if let Some(entries) = arg.as_sym_ints() {
            return entries
                .iter()
                .map(|entry| match entry {
                    SymIntEntry::Int(i) => Ok(Expression::from(i.as_int)),
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

    /// Shape of the node's first output, from its tensor metadata (for ops
    /// whose output size isn't reliably readable from the args).
    pub(crate) fn output_meta_shape(&self, node: &Node) -> Result<Vec<Expression>> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();

        let meta = self
            .tensor_meta(&name)
            .context("Missing tensor meta for output shape")?;
        self.tensor_meta_to_shape(meta)
    }

    /// Dtype of the node's first tensor output. PT2 metadata is authoritative
    /// for PyTorch promotion semantics, including integral inputs promoted to
    /// the current default floating dtype by transcendental and true-divide
    /// operations.
    pub(crate) fn output_meta_dtype(&self, node: &Node) -> Result<DType> {
        let name = node
            .outputs
            .first()
            .and_then(|output| output.as_tensor.as_ref())
            .map(|tensor| tensor.name.clone())
            .unwrap_or_default();

        let meta = self
            .tensor_meta(&name)
            .context("Missing tensor meta for output dtype")?;
        Ok(pt2_util::torch_dtype_int_to_luminal(meta.dtype))
    }

    pub(crate) fn dim_size_to_expr(&self, dim: &DimSize) -> Result<Expression> {
        match dim {
            DimSize::Int(i) => Ok(Expression::from(i.as_int)),
            DimSize::Expr(e) => self.resolve_expr_value(&e.as_expr).with_context(|| {
                format!(
                    "Cannot resolve symbolic dimension expression: {}",
                    e.as_expr.expr_str
                )
            }),
        }
    }

    pub(crate) fn resolve_sym_int(&self, name: &str) -> Option<Expression> {
        let sym_int_values = &self.parsed.program.graph_module.graph.sym_int_values;
        if let Some(val) = sym_int_values.get(name) {
            if let Some(expr_str) = val
                .get("as_expr")
                .and_then(|e| e.get("expr_str"))
                .and_then(|s| s.as_str())
                && let Some(expr) = self.resolve_expr_str(expr_str)
            {
                return Some(expr);
            }
            if let Some(hint) = val
                .get("as_expr")
                .and_then(|e| e.get("hint"))
                .and_then(|h| h.get("as_int"))
                .and_then(|v| v.as_i64())
            {
                return Some(Expression::from(hint));
            }
        }
        None
    }

    pub(crate) fn resolve_arg_as_expression(&self, arg: &Argument) -> Option<Expression> {
        if let Some(v) = arg.as_int() {
            return Some(Expression::from(v));
        }
        if let Some(name) = arg.as_sym_int_name() {
            return self.resolve_sym_int(name);
        }
        if let Argument::Expr(e) = arg {
            return self.resolve_expr_value(&e.as_expr);
        }
        None
    }

    pub(crate) fn resolve_expr_str(&self, expr_str: &str) -> Option<Expression> {
        parse_sympy_expr_with_ranges(expr_str, &self.sym_map.sym_to_symbol, &self.sym_map.ranges)
            .or_else(|| {
                crate::pt2_parser::extract_symbol_name_pub(expr_str)
                    .and_then(|sym| self.sym_map.sym_to_symbol.get(&sym).copied())
                    .map(Expression::from)
            })
    }

    pub(crate) fn resolve_expr_value(&self, expr: &ExprValue) -> Option<Expression> {
        self.resolve_expr_str(&expr.expr_str).or_else(|| {
            expr.hint
                .as_ref()
                .and_then(|h| h.as_int())
                .map(Expression::from)
        })
    }
}
