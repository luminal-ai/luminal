//! PT2 serialized model JSON schema types (torch 2.10+ format).
//!
//! The .pt2 ZIP archive contains `{name}/models/model.json` with this structure.
//! We only model the subset needed for graph translation.

use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ExportedProgram {
    pub graph_module: GraphModule,
    #[serde(default)]
    pub range_constraints: HashMap<String, RangeConstraint>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RangeConstraint {
    pub min_val: i64,
}

#[derive(Debug, Deserialize)]
pub struct GraphModule {
    pub graph: Graph,
    pub signature: Signature,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Graph {
    pub inputs: Vec<TensorRef>,
    pub outputs: Vec<TensorRef>,
    pub nodes: Vec<Node>,
    pub tensor_values: HashMap<String, TensorMeta>,
    #[serde(default)]
    pub sym_int_values: HashMap<String, serde_json::Value>,
}

/// A reference to a tensor by name (used in graph inputs/outputs).
/// Single-output nodes use `as_tensor`, multi-output nodes (split, topk) use `as_tensors`.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorRef {
    pub as_tensor: Option<TensorName>,
    #[serde(default)]
    pub as_tensors: Option<Vec<TensorName>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorName {
    pub name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub target: String,
    pub inputs: Vec<NodeInput>,
    pub outputs: Vec<TensorRef>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeInput {
    pub name: String,
    pub arg: Argument,
    /// 1 = positional, 2 = keyword (not formally documented, but observed)
    #[serde(default)]
    pub kind: u32,
}

/// A node argument — one of several typed variants.
/// ORDER MATTERS for #[serde(untagged)]: more specific variants must come first.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Argument {
    Tensor(TensorArg),
    Int(IntArg),
    Float(FloatArg),
    Bool(BoolArg),
    Ints(IntsArg),
    SymInts(SymIntsArg),
    SymInt(SymIntArg),
    Expr(ExprArg),
    #[allow(dead_code)]
    ScalarType(ScalarTypeArg),
    Tensors(TensorsArg),
    OptionalTensors(OptionalTensorsArg),
    Graph(GraphArg),
    /// Fallback for anything we don't handle (Floats, Str, Layout,
    /// OptionalTensor, None, Device, etc.)
    #[allow(dead_code)]
    Other(serde_json::Value),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorArg {
    pub as_tensor: TensorName,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntArg {
    pub as_int: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FloatArg {
    pub as_float: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BoolArg {
    pub as_bool: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntsArg {
    pub as_ints: Vec<i64>,
}

/// An entry in an optional_tensors list — either a tensor ref or None.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OptionalTensorEntry {
    Tensor(TensorArg),
    #[allow(dead_code)] // NoneArg needed as serde discriminant for untagged enum
    None(NoneArg),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptionalTensorsArg {
    pub as_optional_tensors: Vec<OptionalTensorEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymIntsArg {
    pub as_sym_ints: Vec<SymIntEntry>,
}

/// An entry in a sym_ints list — either a concrete int or a symbolic name reference.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SymIntEntry {
    Int(IntArg),
    Name(SymIntValue),
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymIntArg {
    pub as_sym_int: SymIntValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SymIntValue {
    pub as_name: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExprArg {
    pub as_expr: ExprValue,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExprValue {
    pub expr_str: String,
    pub hint: Option<Box<Argument>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoneArg {
    #[allow(dead_code)] // Serde discriminating key for OptionalTensorEntry untagged enum
    pub as_none: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ScalarTypeArg {
    pub as_scalar_type: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorsArg {
    pub as_tensors: Vec<TensorName>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphArg {
    pub as_graph: SubGraph,
}

/// A subgraph embedded in a higher-order op (e.g. wrap_with_set_grad_enabled).
#[derive(Debug, Clone, Deserialize)]
pub struct SubGraph {
    pub graph: Graph,
}

impl Argument {
    pub fn as_tensor_name(&self) -> Option<&str> {
        match self {
            Argument::Tensor(t) => Some(&t.as_tensor.name),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Argument::Int(i) => Some(i.as_int),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Argument::Float(f) => Some(f.as_float),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Argument::Bool(b) => Some(b.as_bool),
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Argument::Ints(i) => Some(&i.as_ints),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn as_scalar_type(&self) -> Option<u32> {
        match self {
            Argument::ScalarType(s) => Some(s.as_scalar_type),
            _ => None,
        }
    }

    pub fn as_tensors(&self) -> Option<&[TensorName]> {
        match self {
            Argument::Tensors(t) => Some(&t.as_tensors),
            _ => None,
        }
    }

    pub fn as_graph(&self) -> Option<&SubGraph> {
        match self {
            Argument::Graph(g) => Some(&g.as_graph),
            _ => None,
        }
    }

    pub fn as_sym_int_name(&self) -> Option<&str> {
        match self {
            Argument::SymInt(s) => Some(&s.as_sym_int.as_name),
            _ => None,
        }
    }

    pub fn as_sym_ints(&self) -> Option<&[SymIntEntry]> {
        match self {
            Argument::SymInts(s) => Some(&s.as_sym_ints),
            _ => None,
        }
    }

    pub fn as_optional_tensors(&self) -> Option<&[OptionalTensorEntry]> {
        match self {
            Argument::OptionalTensors(t) => Some(&t.as_optional_tensors),
            _ => None,
        }
    }
}

/// Tensor metadata (shape, dtype, strides).
#[derive(Debug, Clone, Deserialize)]
pub struct TensorMeta {
    pub dtype: u32,
    pub sizes: Vec<DimSize>,
}

/// A dimension size — either a concrete integer or a symbolic expression.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DimSize {
    Int(DimInt),
    Expr(DimExpr),
}

#[derive(Debug, Clone, Deserialize)]
pub struct DimInt {
    pub as_int: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DimExpr {
    pub as_expr: ExprValue,
}

impl DimSize {
    pub fn symbol_name(&self) -> Option<&str> {
        match self {
            DimSize::Expr(e) => Some(&e.as_expr.expr_str),
            DimSize::Int(_) => None,
        }
    }

    pub fn hint(&self) -> Option<i64> {
        match self {
            DimSize::Expr(e) => e.as_expr.hint.as_ref().and_then(|h| h.as_int()),
            DimSize::Int(i) => Some(i.as_int),
        }
    }
}

/// Signature describing which inputs are parameters vs user inputs.
#[derive(Debug, Deserialize)]
pub struct Signature {
    pub input_specs: Vec<InputSpec>,
}

/// An input spec — tagged enum via JSON key.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputSpec {
    Parameter(ParameterInput),
    Buffer(BufferInput),
    TensorConstant(TensorConstantInput),
    UserInput(UserInputSpec),
    #[allow(dead_code)] // Serde catch-all for untagged enum
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize)]
pub struct ParameterInput {
    pub parameter: ParameterDetail,
}

#[derive(Debug, Deserialize)]
pub struct ParameterDetail {
    pub arg: ParameterArg,
    pub parameter_name: String,
}

#[derive(Debug, Deserialize)]
pub struct ParameterArg {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct BufferInput {
    pub buffer: BufferDetail,
}

#[derive(Debug, Deserialize)]
pub struct BufferDetail {
    pub arg: ParameterArg,
    pub buffer_name: String,
}

#[derive(Debug, Deserialize)]
pub struct TensorConstantInput {
    pub tensor_constant: TensorConstantDetail,
}

#[derive(Debug, Deserialize)]
pub struct TensorConstantDetail {
    pub arg: ParameterArg,
    pub tensor_constant_name: String,
}

#[derive(Debug, Deserialize)]
pub struct UserInputSpec {
    pub user_input: UserInputDetail,
}

#[derive(Debug, Deserialize)]
pub struct UserInputDetail {
    pub arg: Argument,
}

/// Weights configuration from model_weights_config.json.
#[derive(Debug, Deserialize)]
pub struct WeightsConfig {
    pub config: HashMap<String, WeightEntry>,
}

#[derive(Debug, Deserialize)]
pub struct WeightEntry {
    pub path_name: String,
    pub tensor_meta: TensorMeta,
}
