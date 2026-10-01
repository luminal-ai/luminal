//! The PT2 fixture builders the translator's tests state their programs
//! with: one `ParsedPT2` per test, assembled from tensor metadata and a
//! node list.

use std::collections::HashMap;

use crate::pt2_parser::ParsedPT2;
use crate::pt2_schema::{
    Argument, BoolArg, DimInt, DimSize, ExportedProgram, FloatArg, Graph, GraphModule, IntArg,
    IntsArg, Node, NodeInput, ScalarTypeArg, Signature, TensorArg, TensorMeta, TensorName,
    TensorRef,
};
use crate::translate::Translation;

pub(crate) const INT: u32 = 4;
pub(crate) const HALF: u32 = 6;
pub(crate) const FLOAT: u32 = 7;
pub(crate) const BOOL: u32 = 12;
pub(crate) const BFLOAT16: u32 = 13;

pub(crate) fn tensor_ref(name: &str) -> TensorRef {
    TensorRef {
        as_tensor: Some(TensorName {
            name: name.to_string(),
        }),
        as_tensors: None,
        as_sym_int: None,
        as_sym_float: None,
        as_sym_bool: None,
    }
}

pub(crate) fn tensor_arg(name: &str) -> Argument {
    Argument::Tensor(TensorArg {
        as_tensor: TensorName {
            name: name.to_string(),
        },
    })
}

pub(crate) fn input(name: &str, arg: Argument) -> NodeInput {
    NodeInput {
        name: name.to_string(),
        arg,
        kind: 1,
    }
}

pub(crate) fn tensor_input(name: &str, value: &str) -> NodeInput {
    input(name, tensor_arg(value))
}

pub(crate) fn scalar_float(name: &str, value: f64) -> NodeInput {
    input(name, Argument::Float(FloatArg { as_float: value }))
}

pub(crate) fn scalar_int(name: &str, value: i64) -> NodeInput {
    input(name, Argument::Int(IntArg { as_int: value }))
}

pub(crate) fn scalar_bool(name: &str, value: bool) -> NodeInput {
    input(name, Argument::Bool(BoolArg { as_bool: value }))
}

pub(crate) fn scalar_type(name: &str, code: u32) -> NodeInput {
    input(
        name,
        Argument::ScalarType(ScalarTypeArg {
            as_scalar_type: code,
        }),
    )
}

pub(crate) fn none(name: &str) -> NodeInput {
    input(name, Argument::Other(serde_json::Value::Null))
}

pub(crate) fn ints(name: &str, values: &[i64]) -> NodeInput {
    input(
        name,
        Argument::Ints(IntsArg {
            as_ints: values.to_vec(),
        }),
    )
}

pub(crate) fn sizes(values: &[i64]) -> Vec<DimSize> {
    values
        .iter()
        .map(|value| DimSize::Int(DimInt { as_int: *value }))
        .collect()
}

pub(crate) fn node(target: &str, inputs: Vec<NodeInput>, outputs: &[&str]) -> Node {
    Node {
        target: target.to_string(),
        inputs,
        outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
    }
}

/// Translate one node over the declared tensors. Every tensor operand is a
/// graph input; every declared output is a graph output, in order.
pub(crate) fn translate_one(
    node: Node,
    tensors: &[(&str, u32, &[i64])],
    inputs: &[&str],
    outputs: &[&str],
) -> Translation {
    translate_nodes(vec![node], tensors, inputs, outputs)
}

/// Translate a whole node list over the declared tensors, in the order
/// given.
pub(crate) fn translate_nodes(
    nodes: Vec<Node>,
    tensors: &[(&str, u32, &[i64])],
    inputs: &[&str],
    outputs: &[&str],
) -> Translation {
    crate::translate::translate(&parsed_nodes(nodes, tensors, inputs, outputs))
        .expect("translation failed")
}

/// Build a program without translating it, for symbolic metadata and error tests.
pub(crate) fn parsed_nodes(
    nodes: Vec<Node>,
    tensors: &[(&str, u32, &[i64])],
    inputs: &[&str],
    outputs: &[&str],
) -> ParsedPT2 {
    let mut tensor_values = HashMap::new();
    for (name, dtype, shape) in tensors {
        tensor_values.insert(
            name.to_string(),
            TensorMeta {
                dtype: *dtype,
                sizes: sizes(shape),
                layout: None,
                device: None,
            },
        );
    }
    let program = ExportedProgram {
        graph_module: GraphModule {
            graph: Graph {
                inputs: inputs.iter().map(|name| tensor_ref(name)).collect(),
                outputs: outputs.iter().map(|name| tensor_ref(name)).collect(),
                nodes,
                tensor_values,
                sym_int_values: HashMap::new(),
            },
            signature: Signature {
                input_specs: Vec::new(),
                output_specs: Vec::new(),
            },
        },
        range_constraints: HashMap::new(),
    };
    ParsedPT2 {
        program,
        constants_config: None,
        weights_config: None,
        archive_prefix: "test".to_string(),
        pt2_path: String::new(),
    }
}
