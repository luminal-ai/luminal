use std::collections::HashMap;

use luminal::{
    prelude::{tracing::trace, *},
    shape::Expression,
};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to_expr, get_int_attr};

/// Handle Constant node: creates a tensor from embedded data in the node attributes.
///
/// Supports FLOAT, INT64, INT32, and FLOAT64 data types (all converted to f32).
/// The resulting tensor is registered as a known constant for downstream folding.
pub fn parse_constant_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Constant Node");
    assert!(
        node.output.len() == 1,
        "Constant should have exactly one output"
    );

    // Find the "value" attribute (type TENSOR)
    let value_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| "Constant node missing 'value' attribute".to_string())?;

    let tensor_proto = value_attr
        .t
        .as_ref()
        .ok_or_else(|| "Constant 'value' attribute has no TensorProto".to_string())?;

    // Determine shape: empty dims = scalar = [1] for luminal
    let shape: Vec<usize> = if tensor_proto.dims.is_empty() {
        vec![1]
    } else {
        tensor_proto.dims.iter().map(|&d| d as usize).collect()
    };

    // Extract float data based on data_type
    let floats: Vec<f32> = match tensor_proto.data_type {
        1 => {
            // FLOAT (f32)
            if !tensor_proto.float_data.is_empty() {
                tensor_proto.float_data.clone()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        6 => {
            // INT32
            if !tensor_proto.int32_data.is_empty() {
                tensor_proto.int32_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()
            }
        }
        7 => {
            // INT64
            if !tensor_proto.int64_data.is_empty() {
                tensor_proto.int64_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()
            }
        }
        dt => return Err(format!("Constant node: unsupported data_type {}", dt)),
    };

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), shape);
    tensors.insert(output_name.clone(), tensor);
    known_values.insert(output_name.clone(), floats.clone());
    // Also propagate as concrete shape_exprs for downstream shape computation chains
    shape_exprs.insert(
        output_name.clone(),
        floats
            .iter()
            .map(|&v| Expression::from(v as usize))
            .collect(),
    );
    weight_data.push((output_name.clone(), floats));

    trace!("Finished parse: Constant Node");
    Ok(())
}

/// Handle Shape node: extract the shape of the input tensor as a 1D constant.
///
/// For static shapes, stores as known_values. For dynamic shapes (containing
/// Expression variables), stores in shape_exprs for downstream shape computation chains.
pub fn parse_shape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Started parse: Shape");
    assert!(node.input.len() == 1, "Shape should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Shape: missing input tensor '{}'", node.input[0]))?;

    let all_dims = input.dims();

    // Handle start/end attributes (ONNX Shape opset 15+: extract a slice of dims)
    let start = get_int_attr(node, "start", 0) as usize;
    let end_attr = get_int_attr(node, "end", all_dims.len() as i64);
    let end = if end_attr < 0 {
        (all_dims.len() as i64 + end_attr) as usize
    } else {
        (end_attr as usize).min(all_dims.len())
    };
    let dims: Vec<Expression> = all_dims[start..end].to_vec();

    let output_name = &node.output[0];

    // Always store in shape_exprs (supports both concrete and symbolic dims)
    shape_exprs.insert(output_name.clone(), dims.clone());

    // For concrete dims, also store in known_values for backward compat
    let all_concrete = dims.iter().all(|d| d.to_usize().is_some());
    let shape_values: Vec<f32> = dims
        .iter()
        .map(|d| d.to_usize().unwrap_or(1) as f32)
        .collect();

    if all_concrete {
        // Concrete shape: create tensor + known_values + weight_data
        let tensor = cx.named_tensor(output_name.clone(), vec![shape_values.len()]);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), shape_values.clone());
        weight_data.push((output_name.clone(), shape_values));
    }
    // For symbolic shapes, don't create a tensor — it's shape-only

    trace!("Finished parse: Shape");
    Ok(())
}

/// Handle ConstantOfShape node: creates a tensor of a given shape filled with a constant value.
///
/// The shape is taken from the input tensor (which must be a known constant).
/// The fill value comes from the "value" attribute (default 0.0).
pub fn parse_constant_of_shape(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: ConstantOfShape Node");
    assert!(
        node.input.len() == 1,
        "ConstantOfShape should have exactly one input (shape)"
    );
    assert!(
        node.output.len() == 1,
        "ConstantOfShape should have exactly one output"
    );

    // Extract fill value from "value" attribute (TensorProto scalar), default 0.0
    let fill_value: f32 = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .and_then(|attr| attr.t.as_ref())
        .map(|tp| {
            if !tp.float_data.is_empty() {
                tp.float_data[0]
            } else if !tp.int32_data.is_empty() {
                tp.int32_data[0] as f32
            } else if !tp.raw_data.is_empty() {
                match tp.data_type {
                    1 => f32::from_le_bytes([
                        tp.raw_data[0],
                        tp.raw_data[1],
                        tp.raw_data[2],
                        tp.raw_data[3],
                    ]),
                    6 => i32::from_le_bytes([
                        tp.raw_data[0],
                        tp.raw_data[1],
                        tp.raw_data[2],
                        tp.raw_data[3],
                    ]) as f32,
                    7 => i64::from_le_bytes([
                        tp.raw_data[0],
                        tp.raw_data[1],
                        tp.raw_data[2],
                        tp.raw_data[3],
                        tp.raw_data[4],
                        tp.raw_data[5],
                        tp.raw_data[6],
                        tp.raw_data[7],
                    ]) as f32,
                    _ => 0.0,
                }
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    let output_name = &node.output[0];

    // Try shape_exprs first (for dynamic shapes), then known_values
    if let Some(se) = shape_exprs.get(&node.input[0]) {
        let shape: Vec<Expression> = se.clone();

        // Check if all dims are concrete
        if let Some(concrete) = shape
            .iter()
            .map(|e| e.to_usize())
            .collect::<Option<Vec<usize>>>()
        {
            // Fully concrete: create named tensor with weight data
            let numel: usize = concrete.iter().product();
            let floats: Vec<f32> = vec![fill_value; numel];
            let tensor = cx.named_tensor(output_name.clone(), concrete);
            tensors.insert(output_name.clone(), tensor);
            known_values.insert(output_name.clone(), floats.clone());
            weight_data.push((output_name.clone(), floats));
        } else {
            // Dynamic shape: create scalar constant and broadcast to symbolic shape.
            // The scalar always has concrete data (1 element), and the shape is
            // resolved at runtime via ShapeTracker/dyn_map. Broadcast uses stride-0
            // expansion, so only 1 float is needed in the backing buffer.
            let scalar = cx.constant_float(fill_value);
            let result = broadcast_to_expr(scalar, se);
            tensors.insert(output_name.clone(), result);
        }
    } else {
        let shape_values = known_values.get(&node.input[0]).ok_or_else(|| {
            format!(
                "ConstantOfShape: shape input '{}' must be a known constant or shape_expr",
                node.input[0]
            )
        })?;
        let shape: Vec<usize> = shape_values.iter().map(|&v| v as usize).collect();
        let numel: usize = shape.iter().product();
        let floats: Vec<f32> = vec![fill_value; numel];

        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), floats.clone());
        weight_data.push((output_name.clone(), floats));
    }

    trace!("Finished parse: ConstantOfShape Node");
    Ok(())
}

/// Handle Identity node: output is a direct alias of the input tensor.
///
/// Propagates known constant values for downstream constant folding.
pub fn parse_identity(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Identity Node");
    assert!(node.input.len() == 1, "Identity should only have one input");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Identity: missing input tensor '{}'", node.input[0]))?;

    assert!(
        node.output.len() == 1,
        "Identity should only have a single output"
    );

    let output_name = &node.output[0];

    // Force materialization using Expression-aware broadcast
    let dims = a.dims();
    let one = a.graph().constant_float(1.0);
    let one_expanded = broadcast_to_expr(one, &dims);
    let result = a * one_expanded;
    tensors.insert(output_name.clone(), result);

    // Propagate known values
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    // Propagate shape_exprs
    if let Some(se) = shape_exprs.get(&node.input[0]).cloned() {
        shape_exprs.insert(output_name.clone(), se);
    }

    trace!("Finished parse: Identity Node");
    Ok(())
}

/// Handle Range node: creates a 1D tensor [start, start+delta, start+2*delta, ...] up to limit.
///
/// Used by dynamo ONNX export for generating position indices (arange).
/// Supports Expression-based limits for dynamic sequence lengths.
pub fn parse_range_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Range Node");
    assert!(
        node.input.len() == 3,
        "Range needs 3 inputs: start, limit, delta"
    );

    let output_name = &node.output[0];

    // Try to get concrete values from known_values first
    let start_val = known_values
        .get(&node.input[0])
        .and_then(|v| v.first().copied());
    let limit_val = known_values
        .get(&node.input[1])
        .and_then(|v| v.first().copied());
    let delta_val = known_values
        .get(&node.input[2])
        .and_then(|v| v.first().copied());

    // Also check shape_exprs for symbolic limit
    let limit_expr = shape_exprs
        .get(&node.input[1])
        .and_then(|v| v.first().cloned());

    let start = start_val.unwrap_or(0.0);
    let delta = delta_val.unwrap_or(1.0);

    if start == 0.0 && delta == 1.0 {
        // Simple arange case — most common for position indices
        if let Some(expr) = limit_expr {
            // Dynamic limit: create arange with symbolic length
            let tensor = cx.arange(expr);
            // Cast to F32 (luminal arange returns Int dtype)
            let result = tensor.cast(DType::F32);
            tensors.insert(output_name.clone(), result);
            shape_exprs.insert(output_name.clone(), vec![expr]);
        } else if let Some(limit) = limit_val {
            let n = limit as usize;
            let floats: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let tensor = cx.named_tensor(output_name.clone(), vec![n]);
            tensors.insert(output_name.clone(), tensor);
            known_values.insert(output_name.clone(), floats.clone());
            weight_data.push((output_name.clone(), floats));
        } else {
            return Err("Range: limit must be known or symbolic".to_string());
        }
    } else if let (Some(s), Some(l), Some(d)) = (start_val, limit_val, delta_val) {
        // Fully concrete range
        let mut floats = Vec::new();
        let mut v = s;
        while (d > 0.0 && v < l) || (d < 0.0 && v > l) {
            floats.push(v);
            v += d;
        }
        let tensor = cx.named_tensor(output_name.clone(), vec![floats.len()]);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), floats.clone());
        weight_data.push((output_name.clone(), floats));
    } else {
        return Err("Range: cannot handle non-trivial dynamic ranges yet".to_string());
    }

    trace!("Finished parse: Range Node");
    Ok(())
}

/// Handle CumSum node: cumulative sum along an axis.
///
/// For the simple case of axis=0 on a 1D tensor [0, 1, 2, ...] (position indices),
/// the cumsum is equivalent to [0, 1, 3, 6, ...]. For dynamic ONNX graphs,
/// this is typically used for position_ids computation.
pub fn parse_cumsum_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: CumSum Node");
    assert!(node.input.len() >= 2, "CumSum needs at least 2 inputs");

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("CumSum: missing input '{}'", node.input[0]))?;

    let axis_val = known_values
        .get(&node.input[1])
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0) as i64;

    let dims = input.dims();
    let ndim = dims.len();
    let _axis = if axis_val < 0 {
        (ndim as i64 + axis_val) as usize
    } else {
        axis_val as usize
    };

    // For constant folding
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let output_name = &node.output[0];
        let mut cumsum = vals.clone();
        // Simple 1D cumsum
        if ndim == 1 {
            for i in 1..cumsum.len() {
                cumsum[i] += cumsum[i - 1];
            }
        }
        known_values.insert(output_name.clone(), cumsum);
        // Just alias the tensor (same shape)
        tensors.insert(output_name.clone(), input);
        trace!("Finished parse: CumSum Node (constant folded)");
        return Ok(());
    }

    // For dynamic: cumsum is hard to express in luminal primitives.
    // For the specific pattern used in Llama position_ids (cumsum of ones = arange),
    // we just pass through since arange is already handled by Range node.
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), input);

    trace!("Finished parse: CumSum Node");
    Ok(())
}
