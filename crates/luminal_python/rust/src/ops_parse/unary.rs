use std::collections::HashMap;

use luminal::{
    prelude::{tracing::trace, *},
    shape::Expression,
};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to_expr, get_float_attr, get_int_attr};

/// Handle Softmax node: output = softmax(input[0], axis)
///
/// ONNX axis attribute defaults to -1 (last dimension, opset 13+).
/// Negative axis is normalized against the input rank.
pub fn parse_softmax_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Softmax Node");
    assert!(
        node.input.len() == 1,
        "Softmax nodes need to have one input, {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Softmax nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Softmax: missing input tensor '{}'", node.input[0]))?;

    let ndim = a.dims().len();
    let raw_axis = get_int_attr(node, "axis", -1);
    let axis = if raw_axis < 0 {
        (ndim as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };

    let result = a.softmax(axis);
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Softmax Node");
    Ok(())
}

/// Handle Not node: logical NOT — output = 1.0 - input[0]
pub fn parse_not_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Not Node");
    assert!(
        node.input.len() == 1,
        "Not nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Not nodes only have one output, {} where present",
        node.output.len()
    );
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Not: missing input tensor '{}'", node.input[0]))?;
    let a_f32 = a.cast(DType::F32);
    let result = 1.0_f32 - a_f32;
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: Not Node");
    Ok(())
}

/// Handle Clip node: output = clip(input[0], min, max)
///
/// Equivalent to torch.clamp. min and max are optional tensor inputs
/// (typically constants) residing in known_values.
pub fn parse_clip_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Clip Node");
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Clip: missing input tensor '{}'", node.input[0]))?;

    // input[1] = min (optional), input[2] = max (optional)
    let min_name = node.input.get(1).map(String::as_str).unwrap_or("");
    let max_name = node.input.get(2).map(String::as_str).unwrap_or("");

    let min_val = if min_name.is_empty() {
        None
    } else {
        known_values.get(min_name).map(|v| v[0])
    };
    let max_val = if max_name.is_empty() {
        None
    } else {
        known_values.get(max_name).map(|v| v[0])
    };

    let result = match (min_val, max_val) {
        (Some(lo), Some(hi)) => a.clip(lo, hi),
        (Some(lo), None) => a.maximum_f32(lo),
        (None, Some(hi)) => a.minimum_f32(hi),
        (None, None) => a,
    };

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Clip Node");
    Ok(())
}

/// Handle Floor node: output = floor(input[0])
///
/// Implemented as: trunc(x) - (x < trunc(x) ? 1 : 0)
/// where trunc is truncation toward zero via cast to Int then back to F32.
/// This correctly handles negative non-integer values (e.g. floor(-1.5) = -2).
pub fn parse_floor_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Floor Node");
    assert!(
        node.input.len() == 1,
        "Floor nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Floor nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Floor: missing input tensor '{}'", node.input[0]))?;

    // trunc(x): truncation toward zero
    let trunc = a.cast(DType::Int).cast(DType::F32);
    // For negative non-integers, x < trunc(x), so subtract 1
    // Cast lt result (Bool) to F32 before arithmetic
    let adjustment = a.lt(trunc).cast(DType::F32);
    let result = trunc - adjustment;
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Floor Node");

    Ok(())
}

/// Handle Ceil node: output = ceil(input[0])
///
/// Implemented as: trunc(x) + (x > trunc(x) ? 1 : 0)
/// where trunc is truncation toward zero via cast to Int then back to F32.
/// This correctly handles positive non-integer values (e.g. ceil(1.5) = 2).
pub fn parse_ceil_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Ceil Node");
    assert!(
        node.input.len() == 1,
        "Ceil nodes need to have one input {} where present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Ceil nodes only have one output, {} where present",
        node.output.len(),
    );
    let output_name = &node.output[0];
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Ceil: missing input tensor '{}'", node.input[0]))?;

    // trunc(x): truncation toward zero
    let trunc = a.cast(DType::Int).cast(DType::F32);
    // For positive non-integers, x > trunc(x), so add 1
    let adjustment = a.gt(trunc).cast(DType::F32);
    let result = trunc + adjustment;
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Ceil Node");

    Ok(())
}

pub fn parse_cast_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Cast Node");
    assert!(node.input.len() == 1, "Cast should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Cast: missing input tensor '{}'", node.input[0]))?;

    // ONNX data type enum → luminal DType
    let to = get_int_attr(node, "to", 1);
    let dtype = match to {
        1 => DType::F32,     // FLOAT
        10 => DType::F16,    // FLOAT16
        16 => DType::Bf16,   // BFLOAT16
        6 | 7 => DType::Int, // INT32, INT64
        9 => DType::F32,     // BOOL → treat as F32 (0.0/1.0)
        11 => DType::F32,    // DOUBLE → F32 (downcast)
        _ => DType::F32,     // fallback
    };

    let cast_result = input.cast(dtype);
    let output_name = &node.output[0];

    let result = if cast_result.id == input.id {
        input
    } else {
        cast_result
    };

    tensors.insert(output_name.clone(), result);

    // Propagate known values (cast is a no-op for our f32 storage)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded = if to == 9 {
            vals.iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect()
        } else if to == 6 || to == 7 {
            vals.iter().map(|&v| (v as i64) as f32).collect()
        } else {
            vals
        };
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
    }
    // Propagate shape_exprs
    if let Some(se) = shape_exprs.get(&node.input[0]).cloned() {
        shape_exprs.insert(output_name.clone(), se);
    }

    trace!("Finished parse: Cast Node");
    Ok(())
}

pub fn parse_unary_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() == 1,
        "{} should have 1 input, got {}",
        op_name,
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "{} should have 1 output, got {}",
        op_name,
        node.output.len()
    );
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;
    let result = op(a);
    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}

/// Handle Erf node: output = erf(input[0])
///
/// Uses the Abramowitz & Stegun 7.1.26 polynomial approximation (max error < 1.5e-7):
///   For x ≥ 0: erf(x) ≈ 1 - (a1·t + a2·t² + a3·t³ + a4·t⁴ + a5·t⁵) · exp(-x²)
///   where t = 1 / (1 + 0.3275911·x)
///     a1 =  0.254829592
///     a2 = -0.284496736
///     a3 =  1.421413741
///     a4 = -1.453152027
///     a5 =  1.061405429
/// Extended to all x via odd symmetry: erf(-x) = -erf(x).
pub fn parse_erf_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    parse_unary_op(node, tensors, "Erf", |x| {
        let a = x.abs();
        let t = (1.0_f32 + 0.3275911_f32 * a).reciprocal();
        // Horner evaluation of a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        // poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))))
        let h = t * 1.061_405_4_f32 - 1.453_152_1_f32; // a4 + a5*t
        let h = t * h + 1.421_413_8_f32;
        let h = t * h - 0.284_496_72_f32;
        let h = t * h + 0.254_829_6_f32;
        let poly = t * h;
        let erf_abs = 1.0_f32 - poly * (-a * a).exp();
        x.sign() * erf_abs
    })
}

/// Handle LayerNormalization node (opset 17).
///
/// Inputs: X (required), scale (required), bias (optional)
/// Attributes: axis (default -1), epsilon (default 1e-5)
/// Normalizes over axes [axis, axis+1, ..., rank-1], then applies scale and bias.
/// Only output 0 (the normalized result) is wired; outputs 1/2 (mean, inv_std_var)
/// are training-only and not supported for inference.
pub fn parse_layernorm_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: LayerNormalization Node");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("LayerNorm: missing input '{}'", node.input[0]))?;
    let scale = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("LayerNorm: missing scale '{}'", node.input[1]))?;

    let ndim = input.dims().len();
    let axis_raw = get_int_attr(node, "axis", -1);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };
    let epsilon = get_float_attr(node, "epsilon", 1e-5);
    let axes: Vec<usize> = (axis..ndim).collect();

    let mut result = input.layer_norm(axes, epsilon);

    // Apply scale (broadcast to input shape using Expression-aware broadcast)
    let input_shape = input.dims();
    result *= broadcast_to_expr(scale, &input_shape);

    // Apply optional bias
    if node.input.len() > 2 && !node.input[2].is_empty() {
        let bias = *tensors
            .get(&node.input[2])
            .ok_or_else(|| format!("LayerNorm: missing bias '{}'", node.input[2]))?;
        result += broadcast_to_expr(bias, &input_shape);
    }

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: LayerNormalization Node");
    Ok(())
}

/// Handle GroupNormalization node (opset 18).
///
/// Inputs: X [N, C, spatial...], scale [num_groups], bias [num_groups]
/// Attributes: num_groups (required), epsilon (default 1e-5)
///
/// Normalizes over channels-per-group and spatial dims, then applies per-group scale/bias.
/// Decomposed into: reshape [N, G, C/G, spatial...] -> layer_norm over [C/G, spatial...] ->
/// reshape back to [N, C, spatial...] -> scale + bias (broadcast).
pub fn parse_group_norm_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: GroupNormalization Node");

    assert!(
        node.input.len() >= 3,
        "GroupNormalization needs 3 inputs (X, scale, bias), got {}",
        node.input.len()
    );

    let x = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("GroupNorm: missing input X '{}'", node.input[0]))?;
    let scale = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("GroupNorm: missing scale '{}'", node.input[1]))?;
    let bias = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("GroupNorm: missing bias '{}'", node.input[2]))?;

    let x_dims = x.dims();
    let ndim = x_dims.len();
    assert!(
        ndim >= 3,
        "GroupNorm: input must be at least 3D [N, C, spatial...], got {ndim}D"
    );

    let num_groups = get_int_attr(node, "num_groups", 1) as usize;
    let epsilon = get_float_attr(node, "epsilon", 1e-5);

    let n = x_dims[0]
        .to_usize()
        .expect("GroupNorm: batch must be concrete");
    let c = x_dims[1]
        .to_usize()
        .expect("GroupNorm: channels must be concrete");
    assert_eq!(
        c % num_groups,
        0,
        "GroupNorm: channels {c} must be divisible by num_groups {num_groups}"
    );
    let cpg = c / num_groups; // channels per group

    // Reshape X from [N, C, spatial...] to [N, G, C/G, spatial...]
    let spatial_dims: Vec<Expression> = x_dims[2..].to_vec();
    let mut reshaped = x;
    let mut new_shape = vec![n, num_groups, cpg];
    for d in &spatial_dims {
        new_shape.push(
            d.to_usize()
                .expect("GroupNorm: spatial dims must be concrete"),
        );
    }
    reshaped.shape = ShapeTracker::new(new_shape.clone());

    // Normalize over axes [2, 3, ..., ndim] (C/G + spatial dims)
    let norm_axes: Vec<usize> = (2..new_shape.len()).collect();
    let mut normed = reshaped.layer_norm(norm_axes, epsilon);

    // Reshape back to [N, C, spatial...]
    let mut orig_shape = vec![n, c];
    for d in &spatial_dims {
        orig_shape.push(d.to_usize().unwrap());
    }
    normed *= 1.0;
    normed.shape = ShapeTracker::new(orig_shape.clone());

    // Apply scale and bias (both shape [C], broadcast to [N, C, spatial...])
    let target_shape: Vec<Expression> = orig_shape.iter().map(|&d| Expression::from(d)).collect();
    let result =
        normed * broadcast_to_expr(scale, &target_shape) + broadcast_to_expr(bias, &target_shape);

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: GroupNormalization Node");
    Ok(())
}
