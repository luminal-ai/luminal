use std::collections::HashMap;

use luminal::{
    prelude::{tracing::trace, *},
    shape::Expression,
};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to_expr, compute_broadcast_shape_expr};

/// Handle Where node: conditional select — output[i] = condition[i] ? x[i] : y[i]
///
/// ONNX Where uses numpy-style broadcasting across all three inputs.
pub fn parse_where_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 3, "Where should have 3 inputs");
    let condition = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Where: missing condition tensor '{}'", node.input[0]))?;
    let x = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Where: missing X tensor '{}'", node.input[1]))?;
    let y = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("Where: missing Y tensor '{}'", node.input[2]))?;

    let output_name = &node.output[0];

    // ONNX Where broadcasts all 3 inputs to a common shape
    let bc_shape = compute_broadcast_shape_expr(
        &condition.dims(),
        &compute_broadcast_shape_expr(&x.dims(), &y.dims()),
    );
    let condition = broadcast_to_expr(condition, &bc_shape);
    let x = broadcast_to_expr(x, &bc_shape);
    let y = broadcast_to_expr(y, &bc_shape);

    let result = x.cond(condition, y);
    tensors.insert(output_name.clone(), result);
    Ok(())
}

pub fn parse_binary_broadcast_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() == 2,
        "{} should have 2 inputs, got {}",
        op_name,
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "{} should have 1 output, got {}",
        op_name,
        node.output.len()
    );
    // Shape-only path: if any input is shape-only (not in tensors), do Expression arithmetic
    let a_missing = !tensors.contains_key(&node.input[0]);
    let b_missing = !tensors.contains_key(&node.input[1]);
    if a_missing || b_missing {
        // At least one input is shape-only. Do shape_exprs arithmetic and return.
        let se_a = shape_exprs.get(&node.input[0]).cloned().or_else(|| {
            known_values
                .get(&node.input[0])
                .map(|kv| kv.iter().map(|&v| Expression::from(v as usize)).collect())
        });
        let se_b = shape_exprs.get(&node.input[1]).cloned().or_else(|| {
            known_values
                .get(&node.input[1])
                .map(|kv| kv.iter().map(|&v| Expression::from(v as usize)).collect())
        });
        if let (Some(se_a), Some(se_b)) = (se_a, se_b)
            && se_a.len() == 1
            && se_b.len() == 1
        {
            let result_expr = match op_name {
                "Add" => Some(se_a[0] + se_b[0]),
                "Sub" => Some(se_a[0] - se_b[0]),
                "Mul" => Some(se_a[0] * se_b[0]),
                "Div" => Some(se_a[0] / se_b[0]),
                _ => None,
            };
            if let Some(expr) = result_expr {
                shape_exprs.insert(node.output[0].clone(), vec![expr]);
            }
        }
        trace!("Finished parse: {} Node (shape-only)", op_name);
        return Ok(());
    }

    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input '{}'", op_name, node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("{}: missing input '{}'", op_name, node.input[1]))?;
    let broadcast_shape = compute_broadcast_shape_expr(&a.dims(), &b.dims());
    let a_bc = broadcast_to_expr(a, &broadcast_shape);
    let b_bc = broadcast_to_expr(b, &broadcast_shape);
    let result = op(a_bc, b_bc);
    tensors.insert(node.output[0].clone(), result);

    // Propagate shape_exprs for scalar shape arithmetic (e.g., Add(1, seq_len))
    // At least one input must be in shape_exprs; the other can come from known_values.
    let has_shape_expr =
        shape_exprs.contains_key(&node.input[0]) || shape_exprs.contains_key(&node.input[1]);
    if has_shape_expr {
        let se_a = shape_exprs.get(&node.input[0]).cloned().or_else(|| {
            known_values
                .get(&node.input[0])
                .map(|kv| kv.iter().map(|&v| Expression::from(v as usize)).collect())
        });
        let se_b = shape_exprs.get(&node.input[1]).cloned().or_else(|| {
            known_values
                .get(&node.input[1])
                .map(|kv| kv.iter().map(|&v| Expression::from(v as usize)).collect())
        });
        if let (Some(se_a), Some(se_b)) = (se_a, se_b)
            && se_a.len() == 1
            && se_b.len() == 1
        {
            let result_expr = match op_name {
                "Add" => Some(se_a[0] + se_b[0]),
                "Sub" => Some(se_a[0] - se_b[0]),
                "Mul" => Some(se_a[0] * se_b[0]),
                "Div" => Some(se_a[0] / se_b[0]),
                _ => None,
            };
            if let Some(expr) = result_expr {
                shape_exprs.insert(node.output[0].clone(), vec![expr]);
            }
        }
    }

    trace!("Finished parse: {} Node", op_name);
    Ok(())
}

pub fn parse_variadic_broadcast_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    op_name: &str,
    op: impl Fn(GraphTensor, GraphTensor) -> GraphTensor,
    _shape_exprs: &mut HashMap<String, Vec<Expression>>,
    _known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        node.input.len() >= 2,
        "{} needs at least two inputs, got {}",
        op_name,
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "{} nodes only have one output, got {}",
        op_name,
        node.output.len()
    );

    let mut result = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;

    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, input_name))?;
        let broadcast_shape = compute_broadcast_shape_expr(&result.dims(), &rhs.dims());
        let lhs_bc = broadcast_to_expr(result, &broadcast_shape);
        let rhs_bc = broadcast_to_expr(rhs, &broadcast_shape);
        result = op(lhs_bc, rhs_bc);
    }

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}
