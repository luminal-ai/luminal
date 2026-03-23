use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to_expr, get_float_attr, get_int_attr};

pub fn parse_matmul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Started parse: MatMul Node");
    assert!(node.input.len() == 2, "MatMul should have exactly 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[1]))?;

    //TODO: enforce some kind of check here that they are broadcastable
    let result = a.matmul(b);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: MatMul Node");
    Ok(())
}

/// Handle Gemm node: Y = alpha * (transA ? A.T : A) @ (transB ? B.T : B) + beta * C
///
/// Attributes: transA (default 0), transB (default 0), alpha (default 1.0), beta (default 1.0)
/// Input C (bias) is optional.
pub fn parse_gemm_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Started parse: Gemm Node");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gemm: missing input A '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gemm: missing input B '{}'", node.input[1]))?;

    let trans_a = get_int_attr(node, "transA", 0) != 0;
    let trans_b = get_int_attr(node, "transB", 0) != 0;
    let alpha = get_float_attr(node, "alpha", 1.0);
    let beta = get_float_attr(node, "beta", 1.0);

    let a_mat = if trans_a { a.permute(vec![1, 0]) } else { a };
    let b_mat = if trans_b { b.permute(vec![1, 0]) } else { b };

    let mut result = a_mat.matmul(b_mat);
    if alpha != 1.0 {
        result *= alpha;
    }

    if node.input.len() > 2 && !node.input[2].is_empty() {
        let c = *tensors
            .get(&node.input[2])
            .ok_or_else(|| format!("Gemm: missing bias C '{}'", node.input[2]))?;
        let c_scaled = if beta != 1.0 { c * beta } else { c };
        let result_shape = result.dims();
        result += broadcast_to_expr(c_scaled, &result_shape);
    }

    tensors.insert(node.output[0].clone(), result);
    trace!("Finished parse: Gemm Node");
    Ok(())
}
