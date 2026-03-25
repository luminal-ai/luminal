use std::collections::HashMap;

use luminal::{prelude::*, shape::Expression};
use onnx_protobuf::NodeProto;

use crate::ops_parse::*;

pub fn process_onnx_nodes(
    nodes: &[NodeProto],
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    for node in nodes {
        match node.op_type.as_str() {
            "Add" => parse_binary_broadcast_op(
                node,
                tensors,
                "Add",
                |a, b| a + b,
                shape_exprs,
                known_values,
            )?,
            "Mod" => parse_binary_broadcast_op(
                node,
                tensors,
                "Mod",
                |a, b| a % b,
                shape_exprs,
                known_values,
            )?,
            "Sub" => parse_binary_broadcast_op(
                node,
                tensors,
                "Sub",
                |a, b| a - b,
                shape_exprs,
                known_values,
            )?,
            "Mul" => parse_binary_broadcast_op(
                node,
                tensors,
                "Mul",
                |a, b| a * b,
                shape_exprs,
                known_values,
            )?,
            "Div" => parse_binary_broadcast_op(
                node,
                tensors,
                "Div",
                |a, b| a / b,
                shape_exprs,
                known_values,
            )?,
            "Sqrt" => parse_unary_op(node, tensors, "Sqrt", |a| a.sqrt())?,
            "Transpose" => parse_transpose_node(node, tensors)?,
            "Concat" => parse_concat_node(node, tensors, shape_exprs, known_values)?,
            "Floor" => parse_floor_node(node, tensors)?,
            "Ceil" => parse_ceil_node(node, tensors)?,
            "Sin" => parse_unary_op(node, tensors, "Sin", |a| a.sin())?,
            "Neg" => parse_unary_op(node, tensors, "Neg", |a| -a)?,
            "Cos" => parse_unary_op(node, tensors, "Cos", |a| a.cos())?,
            "Pow" => parse_binary_broadcast_op(
                node,
                tensors,
                "Pow",
                |a, b| a.pow(b),
                shape_exprs,
                known_values,
            )?,
            "Sigmoid" => parse_unary_op(node, tensors, "Sigmoid", |a| a.sigmoid())?,
            "Tanh" => parse_unary_op(node, tensors, "Tanh", |a| a.tanh())?,
            "Relu" => parse_unary_op(node, tensors, "Relu", |a| a.relu())?,
            "Softmax" => parse_softmax_node(node, tensors)?,
            "Abs" => parse_unary_op(node, tensors, "Abs", |a| a.abs())?,
            "Reciprocal" => parse_unary_op(node, tensors, "Reciprocal", |a| a.reciprocal())?,
            "Clip" => parse_clip_node(node, tensors, known_values)?,
            "Equal" => parse_binary_broadcast_op(
                node,
                tensors,
                "Equal",
                |a, b| a.eq(b),
                shape_exprs,
                known_values,
            )?,
            "Where" => parse_where_node(node, tensors)?,
            "Constant" => {
                parse_constant_node(node, tensors, cx, weight_data, known_values, shape_exprs)?
            }
            "ConstantOfShape" => {
                parse_constant_of_shape(node, tensors, cx, weight_data, known_values, shape_exprs)?
            }
            "Cast" => parse_cast_node(node, tensors, weight_data, known_values, shape_exprs)?,
            "MatMul" => parse_matmul_node(node, tensors)?,
            "Reshape" => parse_reshape_node(node, tensors, known_values, shape_exprs)?,
            "Shape" => parse_shape_node(node, tensors, cx, weight_data, known_values, shape_exprs)?,
            "Gather" => {
                parse_gather_node(node, tensors, cx, weight_data, known_values, shape_exprs)?
            }
            "GatherND" => parse_gathernd_node(node, tensors, cx, weight_data, known_values)?,
            "Less" => parse_binary_broadcast_op(
                node,
                tensors,
                "Less",
                |a, b| a.lt(b),
                shape_exprs,
                known_values,
            )?,
            "Greater" => parse_binary_broadcast_op(
                node,
                tensors,
                "Greater",
                |a, b| b.lt(a),
                shape_exprs,
                known_values,
            )?,
            "LessOrEqual" => parse_binary_broadcast_op(
                node,
                tensors,
                "LessOrEqual",
                |a, b| a.le(b),
                shape_exprs,
                known_values,
            )?,
            "GreaterOrEqual" => parse_binary_broadcast_op(
                node,
                tensors,
                "GreaterOrEqual",
                |a, b| a.ge(b),
                shape_exprs,
                known_values,
            )?,
            "Not" => parse_not_node(node, tensors)?,
            "And" => parse_binary_broadcast_op(
                node,
                tensors,
                "And",
                |a, b| a.cast(DType::F32) * b.cast(DType::F32),
                shape_exprs,
                known_values,
            )?,
            "Or" => parse_binary_broadcast_op(
                node,
                tensors,
                "Or",
                |a, b| (a.cast(DType::F32) + b.cast(DType::F32)).minimum_f32(1.0),
                shape_exprs,
                known_values,
            )?,
            "Xor" => parse_binary_broadcast_op(
                node,
                tensors,
                "Xor",
                |a, b| a.ne(b),
                shape_exprs,
                known_values,
            )?,
            "Min" => parse_variadic_broadcast_op(
                node,
                tensors,
                "Min",
                |a, b| a.minimum(b),
                shape_exprs,
                known_values,
            )?,
            "Max" => parse_variadic_broadcast_op(
                node,
                tensors,
                "Max",
                |a, b| a.maximum(b),
                shape_exprs,
                known_values,
            )?,
            "Identity" => parse_identity(node, tensors, known_values, shape_exprs)?,
            "Unsqueeze" => parse_unsqueeze_node(node, tensors, known_values, shape_exprs)?,
            "Squeeze" => parse_squeeze_node(node, tensors, known_values, shape_exprs)?,
            "ReduceSum" => parse_reduce_op(
                node,
                tensors,
                known_values,
                "ReduceSum",
                |t, axes| t.sum(axes),
                |flat, _n| flat.sum(1),
            )?,
            "ReduceMax" => parse_reduce_op(
                node,
                tensors,
                known_values,
                "ReduceMax",
                |t, axes| t.max(axes),
                |flat, _n| flat.max(1),
            )?,
            "ReduceMin" => parse_reduce_op(
                node,
                tensors,
                known_values,
                "ReduceMin",
                |t, axes| t.min(axes),
                |flat, _n| flat.min(1),
            )?,
            "ReduceMean" => parse_reduce_op(
                node,
                tensors,
                known_values,
                "ReduceMean",
                |t, axes| t.mean(axes),
                |flat, n| flat.sum(1) / n as f32,
            )?,
            "Trilu" => parse_trilu_node(node, tensors, cx, known_values)?,
            "GatherElements" => parse_gather_elements_node(node, tensors)?,
            "ScatterElements" => parse_scatter_elements_node(node, tensors)?,
            "ScatterND" => parse_scatter_nd_node(node, tensors)?,
            "Expand" => parse_expand_node(node, tensors, known_values, shape_exprs)?,
            "IsNaN" => parse_unary_op(node, tensors, "IsNaN", |a| a.ne(a))?,
            "LayerNormalization" => parse_layernorm_node(node, tensors)?,
            "Gemm" => parse_gemm_node(node, tensors)?,
            "Erf" => parse_erf_node(node, tensors)?,
            "Slice" => parse_slice_node(node, tensors, known_values, shape_exprs)?,
            "Split" => parse_split_node(node, tensors, known_values)?,
            "TopK" => parse_topk_node(node, tensors, known_values)?,
            "OneHot" => parse_onehot_node(node, tensors, known_values)?,
            "Range" => parse_range_node(node, tensors, cx, weight_data, known_values, shape_exprs)?,
            "CumSum" => parse_cumsum_node(node, tensors, known_values)?,
            "Gelu" => parse_unary_op(node, tensors, "Gelu", |a| a.gelu())?,
            "Conv" => parse_conv_node(node, tensors)?,
            "Pad" => parse_pad_node(node, tensors, known_values)?,
            "Resize" => parse_resize_node(node, tensors, known_values)?,
            "Tile" => parse_tile_node(node, tensors, known_values)?,
            "ReduceL2" => parse_reduce_op(
                node,
                tensors,
                known_values,
                "ReduceL2",
                |t, axes| (t * t).sum(axes).sqrt(),
                |flat, _n| (flat * flat).sum(1).sqrt(),
            )?,
            "GroupNormalization" => parse_group_norm_node(node, tensors)?,
            _ => {
                panic!("Missing Node {}", node.op_type)
            }
        }
    }

    Ok(())
}
