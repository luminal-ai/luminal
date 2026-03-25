use std::{collections::HashMap, fs, path::Path};

use luminal::{prelude::GraphTensor, shape::Expression};
use onnx_protobuf::NodeProto;

/// Maps ONNX dim_param names (e.g. "seq_len") to luminal Expression variable chars ('a'..'w').
pub type DimParamMap = HashMap<String, char>;

// Given a Value from the Onnx proto return its tensor Shape, if it exists
// Note: some times pytorch will create tensors with a 0 shape
// we might want to handle, 0 shape and No shape as seperate ideas
pub fn get_shape_for_onnx_value(value: &onnx_protobuf::ValueInfoProto) -> Vec<usize> {
    if let Some(type_proto) = value.type_.as_ref()
        && let Some(onnx_protobuf::type_proto::Value::TensorType(tensor)) = &type_proto.value
        && let Some(shape) = tensor.shape.as_ref()
    {
        // Scalar (0-dim) tensors have an empty dim list; represent as [1] in luminal
        if shape.dim.is_empty() {
            return vec![1];
        }
        return shape
            .dim
            .iter()
            .map(|dimension| {
                if let Some(onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(v)) =
                    &dimension.value
                {
                    *v as usize
                } else {
                    1
                }
            })
            .collect();
    }

    vec![]
}

/// Like `get_shape_for_onnx_value`, but returns `Vec<Expression>` with symbolic vars for DimParam dims.
/// Allocates new variable chars in `dim_param_map` for unseen dim_param names.
/// `next_char` is updated to the next available char after allocation.
pub fn get_shape_for_onnx_value_expr(
    value: &onnx_protobuf::ValueInfoProto,
    dim_param_map: &mut DimParamMap,
    next_char: &mut char,
) -> Vec<Expression> {
    if let Some(type_proto) = value.type_.as_ref()
        && let Some(onnx_protobuf::type_proto::Value::TensorType(tensor)) = &type_proto.value
        && let Some(shape) = tensor.shape.as_ref()
    {
        if shape.dim.is_empty() {
            return vec![Expression::from(1usize)];
        }
        return shape
            .dim
            .iter()
            .map(|dimension| match &dimension.value {
                Some(onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                    Expression::from(*v as usize)
                }
                Some(onnx_protobuf::tensor_shape_proto::dimension::Value::DimParam(name)) => {
                    let ch = *dim_param_map.entry(name.clone()).or_insert_with(|| {
                        let c = *next_char;
                        *next_char = (c as u8 + 1) as char;
                        c
                    });
                    Expression::from(ch)
                }
                _ => Expression::from(1usize),
            })
            .collect();
    }

    vec![]
}

/// Compute the broadcast output shape for two tensors using Expressions (numpy rules).
pub fn compute_broadcast_shape_expr(a: &[Expression], b: &[Expression]) -> Vec<Expression> {
    let max_rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let a_dim = if i < max_rank - a.len() {
            Expression::from(1usize)
        } else {
            a[i - (max_rank - a.len())]
        };
        let b_dim = if i < max_rank - b.len() {
            Expression::from(1usize)
        } else {
            b[i - (max_rank - b.len())]
        };

        // If both are concrete, use max. If one is 1, use the other.
        // Otherwise, assume they match (same symbolic dim).
        let dim = match (a_dim.to_usize(), b_dim.to_usize()) {
            (Some(a_val), Some(b_val)) => Expression::from(a_val.max(b_val)),
            (Some(1), _) => b_dim,
            (_, Some(1)) => a_dim,
            _ => a_dim, // Both symbolic — assume compatible
        };
        result.push(dim);
    }
    result
}

/// Broadcast a tensor's shape to match a target Expression shape (numpy-style broadcasting).
/// Left-pads with size-1 dims, then expands dims that are 1 to match target.
pub fn broadcast_to_expr(mut tensor: GraphTensor, target_shape: &[Expression]) -> GraphTensor {
    let src_dims = tensor.dims();
    let src_len = src_dims.len();
    let tgt_len = target_shape.len();

    if src_len == tgt_len {
        tensor.shape.expand(target_shape.to_vec());
        return tensor;
    }

    // Left-pad with size-1 dims
    for _ in 0..(tgt_len - src_len) {
        tensor = tensor.expand_dim(0, 1);
    }

    tensor.shape.expand(target_shape.to_vec());
    tensor
}

/// Convert inline data from a TensorProto to f32, based on data_type.
/// Returns None if the tensor has no inline data (e.g. external storage).
fn convert_inline_data(init: &onnx_protobuf::TensorProto) -> Option<Vec<f32>> {
    match init.data_type {
        1 => {
            // FLOAT
            if !init.float_data.is_empty() {
                return Some(init.float_data.clone());
            }
            if !init.raw_data.is_empty() {
                return Some(parse_raw_bytes_as_f32(&init.raw_data, 1));
            }
        }
        7 => {
            // INT64
            if !init.int64_data.is_empty() {
                return Some(init.int64_data.iter().map(|&v| v as f32).collect());
            }
            if !init.raw_data.is_empty() {
                return Some(parse_raw_bytes_as_f32(&init.raw_data, 7));
            }
        }
        6 => {
            // INT32
            if !init.int32_data.is_empty() {
                return Some(init.int32_data.iter().map(|&v| v as f32).collect());
            }
            if !init.raw_data.is_empty() {
                return Some(parse_raw_bytes_as_f32(&init.raw_data, 6));
            }
        }
        9 => {
            // BOOL
            if !init.raw_data.is_empty() {
                return Some(parse_raw_bytes_as_f32(&init.raw_data, 9));
            }
            if !init.int32_data.is_empty() {
                return Some(
                    init.int32_data
                        .iter()
                        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                        .collect(),
                );
            }
        }
        _ => {
            // Fallback: try float_data or interpret raw_data as F32
            if !init.float_data.is_empty() {
                return Some(init.float_data.clone());
            }
            if !init.raw_data.is_empty() {
                return Some(parse_raw_bytes_as_f32(&init.raw_data, 1));
            }
        }
    }
    None
}

/// Parse a raw byte slice as f32 values, respecting the ONNX data_type.
fn parse_raw_bytes_as_f32(bytes: &[u8], data_type: i32) -> Vec<f32> {
    match data_type {
        1 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        7 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        6 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        9 => bytes
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

/// Load float data from a TensorProto, handling inline (float_data/raw_data) and external storage.
/// Prefer `load_all_tensor_floats` for batch loading (avoids redundant file reads).
#[allow(dead_code)]
pub fn load_tensor_floats(init: &onnx_protobuf::TensorProto, model_dir: &Path) -> Option<Vec<f32>> {
    // Try inline data first
    if let Some(floats) = convert_inline_data(init) {
        return Some(floats);
    }
    // Try external data (data_location == EXTERNAL = 1)
    if !init.external_data.is_empty() {
        let mut location: Option<&str> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;
        for entry in &init.external_data {
            match entry.key.as_str() {
                "location" => location = Some(&entry.value),
                "offset" => offset = entry.value.parse().unwrap_or(0),
                "length" => length = entry.value.parse().ok(),
                _ => {}
            }
        }
        if let Some(loc) = location {
            let ext_path = model_dir.join(loc);
            match fs::read(&ext_path) {
                Ok(file_data) => {
                    let start = offset as usize;
                    let end = match length {
                        Some(len) => start + len as usize,
                        None => file_data.len(),
                    };
                    if end > file_data.len() {
                        return None;
                    }
                    return Some(parse_raw_bytes_as_f32(
                        &file_data[start..end],
                        init.data_type,
                    ));
                }
                Err(_) => {
                    return None;
                }
            }
        }
    }
    None
}

/// Batch-load float data from multiple TensorProtos, reading each external file only once.
/// Returns results in the same order as `inits`, with `None` for tensors that couldn't be loaded.
pub fn load_all_tensor_floats(
    inits: &[onnx_protobuf::TensorProto],
    model_dir: &Path,
) -> Vec<(String, Option<Vec<f32>>)> {
    let mut results: Vec<(String, Option<Vec<f32>>)> = Vec::with_capacity(inits.len());

    // Pending external data entries: (result_index, offset, length, data_type)
    // grouped by file location
    type ExternalEntry = (usize, u64, Option<u64>, i32);
    let mut external_pending: HashMap<String, Vec<ExternalEntry>> = HashMap::new();

    for (i, init) in inits.iter().enumerate() {
        // Try inline data first
        if let Some(floats) = convert_inline_data(init) {
            results.push((init.name.clone(), Some(floats)));
            continue;
        }

        // Check for external data
        if !init.external_data.is_empty() {
            let mut location: Option<String> = None;
            let mut offset: u64 = 0;
            let mut length: Option<u64> = None;
            for entry in &init.external_data {
                match entry.key.as_str() {
                    "location" => location = Some(entry.value.clone()),
                    "offset" => offset = entry.value.parse().unwrap_or(0),
                    "length" => length = entry.value.parse().ok(),
                    _ => {}
                }
            }
            if let Some(loc) = location {
                // Push placeholder, will fill in later
                results.push((init.name.clone(), None));
                external_pending
                    .entry(loc)
                    .or_default()
                    .push((i, offset, length, init.data_type));
                continue;
            }
        }

        results.push((init.name.clone(), None));
    }

    // Read each external file once and extract all tensor slices
    for (loc, entries) in &external_pending {
        let ext_path = model_dir.join(loc);
        let file_data = match fs::read(&ext_path) {
            Ok(data) => data,
            Err(_) => continue, // results already have None
        };
        for &(idx, offset, length, data_type) in entries {
            let start = offset as usize;
            let end = match length {
                Some(len) => start + len as usize,
                None => file_data.len(),
            };
            if end > file_data.len() {
                continue;
            }
            results[idx].1 = Some(parse_raw_bytes_as_f32(&file_data[start..end], data_type));
        }
    }

    results
}

/// Load initializer data as f32 values, handling multiple ONNX data types.
/// Used to seed known_values with small constant initializers for constant folding.
pub fn load_initializer_as_f32(init: &onnx_protobuf::TensorProto) -> Option<Vec<f32>> {
    match init.data_type {
        1 => {
            // FLOAT
            if !init.float_data.is_empty() {
                Some(init.float_data.clone())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            } else {
                None
            }
        }
        7 => {
            // INT64
            if !init.int64_data.is_empty() {
                Some(init.int64_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        6 => {
            // INT32
            if !init.int32_data.is_empty() {
                Some(init.int32_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect(),
                )
            } else {
                None
            }
        }
        16 => {
            // BFLOAT16 — 2 bytes per element, upper 16 bits of f32
            if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        9 => {
            // BOOL — 1 byte per element, 0 → 0.0, non-zero → 1.0
            if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .iter()
                        .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                        .collect(),
                )
            } else if !init.int32_data.is_empty() {
                Some(
                    init.int32_data
                        .iter()
                        .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                        .collect(),
                )
            } else {
                None
            }
        }
        11 => {
            // FLOAT64
            if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Transpose weight data from [rows, cols] to [cols, rows] row-major layout
#[cfg(feature = "cuda")]
pub fn transpose_weight_data(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

/// Get an integer attribute from a node, with a default value
pub fn get_int_attr(node: &NodeProto, name: &str, default: i64) -> i64 {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.i;
        }
    }
    default
}

/// Get a string attribute from a node, with a default value
pub fn get_str_attr(node: &NodeProto, name: &str, default: &str) -> String {
    for attr in &node.attribute {
        if attr.name == name {
            return String::from_utf8_lossy(&attr.s).into_owned();
        }
    }
    default.to_string()
}

/// Get a float attribute from a node, with a default value
pub fn get_float_attr(node: &NodeProto, name: &str, default: f32) -> f32 {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.f;
        }
    }
    default
}
