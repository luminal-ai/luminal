use std::collections::HashMap;

use luminal::{
    prelude::{tracing::trace, *},
    shape::{Expression, flatten_strides},
};
use onnx_protobuf::NodeProto;

use crate::util::{broadcast_to_expr, get_int_attr, get_str_attr};

/// Handle GatherElements node: gather elements along an axis using per-element indices.
///
/// ONNX semantics: output[i0,..,ik] = data[i0,..,i_{axis-1}, indices[i0,..,ik], i_{axis+1},..,ik]
pub fn parse_gather_elements_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "GatherElements needs 2 inputs");
    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("GatherElements: missing data '{}'", node.input[0]))?;
    let indices = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("GatherElements: missing indices '{}'", node.input[1]))?;

    let ndim = data.dims().len();
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    let result = data.gather_elements(indices, axis);
    tensors.insert(node.output[0].clone(), result);
    Ok(())
}

/// Handle Expand node: broadcast input to a target shape.
///
/// The target shape is provided as the second input (must be a known constant).
pub fn parse_expand_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Expand needs 2 inputs");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Expand: missing input '{}'", node.input[0]))?;

    // Try shape_exprs first (for dynamic shapes), then known_values
    let target_exprs: Vec<Expression> = if let Some(se) = shape_exprs.get(&node.input[1]) {
        se.clone()
    } else if let Some(shape_vals) = known_values.get(&node.input[1]) {
        shape_vals
            .iter()
            .map(|&v| Expression::from(v as usize))
            .collect()
    } else {
        return Err(format!(
            "Expand: shape '{}' must be a known constant or shape_expr",
            node.input[1]
        ));
    };

    // ONNX Expand uses numpy broadcasting: output_dim = max(input_dim, target_dim).
    let input_dims = input.dims();
    let input_rank = input_dims.len();
    let target_rank = target_exprs.len();
    let out_rank = input_rank.max(target_rank);
    let mut broadcast_shape = Vec::with_capacity(out_rank);
    for i in 0..out_rank {
        let in_dim = if i + input_rank >= out_rank {
            input_dims[i + input_rank - out_rank]
        } else {
            Expression::from(1usize)
        };
        let tgt_dim = if i + target_rank >= out_rank {
            target_exprs[i + target_rank - out_rank]
        } else {
            Expression::from(1usize)
        };

        let dim = match (in_dim.to_usize(), tgt_dim.to_usize()) {
            (Some(a), Some(b)) => Expression::from(a.max(b)),
            (Some(1), _) => tgt_dim,
            (_, Some(1)) => in_dim,
            _ => in_dim, // Both symbolic — assume compatible
        };
        broadcast_shape.push(dim);
    }

    let result = broadcast_to_expr(input, &broadcast_shape);
    tensors.insert(node.output[0].clone(), result);
    Ok(())
}

/// Handle Transpose node: output = permute(input, perm)
///
/// The perm attribute specifies the permutation of dimensions.
/// If perm is not specified, reverses all dimensions (ONNX default).
pub fn parse_transpose_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    trace!("Starting parse: Transpose Node");

    // Validate node structure
    assert!(
        node.input.len() == 1,
        "Transpose nodes must have exactly one input, {} present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Transpose nodes must have exactly one output, {} present",
        node.output.len(),
    );

    // Get input tensor
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Transpose: missing input tensor '{}'", node.input[0]))?;

    let input_rank = input.dims().len();

    // Extract perm attribute or use default (reverse all dims)
    let perm: Vec<usize> = if let Some(attr) = node.attribute.iter().find(|a| a.name == "perm") {
        let perm_i64: &Vec<i64> = &attr.ints;

        // Validate perm length
        if perm_i64.len() != input_rank {
            return Err(format!(
                "Transpose: perm length {} does not match input rank {}",
                perm_i64.len(),
                input_rank
            ));
        }

        // Convert to usize and validate range
        let mut perm_usize: Vec<usize> = Vec::with_capacity(perm_i64.len());
        for &axis in perm_i64.iter() {
            if axis < 0 || axis >= input_rank as i64 {
                return Err(format!(
                    "Transpose: perm axis {} out of range [0, {})",
                    axis, input_rank
                ));
            }
            perm_usize.push(axis as usize);
        }

        // Validate uniqueness (check it's a valid permutation)
        let mut sorted = perm_usize.clone();
        sorted.sort();
        for (i, &val) in sorted.iter().enumerate() {
            if val != i {
                return Err(format!(
                    "Transpose: perm {:?} is not a valid permutation",
                    perm_i64
                ));
            }
        }

        perm_usize
    } else {
        // Default: reverse all dimensions
        (0..input_rank).rev().collect()
    };

    // Apply permute operation
    let result = input.permute(perm);

    // Store output
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    trace!("Finished parse: Transpose Node");
    Ok(())
}

/// Handle Reshape node: change the tensor's shape without modifying data.
///
/// The target shape is read from the second input (must be a known constant).
/// Supports -1 (infer from total elements) and 0 (copy from input dimension).
/// Non-contiguous tensors (e.g., from Expand) are materialized before reshaping.
pub fn parse_reshape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Started parse: Reshape");
    assert!(
        node.input.len() == 2,
        "Reshape should have exactly 2 inputs"
    );

    // Shape-only path: if input is only in shape_exprs (not tensors), propagate
    if !tensors.contains_key(&node.input[0])
        && let Some(se) = shape_exprs.get(&node.input[0]).cloned()
    {
        let output_name = &node.output[0];
        shape_exprs.insert(output_name.clone(), se);
        trace!("Finished parse: Reshape (shape-only)");
        return Ok(());
    }

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Reshape: missing input tensor '{}'", node.input[0]))?;

    let input_dims = input.dims();

    // Try shape_exprs first (for dynamic Reshape targets from Shape→Concat chains)
    let final_shape: Vec<Expression> = if let Some(target_exprs) = shape_exprs.get(&node.input[1]) {
        // Expression-based target shape — resolve 0 and -1
        let mut result_shape: Vec<Expression> = target_exprs.clone();

        // Resolve 0 (copy from input at same position)
        for i in 0..result_shape.len() {
            if result_shape[i].to_usize() == Some(0) && i < input_dims.len() {
                result_shape[i] = input_dims[i];
            }
        }

        // Resolve -1 (infer from total elements / known product)
        let has_neg1 = result_shape
            .iter()
            .any(|e| e.to_usize() == Some(usize::MAX));
        if has_neg1 {
            // total_elements = product of input dims
            let total_elements: Expression = input_dims.iter().cloned().product::<Expression>();
            // known_product = product of non-(-1) dims
            let known_product: Expression = result_shape
                .iter()
                .filter(|e| e.to_usize() != Some(usize::MAX))
                .cloned()
                .product::<Expression>();
            for d in result_shape.iter_mut() {
                if d.to_usize() == Some(usize::MAX) {
                    *d = (total_elements / known_product).simplify();
                }
            }
        }

        result_shape
    } else if let Some(shape_data) = known_values.get(&node.input[1]) {
        // Concrete shape from known_values
        let total_elements: usize = input_dims
            .iter()
            .map(|d| d.to_usize().expect("Reshape: input dims must be concrete"))
            .product();

        let mut target_shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
        for i in 0..target_shape.len() {
            if target_shape[i] == 0 {
                target_shape[i] = input_dims[i].to_usize().unwrap_or(1) as i64;
            }
        }
        let known_product: i64 = target_shape.iter().filter(|&&d| d > 0).product();
        for d in target_shape.iter_mut() {
            if *d == -1 {
                *d = total_elements as i64 / known_product;
            }
        }
        target_shape
            .iter()
            .map(|&d| Expression::from(d as usize))
            .collect()
    } else {
        return Err(format!(
            "Reshape: shape input '{}' must be a known constant or shape_expr",
            node.input[1]
        ));
    };

    let mut result = input;
    // If tensor is not contiguous, materialize
    if !result.shape.is_contiguous() {
        let one = result.graph().constant_float(1.0);
        let src_dims = result.dims();
        let one_expanded = broadcast_to_expr(one, &src_dims);
        result *= one_expanded;
    }
    result.shape = ShapeTracker::new(final_shape.clone());

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values (reshape doesn't change data, just layout)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    // Propagate shape_exprs values unchanged
    if let Some(se) = shape_exprs.get(&node.input[0]).cloned() {
        shape_exprs.insert(output_name.clone(), se);
    }
    trace!("Finished parse: Reshape");
    Ok(())
}

/// Handle Squeeze node: remove size-1 dimensions from a tensor.
///
/// Opset 13+: axes come from the second input. Opset 11: from the "axes" attribute.
/// If no axes are specified, all dimensions of size 1 are removed.
/// Axes are processed in reverse order to avoid index shifting.
pub fn parse_squeeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Squeeze Node");
    assert!(
        !node.input.is_empty(),
        "Squeeze should have at least 1 input"
    );

    // Shape-only path: if input is only in shape_exprs (not tensors), propagate shape_exprs
    if !tensors.contains_key(&node.input[0])
        && let Some(se) = shape_exprs.get(&node.input[0]).cloned()
    {
        let output_name = &node.output[0];
        shape_exprs.insert(output_name.clone(), se);
        trace!("Finished parse: Squeeze Node (shape-only)");
        return Ok(());
    }

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Squeeze: missing input tensor '{}'", node.input[0]))?;

    // Opset 13+: axes come from second input; opset 11: from attribute
    let axes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_data = known_values
            .get(&node.input[1])
            .ok_or_else(|| format!("Squeeze: axes '{}' must be known", node.input[1]))?;
        let ndim = input.dims().len();
        axes_data
            .iter()
            .map(|&v| {
                let a = v as i64;
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        let ndim = input.dims().len();
        attr.ints
            .iter()
            .map(|&v| {
                if v < 0 {
                    (ndim as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        // No axes specified: squeeze all dims of size 1
        input
            .dims()
            .iter()
            .enumerate()
            .filter(|(_, d)| d.to_usize() == Some(1))
            .map(|(i, _)| i)
            .collect()
    };

    // Sort in reverse order so removing earlier axes doesn't shift later ones
    let mut sorted_axes = axes.clone();
    sorted_axes.sort();
    sorted_axes.reverse();

    let mut result = input;
    for &axis in &sorted_axes {
        result = result.squeeze(axis);
    }

    // Handle scalar output (0-dim tensor)
    if result.dims().is_empty() {
        result.shape = ShapeTracker::new(vec![1usize]);
    }

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    // Propagate shape_exprs values unchanged (Squeeze changes tensor shape, not stored values)
    if let Some(se) = shape_exprs.get(&node.input[0]).cloned() {
        shape_exprs.insert(output_name.clone(), se);
    }
    trace!("Finished parse: Squeeze Node");
    Ok(())
}

/// Handle Unsqueeze node: insert size-1 dimensions at the specified axes.
///
/// Axes are read from the second input (must be a known constant).
/// Negative axis values are resolved relative to the output rank.
pub fn parse_unsqueeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Unsqueeze Node");
    assert!(
        node.input.len() == 2,
        "Unsqueeze should have exactly 2 inputs"
    );

    // Shape-only path: if input is only in shape_exprs (not tensors), propagate
    if !tensors.contains_key(&node.input[0])
        && let Some(se) = shape_exprs.get(&node.input[0]).cloned()
    {
        let output_name = &node.output[0];
        shape_exprs.insert(output_name.clone(), se);
        trace!("Finished parse: Unsqueeze Node (shape-only)");
        return Ok(());
    }

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Unsqueeze: missing input tensor '{}'", node.input[0]))?;

    let axes_data = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Unsqueeze: axes '{}' must be known", node.input[1]))?;

    let ndim = input.dims().len();
    let out_ndim = ndim + axes_data.len();
    let mut axes: Vec<usize> = axes_data
        .iter()
        .map(|&v| {
            let a = v as i64;
            if a < 0 {
                (out_ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    axes.sort();

    let mut result = input;
    for &axis in &axes {
        result = result.unsqueeze(axis);
    }

    // Handle scalar output (0-dim tensor)
    if result.dims().is_empty() {
        result.shape = ShapeTracker::new(vec![1usize]);
    }

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    // Propagate shape_exprs values unchanged (Unsqueeze changes tensor shape, not stored values)
    if let Some(se) = shape_exprs.get(&node.input[0]).cloned() {
        shape_exprs.insert(output_name.clone(), se);
    }
    trace!("Finished parse: Unsqueeze Node");
    Ok(())
}

/// Handle Concat node: concatenate multiple tensors along a given axis.
///
/// The axis attribute specifies which dimension to concatenate along.
/// Negative axis values are resolved relative to the input rank.
/// All inputs must have the same shape except along the concat axis.
pub fn parse_concat_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Concat Node");
    assert!(
        !node.input.is_empty(),
        "Concat needs at least 1 input, {} present",
        node.input.len()
    );
    assert!(
        node.output.len() == 1,
        "Concat must have exactly one output, {} present",
        node.output.len()
    );

    let output_name = &node.output[0];

    // Check if this is a shape computation concat (not a tensor concat).
    // At least one input must be in shape_exprs (from a Shape node chain).
    // Other inputs can come from known_values (small initializer constants).
    // All inputs must be small (≤ 8 elements) and rank ≤ 1.
    let has_any_shape_expr = node.input.iter().any(|name| shape_exprs.contains_key(name));
    let all_shape_exprs = has_any_shape_expr
        && node.input.iter().all(|name| {
            let has_se = shape_exprs.contains_key(name);
            let has_kv = known_values.contains_key(name);
            if has_se || has_kv {
                let len = if let Some(se) = shape_exprs.get(name) {
                    se.len()
                } else if let Some(kv) = known_values.get(name) {
                    kv.len()
                } else {
                    return false;
                };
                if let Some(t) = tensors.get(name) {
                    let dims = t.dims();
                    dims.len() <= 1 && len <= 8
                } else {
                    len <= 8
                }
            } else {
                false
            }
        });

    if all_shape_exprs {
        // This is a shape concat (e.g., computing reshape target from shape fragments)
        // Just concatenate the Expression vectors (from shape_exprs or known_values)
        let mut result: Vec<Expression> = Vec::new();
        for input_name in &node.input {
            if let Some(se) = shape_exprs.get(input_name) {
                result.extend(se.iter().cloned());
            } else if let Some(kv) = known_values.get(input_name) {
                result.extend(kv.iter().map(|&v| {
                    let iv = v as i64;
                    if iv < 0 {
                        // Store negative values as large sentinel (usize::MAX for -1, etc.)
                        Expression::from(usize::MAX)
                    } else {
                        Expression::from(v as usize)
                    }
                }));
            }
        }
        shape_exprs.insert(output_name.clone(), result);
        // No tensor entry needed — this is a shape-only computation

        trace!("Finished parse: Concat Node (shape_exprs)");
        return Ok(());
    }

    let first = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Concat: missing input tensor '{}'", node.input[0]))?;

    let ndim = first.dims().len();

    // Extract axis attribute, supporting negative indexing
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // Iteratively concat all inputs along the axis
    let mut result = first;
    for input_name in &node.input[1..] {
        let rhs = *tensors
            .get(input_name)
            .ok_or_else(|| format!("Concat: missing input tensor '{}'", input_name))?;
        result = result.concat_along(rhs, axis);
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: Concat Node");
    Ok(())
}

/// Handle Gather node: index into a tensor along a specified axis.
///
/// For 1D data, performs simple element-wise gathering. For ND data, gathers
/// slices along the specified axis. Supports arbitrary axis values.
/// Supports constant folding when both data and indices are known.
pub fn parse_gather_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting parse: Gather Node");
    assert!(node.input.len() == 2, "Gather should have 2 inputs");

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gather: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let data_ndim = data_dims.len();

    // Handle axis (support negative indexing)
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (data_ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // Shape expression gather: data is from a Shape node (shape_exprs), indices are known constants.
    // This happens in Shape→Gather chains that extract specific dimensions.
    if let Some(se_data) = shape_exprs.get(&node.input[0]).cloned()
        && let Some(idx_vals) = known_values.get(&node.input[1])
    {
        let output_name = &node.output[0];
        let result_exprs: Vec<Expression> = idx_vals
            .iter()
            .map(|&idx_f| {
                let idx = idx_f as i64;
                let idx = if idx < 0 {
                    (se_data.len() as i64 + idx) as usize
                } else {
                    idx as usize
                };
                se_data[idx]
            })
            .collect();
        shape_exprs.insert(output_name.clone(), result_exprs.clone());

        // If all concrete, also store in known_values + weight_data + tensor
        if let Some(concrete) = result_exprs
            .iter()
            .map(|e| e.to_usize())
            .collect::<Option<Vec<usize>>>()
        {
            let floats: Vec<f32> = concrete.iter().map(|&v| v as f32).collect();
            let tensor = cx.named_tensor(output_name.clone(), vec![floats.len()]);
            tensors.insert(output_name.clone(), tensor);
            known_values.insert(output_name.clone(), floats.clone());
            weight_data.push((output_name.clone(), floats));
        }
        // For symbolic results, don't create a tensor — it's shape-only

        trace!("Finished parse: Gather Node (shape_exprs folded)");
        return Ok(());
    }

    // If both inputs are known, fully constant-fold
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let output_name = &node.output[0];

        // Get concrete data dimensions
        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: data dims must be concrete"))
            .collect();

        // Get indices tensor shape
        let indices_tensor = tensors.get(&node.input[1]);
        let indices_shape: Vec<usize> = if let Some(idx_tensor) = indices_tensor {
            idx_tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("Gather: indices dims must be concrete"))
                .collect()
        } else {
            vec![vidx.len()]
        };

        // Compute output shape: data_shape[0..axis] + indices_shape + data_shape[axis+1..]
        let mut output_shape: Vec<usize> = data_shape[..axis].to_vec();
        output_shape.extend(&indices_shape);
        output_shape.extend(&data_shape[axis + 1..]);

        // Gather with arbitrary axis
        let output_size: usize = output_shape.iter().product();
        let mut folded = vec![0.0f32; output_size.max(1)];

        if output_size > 0 {
            let pre_size: usize = data_shape[..axis].iter().product::<usize>().max(1);
            let post_size: usize = data_shape[axis + 1..].iter().product::<usize>().max(1);
            let indices_size: usize = vidx.len();
            let axis_dim = data_shape[axis] as i64;

            for pre in 0..pre_size {
                for (idx_pos, &idx_val) in vidx.iter().enumerate() {
                    // Normalize negative indices: ONNX allows -1 for last element, etc.
                    let idx_raw = idx_val as i64;
                    let idx = (((idx_raw % axis_dim) + axis_dim) % axis_dim) as usize;
                    for post in 0..post_size {
                        let data_flat =
                            pre * (data_shape[axis] * post_size) + idx * post_size + post;
                        let out_flat =
                            pre * (indices_size * post_size) + idx_pos * post_size + post;
                        folded[out_flat] = vdata[data_flat];
                    }
                }
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        trace!("Finished parse: Gather Node (constant folded)");
        return Ok(());
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gather: missing indices tensor '{}'", node.input[1]))?;

    // Normalize negative indices: ONNX allows -1 for last element, -2 for second-to-last, etc.
    // Use conditional normalization instead of modulo to avoid floating-point precision loss:
    // if index < 0 then index + axis_dim else index
    let axis_dim = data_dims[axis].to_usize().ok_or_else(|| {
        "Gather: axis dimension must be concrete for index normalization".to_string()
    })?;
    let axis_dim_f32 = axis_dim as f32;

    // Cast to F32 for normalization arithmetic (ONNX indices may be INT64 → luminal Int,
    // or constant F32 weights). This avoids Bool * F32 panics when lt() returns Bool dtype.
    let indices_f32 = indices_raw.cast(DType::F32);
    let zero = indices_f32
        .graph()
        .constant_float(0.0)
        .expand_rhs(indices_f32.shape);
    let adjustment = indices_f32
        .graph()
        .constant_float(axis_dim_f32)
        .expand_rhs(indices_f32.shape);
    // lt() returns Bool; cast to F32 before arithmetic (same pattern as parse_gathernd_node)
    let is_negative = indices_f32.lt(zero).cast(DType::F32);
    let indices = indices_f32 + (is_negative * adjustment);

    let result = if data_ndim == 1 {
        // 1D data: simple flat element-wise gather
        data.gather(indices.cast(DType::Int))
    } else if axis == 0 {
        // ND data with axis=0: gather entire slices along first dimension
        gather_axis0(data, indices, &data_dims)?
    } else {
        // General case: axis != 0
        // Strategy: permute to bring axis to front, gather, permute back

        // Build permutation to move axis to position 0
        let mut perm: Vec<usize> = (0..data_ndim).collect();
        perm.remove(axis);
        perm.insert(0, axis);

        // Permute data
        let permuted_data = data.permute(perm);
        let permuted_dims = permuted_data.dims();

        // Gather on axis 0 of permuted data
        let gathered = gather_axis0(permuted_data, indices, &permuted_dims)?;

        // Compute gathered shape
        let idx_dims = indices.dims();
        let mut gathered_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &permuted_dims[1..] {
            gathered_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(gathered_shape.clone());

        // Build inverse permutation to restore original axis order
        let indices_rank = idx_dims.len();
        let mut inv_perm: Vec<usize> = Vec::with_capacity(reshaped.dims().len());

        // Dimensions before original axis (they're after indices in gathered result)
        for i in 0..axis {
            inv_perm.push(indices_rank + i);
        }
        // The indices dimensions
        for i in 0..indices_rank {
            inv_perm.push(i);
        }
        // Dimensions after original axis
        for i in axis..(data_ndim - 1) {
            inv_perm.push(indices_rank + i);
        }

        reshaped.permute(inv_perm)
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    trace!("Finished parse: Gather Node");
    Ok(())
}

/// Handle GatherND node: gather elements or slices using multi-dimensional indices.
///
/// The last dimension of `indices` specifies the coordinate depth `q`.
/// For `q=1` (most common from PyTorch's dynamo exporter), degenerates to Gather(axis=0).
/// Supports constant folding when both data and indices are known at graph-build time.
pub fn parse_gathernd_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: GatherND Node");
    assert!(
        node.input.len() >= 2,
        "GatherND should have at least 2 inputs, {} present",
        node.input.len()
    );

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("GatherND: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let n = data_dims.len();

    let b = get_int_attr(node, "batch_dims", 0) as usize;

    let output_name = &node.output[0];

    // --- Constant folding path ---
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let indices_tensor = *tensors
            .get(&node.input[1])
            .ok_or_else(|| format!("GatherND: missing indices tensor '{}'", node.input[1]))?;

        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("GatherND: data dims must be concrete"))
            .collect();
        let idx_shape: Vec<usize> = indices_tensor
            .dims()
            .iter()
            .map(|d| d.to_usize().expect("GatherND: index dims must be concrete"))
            .collect();

        // K = number of outer index dims (excluding coordinate depth dim q)
        let k = idx_shape.len() - 1;
        // q = last dim of indices = coordinate depth
        let q = idx_shape[k];

        // output_shape = data_shape[..b] + idx_shape[b..k] + data_shape[b+q..]
        let mut output_shape: Vec<usize> = data_shape[..b].to_vec();
        output_shape.extend_from_slice(&idx_shape[b..k]);
        output_shape.extend_from_slice(&data_shape[b + q..]);

        let batch_size: usize = data_shape[..b].iter().product::<usize>().max(1);
        let outer_size: usize = idx_shape[b..k].iter().product::<usize>().max(1);
        let slice_size: usize = data_shape[b + q..].iter().product::<usize>().max(1);
        let data_elements_per_batch: usize = data_shape[b..].iter().product::<usize>().max(1);

        // Strides for dims b..b+q of data: stride[j] = product(data_shape[b+j+1..b+q]) * slice_size
        let data_strides: Vec<usize> = (0..q)
            .map(|j| data_shape[b + j + 1..b + q].iter().product::<usize>() * slice_size)
            .collect();

        let total_size = batch_size * outer_size * slice_size;
        let mut folded = vec![0.0f32; total_size.max(1)];

        for batch in 0..batch_size {
            for p in 0..outer_size {
                let idx_offset = (batch * outer_size + p) * q;
                let mut flat_base = batch * data_elements_per_batch;
                for j in 0..q {
                    let coord_raw = vidx[idx_offset + j] as i64;
                    let dim = data_shape[b + j] as i64;
                    let coord = (((coord_raw % dim) + dim) % dim) as usize;
                    flat_base += coord * data_strides[j];
                }
                let out_base = (batch * outer_size + p) * slice_size;
                folded[out_base..(slice_size + out_base)]
                    .copy_from_slice(&vdata[flat_base..(slice_size + flat_base)]);
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        trace!("Finished parse: GatherND Node (constant folded)");
        return Ok(());
    }

    // --- Dynamic path ---
    if b != 0 {
        return Err(format!(
            "GatherND: batch_dims={} > 0 is not yet supported in dynamic mode",
            b
        ));
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("GatherND: missing indices tensor '{}'", node.input[1]))?;

    let data_shape: Vec<usize> = data_dims
        .iter()
        .map(|d| d.to_usize().unwrap_or(1))
        .collect();

    let idx_shape: Vec<usize> = indices_raw
        .dims()
        .iter()
        .map(|d| d.to_usize().unwrap_or(1))
        .collect();

    let k = idx_shape.len() - 1;
    let q = idx_shape[k];

    if q == 1 {
        // Sub-case A: q=1 — GatherND degenerates to Gather(axis=0).
        // Reshape indices from [..., 1] to [...] by dropping the last size-1 dim.
        let squeezed_shape: Vec<usize> = if k == 0 {
            vec![1]
        } else {
            idx_shape[..k].to_vec()
        };
        let mut idx_squeezed = indices_raw;
        idx_squeezed.shape = ShapeTracker::new(squeezed_shape);

        // Cast to F32 for normalization arithmetic (ONNX indices are INT64 → luminal Int)
        let idx_float = idx_squeezed.cast(DType::F32);

        // Normalize negative indices: if index < 0 then index + axis_dim else index
        let axis_dim_f32 = data_shape[0] as f32;
        let zero = idx_float
            .graph()
            .constant_float(0.0)
            .expand_rhs(idx_float.shape);
        let adjustment = idx_float
            .graph()
            .constant_float(axis_dim_f32)
            .expand_rhs(idx_float.shape);
        // lt() returns Bool; cast to F32 before arithmetic (same pattern as parse_floor_node)
        let is_negative = idx_float.lt(zero).cast(DType::F32);
        let idx_normalized = idx_float + (is_negative * adjustment);

        let result = if n == 1 {
            // 1D data: flat element-wise gather
            data.gather(idx_normalized.cast(DType::Int))
        } else {
            // ND data: gather slices along axis 0
            gather_axis0(data, idx_normalized, &data_dims)?
        };

        tensors.insert(output_name.clone(), result);
    } else {
        // Sub-case B: q > 1 — compute flat indices by summing coord_j * stride_j.
        let outer_size: usize = idx_shape[..k].iter().product::<usize>().max(1);
        let slice_size: usize = data_shape[q..].iter().product::<usize>().max(1);

        let indices_flat = indices_raw.flatten();

        // Stride of dim j in data's first q dimensions
        let data_strides: Vec<usize> = (0..q)
            .map(|j| data_shape[j + 1..q].iter().product::<usize>() * slice_size)
            .collect();

        // Build flat_indices (Int) = sum over j of (coord_j * stride_j)
        let mut flat_indices: Option<GraphTensor> = None;
        for j in 0..q {
            // Column j positions in flattened indices: [j, q+j, 2q+j, ...]
            let arange = indices_flat.graph().arange(outer_size); // shape [outer_size], Int
            let q_expanded = indices_flat
                .graph()
                .constant(q as i32)
                .expand_rhs(arange.shape);
            let j_expanded = indices_flat
                .graph()
                .constant(j as i32)
                .expand_rhs(arange.shape);
            let positions = arange * q_expanded + j_expanded; // Int, shape [outer_size]

            // indices_flat may be Int (ONNX INT64); cast to F32 for arithmetic
            let coord_j_f32 = indices_flat.gather(positions).cast(DType::F32); // f32, shape [outer_size]

            // Normalize negative coordinates
            let dim_j_f32 = data_shape[j] as f32;
            let zero = coord_j_f32
                .graph()
                .constant_float(0.0)
                .expand_rhs(coord_j_f32.shape);
            let adj = coord_j_f32
                .graph()
                .constant_float(dim_j_f32)
                .expand_rhs(coord_j_f32.shape);
            // lt() returns Bool; cast to F32 before arithmetic
            let is_neg = coord_j_f32.lt(zero).cast(DType::F32);
            let coord_normalized = coord_j_f32 + (is_neg * adj);

            // Use Int arithmetic to avoid f32 precision loss for large indices
            let coord_int = coord_normalized.cast(DType::Int);
            let stride_expanded = coord_int
                .graph()
                .constant(data_strides[j] as i32)
                .expand_rhs(coord_int.shape);
            let contribution = coord_int * stride_expanded;

            flat_indices = Some(match flat_indices {
                None => contribution,
                Some(acc) => acc + contribution,
            });
        }

        let flat_indices = flat_indices.ok_or_else(|| "GatherND: q must be > 0".to_string())?;

        // Reshape data to [prefix_size, slice_size] for gather_axis0
        let prefix_size: usize = data_shape[..q].iter().product::<usize>().max(1);
        let mut data_reshaped = data;
        data_reshaped.shape = ShapeTracker::new(vec![prefix_size, slice_size]);

        // Reshape flat_indices to the outer shape expected by gather_axis0
        let outer_shape: Vec<usize> = if k == 0 {
            vec![1]
        } else {
            idx_shape[..k].to_vec()
        };
        let mut flat_indices_shaped = flat_indices;
        flat_indices_shaped.shape = ShapeTracker::new(outer_shape.clone());

        // Gather slices: result shape = [outer_shape..., slice_size]
        let data_reshaped_dims = data_reshaped.dims();
        let result = gather_axis0(data_reshaped, flat_indices_shaped, &data_reshaped_dims)?;

        // Reshape result to [outer_shape..., d_q, ..., d_{N-1}]
        let mut output_shape = outer_shape;
        output_shape.extend_from_slice(&data_shape[q..]);
        let mut result_shaped = result;
        result_shaped.shape = ShapeTracker::new(output_shape);

        tensors.insert(output_name.clone(), result_shaped);
    }

    trace!("Finished parse: GatherND Node");
    Ok(())
}

/// Handle Trilu node: apply lower or upper triangular mask to a tensor.
///
/// `upper=1` (default) → upper triangle (triu), `upper=0` → lower triangle (tril).
/// Optional `k` input specifies the diagonal offset (default 0).
/// Luminal's tril/triu require a square matrix; the last two dims must be equal.
pub fn parse_trilu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Trilu Node");

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Trilu: missing input '{}'", node.input[0]))?;

    // Optional k (diagonal offset) — provided as a constant tensor input
    let k: i32 = if node.input.len() > 1 && !node.input[1].is_empty() {
        known_values
            .get(&node.input[1])
            .and_then(|v| v.first())
            .map(|&f| f as i32)
            .unwrap_or(0)
    } else {
        0
    };

    let upper = get_int_attr(node, "upper", 1);

    // Validate that the last two dims form a square matrix
    let dims = input.dims();
    let ndim = dims.len();
    assert!(ndim >= 2, "Trilu: input must have at least 2 dimensions");
    let rows = dims[ndim - 2]
        .to_usize()
        .ok_or_else(|| "Trilu: could not resolve row dimension".to_string())?;
    let cols = dims[ndim - 1]
        .to_usize()
        .ok_or_else(|| "Trilu: could not resolve col dimension".to_string())?;
    assert_eq!(
        rows, cols,
        "Trilu: luminal only supports square matrices, got {}x{}",
        rows, cols
    );
    let size = cols;

    // Generate triangular mask (square, size x size), returns Bool dtype
    let mask = if upper == 0 {
        cx.tril(size, k)
    } else {
        cx.triu(size, k)
    };

    // Cast Bool mask to F32 and broadcast to match input shape (e.g. batch dims)
    let target_shape: Vec<Expression> = dims;
    let mask_f32 = broadcast_to_expr(mask.cast(DType::F32), &target_shape);

    let result = input * mask_f32;
    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Trilu Node");
    Ok(())
}

/// Handle Slice node: extract a sub-tensor using start/end/axes/steps inputs.
///
/// Steps must be 1 (non-unit steps require unfold which is out of scope).
/// Negative starts/ends are normalized against the axis dimension.
/// ONNX INT64_MAX for "to end" is handled by clamping to the axis dimension.
/// Multiple axes are handled by chaining slice_along calls.
pub fn parse_slice_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    shape_exprs: &mut HashMap<String, Vec<Expression>>,
) -> Result<(), String> {
    trace!("Starting Parse: Slice Node");
    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Slice: missing data '{}'", node.input[0]))?;
    let expr_dims = data.dims();
    let rank = expr_dims.len();

    // Get starts/ends from known_values or shape_exprs
    let starts: Vec<i64> = if let Some(kv) = known_values.get(&node.input[1]) {
        kv.iter().map(|&v| v as i64).collect()
    } else if let Some(se) = shape_exprs.get(&node.input[1]) {
        se.iter()
            .map(|e| e.to_usize().unwrap_or(0) as i64)
            .collect()
    } else {
        return Err("Slice: starts must be constant or shape_expr".to_string());
    };

    let ends: Vec<i64> = if let Some(kv) = known_values.get(&node.input[2]) {
        kv.iter().map(|&v| v as i64).collect()
    } else if let Some(se) = shape_exprs.get(&node.input[2]) {
        se.iter()
            .map(|e| e.to_usize().unwrap_or(i64::MAX as usize) as i64)
            .collect()
    } else {
        return Err("Slice: ends must be constant or shape_expr".to_string());
    };

    let axes: Vec<usize> = if node.input.len() > 3 && !node.input[3].is_empty() {
        if let Some(kv) = known_values.get(&node.input[3]) {
            kv.iter()
                .map(|&a| {
                    let a = a as i64;
                    if a < 0 {
                        (a + rank as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else if let Some(se) = shape_exprs.get(&node.input[3]) {
            se.iter()
                .map(|e| {
                    let a = e.to_usize().unwrap_or(0) as i64;
                    if a < 0 {
                        (a + rank as i64) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..starts.len()).collect()
        }
    } else {
        (0..starts.len()).collect()
    };

    // Validate steps=1 if provided
    if node.input.len() > 4
        && !node.input[4].is_empty()
        && let Some(steps) = known_values.get(&node.input[4])
    {
        for &s in steps {
            if s as i64 != 1 {
                return Err(format!(
                    "Slice: step={} not supported (only step=1)",
                    s as i64
                ));
            }
        }
    }

    // Get ends as Expressions (prefer shape_exprs for symbolic values)
    let end_exprs: Vec<Expression> = if let Some(se) = shape_exprs.get(&node.input[2]) {
        se.clone()
    } else {
        ends.iter().map(|&e| Expression::from(e as usize)).collect()
    };

    // Build Expression-based ranges
    let mut ranges: Vec<(Expression, Expression)> = expr_dims
        .iter()
        .map(|d| (Expression::from(0usize), *d))
        .collect();

    for (i, &ax) in axes.iter().enumerate() {
        if i >= starts.len() || i >= end_exprs.len() {
            break;
        }
        let dim_expr = &expr_dims[ax];

        let start_expr = if starts[i] < 0 {
            if let Some(dim_val) = dim_expr.to_usize() {
                Expression::from((dim_val as i64 + starts[i]).max(0) as usize)
            } else {
                *dim_expr - Expression::from((-starts[i]) as usize)
            }
        } else {
            Expression::from(starts[i] as usize)
        };

        let e = &end_exprs[i];
        let end_expr = if let Some(ev) = e.to_usize() {
            // Detect INT64_MAX sentinel (from ONNX "to end" convention).
            // f32 can't represent INT64_MAX exactly, so check for very large values.
            if ev > 1_000_000_000 {
                *dim_expr
            } else {
                Expression::from(ev)
            }
        } else {
            // Symbolic end expression — use directly
            *e
        };

        ranges[ax] = (start_expr, end_expr);
    }

    // Use slice() for concrete ranges, but implement directly for symbolic ranges
    // to avoid luminal's min() clamping which creates unsimplifiable Expression chains.
    let all_concrete = ranges
        .iter()
        .all(|(s, e)| s.to_usize().is_some() && e.to_usize().is_some());

    let result = if all_concrete {
        let concrete_ranges: Vec<(usize, usize)> = ranges
            .iter()
            .map(|(s, e)| (s.to_usize().unwrap(), e.to_usize().unwrap()))
            .collect();
        data.slice(&concrete_ranges)
    } else {
        // Direct implementation without min() clamping:
        // ONNX guarantees start/end are within bounds, so no clamping needed.
        let has_nonzero_start = ranges.iter().any(|(st, _)| *st != Expression::from(0usize));

        if has_nonzero_start {
            // Need iota-based gather for nonzero start
            let dims = data.dims();
            let mut new_dims = vec![];
            let mut index_expressions = vec![];
            let mut phys_size = Expression::from(1);
            for (dim, (start, end)) in dims.into_iter().zip(ranges.iter()).rev() {
                index_expressions.push((Expression::from('z') + *start) * phys_size);
                phys_size *= dim;
                // Direct end - start without min() clamping.
                // .simplify() uses egglog to reduce expressions like (1+a)-1 → a.
                let dim = match (start.to_usize(), end.to_usize()) {
                    (Some(s), Some(e)) => Expression::from(e.saturating_sub(s)),
                    (Some(0), _) => *end,
                    _ => (*end - *start).simplify(),
                };
                new_dims.push(dim);
            }
            new_dims.reverse();
            index_expressions.reverse();
            let index_expression = flatten_strides(&new_dims, &index_expressions);
            let iota = data.graph().iota(index_expression, new_dims);
            data.gather(iota)
        } else {
            // No start offsets — just set dims to end values directly (no min clamping)
            let mut result = data;
            for (sh, (_, end)) in result.shape.dims.iter_mut().zip(ranges) {
                *sh = end;
            }
            result
        }
    };

    tensors.insert(node.output[0].clone(), result);
    trace!("Ending Parse: Slice Node");
    Ok(())
}

/// Handle Split node: split a tensor into multiple chunks along an axis.
///
/// Split sizes come from the optional second input (must be constant) or are
/// computed as equal chunks from the number of outputs.
/// Each non-empty output is inserted into the tensors map under its output name.
pub fn parse_split_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Split: missing data '{}'", node.input[0]))?;
    let rank = data.dims().len();
    let raw_axis = get_int_attr(node, "axis", 0);
    let axis = if raw_axis < 0 {
        (raw_axis + rank as i64) as usize
    } else {
        raw_axis as usize
    };
    let axis_dim = data.dims()[axis]
        .to_usize()
        .expect("Split: axis dim must be concrete");

    let split_sizes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        known_values
            .get(&node.input[1])
            .ok_or("Split: split sizes must be constant")?
            .iter()
            .map(|&v| v as usize)
            .collect()
    } else {
        let n = node.output.len();
        let each = axis_dim / n;
        vec![each; n]
    };

    assert_eq!(
        split_sizes.iter().sum::<usize>(),
        axis_dim,
        "Split: split sizes sum {} != axis_dim {}",
        split_sizes.iter().sum::<usize>(),
        axis_dim
    );

    let mut offset = 0usize;
    for (i, &size) in split_sizes.iter().enumerate() {
        if !node.output[i].is_empty() {
            let segment = data.slice_along(offset..offset + size, axis);
            tensors.insert(node.output[i].clone(), segment);
        }
        offset += size;
    }
    Ok(())
}

/// Handle OneHot node: convert integer indices to one-hot encoded vectors.
///
/// Builds the result as: mask * (on_value - off_value) + off_value
/// where mask = (arange(depth) == indices) broadcasted to output_shape.
/// This avoids needing a dedicated OneHot primitive.
pub fn parse_onehot_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    let indices = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("OneHot: missing indices '{}'", node.input[0]))?;
    let rank = indices.dims().len();

    let depth = known_values
        .get(&node.input[1])
        .ok_or("OneHot: depth must be constant")?[0] as usize;

    let values = known_values
        .get(&node.input[2])
        .ok_or("OneHot: values must be constant")?;
    let (off_value, on_value) = (values[0], values[1]);

    let raw_axis = get_int_attr(node, "axis", -1);
    // Axis is into the OUTPUT shape (rank+1 dims), so normalize against rank+1
    let axis = if raw_axis < 0 {
        (raw_axis + rank as i64 + 1) as usize
    } else {
        raw_axis as usize
    };

    // output_shape: insert depth at position `axis` in indices.dims()
    let output_shape: Vec<usize> = {
        let mut s: Vec<usize> = indices
            .dims()
            .iter()
            .map(|d| d.to_usize().unwrap())
            .collect();
        s.insert(axis, depth);
        s
    };

    // Build arange shaped to [1]*axis + [depth] + [1]*(rank-axis)
    // then broadcast to output_shape
    let mut ar = indices.graph().arange(depth);
    for _ in 0..axis {
        ar = ar.expand_dim(0, 1);
    }
    for _ in (axis + 1)..(rank + 1) {
        let n = ar.dims().len();
        ar = ar.expand_dim(n, 1);
    }
    ar.shape.expand(output_shape.clone());

    // Expand indices to output_shape: insert size-1 at axis, broadcast
    let mut idx = indices.cast(DType::Int);
    idx = idx.expand_dim(axis, 1);
    idx.shape.expand(output_shape.clone());

    // Use ne (not-equal) which returns F32 (0.0=equal, 1.0=not-equal), avoiding Bool dtype
    // At one-hot position: 0.0 * (off - on) + on = on_value
    // Elsewhere:           1.0 * (off - on) + on = off_value
    let ne_mask = idx.cast(DType::F32).ne(ar.cast(DType::F32));

    let result = ne_mask * (off_value - on_value) + on_value;
    tensors.insert(node.output[0].clone(), result);
    Ok(())
}

/// Helper function for gathering along axis 0
fn gather_axis0(
    data: GraphTensor,
    indices: GraphTensor,
    data_dims: &[Expression],
) -> Result<GraphTensor, String> {
    let inner_dim: usize = data_dims[1..]
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "Gather: inner dimensions must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product();

    // Compute flat_indices = indices * inner_dim + offsets
    // IMPORTANT: Use Int arithmetic to avoid f32 precision loss for large indices.
    // f32 only has 24-bit mantissa, so indices > 16.7M lose precision.
    // For vocab_size=50304, embed_dim=768: max flat index = 38.6M
    let indices_int = indices.cast(DType::Int);
    let inner_dim_tensor = indices
        .graph()
        .constant(inner_dim as i32)
        .expand_rhs(indices_int.shape);
    let scaled = indices_int * inner_dim_tensor;
    let idx_ndim = indices.dims().len();
    let scaled_expanded = scaled.expand_dim(idx_ndim, inner_dim);

    // Create column offsets [0, 1, ..., inner_dim-1] and broadcast to indices shape
    // Keep as Int for precise arithmetic
    let offsets = data.graph().arange(inner_dim); // arange returns Int
    let idx_dims = indices.dims();
    let mut offsets_expanded = offsets;
    for i in (0..idx_dims.len()).rev() {
        offsets_expanded = offsets_expanded.expand_dim(0, idx_dims[i]);
    }

    let flat_indices = scaled_expanded + offsets_expanded; // Both Int, result is Int

    // Gather using flat indices into contiguous data buffer
    let gathered = data.gather(flat_indices);

    // Reshape from [..., inner_dim] to [..., D1, D2, ..., Dk]
    if data_dims.len() > 2 {
        let mut output_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &data_dims[1..] {
            output_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(output_shape);
        Ok(reshaped)
    } else {
        Ok(gathered)
    }
}

/// Handle ScatterElements node: scatter updates into data along an axis using per-element indices.
///
/// ONNX semantics: output[i0,..,i_{a-1}, indices[i0,..,ik], i_{a+1},..,ik] = updates[i0,..,ik]
pub fn parse_scatter_elements_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 3, "ScatterElements needs 3 inputs");
    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("ScatterElements: missing data '{}'", node.input[0]))?;
    let indices = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("ScatterElements: missing indices '{}'", node.input[1]))?;
    let updates = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("ScatterElements: missing updates '{}'", node.input[2]))?;

    let ndim = data.dims().len();
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    let result = data.scatter_elements(indices, updates, axis);
    tensors.insert(node.output[0].clone(), result);
    Ok(())
}

/// Handle ScatterND node: scatter updates into data using multi-dimensional index vectors.
///
/// ONNX semantics: for each batch element, use K-element index vector to address into data.
pub fn parse_scatter_nd_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 3, "ScatterND needs 3 inputs");
    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("ScatterND: missing data '{}'", node.input[0]))?;
    let indices = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("ScatterND: missing indices '{}'", node.input[1]))?;
    let updates = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("ScatterND: missing updates '{}'", node.input[2]))?;

    let indices = indices.cast(DType::Int);

    let result = data.scatter_nd(indices, updates);
    tensors.insert(node.output[0].clone(), result);
    Ok(())
}

/// Handle Pad node: pad a tensor along specified dimensions.
///
/// ONNX Pad has inputs: data, pads, [constant_value], [axes]
/// pads is a 1-D tensor of shape [2*num_axes]: [begin_0, begin_1, ..., end_0, end_1, ...]
pub fn parse_pad_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Pad Node");

    assert!(
        node.input.len() >= 2,
        "Pad needs at least 2 inputs (data, pads), got {}",
        node.input.len()
    );

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Pad: missing data '{}'", node.input[0]))?;

    let pads_data = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Pad: pads '{}' must be a known constant", node.input[1]))?;

    let constant_value = if node.input.len() > 2 && !node.input[2].is_empty() {
        if let Some(cv) = known_values.get(&node.input[2]) {
            cv[0]
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mode = get_str_attr(node, "mode", "constant");
    assert_eq!(
        mode, "constant",
        "Pad: only 'constant' mode is supported, got '{mode}'"
    );

    let ndim = data.dims().len();
    let pads: Vec<usize> = pads_data.iter().map(|&v| v as usize).collect();

    // If axes input is provided, only pad those axes; otherwise pad all
    let axes: Vec<usize> = if node.input.len() > 3 && !node.input[3].is_empty() {
        if let Some(axes_data) = known_values.get(&node.input[3]) {
            axes_data
                .iter()
                .map(|&v| {
                    let a = v as i64;
                    if a < 0 {
                        (ndim as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect()
        } else {
            (0..ndim).collect()
        }
    } else {
        (0..ndim).collect()
    };

    let num_axes = axes.len();
    assert_eq!(
        pads.len(),
        2 * num_axes,
        "Pad: pads length {} must be 2 * num_axes {}",
        pads.len(),
        num_axes
    );

    let mut padding: Vec<(Expression, Expression)> = vec![(0.into(), 0.into()); ndim];
    for (i, &axis) in axes.iter().enumerate() {
        padding[axis] = (
            Expression::from(pads[i]),
            Expression::from(pads[num_axes + i]),
        );
    }

    let result = data.pad(padding, constant_value);
    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Pad Node");
    Ok(())
}

/// Handle Resize node: nearest-neighbor upsampling by integer scale factors.
///
/// ONNX Resize inputs: X, [roi], [scales], [sizes]
/// Only nearest-neighbor mode with integer scale factors is supported.
/// Implements upsampling via unsqueeze + expand + merge for each spatial dim.
pub fn parse_resize_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Resize Node");

    let x = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Resize: missing input X '{}'", node.input[0]))?;

    let x_dims = x.dims();
    let ndim = x_dims.len();

    let mode = get_str_attr(node, "mode", "nearest");
    assert!(
        mode == "nearest" || mode == "nearest-exact",
        "Resize: only 'nearest' mode is supported, got '{mode}'"
    );

    // Determine target sizes from either 'scales' (input[2]) or 'sizes' (input[3])
    let target_sizes: Vec<usize> = if node.input.len() > 3
        && !node.input[3].is_empty()
        && known_values.contains_key(&node.input[3])
    {
        // Use sizes directly
        let sizes_data = known_values.get(&node.input[3]).unwrap();
        sizes_data.iter().map(|&v| v as usize).collect()
    } else if node.input.len() > 2
        && !node.input[2].is_empty()
        && known_values.contains_key(&node.input[2])
    {
        // Use scales: target = floor(input_dim * scale)
        let scales_data = known_values.get(&node.input[2]).unwrap();
        x_dims
            .iter()
            .zip(scales_data.iter())
            .map(|(dim, &scale)| {
                let d = dim.to_usize().expect("Resize: dims must be concrete");
                (d as f32 * scale) as usize
            })
            .collect()
    } else {
        return Err("Resize: must provide either 'scales' or 'sizes' input".to_string());
    };

    assert_eq!(
        target_sizes.len(),
        ndim,
        "Resize: target sizes length {} must match input rank {}",
        target_sizes.len(),
        ndim
    );

    // For each dimension, compute integer scale factor (target / input).
    // Use unsqueeze + expand + merge pattern for upsampling.
    let mut result = x;

    for dim_idx in 0..ndim {
        let actual_axis = dim_idx;
        let input_size = x_dims[dim_idx]
            .to_usize()
            .expect("Resize: dims must be concrete");
        let target = target_sizes[dim_idx];

        if target == input_size {
            continue; // no resize needed
        }

        assert!(
            target >= input_size && target.is_multiple_of(input_size),
            "Resize: only integer upsampling is supported (dim {dim_idx}: {input_size} -> {target})"
        );

        let scale = target / input_size;
        // Insert a new dimension after current axis, expand it to scale, then merge
        result = result.expand_dim(actual_axis + 1, scale);
        result = result.merge_dims(actual_axis, actual_axis + 1);
        // No net change to current_axis_offset since we added 1 and merged 1
    }

    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Resize Node");
    Ok(())
}

/// Handle Tile node: repeat a tensor along each dimension.
///
/// ONNX Tile inputs: input, repeats (1-D tensor of repeat counts per dim).
/// Implemented via expand_dim + merge_dims for each dimension with repeat > 1.
pub fn parse_tile_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    trace!("Starting parse: Tile Node");

    assert!(
        node.input.len() == 2,
        "Tile needs 2 inputs (input, repeats), got {}",
        node.input.len()
    );

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Tile: missing input '{}'", node.input[0]))?;

    let repeats_data = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Tile: repeats '{}' must be a known constant", node.input[1]))?;

    let repeats: Vec<usize> = repeats_data.iter().map(|&v| v as usize).collect();
    let ndim = data.dims().len();
    assert_eq!(
        repeats.len(),
        ndim,
        "Tile: repeats length {} must match input rank {}",
        repeats.len(),
        ndim
    );

    let mut result = data;
    for (dim, &r) in repeats.iter().enumerate().take(ndim) {
        if r > 1 {
            // Insert a new dim before the axis, expand to r, merge
            result = result.expand_dim(dim, r);
            result = result.merge_dims(dim, dim + 1);
        }
    }

    tensors.insert(node.output[0].clone(), result);

    trace!("Finished parse: Tile Node");
    Ok(())
}
