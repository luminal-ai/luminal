use std::collections::HashMap;

use luminal::prelude::{tracing::trace, *};
use onnx_protobuf::NodeProto;

use crate::util::get_int_attr;

/// Handle TopK node: return the top-k values and indices along an axis.
///
/// output[0] = values (F32), output[1] = indices (Int, can be empty/unused).
/// For largest=true (default): uses topk_indexes + gather_elements.
/// For largest=false: uses argsort(ascending).slice_along(..k) + gather_elements.
/// Indices output is stored as-is (Int dtype); downstream Cast handles F32 conversion.
/// The "sorted" attribute is ignored — output is always sorted.
pub fn parse_topk_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    let x = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("TopK: missing input '{}'", node.input[0]))?;
    let k = known_values
        .get(&node.input[1])
        .ok_or("TopK: k must be constant")?[0] as usize;

    let rank = x.dims().len() as i64;
    let raw_axis = get_int_attr(node, "axis", -1);
    let axis = if raw_axis < 0 {
        (raw_axis + rank) as usize
    } else {
        raw_axis as usize
    };

    let largest = get_int_attr(node, "largest", 1) != 0;

    // Compute full argsort, then gather all sorted values, then slice both to top-k.
    // This avoids passing a non-contiguous sliced index tensor into gather_elements,
    // which triggers a CUDA kernel bug when data and index sizes differ along the axis.
    let full_argsort = x.argsort(axis, largest);
    let indices = full_argsort.slice_along(..k, axis);
    let values = x.gather_elements(full_argsort, axis).slice_along(..k, axis);

    // ONNX output[0] = values, output[1] = indices
    if !node.output[0].is_empty() {
        tensors.insert(node.output[0].clone(), values);
    }
    if node.output.len() > 1 && !node.output[1].is_empty() {
        // Force materialization of Int indices; downstream Cast(INT64→FLOAT) handles the
        // F32 conversion via the *1.0 workaround in parse_cast_node.
        tensors.insert(node.output[1].clone(), indices * 1.0);
    }
    Ok(())
}

pub fn parse_reduce_op(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
    op_name: &str,
    reduce_op: impl Fn(GraphTensor, Vec<usize>) -> GraphTensor,
    all_axes_op: impl Fn(GraphTensor, usize) -> GraphTensor,
) -> Result<(), String> {
    trace!("Starting parse: {} Node", op_name);
    assert!(
        !node.input.is_empty(),
        "{} should have at least 1 input",
        op_name
    );
    assert!(
        node.output.len() == 1,
        "{} should have exactly 1 output",
        op_name
    );

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("{}: missing input tensor '{}'", op_name, node.input[0]))?;

    let keepdims = get_int_attr(node, "keepdims", 1) != 0;
    let noop_with_empty_axes = get_int_attr(node, "noop_with_empty_axes", 0) != 0;

    let ndim = input.dims().len();

    // Resolve axes from second input (opset 13+) or from attribute (opset 11)
    let raw_axes: Vec<i64> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_vals = known_values.get(&node.input[1]).ok_or_else(|| {
            format!(
                "{}: axes input '{}' must be a known constant",
                op_name, node.input[1]
            )
        })?;
        axes_vals.iter().map(|&v| v as i64).collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        attr.ints.clone()
    } else {
        vec![]
    };

    let output_name = &node.output[0];

    // Handle empty axes: noop or reduce all
    let raw_axes: Vec<i64> = if raw_axes.is_empty() {
        if noop_with_empty_axes {
            tensors.insert(output_name.clone(), input);
            trace!("Finished parse: {} Node (noop)", op_name);
            return Ok(());
        } else {
            (0..ndim as i64).collect()
        }
    } else {
        raw_axes
    };

    // Normalize negative axes and convert to usize
    let mut normalized_axes: Vec<usize> = raw_axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    normalized_axes.sort();
    normalized_axes.dedup();

    // Save original sorted axes for keepdims unsqueeze bookkeeping
    let sorted_axes = normalized_axes.clone();

    let input_dims = input.dims();

    if normalized_axes.len() == ndim {
        // All-axes reduction: flatten to [1, N] and reduce axis 1 → [1].
        // luminal's Expression::product() returns 0 for empty iterators, so a reduce
        // producing a 0-dim tensor causes CUDA to launch with grid (0,1,1), which is
        // invalid. Using [1, N] → reduce(1) → [1] avoids this entirely.
        let total: usize = input_dims
            .iter()
            .map(|d| d.to_usize().expect("reduce: dim must be concrete"))
            .product();
        let mut flat = input;
        flat.shape = ShapeTracker::new(vec![1, total]);
        let mut result = all_axes_op(flat, total);

        if keepdims {
            // Insert (ndim-1) additional size-1 dims to produce [1]*ndim
            for i in 1..ndim {
                result = result.unsqueeze(i);
            }
        }

        tensors.insert(output_name.clone(), result);
        trace!("Finished parse: {} Node (all-axes)", op_name);
        return Ok(());
    }

    // Partial reduction: luminal's ToAxes API handles axis shifting internally
    let mut result = reduce_op(input, normalized_axes);

    // Re-insert size-1 dims at original positions (ascending order keeps positions correct)
    if keepdims {
        for &axis in &sorted_axes {
            result = result.unsqueeze(axis);
        }
    }

    tensors.insert(output_name.clone(), result);
    trace!("Finished parse: {} Node", op_name);
    Ok(())
}
