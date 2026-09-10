//! Price each operation's memory traffic once in the executable DAG.

use crate::{layouts::MetalPlan, symbolic::Expr};
use anyhow::{Result, anyhow};
use luminal::{bufferize::BufferNode, layouts::DecodedLayout, shape::DynMap};

fn tensor_bytes(layout: &DecodedLayout, dims: &DynMap) -> Result<u128> {
    let elements = layout.shape().0.iter().try_fold(1u128, |n, extent| {
        n.checked_mul(Expr(extent.clone()).eval(dims)? as u128)
            .ok_or_else(|| anyhow!("heuristic element count overflow"))
    })?;
    let width = u128::try_from(layout.width_bits())?;
    Ok(elements
        .checked_mul(width)
        .ok_or_else(|| anyhow!("heuristic bit count overflow"))?
        .div_ceil(8))
}

pub fn heuristic_cost_of(plan: &MetalPlan, dims: &DynMap) -> Result<u128> {
    let mut total = 1u128;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = node
        {
            for (index, tensor) in operand_info.iter().enumerate() {
                if op.operand_reads_memory(index) {
                    total = total.saturating_add(tensor_bytes(&tensor.layout, dims)?);
                }
            }
            for (index, tensor) in result_info.iter().enumerate() {
                if op.result_writes_memory(index) {
                    total = total.saturating_add(tensor_bytes(&tensor.layout, dims)?);
                }
            }
        }
    }
    Ok(total)
}
