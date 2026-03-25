use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::*;

use super::Translator;

impl<'a> Translator<'a> {
    pub(crate) fn translate_reduction(
        &mut self,
        node: &Node,
        op: ReductionOp,
    ) -> Result<GraphTensor> {
        let a = self.get_input_tensor(node, 0)?;
        let dims = self.get_ints_arg(node, 1)?;
        let keepdim = if node.inputs.len() > 2 {
            self.get_bool_arg(node, 2).unwrap_or(false)
        } else {
            false
        };

        let ndim = a.shape.len();
        let axes: Vec<usize> = dims.iter().map(|&d| normalize_dim(d, ndim)).collect();

        let mut result = match op {
            ReductionOp::Sum => a.sum(axes.clone()),
            ReductionOp::Mean => a.mean(axes.clone()),
            ReductionOp::Max => a.max(axes.clone()),
            ReductionOp::Min => a.min(axes.clone()),
        };

        if keepdim {
            let mut sorted_axes = axes.clone();
            sorted_axes.sort();
            for &ax in &sorted_axes {
                result = result.unsqueeze(ax);
            }
        }

        Ok(result)
    }
}
