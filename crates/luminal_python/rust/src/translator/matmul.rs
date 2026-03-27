use anyhow::Result;
use luminal::prelude::*;

use crate::pt2_schema::*;
use crate::pt2_util::broadcast_binary;

use super::Translator;

impl<'a> Translator<'a> {
    pub(crate) fn translate_linear(&mut self, node: &Node) -> Result<GraphTensor> {
        let input = self.get_input_tensor(node, 0)?;
        let weight = self.get_input_tensor(node, 1)?;
        let result = input.matmul(weight.t());

        if node.inputs.len() > 2
            && let Ok(bias) = self.get_input_tensor(node, 2)
        {
            let (result, bias) = broadcast_binary(result, bias);
            return Ok(result + bias);
        }
        Ok(result)
    }
}
