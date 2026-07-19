use luminal::prelude::*;

/// A simple linear layer
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    permute: bool,
}

impl Linear {
    pub fn new(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (inp, out)).persist(),
            bias: if bias {
                Some(cx.named_tensor("Bias", out).persist())
            } else {
                None
            },
            permute: false,
        }
    }

    pub fn new_permuted(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (out, inp)).persist(),
            bias: if bias {
                Some(cx.named_tensor("Bias", out).persist())
            } else {
                None
            },
            permute: true,
        }
    }
}

impl Linear {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        let output = input.matmul(if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        });
        if let Some(bias) = self.bias {
            output + bias.expand_lhs(&output.dims()[..output.dims().len() - 1])
        } else {
            output
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::Linear;
//     use luminal::{prelude::*, tests::assert_close};
//     #[test]
//     fn test_linear() {
//         let mut cx = Graph::new();
//         let batch = cx.tensor((2, 3)).set([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
//         let a = cx.tensor(3).set([1.0, 2.0, 3.0]);

//         let model = Linear::new(3, 4, false, &mut cx).init_rand();
//         let mut b = model.forward(a).retrieve();
//         let mut batch_out = model.forward(batch).retrieve();

//         cx.execute();

//         let unoptimized_b = b.data();
//         let unoptimized_batch_out = batch_out.data();

//         cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
//         cx.execute();

//         assert_close(&unoptimized_b, &b.data());
//         assert_close(&unoptimized_batch_out, &batch_out.data());
//     }
// }
