use rand::{rng, Rng};

use luminal::prelude::*;

/// A simple unbiased linear layer
pub struct Linear {
    pub weight: GraphTensor,
    pub bias: Option<GraphTensor>,
    permute: bool,
}

impl Linear {
    pub fn new(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (inp, out)),
            bias: if bias {
                Some(cx.named_tensor("Bias", out))
            } else {
                None
            },
            permute: false,
        }
    }

    pub fn new_permuted(inp: usize, out: usize, bias: bool, cx: &mut Graph) -> Self {
        Self {
            weight: cx.named_tensor("Weight", (out, inp)),
            bias: if bias {
                Some(cx.named_tensor("Bias", out))
            } else {
                None
            },
            permute: true,
        }
    }

    pub fn init_rand(self) -> Self {
        // Init weight as uniform(-1, 1)
        let mut rng = rng();
        self.weight.set(
            (0..self.weight.shape.n_elements().to_usize().unwrap())
                .map(|_| rng.random_range(-1_f32..1_f32))
                .collect::<Vec<_>>(),
        );
        if let Some(bias) = self.bias {
            bias.set(
                (0..bias.shape.n_elements().to_usize().unwrap())
                    .map(|_| rng.random_range(-1_f32..1_f32))
                    .collect::<Vec<_>>(),
            );
        }
        self
    }
}

impl SerializeModule for Linear {
    fn serialize(&self, s: &mut luminal::module::Serializer) {
        s.tensor("weight", self.weight);
        if let Some(bias) = self.bias {
            s.tensor("bias", bias);
        }
    }
}

impl Module<GraphTensor> for Linear {
    type Output = GraphTensor;

    fn forward(&self, input: GraphTensor) -> Self::Output {
        let mut output = input.matmul(if self.permute {
            self.weight.permute((1, 0))
        } else {
            self.weight
        });
        if let Some(bias) = self.bias {
            output += bias.expand(output.shape);
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::Linear;
    use luminal::{prelude::*, tests::assert_close};
    #[test]
    fn test_linear() {
        let mut cx = Graph::new();
        let batch = cx.tensor((2, 3)).set([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let a = cx.tensor(3).set([1.0, 2.0, 3.0]);

        let model = Linear::new(3, 4, false, &mut cx).init_rand();
        let mut b = model.forward(a).retrieve();
        let mut batch_out = model.forward(batch).retrieve();

        cx.execute();

        let unoptimized_b = b.data();
        let unoptimized_batch_out = batch_out.data();

        cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
        cx.execute();

        assert_close(&unoptimized_b, &b.data());
        assert_close(&unoptimized_batch_out, &batch_out.data());
    }
}
