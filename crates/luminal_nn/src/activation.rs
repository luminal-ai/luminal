use luminal::prelude::*;

/// Rectified Linear Unit activation function
#[derive(Default)]
pub struct ReLU;

impl ReLU {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.relu()
    }
}

/// Gaussian Error Linear Unit activation function.
///
/// Uses the exact erf form (`gelu`), matching PyTorch's `nn.GELU()` default
/// (`approximate="none"`). For the cheaper tanh approximation, call
/// `GraphTensor::gelu_fast_tanh_approximation()`.
#[derive(Default)]
pub struct GeLU;

impl GeLU {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.gelu()
    }
}

/// Sigmoid activation function
#[derive(Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.sigmoid()
    }
}

/// Swish activation function
#[derive(Default)]
pub struct Swish;

impl Swish {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.swish()
    }
}

/// Tanh activation function
#[derive(Default)]
pub struct Tanh;

impl Tanh {
    pub fn forward(&self, input: GraphTensor) -> GraphTensor {
        input.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::{ReLU, Sigmoid};
    use luminal::prelude::*;

    /// The M3 ladder end-to-end (full genetic search; scalar-constant
    /// broadcasts route via materialize — rediagnosis 2026-08-12).
    fn run_unary(build: impl Fn(GraphTensor) -> GraphTensor, input: Vec<f32>) -> Vec<f32> {
        let mut cx = Graph::new();
        let x = cx.tensor(input.len());
        let out = build(x).output();
        let mut data = rustc_hash::FxHashMap::default();
        data.insert(x.id, input.clone().into());
        let mut rt = luminal::reference::ReferenceRuntime::load(&cx).expect("native load");
        rt.search(
            &data,
            &luminal::implementation_search::ImplementationSearchOptions::default(),
        )
        .expect("search finds a plan");
        rt.set_data(x.id, input);
        rt.execute().expect("winner executes");
        rt.get_f32(out.id).expect("output").clone()
    }

    #[test]
    fn relu_clamps_negatives_natively() {
        let got = run_unary(|x| ReLU.forward(x), vec![-2., -0.5, 0., 0.5, 2.]);
        for (ours, expected) in got.iter().zip([0., 0., 0., 0.5, 2.]) {
            assert!((ours - expected).abs() < 1e-6, "{got:?}");
        }
    }

    #[test]
    fn sigmoid_matches_reference_natively() {
        let input = vec![-2., -1., 0., 1., 2.];
        let got = run_unary(|x| Sigmoid.forward(x), input.clone());
        for (ours, x) in got.iter().zip(&input) {
            let expected = 1.0 / (1.0 + (-x).exp());
            assert!((ours - expected).abs() < 1e-5, "{got:?}");
        }
    }
}
