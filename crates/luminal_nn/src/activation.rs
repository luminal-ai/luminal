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

// #[cfg(test)]
// mod tests {
//     use super::ReLU;
//     use crate::Linear;
//     use dfdx::prelude::{Module as DfdxModule, *};
//     use luminal::{
//         prelude::{Module, *},
//         tests::assert_close,
//     };

//     #[test]
//     fn test_relu_and_linear() {
//         // Test single and batch, unoptimized and optimized
//         let mut cx = Graph::new();
//         let batch = cx.tensor((2, 3)).set(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
//         let a = cx.tensor(3).set(vec![1.0, 2.0, 3.0]);

//         let model = (
//             Linear::new(3, 4, false, &mut cx),
//             ReLU,
//             Linear::new(4, 2, false, &mut cx),
//         );
//         model
//             .0
//             .weight
//             .set(vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.]);
//         model.2.weight.set(vec![1., 2., 3., 1., 2., 3., 1., 2.]);
//         let mut b = model.forward(a).retrieve();
//         let mut batch_out = model.forward(batch).retrieve();

//         cx.execute();

//         let unoptimized_b = b.data();
//         let unoptimized_batch_out = batch_out.data();

//         cx.compile(GenericCompiler::default(), (&mut b, &mut batch_out));
//         cx.execute();

//         assert_close(&unoptimized_b, &b.data());
//         assert_close(&unoptimized_batch_out, &batch_out.data());

//         // Test against dfdx
//         let dev = Cpu::default();
//         let mut model = <(
//             dfdx::nn::modules::builders::UnbiasedLinear<3, 4>,
//             dfdx::nn::modules::builders::ReLU,
//             dfdx::nn::modules::builders::UnbiasedLinear<4, 2>,
//         )>::build_on_device(&dev);
//         // Set weights
//         model.0.weight = dev
//             .tensor_from_vec(
//                 vec![1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.],
//                 (dfdx::shapes::Const::<3>, dfdx::shapes::Const::<4>),
//             )
//             .permute();
//         model.2.weight = dev
//             .tensor_from_vec(
//                 vec![1., 2., 3., 1., 2., 3., 1., 2.],
//                 (dfdx::shapes::Const::<4>, dfdx::shapes::Const::<2>),
//             )
//             .permute();
//         let a = dev.tensor_from_vec(vec![1.0, 2.0, 3.0], (dfdx::shapes::Const::<3>,));
//         let d_batch = dev.tensor_from_vec(
//             vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
//             (dfdx::shapes::Const::<2>, dfdx::shapes::Const::<3>),
//         );
//         let out = model.forward(a);
//         let d_batch_out = model.forward(d_batch);

//         assert_close(&unoptimized_b, &out.as_vec());
//         assert_close(&unoptimized_batch_out, &d_batch_out.as_vec());
//     }
// }
