//! MiniConvNet — the yolo-family mini model definition (relocated here
//! from luminal_nn per the 2026-08-13 ruling: mini model definitions
//! live in their example crates; luminal_nn keeps only building blocks).

use crate::namespace::Namespace;

use luminal::prelude::*;

/// The yolo-family mini: two valid-padding conv layers with relu, then a
/// linear classification head over the flattened features.
pub struct MiniConvNet {
    pub conv1_weight: GraphTensor,
    pub conv1_bias: Option<GraphTensor>,
    pub conv1_config: luminal_nn::ConvNdConfig,
    pub conv2_weight: GraphTensor,
    pub conv2_bias: Option<GraphTensor>,
    pub conv2_config: luminal_nn::ConvNdConfig,
    pub head_weight: GraphTensor,
    pub head_bias: Option<GraphTensor>,
    classes: usize,
}

impl MiniConvNet {
    /// Input (ch_in, h, w) with h = w = 5 and 3×3 valid convs: 5→3→1.
    pub fn new(ch_in: usize, c1: usize, c2: usize, classes: usize, cx: &mut Graph) -> Self {
        Self {
            conv1_weight: cx.named_tensor(
                Namespace::root().child("conv1").leaf("weight"),
                vec![c1, ch_in * ([3, 3]).iter().product::<usize>()],
                DType::F32,
            ),
            conv1_bias: None,
            conv1_config: luminal_nn::ConvNdConfig::new([3, 3], [1, 1], [1, 1], [0, 0]),
            conv2_weight: cx.named_tensor(
                Namespace::root().child("conv2").leaf("weight"),
                vec![c2, c1 * ([3, 3]).iter().product::<usize>()],
                DType::F32,
            ),
            conv2_bias: None,
            conv2_config: luminal_nn::ConvNdConfig::new([3, 3], [1, 1], [1, 1], [0, 0]),
            head_weight: cx.named_tensor(
                Namespace::root().child("head").leaf("weight"),
                vec![c2, classes],
                DType::F32,
            ),
            head_bias: None,
            classes,
        }
    }

    /// x (1, ch_in, 5, 5) → logits (classes,).
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x =
            luminal_nn::conv_nd(x, self.conv1_weight, self.conv1_bias, &self.conv1_config).relu(); // (1, c1, 3, 3)
        let x =
            luminal_nn::conv_nd(x, self.conv2_weight, self.conv2_bias, &self.conv2_config).relu(); // (1, c2, 1, 1)
        let flat = x.flatten(); // (c2,)
        let logits = luminal_nn::linear(flat.expand_lhs(vec![1]), self.head_weight, self.head_bias); // (1, classes)
        let _ = self.classes;
        logits.squeeze(0)
    }
}
