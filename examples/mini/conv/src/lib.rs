//! MiniConvNet — the yolo-family mini model definition (relocated here
//! from luminal_nn per the 2026-08-13 ruling: mini model definitions
//! live in their example crates; luminal_nn keeps only building blocks).

#[path = "../../../common/model_support.rs"]
mod model_support;

use crate::model_support::{ConvND, Linear, Namespace};
use luminal::prelude::*;

/// The yolo-family mini: two valid-padding conv layers with relu, then a
/// linear classification head over the flattened features.
pub struct MiniConvNet {
    pub conv1: ConvND,
    pub conv2: ConvND,
    pub head: Linear,
    classes: usize,
}

impl MiniConvNet {
    /// Input (ch_in, h, w) with h = w = 5 and 3×3 valid convs: 5→3→1.
    pub fn new(ch_in: usize, c1: usize, c2: usize, classes: usize, cx: &mut Graph) -> Self {
        Self {
            conv1: ConvND::new(
                ch_in,
                c1,
                [3, 3],
                [1, 1],
                [1, 1],
                [0, 0],
                false,
                DType::F32,
                &Namespace::root().child("conv1"),
                cx,
            ),
            conv2: ConvND::new(
                c1,
                c2,
                [3, 3],
                [1, 1],
                [1, 1],
                [0, 0],
                false,
                DType::F32,
                &Namespace::root().child("conv2"),
                cx,
            ),
            head: Linear::new(
                c2,
                classes,
                false,
                DType::F32,
                &Namespace::root().child("head"),
                cx,
            ),
            classes,
        }
    }

    /// x (1, ch_in, 5, 5) → logits (classes,).
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let x = self.conv1.forward(x).relu(); // (1, c1, 3, 3)
        let x = self.conv2.forward(x).relu(); // (1, c2, 1, 1)
        let flat = x.flatten(); // (c2,)
        let logits = self.head.forward(flat.expand_lhs(1)); // (1, classes)
        let _ = self.classes;
        logits.squeeze(0)
    }
}
