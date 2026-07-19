//! Training utilities for luminal.
//!
//! Reverse-mode autograd implemented as an HLIR graph transformation: call
//! [`Backward::backward`] after building the forward graph and before
//! `compile`. Gradients are ordinary HLIR ops appended to the same graph, so
//! the whole forward+backward program is optimized by one e-graph search.
//!
//! ```no_run
//! use luminal::prelude::*;
//! use luminal_training::Backward;
//!
//! let mut cx = Graph::new();
//! let w = cx.tensor((3, 4));
//! let x = cx.tensor((2, 3));
//! let loss = x.matmul(w).sum((0, 1));
//! let grads = cx.backward(loss, &[w]);
//! let grad_out = grads[0].output();
//! // ...compile and execute as usual; read grad_out like any output.
//! ```

mod autograd;
mod optim;
mod reference;
mod view;

pub use autograd::{Backward, backward};
pub use optim::{AdamW, Optimizer, OptimizerStep, SGD};
pub use reference::{Trainer, rebind_buffer, restore_inputs, snapshot_inputs};
pub use view::unview;

pub mod prelude {
    pub use crate::{AdamW, Backward, Optimizer, OptimizerStep, SGD, backward};
}
