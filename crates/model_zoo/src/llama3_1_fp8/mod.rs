//! Logical FP8 Llama 3.1 model definitions.
//!
//! Runtime crates own weight loading, search, execution, and output handling.

pub mod model;
pub mod rope;

pub use model::{Fp8Dims, Llama31Fp8};
