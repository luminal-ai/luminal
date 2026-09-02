//! Logical Qwen3 model definitions.
//!
//! Runtime crates own weight loading, search, execution, and output handling.

pub mod model;

#[path = "../../common/model_support.rs"]
pub mod model_support;

pub use model::{Qwen, QwenDims};
