//! Logical Gemma 3 model definitions.
//!
//! Runtime crates own weight loading, search, execution, and output handling.

pub mod model;

#[path = "../../common/model_support.rs"]
pub mod model_support;

pub use model::{Gemma3, Gemma3Dims};
