//! Logical Gemma 4 MoE model definitions.
//!
//! Runtime crates own weight loading, search, execution, and output handling.

pub mod model;

pub use model::{Gemma4Dims, Gemma4Moe};
