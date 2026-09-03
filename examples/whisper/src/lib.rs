//! Logical Whisper model definitions.
//!
//! Runtime crates own audio preparation, weight loading, search, execution,
//! and output handling.

pub mod model;

#[path = "../../common/model_support.rs"]
pub mod model_support;

pub use model::{Whisper, WhisperDims};
