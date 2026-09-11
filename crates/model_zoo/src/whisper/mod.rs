//! Logical Whisper model definitions.
//!
//! Runtime crates own audio preparation, weight loading, search, execution,
//! and output handling.

pub mod model;

pub use model::{Whisper, WhisperDims};
