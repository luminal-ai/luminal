//! Logical YOLO11 model definitions.
//!
//! Runtime crates own image preparation, weight loading, search, execution,
//! and post-processing.

pub mod model;

pub use model::YoloV11;
