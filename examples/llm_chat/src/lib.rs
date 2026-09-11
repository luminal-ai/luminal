//! A backend-independent chat application. Checkpoint conventions and the
//! common LLM graph contract belong here, alongside the application adapters.
pub mod backend;
pub mod checkpoint;
pub mod graph;
pub mod sampling;
pub mod session;
pub mod tokenizer;

use luminal::prelude::{FxHashMap, NodeIndex};

/// The application's host data; backends own their device representations.
#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}
pub type Inputs = FxHashMap<NodeIndex, TensorData>;
