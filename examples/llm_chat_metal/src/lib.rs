//! Chat with a model-zoo LLM on the Metal runtime. This example is
//! self-contained: the model graph, checkpoint conventions, tokenization,
//! sampling, the session loop and the CLI live here, and [`backend`] holds
//! the Metal runtime's own boundary bindings for the graph.
pub mod app;
pub mod backend;
pub mod checkpoint;
pub mod graph;
pub mod sampling;
pub mod session;
pub mod tokenizer;

use luminal::prelude::{FxHashMap, NodeIndex};

/// The application's host data; the backend owns the device representation.
#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I32(Vec<i32>),
}
pub type Inputs = FxHashMap<NodeIndex, TensorData>;
