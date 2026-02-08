pub mod egglog_utils;
pub mod frontend;
pub mod graph;
pub mod hlir;
pub mod op;
pub mod shape;
pub mod visualization;

#[cfg(test)]
pub mod tests;

pub mod prelude {
    pub use crate::egglog_utils::SerializedEGraph;
    pub use crate::frontend::binary::F32Pow;
    pub use crate::frontend::*;
    pub use crate::graph::*;
    pub use crate::hlir::NativeRuntime;
    pub use crate::op::{DType, Runtime};
    pub use crate::shape::*;
    pub use anyhow;
    pub use egglog;
    pub use egglog::ast as egglog_ast;
    pub use egraph_serialize::NodeId as ENodeId;
    pub use half::{bf16, f16};
    pub use petgraph;
    pub use petgraph::stable_graph::NodeIndex;
    pub use rustc_hash::{FxHashMap, FxHashSet};
    pub use tinyvec;
    pub use tracing;
}

pub use paste::paste;
