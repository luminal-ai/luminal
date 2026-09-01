pub mod dtype;
#[path = "egglog_core/egglog_utils/mod.rs"]
pub mod egglog_utils;
pub mod frontend;
pub mod graph;
pub mod reference_binding;
pub mod shape;

// The logical-SSA layout compiler. Egglog program assembly and registries live
// beside the ops; the fixture suite runs as core tests. See src/egglog_core/
// for the core preamble and fixtures.
pub mod buffer_tensor_ir;
pub mod bufferize;
pub mod dps;
pub mod egglog_snippet;
pub mod extractor;
pub mod implementation_search;
pub mod index_expr;
pub mod layout_ir;
pub mod logical_op;
pub mod reference;
pub mod subst_primitive;
pub mod test_support;
pub mod visualization;

#[cfg(test)]
pub mod tests;

pub mod prelude {
    pub use crate::buffer_tensor_ir::TypedBuffer;
    pub use crate::dtype::DType;
    pub use crate::frontend::binary::F32Pow;
    pub use crate::frontend::*;
    pub use crate::graph::*;
    pub use crate::shape::*;
    pub use crate::visualization::{ToDot, ToHtml};
    pub use anyhow;
    pub use egglog;
    pub use egglog::ast as egglog_ast;
    pub use egraph_serialize;
    pub use egraph_serialize::NodeId as ENodeId;
    pub use float8;
    pub use half::{bf16, f16};
    pub use petgraph;
    pub use petgraph::stable_graph::NodeIndex;
    pub use rustc_hash::{FxHashMap, FxHashSet};
    pub use tinyvec;
    pub use tracing;
}

pub use paste::paste;
