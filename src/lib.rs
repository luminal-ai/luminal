// `Symbol` is an immutable interned identity. Its storage uses interior
// mutability, which makes Clippy reject every map keyed by the public symbol
// type even though the key's hash and ordering never change.
#![allow(clippy::mutable_key_type)]

pub mod dtype;
#[path = "egglog_core/egglog_utils/mod.rs"]
pub mod egglog_utils;
pub mod frontend;
pub mod graph;
pub mod runtime_binding;
pub mod shape;

// The logical-SSA layout compiler. Egglog program assembly and registries live
// beside the ops; the fixture suite runs as core tests. See src/egglog_core/
// for the core preamble and fixtures.
//
// WHAT IS NOT HERE (ruling 2026-09-03, #420/#422 rejoin Phase 1):
// post-saturation SEARCH. The extractor and the selection loop are
// runtime-owned — `luminal_reference::{extractor, search}`,
// `luminal_cuda_lite::{extractor, search}`, `test_runtime::{extractor,
// sampler}` — and core keeps only what every runtime shares: the
// logical program, the egglog assembly, `dps_rewrite`,
// `layouts::decode_layout_table`, `bufferize`, and the IR types.
pub mod buffer_tensor_ir;
pub mod bufferize;
pub mod dps;
pub mod egglog_snippet;
pub mod index_expr;
pub mod layout_ir;
pub mod poison;
pub mod subst_primitive;
// Convenience mirrors of the five egglog Layout constructors + SpanExpr +
// decode_layout, for runtimes to pull from one place. THE BUFFERIZER NEVER
// CALLS ANY OF THIS — the planner stays generic over an opaque layout type,
// and backends may ignore this module entirely (Austin's fold-into-core
// amendment, resident-geometry cleanup 2026-08-31).
pub mod layouts;
pub mod logical_op;
pub mod test_support;
pub mod visualization;

#[cfg(test)]
pub mod tests;

pub mod prelude {
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
