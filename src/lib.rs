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
// WHAT IS NOT HERE: post-saturation SELECTION. Each runtime owns the
// loop that chooses an implementation and everything that decides — the
// op registry, the allow list, the evaluator that prices a plan, the
// option knobs and outcome shape, the finalist policy:
// `luminal_reference::search`, `luminal_cuda_lite::search`.
//
// What core keeps is what decides nothing: the logical program, the
// egglog assembly, `dps_rewrite`, `layouts::decode_layout_table`,
// `bufferize`, the IR types — and, since #420/#422 rejoin Phase 8
// (2026-09-04), the two halves of search that make no choice:
//
//   [`extraction`]      — turning a genome into an `ExtractedGraph`.
//   [`search_support`]  — drawing genomes, plus the vocabulary a search
//                         reports itself in: `RefusalBreakdown`,
//                         `SearchTimings`, `SearchProgress`,
//                         `early_stop_exceeded`.
//
// Both left core in Phase 1 and were duplicated verbatim into the
// runtimes (three copies of the walk and the sampler, two of the
// reporting). Seven phases later the copies differed by one API-shape
// hunk and zero logic lines, so the ruling's own "maybe if there are
// some core utilities, they can belong in core" clause was taken. The
// runtime modules named `extractor` and `sampler` are aliases for
// these, and each runtime's `search` re-exports what it used to define.
//
// The line between the two sides is not "core vs runtime" but
// "describes vs decides". A genome is drawn the same way everywhere; a
// genome is PRICED differently everywhere, and pricing happens in the
// middle of the loop — which is why the loop is not here.
pub mod buffer_tensor_ir;
pub mod bufferize;
pub mod dps;
pub mod egglog_snippet;
pub mod extraction;
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
pub mod search_support;
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
