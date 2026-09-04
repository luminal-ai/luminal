//! The reference runtime: the trusted, loud, out-of-place host
//! executor every other runtime is measured against — now a
//! standalone crate (Step B, ruling 2026-08-17). The core crate is
//! runtime-neutral; everything reference-flavored lives here: the
//! runtime and its ladder, the op registry (structs, matchers,
//! snippets, kernels), and the defaulting conveniences that assume
//! this registry.
//!
//! POST-SATURATION SEARCH IS OURS (ruling 2026-09-03, #420/#422 rejoin
//! Phase 1): [`extractor`] and [`search`] are this crate's own copies of
//! machinery that used to live in core. Core keeps what every runtime
//! shares — the logical program, the e-graph assembly, `dps_rewrite`,
//! `decode_layout_table`, `bufferize` — and nothing that decides which
//! implementation wins.

pub mod bindings;
pub mod extractor;
pub mod harness;
pub mod kernels;
pub mod layouts;
pub mod ops;
pub mod runtime;
pub mod search;
pub mod typed_buffer;

pub use bindings::ReferenceBindings;
pub use harness::{extract_layout_ir, extract_layout_ir_with_genome, producer_index_with_ops};
pub use layouts::ReferencePlan;
pub use runtime::{reference_allow_list, ReferenceRuntime};
pub use search::{
    harness_search_options, search_implementations, search_implementations_with_ops,
    CompileOptions, SearchOutcome,
};
pub use typed_buffer::{ReferenceKernelCtx, TypedBuffer};

/// The reference-registry assembled program (core's
/// `assembled_program_for` with this crate's matchers), memoized.
pub fn assembled_program() -> &'static str {
    use std::sync::OnceLock;
    static PROGRAM: OnceLock<String> = OnceLock::new();
    PROGRAM
        .get_or_init(|| luminal::egglog_snippet::assembled_program_for(&ops::built_in_matchers()))
}
