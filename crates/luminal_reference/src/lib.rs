//! The reference runtime: the trusted, loud, out-of-place host
//! executor every other runtime is measured against — now a
//! standalone crate (Step B, ruling 2026-08-17). The core crate is
//! runtime-neutral; everything reference-flavored lives here: the
//! runtime and its ladder, the op registry (structs, matchers,
//! snippets, kernels), and the defaulting conveniences that assume
//! this registry.

pub mod bindings;
pub mod harness;
pub mod kernels;
pub mod layouts;
pub mod ops;
pub mod runtime;
pub mod search;

pub use bindings::ReferenceBindings;
pub use harness::{extract_layout_ir, extract_layout_ir_with_genome, producer_index_with_ops};
pub use layouts::ReferencePlan;
pub use runtime::{reference_allow_list, ReferenceRuntime};
pub use search::{search_implementations, search_implementations_with_ops, ReferenceProfiler};

/// The reference-registry assembled program (core's
/// `assembled_program_for` with this crate's matchers), memoized.
pub fn assembled_program() -> &'static str {
    use std::sync::OnceLock;
    static PROGRAM: OnceLock<String> = OnceLock::new();
    PROGRAM
        .get_or_init(|| luminal::egglog_snippet::assembled_program_for(&ops::built_in_matchers()))
}
