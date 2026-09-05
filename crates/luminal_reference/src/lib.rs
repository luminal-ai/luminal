//! The reference runtime: the trusted, loud, out-of-place host
//! executor every other runtime is measured against — now a
//! standalone crate (Step B, ruling 2026-08-17). The core crate is
//! runtime-neutral; everything reference-flavored lives here: the
//! runtime and its ladder, the op registry (structs, matchers,
//! snippets, kernels), and the defaulting conveniences that assume
//! this registry.
//!
//! POST-SATURATION SELECTION IS OURS ([`search`]): the op registry, the
//! allow list, the host-timing evaluator, the option knobs, the outcome
//! shape and the GA loop that runs them. What decides nothing — the
//! e-graph walk and the genome sampler — is core's
//! (`luminal::extraction`, `luminal::search_support`), and [`extractor`]
//! below is this crate's name for the walk (#420/#422 rejoin Phase 8,
//! 2026-09-04; it was a verbatim copy here through Phases 1-7).

pub mod bindings;
/// The e-graph walk, in core: every runtime calls it with its own
/// matcher list and it names no runtime type. Kept under this crate's
/// old module name so call sites read the same.
pub use luminal::extraction as extractor;
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
pub use runtime::{ReferenceRuntime, reference_allow_list};
pub use search::{
    CompileOptions, SearchOutcome, harness_search_options, search_implementations,
    search_implementations_with_ops,
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

/// The reference registry's CONSTRUCTOR DECODERS (core's built-ins plus
/// this crate's matcher contributions), memoized — what every reference
/// [`luminal::egglog_utils::eclass::EGraphView`] is built with, and what
/// the assembly tripwire checks the saturated program against.
pub fn decoder_registry() -> &'static luminal::egglog_utils::eclass::ConstructorRegistry {
    use std::sync::OnceLock;
    static REGISTRY: OnceLock<luminal::egglog_utils::eclass::ConstructorRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        luminal::egglog_snippet::decoder_registry_for(&ops::built_in_matchers())
            .expect("the built-in reference registry has no duplicate decoders")
    })
}
