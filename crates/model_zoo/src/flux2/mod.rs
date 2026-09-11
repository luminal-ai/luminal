//! Logical FLUX.2-dev transformer definitions.
//!
//! Runtime crates own checkpoint loading, search, execution, scheduling, and
//! image decoding.

pub mod transformer;

pub use transformer::Flux2Transformer;
