//! Functional neural-network building blocks for Luminal graphs.
//!
//! Every learned parameter, persistent buffer, and runtime state tensor is
//! supplied by the caller. This crate composes those tensors into graph
//! computations; it does not allocate externally bound tensors or assign
//! checkpoint names.

mod convolution;
pub use convolution::*;
mod embedding;
pub use embedding::*;
mod linear;
pub use linear::*;
mod norm;
pub use norm::*;
mod pooling;
pub use pooling::*;
mod moe;
pub use moe::*;
mod attention;
pub use attention::*;
mod cache;
pub use cache::*;

#[cfg(test)]
mod api_boundary_tests {
    #[test]
    fn production_modules_do_not_name_external_tensors() {
        let sources = [
            include_str!("attention.rs"),
            include_str!("cache.rs"),
            include_str!("convolution.rs"),
            include_str!("embedding.rs"),
            include_str!("linear.rs"),
            include_str!("moe.rs"),
            include_str!("norm.rs"),
            include_str!("pooling.rs"),
        ];
        let forbidden_factory = ["named", "tensor"].concat();
        let namespace = ["Name", "space"].concat();
        for source in sources {
            assert!(!source.contains(&forbidden_factory));
            assert!(!source.contains(&namespace));
        }
    }
}
