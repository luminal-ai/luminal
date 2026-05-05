//! Binary-inclusive elementwise kernel fusion.
//!
//! - `markers` — `FusionStart` / `FusionEnd` ops + the seven egglog rule
//!   families that build and extend FE-bracketed regions.
//! - `fused_ops` — eight `FusedX` op variants (interior to a region) so
//!   pair-fuse rules' RHS sit in a different egglog sort than their LHS,
//!   blocking cascade by typing.
//! - `region_codegen` — `kernel_to_host` calls into here to collapse each
//!   FE-rooted region into a single CUDA kernel at compile time.
//!
//! The LLIR keeps `FusionStart` / `FusedX` / `FusionEnd` nodes after
//! extraction; `region_codegen` is the only place that walks them.

pub mod fused_ops;
pub mod markers;
pub mod region_codegen;

pub use fused_ops::{
    FusedAdd, FusedExp, FusedExp2, FusedLog2, FusedMul, FusedRecip, FusedSin, FusedSqrt,
};
pub use markers::{FusionEnd, FusionStart};

/// All fusion-related op types that the egglog runtime needs to know about
/// (markers + interior FusedX variants). Combined into a flat tuple for the
/// `Ops` registry in `kernel::mod`.
pub type Ops = (markers::Ops, fused_ops::Ops);
