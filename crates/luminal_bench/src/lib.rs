//! # Luminal Benchmark Infrastructure
//!
//! Universal benchmark framework for Luminal backends.
//!
//! ## Architecture
//!
//! - **BenchmarkBackend**: Trait that backends implement to enable benchmarking
//! - **BenchmarkPattern**: Trait for defining benchmark workloads
//! - **Micro benchmarks (L1)**: Single-operator performance tests (HLIR primitives)
//! - **Pattern benchmarks (L2)**: Composite operator performance tests
//!
//! Usage 和调试方式见 crate 根目录的 `README.md`。

mod metrics;
mod micro;
mod patterns;

/// Egglog debugging and analysis utilities.
/// This module is backend-agnostic; specific backends are selected via feature flags
/// in the debug_ops example.
pub mod egglog_debug;

pub use metrics::*;
pub use micro::*;
pub use patterns::*;

use luminal::op::Runtime;
use luminal::prelude::*;

/// Hardware information for a backend device
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub device_name: String,
    pub memory_gb: f64,
    /// Peak memory bandwidth in GB/s (if known)
    pub peak_bandwidth_gbps: Option<f64>,
    /// Peak compute throughput in TFLOPS (if known)
    pub peak_tflops: Option<f64>,
}

/// Trait that backends implement to enable benchmarking
pub trait BenchmarkBackend {
    type Runtime: Runtime;

    /// Initialize the runtime
    fn initialize() -> Self::Runtime;

    /// Get backend name (used in reports)
    fn name() -> &'static str;

    /// Get hardware information
    fn hardware_info() -> HardwareInfo;
}

/// Size configuration for benchmarks
#[derive(Debug, Clone, Copy)]
pub struct BenchSize {
    pub name: &'static str,
    pub value: usize,
}

impl BenchSize {
    pub const fn new(name: &'static str, value: usize) -> Self {
        Self { name, value }
    }
}

/// Standard benchmark sizes for micro benchmarks
pub const MICRO_SIZES: &[BenchSize] = &[
    BenchSize::new("1k", 1_000),
    BenchSize::new("100k", 100_000),
    BenchSize::new("1m", 1_000_000),
    BenchSize::new("10m", 10_000_000),
];

/// Trait for defining benchmark workloads (dyn-compatible version)
pub trait BenchmarkPattern {
    /// Pattern name (used in reports)
    fn name(&self) -> &'static str;

    /// Available sizes for this pattern
    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    /// Build the computation graph for this pattern
    fn build_graph(&self, cx: &mut Graph, size: BenchSize);
}

// Re-export backend implementations when features are enabled
#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "metal")]
pub use metal_backend::MetalBenchmark;
