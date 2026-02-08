//! L1 micro benchmark patterns (single-op graphs), used by `benches/micro.rs`.

use crate::{BenchSize, BenchmarkPattern, MICRO_SIZES};
use luminal::prelude::*;

// ============================================================================
// Binary Operators
// ============================================================================

/// Vector addition benchmark: a + b
#[derive(Debug, Default)]
pub struct AddVec;

impl BenchmarkPattern for AddVec {
    fn name(&self) -> &'static str {
        "add_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = (a + b).output();
    }
}

/// Vector multiplication benchmark: a * b
#[derive(Debug, Default)]
pub struct MulVec;

impl BenchmarkPattern for MulVec {
    fn name(&self) -> &'static str {
        "mul_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = (a * b).output();
    }
}

/// Vector modulo benchmark: a % b
#[derive(Debug, Default)]
pub struct ModVec;

impl BenchmarkPattern for ModVec {
    fn name(&self) -> &'static str {
        "mod_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = (a % b).output();
    }
}

/// Vector less-than comparison benchmark: a < b
#[derive(Debug, Default)]
pub struct LessThanVec;

impl BenchmarkPattern for LessThanVec {
    fn name(&self) -> &'static str {
        "less_than_vec"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let b = cx.tensor(size.value);
        let _ = a.lt(b).output();
    }
}

// ============================================================================
// Reduction Operators
// ============================================================================

/// Sum reduction benchmark: sum(a)
#[derive(Debug, Default)]
pub struct SumReduce;

impl BenchmarkPattern for SumReduce {
    fn name(&self) -> &'static str {
        "sum_reduce"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.sum(0).output();
    }
}

/// Max reduction benchmark: max(a)
#[derive(Debug, Default)]
pub struct MaxReduce;

impl BenchmarkPattern for MaxReduce {
    fn name(&self) -> &'static str {
        "max_reduce"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.max(0).output();
    }
}

// ============================================================================
// Unary Operators
// ============================================================================

/// Exp2 benchmark: 2^x
#[derive(Debug, Default)]
pub struct Exp2Bench;

impl BenchmarkPattern for Exp2Bench {
    fn name(&self) -> &'static str {
        "exp2"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.exp2().output();
    }
}

/// Log2 benchmark: log2(x)
#[derive(Debug, Default)]
pub struct Log2Bench;

impl BenchmarkPattern for Log2Bench {
    fn name(&self) -> &'static str {
        "log2"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.log2().output();
    }
}

/// Sin benchmark: sin(x)
#[derive(Debug, Default)]
pub struct SinBench;

impl BenchmarkPattern for SinBench {
    fn name(&self) -> &'static str {
        "sin"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.sin().output();
    }
}

/// Recip benchmark: 1/x
#[derive(Debug, Default)]
pub struct RecipBench;

impl BenchmarkPattern for RecipBench {
    fn name(&self) -> &'static str {
        "recip"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.reciprocal().output();
    }
}

/// Sqrt benchmark: sqrt(x)
#[derive(Debug, Default)]
pub struct SqrtBench;

impl BenchmarkPattern for SqrtBench {
    fn name(&self) -> &'static str {
        "sqrt"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        let _ = a.sqrt().output();
    }
}

// ============================================================================
// Indexing Operators
// ============================================================================

/// Gather benchmark: gather(data, indices)
#[derive(Debug, Default)]
pub struct GatherBench;

impl BenchmarkPattern for GatherBench {
    fn name(&self) -> &'static str {
        "gather"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        // Simple 1D gather: data[indices]
        // data: 1D tensor of size.value elements
        // indices: 1D tensor selecting num_indices elements
        let num_indices = 1024.min(size.value);

        let data = cx.tensor(size.value);
        // Indices must be integer type for gather operation
        let indices = cx.tensor(num_indices).as_dtype(luminal::op::DType::Int);
        let _ = data.gather(indices).output();
    }
}

/// Cast benchmark: type conversion (f32 -> f16 -> f32)
#[derive(Debug, Default)]
pub struct CastBench;

impl BenchmarkPattern for CastBench {
    fn name(&self) -> &'static str {
        "cast"
    }

    fn sizes(&self) -> &[BenchSize] {
        MICRO_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let a = cx.tensor(size.value);
        // Cast to f16 then back to f32 to measure round-trip cost
        let _ = a
            .cast(luminal::op::DType::F16)
            .cast(luminal::op::DType::F32)
            .output();
    }
}

// ============================================================================
// Pattern Registry
// ============================================================================

/// Get all micro benchmark patterns (HLIR primitives supported by Metal)
pub fn all_micro_patterns() -> Vec<Box<dyn BenchmarkPattern>> {
    vec![
        // Binary operators
        Box::new(AddVec),
        Box::new(MulVec),
        Box::new(ModVec),
        Box::new(LessThanVec),
        // Reduction operators
        Box::new(SumReduce),
        Box::new(MaxReduce),
        // Unary operators
        Box::new(Exp2Bench),
        Box::new(Log2Bench),
        Box::new(SinBench),
        Box::new(RecipBench),
        Box::new(SqrtBench),
        // Indexing operators
        Box::new(GatherBench),
        // Note: CastBench removed - Metal backend does not implement Cast yet
    ]
}

/// Calculate bytes transferred for a benchmark pattern
pub fn bytes_for_pattern(pattern_name: &str, size: usize) -> usize {
    let elem_size = std::mem::size_of::<f32>();
    match pattern_name {
        // Binary operators: read 2 inputs + write 1 output = 3 * size * 4 bytes
        "add_vec" | "mul_vec" | "mod_vec" | "less_than_vec" => 3 * size * elem_size,

        // Reduction operators: read 1 input + write 1 output (scalar)
        "sum_reduce" | "max_reduce" => size * elem_size + elem_size,

        // Unary operators: read 1 input + write 1 output = 2 * size * 4 bytes
        "exp2" | "log2" | "sin" | "recip" | "sqrt" => 2 * size * elem_size,

        // Cast: read 1 input (f32) + write intermediate (f16) + read (f16) + write output (f32)
        // Simplified: 2 * size * 4 bytes (f32 in + f32 out)
        "cast" => 2 * size * elem_size,

        // Gather: read indices + read gathered data + write output
        // Simple 1D gather: indices (num_indices * 4) + read data + write output
        "gather" => {
            let num_indices = 1024.min(size);
            // Read indices (i32) + read data (random access, ~num_indices elements) + write output
            num_indices * elem_size + num_indices * elem_size + num_indices * elem_size
        }

        _ => 0,
    }
}
