//! L2 pattern benchmark patterns (composite graphs), used by `benches/patterns.rs`.

use crate::{BenchSize, BenchmarkPattern};
use luminal::prelude::*;

// ============================================================================
// Size Configurations
// ============================================================================

/// Matrix multiplication size configuration
#[derive(Debug, Clone, Copy)]
pub struct MatMulSize {
    pub name: &'static str,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

impl MatMulSize {
    pub const fn new(name: &'static str, m: usize, k: usize, n: usize) -> Self {
        Self { name, m, k, n }
    }
}

/// Dummy size for patterns that handle sizes internally
pub const CUSTOM_SIZE: &[BenchSize] = &[BenchSize::new("custom", 0)];

/// Standard matrix multiplication sizes
pub const MATMUL_SIZES: &[MatMulSize] = &[
    // Square matrices
    MatMulSize::new("128x128", 128, 128, 128),
    MatMulSize::new("512x512", 512, 512, 512),
    MatMulSize::new("1024x1024", 1024, 1024, 1024),
    // LLM-like shapes (batch=1, hidden_dim, ffn_dim)
    MatMulSize::new("1x4096x4096", 1, 4096, 4096),
    // MatMulSize::new("32x4096x4096", 32, 4096, 4096),
];

/// Transformer-like sizes for softmax, layernorm, etc.
pub const TRANSFORMER_SIZES: &[BenchSize] = &[
    BenchSize::new("128x128", 128 * 128), // small attention
    BenchSize::new("512x512", 512 * 512), // medium attention
    BenchSize::new("2048x128", 2048 * 128), // typical seq_len x head_dim
                                          // BenchSize::new("4096x128", 4096 * 128), // long context
];

/// Attention size configurations (seq_len, head_dim)
pub const ATTENTION_SIZES: &[(usize, usize)] = &[
    (128, 64), // small: seq=128, head_dim=64
    (512, 64), // medium: seq=512, head_dim=64
    (1024, 64), // large: seq=1024, head_dim=64
               // (2048, 64), // xlarge: seq=2048, head_dim=64
];

// ============================================================================
// MatMul Pattern
// ============================================================================

/// Matrix multiplication benchmark: C = A @ B
#[derive(Debug, Clone, Copy)]
pub struct MatMulBench {
    pub size: MatMulSize,
}

impl MatMulBench {
    pub fn new(size: MatMulSize) -> Self {
        Self { size }
    }
}

impl BenchmarkPattern for MatMulBench {
    fn name(&self) -> &'static str {
        "matmul"
    }

    fn sizes(&self) -> &[BenchSize] {
        CUSTOM_SIZE
    }

    fn build_graph(&self, cx: &mut Graph, _size: BenchSize) {
        let a = cx.tensor((self.size.m, self.size.k));
        let b = cx.tensor((self.size.k, self.size.n));
        let _ = a.matmul(b).output();
    }
}

// ============================================================================
// Softmax Pattern
// ============================================================================

/// Softmax benchmark: softmax(x, axis=-1)
#[derive(Debug, Default)]
pub struct SoftmaxBench;

impl BenchmarkPattern for SoftmaxBench {
    fn name(&self) -> &'static str {
        "softmax"
    }

    fn sizes(&self) -> &[BenchSize] {
        TRANSFORMER_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        // Reshape to 2D for softmax along last axis
        // Assume size.value = rows * cols, use sqrt for balanced shape
        let dim = (size.value as f64).sqrt() as usize;
        let rows = size.value / dim;
        let cols = dim;

        let x = cx.tensor((rows, cols));
        // Softmax along last axis (axis 1)
        let _ = x.softmax(1).output();
    }
}

// ============================================================================
// LayerNorm Pattern
// ============================================================================

/// Layer normalization benchmark
#[derive(Debug, Default)]
pub struct LayerNormBench;

impl BenchmarkPattern for LayerNormBench {
    fn name(&self) -> &'static str {
        "layer_norm"
    }

    fn sizes(&self) -> &[BenchSize] {
        TRANSFORMER_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        // Typical shape: (batch * seq_len, hidden_dim)
        let hidden_dim = 128;
        let batch_seq = size.value / hidden_dim;

        let x = cx.tensor((batch_seq.max(1), hidden_dim));
        // LayerNorm along last axis with epsilon
        let _ = x.layer_norm(1, 1e-5).output();
    }
}

// ============================================================================
// GeLU Pattern
// ============================================================================

/// GeLU activation benchmark
#[derive(Debug, Default)]
pub struct GeLUBench;

impl BenchmarkPattern for GeLUBench {
    fn name(&self) -> &'static str {
        "gelu"
    }

    fn sizes(&self) -> &[BenchSize] {
        TRANSFORMER_SIZES
    }

    fn build_graph(&self, cx: &mut Graph, size: BenchSize) {
        let x = cx.tensor(size.value);
        let _ = x.gelu().output();
    }
}

// ============================================================================
// Attention Pattern
// ============================================================================

/// Self-attention benchmark: softmax(Q @ K^T / sqrt(d)) @ V
#[derive(Debug, Clone, Copy)]
pub struct AttentionBench {
    pub seq_len: usize,
    pub head_dim: usize,
}

impl AttentionBench {
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        Self { seq_len, head_dim }
    }
}

impl Default for AttentionBench {
    fn default() -> Self {
        Self {
            seq_len: 512,
            head_dim: 64,
        }
    }
}

impl BenchmarkPattern for AttentionBench {
    fn name(&self) -> &'static str {
        "attention"
    }

    fn sizes(&self) -> &[BenchSize] {
        CUSTOM_SIZE
    }

    fn build_graph(&self, cx: &mut Graph, _size: BenchSize) {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;

        // Q, K, V tensors: (seq_len, head_dim)
        let q = cx.tensor((seq_len, head_dim));
        let k = cx.tensor((seq_len, head_dim));
        let v = cx.tensor((seq_len, head_dim));

        // Attention: softmax(Q @ K^T / sqrt(d)) @ V
        // Q @ K^T -> (seq_len, seq_len)
        let scores = q.matmul(k.permute((1, 0)));

        // Scale by 1/sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let scaled_scores = scores * scale;

        // Softmax along last axis
        let attn_weights = scaled_scores.softmax(1);

        // @ V -> (seq_len, head_dim)
        let _ = attn_weights.matmul(v).output();
    }
}

// ============================================================================
// Pattern Registry
// ============================================================================

/// Get all high-priority pattern benchmarks
pub fn all_pattern_benchmarks() -> Vec<Box<dyn BenchmarkPattern>> {
    let mut patterns: Vec<Box<dyn BenchmarkPattern>> = vec![];

    // MatMul patterns with different sizes
    for size in MATMUL_SIZES {
        patterns.push(Box::new(MatMulBench::new(*size)));
    }

    // Softmax
    patterns.push(Box::new(SoftmaxBench));

    // LayerNorm
    patterns.push(Box::new(LayerNormBench));

    // GeLU
    patterns.push(Box::new(GeLUBench));

    // Attention patterns with different sizes
    for (seq_len, head_dim) in ATTENTION_SIZES {
        patterns.push(Box::new(AttentionBench::new(*seq_len, *head_dim)));
    }

    patterns
}

/// Calculate bytes transferred for pattern benchmarks
pub fn bytes_for_pattern_bench(
    pattern_name: &str,
    size: usize,
    extra: Option<(usize, usize, usize)>,
) -> usize {
    let elem_size = std::mem::size_of::<f32>();

    match pattern_name {
        "matmul" => {
            if let Some((m, k, n)) = extra {
                // Read A (m*k) + Read B (k*n) + Write C (m*n)
                (m * k + k * n + m * n) * elem_size
            } else {
                0
            }
        }
        "softmax" => {
            // Read input + Write output (same size)
            2 * size * elem_size
        }
        "layer_norm" => {
            // Read input + Write output
            2 * size * elem_size
        }
        "gelu" => {
            // Read input + Write output
            2 * size * elem_size
        }
        "attention" => {
            if let Some((seq_len, head_dim, _)) = extra {
                // Q, K, V reads: 3 * seq_len * head_dim
                // scores: seq_len * seq_len
                // output: seq_len * head_dim
                (3 * seq_len * head_dim + seq_len * seq_len + seq_len * head_dim) * elem_size
            } else {
                0
            }
        }
        _ => 0,
    }
}
