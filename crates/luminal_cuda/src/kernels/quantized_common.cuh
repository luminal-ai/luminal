#pragma once
#include "common_header.cu"

// Re-export all common quantization functionality
// This header provides the interface for quantized operations

// The common_header.cu already contains:
// - block_q8_0 and block_q4_0 structures
// - warpReduceSum_optimized function
// - load_float8_vectorized function
// - quantize_fp32_to_int8 function

// Additional quantized operation helpers can be added here as needed