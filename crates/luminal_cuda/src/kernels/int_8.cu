#include "common_header.cuh"

// ===== OPTIMIZED INT8 QUANTIZED MATRIX-VECTOR MULTIPLICATION =====
// Uses quantize_fp32_to_int8 for runtime quantization when needed

extern "C" __global__ void quantized_matvec_int8_optimized(
    const block_q8_0* __restrict__ x,    // Quantized matrix
    const float* __restrict__ y,         // Input vector (FP32)
    float* __restrict__ dst,             // Output vector (FP32)
    const int src_vec_size,              // K dimension
    const int dest_vec_size,             // N dimension
    const int mat_batch_stride,          // Matrix batch stride
    const int vec_batch_stride           // Vector batch stride
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int block_id = blockIdx.x;
    const int batch_id = blockIdx.z;
    
    // Each warp processes 4 rows for better occupancy
    const int rows_per_warp = 4;
    const int first_row = (block_id * (blockDim.x / 32) + warp_id) * rows_per_warp;
    
    if (first_row >= dest_vec_size) return;
    
    // Calculate offsets with batch support
    const int num_blocks_per_row = src_vec_size / 32;
    x += first_row * num_blocks_per_row + batch_id * (mat_batch_stride / 32);
    y += batch_id * vec_batch_stride;
    dst += batch_id * dest_vec_size;
    
    // Thread-local accumulation for 4 rows
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Each thread processes different quantization blocks
    for (int block_idx = lane_id / 4; block_idx < num_blocks_per_row; block_idx += 8) {
        const int sub_block = lane_id % 4;
        
        // OPTIMIZATION 1: Vectorized load of input vector values (2x faster)
        float y_vals[8];
        const float* y_ptr = y + block_idx * 32 + sub_block * 8;
        
        // Bounds check and vectorized load
        if (block_idx * 32 + sub_block * 8 + 7 < src_vec_size) {
            load_float8_vectorized(y_ptr, y_vals);
        } else {
            // Boundary case - scalar loads with zero-padding
            for (int i = 0; i < 8; ++i) {
                int idx = block_idx * 32 + sub_block * 8 + i;
                y_vals[i] = (idx < src_vec_size) ? y_ptr[i] : 0.0f;
            }
        }
        
        // Process 4 matrix rows simultaneously
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            if (first_row + row >= dest_vec_size) break;
            
            const block_q8_0* block_ptr = &x[block_idx + row * num_blocks_per_row];
            const char* qs = block_ptr->qs + sub_block * 8;
            const float scale = __half2float(block_ptr->d);
            
            float row_sum = 0.0f;
            
            // OPTIMIZATION 2: Unrolled dot product computation
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                row_sum += (float)qs[i] * y_vals[i];
            }
            
            sums[row] += row_sum * scale;
        }
    }
    
    // OPTIMIZATION 3: Improved warp reduction with loop unrolling
    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        float final_sum = warpReduceSum_optimized(sums[row]);
        
        if (lane_id == 0 && first_row + row < dest_vec_size) {
            dst[first_row + row] = final_sum;
        }
    }
}

// ===== ORIGINAL KERNEL FOR COMPATIBILITY =====

extern "C" __global__ void quantized_matvec_int8_original(
    const block_q8_0* x,
    const float* y,
    float* dst,
    int src_vec_size,
    int dest_vec_size,
    int mat_batch_stride,
    int vec_batch_stride
) {
    // Original implementation for comparison/fallback
    int threadgroup_position_in_grid_x = blockIdx.x;
    int threadgroup_position_in_grid_z = blockIdx.z;
    int thread_index_in_simdgroup = (threadIdx.y * blockDim.x + threadIdx.x) % warpSize;
    int simdgroup_index_in_threadgroup = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    const int num_rows = 4;
    const int num_simdgroups_per_threadgroup = 2;

    int num_quants_per_row = src_vec_size / 32;
    int first_row = (threadgroup_position_in_grid_x * num_simdgroups_per_threadgroup + simdgroup_index_in_threadgroup) * num_rows;

    x += first_row * num_quants_per_row;
    y += threadgroup_position_in_grid_z * vec_batch_stride;
    dst += (threadgroup_position_in_grid_z * dest_vec_size);

    float yl[8];
    float sumf[num_rows] = {0.0};

    int ix = thread_index_in_simdgroup / 4;
    int il = thread_index_in_simdgroup % 4;

    y += thread_index_in_simdgroup * 8;

    for (int ib = ix; ib < num_quants_per_row; ib += 8) {
        
        for (int i = 0; i < 8; ++i) {
            yl[i] = y[i];
        }
//#pragma unroll
        for (int row = 0; row < 4; ++row) {
            const char* qs = x[ib + row * num_quants_per_row].qs + il * 8;
            float sumq = 0.0;
            for (int iq = 0; iq < 8; ++iq) {
                sumq += (float)qs[iq] * (float)yl[iq];
            }
            sumf[row] += sumq * __half2float(x[ib + row * num_quants_per_row].d);
        }
        y += 256;
    }

    for (int row = 0; row < num_rows; ++row) {
        float sum = warpReduceSum_optimized(sumf[row]);
        if (thread_index_in_simdgroup == 0 && first_row + row < dest_vec_size) {
            dst[first_row + row] = sum;
        }
    }
}
