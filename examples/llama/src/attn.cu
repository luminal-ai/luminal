#include <cmath>
{
extern __shared__ float shmem[];
const int WARP_SIZE = 32;
const int WARPS_PER_GROUP = 4;
const int THREADS_PER_GROUP = WARP_SIZE * WARPS_PER_GROUP; // 128
const int NUM_WARP_GROUPS = 8; // 1024 / 128
const int ROWS_PER_GROUP = 4;
const int FULL_MASK = 0xffffffff;

int tid = threadIdx.x;
int lane_id = tid % WARP_SIZE;
int warp_id = tid / WARP_SIZE;
int warp_group_id = tid / THREADS_PER_GROUP; // 0..7
int group_q_row = warp_group_id * ROWS_PER_GROUP; // 0, 4, 8, ..., 28
int warp_id_in_group = (tid % THREADS_PER_GROUP) / WARP_SIZE;
int gtid = tid % THREADS_PER_GROUP;

const int head_dim = payload.head_dim;
const int seq_len = eval_expression(payload.seq_len, 0);
const int num_q_tiles = eval_expression(payload.num_q_tiles, 0);
const int prev_seq = eval_expression(payload.prev_seq, 0);
const int total_seq = prev_seq + seq_len;
const int BR = payload.br;
const int BC = payload.bc;
const int n_heads = payload.n_heads;
const int kv_groups = payload.kv_groups;
const int hidden = payload.hidden;
const int kv_stride = hidden / kv_groups;

// num_q_tiles is the amount of tiles we need to fully cover the sequence length
const int head_idx = current / num_q_tiles;
const int tile_idx = current % num_q_tiles;
const int kv_head_idx = head_idx / kv_groups;
const float rscaled = rsqrtf((float)head_dim);

// Q shape (seq, hidden) == (seq, n_heads * head_dim)
// current thread block handles [tile_idx * BR: tile_idx * BR + BR]
const int q_row = tile_idx * BR;
const int q_pos = prev_seq + q_row + group_q_row + warp_id_in_group; // need for causal attn
const int q_col = head_idx * head_dim;
const int q_offset = q_row * hidden + q_col;
const float* Q_base = source_ptrs[0] + q_offset;
float* O_base = out_ptr + q_offset;

// we only need to know the head start position cause we iterate over all rows
const int kv_col = kv_head_idx * head_dim;
const float* K_base = source_ptrs[1] + kv_col;
const float* V_base = source_ptrs[2] + kv_col;

// similar to reference implementation
float* K_cache = payload.key_cache + kv_head_idx * head_dim;
float* V_cache = payload.val_cache + kv_head_idx * head_dim;
const int group_pos_local = head_idx % kv_groups;
if (tile_idx == 0 && group_pos_local == 0 && K_cache != nullptr && V_cache != nullptr) {
    // only one block needs to write to cache to avoid redundant writes
    for (int r = 0; r < seq_len; r++) {
        for (int col = tid; col < head_dim; col += blockDim.x) {
            int src_offset = r * kv_stride + col;
            int dst_offset = (prev_seq + r) * kv_stride + col;
            K_cache[dst_offset] = K_base[src_offset];
            V_cache[dst_offset] = V_base[src_offset];
        }
    }
}
__syncthreads(); // dont think this is strictly needed 

__shared__ float block_shmem_sum[32];
__shared__ float qk_scores[NUM_WARP_GROUPS][ROWS_PER_GROUP][32];
__shared__ float o_scales[NUM_WARP_GROUPS][ROWS_PER_GROUP];
__shared__ float sums_final[NUM_WARP_GROUPS][ROWS_PER_GROUP];

auto warp_reduce_sum = [&](float val) {
    for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, s);
    }
    return val;
};

auto warp_reduce_max = [&](float val) {
    for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, s));
    }
    return val;
};

auto warp_group_reduce_sum = [&](float val) {
    val = warp_reduce_sum(val);
    if (lane_id == 0) {
        block_shmem_sum[warp_id] = val;
    }
    __syncthreads();
    if (gtid == 0) {
        int group_warp_base = warp_group_id * WARPS_PER_GROUP;
        float4* wsv_ptr = (float4*)(&block_shmem_sum[group_warp_base]);
        float4 wsv = wsv_ptr[0];
        val = wsv.x + wsv.y + wsv.z + wsv.w;
    }
    return val;
};

float q_reg[ROWS_PER_GROUP] = {0};
float o_reg[ROWS_PER_GROUP] = {0};
float max_prev = -INFINITY;
float sum_prev = 0.0f;

// load our warp groups tile
#pragma unroll
for (int i = 0; i < ROWS_PER_GROUP; i++) {
    int global_q_row = q_row + group_q_row + i;
    if (global_q_row < seq_len && gtid < head_dim) {
        int q_local_offset = hidden * (group_q_row + i);
        q_reg[i] = Q_base[q_local_offset + gtid];
    } else {
        q_reg[i] = 0.0f;
    }
}

// iterate over tile chunks of KV
float *sK = shmem;
float *sV = shmem + head_dim * BR;

const int max_kv_idx = prev_seq + q_row + BR - 1;
const int num_kv_tiles_needed = (max_kv_idx / BC) + 1;

for (int j = 0; j < num_kv_tiles_needed; j++) {
    int kv_pos = j * BC + lane_id; // key-value position for causal masking
    __syncthreads();  // make sure previous iteration is finished

    // load K and V into shmem
    #pragma unroll
    for (int i = 0; i < ROWS_PER_GROUP; i++) {
        int kv_tile_row = group_q_row + i;
        int global_kv_row = j * BC + kv_tile_row;
        int kv_shmem_offset = kv_tile_row * head_dim + gtid;
        if (global_kv_row < total_seq && gtid < head_dim) {
            float k_val, v_val;
            if (global_kv_row < prev_seq) {
                int cache_offset = global_kv_row * kv_stride + gtid;
                k_val = K_cache[cache_offset];
                v_val = V_cache[cache_offset];
            } else {
                int local_row = global_kv_row - prev_seq;
                int kv_load_offset = local_row * kv_stride + gtid;
                k_val = K_base[kv_load_offset];
                v_val = V_base[kv_load_offset];
            }
            
            sK[kv_shmem_offset] = k_val;
            sV[kv_shmem_offset] = v_val;
        } else {
            sK[kv_shmem_offset] = 0.0f;
            sV[kv_shmem_offset] = 0.0f;
        }
    } 
    __syncthreads();

    for (int i = 0; i < BC; i++) {
        int global_k_row = j * BC + i;
        float cur_k = (global_k_row < total_seq) ? sK[i * head_dim + gtid] : 0.0f;
        for (int r = 0; r < ROWS_PER_GROUP; r++) {
            float cur_qk = q_reg[r] * cur_k;
            cur_qk = warp_group_reduce_sum(cur_qk);
            if (gtid == 0) {
                qk_scores[warp_group_id][r][i] = cur_qk * rscaled;
            }
        }
    }

    __syncthreads();

    float tid_score = qk_scores[warp_group_id][warp_id_in_group][lane_id];
    if (kv_pos > q_pos || kv_pos >= total_seq) tid_score = -INFINITY;
    float max_curr = warp_reduce_max(tid_score);
    max_curr = __shfl_sync(FULL_MASK, max_curr, 0); // all threads in warp see same max value
    float max_new = fmaxf(max_prev, max_curr);
    float o_scale = (j == 0) ? 1.0f : expf(max_prev - max_new);
    float exp_score = expf(tid_score - max_new);
    qk_scores[warp_group_id][warp_id_in_group][lane_id] = exp_score;
    float sum_curr = warp_reduce_sum(exp_score);
    sum_curr = __shfl_sync(FULL_MASK, sum_curr, 0); // all threads in warp see same sum value
    sum_prev = (sum_prev * o_scale) + sum_curr;
    max_prev = max_new;

    if (lane_id == 0) {
        o_scales[warp_group_id][warp_id_in_group] = o_scale;
        if (j == num_kv_tiles_needed - 1) {
            sums_final[warp_group_id][warp_id_in_group] = sum_prev;
        }
    }
    __syncthreads();

    for (int i = 0; i < ROWS_PER_GROUP; i++) {
        float scale = o_scales[warp_group_id][i];
        float out_val = 0.0f;
        for (int v = 0; v < BC; v++) {
            int global_v_row = j * BC + v;
            if (global_v_row >= total_seq) break;
            float score = qk_scores[warp_group_id][i][v];
            float v_val = sV[v * head_dim + gtid];
            out_val += score * v_val;
        }
        o_reg[i] = o_reg[i] * scale + out_val;
    }
}

__syncthreads();

for (int i = 0; i < ROWS_PER_GROUP; i++) {
    int global_out_row = q_row + group_q_row + i;
    if (global_out_row < seq_len && gtid < head_dim) {
        float curout = o_reg[i];
        float denom = sums_final[warp_group_id][i];
        float inv_sum = (denom != 0) ? (1.0f / denom) : 0.0f;
        curout *= inv_sum;
        int offset = hidden * (group_q_row + i) + gtid;
        O_base[offset] = curout;
    }
}

}