{
extern __shared__ float shmem[];

// register pressure, so define these ahead of time
#define WARP_SIZE 32
#define WARPS_PER_GROUP 4
#define THREADS_PER_GROUP 128
#define NUM_WARP_GROUPS 8
#define ROWS_PER_GROUP 4
#define FULL_MASK 0xffffffff
#define NEG_INF (-__int_as_float(0x7f800000))
#define HEAD_DIM 128
#define BR 32
#define BC 32
#define HIDDEN 4096

int tmp; // reuse registers
int tid = threadIdx.x;
int lane_id = tid % WARP_SIZE;
int warp_id = tid / WARP_SIZE;
int warp_group_id = tid / THREADS_PER_GROUP; // 0..7
int group_q_row = warp_group_id * ROWS_PER_GROUP; // 0, 4, 8, ..., 28
int warp_id_in_group = (tid % THREADS_PER_GROUP) / WARP_SIZE;
int gtid = tid % THREADS_PER_GROUP;

const int seq_len = eval_expression(payload.seq_len, 0);
const int num_q_tiles = eval_expression(payload.num_q_tiles, 0);
const int prev_seq = eval_expression(payload.prev_seq, 0);
const int total_seq = prev_seq + seq_len;
// kv can have fewer heads than q
const int kv_stride = HIDDEN / payload.kv_groups;

// num_q_tiles is the amount of tiles we need to fully cover the sequence length
const int head_idx = current / num_q_tiles;
const int tile_idx = current % num_q_tiles;
// map q head to kv head
const int kv_head_idx = head_idx / payload.kv_groups;
const float rscaled = rsqrtf((float)HEAD_DIM);

// Q shape (seq, hidden) == (seq, n_heads * head_dim)
// current thread block handles [tile_idx * BR: tile_idx * BR + BR]
const int q_row = tile_idx * BR;
const int q_pos = prev_seq + q_row + group_q_row + warp_id_in_group; // need for causal attn
tmp = q_row * HIDDEN + head_idx * HEAD_DIM; // q_offset
const float* Q_base = source_ptrs[0] + tmp;
float* O_base = out_ptr + tmp;

// we only need to know the head start position cause we iterate over all rows
tmp = kv_head_idx * HEAD_DIM; // kv_col
// KV source_ptrs point to new tokens (i.e. prev_seq : total_seq - 1)
const float* K_base = source_ptrs[1] + tmp;
const float* V_base = source_ptrs[2] + tmp;

// similar to reference implementation
tmp = kv_head_idx * HEAD_DIM;
float* K_cache = payload.key_cache + tmp;
float* V_cache = payload.val_cache + tmp;
// only first q head and first q tile in group writes to cache 
if (
    tile_idx == 0 &&
    (head_idx % payload.kv_groups) == 0 &&
    payload.key_cache != nullptr &&
    payload.val_cache != nullptr)
{
    // only one block needs to write to cache to avoid redundant writes
    for (int r = 0; r < seq_len; r++) {
        for (int col = tid; col < HEAD_DIM; col += blockDim.x) {
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

// macros instead of lambdas to help alleviate register pressure
#define warp_reduce_sum(val) \
    do { \
        for (int _s = WARP_SIZE / 2; _s > 0; _s >>= 1) { \
            (val) += __shfl_down_sync(FULL_MASK, (val), _s); \
        } \
    } while(0)

#define warp_reduce_max(val) \
    do { \
        for (int _s = WARP_SIZE / 2; _s > 0; _s >>= 1) { \
            (val) = fmaxf((val), __shfl_down_sync(FULL_MASK, (val), _s)); \
        } \
    } while(0)

#define warp_group_reduce_sum(val) \
    do { \
        for (int _s = WARP_SIZE / 2; _s > 0; _s >>= 1) { \
            (val) += __shfl_down_sync(FULL_MASK, (val), _s); \
        } \
        if (lane_id == 0) { \
            block_shmem_sum[warp_id] = (val); \
        } \
        __syncthreads(); \
        if (gtid == 0) { \
            (val) = block_shmem_sum[warp_group_id * WARPS_PER_GROUP] \
                  + block_shmem_sum[warp_group_id * WARPS_PER_GROUP + 1] \
                  + block_shmem_sum[warp_group_id * WARPS_PER_GROUP + 2] \
                  + block_shmem_sum[warp_group_id * WARPS_PER_GROUP + 3]; \
        } \
    } while(0)


float q_reg[ROWS_PER_GROUP] = {0};
float o_reg[ROWS_PER_GROUP] = {0};
float max_prev = NEG_INF;
float sum_prev = 0.0f;

// load our warp groups tile
for (int i = 0; i < ROWS_PER_GROUP; i++) {
    int global_q_row = q_row + group_q_row + i;
    if (global_q_row < seq_len && gtid < HEAD_DIM) {
        int q_local_offset = HIDDEN * (group_q_row + i);
        q_reg[i] = Q_base[q_local_offset + gtid];
    } else {
        q_reg[i] = 0.0f;
    }
}

float *sK = shmem;
float *sV = shmem + HEAD_DIM * BC;

// const int max_kv_idx = prev_seq + q_row + BR - 1;
tmp = prev_seq + q_row + BR - 1; // max_kv_idx
const int num_kv_tiles_needed = ( tmp / BC) + 1;

for (int j = 0; j < num_kv_tiles_needed; j++) {
    int kv_pos = j * BC + lane_id; // key-value position for causal masking
    __syncthreads();  // make sure previous iteration is finished

    // load K and V into shmem
    for (int idx = tid; idx < BC * HEAD_DIM; idx += blockDim.x) {
        int global_kv_row = j * BC + (idx / HEAD_DIM);
        int col = idx % HEAD_DIM;
        if (global_kv_row < total_seq) {
            float k_val, v_val;
            if (global_kv_row < prev_seq) {
                int offset = global_kv_row * kv_stride + col;
                k_val = K_cache[offset];
                v_val = V_cache[offset];
            } else {
                int offset = (global_kv_row - prev_seq) * kv_stride + col;
                k_val = K_base[offset];
                v_val = V_base[offset];
            }
            sK[idx] = k_val;
            sV[idx] = v_val;
        } else {
            sK[idx] = 0.0f;
            sV[idx] = 0.0f;
        }
    }
    __syncthreads();

    for (int i = 0; i < BC; i++) {
        int global_k_row = j * BC + i;
        float cur_k = (global_k_row < total_seq) ? sK[i * HEAD_DIM + gtid] : 0.0f;
        for (int r = 0; r < ROWS_PER_GROUP; r++) {
            float cur_qk = q_reg[r] * cur_k;
            warp_group_reduce_sum(cur_qk);
            if (gtid == 0) {
                qk_scores[warp_group_id][r][i] = cur_qk * rscaled;
            }
        }
    }

    __syncthreads();

    float tid_score = qk_scores[warp_group_id][warp_id_in_group][lane_id];
    if (kv_pos > q_pos || kv_pos >= total_seq) tid_score = NEG_INF;
    float max_curr = tid_score; warp_reduce_max(max_curr); // these are macros
    max_curr = __shfl_sync(FULL_MASK, max_curr, 0); // all threads in warp see same max value
    float max_new = fmaxf(max_prev, max_curr);
    float o_scale = (j == 0) ? 1.0f : expf(max_prev - max_new);
    float exp_score = expf(tid_score - max_new);
    qk_scores[warp_group_id][warp_id_in_group][lane_id] = exp_score;
    float sum_curr = exp_score; warp_reduce_sum(sum_curr); // is a macro
    sum_curr = __shfl_sync(FULL_MASK, sum_curr, 0); // all threads in warp see same sum value
    sum_prev = (sum_prev * o_scale) + sum_curr;
    max_prev = max_new;

    if (lane_id == 0) {
        o_scales[warp_group_id][warp_id_in_group] = o_scale;
        if (j == num_kv_tiles_needed - 1) {
            // only need to store final sum for output write
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
            float v_val = sV[v * HEAD_DIM + gtid];
            out_val += score * v_val;
        }
        o_reg[i] = o_reg[i] * scale + out_val;
    }

}

__syncthreads();

for (int i = 0; i < ROWS_PER_GROUP; i++) {
    int global_out_row = q_row + group_q_row + i;
    if (global_out_row < seq_len && gtid < HEAD_DIM) {
        float denom = sums_final[warp_group_id][i];
        float inv_sum = (denom != 0) ? (1.0f / denom) : 0.0f;
        int offset = HIDDEN * (group_q_row + i) + gtid;
        O_base[offset] = o_reg[i] * inv_sum;
    }
}

// clean up macros
#undef WARP_SIZE
#undef WARPS_PER_GROUP
#undef THREADS_PER_GROUP
#undef NUM_WARP_GROUPS
#undef ROWS_PER_GROUP
#undef FULL_MASK
#undef NEG_INF
#undef HEAD_DIM
#undef BR
#undef BC
#undef HIDDEN
#undef warp_reduce_sum
#undef warp_reduce_max
#undef warp_group_reduce_sum

}