const int N_OPS = 0;

enum OpCode {
  //%extra_op_codes%
};

//%extra_op_structs%

union Payload {
  //%extra_op_payloads%
};

struct Task {
  OpCode op;
  int range;
  int remaining;
  int in_dep_a_stride;
  int in_dep_a_base;
  int in_dep_b_stride;
  int in_dep_b_base;
  int in_dep_c_stride;
  int in_dep_c_base;
  int out_dep_stride;
  int out_dep_base;
  const float *source_ptrs[3];
  float *out_ptr;
  Payload payload;
};

struct SMEvent {
  unsigned long long start;
  unsigned long long stop;
  int event;
};

//%constants%

__device__ __noinline__ int eval_expression(int expression, int const_z) {
  switch (expression) {
    //%expr_fns%
  }
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

//%extra_op_functions%

//%extra_prologue_functions%

__device__ __forceinline__ void nanosleep(unsigned int cycles) {
  asm volatile("nanosleep.u32 %0;" ::"r"(cycles));
}

__device__ __forceinline__ int atomic_load_acquire(int *addr) {
  int val;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

struct NextTask {
  int current;
  int task_idx;
};

// Lock-free task fetching using atomicSub for claiming (reduces CAS contention)
// remaining encoding:
//   -1 = uninitialized
//   > 0 = iterations remaining (atomicSub to claim, iteration = old - 1)
//   <= 0 = exhausted
__device__ inline bool fetch_next_task(Task *tasks, int num_tasks, int *head,
                                       NextTask *out, int *queue_lock) {
  while (true) {
    int idx = atomic_load_acquire(head);
    if (idx >= num_tasks)
      return false;

    Task *t = &tasks[idx];
    int remaining = atomicAdd(&t->remaining, 0);

    // Handle uninitialized task - one CAS to initialize
    if (remaining == -1) {
      int range = eval_expression(t->range, 0);
      atomicCAS(&t->remaining, -1, range);
      continue;
    }

    // Task already exhausted, advance head
    if (remaining <= 0) {
      atomicMax(head, idx + 1);
      continue;
    }

    // Claim via atomicSub - guaranteed to make progress, no CAS retry
    int old = atomicSub(&t->remaining, 1);

    if (old > 0) {
      out->task_idx = idx;
      out->current = old - 1;
      if (old == 1) {
        atomicMax(head, idx + 1);
      }
      return true;
    }

    // Race: exhausted between check and atomicSub, advance head
    atomicMax(head, idx + 1);
  }
}

__device__ inline void record_event(SMEvent *__restrict__ timings,
                                    int *event_idx, int event_type) {
  if (*event_idx < 1000) {
    unsigned long long now = read_globaltimer();
    if (*event_idx > 0) { // record the end of the previous op
      timings[*event_idx - 1].stop = now;
    }
    timings[*event_idx].start = now;
    timings[*event_idx].stop = 0ull;
    timings[*event_idx].event = event_type;
    (*event_idx)++;
  }
}

extern "C" {
__global__ void worker_kernel(Task *__restrict__ tasks, int num_tasks,
                              int *__restrict__ head, int *__restrict__ ready,
                              int *__restrict__ queue_lock,
                              SMEvent *__restrict__ timings,
                              unsigned long long *__restrict__ start_times) {
  __shared__ NextTask nt;
  __shared__ int done;
  __shared__ int dep_out;
  __shared__ bool run_a_prologue;
  __shared__ bool run_b_prologue;
  __shared__ bool run_c_prologue;
  __shared__ bool stop_wait_loop;
  __shared__ float scratchpad[8192]; // 32 KB scratchpad
  int recorded_event = 0;
  timings += blockIdx.x * 1000;
  if (threadIdx.x == 0) {
    start_times[blockIdx.x] = read_globaltimer();
  }
  while (true) {
    if (threadIdx.x == 0) {
      record_event(timings, &recorded_event, 0); // Record issue start
      done = !fetch_next_task(tasks, num_tasks, head, &nt, queue_lock);
    }
    __syncthreads();
    if (done)
      break;

    const Task *t = &tasks[nt.task_idx];
    int dep_a = 0;
    int dep_b = 0;
    int dep_c = 0;

    // Thread 0 calculates dependencies and waits for inputs
    if (threadIdx.x == 0) {
      // Note: atomic_load_acquire provides visibility for ready array
      dep_a = (t->in_dep_a_base == -1
                   ? 0
                   : (eval_expression(t->in_dep_a_base, 0) +
                      eval_expression(t->in_dep_a_stride, nt.current)));
      dep_b = (t->in_dep_b_base == -1
                   ? 0
                   : (eval_expression(t->in_dep_b_base, 0) +
                      eval_expression(t->in_dep_b_stride, nt.current)));
      dep_c = (t->in_dep_c_base == -1
                   ? 0
                   : (eval_expression(t->in_dep_c_base, 0) +
                      eval_expression(t->in_dep_c_stride, nt.current)));
      dep_out = eval_expression(t->out_dep_base, 0) +
                eval_expression(t->out_dep_stride, nt.current);

      // Increment the output barrier to signal an op is in-flight
      atomicAdd(&ready[dep_out], 1);

      record_event(timings, &recorded_event, 1); // Record wait start

      // Wait on input dependencies and run prologues as inputs become ready
      run_a_prologue = false;
      run_b_prologue = false;
      run_c_prologue = false;
      stop_wait_loop = false;
    }
    __syncthreads();

    bool a_done = false, b_done = false, c_done = false, tmp;
    // Optimize: if deps are same, reuse atomic load result
    const bool ab_same = (dep_a == dep_b);
    const bool ac_same = (dep_a == dep_c);
    const bool bc_same = (dep_b == dep_c);

    while (true) {
      if (threadIdx.x == 0) {
        // Derive x_done and run_x_prologue with optimized atomic loads
        if (!a_done) {
          tmp = atomic_load_acquire(&ready[dep_a]) <= 0;
          if (tmp) {
            run_a_prologue = true;
            a_done = true;
            // Propagate to same deps
            if (ab_same) { run_b_prologue = true; b_done = true; }
            if (ac_same) { run_c_prologue = true; c_done = true; }
          }
        }
        if (!b_done && !ab_same) {
          tmp = atomic_load_acquire(&ready[dep_b]) <= 0;
          if (tmp) {
            run_b_prologue = true;
            b_done = true;
            if (bc_same) { run_c_prologue = true; c_done = true; }
          }
        }
        if (!c_done && !ac_same && !bc_same) {
          tmp = atomic_load_acquire(&ready[dep_c]) <= 0;
          if (tmp) {
            run_c_prologue = true;
            c_done = true;
          }
        }
        if (a_done && b_done && c_done)
          stop_wait_loop = true;
      }
      __syncthreads();

      // Early exit if all dependencies satisfied (skip prologue checks)
      if (stop_wait_loop)
        break;

      if (run_a_prologue) {
        switch (t->op) {
          //%prologue_a_calls%
        }
        if (threadIdx.x == 0) {
          run_a_prologue = false;
        }
      }
      if (run_b_prologue) {
        switch (t->op) {
          //%prologue_b_calls%
        }
        if (threadIdx.x == 0) {
          run_b_prologue = false;
        }
      }
      if (run_c_prologue) {
        switch (t->op) {
          //%prologue_c_calls%
        }
        if (threadIdx.x == 0) {
          run_c_prologue = false;
        }
      }

      __syncthreads();
    }
    if (threadIdx.x == 0)
      record_event(timings, &recorded_event,
                   t->op + 2); // Record main op, ends Wait

    // Execute main operation
    switch (t->op) {
      //%extra_op_calls%
    }
    __syncthreads();

    // Arrive at output barrier
    if (threadIdx.x == 0) {
      __threadfence();
      atomicSub(&ready[dep_out], 1);
    }
  }

  if (threadIdx.x == 0 && recorded_event > 0) {
    timings[recorded_event - 1].stop = read_globaltimer();
  }
}
}
