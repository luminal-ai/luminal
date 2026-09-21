# Memory retention options — 2026-09-21

CUDA memory pools are usable for both device memory and pinned host staging on this machine. The CUDA driver reports API version 13000 (driver 580.126.09); installed build headers are CUDA 12.8. The small capability probe created DEVICE, HOST_NUMA (node 0), and HOST pools successfully, allocated/freed/reallocated from each, and validated host writes plus a 1 MiB H2D transfer and a one-byte readback for both host locations.

The current `cudarc` dependency is the Luminal fork at `a42f3897b6d00aded8e6ac7ad258c72d8bad5596`. Its `CudaStream::alloc_zeros` already calls `cuMemAllocAsync` when `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED` is true, as it is here. The fresh process's default device pool release threshold was 0. The expensive `Pinned` wrapper instead calls `cuMemHostAlloc` and `cuMemFreeHost` directly.

## Full-size host-pool probe

`pool_probe_full.c` uses a private HOST_NUMA pool on node 0, a 64 GiB release threshold, and a 32,658,510,964-byte allocation. Allocation/free measurements include stream synchronization. It writes every byte on the CPU after allocation completes. Each trial also checks a 1 MiB transfer to the GPU and reads one byte back. This is an isolated allocator experiment, not a rerun of model search and not a full-bandwidth transfer benchmark.

| Operation | First use | Reuse |
|---|---:|---:|
| Allocate + synchronize | 3548.704 ms | 0.015 ms |
| Fill entire CPU buffer | 2056.346 ms | 1157.852 ms |
| Free to pool + synchronize | 0.014 ms | 0.012 ms |

After each logical free the pool retained 32,682,016,768 bytes. Destroying the pool at the end took 1643.642 ms. The first allocation and first full touch together took 5.605 seconds. CPU filling/copying remains necessary when data must be initialized; the fast reuse number excludes that work. Prior direct-allocation measurements were 16.453 s allocation, 0.420 s zeroing, and 6.422 s free, in a separate process/run.

## Follow-up: grant the GPU access to the host pool

The full-search experiment exposed a transfer requirement that the small capability probe did not measure. A HOST_NUMA pool is initially CPU-accessible. Before allocating staging, grant the selected CUDA device `CU_MEM_ACCESS_FLAGS_PROT_READWRITE` using `cuMemPoolSetAccess`. This preserves direct asynchronous transfer performance on this machine.

An isolated 32,658,510,964-byte H2D test measured three repetitions per allocator:

| Host allocation | Transfer wall time | Time inside the async-copy API call |
|---|---:|---:|
| Original `cuMemHostAlloc` | 0.569 s | 9–20 microseconds |
| HOST_NUMA pool with default CPU access | 1.500–1.501 s | 1.500–1.501 s |
| HOST_NUMA pool with device READWRITE access | 0.569 s | 11–14 microseconds |

The test transferred every byte, then checked the first and last GPU bytes. CPU initialization and allocation were outside the copy timers. Raw source and results are in `pool-comparison-cpu-only/transfer_probe.c` and `transfer_probe.log`.

## Options for the runtime

1. **Retain the existing host buffer and device arena.** Split candidate graph teardown from allocation release. Keep the allocations until they need to grow, an explicit trim, or runtime destruction. This is the smallest source change and removes repeated allocation/free without requiring a host-pool API. Every new candidate must still initialize all state it depends on; reuse must not leak KV contents between candidates.
2. **Use private CUDA pools with a retention policy.** Grant the GPU READWRITE access to a HOST_NUMA pool, allocate host staging via `cuMemAllocFromPoolAsync`, free via `cuMemFreeAsync`, and keep the pool alive across candidates/searches. Keep its release threshold large enough for the intended working set and provide an explicit `cuMemPoolTrimTo` path. The threshold is a retention policy, not a hard cache-size limit. Default threshold 0 allows reclamation at synchronization; this runtime synchronizes each execution. Device pooling is already active, so device retention is a policy improvement rather than a switch from synchronous allocation.
3. **Keep model weights resident.** Give profiling the serving runtime's resident bindings so warmups/trials can retain immutable GPU weights and use a small reusable upload buffer. Cross-candidate reuse additionally needs stable weight storage separate from candidate scratch/graph lifetimes. Reusing an allocation alone does not preserve valid weight bindings or remove repeated weight copies. Reset mutable KV/scratch state as required.
4. **Retain the owner across calls.** Repeated in-process searches/runs can share a live runtime/allocator owner. The current plan invalidation calls `release_slab`, so the retained allocations must have an independent lifetime. Across separate CLI invocations, a long-lived worker/service can own the context, pools, modules, and weights. A CUDA pool is not a persistent disk cache after its owner exits.
5. **Caller-owned GPU arena.** The existing `CudaDevice::set_external_arena` hook allows a caller to retain device storage. Pinned staging currently needs a corresponding retention policy; this GPU-only hook does not address its allocation cost.

A practical implementation can use one retained pinned staging buffer for serial candidate evaluation, use a host pool when multiple sizes/lifetimes need reuse, and expose separate operations for clearing candidate plans and trimming memory. Memory limits and initialized state must be tracked independently of retained capacity.

## Sources and reproduction

- [CUDA 12.8 host-NUMA pool creation](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html): specifies HOST_NUMA pool properties and asynchronous allocation ordering.
- [CUDA pool release threshold](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__TYPES.html): the default is 0; synchronization may release unused backing memory above the threshold.
- `pool_probe.c`, `pool_probe.log`: capability and small-allocation checks.
- `pool_probe_full.c`, `pool_probe_full.log`: full-size measurements above.

Compile either probe with `gcc -O2 -I/usr/local/cuda/include pool_probe_full.c -lcuda -o /tmp/pool_probe_full`, then run `/tmp/pool_probe_full`. The full probe temporarily reserves about 32.7 GB of pinned RAM and releases its private pool at exit. These probes do not change production allocator code or pool defaults.
