## luminal_cuda_lite

This crate contains the CUDA backend for Luminal.

The backend can be broken down into several main types of ops. Starting from the highest level and going lower:

#### Host Ops

Host ops are opaque operations executed from the host (can execute on device, simply launched in an opaque manner). cuBLAS is a good example of this type of op. Luminal can't assume much about these operations since they are so opaque. These ops implement the `HostOp` trait.

#### Kernel Ops

Kernel ops are operations encoded as a kernel and launch parameters. Luminal can put these into CUDA graphs. Cutlass kernels are good examples of these. These ops implement the `KernelOp` trait.

#### Block Ops

Block ops are operations encoded on the threadblock level, which implement an operation that runs for a duration within a single threadblock. These are required to use a fixed number of threads per threadblock (or gate unused threads out), and are given a fixed-size shared memory scratchpad. Luminal can fuse these operations into megakernels. These ops impelement the `BlockOp` trait.

#### Warp Ops

Warp ops are not yet merged. Stay tuned!

#### Thread Ops

Thread ops are not yet merged. Stay tuned!

### Architecture

`luminal_cuda_lite` can model a joint search space that smoothly searches through various mixed configurations of these ops. At compile time, a waterfall process takes place to iteratively raise each op to the level above, resulting in all host-level ops in the final runtime graph. For instance, block ops get combined into megakernels, implemented as kernel ops. Kernel ops get combined into cuda graphs, implemented as host ops.

### Semantic search contract

Backend rewrites add legal implementations with `union`; they do not remove a
legal implementation merely because another implementation is usually faster.
The profiling search, rather than cleanup, chooses between alternatives such as
generic kernels, specialized kernels, and host-library calls.

That includes GenericMatmul/cuBLASLt/GEMV, direct/decomposed Conv2D,
materialized/absorbed fusion and casts, copying/no-copy scatter, and
materialized/fused RoPE-scatter paths. These alternatives are matched in
egglog; selected LLIR is not rewritten into a different operator pattern after
extraction.

Cleanup may remove only representations that are not executable plans: cycles,
malformed shape/stride metadata, unsupported type/layout combinations, and
proven alias or ownership violations. Candidate resource checks may reject a
plan that cannot fit or launch on the target device. The intermediate-memory
cap applies to the peak planned bucket arena. Bucket dispatch drops the active
arena before allocating another, so bucket arenas peak rather than coexist. The
device-memory check separately includes that peak arena, persistent host-op state
retained by all compiled buckets, the peak transient host-op allocation, and
deduplicated shared workspaces. That check is a necessary planned-capacity bound,
not an available-memory guarantee: external allocations, CUDA context and
allocator overhead, and pool reservations are not observable in the plan. Arena
growth likewise drops the synchronized old arena before allocating its
replacement, so replacement itself does not introduce an old-plus-new peak. The
intermediate-memory and synchronous-NVRTC source budgets are reported as resource
rejections and can be adjusted independently of rewrite semantics. Otherwise, a
plan that is legal but merely expensive remains available for measured search.

Choice-set validation detects correlated e-class cycles before LLIR loading.
Random initial genomes repair only those reachable cycles; later mutations may
still produce them, in which case candidate filtering discards them without
profiling and continues searching the remaining legal alternatives.
