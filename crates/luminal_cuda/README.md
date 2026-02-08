## luminal_cuda

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

`luminal_cuda` can model a joint search space that smoothly searches through various mixed configurations of these ops. At compile time, a waterfall process takes place to iteratively raise each op to the level above, resulting in all host-level ops in the final runtime graph. For instance, block ops get combined into megakernels, implemented as kernel ops. Kernel ops get combined into cuda graphs, implemented as host ops.
