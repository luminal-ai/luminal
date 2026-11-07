# AllGather

`tensor.all_gather(group, width, sequence)` stacks one equal-shaped tensor from
each rank, in communicator order. An input of shape `[M, N]` produces
`[width, M, N]` with the same dtype. This is AllGather, not root-to-all broadcast
or AllToAll. Group and sequence are part of the logical operation's identity.
Sequences must be unique within each group in a program.

All participants must execute the same program with matching group membership,
sequence order, shapes and dtypes. Each rank supplies only its local input.
Compile with `runtime.compile(&bounds, &dimensions)`, then execute concurrently
on the participating ranks. The compile method uses existing egglog rewrites,
extraction and bufferization to select a deterministic viable plan without
running it. It does not profile or claim a timing result. Ordinary rank-local
`search`/profiling deliberately refuses collective graphs: differing candidate
counts or early stopping could otherwise issue unmatched collectives.

## Reference and Metal

These backends use the core `AllGatherTransport` interface. `TcpAllGather` is a
complete built-in transport for threads or separate processes; applications can
also supply their own transport. Each named group has a separate root address.
Rank zero listens, and the other ranks connect. Construct all endpoints
concurrently, then bind the endpoint around execution:

```rust,no_run
use std::{sync::Arc, time::Duration};
use luminal::collective::{TcpAllGather, with_group};
# fn example(rank: usize) -> anyhow::Result<()> {
let endpoint = Arc::new(TcpAllGather::connect(
    "127.0.0.1:29501".parse()?, rank, 2, Duration::from_secs(30),
)?);
with_group("workers", endpoint, || {
    // runtime.execute(...)
});
# Ok(()) }
```

The transport checks each call's group, sequence, iteration, dtype and shape
before accepting payloads. A failed exchange poisons that endpoint; create a new
group rather than attempting to resume half an exchange. Connection/read/write
timeouts bound failures. Payloads preserve storage bits, including NaNs, narrow
float encodings, and full-width integers.

Reference uses its typed host storage. Metal finishes preceding GPU work,
copies the local contiguous input to shared host storage, exchanges the bytes,
and uploads the result before later GPU commands. Metal's path is functional
host-staged communication; it does not claim GPU-direct transport. Existing
layout rewrites materialize noncontiguous GPU inputs before AllGather.

## CUDA Lite

The `device` feature enables native NCCL. Tensor payloads stay on the GPU:
`ncclAllGather` records on the runtime's CUDA stream and is retained in the
captured graph. Transfer counts use physical storage bytes, preserving every
supported dtype's encoding. A one-rank group uses a device copy and needs no
NCCL communicator.

Generate one ID with `NcclCommunicator::unique_id()`, distribute those 128 bytes
using the application's rendezvous mechanism, and concurrently initialize each
rank with `NcclCommunicator::init(&device, id, rank, width)`. Use the same device
for the runtime and bind the communicator while first preparing/executing it:

```rust,ignore
use luminal_cuda_lite::collective::{NcclCommunicator, with_group};
let group = NcclCommunicator::init(&device, id, rank, width)?;
let mut runtime = CudaRuntime::load(&graph)?.with_device(&device)?;
runtime.compile(&bounds, &dimensions)?;
with_group("workers", group, || runtime.execute(arena, &mut staging))?;
```

NCCL must be installed on CUDA execution hosts. Initialization, graph capture
and replay must be coordinated across ranks as required by NCCL. The prepared
operation retains its communicator and CUDA context for graph lifetime. A
borrowed communicator can be installed through unsafe `from_raw`; its owner
must outlive every graph using it. Rebinding a name does not change an already
captured graph; create a fresh runtime to change communicators.

The reference artifact format saves AllGather metadata but never sockets or
communicators. Restore the artifact and bind an appropriate local endpoint
before execution. The PT2 translator recognizes
`luminal_collective.all_gather.default` with `(input, group, width, sequence)`.
Existing PyTorch AOT region execution may continue to delegate functional
collectives to PyTorch; native Rust AllGather uses the runtime API above.

## Validation

`cargo test -p luminal --test all_gather` checks identity, invalid metadata,
profiling refusal and transport mismatches. The reference AllGather integration
tests exercise real TCP exchanges, dynamic and empty tensors, scalar and strided
inputs, encoding preservation, repeated calls, and artifact serialization.
CUDA and Metal host tests verify registration and extraction. Device tests live
in each backend's `tests/all_gather.rs`; the NCCL test is explicitly enabled with
`--features device -- --ignored` and requires two CUDA devices. Metal's device
case runs on macOS and can use two rank threads on one GPU.
