import os
import socket
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from luminal_reference import Compiler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def model(x, weight):
    return (x @ weight).redistribute(placements=[Replicate()]).to_local()


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        torch.manual_seed(0)
        full_x = torch.randn(4, 8)
        full_weight = torch.randn(8, 2)

        # Split the contraction dimension: [4, 4] @ [4, 2] on each rank.
        x = distribute_tensor(full_x, mesh, [Shard(1)])
        weight = distribute_tensor(full_weight, mesh, [Shard(0)])
        compiled = torch.compile(
            model, backend=Compiler(log=True), fullgraph=True, dynamic=True
        )
        with torch.no_grad():
            output = compiled(x, weight)
        torch.testing.assert_close(output, full_x @ full_weight)

        if dist.get_rank() == 0:
            print(output)
            print(
                f"Local shards: x={tuple(x.shape)} global, {tuple(x.to_local().shape)} local;"
            )
            print(
                f"              weight={tuple(weight.shape)} global, {tuple(weight.to_local().shape)} local"
            )
    finally:
        dist.destroy_process_group()


def main():
    interfaces = {name for _, name in socket.if_nameindex()}
    loopback = next((name for name in ("lo0", "lo") if name in interfaces), None)
    if loopback is None:
        raise RuntimeError("This local example requires a lo0 or lo loopback interface")
    os.environ["GLOO_SOCKET_IFNAME"] = loopback
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="luminal-spmd-") as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
