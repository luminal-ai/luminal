"""Real two-rank DTensor training through ReferenceRuntime and Gloo."""

import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from luminal_reference import Compiler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cpu", (2,))
        torch.manual_seed(83)
        full_weight = torch.randn(4, 3)
        weight = torch.nn.Parameter(
            distribute_tensor(full_weight.clone(), mesh, [Shard(0)])
        )
        eager_weight = torch.nn.Parameter(
            distribute_tensor(full_weight.clone(), mesh, [Shard(0)])
        )
        dense_weight = torch.nn.Parameter(full_weight.clone())
        optimizers = [
            torch.optim.SGD([w], lr=0.01) for w in (weight, eager_weight, dense_weight)
        ]
        backend = Compiler()

        def model(x, w):
            # Contracting shards produce Partial(sum); DTensor inserts the
            # reduction when requesting a replicated output.
            first = (x @ w).redistribute(placements=[Replicate()]).to_local()
            second = (x @ (w * 0.5)).redistribute(placements=[Replicate()]).to_local()
            return first.square() + second

        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        for _ in range(2):
            full_x = torch.randn(2, 4)
            x = distribute_tensor(full_x.clone(), mesh, [Shard(1)]).requires_grad_()
            eager_x = distribute_tensor(
                full_x.clone(), mesh, [Shard(1)]
            ).requires_grad_()
            dense_x = full_x.clone().requires_grad_()
            actual = compiled(x, weight)
            eager = model(eager_x, eager_weight)
            dense = (dense_x @ dense_weight).square() + dense_x @ (dense_weight * 0.5)
            torch.testing.assert_close(actual, eager)
            torch.testing.assert_close(actual, dense)
            actual.square().sum().backward()
            eager.square().sum().backward()
            dense.square().sum().backward()
            torch.testing.assert_close(
                weight.grad.to_local(), eager_weight.grad.to_local()
            )
            torch.testing.assert_close(x.grad.to_local(), eager_x.grad.to_local())
            torch.testing.assert_close(weight.grad.full_tensor(), dense_weight.grad)
            torch.testing.assert_close(x.grad.full_tensor(), dense_x.grad)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            torch.testing.assert_close(weight.full_tensor(), dense_weight)

        assert (backend.leader_compilations > 0) == (rank == 0)
        assert {r.phase for r in backend.regions} == {"forward", "backward"}
        assert all(r.executions >= 2 for r in backend.regions)
        forward_shapes = {
            shape
            for r in backend.regions
            if r.phase == "forward"
            for shape in r.input_shapes
        }
        assert (2, 2) in forward_shapes and (2, 3) in forward_shapes
        assert (2, 4) not in forward_shapes and (4, 3) not in forward_shapes
        assert any("all_reduce" in op for g in backend.graphs for op in g.collectives)
        assert any("wait_tensor" in op for g in backend.graphs for op in g.collectives)
        assert (
            sum(
                "all_reduce" in op
                for g in backend.graphs
                if g.phase == "forward"
                for op in g.collectives
            )
            == 2
        )
        dynamic_backend = Compiler()

        def dynamic_model(x, w):
            return (x @ w).redistribute(placements=[Replicate()]).to_local()

        dynamic_compiled = torch.compile(
            dynamic_model, backend=dynamic_backend, fullgraph=True, dynamic=True
        )
        with torch.no_grad():
            for batch in (5, 7, 9):
                dense_input = torch.randn(batch, 4)
                sharded = distribute_tensor(dense_input, mesh, [Shard(1)])
                torch.testing.assert_close(
                    dynamic_compiled(sharded, weight), dense_input @ dense_weight
                )
        assert len(dynamic_backend.graphs) == 1
        assert len(dynamic_backend.regions) == 1
        assert dynamic_backend.regions[0].targets == ("aten.mm.default",)
        assert dynamic_backend.leader_compilations == (1 if rank == 0 else 0)
        assert all(region.executions == 3 for region in dynamic_backend.regions)
    finally:
        dist.destroy_process_group()


def test_two_rank_dtensor_training(tmp_path):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=False)
    deadline = time.monotonic() + 240
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("two-rank reference training exceeded 240 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
