"""Three-rank integration coverage for PyTorch collectives between local regions."""

import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from luminal_reference import Compiler


def _check(model, value, expected, gradient=None):
    if gradient is not None:
        value = value.detach().requires_grad_()
    backend = Compiler()
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    first = compiled(value)
    torch.testing.assert_close(first, expected)
    if gradient is not None:
        first.square().sum().backward()
        torch.testing.assert_close(value.grad, gradient)
        value.grad = None
    result = compiled(value)
    torch.testing.assert_close(result, expected)
    if gradient is not None:
        result.square().sum().backward()
        torch.testing.assert_close(value.grad, gradient)
    assert any(graph.collectives for graph in backend.graphs)
    assert all(
        "c10d" not in target for region in backend.regions for target in region.targets
    )
    assert all(region.executions == 2 for region in backend.regions)
    assert result.data_ptr() != value.data_ptr() or not value.numel()


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=60),
    )
    ops = torch.ops._c10d_functional
    group = dist.group.WORLD.group_name
    try:
        # Noncontiguous source, non-power-of-two world, all supported reductions.
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3).t() + rank + 1
        for reduction in ("sum", "product", "min", "max", "avg"):

            def reduce(value, reduction=reduction):
                return ops.wait_tensor(ops.all_reduce(value, reduction, group))

            peers = torch.stack([x - rank + peer for peer in range(3)])
            expected = {
                "sum": peers.sum(0),
                "product": peers.prod(0),
                "min": peers.amin(0),
                "max": peers.amax(0),
                "avg": peers.mean(0),
            }[reduction]
            _check(reduce, x, expected)

        def gather(value):
            return ops.wait_tensor(ops.all_gather_into_tensor(value, 3, group))

        _check(gather, x, torch.cat([x - rank + peer for peer in range(3)]), 6 * x)

        def scatter(value):
            return ops.wait_tensor(ops.reduce_scatter_tensor(value, "sum", 3, group))

        _check(
            scatter, x, (3 * (x - rank) + 3)[rank : rank + 1], 2 * (3 * (x - rank) + 3)
        )

        # Matching send/receive matrix includes uneven and empty peer payloads.
        counts = [[1, 0, 2], [2, 1, 0], [0, 2, 1]]
        ins = counts[rank]
        outs = [row[rank] for row in counts]
        value = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 10 * rank

        def exchange(value):
            return ops.wait_tensor(ops.all_to_all_single(value, outs, ins, group))

        pieces = []
        for peer in range(3):
            start = sum(counts[peer][:rank])
            source = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 10 * peer
            pieces.append(source[start : start + counts[peer][rank]])
        _check(exchange, value, torch.cat(pieces), 2 * value)

        # Reuse forward/backward regions as leading dimensions and split sizes change.
        def dynamic_reduce(value):
            return ops.wait_tensor(ops.all_reduce(value, "sum", group))

        def dynamic_gather(value):
            return ops.wait_tensor(ops.all_gather_into_tensor(value, 3, group))

        def dynamic_scatter(value):
            return ops.wait_tensor(ops.reduce_scatter_tensor(value, "sum", 3, group))

        def dynamic_exchange(value):
            splits = [value.shape[0] // 3] * 3
            return ops.wait_tensor(ops.all_to_all_single(value, splits, splits, group))

        def dynamic_uneven_exchange(value):
            width = value.shape[0] // 3
            splits = [width - 1, width, width + 1]
            return ops.wait_tensor(
                ops.all_to_all_single(value, [splits[rank]] * 3, splits, group)
            )

        for kind, model in enumerate(
            (
                dynamic_reduce,
                dynamic_gather,
                dynamic_scatter,
                dynamic_exchange,
                dynamic_uneven_exchange,
            )
        ):
            backend = Compiler()
            compiled = torch.compile(
                model, backend=backend, fullgraph=True, dynamic=True
            )
            for length in (9, 12, 15) if kind == 4 else (6, 9, 12):
                value = (
                    torch.arange(length * 2, dtype=torch.float32).reshape(length, 2)
                    + rank
                ).requires_grad_()
                base = value.detach() - rank
                total = 3 * base + 3
                width = length // 3
                if kind == 0:
                    expected, gradient = total, 6 * total
                elif kind == 1:
                    expected, gradient = (
                        torch.cat([base + peer for peer in range(3)]),
                        6 * value.detach(),
                    )
                elif kind == 2:
                    expected, gradient = (
                        total[rank * width : (rank + 1) * width],
                        2 * total,
                    )
                else:
                    if kind == 4:
                        sizes = [width - 1, width, width + 1]
                        start, end = sum(sizes[:rank]), sum(sizes[: rank + 1])
                    else:
                        start, end = rank * width, (rank + 1) * width
                    expected = torch.cat(
                        [(base + peer)[start:end] for peer in range(3)]
                    )
                    gradient = 2 * value.detach()
                result = compiled(value)
                torch.testing.assert_close(result, expected)
                result.square().sum().backward()
                torch.testing.assert_close(value.grad, gradient)
            assert len(backend.graphs) == 2, (kind, len(backend.graphs))
            assert all(region.executions == 3 for region in backend.regions)

        # Group-relative source differs from global source; nonmembers skip work.
        subgroup = dist.new_group([0, 2], backend="gloo")
        if rank in (0, 2):
            subgroup_name = subgroup.group_name

            def broadcast(value):
                return ops.wait_tensor(ops.broadcast(value, 1, subgroup_name))

            torch.testing.assert_close(broadcast(x), x - rank + 2)
            _check(broadcast, x, x - rank + 2)
        dist.barrier()
        single = dist.new_group([0], backend="gloo")
        if rank == 0:
            single_name = single.group_name

            def singleton(value):
                return ops.wait_tensor(ops.all_reduce(value, "sum", single_name))

            _check(singleton, x, x)
    finally:
        dist.destroy_process_group()


def test_three_rank_collectives(tmp_path):
    context = mp.spawn(
        _worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3, join=False
    )
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("three-rank collective test exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
