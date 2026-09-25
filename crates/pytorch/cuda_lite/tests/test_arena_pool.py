from contextlib import nullcontext
from types import SimpleNamespace

from luminal_cuda_lite.arena_pool import _HighWaterArenaPool


class _Tensor:
    def __init__(self, size):
        self.size = size
        self.streams = []

    def numel(self):
        return self.size

    def record_stream(self, stream):
        self.streams.append(stream)


def _fake_allocator(monkeypatch):
    allocations = []

    def empty(size, **_kwargs):
        tensor = _Tensor(size)
        allocations.append(tensor)
        return tensor

    monkeypatch.setattr("luminal_cuda_lite.arena_pool.torch.empty", empty)
    monkeypatch.setattr(
        "luminal_cuda_lite.arena_pool.torch.cuda.stream", lambda _stream: nullcontext()
    )
    return allocations


def test_arena_reuses_then_grows_to_the_high_water_mark(monkeypatch):
    allocations = _fake_allocator(monkeypatch)
    pool = _HighWaterArenaPool()
    device = SimpleNamespace(index=0)
    caller = SimpleNamespace(cuda_stream=11)
    side = SimpleNamespace(cuda_stream=12)

    first = pool.acquire(64, device, caller, side)
    reused = pool.acquire(32, device, caller, side)
    grown = pool.acquire(128, device, caller, side)

    assert reused is first
    assert grown is not first
    assert [tensor.size for tensor in allocations] == [64, 128]
    assert pool.stats() == {
        "lanes": 1,
        "capacity_bytes": 128,
        "max_capacity_bytes": 128,
        "growths": 2,
    }
    assert first.streams == [side, side]
    assert grown.streams == [side]


def test_caller_streams_are_independent_execution_lanes(monkeypatch):
    _fake_allocator(monkeypatch)
    pool = _HighWaterArenaPool()
    device = SimpleNamespace(index=0)
    side = SimpleNamespace(cuda_stream=12)

    left = pool.acquire(64, device, SimpleNamespace(cuda_stream=1), side)
    right = pool.acquire(32, device, SimpleNamespace(cuda_stream=2), side)

    assert left is not right
    assert pool.stats()["lanes"] == 2
    pool.clear()
    assert pool.stats()["capacity_bytes"] == 0


def test_external_capture_reuses_warmup_arena_across_capture_stream(monkeypatch):
    _fake_allocator(monkeypatch)
    pool = _HighWaterArenaPool()
    device = SimpleNamespace(index=0)
    warmup = SimpleNamespace(cuda_stream=1)
    capture = SimpleNamespace(cuda_stream=2)

    first = pool.acquire(64, device, warmup, warmup, device_wide=True)
    captured = pool.acquire(64, device, capture, capture, device_wide=True)

    assert captured is first
    assert pool.stats()["lanes"] == 1
