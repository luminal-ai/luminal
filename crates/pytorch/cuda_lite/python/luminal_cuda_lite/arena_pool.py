"""Process-local, grow-only CUDA scratch shared by compiled regions."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


@dataclass
class _Arena:
    tensor: torch.Tensor
    growths: int = 1


class _HighWaterArenaPool:
    """One persistent arena per CUDA device and caller execution stream.

    Calls issued on one caller stream are one execution lane: the Luminal
    side-stream handshake orders every later call after the previous one, so
    all compiled regions on that lane can safely reuse the same bytes.
    """

    def __init__(self) -> None:
        self._arenas: dict[tuple[int, int], _Arena] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(device: torch.device, caller_stream: torch.cuda.Stream) -> tuple[int, int]:
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return index, caller_stream.cuda_stream

    def acquire(
        self,
        required_bytes: int,
        device: torch.device,
        caller_stream: torch.cuda.Stream,
        use_stream: torch.cuda.Stream,
        *,
        device_wide: bool = False,
    ) -> torch.Tensor:
        """Return this lane's arena, growing it only when required."""
        required_bytes = max(required_bytes, 1)
        key = self._key(device, caller_stream)
        if device_wide:
            # An enclosing CUDA graph captures on a temporary capture stream,
            # not the eager warmup stream. It must nevertheless see the exact
            # arena address prepared during warmup.
            key = (key[0], 0)
        with self._lock:
            arena = self._arenas.get(key)
            if arena is None or arena.tensor.numel() < required_bytes:
                growths = 1 if arena is None else arena.growths + 1
                with torch.cuda.stream(use_stream):
                    tensor = torch.empty(
                        required_bytes, dtype=torch.uint8, device=device
                    )
                arena = _Arena(tensor=tensor, growths=growths)
                self._arenas[key] = arena
            # Luminal uses the raw pointer, outside PyTorch's dispatcher. Tell
            # the allocator which stream must finish before a replaced arena's
            # storage may be recycled.
            arena.tensor.record_stream(use_stream)
            return arena.tensor

    def clear(self) -> None:
        """Release retained tensors; recorded streams preserve safe reuse."""
        with self._lock:
            self._arenas.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            capacities = [arena.tensor.numel() for arena in self._arenas.values()]
            return {
                "lanes": len(capacities),
                "capacity_bytes": sum(capacities),
                "max_capacity_bytes": max(capacities, default=0),
                "growths": sum(arena.growths for arena in self._arenas.values()),
            }


_POOL = _HighWaterArenaPool()


def acquire_arena(
    required_bytes: int,
    device: torch.device,
    caller_stream: torch.cuda.Stream,
    use_stream: torch.cuda.Stream,
    *,
    device_wide: bool = False,
) -> torch.Tensor:
    return _POOL.acquire(
        required_bytes,
        device,
        caller_stream,
        use_stream,
        device_wide=device_wide,
    )


def clear_arena_pool() -> None:
    """Release every process-local high-water arena."""
    _POOL.clear()


def arena_pool_stats() -> dict[str, int]:
    """Return aggregate capacities and growth counts for diagnostics."""
    return _POOL.stats()
