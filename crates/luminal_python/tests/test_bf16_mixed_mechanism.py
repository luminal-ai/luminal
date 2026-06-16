"""Pin the cublaslt_mixed_dtype mechanism: does the unsound candidate need a
beta/C-input (residual fused into the matmul), or does matmul+cast alone break?

Each module ends in .float() to trigger the mixed_dtype (bf16 matmul -> F32 out)
rewrite. M_resid additionally adds a residual before the cast (fusable as the
cuBLASLt C/beta input). Compiles each many times; reports broken rate.

    LUMINAL_TEST_DEVICE=cuda uv run --group dev pytest tests/_bf16_mixed_mechanism.py -v -s
"""

import pytest
import torch
import torch._dynamo
from luminal import luminal_backend

H = 4096


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatmulCast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(H, H, bias=False)

    def forward(self, x):
        return self.l(x).float()  # bf16 matmul -> cast f32  (beta=0)


class MatmulResidCast(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(H, H, bias=False)

    def forward(self, x):
        return (self.l(x) + x).float()  # matmul + residual(C) -> cast f32 (beta!=0)


def _rate(name, ctor, device, n=12):
    torch.manual_seed(0)
    mod = ctor().eval().to(device).to(torch.bfloat16)
    x = torch.randn(1, 4, H, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = mod(x).detach().clone()
    errs = []
    for i in range(n):
        torch._dynamo.reset()
        c = torch.compile(mod, backend=luminal_backend)
        with torch.no_grad():
            out = c(x).detach().clone()
        e = (out.float() - ref.float()).abs().max().item()
        errs.append(e)
    nb = sum(1 for e in errs if e > 1 or e != e)
    finite = [e for e in errs if e == e]
    print(f"{name:18} broken {nb}/{n}  min={min(finite):.2e} max={max(finite):.2e}")
    assert nb == 0, (
        f"{name}: {nb}/{n} compiles broke — the bf16 matmul C-input (beta) is being "
        f"misread; cublaslt_mixed_dtype must not fire on beta!=0 matmuls"
    )


@pytest.mark.slow
def test_mechanism():
    device = _device()
    if device.type != "cuda":
        pytest.skip("requires the CUDA backend")
    # matmul+cast (beta=0) was always fine; matmul+residual+cast (beta!=0) was
    # the unsound case (~90% broken before the fix). Both must be 0/N now.
    _rate("matmul+cast", MatmulCast, device)
    _rate("matmul+resid+cast", MatmulResidCast, device)
