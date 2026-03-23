A couple of short things to keep in mind

## Lessons Learned

At the end of any session that involved a hard or non-obvious bug, append an entry to
`LessonsLearned.md` in this directory. A "hard bug" means any bug that required significant
investigation — intermittent failures, wrong output without a crash, egglog/optimizer issues,
or anything that took more than a few minutes to locate.

Each entry should cover:
1. **What the symptom was** (test failure, wrong output, panic, etc.)
2. **What the actual root cause was** (the specific code/logic that was wrong)
3. **Why it was hard to find** (what made it non-obvious or intermittent)
4. **The fix** (what changed and why it works)
5. **A general principle** extracted from the bug — something that helps avoid the same
   class of mistake in future code

The goal is to build a living record of codebase-specific pitfalls that future sessions can
consult before writing new egglog rules, CUDA kernels, or optimizer passes.
1. If you want to run tests:
   - `./run_test.sh` - runs tests with the native backend
   - `./run_tests_cuda.sh` - runs tests with the CUDA backend

## Testing Best Practices

### Overview
The luminal_python crate provides a bridge between PyTorch models and the luminal library via ONNX. Tests should verify this integration end-to-end by testing the actual user workflow: PyTorch model → torch.compile → luminal backend.

### Test Pattern (CORRECT)

All tests should follow this standard pattern:

```python
def test_operation():
    """Brief description of what operation is being tested."""
    # 1. Instantiate PyTorch model
    model: torch.nn.Module = OperationTestModel()

    # 2. Compile with luminal backend
    model_compiled: Callable = torch.compile(model, backend=luminal_backend)

    # 3. Create test input
    x: torch.Tensor = torch.tensor([...])  # or torch.rand(...)

    # 4. Run both original and compiled versions
    original: torch.Tensor = model(x)
    output: torch.Tensor = model_compiled(x)

    # 5. Verify outputs match
    assert torch.allclose(output, original)
```

### Test Models

- Define test model classes in `tests/test_models.py`
- Each model should be a simple `torch.nn.Module` that demonstrates one operation or pattern
- Use clear, descriptive class names (e.g., `AddTestModel`, `TransposeTestModel`)
- Include docstrings explaining what the model tests

Example:
```python
class AddTestModel(torch.nn.Module):
    """Tests element-wise addition."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x
```

### What NOT to Do

**❌ DO NOT create ONNX files directly in tests:**
```python
# WRONG - bypasses the PyTorch integration
model_path = create_onnx_model(...)
graph_result = luminal.process_onnx(model_path, backend='native')
```

**✓ DO create PyTorch models and use torch.compile:**
```python
# CORRECT - tests actual user workflow
model: torch.nn.Module = MyTestModel()
model_compiled = torch.compile(model, backend=luminal_backend)
```

### Rationale

- **End-to-end testing**: Tests verify the complete PyTorch → ONNX → luminal pipeline
- **User-facing API**: Tests use the same API that users will use (torch.compile)
- **Correctness**: Comparing compiled vs original PyTorch output ensures correctness
- **Maintainability**: Consistent pattern across all tests makes the codebase easier to understand
- **Simplicity**: No manual ONNX file creation, no tempfile cleanup, no numpy comparisons

### Special Cases

**Testing constants:**
Use inline tensor literals in the forward method - PyTorch exports these as ONNX Constant nodes:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    constant = torch.tensor([1.0, 2.0, 3.0])
    return x + constant
```

**Testing type casts:**
Use `.to(dtype)` method - PyTorch exports these as ONNX Cast nodes:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)
```

**Testing complex operations:**
Chain operations naturally in PyTorch - ONNX export handles the conversion:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    transposed = x.transpose(0, 1)
    scaled = transposed * 2.0
    return scaled + 1.0
```
