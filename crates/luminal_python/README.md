# luminal_python

PyTorch `torch.compile` integration for Luminal.

## CUDA Tests

The Python CUDA CI job builds the Rust extension with the CUDA feature and runs
the non-slow pytest suite:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s -m "not slow"
```

The slow tests are explicit opt-in. They include large/pretrained model tests,
full-width architecture compiles, Whisper end-to-end cases, and other cases that
can take a long time or need a large GPU / Hugging Face cache.

Run the full Python CUDA suite, including slow tests:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s
```

Run only the slow Python CUDA tests:

```bash
cd crates/luminal_python
RUST_BACKTRACE=1 \
LUMINAL_TEST_DEVICE=cuda \
MATURIN_PEP517_ARGS="--features cuda --profile release" \
CUDARC_CUDA_VERSION=12080 \
uv run --group dev python -m pytest tests/ -v -s -m slow
```

The helper script follows the same convention:

```bash
cd crates/luminal_python
./run_tests_cuda.sh              # non-slow CUDA suite
./run_tests_cuda.sh --slow-only  # only slow CUDA tests
./run_tests_cuda.sh --include-slow
```

The GitHub/Modal entrypoint uses the same marker split:

```bash
cd crates/luminal_python
modal run modal_pytest_runner.py --gpu A100 --timeout 7200 tests/ -v -s -m "not slow"
modal run modal_pytest_runner.py --gpu A100 --timeout 7200 tests/ -v -s
```
