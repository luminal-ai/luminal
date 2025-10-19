## CUDA Training Setup

Luminal supports CUDA for accelerated ML training. To enable and debug CUDA workflows:
1. Ensure you have the latest NVIDIA drivers and the CUDA toolkit installed (>= 11.x).
2. Set the environment variable `CUDA_HOME` to your CUDA installation path.
3. When using GitHub Actions, use a self-hosted runner with GPU. Example snippet:
    ```
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Set up CUDA
        run: sudo apt-get install -y cuda-toolkit-11-2
      - name: Test CUDA build
        run: make test-cuda
    ```
4. For troubleshooting, run:
    ```
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"
    ```
5. If you encounter build errors, see `docs/troubleshooting.md`.

## Google Colab (L4) / cudarc compatibility

Issue summary
- Some Colab VMs with the NVIDIA L4 expose CUDA 12.2 (nvcc V12.2.x). Users reported a runtime panic referring to an undefined symbol such as:

```
Expected symbol in library: DlSym { desc: "/usr/lib64-nvidia/libcuda.so: undefined symbol: cuCtxCreate_v4" }
```

What causes this
- The Rust crate `cudarc` exposes bindings to the native CUDA driver API. `cudarc` releases are compiled against specific CUDA versions/ABI features and expose feature flags like `cuda-12080` to select the matching CUDA symbols and behavior. If the system CUDA/runtime (or driver) is newer and exposes symbols that the `cudarc` build does not expect (or vice-versa), you can get unresolved symbol errors such as the one above.

Quick workarounds / fixes

1) Try using the matching `cudarc` feature for CUDA 12.2
- Edit the `cudarc` dependency feature flags in the `Cargo.toml` files that enable CUDA (for example `crates/luminal_cuda/Cargo.toml`, `crates/luminal_2/Cargo.toml`, and `demos/matmul/Cargo.toml`) and replace the `cuda-12080` feature with the `cuda-12200` feature (or the feature that matches your system CUDA minor version). Example snippet:

```toml
cudarc = { version = "0.16.6", features = [
    "f16",
    "cuda-12200",
] }
```

- After editing, run:

```powershell
cargo update -p cudarc
cargo build --release --features cuda
```

If `cargo update -p cudarc` fails to resolve an appropriate `cudarc` build with `cuda-12200`, you may need to bump the `cudarc` crate version (see next step).

2) Bump `cudarc` to a newer release that supports CUDA 12.2
- If a newer `cudarc` release adds first-class support for CUDA 12.2 (features and compatible driver ABI), update the `version = "..."` to that release in the same `Cargo.toml` files and run `cargo update` / `cargo build`.

3) As a temporary workaround, build `cudarc` from source with the matching feature
- You can point the dependency to a Git revision/branch that contains the necessary CUDA 12.2 support and enable the `cuda-12200` feature there. Example (replace repo/branch with the upstream repository and branch that contains the fix):

```toml
cudarc = { git = "https://github.com/rust-cuda/cudarc.git", branch = "main", features = ["f16", "cuda-12200"] }
```

Colab-specific setup notes
- On Colab you may need to install or switch to a CUDA toolkit that matches the driver exposed by the runtime. A minimal sequence (run in a notebook cell) is:

```bash
# Install CUDA 12.2 toolkit (or the version matching the runtime) -- example only; exact package names can vary
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-2
export CUDA_HOME=/usr/local/cuda-12.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

- Verify the driver and toolkit in Colab before building:

```bash
nvidia-smi
nvcc --version
ldconfig -p | grep libcuda
```

What to report back when opening an issue (helps maintainers)
- Exact Colab environment output: `nvidia-smi`, `nvcc --version`, and `ldconfig -p | grep libcuda`.
- The `cudarc` version and feature flags from your `Cargo.toml` (copy the `[dependencies]` section where `cudarc` is declared).
- The exact cargo build/run command and the full error (stack trace) you observed.

Long-term
- If upgrading `cudarc` or enabling the correct `cuda-12200` feature fixes the problem for Colab/L4, we should update the repository `Cargo.toml` files to use the newer `cudarc` version and feature or add conditional features that cover CUDA 12.2. If this repo maintainers prefer, we can make a PR that:
  - bumps `cudarc` where needed,
  - adds `cuda-12200` feature selections, and
  - updates `docs/cuda_setup.md` with the Colab/L4 notes above.

If you want, I can prepare a PR that:
- switches the `cudarc` features to include `cuda-12200` (or a version bump),
- runs `cargo check` for the affected crates locally (where possible),
- and includes this documentation update.

