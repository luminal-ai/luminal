import modal
import subprocess
import os

gpu_type = os.environ.get("GPU_TYPE", "T4")
CUDARC_CUDA_VERSION = "12080"

app = modal.App("luminal-ci-cargo-test")

WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.03-py3")
    .apt_install("protobuf-compiler")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env(
        {
            "PATH": "/root/.cargo/bin:$PATH",
            "CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION,
        }
    )
    .add_local_dir(".", remote_path=WORKDIR, copy=True)
)


@app.function(
    image=cuda_image,
    gpu=gpu_type,
    timeout=1800,  # 30 minutes
)
def run_cargo_test():
    """Run cargo test for luminal_cuda_lite on a Modal GPU."""
    subprocess.run(["nvidia-smi"], check=True)

    # Detect GPU compute capability
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    compute_cap = result.stdout.strip().replace(".", "")

    subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "luminal_cuda_lite",
            "--verbose",
            "--",
            "--test-threads=1",
        ],
        cwd=WORKDIR,
        env={
            **os.environ,
            "CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION,
            "CUDA_COMPUTE_CAP": compute_cap,
        },
        check=True,
    )


@app.local_entrypoint()
def main():
    run_cargo_test.remote()
