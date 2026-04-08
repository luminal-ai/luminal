import modal
import subprocess
import os

example = os.environ.get("EXAMPLE", "llama")
gpu_type = os.environ.get("GPU_TYPE", "A100-80GB")
CUDARC_CUDA_VERSION = "12080"
HF_CACHE_VOLUME_NAME = "luminal-hf-cache-v2"
HF_CACHE_PATH = "/root/.cache/huggingface"

app = modal.App(f"luminal-ci-{example}")

hf_cache = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.03-py3"
    )
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
    timeout=3600,  # 60 minutes
    volumes={
        HF_CACHE_PATH: hf_cache,
    },
)
def run_example(example: str):
    """Build and run a luminal example on a Modal GPU."""
    subprocess.run(["nvidia-smi"], check=True)

    subprocess.run(
        ["cargo", "run", "--release"],
        cwd=f"{WORKDIR}/examples/{example}",
        env={
            **os.environ,
            "CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION,
            "HF_HOME": HF_CACHE_PATH,
        },
        check=True,
    )

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_example.remote(example)
