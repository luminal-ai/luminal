import modal
import subprocess
import os

example = os.environ.get("EXAMPLE", "llama")
gpu_type = os.environ.get("GPU_TYPE", "A100-80GB")

app = modal.App(f"luminal-ci-{example}")

hf_cache = modal.Volume.from_name("luminal-hf-cache", create_if_missing=True)

WORKDIR = "/workspace/luminal"

cuda_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.03-py3"
    )
    .apt_install("protobuf-compiler")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH"})
    .add_local_dir(".", remote_path=WORKDIR)
)


@app.function(
    image=cuda_image,
    gpu=gpu_type,
    timeout=3600,  # 60 minutes
    volumes={
        "/root/.cache/huggingface": hf_cache,
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
            "HF_HOME": "/root/.cache/huggingface",
        },
        check=True,
    )

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_example.remote(example)
