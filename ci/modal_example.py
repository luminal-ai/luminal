import os
import subprocess
import sys

import modal

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

EXAMPLE_CARGO_ARGS = {
    "qwen": ["--features", "cuda"],
}


def run_and_capture(command: list[str], *, cwd: str, env: dict[str, str]) -> str:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None

    chunks = []
    while True:
        chunk = process.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        chunks.append(chunk)

    return_code = process.wait()
    output = b"".join(chunks).decode("utf-8", errors="replace")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


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
    timeout=7200,  # 2 hours
    volumes={
        HF_CACHE_PATH: hf_cache,
    },
)
def run_example(example: str):
    """Build and run a luminal example on a Modal GPU."""
    subprocess.run(["nvidia-smi"], check=True)
    sys.path.insert(0, f"{WORKDIR}/ci")
    from example_output import validate_output

    run_env = {
        **os.environ,
        "CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION,
        "HF_HOME": HF_CACHE_PATH,
        # Reduce buffering so more child output reaches the pipe before a crash.
        "PYTHONUNBUFFERED": "1",
        "RUST_BACKTRACE": "1",
    }
    try:
        output = run_and_capture(
            ["cargo", "run", "--release", *EXAMPLE_CARGO_ARGS.get(example, [])],
            cwd=f"{WORKDIR}/examples/{example}",
            env=run_env,
        )
    except subprocess.CalledProcessError as e:
        # Surface the captured output in the exception message, which CI shows
        # reliably (unlike the streamed remote stdout), so crashes aren't opaque.
        captured = e.output or "(no output captured before crash)"
        print(
            f"\n===== Captured output from '{example}' before failure "
            f"(rc={e.returncode}) =====\n{captured}\n===== end captured output =====",
            flush=True,
        )
        tail = captured[-4000:]
        raise RuntimeError(
            f"Example '{example}' failed (exit/signal {e.returncode}). "
            f"Last {len(tail)} chars of output:\n{tail}"
        ) from e
    validate_output(example, output)

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_example.remote(example)
