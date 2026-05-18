import os
import re
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

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

EXPECTED_OUTPUT = {
    "qwen": [
        "computational model inspired by the structure and function of the human brain",
    ],
    "gemma4_moe": [
        "city of romance, art and culture",
    ],
    "whisper": [
        "ask not what your country can do for you",
    ],
}

EXPECTED_CONCEPTS = {
    "llama": [
        ["layers"],
        ["neurons", "nodes"],
        ["learn", "learning", "adapt"],
        ["data", "patterns", "features"],
    ],
    "gemma": [
        ["neural network", "neural networks"],
        ["nodes", "neurons"],
        ["layers"],
        ["weights"],
        ["training", "learn", "learns"],
    ],
    "qwen3_moe": [
        ["capital"],
        ["france"],
        ["paris"],
    ],
}

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


def normalize_output(output: str) -> str:
    output = ANSI_ESCAPE.sub("", output)
    output = output.replace("\r", "\n")
    return re.sub(r"\s+", " ", output).casefold()


def validate_output(example: str, output: str):
    normalized_output = normalize_output(output)

    expected_concepts = EXPECTED_CONCEPTS.get(example)
    if expected_concepts is not None:
        missing = [
            concept_group
            for concept_group in expected_concepts
            if not any(normalize_output(term) in normalized_output for term in concept_group)
        ]
        if missing:
            expected = "\n  - ".join(" / ".join(group) for group in expected_concepts)
            missing_terms = "\n  - ".join(" / ".join(group) for group in missing)
            raise AssertionError(
                f"Output check failed for {example!r}.\n"
                f"Expected concept groups:\n  - {expected}\n"
                f"Missing concept groups:\n  - {missing_terms}"
            )

        expected = ", ".join(" / ".join(group) for group in expected_concepts)
        print(f"\nOutput check passed for {example!r}: found concepts {expected}")
        return

    expected_phrases = EXPECTED_OUTPUT.get(example)
    if expected_phrases is None:
        raise ValueError(f"No expected output phrases configured for example {example!r}")

    for phrase in expected_phrases:
        if normalize_output(phrase) in normalized_output:
            print(f"\nOutput check passed for {example!r}: found {phrase!r}")
            return

    expected = "\n  - ".join(expected_phrases)
    raise AssertionError(
        f"Output check failed for {example!r}. Expected one of:\n  - {expected}"
    )

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

    run_env = {
        **os.environ,
        "CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION,
        "HF_HOME": HF_CACHE_PATH,
    }
    output = run_and_capture(
        ["cargo", "run", "--release", *EXAMPLE_CARGO_ARGS.get(example, [])],
        cwd=f"{WORKDIR}/examples/{example}",
        env=run_env,
    )
    validate_output(example, output)

    hf_cache.commit()


@app.local_entrypoint()
def main():
    run_example.remote(example)
