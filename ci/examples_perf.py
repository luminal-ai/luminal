import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from example_output import validate_output


DEFAULT_EXAMPLES = ["llama", "gemma", "qwen", "qwen3_moe", "gemma4_moe", "whisper"]

EXAMPLE_CARGO_ARGS = {
    "llama": ["run", "--release", "-p", "llama"],
    "gemma": ["run", "--release", "-p", "gemma"],
    "qwen": ["run", "--release", "-p", "qwen", "--features", "cuda"],
    "qwen3_moe": ["run", "--release", "-p", "qwen3_moe"],
    "gemma4_moe": ["run", "--release", "-p", "gemma4_moe"],
    "whisper": ["run", "--release", "-p", "whisper"],
}


@dataclass
class Metrics:
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    tps: float | None = None


@dataclass
class ExampleResult:
    name: str
    ok: bool
    metrics: Metrics = field(default_factory=Metrics)
    wall_s: float = 0.0
    error: str | None = None


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg != "--"]
    if any(arg in {"-h", "--help"} for arg in args):
        print_help()
        return
    if "--list" in args:
        print("\n".join(DEFAULT_EXAMPLES))
        return

    examples = args or DEFAULT_EXAMPLES
    results = [run_example(example) for example in examples]
    print_table(results)
    if any(not result.ok for result in results):
        raise SystemExit(1)


def print_help() -> None:
    print(
        "Run validated Luminal examples, validate textual output, and summarize perf.\n"
        "\n"
        "Usage:\n"
        "  cargo examples\n"
        "  cargo examples llama qwen whisper\n"
        "\n"
        "Options:\n"
        "  --list    Print the default validated examples\n"
        "  -h, --help\n"
        "\n"
        f"The default set matches the Modal examples CI: {', '.join(DEFAULT_EXAMPLES)}."
    )


def run_example(example: str) -> ExampleResult:
    cargo_args = EXAMPLE_CARGO_ARGS.get(example)
    if cargo_args is None:
        known = ", ".join(DEFAULT_EXAMPLES)
        return ExampleResult(example, False, error=f"unknown example; known examples: {known}")

    print(f"\n=== Running {example} ===")
    print(f"$ cargo {' '.join(cargo_args)}")
    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault("CUDARC_CUDA_VERSION", "12080")
    process = subprocess.Popen(
        ["cargo", *cargo_args],
        cwd=repo_root(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None

    chunks: list[bytes] = []
    while True:
        chunk = process.stdout.read1(4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        chunks.append(chunk)

    return_code = process.wait()
    output = b"".join(chunks).decode("utf-8", errors="replace")
    wall_s = time.monotonic() - started
    metrics = parse_metrics(output)

    if return_code:
        return ExampleResult(
            example,
            False,
            metrics=metrics,
            wall_s=wall_s,
            error=f"process exited with code {return_code}",
        )

    try:
        validate_output(example, output)
    except Exception as exc:
        return ExampleResult(example, False, metrics=metrics, wall_s=wall_s, error=str(exc))

    return ExampleResult(example, True, metrics=metrics, wall_s=wall_s)


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_metrics(output: str) -> Metrics:
    metrics = Metrics()
    for line in output.splitlines():
        if "TTFT:" in line:
            metrics.ttft_ms = parse_number_after(line, "TTFT:")
        if "TPOT:" in line:
            metrics.tpot_ms = parse_number_after(line, "TPOT:")
        if "tok/s" in line:
            metrics.tps = parse_tok_per_second(line)
    if metrics.tps is None and metrics.tpot_ms:
        metrics.tps = 1000.0 / metrics.tpot_ms
    return metrics


def parse_number_after(line: str, marker: str) -> float | None:
    tail = line.split(marker, 1)[1].lstrip()
    chars = []
    for char in tail:
        if char.isdigit() or char == ".":
            chars.append(char)
        else:
            break
    if not chars:
        return None
    return float("".join(chars))


def parse_tok_per_second(line: str) -> float | None:
    head = line.split("tok/s", 1)[0].rstrip(" (")
    parts = head.split()
    if not parts:
        return None
    try:
        return float(parts[-1])
    except ValueError:
        return None


def print_table(results: list[ExampleResult]) -> None:
    print("\nSummary")
    print(f"{'example':<14} {'status':<8} {'TTFT ms':>10} {'TPOT ms':>10} {'tok/s':>10} {'wall s':>10}")
    print("-" * 68)
    for result in results:
        status = "ok" if result.ok else "failed"
        print(
            f"{result.name:<14} {status:<8} "
            f"{format_metric(result.metrics.ttft_ms):>10} "
            f"{format_metric(result.metrics.tpot_ms):>10} "
            f"{format_metric(result.metrics.tps):>10} "
            f"{result.wall_s:>10.1f}"
        )
        if result.error:
            print(f"  error: {result.error}")


def format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
