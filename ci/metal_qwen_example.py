import os
import subprocess
import sys

from example_output import validate_output

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


def main():
    repo_root = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    output = run_and_capture(
        ["cargo", "run", "--release", "-p", "qwen", "--features", "metal"],
        cwd=repo_root,
        env=os.environ.copy(),
    )
    if "TTFT:" not in output or "TPOT:" not in output:
        raise AssertionError("qwen Metal example did not complete generation")
    validate_output("qwen", output)


if __name__ == "__main__":
    main()
