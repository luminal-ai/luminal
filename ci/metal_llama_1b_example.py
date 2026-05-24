import os
import subprocess
import sys


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
    sys.path.insert(0, os.path.join(repo_root, "ci"))
    from example_output import validate_output

    output = run_and_capture(
        ["cargo", "run", "--release", "-p", "luminal_metal", "--example", "llama_1b"],
        cwd=repo_root,
        env=os.environ.copy(),
    )
    if "TTFT:" not in output or "TPOT:" not in output:
        raise AssertionError("Llama 1B Metal example did not complete generation")
    validate_output("llama", output)


if __name__ == "__main__":
    main()
