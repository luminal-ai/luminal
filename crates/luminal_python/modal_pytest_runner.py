"""Run pytest on Modal with a dynamically selected GPU.

Usage:
    uv run modal run modal_pytest_runner.py --gpu A100 tests/test_llama3.py::test_hf_llama3_full -v
    uv run modal run modal_pytest_runner.py --gpu T4 tests/
    uv run modal run modal_pytest_runner.py --gpu A100 --profile tests/ -v
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from modal.volume import FileEntryType

app = modal.App("luminal-tests")

DEFAULT_TIMEOUT = 30 * 60
CUDARC_CUDA_VERSION = "12080"
LOCAL_PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = "/root/luminal/crates/luminal_python"
VENV_PATH = "/root/.cache/luminal/uv-project-environments/luminal_python"
SRC_PATH = f"{PROJECT_DIR}/src"
PROFILE_VOLUME_NAME = "luminal-pytest-profiling"
PROFILE_VOLUME_PATH = "/root/pytest-profile-artifacts"
PROFILE_LOCAL_DEFAULT_ROOT = "luminal_artifacts/pytest-profiling"
PROFILE_SCRATCH_ROOT = "/tmp/luminal-pytest-profiling"
HF_CACHE_VOLUME_NAME = "luminal-hf-cache-v2"
HF_CACHE_PATH = "/root/.cache/huggingface"
HF_TOKEN_ENV_KEY = "HF_TOKEN"
PROFILE_VOLUME = modal.Volume.from_name(PROFILE_VOLUME_NAME, create_if_missing=True)
HF_CACHE_VOLUME = modal.Volume.from_name(
    HF_CACHE_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)

image = (
    modal.Image.from_registry("ghcr.io/luminal-ai/luminal-docker:cuda")
    .env({"CUDARC_CUDA_VERSION": CUDARC_CUDA_VERSION})
    .uv_sync(
        str(LOCAL_PROJECT_DIR),
        frozen=False,
        groups=["dev"],
        env={"UV_PROJECT_ENVIRONMENT": VENV_PATH},
    )
    .workdir(PROJECT_DIR)
    .add_local_dir(
        str(LOCAL_PROJECT_DIR.parent.parent),
        remote_path="/root/luminal",
        copy=True,
        ignore=[
            ".git",
            ".claude-project",
            ".cargo-local",
            "**/.venv",
            "**/.pytest_cache",
            "**/__pycache__",
            "**/luminal_artifacts",
            "**/target",
            "docs",
        ],
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hf_token_secret() -> modal.Secret | None:
    hf_token = os.environ.get(HF_TOKEN_ENV_KEY)
    if not hf_token:
        return None
    return modal.Secret.from_dict({HF_TOKEN_ENV_KEY: hf_token})


def _has_pytest_flag(pytest_args: list[str], flag: str) -> bool:
    return any(arg == flag for arg in pytest_args)


def _profiling_enabled(cli_profile: bool, pytest_args: list[str]) -> bool:
    return (
        cli_profile
        or _has_pytest_flag(pytest_args, "--profile")
        or _has_pytest_flag(pytest_args, "--profile-svg")
    )


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _prepare_scratch_dir(scratch_dir: Path) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    linked_names = {
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "luminal_artifacts",
        "prof",
    }
    for entry in Path(PROJECT_DIR).iterdir():
        if entry.name in linked_names:
            continue

        target = scratch_dir / entry.name
        if target.exists() or target.is_symlink():
            continue

        target.symlink_to(entry, target_is_directory=entry.is_dir())


def _default_profile_output_dir(run_id: str) -> Path:
    return (LOCAL_PROJECT_DIR / PROFILE_LOCAL_DEFAULT_ROOT / run_id).resolve()


def _prepare_local_profile_dir(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    prof_dir = output_dir / "prof"
    if prof_dir.exists():
        shutil.rmtree(prof_dir)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _download_profile_artifacts(run_id: str, output_dir: Path) -> None:
    entries = PROFILE_VOLUME.listdir(run_id, recursive=True)
    _prepare_local_profile_dir(output_dir)

    for entry in entries:
        relative_path = Path(entry.path).relative_to(run_id)
        if relative_path == Path("."):
            continue

        destination = output_dir / relative_path
        if entry.type == FileEntryType.DIRECTORY:
            destination.mkdir(parents=True, exist_ok=True)
            continue

        if entry.type != FileEntryType.FILE:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in PROFILE_VOLUME.read_file(entry.path):
                handle.write(chunk)


def _cleanup_remote_profile_artifacts(run_id: str) -> None:
    try:
        PROFILE_VOLUME.remove_file(run_id, recursive=True)
    except FileNotFoundError:
        return


@app.cls(image=image, timeout=DEFAULT_TIMEOUT)
class TestRunner:
    @modal.method()
    def run(
        self,
        pytest_args: list[str],
        pytest_addopts: str = "",
        profile_enabled: bool = False,
    ) -> dict[str, Any]:
        started_at = _utc_now()
        run_id = _run_id() if profile_enabled else None
        scratch_dir = Path(PROFILE_SCRATCH_ROOT) / run_id if run_id else None
        if scratch_dir is not None:
            _prepare_scratch_dir(scratch_dir)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{SRC_PATH}:{existing}" if existing else SRC_PATH
        env["LUMINAL_TEST_DEVICE"] = "cuda"
        env["UV_PROJECT_ENVIRONMENT"] = VENV_PATH
        env["MATURIN_PEP517_ARGS"] = "--features cuda --profile release"
        env["CUDARC_CUDA_VERSION"] = CUDARC_CUDA_VERSION
        env["HF_HOME"] = HF_CACHE_PATH
        if pytest_addopts:
            env["PYTEST_ADDOPTS"] = pytest_addopts

        original_svg_requested = _has_pytest_flag(pytest_args, "--profile-svg")
        dot_available = shutil.which("dot") is not None
        sanitized_pytest_args = [
            arg for arg in pytest_args if arg not in {"--profile", "--profile-svg"}
        ]
        if profile_enabled:
            sanitized_pytest_args.append("--profile")
            if dot_available:
                sanitized_pytest_args.append("--profile-svg")
            elif original_svg_requested:
                print(
                    "Graphviz 'dot' is unavailable in the Modal container; "
                    "falling back to raw .prof artifacts only.",
                    file=sys.stderr,
                )

        svg_requested = profile_enabled and dot_available
        cmd = [
            "uv",
            "run",
            "--project",
            PROJECT_DIR,
            "--group",
            "dev",
            "--reinstall-package",
            "luminal_python",
            "python",
            "-m",
            "pytest",
            *sanitized_pytest_args,
        ]
        exit_code = subprocess.run(
            cmd,
            env=env,
            cwd=str(scratch_dir) if scratch_dir is not None else PROJECT_DIR,
        ).returncode
        HF_CACHE_VOLUME.commit()
        finished_at = _utc_now()

        if not profile_enabled:
            return {
                "exit_code": exit_code,
                "run_id": None,
                "profile_enabled": False,
                "remote_profile_dir": None,
                "local_default_dirname": None,
            }

        volume_root = Path(PROFILE_VOLUME_PATH)
        if not volume_root.exists():
            raise RuntimeError(
                "Profiling requested but the profile volume is not mounted."
            )

        remote_run_dir = volume_root / run_id
        remote_run_dir.mkdir(parents=True, exist_ok=True)

        prof_dir = scratch_dir / "prof"
        if prof_dir.is_dir():
            shutil.copytree(prof_dir, remote_run_dir / "prof")

        svg_generated = (remote_run_dir / "prof" / "combined.svg").is_file()
        manifest = {
            "exit_code": exit_code,
            "finished_at": finished_at,
            "profile_enabled": True,
            "pytest_args": sanitized_pytest_args,
            "run_id": run_id,
            "started_at": started_at,
            "svg_generated": svg_generated,
            "svg_requested": svg_requested,
        }
        (remote_run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        PROFILE_VOLUME.commit()

        return {
            "exit_code": exit_code,
            "run_id": run_id,
            "profile_enabled": True,
            "remote_profile_dir": f"{PROFILE_VOLUME_PATH}/{run_id}",
            "local_default_dirname": run_id,
            "svg_generated": svg_generated,
            "svg_requested": svg_requested,
        }


def _parse_cli_args(
    cli_args: tuple[str, ...],
) -> tuple[str, int | None, bool, str | None, list[str]]:
    parser = argparse.ArgumentParser(
        prog="modal run modal_pytest_runner.py",
        add_help=False,
        allow_abbrev=False,
        description="Run pytest on Modal with a dynamically selected GPU.",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        help="GPU type to request from Modal (for example: A100, T4, H100).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Optional Modal execution timeout in seconds. Defaults to 1800 seconds.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable pytest-profiling and download the resulting artifacts locally.",
    )
    parser.add_argument(
        "--profile-output-dir",
        help="Directory to download profiling artifacts into when profiling is enabled.",
    )
    parsed, pytest_args = parser.parse_known_args(cli_args)

    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["tests/"]

    return (
        parsed.gpu,
        parsed.timeout,
        parsed.profile,
        parsed.profile_output_dir,
        pytest_args,
    )


@app.local_entrypoint()
def main(*cli_args: str):
    gpu, timeout, cli_profile, profile_output_dir, pytest_args = _parse_cli_args(
        cli_args
    )
    profile_enabled = _profiling_enabled(cli_profile, pytest_args)
    pytest_addopts = os.environ.get("PYTEST_ADDOPTS", "")
    runner_options = {"gpu": gpu}
    hf_token_secret = _hf_token_secret()
    runner_volumes = {HF_CACHE_PATH: HF_CACHE_VOLUME}
    if timeout is not None:
        runner_options["timeout"] = timeout
    if profile_enabled:
        runner_volumes[PROFILE_VOLUME_PATH] = PROFILE_VOLUME
    runner_options["volumes"] = runner_volumes
    if hf_token_secret is not None:
        runner_options["secrets"] = [hf_token_secret]
    runner = TestRunner.with_options(**runner_options)()
    result = runner.run.remote(
        pytest_args=pytest_args,
        pytest_addopts=pytest_addopts,
        profile_enabled=profile_enabled,
    )

    if result["profile_enabled"] and result["run_id"] is not None:
        if profile_output_dir:
            output_dir = Path(profile_output_dir).expanduser().resolve()
        else:
            output_dir = _default_profile_output_dir(result["local_default_dirname"])

        try:
            _download_profile_artifacts(result["run_id"], output_dir)
            print(f"Profile artifacts downloaded to {output_dir}")
            _cleanup_remote_profile_artifacts(result["run_id"])
        except FileNotFoundError as exc:
            print(f"Unable to download profile artifacts: {exc}", file=sys.stderr)
        except OSError as exc:
            print(f"Failed to write local profile artifacts: {exc}", file=sys.stderr)

    sys.exit(result["exit_code"])
