"""Run reference tests on Modal.

Pytest paths are relative to crates/pytorch. With no paths, this runs the
backend tests and translator parity (plus shared model cases for CUDA).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ci.modal_pytorch_tests import create_app

app, run_tests = create_app("reference", Path(__file__).resolve().parent)


@app.local_entrypoint()
def main(*cli_args: str):
    run_tests(*cli_args)
