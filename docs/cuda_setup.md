## CUDA Training Setup

Luminal supports CUDA for accelerated ML training. To enable and debug CUDA workflows:
1. Ensure you have the latest NVIDIA drivers and the CUDA toolkit installed (>= 11.x).
2. Set the environment variable `CUDA_HOME` to your CUDA installation path.
3. When using GitHub Actions, use a self-hosted runner with GPU. Example snippet:
    ```
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Set up CUDA
        run: sudo apt-get install -y cuda-toolkit-11-2
      - name: Test CUDA build
        run: make test-cuda
    ```
4. For troubleshooting, run:
    ```
    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"
    ```
5. If you encounter build errors, see `docs/troubleshooting.md`.
