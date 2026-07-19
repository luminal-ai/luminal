# Luminal CUDA+Rust devel image. Tags: luminal-docker:cuda-<tag> (local) and
# ghcr.io/luminal-ai/luminal-docker:cuda-<tag> (remote; see make cuda-devel-image-push).
ARG CUDA_BASE_IMAGE=nvidia/cuda:13.3.0-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config git make && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /work
