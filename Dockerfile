# Base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN . "$HOME/.cargo/env" && rustup default stable

# Verify CUDA installation
RUN nvcc --version && \
    printf "#include <cuda_runtime.h>\nint main() { return 0; }" > test.cu && \
    nvcc test.cu -o test && \
    rm test.cu test

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Build in release mode
RUN . "$HOME/.cargo/env" && cargo build --release

# Run tests
CMD ["/bin/bash", "-c", "source $HOME/.cargo/env && cargo test --release"]