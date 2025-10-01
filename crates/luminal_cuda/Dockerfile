FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y curl && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

COPY . .

WORKDIR /app/crates/luminal_cuda

CMD ["cargo", "test", "--release", "--", "--test-threads=1"]