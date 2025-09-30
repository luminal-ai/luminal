#base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

#environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

#install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y RUN rustup default stable

#work dir inside container
WORKDIR /workspace

#copy into container
COPY . .

#build cargo in release mode
RUN cargo build --release

#run unit tests
CMD ["cargo", "test", "--release"]