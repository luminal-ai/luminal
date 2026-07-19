APP_ROOT := $(abspath .)
CUDA_PKG := luminal_cuda_lite
CUDA_LITE_TARGET_BASE := $(APP_ROOT)/target/cuda-lite

ifneq ($(CUDARC_VERSION),)
export CUDARC_CUDA_VERSION := $(CUDARC_VERSION)
export CARGO_TARGET_DIR := $(CUDA_LITE_TARGET_BASE)/$(CUDARC_VERSION)
endif

CUDA_TAG                ?= 13.3.0
CUDA_BASE_IMAGE         ?= nvidia/cuda:$(CUDA_TAG)-devel-ubuntu22.04
LUMINAL_DOCKER_IMAGE    ?= luminal-docker
LUMINAL_DOCKER_REGISTRY ?= ghcr.io/luminal-ai
# Local: luminal-docker:cuda-13.3.0 — same name family as ghcr.io/luminal-ai/luminal-docker:cuda
LUMINAL_DOCKER_CUDA_IMAGE        ?= $(LUMINAL_DOCKER_IMAGE):cuda-$(CUDA_TAG)
LUMINAL_DOCKER_CUDA_IMAGE_REMOTE ?= $(LUMINAL_DOCKER_REGISTRY)/$(LUMINAL_DOCKER_IMAGE):cuda-$(CUDA_TAG)
CUDA_DOCKER_TARGET := $(CUDA_LITE_TARGET_BASE)/docker-$(CUDA_TAG)

.PHONY: help \
	cuda-lite-test \
	cuda-lite-test-version \
	cuda-lite-test-unit \
	cuda-lite-test-graph \
	cuda-lite-test-ignored \
	cuda-lite-test-all \
	cuda-devel-image \
	cuda-devel-image-push \
	cuda-lite-docker-test \
	cuda-lite-docker-test-all
.SILENT:

.DEFAULT_GOAL := help

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-28s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

cuda-lite-test: cuda-lite-test-version cuda-lite-test-unit cuda-lite-test-graph ## Default luminal_cuda_lite smoke (version + unit + graph)

cuda-lite-test-version: ## luminal_cuda_lite CPU parser tests (cuda_version_detect)
	cargo test -p $(CUDA_PKG) --test cuda_version_detect

cuda-lite-test-unit: ## luminal_cuda_lite GPU unit suite (skips #[ignore])
	cargo test -p $(CUDA_PKG)

cuda-lite-test-graph: ## luminal_cuda_lite cuda_graph::tests smoke
	cargo test -p $(CUDA_PKG) cuda_graph::tests

cuda-lite-test-ignored: ## luminal_cuda_lite expensive GPU sweeps (--ignored)
	cargo test -p $(CUDA_PKG) -- --ignored

cuda-lite-test-all: cuda-lite-test-version cuda-lite-test-unit cuda-lite-test-graph cuda-lite-test-ignored ## luminal_cuda_lite full run including ignored

cuda-devel-image: ## Build luminal-docker:cuda-$(CUDA_TAG) (CUDA+Rust devel; pushable to GHCR)
	docker build -f scripts/cuda-devel.Dockerfile \
		--build-arg CUDA_BASE_IMAGE=$(CUDA_BASE_IMAGE) \
		-t $(LUMINAL_DOCKER_CUDA_IMAGE) \
		-t $(LUMINAL_DOCKER_CUDA_IMAGE_REMOTE) \
		$(APP_ROOT)

cuda-devel-image-push: cuda-devel-image ## Push to $(LUMINAL_DOCKER_CUDA_IMAGE_REMOTE)
	docker push $(LUMINAL_DOCKER_CUDA_IMAGE_REMOTE)

cuda-lite-docker-test: cuda-devel-image ## luminal_cuda_lite tests in luminal-docker:cuda-$(CUDA_TAG)
	mkdir -p "$(CUDA_LITE_TARGET_BASE)/tmp" "$(CUDA_DOCKER_TARGET)"
	docker run --rm --gpus all \
		-v "$(APP_ROOT):/work" -w /work \
		-e CARGO_TARGET_DIR=/work/target/cuda-lite/docker-$(CUDA_TAG) \
		-e TMPDIR=/work/target/cuda-lite/tmp \
		$(LUMINAL_DOCKER_CUDA_IMAGE) \
		bash -lc 'unset CUDARC_CUDA_VERSION; nvcc --version; nvidia-smi; make cuda-lite-test'

cuda-lite-docker-test-all: cuda-devel-image ## luminal_cuda_lite full suite in luminal-docker:cuda-$(CUDA_TAG)
	mkdir -p "$(CUDA_LITE_TARGET_BASE)/tmp" "$(CUDA_DOCKER_TARGET)"
	docker run --rm --gpus all \
		-v "$(APP_ROOT):/work" -w /work \
		-e CARGO_TARGET_DIR=/work/target/cuda-lite/docker-$(CUDA_TAG) \
		-e TMPDIR=/work/target/cuda-lite/tmp \
		$(LUMINAL_DOCKER_CUDA_IMAGE) \
		bash -lc 'unset CUDARC_CUDA_VERSION; nvcc --version; nvidia-smi; make cuda-lite-test-all'
