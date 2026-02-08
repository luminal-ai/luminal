#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "safetensors",
#     "torch",
#     "packaging",
#     "numpy",
#     "huggingface-hub"
# ]
# ///

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


def download_model_files(repo_id: str, output_dir: Path):
    print(f"Listing files in {repo_id}...")
    all_files = list_repo_files(repo_id)

    files_to_download = []
    for file in all_files:
        if (
            file == "tokenizer.json"
            or file.endswith(".safetensors")
            or file == "model.safetensors.index.json"
        ):
            files_to_download.append(file)

    print(f"Found {len(files_to_download)} files to download")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in files_to_download:
        print(f"  Downloading {filename}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=None, local_dir=output_dir
        )
        print(f"    Saved to {downloaded_path}")

    print("All files downloaded successfully!")


def combine_and_convert_safetensors_to_fp32(model_dir: Path):
    """
    Combine sharded safetensors into a single file, converting tensors to FP32 on the fly.
    Outputs: model_combined_fp32.safetensors
    """
    output_path = model_dir / "model_combined.safetensors"
    if output_path.exists():
        print(f"FP32 combined safetensors file already exists at {output_path}")
        print("Skipping combine+convert step.")
        return

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})
    shard_files = sorted(set(weight_map.values()))

    # Stream through shards; convert tensors to fp32 as we read.
    all_tensors = {}

    print(f"Loading {len(shard_files)} shard files (converting to fp32)...")
    for shard_file in shard_files:
        shard_path = model_dir / shard_file
        print(f"  Loading {shard_file}...")
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                t = sf.get_tensor(key)

                # Convert float dtypes to fp32; keep non-floats as-is (e.g., int tensors, masks).
                if torch.is_floating_point(t) and t.dtype != torch.float32:
                    t = t.to(dtype=torch.float32)

                all_tensors[key] = t

    print(f"Saving combined FP32 model to {output_path}...")
    save_file(all_tensors, output_path)
    print(f"Combined FP32 model saved successfully to {output_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    repo_id = "NousResearch/Meta-Llama-3-8B-Instruct"

    download_model_files(repo_id, script_dir)

    print("\nCombining + converting safetensors to FP32...")
    combine_and_convert_safetensors_to_fp32(script_dir)

    print("\nDone!")
