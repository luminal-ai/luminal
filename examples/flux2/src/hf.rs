//! HuggingFace download helpers for the Flux 2 multi-folder repo layout.
//!
//! Unlike the LLM examples we deliberately do **not** combine shards into a
//! single FP32 file: Flux 2's transformer weights are 70 GB on disk in BF16
//! (~140 GB if upcast to F32) and the text encoder is ~48 GB. We download the
//! original BF16 / F32 shards as-is and load them directly via
//! `runtime.load_safetensors`, which already supports BF16 in luminal_cuda_lite.

use hf_hub::api::sync::Api;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

const REPO_ID: &str = "black-forest-labs/FLUX.2-dev";

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

fn api() -> Result<hf_hub::api::sync::ApiRepo, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    Ok(api.model(REPO_ID.to_string()))
}

/// Download a single file from a sub-folder under the Flux 2 repo, returning
/// the local cache path.
pub fn fetch(path_in_repo: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(api()?.get(path_in_repo)?)
}

/// Resolve every shard listed in a sub-folder's safetensors index, downloading
/// what's missing. Returns the absolute local paths in shard order.
pub fn fetch_sharded(folder: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let index_path = fetch(&format!(
        "{folder}/diffusion_pytorch_model.safetensors.index.json"
    ))
    .or_else(|_| fetch(&format!("{folder}/model.safetensors.index.json")))?;

    let raw = std::fs::read_to_string(&index_path)?;
    let idx: SafetensorsIndex = serde_json::from_str(&raw)?;

    let mut files: Vec<String> = idx.weight_map.values().cloned().collect();
    files.sort();
    files.dedup();

    let mut paths = Vec::with_capacity(files.len());
    for f in files {
        paths.push(fetch(&format!("{folder}/{f}"))?);
    }
    Ok(paths)
}

/// Convenience wrapper for the small VAE (single safetensors file).
pub fn fetch_vae() -> Result<PathBuf, Box<dyn std::error::Error>> {
    fetch("vae/diffusion_pytorch_model.safetensors")
}

/// Tokenizer JSON (Pixtral / Mistral tokenizer used by Flux 2's text encoder).
pub fn fetch_tokenizer() -> Result<PathBuf, Box<dyn std::error::Error>> {
    fetch("tokenizer/tokenizer.json")
}
