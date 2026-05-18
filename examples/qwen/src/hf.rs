use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

struct StoredTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else {
        format!("{bytes} bytes")
    }
}

fn tensor_to_f32(tensor: &safetensors::tensor::TensorView<'_>) -> Vec<u8> {
    match tensor.dtype() {
        Dtype::F32 => tensor.data().to_vec(),
        Dtype::F16 => bytemuck::cast_slice::<f32, u8>(
            &bytemuck::cast_slice::<u8, f16>(tensor.data())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        Dtype::BF16 => bytemuck::cast_slice::<f32, u8>(
            &bytemuck::cast_slice::<u8, bf16>(tensor.data())
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>(),
        )
        .to_vec(),
        dtype => panic!("Unsupported safetensors dtype {dtype:?}"),
    }
}

fn combine_safetensors_to_fp32(model_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");
    if output_path.exists() {
        let existing_bytes = std::fs::metadata(&output_path)?.len();
        println!(
            "Using existing combined FP32 model at {} ({})",
            output_path.display(),
            format_bytes(existing_bytes)
        );
        return Ok(());
    }

    let single_path = model_dir.join("model.safetensors");
    if single_path.exists() {
        let bytes = std::fs::metadata(&single_path)?.len();
        println!(
            "Using single safetensors model at {} ({})",
            single_path.display(),
            format_bytes(bytes)
        );
        return Ok(());
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    let original_bytes = shard_files.iter().try_fold(0u64, |acc, shard_file| {
        Ok::<_, std::io::Error>(acc + std::fs::metadata(model_dir.join(shard_file))?.len())
    })?;

    println!(
        "Loading {} shard files ({} original bytes, converting to FP32)...",
        shard_files.len(),
        format_bytes(original_bytes)
    );

    let mut tensors = Vec::new();
    for shard_file in &shard_files {
        println!("  Loading {shard_file}...");
        let shard_path = model_dir.join(shard_file);
        let file = File::open(&shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;

        for name in safetensors.names() {
            let tensor = safetensors.tensor(name)?;
            tensors.push(StoredTensor {
                name: name.to_string(),
                shape: tensor.shape().to_vec(),
                data: tensor_to_f32(&tensor),
            });
        }
    }

    let total_params: usize = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().product::<usize>())
        .sum();
    let raw_fp32_bytes: u64 = (total_params * std::mem::size_of::<f32>()) as u64;
    println!(
        "Extracted {} tensors: {} params, {} raw FP32 tensor payload",
        tensors.len(),
        total_params,
        format_bytes(raw_fp32_bytes)
    );

    let tensor_map: HashMap<String, TensorView<'_>> = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                TensorView::new(Dtype::F32, tensor.shape.clone(), &tensor.data).unwrap(),
            )
        })
        .collect();
    let serialized = safetensors::serialize(&tensor_map, None)?;
    println!(
        "Serialized combined FP32 safetensors size: {}",
        format_bytes(serialized.len() as u64)
    );

    println!("Removing original shards before saving combined FP32 model...");
    for shard_file in &shard_files {
        std::fs::remove_file(model_dir.join(shard_file))?;
    }
    std::fs::remove_file(&index_path)?;

    println!("Saving combined FP32 model to {}...", output_path.display());
    let mut output_file = File::create(&output_path)?;
    output_file.write_all(&serialized)?;
    println!("Done!");

    Ok(())
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    // Try to download single shard model first
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }
    // Otherwise download sharded model
    let index_path = repo.get("model.safetensors.index.json")?;
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    // Download each shard
    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }
    Ok(model_dir)
}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - model.safetensors or model_combined.safetensors
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_fp32(&model_dir)?;
    Ok(model_dir)
}
