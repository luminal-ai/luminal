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

use crate::model::{HIDDEN, LAYERS, MOE_INTERMEDIATE, NUM_EXPERTS};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor with raw bytes and dtype
struct StoredTensor {
    shape: Vec<usize>,
    data: Vec<u8>, // raw bytes in the stored dtype
    dtype: Dtype,
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

/// Convert tensor data to f32 bytes (as raw u8)
fn tensor_to_f32_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    let dtype = tensor.dtype();
    let data = tensor.data();

    let f32_data: Vec<f32> = match dtype {
        Dtype::F32 => return data.to_vec(), // already F32 bytes
        Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => {
            panic!("Unsupported dtype for conversion: {other:?}");
        }
    };
    bytemuck::cast_slice(&f32_data).to_vec()
}

/// Check if a tensor name is an expert weight (large, should be stored as BF16)
fn is_expert_weight(name: &str) -> bool {
    name.contains(".mlp.experts.")
        || name.contains(".mlp.gate_up_weights")
        || name.contains(".mlp.down_weights")
}

/// Combines sharded safetensors files into a single mixed-precision file.
///
/// Expert weights are stored as BF16 (saves ~60GB), non-expert weights as F32 (~6GB).
pub fn combine_safetensors(model_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

    // Determine which shard files to load
    let shard_files: Vec<PathBuf> = if single_shard_path.exists() && !index_path.exists() {
        println!("Single shard model detected...");
        vec![single_shard_path]
    } else if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();

        println!("Loading {} shard files...", files.len());
        files.into_iter().map(|f| model_dir.join(f)).collect()
    } else {
        return Err("No model.safetensors or model.safetensors.index.json found".into());
    };

    // Load all tensors: expert weights kept as BF16, non-expert converted to F32
    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    for shard_path in &shard_files {
        println!(
            "  Loading {}...",
            shard_path.file_name().unwrap().to_string_lossy()
        );
        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();

            if is_expert_weight(name) {
                // Expert weights as BF16 (saves ~54 GB → ~27 GB)
                let data = match tensor.dtype() {
                    Dtype::BF16 => tensor.data().to_vec(),
                    Dtype::F16 => {
                        let f16_slice: &[f16] = bytemuck::cast_slice(tensor.data());
                        let bf16_data: Vec<bf16> = f16_slice
                            .iter()
                            .map(|x| bf16::from_f32(x.to_f32()))
                            .collect();
                        bytemuck::cast_slice(&bf16_data).to_vec()
                    }
                    Dtype::F32 => {
                        let f32_slice: &[f32] = bytemuck::cast_slice(tensor.data());
                        let bf16_data: Vec<bf16> =
                            f32_slice.iter().map(|x| bf16::from_f32(*x)).collect();
                        bytemuck::cast_slice(&bf16_data).to_vec()
                    }
                    other => panic!("Unsupported dtype: {other:?}"),
                };
                all_tensors.insert(
                    name.to_string(),
                    StoredTensor {
                        shape,
                        data,
                        dtype: Dtype::BF16,
                    },
                );
            } else {
                // Non-expert weights as F32
                let data = tensor_to_f32_bytes(&tensor);
                all_tensors.insert(
                    name.to_string(),
                    StoredTensor {
                        shape,
                        data,
                        dtype: Dtype::F32,
                    },
                );
            }
        }
    }

    println!("Extracted {} tensors", all_tensors.len());

    // Stack per-expert weights into combined tensors
    println!("Stacking expert weights (BF16)...");
    let gate_size_bf16 = MOE_INTERMEDIATE * HIDDEN * 2; // bytes: 768 * 2048 * 2
    let gate_up_size_bf16 = MOE_INTERMEDIATE * 2 * HIDDEN * 2; // bytes after concat
    let down_size_bf16 = HIDDEN * MOE_INTERMEDIATE * 2; // bytes: 2048 * 768 * 2

    for l in 0..LAYERS {
        // Concatenate gate_proj + up_proj per expert, then stack
        let mut gate_up_data = Vec::with_capacity(NUM_EXPERTS * gate_up_size_bf16);
        for e in 0..NUM_EXPERTS {
            let gate_key = format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight");
            let up_key = format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight");
            let gate = all_tensors
                .remove(&gate_key)
                .unwrap_or_else(|| panic!("Missing tensor: {gate_key}"));
            let up = all_tensors
                .remove(&up_key)
                .unwrap_or_else(|| panic!("Missing tensor: {up_key}"));
            assert_eq!(
                gate.data.len(),
                gate_size_bf16,
                "gate_proj size mismatch layer {l} expert {e}"
            );
            assert_eq!(
                up.data.len(),
                gate_size_bf16,
                "up_proj size mismatch layer {l} expert {e}"
            );
            // Concatenate: gate first, then up
            gate_up_data.extend_from_slice(&gate.data);
            gate_up_data.extend_from_slice(&up.data);
        }
        all_tensors.insert(
            format!("model.layers.{l}.mlp.gate_up_weights"),
            StoredTensor {
                shape: vec![NUM_EXPERTS, MOE_INTERMEDIATE * 2, HIDDEN],
                data: gate_up_data,
                dtype: Dtype::BF16,
            },
        );

        // Stack down_proj weights
        let mut down_data = Vec::with_capacity(NUM_EXPERTS * down_size_bf16);
        for e in 0..NUM_EXPERTS {
            let key = format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight");
            let tensor = all_tensors
                .remove(&key)
                .unwrap_or_else(|| panic!("Missing tensor: {key}"));
            assert_eq!(
                tensor.data.len(),
                down_size_bf16,
                "down_proj size mismatch layer {l} expert {e}"
            );
            down_data.extend_from_slice(&tensor.data);
        }
        all_tensors.insert(
            format!("model.layers.{l}.mlp.down_weights"),
            StoredTensor {
                shape: vec![NUM_EXPERTS, HIDDEN, MOE_INTERMEDIATE],
                data: down_data,
                dtype: Dtype::BF16,
            },
        );

        if (l + 1) % 10 == 0 {
            println!("  Stacked experts for {}/{} layers", l + 1, LAYERS);
        }
    }

    println!(
        "After stacking: {} tensors in combined file",
        all_tensors.len()
    );

    // Serialize to combined file
    println!("Saving combined model (BF16 experts + F32 rest)...");

    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let view = TensorView::new(stored.dtype, stored.shape.clone(), &stored.data).unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::serialize(&tensor_views, None)?;

    let mut file = File::create(&output_path)?;
    file.write_all(&serialized)?;

    println!("Combined model saved successfully!");
    Ok(output_path)
}

/// Downloads a model from HuggingFace and prepares it for use.
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors(&model_dir)?;
    Ok(model_dir)
}
