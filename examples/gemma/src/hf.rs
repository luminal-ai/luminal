use half::{bf16, f16};
use hf_hub::api::sync::Api;
use memmap2::MmapOptions;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
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

/// Stored tensor data with shape and converted FP32 bytes
struct StoredTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
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

/// Convert tensor data to f32 vec
fn tensor_to_f32(tensor: &safetensors::tensor::TensorView) -> Vec<f32> {
    let dtype = tensor.dtype();
    let data = tensor.data();

    match dtype {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data).to_vec(),
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
    }
}

/// Combines sharded safetensors files into a single FP32 file with Gemma-specific transformations.
///
/// This function:
/// 1. Loads tensors from shard(s)
/// 2. Filters out vision model weights (for multimodal models)
/// 3. Strips 'language_model.' prefix if present
/// 4. Pre-scales embeddings by sqrt(hidden_size)
/// 5. Creates lm_head from unscaled embeddings
/// 6. Pre-adds 1.0 to RMSNorm weights
/// 7. Converts all to FP32 and writes combined file
pub fn combine_safetensors_to_fp32(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

    // Determine which shard files to load
    let shard_files: Vec<PathBuf> = if single_shard_path.exists() && !index_path.exists() {
        println!("Single shard model detected, converting to FP32...");
        vec![single_shard_path]
    } else if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();

        println!(
            "Loading {} shard files (converting to FP32)...",
            files.len()
        );
        files.into_iter().map(|f| model_dir.join(f)).collect()
    } else {
        return Err("No model.safetensors or model.safetensors.index.json found".into());
    };

    // Load and convert all tensors
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
            // Filter out vision model weights for multimodal Gemma
            if name.starts_with("vision_tower.") || name.starts_with("multi_modal_projector.") {
                continue;
            }

            // Strip language_model. prefix if present
            let new_name = if name.starts_with("language_model.") {
                name.strip_prefix("language_model.").unwrap().to_string()
            } else {
                name.to_string()
            };

            let tensor = st.tensor(name)?;
            let shape: Vec<usize> = tensor.shape().to_vec();
            let fp32_data = tensor_to_f32(&tensor);

            all_tensors.insert(
                new_name,
                StoredTensor {
                    shape,
                    data: fp32_data,
                },
            );
        }
    }

    println!("Extracted {} language model tensors", all_tensors.len());

    // Gemma 3 scaling:
    // - Input embeddings are scaled by sqrt(hidden_size) = sqrt(2560) ≈ 50.596
    // - lm_head uses unscaled embedding weights (tie_word_embeddings)
    let hidden_size: f32 = 2560.0;
    let embed_scale = hidden_size.sqrt();

    let embed_key = "model.embed_tokens.weight";
    if all_tensors.contains_key(embed_key) {
        println!(
            "Pre-scaling embedding by sqrt({}) = {:.4}...",
            hidden_size as i32, embed_scale
        );

        // Clone data for lm_head (unscaled) and compute scaled data
        let embed_tensor = all_tensors.get(embed_key).unwrap();
        let lm_head = StoredTensor {
            shape: embed_tensor.shape.clone(),
            data: embed_tensor.data.clone(),
        };
        let scaled_data: Vec<f32> = embed_tensor.data.iter().map(|x| x * embed_scale).collect();

        // Now we can mutate
        all_tensors.insert("lm_head.weight".to_string(), lm_head);
        all_tensors.get_mut(embed_key).unwrap().data = scaled_data;
    }

    // Gemma 3 RMSNorm uses (1 + weight) instead of just weight
    // Pre-add 1.0 to all norm weights so the model can use simple multiplication
    println!("Pre-adding 1.0 to RMSNorm weights (Gemma uses 1+weight pattern)...");
    let norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "model.norm.weight",
        "q_norm.weight",
        "k_norm.weight",
    ];

    let norm_keys: Vec<String> = all_tensors
        .keys()
        .filter(|key| norm_patterns.iter().any(|p| key.contains(p)))
        .cloned()
        .collect();

    for key in &norm_keys {
        if let Some(tensor) = all_tensors.get_mut(key) {
            tensor.data = tensor.data.iter().map(|x| x + 1.0).collect();
        }
    }
    println!("  Transformed {} norm weight tensors", norm_keys.len());

    // Serialize to combined file
    println!("Saving combined FP32 model to {}...", output_path.display());

    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let data_bytes: &[u8] = bytemuck::cast_slice(&stored.data);
            let view = TensorView::new(Dtype::F32, stored.shape.clone(), data_bytes).unwrap();
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::serialize(&tensor_views, None)?;

    let mut file = File::create(&output_path)?;
    file.write_all(&serialized)?;

    println!("Combined FP32 model saved successfully!");
    Ok(output_path)
}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - model_combined.safetensors (FP32)
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_fp32(&model_dir)?;
    Ok(model_dir)
}
