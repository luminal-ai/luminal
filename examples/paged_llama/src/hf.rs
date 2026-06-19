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

/// Stored tensor data with shape, dtype and serialized bytes.
struct StoredTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
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

fn tensor_to_f32_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    let fp32 = tensor_to_f32(tensor);
    bytemuck::cast_slice(&fp32).to_vec()
}

fn tensor_to_bf16_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    match tensor.dtype() {
        Dtype::BF16 => tensor.data().to_vec(),
        _ => tensor_to_f32(tensor)
            .into_iter()
            .flat_map(|x| bf16::from_f32(x).to_le_bytes())
            .collect(),
    }
}

/// Norm weights stay F32 in the bf16 pipeline: the model computes norms in
/// F32 (explicit casts) and only the linear/embedding weights are bf16.
fn keep_f32_in_bf16_pipeline(name: &str) -> bool {
    name.contains("norm")
}

fn stored_tensor_bf16(name: &str, tensor: &safetensors::tensor::TensorView) -> StoredTensor {
    let shape = tensor.shape().to_vec();
    if keep_f32_in_bf16_pipeline(name) {
        StoredTensor {
            shape,
            dtype: Dtype::F32,
            data: tensor_to_f32_bytes(tensor),
        }
    } else {
        StoredTensor {
            shape,
            dtype: Dtype::BF16,
            data: tensor_to_bf16_bytes(tensor),
        }
    }
}

fn model_shard_files(model_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

    if single_shard_path.exists() && !index_path.exists() {
        Ok(vec![single_shard_path])
    } else if index_path.exists() {
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

        let mut files: Vec<String> = index.weight_map.values().cloned().collect();
        files.sort();
        files.dedup();

        Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
    } else {
        Err("No model.safetensors or model.safetensors.index.json found".into())
    }
}

/// Combines sharded safetensors into a single BF16 file (norm weights kept
/// F32). The paged_llama model is bf16-only: linears, embedding and lm_head
/// are bf16; norms compute in F32 from F32 weights.
pub fn combine_safetensors_to_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined_bf16_v1.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    println!(
        "Loading {} shard files (converting to BF16, norms F32)...",
        shard_files.len()
    );

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
            let tensor = st.tensor(name)?;
            all_tensors.insert(name.to_string(), stored_tensor_bf16(name, &tensor));
        }
    }

    println!("Extracted {} language model tensors", all_tensors.len());
    println!("Saving combined BF16 model to {}...", output_path.display());

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

    println!("Combined BF16 model saved successfully!");
    Ok(output_path)
}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - model_combined_bf16_v1.safetensors (BF16 linears/embeddings, F32 norms)
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors_to_bf16(&model_dir)?;
    Ok(model_dir)
}
