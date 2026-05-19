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
use tracing::info;

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor data with shape, dtype, and serialized bytes.
struct StoredTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Fp32,
    Fp8,
}

pub struct PreparedModel {
    pub model_dir: PathBuf,
    pub weight_files: Vec<PathBuf>,
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    info!("Downloading model from HuggingFace: {repo_id}");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    // Download tokenizer
    info!("Downloading tokenizer.json...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    info!("Model cache directory: {}", model_dir.display());
    // Try to download single shard model first
    info!("Checking for single-shard model...");
    if repo.get("model.safetensors").is_ok() {
        info!("Single-shard model downloaded successfully.");
        return Ok(model_dir);
    }
    // Otherwise download sharded model
    info!("Single shard not found, downloading sharded model index...");
    let index_path = repo.get("model.safetensors.index.json")?;
    // Parse index to find shard files
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    // Get unique shard files
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    info!("Found {} shard files to download.", shard_files.len());
    // Download each shard
    for (i, shard_file) in shard_files.iter().enumerate() {
        info!(
            "Downloading shard {}/{}: {shard_file}",
            i + 1,
            shard_files.len()
        );
        repo.get(shard_file)?;
    }
    info!("All shards downloaded successfully.");
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

fn stored_tensor_from_view(
    tensor: &safetensors::tensor::TensorView,
    preserve_fp8: bool,
) -> StoredTensor {
    let shape = tensor.shape().to_vec();
    let dtype = tensor.dtype();
    match dtype {
        Dtype::F32 if preserve_fp8 => StoredTensor {
            shape,
            dtype,
            data: tensor.data().to_vec(),
        },
        Dtype::F8_E4M3 | Dtype::F8_E5M2 | Dtype::F8_E8M0 if preserve_fp8 => StoredTensor {
            shape,
            dtype,
            data: tensor.data().to_vec(),
        },
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => StoredTensor {
            shape,
            dtype: Dtype::F32,
            data: tensor_to_f32_bytes(tensor),
        },
        other => {
            panic!("Unsupported dtype for model preparation: {other:?}");
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

/// Combines sharded safetensors files into a single FP32 file.
///
/// This function:
/// 1. Loads tensors from shard(s)
/// 2. Converts all to FP32
/// 3. Writes combined file
pub fn combine_safetensors_to_fp32(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    // Skip if already combined
    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    info!(
        "Loading {} shard files (converting to FP32)...",
        shard_files.len()
    );

    // Load and convert all tensors
    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    for shard_path in &shard_files {
        info!(
            "  Loading {}...",
            shard_path.file_name().unwrap().to_string_lossy()
        );
        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            all_tensors.insert(name.to_string(), stored_tensor_from_view(&tensor, false));
        }
    }

    info!("Extracted {} language model tensors", all_tensors.len());

    // Serialize to combined file
    info!("Saving combined FP32 model to {}...", output_path.display());

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

    info!("Combined FP32 model saved successfully!");
    Ok(output_path)
}

/// Combines sharded safetensors files into one file while preserving FP8 tensors.
///
/// Non-FP8 tensors are converted to FP32 so the existing embedding, norm, and
/// output-head graph inputs can still load without changing their dtypes.
pub fn combine_safetensors_preserve_fp8(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined_fp8.safetensors");

    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    info!(
        "Loading {} shard files (preserving FP8 tensors)...",
        shard_files.len()
    );

    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    for shard_path in &shard_files {
        info!(
            "  Loading {}...",
            shard_path.file_name().unwrap().to_string_lossy()
        );
        let file = File::open(shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            let tensor = st.tensor(name)?;
            all_tensors.insert(name.to_string(), stored_tensor_from_view(&tensor, true));
        }
    }

    info!("Extracted {} language model tensors", all_tensors.len());
    info!(
        "Saving mixed FP8/FP32 model to {}...",
        output_path.display()
    );

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

    info!("Combined mixed FP8/FP32 model saved successfully!");
    Ok(output_path)
}

/// Downloads a model from HuggingFace and prepares it for use.
///
/// Returns the path to the model directory containing:
/// - tokenizer.json
/// - a combined safetensors file for the requested weight format
pub fn prepare_hf_model(
    repo_id: &str,
    weight_format: WeightFormat,
) -> Result<PreparedModel, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    let weights_path = match weight_format {
        WeightFormat::Fp32 => combine_safetensors_to_fp32(&model_dir)?,
        WeightFormat::Fp8 => combine_safetensors_preserve_fp8(&model_dir)?,
    };
    Ok(PreparedModel {
        model_dir,
        weight_files: vec![weights_path],
    })
}
