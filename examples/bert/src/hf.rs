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

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

struct StoredTensor {
    shape: Vec<usize>,
    dtype: Dtype,
    data: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    Bf16,
}

pub struct PreparedModel {
    pub model_dir: PathBuf,
    pub weight_files: Vec<PathBuf>,
}

pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    info!("Downloading model from HuggingFace: {repo_id}");
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());
    info!("Downloading tokenizer.json...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();
    info!("Model cache directory: {}", model_dir.display());
    info!("Checking for single-shard model...");
    if repo.get("model.safetensors").is_ok() {
        info!("Single-shard model downloaded successfully.");
        return Ok(model_dir);
    }
    info!("Single shard not found, downloading sharded model index...");
    let index_path = repo.get("model.safetensors.index.json")?;
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;
    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();
    info!("Found {} shard files to download.", shard_files.len());
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
        other => panic!("Unsupported dtype for conversion: {other:?}"),
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

fn keep_f32_in_bf16_pipeline(name: &str) -> bool {
    name.contains("LayerNorm")
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

fn stored_tensor_f32(_name: &str, tensor: &safetensors::tensor::TensorView) -> StoredTensor {
    StoredTensor {
        shape: tensor.shape().to_vec(),
        dtype: Dtype::F32,
        data: tensor_to_f32_bytes(tensor),
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

pub fn combine_safetensors(
    model_dir: &Path,
    convert_bf16: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let suffix = if convert_bf16 { "bf16" } else { "f32" };
    // Bump the filename so existing cached combined files are regenerated
    // after key-mapping changes (e.g. LayerNorm.gamma -> LayerNorm.weight).
    let output_path = model_dir.join(format!("model_combined_v2_{suffix}.safetensors"));

    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    info!(
        "Loading {} shard files (converting to {})...",
        shard_files.len(),
        if convert_bf16 {
            "BF16, norms F32"
        } else {
            "F32"
        }
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
            // HF BERT checkpoints use LayerNorm.gamma / LayerNorm.beta, but the
            // Luminal model declares them as LayerNorm.weight / LayerNorm.bias.
            let mapped_name = if name.ends_with("LayerNorm.gamma") {
                name.replace("LayerNorm.gamma", "LayerNorm.weight")
            } else if name.ends_with("LayerNorm.beta") {
                name.replace("LayerNorm.beta", "LayerNorm.bias")
            } else {
                name.to_string()
            };
            let tensor = st.tensor(name)?;
            all_tensors.insert(
                mapped_name,
                if convert_bf16 {
                    stored_tensor_bf16(name, &tensor)
                } else {
                    stored_tensor_f32(name, &tensor)
                },
            );
        }
    }

    info!("Extracted {} tensors", all_tensors.len());
    info!("Saving combined model to {}...", output_path.display());

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

    info!("Combined model saved successfully!");
    Ok(output_path)
}

pub fn prepare_hf_model(
    repo_id: &str,
    weight_format: WeightFormat,
) -> Result<PreparedModel, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    let convert_bf16 = weight_format == WeightFormat::Bf16;
    let weights_path = combine_safetensors(&model_dir, convert_bf16)?;
    Ok(PreparedModel {
        model_dir,
        weight_files: vec![weights_path],
    })
}
