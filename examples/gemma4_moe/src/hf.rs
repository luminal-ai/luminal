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

use crate::model::HIDDEN;

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

enum TensorData {
    F32(Vec<f32>),
    BF16(Vec<u8>),
}

struct StoredTensor {
    shape: Vec<usize>,
    data: TensorData,
}

pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();

    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }

    let index_path = repo.get("model.safetensors.index.json")?;
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    for shard_file in &shard_files {
        repo.get(shard_file)?;
    }

    Ok(model_dir)
}

fn tensor_to_f32(tensor: &safetensors::tensor::TensorView) -> Vec<f32> {
    match tensor.dtype() {
        Dtype::F32 => bytemuck::cast_slice::<u8, f32>(tensor.data()).to_vec(),
        Dtype::F16 => {
            let f16_slice: &[f16] = bytemuck::cast_slice(tensor.data());
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(tensor.data());
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => panic!("Unsupported dtype for conversion: {other:?}"),
    }
}

fn tensor_to_bf16_bytes(tensor: &safetensors::tensor::TensorView) -> Vec<u8> {
    match tensor.dtype() {
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
            let bf16_data: Vec<bf16> = f32_slice.iter().map(|x| bf16::from_f32(*x)).collect();
            bytemuck::cast_slice(&bf16_data).to_vec()
        }
        other => panic!("Unsupported dtype for conversion: {other:?}"),
    }
}

fn is_text_weight(name: &str) -> bool {
    name.starts_with("model.language_model.")
}

fn is_expert_weight(name: &str) -> bool {
    name.contains(".experts.")
}

pub fn combine_safetensors(model_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");
    if output_path.exists() {
        return Ok(output_path);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let single_shard_path = model_dir.join("model.safetensors");

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
            if !is_text_weight(name) {
                continue;
            }

            let new_name = name.replacen("model.language_model.", "model.", 1);
            let tensor = st.tensor(name)?;

            if new_name.ends_with(".layer_scalar") {
                let scalar = tensor_to_f32(&tensor);
                let scalar = *scalar.first().expect("layer_scalar tensor is empty");
                all_tensors.insert(
                    new_name,
                    StoredTensor {
                        shape: vec![HIDDEN],
                        data: TensorData::F32(vec![scalar; HIDDEN]),
                    },
                );
                continue;
            }

            let shape = tensor.shape().to_vec();
            let data = if is_expert_weight(&new_name) {
                TensorData::BF16(tensor_to_bf16_bytes(&tensor))
            } else {
                TensorData::F32(tensor_to_f32(&tensor))
            };

            all_tensors.insert(new_name, StoredTensor { shape, data });
        }
    }

    println!("Extracted {} text tensors", all_tensors.len());

    let embed_key = "model.embed_tokens.weight";
    if let Some(embed_tensor) = all_tensors.get(embed_key) {
        let (shape, embed_data) = match &embed_tensor.data {
            TensorData::F32(data) => (embed_tensor.shape.clone(), data.clone()),
            TensorData::BF16(_) => unreachable!("Embedding weights should stay in F32"),
        };

        all_tensors.insert(
            "lm_head.weight".to_string(),
            StoredTensor {
                shape,
                data: TensorData::F32(embed_data.clone()),
            },
        );

        let embed_scale = (HIDDEN as f32).sqrt();
        if let Some(stored) = all_tensors.get_mut(embed_key) {
            match &mut stored.data {
                TensorData::F32(data) => {
                    for value in data {
                        *value *= embed_scale;
                    }
                }
                TensorData::BF16(_) => unreachable!("Embedding weights should stay in F32"),
            }
        }
    }

    println!("Saving combined model (BF16 experts + F32 rest)...");
    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let view = match &stored.data {
                TensorData::F32(data) => {
                    let bytes: &[u8] = bytemuck::cast_slice(data);
                    TensorView::new(Dtype::F32, stored.shape.clone(), bytes).unwrap()
                }
                TensorData::BF16(bytes) => {
                    TensorView::new(Dtype::BF16, stored.shape.clone(), bytes).unwrap()
                }
            };
            (name.clone(), view)
        })
        .collect();

    let serialized = safetensors::serialize(&tensor_views, None)?;
    let mut file = File::create(&output_path)?;
    file.write_all(&serialized)?;

    println!("Combined model saved successfully!");
    Ok(output_path)
}

pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors(&model_dir)?;
    Ok(model_dir)
}
