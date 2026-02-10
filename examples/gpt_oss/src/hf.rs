use half::bf16;
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

use luminal_cuda::cudarc::driver::CudaStream;
use std::sync::Arc;

use crate::model::{
    ExpertWeightBuffers, SinkBuffers, FUSED_INTERMEDIATE, HIDDEN, INTERMEDIATE, LAYERS,
    NUM_EXPERTS, N_HEADS,
};

/// Index file structure for sharded safetensors models
#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stored tensor: either FP32 data or raw U8 bytes
enum StoredTensor {
    F32 { shape: Vec<usize>, data: Vec<f32> },
}

/// Downloads model files from HuggingFace and returns the cache directory path.
pub fn download_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model(repo_id.to_string());

    // Download tokenizer
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_dir = tokenizer_path.parent().unwrap().to_path_buf();

    // Download sharded model
    let index_path = repo.get("model.safetensors.index.json")?;
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    println!("Downloading {} shard files...", shard_files.len());
    for shard_file in &shard_files {
        println!("  Downloading {shard_file}...");
        repo.get(shard_file)?;
    }

    Ok(model_dir)
}

/// Check if a tensor is an expert weight that should NOT go in combined safetensors
fn is_expert_tensor(name: &str) -> bool {
    name.contains("mlp.experts.")
}

/// Combine non-quantized tensors into a single model_combined.safetensors file.
/// Expert weights are loaded separately via load_expert_weights().
pub fn combine_safetensors(model_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined.safetensors");

    if output_path.exists() {
        println!("Combined model already exists, skipping...");
        return Ok(output_path);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
    shard_files.sort();
    shard_files.dedup();

    let mut all_tensors: HashMap<String, StoredTensor> = HashMap::new();

    for shard_file in &shard_files {
        let shard_path = model_dir.join(shard_file);
        println!("  Loading {shard_file}...");
        let file = File::open(&shard_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        for name in st.names() {
            // Skip expert weights (loaded separately into GPU buffers)
            if is_expert_tensor(name) {
                continue;
            }

            let tensor = st.tensor(name)?;

            // Convert BF16/F16 to FP32
            let fp32_data = match tensor.dtype() {
                Dtype::F32 => bytemuck::cast_slice::<u8, f32>(tensor.data()).to_vec(),
                Dtype::F16 => {
                    let f16_slice: &[half::f16] = bytemuck::cast_slice(tensor.data());
                    f16_slice.iter().map(|x| x.to_f32()).collect()
                }
                Dtype::BF16 => {
                    let bf16_slice: &[bf16] = bytemuck::cast_slice(tensor.data());
                    bf16_slice.iter().map(|x| x.to_f32()).collect()
                }
                other => {
                    println!("  Skipping {name} (unsupported dtype {other:?})");
                    continue;
                }
            };

            println!(
                "  Converting {name}: {:?} ({:?} -> F32)",
                tensor.shape(),
                tensor.dtype()
            );
            all_tensors.insert(
                name.to_string(),
                StoredTensor::F32 {
                    shape: tensor.shape().to_vec(),
                    data: fp32_data,
                },
            );
        }
    }

    println!(
        "Prepared {} non-quantized tensors for output",
        all_tensors.len()
    );
    println!("Saving combined model to {}...", output_path.display());

    let tensor_views: HashMap<String, TensorView<'_>> = all_tensors
        .iter()
        .map(|(name, stored)| {
            let view = match stored {
                StoredTensor::F32 { shape, data } => {
                    let data_bytes: &[u8] = bytemuck::cast_slice(data);
                    TensorView::new(Dtype::F32, shape.clone(), data_bytes).unwrap()
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

/// MXFP4 buffer sizes
const PACKED_PER_COL_GU: usize = HIDDEN / 2; // 1440
const SCALES_PER_COL_GU: usize = HIDDEN / 32; // 90

const PACKED_PER_COL_D: usize = INTERMEDIATE / 2; // 1440
const SCALES_PER_COL_D: usize = INTERMEDIATE / 32; // 90

/// Interleave blocks and scales per-column for MXFP4 kernel layout.
///
/// Input: blocks [num_cols, K/2] U8, scales [num_cols, K/32] U8
/// Output: [num_cols, K/2 + K/32] U8 interleaved per column
fn interleave_mxfp4(
    blocks: &[u8],
    scales: &[u8],
    num_cols: usize,
    packed_per_col: usize,
    scales_per_col: usize,
) -> Vec<u8> {
    let col_stride = packed_per_col + scales_per_col;
    let mut buf = vec![0u8; num_cols * col_stride];

    for col in 0..num_cols {
        let dst = col * col_stride;
        // Copy packed blocks
        let src_blocks = col * packed_per_col;
        buf[dst..dst + packed_per_col]
            .copy_from_slice(&blocks[src_blocks..src_blocks + packed_per_col]);
        // Copy scales
        let src_scales = col * scales_per_col;
        buf[dst + packed_per_col..dst + col_stride]
            .copy_from_slice(&scales[src_scales..src_scales + scales_per_col]);
    }

    buf
}

/// Load expert weights from HF safetensors shards directly into GPU buffers.
///
/// For each layer:
///   - gate_up_proj_blocks [128, 5760, 1440] + gate_up_proj_scales [128, 5760, 90]
///     → interleave into [128 * 5760, 1530] and upload to GPU
///   - down_proj_blocks [128, 2880, 1440] + down_proj_scales [128, 2880, 90]
///     → interleave into [128 * 2880, 1530] and upload to GPU
///   - gate_up_proj_bias [128, 5760] BF16 → FP32 → GPU
///   - down_proj_bias [128, 2880] BF16 → FP32 → GPU
pub fn load_expert_weights(
    model_dir: &Path,
    expert_weights: &mut ExpertWeightBuffers,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index_content = std::fs::read_to_string(&index_path)?;
    let index: SafetensorsIndex = serde_json::from_str(&index_content)?;

    // Build mapping: tensor_name → shard_file
    let tensor_to_shard: HashMap<String, String> = index.weight_map;

    // Group shard files by what layers they contain
    // Process layer by layer to minimize memory usage
    for layer in 0..LAYERS {
        println!("  Loading expert weights for layer {layer}...");

        let gu_blocks_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj_blocks");
        let gu_scales_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj_scales");
        let gu_bias_name = format!("model.layers.{layer}.mlp.experts.gate_up_proj_bias");
        let d_blocks_name = format!("model.layers.{layer}.mlp.experts.down_proj_blocks");
        let d_scales_name = format!("model.layers.{layer}.mlp.experts.down_proj_scales");
        let d_bias_name = format!("model.layers.{layer}.mlp.experts.down_proj_bias");

        // Find which shards contain these tensors
        let names = [
            &gu_blocks_name,
            &gu_scales_name,
            &gu_bias_name,
            &d_blocks_name,
            &d_scales_name,
            &d_bias_name,
        ];
        let mut needed_shards: Vec<String> = names
            .iter()
            .filter_map(|n| tensor_to_shard.get(*n).cloned())
            .collect();
        needed_shards.sort();
        needed_shards.dedup();

        // Load needed shards and extract tensors
        let mut raw_data: HashMap<String, (Vec<u8>, Dtype, Vec<usize>)> = HashMap::new();

        for shard in &needed_shards {
            let shard_path = model_dir.join(shard);
            let file = File::open(&shard_path)?;
            let mmap = unsafe { MmapOptions::new().map(&file)? };
            let st = SafeTensors::deserialize(&mmap)?;

            for name in &names {
                if tensor_to_shard
                    .get(*name)
                    .map(|s| s == shard)
                    .unwrap_or(false)
                {
                    if let Ok(tensor) = st.tensor(name) {
                        raw_data.insert(
                            name.to_string(),
                            (
                                tensor.data().to_vec(),
                                tensor.dtype(),
                                tensor.shape().to_vec(),
                            ),
                        );
                    }
                }
            }
        }

        // Process gate_up
        {
            let (blocks_data, _, blocks_shape) = raw_data
                .get(&gu_blocks_name)
                .unwrap_or_else(|| panic!("Missing {gu_blocks_name}"));
            let (scales_data, _, scales_shape) = raw_data
                .get(&gu_scales_name)
                .unwrap_or_else(|| panic!("Missing {gu_scales_name}"));

            assert_eq!(blocks_shape[0], NUM_EXPERTS);
            assert_eq!(blocks_shape[1], FUSED_INTERMEDIATE);
            assert_eq!(scales_shape[0], NUM_EXPERTS);
            assert_eq!(scales_shape[1], FUSED_INTERMEDIATE);

            let total_cols = NUM_EXPERTS * FUSED_INTERMEDIATE;
            let interleaved = interleave_mxfp4(
                blocks_data,
                scales_data,
                total_cols,
                PACKED_PER_COL_GU,
                SCALES_PER_COL_GU,
            );

            stream
                .memcpy_htod(&interleaved, &mut expert_weights.gate_up[layer])
                .unwrap();
        }

        // Process down
        {
            let (blocks_data, _, blocks_shape) = raw_data
                .get(&d_blocks_name)
                .unwrap_or_else(|| panic!("Missing {d_blocks_name}"));
            let (scales_data, _, scales_shape) = raw_data
                .get(&d_scales_name)
                .unwrap_or_else(|| panic!("Missing {d_scales_name}"));

            assert_eq!(blocks_shape[0], NUM_EXPERTS);
            assert_eq!(blocks_shape[1], HIDDEN);
            assert_eq!(scales_shape[0], NUM_EXPERTS);
            assert_eq!(scales_shape[1], HIDDEN);

            let total_cols = NUM_EXPERTS * HIDDEN;
            let interleaved = interleave_mxfp4(
                blocks_data,
                scales_data,
                total_cols,
                PACKED_PER_COL_D,
                SCALES_PER_COL_D,
            );

            stream
                .memcpy_htod(&interleaved, &mut expert_weights.down[layer])
                .unwrap();
        }

        // Process gate_up bias: BF16 → FP32
        {
            let (bias_data, bias_dtype, _) = raw_data
                .get(&gu_bias_name)
                .unwrap_or_else(|| panic!("Missing {gu_bias_name}"));
            let fp32_data = convert_to_f32(bias_data, *bias_dtype);
            let fp32_bytes: &[u8] = bytemuck::cast_slice(&fp32_data);
            stream
                .memcpy_htod(fp32_bytes, &mut expert_weights.gate_up_bias[layer])
                .unwrap();
        }

        // Process down bias: BF16 → FP32
        {
            let (bias_data, bias_dtype, _) = raw_data
                .get(&d_bias_name)
                .unwrap_or_else(|| panic!("Missing {d_bias_name}"));
            let fp32_data = convert_to_f32(bias_data, *bias_dtype);
            let fp32_bytes: &[u8] = bytemuck::cast_slice(&fp32_data);
            stream
                .memcpy_htod(fp32_bytes, &mut expert_weights.down_bias[layer])
                .unwrap();
        }
    }

    println!("Expert weights loaded successfully!");
    Ok(())
}

fn convert_to_f32(data: &[u8], dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::F32 => bytemuck::cast_slice(data).to_vec(),
        Dtype::BF16 => {
            let bf16_slice: &[bf16] = bytemuck::cast_slice(data);
            bf16_slice.iter().map(|x| x.to_f32()).collect()
        }
        Dtype::F16 => {
            let f16_slice: &[half::f16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|x| x.to_f32()).collect()
        }
        other => panic!("Unsupported dtype {other:?} for bias conversion"),
    }
}

/// Load per-layer sink values from the combined safetensors into GPU buffers.
///
/// Each layer has a `self_attn.sinks` tensor of shape [num_attention_heads] = [64]
/// that acts as a learnable per-head scalar logit in the attention softmax.
pub fn load_sinks(
    model_dir: &Path,
    sink_buffers: &mut SinkBuffers,
    stream: &Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let combined_path = model_dir.join("model_combined.safetensors");
    let file = File::open(&combined_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    for layer in 0..LAYERS {
        let name = format!("model.layers.{layer}.self_attn.sinks");
        let tensor = st.tensor(&name)?;
        let fp32_data = convert_to_f32(tensor.data(), tensor.dtype());
        assert_eq!(
            fp32_data.len(),
            N_HEADS,
            "Sinks tensor should have {N_HEADS} elements, got {}",
            fp32_data.len()
        );
        let fp32_bytes: &[u8] = bytemuck::cast_slice(&fp32_data);
        stream
            .memcpy_htod(fp32_bytes, &mut sink_buffers.layers[layer])
            .unwrap();
    }

    Ok(())
}

/// Downloads and prepares the GPT-OSS model.
///
/// Returns the model directory containing:
/// - tokenizer.json
/// - model_combined.safetensors (non-quantized weights only, F32)
/// - Original shard files (for expert weight loading)
pub fn prepare_hf_model(repo_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let model_dir = download_hf_model(repo_id)?;
    combine_safetensors(&model_dir)?;
    Ok(model_dir)
}
