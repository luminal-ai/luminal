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
    Bf16,
    /// FP8 linear weights with bf16 embeddings/lm_head and F32 norms — the
    /// fp8 + bf16-activation pipeline.
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

/// Combines sharded safetensors into a bf16 file (norm weights kept F32).
pub fn combine_safetensors_to_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // "_fused": q/k/v and gate/up are concatenated into qkv_proj /
    // gate_up_proj rows so each decode step runs one GEMV per group.
    // v2: qkv fused V-FIRST ([v; q; k]) so the V slice is a start-0 strided
    // view (offset slices lower to Gather materializations); rope reads q/k
    // at their offsets inside the row.
    let output_path = model_dir.join("model_combined_bf16_fused_v2.safetensors");

    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    info!(
        "Loading {} shard files (converting to BF16, norms F32)...",
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
            all_tensors.insert(name.to_string(), stored_tensor_bf16(name, &tensor));
        }
    }

    fuse_projection_tensors(&mut all_tensors);

    info!("Extracted {} language model tensors", all_tensors.len());
    info!("Saving combined BF16 model to {}...", output_path.display());

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

    info!("Combined BF16 model saved successfully!");
    Ok(output_path)
}

/// Combines shards for the fp8 + bf16-activation pipeline: FP8 linears and
/// their F32 scales pass through, norms convert to F32, everything else
/// (embeddings, lm_head) converts to BF16.
pub fn combine_safetensors_fp8_bf16(
    model_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output_path = model_dir.join("model_combined_fp8_bf16_fused_v1.safetensors");

    if output_path.exists() {
        return Ok(output_path);
    }

    let shard_files = model_shard_files(model_dir)?;
    info!(
        "Loading {} shard files (FP8 linears, BF16 embeddings, F32 norms)...",
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
            let shape = tensor.shape().to_vec();
            let stored = match tensor.dtype() {
                // FP8 weights and their F32 scale scalars pass through.
                Dtype::F8_E4M3 | Dtype::F8_E5M2 | Dtype::F8_E8M0 => StoredTensor {
                    shape,
                    dtype: tensor.dtype(),
                    data: tensor.data().to_vec(),
                },
                Dtype::F32 => StoredTensor {
                    shape,
                    dtype: Dtype::F32,
                    data: tensor.data().to_vec(),
                },
                _ if keep_f32_in_bf16_pipeline(name) => StoredTensor {
                    shape,
                    dtype: Dtype::F32,
                    data: tensor_to_f32_bytes(&tensor),
                },
                _ => StoredTensor {
                    shape,
                    dtype: Dtype::BF16,
                    data: tensor_to_bf16_bytes(&tensor),
                },
            };
            all_tensors.insert(name.to_string(), stored);
        }
    }

    fuse_projection_tensors_fp8(&mut all_tensors);

    info!("Extracted {} language model tensors", all_tensors.len());
    info!(
        "Saving combined FP8/BF16 model to {}...",
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

    info!("Combined FP8/BF16 model saved successfully!");
    Ok(output_path)
}

/// Concatenate per-layer q/k/v into qkv_proj and gate/up into gate_up_proj.
///
/// Linear weights are stored row-major (out_features, in_features), so output
/// concatenation is a plain byte concat. The originals are removed; the bf16
/// model graph references only the fused names.
fn fuse_projection_tensors(all_tensors: &mut HashMap<String, StoredTensor>) {
    let fuse =
        |tensors: &mut HashMap<String, StoredTensor>, parts: &[String], fused_name: String| {
            let mut shape: Option<Vec<usize>> = None;
            let mut data = Vec::new();
            for part in parts {
                let t = tensors
                    .remove(part)
                    .unwrap_or_else(|| panic!("missing tensor {part} for fusion"));
                match &mut shape {
                    None => {
                        assert_eq!(t.shape.len(), 2, "{part} must be 2-D");
                        shape = Some(t.shape.clone());
                    }
                    Some(shape) => {
                        assert_eq!(shape[1], t.shape[1], "{part} in_features mismatch");
                        shape[0] += t.shape[0];
                    }
                }
                data.extend_from_slice(&t.data);
            }
            let shape = shape.unwrap();
            let dtype = Dtype::BF16;
            tensors.insert(fused_name, StoredTensor { shape, dtype, data });
        };

    let mut layer = 0usize;
    loop {
        let prefix = format!("model.layers.{layer}.");
        if !all_tensors.contains_key(&format!("{prefix}self_attn.q_proj.weight")) {
            break;
        }
        fuse(
            all_tensors,
            &[
                format!("{prefix}self_attn.v_proj.weight"),
                format!("{prefix}self_attn.q_proj.weight"),
                format!("{prefix}self_attn.k_proj.weight"),
            ],
            format!("{prefix}self_attn.vqk_proj.weight"),
        );
        fuse(
            all_tensors,
            &[
                format!("{prefix}mlp.gate_proj.weight"),
                format!("{prefix}mlp.up_proj.weight"),
            ],
            format!("{prefix}mlp.gate_up_proj.weight"),
        );
        layer += 1;
    }
    info!("Fused projections for {layer} layers");
}

/// Decode one e4m3 byte to f32 (NaN for the 0x7F/0xFF codes).
fn f8e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    if exp == 0xF && (b & 0x7) == 0x7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * (man / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// Encode f32 to the nearest e4m3 byte: round-to-nearest-even, saturating to
/// ±448 (matching CUDA's saturating `__nv_fp8_e4m3` conversion).
fn f8e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    if bits & 0x7FFF_FFFF == 0 {
        return sign;
    }
    let mut exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let man = (bits & 0x7F_FFFF) | 0x80_0000; // 24-bit significand, implicit 1
    // Quantize to q = round(|v| / 2^(exp-3)): for normals q = 8 + mantissa;
    // clamping exp to the e4m3 subnormal regime makes q count units of 2^-9.
    let mut shift = 20;
    if exp < -6 {
        shift += -6 - exp;
        exp = -6;
    }
    if shift >= 32 {
        return sign; // underflows to zero
    }
    let half = 1u32 << (shift - 1);
    let low = man & ((1u32 << shift) - 1);
    let mut q = man >> shift;
    if low > half || (low == half && q & 1 == 1) {
        q += 1;
    }
    if q >= 16 {
        q >>= 1;
        exp += 1;
    }
    if exp > 8 || (exp == 8 && q == 15) {
        return sign | 0x7E; // saturate to ±448 (0x7F is NaN)
    }
    if q < 8 {
        return sign | q as u8; // subnormal: biased exponent 0
    }
    sign | (((exp + 7) as u8) << 3) | (q as u8 - 8)
}

/// FP8 variant of [`fuse_projection_tensors`]: concatenates each layer's
/// v/q/k (and gate/up) FP8 weights into single tensors. Per-tensor scales
/// are fused to the max of the part scales (weight and input separately);
/// parts whose weight scale changed are requantized — decoded with the old
/// scale, re-encoded with the shared one — costing one extra e4m3 rounding
/// only on those parts. Input scales are calibrated on the same activation,
/// so taking the max only loosens quantization, never overflows it.
fn fuse_projection_tensors_fp8(all_tensors: &mut HashMap<String, StoredTensor>) {
    let read_scalar = |t: &StoredTensor| -> f32 {
        assert_eq!(t.dtype, Dtype::F32, "fp8 scale must be F32");
        f32::from_le_bytes(t.data[..4].try_into().unwrap())
    };

    let fuse =
        |tensors: &mut HashMap<String, StoredTensor>, parts: &[String], fused_prefix: String| {
            let w_scales: Vec<f32> = parts
                .iter()
                .map(|p| read_scalar(&tensors[&format!("{p}.weight_scale")]))
                .collect();
            let in_scales: Vec<f32> = parts
                .iter()
                .map(|p| read_scalar(&tensors[&format!("{p}.input_scale")]))
                .collect();
            let shared_w = w_scales.iter().copied().fold(f32::MIN, f32::max);
            let shared_in = in_scales.iter().copied().fold(f32::MIN, f32::max);

            let mut shape: Option<Vec<usize>> = None;
            let mut data = Vec::new();
            for (part, &old_w) in parts.iter().zip(&w_scales) {
                let t = tensors
                    .remove(&format!("{part}.weight"))
                    .unwrap_or_else(|| panic!("missing tensor {part}.weight for fp8 fusion"));
                assert_eq!(t.dtype, Dtype::F8_E4M3, "{part} must be e4m3");
                match &mut shape {
                    None => {
                        assert_eq!(t.shape.len(), 2, "{part} must be 2-D");
                        shape = Some(t.shape.clone());
                    }
                    Some(shape) => {
                        assert_eq!(shape[1], t.shape[1], "{part} in_features mismatch");
                        shape[0] += t.shape[0];
                    }
                }
                if old_w == shared_w {
                    data.extend_from_slice(&t.data);
                } else {
                    data.extend(
                        t.data
                            .iter()
                            .map(|&b| f8e4m3_encode(f8e4m3_decode(b) * (old_w / shared_w))),
                    );
                }
            }
            // Keep the scalar scale tensors' original shape for the loader.
            let scale_shape = tensors[&format!("{}.weight_scale", parts[0])].shape.clone();
            for part in parts {
                tensors.remove(&format!("{part}.weight_scale"));
                tensors.remove(&format!("{part}.input_scale"));
            }
            tensors.insert(
                format!("{fused_prefix}.weight"),
                StoredTensor {
                    shape: shape.unwrap(),
                    dtype: Dtype::F8_E4M3,
                    data,
                },
            );
            tensors.insert(
                format!("{fused_prefix}.weight_scale"),
                StoredTensor {
                    shape: scale_shape.clone(),
                    dtype: Dtype::F32,
                    data: shared_w.to_le_bytes().to_vec(),
                },
            );
            tensors.insert(
                format!("{fused_prefix}.input_scale"),
                StoredTensor {
                    shape: scale_shape,
                    dtype: Dtype::F32,
                    data: shared_in.to_le_bytes().to_vec(),
                },
            );
        };

    let mut layer = 0usize;
    loop {
        let prefix = format!("model.layers.{layer}.");
        if !all_tensors.contains_key(&format!("{prefix}self_attn.q_proj.weight")) {
            break;
        }
        fuse(
            all_tensors,
            &[
                format!("{prefix}self_attn.v_proj"),
                format!("{prefix}self_attn.q_proj"),
                format!("{prefix}self_attn.k_proj"),
            ],
            format!("{prefix}self_attn.vqk_proj"),
        );
        fuse(
            all_tensors,
            &[
                format!("{prefix}mlp.gate_proj"),
                format!("{prefix}mlp.up_proj"),
            ],
            format!("{prefix}mlp.gate_up_proj"),
        );
        layer += 1;
    }
    info!("Fused FP8 projections (shared requantized scales) for {layer} layers");
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
        WeightFormat::Bf16 => combine_safetensors_to_bf16(&model_dir)?,
        WeightFormat::Fp8 => combine_safetensors_fp8_bf16(&model_dir)?,
    };
    Ok(PreparedModel {
        model_dir,
        weight_files: vec![weights_path],
    })
}

#[cfg(test)]
mod fp8_codec_tests {
    use super::{f8e4m3_decode, f8e4m3_encode};

    #[test]
    fn e4m3_round_trips_all_codes() {
        for b in 0..=255u8 {
            if b & 0x7F == 0x7F {
                continue; // NaN codes
            }
            let v = f8e4m3_decode(b);
            assert_eq!(
                f8e4m3_encode(v),
                b,
                "code {b:#04x} (value {v}) failed round-trip"
            );
        }
    }

    #[test]
    fn e4m3_rounds_and_saturates() {
        // Midpoint between 20 and 22 rounds to even mantissa (20).
        assert_eq!(f8e4m3_encode(21.0), f8e4m3_encode(20.0));
        // Midpoint between 22 and 24 rounds to 24 (mantissa even after carry).
        assert_eq!(f8e4m3_encode(23.0), f8e4m3_encode(24.0));
        // Saturation at the top of the range.
        assert_eq!(f8e4m3_encode(1e6), f8e4m3_encode(448.0));
        assert_eq!(f8e4m3_encode(-1e6), f8e4m3_encode(-448.0));
        // Below half the smallest subnormal flushes to zero.
        assert_eq!(f8e4m3_decode(f8e4m3_encode(2f32.powi(-11))), 0.0);
        // Smallest subnormal survives.
        assert_eq!(f8e4m3_decode(f8e4m3_encode(2f32.powi(-9))), 2f32.powi(-9));
    }
}
