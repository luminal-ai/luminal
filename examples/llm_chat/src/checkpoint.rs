//! Safetensors names are interpreted only by the application. Runtimes receive
//! resolved graph bindings and never inspect checkpoint names.
use crate::{Inputs, TensorData, graph::ModelType};
use anyhow::{Context, Result, bail, ensure};
use luminal::{dtype::DType, graph::InputSpec, prelude::NodeIndex};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    path::{Component, Path},
};

#[derive(Debug, Clone)]
pub struct Parameter {
    pub input: NodeIndex,
    pub namespace: String,
    pub checkpoint_name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub transpose: bool,
}
impl Parameter {
    pub fn from_namespace(_model: ModelType, input: &InputSpec) -> Result<Self> {
        ensure!(
            matches!(input.dtype, DType::F32 | DType::F16 | DType::Bf16),
            "the chat adapter does not support {:?} parameter {}",
            input.dtype,
            input.label
        );
        let namespace = input.label.clone();
        let shape = input
            .dims
            .iter()
            .map(|d| {
                d.to_usize()
                    .ok_or_else(|| anyhow::anyhow!("parameter {namespace} has dynamic dimensions"))
            })
            .collect::<Result<Vec<_>>>()?;
        // Explicit conventions of the currently registered zoo adapters. Expert
        // matrices are authored in checkpoint order; ordinary Linear weights
        // are authored as [in, out]. Embeddings and norm vectors keep their order.
        let transpose = !namespace.contains(".experts.")
            && [
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
                "down_proj.weight",
                "lm_head.weight",
                "mlp.gate.weight",
                "router.proj.weight",
            ]
            .iter()
            .any(|suffix| namespace.ends_with(suffix));
        ensure!(
            !transpose || shape.len() == 2,
            "linear parameter {namespace} must be a matrix"
        );
        Ok(Self {
            input: input.id,
            checkpoint_name: namespace.clone(),
            namespace,
            shape,
            dtype: input.dtype,
            transpose,
        })
    }
    pub fn decode(&self, tensor: &TensorView<'_>) -> Result<TensorData> {
        let expected: Vec<_> = if self.transpose {
            self.shape.iter().copied().rev().collect()
        } else {
            self.shape.clone()
        };
        ensure!(
            tensor.shape() == expected,
            "{} -> {}: checkpoint shape {:?}, expected {:?}",
            self.checkpoint_name,
            self.namespace,
            tensor.shape(),
            expected
        );
        let values = decode_f32(&self.checkpoint_name, tensor)?;
        if !self.transpose {
            return TensorData::from_f32(self.dtype, values);
        }
        let (rows, cols) = (expected[0], expected[1]);
        let mut transposed = vec![0.; values.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = values[r * cols + c];
            }
        }
        TensorData::from_f32(self.dtype, transposed)
    }
}

fn decode_f32(name: &str, tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    Ok(match tensor.dtype() {
        Dtype::F32 => tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        Dtype::BF16 => tensor
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect(),
        dtype => {
            bail!("{name}: checkpoint dtype {dtype:?} is unsupported by this chat adapter")
        }
    })
}

pub fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_reader(File::open(path).with_context(|| format!("open {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

/// Optional JSON object: {"safetensors.name": "model.namespace"}. Mappings
/// override the adapter's defaults; unknown/duplicate destinations are errors.
pub fn apply_name_map(
    parameters: &mut [Parameter],
    mappings: &BTreeMap<String, String>,
) -> Result<()> {
    let mut seen = BTreeSet::new();
    for (source, destination) in mappings {
        ensure!(
            seen.insert(destination),
            "multiple checkpoint tensors map to {destination}"
        );
        let p = parameters
            .iter_mut()
            .find(|p| p.namespace == *destination)
            .ok_or_else(|| anyhow::anyhow!("unknown model namespace {destination}"))?;
        p.checkpoint_name = source.clone();
    }
    Ok(())
}

pub fn load(directory: &Path, parameters: &[Parameter]) -> Result<Inputs> {
    let index_path = directory.join("model.safetensors.index.json");
    let index: BTreeMap<String, String> = if index_path.exists() {
        serde_json::from_value(read_json(&index_path)?["weight_map"].clone())
            .context("safetensors index weight_map")?
    } else {
        parameters
            .iter()
            .map(|p| (p.checkpoint_name.clone(), "model.safetensors".to_owned()))
            .collect()
    };
    let mut shards: BTreeMap<&str, Vec<&Parameter>> = BTreeMap::new();
    let mut split_expert_banks = Vec::new();
    for p in parameters {
        let Some(shard) = index.get(&p.checkpoint_name) else {
            if p.namespace.ends_with(".experts.gate_up_proj")
                || p.namespace.ends_with(".experts.down_proj")
            {
                split_expert_banks.push(p);
                continue;
            }
            bail!(
                "checkpoint index has no {} (model {})",
                p.checkpoint_name,
                p.namespace
            );
        };
        ensure!(
            Path::new(shard)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
            "invalid checkpoint shard path {shard}"
        );
        shards.entry(shard).or_default().push(p);
    }
    let mut inputs = Inputs::default();
    for (shard, parameters) in shards {
        let path = directory.join(shard);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: checkpoint files must remain unchanged while this process
        // loads them. The read-only map is dropped before loading the next shard.
        let mapped = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&mapped).with_context(|| format!("read {shard}"))?;
        for p in parameters {
            let tensor = tensors
                .tensor(&p.checkpoint_name)
                .with_context(|| format!("{} -> {}", p.checkpoint_name, p.namespace))?;
            inputs.insert(p.input, p.decode(&tensor)?);
        }
    }
    for p in split_expert_banks {
        inputs.insert(p.input, load_split_expert_bank(directory, &index, p)?);
    }
    Ok(inputs)
}

/// Stack Qwen's per-expert checkpoint matrices into the rank-3 banks used by
/// the logical model. This is a layout conversion only; expert and row order
/// are preserved exactly.
fn load_split_expert_bank(
    directory: &Path,
    index: &BTreeMap<String, String>,
    parameter: &Parameter,
) -> Result<TensorData> {
    ensure!(parameter.shape.len() == 3, "expert bank must be rank 3");
    let experts = parameter.shape[0];
    let gate_up = parameter.namespace.ends_with(".experts.gate_up_proj");
    let suffix = if gate_up { "gate_up_proj" } else { "down_proj" };
    let prefix = parameter
        .checkpoint_name
        .strip_suffix(suffix)
        .ok_or_else(|| anyhow::anyhow!("invalid expert bank name {}", parameter.checkpoint_name))?;
    let kinds: &[&str] = if gate_up {
        &["gate_proj.weight", "up_proj.weight"]
    } else {
        &["down_proj.weight"]
    };
    let rows_per_part = if gate_up {
        parameter.shape[1] / 2
    } else {
        parameter.shape[1]
    };
    ensure!(
        !gate_up || parameter.shape[1].is_multiple_of(2),
        "gate/up bank rows must be even"
    );
    let cols = parameter.shape[2];
    let part_elements = rows_per_part * cols;
    let expert_elements = parameter.shape[1] * cols;
    let mut requests: BTreeMap<&str, Vec<(String, usize)>> = BTreeMap::new();
    for expert in 0..experts {
        for (part, kind) in kinds.iter().enumerate() {
            let name = format!("{prefix}{expert}.{kind}");
            let shard = index.get(&name).ok_or_else(|| {
                anyhow::anyhow!("checkpoint index has no {name} for {}", parameter.namespace)
            })?;
            ensure!(
                Path::new(shard)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
                "invalid checkpoint shard path {shard}"
            );
            requests
                .entry(shard)
                .or_default()
                .push((name, expert * expert_elements + part * part_elements));
        }
    }
    let mut values = vec![0.; experts * expert_elements];
    for (shard, tensors_in_shard) in requests {
        let path = directory.join(shard);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: see the ordinary shard loader above.
        let mapped = unsafe { MmapOptions::new().map(&file)? };
        let tensors = SafeTensors::deserialize(&mapped).with_context(|| format!("read {shard}"))?;
        for (name, offset) in tensors_in_shard {
            let tensor = tensors.tensor(&name).with_context(|| name.clone())?;
            ensure!(
                tensor.shape() == [rows_per_part, cols],
                "{name}: checkpoint shape {:?}, expected {:?}",
                tensor.shape(),
                [rows_per_part, cols]
            );
            let part = decode_f32(&name, &tensor)?;
            values[offset..offset + part_elements].copy_from_slice(&part);
        }
    }
    TensorData::from_f32(parameter.dtype, values)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn param(transpose: bool) -> Parameter {
        Parameter {
            input: NodeIndex::new(0),
            namespace: "model.proj.weight".into(),
            checkpoint_name: "checkpoint.w".into(),
            shape: vec![2, 3],
            dtype: DType::F32,
            transpose,
        }
    }
    #[test]
    fn checkpoint_mapping_and_non_square_bf16_transpose() {
        let mut parameters = vec![param(true)];
        apply_name_map(
            &mut parameters,
            &[("other.w".into(), "model.proj.weight".into())].into(),
        )
        .unwrap();
        assert_eq!(parameters[0].checkpoint_name, "other.w");
        let bytes: Vec<u8> = [1., 2., 3., 4., 5., 6.]
            .into_iter()
            .flat_map(|x| half::bf16::from_f32(x).to_bits().to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::BF16, vec![3, 2], &bytes).unwrap();
        let TensorData::F32(values) = parameters[0].decode(&view).unwrap() else {
            panic!()
        };
        assert_eq!(values, vec![1., 3., 5., 2., 4., 6.]);
        assert!(param(false).decode(&view).is_err());
        assert!(apply_name_map(&mut parameters, &[("x".into(), "unknown".into())].into()).is_err());

        let mut native = param(false);
        native.dtype = DType::Bf16;
        let f32_bytes: Vec<_> = [1f32, 2., 3., 4., 5., 6.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let view = TensorView::new(Dtype::F32, vec![2, 3], &f32_bytes).unwrap();
        let TensorData::BF16(values) = native.decode(&view).unwrap() else {
            panic!()
        };
        assert_eq!(
            values,
            (1..=6)
                .map(|x| half::bf16::from_f32(x as f32).to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[cfg(test)]
mod shard_tests {
    use super::*;
    fn f32_view(shape: Vec<usize>, values: &[f32]) -> TensorView<'static> {
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>()
            .leak();
        TensorView::new(Dtype::F32, shape, bytes).unwrap()
    }

    #[test]
    fn indexed_shards_load_through_application_name_mapping() {
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<_> = [1f32, 2., 3., 4., 5., 6.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let view = TensorView::new(Dtype::F32, vec![3, 2], &bytes).unwrap();
        std::fs::write(
            dir.path().join("a.safetensors"),
            safetensors::serialize([("vendor.linear", view)], None).unwrap(),
        )
        .unwrap();
        let view = TensorView::new(Dtype::F32, vec![2, 3], &bytes).unwrap();
        std::fs::write(
            dir.path().join("b.safetensors"),
            safetensors::serialize([("vendor.embed", view)], None).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("model.safetensors.index.json"),serde_json::json!({"weight_map":{"vendor.linear":"a.safetensors","vendor.embed":"b.safetensors"}}).to_string()).unwrap();
        let mut parameters = vec![
            Parameter {
                input: NodeIndex::new(1),
                namespace: "model.proj.weight".into(),
                checkpoint_name: "model.proj.weight".into(),
                shape: vec![2, 3],
                dtype: DType::F32,
                transpose: true,
            },
            Parameter {
                input: NodeIndex::new(2),
                namespace: "model.embed_tokens.weight".into(),
                checkpoint_name: "model.embed_tokens.weight".into(),
                shape: vec![2, 3],
                dtype: DType::F32,
                transpose: false,
            },
        ];
        apply_name_map(
            &mut parameters,
            &[
                ("vendor.linear".into(), "model.proj.weight".into()),
                ("vendor.embed".into(), "model.embed_tokens.weight".into()),
            ]
            .into(),
        )
        .unwrap();
        let loaded = load(dir.path(), &parameters).unwrap();
        let TensorData::F32(linear) = &loaded[&NodeIndex::new(1)] else {
            panic!()
        };
        let TensorData::F32(embed) = &loaded[&NodeIndex::new(2)] else {
            panic!()
        };
        assert_eq!(linear, &[1., 3., 5., 2., 4., 6.]);
        assert_eq!(embed, &[1., 2., 3., 4., 5., 6.]);
        parameters[0].checkpoint_name = "missing".into();
        assert!(load(dir.path(), &parameters).is_err());
    }

    #[test]
    fn split_qwen_experts_are_stacked_in_expert_gate_up_order() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = [
            (
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                f32_view(vec![2, 3], &[1., 2., 3., 4., 5., 6.]),
            ),
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                f32_view(vec![2, 3], &[7., 8., 9., 10., 11., 12.]),
            ),
            (
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                f32_view(vec![2, 3], &[13., 14., 15., 16., 17., 18.]),
            ),
            (
                "model.layers.0.mlp.experts.1.up_proj.weight",
                f32_view(vec![2, 3], &[19., 20., 21., 22., 23., 24.]),
            ),
            (
                "model.layers.0.mlp.experts.0.down_proj.weight",
                f32_view(vec![3, 2], &[25., 26., 27., 28., 29., 30.]),
            ),
            (
                "model.layers.0.mlp.experts.1.down_proj.weight",
                f32_view(vec![3, 2], &[31., 32., 33., 34., 35., 36.]),
            ),
        ];
        let weight_map = tensors
            .iter()
            .map(|(name, _)| (*name, "experts.safetensors"))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            dir.path().join("experts.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": weight_map}).to_string(),
        )
        .unwrap();
        let parameters = [
            Parameter {
                input: NodeIndex::new(1),
                namespace: "model.layers.0.mlp.experts.gate_up_proj".into(),
                checkpoint_name: "model.layers.0.mlp.experts.gate_up_proj".into(),
                shape: vec![2, 4, 3],
                dtype: DType::F32,
                transpose: false,
            },
            Parameter {
                input: NodeIndex::new(2),
                namespace: "model.layers.0.mlp.experts.down_proj".into(),
                checkpoint_name: "model.layers.0.mlp.experts.down_proj".into(),
                shape: vec![2, 3, 2],
                dtype: DType::F32,
                transpose: false,
            },
        ];
        let loaded = load(dir.path(), &parameters).unwrap();
        let TensorData::F32(gate_up) = &loaded[&NodeIndex::new(1)] else {
            panic!()
        };
        let TensorData::F32(down) = &loaded[&NodeIndex::new(2)] else {
            panic!()
        };
        assert_eq!(gate_up, &(1..=24).map(|x| x as f32).collect::<Vec<_>>());
        assert_eq!(down, &(25..=36).map(|x| x as f32).collect::<Vec<_>>());
    }
}
