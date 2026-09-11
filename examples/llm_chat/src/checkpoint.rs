//! Safetensors names are interpreted only by the application. Runtimes receive
//! resolved graph bindings and never inspect checkpoint names.
use crate::{Inputs, TensorData, graph::ModelType};
use anyhow::{Context, Result, bail, ensure};
use luminal::{graph::InputSpec, prelude::NodeIndex};
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
    pub transpose: bool,
}
impl Parameter {
    pub fn from_namespace(_model: ModelType, input: &InputSpec) -> Result<Self> {
        ensure!(
            input.dtype == luminal::dtype::DType::F32,
            "the chat adapter expects F32 parameter {}",
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
        let values = match tensor.dtype() {
            Dtype::F32 => tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
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
            dtype => bail!(
                "{}: checkpoint dtype {dtype:?} is unsupported by this F32 model adapter",
                self.checkpoint_name
            ),
        };
        if !self.transpose {
            return Ok(TensorData::F32(values));
        }
        let (rows, cols) = (expected[0], expected[1]);
        let mut transposed = vec![0.; values.len()];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = values[r * cols + c];
            }
        }
        Ok(TensorData::F32(transposed))
    }
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
    for p in parameters {
        let shard = index.get(&p.checkpoint_name).ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint index has no {} (model {})",
                p.checkpoint_name,
                p.namespace
            )
        })?;
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
    Ok(inputs)
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
    }
}

#[cfg(test)]
mod shard_tests {
    use super::*;
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
                transpose: true,
            },
            Parameter {
                input: NodeIndex::new(2),
                namespace: "model.embed_tokens.weight".into(),
                checkpoint_name: "model.embed_tokens.weight".into(),
                shape: vec![2, 3],
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
}
