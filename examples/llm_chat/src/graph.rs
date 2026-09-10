//! Application-owned adapters around the zoo's logical model APIs.
use crate::{Inputs, TensorData, checkpoint::Parameter};
use anyhow::{Result, ensure};
use clap::ValueEnum;
use luminal::prelude::*;
use luminal_nn::{rope_pairing_matrix, rope_tables_split_half};
use model_zoo::{
    gemma3::{Gemma3, Gemma3Dims},
    llama3::{Llama3, Llama3Dims},
    model_support::{Namespace, named_heterogeneous_kv_cache_pool},
    qwen3::{Qwen, QwenDims},
    qwen3_moe::{Qwen3Moe, Qwen3MoeDims},
};
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ModelType {
    Llama3,
    Qwen3,
    Gemma3,
    Qwen3Moe,
}

pub enum ModelConfig {
    Llama3(Llama3Dims),
    Qwen3(QwenDims),
    Gemma3(Gemma3Dims),
    Qwen3Moe(Qwen3MoeDims),
}

fn number(config: &Value, key: &str) -> Result<usize> {
    let n = config[key]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("config.json: missing integer {key}"))?;
    let n = usize::try_from(n)?;
    ensure!(n > 0, "config.json: {key} must be positive");
    Ok(n)
}
fn real(config: &Value, key: &str) -> Result<f32> {
    let n = config[key]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("config.json: missing number {key}"))? as f32;
    ensure!(n.is_finite() && n > 0., "config.json: invalid {key}");
    Ok(n)
}
fn equal(config: &Value, key: &str, expected: usize) -> Result<()> {
    ensure!(
        number(config, key)? == expected,
        "this zoo definition requires {key}={expected}"
    );
    Ok(())
}

impl ModelConfig {
    pub fn from_checkpoint(kind: ModelType, root: &Value) -> Result<Self> {
        let config = root.get("text_config").unwrap_or(root);
        let expected = match kind {
            ModelType::Llama3 => "llama",
            ModelType::Qwen3 => "qwen3",
            ModelType::Gemma3 => "gemma3_text",
            ModelType::Qwen3Moe => "qwen3_moe",
        };
        ensure!(
            config["model_type"].as_str() == Some(expected),
            "selected {kind:?} requires model_type={expected}"
        );
        ensure!(
            config.get("quantization_config").is_none()
                && root.get("quantization_config").is_none(),
            "quantized checkpoints require a matching zoo/backend adapter"
        );
        let hidden = number(config, "hidden_size")?;
        let heads = number(config, "num_attention_heads")?;
        let kv_heads = number(config, "num_key_value_heads")?;
        ensure!(
            heads.is_multiple_of(kv_heads),
            "attention heads must be divisible by KV heads"
        );
        let head_dim = config
            .get("head_dim")
            .map(|_| number(config, "head_dim"))
            .transpose()?
            .unwrap_or(hidden / heads);
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(2),
            "RoPE requires an even head dimension"
        );
        let common = (
            number(config, "vocab_size")?,
            number(config, "num_hidden_layers")?,
        );
        match kind {
            ModelType::Llama3 | ModelType::Qwen3 => {
                ensure!(
                    config.get("rope_scaling").is_none_or(Value::is_null),
                    "this zoo adapter uses unscaled RoPE; select a matching model for scaled RoPE"
                );
                ensure!(
                    config["hidden_act"] == "silu",
                    "expected SwiGLU (hidden_act=silu)"
                );
                ensure!(
                    !config["attention_bias"].as_bool().unwrap_or(false)
                        && !config["mlp_bias"].as_bool().unwrap_or(false),
                    "the selected zoo model has bias-free projections"
                );
                ensure!(
                    !config["use_sliding_window"].as_bool().unwrap_or(false),
                    "this adapter uses full causal attention"
                );
                ensure!(
                    config["partial_rotary_factor"].as_f64().unwrap_or(1.) == 1.,
                    "this adapter requires full rotary embeddings"
                );
                let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
                let theta = real(config, "rope_theta")?;
                let eps = real(config, "rms_norm_eps")?;
                let intermediate = number(config, "intermediate_size")?;
                if kind == ModelType::Llama3 {
                    ensure!(!tied, "the Llama3 zoo definition has an untied lm_head");
                    ensure!(
                        hidden == heads * head_dim,
                        "Llama3 hidden/head dimensions disagree"
                    );
                    Ok(Self::Llama3(Llama3Dims {
                        vocab: common.0,
                        hidden,
                        intermediate,
                        head_dim,
                        n_heads: heads,
                        n_kv_heads: kv_heads,
                        layers: common.1,
                        rope_theta: theta,
                        rms_eps: eps,
                    }))
                } else {
                    ensure!(tied, "the Qwen3 zoo definition requires tied embeddings");
                    ensure!(
                        (eps - 1e-6).abs() < 1e-12,
                        "Qwen3's zoo Q/K norms require rms_norm_eps=1e-6"
                    );
                    Ok(Self::Qwen3(QwenDims {
                        vocab: common.0,
                        hidden,
                        intermediate,
                        head_dim,
                        n_heads: heads,
                        n_kv_heads: kv_heads,
                        layers: common.1,
                        rope_theta: theta,
                        rms_eps: eps,
                    }))
                }
            }
            ModelType::Gemma3 => {
                let d = Gemma3Dims::gemma3_4b();
                for (key, value) in [
                    ("hidden_size", d.hidden),
                    ("vocab_size", d.vocab),
                    ("num_hidden_layers", d.layers),
                    ("intermediate_size", d.intermediate),
                    ("num_attention_heads", d.n_heads),
                    ("num_key_value_heads", d.n_kv_heads),
                    ("head_dim", d.head_dim),
                    ("sliding_window", d.window),
                ] {
                    equal(config, key, value)?;
                }
                ensure!(
                    config["tie_word_embeddings"]
                        .as_bool()
                        .or_else(|| root["tie_word_embeddings"].as_bool())
                        // Gemma3TextConfig defaults to tied embeddings. The
                        // published 4B checkpoint omits this default field.
                        .unwrap_or(true),
                    "Gemma3 requires tied embeddings"
                );
                ensure!(
                    real(config, "rope_theta")? == 1_000_000.0
                        && real(config, "rope_local_base_freq")? == 10_000.0
                        && config["rope_scaling"]["factor"].as_f64() == Some(8.0),
                    "Gemma3 requires the zoo's dual-theta, /8 global RoPE configuration"
                );
                ensure!(
                    config["rope_scaling"]["rope_type"]
                        .as_str()
                        .or_else(|| config["rope_scaling"]["type"].as_str())
                        == Some("linear"),
                    "Gemma3 requires linear global RoPE scaling"
                );
                ensure!(
                    real(config, "rms_norm_eps")? == d.rms_eps
                        && number(config, "query_pre_attn_scalar")? == d.head_dim,
                    "Gemma3 norm/attention scaling differs from the zoo definition"
                );
                ensure!(
                    config["sliding_window_pattern"].as_u64().unwrap_or(6) == 6
                        && config["hidden_activation"] == "gelu_pytorch_tanh",
                    "Gemma3 layer pattern/activation differs from the zoo definition"
                );
                Ok(Self::Gemma3(d))
            }
            ModelType::Qwen3Moe => {
                let d = Qwen3MoeDims::qwen3_30b_a3b();
                for (key, value) in [
                    ("hidden_size", d.hidden),
                    ("vocab_size", d.vocab),
                    ("num_hidden_layers", d.layers),
                    ("moe_intermediate_size", d.moe_intermediate),
                    ("num_attention_heads", d.n_heads),
                    ("num_key_value_heads", d.n_kv_heads),
                    ("head_dim", d.head_dim),
                    ("num_experts", d.experts),
                    ("num_experts_per_tok", d.top_k),
                ] {
                    equal(config, key, value)?;
                }
                ensure!(
                    config.get("rope_scaling").is_none_or(Value::is_null)
                        && real(config, "rope_theta")? == d.rope_theta,
                    "Qwen3 MoE RoPE differs from the zoo definition"
                );
                ensure!(
                    !config["tie_word_embeddings"].as_bool().unwrap_or(false),
                    "Qwen3 MoE requires an untied output head"
                );
                ensure!(
                    real(config, "rms_norm_eps")? == d.rms_eps && config["hidden_act"] == "silu",
                    "Qwen3 MoE norm/activation differs from the zoo definition"
                );
                ensure!(
                    config["norm_topk_prob"].as_bool() == Some(true)
                        && config["decoder_sparse_step"].as_u64().unwrap_or(1) == 1
                        && config["mlp_only_layers"]
                            .as_array()
                            .is_none_or(Vec::is_empty),
                    "Qwen3 MoE routing/layer pattern differs from the zoo definition"
                );
                Ok(Self::Qwen3Moe(d))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct StateBinding {
    pub input: NodeIndex,
    pub output: NodeIndex,
    pub elements: usize,
}
struct RopeInputs {
    cos: GraphTensor,
    sin: GraphTensor,
    rot: GraphTensor,
    width: usize,
    theta: f32,
    scale: f32,
}
impl RopeInputs {
    fn new(
        cx: &mut Graph,
        width: usize,
        theta: f32,
        scale: f32,
        rotation: Option<GraphTensor>,
    ) -> Self {
        Self {
            cos: cx.tensor(('q', width), DType::F32),
            sin: cx.tensor(('q', width), DType::F32),
            rot: rotation.unwrap_or_else(|| cx.tensor((width, width), DType::F32)),
            width,
            theta,
            scale,
        }
    }
}

/// A common execution contract owned by this application, not by the model zoo.
pub struct LlmGraph {
    pub graph: Graph,
    pub parameters: Vec<Parameter>,
    pub state: Vec<StateBinding>,
    pub logits: NodeIndex,
    pub vocab: usize,
    pub capacity: usize,
    pub chunk_size: usize,
    tokens: GraphTensor,
    positions: GraphTensor,
    gather: GraphTensor,
    scatter: GraphTensor,
    last: GraphTensor,
    ropes: Vec<RopeInputs>,
}
impl LlmGraph {
    pub fn build(config: ModelConfig, capacity: usize, chunk_size: usize) -> Result<Self> {
        ensure!(
            capacity > 0
                && capacity <= i32::MAX as usize
                && chunk_size > 0
                && chunk_size <= capacity,
            "require 1 <= prefill-chunk <= max-context <= i32::MAX"
        );
        let mut cx = Graph::new();
        let tokens = cx.tensor('q', DType::Int);
        let positions = cx.tensor('q', DType::Int);
        let gather = cx.tensor('c', DType::Int);
        let scatter = cx.tensor('q', DType::Int);
        let last = cx.tensor(1, DType::Int);
        let (kind, vocab, widths, roles) = match &config {
            ModelConfig::Llama3(d) => (
                ModelType::Llama3,
                d.vocab,
                vec![d.kv_dim(); d.layers],
                vec![(d.head_dim, d.rope_theta, 1.)],
            ),
            ModelConfig::Qwen3(d) => (
                ModelType::Qwen3,
                d.vocab,
                vec![d.kv_dim(); d.layers],
                vec![(d.head_dim, d.rope_theta, 1.)],
            ),
            ModelConfig::Gemma3(d) => (
                ModelType::Gemma3,
                d.vocab,
                vec![d.kv_dim(); d.layers],
                vec![(d.head_dim, 10_000., 1.), (d.head_dim, 1_000_000., 1. / 8.)],
            ),
            ModelConfig::Qwen3Moe(d) => (
                ModelType::Qwen3Moe,
                d.vocab,
                vec![d.kv_dim(); d.layers],
                vec![(d.head_dim, d.rope_theta, 1.)],
            ),
        };
        let mut ropes: Vec<RopeInputs> = vec![];
        for (width, theta, scale) in roles {
            let rotation = ropes.iter().find(|r| r.width == width).map(|r| r.rot);
            ropes.push(RopeInputs::new(&mut cx, width, theta, scale, rotation));
        }
        let pool = named_heterogeneous_kv_cache_pool(
            &mut cx,
            capacity,
            &widths,
            DType::F32,
            &Namespace::root().child("chat_cache"),
        );
        let r = &ropes[0];
        let (logits, caches) = match config {
            ModelConfig::Llama3(d) => Llama3::init(&mut cx, &d).forward(
                tokens, positions, r.cos, r.sin, r.rot, &pool, gather, scatter,
            ),
            ModelConfig::Qwen3(d) => Qwen::init(&mut cx, &d).forward(
                tokens,
                positions,
                r.cos,
                r.sin,
                r.rot,
                &pool.layers,
                gather,
                scatter,
            ),
            ModelConfig::Gemma3(d) => Gemma3::init(&mut cx, &d).forward(
                tokens,
                positions,
                (r.cos, r.sin),
                (ropes[1].cos, ropes[1].sin),
                r.rot,
                &pool,
                gather,
                scatter,
            ),
            ModelConfig::Qwen3Moe(d) => Qwen3Moe::init(&mut cx, &d).forward(
                tokens, positions, r.cos, r.sin, r.rot, &pool, gather, scatter,
            ),
        };
        let logits = luminal_nn::gather_rows(logits, last).output().id;
        let state: Vec<_> = pool
            .layers
            .iter()
            .zip(caches)
            .zip(widths)
            .flat_map(|(((ki, vi), (ko, vo)), w)| {
                [
                    StateBinding {
                        input: ki.id,
                        output: ko.output().id,
                        elements: capacity * w,
                    },
                    StateBinding {
                        input: vi.id,
                        output: vo.output().id,
                        elements: capacity * w,
                    },
                ]
            })
            .collect();
        let mut app_inputs = vec![tokens.id, positions.id, gather.id, scatter.id, last.id];
        app_inputs.extend(state.iter().map(|s| s.input));
        app_inputs.extend(ropes.iter().flat_map(|r| [r.cos.id, r.sin.id, r.rot.id]));
        let parameters = cx
            .logical
            .input_specs()
            .into_iter()
            .filter(|s| !app_inputs.contains(&s.id))
            .map(|s| Parameter::from_namespace(kind, &s))
            .collect::<Result<_>>()?;
        Ok(Self {
            graph: cx,
            parameters,
            state,
            logits,
            vocab,
            capacity,
            chunk_size,
            tokens,
            positions,
            gather,
            scatter,
            last,
            ropes,
        })
    }
    pub fn initial_inputs(&self) -> Inputs {
        let mut out = Inputs::default();
        for s in &self.state {
            out.insert(s.input, TensorData::F32(vec![0.; s.elements]));
        }
        for r in &self.ropes {
            out.entry(r.rot.id)
                .or_insert_with(|| TensorData::F32(rope_pairing_matrix(r.width, false)));
        }
        out
    }
    pub fn step_inputs(&self, tokens: &[u32], offset: usize) -> Result<Inputs> {
        let end = offset
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context overflow"))?;
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.chunk_size && end <= self.capacity,
            "step exceeds configured query/context bounds"
        );
        ensure!(
            tokens
                .iter()
                .all(|&t| (t as usize) < self.vocab && t <= i32::MAX as u32),
            "token ID outside model vocabulary"
        );
        let pos: Vec<i32> = (offset..end).map(|p| p as i32).collect();
        let mut out: Inputs = [
            (
                self.tokens.id,
                TensorData::I32(tokens.iter().map(|&t| t as i32).collect()),
            ),
            (self.positions.id, TensorData::I32(pos.clone())),
            (self.scatter.id, TensorData::I32(pos)),
            (self.gather.id, TensorData::I32((0..end as i32).collect())),
            (self.last.id, TensorData::I32(vec![tokens.len() as i32 - 1])),
        ]
        .into_iter()
        .collect();
        let positions: Vec<f32> = (offset..end).map(|p| p as f32).collect();
        for r in &self.ropes {
            let (cos, sin) = rope_tables_split_half(&positions, r.width, r.theta, r.scale);
            out.insert(r.cos.id, TensorData::F32(cos));
            out.insert(r.sin.id, TensorData::F32(sin));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adapters_keep_parameter_mapping_and_runtime_inputs_separate() {
        for config in [
            ModelConfig::Llama3(Llama3Dims::tiny()),
            ModelConfig::Qwen3(QwenDims::tiny()),
            ModelConfig::Gemma3(Gemma3Dims::tiny()),
        ] {
            let graph = LlmGraph::build(config, 8, 2).unwrap();
            let runtime = graph.initial_inputs();
            let step = graph.step_inputs(&[1, 2], 3).unwrap();
            assert!(
                graph
                    .parameters
                    .iter()
                    .all(|p| !runtime.contains_key(&p.input) && !step.contains_key(&p.input))
            );
            assert_eq!(
                graph.graph.logical.input_specs().len(),
                graph.parameters.len() + runtime.len() + step.len()
            );
            assert!(graph.parameters.iter().any(|p| p.transpose));
            assert!(
                graph
                    .parameters
                    .iter()
                    .any(|p| p.namespace.ends_with("embed_tokens.weight") && !p.transpose)
            );
            assert!(graph.step_inputs(&[1, 2], 7).is_err());
        }
    }
    #[test]
    fn incompatible_checkpoint_architecture_is_rejected_before_loading() {
        let config = serde_json::json!({"model_type":"llama","hidden_size":16,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":29,"num_hidden_layers":2,"hidden_act":"silu","intermediate_size":24,"rope_theta":10000.,"rms_norm_eps":1e-5,"rope_scaling":{"factor":8.}});
        assert!(ModelConfig::from_checkpoint(ModelType::Llama3, &config).is_err());
        assert!(ModelConfig::from_checkpoint(ModelType::Qwen3, &config).is_err());
    }

    #[test]
    fn gemma_checkpoint_uses_its_default_tied_embeddings_and_rejects_untied() {
        let mut root = serde_json::json!({"model_type":"gemma3", "text_config": {
            "model_type":"gemma3_text", "hidden_size":2560, "vocab_size":262208,
            "num_hidden_layers":34, "intermediate_size":10240,
            "num_attention_heads":8, "num_key_value_heads":4, "head_dim":256,
            "sliding_window":1024, "rope_theta":1000000., "rope_local_base_freq":10000.,
            "rope_scaling":{"factor":8., "rope_type":"linear"}, "rms_norm_eps":1e-6,
            "query_pre_attn_scalar":256, "hidden_activation":"gelu_pytorch_tanh"
        }});
        assert!(ModelConfig::from_checkpoint(ModelType::Gemma3, &root).is_ok());
        root["tie_word_embeddings"] = false.into();
        assert!(ModelConfig::from_checkpoint(ModelType::Gemma3, &root).is_err());
        root["tie_word_embeddings"] = true.into();
        root["text_config"]["tie_word_embeddings"] = false.into();
        assert!(ModelConfig::from_checkpoint(ModelType::Gemma3, &root).is_err());
    }
}
