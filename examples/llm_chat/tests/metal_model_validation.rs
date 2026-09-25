//! Small instances of all chat model families, with deterministic weights.
//! These test GPU semantics, not checkpoint loading or language quality.
#![cfg(all(feature = "metal", target_os = "macos"))]
use llm_chat::{
    Inputs, TensorData,
    backend::GpuBackend,
    graph::{LlmGraph, ModelConfig},
};
use luminal::prelude::*;
use luminal_metal::harness_search_options;
use model_zoo::{gemma3::Gemma3Dims, llama3::Llama3Dims, qwen3::QwenDims, qwen3_moe::Qwen3MoeDims};

fn fixture(model: ModelConfig) -> (LlmGraph, Inputs) {
    let graph = LlmGraph::build(model, 6, 2).unwrap();
    let weights: Inputs = graph
        .parameters
        .iter()
        .enumerate()
        .map(|(seed, p)| {
            let n = p.shape.iter().product();
            let values = (0..n)
                .map(|i| {
                    if p.namespace.contains("norm") {
                        1.
                    } else {
                        (((i * 17 + seed * 13) % 31) as f32 - 15.) / 100.
                    }
                })
                .collect();
            (p.input, TensorData::F32(values))
        })
        .collect();
    (graph, weights)
}

fn check_model(model: ModelConfig) {
    let (graph, weights) = fixture(model);
    let mut backend =
        GpuBackend::compile(&graph, weights.clone(), &harness_search_options()).unwrap();
    let mut first_logits = None;
    let mut reference_state = graph.initial_inputs();
    let mut read_back = vec![graph.logits];
    read_back.extend(graph.state.iter().map(|state| state.output));
    for (tokens, offset) in [(vec![1, 2], 0), (vec![3], 2), (vec![4, 5], 3), (vec![6], 5)] {
        let step = graph.step_inputs(&tokens, offset).unwrap();
        let bindings =
            luminal_reference::ReferenceBindings::dense(&graph.graph.logical, &read_back);
        let mut reference =
            luminal_reference::ReferenceRuntime::load_with(&graph.graph, bindings).unwrap();
        reference
            .bind_dyn_range('q', tokens.len() as u64, tokens.len() as u64)
            .unwrap();
        reference
            .bind_dyn_range(
                'c',
                (offset + tokens.len()) as u64,
                (offset + tokens.len()) as u64,
            )
            .unwrap();
        let mut data = weights.clone();
        data.extend(reference_state.clone());
        data.extend(step.clone());
        let data: FxHashMap<_, luminal_reference::TypedBuffer> = data
            .into_iter()
            .map(|(id, v)| {
                (
                    id,
                    match v {
                        TensorData::F32(v) => v.into(),
                        TensorData::I32(v) => v.into(),
                        TensorData::BF16(_) | TensorData::F16(_) => {
                            panic!("the reference fixture is explicitly F32")
                        }
                    },
                )
            })
            .collect();
        reference
            .search(&data, &luminal_reference::harness_search_options())
            .unwrap();
        for (id, data) in data {
            reference.set_data(id, data);
        }
        reference.execute().unwrap();
        let expected = reference.get_f32(graph.logits).unwrap();
        let actual = backend
            .step(step, tokens.len(), offset + tokens.len())
            .unwrap();
        assert_eq!(actual.len(), expected.len());
        if first_logits.is_none() {
            first_logits = Some(expected.clone());
        }
        for (&a, &b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-4, "GPU {a} != reference {b}");
        }
        for state in &graph.state {
            reference_state.insert(
                state.input,
                TensorData::F32(reference.get_f32(state.output).unwrap().clone()),
            );
        }
    }
    backend.reset().unwrap();
    let restarted = backend
        .step(graph.step_inputs(&[1, 2], 0).unwrap(), 2, 2)
        .unwrap();
    let first = first_logits.unwrap();
    assert_eq!(restarted.len(), first.len());
    for (&a, &b) in restarted.iter().zip(&first) {
        assert!(
            (a - b).abs() < 1e-4,
            "reset GPU {a} != initial reference {b}"
        );
    }
}

#[test]
fn llama3_metal() {
    check_model(ModelConfig::Llama3(Llama3Dims {
        vocab: 31,
        hidden: 16,
        intermediate: 24,
        head_dim: 4,
        n_heads: 4,
        n_kv_heads: 2,
        layers: 2,
        rope_theta: 10000.,
        rms_eps: 1e-5,
    }));
}
#[test]
fn qwen3_metal() {
    check_model(ModelConfig::Qwen3(QwenDims::tiny()));
}
#[test]
fn gemma3_metal() {
    check_model(ModelConfig::Gemma3(Gemma3Dims::tiny()));
}
#[test]
fn qwen3_moe_metal() {
    check_model(ModelConfig::Qwen3Moe(Qwen3MoeDims {
        vocab: 31,
        hidden: 16,
        moe_intermediate: 24,
        head_dim: 4,
        n_heads: 4,
        n_kv_heads: 2,
        layers: 2,
        experts: 4,
        top_k: 2,
        rope_theta: 10000.,
        rms_eps: 1e-6,
    }));
}
