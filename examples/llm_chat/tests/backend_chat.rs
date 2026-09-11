#![cfg(any(feature = "cuda_lite", feature = "metal"))]
#[cfg(any(not(feature = "metal"), target_os = "macos"))]
use llm_chat::backend::Backend;
use llm_chat::{
    Inputs, TensorData,
    backend::{GpuBackend, harness_search_options},
    graph::{LlmGraph, ModelConfig},
};
#[cfg(any(not(feature = "metal"), target_os = "macos"))]
use luminal::prelude::*;
use model_zoo::llama3::Llama3Dims;

fn fixture() -> (LlmGraph, Inputs) {
    let dims = Llama3Dims {
        vocab: 11,
        hidden: 8,
        intermediate: 12,
        head_dim: 4,
        n_heads: 2,
        n_kv_heads: 1,
        layers: 1,
        rope_theta: 10000.,
        rms_eps: 1e-5,
    };
    let graph = LlmGraph::build(ModelConfig::Llama3(dims), 4, 2).unwrap();
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

#[cfg(all(feature = "metal", not(target_os = "macos")))]
#[test]
fn metal_compiles_the_shared_chat_graph_without_a_device() {
    let (graph, weights) = fixture();
    GpuBackend::compile(&graph, weights, &harness_search_options()).unwrap();
}

#[cfg(any(not(feature = "metal"), target_os = "macos"))]
#[test]
fn prefill_and_decode_use_resident_state_and_match_reference() {
    let (graph, weights) = fixture();
    let mut backend =
        GpuBackend::compile(&graph, weights.clone(), &harness_search_options()).unwrap();
    let mut first_logits = None;
    let mut reference_state = graph.initial_inputs();
    for (tokens, offset) in [(vec![1, 2], 0), (vec![3], 2)] {
        let step = graph.step_inputs(&tokens, offset).unwrap();
        let mut reference = luminal_reference::ReferenceRuntime::load(&graph.graph).unwrap();
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
