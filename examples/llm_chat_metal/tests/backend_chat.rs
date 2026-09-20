//! The Metal chat backend, against the reference runtime.
use llm_chat_metal::{
    Inputs, TensorData,
    backend::MetalBackend,
    graph::{LlmGraph, ModelConfig},
};
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

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_compiles_the_shared_chat_graph_without_a_device() {
    let (graph, weights) = fixture();
    MetalBackend::compile(&graph, weights, &luminal_metal::harness_search_options()).unwrap();
}

#[cfg(target_os = "macos")]
#[test]
fn prefill_and_decode_use_resident_state_and_match_reference() {
    use luminal::prelude::*;
    let (graph, weights) = fixture();
    let mut backend = MetalBackend::compile(
        &graph,
        weights.clone(),
        &luminal_metal::harness_search_options(),
    )
    .unwrap();
    // The reference reads back what this comparison needs: the logits and
    // every cache output. Its own state inputs are ordinary inputs, fed the
    // previous step's cache outputs — no aliasing, so the reference's
    // arithmetic is what is compared, not a runtime's storage policy.
    let mut read_back = vec![graph.logits];
    read_back.extend(graph.state.iter().map(|s| s.output));
    let mut first_logits = None;
    let mut reference_state = graph.initial_inputs();
    for (tokens, offset) in [(vec![1, 2], 0), (vec![3], 2)] {
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

/// What the chat loop does with the cache: prefill in chunks, decode one
/// token at a time, refuse a prompt that fills the context, and rebuild
/// the cache whenever the prompt is not an extension of it.
#[cfg(target_os = "macos")]
#[test]
fn chunked_prefill_history_reuse_and_reset_drive_one_session() {
    use llm_chat_metal::{sampling::Sampler, session::Session};
    use std::collections::BTreeSet;
    let (graph, weights) = fixture();
    let backend =
        MetalBackend::compile(&graph, weights, &luminal_metal::harness_search_options()).unwrap();
    let mut session = Session::new(graph, backend);
    let mut sampler = Sampler::new(0., 1., 0).unwrap();
    let stops = BTreeSet::new();
    // Three prompt tokens at a prefill chunk of two: two prefill steps,
    // then one decode step.
    let first = session
        .generate(&[1, 2, 3], 1, &stops, &mut sampler, |_| Ok(()))
        .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(session.cached(), [1, 2, 3, first[0]]);
    // A prompt that fills the context is refused, never truncated.
    assert!(
        session
            .generate(&[1, 2, 3, 9], 1, &stops, &mut sampler, |_| Ok(()))
            .is_err()
    );
    session.reset().unwrap();
    assert!(session.cached().is_empty());
    // The same prompt greedily decodes to the same token from a reset
    // cache: the state the backend mutated in place is gone.
    let again = session
        .generate(&[1, 2, 3], 1, &stops, &mut sampler, |_| Ok(()))
        .unwrap();
    assert_eq!(again, first);
    // A prompt the cached history does not prefix rebuilds the cache.
    let other = session
        .generate(&[1, 2], 1, &stops, &mut sampler, |_| Ok(()))
        .unwrap();
    assert_eq!(session.cached(), [1, 2, other[0]]);
}
