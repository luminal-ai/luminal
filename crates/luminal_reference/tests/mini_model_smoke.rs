//! Execution-only smoke coverage for every mini model family.
//!
//! These tests deliberately do not carry scalar expected values or compare
//! runtimes. Numerical correctness belongs to the operation/runtime test
//! suites; a mini model smoke test only proves that its complete small graph
//! can be searched, staged, executed, and read back.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_reference::ReferenceRuntime;

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

fn run(
    cx: Graph,
    outputs: &[GraphTensor],
    overrides: impl IntoIterator<Item = (NodeIndex, TypedBuffer)>,
) {
    let mut overrides = overrides
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    let pairs = cx
        .logical
        .input_specs()
        .into_iter()
        .enumerate()
        .map(|(seed, spec)| {
            let elements = spec
                .dims
                .iter()
                .map(|dim| dim.to_usize().expect("static mini dimension"))
                .product::<usize>();
            let value = if let Some(value) = overrides.remove(&spec.id) {
                value
            } else {
                match spec.dtype {
                    DType::F32 => TypedBuffer::F32(values(elements, seed)),
                    DType::Int if elements == 1 => TypedBuffer::I32(vec![1]),
                    DType::Int => TypedBuffer::I32((0..elements as i32).collect()),
                    DType::I64 if elements == 1 => TypedBuffer::I64(vec![1]),
                    DType::I64 => TypedBuffer::I64((0..elements as i64).collect()),
                    other => panic!("unsupported mini smoke input dtype {other:?}"),
                }
            };
            (spec.id, value)
        })
        .collect::<Vec<(NodeIndex, TypedBuffer)>>();
    let data = pairs.iter().cloned().collect();
    let mut runtime = ReferenceRuntime::load(&cx).expect("reference load");
    runtime
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("mini graph searches");
    for (id, value) in pairs {
        runtime.set_data(id, value);
    }
    runtime.execute().expect("mini graph executes");
    for output in outputs {
        assert!(
            !runtime
                .get_f32(output.id)
                .expect("mini output readback")
                .is_empty(),
            "mini output is empty"
        );
    }
}

#[test]
fn mini_conv_runs() {
    use mini_conv::MiniConvNet;

    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let input = cx.tensor((1, 1, 5, 5), DType::F32);
    let output = model.forward(input).output();
    run(cx, &[output], []);
}

#[test]
fn mini_llama3_runs() {
    use mini_llama3::MiniLlama3;

    let mut cx = Graph::new();
    let model = MiniLlama3::new(5, 8, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let caches = vec![(cx.tensor((4, 4), DType::F32), cx.tensor((4, 4), DType::F32))];
    let gather = cx.tensor(2, DType::Int);
    let scatter = cx.tensor(1, DType::Int);
    let (output, cache_outputs) =
        model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));
    let mut outputs = vec![output.output()];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key.output(), value.output()]),
    );
    run(
        cx,
        &outputs,
        [
            (ids.id, vec![3i32].into()),
            (gather.id, vec![0i32, 1].into()),
            (scatter.id, vec![1i32].into()),
        ],
    );
}

#[test]
fn mini_qwen3_runs() {
    use mini_qwen3::MiniQwen3;

    let mut cx = Graph::new();
    let model = MiniQwen3::new(5, 8, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let caches = vec![(cx.tensor((4, 4), DType::F32), cx.tensor((4, 4), DType::F32))];
    let gather = cx.tensor(2, DType::Int);
    let scatter = cx.tensor(1, DType::Int);
    let (output, cache_outputs) =
        model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));
    let mut outputs = vec![output.output()];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key.output(), value.output()]),
    );
    run(
        cx,
        &outputs,
        [
            (ids.id, vec![3i32].into()),
            (gather.id, vec![0i32, 1].into()),
            (scatter.id, vec![1i32].into()),
        ],
    );
}

#[test]
fn mini_gemma3_runs() {
    use mini_gemma3::MiniGemma3;

    const LAYERS: usize = 2;
    const HEAD_DIM: usize = 4;
    let mut cx = Graph::new();
    let model = MiniGemma3::new(5, 6, 8, 2, 1, HEAD_DIM, LAYERS, 1, 2, &mut cx);
    let ids = cx.tensor(1, DType::Int);
    let caches = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor((4, HEAD_DIM), DType::F32),
                cx.tensor((4, HEAD_DIM), DType::F32),
            )
        })
        .collect::<Vec<_>>();
    let gather = cx.tensor(2, DType::Int);
    let scatter = cx.tensor(1, DType::Int);
    let rope = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor((1, HEAD_DIM), DType::F32),
                cx.tensor((1, HEAD_DIM), DType::F32),
            )
        })
        .collect::<Vec<_>>();
    let rotation = cx.tensor((HEAD_DIM, HEAD_DIM), DType::F32);
    let (output, cache_outputs) = model.forward(
        ids,
        &caches,
        gather,
        scatter,
        IntExpr::from(1usize),
        &rope,
        rotation,
    );
    let mut outputs = vec![output.output()];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key.output(), value.output()]),
    );
    run(
        cx,
        &outputs,
        [
            (ids.id, vec![3i32].into()),
            (gather.id, vec![0i32, 1].into()),
            (scatter.id, vec![1i32].into()),
        ],
    );
}

fn mini_moe<M>(
    build: impl FnOnce(&mut Graph) -> M,
    forward: impl FnOnce(M, &mut Graph) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>),
) {
    let mut cx = Graph::new();
    let model = build(&mut cx);
    let (output, cache_outputs) = forward(model, &mut cx);
    let mut outputs = vec![output.output()];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key.output(), value.output()]),
    );
    run(cx, &outputs, []);
}

#[test]
fn mini_qwen3_moe_runs() {
    use mini_qwen3_moe::MiniQwen3Moe;

    mini_moe(
        |cx| MiniQwen3Moe::new(5, 4, 2, 1, 2, 1, cx),
        |model, cx| {
            let ids = cx.tensor(1, DType::Int);
            let caches = vec![(cx.tensor((4, 4), DType::F32), cx.tensor((4, 4), DType::F32))];
            let gather = cx.tensor(2, DType::Int);
            let scatter = cx.tensor(1, DType::Int);
            model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize))
        },
    );
}

#[test]
fn mini_gemma4_moe_runs() {
    use mini_gemma4_moe::MiniGemma4Moe;

    mini_moe(
        |cx| MiniGemma4Moe::new(5, 4, 2, 1, 2, 1, cx),
        |model, cx| {
            let ids = cx.tensor(1, DType::Int);
            let caches = vec![(cx.tensor((4, 4), DType::F32), cx.tensor((4, 4), DType::F32))];
            let gather = cx.tensor(2, DType::Int);
            let scatter = cx.tensor(1, DType::Int);
            model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize))
        },
    );
}

#[test]
fn mini_whisper_runs() {
    use mini_whisper::MiniWhisper;

    let mut cx = Graph::new();
    let model = MiniWhisper::new(4, 6, 2, &mut cx);
    let audio = cx.tensor((2, 4), DType::F32);
    let tokens = cx.tensor((1, 4), DType::F32);
    let output = model.forward(audio, tokens).output();
    run(cx, &[output], []);
}

#[test]
#[ignore = "blocked by the known adaLN rejoin-divergence search issue"]
fn mini_flux_runs() {
    use mini_flux::MiniDit;

    const TEXT_TOKENS: usize = 2;
    const IMAGE_TOKENS: usize = 4;
    const HEAD_DIM: usize = 8;
    const HIDDEN: usize = 16;
    let mut cx = Graph::new();
    let model = MiniDit::new(4, 6, HIDDEN, 2, 6, 2, TEXT_TOKENS, &mut cx);
    let latent = cx.tensor((IMAGE_TOKENS, 4), DType::F32);
    let text = cx.tensor((TEXT_TOKENS, 6), DType::F32);
    let timestep = cx.tensor(1, DType::F32);
    let guidance = cx.tensor(1, DType::F32);
    let rope_cos = cx.tensor((TEXT_TOKENS + IMAGE_TOKENS, HEAD_DIM), DType::F32);
    let rope_sin = cx.tensor((TEXT_TOKENS + IMAGE_TOKENS, HEAD_DIM), DType::F32);
    let rope_rotation = cx.tensor((HEAD_DIM, HEAD_DIM), DType::F32);
    let joint_base = cx.tensor((TEXT_TOKENS + IMAGE_TOKENS, HIDDEN), DType::F32);
    let output = model
        .forward(
            latent,
            text,
            timestep,
            guidance,
            rope_cos,
            rope_sin,
            rope_rotation,
            joint_base,
        )
        .output();
    run(cx, &[output], []);
}
