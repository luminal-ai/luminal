//! Execution-only smoke coverage for every mini model family.
//!
//! These tests deliberately do not carry scalar expected values or compare
//! runtimes. Numerical correctness belongs to the operation/runtime test
//! suites; a mini model smoke test only proves that its complete small graph
//! can be searched, staged, executed, and read back.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, NodeIndex};
use luminal::shape::IntExpr;
use luminal_reference::ReferenceBindings;
use luminal_reference::ReferenceRuntime;
use luminal_reference::TypedBuffer;

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
                    // E4M3FN saturates to +/-448, matching the reference cast
                    // kernel, so the clamp is applied here too.
                    DType::F8E4M3FN => TypedBuffer::F8E4M3FN(
                        values(elements, seed)
                            .into_iter()
                            .map(|v| float8::F8E4M3::from_f32(v.clamp(-448.0, 448.0)))
                            .collect(),
                    ),
                    other => panic!("unsupported mini smoke input dtype {other:?}"),
                }
            };
            (spec.id, value)
        })
        .collect::<Vec<(NodeIndex, TypedBuffer)>>();
    let data = pairs.iter().cloned().collect();
    // The KV-cache outputs are consumed by the same graph (attention reads
    // the updated cache), so they are not leaves and the default binding
    // would not carry them: bind every requested readback by name.
    let readbacks: Vec<_> = outputs.iter().map(|tensor| tensor.id).collect();
    let mut runtime =
        ReferenceRuntime::load_with(&cx, ReferenceBindings::dense(&cx.logical, &readbacks))
            .expect("reference load");
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &luminal_reference::harness_search_options(),
        )
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
    use model_zoo::mini::conv::MiniConvNet;

    let mut cx = Graph::new();
    let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
    let input = cx.tensor(vec![1, 1, 5, 5], DType::F32);
    let output = model.forward(input);
    run(cx, &[output], []);
}

#[test]
fn mini_llama3_runs() {
    use model_zoo::mini::llama3::MiniLlama3;

    let mut cx = Graph::new();
    let model = MiniLlama3::new(5, 8, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let caches = vec![(
        cx.tensor(vec![4, 4], DType::F32),
        cx.tensor(vec![4, 4], DType::F32),
    )];
    let gather = cx.tensor(vec![2], DType::Int);
    let scatter = cx.tensor(vec![1], DType::Int);
    let (output, cache_outputs) =
        model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));
    let mut outputs = vec![output];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key, value]),
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
    use model_zoo::mini::qwen3::MiniQwen3;

    let mut cx = Graph::new();
    let model = MiniQwen3::new(5, 8, 12, 4, 2, 1, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let caches = vec![(
        cx.tensor(vec![4, 4], DType::F32),
        cx.tensor(vec![4, 4], DType::F32),
    )];
    let gather = cx.tensor(vec![2], DType::Int);
    let scatter = cx.tensor(vec![1], DType::Int);
    let (output, cache_outputs) =
        model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize));
    let mut outputs = vec![output];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key, value]),
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
    use model_zoo::mini::gemma3::MiniGemma3;

    const LAYERS: usize = 2;
    const HEAD_DIM: usize = 4;
    let mut cx = Graph::new();
    let model = MiniGemma3::new(5, 6, 8, 2, 1, HEAD_DIM, LAYERS, 1, 2, &mut cx);
    let ids = cx.tensor(vec![1], DType::Int);
    let caches = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor(vec![4, HEAD_DIM], DType::F32),
                cx.tensor(vec![4, HEAD_DIM], DType::F32),
            )
        })
        .collect::<Vec<_>>();
    let gather = cx.tensor(vec![2], DType::Int);
    let scatter = cx.tensor(vec![1], DType::Int);
    let rope = (0..LAYERS)
        .map(|_| {
            (
                cx.tensor(vec![1, HEAD_DIM], DType::F32),
                cx.tensor(vec![1, HEAD_DIM], DType::F32),
            )
        })
        .collect::<Vec<_>>();
    let rotation = cx.tensor(vec![HEAD_DIM, HEAD_DIM], DType::F32);
    let (output, cache_outputs) = model.forward(
        ids,
        &caches,
        gather,
        scatter,
        IntExpr::from(1usize),
        &rope,
        rotation,
    );
    let mut outputs = vec![output];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key, value]),
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
    let mut outputs = vec![output];
    outputs.extend(
        cache_outputs
            .into_iter()
            .flat_map(|(key, value)| [key, value]),
    );
    run(cx, &outputs, []);
}

#[test]
fn mini_qwen3_moe_runs() {
    use model_zoo::mini::qwen3_moe::MiniQwen3Moe;

    mini_moe(
        |cx| MiniQwen3Moe::new(5, 4, 2, 1, 2, 1, cx),
        |model, cx| {
            let ids = cx.tensor(vec![1], DType::Int);
            let caches = vec![(
                cx.tensor(vec![4, 4], DType::F32),
                cx.tensor(vec![4, 4], DType::F32),
            )];
            let gather = cx.tensor(vec![2], DType::Int);
            let scatter = cx.tensor(vec![1], DType::Int);
            model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize))
        },
    );
}

#[test]
fn mini_gemma4_moe_runs() {
    use model_zoo::mini::gemma4_moe::MiniGemma4Moe;

    mini_moe(
        |cx| MiniGemma4Moe::new(5, 4, 2, 1, 2, 1, cx),
        |model, cx| {
            let ids = cx.tensor(vec![1], DType::Int);
            let caches = vec![(
                cx.tensor(vec![4, 4], DType::F32),
                cx.tensor(vec![4, 4], DType::F32),
            )];
            let gather = cx.tensor(vec![2], DType::Int);
            let scatter = cx.tensor(vec![1], DType::Int);
            model.forward(ids, &caches, gather, scatter, IntExpr::from(1usize))
        },
    );
}

#[test]
fn mini_whisper_runs() {
    use model_zoo::mini::whisper::MiniWhisper;

    let mut cx = Graph::new();
    let model = MiniWhisper::new(4, 6, 2, &mut cx);
    let audio = cx.tensor(vec![2, 4], DType::F32);
    let tokens = cx.tensor(vec![1, 4], DType::F32);
    let output = model.forward(audio, tokens);
    run(cx, &[output], []);
}

#[test]
#[ignore = "blocked by the known adaLN rejoin-divergence search issue"]
fn mini_flux_runs() {
    use model_zoo::mini::flux::MiniDit;

    const TEXT_TOKENS: usize = 2;
    const IMAGE_TOKENS: usize = 4;
    const HEAD_DIM: usize = 8;
    const HIDDEN: usize = 16;
    let mut cx = Graph::new();
    let model = MiniDit::new(4, 6, HIDDEN, 2, 6, 2, TEXT_TOKENS, &mut cx);
    let latent = cx.tensor(vec![IMAGE_TOKENS, 4], DType::F32);
    let text = cx.tensor(vec![TEXT_TOKENS, 6], DType::F32);
    let timestep = cx.tensor(vec![1], DType::F32);
    let guidance = cx.tensor(vec![1], DType::F32);
    let rope_cos = cx.tensor(vec![TEXT_TOKENS + IMAGE_TOKENS, HEAD_DIM], DType::F32);
    let rope_sin = cx.tensor(vec![TEXT_TOKENS + IMAGE_TOKENS, HEAD_DIM], DType::F32);
    let rope_rotation = cx.tensor(vec![HEAD_DIM, HEAD_DIM], DType::F32);
    let joint_base = cx.tensor(vec![TEXT_TOKENS + IMAGE_TOKENS, HIDDEN], DType::F32);
    let output = model.forward(
        latent,
        text,
        timestep,
        guidance,
        rope_cos,
        rope_sin,
        rope_rotation,
        joint_base,
    );
    run(cx, &[output], []);
}

#[test]
fn llama3_1_fp8_runs() {
    use luminal_nn::{KvCache, KvCachePool, rope_pairing_matrix, rope_tables_split_half};
    use model_zoo::llama3_1_fp8::Fp8Dims;
    use model_zoo::llama3_1_fp8::model::Llama31Fp8;

    const SLOTS: usize = 4;
    // Small enough to search quickly, structurally identical to the 8B model.
    let dims = Fp8Dims {
        vocab: 32,
        hidden: 16,
        intermediate: 32,
        head_dim: 8,
        n_heads: 2,
        n_kv_heads: 1,
        layers: 2,
        rope_theta: 10_000.0,
        rope_factor: 1.0,
        rope_low_freq_factor: 1.0,
        rope_high_freq_factor: 4.0,
        rope_original_max: 8192.0,
        rms_eps: 1e-5,
    };

    let mut cx = Graph::new();
    let model = Llama31Fp8::init(&mut cx, &dims);
    let token = cx.tensor(vec![1], DType::Int);
    let q_pos = cx.tensor(vec![1], DType::Int);
    let rope_cos = cx.tensor(vec![1, dims.head_dim], DType::F32);
    let rope_sin = cx.tensor(vec![1, dims.head_dim], DType::F32);
    let rope_rot = cx.tensor(vec![dims.head_dim, dims.head_dim], DType::F32);
    let gather_idx = cx.tensor(vec![SLOTS], DType::Int);
    let scatter_idx = cx.tensor(vec![1], DType::Int);
    let pool = KvCachePool::from_layers((0..dims.layers).map(|_| {
        KvCache::new(
            cx.tensor(vec![SLOTS, dims.kv_dim()], DType::F32),
            cx.tensor(vec![SLOTS, dims.kv_dim()], DType::F32),
        )
    }));

    let (logits, caches) = model.forward(
        token,
        q_pos,
        rope_cos,
        rope_sin,
        rope_rot,
        &pool,
        gather_idx,
        scatter_idx,
    );

    // Every fp8 projection carries rank-0 `input_scale`/`weight_scale`, and
    // `fp8_linear` takes `input_scale.reciprocal()`. The generic F32 filler can
    // hand those a zero, which yields NaN logits that still satisfy the
    // non-empty readback assertion. Pin every rank-0 F32 input to 1.0.
    let scale_overrides = cx
        .logical
        .input_specs()
        .into_iter()
        .filter(|spec| spec.dtype == DType::F32 && spec.dims.is_empty())
        .map(|spec| (spec.id, TypedBuffer::F32(vec![1.0])))
        .collect::<Vec<_>>();
    assert!(
        !scale_overrides.is_empty(),
        "fp8 model should expose rank-0 scale inputs"
    );

    let (cos, sin) = rope_tables_split_half(&[1.0], dims.head_dim, dims.rope_theta, 1.0);
    let overrides = scale_overrides.into_iter().chain([
        (rope_cos.id, TypedBuffer::F32(cos)),
        (rope_sin.id, TypedBuffer::F32(sin)),
        (
            rope_rot.id,
            TypedBuffer::F32(rope_pairing_matrix(dims.head_dim, false)),
        ),
        (token.id, TypedBuffer::I32(vec![3])),
        (q_pos.id, TypedBuffer::I32(vec![1])),
        (gather_idx.id, TypedBuffer::I32((0..SLOTS as i32).collect())),
        (scatter_idx.id, TypedBuffer::I32(vec![1])),
    ]);

    let mut outputs = vec![logits];
    for (k, v) in caches {
        outputs.push(k);
        outputs.push(v);
    }
    run(cx, &outputs, overrides);
}
