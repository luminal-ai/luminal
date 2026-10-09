//! Times search over enough genomes that session-level caches and the
//! extraction hot path can be seen. `harness_search_options` runs two
//! genomes, which is below the point where a genome-independent cache starts
//! hitting, so the smoke suites cannot show extraction cost either way.
//!
//! Run explicitly: `cargo test --release -p luminal_reference --test
//! measure_genome_search -- --ignored --nocapture`.

use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{GraphTensor, NodeIndex};
use luminal::shape::{IntExpr, SymbolBounds};
use luminal_reference::{
    CompileOptions, ReferenceBindings, ReferenceRuntime, TypedBuffer, harness_search_options,
};
use rustc_hash::FxHashMap;
use std::time::Instant;

fn values(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// Search `cx` over `generations * 4` genomes and return the wall time.
fn time_search(cx: Graph, outputs: &[GraphTensor], generations: usize) -> u128 {
    let data: FxHashMap<NodeIndex, TypedBuffer> = cx
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
            let value = match spec.dtype {
                DType::F32 => TypedBuffer::F32(values(elements, seed)),
                DType::Int if elements == 1 => TypedBuffer::I32(vec![1]),
                DType::Int => TypedBuffer::I32((0..elements as i32).collect()),
                DType::I64 if elements == 1 => TypedBuffer::I64(vec![1]),
                DType::I64 => TypedBuffer::I64((0..elements as i64).collect()),
                other => panic!("unsupported input dtype {other:?}"),
            };
            (spec.id, value)
        })
        .collect();

    let readbacks: Vec<_> = outputs.iter().map(|tensor| tensor.id).collect();
    let bindings = ReferenceBindings::dense(&cx.logical, &readbacks);
    let mut runtime = ReferenceRuntime::load_with(&cx, bindings).expect("load");
    let options = CompileOptions {
        generations,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
        ..harness_search_options()
    };
    let start = Instant::now();
    runtime
        .search(
            &SymbolBounds::default(),
            &FxHashMap::default(),
            &data,
            &options,
        )
        .expect("search");
    start.elapsed().as_millis()
}

#[test]
#[ignore = "measurement — run explicitly by name (release)"]
fn measure_many_genome_search() {
    const LAYERS: usize = 2;
    const HEAD_DIM: usize = 4;

    // gemma3: GQA with sliding-window bias and per-head QK norm.
    {
        use model_zoo::mini::gemma3::MiniGemma3;
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
        outputs.extend(cache_outputs.into_iter().flat_map(|(k, v)| [k, v]));
        println!("GEMMA3_MS={}", time_search(cx, &outputs, 12));
    }

    // llama3: dense GQA attention with a KV cache, no sliding window.
    {
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
        outputs.extend(cache_outputs.into_iter().flat_map(|(k, v)| [k, v]));
        println!("LLAMA3_MS={}", time_search(cx, &outputs, 12));
    }

    // qwen3_moe: expert routing on top of the attention block.
    {
        use model_zoo::mini::qwen3_moe::MiniQwen3Moe;
        let mut cx = Graph::new();
        let model = MiniQwen3Moe::new(5, 4, 2, 1, 2, 1, &mut cx);
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
        outputs.extend(cache_outputs.into_iter().flat_map(|(k, v)| [k, v]));
        println!("QWEN3_MOE_MS={}", time_search(cx, &outputs, 12));
    }

    // conv: the convnet family, no attention and no cache.
    {
        use model_zoo::mini::conv::MiniConvNet;
        let mut cx = Graph::new();
        let model = MiniConvNet::new(1, 2, 3, 2, &mut cx);
        let input = cx.tensor(vec![1, 1, 5, 5], DType::F32);
        let output = model.forward(input);
        println!("CONV_MS={}", time_search(cx, &[output], 12));
    }
}
