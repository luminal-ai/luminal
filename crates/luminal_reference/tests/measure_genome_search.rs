//! Times a mini-model search over enough genomes that the session-level
//! memos can be seen. `harness_search_options` runs two genomes, which is
//! below the point where a genome-independent cache starts hitting, so the
//! smoke suites cannot show extraction-assembly cost either way.
use luminal::prelude::*;
use luminal_reference::CompileOptions;
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};
use std::time::Instant;

#[test]
#[ignore = "measurement — run explicitly by name (release)"]
fn measure_many_genome_search() {
    use model_zoo::mini::gemma3::MiniGemma3;
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
    let mut outputs = vec![output];
    outputs.extend(cache_outputs.into_iter().flat_map(|(k, v)| [k, v]));

    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    for (seed, spec) in cx.logical.input_specs().into_iter().enumerate() {
        let n: usize = spec.dims.iter().map(|d| d.to_usize().unwrap()).product();
        let v = match spec.dtype {
            DType::Int => TypedBuffer::I32(vec![0i32; n]),
            _ => TypedBuffer::F32((0..n).map(|i| ((i + seed) % 7) as f32 * 0.1).collect()),
        };
        data.insert(spec.id, v);
    }
    for (id, v) in [
        (ids.id, TypedBuffer::I32(vec![3])),
        (gather.id, TypedBuffer::I32(vec![0, 1])),
        (scatter.id, TypedBuffer::I32(vec![1])),
    ] {
        data.insert(id, v);
    }

    let ids_out: Vec<_> = outputs.iter().map(|t| t.id).collect();
    let bindings = ReferenceBindings::dense(&cx.logical, &ids_out);
    let mut rt = ReferenceRuntime::load_with(&cx, bindings).expect("load");
    let options = CompileOptions {
        generations: 12,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
        ..luminal_reference::harness_search_options()
    };
    let t = Instant::now();
    rt.search(&data, &options).expect("search");
    println!("GENOME_SEARCH_MS={}", t.elapsed().as_millis());
}
