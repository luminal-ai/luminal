#![cfg(feature = "device")]

use luminal::graph::DimBucket;
use luminal::prelude::*;
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer};

#[test]
fn representative_payloads_override_shared_inputs_in_search_and_finalist_validation() {
    let mut graph = Graph::new();
    let input = graph.tensor('q', DType::F32);
    let output = input + 1.;
    let create = || {
        let mut runtime = CudaRuntime::load(&graph).unwrap();
        runtime
            .bind_dim_buckets(
                'q',
                vec![
                    DimBucket::new(1, 1),
                    DimBucket::new(2, 128).representative(128),
                ],
            )
            .unwrap();
        runtime
    };
    let options = CompileOptions {
        generations: 1,
        generation_size: 1,
        trials: 1,
        search_log: false,
        ..Default::default()
    };
    // This fallback has the wrong dtype: success proves each representative
    // and its finalist received the override, not resized fallback data.
    let shared: FxHashMap<_, HostBuffer> = [(input.id, vec![0i32].into())].into_iter().collect();
    let profiles: Vec<(luminal::shape::DynMap, FxHashMap<_, HostBuffer>)> = [1usize, 128]
        .into_iter()
        .map(|q| {
            (
                [('q'.into(), q)].into_iter().collect(),
                [(input.id, vec![q as f32; q].into())].into_iter().collect(),
            )
        })
        .collect();
    let mut runtime = create();
    runtime
        .search_with_profile_inputs(&shared, &profiles, &options)
        .unwrap();
    assert_eq!(runtime.bucket_plans().len(), 2);
    assert_eq!(runtime.graph_stats().unwrap().launches, 6);
    for q in [128usize, 1, 7] {
        runtime.set_dim('q', q);
        runtime.set_data(input.id, vec![q as f32; q]).unwrap();
        runtime.execute().unwrap();
        assert_eq!(runtime.get_f32(output.id).unwrap(), vec![q as f32 + 1.; q]);
    }
    assert_eq!(shared[&input.id].as_i32().unwrap(), vec![0]);
    let missing = create()
        .search_with_profile_inputs(&shared, &profiles[..1], &options)
        .unwrap_err();
    assert!(format!("{missing:#}").contains("no profiling inputs"));
    let duplicate = [
        profiles[0].clone(),
        profiles[0].clone(),
        profiles[1].clone(),
    ];
    let ambiguous = create()
        .search_with_profile_inputs(&shared, &duplicate, &options)
        .unwrap_err();
    assert!(format!("{ambiguous:#}").contains("ambiguous profiling inputs"));
}
