//! cargo run --release -p luminal_cuda_lite --example representative_profiles
//! Ordinary matrix arithmetic; no inference-server or model dependencies.
use luminal::prelude::*;
use luminal_cuda_lite::runtime::{CudaRuntime, ProfileWorkload};
use rand::{SeedableRng, rngs::SmallRng};

fn main() -> anyhow::Result<()> {
    let mut graph = Graph::new();
    let a = graph.tensor(('m', 4)).persist();
    let b = graph.tensor((4, 4)).persist();
    let output = a.matmul(b).output();
    graph.build_search_space::<CudaRuntime>(
        CompileOptions::default().dim_buckets('m', &[DimBucket::new(1, 4).representative(2)]),
    );
    let mut runtime = CudaRuntime::new()?;
    runtime.set_data(b, vec![1f32; 16]);
    let mut workload = ProfileWorkload::new().shared_input(b);
    for rows in [2, 4] {
        graph.set_dim('m', rows);
        runtime.set_data(a, vec![rows as f32; rows * 4]);
        let snapshot = runtime.capture_profile_inputs(&[a], &graph.dyn_map)?;
        workload = workload.case(format!("rows-{rows}"), graph.dyn_map.clone(), snapshot, 1.);
    }
    runtime.set_profile_workload(&graph, workload)?;
    runtime = graph.search_with_rng(
        runtime,
        CompileOptions::default().search_graph_limit(5).trials(3),
        &mut SmallRng::seed_from_u64(43),
    );
    for evaluation in runtime
        .profile_evaluations()
        .iter()
        .filter(|e| e.cuda_graph)
    {
        println!(
            "CUDA-graph workload score {:?}: {:?}",
            evaluation.weighted_cost, evaluation.cases
        );
    }
    // Examples tune performance; they do not restrict the bucket's valid shapes.
    graph.set_dim('m', 3);
    runtime.set_data(a, vec![2f32; 12]);
    runtime.execute(&graph.dyn_map);
    assert_eq!(runtime.get_f32(output), vec![8f32; 12]);
    Ok(())
}
