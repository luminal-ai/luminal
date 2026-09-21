#![cfg(feature = "device")]

use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::*;
use luminal::resident::ResidentBindings;
use luminal_cuda_lite::{CompileOptions, CudaBindings, CudaRuntime, HostBuffer};

#[test]
fn profiling_retains_weights_resets_mutable_state_and_suppresses_resident_readback() {
    let mut graph = Graph::new();
    let weights = graph.tensor(4, DType::F32);
    let state = graph.tensor(4, DType::F32);
    let input = graph.tensor(1, DType::F32);
    let previous = state.sum(0);
    let next = state + weights * input.sum(0).expand_dim(0, 4);
    let mut bindings = CudaBindings::new();
    bindings.input_resident(weights.id);
    let home = bindings.input_resident(state.id);
    bindings.declare(home, Access::ReadWrite, FreedBy::Caller);
    bindings.input(input.id);
    bindings.output(previous.id);
    bindings.output_on(next.id, home);
    let mut runtime =
        CudaRuntime::load_with(&graph, bindings, luminal_cuda_lite::cuda_registry()).unwrap();
    let data: FxHashMap<NodeIndex, HostBuffer> = [
        (weights.id, vec![1f32, 2., 3., 4.].into()),
        (state.id, vec![10f32, 20., 30., 40.].into()),
        (input.id, vec![2f32].into()),
    ]
    .into_iter()
    .collect();
    let options = CompileOptions {
        generations: 1,
        generation_size: 1,
        trials: 3,
        search_log: false,
        ..Default::default()
    };
    let outcome = runtime.search(&data, &options).unwrap();
    assert_eq!(outcome.plans_profiled, 1);
    let stats = runtime.graph_stats().unwrap();
    assert_eq!(
        stats.launches, 5,
        "warmup, three trials, and finalist validation"
    );
    // Each preparation uploads 16 bytes of weights + 16 of state. Only state
    // is uploaded for the three resets. Finalist validation also honors residency.
    assert_eq!(stats.resident_upload_bytes, 2 * 32 + 3 * 16);

    let staged: FxHashMap<_, _> = data
        .iter()
        .map(|(id, data)| (runtime.input_buffer(*id).unwrap(), data))
        .collect();
    let residents = ResidentBindings {
        inputs: runtime.residents().clone(),
        externals: Default::default(),
    };
    let transient = staged
        .iter()
        .filter(|(lit, _)| !residents.inputs.contains(lit))
        .map(|(lit, data)| (*lit, *data))
        .collect();
    let mut device = luminal_cuda_lite::device::CudaDevice::new(0).unwrap();
    for _ in 0..2 {
        let before = device.stats();
        let measurement = luminal_cuda_lite::profile::profile_candidate(
            &mut device,
            runtime.plan().unwrap(),
            &staged,
            &residents,
            3,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(
            measurement,
            luminal_cuda_lite::profile::Measurement::Timed {
                completed_trials: 3,
                ..
            }
        ));
        let after = device.stats();
        assert_eq!(after.launches - before.launches, 4);
        assert_eq!(
            after.resident_upload_bytes - before.resident_upload_bytes,
            32 + 3 * 16
        );

        // The last trial leaves exactly ONE update to the initial state,
        // regardless of the warmup, previous trials, or previous candidate.
        let outputs = device.execute(0, &transient, &Default::default()).unwrap();
        assert_eq!(
            outputs.len(),
            1,
            "resident mutation output must stay on device"
        );
        let (data, slot) = outputs.values().next().unwrap();
        let actual =
            luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &slot.layout).unwrap();
        assert_eq!(actual, vec![120.]);
        assert_eq!(
            device.stats().resident_upload_bytes,
            after.resident_upload_bytes
        );
        assert_eq!(data.dtype, luminal::dtype::PlanDtype::F32);
    }
    assert_eq!(data[&state.id].as_f32().unwrap(), vec![10., 20., 30., 40.]);
}
