//! Shared F32 input-rounding execution and reference checks.
use super::TestTransfers;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaBindings, CudaRuntime, HostBuffer, RegisteredOp};
use luminal_reference::{ReferenceBindings, ReferenceRuntime, TypedBuffer};

pub fn reference(graph: &Graph, out: GraphTensor, inputs: &[(GraphTensor, Vec<f32>)]) -> Vec<f32> {
    let mut runtime =
        ReferenceRuntime::load_with(graph, ReferenceBindings::dense(&graph.logical, &[out.id]))
            .unwrap();
    let data: FxHashMap<_, _> = inputs
        .iter()
        .map(|(x, data)| (x.id, TypedBuffer::F32(data.clone())))
        .collect();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &luminal_reference::harness_search_options(),
        )
        .unwrap();
    for (id, data) in data {
        runtime.set_data(id, data);
    }
    runtime.execute().unwrap();
    runtime.get_f32(out.id).unwrap().clone()
}

pub fn execute(
    graph: &Graph,
    out: GraphTensor,
    bindings: CudaBindings,
    inputs: &[(GraphTensor, Vec<f32>)],
    registry: Vec<RegisteredOp>,
) -> (CudaRuntime, Vec<f32>) {
    let kept_inputs: Vec<_> = inputs
        .iter()
        .filter(|(x, _)| bindings.outputs().iter().any(|out| out.value == x.id))
        .cloned()
        .collect();
    let mut runtime = CudaRuntime::load_with(graph, bindings, registry).unwrap();
    let data: FxHashMap<_, _> = inputs
        .iter()
        .map(|(x, data)| (x.id, HostBuffer::from(data.clone())))
        .collect();
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &luminal_cuda_lite::harness_search_options(),
        )
        .unwrap();
    let mut arena = super::allocate_arena(&runtime).unwrap();
    let mut staging = runtime.allocate_staging().unwrap();
    for (id, data) in data {
        runtime.upload(&mut arena, id, data).unwrap();
    }
    runtime.execute(arena.arena(), &mut staging).unwrap();
    for (input, expected) in kept_inputs {
        let (data, binding) = runtime.download(&arena, input.id).unwrap();
        assert_eq!(
            luminal_cuda_lite::layouts::dense_f32(&data.as_f32().unwrap(), &binding.layout)
                .unwrap(),
            expected,
            "a kept input must survive accumulation"
        );
    }
    let (data, binding) = runtime.download(&arena, out.id).unwrap();
    let raw = match data.dtype {
        luminal::dtype::PlanDtype::F32 => data.as_f32().unwrap(),
        dtype => data
            .bytes
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_ne_bytes(bytes.try_into().unwrap());
                match dtype {
                    luminal::dtype::PlanDtype::Bf16 => half::bf16::from_bits(bits).to_f32(),
                    luminal::dtype::PlanDtype::F16 => half::f16::from_bits(bits).to_f32(),
                    _ => panic!("unexpected output {dtype:?}"),
                }
            })
            .collect(),
    };
    let dense = luminal_cuda_lite::layouts::dense_f32(&raw, &binding.layout).unwrap();
    (runtime, dense)
}
