#[cfg(feature = "device")]
mod support;
use luminal::prelude::*;
use luminal_cuda_lite::CudaRuntime;

#[test]
fn all_gather_is_claimed_and_uses_a_host_operation() {
    for dtype in [DType::F32, DType::I64, DType::Bool] {
        let mut graph = Graph::new();
        let input = graph.tensor([2, 3], dtype);
        input.all_gather("workers", 2, 0);
        let rt = CudaRuntime::load(&graph).unwrap();
        assert!(rt.active_allow_list().contains(&"LayoutTensorOpAllGather"));
        let snapshot = rt.saturated_egraph(&Default::default()).unwrap();
        let registry = luminal_cuda_lite::cuda_registry();
        let matchers: Vec<_> = registry.into_iter().map(|r| r.matcher).collect();
        let plan = luminal::collective::compile_first(
            &snapshot,
            rt.decoders(),
            &matchers,
            rt.active_allow_list(),
        )
        .unwrap();
        let operation = plan
            .dag
            .node_weights()
            .find_map(|n| match n {
                luminal::bufferize::BufferNode::Compute { op, .. } if op.label() == "AllGather" => {
                    Some(op)
                }
                _ => None,
            })
            .unwrap();
        assert!(luminal_cuda_lite::as_host_op(operation.as_ref()).is_some());
        assert!(luminal::collective::ensure_local_profile(&plan).is_err());
    }
}

#[test]
fn collective_compile_retains_dynamic_capacity_bounds() {
    let mut graph = Graph::new();
    let input = graph.tensor(['n'], DType::F32);
    input.all_gather("workers", 2, 0);
    let bounds = SymbolBounds::from_ranges([('n'.into(), (0, 7))]).unwrap();
    let dims = [('n'.into(), 3)].into_iter().collect();
    let mut rt = CudaRuntime::load(&graph).unwrap();

    rt.compile(&bounds, &dims).unwrap();

    assert!(rt.staging_bytes().unwrap() >= 1);
}

#[cfg(feature = "device")]
mod native {
    use super::support::TestTransfers;
    use super::*;
    use luminal_cuda_lite::{
        collective::{NcclCommunicator, with_group},
        device::CudaDevice,
    };

    #[test]
    #[ignore = "requires two CUDA devices and NCCL; run explicitly on the multi-GPU runner"]
    fn nccl_all_gather_replays_with_new_shapes_and_arenas() {
        let id = NcclCommunicator::unique_id().unwrap();
        let threads: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let device = CudaDevice::new(rank).unwrap();
                    let group = NcclCommunicator::init(&device, id, rank, 2).unwrap();
                    let mut graph = Graph::new();
                    let input = graph.tensor(['n'], DType::F32);
                    let output = (input + 1.).all_gather("workers", 2, 0) + 10.;
                    let bounds = SymbolBounds::from_ranges([('n'.into(), (0, 7))]).unwrap();
                    let mut rt = CudaRuntime::load(&graph)
                        .unwrap()
                        .with_device(&device)
                        .unwrap();
                    rt.compile(&bounds, &[('n'.into(), 3)].into_iter().collect())
                        .unwrap();
                    let mut staging = rt.allocate_staging().unwrap();
                    with_group("workers", group, || {
                        for n in [3, 3, 0, 7, 1] {
                            rt.set_dim('n', n);
                            let mut arena = support::allocate_arena(&rt).unwrap();
                            rt.upload(&mut arena, input.id, vec![rank as f32 + 1.; n])
                                .unwrap();
                            rt.execute(arena.arena(), &mut staging).unwrap();
                            assert_eq!(
                                rt.read_f32(&arena, output.id).unwrap(),
                                [vec![12.; n], vec![13.; n]].concat()
                            );
                        }
                    });
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
