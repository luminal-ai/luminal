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

#[cfg(feature = "device")]
mod native {
    use super::support::TestTransfers;
    use super::*;
    use luminal_cuda_lite::{
        collective::{NcclCommunicator, with_group},
        device::CudaDevice,
    };

    #[test]
    fn width_one_collective_uses_the_normal_search_and_installs_its_finalist() {
        let device = CudaDevice::new(0).unwrap();
        let id = NcclCommunicator::unique_id().unwrap();
        let group = NcclCommunicator::init(&device, id, 0, 1).unwrap();
        let mut graph = Graph::new();
        let input = graph.tensor([3], DType::F32);
        let output = (input + 1.).all_gather("workers", 1, 0) + 10.;
        let mut rt = CudaRuntime::load(&graph)
            .unwrap()
            .with_device(&device)
            .unwrap();
        let options = luminal_cuda_lite::harness_search_options();

        let refusal = rt
            .search(
                &Default::default(),
                &Default::default(),
                &Default::default(),
                &options,
            )
            .unwrap_err();
        assert!(refusal.to_string().contains("search_collective"));

        with_group("workers", group.clone(), || {
            let outcome = rt
                .search_collective(
                    &Default::default(),
                    &Default::default(),
                    &Default::default(),
                    &options,
                    group,
                )
                .unwrap()
                .expect("rank zero owns the search outcome");
            assert!(outcome.plans_profiled > 0);

            let mut staging = rt.allocate_staging().unwrap();
            let mut arena = support::allocate_arena(&rt).unwrap();
            rt.upload(&mut arena, input.id, vec![2., 4., 8.]).unwrap();
            rt.execute(arena.arena(), &mut staging).unwrap();
            assert_eq!(rt.read_f32(&arena, output.id).unwrap(), [13., 15., 19.]);
        });
    }

    #[test]
    #[ignore = "requires two CUDA devices and NCCL; run explicitly on the multi-GPU runner"]
    fn nccl_all_gather_searches_once_and_replays_on_every_rank() {
        if cudarc::driver::CudaContext::device_count().unwrap() < 2 {
            eprintln!("skipping: this runner exposes fewer than two CUDA devices");
            return;
        }
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
                    let dims = [('n'.into(), 3)].into_iter().collect();
                    let options = luminal_cuda_lite::harness_search_options();
                    with_group("workers", group.clone(), || {
                        let outcome = rt
                            .search_collective(&bounds, &dims, &Default::default(), &options, group)
                            .unwrap();
                        assert_eq!(outcome.is_some(), rank == 0);
                        let mut staging = rt.allocate_staging().unwrap();
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
