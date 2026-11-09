#[cfg(target_os = "macos")]
mod support;
use luminal::prelude::*;
use luminal_metal::MetalRuntime;

#[test]
fn all_gather_is_claimed_and_has_a_host_execution_interface() {
    for dtype in [DType::F32, DType::I64, DType::Bool] {
        let mut graph = Graph::new();
        let input = graph.tensor([2, 3], dtype);
        input.all_gather("workers", 2, 0);
        let rt = MetalRuntime::load(&graph).unwrap();
        assert!(rt.active_allow_list().contains(&"LayoutTensorOpAllGather"));
        let snapshot = rt.saturated_egraph(&Default::default()).unwrap();
        let registry = luminal_metal::metal_registry();
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
        assert!(luminal_metal::as_host_op(operation.as_ref()).is_some());
        assert!(luminal::collective::ensure_local_profile(&plan).is_err());
    }
}

#[cfg(target_os = "macos")]
mod native {
    use super::support::TestTransfers;
    use super::*;
    use luminal::collective::{TcpAllGather, with_group};
    use std::{net::TcpListener, sync::Arc, time::Duration};

    #[test]
    fn metal_all_gather_replays_with_new_shapes_and_arenas() {
        let address = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let threads: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let group = Arc::new(
                        TcpAllGather::connect(address, rank, 2, Duration::from_secs(30)).unwrap(),
                    );
                    let mut graph = Graph::new();
                    let input = graph.tensor(['n'], DType::F32);
                    let output = (input + 1.).all_gather("workers", 2, 0) + 10.;
                    let bounds = SymbolBounds::from_ranges([('n'.into(), (0, 7))]).unwrap();
                    let mut rt = MetalRuntime::load(&graph).unwrap();
                    rt.compile(&bounds, &[('n'.into(), 3)].into_iter().collect())
                        .unwrap();
                    with_group("workers", group, || {
                        for n in [3, 3, 0, 7, 1] {
                            rt.set_dim('n', n);
                            let mut arena = support::allocate_arena(&rt).unwrap();
                            rt.upload(&mut arena, input.id, vec![rank as f32 + 1.; n])
                                .unwrap();
                            rt.execute(arena.buffer()).unwrap();
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
