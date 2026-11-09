use luminal::{
    collective::{TcpAllGather, with_group},
    prelude::*,
};
use luminal_reference::{ReferenceRuntime, TypedBuffer};
use std::{net::TcpListener, sync::Arc, time::Duration};

fn address() -> std::net::SocketAddr {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

#[test]
fn gathered_values_preserve_rank_order_dtype_and_repeated_execution() {
    for dtype in [DType::F32, DType::I64, DType::Bool] {
        let address = address();
        let threads: Vec<_> = (0..3).map(|rank| std::thread::spawn(move || {
            let group = Arc::new(TcpAllGather::connect(address, rank, 3, Duration::from_secs(20)).unwrap());
            let mut graph = Graph::new();
            let input = graph.tensor(['n'], dtype);
            let output = input.all_gather("workers", 3, 7);
            let bounds = SymbolBounds::from_ranges([('n'.into(), (0, 4))]).unwrap();
            let mut rt = ReferenceRuntime::load(&graph).unwrap();
            rt.compile(&bounds, &[('n'.into(), 3)].into_iter().collect()).unwrap();
            let wire = luminal_reference::compiled_artifact::serialize(&luminal_reference::compiled_artifact::CompiledProgram {
                bounds: bounds.clone(), input_buffers: vec![], output_buffers: vec![], plan: rt.plan().unwrap().clone(),
            }).unwrap();
            let decoded = luminal_reference::compiled_artifact::deserialize(&wire).unwrap();
            assert!(decoded.plan.dag.node_weights().any(|n| matches!(n, luminal::bufferize::BufferNode::Compute { op, .. } if op.label() == "AllGather")));
            with_group("workers", group, || for n in [3, 0, 1, 4] {
                rt.set_dim('n', n);
                match dtype {
                    DType::F32 => rt.set_data(input.id, vec![f32::from_bits(0x7fc0_0040 + rank as u32); n]),
                    DType::I64 => rt.set_data(input.id, vec![i64::MAX - rank as i64; n]),
                    DType::Bool => rt.set_data(input.id, TypedBuffer::bool8(vec![(rank % 2) as u8; n]).unwrap()),
                    _ => unreachable!(),
                }
                rt.execute().unwrap();
                match dtype {
                    DType::F32 => assert_eq!(rt.get_f32(output.id).unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>(), (0..3).flat_map(|r| vec![0x7fc0_0040 + r; n]).collect::<Vec<_>>()),
                    DType::I64 => assert_eq!(*rt.get_i64(output.id).unwrap(), (0..3).flat_map(|r| vec![i64::MAX - r; n]).collect::<Vec<_>>()),
                    DType::Bool => assert_eq!(*rt.get_bool8(output.id).unwrap(), (0..3).flat_map(|r| vec![(r % 2) as u8; n]).collect::<Vec<_>>()),
                    _ => unreachable!(),
                }
            });
        })).collect();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}

#[test]
fn single_rank_materializes_views_and_preserves_scalar_shape() {
    let mut graph = Graph::new();
    let input = graph.tensor([2, 3], DType::F32);
    let output = input.permute(&[1, 0]).all_gather("local", 1, 0);
    let mut rt = ReferenceRuntime::load(&graph).unwrap();
    rt.compile(&Default::default(), &Default::default())
        .unwrap();
    rt.set_data(input.id, vec![1f32, 2., 3., 4., 5., 6.]);
    rt.execute().unwrap();
    assert_eq!(*rt.get_f32(output.id).unwrap(), [1., 4., 2., 5., 3., 6.]);
    let mut graph = Graph::new();
    let input = graph.tensor([0usize; 0], DType::F32);
    let output = input.all_gather("local", 1, 0);
    assert_eq!(output.dims(), [IntExpr::from(1)]);
    let mut rt = ReferenceRuntime::load(&graph).unwrap();
    rt.compile(&Default::default(), &Default::default())
        .unwrap();
    rt.set_data(input.id, vec![7f32]);
    rt.execute().unwrap();
    assert_eq!(*rt.get_f32(output.id).unwrap(), [7.]);
}

#[test]
fn multiple_collectives_preserve_dependencies() {
    let address = address();
    let ranks: Vec<_> = (0..2)
        .map(|rank| {
            std::thread::spawn(move || {
                let group = Arc::new(
                    TcpAllGather::connect(address, rank, 2, Duration::from_secs(20)).unwrap(),
                );
                let mut graph = Graph::new();
                let input = graph.tensor([2], DType::F32);
                let first = (input + 1.).all_gather("workers", 2, 0);
                let output = (first + 10.).all_gather("workers", 2, 1) + 100.;
                let mut rt = ReferenceRuntime::load(&graph).unwrap();
                rt.compile(&Default::default(), &Default::default())
                    .unwrap();
                rt.set_data(input.id, vec![rank as f32 + 1.; 2]);
                with_group("workers", group, || {
                    for _ in 0..2 {
                        rt.execute().unwrap();
                        assert_eq!(
                            *rt.get_f32(output.id).unwrap(),
                            vec![112., 112., 113., 113., 112., 112., 113., 113.]
                        );
                    }
                });
            })
        })
        .collect();
    for rank in ranks {
        rank.join().unwrap();
    }
}
