use luminal::{collective::*, prelude::*};
use std::{net::TcpListener, sync::Arc, time::Duration};

#[test]
fn malformed_collectives_are_rejected_before_execution() {
    for (width, seq) in [(0, 0), (2, usize::MAX)] {
        assert!(
            std::panic::catch_unwind(|| {
                let mut g = Graph::new();
                let x = g.tensor([3], DType::F32);
                x.all_gather("g", width, seq);
            })
            .is_err()
        );
    }
    assert!(
        std::panic::catch_unwind(|| {
            let mut g = Graph::new();
            let x = g.tensor([3], DType::F32);
            x.all_gather("g", 2, 0);
            x.all_gather("g", 2, 0);
        })
        .is_err()
    );
}

#[test]
fn collective_identity_survives_saturation_and_refuses_rank_local_profiling() {
    let mut g = Graph::new();
    let x = g.tensor([3], DType::F32);
    let a = x.all_gather("g", 2, 0);
    let b = x.all_gather("g", 2, 1);
    assert_ne!(a.id, b.id);
    let mut rt = luminal_reference::ReferenceRuntime::load(&g).unwrap();
    rt.compile(&Default::default(), &Default::default())
        .unwrap();
    let count = rt
        .plan()
        .unwrap()
        .dag
        .node_weights()
        .filter(|n| {
            matches!(n,
        luminal::bufferize::BufferNode::Compute { op, .. } if op.label() == "AllGather")
        })
        .count();
    assert_eq!(count, 2);
    assert!(
        rt.search(
            &Default::default(),
            &Default::default(),
            &Default::default(),
            &luminal_reference::harness_search_options()
        )
        .unwrap_err()
        .to_string()
        .contains("compile()")
    );
}

#[test]
fn tcp_metadata_mismatch_fails_every_rank_and_poisoned_groups_stay_failed() {
    let address = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let threads: Vec<_> = (0..2)
        .map(|rank| {
            std::thread::spawn(move || {
                let group = Arc::new(
                    TcpAllGather::connect(address, rank, 2, Duration::from_secs(5)).unwrap(),
                );
                with_group("g", group, || {
                    let spec = AllGatherSpec {
                        group: "g".into(),
                        width: 2,
                        sequence: rank,
                    };
                    let shape = TensorSignature {
                        shape: vec![1],
                        bits: 8,
                        dtype: "U8".into(),
                    };
                    assert!(gather(&spec, &shape, &[rank as u8]).is_err());
                    assert!(gather(&spec, &shape, &[rank as u8]).is_err());
                });
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}
