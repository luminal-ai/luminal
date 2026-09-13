use super::*;
use crate::{
    runtime::CudaRuntime,
    tests::utilities::{ForcedExtractionConfig, get_cuda_stream, try_extract_forced_op_llir_where},
};

fn data(len: usize, weight: bool) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            bf16::from_f32(if weight {
                (((i * 17 + 5) % 29) as i32 - 14) as f32 / 128.0
            } else {
                (((i * 13 + 3) % 31) as i32 - 15) as f32 / 128.0
            })
        })
        .collect()
}

#[test]
fn ranked_algorithms_keep_full_precision_for_unaligned_inputs() {
    let stream = get_cuda_stream().expect("CUDA required for algorithm regression");
    let lt = try_create_cublaslt(stream.clone()).unwrap();
    let (m, n, k) = (512, 2, 2880);
    let (a, b) = (data(m * k, true), data(n * k, false));
    let expected: Vec<_> = (0..n)
        .flat_map(|j| {
            let (a, b) = (&a, &b);
            (0..m).map(move |i| {
                bf16::from_f32(
                    (0..k)
                        .map(|z| a[i * k + z].to_f32() * b[j * k + z].to_f32())
                        .sum(),
                )
            })
        })
        .collect();
    let mut padded_a = vec![bf16::ZERO];
    padded_a.extend(a);
    let mut padded_b = vec![bf16::ZERO];
    padded_b.extend(b);
    let da = stream
        .clone_htod(&padded_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        .unwrap();
    let db = stream
        .clone_htod(&padded_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        .unwrap();
    let dc = stream.alloc_zeros::<u16>(m * n + 1).unwrap();
    let ptrs = LtMatmulPointers {
        a: da.device_ptr(&stream).0 + 2,
        b: db.device_ptr(&stream).0 + 2,
        c: dc.device_ptr(&stream).0 + 2,
        d: dc.device_ptr(&stream).0 + 2,
        bias: None,
        a_scale: None,
        b_scale: None,
    };
    let op = CuBlasLt {
        m: m.into(),
        n: n.into(),
        k: k.into(),
        a_layout: cublasOperation_t::CUBLAS_OP_T,
        b_layout: cublasOperation_t::CUBLAS_OP_N,
        lda: k.into(),
        ldb: k.into(),
        ldc: m.into(),
        ldd: m.into(),
        a_dtype: DType::Bf16,
        b_dtype: DType::Bf16,
        c_dtype: DType::Bf16,
        d_dtype: DType::Bf16,
        compute_type: cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        scale_dtype: DType::F32,
        ..Default::default()
    };
    let mut spec = op
        .resolve_matmul_spec(&FxHashMap::default())
        .unwrap()
        .with_pointers(ptrs);
    assert_eq!(spec.alignments, [2; 4]);
    for rank in 0..CUBLASLT_ALGORITHM_CHOICES {
        spec.algorithm_rank = rank;
        let prepared = prepare_cublaslt_matmul(&stream, &lt, &spec, ptrs).unwrap();
        prepared.enqueue(&stream, ptrs).unwrap();
        stream.synchronize().unwrap();
        let result: Vec<_> = stream
            .clone_dtoh(&dc)
            .unwrap()
            .into_iter()
            .map(bf16::from_bits)
            .collect();
        assert_eq!(result[0], bf16::ZERO, "output sentinel at rank {rank}");
        assert_eq!(
            &result[1..],
            expected.as_slice(),
            "rank {rank} must round only after FP32 reduction"
        );
    }
}

#[test]
fn ranked_rewrites_execute_and_rebind_pointer_alignment() {
    let stream = get_cuda_stream().expect("CUDA required for ranked graph regression");
    let (m, n, k) = (2, 16, 32);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k)).as_dtype(DType::Bf16).persist();
    let b = cx.tensor((n, k)).as_dtype(DType::Bf16).persist();
    let out = a.matmul(b.t()).cast(DType::Bf16).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let ad = data(m * k, false);
    let bd = data(n * k, true);
    let expected: Vec<_> = (0..m)
        .flat_map(|i| {
            let (ad, bd) = (&ad, &bd);
            (0..n).map(move |j| {
                bf16::from_f32(
                    (0..k)
                        .map(|z| ad[i * k + z].to_f32() * bd[j * k + z].to_f32())
                        .sum(),
                )
            })
        })
        .collect();
    let mut padded = vec![bf16::ZERO];
    padded.extend(ad.clone());
    let misaligned = stream
        .clone_htod(&padded.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        .unwrap();
    for rank in 0..CUBLASLT_ALGORITHM_CHOICES {
        let llir = try_extract_forced_op_llir_where(
            &cx,
            &["cublaslt_algorithm"],
            ForcedExtractionConfig::new(731).attempts_per_node(64),
            |llir| {
                llir.node_weights().any(|op| {
                    op.to_dialect::<dyn HostOp>()
                        .and_then(|host| host.as_any().downcast_ref::<CuBlasLt>())
                        .is_some_and(|op| op.algorithm_rank == rank)
                })
            },
        )
        .unwrap_or_else(|error| panic!("rank {rank} was not reachable: {error}"));
        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(a, ad.clone());
        rt.set_data(b, bd.clone());
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_bf16(out), expected, "aligned rank {rank}");
        unsafe {
            rt.set_device_ptr(a, misaligned.device_ptr(&stream).0 + 2, ad.len() * 2);
        }
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_bf16(out), expected, "unaligned rank {rank}");
        rt.set_data(a, ad.clone());
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_bf16(out), expected, "restored aligned rank {rank}");
    }
}

#[test]
fn ranked_row_and_column_lowerings_agree_for_wide_outputs() {
    let stream = get_cuda_stream().expect("CUDA required for layout regression");
    let (m, n, k) = (8, 2880, 4096);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k)).as_dtype(DType::Bf16).persist();
    let b = cx.tensor((n, k)).as_dtype(DType::Bf16).persist();
    let out = a.matmul(b.t()).cast(DType::Bf16).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let ad = data(m * k, false);
    let bd = data(n * k, true);
    // Dyadic products and bounded sums are exactly representable in FP32.
    // This catches layout/reduction errors without demanding an unspecified
    // floating-point accumulation order for arbitrary real-valued inputs.
    let expected: Vec<_> = (0..m)
        .flat_map(|i| {
            let (ad, bd) = (&ad, &bd);
            (0..n).map(move |j| {
                bf16::from_f32(
                    (0..k)
                        .map(|z| ad[i * k + z].to_f32() * bd[j * k + z].to_f32())
                        .sum(),
                )
            })
        })
        .collect();
    for order in [
        cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        cublasLtOrder_t::CUBLASLT_ORDER_COL,
    ] {
        for rank in 0..CUBLASLT_ALGORITHM_CHOICES {
            let llir = try_extract_forced_op_llir_where(
                &cx,
                &["cublaslt_algorithm"],
                ForcedExtractionConfig::new(917).attempts_per_node(128),
                |llir| {
                    llir.node_weights().any(|op| {
                        op.to_dialect::<dyn HostOp>()
                            .and_then(|host| host.as_any().downcast_ref::<CuBlasLt>())
                            .is_some_and(|op| op.algorithm_rank == rank && op.d_order == order)
                    })
                },
            )
            .unwrap_or_else(|error| panic!("rank {rank}, order {order:?}: {error}"));
            let mut rt = CudaRuntime::initialize(stream.clone());
            rt.load_llir(&llir);
            rt.set_data(a, ad.clone());
            rt.set_data(b, bd.clone());
            rt.execute(&cx.dyn_map);
            assert_eq!(rt.get_bf16(out), expected, "rank {rank}, order {order:?}");
        }
    }
}

#[test]
fn ranked_scaled_algorithms_preserve_mutable_fp8_scales() {
    let stream = get_cuda_stream().expect("CUDA required for scaled algorithm regression");
    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k)).persist();
    let input_scale = cx.tensor(()).persist();
    let weight_scale = cx.tensor(()).persist();
    let b = cx.tensor((n, k)).as_dtype(DType::F8E4M3).persist();
    let quant = (a / input_scale.expand_rhs((m, k))).cast(DType::F8E4M3);
    let out = (quant.matmul(b.t()).cast(DType::F32)
        * (input_scale * weight_scale).expand_rhs((m, n)))
    .output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let av: Vec<_> = (0..m * k)
        .map(|i| ((i * 3 + i / 7) % 5) as f32 - 2.0)
        .collect();
    let bv: Vec<_> = (0..n * k)
        .map(|i| ((i * 2 + i / 9) % 5) as f32 - 2.0)
        .collect();
    let bytes: Vec<u8> = bv
        .iter()
        .map(|v| [0xC0, 0xB8, 0, 0x38, 0x40][(*v + 2.0) as usize])
        .collect();
    let reference: Vec<f32> = (0..m)
        .flat_map(|i| {
            let (av, bv) = (&av, &bv);
            (0..n).map(move |j| (0..k).map(|z| av[i * k + z] * bv[j * k + z]).sum())
        })
        .collect();
    for rank in 0..CUBLASLT_ALGORITHM_CHOICES {
        let llir = try_extract_forced_op_llir_where(
            &cx,
            &["cublaslt_scaled_algorithm"],
            ForcedExtractionConfig::new(922).attempts_per_node(64),
            |llir| {
                llir.node_weights().any(|op| {
                    op.to_dialect::<dyn HostOp>()
                        .and_then(|host| host.as_any().downcast_ref::<CuBlasLt>())
                        .is_some_and(|op| {
                            op.algorithm_rank == rank && op.a_scale_input && op.b_scale_input
                        })
                })
            },
        )
        .unwrap_or_else(|error| panic!("scaled rank {rank} was not reachable: {error}"));
        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(b, bytes.clone());
        for (xs, ws) in [(0.25, 2.0), (0.5, 1.5)] {
            rt.set_data(a, av.iter().map(|v| v * xs).collect::<Vec<_>>());
            rt.set_data(input_scale, vec![xs]);
            rt.set_data(weight_scale, vec![ws]);
            rt.execute(&cx.dyn_map);
            assert_eq!(
                rt.get_f32(out),
                reference.iter().map(|v| v * xs * ws).collect::<Vec<_>>(),
                "scaled rank {rank}"
            );
        }
    }
}
