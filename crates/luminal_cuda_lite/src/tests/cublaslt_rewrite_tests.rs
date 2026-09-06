use half::bf16;
use luminal::{
    dtype::DType,
    egglog_utils::{ClassId, NodeId, SerializedEGraph, random_initial_choice, validate_choice_set},
    prelude::*,
};
use rand::{SeedableRng, rngs::StdRng};
use std::{collections::BTreeMap, sync::Arc};

use crate::{
    host::{
        CublasLtMatrixOrders, CublasLtScaleValues, CublasLtTransposeOps, CublasLtTypeTuple, HostOp,
        cublaslt::{cublaslt_prepare_count_for_test, reset_cublaslt_prepare_count_for_test},
        cublaslt_c_d_layouts_match, cublaslt_epilogue, cublaslt_matrix_orders,
        cublaslt_scale_values, cublaslt_tensor_scale_inputs, cublaslt_transpose_ops,
        cublaslt_type_tuple,
    },
    runtime::CudaRuntime,
};

use super::utilities::{
    ForcedExtractionConfig, assert_close, extract_llir_for_choices, get_cuda_stream,
    gpu_supports_dtype, op_ir_nodes, random_f32_vec, try_extract_forced_op_llir_where,
};

// Broad cuBLASLt rewrite coverage is intentionally opt-in: these tests rerun the
// egglog optimizer across many layout and epilogue combinations and dominate the
// serialized CUDA unit-test runtime. Use `cargo test -p luminal_cuda_lite -- --ignored`
// when changing cuBLASLt rewrites or extraction logic.

#[derive(Debug, Clone, Copy)]
struct LayoutCase {
    name: &'static str,
    a_col_major: bool,
    b_col_major: bool,
}

const LAYOUT_CASES: [LayoutCase; 4] = [
    LayoutCase {
        name: "row-major x row-major",
        a_col_major: false,
        b_col_major: false,
    },
    LayoutCase {
        name: "row-major x column-major",
        a_col_major: false,
        b_col_major: true,
    },
    LayoutCase {
        name: "column-major x row-major",
        a_col_major: true,
        b_col_major: false,
    },
    LayoutCase {
        name: "column-major x column-major",
        a_col_major: true,
        b_col_major: true,
    },
];

fn row_order_tuple(case: LayoutCase) -> CublasLtMatrixOrders {
    (
        if case.a_col_major { "COL" } else { "ROW" },
        if case.b_col_major { "COL" } else { "ROW" },
        "ROW",
        "ROW",
    )
}

fn case_seed(case: LayoutCase) -> u64 {
    match (case.a_col_major, case.b_col_major) {
        (false, false) => 0,
        (false, true) => 1,
        (true, false) => 2,
        (true, true) => 3,
    }
}

fn gelu_approx(x: f32) -> f32 {
    x / (1.0 + (-(1.595_769_2 * x * (1.0 + 0.044_715 * x * x))).exp())
}

#[derive(Debug, Clone, Copy)]
enum PostOp {
    Identity,
    Relu,
    Gelu,
}

fn apply_postop(x: f32, post: PostOp) -> f32 {
    match post {
        PostOp::Identity => x,
        PostOp::Relu => x.max(0.0),
        PostOp::Gelu => gelu_approx(x),
    }
}

fn fp8_exact_bytes(dtype: DType, len: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
    const VALUES: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let mut bytes = Vec::with_capacity(len);
    let mut decoded = Vec::with_capacity(len);
    for i in 0..len {
        let value = VALUES[(i * 5 + seed) % VALUES.len()];
        bytes.push(match (dtype, value) {
            (DType::F8E4M3, -2.0) => 0xC0,
            (DType::F8E4M3, -1.0) => 0xB8,
            (DType::F8E4M3, -0.5) => 0xB0,
            (DType::F8E4M3, 0.0) => 0x00,
            (DType::F8E4M3, 0.5) => 0x30,
            (DType::F8E4M3, 1.0) => 0x38,
            (DType::F8E4M3, 2.0) => 0x40,
            (DType::F8E5M2, -2.0) => 0xC0,
            (DType::F8E5M2, -1.0) => 0xBC,
            (DType::F8E5M2, -0.5) => 0xB8,
            (DType::F8E5M2, 0.0) => 0x00,
            (DType::F8E5M2, 0.5) => 0x38,
            (DType::F8E5M2, 1.0) => 0x3C,
            (DType::F8E5M2, 2.0) => 0x40,
            _ => panic!("unsupported FP8 exact test value {value} for {dtype:?}"),
        });
        decoded.push(value);
    }
    (bytes, decoded)
}

fn reference_matmul_2d(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut expected = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            expected[row * n + col] = acc;
        }
    }
    expected
}

fn reference_mixed_chain(
    a: &[f32],
    pre: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut expected = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += (a[row * k + inner] + pre[row * k + inner]) * b[inner * n + col];
            }
            expected[row * n + col] = acc.exp();
        }
    }
    expected
}

fn cublaslt_available_for_runtime(stream: &Arc<cudarc::driver::CudaStream>) -> bool {
    crate::try_create_cublaslt(stream.clone()).is_ok()
}

fn build_mixed_chain_graph(
    m: impl Into<Expression>,
    n: usize,
    k: usize,
) -> (Graph, NodeIndex, NodeIndex, NodeIndex, NodeIndex) {
    let m = m.into();
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let pre = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = ((a + pre).matmul(b).exp()).output();
    (cx, a.id, pre.id, b.id, out.id)
}

fn add_in_place(values: &mut [f32], addends: &[f32]) {
    for (value, addend) in values.iter_mut().zip(addends) {
        *value += *addend;
    }
}

fn reference_matmul_2d_layout(
    case: LayoutCase,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut expected = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            expected[row * n + col] = (0..k)
                .map(|inner| {
                    let a_value = if case.a_col_major {
                        a[inner * m + row]
                    } else {
                        a[row * k + inner]
                    };
                    let b_value = if case.b_col_major {
                        b[col * k + inner]
                    } else {
                        b[inner * n + col]
                    };
                    a_value * b_value
                })
                .sum();
        }
    }
    expected
}

fn reference_matmul_2d_plus_strided_c(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    dims: (usize, usize, usize),
    c_layout: (usize, usize, usize),
) -> Vec<f32> {
    let (m, n, k) = dims;
    let (c_row_stride, row_offset, col_offset) = c_layout;
    let mut expected = reference_matmul_2d(a, b, m, n, k);
    for row in 0..m {
        for col in 0..n {
            expected[row * n + col] += c[(row + row_offset) * c_row_stride + col + col_offset];
        }
    }
    expected
}

fn logical_b_from_column_major_storage(storage: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut logical = vec![0.0; k * n];
    for col in 0..n {
        for inner in 0..k {
            logical[inner * n + col] = storage[col * k + inner];
        }
    }
    logical
}

fn logical_b_from_batched_column_major_storage(
    storage: &[f32],
    batch: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut logical = vec![0.0; batch * k * n];
    for batch_idx in 0..batch {
        let storage_offset = batch_idx * n * k;
        let logical_offset = batch_idx * k * n;
        logical[logical_offset..logical_offset + k * n].copy_from_slice(
            &logical_b_from_column_major_storage(
                &storage[storage_offset..storage_offset + n * k],
                n,
                k,
            ),
        );
    }
    logical
}

fn reference_matmul_batched(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut expected = vec![0.0; batch * m * n];
    for batch_idx in 0..batch {
        let a_offset = batch_idx * m * k;
        let b_offset = batch_idx * k * n;
        let out_offset = batch_idx * m * n;
        expected[out_offset..out_offset + m * n].copy_from_slice(&reference_matmul_2d(
            &a[a_offset..a_offset + m * k],
            &b[b_offset..b_offset + k * n],
            m,
            n,
            k,
        ));
    }
    expected
}

fn reference_matmul_batched_plus_strided_c(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    dims: (usize, usize, usize, usize),
    c_layout: (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let (batch, m, n, k) = dims;
    let (c_batch_stride, c_row_stride, batch_offset, row_offset, col_offset) = c_layout;
    let mut expected = reference_matmul_batched(a, b, batch, m, n, k);
    for batch_idx in 0..batch {
        for row in 0..m {
            for col in 0..n {
                let out_idx = batch_idx * m * n + row * n + col;
                let c_idx = (batch_idx + batch_offset) * c_batch_stride
                    + (row + row_offset) * c_row_stride
                    + col
                    + col_offset;
                expected[out_idx] += c[c_idx];
            }
        }
    }
    expected
}

fn reference_scaled_alpha_beta(matmul: &[f32], c: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    matmul
        .iter()
        .zip(c)
        .map(|(acc, c_value)| alpha * acc + beta * c_value)
        .collect()
}

fn reference_column_bias_postop(
    mut matmul: Vec<f32>,
    bias: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    post: PostOp,
) -> Vec<f32> {
    for batch_idx in 0..batch {
        for row in 0..m {
            for (col, bias_value) in bias.iter().enumerate().take(n) {
                let idx = batch_idx * m * n + row * n + col;
                matmul[idx] = apply_postop(matmul[idx] + bias_value, post);
            }
        }
    }
    matmul
}

fn reference_postop(mut values: Vec<f32>, post: PostOp) -> Vec<f32> {
    for value in &mut values {
        *value = apply_postop(*value, post);
    }
    values
}

fn gpu_supports_cublaslt_fp8_launch(dtype: DType) -> bool {
    gpu_supports_cublaslt_fp8_launch_pair(dtype, dtype)
}

fn gpu_supports_cublaslt_fp8_launch_pair(a_dtype: DType, b_dtype: DType) -> bool {
    gpu_supports_dtype(a_dtype)
        && gpu_supports_dtype(b_dtype)
        && matches!(
            (a_dtype, b_dtype),
            (DType::F8E4M3, DType::F8E4M3)
                | (DType::F8E4M3, DType::F8E5M2)
                | (DType::F8E5M2, DType::F8E4M3)
        )
}

const CUBLASLT_FP8_F32_PAIRS: [(DType, DType); 3] = [
    (DType::F8E4M3, DType::F8E4M3),
    (DType::F8E4M3, DType::F8E5M2),
    (DType::F8E5M2, DType::F8E4M3),
];

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_layout_pairs() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(build_2d_matmul_graph(case, DType::F32), case.name, |_| true);
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_layout_pairs() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_batched_matmul_graph(case, DType::F32),
            case.name,
            |_| true,
        );
    }
}

#[test]
fn cublaslt_rewrites_preserve_explicit_same_type_tuple() {
    for (dtype, compute_type, scale_dtype) in [
        (DType::F32, "32F", DType::F32),
        (DType::F16, "32F", DType::F32),
        (DType::Bf16, "32F_FAST_16BF", DType::F32),
    ] {
        let cx = build_2d_matmul_graph(
            LayoutCase {
                name: "dtype smoke",
                a_col_major: false,
                b_col_major: false,
            },
            dtype,
        );
        let llir = extract_forced_cublaslt_llir(cx, &format!("row-major x row-major {dtype:?}"));
        assert_eq!(
            cublaslt_type_tuples(&llir),
            vec![(dtype, dtype, dtype, dtype, compute_type, scale_dtype)],
            "current rewrites should emit explicit A/B/C/D plus default compute/scale types for {dtype:?}"
        );
    }
}

#[test]
fn cublaslt_rewrites_emit_default_scale_values() {
    let cx = build_2d_matmul_graph(
        LayoutCase {
            name: "default alpha beta",
            a_col_major: false,
            b_col_major: false,
        },
        DType::F32,
    );
    let llir = extract_forced_cublaslt_llir(cx, "default alpha beta");
    assert_eq!(
        cublaslt_scale_value_tuples(&llir),
        vec![(1.0, 0.0)],
        "current rewrites should emit alpha=1 and beta=0"
    );
}

#[test]
fn cublaslt_rewrites_emit_default_epilogue() {
    let cx = build_2d_matmul_graph(
        LayoutCase {
            name: "default epilogue",
            a_col_major: false,
            b_col_major: false,
        },
        DType::F32,
    );
    let llir = extract_forced_cublaslt_llir(cx, "default epilogue");
    assert_eq!(
        cublaslt_epilogues(&llir),
        vec!["DEFAULT"],
        "base rewrites should emit the default cuBLASLt epilogue"
    );
}

#[test]
fn cublaslt_rewrites_emit_default_matrix_orders() {
    let col_orders = ("COL", "COL", "COL", "COL");
    assert_cublaslt_rewrite(
        build_2d_matmul_graph(
            LayoutCase {
                name: "default matrix orders",
                a_col_major: false,
                b_col_major: false,
            },
            DType::F32,
        ),
        "default matrix orders",
        |llir| cublaslt_matrix_order_tuples(llir) == vec![col_orders],
    );
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_row_order_layout_pairs() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(build_2d_matmul_graph(case, DType::F32), case.name, |llir| {
            cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
        });
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_row_order_layout_pairs() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_batched_matmul_graph(case, DType::F32),
            case.name,
            |llir| cublaslt_matrix_order_tuples(llir).contains(&expected_orders),
        );
    }
}

#[test]
fn cublaslt_rewrites_cover_flux2_qk_transposed_matmul() {
    let mut cx = Graph::new();
    let q = cx.tensor((8usize, 4usize));
    let k = cx.tensor((8usize, 4usize));
    let _out = q.matmul(k.t()).output();

    assert_cublaslt_rewrite(cx, "flux2 q @ k.t()", |llir| {
        cublaslt_matrix_order_tuples(llir).contains(&("ROW", "COL", "ROW", "ROW"))
            || cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
    });
}

#[test]
fn cublaslt_rewrites_cover_flux2_linear_bias_epilogue() {
    let mut cx = Graph::new();
    let x = cx.tensor((8usize, 4usize));
    let weight = cx.tensor((6usize, 4usize));
    let bias = cx.tensor(6usize);
    let _out = (x.matmul(weight.t()) + bias.expand_dim(0, 8usize)).output();

    assert_cublaslt_epilogue_rewrite(
        cx,
        "flux2 x @ weight.t() + bias",
        "BIAS",
        Some(("COL", "COL", "COL", "COL")),
    );
}

#[test]
fn cublaslt_preserves_legal_matmul_backend_choices() {
    let mut cx = Graph::new();
    let q = cx.tensor((8usize, 4usize));
    let k = cx.tensor((8usize, 4usize));
    let _out = q.matmul(k.t()).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");
    assert!(
        !cublaslt_ir_nodes(egraph).is_empty(),
        "Flux2 q @ k.t() should have at least one cuBLASLt candidate"
    );
    assert!(
        ir_class_has_op_kinds(egraph, &["KernelSum", "GenericMatmul", "cublaslt"]),
        "the lowered reduction, generic matmul, and cuBLASLt implementations should coexist in one output e-class"
    );
}

/// Spell the frontend's 2D matmul decomposition while inserting a real
/// movement permutation between the elementwise product and its reduction.
/// The Mul still has logical shape [m, n, k] and canonical operand strides,
/// but the Sum observes columns in split/transpose/merge order.
fn matmul_with_permuted_reduction_columns(lhs: GraphTensor, rhs: GraphTensor) -> GraphTensor {
    let (m, _) = lhs.dims2();
    let (_, n) = rhs.dims2();
    let product = lhs.expand_dim(1, n) * rhs.permute((1, 0)).expand_dim(0, m);
    product
        .split_dims(1, 2usize)
        .permute((0, 2, 1, 3))
        .merge_dims(1, 2)
        .sum(2)
}

/// Keep a tensor's [m, n] shape while changing its logical column mapping.
fn permute_matmul_columns(tensor: GraphTensor) -> GraphTensor {
    tensor
        .split_dims(1, 2usize)
        .permute((0, 2, 1))
        .merge_dims(1, 2)
}

#[test]
fn cublaslt_and_gemv_reject_permuted_mul_to_sum_layout() {
    let mut cx = Graph::new();
    let activation = cx.tensor((1usize, 5usize)).as_dtype(DType::Bf16);
    let weight = cx.tensor((6usize, 5usize)).as_dtype(DType::Bf16);
    matmul_with_permuted_reduction_columns(activation, weight.t()).output();

    // Mirror GraphTensor::matmul's explicit FP8 -> F32 semantic promotion so
    // the same malformed reduction also exercises the FP8 cuBLASLt/GEMV
    // consumers, not only the ordinary 16-bit families.
    let fp8_activation = cx.tensor((1usize, 5usize)).as_dtype(DType::F8E4M3);
    let fp8_weight = cx.tensor((6usize, 5usize)).as_dtype(DType::F8E4M3);
    matmul_with_permuted_reduction_columns(
        fp8_activation.cast(DType::F32),
        fp8_weight.t().cast(DType::F32),
    )
    .output();

    // Exercise the scaled-quantized spelling consumed by cublaslt_scaled.
    let raw_activation = cx.tensor((1usize, 5usize));
    let input_scale = cx.tensor(());
    let weight_scale = cx.tensor(());
    let quantized = (raw_activation / input_scale.expand_rhs((1usize, 5usize))).cast(DType::F8E4M3);
    let scaled_sum = matmul_with_permuted_reduction_columns(
        quantized.cast(DType::F32),
        fp8_weight.t().cast(DType::F32),
    );
    (scaled_sum * (input_scale * weight_scale).expand_rhs((1usize, 6usize)))
        .cast(DType::Bf16)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        cublaslt_ir_nodes(egraph).is_empty(),
        "cuBLASLt must not replace a reduction whose surviving columns are permuted"
    );
    assert!(
        op_ir_nodes(egraph, "KernelGemv").is_empty(),
        "GEMV must not replace a reduction whose surviving columns are permuted"
    );
    assert!(
        ir_class_has_op_kinds(egraph, &["KernelSum", "GenericMatmul"]),
        "the semantic reduction and GenericMatmul fallback must remain selectable"
    );
}

#[test]
fn cublaslt_post_matmul_fusions_reject_permuted_output_layout() {
    let mut alpha_cx = Graph::new();
    let a = alpha_cx.tensor((4usize, 5usize));
    let b = alpha_cx.tensor((5usize, 6usize));
    (permute_matmul_columns(a.matmul(b)) * 1.5).output();
    assert_no_forced_cublaslt_llir_where(&mut alpha_cx, "permuted matmul alpha", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.5, 0.0))
    });

    let mut beta_cx = Graph::new();
    let a = beta_cx.tensor((4usize, 5usize));
    let b = beta_cx.tensor((5usize, 6usize));
    let c = beta_cx.tensor((4usize, 6usize));
    (permute_matmul_columns(a.matmul(b)) + c).output();
    assert_no_forced_cublaslt_llir_where(&mut beta_cx, "permuted matmul beta", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });
}

#[test]
fn cublaslt_beta_rejects_clamped_c_leading_dimension() {
    let z = Expression::from('z');

    let mut ld_cx = Graph::new();
    let a = ld_cx.tensor((4usize, 5usize));
    let b = ld_cx.tensor((5usize, 6usize));
    let mut c = ld_cx.tensor((4usize, 6usize));
    // Logical rows overlap: ldc=3 is smaller than the six columns. Runtime
    // clamping ldc to six would silently change which C values are added.
    c.shape = ShapeTracker::new_strided((4usize, 6usize), (z * 3usize, z));
    (a.matmul(b) + c).output();
    assert_no_forced_cublaslt_llir_where(&mut ld_cx, "too-small C leading dimension", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });
}

#[test]
fn cublaslt_beta_preserves_batched_broadcast_c() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 4usize, 5usize));
    let b = cx.tensor((2usize, 5usize, 6usize));
    let c = cx.tensor((4usize, 6usize)).expand_dim(0, 2usize);
    (a.matmul(b) + c).output();

    assert_cublaslt_rewrite(cx, "batched broadcast C", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });
}

#[test]
#[ignore = "expensive CUDA rewrite regression; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_layout_witnesses_are_consumer_demanded() {
    let mut cx = Graph::new();

    let a = cx.tensor((4usize, 5usize));
    let b = cx.tensor((5usize, 6usize));
    let sliced_c = cx.tensor((4usize, 9usize)).slice((0..4usize, 0..6usize));
    (a.matmul(b) + sliced_c).output();

    let batched_a = cx.tensor((2usize, 4usize, 5usize));
    let batched_b = cx.tensor((2usize, 5usize, 6usize));
    let batched_c = cx.tensor((2usize, 4usize, 6usize));
    (batched_c * 0.5 + batched_a.matmul(batched_b) * 1.5).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert_relation_facts_are_requested(
        egraph,
        "cublaslt_valid_leading_dimension",
        "cublaslt_leading_dimension_request",
    );
    assert_relation_facts_are_requested(
        egraph,
        "cublaslt_exact_matrix_stride",
        "cublaslt_matrix_stride_request",
    );
}

#[test]
fn cublaslt_epilogues_reject_permuted_output_layout() {
    let mut relu_cx = Graph::new();
    let a = relu_cx.tensor((4usize, 5usize));
    let b = relu_cx.tensor((5usize, 6usize));
    permute_matmul_columns(a.matmul(b)).relu().output();
    assert_no_forced_cublaslt_epilogue_rewrite(relu_cx, "permuted matmul relu", "RELU", None);

    let mut gelu_cx = Graph::new();
    let a = gelu_cx.tensor((4usize, 5usize));
    let b = gelu_cx.tensor((5usize, 6usize));
    permute_matmul_columns(a.matmul(b))
        .gelu_fast_tanh_approximation()
        .output();
    assert_no_forced_cublaslt_epilogue_rewrite(gelu_cx, "permuted matmul gelu", "GELU", None);
}

#[test]
fn cublaslt_rank4_matmuls_do_not_drop_leading_batch_axes() {
    let (outer, batch, m, n, k) = (2usize, 3usize, 7usize, 11usize, 5usize);
    let mut cx = Graph::new();

    // Exercise every physical A/B layout family in one search space. The
    // current cuBLASLt descriptors encode one strided-batch dimension, so a
    // rank-4 matmul must stay on the semantic reduction/GenericMatmul paths
    // until a rewrite explicitly proves that both leading axes can flatten.
    for case in LAYOUT_CASES {
        let a = cx.tensor(if case.a_col_major {
            (outer, batch, k, m)
        } else {
            (outer, batch, m, k)
        });
        let b = cx.tensor(if case.b_col_major {
            (outer, batch, n, k)
        } else {
            (outer, batch, k, n)
        });
        let a = if case.a_col_major {
            a.transpose(2, 3)
        } else {
            a
        };
        let b = if case.b_col_major {
            b.transpose(2, 3)
        } else {
            b
        };
        a.matmul(b).output();
    }

    // Also cover promoted FP8 operands and the scaled-linear spelling. These
    // could otherwise reach both the ordinary F32 and FP8-specific batched
    // rules after GraphTensor::matmul inserts its semantic F32 casts.
    let activation = cx.tensor((outer, batch, m, k));
    let weight_storage = cx.tensor((outer, batch, n, k)).as_dtype(DType::F8E4M3);
    let input_scale = cx.tensor(());
    let weight_scale = cx.tensor(());
    let quantized = (activation / input_scale.expand_rhs((outer, batch, m, k))).cast(DType::F8E4M3);
    let dequant_scale = (input_scale * weight_scale).expand_rhs((outer, batch, m, n));
    (quantized.matmul(weight_storage.transpose(2, 3)) * dequant_scale).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");
    assert!(
        cublaslt_ir_nodes(egraph).is_empty(),
        "a one-axis strided-batch descriptor must not represent rank-4 matmul by dropping the leading batch axis"
    );
    assert!(
        op_ir_nodes(egraph, "GenericMatmul").len() > LAYOUT_CASES.len(),
        "all rank-4 matmuls must retain their semantic GenericMatmul fallback"
    );
    assert!(
        op_ir_nodes(egraph, "KernelSum").len() > LAYOUT_CASES.len(),
        "all rank-4 matmuls must retain their lowered reduction fallback"
    );
}

#[test]
fn cublaslt_rewrites_keep_c_and_d_layouts_equal_initially() {
    for case in LAYOUT_CASES {
        let cx = build_batched_matmul_graph(case, DType::F32);
        let llir = extract_forced_cublaslt_llir(cx, case.name);
        assert_eq!(
            cublaslt_c_d_layout_matches(&llir),
            vec![true],
            "current rewrites should emit identical C and D ld/stride for {}",
            case.name
        );
    }
}

#[test]
fn mixed_cuda_graph_cublaslt_kernel_chain_executes_correctly() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (m, n, k) = (7, 11, 5);
    let (mut cx, a, pre, b, out) = build_mixed_chain_graph(m, n, k);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    // Fixed-seed forcing is background-sensitive; this graph needs a few
    // attempts to land a background that keeps the plain-scale cuBLASLt
    // reachable now that the fusion alternatives are gone.
    let llir = try_extract_forced_op_llir_where(
        &cx,
        &["cublaslt", "cublaslt_scaled"],
        ForcedExtractionConfig::new(0x00C0_B1A5).attempts_per_node(8),
        |llir| {
            !cublaslt_type_tuples(llir).is_empty()
                && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
        },
    )
    .unwrap_or_else(|error| {
        panic!("expected to extract a CuBlasLt HostOp for mixed graph chain: {error}")
    });

    let a_data = random_f32_vec(m * k, 0xCAFE_0001, -0.08, 0.08);
    let pre_data = random_f32_vec(m * k, 0xCAFE_0002, -0.03, 0.03);
    let b_data = random_f32_vec(k * n, 0xCAFE_0003, -0.08, 0.08);
    let expected = reference_mixed_chain(&a_data, &pre_data, &b_data, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(pre, pre_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
    let summaries = rt.debug_cuda_graph_summaries();
    let mixed = summaries
        .iter()
        .find(|summary| summary.n_cublaslt == 1)
        .expect("expected one CudaGraphOp to capture the cuBLASLt island");
    assert!(mixed.n_kernels >= 2, "expected kernels around cuBLASLt");
    assert_eq!(mixed.n_steps, mixed.n_kernels + mixed.n_cublaslt);
    assert_eq!(mixed.absorbed_host_nodes.len(), 1);
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn cuda_graph_cublaslt_only_executes_correctly() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "cuBLASLt-only graph", |_| true);

    let a_data = random_f32_vec(m * k, 0xC001_0001, -0.08, 0.08);
    let b_data = random_f32_vec(k * n, 0xC001_0002, -0.08, 0.08);
    let expected = reference_matmul_2d(&a_data, &b_data, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a.id, a_data);
    rt.set_data(b.id, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    let summary = rt
        .debug_cuda_graph_summaries()
        .into_iter()
        .find(|summary| summary.n_cublaslt == 1)
        .expect("expected a cuBLASLt-only CudaGraphOp");
    assert_eq!(summary.n_kernels, 0);
    assert_eq!(summary.n_steps, 1);
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn cuda_graph_cublaslt_bf16_output_keeps_storage_dtype() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream)
        || !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream)
    {
        return;
    }

    let (m, n, k) = (1, 96, 64);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k)).as_dtype(DType::Bf16);
    let b = cx.tensor((k, n)).as_dtype(DType::Bf16);
    let out = a.matmul(b).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "BF16 cuBLASLt output dtype", |llir| {
        cublaslt_type_tuples(llir)
            .iter()
            .any(|types| types.3 == DType::Bf16)
    });

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, vec![bf16::from_f32(0.25); m * k]);
    rt.set_data(b, vec![bf16::from_f32(0.5); k * n]);
    rt.execute(&cx.dyn_map);

    let result = rt.get_bf16(out);
    assert_eq!(result.len(), m * n);
    assert!(result.iter().all(|value| value.to_f32() == 8.0));
}

#[test]
fn mixed_cuda_graph_reuses_prepared_for_ordered_matching_cublaslt_ops() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (n, k) = (8, 8);
    let mut cx = Graph::new();
    let a = cx.tensor(('m', k));
    let b = cx.tensor((k, n));
    let first = a.matmul(b);
    let out = (a + first.sin()).matmul(b).output();
    cx.set_dim('m', 5);
    let llir = extract_forced_distinct_cublaslt_classes_llir_where(
        &mut cx,
        "ordered matching cuBLASLt prepared reuse",
        |llir| {
            let orders = cublaslt_matrix_order_tuples(llir);
            orders.len() == 2 && orders[0] == orders[1]
        },
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    for (m, seed) in [(5, 0xC001_1001), (7, 0xC001_1011)] {
        cx.set_dim('m', m);
        let a_data = random_f32_vec(m * k, seed, -0.08, 0.08);
        let b_data = random_f32_vec(k * n, seed + 1, -0.08, 0.08);
        let first = reference_matmul_2d(&a_data, &b_data, m, n, k);
        let dep = a_data
            .iter()
            .zip(&first)
            .map(|(a, first)| a + first.sin())
            .collect::<Vec<_>>();
        let expected = reference_matmul_2d(&dep, &b_data, m, n, k);

        rt.set_data(a.id, a_data);
        rt.set_data(b.id, b_data);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    }
    let summary = rt
        .debug_cuda_graph_summaries()
        .into_iter()
        .find(|summary| summary.n_cublaslt == 2)
        .expect("expected one mixed CudaGraphOp with two cuBLASLt islands");
    assert_eq!(
        summary.n_cublaslt_prepared, 1,
        "dependency-ordered matching cuBLASLt calls should share prepared resources"
    );
    assert_eq!(
        summary.cublaslt_capture_counts,
        vec![2, 2],
        "both ordered cuBLASLt children should recapture after the shared dynamic shape changes"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn cuda_graph_cublaslt_skips_prepare_when_unrelated_dyn_dim_changes() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();
    a.output();
    b.output();
    cx.set_dim('p', 1);
    let llir = extract_forced_cublaslt_llir_where(
        &mut cx,
        "cuBLASLt unchanged under unrelated dyn dim",
        |_| true,
    );

    let a_data = random_f32_vec(m * k, 0xC004_0001, -0.08, 0.08);
    let b_data = random_f32_vec(k * n, 0xC004_0002, -0.08, 0.08);
    let expected = reference_matmul_2d(&a_data, &b_data, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a.id, a_data);
    rt.set_data(b.id, b_data);

    reset_cublaslt_prepare_count_for_test();
    rt.execute(&cx.dyn_map);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    let first_prepare_count = cublaslt_prepare_count_for_test();
    assert!(
        first_prepare_count > 0,
        "first execution should prepare the captured cuBLASLt island"
    );

    cx.set_dim('p', 2);
    rt.execute(&cx.dyn_map);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    assert_eq!(
        cublaslt_prepare_count_for_test(),
        first_prepare_count,
        "unrelated dyn dim changes should not redo expensive cuBLASLt prepare"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn cuda_graph_cublaslt_only_recaptures_on_dynamic_shape_change() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (n, k) = (11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor(('m', k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();
    // Keep the inputs alive so this test isolates dynamic-shape changes from
    // pointer changes. Pointer-driven recapture is covered separately.
    a.output();
    b.output();
    cx.set_dim('m', 7);
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "cuBLASLt-only dynamic graph", |_| true);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    let mut workspace_ptr = None;
    let cases = [
        (7usize, 0xC002_0001),
        (9usize, 0xC002_0002),
        (7usize, 0xC002_0003),
    ];
    let max_m = cases.iter().map(|(m, _)| *m).max().unwrap();
    rt.set_data_with_capacity(
        a.id,
        Vec::<f32>::new(),
        max_m * k * std::mem::size_of::<f32>(),
    );
    rt.set_data_with_capacity(b.id, Vec::<f32>::new(), k * n * std::mem::size_of::<f32>());

    for (m, seed) in cases {
        cx.set_dim('m', m);
        let a_data = random_f32_vec(m * k, seed, -0.08, 0.08);
        let b_data = random_f32_vec(k * n, seed + 10, -0.08, 0.08);
        let expected = reference_matmul_2d(&a_data, &b_data, m, n, k);

        rt.set_data(a.id, a_data);
        rt.set_data(b.id, b_data);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
        let current_workspace_ptr = rt
            .debug_cuda_graph_summaries()
            .into_iter()
            .find(|summary| summary.n_cublaslt == 1)
            .and_then(|summary| summary.cublaslt_workspace_ptrs.into_iter().next())
            .expect("captured cuBLASLt should retain a prepared workspace");
        if let Some(previous_workspace_ptr) = workspace_ptr {
            assert_eq!(
                current_workspace_ptr, previous_workspace_ptr,
                "dynamic-shape recapture should recycle the prepared workspace"
            );
        }
        workspace_ptr = Some(current_workspace_ptr);
    }

    let summary = rt
        .debug_cuda_graph_summaries()
        .into_iter()
        .find(|summary| summary.n_cublaslt == 1)
        .expect("expected a cuBLASLt-only CudaGraphOp after recapture");
    assert_eq!(summary.n_kernels, 0);
    assert_eq!(summary.n_steps, 1);
    assert_eq!(
        summary.cublaslt_capture_counts,
        vec![2],
        "m=7 should reuse its first captured child graph after visiting m=9"
    );
    assert_eq!(summary.cublaslt_capture_cache_hits, vec![1]);
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

/// Gemma profiles attention at (s=16, c=512), then prefills 19 tokens.
/// Updating the captured BF16 child graph across that change must update the
/// executable as well. On B300, cuGraphExecUpdate can report success while
/// the old executable returns zeros; a fresh executable gives the reference.
#[test]
fn cuda_graph_bf16_batched_matmul_profile_to_prefill_transition() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream)
        || !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream)
    {
        return;
    }
    let (heads, width) = (8usize, 256usize);
    let mut cx = Graph::new();
    let probabilities = cx.tensor((heads, 's', 'c')).as_dtype(DType::Bf16);
    let values = cx.tensor((heads, 'c', width)).as_dtype(DType::Bf16);
    let out = probabilities.matmul(values).output();
    cx.set_dim('s', 16);
    cx.set_dim('c', 512);
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "Gemma BF16 attention transition", |llir| {
            llir.node_weights()
                .filter_map(|op| op.to_dialect::<dyn HostOp>())
                .any(|op| {
                    cublaslt_matrix_orders(op.as_ref().as_ref())
                        == Some(("COL", "COL", "COL", "COL"))
                        && cublaslt_transpose_ops(op.as_ref().as_ref()) == Some(("N", "N"))
                })
        });
    let mut rt = CudaRuntime::initialize(stream);
    // Stable, generously sized bindings isolate the shape change from pointer
    // relocation. Each probability row selects one value row, so the expected
    // result is independent of matmul accumulation/rounding details.
    let mut probs = vec![bf16::ZERO; heads * 64 * 512];
    let mut vals = vec![bf16::ZERO; heads * 512 * width];
    for head in 0..heads {
        for row in 0..19 {
            probs[(head * 19 + row) * 19 + row] = bf16::ONE;
            for col in 0..width {
                vals[(head * 19 + row) * width + col] =
                    bf16::from_f32(((head * 17 + row * 3 + col) % 31) as f32 / 16.0 - 1.0);
            }
        }
    }
    let expected = vals[..heads * 19 * width].to_vec();
    rt.set_data(probabilities, probs);
    rt.set_data(values, vals);
    rt.load_llir_buckets(
        &FxHashMap::default(),
        &[(FxHashMap::default(), cx.dyn_map.clone(), llir)],
    );
    cx.set_dim('s', 19);
    cx.set_dim('c', 19);
    rt.execute(&cx.dyn_map);
    let actual = rt.get_bf16(out);
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b);
    assert!(
        mismatch.is_none(),
        "captured BF16 attention mismatch after profile-to-prefill shape change: {mismatch:?}"
    );
}

#[test]
fn cublaslt_with_dynamic_c_spec_is_captured() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }

    let (n, k) = (11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor(('c', k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();
    cx.set_dim('c', 7);
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "dynamic c cuBLASLt graph", |_| true);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);

    for (c, seed) in [(7usize, 0xC003_0001), (9usize, 0xC003_0002)] {
        cx.set_dim('c', c);
        let a_data = random_f32_vec(c * k, seed, -0.08, 0.08);
        let b_data = random_f32_vec(k * n, seed + 10, -0.08, 0.08);
        let expected = reference_matmul_2d(&a_data, &b_data, c, n, k);

        rt.set_data(a.id, a_data);
        rt.set_data(b.id, b_data);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    }

    assert!(
        rt.debug_cuda_graph_summaries()
            .iter()
            .any(|summary| summary.n_cublaslt == 1),
        "c-dependent cuBLASLt should be absorbed into a CUDA graph"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn bucket_range_and_singleton_cublaslt_buckets_are_captured() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (n, k) = (11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor(('s', k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).output();
    cx.set_dim('s', 1);
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "bucketed s cuBLASLt graph capture", |_| true);

    let dim_buckets = [(
        Symbol::from('s'),
        vec![DimBucket::new(1, 1), DimBucket::new(2, 4)],
    )]
    .into_iter()
    .collect();
    let bucket_llirs = vec![
        (
            [(Symbol::from('s'), 0usize)].into_iter().collect(),
            [(Symbol::from('s'), 1usize)].into_iter().collect(),
            llir.clone(),
        ),
        (
            [(Symbol::from('s'), 1usize)].into_iter().collect(),
            [(Symbol::from('s'), 3usize)].into_iter().collect(),
            llir,
        ),
    ];

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir_buckets(&dim_buckets, &bucket_llirs);

    cx.set_dim('s', 1);
    let a_data = random_f32_vec(k, 0xB001_0001, -0.08, 0.08);
    let b_data = random_f32_vec(k * n, 0xB001_0002, -0.08, 0.08);
    let expected = reference_matmul_2d(&a_data, &b_data, 1, n, k);
    rt.set_data(a.id, a_data);
    rt.set_data(b.id, b_data.clone());
    rt.execute(&cx.dyn_map);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    assert!(
        rt.debug_cuda_graph_summaries()
            .iter()
            .any(|summary| summary.n_cublaslt == 1),
        "singleton s bucket should capture s-dependent cuBLASLt"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
    assert!(
        rt.debug_active_bucket_stabilizes_intermediate_pointers(),
        "bucket with captured cuBLASLt needs stable intermediate pointers"
    );

    cx.set_dim('s', 3);
    let a_data = random_f32_vec(3 * k, 0xB001_0003, -0.08, 0.08);
    let expected = reference_matmul_2d(&a_data, &b_data, 3, n, k);
    rt.set_data(a.id, a_data);
    rt.set_data(b.id, b_data);
    rt.execute(&cx.dyn_map);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    assert!(
        rt.debug_cuda_graph_summaries()
            .iter()
            .any(|summary| summary.n_cublaslt == 1),
        "range s bucket should capture s-dependent cuBLASLt"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
    assert!(
        rt.debug_active_bucket_stabilizes_intermediate_pointers(),
        "bucket with captured cuBLASLt needs stable intermediate pointers"
    );
}

#[test]
fn warmup_cublaslt_preserves_exact_dynamic_bucket_values() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream)
        || !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream)
    {
        return;
    }
    let n = 11;
    let mut cx = Graph::new();
    let a = cx.tensor(('s', 'c')).persist();
    let b = cx.tensor(('c', n)).persist();
    let out = a.matmul(b).output();
    cx.set_dim('s', 2);
    cx.set_dim('c', 16);
    // Exercise the unfused alternative regardless of which implementation
    // happens to win a performance search on the test GPU.
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "dynamic warmup", |_| true);
    let dim_buckets = [
        (
            Symbol::from('s'),
            vec![DimBucket::new(1, 4), DimBucket::new(5, 8)],
        ),
        (
            Symbol::from('c'),
            vec![DimBucket::new(1, 32), DimBucket::new(33, 64)],
        ),
    ]
    .into_iter()
    .collect();
    let mut buckets = Vec::new();
    for (si, s) in [(0, 2), (1, 6)] {
        for (ci, c) in [(0, 16), (1, 48)] {
            buckets.push((
                [(Symbol::from('s'), si), (Symbol::from('c'), ci)]
                    .into_iter()
                    .collect(),
                [(Symbol::from('s'), s), (Symbol::from('c'), c)]
                    .into_iter()
                    .collect(),
                llir.clone(),
            ));
        }
    }
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data_with_capacity(a, vec![0.0f32; 8 * 64], 8 * 64 * 4);
    rt.set_data_with_capacity(b, vec![0.0f32; 64 * n], 64 * n * 4);
    rt.load_llir_buckets(&dim_buckets, &buckets);
    // c is capture-sensitive in cuBLASLt, but is deliberately not a cache
    // key. Startup must not demand knowledge of the selected op's dependencies.
    rt.begin_cuda_graph_warmup(&['s'.into()]);
    for s in [2, 6] {
        for c in [16, 48] {
            cx.set_dim('s', s);
            cx.set_dim('c', c);
            rt.set_data(a, vec![0.0f32; s * c]);
            rt.set_data(b, vec![0.0f32; c * n]);
            rt.prepare_cuda_graphs(&cx.dyn_map).unwrap();
        }
    }
    // Explicit strict freezing must still reject this incomplete cache key.
    assert!(
        rt.finish_cuda_graph_preparation()
            .unwrap_err()
            .to_string()
            .contains("uncached capture-sensitive dimensions")
    );
    let cases = [
        (2, 7),
        (2, 19),
        (2, 7), // same cached variant: grow, shrink, reset
        (1, 1),
        (4, 32),
        (5, 33),
        (8, 64), // both bucket boundaries
        (3, 7),
        (7, 51),
        (2, 19),
        (1, 1), // unseen and reused cached values
    ];
    for (iteration, (s, c)) in cases.into_iter().enumerate() {
        cx.set_dim('s', s);
        cx.set_dim('c', c);
        let seed = 0xC004_0000 + iteration as u64;
        let av = random_f32_vec(s * c, seed, -0.08, 0.08);
        let bv = random_f32_vec(c * n, seed + 100, -0.08, 0.08);
        let expected = reference_matmul_2d(&av, &bv, s, n, c);
        rt.set_data(a, av);
        rt.set_data(b, bv);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
        assert!(
            rt.debug_cuda_graph_summaries()
                .iter()
                .any(|summary| summary.n_cublaslt == 1)
        );
    }
    // Dimension changes alone must invalidate the affected library plan,
    // even when persistent input pointers and contents are unchanged.
    let av = random_f32_vec(8 * 64, 0xC004_1000, -0.08, 0.08);
    let bv = random_f32_vec(64 * n, 0xC004_1001, -0.08, 0.08);
    rt.set_data(a, av.clone());
    rt.set_data(b, bv.clone());
    for (s, c) in [(2, 16), (2, 7), (2, 19), (2, 7)] {
        cx.set_dim('s', s);
        cx.set_dim('c', c);
        let expected = reference_matmul_2d(&av[..s * c], &bv[..c * n], s, n, c);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
    }
}

#[test]
fn resident_cublaslt_variants_switch_without_recapture() {
    check_cublaslt_variants_switch_without_recapture(true);
}

#[test]
fn warmup_cublaslt_variants_switch_without_recapture() {
    check_cublaslt_variants_switch_without_recapture(false);
}

fn check_cublaslt_variants_switch_without_recapture(strict: bool) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream)
        || !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream)
    {
        return;
    }
    let (k, n) = (5, 11);
    let mut cx = Graph::new();
    let a = cx.tensor(('s', k)).persist();
    let b = cx.tensor((k, n)).persist();
    let out = a.matmul(b).output();
    cx.set_dim('s', 4);
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "resident cuBLASLt variants", |_| true);
    let dim_buckets = [(
        Symbol::from('s'),
        vec![DimBucket::new(1, 1), DimBucket::new(2, 4)],
    )]
    .into_iter()
    .collect();
    let buckets = [(0, 1), (1, 4)].map(|(index, size)| {
        (
            [(Symbol::from('s'), index)].into_iter().collect(),
            [(Symbol::from('s'), size)].into_iter().collect(),
            llir.clone(),
        )
    });
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data_with_capacity(a, vec![0.0f32; 4 * k], 4 * k * 4);
    let b_values = random_f32_vec(k * n, 17, -0.1, 0.1);
    rt.set_data(b, b_values.clone());
    rt.load_llir_buckets(&dim_buckets, &buckets);
    if strict {
        rt.begin_cuda_graph_preparation(&['s'.into()]);
    } else {
        rt.begin_cuda_graph_warmup(&['s'.into()]);
    }
    assert!(rt.debug_retained_host_memory_bytes() >= 2 * 32 * 1024 * 1024);
    for s in [4, 2, 1] {
        cx.set_dim('s', s);
        rt.set_data(a, vec![0.0f32; s * k]);
        rt.prepare_cuda_graphs(&cx.dyn_map).unwrap();
    }
    if strict {
        rt.finish_cuda_graph_preparation().unwrap();
    }
    // These tiny GEMMs do not need the 32 MiB workspace preference limit.
    // Final residency validation must charge their actual allocations, even
    // with several retained shapes, instead of multiplying that limit.
    assert!(rt.debug_retained_host_memory_bytes() < 32 * 1024 * 1024);
    let baseline = rt.cuda_graph_residency_stats();
    let arena = rt.intermediate_buffer_bytes();
    assert_eq!(baseline.1, 3);
    for iteration in 0..24 {
        let s = [1, 4, 2, 4, 1, 2][iteration % 6];
        cx.set_dim('s', s);
        let values = random_f32_vec(s * k, 123 + iteration as u64, -0.1, 0.1);
        let expected = reference_matmul_2d(&values, &b_values, s, n, k);
        rt.set_data(a, values);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
        assert_eq!(rt.cuda_graph_residency_stats(), baseline);
        assert_eq!(rt.intermediate_buffer_bytes(), arena);
    }
    cx.set_dim('s', 3);
    rt.set_data(a, vec![0.0f32; 3 * k]);
    if strict {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.execute(&cx.dyn_map)))
                .is_err()
        );
        assert_eq!(rt.cuda_graph_residency_stats(), baseline);
    } else {
        rt.execute(&cx.dyn_map);
        assert_eq!(rt.get_f32(out), vec![0.0; 3 * n]);
    }
}

#[test]
fn mixed_cuda_graph_cublaslt_recaptures_on_input_pointer_change() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (m, n, k) = (7, 11, 5);
    let (mut cx, a, pre, b, out) = build_mixed_chain_graph(m, n, k);
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "mixed graph pointer recapture", |_| true);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);

    reset_cublaslt_prepare_count_for_test();
    let mut first_prepare_count = None;
    for seed in [0xCC00_0001, 0xCC00_0002] {
        let a_data = random_f32_vec(m * k, seed, -0.08, 0.08);
        let pre_data = random_f32_vec(m * k, seed + 10, -0.03, 0.03);
        let b_data = random_f32_vec(k * n, seed + 20, -0.08, 0.08);
        let expected = reference_mixed_chain(&a_data, &pre_data, &b_data, m, n, k);

        rt.set_data(a, a_data);
        rt.set_data(pre, pre_data);
        rt.set_data(b, b_data);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
        if first_prepare_count.is_none() {
            first_prepare_count = Some(cublaslt_prepare_count_for_test());
        }
    }
    assert_eq!(
        cublaslt_prepare_count_for_test(),
        first_prepare_count.unwrap(),
        "A/B/C/D pointer-only recapture should reuse prepared cuBLASLt resources"
    );

    let summaries = rt.debug_cuda_graph_summaries();
    assert!(
        summaries.iter().any(|summary| summary.n_cublaslt == 1),
        "expected cuBLASLt to remain captured after pointer recapture"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
fn mixed_cuda_graph_cublaslt_recaptures_on_dynamic_shape_change() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !cublaslt_available_for_runtime(&stream) {
        return;
    }
    if !crate::host::cublaslt::cublaslt_graph_capture_supported(&stream) {
        return;
    }

    let (n, k) = (11, 5);
    let (mut cx, a, pre, b, out) = build_mixed_chain_graph('m', n, k);
    cx.set_dim('m', 7);
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "mixed graph dynamic recapture", |_| true);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);

    for (m, seed) in [(7usize, 0xDD00_0001), (9usize, 0xDD00_0002)] {
        cx.set_dim('m', m);
        let a_data = random_f32_vec(m * k, seed, -0.08, 0.08);
        let pre_data = random_f32_vec(m * k, seed + 10, -0.03, 0.03);
        let b_data = random_f32_vec(k * n, seed + 20, -0.08, 0.08);
        let expected = reference_mixed_chain(&a_data, &pre_data, &b_data, m, n, k);

        rt.set_data(a, a_data);
        rt.set_data(pre, pre_data);
        rt.set_data(b, b_data);
        rt.execute(&cx.dyn_map);
        assert_close(&rt.get_f32(out), &expected, 1e-5, 1e-5);
    }

    let summaries = rt.debug_cuda_graph_summaries();
    assert!(
        summaries.iter().any(|summary| summary.n_cublaslt == 1),
        "expected cuBLASLt to remain captured after dynamic-shape recapture"
    );
    assert_eq!(rt.debug_standalone_cublaslt_host_ops(), 0);
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_c_beta_one() {
    for case in LAYOUT_CASES {
        let cx = build_2d_matmul_plus_c_graph(case, DType::F32, false);
        assert_cublaslt_rewrite(cx, case.name, |llir| {
            cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                && cublaslt_c_d_layout_matches(llir).contains(&true)
        });
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_c_graph(case, DType::F32, true),
            case.name,
            |llir| cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0)),
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_c_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_c_graph(case, DType::F32, false),
            case.name,
            |llir| cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0)),
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_c_graph(case, DType::F32, true),
            case.name,
            |llir| cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0)),
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_row_order_matmul_plus_c_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_row_order_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_c_graph(case, DType::F32, true),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_row_order_matmul_plus_c_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_row_order_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_c_graph(case, DType::F32, true),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_sliced_c_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_sliced_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_sliced_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_sliced_c_graph(case, DType::F32, true),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_sliced_c_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_sliced_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_sliced_c_plus_matmul_beta_one() {
    for case in LAYOUT_CASES {
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_sliced_c_graph(case, DType::F32, true),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_row_order_matmul_plus_sliced_c_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_2d_matmul_plus_sliced_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_row_order_matmul_plus_sliced_c_beta_one() {
    for case in LAYOUT_CASES {
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_batched_matmul_plus_sliced_c_graph(case, DType::F32, false),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                    && cublaslt_c_d_layout_matches(llir).contains(&false)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_fuse_2d_transposed_c_beta_one() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            let mut cx = build_2d_matmul_plus_transposed_c_graph(case, DType::F32, commuted);
            assert_no_forced_cublaslt_llir_where(&mut cx, case.name, |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            });
        }
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_fuse_batched_transposed_c_beta_one() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            let mut cx = build_batched_matmul_plus_transposed_c_graph(case, DType::F32, commuted);
            assert_no_forced_cublaslt_llir_where(&mut cx, case.name, |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            });
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_offset_c_beta_one() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_rewrite(
                build_2d_matmul_plus_offset_c_graph(case, DType::F32, commuted),
                case.name,
                |llir| {
                    cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                        && cublaslt_c_d_layout_matches(llir).contains(&true)
                },
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_offset_c_beta_one() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_rewrite(
                build_batched_matmul_plus_offset_c_graph(case, DType::F32, commuted),
                case.name,
                |llir| {
                    cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                        && cublaslt_c_d_layout_matches(llir).contains(&true)
                },
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_scaled_alpha_beta() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            let expected_orders = row_order_tuple(case);
            assert_cublaslt_rewrite(
                build_2d_scaled_alpha_beta_graph(case, DType::F32, commuted),
                case.name,
                |llir| {
                    cublaslt_scale_value_tuples(llir).contains(&(1.5, 0.5))
                        && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                },
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_scaled_alpha_beta() {
    for commuted in [false, true] {
        let case = LayoutCase {
            name: "batched row-major scaled alpha beta",
            a_col_major: false,
            b_col_major: false,
        };
        let expected_orders = row_order_tuple(case);
        assert_cublaslt_rewrite(
            build_batched_scaled_alpha_beta_graph(case, DType::F32, commuted),
            case.name,
            |llir| {
                cublaslt_scale_value_tuples(llir).contains(&(1.5, 0.5))
                    && cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA FP8 rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_supported_pairs_execute_2d_matmul_f32_output() {
    for (a_dtype, b_dtype) in CUBLASLT_FP8_F32_PAIRS {
        cublaslt_fp8_candidate_executes_2d_matmul_f32_output(a_dtype, b_dtype);
    }
}

#[test]
#[ignore = "expensive CUDA FP8 rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_supported_pairs_execute_batched_matmul_f32_output() {
    for (a_dtype, b_dtype) in CUBLASLT_FP8_F32_PAIRS {
        cublaslt_fp8_candidate_executes_batched_matmul_f32_output(a_dtype, b_dtype);
    }
}

#[test]
#[ignore = "expensive CUDA FP8 rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_e5m2_same_type_does_not_match_f32_output() {
    cublaslt_fp8_same_type_does_not_match_2d_matmul_f32_output(DType::F8E5M2);
    cublaslt_fp8_same_type_does_not_match_batched_matmul_f32_output(DType::F8E5M2);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_e4m3_beta_candidate_executes_2d_matmul_plus_f32_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_cublaslt_fp8_launch(DType::F8E4M3) {
        return;
    }

    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k)).as_dtype(DType::F8E4M3);
    let b_input = cx.tensor((n, k)).as_dtype(DType::F8E4M3);
    let b = b_input.t();
    let c = cx.tensor((m, n)).as_dtype(DType::F32);
    let out = (a.matmul(b).cast(DType::F32) + c).output();
    let expected_tuple = (
        DType::F8E4M3,
        DType::F8E4M3,
        DType::F32,
        DType::F32,
        "32F",
        DType::F32,
    );
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional fp8 beta", |llir| {
        cublaslt_type_tuples(llir).contains(&expected_tuple)
            && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            && cublaslt_transpose_op_tuples(llir).contains(&("T", "N"))
            && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
    });

    let (a_bytes, a_values) = fp8_exact_bytes(DType::F8E4M3, m * k, 1);
    let (b_bytes, b_storage_values) = fp8_exact_bytes(DType::F8E4M3, k * n, 3);
    let b_values = logical_b_from_column_major_storage(&b_storage_values, n, k);
    let c_data = random_f32_vec(m * n, 0xF800_C0DE, -0.5, 0.5);
    let mut expected = reference_matmul_2d(&a_values, &b_values, m, n, k);
    for (value, c_value) in expected.iter_mut().zip(&c_data) {
        *value += *c_value;
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_bytes);
    rt.set_data(b_input, b_bytes);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA FP8 rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_scaled_candidate_executes_2d_matmul_f32_output() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_cublaslt_fp8_launch(DType::F8E4M3) {
        return;
    }

    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let a_scale = cx.tensor(());
    let b_scale = cx.tensor(());
    let b_input = cx.tensor((n, k)).as_dtype(DType::F8E4M3);
    let b = b_input.t();
    let scaled_a = (a / a_scale.expand_rhs((m, k))).cast(DType::F8E4M3);
    let out =
        (scaled_a.matmul(b).cast(DType::F32) * (a_scale * b_scale).expand_rhs((m, n))).output();
    let expected_tuple = (
        DType::F8E4M3,
        DType::F8E4M3,
        DType::F32,
        DType::F32,
        "32F",
        DType::F32,
    );
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional scaled fp8", |llir| {
        cublaslt_type_tuples(llir).contains(&expected_tuple)
            && cublaslt_tensor_scale_input_tuples(llir).contains(&(true, true))
            && cublaslt_transpose_op_tuples(llir).contains(&("T", "N"))
            && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
    });

    let input_scale = 0.25f32;
    let weight_scale = 2.0f32;
    let (a_fp8_bytes, a_values) = fp8_exact_bytes(DType::F8E4M3, m * k, 7);
    let a_data = a_values
        .iter()
        .map(|value| value * input_scale)
        .collect::<Vec<_>>();
    let (b_bytes, b_storage_values) = fp8_exact_bytes(DType::F8E4M3, k * n, 9);
    let b_values = logical_b_from_column_major_storage(&b_storage_values, n, k);
    let mut expected = reference_matmul_2d(&a_values, &b_values, m, n, k);
    for value in &mut expected {
        *value *= input_scale * weight_scale;
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(a_scale, vec![input_scale]);
    rt.set_data(b_scale, vec![weight_scale]);
    rt.set_data(b_input, b_bytes);
    rt.execute(&cx.dyn_map);

    // Keep the raw bytes live in the test construction: a_data was chosen so
    // the explicit scaled cast quantizes to these exact FP8 values.
    assert_eq!(a_fp8_bytes.len(), m * k);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
fn cublaslt_fp8_scaled_candidate_reaches_fused_output_scale_consumer() {
    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let a_scale = cx.tensor(());
    let b_scale = cx.tensor(());
    let b_input = cx.tensor((n, k)).as_dtype(DType::F8E4M3);
    let b = b_input.t();
    let side = cx.tensor((m, n));
    let scaled_a = (a / a_scale.expand_rhs((m, k))).cast(DType::F8E4M3);
    let scaled_out = scaled_a.matmul(b).cast(DType::F32) * (a_scale * b_scale).expand_rhs((m, n));
    (scaled_out * side).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        dataflow_reachable_cublaslt_scaled_count(egraph) > 0,
        "scaled cuBLASLt must remain reachable when fusion growth consumes the output-scale multiply internally"
    );
    assert!(
        dataflow_reachable_cublaslt_raw_fp8_count(egraph) > 0,
        "the raw FP8 matmul plus output scaling must remain reachable beside the scaled cuBLASLt implementation"
    );
}

#[test]
fn cublaslt_fp8_scaled_candidates_reach_fused_mlp_consumer() {
    let (m, n, k) = (16, 32, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let gate_input_scale = cx.tensor(());
    let gate_weight_scale = cx.tensor(());
    let up_input_scale = cx.tensor(());
    let up_weight_scale = cx.tensor(());
    let gate_weight = cx.tensor((n, k)).as_dtype(DType::F8E4M3);
    let up_weight = cx.tensor((n, k)).as_dtype(DType::F8E4M3);

    let scaled_gate_a = (a / gate_input_scale.expand_rhs((m, k))).cast(DType::F8E4M3);
    let gate = scaled_gate_a.matmul(gate_weight.t()).cast(DType::F32)
        * (gate_input_scale * gate_weight_scale).expand_rhs((m, n));
    let scaled_up_a = (a / up_input_scale.expand_rhs((m, k))).cast(DType::F8E4M3);
    let up = scaled_up_a.matmul(up_weight.t()).cast(DType::F32)
        * (up_input_scale * up_weight_scale).expand_rhs((m, n));
    (gate.swish() * up).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        dataflow_reachable_cublaslt_scaled_count(egraph) >= 2,
        "scaled cuBLASLt candidates must remain reachable through fused MLP gate/up consumers"
    );
    assert!(
        dataflow_reachable_cublaslt_raw_fp8_count(egraph) >= 2,
        "raw FP8 matmul plus output scaling must remain reachable for both MLP branches"
    );
}

#[test]
#[ignore = "expensive CUDA FP8 rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_fp8_scaled_candidate_executes_batched_matmul_f32_output() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_cublaslt_fp8_launch(DType::F8E4M3) {
        return;
    }

    let (batch, m, n, k) = (2, 16, 16, 16);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let a_scale = cx.tensor(());
    let b_scale = cx.tensor(());
    let b_input = cx.tensor((batch, n, k)).as_dtype(DType::F8E4M3);
    let b = b_input.transpose(1, 2);
    let scaled_a = (a / a_scale.expand_rhs((batch, m, k))).cast(DType::F8E4M3);
    let matmul = scaled_a.matmul(b);
    let out = (matmul * (a_scale * b_scale).expand_rhs((batch, m, n))).output();
    let expected_tuple = (
        DType::F8E4M3,
        DType::F8E4M3,
        DType::F32,
        DType::F32,
        "32F",
        DType::F32,
    );
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional scaled batched fp8", |llir| {
            cublaslt_type_tuples(llir).contains(&expected_tuple)
                && cublaslt_tensor_scale_input_tuples(llir).contains(&(true, true))
                && cublaslt_transpose_op_tuples(llir).contains(&("T", "N"))
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        });

    let input_scale = 0.5f32;
    let weight_scale = 1.5f32;
    let (a_fp8_bytes, a_values) = fp8_exact_bytes(DType::F8E4M3, batch * m * k, 11);
    let a_data = a_values
        .iter()
        .map(|value| value * input_scale)
        .collect::<Vec<_>>();
    let (b_bytes, b_storage_values) = fp8_exact_bytes(DType::F8E4M3, batch * k * n, 13);
    let b_values = logical_b_from_batched_column_major_storage(&b_storage_values, batch, n, k);
    let mut expected = reference_matmul_batched(&a_values, &b_values, batch, m, n, k);
    for value in &mut expected {
        *value *= input_scale * weight_scale;
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(a_scale, vec![input_scale]);
    rt.set_data(b_scale, vec![weight_scale]);
    rt.set_data(b_input, b_bytes);
    rt.execute(&cx.dyn_map);

    assert_eq!(a_fp8_bytes.len(), batch * m * k);
    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

fn cublaslt_fp8_candidate_executes_2d_matmul_f32_output(a_dtype: DType, b_dtype: DType) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_cublaslt_fp8_launch_pair(a_dtype, b_dtype) {
        return;
    }

    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let (a, b_input, out) = build_fp8_2d_cast_matmul_f32_graph(&mut cx, a_dtype, b_dtype, m, n, k);
    let expected_tuple = (b_dtype, a_dtype, DType::F32, DType::F32, "32F", DType::F32);
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional fp8 f32 output", |llir| {
        cublaslt_type_tuples(llir).contains(&expected_tuple)
            && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
            && cublaslt_transpose_op_tuples(llir).contains(&("T", "N"))
            && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
    });

    let (a_bytes, a_values) = fp8_exact_bytes(a_dtype, m * k, 2);
    let (b_bytes, b_storage_values) = fp8_exact_bytes(b_dtype, k * n, 4);
    let b_values = logical_b_from_column_major_storage(&b_storage_values, n, k);
    let expected = reference_matmul_2d(&a_values, &b_values, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_bytes);
    rt.set_data(b_input, b_bytes);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

fn cublaslt_fp8_same_type_does_not_match_2d_matmul_f32_output(dtype: DType) {
    let (m, n, k) = (16, 16, 16);
    let mut cx = Graph::new();
    let (_a, _b_input, _out) = build_fp8_2d_cast_matmul_f32_graph(&mut cx, dtype, dtype, m, n, k);

    let expected_tuple = (dtype, dtype, DType::F32, DType::F32, "32F", DType::F32);
    assert_no_cublaslt_llir_where(&mut cx, "illegal fp8 same-type f32 output", |llir| {
        cublaslt_type_tuples(llir).contains(&expected_tuple)
    });
}

fn cublaslt_fp8_candidate_executes_batched_matmul_f32_output(a_dtype: DType, b_dtype: DType) {
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_cublaslt_fp8_launch_pair(a_dtype, b_dtype) {
        return;
    }

    let (batch, m, n, k) = (2, 16, 16, 16);
    let mut cx = Graph::new();
    let (a, b_input, out) =
        build_fp8_batched_cast_matmul_f32_graph(&mut cx, a_dtype, b_dtype, batch, m, n, k);
    let expected_tuple = (b_dtype, a_dtype, DType::F32, DType::F32, "32F", DType::F32);
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched fp8 f32 output", |llir| {
            cublaslt_type_tuples(llir).contains(&expected_tuple)
                && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
                && cublaslt_transpose_op_tuples(llir).contains(&("T", "N"))
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        });

    let (a_bytes, a_values) = fp8_exact_bytes(a_dtype, batch * m * k, 5);
    let (b_bytes, b_storage_values) = fp8_exact_bytes(b_dtype, batch * k * n, 6);
    let b_values = logical_b_from_batched_column_major_storage(&b_storage_values, batch, n, k);
    let expected = reference_matmul_batched(&a_values, &b_values, batch, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_bytes);
    rt.set_data(b_input, b_bytes);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

fn cublaslt_fp8_same_type_does_not_match_batched_matmul_f32_output(dtype: DType) {
    let (batch, m, n, k) = (2, 16, 16, 16);
    let mut cx = Graph::new();
    let (_a, _b_input, _out) =
        build_fp8_batched_cast_matmul_f32_graph(&mut cx, dtype, dtype, batch, m, n, k);

    let expected_tuple = (dtype, dtype, DType::F32, DType::F32, "32F", DType::F32);
    assert_no_cublaslt_llir_where(
        &mut cx,
        "illegal batched fp8 same-type f32 output",
        |llir| cublaslt_type_tuples(llir).contains(&expected_tuple),
    );
}

fn build_fp8_2d_cast_matmul_f32_graph(
    cx: &mut Graph,
    a_dtype: DType,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let a = cx.tensor((m, k)).as_dtype(a_dtype);
    let b_input = cx.tensor((n, k)).as_dtype(b_dtype);
    let b = b_input.t();

    let out = a.matmul(b).output();
    (a, b_input, out)
}

fn build_fp8_batched_cast_matmul_f32_graph(
    cx: &mut Graph,
    a_dtype: DType,
    b_dtype: DType,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let a = cx.tensor((batch, m, k)).as_dtype(a_dtype);
    let b_input = cx.tensor((batch, n, k)).as_dtype(b_dtype);
    let b = b_input.transpose(1, 2);

    let out = a.matmul(b).output();
    (a, b_input, out)
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_column_bias_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_2d_matmul_plus_column_bias_graph(case, DType::F32, commuted),
                case.name,
                "BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_column_bias_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_batched_matmul_plus_column_bias_graph(case, DType::F32, commuted),
                case.name,
                "BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_emit_row_order_row_bias_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_epilogue_rewrite(
                build_2d_matmul_plus_row_bias_graph(case, DType::F32, commuted),
                case.name,
                "BIAS",
                Some(row_order_tuple(case)),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_emit_batched_row_order_row_bias_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_epilogue_rewrite(
                build_batched_matmul_plus_row_bias_graph(case, DType::F32, commuted),
                case.name,
                "BIAS",
                Some(row_order_tuple(case)),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_relu_epilogue() {
    for case in LAYOUT_CASES {
        assert_cublaslt_epilogue_rewrite(
            build_2d_matmul_relu_graph(case, DType::F32),
            case.name,
            "RELU",
            None,
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_relu_epilogue() {
    for case in LAYOUT_CASES {
        assert_cublaslt_epilogue_rewrite(
            build_batched_matmul_relu_graph(case, DType::F32),
            case.name,
            "RELU",
            None,
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_column_bias_relu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_2d_matmul_plus_column_bias_relu_graph(case, DType::F32, commuted),
                case.name,
                "RELU_BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_column_bias_relu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_batched_matmul_plus_column_bias_relu_graph(case, DType::F32, commuted),
                case.name,
                "RELU_BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_gelu_epilogue() {
    for case in LAYOUT_CASES {
        assert_cublaslt_epilogue_rewrite(
            build_2d_matmul_gelu_graph(case, DType::F32),
            case.name,
            "GELU",
            None,
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_gelu_epilogue() {
    for case in LAYOUT_CASES {
        assert_cublaslt_epilogue_rewrite(
            build_batched_matmul_gelu_graph(case, DType::F32),
            case.name,
            "GELU",
            None,
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_2d_matmul_plus_column_bias_gelu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_2d_matmul_plus_column_bias_gelu_graph(case, DType::F32, commuted),
                case.name,
                "GELU_BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_matmul_plus_column_bias_gelu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_cublaslt_epilogue_rewrite(
                build_batched_matmul_plus_column_bias_gelu_graph(case, DType::F32, commuted),
                case.name,
                "GELU_BIAS",
                Some(("COL", "COL", "COL", "COL")),
            );
        }
    }
}

#[test]
fn cublaslt_beta_rewrite_does_not_cross_activation_epilogues() {
    let case = LAYOUT_CASES[0];
    assert_no_forced_cublaslt_llir_where(
        &mut build_2d_matmul_plus_column_bias_activation_plus_c_graph(
            case,
            DType::F32,
            false,
            |x| x.relu(),
        ),
        case.name,
        |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("RELU_BIAS", (1.0, 1.0))),
    );
    assert_no_forced_cublaslt_llir_where(
        &mut build_2d_matmul_plus_column_bias_activation_plus_c_graph(
            case,
            DType::F32,
            false,
            |x| x.gelu_fast_tanh_approximation(),
        ),
        case.name,
        |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("GELU_BIAS", (1.0, 1.0))),
    );
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_rewrite_does_not_cross_activation_epilogues_exhaustive() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_llir_where(
                &mut build_2d_matmul_plus_column_bias_activation_plus_c_graph(
                    case,
                    DType::F32,
                    commuted,
                    |x| x.relu(),
                ),
                case.name,
                |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("RELU_BIAS", (1.0, 1.0))),
            );
            assert_no_forced_cublaslt_llir_where(
                &mut build_2d_matmul_plus_column_bias_activation_plus_c_graph(
                    case,
                    DType::F32,
                    commuted,
                    |x| x.gelu_fast_tanh_approximation(),
                ),
                case.name,
                |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("GELU_BIAS", (1.0, 1.0))),
            );
        }
    }
}

#[test]
fn cublaslt_alpha_scale_rewrite_does_not_cross_bias_epilogue() {
    let case = LAYOUT_CASES[0];
    assert_no_forced_cublaslt_llir_where(
        &mut build_2d_matmul_plus_column_bias_scaled_graph(case, DType::F32, false),
        case.name,
        |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("BIAS", (1.5, 0.0))),
    );
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_alpha_scale_rewrite_does_not_cross_bias_epilogue_exhaustive() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_llir_where(
                &mut build_2d_matmul_plus_column_bias_scaled_graph(case, DType::F32, commuted),
                case.name,
                |llir| cublaslt_epilogue_scale_tuples(llir).contains(&("BIAS", (1.5, 0.0))),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_emit_row_order_row_bias_relu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_epilogue_rewrite(
                build_2d_matmul_plus_row_bias_relu_graph(case, DType::F32, commuted),
                case.name,
                "RELU_BIAS",
                Some(row_order_tuple(case)),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA negative rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_do_not_emit_batched_row_order_row_bias_relu_epilogue() {
    for case in LAYOUT_CASES {
        for commuted in [false, true] {
            assert_no_forced_cublaslt_epilogue_rewrite(
                build_batched_matmul_plus_row_bias_relu_graph(case, DType::F32, commuted),
                case.name,
                "RELU_BIAS",
                Some(row_order_tuple(case)),
            );
        }
    }
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_2d_matmul_plus_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let c = cx.tensor((m, n));
    let out = (a.matmul(b) + c).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional beta=1", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });

    let a_data = random_f32_vec(m * k, 0xA11CE, -0.5, 0.5);
    let b_data = random_f32_vec(k * n, 0xB0B, -0.5, 0.5);
    let c_data = random_f32_vec(m * n, 0xC0DE, -0.5, 0.5);
    let mut expected = reference_matmul_2d(&a_data, &b_data, m, n, k);
    add_in_place(&mut expected, &c_data);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_2d_matmul_plus_sliced_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k, c_padding) = (7, 11, 5, 3);
    let c_parent_n = n + c_padding;
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let c_base = cx.tensor((m, c_parent_n));
    let c = c_base.slice((0..m, 0..n));
    let out = (a.matmul(b) + c).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional beta=1 sliced C", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            && cublaslt_c_d_layout_matches(llir).contains(&false)
    });

    let a_data = random_f32_vec(m * k, 0x05A1_1CEA, -0.5, 0.5);
    let b_data = random_f32_vec(k * n, 0x05A1_1CEB, -0.5, 0.5);
    let c_data = random_f32_vec(m * c_parent_n, 0x05A1_1CEC, -0.5, 0.5);
    let expected = reference_matmul_2d_plus_strided_c(
        &a_data,
        &b_data,
        &c_data,
        (m, n, k),
        (c_parent_n, 0, 0),
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c_base, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_batched_matmul_plus_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let c = cx.tensor((batch, m, n));
    let out = (a.matmul(b) + c).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional batched beta=1", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_A11CE, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0x0BA7_CB0B, -0.5, 0.5);
    let c_data = random_f32_vec(batch * m * n, 0xBA7C_C0DE, -0.5, 0.5);
    let mut expected = reference_matmul_batched(&a_data, &b_data, batch, m, n, k);
    add_in_place(&mut expected, &c_data);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_batched_matmul_plus_sliced_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k, c_padding) = (3, 5, 4, 6, 2);
    let c_parent_n = n + c_padding;
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let c_base = cx.tensor((batch, m, c_parent_n));
    let c = c_base.slice((0..batch, 0..m, 0..n));
    let out = (a.matmul(b) + c).output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched beta=1 sliced C", |llir| {
            cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                && cublaslt_c_d_layout_matches(llir).contains(&false)
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_5A11, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_5B11, -0.5, 0.5);
    let c_data = random_f32_vec(batch * m * c_parent_n, 0xBA7C_5C11, -0.5, 0.5);
    let expected = reference_matmul_batched_plus_strided_c(
        &a_data,
        &b_data,
        &c_data,
        (batch, m, n, k),
        (m * c_parent_n, c_parent_n, 0, 0, 0),
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c_base, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_2d_matmul_plus_offset_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let (c_parent_m, c_parent_n) = (m + 2, n + 3);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let c_base = cx.tensor((c_parent_m, c_parent_n));
    let c = c_base.slice((1..(m + 1), 2..(n + 2)));
    let out = (a.matmul(b) + c).output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional beta=1 offset C", |llir| {
        cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            && cublaslt_c_d_layout_matches(llir).contains(&true)
    });

    let a_data = random_f32_vec(m * k, 0x0FF5_E7A1, -0.5, 0.5);
    let b_data = random_f32_vec(k * n, 0x0FF5_E7B1, -0.5, 0.5);
    let c_data = random_f32_vec(c_parent_m * c_parent_n, 0x0FF5_E7C1, -0.5, 0.5);
    let expected = reference_matmul_2d_plus_strided_c(
        &a_data,
        &b_data,
        &c_data,
        (m, n, k),
        (c_parent_n, 1, 2),
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c_base, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_beta_one_candidate_executes_batched_matmul_plus_offset_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let (c_parent_batch, c_parent_m, c_parent_n) = (batch + 1, m + 2, n + 3);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let c_base = cx.tensor((c_parent_batch, c_parent_m, c_parent_n));
    let c = c_base.slice((1..(batch + 1), 1..(m + 1), 2..(n + 2)));
    let out = (a.matmul(b) + c).output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched beta=1 offset C", |llir| {
            cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
                && cublaslt_c_d_layout_matches(llir).contains(&true)
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_0FF5, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_0FF6, -0.5, 0.5);
    let c_data = random_f32_vec(
        c_parent_batch * c_parent_m * c_parent_n,
        0xBA7C_0FF7,
        -0.5,
        0.5,
    );
    let expected = reference_matmul_batched_plus_strided_c(
        &a_data,
        &b_data,
        &c_data,
        (batch, m, n, k),
        (c_parent_m * c_parent_n, c_parent_n, 1, 1, 2),
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c_base, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_scaled_alpha_beta_candidate_executes_2d_matmul_plus_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let (alpha, beta) = (1.5, 0.5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let c = cx.tensor((m, n));
    let out = (a.matmul(b) * alpha + c * beta).output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional scaled alpha beta", |llir| {
            cublaslt_scale_value_tuples(llir).contains(&(alpha as f64, beta as f64))
                && cublaslt_matrix_order_tuples(llir).contains(&("ROW", "ROW", "ROW", "ROW"))
        });

    let a_data = random_f32_vec(m * k, 0x5CA1_EDA1, -0.5, 0.5);
    let b_data = random_f32_vec(k * n, 0x5CA1_EDB1, -0.5, 0.5);
    let c_data = random_f32_vec(m * n, 0x5CA1_EDC1, -0.5, 0.5);
    let matmul = reference_matmul_2d(&a_data, &b_data, m, n, k);
    let expected = reference_scaled_alpha_beta(&matmul, &c_data, alpha, beta);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_scaled_alpha_beta_candidate_executes_batched_matmul_plus_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let (alpha, beta) = (1.5, 0.5);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let c = cx.tensor((batch, m, n));
    let out = (a.matmul(b) * alpha + c * beta).output();
    let llir = extract_forced_cublaslt_llir_where(
        &mut cx,
        "functional batched scaled alpha beta",
        |llir| {
            cublaslt_scale_value_tuples(llir).contains(&(alpha as f64, beta as f64))
                && cublaslt_matrix_order_tuples(llir).contains(&("ROW", "ROW", "ROW", "ROW"))
        },
    );

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_5CA1, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_5CB1, -0.5, 0.5);
    let c_data = random_f32_vec(batch * m * n, 0xBA7C_5CC1, -0.5, 0.5);
    let matmul = reference_matmul_batched(&a_data, &b_data, batch, m, n, k);
    let expected = reference_scaled_alpha_beta(&matmul, &c_data, alpha, beta);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_bias_epilogue_candidate_executes_2d_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    for case in LAYOUT_CASES {
        let (m, n, k) = (7, 11, 5);
        let mut cx = Graph::new();
        let a_storage = cx.tensor(if case.a_col_major { (k, m) } else { (m, k) });
        let b_storage = cx.tensor(if case.b_col_major { (n, k) } else { (k, n) });
        let a = if case.a_col_major {
            a_storage.t()
        } else {
            a_storage
        };
        let b = if case.b_col_major {
            b_storage.t()
        } else {
            b_storage
        };
        let bias = cx.tensor(n);
        let bias_expanded = bias.expand_dim(0, m);
        let out = (a.matmul(b) + bias_expanded).output();
        let llir = extract_forced_cublaslt_llir_where(
            &mut cx,
            &format!("functional bias epilogue {}", case.name),
            |llir| {
                cublaslt_epilogues(llir).contains(&"BIAS")
                    && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
            },
        );

        let a_data = random_f32_vec(m * k, 0xB1A5_EDA1 + case_seed(case), -0.5, 0.5);
        let b_data = random_f32_vec(k * n, 0xB1A5_EDB1 + case_seed(case), -0.5, 0.5);
        let bias_data = random_f32_vec(n, 0xB1A5_EDC1 + case_seed(case), -0.5, 0.5);
        let expected = reference_column_bias_postop(
            reference_matmul_2d_layout(case, &a_data, &b_data, m, n, k),
            &bias_data,
            1,
            m,
            n,
            PostOp::Identity,
        );

        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(a_storage, a_data);
        rt.set_data(b_storage, b_data);
        rt.set_data(bias, bias_data);
        rt.execute(&cx.dyn_map);

        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    }
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_bias_epilogue_candidate_executes_batched_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let bias = cx.tensor(n);
    let bias_expanded = bias.expand_dim(0, m).expand_dim(0, batch);
    let out = (a.matmul(b) + bias_expanded).output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched bias epilogue", |llir| {
            cublaslt_epilogues(llir).contains(&"BIAS")
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_B1A5, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_B1A6, -0.5, 0.5);
    let bias_data = random_f32_vec(n, 0xBA7C_B1A7, -0.5, 0.5);
    let expected = reference_column_bias_postop(
        reference_matmul_batched(&a_data, &b_data, batch, m, n, k),
        &bias_data,
        batch,
        m,
        n,
        PostOp::Identity,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(bias, bias_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_relu_bias_epilogue_candidate_executes_2d_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let bias = cx.tensor(n);
    let bias_expanded = bias.expand_dim(0, m);
    let out = (a.matmul(b) + bias_expanded).relu().output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional relu bias epilogue", |llir| {
            cublaslt_epilogues(llir).contains(&"RELU_BIAS")
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        });

    let a_data = random_f32_vec(m * k, 0x2E1F_B1A5, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 0x2E1F_B1A6, -1.0, 1.0);
    let bias_data = random_f32_vec(n, 0x2E1F_B1A7, -0.5, 0.5);
    let expected = reference_column_bias_postop(
        reference_matmul_2d(&a_data, &b_data, m, n, k),
        &bias_data,
        1,
        m,
        n,
        PostOp::Relu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(bias, bias_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_relu_bias_epilogue_candidate_executes_batched_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let bias = cx.tensor(n);
    let bias_expanded = bias.expand_dim(0, m).expand_dim(0, batch);
    let out = (a.matmul(b) + bias_expanded).relu().output();
    let llir = extract_forced_cublaslt_llir_where(
        &mut cx,
        "functional batched relu column bias epilogue",
        |llir| {
            cublaslt_epilogues(llir).contains(&"RELU_BIAS")
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        },
    );

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_2E1F, -1.0, 1.0);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_2E20, -1.0, 1.0);
    let bias_data = random_f32_vec(n, 0xBA7C_2E21, -0.5, 0.5);
    let expected = reference_column_bias_postop(
        reference_matmul_batched(&a_data, &b_data, batch, m, n, k),
        &bias_data,
        batch,
        m,
        n,
        PostOp::Relu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(bias, bias_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_gelu_epilogue_candidate_executes_2d_matmul() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).gelu_fast_tanh_approximation().output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional gelu epilogue", |llir| {
        cublaslt_epilogues(llir).contains(&"GELU")
    });

    let a_data = random_f32_vec(m * k, 0x9E1F_2EDA, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 0x9E1F_2EDB, -1.0, 1.0);
    let expected = reference_postop(reference_matmul_2d(&a_data, &b_data, m, n, k), PostOp::Gelu);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 5e-4, 5e-4);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_gelu_epilogue_candidate_executes_batched_matmul() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let out = a.matmul(b).gelu_fast_tanh_approximation().output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched gelu epilogue", |llir| {
            cublaslt_epilogues(llir).contains(&"GELU")
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_9E1F, -1.0, 1.0);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_9E20, -1.0, 1.0);
    let expected = reference_postop(
        reference_matmul_batched(&a_data, &b_data, batch, m, n, k),
        PostOp::Gelu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 5e-4, 5e-4);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_gelu_bias_epilogue_candidate_executes_2d_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let bias = cx.tensor(n);
    let bias_expanded = bias.expand_dim(0, m);
    let out = (a.matmul(b) + bias_expanded)
        .gelu_fast_tanh_approximation()
        .output();
    let llir = extract_forced_cublaslt_llir_where(
        &mut cx,
        "functional gelu column bias epilogue",
        |llir| {
            cublaslt_epilogues(llir).contains(&"GELU_BIAS")
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        },
    );

    let a_data = random_f32_vec(m * k, 0x9E1F_B1A5, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 0x9E1F_B1A6, -1.0, 1.0);
    let bias_data = random_f32_vec(n, 0x9E1F_B1A7, -0.5, 0.5);
    let expected = reference_column_bias_postop(
        reference_matmul_2d(&a_data, &b_data, m, n, k),
        &bias_data,
        1,
        m,
        n,
        PostOp::Gelu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(bias, bias_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 5e-4, 5e-4);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_gelu_bias_epilogue_candidate_executes_batched_matmul_plus_column_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let bias = cx.tensor(n);
    let bias_expanded = bias.expand_dim(0, m).expand_dim(0, batch);
    let out = (a.matmul(b) + bias_expanded)
        .gelu_fast_tanh_approximation()
        .output();
    let llir = extract_forced_cublaslt_llir_where(
        &mut cx,
        "functional batched gelu column bias epilogue",
        |llir| {
            cublaslt_epilogues(llir).contains(&"GELU_BIAS")
                && cublaslt_matrix_order_tuples(llir).contains(&("COL", "COL", "COL", "COL"))
        },
    );

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_9E1F, -1.0, 1.0);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_9E20, -1.0, 1.0);
    let bias_data = random_f32_vec(n, 0xBA7C_9E21, -0.5, 0.5);
    let expected = reference_column_bias_postop(
        reference_matmul_batched(&a_data, &b_data, batch, m, n, k),
        &bias_data,
        batch,
        m,
        n,
        PostOp::Gelu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(bias, bias_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 5e-4, 5e-4);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_relu_epilogue_candidate_executes_2d_matmul() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));
    let out = a.matmul(b).relu().output();
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "functional relu epilogue", |llir| {
        cublaslt_epilogues(llir).contains(&"RELU")
    });

    let a_data = random_f32_vec(m * k, 0x5E1F_2EDA, -1.0, 1.0);
    let b_data = random_f32_vec(k * n, 0x5E1F_2EDB, -1.0, 1.0);
    let expected = reference_postop(reference_matmul_2d(&a_data, &b_data, m, n, k), PostOp::Relu);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_relu_epilogue_candidate_executes_batched_matmul() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let out = a.matmul(b).relu().output();
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "functional batched relu epilogue", |llir| {
            cublaslt_epilogues(llir).contains(&"RELU")
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_2EDA, -1.0, 1.0);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_2EDB, -1.0, 1.0);
    let expected = reference_postop(
        reference_matmul_batched(&a_data, &b_data, batch, m, n, k),
        PostOp::Relu,
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_row_order_beta_one_candidate_executes_2d_layout_pairs() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    for case in LAYOUT_CASES {
        let (m, n, k) = (5, 7, 4);
        let a_shape = if case.a_col_major { (k, m) } else { (m, k) };
        let b_shape = if case.b_col_major { (n, k) } else { (k, n) };

        let mut cx = Graph::new();
        let a_input = cx.tensor(a_shape);
        let b_input = cx.tensor(b_shape);
        let c = cx.tensor((m, n));
        let a = if case.a_col_major {
            a_input.t()
        } else {
            a_input
        };
        let b = if case.b_col_major {
            b_input.t()
        } else {
            b_input
        };
        let out = (a.matmul(b) + c).output();
        let expected_orders = row_order_tuple(case);
        let llir = extract_forced_cublaslt_llir_where(&mut cx, case.name, |llir| {
            cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
        });

        let a_data = random_f32_vec(m * k, 0xE21A_0000 + case_seed(case), -0.5, 0.5);
        let b_data = random_f32_vec(k * n, 0xE21B_0000 + case_seed(case), -0.5, 0.5);
        let c_data = random_f32_vec(m * n, 0xE21C_0000 + case_seed(case), -0.5, 0.5);
        let mut expected = reference_matmul_2d_layout(case, &a_data, &b_data, m, n, k);
        add_in_place(&mut expected, &c_data);

        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(a_input, a_data);
        rt.set_data(b_input, b_data);
        rt.set_data(c, c_data);
        rt.execute(&cx.dyn_map);

        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    }
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_row_order_beta_one_candidate_executes_batched_row_major_matmul_plus_c() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let c = cx.tensor((batch, m, n));
    let out = (a.matmul(b) + c).output();
    let expected_orders = ("ROW", "ROW", "ROW", "ROW");
    let llir =
        extract_forced_cublaslt_llir_where(&mut cx, "batched row-order beta=1 row-major", |llir| {
            cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
        });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_E21A, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_E21B, -0.5, 0.5);
    let c_data = random_f32_vec(batch * m * n, 0xBA7C_E21C, -0.5, 0.5);
    let mut expected = reference_matmul_batched(&a_data, &b_data, batch, m, n, k);
    add_in_place(&mut expected, &c_data);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_row_order_candidate_executes_2d_layout_pairs() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    for case in LAYOUT_CASES {
        let (m, n, k) = (5, 7, 4);
        let a_shape = if case.a_col_major { (k, m) } else { (m, k) };
        let b_shape = if case.b_col_major { (n, k) } else { (k, n) };

        let mut cx = Graph::new();
        let a_input = cx.tensor(a_shape);
        let b_input = cx.tensor(b_shape);
        let a = if case.a_col_major {
            a_input.t()
        } else {
            a_input
        };
        let b = if case.b_col_major {
            b_input.t()
        } else {
            b_input
        };
        let out = a.matmul(b).output();
        let expected_orders = row_order_tuple(case);
        let llir = extract_forced_cublaslt_llir_where(&mut cx, case.name, |llir| {
            cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
                && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
        });

        let a_data = random_f32_vec(m * k, 0xE20A_0000 + case_seed(case), -0.5, 0.5);
        let b_data = random_f32_vec(k * n, 0xE20B_0000 + case_seed(case), -0.5, 0.5);
        let expected = reference_matmul_2d_layout(case, &a_data, &b_data, m, n, k);

        let mut rt = CudaRuntime::initialize(stream.clone());
        rt.load_llir(&llir);
        rt.set_data(a_input, a_data);
        rt.set_data(b_input, b_data);
        rt.execute(&cx.dyn_map);

        assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
    }
}

#[test]
#[ignore = "large row-order CUDA functional repro for llama lm_head shape"]
fn cublaslt_row_order_candidate_executes_large_lm_head_like_projection() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (1, 128_256, 64);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b_input = cx.tensor((n, k));
    let b = b_input.t();
    let out = a.matmul(b).output();
    let expected_orders = ("ROW", "COL", "ROW", "ROW");
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "lm_head-like row-order", |llir| {
        cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
    });

    let a_data = random_f32_vec(m * k, 0x1A11_A000, -0.5, 0.5);
    let b_data = random_f32_vec(n * k, 0x1A11_B000, -0.5, 0.5);
    let mut expected = vec![0.0f32; m * n];
    for col in 0..n {
        let mut sum = 0.0f32;
        for kk in 0..k {
            sum += a_data[kk] * b_data[col * k + kk];
        }
        expected[col] = sum;
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b_input, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "large row-order CUDA functional repro for llama MLP residual beta=1 shape"]
fn cublaslt_row_order_beta_one_candidate_executes_llama_mlp_residual_like_projection() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (m, n, k) = (1, 4096, 64);
    let mut cx = Graph::new();
    let a = cx.tensor((m, k));
    let b_input = cx.tensor((n, k));
    let b = b_input.t();
    let c = cx.tensor((m, n));
    let out = (a.matmul(b) + c).output();
    let expected_orders = ("ROW", "COL", "ROW", "ROW");
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "mlp residual row-order", |llir| {
        cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
    });

    let a_data = random_f32_vec(m * k, 0x1A12_A000, -0.5, 0.5);
    let b_data = random_f32_vec(n * k, 0x1A12_B000, -0.5, 0.5);
    let c_data = random_f32_vec(m * n, 0x1A12_C000, -0.5, 0.5);
    let mut expected = c_data.clone();
    for col in 0..n {
        for kk in 0..k {
            expected[col] += a_data[kk] * b_data[col * k + kk];
        }
    }

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b_input, b_data);
    rt.set_data(c, c_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
#[ignore = "expensive CUDA functional candidate sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_row_order_candidate_executes_batched_row_major_matmul() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (batch, m, n, k) = (3, 5, 4, 6);
    let mut cx = Graph::new();
    let a = cx.tensor((batch, m, k));
    let b = cx.tensor((batch, k, n));
    let out = a.matmul(b).output();
    let expected_orders = ("ROW", "ROW", "ROW", "ROW");
    let llir = extract_forced_cublaslt_llir_where(&mut cx, "batched row-order row-major", |llir| {
        cublaslt_matrix_order_tuples(llir).contains(&expected_orders)
            && cublaslt_scale_value_tuples(llir).contains(&(1.0, 0.0))
    });

    let a_data = random_f32_vec(batch * m * k, 0xBA7C_E20A, -0.5, 0.5);
    let b_data = random_f32_vec(batch * k * n, 0xBA7C_E20B, -0.5, 0.5);
    let expected = reference_matmul_batched(&a_data, &b_data, batch, m, n, k);

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(a, a_data);
    rt.set_data(b, b_data);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

fn add_commuted(lhs: GraphTensor, rhs: GraphTensor, commuted: bool) -> GraphTensor {
    if commuted { rhs + lhs } else { lhs + rhs }
}

fn build_2d_layout_graph(
    case: LayoutCase,
    a_dtype: DType,
    b_dtype: DType,
    build: impl FnOnce(&mut Graph, GraphTensor, GraphTensor, usize, usize, usize) -> GraphTensor,
) -> Graph {
    let (m, n, k) = (7, 11, 5);
    let mut cx = Graph::new();
    let a = cx
        .tensor(if case.a_col_major { (k, m) } else { (m, k) })
        .as_dtype(a_dtype);
    let b = cx
        .tensor(if case.b_col_major { (n, k) } else { (k, n) })
        .as_dtype(b_dtype);
    let a = if case.a_col_major { a.t() } else { a };
    let b = if case.b_col_major { b.t() } else { b };
    build(&mut cx, a, b, m, n, k).output();
    cx
}

fn build_same_dtype_2d_graph(
    case: LayoutCase,
    dtype: DType,
    build: impl FnOnce(&mut Graph, GraphTensor, GraphTensor, usize, usize, usize) -> GraphTensor,
) -> Graph {
    build_2d_layout_graph(case, dtype, dtype, build)
}

fn build_batched_layout_graph(
    case: LayoutCase,
    a_dtype: DType,
    b_dtype: DType,
    build: impl FnOnce(&mut Graph, GraphTensor, GraphTensor, usize, usize, usize, usize) -> GraphTensor,
) -> Graph {
    let (batch, m, n, k) = (3, 7, 11, 5);
    let mut cx = Graph::new();
    let a = cx
        .tensor(if case.a_col_major {
            (batch, k, m)
        } else {
            (batch, m, k)
        })
        .as_dtype(a_dtype);
    let b = cx
        .tensor(if case.b_col_major {
            (batch, n, k)
        } else {
            (batch, k, n)
        })
        .as_dtype(b_dtype);
    let a = if case.a_col_major {
        a.transpose(1, 2)
    } else {
        a
    };
    let b = if case.b_col_major {
        b.transpose(1, 2)
    } else {
        b
    };
    build(&mut cx, a, b, batch, m, n, k).output();
    cx
}

fn build_same_dtype_batched_graph(
    case: LayoutCase,
    dtype: DType,
    build: impl FnOnce(&mut Graph, GraphTensor, GraphTensor, usize, usize, usize, usize) -> GraphTensor,
) -> Graph {
    build_batched_layout_graph(case, dtype, dtype, build)
}

fn build_2d_matmul_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |_, a, b, _, _, _| a.matmul(b))
}

fn build_2d_matmul_plus_c_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let c = cx.tensor((m, n)).as_dtype(dtype);
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_2d_matmul_plus_sliced_c_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let c = cx.tensor((m, n + 3)).as_dtype(dtype).slice((0..m, 0..n));
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_2d_matmul_plus_offset_c_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let c = cx
            .tensor((m + 2, n + 3))
            .as_dtype(dtype)
            .slice((1..(m + 1), 2..(n + 2)));
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_2d_matmul_plus_transposed_c_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let c = cx.tensor((n, m)).as_dtype(dtype).t();
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_2d_scaled_alpha_beta_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let c = cx.tensor((m, n)).as_dtype(dtype);
        add_commuted(a.matmul(b) * 1.5, c * 0.5, commuted)
    })
}

fn build_2d_matmul_plus_column_bias_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(n).as_dtype(dtype).expand_dim(0, m);
        add_commuted(a.matmul(b), bias, commuted)
    })
}

fn build_2d_matmul_plus_row_bias_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(m).as_dtype(dtype).expand_dim(1, n);
        add_commuted(a.matmul(b), bias, commuted)
    })
}

fn build_2d_matmul_plus_column_bias_relu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(n).as_dtype(dtype).expand_dim(0, m);
        add_commuted(a.matmul(b), bias, commuted).relu()
    })
}

fn build_2d_matmul_plus_column_bias_gelu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(n).as_dtype(dtype).expand_dim(0, m);
        add_commuted(a.matmul(b), bias, commuted).gelu_fast_tanh_approximation()
    })
}

fn build_2d_matmul_plus_column_bias_activation_plus_c_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
    activation: impl FnOnce(GraphTensor) -> GraphTensor,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(n).as_dtype(dtype).expand_dim(0, m);
        let residual = cx.tensor((m, n)).as_dtype(dtype);
        add_commuted(activation(a.matmul(b) + bias), residual, commuted)
    })
}

fn build_2d_matmul_plus_column_bias_scaled_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(n).as_dtype(dtype).expand_dim(0, m);
        add_commuted(a.matmul(b), bias, commuted) * 1.5
    })
}

fn build_2d_matmul_plus_row_bias_relu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |cx, a, b, m, n, _| {
        let bias = cx.tensor(m).as_dtype(dtype).expand_dim(1, n);
        add_commuted(a.matmul(b), bias, commuted).relu()
    })
}

fn build_2d_matmul_relu_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |_, a, b, _, _, _| a.matmul(b).relu())
}

fn build_2d_matmul_gelu_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_2d_graph(case, dtype, |_, a, b, _, _, _| {
        a.matmul(b).gelu_fast_tanh_approximation()
    })
}

fn build_batched_matmul_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |_, a, b, _, _, _, _| a.matmul(b))
}

fn build_batched_matmul_plus_c_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let c = cx.tensor((batch, m, n)).as_dtype(dtype);
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_batched_matmul_plus_sliced_c_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let c = cx
            .tensor((batch, m, n + 3))
            .as_dtype(dtype)
            .slice((0..batch, 0..m, 0..n));
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_batched_matmul_plus_offset_c_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let c = cx.tensor((batch + 1, m + 2, n + 3)).as_dtype(dtype).slice((
            1..(batch + 1),
            1..(m + 1),
            2..(n + 2),
        ));
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_batched_matmul_plus_transposed_c_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let c = cx.tensor((batch, n, m)).as_dtype(dtype).transpose(1, 2);
        add_commuted(a.matmul(b), c, commuted)
    })
}

fn build_batched_scaled_alpha_beta_graph(case: LayoutCase, dtype: DType, commuted: bool) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let c = cx.tensor((batch, m, n)).as_dtype(dtype);
        add_commuted(a.matmul(b) * 1.5, c * 0.5, commuted)
    })
}

fn build_batched_matmul_plus_column_bias_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let bias = cx
            .tensor(n)
            .as_dtype(dtype)
            .expand_dim(0, m)
            .expand_dim(0, batch);
        add_commuted(a.matmul(b), bias, commuted)
    })
}

fn build_batched_matmul_plus_row_bias_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let bias = cx
            .tensor(m)
            .as_dtype(dtype)
            .expand_dim(1, n)
            .expand_dim(0, batch);
        add_commuted(a.matmul(b), bias, commuted)
    })
}

fn build_batched_matmul_plus_column_bias_relu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let bias = cx
            .tensor(n)
            .as_dtype(dtype)
            .expand_dim(0, m)
            .expand_dim(0, batch);
        add_commuted(a.matmul(b), bias, commuted).relu()
    })
}

fn build_batched_matmul_plus_column_bias_gelu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let bias = cx
            .tensor(n)
            .as_dtype(dtype)
            .expand_dim(0, m)
            .expand_dim(0, batch);
        add_commuted(a.matmul(b), bias, commuted).gelu_fast_tanh_approximation()
    })
}

fn build_batched_matmul_plus_row_bias_relu_graph(
    case: LayoutCase,
    dtype: DType,
    commuted: bool,
) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |cx, a, b, batch, m, n, _| {
        let bias = cx
            .tensor(m)
            .as_dtype(dtype)
            .expand_dim(1, n)
            .expand_dim(0, batch);
        add_commuted(a.matmul(b), bias, commuted).relu()
    })
}

fn build_batched_matmul_relu_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |_, a, b, _, _, _, _| a.matmul(b).relu())
}

fn build_batched_matmul_gelu_graph(case: LayoutCase, dtype: DType) -> Graph {
    build_same_dtype_batched_graph(case, dtype, |_, a, b, _, _, _, _| {
        a.matmul(b).gelu_fast_tanh_approximation()
    })
}

fn extract_forced_cublaslt_llir(mut cx: Graph, case_name: &str) -> LLIRGraph {
    extract_forced_cublaslt_llir_where(&mut cx, case_name, |_| true)
}

fn assert_cublaslt_rewrite(mut cx: Graph, case_name: &str, matches: impl Fn(&LLIRGraph) -> bool) {
    let _llir = extract_forced_cublaslt_llir_where(&mut cx, case_name, matches);
}

fn llir_has_epilogue_with_orders(
    llir: &LLIRGraph,
    epilogue: &str,
    orders: Option<CublasLtMatrixOrders>,
) -> bool {
    cublaslt_epilogues(llir).contains(&epilogue)
        && match orders {
            Some(expected_orders) => cublaslt_matrix_order_tuples(llir).contains(&expected_orders),
            None => true,
        }
}

fn assert_cublaslt_epilogue_rewrite(
    cx: Graph,
    case_name: &str,
    epilogue: &str,
    orders: Option<CublasLtMatrixOrders>,
) {
    assert_cublaslt_rewrite(cx, case_name, |llir| {
        llir_has_epilogue_with_orders(llir, epilogue, orders)
    });
}

fn assert_no_forced_cublaslt_epilogue_rewrite(
    mut cx: Graph,
    case_name: &str,
    epilogue: &str,
    orders: Option<CublasLtMatrixOrders>,
) {
    assert_no_forced_cublaslt_llir_where(&mut cx, case_name, |llir| {
        llir_has_epilogue_with_orders(llir, epilogue, orders)
    });
}

fn extract_forced_cublaslt_llir_where(
    cx: &mut Graph,
    case_name: &str,
    matches: impl Fn(&LLIRGraph) -> bool,
) -> LLIRGraph {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    try_extract_forced_op_llir_where(
        cx,
        &["cublaslt", "cublaslt_scaled"],
        ForcedExtractionConfig::new(0x00C0_B1A5),
        |llir| !cublaslt_type_tuples(llir).is_empty() && matches(llir),
    )
    .unwrap_or_else(|error| {
        panic!("expected to extract a CuBlasLt HostOp for {case_name}: {error}")
    })
}

/// Force exactly one cuBLASLt alternative in every distinct IR e-class.
///
/// Multi-matmul tests need this stronger form because each output e-class also
/// intentionally retains a GenericMatmul fallback. Forcing every cuBLASLt
/// enode directly would be contradictory when several of those enodes are
/// mutually-exclusive alternatives in the same e-class, so choices are made
/// per class and candidate combinations are validated before extraction.
fn extract_forced_distinct_cublaslt_classes_llir_where(
    cx: &mut Graph,
    case_name: &str,
    matches: impl Fn(&LLIRGraph) -> bool,
) -> LLIRGraph {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .egglog_ops()
        .expect("search space should have registered egglog ops");
    let mut nodes_by_class: BTreeMap<ClassId, Vec<&NodeId>> = BTreeMap::new();
    for node in cublaslt_ir_nodes(egraph) {
        nodes_by_class
            .entry(egraph.node_to_class[node].clone())
            .or_default()
            .push(node);
    }
    assert!(
        nodes_by_class.len() > 1,
        "expected distinct cuBLASLt output e-classes for {case_name}"
    );
    for nodes in nodes_by_class.values_mut() {
        nodes.sort_unstable();
    }

    let combination_count = nodes_by_class
        .values()
        .try_fold(1usize, |count, nodes| count.checked_mul(nodes.len()))
        .expect("cuBLASLt test alternative combination count overflowed");
    let mut last_error = None;

    // Retry each per-class combination against several deterministic random
    // backgrounds for the non-forced e-classes. This preserves all semantic
    // fallbacks while ensuring both logical matmuls are represented by one
    // compatible cuBLASLt choice in the selected LLIR.
    for background in 0..16usize {
        for combination in 0..combination_count {
            let mut rng = StdRng::seed_from_u64(
                0xD157_1C7C + (background * combination_count + combination) as u64,
            );
            let mut choices = random_initial_choice(egraph, &mut rng);
            let mut selection = combination;
            for (class, nodes) in &nodes_by_class {
                let selected = nodes[selection % nodes.len()];
                selection /= nodes.len();
                choices.insert(class, selected);
            }

            if let Err(err) = validate_choice_set(egraph, &choices, ops) {
                last_error = Some(err);
                continue;
            }

            let llir = extract_llir_for_choices(cx, choices);
            if cublaslt_type_tuples(&llir).len() == nodes_by_class.len() && matches(&llir) {
                return llir;
            }

            last_error =
                Some("per-class cuBLASLt choices did not satisfy requested extracted shape".into());
        }
    }

    panic!(
        "expected one compatible CuBlasLt HostOp from each distinct e-class for {case_name}; last error: {}",
        last_error.unwrap_or_else(|| "no per-class candidate could be forced".into())
    );
}

fn assert_no_forced_cublaslt_llir_where(
    cx: &mut Graph,
    case_name: &str,
    matches: impl Fn(&LLIRGraph) -> bool,
) {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .egglog_ops()
        .expect("search space should have registered egglog ops");
    let cublaslt_nodes = cublaslt_ir_nodes(egraph);
    assert!(
        !cublaslt_nodes.is_empty(),
        "expected at least the base cuBLASLt matmul candidate for {case_name}"
    );

    for (idx, cublaslt_node) in cublaslt_nodes.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0xBAD_C0DE + idx as u64);
        let mut choices = random_initial_choice(egraph, &mut rng);
        let cublaslt_class = &egraph.node_to_class[*cublaslt_node];
        choices.insert(cublaslt_class, cublaslt_node);

        if validate_choice_set(egraph, &choices, ops).is_err() {
            continue;
        }

        let llir = extract_llir_for_choices(cx, choices);

        assert!(
            !llir_has_cublaslt(&llir) || !matches(&llir),
            "unexpected cuBLASLt candidate matched forbidden shape for {case_name}: types={:?}, scales={:?}, orders={:?}",
            cublaslt_type_tuples(&llir),
            cublaslt_scale_value_tuples(&llir),
            cublaslt_matrix_order_tuples(&llir)
        );
    }
}

fn assert_no_cublaslt_llir_where(
    cx: &mut Graph,
    case_name: &str,
    matches: impl Fn(&LLIRGraph) -> bool,
) {
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .egglog_ops()
        .expect("search space should have registered egglog ops");

    for (idx, cublaslt_node) in cublaslt_ir_nodes(egraph).iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0xBAD_C0DE + idx as u64);
        let mut choices = random_initial_choice(egraph, &mut rng);
        let cublaslt_class = &egraph.node_to_class[*cublaslt_node];
        choices.insert(cublaslt_class, cublaslt_node);

        if validate_choice_set(egraph, &choices, ops).is_err() {
            continue;
        }

        let llir = extract_llir_for_choices(cx, choices);

        assert!(
            !llir_has_cublaslt(&llir) || !matches(&llir),
            "unexpected cuBLASLt candidate matched forbidden shape for {case_name}: types={:?}, scales={:?}, orders={:?}, transposes={:?}",
            cublaslt_type_tuples(&llir),
            cublaslt_scale_value_tuples(&llir),
            cublaslt_matrix_order_tuples(&llir),
            cublaslt_transpose_op_tuples(&llir)
        );
    }
}

fn cublaslt_ir_nodes(egraph: &SerializedEGraph) -> Vec<&NodeId> {
    op_ir_nodes(egraph, "cublaslt")
        .into_iter()
        .chain(op_ir_nodes(egraph, "cublaslt_scaled"))
        .collect()
}

fn assert_relation_facts_are_requested(
    egraph: &SerializedEGraph,
    proof_relation: &str,
    request_relation: &str,
) {
    let relation_facts = |label: &str| {
        egraph
            .enodes
            .values()
            .filter_map(|(node_label, children)| (node_label == label).then_some(children.clone()))
            .collect::<FxHashSet<_>>()
    };
    let proofs = relation_facts(proof_relation);
    let requests = relation_facts(request_relation);

    assert!(
        !proofs.is_empty(),
        "expected at least one {proof_relation} fact"
    );
    assert!(
        proofs.is_subset(&requests),
        "every {proof_relation} fact must have an identical {request_relation} tuple"
    );
}

fn ir_class_has_op_kinds(egraph: &SerializedEGraph, kind_labels: &[&str]) -> bool {
    egraph.eclasses.values().any(|(sort, nodes)| {
        sort == "IR"
            && kind_labels.iter().all(|kind_label| {
                nodes.iter().any(|node| {
                    let Some((label, children)) = egraph.enodes.get(node) else {
                        return false;
                    };
                    label == "Op"
                        && children.first().is_some_and(|kind_class| {
                            egraph.eclasses[kind_class]
                                .1
                                .iter()
                                .any(|kind_node| egraph.enodes[kind_node].0.as_str() == *kind_label)
                        })
                })
            })
    })
}

fn dataflow_reachable_cublaslt_scaled_count(egraph: &SerializedEGraph) -> usize {
    dataflow_reachable_cublaslt_count(egraph, true)
}

fn dataflow_reachable_cublaslt_raw_fp8_count(egraph: &SerializedEGraph) -> usize {
    dataflow_reachable_cublaslt_count(egraph, false)
}

fn dataflow_reachable_cublaslt_count(egraph: &SerializedEGraph, scaled: bool) -> usize {
    let reachable = dataflow_reachable_ir_classes(egraph);
    egraph
        .enodes
        .iter()
        .filter(|(node, (label, children))| {
            label == "Op"
                && reachable.contains(&egraph.node_to_class[*node])
                && children.first().is_some_and(|kind_class| {
                    egraph
                        .eclasses
                        .get(kind_class)
                        .is_some_and(|(_, kind_nodes)| {
                            kind_nodes.iter().any(|kind_node| {
                                egraph.enodes.get(kind_node).is_some_and(|(kind_label, _)| {
                                    if scaled {
                                        kind_label == "cublaslt_scaled"
                                    } else {
                                        kind_label == "cublaslt"
                                    }
                                })
                            })
                        })
                })
        })
        .count()
}

fn dataflow_reachable_ir_classes(egraph: &SerializedEGraph) -> FxHashSet<ClassId> {
    let mut reachable = FxHashSet::default();
    let mut stack = egraph.roots.clone();
    while let Some(class) = stack.pop() {
        if !reachable.insert(class.clone()) {
            continue;
        }
        let Some((sort, nodes)) = egraph.eclasses.get(&class) else {
            continue;
        };
        for node in nodes {
            let Some((label, children)) = egraph.enodes.get(node) else {
                continue;
            };
            match (sort.as_str(), label.as_str()) {
                ("IR", "Output") => {
                    if let Some(child) = children.first() {
                        stack.push(child.clone());
                    }
                }
                ("IR", "OutputJoin") => stack.extend(children.iter().cloned()),
                ("IR", "Op") => {
                    if let Some(inputs) = children.get(1) {
                        stack.push(inputs.clone());
                    }
                }
                ("IR", _) => {
                    for child in children {
                        if egraph
                            .eclasses
                            .get(child)
                            .is_some_and(|(child_sort, _)| child_sort == "IR")
                        {
                            stack.push(child.clone());
                        }
                    }
                }
                ("IList", "ICons") => stack.extend(children.iter().cloned()),
                _ => {}
            }
        }
    }
    reachable
}

fn llir_has_cublaslt(llir: &LLIRGraph) -> bool {
    !cublaslt_type_tuples(llir).is_empty()
}

fn cublaslt_type_tuples(llir: &LLIRGraph) -> Vec<CublasLtTypeTuple> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_type_tuple(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_scale_value_tuples(llir: &LLIRGraph) -> Vec<CublasLtScaleValues> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_scale_values(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_tensor_scale_input_tuples(llir: &LLIRGraph) -> Vec<(bool, bool)> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_tensor_scale_inputs(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_epilogues(llir: &LLIRGraph) -> Vec<&'static str> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_epilogue(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_epilogue_scale_tuples(llir: &LLIRGraph) -> Vec<(&'static str, CublasLtScaleValues)> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| {
            let host_op = host_op.as_ref().as_ref();
            Some((cublaslt_epilogue(host_op)?, cublaslt_scale_values(host_op)?))
        })
        .collect()
}

fn cublaslt_matrix_order_tuples(llir: &LLIRGraph) -> Vec<CublasLtMatrixOrders> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_matrix_orders(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_transpose_op_tuples(llir: &LLIRGraph) -> Vec<CublasLtTransposeOps> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_transpose_ops(host_op.as_ref().as_ref()))
        .collect()
}

fn cublaslt_c_d_layout_matches(llir: &LLIRGraph) -> Vec<bool> {
    llir.node_weights()
        .filter_map(|op| op.to_dialect::<dyn HostOp>())
        .filter_map(|host_op| cublaslt_c_d_layouts_match(host_op.as_ref().as_ref()))
        .collect()
}
