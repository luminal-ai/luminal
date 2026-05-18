use luminal::{
    dtype::DType,
    egglog_utils::{
        NodeId, SerializedEGraph, egglog_to_llir, random_initial_choice, validate_choice_set,
    },
    prelude::*,
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    host::{
        CublasLtMatrixOrders, CublasLtScaleValues, CublasLtTransposeOps, CublasLtTypeTuple, HostOp,
        cublaslt_c_d_layouts_match, cublaslt_epilogue, cublaslt_matrix_orders,
        cublaslt_scale_values, cublaslt_transpose_ops, cublaslt_type_tuple,
    },
    runtime::CudaRuntime,
};

use super::utilities::{assert_close, get_cuda_stream, gpu_supports_dtype, random_f32_vec};

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
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_mixed_low_precision_inputs_f32_output_and_c() {
    for (dtype, compute_type) in [(DType::F16, "32F"), (DType::Bf16, "32F_FAST_16BF")] {
        let expected_tuple = (
            dtype,
            dtype,
            DType::F32,
            DType::F32,
            compute_type,
            DType::F32,
        );
        assert_cublaslt_rewrite(
            build_2d_cast_matmul_plus_c_graph(
                LayoutCase {
                    name: "mixed dtype row-major",
                    a_col_major: false,
                    b_col_major: false,
                },
                dtype,
                DType::F32,
                false,
            ),
            "mixed dtype f32 output",
            |llir| {
                cublaslt_type_tuples(llir).contains(&expected_tuple)
                    && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
            },
        );
    }
}

#[test]
#[ignore = "expensive CUDA rewrite sweep; run with cargo test -p luminal_cuda_lite -- --ignored"]
fn cublaslt_rewrites_cover_batched_mixed_low_precision_inputs_f32_output_and_c() {
    for (dtype, compute_type) in [(DType::F16, "32F"), (DType::Bf16, "32F_FAST_16BF")] {
        let expected_tuple = (
            dtype,
            dtype,
            DType::F32,
            DType::F32,
            compute_type,
            DType::F32,
        );
        assert_cublaslt_rewrite(
            build_batched_cast_matmul_plus_c_graph(
                LayoutCase {
                    name: "batched mixed dtype row-major",
                    a_col_major: false,
                    b_col_major: false,
                },
                dtype,
                DType::F32,
                false,
            ),
            "batched mixed dtype f32 output",
            |llir| {
                cublaslt_type_tuples(llir).contains(&expected_tuple)
                    && cublaslt_scale_value_tuples(llir).contains(&(1.0, 1.0))
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

    let lhs = a.expand_dim(1, n);
    let rhs = b.permute((1, 0)).expand_dim(0, m);
    let mul = unchecked_mul_same_shape(lhs, rhs, a_dtype);
    let out = mul.sum(2).cast(DType::F32).output();
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

    let lhs = a.expand_dim(2, n);
    let rhs = b.permute((0, 2, 1)).expand_dim(1, m);
    let mul = unchecked_mul_same_shape(lhs, rhs, a_dtype);
    let out = mul.sum(3).cast(DType::F32).output();
    (a, b_input, out)
}

fn unchecked_mul_same_shape(lhs: GraphTensor, rhs: GraphTensor, dtype: DType) -> GraphTensor {
    let shape = lhs.shape.contiguous();
    let new_id = lhs.graph().add_op(
        luminal::hlir::Mul {
            input_shapes: vec![lhs.shape, rhs.shape],
            ..Default::default()
        },
        &[lhs.id, rhs.id],
    );
    GraphTensor::from_id(new_id, shape, lhs.graph_ref, dtype)
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
            |x| x.gelu(),
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
                    |x| x.gelu(),
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
    let out = a.matmul(b).gelu().output();
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
    let out = a.matmul(b).gelu().output();
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
    let out = (a.matmul(b) + bias_expanded).gelu().output();
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
    let out = (a.matmul(b) + bias_expanded).gelu().output();
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

fn build_2d_cast_matmul_plus_c_graph(
    case: LayoutCase,
    input_dtype: DType,
    output_dtype: DType,
    commuted: bool,
) -> Graph {
    build_2d_layout_graph(case, input_dtype, input_dtype, |cx, a, b, m, n, _| {
        let c = cx.tensor((m, n)).as_dtype(output_dtype);
        add_commuted(a.matmul(b).cast(output_dtype), c, commuted)
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
        add_commuted(a.matmul(b), bias, commuted).gelu()
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
    build_same_dtype_2d_graph(case, dtype, |_, a, b, _, _, _| a.matmul(b).gelu())
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

fn build_batched_cast_matmul_plus_c_graph(
    case: LayoutCase,
    input_dtype: DType,
    output_dtype: DType,
    commuted: bool,
) -> Graph {
    build_batched_layout_graph(
        case,
        input_dtype,
        input_dtype,
        |cx, a, b, batch, m, n, _| {
            let c = cx.tensor((batch, m, n)).as_dtype(output_dtype);
            add_commuted(a.matmul(b).cast(output_dtype), c, commuted)
        },
    )
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
        add_commuted(a.matmul(b), bias, commuted).gelu()
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
    build_same_dtype_batched_graph(case, dtype, |_, a, b, _, _, _, _| a.matmul(b).gelu())
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
    cx.build_search_space::<CudaRuntime>();

    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .egglog_ops()
        .expect("search space should have registered egglog ops");
    let cublaslt_nodes = cublaslt_ir_nodes(egraph);
    assert!(
        !cublaslt_nodes.is_empty(),
        "expected a cublasLt rewrite candidate for {case_name}, but no cublaslt Op appeared"
    );

    let mut last_error = None;
    for (idx, cublaslt_node) in cublaslt_nodes.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0x00C0_B1A5 + idx as u64);
        let mut choices = random_initial_choice(egraph, &mut rng);
        let cublaslt_class = &egraph.node_to_class[*cublaslt_node];
        choices.insert(cublaslt_class, cublaslt_node);

        if let Err(err) = validate_choice_set(egraph, &choices, ops) {
            last_error = Some(err);
            continue;
        }

        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );

        if !cublaslt_type_tuples(&llir).is_empty() && matches(&llir) {
            return llir;
        }

        last_error =
            Some("forced cublaslt candidate did not satisfy requested extracted shape".into());
    }

    panic!(
        "expected to extract a CuBlasLt HostOp for {case_name}; last error: {}",
        last_error.unwrap_or_else(|| "no candidate could be forced".into())
    );
}

fn assert_no_forced_cublaslt_llir_where(
    cx: &mut Graph,
    case_name: &str,
    matches: impl Fn(&LLIRGraph) -> bool,
) {
    cx.build_search_space::<CudaRuntime>();

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

        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );

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
    cx.build_search_space::<CudaRuntime>();

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

        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );

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
    let cublaslt_kind_classes = egraph
        .enodes
        .iter()
        .filter(|(_, (label, _))| label == "cublaslt")
        .map(|(node, _)| egraph.node_to_class[node].clone())
        .collect::<Vec<_>>();

    egraph
        .enodes
        .iter()
        .filter_map(|(node, (label, children))| {
            (label == "Op"
                && children
                    .first()
                    .is_some_and(|kind| cublaslt_kind_classes.contains(kind)))
            .then_some(node)
        })
        .collect()
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
