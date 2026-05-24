use std::sync::{Arc, OnceLock};

use half::{bf16, f16};
use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND, STRING},
        extract_dtype, extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::{
        tracing::{Level, span, trace},
        *,
    },
};

use crate::{
    cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasComputeType_t, cublasLtEpilogue_t, cublasLtMatmul,
                cublasLtMatmulAlgoGetHeuristic, cublasLtMatmulDesc_t,
                cublasLtMatmulDescAttributes_t, cublasLtMatmulDescCreate,
                cublasLtMatmulDescDestroy, cublasLtMatmulDescSetAttribute,
                cublasLtMatmulHeuristicResult_t, cublasLtMatmulPreference_t,
                cublasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceCreate,
                cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
                cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t,
                cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
                cublasLtMatrixLayoutSetAttribute, cublasLtOrder_t, cudaDataType,
            },
        },
        driver::{CudaStream, DevicePtr},
    },
    host::{DeviceBuffer, HostOp},
    try_create_cublaslt,
};

fn parse_cublas_op(s: &str) -> cublasOperation_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "T" => cublasOperation_t::CUBLAS_OP_T,
        "N" => cublasOperation_t::CUBLAS_OP_N,
        "C" => cublasOperation_t::CUBLAS_OP_C,
        other => panic!("Unknown cuBLAS operation: '{other}' (original: '{s}')"),
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    a_order: cublasLtOrder_t,
    b_order: cublasLtOrder_t,
    c_order: cublasLtOrder_t,
    d_order: cublasLtOrder_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    ldd: Expression,
    batch_count: Expression,
    stride_a: Expression,
    stride_b: Expression,
    stride_c: Expression,
    stride_d: Expression,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    compute_type: cublasComputeType_t,
    scale_dtype: DType,
    alpha: f64,
    beta: f64,
    epilogue: cublasLtEpilogue_t,
    a_scale_input: bool,
    b_scale_input: bool,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
}

// Useless default for IntoEgglogOp
impl Default for CuBlasLt {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: cublasOperation_t::CUBLAS_OP_N,
            b_layout: cublasOperation_t::CUBLAS_OP_T,
            a_order: cublasLtOrder_t::CUBLASLT_ORDER_COL,
            b_order: cublasLtOrder_t::CUBLASLT_ORDER_COL,
            c_order: cublasLtOrder_t::CUBLASLT_ORDER_COL,
            d_order: cublasLtOrder_t::CUBLASLT_ORDER_COL,
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            ldd: Expression::default(),
            batch_count: 1.into(),
            stride_a: 0.into(),
            stride_b: 0.into(),
            stride_c: 0.into(),
            stride_d: 0.into(),
            a_dtype: DType::F32,
            b_dtype: DType::F32,
            c_dtype: DType::F32,
            d_dtype: DType::F32,
            compute_type: cublasComputeType_t::CUBLAS_COMPUTE_32F,
            scale_dtype: DType::F32,
            alpha: 1.0,
            beta: 0.0,
            epilogue: cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            a_scale_input: false,
            b_scale_input: false,
            cublaslt: OnceLock::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct CuBlasLtScaled;

fn cublaslt_sort(name: &'static str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[
            ("m", EXPRESSION),
            ("n", EXPRESSION),
            ("k", EXPRESSION),
            ("a_layout", STRING),
            ("b_layout", STRING),
            ("a_order", STRING),
            ("b_order", STRING),
            ("c_order", STRING),
            ("d_order", STRING),
            ("lda", EXPRESSION),
            ("ldb", EXPRESSION),
            ("ldc", EXPRESSION),
            ("ldd", EXPRESSION),
            ("batch_count", EXPRESSION),
            ("stride_a", EXPRESSION),
            ("stride_b", EXPRESSION),
            ("stride_c", EXPRESSION),
            ("stride_d", EXPRESSION),
            ("a_dtype", DTYPE),
            ("b_dtype", DTYPE),
            ("c_dtype", DTYPE),
            ("d_dtype", DTYPE),
            ("compute_type", STRING),
            ("scale_dtype", STRING),
            ("alpha", F64),
            ("beta", F64),
            ("epilogue", STRING),
        ],
    )
}

impl EgglogOp for CuBlasLt {
    fn sort(&self) -> SortDef {
        cublaslt_sort("cublaslt")
    }

    fn n_inputs(&self) -> usize {
        let c_input = usize::from(self.beta != 0.0);
        let bias_input = usize::from(epilogue_uses_bias(self.epilogue));
        let scale_inputs = usize::from(self.a_scale_input) + usize::from(self.b_scale_input);
        2 + c_input + bias_input + scale_inputs
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(relation cublaslt_base_dtype (DType))
                 (cublaslt_base_dtype (F32))
                 (cublaslt_base_dtype (F16))
                 (cublaslt_base_dtype (Bf16))
                 (cublaslt_base_dtype (TF32))
                 (relation cublaslt_fp8_dtype (DType))
                 (cublaslt_fp8_dtype (F8E4M3))
                 (cublaslt_fp8_dtype (F8E5M2))
                 (relation cublaslt_fp8_f32_output_pair (DType DType))
                 (cublaslt_fp8_f32_output_pair (F8E4M3) (F8E4M3))
                 (cublaslt_fp8_f32_output_pair (F8E4M3) (F8E5M2))
                 (cublaslt_fp8_f32_output_pair (F8E5M2) (F8E4M3))",
            ),
            Rule::raw(include_str!["cublaslt_RmRm_rewrite.egg"]), // row row
            Rule::raw(include_str!["cublaslt_RmCm_rewrite.egg"]), // row col
            Rule::raw(include_str!["cublaslt_CmRm_rewrite.egg"]), // col row
            Rule::raw(include_str!["cublaslt_CmCm_rewrite.egg"]), // col col
            Rule::raw(include_str!["cublaslt_fp8_rewrite.egg"]),
            Rule::raw(include_str!["cublaslt_mixed_dtype_rewrite.egg"]),
            Rule::raw(include_str!["cublaslt_scale_rewrite.egg"]),
            Rule::raw(include_str!["cublaslt_beta_rewrite.egg"]),
            Rule::raw(include_str!["cublaslt_epilogue_rewrite.egg"]),
            Rule::raw(include_str!["cublaslt_row_order_rewrite.egg"]),
            // cuBLASLt now specializes GenericMatmul, so cleanup should prune
            // the matmul output alternatives directly. Do not delete the
            // broadcast Mul here; it may still have non-matmul consumers.
            Rule::raw("(rule
                ((= ?sum (Op (Sum ?shape ?k ?sis ?sks ?sos) ?inputs))
                 (= ?sum (Op (cublaslt ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (Sum ?shape ?k ?sis ?sks ?sos) ?inputs)))
                :ruleset cleanup
                :name \"delete-sum-when-cublaslt-exists\"
            )"),
            Rule::raw("(rule
                ((= ?sum (Op (KernelSum ?shape ?k ?sis ?sks ?sos ?dt) ?inputs))
                 (= ?sum (Op (cublaslt ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (KernelSum ?shape ?k ?sis ?sks ?sos ?dt) ?inputs)))
                :ruleset cleanup
                :name \"delete-kernel-sum-when-cublaslt-exists\"
            )"),
            Rule::raw("(rule
                ((= ?sum (Op (Sum ?shape ?k ?sis ?sks ?sos) ?inputs))
                 (= ?sum (Op (cublaslt_scaled ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (Sum ?shape ?k ?sis ?sks ?sos) ?inputs)))
                :ruleset cleanup
                :name \"delete-sum-when-scaled-cublaslt-exists\"
            )"),
            Rule::raw("(rule
                ((= ?sum (Op (KernelSum ?shape ?k ?sis ?sks ?sos ?dt) ?inputs))
                 (= ?sum (Op (cublaslt_scaled ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?ci)))
                ((delete (Op (KernelSum ?shape ?k ?sis ?sks ?sos ?dt) ?inputs)))
                :ruleset cleanup
                :name \"delete-kernel-sum-when-scaled-cublaslt-exists\"
            )"),
            Rule::raw("(rule
                ((= ?sum (Op (GenericMatmul ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?gdt) ?generic_inputs))
                 (= ?sum (Op (cublaslt ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?cublas_inputs)))
                ((delete (Op (GenericMatmul ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?gdt) ?generic_inputs)))
                :ruleset cleanup
                :name \"prefer-cublaslt-over-generic-matmul\"
            )"),
            Rule::raw("(rule
                ((= ?sum (Op (GenericMatmul ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?gdt) ?generic_inputs))
                 (= ?sum (Op (cublaslt_scaled ?cm ?cn ?ck ?cta ?ctb ?cao ?cbo ?cco ?cdo ?clda ?cldb ?cldc ?cldd ?cbc ?csa ?csb ?csc ?csd ?cadt ?cbdt ?ccdt ?cddt ?ccompute ?cscale ?calpha ?cbeta ?cepilogue) ?cublas_inputs)))
                ((delete (Op (GenericMatmul ?go ?gm ?gk ?gls ?grs ?gsis ?gsit ?gos ?gdt) ?generic_inputs)))
                :ruleset cleanup
                :name \"prefer-scaled-cublaslt-over-generic-matmul\"
            )"),
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let n = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();

        // Extract transpose/layout strings from egglog
        let a_layout_str = &egraph.enodes[kind_children[3]].0;
        let b_layout_str = &egraph.enodes[kind_children[4]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);
        let a_order = parse_cublaslt_order(&egraph.enodes[kind_children[5]].0);
        let b_order = parse_cublaslt_order(&egraph.enodes[kind_children[6]].0);
        let c_order = parse_cublaslt_order(&egraph.enodes[kind_children[7]].0);
        let d_order = parse_cublaslt_order(&egraph.enodes[kind_children[8]].0);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[10], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[11], expr_cache).unwrap();
        let ldd = extract_expr(egraph, kind_children[12], expr_cache).unwrap();

        // Extract batch parameters
        let batch_count = extract_expr(egraph, kind_children[13], expr_cache).unwrap();
        let stride_a = extract_expr(egraph, kind_children[14], expr_cache).unwrap();
        let stride_b = extract_expr(egraph, kind_children[15], expr_cache).unwrap();
        let stride_c = extract_expr(egraph, kind_children[16], expr_cache).unwrap();
        let stride_d = extract_expr(egraph, kind_children[17], expr_cache).unwrap();

        // Extract cuBLASLt type tuple from egglog. Existing rewrites emit the
        // same dtype for A/B/C/D, but keeping these fields separate lets later
        // rewrites model mixed-input and mixed-output matmuls without changing
        // the host launch helper again.
        let a_dtype = extract_dtype(egraph, kind_children[18]);
        let b_dtype = extract_dtype(egraph, kind_children[19]);
        let c_dtype = extract_dtype(egraph, kind_children[20]);
        let d_dtype = extract_dtype(egraph, kind_children[21]);
        let compute_type_str = &egraph.enodes[kind_children[22]].0;
        let scale_dtype_str = &egraph.enodes[kind_children[23]].0;
        let compute_type = parse_cublaslt_compute_type(compute_type_str, a_dtype);
        let scale_dtype = parse_cublaslt_scale_dtype(scale_dtype_str, a_dtype);
        let alpha = parse_cublaslt_scalar(&egraph.enodes[kind_children[24]].0);
        let beta = parse_cublaslt_scalar(&egraph.enodes[kind_children[25]].0);
        let epilogue = parse_cublaslt_epilogue(&egraph.enodes[kind_children[26]].0);

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            a_order,
            b_order,
            c_order,
            d_order,
            lda,
            ldb,
            ldc,
            ldd,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            stride_d,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            compute_type,
            scale_dtype,
            alpha,
            beta,
            epilogue,
            a_scale_input: false,
            b_scale_input: false,
            cublaslt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl EgglogOp for CuBlasLtScaled {
    fn sort(&self) -> SortDef {
        cublaslt_sort("cublaslt_scaled")
    }

    fn n_inputs(&self) -> usize {
        4
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let m = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let n = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();

        let a_layout = parse_cublas_op(&egraph.enodes[kind_children[3]].0);
        let b_layout = parse_cublas_op(&egraph.enodes[kind_children[4]].0);
        let a_order = parse_cublaslt_order(&egraph.enodes[kind_children[5]].0);
        let b_order = parse_cublaslt_order(&egraph.enodes[kind_children[6]].0);
        let c_order = parse_cublaslt_order(&egraph.enodes[kind_children[7]].0);
        let d_order = parse_cublaslt_order(&egraph.enodes[kind_children[8]].0);

        let lda = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[10], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[11], expr_cache).unwrap();
        let ldd = extract_expr(egraph, kind_children[12], expr_cache).unwrap();

        let batch_count = extract_expr(egraph, kind_children[13], expr_cache).unwrap();
        let stride_a = extract_expr(egraph, kind_children[14], expr_cache).unwrap();
        let stride_b = extract_expr(egraph, kind_children[15], expr_cache).unwrap();
        let stride_c = extract_expr(egraph, kind_children[16], expr_cache).unwrap();
        let stride_d = extract_expr(egraph, kind_children[17], expr_cache).unwrap();

        let a_dtype = extract_dtype(egraph, kind_children[18]);
        let b_dtype = extract_dtype(egraph, kind_children[19]);
        let c_dtype = extract_dtype(egraph, kind_children[20]);
        let d_dtype = extract_dtype(egraph, kind_children[21]);
        let compute_type_str = &egraph.enodes[kind_children[22]].0;
        let scale_dtype_str = &egraph.enodes[kind_children[23]].0;
        let compute_type = parse_cublaslt_compute_type(compute_type_str, a_dtype);
        let scale_dtype = parse_cublaslt_scale_dtype(scale_dtype_str, a_dtype);
        let alpha = parse_cublaslt_scalar(&egraph.enodes[kind_children[24]].0);
        let beta = parse_cublaslt_scalar(&egraph.enodes[kind_children[25]].0);
        let epilogue = parse_cublaslt_epilogue(&egraph.enodes[kind_children[26]].0);

        let extracted_state = CuBlasLt {
            m,
            n,
            k,
            a_layout,
            b_layout,
            a_order,
            b_order,
            c_order,
            d_order,
            lda,
            ldb,
            ldc,
            ldd,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            stride_d,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            compute_type,
            scale_dtype,
            alpha,
            beta,
            epilogue,
            a_scale_input: true,
            b_scale_input: true,
            cublaslt: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

/// Convert DType to CUDA matrix/scale type for cuBLAS LT.
fn dtype_to_cuda_dtype(dtype: DType) -> cudaDataType {
    match dtype {
        DType::F64 => cudaDataType::CUDA_R_64F,
        DType::F32 => cudaDataType::CUDA_R_32F,
        DType::F16 => cudaDataType::CUDA_R_16F,
        DType::Bf16 => cudaDataType::CUDA_R_16BF,
        // TF32 is a compute mode over f32 storage.
        DType::TF32 => cudaDataType::CUDA_R_32F,
        DType::F8E4M3 => cudaDataType::CUDA_R_8F_E4M3,
        DType::F8E5M2 => cudaDataType::CUDA_R_8F_E5M2,
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        other => todo!("cuBLAS LT matmul not yet implemented for {other}"),
    }
}

fn default_compute_and_scale_dtype(dtype: DType) -> (cublasComputeType_t, DType) {
    match dtype {
        DType::F64 => (cublasComputeType_t::CUBLAS_COMPUTE_64F, DType::F64),
        DType::F32 => (cublasComputeType_t::CUBLAS_COMPUTE_32F, DType::F32),
        DType::F16 => (cublasComputeType_t::CUBLAS_COMPUTE_32F, DType::F32),
        DType::Bf16 => (
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            DType::F32,
        ),
        DType::TF32 => (
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            DType::F32,
        ),
        DType::F8E4M3 | DType::F8E5M2 => (cublasComputeType_t::CUBLAS_COMPUTE_32F, DType::F32),
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        other => todo!("cuBLAS LT matmul not yet implemented for {other}"),
    }
}

fn parse_cublaslt_compute_type(s: &str, default_dtype: DType) -> cublasComputeType_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "default" => default_compute_and_scale_dtype(default_dtype).0,
        "64F" => cublasComputeType_t::CUBLAS_COMPUTE_64F,
        "32F" => cublasComputeType_t::CUBLAS_COMPUTE_32F,
        "32F_FAST_16BF" => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        "32F_FAST_TF32" => cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
        other => panic!("Unknown cuBLASLt compute type: '{other}' (original: '{s}')"),
    }
}

fn parse_cublaslt_scale_dtype(s: &str, default_dtype: DType) -> DType {
    let stripped = s.trim_matches('"');
    match stripped {
        "default" => default_compute_and_scale_dtype(default_dtype).1,
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "Bf16" => DType::Bf16,
        "TF32" => DType::TF32,
        "F8E4M3" => DType::F8E4M3,
        "F8E5M2" => DType::F8E5M2,
        other => panic!("Unknown cuBLASLt scale dtype: '{other}' (original: '{s}')"),
    }
}

fn parse_cublaslt_scalar(s: &str) -> f64 {
    let stripped = s.trim_matches('"');
    stripped.parse::<f64>().unwrap_or_else(|_| {
        panic!("Unknown cuBLASLt scalar literal: '{stripped}' (original: '{s}')")
    })
}

fn parse_cublaslt_order(s: &str) -> cublasLtOrder_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "COL" => cublasLtOrder_t::CUBLASLT_ORDER_COL,
        "ROW" => cublasLtOrder_t::CUBLASLT_ORDER_ROW,
        "COL32" => cublasLtOrder_t::CUBLASLT_ORDER_COL32,
        "COL4_4R2_8C" => cublasLtOrder_t::CUBLASLT_ORDER_COL4_4R2_8C,
        "COL32_2R_4R4" => cublasLtOrder_t::CUBLASLT_ORDER_COL32_2R_4R4,
        other => panic!("Unknown cuBLASLt matrix order: '{other}' (original: '{s}')"),
    }
}

fn parse_cublaslt_epilogue(s: &str) -> cublasLtEpilogue_t {
    let stripped = s.trim_matches('"');
    match stripped {
        "DEFAULT" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
        "BIAS" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
        "RELU" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
        "RELU_BIAS" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
        "GELU" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
        "GELU_BIAS" => cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        other => panic!("Unknown cuBLASLt epilogue: '{other}' (original: '{s}')"),
    }
}

#[cfg(test)]
fn compute_type_name(compute_type: cublasComputeType_t) -> &'static str {
    match compute_type {
        cublasComputeType_t::CUBLAS_COMPUTE_64F => "64F",
        cublasComputeType_t::CUBLAS_COMPUTE_32F => "32F",
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF => "32F_FAST_16BF",
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32 => "32F_FAST_TF32",
        _ => "other",
    }
}

#[cfg(test)]
fn order_name(order: cublasLtOrder_t) -> &'static str {
    match order {
        cublasLtOrder_t::CUBLASLT_ORDER_COL => "COL",
        cublasLtOrder_t::CUBLASLT_ORDER_ROW => "ROW",
        cublasLtOrder_t::CUBLASLT_ORDER_COL32 => "COL32",
        cublasLtOrder_t::CUBLASLT_ORDER_COL4_4R2_8C => "COL4_4R2_8C",
        cublasLtOrder_t::CUBLASLT_ORDER_COL32_2R_4R4 => "COL32_2R_4R4",
    }
}

#[cfg(test)]
fn transpose_op_name(op: cublasOperation_t) -> &'static str {
    match op {
        cublasOperation_t::CUBLAS_OP_N => "N",
        cublasOperation_t::CUBLAS_OP_T => "T",
        cublasOperation_t::CUBLAS_OP_C => "C",
        cublasOperation_t::CUBLAS_OP_CONJG => "CONJG",
    }
}

#[cfg(test)]
fn epilogue_name(epilogue: cublasLtEpilogue_t) -> &'static str {
    match epilogue {
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT => "DEFAULT",
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS => "BIAS",
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU => "RELU",
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS => "RELU_BIAS",
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU => "GELU",
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS => "GELU_BIAS",
        _ => "other",
    }
}

#[derive(Debug, Clone, Copy)]
enum LtScalar {
    F64(f64),
    F32(f32),
    F16(f16),
    Bf16(bf16),
}

impl LtScalar {
    #[cfg(test)]
    fn one(dtype: DType) -> anyhow::Result<Self> {
        Self::from_f64(dtype, 1.0)
    }

    #[cfg(test)]
    fn zero(dtype: DType) -> anyhow::Result<Self> {
        Self::from_f64(dtype, 0.0)
    }

    fn from_f64(dtype: DType, value: f64) -> anyhow::Result<Self> {
        match dtype {
            DType::F64 => Ok(Self::F64(value)),
            DType::F32 => Ok(Self::F32(value as f32)),
            DType::F16 => Ok(Self::F16(f16::from_f32(value as f32))),
            DType::Bf16 => Ok(Self::Bf16(bf16::from_f32(value as f32))),
            other => Err(anyhow::anyhow!(
                "cuBLASLt scale dtype {other} is not supported for host alpha/beta scalars"
            )),
        }
    }

    fn as_ptr(&self) -> *const std::ffi::c_void {
        match self {
            Self::F64(value) => value as *const _ as *const std::ffi::c_void,
            Self::F32(value) => value as *const _ as *const std::ffi::c_void,
            Self::F16(value) => value as *const _ as *const std::ffi::c_void,
            Self::Bf16(value) => value as *const _ as *const std::ffi::c_void,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulProblem {
    m: u64,
    n: u64,
    k: u64,
    batch_count: i32,
}

#[derive(Debug, Clone, Copy)]
struct LtMatrixSpec {
    dtype: cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
    batch_stride: i64,
    order: cublasLtOrder_t,
}

#[derive(Debug, Clone, Copy)]
struct LtComputeSpec {
    compute_type: cublasComputeType_t,
    scale_dtype: cudaDataType,
    alpha: LtScalar,
    beta: LtScalar,
    epilogue: cublasLtEpilogue_t,
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulSpec {
    problem: LtMatmulProblem,
    trans_a: cublasOperation_t,
    trans_b: cublasOperation_t,
    a: LtMatrixSpec,
    b: LtMatrixSpec,
    c: LtMatrixSpec,
    d: LtMatrixSpec,
    compute: LtComputeSpec,
    workspace_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct LtMatmulPointers {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    bias: Option<u64>,
    a_scale: Option<u64>,
    b_scale: Option<u64>,
}

struct LtRawDescriptors {
    matmul_desc: cublasLtMatmulDesc_t,
    a_desc: cublasLtMatrixLayout_t,
    b_desc: cublasLtMatrixLayout_t,
    c_desc: cublasLtMatrixLayout_t,
    d_desc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
}

impl Default for LtRawDescriptors {
    fn default() -> Self {
        Self {
            matmul_desc: std::ptr::null_mut(),
            a_desc: std::ptr::null_mut(),
            b_desc: std::ptr::null_mut(),
            c_desc: std::ptr::null_mut(),
            d_desc: std::ptr::null_mut(),
            preference: std::ptr::null_mut(),
        }
    }
}

impl Drop for LtRawDescriptors {
    fn drop(&mut self) {
        unsafe {
            if !self.preference.is_null() {
                let _ = cublasLtMatmulPreferenceDestroy(self.preference);
            }
            if !self.d_desc.is_null() {
                let _ = cublasLtMatrixLayoutDestroy(self.d_desc);
            }
            if !self.c_desc.is_null() {
                let _ = cublasLtMatrixLayoutDestroy(self.c_desc);
            }
            if !self.b_desc.is_null() {
                let _ = cublasLtMatrixLayoutDestroy(self.b_desc);
            }
            if !self.a_desc.is_null() {
                let _ = cublasLtMatrixLayoutDestroy(self.a_desc);
            }
            if !self.matmul_desc.is_null() {
                let _ = cublasLtMatmulDescDestroy(self.matmul_desc);
            }
        }
    }
}

fn create_matrix_layout(
    desc: &mut cublasLtMatrixLayout_t,
    spec: LtMatrixSpec,
) -> anyhow::Result<()> {
    unsafe {
        cublasLtMatrixLayoutCreate(desc, spec.dtype, spec.rows, spec.cols, spec.ld).result()?;
        cublasLtMatrixLayoutSetAttribute(
            *desc,
            cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER,
            &spec.order as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasLtOrder_t>(),
        )
        .result()?;
    }
    Ok(())
}

fn clamp_ld_for_order(ld: i64, rows: u64, cols: u64, order: cublasLtOrder_t) -> i64 {
    let min_ld = match order {
        cublasLtOrder_t::CUBLASLT_ORDER_COL => rows,
        cublasLtOrder_t::CUBLASLT_ORDER_ROW => cols,
        cublasLtOrder_t::CUBLASLT_ORDER_COL32
        | cublasLtOrder_t::CUBLASLT_ORDER_COL4_4R2_8C
        | cublasLtOrder_t::CUBLASLT_ORDER_COL32_2R_4R4 => rows,
    };
    std::cmp::max(ld, min_ld as i64)
}

fn set_strided_batch_layout(
    desc: cublasLtMatrixLayout_t,
    batch_count: i32,
    batch_stride: i64,
) -> anyhow::Result<()> {
    unsafe {
        cublasLtMatrixLayoutSetAttribute(
            desc,
            cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batch_count as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<i32>(),
        )
        .result()?;
        cublasLtMatrixLayoutSetAttribute(
            desc,
            cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
            &batch_stride as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<i64>(),
        )
        .result()?;
    }
    Ok(())
}

fn cuda_dtype_needs_tensorwide_scale(dtype: cudaDataType) -> bool {
    matches!(
        dtype,
        cudaDataType::CUDA_R_8F_E4M3 | cudaDataType::CUDA_R_8F_E5M2
    )
}

fn set_scalar_scale_pointer(
    matmul_desc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    ptr: u64,
) -> anyhow::Result<()> {
    unsafe {
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            attr,
            &ptr as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<u64>(),
        )
        .result()?;
    }
    Ok(())
}

fn run_cublaslt_matmul(
    stream: &Arc<CudaStream>,
    cublaslt: &Arc<CudaBlasLT>,
    spec: &LtMatmulSpec,
    ptrs: LtMatmulPointers,
) -> anyhow::Result<()> {
    if spec.problem.m == 0 || spec.problem.n == 0 || spec.problem.k == 0 {
        return Err(anyhow::anyhow!(
            "cuBLASLT matmul got zero-sized dimensions: m={}, n={}, k={}",
            spec.problem.m,
            spec.problem.n,
            spec.problem.k
        ));
    }

    let mut resources = LtRawDescriptors::default();
    let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    let workspace = unsafe { stream.alloc::<u8>(spec.workspace_size)? };
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

    let a_scale = if cuda_dtype_needs_tensorwide_scale(spec.a.dtype) && ptrs.a_scale.is_none() {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let b_scale = if cuda_dtype_needs_tensorwide_scale(spec.b.dtype) && ptrs.b_scale.is_none() {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let c_scale = if cuda_dtype_needs_tensorwide_scale(spec.c.dtype) {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };
    let d_scale = if cuda_dtype_needs_tensorwide_scale(spec.d.dtype) {
        Some(stream.clone_htod(&[1.0f32])?)
    } else {
        None
    };

    unsafe {
        cublasLtMatmulDescCreate(
            &mut resources.matmul_desc,
            spec.compute.compute_type,
            spec.compute.scale_dtype,
        )
        .result()?;

        cublasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &spec.trans_a as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &spec.trans_b as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
            &spec.compute.epilogue as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasLtEpilogue_t>(),
        )
        .result()?;
        if let Some(bias_ptr) = ptrs.bias {
            cublasLtMatmulDescSetAttribute(
                resources.matmul_desc,
                cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias_ptr as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<u64>(),
            )
            .result()?;
        }
    }

    let (a_scale_ptr, _a_scale_guard) = if let Some(ptr) = ptrs.a_scale {
        (Some(ptr), None)
    } else if let Some(scale) = &a_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (b_scale_ptr, _b_scale_guard) = if let Some(ptr) = ptrs.b_scale {
        (Some(ptr), None)
    } else if let Some(scale) = &b_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (c_scale_ptr, _c_scale_guard) = if let Some(scale) = &c_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    let (d_scale_ptr, _d_scale_guard) = if let Some(scale) = &d_scale {
        let (ptr, guard) = scale.device_ptr(stream);
        (Some(ptr), Some(guard))
    } else {
        (None, None)
    };
    if let Some(ptr) = a_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = b_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = c_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
            ptr,
        )?;
    }
    if let Some(ptr) = d_scale_ptr {
        set_scalar_scale_pointer(
            resources.matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
            ptr,
        )?;
    }

    create_matrix_layout(&mut resources.a_desc, spec.a)?;
    create_matrix_layout(&mut resources.b_desc, spec.b)?;
    create_matrix_layout(&mut resources.c_desc, spec.c)?;
    create_matrix_layout(&mut resources.d_desc, spec.d)?;

    if spec.problem.batch_count > 1 {
        for (desc, matrix) in [
            (resources.a_desc, spec.a),
            (resources.b_desc, spec.b),
            (resources.c_desc, spec.c),
            (resources.d_desc, spec.d),
        ] {
            set_strided_batch_layout(desc, spec.problem.batch_count, matrix.batch_stride)?;
        }
    }

    unsafe {
        cublasLtMatmulPreferenceCreate(&mut resources.preference).result()?;
        cublasLtMatmulPreferenceSetAttribute(
            resources.preference,
            cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &spec.workspace_size as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        cublasLtMatmulAlgoGetHeuristic(
            *cublaslt.handle(),
            resources.matmul_desc,
            resources.a_desc,
            resources.b_desc,
            resources.c_desc,
            resources.d_desc,
            resources.preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
        }

        let alpha_ptr = spec.compute.alpha.as_ptr();
        let beta_ptr = spec.compute.beta.as_ptr();
        cublasLtMatmul(
            *cublaslt.handle(),
            resources.matmul_desc,
            alpha_ptr,
            ptrs.a as *const std::ffi::c_void,
            resources.a_desc,
            ptrs.b as *const std::ffi::c_void,
            resources.b_desc,
            beta_ptr,
            ptrs.c as *const std::ffi::c_void,
            resources.c_desc,
            ptrs.d as *mut std::ffi::c_void,
            resources.d_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            spec.workspace_size,
            stream.cu_stream() as *mut _,
        )
        .result()?;
    }

    Ok(())
}

fn resolve_cublaslt_pointers(
    self_node: NodeIndex,
    inputs: &[NodeIndex],
    buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
    beta: f64,
    epilogue: cublasLtEpilogue_t,
    a_scale_input: bool,
    b_scale_input: bool,
) -> anyhow::Result<LtMatmulPointers> {
    if inputs.len() < 2 {
        return Err(anyhow::anyhow!(
            "cuBLASLt matmul expected at least 2 inputs (A, B[, C]), got {}",
            inputs.len()
        ));
    }

    let a = buffers
        .get(&inputs[0])
        .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt A input buffer"))?
        .ptr();
    let b = buffers
        .get(&inputs[1])
        .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt B input buffer"))?
        .ptr();
    let d = buffers
        .get(&self_node)
        .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt output buffer"))?
        .ptr();
    let mut next_input = 2;
    let c = if beta == 0.0 {
        d
    } else {
        let c_input = inputs.get(next_input).ok_or_else(|| {
            anyhow::anyhow!("cuBLASLt matmul with beta={beta} requires a third C input")
        })?;
        next_input += 1;
        buffers
            .get(c_input)
            .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt C input buffer"))?
            .ptr()
    };

    let bias = if epilogue_uses_bias(epilogue) {
        let bias_input = inputs.get(next_input).ok_or_else(|| {
            anyhow::anyhow!("cuBLASLt matmul with {epilogue:?} epilogue requires a bias input")
        })?;
        next_input += 1;
        Some(
            buffers
                .get(bias_input)
                .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt bias input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    let a_scale = if a_scale_input {
        let scale_input = inputs
            .get(next_input)
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt matmul requires an A scale input pointer"))?;
        next_input += 1;
        Some(
            buffers
                .get(scale_input)
                .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt A scale input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    let b_scale = if b_scale_input {
        let scale_input = inputs
            .get(next_input)
            .ok_or_else(|| anyhow::anyhow!("cuBLASLt matmul requires a B scale input pointer"))?;
        Some(
            buffers
                .get(scale_input)
                .ok_or_else(|| anyhow::anyhow!("missing cuBLASLt B scale input buffer"))?
                .ptr(),
        )
    } else {
        None
    };

    Ok(LtMatmulPointers {
        a,
        b,
        c,
        d,
        bias,
        a_scale,
        b_scale,
    })
}

fn epilogue_uses_bias(epilogue: cublasLtEpilogue_t) -> bool {
    matches!(
        epilogue,
        cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS
            | cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS
            | cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX_BIAS
            | cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
            | cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX_BIAS
    )
}

impl CuBlasLt {
    fn get_cublaslt(&self, stream: &Arc<CudaStream>) -> anyhow::Result<Arc<CudaBlasLT>> {
        if let Some(cublaslt) = self.cublaslt.get() {
            return Ok(cublaslt.clone());
        }
        let created = try_create_cublaslt(stream.clone()).map_err(|message| {
            anyhow::anyhow!("cuBLASLt unavailable on this machine: {message}")
        })?;
        let _ = self.cublaslt.set(created.clone());
        Ok(created)
    }

    #[cfg(test)]
    pub(crate) fn type_tuple(&self) -> (DType, DType, DType, DType, &'static str, DType) {
        (
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.d_dtype,
            compute_type_name(self.compute_type),
            self.scale_dtype,
        )
    }

    #[cfg(test)]
    pub(crate) fn scale_values(&self) -> (f64, f64) {
        (self.alpha, self.beta)
    }

    #[cfg(test)]
    pub(crate) fn epilogue(&self) -> &'static str {
        epilogue_name(self.epilogue)
    }

    #[cfg(test)]
    pub(crate) fn matrix_orders(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        (
            order_name(self.a_order),
            order_name(self.b_order),
            order_name(self.c_order),
            order_name(self.d_order),
        )
    }

    #[cfg(test)]
    pub(crate) fn transpose_ops(&self) -> (&'static str, &'static str) {
        (
            transpose_op_name(self.a_layout),
            transpose_op_name(self.b_layout),
        )
    }

    #[cfg(test)]
    pub(crate) fn c_d_layouts_match(&self) -> bool {
        let normalize = |expr: Expression| expr.substitute('z', Expression::from(1)).simplify();
        normalize(self.ldc) == normalize(self.ldd)
            && normalize(self.stride_c) == normalize(self.stride_d)
            && self.c_order == self.d_order
    }

    #[cfg(test)]
    pub(crate) fn tensor_scale_inputs(&self) -> (bool, bool) {
        (self.a_scale_input, self.b_scale_input)
    }
}

impl HostOp for CuBlasLt {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters — resolve z→1 for element stride before exec
        let resolve = |e: &Expression| -> Expression { e.substitute('z', Expression::from(1)) };
        let m = resolve(&self.m).exec(dyn_map).unwrap() as u64;
        let n = resolve(&self.n).exec(dyn_map).unwrap() as u64;
        let k = resolve(&self.k).exec(dyn_map).unwrap() as u64;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = resolve(&self.lda).exec(dyn_map).unwrap() as i64;
        let ldb = resolve(&self.ldb).exec(dyn_map).unwrap() as i64;
        let ldc = resolve(&self.ldc).exec(dyn_map).unwrap() as i64;
        let ldd = resolve(&self.ldd).exec(dyn_map).unwrap() as i64;
        let batch_count = resolve(&self.batch_count).exec(dyn_map).unwrap() as i32;
        let stride_a = resolve(&self.stride_a).exec(dyn_map).unwrap() as i64;
        let stride_b = resolve(&self.stride_b).exec(dyn_map).unwrap() as i64;
        let stride_c = resolve(&self.stride_c).exec(dyn_map).unwrap() as i64;
        let stride_d = resolve(&self.stride_d).exec(dyn_map).unwrap() as i64;

        // Get CUDA types based on the explicit cuBLASLt type tuple.
        let a_cuda_dtype = dtype_to_cuda_dtype(self.a_dtype);
        let b_cuda_dtype = dtype_to_cuda_dtype(self.b_dtype);
        let c_cuda_dtype = dtype_to_cuda_dtype(self.c_dtype);
        let d_cuda_dtype = dtype_to_cuda_dtype(self.d_dtype);
        let scale_cuda_dtype = dtype_to_cuda_dtype(self.scale_dtype);
        let element_size = (self.d_dtype.bits() / 8) as u64;
        assert!(
            element_size > 0,
            "cuBLAS LT does not support sub-byte dtype {}",
            self.d_dtype
        );

        let alpha = LtScalar::from_f64(self.scale_dtype, self.alpha)?;
        let beta = LtScalar::from_f64(self.scale_dtype, self.beta)?;

        let ptrs = resolve_cublaslt_pointers(
            self_node,
            inputs,
            buffers,
            self.beta,
            self.epilogue,
            self.a_scale_input,
            self.b_scale_input,
        )?;

        let (a_rows, a_cols) = if a_layout == cublasOperation_t::CUBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_layout == cublasOperation_t::CUBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };
        let lda = clamp_ld_for_order(lda, a_rows, a_cols, self.a_order);
        let ldb = clamp_ld_for_order(ldb, b_rows, b_cols, self.b_order);
        let ldc = clamp_ld_for_order(ldc, m, n, self.c_order);
        let ldd = clamp_ld_for_order(ldd, m, n, self.d_order);

        let _span = span!(
            Level::TRACE,
            "cuBLASLT",
            m, n, k, lda, ldb, ldc, ldd, batch_count, ?a_layout, ?b_layout,
            ?self.a_order, ?self.b_order, ?self.c_order, ?self.d_order,
            ?self.a_dtype, ?self.b_dtype, ?self.c_dtype, ?self.d_dtype,
            ?self.compute_type, ?self.scale_dtype, self.alpha, self.beta,
            ?self.epilogue,
        )
        .entered();

        let cublaslt = self.get_cublaslt(stream)?;

        // Allocate workspace (32 MiB)
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let c_spec = LtMatrixSpec {
            dtype: c_cuda_dtype,
            rows: m,
            cols: n,
            ld: ldc,
            batch_stride: stride_c,
            order: self.c_order,
        };
        let d_spec = LtMatrixSpec {
            dtype: d_cuda_dtype,
            rows: m,
            cols: n,
            ld: ldd,
            batch_stride: stride_d,
            order: self.d_order,
        };
        let spec = LtMatmulSpec {
            problem: LtMatmulProblem {
                m,
                n,
                k,
                batch_count,
            },
            trans_a: a_layout,
            trans_b: b_layout,
            a: LtMatrixSpec {
                dtype: a_cuda_dtype,
                rows: a_rows,
                cols: a_cols,
                ld: lda,
                batch_stride: stride_a,
                order: self.a_order,
            },
            b: LtMatrixSpec {
                dtype: b_cuda_dtype,
                rows: b_rows,
                cols: b_cols,
                ld: ldb,
                batch_stride: stride_b,
                order: self.b_order,
            },
            c: c_spec,
            d: d_spec,
            compute: LtComputeSpec {
                compute_type: self.compute_type,
                scale_dtype: scale_cuda_dtype,
                alpha,
                beta,
                epilogue: self.epilogue,
            },
            workspace_size: WORKSPACE_SIZE,
        };

        run_cublaslt_matmul(stream, &cublaslt, &spec, ptrs)?;

        // No stream.synchronize() here — CUDA stream ordering guarantees
        // sequential execution. The runtime syncs once at the end of execute().
        Ok(())
    }

    fn output_size(&self) -> Expression {
        let resolve = |e: &Expression| -> Expression { e.substitute('z', Expression::from(1)) };
        resolve(&self.batch_count) * resolve(&self.m) * resolve(&self.n)
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.d_dtype.bits()).ceil_div(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lt_scalar_packs_f32_scale_values() {
        match LtScalar::one(DType::F32).unwrap() {
            LtScalar::F32(value) => assert_eq!(value, 1.0),
            other => panic!("expected f32 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::F32).unwrap() {
            LtScalar::F32(value) => assert_eq!(value, 0.0),
            other => panic!("expected f32 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_packs_f64_scale_values() {
        match LtScalar::one(DType::F64).unwrap() {
            LtScalar::F64(value) => assert_eq!(value, 1.0),
            other => panic!("expected f64 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::F64).unwrap() {
            LtScalar::F64(value) => assert_eq!(value, 0.0),
            other => panic!("expected f64 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_packs_low_precision_scale_values() {
        match LtScalar::one(DType::F16).unwrap() {
            LtScalar::F16(value) => assert_eq!(f32::from(value), 1.0),
            other => panic!("expected f16 scalar, got {other:?}"),
        }

        match LtScalar::zero(DType::Bf16).unwrap() {
            LtScalar::Bf16(value) => assert_eq!(f32::from(value), 0.0),
            other => panic!("expected bf16 scalar, got {other:?}"),
        }
    }

    #[test]
    fn lt_scalar_rejects_non_host_scalar_scale_dtypes() {
        assert!(LtScalar::one(DType::TF32).is_err());
        assert!(LtScalar::zero(DType::F8E4M3).is_err());
    }

    #[test]
    fn fp8_cuda_dtypes_request_tensorwide_scales() {
        assert!(cuda_dtype_needs_tensorwide_scale(
            cudaDataType::CUDA_R_8F_E4M3
        ));
        assert!(cuda_dtype_needs_tensorwide_scale(
            cudaDataType::CUDA_R_8F_E5M2
        ));
        assert!(!cuda_dtype_needs_tensorwide_scale(cudaDataType::CUDA_R_32F));
    }

    #[test]
    fn cublaslt_pointers_alias_output_as_c_for_two_input_beta_zero() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let ptrs = resolve_cublaslt_pointers(
            output,
            &[a, b],
            &buffers,
            0.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn cublaslt_pointers_ignore_extra_inputs_for_beta_zero() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let extra = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (extra, 0xEEEE)]);

        let ptrs = resolve_cublaslt_pointers(
            output,
            &[a, b, extra],
            &buffers,
            0.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn cublaslt_pointers_use_distinct_c_input_when_present() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let c = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (c, 0xC000)]);

        let ptrs = resolve_cublaslt_pointers(
            output,
            &[a, b, c],
            &buffers,
            1.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xC000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
    }

    #[test]
    fn cublaslt_pointers_use_bias_input_for_bias_epilogue() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let bias = NodeIndex::new(3);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000), (bias, 0xB1A5)]);

        let ptrs = resolve_cublaslt_pointers(
            output,
            &[a, b, bias],
            &buffers,
            0.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            false,
            false,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, Some(0xB1A5));
    }

    #[test]
    fn cublaslt_pointers_use_tensor_scale_inputs_after_base_inputs() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let a_scale = NodeIndex::new(3);
        let b_scale = NodeIndex::new(4);
        let buffers = buffers_for(&[
            (output, 0xD000),
            (a, 0xA000),
            (b, 0xB000),
            (a_scale, 0xA5A5),
            (b_scale, 0xB5B5),
        ]);

        let ptrs = resolve_cublaslt_pointers(
            output,
            &[a, b, a_scale, b_scale],
            &buffers,
            0.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            true,
            true,
        )
        .unwrap();

        assert_eq!(ptrs.a, 0xA000);
        assert_eq!(ptrs.b, 0xB000);
        assert_eq!(ptrs.c, 0xD000);
        assert_eq!(ptrs.d, 0xD000);
        assert_eq!(ptrs.bias, None);
        assert_eq!(ptrs.a_scale, Some(0xA5A5));
        assert_eq!(ptrs.b_scale, Some(0xB5B5));
    }

    #[test]
    fn cublaslt_pointers_reject_two_input_nonzero_beta() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let err = resolve_cublaslt_pointers(
            output,
            &[a, b],
            &buffers,
            1.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            false,
            false,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("requires a third C input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cublaslt_pointers_reject_missing_bias_input() {
        let output = NodeIndex::new(0);
        let a = NodeIndex::new(1);
        let b = NodeIndex::new(2);
        let buffers = buffers_for(&[(output, 0xD000), (a, 0xA000), (b, 0xB000)]);

        let err = resolve_cublaslt_pointers(
            output,
            &[a, b],
            &buffers,
            0.0,
            cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            false,
            false,
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("requires a bias input"),
            "unexpected error: {err}"
        );
    }

    fn buffers_for(entries: &[(NodeIndex, u64)]) -> FxHashMap<NodeIndex, DeviceBuffer> {
        entries
            .iter()
            .map(|(node, ptr)| (*node, DeviceBuffer::new(*ptr, 16)))
            .collect()
    }
}
