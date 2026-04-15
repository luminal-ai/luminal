use std::sync::{Arc, OnceLock};

use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, OP_KIND, STRING},
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
                cublasComputeType_t, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmulDesc_t, cublasLtMatmulDescCreate, cublasLtMatmulDescDestroy,
                cublasLtMatmulDescSetAttribute, cublasLtMatmulHeuristicResult_t,
                cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t,
                cublasLtMatmulPreferenceCreate, cublasLtMatmulPreferenceDestroy,
                cublasLtMatmulPreferenceSetAttribute, cublasLtMatrixLayout_t,
                cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy, cudaDataType,
            },
        },
        driver::{CudaSlice, CudaStream, DevicePtr},
    },
    host::{HostOp, cublas::parse_cublas_op},
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct CuBlasLt {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    batch_count: Expression,
    stride_a: Expression,
    stride_b: Expression,
    stride_c: Expression,
    dtype: DType,
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
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            batch_count: 1.into(),
            stride_a: 0.into(),
            stride_b: 0.into(),
            stride_c: 0.into(),
            dtype: DType::F32,
            cublaslt: OnceLock::new(),
        }
    }
}

impl EgglogOp for CuBlasLt {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "cublaslt",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("a_layout", STRING),
                ("b_layout", STRING),
                ("lda", EXPRESSION),
                ("ldb", EXPRESSION),
                ("ldc", EXPRESSION),
                ("batch_count", EXPRESSION),
                ("stride_a", EXPRESSION),
                ("stride_b", EXPRESSION),
                ("stride_c", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(include_str!["cublaslt_RmRm_rewrite.egg"]), // row row
            Rule::raw(include_str!["cublaslt_RmCm_rewrite.egg"]), // row col
            Rule::raw(include_str!["cublaslt_CmRm_rewrite.egg"]), // col row
            Rule::raw(include_str!["cublaslt_CmCm_rewrite.egg"]), // col col
            // Delete KernelMul matmul broadcast intermediates when the Sum eclass
            // has a cublaslt or KernelBatchMatMul alternative. This prevents OOM
            // from O(m*k*n) intermediates at large seq_len. cuBLAS, TileMatmulFullSplit,
            // KernelBatchMatVec, and KernelBatchMatMul all take original inputs
            // (not the Mul eclass), so they survive the cascade.
            Rule::raw("(rule
                ((= ?mul (Op (KernelMul ?shape ?as ?bs ?os ?dt) ?inputs))
                 (= (MNum 0) (nth_from_end ?as 1))
                 (= (MNum 0) (nth_from_end ?bs 2))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (cublaslt ?cm ?cn ?ck ?cta ?ctb ?clda ?cldb ?cldc ?cbc ?csa ?csb ?csc ?cdt) ?ci)))
                ((delete (Op (KernelMul ?shape ?as ?bs ?os ?dt) ?inputs)))
                :ruleset cleanup
            )"),
            Rule::raw("(rule
                ((= ?mul (Op (KernelMul ?shape ?as ?bs ?os ?dt) ?inputs))
                 (= (MNum 0) (nth_from_end ?as 1))
                 (= (MNum 0) (nth_from_end ?bs 2))
                 (= ?sum (Op (Sum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                 (= ?sum (Op (KernelBatchMatMul ?bos ?bk ?bas ?baks ?bbs ?bbks ?bouts ?bdt) ?bi)))
                ((delete (Op (KernelMul ?shape ?as ?bs ?os ?dt) ?inputs)))
                :ruleset cleanup
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

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[kind_children[3]].0;
        let b_layout_str = &egraph.enodes[kind_children[4]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let ldb = extract_expr(egraph, kind_children[6], expr_cache).unwrap();
        let ldc = extract_expr(egraph, kind_children[7], expr_cache).unwrap();

        // Extract batch parameters
        let batch_count = extract_expr(egraph, kind_children[8], expr_cache).unwrap();
        let stride_a = extract_expr(egraph, kind_children[9], expr_cache).unwrap();
        let stride_b = extract_expr(egraph, kind_children[10], expr_cache).unwrap();
        let stride_c = extract_expr(egraph, kind_children[11], expr_cache).unwrap();

        // Extract dtype from egglog
        let dtype = extract_dtype(egraph, kind_children[12]);

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            batch_count,
            stride_a,
            stride_b,
            stride_c,
            dtype,
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

/// Convert DType to CUDA types for cuBLAS LT
/// Returns (matrix_dtype, compute_type, scale_dtype)
fn dtype_to_cuda_types(dtype: DType) -> (cudaDataType, cublasComputeType_t, cudaDataType) {
    match dtype {
        // F64: matrix=f64, compute=f64, scale=f64
        DType::F64 => (
            cudaDataType::CUDA_R_64F,
            cublasComputeType_t::CUBLAS_COMPUTE_64F,
            cudaDataType::CUDA_R_64F,
        ),
        // F32: matrix=f32, compute=f32, scale=f32
        DType::F32 => (
            cudaDataType::CUDA_R_32F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // F16: matrix=f16, compute=f32 (FP32 accumulation for accuracy), scale=f32
        DType::F16 => (
            cudaDataType::CUDA_R_16F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // BF16: matrix=bf16, compute=f32 with tensor cores, scale=f32
        DType::Bf16 => (
            cudaDataType::CUDA_R_16BF,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cudaDataType::CUDA_R_32F,
        ),
        // TF32: stored as f32, use fast TF32 tensor core path
        DType::TF32 => (
            cudaDataType::CUDA_R_32F,
            cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            cudaDataType::CUDA_R_32F,
        ),
        // FP8 E4M3: matrix=fp8_e4m3, compute=f32, scale=f32
        DType::F8E4M3 => (
            cudaDataType::CUDA_R_8F_E4M3,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        // FP8 E5M2: matrix=fp8_e5m2, compute=f32, scale=f32
        DType::F8E5M2 => (
            cudaDataType::CUDA_R_8F_E5M2,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType::CUDA_R_32F,
        ),
        DType::Int => panic!("cuBLAS LT does not support integer matmul"),
        DType::Bool => panic!("cuBLAS LT does not support bool matmul"),
        other => todo!("cuBLAS LT matmul not yet implemented for {other}"),
    }
}

impl HostOp for CuBlasLt {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, &CudaSlice<u8>>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        use crate::cudarc::cublaslt::sys::{
            cublasLtMatrixLayoutAttribute_t, cublasLtMatrixLayoutSetAttribute,
        };

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
        let batch_count = resolve(&self.batch_count).exec(dyn_map).unwrap() as i32;
        let stride_a = resolve(&self.stride_a).exec(dyn_map).unwrap() as i64;
        let stride_b = resolve(&self.stride_b).exec(dyn_map).unwrap() as i64;
        let stride_c = resolve(&self.stride_c).exec(dyn_map).unwrap() as i64;

        // Get CUDA types based on dtype
        let (cuda_dtype, compute_type, scale_dtype) = dtype_to_cuda_types(self.dtype);
        let element_size = (self.dtype.bits() / 8) as u64;
        assert!(
            element_size > 0,
            "cuBLAS LT does not support sub-byte dtype {}",
            self.dtype
        );

        // Alpha/beta scale values (all dtypes use F32 scale type)
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;

        // Get buffers: output is self_node, inputs are from graph edges
        let c_buf = buffers[&self_node];
        let a_buf = buffers[&inputs[0]];
        let b_buf = buffers[&inputs[1]];

        // Get device pointers
        let (a_ptr, _a_guard) = a_buf.device_ptr(stream);
        let (b_ptr, _b_guard) = b_buf.device_ptr(stream);
        let (c_ptr, _c_guard) = c_buf.device_ptr(stream);

        // Clamp leading dimensions to minimum valid values.
        // When a dimension is 1 (e.g., k=1 outer product), the stride along that
        // dimension may be 0 in the egglog representation, but cuBLAS requires
        // lda >= rows_of_A and ldb >= rows_of_B.
        let a_ld_min = if a_layout == cublasOperation_t::CUBLAS_OP_N {
            m
        } else {
            k
        };
        let b_ld_min = if b_layout == cublasOperation_t::CUBLAS_OP_N {
            k
        } else {
            n
        };
        let lda = std::cmp::max(lda, a_ld_min as i64);
        let ldb = std::cmp::max(ldb, b_ld_min as i64);
        let ldc = std::cmp::max(ldc, m as i64);

        let _span = span!(
            Level::TRACE,
            "cuBLASLT",
            m, n, k, lda, ldb, ldc, batch_count, ?a_layout, ?b_layout, ?self.dtype,
        )
        .entered();

        let cublaslt = self
            .cublaslt
            .get_or_init(|| Arc::new(CudaBlasLT::new(stream.clone()).unwrap()));

        let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
        let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
        let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
        let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
        let mut algo_count: i32 = 0;

        // Allocate workspace (32 MiB)
        const WORKSPACE_SIZE: usize = 32 * 1024 * 1024;
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE)? };
        let (workspace_ptr, _workspace_guard) = workspace.device_ptr(stream);

        unsafe {
            // Create matmul descriptor (compute_type, scale_type for alpha/beta)
            cublasLtMatmulDescCreate(&mut matmul_desc, compute_type, scale_dtype).result()?;

            // Set transpose attributes
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                &a_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;
            cublasLtMatmulDescSetAttribute(
                matmul_desc,
                cudarc::cublaslt::sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                &b_layout as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<cublasOperation_t>(),
            )
            .result()?;

            // Create matrix layout descriptors
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

            cublasLtMatrixLayoutCreate(&mut a_desc, cuda_dtype, a_rows, a_cols, lda).result()?;
            cublasLtMatrixLayoutCreate(&mut b_desc, cuda_dtype, b_rows, b_cols, ldb).result()?;
            cublasLtMatrixLayoutCreate(&mut c_desc, cuda_dtype, m, n, ldc).result()?;

            // Set batched GEMM attributes if batch_count > 1
            if batch_count > 1 {
                for (desc, stride) in [(a_desc, stride_a), (b_desc, stride_b), (c_desc, stride_c)] {
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
                        &stride as *const _ as *const std::ffi::c_void,
                        std::mem::size_of::<i64>(),
                    )
                    .result()?;
                }
            }

            // Create preference and set workspace size
            cublasLtMatmulPreferenceCreate(&mut preference).result()?;
            cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            )
            .result()?;

            // Get heuristic (best algorithm)
            cublasLtMatmulAlgoGetHeuristic(
                *cublaslt.handle(),
                matmul_desc,
                a_desc,
                b_desc,
                c_desc,
                c_desc, // D layout same as C
                preference,
                1, // Request 1 result
                &mut heuristic,
                &mut algo_count,
            )
            .result()?;

            if algo_count == 0 {
                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(c_desc);
                cublasLtMatrixLayoutDestroy(b_desc);
                cublasLtMatrixLayoutDestroy(a_desc);
                cublasLtMatmulDescDestroy(matmul_desc);
                return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
            }

            let alpha_ptr = &alpha_f32 as *const _ as *const std::ffi::c_void;
            let beta_ptr = &beta_f32 as *const _ as *const std::ffi::c_void;
            cublasLtMatmul(
                *cublaslt.handle(),
                matmul_desc,
                alpha_ptr,
                a_ptr as *const std::ffi::c_void,
                a_desc,
                b_ptr as *const std::ffi::c_void,
                b_desc,
                beta_ptr,
                c_ptr as *const std::ffi::c_void,
                c_desc,
                c_ptr as *mut std::ffi::c_void,
                c_desc,
                &heuristic.algo,
                workspace_ptr as *mut std::ffi::c_void,
                WORKSPACE_SIZE,
                stream.cu_stream() as *mut _,
            )
            .result()?;

            // Cleanup
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
        }

        // No stream.synchronize() here — CUDA stream ordering guarantees
        // sequential execution. The runtime syncs once at the end of execute().
        Ok(())
    }

    fn output_size(&self) -> Expression {
        let resolve = |e: &Expression| -> Expression { e.substitute('z', Expression::from(1)) };
        resolve(&self.batch_count) * resolve(&self.m) * resolve(&self.n)
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
}
