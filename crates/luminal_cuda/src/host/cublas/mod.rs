use cudarc::cublas::{
    CudaBlas,
    sys::{cublasOperation_t, cublasSetStream_v2, cublasSgemm_v2, cublasStatus_t},
};
use cudarc::driver::{CudaStream, DevicePtr};
use luminal::{
    egglog_utils::extract_expr,
    op::{
        EgglogOp, LLIROp,
        OpParam::{self, *},
    },
    prelude::*,
};
use std::sync::{Arc, OnceLock};
use tracing::{Level, span, trace};

use crate::{cudarc::driver::CudaSlice, host::HostOp};

/// Global shared cuBLAS handle to avoid per-operation workspace allocation
static SHARED_CUBLAS: OnceLock<Arc<CudaBlas>> = OnceLock::new();

/// Parse cuBLAS operation from egglog string (e.g., "\"T\"" -> CUBLAS_OP_T)
fn parse_cublas_op(s: &str) -> cublasOperation_t {
    // Strip quotes if present (egglog strings are stored with quotes)
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
pub struct CuBlasSgemmV2 {
    m: Expression,
    n: Expression,
    k: Expression,
    a_layout: cublasOperation_t,
    b_layout: cublasOperation_t,
    lda: Expression,
    ldb: Expression,
    ldc: Expression,
    /// Lazily initialized cuBLAS handle - created on first execute
    cublas: OnceLock<Arc<CudaBlas>>,
}

// Useless default for IntoEgglogOp
impl Default for CuBlasSgemmV2 {
    fn default() -> Self {
        Self {
            m: Expression::default(),
            n: Expression::default(),
            k: Expression::default(),
            a_layout: cublasOperation_t::CUBLAS_OP_N, // IGNORE NOT REAL
            b_layout: cublasOperation_t::CUBLAS_OP_T, // IGNORE NOT REAL
            lda: Expression::default(),
            ldb: Expression::default(),
            ldc: Expression::default(),
            cublas: OnceLock::new(),
        }
    }
}

impl EgglogOp for CuBlasSgemmV2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "cublasSgemmV2".to_string(),
            //    A      B      m     n      k  , A input Layout, B input Layout,
            vec![Input, Input, Expr, Expr, Expr, Str, Str, Expr, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            include_str!["sgemm_v2_RmRm_rewrite.egg"].to_string(), // row row
            include_str!["sgemm_v2_RmCm_rewrite.egg"].to_string(), // row col
            include_str!["sgemm_v2_CmRm_rewrite.egg"].to_string(), // col row
            include_str!["sgemm_v2_CmCm_rewrite.egg"].to_string(), // col col
        ]
    }

    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        // Extract dimensions from egglog
        let m = extract_expr(egraph, children[2], expr_cache).unwrap();
        let n = extract_expr(egraph, children[3], expr_cache).unwrap();
        let k = extract_expr(egraph, children[4], expr_cache).unwrap();

        // Extract layout strings from egglog
        let a_layout_str = &egraph.enodes[children[5]].0;
        let b_layout_str = &egraph.enodes[children[6]].0;
        let a_layout = parse_cublas_op(a_layout_str);
        let b_layout = parse_cublas_op(b_layout_str);

        // Extract leading dimensions from egglog
        let lda = extract_expr(egraph, children[7], expr_cache).unwrap();
        let ldb = extract_expr(egraph, children[8], expr_cache).unwrap();
        let ldc = extract_expr(egraph, children[9], expr_cache).unwrap();

        let extracted_state = Self {
            m,
            n,
            k,
            a_layout,
            b_layout,
            lda,
            ldb,
            ldc,
            cublas: OnceLock::new(),
        };
        trace!(?extracted_state);

        let extracted = LLIROp::new::<dyn HostOp>(Box::new(extracted_state) as Box<dyn HostOp>);

        (extracted, vec![children[0], children[1]])
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for CuBlasSgemmV2 {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[&CudaSlice<u8>],
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        // GEMM parameters
        let m = self.m.exec(dyn_map).unwrap() as i32;
        let n = self.n.exec(dyn_map).unwrap() as i32;
        let k = self.k.exec(dyn_map).unwrap() as i32;
        let a_layout = self.a_layout;
        let b_layout = self.b_layout;
        let lda = self.lda.exec(dyn_map).unwrap() as i32;
        let ldb = self.ldb.exec(dyn_map).unwrap() as i32;
        let ldc = self.ldc.exec(dyn_map).unwrap() as i32;

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // Get device pointers
        let (a_ptr, _a_guard) = inputs[1].device_ptr(stream);
        let (b_ptr, _b_guard) = inputs[2].device_ptr(stream);
        let (c_ptr, _c_guard) = inputs[0].device_ptr(stream);

        // Debug: Check buffer sizes
        trace!(
            "buffer_validation {}=={},{}=={},{}=={}",
            inputs[1].len(),
            m * k * 4,
            inputs[2].len(),
            k * n * 4,
            inputs[0].len(),
            m * n * 4
        );
        let _sgemm_span = span!(
            Level::TRACE,
            "cuBLAS_SGEMM_V2",
            m,
            n,
            k,
            alpha,
            beta,
            lda,
            ldb,
            ldc,
            ?a_layout,
            ?b_layout,
        )
        .entered();

        // Use shared cuBLAS handle to avoid per-operation workspace allocation
        let cublas = SHARED_CUBLAS.get_or_init(|| Arc::new(CudaBlas::new(stream.clone()).unwrap()));

        // Set the stream for this operation (cuBLAS handle can work with any stream)
        // The CUstream types from cublas::sys and driver::sys are compatible, just cast
        unsafe {
            cublasSetStream_v2(*cublas.handle(), stream.cu_stream() as _);
        }

        let status = unsafe {
            cublasSgemm_v2(
                *cublas.handle(),
                a_layout,
                b_layout,
                m,
                n,
                k,
                &alpha as *const f32,
                a_ptr as *const f32,
                lda,
                b_ptr as *const f32,
                ldb,
                &beta as *const f32,
                c_ptr as *mut f32,
                ldc,
            )
        };
        stream.synchronize().unwrap();

        if status != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(anyhow::anyhow!(
                "cuBLAS SGEMM TN failed with status: {:?}",
                status
            ));
        }

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }
}
