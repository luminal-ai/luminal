//! Direct 2D matmul kernel — bypasses egglog rewrites, used as a custom op
//! for matmul shapes where the cublaslt egg rules don't reliably fire.
//!
//! The cublaslt 2D rules in `host/cublaslt/cublaslt_*Cm_rewrite.egg` /
//! `cublaslt_Rm*_rewrite.egg` are *supposed* to match any 2D matmul whose
//! Mul + SumReduce broadcast lowering has the expected stride patterns,
//! and the conditional `KernelMul` cleanup is *supposed* to delete the
//! KernelMul + KernelSumReduce fallback whenever a cublaslt alternative
//! exists. In practice both fail to fire reliably for the VAE's mid-block
//! `AttnBlock` matmuls — at 1024² that lets the search occasionally pick
//! the broadcast-Mul path for `q @ kᵀ`, generating a `(HW, HW, C) =
//! (16384, 16384, 512)` ≈ 524 GiB single intermediate that OOMs the GPU.
//!
//! Same approach as `kernel::conv2d`: define a `KernelOp`, wrap it in a
//! `CustomOp`, expose a tiny `pub fn` so callers don't see the
//! `cx.custom_op` plumbing. This is opaque to egglog by design — we
//! aren't trying to fuse with surrounding ops, just guarantee a sane
//! lowering for the matmuls we know are problematic.
//!
//! The CUDA implementation is a textbook 2D-blocked SGEMM:
//!   * 16×16 output tile per block (256 threads)
//!   * Tiled load of A and B into shared memory in K-size chunks
//!   * Each thread accumulates one output element across all K-tiles
//!   * Optional bias broadcast along the M axis at write-out
//!   * `transpose_b` toggles between row-major B `(K, N)` and row-major
//!     B `(N, K)` (i.e. the `A @ Bᵀ` pattern that linear/projection
//!     layers use).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

/// Direct F32 2D matmul `(M, K) × {(K, N) | (N, K)} → (M, N)` with optional
/// per-output-column bias. All shape parameters are static (baked into the
/// CUDA source via #defines), so each (M, N, K, transpose_b, has_bias)
/// shape gets its own compiled kernel.
#[derive(Debug, Clone)]
pub struct Matmul2DKernel {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// If `true`, B is interpreted as `(N, K)` row-major and accessed as
    /// `B[n][k]` (i.e. `A @ Bᵀ`). If `false`, B is `(K, N)` row-major and
    /// accessed as `B[k][n]` (i.e. `A @ B`).
    pub transpose_b: bool,
    pub has_bias: bool,
}

const TILE: usize = 16;

impl KernelOp for Matmul2DKernel {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let bias_param = if self.has_bias {
            ", const float* __restrict__ bias"
        } else {
            ""
        };
        let bias_add = if self.has_bias {
            "    acc += bias[n];\n"
        } else {
            ""
        };
        // We want Bs[ty][tx] = B_effective[k0+ty][b_n_base+tx] where:
        //   transpose_b=false: B is (K, N) row-major → B[(k0+ty)*N + (b_n_base+tx)]
        //   transpose_b=true:  B is (N, K) row-major → B[(b_n_base+tx)*K + (k0+ty)]
        let b_index = if self.transpose_b {
            "B[(b_n_base + tx) * K + (k0 + ty)]"
        } else {
            "B[(k0 + ty) * N + (b_n_base + tx)]"
        };

        let kernel = format!(
            "
extern \"C\" __global__ void matmul_2d_kernel(
    float* __restrict__ C,
    const float* __restrict__ A,
    const float* __restrict__ B{bias_param}
) {{
    const int M = {m};
    const int N = {n};
    const int K = {k};
    const int TILE = {tile};

    __shared__ float As[{tile}][{tile}];
    __shared__ float Bs[{tile}][{tile}];

    int bx = blockIdx.x;  // tile column (n)
    int by = blockIdx.y;  // tile row (m)
    int tx = threadIdx.x; // 0..TILE-1, output col within tile
    int ty = threadIdx.y; // 0..TILE-1, output row within tile

    int m_global = by * TILE + ty;
    int n_global = bx * TILE + tx;

    int a_m_base = by * TILE;
    int b_n_base = bx * TILE;

    float acc = 0.0f;

    int n_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < n_tiles; ++t) {{
        int k0 = t * TILE;

        // Load A tile (TILE, TILE) row-major from A[m, k]: A[(by*TILE+ty)*K + (k0+tx)]
        int a_m = a_m_base + ty;
        int a_k = k0 + tx;
        As[ty][tx] = (a_m < M && a_k < K) ? A[a_m * K + a_k] : 0.0f;

        // Load B tile depending on transpose_b
        int b_n_or_k = b_n_base + tx;  // for transpose_b=true this is N; for =false this is N
        int b_k_or_k = k0 + ty;        // similarly
        // We compute Bs[ty][tx] such that the inner loop reads Bs[k_local][n_local] = B[k][n].
        // For transpose_b=true (B is (N,K)):  B[k][n] in math = B_storage[n][k] = B[(b_n_base+tx)*K + (k0+ty)]
        // For transpose_b=false (B is (K,N)): B[k][n] in math = B_storage[k][n] = B[(k0+ty)*N + (b_n_base+tx)]
        bool b_in_bounds = ({transpose_b} ? (b_n_or_k < N && b_k_or_k < K)
                                          : (b_k_or_k < K && b_n_or_k < N));
        Bs[ty][tx] = b_in_bounds ? {b_index} : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < {tile}; ++kk) {{
            acc += As[ty][kk] * Bs[kk][tx];
        }}
        __syncthreads();
    }}

    if (m_global < M && n_global < N) {{
        int n = n_global;
{bias_add}        C[m_global * N + n_global] = acc;
    }}
}}
",
            m = self.m,
            n = self.n,
            k = self.k,
            tile = TILE,
            transpose_b = self.transpose_b,
            b_index = b_index,
            bias_param = bias_param,
            bias_add = bias_add,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("matmul_2d_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let grid_x = self.n.div_ceil(TILE);
        let grid_y = self.m.div_ceil(TILE);
        (
            func,
            module,
            kernel,
            (
                Expression::from(grid_x),
                Expression::from(grid_y),
                Expression::from(1usize),
            ),
            (
                Expression::from(TILE),
                Expression::from(TILE),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        Expression::from(self.m * self.n)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // Each output: K elements from A and K elements from B (plus 1 bias).
        let per_out = self.k * 2 + if self.has_bias { 1 } else { 0 };
        Expression::from(self.m * self.n * per_out * 4)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        let per_out = self.k * 2 + if self.has_bias { 1 } else { 0 };
        Expression::from(self.m * self.n * per_out)
    }

    fn kernel_name(&self) -> &'static str {
        "Matmul2D"
    }
}

/// CustomOp wrapper for [`Matmul2DKernel`].
#[derive(Debug, Clone)]
pub struct Matmul2DCustom(pub Matmul2DKernel);

impl CustomOp for Matmul2DCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// `(M, K) @ (K, N) -> (M, N)` for row-major F32 inputs. No bias.
pub fn matmul_2d(a: GraphTensor, b: GraphTensor) -> GraphTensor {
    matmul_2d_inner(a, b, /*transpose_b=*/ false, None)
}

/// `(M, K) @ (N, K)ᵀ -> (M, N)` for row-major F32 inputs. No bias.
/// Use this for `A @ Bᵀ` where B is stored row-major as `(N, K)` — the
/// pattern produced by linear / projection layers (`x @ w.t()`).
pub fn matmul_2d_t(a: GraphTensor, b: GraphTensor) -> GraphTensor {
    matmul_2d_inner(a, b, /*transpose_b=*/ true, None)
}

/// Linear projection with bias: `(M, K) @ (N, K)ᵀ + bias` where bias is
/// `(N,)`, row-major F32 throughout.
pub fn linear_bias(a: GraphTensor, b: GraphTensor, bias: GraphTensor) -> GraphTensor {
    matmul_2d_inner(a, b, /*transpose_b=*/ true, Some(bias))
}

fn matmul_2d_inner(
    a: GraphTensor,
    b: GraphTensor,
    transpose_b: bool,
    bias: Option<GraphTensor>,
) -> GraphTensor {
    assert_eq!(a.dtype, DType::F32, "matmul_2d requires F32 inputs");
    assert_eq!(b.dtype, DType::F32, "matmul_2d requires F32 inputs");
    let a_dims = a.dims();
    let b_dims = b.dims();
    assert_eq!(a_dims.len(), 2, "matmul_2d expects 2D A");
    assert_eq!(b_dims.len(), 2, "matmul_2d expects 2D B");
    let m = a_dims[0].to_usize().expect("M must be a static dim");
    let k_a = a_dims[1].to_usize().expect("K (A) must be a static dim");
    let (n, k_b) = if transpose_b {
        // B is (N, K)
        let n = b_dims[0].to_usize().expect("N must be a static dim");
        let k = b_dims[1].to_usize().expect("K (B) must be a static dim");
        (n, k)
    } else {
        // B is (K, N)
        let k = b_dims[0].to_usize().expect("K (B) must be a static dim");
        let n = b_dims[1].to_usize().expect("N must be a static dim");
        (n, k)
    };
    assert_eq!(k_a, k_b, "matmul_2d K mismatch: A K={k_a}, B K={k_b}");
    let k = k_a;

    let has_bias = bias.is_some();
    if let Some(bias) = bias {
        let bdims = bias.dims();
        assert_eq!(bdims.len(), 1, "matmul_2d bias must be 1D");
        assert_eq!(
            bdims[0].to_usize().expect("bias dim must be static"),
            n,
            "matmul_2d bias size must equal N"
        );
        assert_eq!(bias.dtype, DType::F32, "matmul_2d bias must be F32");
    }

    let kern = Matmul2DKernel {
        m,
        n,
        k,
        transpose_b,
        has_bias,
    };
    let cx = unsafe { &mut *a.graph_ref };
    let inputs: Vec<GraphTensor> = if let Some(bias) = bias {
        vec![a, b, bias]
    } else {
        vec![a, b]
    };
    cx.custom_op(Matmul2DCustom(kern), inputs, (m, n), DType::F32)
}
