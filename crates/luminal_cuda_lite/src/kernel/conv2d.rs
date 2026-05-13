//! Direct conv2d_bias kernel — fuses unfold + matmul + bias into one
//! CUDA kernel with no `(H_out*W_out, C_in*K*K)` intermediate matrix.
//!
//! This is exposed as a luminal `CustomOp`, not a standard egglog-rewritten
//! `KernelOp`, because the conv has no useful fusion opportunities with
//! surrounding ops in the graphs it's used in (the VAE's resnet blocks),
//! and pattern-matching the unfold+permute+merge_dims+matmul+bias chain
//! reliably from egglog is significantly more work than just bypassing
//! the egglog rewrite path entirely.
//!
//! The kernel is one-thread-per-output: each thread computes
//!   `out[co, ho, wo] = bias[co] + sum_{ci,ki,kj} input[ci, ho*S+ki-P, wo*S+kj-P] * weight[co, ci, ki, kj]`
//! with bounds checks on the spatial dims for padding. This is far from
//! peak FLOPs (no shared-memory tiling, no warp-level reduction over K)
//! but it's correct and the memory footprint is just the input + weight +
//! bias + output buffers — no `(M, K)` or `(M, N, K)` intermediate, so it
//! scales linearly with the actual conv FLOPs rather than blowing up at
//! large H/W like the unfold-based formulation.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::FxHashMap;
use luminal::{
    dtype::DType, graph::Graph, op::CustomOp, op::LLIROp, prelude::GraphTensor, shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

/// Direct conv2d-with-bias kernel. All shape/kernel params are static
/// (baked into the CUDA source via #defines), so each conv shape gets
/// its own compiled kernel. Inputs (in order): input `(C_in, H_in, W_in)`,
/// weight `(C_out, C_in*K*K)` (i.e. flattened `(C_out, C_in, K, K)`), bias
/// `(C_out,)`. Output: `(C_out, H_out, W_out)`.
#[derive(Debug, Clone)]
pub struct Conv2DKernel {
    pub c_in: usize,
    pub h_in: usize,
    pub w_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub h_out: usize,
    pub w_out: usize,
}

impl Conv2DKernel {
    fn output_elements(&self) -> usize {
        self.c_out * self.h_out * self.w_out
    }
}

const THREADS_PER_BLOCK: usize = 256;

impl KernelOp for Conv2DKernel {
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
        let total = self.output_elements();
        let grid = total.div_ceil(THREADS_PER_BLOCK);

        let kernel = format!(
            "
extern \"C\" __global__ void conv2d_bias_kernel(
    float* __restrict__ out,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias
) {{
    const int TOTAL = {total};
    const int CIN  = {c_in};
    const int H    = {h_in};
    const int W    = {w_in};
    const int HOUT = {h_out};
    const int WOUT = {w_out};
    const int K    = {k};
    const int S    = {s};
    const int P    = {p};

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL) return;
    int hw = HOUT * WOUT;
    int co = idx / hw;
    int rem = idx - co * hw;
    int ho = rem / WOUT;
    int wo = rem - ho * WOUT;

    float acc = bias[co];
    int weight_co_base = co * (CIN * K * K);
    for (int ci = 0; ci < CIN; ci++) {{
        int input_ci_base = ci * (H * W);
        int weight_ci_base = weight_co_base + ci * (K * K);
        #pragma unroll
        for (int ki = 0; ki < K; ki++) {{
            int hi = ho * S + ki - P;
            if (hi < 0 || hi >= H) continue;
            int input_row_base = input_ci_base + hi * W;
            int weight_row_base = weight_ci_base + ki * K;
            #pragma unroll
            for (int kj = 0; kj < K; kj++) {{
                int wj = wo * S + kj - P;
                if (wj < 0 || wj >= W) continue;
                acc += input[input_row_base + wj] * weight[weight_row_base + kj];
            }}
        }}
    }}
    out[idx] = acc;
}}
",
            total = total,
            c_in = self.c_in,
            h_in = self.h_in,
            w_in = self.w_in,
            h_out = self.h_out,
            w_out = self.w_out,
            k = self.kernel,
            s = self.stride,
            p = self.padding,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("conv2d_bias_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                Expression::from(grid),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(THREADS_PER_BLOCK),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        Expression::from(self.output_elements())
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // Per output: C_in * K * K input loads + same many weight loads + 1 bias load.
        let per_out = self.c_in * self.kernel * self.kernel * 2 + 1;
        Expression::from(self.output_elements() * per_out * 4)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // 2 * C_in * K * K mul-adds per output, plus the bias add = +1.
        let per_out = self.c_in * self.kernel * self.kernel * 2 + 1;
        Expression::from(self.output_elements() * per_out)
    }

    fn kernel_name(&self) -> &'static str {
        "Conv2DBias"
    }
}

/// luminal `CustomOp` that wraps `Conv2DKernel`. Lets us drop the kernel
/// straight into an HLIR graph via `cx.custom_op(...)` without going
/// through egglog rewrites.
#[derive(Debug, Clone)]
pub struct Conv2DCustom(pub Conv2DKernel);

impl CustomOp for Conv2DCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// 2D conv-with-bias on a `(C_in, H, W)` F32 input tensor, with weights
/// stored as `(C_out, C_in*K*K)` and bias as `(C_out,)`. Stride/padding/kernel
/// are static. Output: `(C_out, H_out, W_out)`.
///
/// This is a thin wrapper over [`Conv2DKernel`] that hides the
/// `cx.custom_op` plumbing. All inputs MUST be `DType::F32` and contiguous
/// row-major; pass `tensor * 1.0_f32` first if you have a strided view.
pub fn conv2d_bias(
    input: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> GraphTensor {
    assert_eq!(input.dtype, DType::F32, "conv2d_bias requires F32 input");
    assert_eq!(weight.dtype, DType::F32, "conv2d_bias requires F32 weight");
    assert_eq!(bias.dtype, DType::F32, "conv2d_bias requires F32 bias");

    let dims = input.dims();
    assert_eq!(dims.len(), 3, "conv2d_bias expects (C_in, H, W) input");
    let c_in = dims[0].to_usize().expect("C_in must be a static dim");
    let h_in = dims[1].to_usize().expect("H must be a static dim");
    let w_in = dims[2].to_usize().expect("W must be a static dim");

    let w_dims = weight.dims();
    assert_eq!(
        w_dims.len(),
        2,
        "conv2d_bias expects weight (C_out, C_in*K*K)"
    );
    let c_out = w_dims[0].to_usize().expect("C_out must be a static dim");
    let w_kk = w_dims[1]
        .to_usize()
        .expect("weight inner dim must be static");
    assert_eq!(
        w_kk,
        c_in * kernel * kernel,
        "weight inner dim {w_kk} != C_in*K*K = {}",
        c_in * kernel * kernel,
    );

    let b_dims = bias.dims();
    assert_eq!(b_dims.len(), 1, "conv2d_bias expects bias (C_out,)");
    assert_eq!(
        b_dims[0].to_usize().expect("bias dim must be static"),
        c_out
    );

    assert!(
        h_in + 2 * padding >= kernel,
        "padded H_in ({}) is smaller than kernel ({})",
        h_in + 2 * padding,
        kernel,
    );
    assert!(
        w_in + 2 * padding >= kernel,
        "padded W_in ({}) is smaller than kernel ({})",
        w_in + 2 * padding,
        kernel,
    );
    let h_out = (h_in + 2 * padding - kernel) / stride + 1;
    let w_out = (w_in + 2 * padding - kernel) / stride + 1;

    let kern = Conv2DKernel {
        c_in,
        h_in,
        w_in,
        c_out,
        kernel,
        stride,
        padding,
        h_out,
        w_out,
    };
    let cx: &mut Graph = unsafe { &mut *input.graph_ref };
    cx.custom_op(
        Conv2DCustom(kern),
        vec![input, weight, bias],
        (c_out, h_out, w_out),
        DType::F32,
    )
}
