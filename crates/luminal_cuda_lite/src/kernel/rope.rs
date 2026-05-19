//! Fused RoPE (rotary position embedding) — interleaved-pair convention.
//!
//! Replaces flux2's 6-op RoPE chain (split / slice / squeeze / neg / concat /
//! merge_dims / 4× cast / mul / add) with a single kernel launch per call.
//! ~120 RoPE calls per forward pass at full DiT depth.
//!
//! Convention: `repeat_interleave_real=True` (Flux 2 / diffusers), so adjacent
//! dim pairs rotate together. For an input `[a0, b0, a1, b1, ...]` and per-
//! position `(cos, sin)`, the output is
//!   `out[2j]   = x[2j]   * cos[2j]   - x[2j+1] * sin[2j]`
//!   `out[2j+1] = x[2j+1] * cos[2j+1] + x[2j]   * sin[2j+1]`
//!
//! Layout: x `(S, H, D)`, cos/sin `(S, D)` (broadcast across H).

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

#[derive(Debug, Clone)]
pub struct RoPEKernel {
    pub s: usize,
    pub h: usize,
    pub d: usize,
}

const TPB: usize = 64;

impl KernelOp for RoPEKernel {
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
        let s = self.s;
        let h = self.h;
        let d = self.d;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let kernel = format!(
            r#"
extern "C" __global__ void rope_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_
) {{
    const int S = {s};
    const int H = {h};
    const int D = {d};
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int tid = threadIdx.x;

    const float* xr   = x    + sh    * D;
    const float* cosr = cos_ + s_idx * D;
    const float* sinr = sin_ + s_idx * D;
    float* yr = out + sh * D;

    for (int i = tid; i < D; i += {TPB}) {{
        float xi = xr[i];
        float xpair;
        if ((i & 1) == 0) {{
            // even: paired with i+1, rotated value is -x[i+1]
            xpair = -xr[i + 1];
        }} else {{
            // odd: paired with i-1, rotated value is +x[i-1]
            xpair = xr[i - 1];
        }}
        yr[i] = xi * cosr[i] + xpair * sinr[i];
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rope_kernel".to_string(),
            (
                Expression::from(s * h),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        Expression::from(self.s * self.h * self.d)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // x: full (S,H,D); cos/sin: (S,D) read H times each but cached.
        Expression::from(self.s * self.h * self.d * 4 + self.s * self.d * 4 * 2)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // 4 per output element (mul, neg/load, mul, add).
        Expression::from(self.s * self.h * self.d * 4)
    }

    fn kernel_name(&self) -> &'static str {
        "RoPE"
    }
}

#[derive(Debug, Clone)]
pub struct RoPECustom(pub RoPEKernel);

impl CustomOp for RoPECustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Apply RoPE: `x` shape `(S, H, D)` F32, `cos`/`sin` shape `(S, D)` F32.
/// Returns `(S, H, D)` F32.
pub fn apply_rope(x: GraphTensor, cos: GraphTensor, sin: GraphTensor) -> GraphTensor {
    assert_eq!(x.dtype, DType::F32, "RoPE x must be F32");
    let cos = if cos.dtype == DType::F32 {
        cos
    } else {
        cos.cast(DType::F32)
    };
    let sin = if sin.dtype == DType::F32 {
        sin
    } else {
        sin.cast(DType::F32)
    };
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 3, "RoPE x must be 3-D (S, H, D)");
    let s = x_dims[0].to_usize().expect("RoPE: S must be static");
    let h = x_dims[1].to_usize().expect("RoPE: H must be static");
    let d = x_dims[2].to_usize().expect("RoPE: D must be static");
    let cos_dims = cos.dims();
    let sin_dims = sin.dims();
    assert_eq!(cos_dims.len(), 2, "RoPE cos must be 2-D (S, D)");
    assert_eq!(sin_dims.len(), 2, "RoPE sin must be 2-D (S, D)");
    assert_eq!(cos_dims[0].to_usize().unwrap(), s, "RoPE cos S mismatch");
    assert_eq!(cos_dims[1].to_usize().unwrap(), d, "RoPE cos D mismatch");
    assert_eq!(sin_dims[0].to_usize().unwrap(), s, "RoPE sin S mismatch");
    assert_eq!(sin_dims[1].to_usize().unwrap(), d, "RoPE sin D mismatch");

    let kern = RoPEKernel { s, h, d };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RoPECustom(kern), vec![x, cos, sin], (s, h, d), DType::F32)
}
