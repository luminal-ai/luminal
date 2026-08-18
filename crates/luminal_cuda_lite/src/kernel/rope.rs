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

use std::{fmt::Display, sync::Arc};

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType,
    egglog_utils::list_to_egglog,
    op::{CustomOp, HLIROp, LLIROp},
    prelude::{DynMap, FxHashMap, FxHashSet, GraphTensor, NodeIndex, ShapeTracker, Symbol},
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
        FxHashMap<Symbol, CudaSlice<u8>>,
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
            kernel,
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

// ═══════════════════════════════════════════════════════════
// Half-rotation RoPE (Llama 3 convention), dtype-aware, dynamic S.
//
// Rotates the two halves of each head: for j in 0..D/2
//   out[j]       = x[j]       * cos[j] - x[j + D/2] * sin[j]
//   out[j + D/2] = x[j + D/2] * cos[j] + x[j]       * sin[j]
//
// The input is read from a row of a (possibly wider) projection output via
// `pitch` (row stride in elements) and `offset` (column offset), so q and k
// can be roped straight out of a fused QKV GEMM without materializing
// slices. cos/sin are (S, D/2) F32; x/out are `dtype` (F32 or 16-bit, math
// in F32). S is dynamic: the kernel derives everything from blockIdx, so
// only the grid expression carries the dyn dim.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEHalfKernel {
    pub s: Expression,
    pub h: usize,
    pub d: usize,
    /// Input row stride in elements (e.g. 6144 for a fused QKV row).
    pub pitch: usize,
    /// Column offset of this head group within the input row.
    pub offset: usize,
    pub dtype: DType,
}

impl Default for RoPEHalfKernel {
    fn default() -> Self {
        Self {
            s: 1.into(),
            h: 1,
            d: 2,
            pitch: 2,
            offset: 0,
            dtype: DType::F32,
        }
    }
}

impl Display for RoPEHalfKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RoPEHalf")
    }
}

impl HLIROp for RoPEHalfKernel {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        assert_eq!(inputs.len(), 3, "RoPEHalf has x, cos, and sin inputs");
        format!(
            "(Op (KernelRoPEHalf {} {} {} {} {} {} ({:?})) {})",
            self.s.to_egglog(),
            Expression::from(self.h).to_egglog(),
            Expression::from(self.d).to_egglog(),
            Expression::from(self.h * self.d).to_egglog(),
            Expression::from(self.pitch).to_egglog(),
            Expression::from(self.offset).to_egglog(),
            self.dtype,
            list_to_egglog(&[&inputs[0].1, &inputs[1].1, &inputs[2].1], "ICons", "INil"),
        )
    }
}

impl EgglogOp for RoPEHalfKernel {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPEHalf",
            &[
                ("s", EXPRESSION),
                ("h", EXPRESSION),
                ("d", EXPRESSION),
                ("out_width", EXPRESSION),
                ("pitch", EXPRESSION),
                ("offset", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                ((= ?rope (Op (KernelRoPEHalf ?s ?h ?d ?out_width ?pitch ?offset ?dt) ?inputs)))
                ((set (dtype ?rope) ?dt))
                :ruleset dtype_prop
                :name \"kernel-rope-half-dtype\"
            )",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let s = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let h = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf h must be static");
        let d = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf d must be static");
        let pitch = extract_expr(egraph, kind_children[4], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf pitch must be static");
        let offset = extract_expr(egraph, kind_children[5], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf offset must be static");
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                s,
                h,
                d,
                pitch,
                offset,
                dtype: extract_dtype(egraph, kind_children[6]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for RoPEHalfKernel {
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let h = self.h;
        let d = self.d;
        let pitch = self.pitch;
        let offset = self.offset;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let half = d / 2;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        let kernel = format!(
            r#"{includes}
extern "C" __global__ void rope_half_kernel(
    {ty}* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_
) {{
    const int H = {h};
    const int D = {d};
    const int HALF = {half};
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int h_idx = sh - s_idx * H;
    int tid = threadIdx.x;

    const {ty}* xr   = x    + (long long)s_idx * {pitch} + {offset} + h_idx * D;
    const float* cosr = cos_ + (long long)s_idx * HALF;
    const float* sinr = sin_ + (long long)s_idx * HALF;
    {ty}* yr = out + (long long)sh * D;

    for (int j = tid; j < HALF; j += {TPB}) {{
        float x0 = (float)xr[j];
        float x1 = (float)xr[j + HALF];
        float c = cosr[j];
        float s = sinr[j];
        yr[j] = ({ty})(x0 * c - x1 * s);
        yr[j + HALF] = ({ty})(x1 * c + x0 * s);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_half_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.s * self.h,
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
        self.s * self.h * self.d
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        (self.s * self.h * self.d * self.dtype.bits()).ceil_div(8) + self.s * self.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.s * self.h * self.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEHalf"
    }
}

#[derive(Debug, Clone)]
pub struct RoPEHalfCustom(pub RoPEHalfKernel);

impl CustomOp for RoPEHalfCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Half-rotation RoPE over a head group inside a projection output.
///
/// `x` is `(S, pitch)` (e.g. a fused QKV output), `offset`/`h`/`d` select the
/// head group, `cos`/`sin` are `(S, d/2)` F32. Returns contiguous `(S, h*d)`
/// in `x`'s dtype. One kernel replaces the ~10-op rope chain per projection.
pub fn apply_rope_half(
    x: GraphTensor,
    offset: usize,
    h: usize,
    d: usize,
    cos: GraphTensor,
    sin: GraphTensor,
) -> GraphTensor {
    assert_eq!(cos.dtype, DType::F32, "RoPE cos must be F32");
    assert_eq!(sin.dtype, DType::F32, "RoPE sin must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RoPE x must be 2-D (S, pitch)");
    let s = x_dims[0];
    let pitch = x_dims[1].to_usize().expect("RoPE: pitch must be static");
    assert!(offset + h * d <= pitch, "RoPE head group exceeds row pitch");

    let kern = RoPEHalfKernel {
        s,
        h,
        d,
        pitch,
        offset,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    let id = cx.add_op(kern, &[x.id, cos.id, sin.id]);
    GraphTensor::from_id(
        id,
        ShapeTracker::new_with_element_bits((s, h * d), x.dtype.bits()),
        cx,
        x.dtype,
    )
}

// ═══════════════════════════════════════════════════════════
// Fused RoPE + KV-cache scatter
//
// The fused kernels below are selectable egglog alternatives to materialized
// RoPE followed by in-place scatter. They write rotated values straight to the
// cache slots, removing one launch and one `(s, kv_dim)` temporary when measured
// search selects them. The materialized form remains in the same e-class.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEScatterKernel {
    pub rope: RoPEHalfKernel,
    /// Total element count of the scatter destination (the cache pool).
    dest_size: Expression,
    /// Flattened scatter-index expression over `z`, where `z` is the element
    /// position in the rope output's contiguous (s, h·d) layout.
    idx_flat: Expression,
}

impl Default for RoPEScatterKernel {
    fn default() -> Self {
        Self {
            rope: RoPEHalfKernel::default(),
            dest_size: 1.into(),
            idx_flat: Expression::from('z'),
        }
    }
}

impl EgglogOp for RoPEScatterKernel {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPEHalfScatter",
            &[
                ("s", EXPRESSION),
                ("h", EXPRESSION),
                ("d", EXPRESSION),
                ("out_width", EXPRESSION),
                ("pitch", EXPRESSION),
                ("offset", EXPRESSION),
                ("dest_shape", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        5
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    ; The source grid is exactly RoPEHalf's contiguous
                    ; `(s, out_width)` output. `out_width` is emitted by the
                    ; RoPEHalf HLIROp as the derived `h*d` semantic field.
                    (= ?index_shape (ECons ?s (ECons ?out_width (ENil))))
                    (= ?src_strides
                        (ECons (MMul (MIter) ?out_width)
                            (ECons (MIter) (ENil))))
                    (= ?dest_strides ?out_strides)
                    (= ?rope (Op
                        (KernelRoPEHalf ?s ?h ?d ?out_width ?pitch ?offset ?dt)
                        (ICons ?x (ICons ?cos (ICons ?sin (INil))))))
                    (= ?dt (dtype ?x))
                    (= ?dt (dtype ?dest))
                    (= (Int) (dtype ?indexes))
                    (= (F32) (dtype ?cos))
                    (= (F32) (dtype ?sin))
                    (= ?scatter (Op
                        (KernelScatterNoCopy ?dest_shape ?dest_strides
                            ?index_shape ?index_strides ?src_strides ?out_strides ?dt)
                        (ICons ?dest (ICons ?indexes (ICons ?rope (INil))))))
                )
                (
                    (let ?fused (Op
                        (KernelRoPEHalfScatter ?s ?h ?d ?out_width ?pitch ?offset
                            ?dest_shape ?index_shape ?index_strides ?dt)
                        (ICons ?dest (ICons ?indexes
                            (ICons ?x (ICons ?cos (ICons ?sin (INil))))))))
                    (union ?scatter ?fused)
                    (set (dtype ?fused) ?dt)
                )
                :ruleset kernel_fuse_late2_rope
                :name \"kernel rope-half scatter exact-layout\"
            )",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let s = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let h = extract_expr(egraph, kind_children[1], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf h must be static");
        let d = extract_expr(egraph, kind_children[2], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf d must be static");
        let pitch = extract_expr(egraph, kind_children[4], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf pitch must be static");
        let offset = extract_expr(egraph, kind_children[5], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPEHalf offset must be static");
        let dest_shape =
            extract_expr_list(egraph, kind_children[6], list_cache, expr_cache).unwrap();
        let index_shape =
            extract_expr_list(egraph, kind_children[7], list_cache, expr_cache).unwrap();
        let index_strides =
            extract_expr_list(egraph, kind_children[8], list_cache, expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                rope: RoPEHalfKernel {
                    s,
                    h,
                    d,
                    pitch,
                    offset,
                    dtype: extract_dtype(egraph, kind_children[9]),
                },
                dest_size: dest_shape.into_iter().product(),
                idx_flat: luminal::shape::flatten_strides(&index_shape, &index_strides),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for RoPEScatterKernel {
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let h = self.rope.h;
        let d = self.rope.d;
        let pitch = self.rope.pitch;
        let offset = self.rope.offset;
        let half = d / 2;
        let ty = crate::cuda_dtype(self.rope.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.rope.dtype]);

        let vars: FxHashSet<Symbol> = self
            .idx_flat
            .dyn_vars()
            .into_iter()
            .chain(self.dest_size.dyn_vars())
            .collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let idx_expr = self.idx_flat.to_kernel();
        let dest_n = self.dest_size.to_kernel();

        let kernel = format!(
            r#"{includes}
{dyn_defines}
extern "C" __global__ void rope_scatter_kernel(
    {ty}* __restrict__ dest,
    const int* __restrict__ indexes,
    const {ty}* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_{dyn_dims_param}
) {{
    const int H = {h};
    const int D = {d};
    const int HALF = {half};
    const long long KVD = (long long)H * D;
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int h_idx = sh - s_idx * H;
    int tid = threadIdx.x;

    const {ty}* xr   = x    + (long long)s_idx * {pitch} + {offset} + h_idx * D;
    const float* cosr = cos_ + (long long)s_idx * HALF;
    const float* sinr = sin_ + (long long)s_idx * HALF;

    for (int j = tid; j < HALF; j += {TPB}) {{
        float x0 = (float)xr[j];
        float x1 = (float)xr[j + HALF];
        float c = cosr[j];
        float s = sinr[j];
        {{
            long long const_z = (long long)s_idx * KVD + (long long)h_idx * D + j;
            int idx = indexes[{idx_expr}];
            if (idx >= 0 && idx < ({dest_n})) {{
                dest[idx] = ({ty})(x0 * c - x1 * s);
            }}
        }}
        {{
            long long const_z = (long long)s_idx * KVD + (long long)h_idx * D + j + HALF;
            int idx = indexes[{idx_expr}];
            if (idx >= 0 && idx < ({dest_n})) {{
                dest[idx] = ({ty})(x1 * c + x0 * s);
            }}
        }}
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_scatter_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.rope.s * self.rope.h,
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

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[cudarc::driver::CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // rope_scatter_kernel: (dest, indexes, x, cos, sin [, dyn_dims]).
        // Writes in place through dest (input 0), not through output_ptr.
        let mut params = vec![
            input_ptrs[0],
            input_ptrs[1],
            input_ptrs[2],
            input_ptrs[3],
            input_ptrs[4],
        ];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn mutates_aliased_input(&self) -> bool {
        true
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.idx_flat.collect_dyn_vars_into(vars);
        self.dest_size.collect_dyn_vars_into(vars);
        self.rope.s.collect_dyn_vars_into(vars);
    }

    fn output_size(&self) -> Expression {
        self.dest_size
    }

    fn output_bytes(&self) -> Expression {
        (self.dest_size * self.rope.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.rope.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        let rotated = self.rope.s * self.rope.h * self.rope.d;
        (rotated * self.rope.dtype.bits()).ceil_div(8) + rotated * 4 + self.rope.s * self.rope.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        (self.rope.s * self.rope.h * self.rope.d * self.rope.dtype.bits()).ceil_div(8)
    }

    fn flops(&self) -> Expression {
        self.rope.s * self.rope.h * self.rope.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEScatter"
    }
}

// ═══════════════════════════════════════════════════════════
// KernelRoPE — egglog-matched fused rotary (half convention, bf16).
//
// Matches the full HLIR rotary chain the qwen/gemma models spell:
// inv-freq (iota·2 → cast → ×(1/hd) → ×ln(theta) → ×log2(e) → exp2 → recip),
// angles (cast(pos) × inv_freq → sum), sin / cos-as-sin(π/2−x), bf16 casts,
// the x0 strided view + x1 offset-slice gather, the rotation arithmetic, and
// the concat (2 clamped gathers + 2 mask iotas). The rule roots at the
// concat Add eclass — the last materialized tensor of the chain; the
// trailing transpose+merge is a view applied by consumers, so the fused
// kernel writes the same (heads, seq, hd) buffer the concat produces and
// every consumer works unchanged. One kernel replaces ~13 launches per rope
// call; the angle chain (shared by the q and k calls within a layer)
// becomes dead.
//
// The kernel mirrors the decomposed numerics exactly: angle math in F32 with
// the same op order/spellings (exp2f, 1.0f/x, sinf, sin(−x+π/2)), and a bf16
// rounding at every decomposed op boundary (cos/sin casts, each mul, the −1
// mul). The concat's ×{0,1} masks and +0 are value-preserving and elided.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRoPE {
    /// `(heads, seq, head_dim)` — seq may be dynamic.
    out_shape: Vec<Expression>,
    /// Row stride of the x input in elements (the projection width).
    width: usize,
    ln_theta: f64,
    inv_hd: f64,
}

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::EgglogOp,
    prelude::{ENodeId, SerializedEGraph},
};

impl EgglogOp for KernelRoPE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPE",
            &[
                ("out_shape", ELIST),
                ("out_width", EXPRESSION),
                ("head_stride", EXPRESSION),
                ("width", EXPRESSION),
                ("ln_theta", F64),
                ("inv_hd", F64),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Two-stage match via an intermediate relation. A single ~45-atom
        // join blew up egglog's query planner on real graphs (4m56s on the
        // 279-node mini layer); splitting at the angle chain keeps each join
        // anchored: stage 1 is pinned by the Constant atoms (log2e, pi/2,
        // ln theta chain) and emits one `rope_angles` fact per rope site,
        // stage 2 joins the rotation/concat with ?cosb/?sinb already bound,
        // which makes every Mul atom selective.
        let angle_stage: &str = "
            (relation rope_invf (IR f64 f64))
            (relation rope_angles (IR IR IR f64 f64))
            (relation rope_rotated (IR IR IR IR Expression Expression Expression Expression f64 f64))
            (relation rope_safe_pad_left (IR IR EList Expression Expression Expression))
            (relation rope_safe_pad_right (IR IR EList Expression Expression Expression))
            (relation rope_row_dims_candidate (Expression Expression Expression Expression Expression))
            (relation rope_row_dims (Expression Expression Expression Expression Expression))
            (relation rope_tensor_range_candidate (Expression Expression Expression))
            (relation rope_tensor_range (Expression Expression Expression))
            ; Seed dimension proofs only from the distinctive offset-half
            ; gather used by RoPE.  Unanchored numeric-product rules form a
            ; combinatorial join over every dimension in large models.
            (rule
                (
                    (= ?x1idx (Op (Iota
                        (MAdd (MAdd (MAdd (MMod (MIter) ?hd2) ?hd2)
                                    (MMul (MMod (MDiv (MIter) ?hd2) ?seq) ?hd))
                              (MMul (MDiv (MIter) ?ch) ?hs))
                        ?range) (INil)))
                )
                ((rope_row_dims_candidate ?hd ?hd2 ?seq ?hs ?ch))
                :ruleset kernel_fuse_late
                :name \"kernel rope row dimension candidate\"
            )
            ; Static dimensions are folded before specialization, so prove
            ; their products in egglog's i64 domain.
            (rule
                (
                    (rope_row_dims_candidate ?hd ?hd2 ?seq ?hs ?ch)
                    (= ?hd (MNum ?hd_n))
                    (= ?hd2 (MNum ?hd2_n))
                    (= ?seq (MNum ?seq_n))
                    (= ?hs (MNum ?hs_n))
                    (= ?ch (MNum ?ch_n))
                    (= ?hd_n (* ?hd2_n 2))
                    (= ?hs_n (* ?hd_n ?seq_n))
                    (= ?ch_n (* ?hd2_n ?seq_n))
                )
                ((rope_row_dims ?hd ?hd2 ?seq ?hs ?ch))
                :ruleset kernel_fuse_late
                :name \"kernel rope static row dimensions\"
            )
            ; With a symbolic sequence length the product expressions remain
            ; live. Accept either canonical operand order.
            (rule
                (
                    (rope_row_dims_candidate ?hd ?hd2 ?seq ?hs ?ch)
                    (= ?hd (MNum ?hd_n))
                    (= ?hd2 (MNum ?hd2_n))
                    (= ?hd_n (* ?hd2_n 2))
                    (= ?hs (MMul ?hd ?seq))
                    (= ?ch (MMul ?hd2 ?seq))
                )
                ((rope_row_dims ?hd ?hd2 ?seq ?hs ?ch))
                :ruleset kernel_fuse_late
                :name \"kernel rope symbolic row dimensions\"
            )
            (rule
                (
                    (rope_row_dims_candidate ?hd ?hd2 ?seq ?hs ?ch)
                    (= ?hd (MNum ?hd_n))
                    (= ?hd2 (MNum ?hd2_n))
                    (= ?hd_n (* ?hd2_n 2))
                    (= ?hs (MMul ?seq ?hd))
                    (= ?ch (MMul ?seq ?hd2))
                )
                ((rope_row_dims ?hd ?hd2 ?seq ?hs ?ch))
                :ruleset kernel_fuse_late
                :name \"kernel rope symbolic row dimensions commuted\"
            )
            ; The concat gather binds the tensor shape to the same flat range.
            (rule
                (
                    (= ?cat_sh (ECons ?heads (ECons ?seq (ECons ?hd (ENil)))))
                    (= ?x0g (Op (Gather ?cat_sh ?cat_st ?x0_dsh ?x0_dstr)
                        (ICons ?c0idx (ICons ?x0out (INil)))))
                    (= ?c0idx (Op (Iota
                        (MAdd (MAdd (MMin (MMod (MIter) ?hd) ?hdm1)
                                    (MMul (MMod (MDiv (MIter) ?hd) ?seq) ?hd2))
                              (MMul (MDiv (MIter) ?hs) ?ch))
                        ?range) (INil)))
                )
                ((rope_tensor_range_candidate ?heads ?hs ?range))
                :ruleset kernel_fuse_late
                :name \"kernel rope tensor range candidate\"
            )
            (rule
                (
                    (rope_tensor_range_candidate ?heads ?hs ?range)
                    (= ?heads (MNum ?heads_n))
                    (= ?hs (MNum ?hs_n))
                    (= ?range (MNum ?range_n))
                    (= ?range_n (* ?heads_n ?hs_n))
                )
                ((rope_tensor_range ?heads ?hs ?range))
                :ruleset kernel_fuse_late
                :name \"kernel rope static tensor range\"
            )
            (rule
                (
                    (rope_tensor_range_candidate ?heads ?hs ?range)
                    (= ?range (MMul ?heads ?hs))
                )
                ((rope_tensor_range ?heads ?hs ?range))
                :ruleset kernel_fuse_late
                :name \"kernel rope symbolic tensor range\"
            )
            (rule
                (
                    (rope_tensor_range_candidate ?heads ?hs ?range)
                    (= ?range (MMul ?hs ?heads))
                )
                ((rope_tensor_range ?heads ?hs ?range))
                :ruleset kernel_fuse_late
                :name \"kernel rope symbolic tensor range commuted\"
            )
            (rule
                (
                    ; inv_freq = recip(exp2(((2i x 1/hd) x ln theta) x log2 e))
                    ; anchored at the Iota(z*2) only rope inv-freq chains have;
                    ; every atom keys off the previous one.
                    (= ?iot2 (Op (Iota (MMul (MIter) (MNum 2)) ?if_range) (INil)))
                    (= ?iotf (Op (Cast ?ic_size (F32)) (ICons ?iot2 (INil))))
                    (= ?fr1 (Op (Mul ?f1_sh ?f1_a ?f1_b ?f1_o)
                        (ICons ?iotf (ICons ?invhd_c (INil)))))
                    (= ?invhd_c (Op (Constant ?inv_hd) (INil)))
                    (= ?fr2 (Op (Mul ?f2_sh ?f2_a ?f2_b ?f2_o)
                        (ICons ?fr1 (ICons ?lnt_c (INil)))))
                    (= ?lnt_c (Op (Constant ?ln_theta) (INil)))
                    (= ?fr3 (Op (Mul ?f3_sh ?f3_a ?f3_b ?f3_o)
                        (ICons ?fr2 (ICons ?l2e (INil)))))
                    (= ?l2e (Op (Constant 1.442695) (INil)))
                    (= ?ex (Op (Exp2 ?ex_sh ?ex_in ?ex_out) (ICons ?fr3 (INil))))
                    (= ?invf (Op (Recip ?rc_sh ?rc_in ?rc_out) (ICons ?ex (INil))))
                )
                (
                    (rope_invf ?invf ?ln_theta ?inv_hd)
                )
                :ruleset kernel_fuse_late_pre_rope
                :name \"kernel rope invf stage\"
            )
            (rule
                (
                    (rope_invf ?invf ?ln_theta ?inv_hd)

                    ; emb = cast(pos) x inv_freq (1xk matmul as mul+sum),
                    ; keyed by the bound ?invf
                    (= ?embm (Op (Mul ?em_sh ?em_a ?em_b ?em_o)
                        (ICons ?posf (ICons ?invf (INil)))))
                    (= ?posf (Op (Cast ?pc_size (F32)) (ICons ?pos (INil))))
                    (= (Int) (dtype ?pos))
                    (= ?emb (Op (Sum ?es_sh ?es_dim ?es_in ?es_k ?es_out)
                        (ICons ?embm (INil))))

                    ; cos = sin(-emb + pi/2), sin = sin(emb), both cast bf16
                    (= ?sinv (Op (Sin ?s1_sh ?s1_in ?s1_out) (ICons ?emb (INil))))
                    (= ?sinb (Op (Cast ?sb_size (Bf16)) (ICons ?sinv (INil))))
                    (= ?neg (Op (Mul ?ng_sh ?ng_a ?ng_b ?ng_o)
                        (ICons ?emb2 (ICons ?m1c (INil)))))
                    (= ?emb ?emb2)
                    (= ?m1c (Op (Constant -1.000000) (INil)))
                    (= ?shift (Op (Add ?sh_sh ?sh_a ?sh_b ?sh_o)
                        (ICons ?neg (ICons ?hpi (INil)))))
                    ; pi/2: tolerance window, not exact text match
                    (= ?hpi (Op (Constant ?hpi_val) (INil)))
                    (> ?hpi_val 1.57078)
                    (< ?hpi_val 1.57081)
                    (= ?cosv (Op (Sin ?s2_sh ?s2_in ?s2_out) (ICons ?shift (INil))))
                    (= ?cosb (Op (Cast ?cb_size (Bf16)) (ICons ?cosv (INil))))
                )
                (
                    (rope_angles ?cosb ?sinb ?pos ?ln_theta ?inv_hd)
                )
                :ruleset kernel_fuse_late
                :name \"kernel rope angles stage\"
            )";
        let rotation_stage: &str = "
            (rule
                (
                    (rope_angles ?cosb ?sinb ?pos ?ln_theta ?inv_hd)

                    ; rotation: x0_out = x0*cos - x1*sin ; x1_out = x1*cos + x0*sin
                    ; (every Mul keyed by the bound ?cosb / ?sinb second input)
                    (= ?x0c (Op (Mul ?m1_sh
                        (ECons (MMul (MIter) ?e_hd) (ECons (MMul (MIter) ?e_w) (ECons (MIter) (ENil))))
                        (ECons (MNum 0) (ECons (MMul (MIter) ?e_hd2) (ECons (MIter) (ENil))))
                        ?m1_o)
                        (ICons ?x (ICons ?cosb (INil)))))
                    (= ?x0out (Op (Add ?a1_sh ?a1_a ?a1_b ?a1_o)
                        (ICons ?x0c (ICons ?x1sn (INil)))))
                    (= ?x1sn (Op (Mul ?m3_sh ?m3_a ?m3_b ?m3_o)
                        (ICons ?x1s (ICons ?negb (INil)))))
                    (= ?x1s (Op (Mul ?m2_sh ?m2_a ?m2_b ?m2_o)
                        (ICons ?x1 (ICons ?sinb (INil)))))
                    (= ?negb (Op (Cast ?nb_size (Bf16)) (ICons ?m1c (INil))))
                    (= ?m1c (Op (Constant -1.000000) (INil)))
                    (= ?x1out (Op (Add ?a2_sh ?a2_a ?a2_b ?a2_o)
                        (ICons ?x1c (ICons ?x0s (INil)))))
                    (= ?x1c (Op (Mul ?m4_sh ?m4_a ?m4_b ?m4_o)
                        (ICons ?x1 (ICons ?cosb (INil)))))
                    (= ?x0s (Op (Mul ?m5_sh
                        (ECons (MMul (MIter) ?e_hd) (ECons (MMul (MIter) ?e_w) (ECons (MIter) (ENil))))
                        ?m5_b ?m5_o)
                        (ICons ?x2 (ICons ?sinb (INil)))))
                    (= ?x ?x2)

                    ; x1 = offset-slice gather of the (heads, seq, hd) view of x
                    (= ?x1 (Op (Gather ?g1_osh ?g1_ostr ?g1_dsh
                        (ECons (MMul (MIter) ?e_hd) (ECons (MMul (MIter) ?e_w) (ECons (MIter) (ENil)))))
                        (ICons ?x1idx (ICons ?x3 (INil)))))
                    (= ?x ?x3)
                    (= ?x1idx (Op (Iota
                        (MAdd (MAdd (MAdd (MMod (MIter) ?e_hd2) ?e_hd2)
                                    (MMul (MMod (MDiv (MIter) ?e_hd2) ?e_seq) ?e_hd))
                              (MMul (MDiv (MIter) ?e_ch) ?e_hs2))
                        ?x1_range) (INil)))

                    (rope_row_dims ?e_hd ?e_hd2 ?e_seq ?e_hs2 ?e_ch)

                    (= (Bf16) (dtype ?x))
                )
                (
                    (rope_rotated ?x0out ?x1out ?x ?pos
                        ?e_hd ?e_hd2 ?e_w ?e_seq ?ln_theta ?inv_hd)
                )
                :ruleset kernel_fuse_late
                :name \"kernel rope rotation stage\"
            )";

        // Stage-2 conditions in dependency order, segmented for readability.
        let segments: Vec<&str> = vec![
            "
                    (rope_rotated ?x0out ?x1out ?x ?pos ?e_hd ?e_hd2 ?e_w ?e_seq ?ln_theta ?inv_hd)",
            "
                    ; root: the concat Add - (heads, seq, hd) contiguous
                    (= ?x0g (Op (Gather ?g2_osh ?g2_ostr ?g2_dsh ?g2_dstr)
                        (ICons ?c0idx (ICons ?x0out (INil)))))
                    (= ?x0m (Op (Mul ?m6_sh ?m6_a ?m6_b ?m6_o)
                        (ICons ?x0g (ICons ?mk0b (INil)))))
                    (= ?cat (Op (Add ?a3_sh ?a3_a ?a3_b ?a3_o)
                        (ICons ?x0m (ICons ?x1m (INil)))))
                    (= ?a3_sh (ECons ?heads (ECons ?seqd (ECons ?hdd (ENil)))))",
            "
                    ; concat half 0 pins
                    (= ?c0idx (Op (Iota
                        (MAdd (MAdd (MMin (MMod (MIter) ?e_hd) ?e_hdm1)
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (= ?mk0b (Op (Cast ?k0_size (Bf16)) (ICons ?mk0 (INil))))
                    (= ?mk0 (Op (Iota (MLt (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))",
            "
                    ; concat half 1
                    (= ?x1g (Op (Gather ?g3_osh ?g3_ostr ?g3_dsh ?g3_dstr)
                        (ICons ?c1idx (ICons ?x1out (INil)))))
                    (= ?x1m (Op (Mul ?m7_sh ?m7_a ?m7_b ?m7_o)
                        (ICons ?x1g (ICons ?mk1b (INil)))))
                    (= ?c1idx (Op (Iota
                        (MAdd (MAdd (MMax (MSub (MMod (MIter) ?e_hd) ?e_hd2) (MNum 0))
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (= ?mk1b (Op (Cast ?k1_size (Bf16)) (ICons ?mk1 (INil))))
                    (= ?mk1 (Op (Iota (MGte (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))",
            "
                    ; layout consistency
                    (= ?hdd ?e_hd)
                    (= ?seqd ?e_seq)
                    (= ?e_hd2 (MNum ?e_hd2_n))
                    (= ?e_hdm1 (MNum ?e_hdm1_n))
                    (= ?e_hdm1_n (- ?e_hd2_n 1))
                    (rope_row_dims ?e_hd ?e_hd2 ?e_seq ?e_hs ?e_ch)
                    (rope_tensor_range ?heads ?e_hs ?cat_range)",
        ];

        let concat_rule = format!(
            "(rule
                (
                    {}
                )
                (
                    (let ?kr (Op (KernelRoPE ?a3_sh (MMul ?heads ?e_hd)
                        (MMul ?e_hd ?e_seq) ?e_w ?ln_theta ?inv_hd)
                        (ICons ?x (ICons ?pos (INil)))))
                    (union ?cat ?kr)
                    (set (dtype ?kr) (Bf16))
                )
                :ruleset kernel_fuse_late2_rope
                :name \"kernel rope half bf16\"
            )",
            segments.join("\n")
        );

        // `GraphTensor::pad_with` selects between the gathered value and an
        // exact typed zero through an interleaved scatter/gather.  This avoids
        // the IEEE-invalid `0 * NaN` arithmetic mask used by the historical
        // padding spelling.  Match both safe padded halves independently so
        // the final RoPE join stays selective on large model graphs.
        //
        // The scatter/gather constraints below are the semantic proof for the
        // normalization: even slots are overwritten with zero, odd slots with
        // the gathered half, and the final gather chooses `2*z + mask(z)`.
        // Consequently the left relation contributes x0 only for i < hd/2,
        // the right relation contributes x1 only for i >= hd/2, and their Add
        // is exactly concat(x0, x1), including NaN and infinity behavior.
        let safe_concat_stage: &str = "
            (rule
                (
                    ; Contiguous logical and packed layouts for a 3-D rope row.
                    (= ?cat_sh (ECons ?heads (ECons ?e_seq (ECons ?e_hd (ENil)))))
                    (= ?cat_st (ECons (MMul (MMul (MIter) ?e_hd) ?e_seq)
                        (ECons (MMul (MIter) ?e_hd) (ECons (MIter) (ENil)))))
                    (= ?packed_sh (ECons ?heads (ECons ?e_seq
                        (ECons ?e_hd (ECons (MNum 2) (ENil))))))
                    (= ?packed_st (ECons (MMul (MMul (MMul (MIter) (MNum 2)) ?e_hd) ?e_seq)
                        (ECons (MMul (MMul (MIter) (MNum 2)) ?e_hd)
                            (ECons (MMul (MIter) (MNum 2)) (ECons (MIter) (ENil))))))
                    (= ?zero_st (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil)))))

                    ; The actual left half, clamped only in the inactive half.
                    (= ?x0g (Op (Gather ?cat_sh ?cat_st ?x0_dsh ?x0_dstr)
                        (ICons ?c0idx (ICons ?x0out (INil)))))
                    (= ?c0idx (Op (Iota
                        (MAdd (MAdd (MMin (MMod (MIter) ?e_hd) ?e_hdm1)
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (= ?e_hd2 (MNum ?e_hd2_n))
                    (= ?e_hdm1 (MNum ?e_hdm1_n))
                    (= ?e_hdm1_n (- ?e_hd2_n 1))
                    (rope_row_dims ?e_hd ?e_hd2 ?e_seq ?e_hs ?e_ch)
                    (rope_tensor_range ?heads ?e_hs ?cat_range)

                    ; select(mask, x0g, 0) encoded as two scatters and a gather.
                    (= ?mk0 (Op (Iota (MLt (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))
                    (= ?even0 (Op (Iota (MMul (MIter) (MNum 2)) ?cat_range) (INil)))
                    (= ?odd0 (Op (Iota (MAdd (MMul (MIter) (MNum 2)) (MNum 1)) ?cat_range) (INil)))
                    (= ?base0 (Op (Iota (MMul (MIter) (MNum 2)) ?cat_range) (INil)))
                    (= ?pick0 (Op (Add ?cat_sh ?cat_st ?cat_st ?cat_st)
                        (ICons ?base0 (ICons ?mk0 (INil)))))
                    (= ?zero0 (Op (Cast ?zero0_size (Bf16)) (ICons ?zero0c (INil))))
                    (= ?zero0c (Op (Constant 0.000000) (INil)))
                    (= ?zeros0 (Op (Scatter ?packed_sh ?packed_st ?cat_sh ?cat_st ?zero_st)
                        (ICons ?dest0 (ICons ?even0 (ICons ?zero0 (INil))))))
                    (= ?packed0 (Op (Scatter ?packed_sh ?packed_st ?cat_sh ?cat_st ?cat_st)
                        (ICons ?zeros0 (ICons ?odd0 (ICons ?x0g (INil))))))
                    (= ?selected0 (Op (Gather ?cat_sh ?cat_st ?packed_sh ?packed_st)
                        (ICons ?pick0 (ICons ?packed0 (INil)))))
                )
                (
                    (rope_safe_pad_left ?selected0 ?x0out ?cat_sh
                        ?e_hd ?e_hd2 ?e_seq)
                )
                :ruleset kernel_fuse_late
                :name \"kernel rope IEEE-safe left pad\"
            )
            (rule
                (
                    (= ?cat_sh (ECons ?heads (ECons ?e_seq (ECons ?e_hd (ENil)))))
                    (= ?cat_st (ECons (MMul (MMul (MIter) ?e_hd) ?e_seq)
                        (ECons (MMul (MIter) ?e_hd) (ECons (MIter) (ENil)))))
                    (= ?packed_sh (ECons ?heads (ECons ?e_seq
                        (ECons ?e_hd (ECons (MNum 2) (ENil))))))
                    (= ?packed_st (ECons (MMul (MMul (MMul (MIter) (MNum 2)) ?e_hd) ?e_seq)
                        (ECons (MMul (MMul (MIter) (MNum 2)) ?e_hd)
                            (ECons (MMul (MIter) (MNum 2)) (ECons (MIter) (ENil))))))
                    (= ?zero_st (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil)))))

                    ; The actual right half, clamped only in the inactive half.
                    (= ?x1g (Op (Gather ?cat_sh ?cat_st ?x1_dsh ?x1_dstr)
                        (ICons ?c1idx (ICons ?x1out (INil)))))
                    (= ?c1idx (Op (Iota
                        (MAdd (MAdd (MMax (MSub (MMod (MIter) ?e_hd) ?e_hd2) (MNum 0))
                                    (MMul (MMod (MDiv (MIter) ?e_hd) ?e_seq) ?e_hd2))
                              (MMul (MDiv (MIter) ?e_hs) ?e_ch))
                        ?cat_range) (INil)))
                    (rope_row_dims ?e_hd ?e_hd2 ?e_seq ?e_hs ?e_ch)
                    (rope_tensor_range ?heads ?e_hs ?cat_range)

                    ; select(mask, x1g, 0) with the complementary half mask.
                    (= ?mk1 (Op (Iota (MGte (MMod (MIter) ?e_hd) ?e_hd2) ?cat_range) (INil)))
                    (= ?even1 (Op (Iota (MMul (MIter) (MNum 2)) ?cat_range) (INil)))
                    (= ?odd1 (Op (Iota (MAdd (MMul (MIter) (MNum 2)) (MNum 1)) ?cat_range) (INil)))
                    (= ?base1 (Op (Iota (MMul (MIter) (MNum 2)) ?cat_range) (INil)))
                    (= ?pick1 (Op (Add ?cat_sh ?cat_st ?cat_st ?cat_st)
                        (ICons ?base1 (ICons ?mk1 (INil)))))
                    (= ?zero1 (Op (Cast ?zero1_size (Bf16)) (ICons ?zero1c (INil))))
                    (= ?zero1c (Op (Constant 0.000000) (INil)))
                    (= ?zeros1 (Op (Scatter ?packed_sh ?packed_st ?cat_sh ?cat_st ?zero_st)
                        (ICons ?dest1 (ICons ?even1 (ICons ?zero1 (INil))))))
                    (= ?packed1 (Op (Scatter ?packed_sh ?packed_st ?cat_sh ?cat_st ?cat_st)
                        (ICons ?zeros1 (ICons ?odd1 (ICons ?x1g (INil))))))
                    (= ?selected1 (Op (Gather ?cat_sh ?cat_st ?packed_sh ?packed_st)
                        (ICons ?pick1 (ICons ?packed1 (INil)))))
                )
                (
                    (rope_safe_pad_right ?selected1 ?x1out ?cat_sh
                        ?e_hd ?e_hd2 ?e_seq)
                )
                :ruleset kernel_fuse_late
                :name \"kernel rope IEEE-safe right pad\"
            )
            (rule
                (
                    (rope_rotated ?x0out ?x1out ?x ?pos
                        ?e_hd ?e_hd2 ?e_w ?e_seq ?ln_theta ?inv_hd)
                    (rope_safe_pad_left ?selected0 ?x0out ?cat_sh
                        ?e_hd ?e_hd2 ?e_seq)
                    (rope_safe_pad_right ?selected1 ?x1out ?cat_sh
                        ?e_hd ?e_hd2 ?e_seq)
                    (= ?cat (Op (Add ?cat_sh ?cat_st0 ?cat_st1 ?cat_out_st)
                        (ICons ?selected0 (ICons ?selected1 (INil)))))
                    (= ?cat_sh (ECons ?heads (ECons ?seqd (ECons ?hdd (ENil)))))
                    (= ?hdd ?e_hd)
                    (= ?seqd ?e_seq)
                )
                (
                    (let ?kr (Op (KernelRoPE ?cat_sh (MMul ?heads ?e_hd)
                        (MMul ?e_hd ?e_seq) ?e_w ?ln_theta ?inv_hd)
                        (ICons ?x (ICons ?pos (INil)))))
                    (union ?cat ?kr)
                    (set (dtype ?kr) (Bf16))
                )
                :ruleset kernel_fuse_late2_rope
                :name \"kernel rope IEEE-safe concat\"
            )";
        vec![Rule::raw(format!(
            "{angle_stage}\n{rotation_stage}\n{concat_rule}\n{safe_concat_stage}"
        ))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let width = extract_expr(egraph, kind_children[3], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPE width must be static");
        let ln_theta: f64 = egraph.enodes[kind_children[4]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        let inv_hd: f64 = egraph.enodes[kind_children[5]]
            .0
            .replace('"', "")
            .parse()
            .unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape,
                width,
                ln_theta,
                inv_hd,
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelRoPE {
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let heads = self.out_shape[0].to_usize().expect("RoPE heads is static");
        let seq = self.out_shape[1];
        let hd = self.out_shape[2]
            .to_usize()
            .expect("RoPE head_dim is static");
        let w = self.width;
        let half = hd / 2;
        let lnt = self.ln_theta as f32;
        let inv_hd = self.inv_hd as f32;

        let vars: FxHashSet<Symbol> = seq.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let seq_expr = seq.to_kernel();

        let kernel = format!(
            "#include <cuda_bf16.h>
{dyn_defines}
extern \"C\" {{
    __global__ void rope_k(__nv_bfloat16 *out, const __nv_bfloat16 *x, const int *pos{dyn_dims_param}) {{
        long long s = blockIdx.y;
        int h = blockIdx.z;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {hd}) return;
        int fi = (i < {half}) ? i : (i - {half});
        // angle chain in F32 with the decomposed op spellings
        float v = (float)(2 * fi);
        float freq = ((v * {inv_hd:.10e}f) * {lnt:.10e}f) * 1.442695f;
        float invf = 1.0f / exp2f(freq);
        float angle = (float)pos[s] * invf;
        // bf16 rounding at each decomposed op boundary
        float sb = __bfloat162float(__float2bfloat16(sinf(angle)));
        float cb = __bfloat162float(__float2bfloat16(sinf(angle * -1.0f + 1.570796f)));
        long long xbase = s * {w} + (long long)h * {hd};
        float out_v;
        if (i < {half}) {{
            float x0 = __bfloat162float(x[xbase + i]);
            float x1 = __bfloat162float(x[xbase + i + {half}]);
            float x0c = __bfloat162float(__float2bfloat16(x0 * cb));
            float x1s = __bfloat162float(__float2bfloat16(x1 * sb));
            float x1sn = __bfloat162float(__float2bfloat16(x1s * -1.0f));
            out_v = x0c + x1sn;
        }} else {{
            float x1 = __bfloat162float(x[xbase + i]);
            float x0 = __bfloat162float(x[xbase + i - {half}]);
            float x1c = __bfloat162float(__float2bfloat16(x1 * cb));
            float x0s = __bfloat162float(__float2bfloat16(x0 * sb));
            out_v = x1c + x0s;
        }}
        out[((long long)h * ({seq_expr}) + s) * {hd} + i] = __float2bfloat16(out_v);
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = hd.min(256);
        (
            func,
            module,
            kernel,
            (
                Expression::from(hd.div_ceil(tpb)),
                seq,
                Expression::from(heads),
            ),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.out_shape[1].collect_dyn_vars_into(vars);
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 2 + self.out_shape[1] * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * 16
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEFused"
    }
}

// ═══════════════════════════════════════════════════════════
// KernelRoPEScatterFused — egglog-selected alternative to KernelRoPE followed
// by KernelScatterNoCopy through the exact `(s, heads*hd)` deinterleave view.
// It rotates and writes straight into the cache pool; output aliases dest.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct KernelRoPEScatterFused {
    rope: KernelRoPE,
    /// Total element count of the scatter destination (the cache pool).
    dest_size: Expression,
    /// Flattened scatter-index expression over `z`, where `z` is the element
    /// position in the (s, heads·hd) deinterleaved index grid.
    idx_flat: Expression,
}

impl Default for KernelRoPEScatterFused {
    fn default() -> Self {
        Self {
            rope: KernelRoPE::default(),
            dest_size: 1.into(),
            idx_flat: Expression::from('z'),
        }
    }
}

impl EgglogOp for KernelRoPEScatterFused {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelRoPEScatterFused",
            &[
                ("out_shape", ELIST),
                ("out_width", EXPRESSION),
                ("head_stride", EXPRESSION),
                ("width", EXPRESSION),
                ("ln_theta", F64),
                ("inv_hd", F64),
                ("dest_shape", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        4
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
                (
                    (= ?out_shape
                        (ECons ?heads (ECons ?seq (ECons ?hd (ENil)))))
                    (= ?index_shape (ECons ?seq (ECons ?out_width (ENil))))
                    ; Exact deinterleave view of KernelRoPE's physical
                    ; `(heads, seq, hd)` output:
                    ; (z/hd)*(hd*seq) + z%hd.
                    (= ?src_strides
                        (ECons (MMul (MIter) ?hd)
                            (ECons
                                (MAdd
                                    (MMul (MDiv (MIter) ?hd) ?head_stride)
                                    (MMod (MIter) ?hd))
                                (ENil))))
                    (= ?rope (Op
                        (KernelRoPE ?out_shape ?out_width ?head_stride
                            ?width ?ln_theta ?inv_hd)
                        (ICons ?x (ICons ?pos (INil)))))
                    (= ?scatter (Op
                        (KernelScatterNoCopy ?dest_shape ?dest_strides
                            ?index_shape ?index_strides ?src_strides ?out_strides (Bf16))
                        (ICons ?dest (ICons ?indexes (ICons ?rope (INil))))))
                    (= ?dest_strides ?out_strides)
                    (= (Bf16) (dtype ?x))
                    (= (Bf16) (dtype ?dest))
                    (= (Int) (dtype ?indexes))
                    (= (Int) (dtype ?pos))
                )
                (
                    (let ?fused (Op
                        (KernelRoPEScatterFused ?out_shape ?out_width ?head_stride
                            ?width ?ln_theta ?inv_hd ?dest_shape ?index_shape ?index_strides)
                        (ICons ?dest (ICons ?indexes (ICons ?x (ICons ?pos (INil)))))))
                    (union ?scatter ?fused)
                    (set (dtype ?fused) (Bf16))
                )
                :ruleset post_cleanup
                :name \"kernel rope scatter exact-deinterleave\"
            )",
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let out_shape =
            extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let width = extract_expr(egraph, kind_children[3], expr_cache)
            .unwrap()
            .to_usize()
            .expect("RoPE width must be static");
        let parse_f64 = |child: &'a ENodeId| {
            egraph.enodes[child]
                .0
                .replace('"', "")
                .parse::<f64>()
                .unwrap()
        };
        let dest_shape =
            extract_expr_list(egraph, kind_children[6], list_cache, expr_cache).unwrap();
        let index_shape =
            extract_expr_list(egraph, kind_children[7], list_cache, expr_cache).unwrap();
        let index_strides =
            extract_expr_list(egraph, kind_children[8], list_cache, expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                rope: KernelRoPE {
                    out_shape,
                    width,
                    ln_theta: parse_f64(kind_children[4]),
                    inv_hd: parse_f64(kind_children[5]),
                },
                dest_size: dest_shape.into_iter().product(),
                idx_flat: luminal::shape::flatten_strides(&index_shape, &index_strides),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelRoPEScatterFused {
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        let heads = self.rope.out_shape[0]
            .to_usize()
            .expect("RoPE heads is static");
        let seq = self.rope.out_shape[1];
        let hd = self.rope.out_shape[2]
            .to_usize()
            .expect("RoPE head_dim is static");
        let w = self.rope.width;
        let half = hd / 2;
        let lnt = self.rope.ln_theta as f32;
        let inv_hd = self.rope.inv_hd as f32;
        let kvd = heads * hd;

        let vars: FxHashSet<Symbol> = self.all_dyn_vars();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let idx_expr = self.idx_flat.to_kernel();
        let dest_n = self.dest_size.to_kernel();

        let kernel = format!(
            "#include <cuda_bf16.h>
{dyn_defines}
extern \"C\" {{
    __global__ void rope_scatter_fused_k(__nv_bfloat16 *dest, const int *indexes, const __nv_bfloat16 *x, const int *pos{dyn_dims_param}) {{
        long long s = blockIdx.y;
        int h = blockIdx.z;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {hd}) return;
        int fi = (i < {half}) ? i : (i - {half});
        float v = (float)(2 * fi);
        float freq = ((v * {inv_hd:.10e}f) * {lnt:.10e}f) * 1.442695f;
        float invf = 1.0f / exp2f(freq);
        float angle = (float)pos[s] * invf;
        float sb = __bfloat162float(__float2bfloat16(sinf(angle)));
        float cb = __bfloat162float(__float2bfloat16(sinf(angle * -1.0f + 1.570796f)));
        long long xbase = s * {w} + (long long)h * {hd};
        float out_v;
        if (i < {half}) {{
            float x0 = __bfloat162float(x[xbase + i]);
            float x1 = __bfloat162float(x[xbase + i + {half}]);
            float x0c = __bfloat162float(__float2bfloat16(x0 * cb));
            float x1s = __bfloat162float(__float2bfloat16(x1 * sb));
            float x1sn = __bfloat162float(__float2bfloat16(x1s * -1.0f));
            out_v = x0c + x1sn;
        }} else {{
            float x1 = __bfloat162float(x[xbase + i]);
            float x0 = __bfloat162float(x[xbase + i - {half}]);
            float x1c = __bfloat162float(__float2bfloat16(x1 * cb));
            float x0s = __bfloat162float(__float2bfloat16(x0 * sb));
            out_v = x1c + x0s;
        }}
        long long const_z = s * {kvd} + (long long)h * {hd} + i;
        int idx = indexes[{idx_expr}];
        if (idx >= 0 && idx < ({dest_n})) {{
            dest[idx] = __float2bfloat16(out_v);
        }}
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_scatter_fused_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = hd.min(256);
        (
            func,
            module,
            kernel,
            (
                Expression::from(hd.div_ceil(tpb)),
                seq,
                Expression::from(heads),
            ),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[cudarc::driver::CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // rope_scatter_fused_k: (dest, indexes, x, pos [, dyn_dims]).
        // Writes in place through dest (input 0), not through output_ptr.
        let mut params = vec![input_ptrs[0], input_ptrs[1], input_ptrs[2], input_ptrs[3]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn mutates_aliased_input(&self) -> bool {
        true
    }

    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.idx_flat.collect_dyn_vars_into(vars);
        self.dest_size.collect_dyn_vars_into(vars);
        self.rope.out_shape[1].collect_dyn_vars_into(vars);
    }

    fn output_size(&self) -> Expression {
        self.dest_size
    }

    fn output_bytes(&self) -> Expression {
        self.dest_size * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn bytes_loaded(&self) -> Expression {
        let rotated: Expression = self.rope.out_shape.iter().copied().product();
        rotated * 2 + rotated * 4 + self.rope.out_shape[1] * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.rope.out_shape.iter().copied().product::<Expression>() * 2
    }

    fn flops(&self) -> Expression {
        self.rope.out_shape.iter().copied().product::<Expression>() * 16
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEScatterFused"
    }
}
