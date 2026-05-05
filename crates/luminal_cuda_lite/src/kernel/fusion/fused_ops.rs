// =========================================================================
// Fused elementwise op variants used inside FusionStart/FusionEnd regions.
//
// Each `FusedX` struct mirrors its un-fused `KernelX` sibling field-for-field
// and serves a single purpose: give the egglog rules a distinct sort to
// rewrite into so a pair-fuse rule's RHS can never re-match its own LHS
// pattern. Cascade prevention by typing.
//
// `compile()` is a *fallback* path. The fast path collapses each FE-rooted
// region into one CUDA kernel inside `region_codegen` and FusedX/FS/FE
// never reach kernel_to_host's compile loop. But extraction can produce
// LLIR shapes the detector doesn't sweep into a region, so each FusedX's
// standalone `compile()` falls back to emitting the same kernel its
// un-fused KernelX sibling would — correct, just one launch per op.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, OP_KIND},
        extract_dtype, extract_expr_list,
    },
    op::*,
    prelude::*,
};

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
};

pub type Ops = (
    FusedSin,
    FusedSqrt,
    FusedExp,
    FusedExp2,
    FusedLog2,
    FusedRecip,
    FusedAdd,
    FusedMul,
);

// Standard `compile()` return tuple (matches the trait signature).
type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
);

// =========================================================================
// Fallback kernel templates — used when a FusedX op reaches
// `kernel_to_host` standalone (region detection missed it). Same CUDA as
// the matching un-fused KernelX would emit, parameterised by the per-op
// body expression. The fast path goes through `region_codegen`.
// =========================================================================

#[allow(clippy::too_many_arguments)]
fn compile_unary_fallback(
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    kernel_name: &str,
    body_expr: &str, // CUDA expression on `in[{in_idx}]`, e.g. "sinf(in[{in_idx}])"
    shape: &[Expression],
    in_strides: &[Expression],
    out_strides: &[Expression],
    dtype: DType,
) -> CompileOut {
    let vars = shape
        .iter()
        .flat_map(|e| e.dyn_vars())
        .chain(in_strides.iter().flat_map(|e| e.dyn_vars()))
        .chain(out_strides.iter().flat_map(|e| e.dyn_vars()))
        .collect::<FxHashSet<_>>();
    let cuda_ty = cuda_dtype(dtype);
    let includes = dtype_includes(&[dtype]);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
    let dyn_dims_param = if vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };
    let n_elements = shape.iter().copied().product::<Expression>().to_kernel();
    let out_idx = flatten_strides(shape, out_strides).to_kernel();
    let in_idx = flatten_strides(shape, in_strides).to_kernel();
    let body = body_expr.replace("{in_idx}", &in_idx);
    let kernel = format!(
        "{includes}\n{dyn_defines}\nextern \"C\" {{\n\
         \x20   __global__ void {kernel_name}({cuda_ty} *out, const {cuda_ty} *in{dyn_dims_param}) {{\n\
         \x20       long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20       if (const_z >= {n_elements}) return;\n\
         \x20       out[{out_idx}] = {body};\n\
         \x20   }}\n}}"
    );
    let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
        (m.clone(), f.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function(kernel_name).unwrap();
        compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
        (module, func)
    };
    let out_size = shape.iter().copied().product::<Expression>();
    (
        func,
        module,
        kernel,
        (out_size.ceil_div(256), 1.into(), 1.into()),
        (out_size.min(256), 1.into(), 1.into()),
        0.into(),
        FxHashMap::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn compile_binary_fallback(
    stream: &Arc<CudaStream>,
    compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    kernel_name: &str,
    op_str: &str, // CUDA infix operator, e.g. "+", "*"
    out_shape: &[Expression],
    a_stride: &[Expression],
    b_stride: &[Expression],
    out_stride: &[Expression],
    dtype: DType,
) -> CompileOut {
    let vars = out_shape
        .iter()
        .flat_map(|e| e.dyn_vars())
        .chain(a_stride.iter().flat_map(|e| e.dyn_vars()))
        .chain(b_stride.iter().flat_map(|e| e.dyn_vars()))
        .chain(out_stride.iter().flat_map(|e| e.dyn_vars()))
        .collect::<FxHashSet<_>>();
    let cuda_ty = cuda_dtype(dtype);
    let includes = dtype_includes(&[dtype, dtype]);
    let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
    let dyn_dims_param = if vars.is_empty() {
        ""
    } else {
        ", const int* dyn_dims"
    };
    let n_elements = out_shape
        .iter()
        .copied()
        .product::<Expression>()
        .to_kernel();
    let out_idx = flatten_strides(out_shape, out_stride).to_kernel();
    let a_idx = flatten_strides(out_shape, a_stride).to_kernel();
    let b_idx = flatten_strides(out_shape, b_stride).to_kernel();
    let kernel = format!(
        "{includes}\n{dyn_defines}\nextern \"C\" {{\n\
         \x20   __global__ void {kernel_name}({cuda_ty} *C, const {cuda_ty} *A, const {cuda_ty} *B{dyn_dims_param}) {{\n\
         \x20       long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20       if (const_z >= {n_elements}) return;\n\
         \x20       C[{out_idx}] = A[{a_idx}] {op_str} B[{b_idx}];\n\
         \x20   }}\n}}"
    );
    let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
        (m.clone(), f.clone())
    } else {
        let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function(kernel_name).unwrap();
        compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
        (module, func)
    };
    let out_size = out_shape.iter().copied().product::<Expression>();
    (
        func,
        module,
        kernel,
        (out_size.ceil_div(256), 1.into(), 1.into()),
        (out_size.min(256), 1.into(), 1.into()),
        0.into(),
        FxHashMap::default(),
    )
}

/// Generate `pub struct $Name { … unary fields … }` plus its `EgglogOp` and
/// `KernelOp` impls. `$kernel_name` names the CUDA function (and the cache
/// key); `$body` is the per-op CUDA expression, e.g. `"sinf(in[{in_idx}])"`.
macro_rules! impl_fused_unary {
    ($Name:ident, $sort:literal, $kernel_name:literal, $body:literal) => {
        #[derive(Default, Debug, Clone)]
        pub struct $Name {
            pub(crate) shape: Vec<Expression>,
            pub(crate) in_strides: Vec<Expression>,
            pub(crate) out_strides: Vec<Expression>,
            pub(crate) dtype: DType,
        }

        impl EgglogOp for $Name {
            fn sort(&self) -> SortDef {
                sort(
                    OP_KIND,
                    $sort,
                    &[
                        ("shape", ELIST),
                        ("strides", ELIST),
                        ("out_strides", ELIST),
                        ("dtype", DTYPE),
                    ],
                )
            }
            fn n_inputs(&self) -> usize {
                1
            }
            fn rewrites(&self) -> Vec<Rule> {
                Vec::new()
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
                (
                    LLIROp::new::<dyn KernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                            .unwrap(),
                        in_strides: extract_expr_list(
                            egraph,
                            kind_children[1],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        out_strides: extract_expr_list(
                            egraph,
                            kind_children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        dtype: extract_dtype(egraph, kind_children[3]),
                    })),
                    input_enodes,
                )
            }
        }

        impl KernelOp for $Name {
            fn compile(
                &self,
                stream: &Arc<CudaStream>,
                compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
            ) -> CompileOut {
                compile_unary_fallback(
                    stream,
                    compile_cache,
                    $kernel_name,
                    $body,
                    &self.shape,
                    &self.in_strides,
                    &self.out_strides,
                    self.dtype,
                )
            }
            fn output_size(&self) -> Expression {
                self.shape.iter().copied().product()
            }
            fn output_bytes(&self) -> Expression {
                (self.output_size() * self.dtype.bits()).ceil_div(8)
            }
            fn bytes_loaded(&self) -> Expression {
                self.output_bytes()
            }
            fn bytes_stored(&self) -> Expression {
                self.output_bytes()
            }
            fn flops(&self) -> Expression {
                self.shape.iter().copied().product()
            }
            fn output_dtype(&self) -> DType {
                self.dtype
            }
            fn kernel_name(&self) -> &'static str {
                $sort
            }
        }
    };
}

/// As `impl_fused_unary!` but for binary ops: 5-field sort signature
/// (shape + per-input strides + out_stride + dtype), n_inputs = 2.
/// `$op_str` is the CUDA infix operator, e.g. `"+"`, `"*"`.
macro_rules! impl_fused_binary {
    ($Name:ident, $sort:literal, $kernel_name:literal, $op_str:literal) => {
        #[derive(Default, Debug, Clone)]
        pub struct $Name {
            pub(crate) out_shape: Vec<Expression>,
            pub(crate) a_stride: Vec<Expression>,
            pub(crate) b_stride: Vec<Expression>,
            pub(crate) out_stride: Vec<Expression>,
            pub(crate) dtype: DType,
        }

        impl EgglogOp for $Name {
            fn sort(&self) -> SortDef {
                sort(
                    OP_KIND,
                    $sort,
                    &[
                        ("shape", ELIST),
                        ("a_strides", ELIST),
                        ("b_strides", ELIST),
                        ("out_strides", ELIST),
                        ("dtype", DTYPE),
                    ],
                )
            }
            fn n_inputs(&self) -> usize {
                2
            }
            fn rewrites(&self) -> Vec<Rule> {
                Vec::new()
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
                (
                    LLIROp::new::<dyn KernelOp>(Box::new(Self {
                        out_shape: extract_expr_list(
                            egraph,
                            kind_children[0],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        a_stride: extract_expr_list(
                            egraph,
                            kind_children[1],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        b_stride: extract_expr_list(
                            egraph,
                            kind_children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        out_stride: extract_expr_list(
                            egraph,
                            kind_children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        dtype: extract_dtype(egraph, kind_children[4]),
                    })),
                    input_enodes,
                )
            }
        }

        impl KernelOp for $Name {
            fn compile(
                &self,
                stream: &Arc<CudaStream>,
                compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
            ) -> CompileOut {
                compile_binary_fallback(
                    stream,
                    compile_cache,
                    $kernel_name,
                    $op_str,
                    &self.out_shape,
                    &self.a_stride,
                    &self.b_stride,
                    &self.out_stride,
                    self.dtype,
                )
            }
            fn output_size(&self) -> Expression {
                self.out_shape.iter().copied().product()
            }
            fn output_bytes(&self) -> Expression {
                (self.output_size() * self.dtype.bits()).ceil_div(8)
            }
            fn bytes_loaded(&self) -> Expression {
                let bytes = (self.output_size() * self.dtype.bits()).ceil_div(8);
                bytes + bytes
            }
            fn bytes_stored(&self) -> Expression {
                self.output_bytes()
            }
            fn flops(&self) -> Expression {
                self.out_shape.iter().copied().product()
            }
            fn output_dtype(&self) -> DType {
                self.dtype
            }
            fn kernel_name(&self) -> &'static str {
                $sort
            }
        }
    };
}

impl_fused_unary!(FusedSin, "FusedSin", "fused_sin_k", "sinf(in[{in_idx}])");
impl_fused_unary!(
    FusedSqrt,
    "FusedSqrt",
    "fused_sqrt_k",
    "sqrtf(in[{in_idx}])"
);
impl_fused_unary!(FusedExp, "FusedExp", "fused_exp_k", "expf(in[{in_idx}])");
impl_fused_unary!(
    FusedExp2,
    "FusedExp2",
    "fused_exp2_k",
    "exp2f(in[{in_idx}])"
);
impl_fused_unary!(
    FusedLog2,
    "FusedLog2",
    "fused_log2_k",
    "log2f(in[{in_idx}])"
);
impl_fused_unary!(
    FusedRecip,
    "FusedRecip",
    "fused_recip_k",
    "1.0f / in[{in_idx}]"
);

impl_fused_binary!(FusedAdd, "FusedAdd", "fused_add_k", "+");
impl_fused_binary!(FusedMul, "FusedMul", "fused_mul_k", "*");
