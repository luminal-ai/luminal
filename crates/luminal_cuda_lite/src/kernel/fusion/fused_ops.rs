// =========================================================================
// Fused elementwise op variants used inside FusionStart/FusionEnd regions.
//
// Each `FusedX` struct mirrors its un-fused `KernelX` sibling field-for-field
// and serves a single purpose: give the egglog rules a distinct sort to
// rewrite into so a pair-fuse rule's RHS can never re-match its own LHS
// pattern. Cascade prevention by typing.
//
// Each FusedX must be absorbed into a FusionEnd-rooted region and compiled by
// `region_codegen`; standalone compilation is intentionally unsupported.
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

use crate::kernel::KernelOp;

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
                _stream: &Arc<CudaStream>,
                _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
            ) -> CompileOut {
                unreachable!(concat!(
                    $sort,
                    " must be compiled through fusion region codegen"
                ))
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
                _stream: &Arc<CudaStream>,
                _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
            ) -> CompileOut {
                unreachable!(concat!(
                    $sort,
                    " must be compiled through fusion region codegen"
                ))
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
