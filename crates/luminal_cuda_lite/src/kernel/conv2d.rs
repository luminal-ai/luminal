//! CUDA conv2d-with-bias backend rewrite.
//!
//! `KernelConv2D` is selected by egglog from pure HLIR conv graphs and lowers
//! to a one-thread-per-output CUDA kernel. It avoids materializing unfold/im2col
//! intermediates while keeping model code free of custom ops.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::FxHashMap;
use luminal::{
    dtype::DType,
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::{EgglogOp, LLIROp},
    prelude::FxHashSet,
    shape::{Expression, flatten_strides},
};

use crate::compile_module_image_for_current_device;
use crate::kernel::{KernelOp, hlir::generate_dyn_dims_defines};

#[derive(Default, Debug, Clone)]
pub struct KernelConv2D {
    out_shape: Vec<Expression>,
    input_shape: Vec<Expression>,
    input_stride: Vec<Expression>,
    weight_co_stride: Expression,
    weight_inner_stride: Expression,
    bias_c_stride: Expression,
    out_stride: Vec<Expression>,
    kernel_h: Expression,
    kernel_w: Expression,
    stride_h: Expression,
    stride_w: Expression,
    dilation_h: Expression,
    dilation_w: Expression,
    pad_h: Expression,
    pad_w: Expression,
    dtype: DType,
}

impl EgglogOp for KernelConv2D {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "KernelConv2D",
            &[
                ("out_shape", ELIST),
                ("input_shape", ELIST),
                ("input_stride", ELIST),
                ("weight_co_stride", EXPRESSION),
                ("weight_inner_stride", EXPRESSION),
                ("bias_c_stride", EXPRESSION),
                ("out_stride", ELIST),
                ("kernel_h", EXPRESSION),
                ("kernel_w", EXPRESSION),
                ("stride_h", EXPRESSION),
                ("stride_w", EXPRESSION),
                ("dilation_h", EXPRESSION),
                ("dilation_w", EXPRESSION),
                ("pad_h", EXPRESSION),
                ("pad_w", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            // 1x1 convs in Flux2's VAE are represented without `unfold`:
            //
            //   input.permute([H,W,C]).merge(H,W)
            //     -> matmul(weight.t())
            //     -> split/permute back to [C_out,H,W]
            //     -> + channel bias
            //
            // The lowered form is still the same Mul -> KernelSum -> Add
            // matmul skeleton, but the lhs FusionStart reads directly from the
            // original input instead of a KernelGather window tensor.
            Rule::raw(
                "(rule
                    (
                        (= ?out (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (= ?add_elem (Op (CudaBinaryElementwise \"Add\" ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride (F32)) (ICons ?sum_fs (ICons ?bias_fs (INil)))))
                        (= ?sum_fs (Op (FusionStart ?out_shape ?sum_add_stride (F32)) (ICons ?sum (INil))))
                        (= ?bias_fs (Op (FusionStart ?out_shape ?bias_add_stride (F32)) (ICons ?bias (INil))))

                        (= ?sum (Op (KernelSum ?matmul_out_shape ?c_in ?sum_in_stride ?k_stride ?sum_out_stride (F32)) (ICons ?mul_fe (INil))))
                        (= ?mul_fe (Op (FusionEnd ?mul_shape ?mul_out_stride (F32)) (ICons ?mul_elem (INil))))
                        (= ?mul_elem (Op (CudaBinaryElementwise \"Mul\" ?mul_shape ?input_1x1_stride ?weight_stride ?mul_out_stride (F32)) (ICons ?input_fs (ICons ?weight_fs (INil)))))
                        (= ?input_fs (Op (FusionStart ?mul_shape ?input_1x1_stride (F32)) (ICons ?input (INil))))
                        (= ?weight_fs (Op (FusionStart ?mul_shape ?weight_stride (F32)) (ICons ?weight (INil))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))
                        (= ?mul_shape (ECons ?m (ECons ?c_out (ECons ?c_in (ENil)))))
                        (= ?input_1x1_stride (ECons ?flat_stride (ECons (MNum 0) (ECons ?input_c_stride (ENil)))))
                        (= ?flat_stride (MIter))

                        (= ?k_stride (MIter))
                        (= ?sum_in_stride (ECons ?sum_m_stride (ECons ?sum_c_stride (ENil))))
                        (= ?sum_out_stride (ECons ?sum_out_m_stride (ECons ?sum_out_c_stride (ENil))))
                        (= ?sum_add_stride (ECons ?sum_add_c_stride (ECons ?sum_add_h_stride (ECons ?sum_add_w_stride (ENil)))))
                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))
                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                            (ECons ?input_c_stride (ECons (MMul ?w_out ?flat_stride) (ECons ?flat_stride (ENil))))
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?out_stride
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?out ?conv)
                        (subsume (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_lower
                    :name \"kernel conv2d 1x1 from cuda lowered matmul bias\"
                )",
            ),
            Rule::raw(
                "(rule
                    (
                        (= ?out (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (= ?add_elem (Op (CudaBinaryElementwise \"Add\" ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride (F32)) (ICons ?bias_fs (ICons ?sum_fs (INil)))))
                        (= ?sum_fs (Op (FusionStart ?out_shape ?sum_add_stride (F32)) (ICons ?sum (INil))))
                        (= ?bias_fs (Op (FusionStart ?out_shape ?bias_add_stride (F32)) (ICons ?bias (INil))))

                        (= ?sum (Op (KernelSum ?matmul_out_shape ?c_in ?sum_in_stride ?k_stride ?sum_out_stride (F32)) (ICons ?mul_fe (INil))))
                        (= ?mul_fe (Op (FusionEnd ?mul_shape ?mul_out_stride (F32)) (ICons ?mul_elem (INil))))
                        (= ?mul_elem (Op (CudaBinaryElementwise \"Mul\" ?mul_shape ?input_1x1_stride ?weight_stride ?mul_out_stride (F32)) (ICons ?input_fs (ICons ?weight_fs (INil)))))
                        (= ?input_fs (Op (FusionStart ?mul_shape ?input_1x1_stride (F32)) (ICons ?input (INil))))
                        (= ?weight_fs (Op (FusionStart ?mul_shape ?weight_stride (F32)) (ICons ?weight (INil))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))
                        (= ?mul_shape (ECons ?m (ECons ?c_out (ECons ?c_in (ENil)))))
                        (= ?input_1x1_stride (ECons ?flat_stride (ECons (MNum 0) (ECons ?input_c_stride (ENil)))))
                        (= ?flat_stride (MIter))

                        (= ?k_stride (MIter))
                        (= ?sum_in_stride (ECons ?sum_m_stride (ECons ?sum_c_stride (ENil))))
                        (= ?sum_out_stride (ECons ?sum_out_m_stride (ECons ?sum_out_c_stride (ENil))))
                        (= ?sum_add_stride (ECons ?sum_add_c_stride (ECons ?sum_add_h_stride (ECons ?sum_add_w_stride (ENil)))))
                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))
                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                            (ECons ?input_c_stride (ECons (MMul ?w_out ?flat_stride) (ECons ?flat_stride (ENil))))
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?out_stride
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?out ?conv)
                        (subsume (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_lower
                    :name \"kernel conv2d 1x1 from cuda lowered bias matmul\"
                )",
            ),
            // Match the same conv after generic CUDA lowering has normalized
            // the elementwise pieces into fusion regions:
            //
            //   KernelGather(input windows)
            //     -> CudaBinaryElementwise("Mul", weight)
            //     -> KernelSum(reduce K)
            //     -> CudaBinaryElementwise("Add", bias)
            //
            // This is the form that survives long enough for CUDA search in
            // real models. The KernelConv2D op consumes the pre-gather input
            // and avoids materializing both the im2col window tensor and the
            // elementwise product tensor.
            //
            // TODO(egglog-shapes): the current e-graph does not reliably prove
            // the derived arithmetic equalities for this chain after CUDA
            // normalization:
            //   * `M == H_out * W_out`
            //   * `K == C_in * KH * KW`
            //   * separately-derived but structurally identical stride
            //     expressions, e.g. the Mul output stride and KernelSum input
            //     stride, belong to the same e-class.
            // Keep the rewrite anchored on the stable conv layout facts the
            // graph does carry today: six-axis unfold window shape, flattened
            // `[M, C_out, K]` product, reduction over `K`, the three-axis
            // `[C_out, H_out, W_out]` output view, and channel-only bias
            // broadcast. Once expression/list canonicalization can prove those
            // equalities, tighten this rule and its regression tests.
            Rule::raw(
                "(rule
                    (
                        (= ?out (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (= ?add_elem (Op (CudaBinaryElementwise \"Add\" ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride (F32)) (ICons ?sum_fs (ICons ?bias_fs (INil)))))
                        (= ?sum_fs (Op (FusionStart ?out_shape ?sum_add_stride (F32)) (ICons ?sum (INil))))
                        (= ?bias_fs (Op (FusionStart ?out_shape ?bias_add_stride (F32)) (ICons ?bias (INil))))

                        (= ?sum (Op (KernelSum ?matmul_out_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride (F32)) (ICons ?mul_fe (INil))))
                        (= ?mul_fe (Op (FusionEnd ?mul_shape ?mul_out_stride (F32)) (ICons ?mul_elem (INil))))
                        (= ?mul_elem (Op (CudaBinaryElementwise \"Mul\" ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride (F32)) (ICons ?patch_fs (ICons ?weight_fs (INil)))))
                        (= ?patch_fs (Op (FusionStart ?mul_shape ?patch_stride (F32)) (ICons ?patches (INil))))
                        (= ?weight_fs (Op (FusionStart ?mul_shape ?weight_stride (F32)) (ICons ?weight (INil))))
                        (= ?patches (Op (KernelGather ?idx_shape ?idx_stride ?input_shape ?input_stride ?gather_out_stride (F32)) (ICons ?indices (ICons ?input (INil)))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?input_shape (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                        (= ?idx_shape (ECons ?c_in (ECons ?h_out (ECons ?w_out (ECons (MNum 1) (ECons ?kernel_h (ECons ?kernel_w (ENil))))))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))
                        (= ?mul_shape (ECons ?m (ECons ?c_out (ECons ?k_dim (ENil)))))

                        (= ?k_stride (MIter))
                        (= ?sum_in_stride (ECons ?sum_m_stride (ECons ?sum_c_stride (ENil))))
                        (= ?sum_out_stride (ECons ?sum_out_m_stride (ECons ?sum_out_c_stride (ENil))))
                        (= ?sum_add_stride (ECons ?sum_add_c_stride (ECons ?sum_add_h_stride (ECons ?sum_add_w_stride (ENil)))))
                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))
                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            ?input_shape
                            ?input_stride
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?out_stride
                            ?kernel_h
                            ?kernel_w
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?out ?conv)
                        (subsume (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_lower
                    :name \"kernel conv2d from cuda lowered unfold matmul bias\"
                )",
            ),
            Rule::raw(
                "(rule
                    (
                        (= ?out (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (= ?add_elem (Op (CudaBinaryElementwise \"Add\" ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride (F32)) (ICons ?bias_fs (ICons ?sum_fs (INil)))))
                        (= ?sum_fs (Op (FusionStart ?out_shape ?sum_add_stride (F32)) (ICons ?sum (INil))))
                        (= ?bias_fs (Op (FusionStart ?out_shape ?bias_add_stride (F32)) (ICons ?bias (INil))))

                        (= ?sum (Op (KernelSum ?matmul_out_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride (F32)) (ICons ?mul_fe (INil))))
                        (= ?mul_fe (Op (FusionEnd ?mul_shape ?mul_out_stride (F32)) (ICons ?mul_elem (INil))))
                        (= ?mul_elem (Op (CudaBinaryElementwise \"Mul\" ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride (F32)) (ICons ?patch_fs (ICons ?weight_fs (INil)))))
                        (= ?patch_fs (Op (FusionStart ?mul_shape ?patch_stride (F32)) (ICons ?patches (INil))))
                        (= ?weight_fs (Op (FusionStart ?mul_shape ?weight_stride (F32)) (ICons ?weight (INil))))
                        (= ?patches (Op (KernelGather ?idx_shape ?idx_stride ?input_shape ?input_stride ?gather_out_stride (F32)) (ICons ?indices (ICons ?input (INil)))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?input_shape (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                        (= ?idx_shape (ECons ?c_in (ECons ?h_out (ECons ?w_out (ECons (MNum 1) (ECons ?kernel_h (ECons ?kernel_w (ENil))))))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))
                        (= ?mul_shape (ECons ?m (ECons ?c_out (ECons ?k_dim (ENil)))))

                        (= ?k_stride (MIter))
                        (= ?sum_in_stride (ECons ?sum_m_stride (ECons ?sum_c_stride (ENil))))
                        (= ?sum_out_stride (ECons ?sum_out_m_stride (ECons ?sum_out_c_stride (ENil))))
                        (= ?sum_add_stride (ECons ?sum_add_c_stride (ECons ?sum_add_h_stride (ECons ?sum_add_w_stride (ENil)))))
                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))
                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            ?input_shape
                            ?input_stride
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?out_stride
                            ?kernel_h
                            ?kernel_w
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?out ?conv)
                        (subsume (Op (FusionEnd ?out_shape ?out_stride (F32)) (ICons ?add_elem (INil))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_lower
                    :name \"kernel conv2d from cuda lowered bias unfold matmul\"
                )",
            ),
            // Match the im2col-style HLIR conv used by Flux2:
            //
            //   input.unfold([1, kh, kw], [1, 1, 1], [1, 1, 1])
            //     -> squeeze/permute/merge view
            //     -> matmul(weight.t())
            //     -> split/permute view
            //     -> + bias.expand_dim(1, h_out).expand_dim(2, w_out)
            //
            // The kernel consumes the pre-unfold input directly. That input may
            // already be a padded HLIR tensor, so the rewrite is still correct
            // for Flux2's padded convs while removing the large patch matrix.
            Rule::raw(
                "(rule
                    (
                        (= ?add (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?add_out_stride) (ICons ?sum (ICons ?bias (INil)))))
                        (= ?sum (Op (Sum ?matmul_out_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride) (ICons ?mul (INil))))
                        (= ?mul (Op (Mul ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride) (ICons ?patches (ICons ?weight (INil)))))
                        (= ?patches (Op (Gather ?idx_shape ?idx_stride ?input_shape ?input_stride) (ICons ?indices (ICons ?input (INil)))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?input_shape (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                        (= ?idx_shape (ECons ?c_in (ECons ?h_out (ECons ?w_out (ECons (MNum 1) (ECons ?kernel_h (ECons ?kernel_w (ENil))))))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))

                        ; This rewrite is for stride=1, dilation=1 over the
                        ; tensor passed to unfold. Padded HLIR inputs are already
                        ; represented as their own tensor, so padding is 0 here.
                        (= ?h_out (MAdd (MSub ?h_in ?kernel_h) (MNum 1)))
                        (= ?w_out (MAdd (MSub ?w_in ?kernel_w) (MNum 1)))
                        (= ?m (MMul ?h_out ?w_out))
                        (= ?k_dim (MMul ?c_in (MMul ?kernel_h ?kernel_w)))
                        (= ?k_stride (MIter))

                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))

                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))

                        (= (F32) (dtype ?input))
                        (= (F32) (dtype ?weight))
                        (= (F32) (dtype ?bias))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            ?input_shape
                            ?input_stride
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?add_out_stride
                            ?kernel_h
                            ?kernel_w
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?add ?conv)
                        (subsume (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?add_out_stride) (ICons ?sum (ICons ?bias (INil)))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_specialize
                    :name \"kernel conv2d from unfold matmul bias\"
                )",
            ),
            Rule::raw(
                "(rule
                    (
                        (= ?add (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?add_out_stride) (ICons ?bias (ICons ?sum (INil)))))
                        (= ?sum (Op (Sum ?matmul_out_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride) (ICons ?mul (INil))))
                        (= ?mul (Op (Mul ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride) (ICons ?patches (ICons ?weight (INil)))))
                        (= ?patches (Op (Gather ?idx_shape ?idx_stride ?input_shape ?input_stride) (ICons ?indices (ICons ?input (INil)))))

                        (= ?out_shape (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                        (= ?input_shape (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                        (= ?idx_shape (ECons ?c_in (ECons ?h_out (ECons ?w_out (ECons (MNum 1) (ECons ?kernel_h (ECons ?kernel_w (ENil))))))))
                        (= ?matmul_out_shape (ECons ?m (ECons ?c_out (ENil))))

                        (= ?h_out (MAdd (MSub ?h_in ?kernel_h) (MNum 1)))
                        (= ?w_out (MAdd (MSub ?w_in ?kernel_w) (MNum 1)))
                        (= ?m (MMul ?h_out ?w_out))
                        (= ?k_dim (MMul ?c_in (MMul ?kernel_h ?kernel_w)))
                        (= ?k_stride (MIter))

                        (= ?weight_co_stride (nth_from_end ?weight_stride 1))
                        (= ?weight_inner_stride (nth_from_end ?weight_stride 0))
                        (= (MNum 0) (nth_from_end ?weight_stride 2))

                        (= ?bias_add_stride (ECons ?bias_c_stride (ECons (MNum 0) (ECons (MNum 0) (ENil)))))

                        (= (F32) (dtype ?input))
                        (= (F32) (dtype ?weight))
                        (= (F32) (dtype ?bias))
                    )
                    (
                        (let ?conv (Op (KernelConv2D
                            ?out_shape
                            ?input_shape
                            ?input_stride
                            ?weight_co_stride
                            ?weight_inner_stride
                            ?bias_c_stride
                            ?add_out_stride
                            ?kernel_h
                            ?kernel_w
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 1)
                            (MNum 0)
                            (MNum 0)
                            (F32))
                            (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                        (union ?add ?conv)
                        (subsume (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?add_out_stride) (ICons ?bias (ICons ?sum (INil)))))
                        (set (dtype ?conv) (F32))
                    )
                    :ruleset kernel_specialize
                    :name \"kernel conv2d from bias unfold matmul\"
                )",
            ),
            Rule::raw(
                "(rule
                    (
                        (= ?add (Op (Add ?shape ?as ?bs ?os) ?inputs))
                        (= ?add (Op (KernelConv2D ?out_shape ?input_shape ?input_stride ?wco ?wi ?bc ?out_stride ?kh ?kw ?sh ?sw ?dh ?dw ?ph ?pw ?dt) ?conv_inputs))
                    )
                    ((delete (Op (Add ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                )",
            ),
            Rule::raw(
                "(rule
                    (
                        (= ?fe (Op (FusionEnd ?shape ?os ?dt) ?inputs))
                        (= ?fe (Op (KernelConv2D ?out_shape ?input_shape ?input_stride ?wco ?wi ?bc ?out_stride ?kh ?kw ?sh ?sw ?dh ?dw ?ph ?pw ?conv_dt) ?conv_inputs))
                    )
                    ((delete (Op (FusionEnd ?shape ?os ?dt) ?inputs)))
                    :ruleset cleanup
                )",
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a luminal::egglog_utils::NodeId],
        input_enodes: Vec<&'a luminal::egglog_utils::NodeId>,
        list_cache: &mut FxHashMap<&'a luminal::egglog_utils::NodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a luminal::egglog_utils::NodeId, Expression>,
    ) -> (LLIROp, Vec<&'a luminal::egglog_utils::NodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                input_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                input_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                weight_co_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                weight_inner_stride: extract_expr(egraph, kind_children[4], expr_cache).unwrap(),
                bias_c_stride: extract_expr(egraph, kind_children[5], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                kernel_h: extract_expr(egraph, kind_children[7], expr_cache).unwrap(),
                kernel_w: extract_expr(egraph, kind_children[8], expr_cache).unwrap(),
                stride_h: extract_expr(egraph, kind_children[9], expr_cache).unwrap(),
                stride_w: extract_expr(egraph, kind_children[10], expr_cache).unwrap(),
                dilation_h: extract_expr(egraph, kind_children[11], expr_cache).unwrap(),
                dilation_w: extract_expr(egraph, kind_children[12], expr_cache).unwrap(),
                pad_h: extract_expr(egraph, kind_children[13], expr_cache).unwrap(),
                pad_w: extract_expr(egraph, kind_children[14], expr_cache).unwrap(),
                dtype: extract_dtype(egraph, kind_children[15]),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelConv2D {
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
        assert_eq!(self.dtype, DType::F32, "KernelConv2D currently emits F32");

        let vars: FxHashSet<char> = self
            .out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .flat_map(|e| e.dyn_vars())
            .chain(self.weight_co_stride.dyn_vars())
            .chain(self.weight_inner_stride.dyn_vars())
            .chain(self.bias_c_stride.dyn_vars())
            .chain(self.kernel_h.dyn_vars())
            .chain(self.kernel_w.dyn_vars())
            .chain(self.stride_h.dyn_vars())
            .chain(self.stride_w.dyn_vars())
            .chain(self.dilation_h.dyn_vars())
            .chain(self.dilation_w.dyn_vars())
            .chain(self.pad_h.dyn_vars())
            .chain(self.pad_w.dyn_vars())
            .collect();

        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let c_out = self.out_shape[0].to_kernel();
        let h_out = self.out_shape[1].to_kernel();
        let w_out = self.out_shape[2].to_kernel();
        let c_in = self.input_shape[0].to_kernel();
        let h_in = self.input_shape[1].to_kernel();
        let w_in = self.input_shape[2].to_kernel();
        let weight_co_stride = self
            .weight_co_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let weight_inner_stride = self
            .weight_inner_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let bias_c_stride = self
            .bias_c_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let kh = self.kernel_h.to_kernel();
        let kw = self.kernel_w.to_kernel();
        let stride_h = self.stride_h.to_kernel();
        let stride_w = self.stride_w.to_kernel();
        let dilation_h = self.dilation_h.to_kernel();
        let dilation_w = self.dilation_w.to_kernel();
        let pad_h = self.pad_h.to_kernel();
        let pad_w = self.pad_w.to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let input_idx = flatten_strides(&self.input_shape, &self.input_stride)
            .to_kernel()
            .replace("const_z", "input_linear");
        let n_outputs: Expression = self.out_shape.iter().copied().product();

        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void generic_conv2d_bias(
        float* __restrict__ out,
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias{dyn_dims_param}
    ) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        const long long total = {total};
        if (const_z >= total) return;

        const long long COUT = {c_out};
        const long long HOUT = {h_out};
        const long long WOUT = {w_out};
        const long long CIN = {c_in};
        const long long HIN = {h_in};
        const long long WIN = {w_in};
        const long long KH = {kh};
        const long long KW = {kw};
        const long long SH = {stride_h};
        const long long SW = {stride_w};
        const long long DH = {dilation_h};
        const long long DW = {dilation_w};
        const long long PH = {pad_h};
        const long long PW = {pad_w};
        const long long W_CO_STRIDE = {weight_co_stride};
        const long long W_INNER_STRIDE = {weight_inner_stride};
        const long long BIAS_C_STRIDE = {bias_c_stride};

        long long co = const_z / (HOUT * WOUT);
        long long rem = const_z - co * HOUT * WOUT;
        long long oh = rem / WOUT;
        long long ow = rem - oh * WOUT;

        float acc = bias[co * BIAS_C_STRIDE];
        for (long long ci = 0; ci < CIN; ++ci) {{
            for (long long r = 0; r < KH; ++r) {{
                long long ih = oh * SH + r * DH - PH;
                if (ih < 0 || ih >= HIN) continue;
                for (long long s = 0; s < KW; ++s) {{
                    long long iw = ow * SW + s * DW - PW;
                    if (iw < 0 || iw >= WIN) continue;
                    long long input_linear = (ci * HIN + ih) * WIN + iw;
                    long long input_idx = {input_idx};
                    long long inner = (ci * KH + r) * KW + s;
                    long long weight_idx = co * W_CO_STRIDE + inner * W_INNER_STRIDE;
                    acc += input[input_idx] * weight[weight_idx];
                }}
            }}
        }}
        out[{out_idx}] = acc;
    }}
}}",
            total = n_outputs.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_conv2d_bias").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs.ceil_div(256), 1.into(), 1.into()),
            (n_outputs.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .flat_map(|e| e.dyn_vars())
            .chain(self.weight_co_stride.dyn_vars())
            .chain(self.weight_inner_stride.dyn_vars())
            .chain(self.bias_c_stride.dyn_vars())
            .chain(self.kernel_h.dyn_vars())
            .chain(self.kernel_w.dyn_vars())
            .chain(self.stride_h.dyn_vars())
            .chain(self.stride_w.dyn_vars())
            .chain(self.dilation_h.dyn_vars())
            .chain(self.dilation_w.dyn_vars())
            .chain(self.pad_h.dyn_vars())
            .chain(self.pad_w.dyn_vars())
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2 * 4 + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericConv2D"
    }
}
