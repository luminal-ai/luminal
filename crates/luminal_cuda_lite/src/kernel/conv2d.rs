//! CUDA conv2d-with-bias backend rewrite.
//!
//! `KernelConv2D` is selected by egglog from pure HLIR conv graphs and lowers
//! to a one-thread-per-output CUDA kernel. It avoids materializing unfold/im2col
//! intermediates while keeping model code free of custom ops.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::{FxHashMap, Symbol};
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
        vec![Rule::raw(
            "; A semantic witness is produced only after the complete matmul
            ; decomposition has been proven. The final bias rules consume the
            ; witness in either Add operand order. Keeping the witness separate
            ; also prevents a partially lowered CUDA spelling from weakening
            ; the Gather/index/layout contract.
            (relation conv2d_unfold_matmul
                (IR IR IR EList EList
                 Expression Expression Expression Expression Expression
                 Expression Expression Expression Expression Expression))
            (relation conv2d_1x1_matmul
                (IR IR IR Expression Expression Expression
                 Expression Expression Expression))

            (rule
                (
                    (= ?sum (Op (Sum ?matmul_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride)
                        (ICons ?mul (INil))))
                    (= ?mul (Op (Mul ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride)
                        (ICons ?patches (ICons ?weight (INil)))))
                    (= ?patches (Op (Gather ?idx_shape ?idx_stride ?input_shape ?input_stride)
                        (ICons ?indices (ICons ?input (INil)))))
                    (= ?indices (Op (Iota ?unfold_index ?index_range) (INil)))
                    (= ?input_shape (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                    (= ?idx_shape (ECons ?c_in (ECons ?h_out (ECons ?w_out (ECons (MNum 1) (ECons ?kernel_h (ECons ?kernel_w (ENil))))))))
                    (= ?matmul_shape (ECons ?m (ECons ?c_out (ENil))))
                    (= ?mul_shape (ECons ?m (ECons ?c_out (ECons ?k_dim (ENil)))))

                    ; Constants are already folded (and their arithmetic rows
                    ; subsumed) before kernel_specialize. Prove the same
                    ; equalities directly in egglog's i64 domain instead of
                    ; requiring those hidden MMul/MAdd rows to remain live.
                    (= ?c_in (MNum ?c_in_n))
                    (= ?h_in (MNum ?h_in_n))
                    (= ?w_in (MNum ?w_in_n))
                    (= ?c_out (MNum ?c_out_n))
                    (= ?h_out (MNum ?h_out_n))
                    (= ?w_out (MNum ?w_out_n))
                    (= ?kernel_h (MNum ?kernel_h_n))
                    (= ?kernel_w (MNum ?kernel_w_n))
                    (= ?m (MNum ?m_n))
                    (= ?k_dim (MNum ?k_dim_n))
                    (= ?kernel_area (MNum ?kernel_area_n))
                    (= ?row_kernel (MNum ?row_kernel_n))
                    (= ?spatial_kernel (MNum ?spatial_kernel_n))
                    (= ?input_hw (MNum ?input_hw_n))
                    (= ?m_kernel (MNum ?m_kernel_n))
                    (= ?index_range (MNum ?index_range_n))
                    (= ?h_out_n (+ (- ?h_in_n ?kernel_h_n) 1))
                    (= ?w_out_n (+ (- ?w_in_n ?kernel_w_n) 1))
                    (= ?m_n (* ?h_out_n ?w_out_n))
                    (= ?kernel_area_n (* ?kernel_w_n ?kernel_h_n))
                    (= ?k_dim_n (* ?c_in_n ?kernel_area_n))
                    (= ?row_kernel_n (* ?kernel_area_n ?w_out_n))
                    (= ?spatial_kernel_n (* ?row_kernel_n ?h_out_n))
                    (= ?input_hw_n (* ?w_in_n ?h_in_n))
                    (= ?m_kernel_n (* ?m_n ?kernel_area_n))
                    (= ?index_range_n (* ?c_in_n ?m_kernel_n))

                    (= ?unfold_index
                        (MAdd
                         (MAdd
                          (MAdd
                           (MMod (MDiv (MIter) ?kernel_area) ?w_out)
                           (MAdd
                            (MMod (MIter) ?kernel_w)
                            (MMul
                             (MMod (MDiv (MIter) ?kernel_w) ?kernel_h)
                             ?w_in)))
                          (MMul
                           (MMod (MDiv (MIter) ?row_kernel) ?h_out)
                           ?w_in))
                         (MMul (MDiv (MIter) ?spatial_kernel) ?input_hw)))
                    (= ?idx_stride
                        (ECons
                         (MMul (MMul (MMul (MMul (MIter) ?kernel_w) ?kernel_h) ?w_out) ?h_out)
                         (ECons
                          (MMul (MMul (MMul (MIter) ?kernel_w) ?kernel_h) ?w_out)
                          (ECons
                           (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                           (ECons
                            (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                            (ECons (MMul (MIter) ?kernel_w)
                            (ECons (MIter) (ENil))))))))
                    (= ?patch_stride
                        (ECons (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                         (ECons (MNum 0)
                         (ECons
                          (MAdd
                           (MMul (MDiv (MIter) ?kernel_area) ?m_kernel)
                           (MMod (MIter) ?kernel_area))
                          (ENil)))))
                    (= ?weight_stride
                        (ECons (MNum 0)
                         (ECons ?weight_co_stride
                         (ECons ?weight_inner_stride (ENil)))))
                    (= ?mul_out_stride
                        (ECons
                         (MMul (MMul (MIter) ?k_dim) ?c_out)
                         (ECons (MMul (MIter) ?k_dim)
                         (ECons (MIter) (ENil)))))
                    (= ?sum_in_stride
                        (ECons
                         (MMul (MMul (MIter) ?k_dim) ?c_out)
                         (ECons (MMul (MIter) ?k_dim) (ENil))))
                    (= ?k_stride (MIter))
                    (= ?sum_out_stride
                        (ECons (MMul (MIter) ?c_out) (ECons (MIter) (ENil))))
                    (= (F32) (dtype ?input))
                    (= (F32) (dtype ?weight))
                )
                ((conv2d_unfold_matmul ?sum ?input ?weight ?input_shape ?input_stride
                    ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out ?weight_co_stride
                    ?weight_inner_stride ?kernel_h ?kernel_w))
                :ruleset kernel_specialize
                :name \"prove static conv2d unfold matmul semantics\"
            )

            ; Prove the exact stride-1, dilation-1 unfold represented by
            ;
            ;   [C,Hout,Wout,1,KH,KW] Gather -> [M,Cout,K] Mul -> Sum(K).
            ;
            ; Shape alone is not sufficient: the Iota expression proves which
            ; input element each window position reads, while ?patch_stride
            ; proves the squeeze/permute/merge view used by the matmul lhs.
            (rule
                (
                    (= ?sum (Op (Sum ?matmul_shape ?k_dim ?sum_in_stride ?k_stride ?sum_out_stride)
                        (ICons ?mul (INil))))
                    (= ?mul (Op (Mul ?mul_shape ?patch_stride ?weight_stride ?mul_out_stride)
                        (ICons ?patches (ICons ?weight (INil)))))
                    (= ?patches (Op (Gather ?idx_shape ?idx_stride ?input_shape ?input_stride)
                        (ICons ?indices (ICons ?input (INil)))))
                    (= ?indices (Op (Iota ?unfold_index ?index_range) (INil)))

                    (= ?input_shape
                        (ECons ?c_in (ECons ?h_in (ECons ?w_in (ENil)))))
                    (= ?idx_shape
                        (ECons ?c_in
                         (ECons ?h_out
                         (ECons ?w_out
                         (ECons (MNum 1)
                         (ECons ?kernel_h
                         (ECons ?kernel_w (ENil))))))))
                    (= ?matmul_shape (ECons ?m (ECons ?c_out (ENil))))
                    (= ?mul_shape
                        (ECons ?m (ECons ?c_out (ECons ?k_dim (ENil)))))

                    ; The logical convolution dimensions must be the dimensions
                    ; actually flattened into M and K.
                    (= ?h_out (MAdd (MSub ?h_in ?kernel_h) (MNum 1)))
                    (= ?w_out (MAdd (MSub ?w_in ?kernel_w) (MNum 1)))
                    (= ?m (MMul ?h_out ?w_out))
                    (= ?k_dim (MMul ?c_in (MMul ?kernel_h ?kernel_w)))

                    ; The index tensor itself must be the contiguous unfold
                    ; Iota, not merely an arbitrary tensor with the same shape.
                    (= ?idx_stride
                        (ECons
                         (MMul (MMul (MMul (MMul (MIter) ?kernel_w) ?kernel_h) ?w_out) ?h_out)
                         (ECons
                          (MMul (MMul (MMul (MIter) ?kernel_w) ?kernel_h) ?w_out)
                          (ECons
                           (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                           (ECons
                            (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                            (ECons (MMul (MIter) ?kernel_w)
                            (ECons (MIter) (ENil))))))))
                    (= ?index_range (n_elements ?idx_shape))
                    (= ?unfold_index
                        (MAdd
                         (MAdd
                          (MAdd
                           (MMod
                            (MDiv (MIter) (MMul ?kernel_w ?kernel_h))
                            ?w_out)
                           (MAdd
                            (MMod (MIter) ?kernel_w)
                            (MMul
                             (MMod (MDiv (MIter) ?kernel_w) ?kernel_h)
                             ?w_in)))
                          (MMul
                           (MMod
                            (MDiv
                             (MIter)
                             (MMul (MMul ?kernel_w ?kernel_h) ?w_out))
                            ?h_out)
                           ?w_in))
                         (MMul
                          (MDiv
                           (MIter)
                           (MMul
                            (MMul (MMul ?kernel_w ?kernel_h) ?w_out)
                            ?h_out))
                          (MMul ?w_in ?h_in))))

                    ; Patches are viewed as [M,Cout,K]. The M coordinate walks
                    ; one spatial window, while K walks [Cin,KH,KW] across the
                    ; channel-major Gather materialization.
                    (= ?kernel_area (MMul ?kernel_w ?kernel_h))
                    (= ?patch_stride
                        (ECons (MMul (MMul (MIter) ?kernel_w) ?kernel_h)
                         (ECons (MNum 0)
                         (ECons
                          (MAdd
                           (MMul
                            (MDiv (MIter) ?kernel_area)
                            (MMul (MMul ?m ?kernel_w) ?kernel_h))
                           (MMod (MIter) ?kernel_area))
                          (ENil)))))
                    (= ?weight_stride
                        (ECons (MNum 0)
                         (ECons ?weight_co_stride
                         (ECons ?weight_inner_stride (ENil)))))

                    ; Sum must reduce the contiguous K axis of this exact Mul,
                    ; and its materialized output must be [M,Cout] row-major.
                    (= ?mul_out_stride
                        (ECons
                         (MMul (MMul (MIter) ?k_dim) ?c_out)
                         (ECons (MMul (MIter) ?k_dim)
                         (ECons ?k_stride (ENil)))))
                    (= ?sum_in_stride
                        (ECons
                         (MMul (MMul (MIter) ?k_dim) ?c_out)
                         (ECons (MMul (MIter) ?k_dim) (ENil))))
                    (= ?k_stride (MIter))
                    (= ?sum_out_stride
                        (ECons (MMul (MIter) ?c_out) (ECons (MIter) (ENil))))

                    (= (F32) (dtype ?input))
                    (= (F32) (dtype ?weight))
                )
                ((conv2d_unfold_matmul
                    ?sum ?input ?weight ?input_shape ?input_stride
                    ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out
                    ?weight_co_stride ?weight_inner_stride
                    ?kernel_h ?kernel_w))
                :ruleset kernel_specialize
                :name \"prove conv2d unfold matmul semantics\"
            )

            ; The 1x1 spelling has no Gather. Its flattened spatial coordinate
            ; must be contiguous, and M is tied to Hout*Wout by the bias rule.
            (rule
                (
                    (= ?sum (Op (Sum ?matmul_shape ?c_in ?sum_in_stride ?k_stride ?sum_out_stride)
                        (ICons ?mul (INil))))
                    (= ?mul (Op (Mul ?mul_shape ?input_stride ?weight_stride ?mul_out_stride)
                        (ICons ?input (ICons ?weight (INil)))))
                    (= ?matmul_shape (ECons ?m (ECons ?c_out (ENil))))
                    (= ?mul_shape
                        (ECons ?m (ECons ?c_out (ECons ?c_in (ENil)))))
                    (= ?input_stride
                        (ECons (MIter)
                         (ECons (MNum 0) (ECons ?input_c_stride (ENil)))))
                    (= ?weight_stride
                        (ECons (MNum 0)
                         (ECons ?weight_co_stride
                         (ECons ?weight_inner_stride (ENil)))))
                    (= ?mul_out_stride
                        (ECons
                         (MMul (MMul (MIter) ?c_in) ?c_out)
                         (ECons (MMul (MIter) ?c_in)
                         (ECons ?k_stride (ENil)))))
                    (= ?sum_in_stride
                        (ECons
                         (MMul (MMul (MIter) ?c_in) ?c_out)
                         (ECons (MMul (MIter) ?c_in) (ENil))))
                    (= ?k_stride (MIter))
                    (= ?sum_out_stride
                        (ECons (MMul (MIter) ?c_out) (ECons (MIter) (ENil))))
                    (= (F32) (dtype ?input))
                    (= (F32) (dtype ?weight))
                )
                ((conv2d_1x1_matmul
                    ?sum ?input ?weight ?c_in ?c_out ?m
                    ?input_c_stride ?weight_co_stride ?weight_inner_stride))
                :ruleset kernel_specialize
                :name \"prove conv2d 1x1 matmul semantics\"
            )

            (rule
                (
                    (conv2d_unfold_matmul
                        ?sum ?input ?weight ?input_shape ?input_stride
                        ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out
                        ?weight_co_stride ?weight_inner_stride
                        ?kernel_h ?kernel_w)
                    (= ?add (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride)
                        (ICons ?sum (ICons ?bias (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?c_out (MNum ?c_out_n))
                    (= ?w_out (MNum ?w_out_n))
                    (= ?sum_row_width (MNum ?sum_row_width_n))
                    (= ?sum_row_width_n (* ?w_out_n ?c_out_n))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MIter) ?sum_row_width)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape ?input_shape ?input_stride
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride ?kernel_h ?kernel_w
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel conv2d from proven unfold matmul bias\"
            )
            (rule
                (
                    (conv2d_unfold_matmul
                        ?sum ?input ?weight ?input_shape ?input_stride
                        ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out
                        ?weight_co_stride ?weight_inner_stride
                        ?kernel_h ?kernel_w)
                    (= ?add (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride)
                        (ICons ?bias (ICons ?sum (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?c_out (MNum ?c_out_n))
                    (= ?w_out (MNum ?w_out_n))
                    (= ?sum_row_width (MNum ?sum_row_width_n))
                    (= ?sum_row_width_n (* ?w_out_n ?c_out_n))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MIter) ?sum_row_width)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape ?input_shape ?input_stride
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride ?kernel_h ?kernel_w
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel conv2d from proven bias unfold matmul\"
            )

            (rule
                (
                    (conv2d_1x1_matmul
                        ?sum ?input ?weight ?c_in ?c_out ?m
                        ?input_c_stride ?weight_co_stride ?weight_inner_stride)
                    (= ?add (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride)
                        (ICons ?sum (ICons ?bias (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?m (MNum ?m_n))
                    (= ?h_out (MNum ?h_out_n))
                    (= ?w_out (MNum ?w_out_n))
                    (= ?c_out (MNum ?c_out_n))
                    (= ?sum_row_width (MNum ?sum_row_width_n))
                    (= ?m_n (* ?h_out_n ?w_out_n))
                    (= ?sum_row_width_n (* ?w_out_n ?c_out_n))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MIter) ?sum_row_width)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape
                        (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                        (ECons ?input_c_stride
                         (ECons (MMul (MIter) ?w_out) (ECons (MIter) (ENil))))
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride (MNum 1) (MNum 1)
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel conv2d 1x1 from proven matmul bias\"
            )
            (rule
                (
                    (conv2d_1x1_matmul
                        ?sum ?input ?weight ?c_in ?c_out ?m
                        ?input_c_stride ?weight_co_stride ?weight_inner_stride)
                    (= ?add (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride)
                        (ICons ?bias (ICons ?sum (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?m (MNum ?m_n))
                    (= ?h_out (MNum ?h_out_n))
                    (= ?w_out (MNum ?w_out_n))
                    (= ?c_out (MNum ?c_out_n))
                    (= ?sum_row_width (MNum ?sum_row_width_n))
                    (= ?m_n (* ?h_out_n ?w_out_n))
                    (= ?sum_row_width_n (* ?w_out_n ?c_out_n))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MIter) ?sum_row_width)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape
                        (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                        (ECons ?input_c_stride
                         (ECons (MMul (MIter) ?w_out) (ECons (MIter) (ENil))))
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride (MNum 1) (MNum 1)
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel conv2d 1x1 from proven bias matmul\"
            )

            ; Symbolic dimensions retain their arithmetic rows through expr
            ; saturation, so these consumers prove the same output-layout
            ; contract structurally. The static consumers above use i64 facts
            ; because constant-folding deliberately subsumes those rows.
            (rule
                (
                    (conv2d_unfold_matmul
                        ?sum ?input ?weight ?input_shape ?input_stride
                        ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out
                        ?weight_co_stride ?weight_inner_stride
                        ?kernel_h ?kernel_w)
                    (= ?add (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride)
                        (ICons ?sum (ICons ?bias (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MMul (MIter) ?w_out) ?c_out)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape ?input_shape ?input_stride
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride ?kernel_h ?kernel_w
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel symbolic conv2d from proven unfold matmul bias\"
            )
            (rule
                (
                    (conv2d_unfold_matmul
                        ?sum ?input ?weight ?input_shape ?input_stride
                        ?c_in ?h_in ?w_in ?c_out ?h_out ?w_out
                        ?weight_co_stride ?weight_inner_stride
                        ?kernel_h ?kernel_w)
                    (= ?add (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride)
                        (ICons ?bias (ICons ?sum (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MMul (MIter) ?w_out) ?c_out)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape ?input_shape ?input_stride
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride ?kernel_h ?kernel_w
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel symbolic conv2d from proven bias unfold matmul\"
            )
            (rule
                (
                    (conv2d_1x1_matmul
                        ?sum ?input ?weight ?c_in ?c_out ?m
                        ?input_c_stride ?weight_co_stride ?weight_inner_stride)
                    (= ?add (Op (Add ?out_shape ?sum_add_stride ?bias_add_stride ?out_stride)
                        (ICons ?sum (ICons ?bias (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?m (MMul ?h_out ?w_out))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MMul (MIter) ?w_out) ?c_out)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape
                        (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                        (ECons ?input_c_stride
                         (ECons (MMul (MIter) ?w_out) (ECons (MIter) (ENil))))
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride (MNum 1) (MNum 1)
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel symbolic conv2d 1x1 from proven matmul bias\"
            )
            (rule
                (
                    (conv2d_1x1_matmul
                        ?sum ?input ?weight ?c_in ?c_out ?m
                        ?input_c_stride ?weight_co_stride ?weight_inner_stride)
                    (= ?add (Op (Add ?out_shape ?bias_add_stride ?sum_add_stride ?out_stride)
                        (ICons ?bias (ICons ?sum (INil)))))
                    (= ?out_shape
                        (ECons ?c_out (ECons ?h_out (ECons ?w_out (ENil)))))
                    (= ?m (MMul ?h_out ?w_out))
                    (= ?sum_add_stride
                        (ECons (MIter)
                         (ECons (MMul (MMul (MIter) ?w_out) ?c_out)
                         (ECons (MMul (MIter) ?c_out) (ENil)))))
                    (= ?bias_add_stride
                        (ECons ?bias_c_stride
                         (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                    (= (F32) (dtype ?bias))
                )
                (
                    (let ?conv (Op (KernelConv2D
                        ?out_shape
                        (ECons ?c_in (ECons ?h_out (ECons ?w_out (ENil))))
                        (ECons ?input_c_stride
                         (ECons (MMul (MIter) ?w_out) (ECons (MIter) (ENil))))
                        ?weight_co_stride ?weight_inner_stride ?bias_c_stride
                        ?out_stride (MNum 1) (MNum 1)
                        (MNum 1) (MNum 1) (MNum 1) (MNum 1)
                        (MNum 0) (MNum 0) (F32))
                        (ICons ?input (ICons ?weight (ICons ?bias (INil))))))
                    (union ?add ?conv)
                    (set (dtype ?conv) (F32))
                )
                :ruleset kernel_specialize
                :name \"kernel symbolic conv2d 1x1 from proven bias matmul\"
            )",
        )]
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
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        assert_eq!(self.dtype, DType::F32, "KernelConv2D currently emits F32");

        let vars: FxHashSet<Symbol> = self
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
            .to_kernel_with_index("input_linear");
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

    fn all_dyn_vars(&self) -> FxHashSet<Symbol> {
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
