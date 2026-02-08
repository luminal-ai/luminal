use std::{fmt::Debug, sync::Arc};

use cudarc::driver::CudaStream;
use luminal::{
    egglog_utils::{extract_expr, extract_expr_list},
    op::OpParam::*,
    op::*,
    prelude::*,
};

use crate::block::{BlockOp, CStruct};

pub type Ops = (
    RowAdd,
    RowSwishMul,
    RowRMSNorm,
    RowRope,
    TileMatmulFullSplit,
    // TileMatmulSplitK, // TODO: Fix rewrite rule to not use TileSum and CubeMul
    RowEmbed,
);

#[derive(Debug, Default)]
pub struct RowAdd {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec!["(rule
            (
                ; get add
                (= ?sa (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
                (= ?row_width (nth_from_end ?shape 0))
                (= (MNum ?row_width_num) ?row_width)
                (<= ?row_width_num 4096) ; currently load full row to sram, should instead load chunks in up to capacity and stream rest in
                ; assert the row is contiguous
                (= (MNum 1) (nth_from_end ?a_stride 0))
                (= (MNum 1) (nth_from_end ?b_stride 0))
                (= (MNum 1) (nth_from_end ?out_stride 0))
                ;(= (F32) (dtype ?a))
                ;(= (F32) (dtype ?b))
            )
            (
                (let ?new_shape (RemoveNthFromEnd ?shape 0))
                (let ?new_a_stride (RemoveNthFromEnd ?a_stride 0))
                (let ?new_b_stride (RemoveNthFromEnd ?b_stride 0))
                (let ?new_out_stride (RemoveNthFromEnd ?out_stride 0))
                (let ?ra (RowAdd ?new_shape ?a ?new_a_stride ?b ?new_b_stride ?new_out_stride ?row_width))
                (union ?sa ?ra)
                (set (dtype ?ra) (F32))
            )
            :name \"row add\"
        )".to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[6], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl BlockOp for RowAdd {
    fn op_name(&self) -> &'static str {
        "RowAdd"
    }

    fn launch_range(&self) -> Vec<Expression> {
        if self.range.is_empty() {
            vec![1.into()]
        } else {
            self.range.clone()
        }
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // 1 add per element
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width
    }

    fn bytes_loaded(&self) -> Expression {
        // Load 2 input rows (a + b) per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 2 * 4
    }

    fn cuda_function(&self) -> String {
        "
        const float* a = source_ptrs[0] + eval_expression(payload.a_strides, current);
        const float* b = source_ptrs[1] + eval_expression(payload.b_strides, current);
        float* out = out_ptr + eval_expression(payload.out_strides, current);
        int row_width = eval_expression(payload.row_width, 0);
        for (int idx = t; idx < row_width; idx += blockDim.x) {
            out[idx] = a[idx] + b[idx];
        }
        "
        .to_string()
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr(
                "a_strides",
                flatten_mul_strides(&self.range, &self.a_stride),
            )
            .expr(
                "b_strides",
                flatten_mul_strides(&self.range, &self.b_stride),
            )
            .expr(
                "out_strides",
                flatten_mul_strides(&self.range, &self.out_stride),
            )
            .expr("row_width", self.row_width)
    }
}

#[derive(Debug, Default)]
pub struct RowSwishMul {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    row_width: Expression,
    sm_count: Expression,
}

impl EgglogOp for RowSwishMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowSwishMul".to_string(),
            vec![EList, Input, EList, Input, EList, Expr, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
            (
                (= ?sigmoid (Sigmoid
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                (= ?swish (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?self
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    ?sigmoid
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                (= ?swishmul (Mul
                    (ECons ?batch (ECons ?width (ENil)))
                    ?swish
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    ?other
                    (ECons ?width (ECons (MNum 1) (ENil)))
                    (ECons ?width (ECons (MNum 1) (ENil)))
                ))
                ;(= (F32) (dtype ?self))
                ;(= (F32) (dtype ?other))
            )
            (
                (let ?rsm (RowSwishMul
                    (ECons ?batch (ENil))
                    ?self
                    (ECons ?width (ENil))
                    ?other
                    (ECons ?width (ENil))
                    ?width
                    (MNum 4)
                ))
                (union ?swishmul ?rsm)
                (set (dtype ?rsm) (F32))
            )
            :name \"row swish mul\"
        )"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[5], expr_cache).unwrap(),
                sm_count: extract_expr(egraph, children[6], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl BlockOp for RowSwishMul {
    fn op_name(&self) -> &'static str {
        "RowSwishMul"
    }

    fn launch_range(&self) -> Vec<Expression> {
        // Split across SMs: [batch..., sm_count]
        let mut range = self.range.clone();
        range.push(self.sm_count);
        if range.is_empty() {
            vec![self.sm_count]
        } else {
            range
        }
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        // Batch dims separate, SM dim shared (all SMs contribute to same output)
        let mut barriers = vec![true; self.range.len()];
        barriers.push(false); // SM dimension - shared barrier
        barriers
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        let launch_len = self.launch_range().len();
        vec![vec![true; launch_len], vec![true; launch_len]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load 2 input rows (a + b) per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // swish(x) * b[idx] = x / (1 + exp(-x)) * b
        // ~5 ops per element: neg, exp, add, div, mul
        self.range.iter().copied().product::<Expression>().max(1) * self.row_width * 5
    }

    fn cuda_function(&self) -> String {
        "
        const int row_width = eval_expression(payload.row_width, 0);
        const int sm_count = eval_expression(payload.sm_count, 0);
        const float* a = source_ptrs[0] + eval_expression(payload.a, current);
        const float* b = source_ptrs[1] + eval_expression(payload.b, current);
        float* out = out_ptr + eval_expression(payload.out, current);

        // Split row across SMs
        const int sm_idx = current % sm_count;
        const int elems_per_sm = (row_width + sm_count - 1) / sm_count;
        const int start = sm_idx * elems_per_sm;
        const int end = min(start + elems_per_sm, row_width);

        // Process assigned slice
        for (int idx = start + t; idx < end; idx += blockDim.x) {
            float x = a[idx];
            float sw = x / (1.0f + __expf(-x)); // swish(x)
            out[idx] = sw * b[idx];
        }
        "
        .to_string()
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        // Extend strides with 0 for the SM dimension
        let mut a_stride_ext = self.a_stride.clone();
        a_stride_ext.push(0.into());
        let mut b_stride_ext = self.b_stride.clone();
        b_stride_ext.push(0.into());

        let launch_range = self.launch_range();
        payload
            .expr("a", flatten_mul_strides(&launch_range, &a_stride_ext))
            .expr("b", flatten_mul_strides(&launch_range, &b_stride_ext))
            .expr("out", flatten_mul_strides(&launch_range, &a_stride_ext))
            .expr("row_width", self.row_width)
            .expr("sm_count", self.sm_count)
    }
}

#[derive(Debug, Default)]
pub struct RowRMSNorm {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowRMSNorm {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowRMSNorm".to_string(),
            vec![EList, Input, EList, Expr, Input],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
            (
                (= ?square (Mul ?inp_range ?x ?inp_stride ?x ?inp_stride ?square_stride))
                (= ?width (nth_from_end ?inp_range 0))
                (= ?batch (nth_from_end ?inp_range 1))
                (= ?square_summed
                    (Sum
                        (ECons ?batch (ENil))
                        ?width
                        ?square
                        (ECons ?width (ENil))
                        (MNum 1)
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?inv_div_factor
                    (Recip (ECons ?batch (ENil)) (Cast (Iota ?width (MNum 1)) (MNum 1) (F32))
                                    (ECons (MNum 0) (ENil))  ; broadcast the constant
                                    (ECons (MNum 1) (ENil)))) ; produce per-batch vector

                (= ?mean
                    (Mul (ECons ?batch (ENil))
                                ?square_summed (ECons (MNum 1) (ENil))
                                ?inv_div_factor (ECons (MNum 1) (ENil))
                                (ECons (MNum 1) (ENil))))
                (= ?eps_add
                    (Add
                        (ECons ?batch (ENil))
                        ?mean
                        (ECons (MNum 1) (ENil))
                        (Constant ?eps)
                        (ECons (MNum 0) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?sqrt
                    (Sqrt
                        (ECons ?batch (ENil))
                        ?eps_add
                        (ECons (MNum 1) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?recip
                    (Recip
                        (ECons ?batch (ENil))
                        ?sqrt
                        (ECons (MNum 1) (ENil))
                        (ECons (MNum 1) (ENil))
                    )
                )
                (= ?std_normed
                    (Mul
                        ?inp_range
                        ?recip
                        (ECons (MNum 1) (ECons (MNum 0) (ENil)))
                        ?x
                        ?inp_stride
                        ?inp_stride
                    )
                )
                (= ?final
                    (Mul
                        ?inp_range
                        ?std_normed
                        ?inp_stride
                        ?weight
                        (ECons (MNum 0) (ECons (MNum 1) (ENil)))
                        ?inp_stride
                    )
                )
               ;(= (F32) (dtype ?x))
            )
            (
                (let ?new
                    (RowRMSNorm
                        (ECons ?batch (ENil))
                        ?x
                        (ECons ?width (ENil))
                        ?width
                        ?weight
                    )
                )
                (union ?final ?new)
                (set (dtype ?new) (F32))
            )
            :name \"row rms norm\"
        )"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[3], expr_cache).unwrap(),
            })),
            vec![children[1], children[4]],
        )
    }
}

impl BlockOp for RowRMSNorm {
    fn op_name(&self) -> &'static str {
        "RowRMSNorm"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load input row + weight row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 2 * 4
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // Per row: d squares, d-1 adds for sum, div by d, add eps, sqrt, recip, then 2d muls (inp * inv_rms * weight)
        // Approximate: 5*d ops per row
        self.range.iter().copied().product::<Expression>() * self.row_width * 5
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr("inp", flatten_mul_strides(&self.range, &self.a_stride))
            .expr("out", flatten_mul_strides(&self.range, &self.a_stride))
            .expr("row_width", self.row_width)
    }

    fn cuda_function(&self) -> String {
        "
        const float* inp = source_ptrs[0] + eval_expression(payload.inp, current);
        float*       out = out_ptr + eval_expression(payload.out, current);

        const int d       = eval_expression(payload.row_width, 0);
        const float eps   = 1e-5f;
        const int nthreads = blockDim.x;

        // Shared partial sums (double for accuracy)
        __shared__ double s_partials[1024];  // assumes blockDim.x <= 1024
        __shared__ float  s_inv_rms;

        // 1) Each thread computes a partial sum of squares over its stripe
        double ss_local = 0.0;
        for (int j = t; j < d; j += nthreads) {
            float x = inp[j];
            ss_local += (double)x * (double)x;
        }

        s_partials[t] = ss_local;
        __syncthreads();

        // 2) Parallel reduction in shared memory to get total sum of squares
        for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
            if (t < offset) {
                s_partials[t] += s_partials[t + offset];
            }
            __syncthreads();
        }

        // 3) Thread 0 computes inv_rms and broadcasts it
        if (t == 0) {
            double ss_total = s_partials[0];
            float denom     = sqrtf((float)(ss_total / (double)d) + eps);
            s_inv_rms       = 1.0f / denom;
        }
        __syncthreads();

        float inv_rms = s_inv_rms;

        // 4) All threads normalize their stripe
        for (int j = t; j < d; j += nthreads) {
            out[j] = inp[j] * inv_rms * source_ptrs[1][j];
        }
        "
        .to_string()
    }
}

// TODO: generalize elementwise fusion and remove rope operations
#[derive(Debug, Default)]
pub struct RowRope {
    range: Vec<Expression>,
    a_stride: Vec<Expression>,
    row_width: Expression,
}

impl EgglogOp for RowRope {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowRope".to_string(),
            vec![EList, Input, EList, Expr, Input],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                row_width: extract_expr(egraph, children[3], expr_cache).unwrap(),
            })),
            vec![children[1], children[4]],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           (
                (= ?e (RowRope ?shape ?inp ?stride ?row_width ?pos_ids))
                (= (F32) (dtype ?inp))
            )
           ((set (dtype ?e) (F32)))
        )"
            .to_string(),
        ]
    }

    fn early_rewrites(&self) -> Vec<String> {
        vec![
        r#"
            (rule
              (
                ;; Bind the head-count and hidden-dim directly from the places they appear.
                ;; This matches graphs where these are literals (e.g. 32, 4096) *or* already-simplified expressions.
                (= ?inp_strides (ECons (MNum 128) (ECons ?hidden_dim (ECons (MNum 2) (ECons (MNum 1) (ENil))))))

                ;; -----------------------------
                ;; inv_freq construction (exact literals as in dump)
                ;; -----------------------------
                (= ?freq_indices        (Cast (Iota (MMul (MIter) (MNum 2)) (MNum 64)) (MNum 64) (F32)))
                (= ?c_inv_head_dim      (Constant 0.007812))
                (= ?freq_scaled         (Mul (ECons (MNum 64) (ENil)) ?freq_indices
                                             (ECons (MNum 1) (ENil)) ?c_inv_head_dim
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?c_ln_theta          (Constant 13.122363))
                (= ?log_arg             (Mul (ECons (MNum 64) (ENil)) ?freq_scaled
                                             (ECons (MNum 1) (ENil)) ?c_ln_theta
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?c_log2e             (Constant 1.442695))
                (= ?exp2_arg            (Mul (ECons (MNum 64) (ENil)) ?log_arg
                                             (ECons (MNum 1) (ENil)) ?c_log2e
                                             (ECons (MNum 0) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?pow_theta           (Exp2 (ECons (MNum 64) (ENil)) ?exp2_arg
                                              (ECons (MNum 1) (ENil)) (ECons (MNum 1) (ENil))))
                (= ?inv_freq            (Recip (ECons (MNum 64) (ENil)) ?pow_theta
                                               (ECons (MNum 1) (ENil)) (ECons (MNum 1) (ENil))))

                ;; -----------------------------
                ;; emb = pos_ids @ inv_freq
                ;; -----------------------------
                (= ?pos_f32             (Cast ?pos_ids ?cast_sh (F32)))
                (= ?pos_times_invfreq_bcast
                   (Mul (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil))))
                        ?pos_f32
                        (ECons (MNum 1) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                        ?inv_freq
                        (ECons (MNum 0) (ECons (MNum 1) (ECons (MNum 0) (ENil))))
                        (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil))))))
                (= ?emb
                   (Sum (ECons (MVar "s") (ECons (MNum 64) (ENil)))
                        (MNum 1)
                        ?pos_times_invfreq_bcast
                        (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                        (MNum 1)
                        (ECons (MNum 64) (ECons (MNum 1) (ENil)))))

                ;; -----------------------------
                ;; Gather odd lane (x1) from inp (structure preserved; 32 -> ?n_heads, 4096 -> ?hidden_dim)
                ;; -----------------------------
                (= ?odd_lane_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MNum 1)
                           (MMul (MMod (MIter) (MNum 64)) (MNum 2)))
                         (MMul (MMod (MDiv (MIter) (MNum 64)) (MVar "s")) (MNum 128)))
                       (MMul (MDiv (MIter) (MMul (MNum 64) (MVar "s")))
                             (MMul (MNum 128) (MVar "s"))))
                     (MMul (MMul ?n_heads (MVar "s")) (MNum 64))))
                (= ?odd_lane
                   (Gather
                     ?odd_lane_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                     ?inp
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     ?inp_strides))
                ;(= (F32) (dtype ?inp))

                ;; -----------------------------
                ;; cos(emb) = sin(-emb + pi/2), sin(emb)
                ;; -----------------------------
                (= ?c_neg1    (Constant -1.000000))
                (= ?neg_emb   (Mul (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil))) ?c_neg1
                                   (ECons (MNum 0) (ECons (MNum 0) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?c_pihalf  (Constant 1.570796))
                (= ?cos_phase (Add (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?neg_emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil))) ?c_pihalf
                                   (ECons (MNum 0) (ECons (MNum 0) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?cos_emb   (Sin (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?cos_phase
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                (= ?sin_emb   (Sin (ECons (MVar "s") (ECons (MNum 64) (ENil))) ?emb
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))
                                   (ECons (MNum 64) (ECons (MNum 1) (ENil)))))

                ;; -----------------------------
                ;; even_lane_rot = x0*cos - x1*sin
                ;; -----------------------------
                (= ?x0_cos
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?inp
                        ?inp_strides
                        ?cos_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?x1_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?odd_lane
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?sin_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?c_neg1b (Constant -1.000000))
                (= ?neg_x1_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x1_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?c_neg1b
                        (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?even_lane_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x0_cos
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?neg_x1_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))

                ;; -----------------------------
                ;; odd_lane_rot = x0*sin + x1*cos
                ;; -----------------------------
                (= ?x0_sin
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?inp
                        ?inp_strides
                        ?sin_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?x1_cos
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?odd_lane
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?cos_emb
                        (ECons (MNum 0) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 0) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?odd_lane_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                        ?x0_sin
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        ?x1_cos
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))

                ;; -----------------------------
                ;; Scatter + masks (keep the same MMul nesting as original)
                ;; -----------------------------
                (= ?scatter_even_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MMin (MMod (MIter) (MNum 2)) (MNum 0))
                           (MMod (MDiv (MIter) (MNum 2)) (MNum 64)))
                         (MMul (MMod (MDiv (MIter) (MNum 128)) (MVar "s")) (MNum 64)))
                       (MMul (MDiv (MIter) (MMul (MNum 128) (MVar "s"))) (MMul (MNum 64) (MVar "s"))))
                     (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?scattered_even
                   (Gather
                     ?scatter_even_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                     ?even_lane_rot
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?even_mask
                   (Iota (MLt (MMod (MIter) (MNum 2)) (MNum 1))
                         (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?even_masked
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?scattered_even
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?even_mask
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                (= ?scatter_odd_index
                   (Iota
                     (MAdd
                       (MAdd
                         (MAdd
                           (MMax (MSub (MMod (MIter) (MNum 2)) (MNum 1)) (MNum 0))
                           (MMod (MDiv (MIter) (MNum 2)) (MNum 64)))
                         (MMul (MMod (MDiv (MIter) (MNum 128)) (MVar "s")) (MNum 64)))
                       (MMul (MDiv (MIter) (MMul (MNum 128) (MVar "s"))) (MMul (MNum 64) (MVar "s"))))
                     (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?scattered_odd
                   (Gather
                     ?scatter_odd_index
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                     (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                     ?odd_lane_rot
                     (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 1) (ENil)))))
                     (ECons (MMul (MNum 64) (MVar "s")) (ECons (MNum 64) (ECons (MNum 1) (ECons (MNum 1) (ENil)))))))
                (= ?odd_mask
                   (Iota (MGte (MMod (MIter) (MNum 2)) (MNum 1))
                         (MMul (MMul (MMul ?n_heads (MVar "s")) (MNum 64)) (MNum 2))))
                (= ?odd_masked
                   (Mul (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?scattered_odd
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?odd_mask
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                (= ?interleaved_rot
                   (Add (ECons ?n_heads (ECons (MVar "s") (ECons (MNum 64) (ECons (MNum 2) (ENil)))))
                        ?even_masked
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        ?odd_masked
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))
                        (ECons (MMul (MNum 128) (MVar "s")) (ECons (MNum 128) (ECons (MNum 2) (ECons (MNum 1) (ENil)))))))

                ;; Final identity mul "* 1.0" with output shape/strides
                (= ?c_one (Constant 1.000000))
                (= ?rope_out
                   (Mul (ECons (MVar "s") (ECons ?n_heads (ECons (MNum 128) (ENil))))
                        ?interleaved_rot
                        (ECons (MNum 128) (ECons (MMul (MVar "s") (MNum 128)) (ECons (MNum 1) (ENil))))
                        ?c_one
                        (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                        (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil)))))
              )
              )
              (
                (union ?rope_out
                  (RowRope
                    (ECons (MVar "s") (ENil))
                    ?inp
                    (ECons ?hidden_dim (ENil))
                    ?hidden_dim
                    ?pos_ids))
                ; we want to subsume all terms up to ?inp and ?pos_ids. don't know how to do this.
                (delete (Mul (ECons (MVar "s") (ECons ?n_heads (ECons (MNum 128) (ENil))))
                     ?interleaved_rot
                     (ECons (MNum 128) (ECons (MMul (MVar "s") (MNum 128)) (ECons (MNum 1) (ENil))))
                     ?c_one
                     (ECons (MNum 0) (ECons (MNum 0) (ECons (MNum 0) (ENil))))
                     (ECons ?hidden_dim (ECons (MNum 128) (ECons (MNum 1) (ENil))))))
              )
              :name "row rope"
            )
        "#.to_string()]
    }
}

impl BlockOp for RowRope {
    fn op_name(&self) -> &'static str {
        "RowRope"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>() * self.row_width
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load input row (row_width floats) + token_ids (1 int per row)
        self.range.iter().copied().product::<Expression>() * (self.row_width * 4 + 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store 1 output row per launch
        self.range.iter().copied().product::<Expression>() * self.row_width * 4
    }

    fn flops(&self) -> Expression {
        // Per pair of elements: pow, sincos, 4 muls, 2 adds ≈ 10 ops
        // row_width/2 pairs per row
        self.range.iter().copied().product::<Expression>() * self.row_width * 5
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr("inp", flatten_mul_strides(&self.range, &self.a_stride))
            .expr("out", flatten_mul_strides(&self.range, &self.a_stride))
            .expr("row_width", self.row_width)
            .expr("token_ids", 'z')
    }

    fn cuda_function(&self) -> String {
        "
        const float* inp = source_ptrs[0] + eval_expression(payload.inp, current);
        float*       out = out_ptr + eval_expression(payload.out, current);
        const int* token_ids = (const int*)source_ptrs[1] + eval_expression(payload.token_ids, current);

        const int D_total = eval_expression(payload.row_width, 0);    // = n_heads * d_head
        const int d_head  = 128;            // head_dim
        const int n_heads = D_total / d_head;

        const int   pos  = token_ids[0];   // must match position_ids[batch, seq]
        const float base = 500000.0f;

        const int half = d_head / 2;            // 64 when d_head = 128

        for (int h = 0; h < n_heads; ++h) {
            const float* head_in  = inp + h * d_head;
            float*       head_out = out + h * d_head;

            // k indexes within the first half [0 .. half-1]
            for (int k = t; k < half; k += blockDim.x) {
                const int j0 = k;           // first half index
                const int j1 = k + half;    // corresponding second-half index

                // exponent = -(2*k / d_head) to match inv_freq = base^{-(arange(0,dim,2)/dim)}
                const float exponent = -(2.0f * (float)k) / (float)d_head;
                const float theta    = (float)pos * __powf(base, exponent);

                float s, c;
                __sincosf(theta, &s, &c);

                const float x0 = head_in[j0];
                const float x1 = head_in[j1];

                head_out[j0] = x0 * c - x1 * s;
                head_out[j1] = x1 * c + x0 * s;
            }
        }
        "
        .to_string()
    }
}

pub const TILE_SIZE: u32 = 64;
const K_CHUNK_SIZE: usize = 4096;

#[derive(Debug, Default)]
pub struct TileMatmulSplitK {
    range: Vec<Expression>,         // [batch..., tiled_m, tiled_n, k_chunks]
    untiled_range: Vec<Expression>, // [M, N]
    total_k: Expression,
    a_stride: Vec<Expression>,
    a_m_stride: Expression,
    b_stride: Vec<Expression>,
    b_n_stride: Expression,
    out_stride: Vec<Expression>,
    out_m_stride: Expression,
    k_chunk: Expression,
}

impl EgglogOp for TileMatmulSplitK {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "TileMatmulSplitK".to_string(),
            vec![
                EList, EList, Expr, Input, EList, Expr, Expr, Input, EList, Expr, Expr, EList,
                Expr, Expr, Expr,
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            // Direct Mul -> Sum -> TileMatmulSplitK (A row-major, B col-major, C row-major)
            format!(
                "
        (rule
            (
                ; Match Mul node
                (= ?mul (Mul ?mul_shape ?a ?a_stride ?b ?b_stride ?mul_out_stride))

                ; Match Sum that reduces the Mul (k dimension)
                (= ?sum (Sum ?out_shape ?k ?mul ?sum_in_stride ?k_stride ?sum_out_stride))

                ; Get dimensions from output shape
                (= ?m (nth_from_end ?out_shape 1))
                (= ?n (nth_from_end ?out_shape 0))
                (!= ?m (MNum 0))
                (!= ?n (MNum 0))

                ; Get output strides
                (= ?sum_out_m_stride (nth_from_end ?sum_out_stride 1))
                (= ?sum_out_n_stride (nth_from_end ?sum_out_stride 0))

                ; Get A strides
                (= ?a_m_stride (nth_from_end ?a_stride 2))
                (= ?a_n_stride (nth_from_end ?a_stride 1))
                (= ?a_k_stride (nth_from_end ?a_stride 0))

                ; Get B strides
                (= ?b_m_stride (nth_from_end ?b_stride 2))
                (= ?b_n_stride (nth_from_end ?b_stride 1))
                (= ?b_k_stride (nth_from_end ?b_stride 0))

                ; Assert contiguous k stride on output (required for reduction)
                (= ?k_stride (MNum 1))

                ; Assert A has contiguous k (row-major A)
                (= ?a_k_stride (MNum 1))

                ; Assert B has contiguous k (col-major B / transposed)
                (= ?b_k_stride (MNum 1))

                ;(= (F32) (dtype ?a))
            )
            (
                ; Create tiled shape with K chunks
                (let ?tiled_m (MCeilDiv ?m (MNum {ts})))
                (let ?tiled_n (MCeilDiv ?n (MNum {ts})))
                ;(let ?total_output_tiles (MMul ?tiled_m ?tiled_n))
                ;(let ?k_chunk_size (MCeilDiv ))
                (let ?k_chunks (MCeilDiv ?k (MNum {kc})))
                (let ?tiled_shape
                    (ECons ?k_chunks
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd ?out_shape ?tiled_n 0)
                        ?tiled_m 1)))

                ; Create tiled strides for A: scale m and n strides, remove k
                (let ?scaled_a_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?a_stride
                            (MMul ?a_n_stride (MNum {ts})) 1)
                        (MMul ?a_m_stride (MNum {ts})) 2))
                (let ?tiled_a_stride (ECons (MNum 0) (RemoveNthFromEnd ?scaled_a_stride 0)))

                ; Create tiled strides for B: scale m and n strides, remove k
                (let ?scaled_b_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?b_stride
                            (MMul ?b_n_stride (MNum {ts})) 1)
                        (MMul ?b_m_stride (MNum {ts})) 2))
                (let ?tiled_b_stride (ECons (MNum 0) (RemoveNthFromEnd ?scaled_b_stride 0)))

                ; Create tiled output strides (k_chunk dimension has 0 stride since all chunks write to same output)
                (let ?tiled_out_stride
                    (ECons (MNum 0)
                        (ReplaceNthFromEnd
                            (ReplaceNthFromEnd ?sum_out_stride (MMul ?sum_out_n_stride (MNum {ts})) 0)
                        (MMul ?sum_out_m_stride (MNum {ts})) 1)))

                (let ?tm (TileMatmulSplitK
                    ?tiled_shape ?out_shape ?k
                    ?a ?tiled_a_stride ?a_m_stride (MNum 1)
                    ?b ?tiled_b_stride (MNum 1) ?b_n_stride
                    ?tiled_out_stride ?sum_out_m_stride (MNum 1) (MNum {kc})))
                (union ?sum ?tm)
                (set (dtype ?tm) (F32))
                ; Subsume TileSum and CubeMul so they aren't chosen over TileMatmul
                (subsume (TileSum ?sum_shape ?untiled_sum_shape ?iters ?cm ?sum_in_stride ?sum_in_m_stride ?sum_in_n_stride ?sum_in_k_stride ?sum_out_stride ?sum_out_m_stride ?sum_out_n_stride))
                (subsume (CubeMul ?mul_shape ?untiled_mul_shape ?a ?a_stride ?a_m_stride ?a_n_stride ?a_k_stride ?b ?b_stride ?b_m_stride ?b_n_stride ?b_k_stride ?out_stride ?out_m_stride ?out_n_stride ?out_k_stride))
            )
            :name \"tile matmul split k\"
        )",
                ts = TILE_SIZE,
                kc = K_CHUNK_SIZE
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                untiled_range: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                total_k: extract_expr(egraph, children[2], expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                a_m_stride: extract_expr(egraph, children[5], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[8], list_cache, expr_cache).unwrap(),
                b_n_stride: extract_expr(egraph, children[10], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[11], list_cache, expr_cache)
                    .unwrap(),
                out_m_stride: extract_expr(egraph, children[12], expr_cache).unwrap(),
                k_chunk: extract_expr(egraph, children[14], expr_cache).unwrap(),
            })),
            vec![children[3], children[7]],
        )
    }
}

impl BlockOp for TileMatmulSplitK {
    fn op_name(&self) -> &'static str {
        "TileMatmulSplitK"
    }

    fn launch_range(&self) -> Vec<Expression> {
        self.range.clone()
    }

    fn output_size(&self) -> Expression {
        self.untiled_range.iter().copied().product::<Expression>()
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        // All dimensions are separable except k_chunks (at index 0)
        // since multiple k_chunks write to the same output tile via atomicAdd
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        let mut sep = vec![true; self.range.len()];
        sep[0] = false; // k_chunk dimension at index 0 is NOT separable
        sep
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        // For input A: all dims except n (at index len-1)
        let mut a = vec![true; self.range.len()];
        a[self.range.len() - 1] = false; // n dimension
        // For input B: all dims except m (at index len-2)
        let mut b = vec![true; self.range.len()];
        b[self.range.len() - 2] = false; // m dimension
        vec![a, b]
    }

    fn bytes_stored(&self) -> Expression {
        // Store C (M * N) floats - each k_chunk atomically adds
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        batch * m * n * 4
    }

    fn flops(&self) -> Expression {
        // Matmul FLOPs: 2 * M * N * K (one mul + one add per output element per K iteration)
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        batch * m * n * k * 2
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        assert_eq!(self.untiled_range.len(), 2);
        // Range layout: [k_chunks, batch..., tiled_m, tiled_n]
        // k_chunk is at index 0
        let mut k_chunk_stride = vec![0.into(); self.range.len()];
        k_chunk_stride[0] = 1.into();
        // m_pos (tiled_m) is at index len-2
        let mut m_pos_stride = vec![0.into(); self.range.len()];
        m_pos_stride[self.range.len() - 2] = 1.into();
        // n_pos (tiled_n) is at index len-1
        let mut n_pos_stride = vec![0.into(); self.range.len()];
        n_pos_stride[self.range.len() - 1] = 1.into();
        payload
            .expr_arr("untiled_range", &self.untiled_range)
            .expr("a", flatten_mul_strides(&self.range, &self.a_stride))
            .expr("b", flatten_mul_strides(&self.range, &self.b_stride))
            .expr("c", flatten_mul_strides(&self.range, &self.out_stride))
            .expr("total_k", self.total_k)
            .expr("a_width", self.a_m_stride)
            .expr("b_width", self.b_n_stride)
            .expr("c_width", self.out_m_stride)
            .expr(
                "m_pos_stride",
                flatten_mul_strides(&self.range, &m_pos_stride),
            )
            .expr(
                "n_pos_stride",
                flatten_mul_strides(&self.range, &n_pos_stride),
            )
            .expr(
                "k_chunk_stride",
                flatten_mul_strides(&self.range, &k_chunk_stride),
            )
            .expr("k_chunk_size", self.k_chunk)
    }

    fn bytes_loaded(&self) -> Expression {
        // Load A (M * K) + B (K * N) per batch
        let batch: Expression = if self.range.len() > 3 {
            self.range[..self.range.len() - 3].iter().copied().product()
        } else {
            1.into()
        };
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        batch * (m * k + k * n) * 4
    }

    fn cuda_function(&self) -> String {
        format!("
        auto warp_reduce_sum = [](float val) {{
            for (int offset = 16; offset > 0; offset >>= 1) {{
                val += __shfl_down_sync(0xffffffff, val, offset);
            }}
            return val;
        }};
        const int k_chunk = eval_expression(payload.k_chunk_stride, current);
        const int k_chunk_size = eval_expression(payload.k_chunk_size, current);
        const int total_K = eval_expression(payload.total_k, 0);
        const int k_start = k_chunk * k_chunk_size;
        const int k_end = min(k_start + k_chunk_size, total_K);
        const int K = k_end - k_start;

        if (K <= 0) return;

        const int a_offset = eval_expression(payload.a, current);
        const int b_offset = eval_expression(payload.b, current);
        const int c_offset = eval_expression(payload.c, current);
        const float* a = source_ptrs[0] + a_offset + k_start;
        const float* b = source_ptrs[1] + b_offset + k_start;
        float*       c = out_ptr + c_offset;
        const int m_pos = eval_expression(payload.m_pos_stride, current);
        const int n_pos = eval_expression(payload.n_pos_stride, current);

        const int threads   = blockDim.x;
        const int lane      = t & 31;
        const int warp_id   = t >> 5;
        const int num_warps = threads >> 5;

        constexpr int TILE_SIZE = {ts};

        const int global_m0 = m_pos * TILE_SIZE;
        const int global_n0 = n_pos * TILE_SIZE;
        const int M = eval_expression(payload.untiled_range[0], 0);
        const int N = eval_expression(payload.untiled_range[1], 0);

        const int rows_left = M - global_m0;
        const int cols_left = N - global_n0;
        if (rows_left <= 0 || cols_left <= 0) return;

        const int tile_m = min(rows_left, TILE_SIZE);
        const int tile_n = min(cols_left, TILE_SIZE);

        const int b_width = eval_expression(payload.b_width, 0);

        // Fast path for M=1 decode: warps parallelize over columns with K reduction
        if (tile_m == 1 && num_warps > 0) {{
            constexpr int COLS_PER_WARP = 4;
            for (int col_base = warp_id * COLS_PER_WARP; col_base < tile_n; col_base += num_warps * COLS_PER_WARP) {{
                float partial[COLS_PER_WARP] = {{0.0f, 0.0f, 0.0f, 0.0f}};

                // Stream K elements from this chunk
                for (int k = lane; k < K; k += 32) {{
                    float a_val = a[k];
                    #pragma unroll
                    for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                        if (col_base + ci < tile_n) {{
                            partial[ci] += a_val * b[(col_base + ci) * b_width + k];
                        }}
                    }}
                }}

                // Warp reduction for each column
                #pragma unroll
                for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                    partial[ci] = warp_reduce_sum(partial[ci]);
                }}
                // Lane 0 atomically adds results
                if (lane == 0) {{
                    #pragma unroll
                    for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                        if (col_base + ci < tile_n) {{
                            atomicAdd(&c[col_base + ci], partial[ci]);
                        }}
                    }}
                }}
            }}
        }} else {{
            // Generic path: handle any M, N tile
            const int tile_elems = tile_m * tile_n;
            const int c_width = eval_expression(payload.c_width, 0);
            const int a_width = eval_expression(payload.a_width, 0);

            for (int idx = t; idx < tile_elems; idx += threads) {{
                int ty = idx / tile_n;
                int tx = idx % tile_n;

                const float* A0 = a + ty * a_width;
                const float* B0 = b + tx * b_width;
                float*       C0 = c + ty * c_width + tx;

                float acc = 0.f;
                for (int k = 0; k < K; ++k) {{
                    acc += A0[k] * B0[k];
                }}

                atomicAdd(C0, acc);
            }}
        }}
        ", ts = TILE_SIZE)
    }
}

/// TileMatmulFullSplit: Optimally splits matmul work across SMs by computing a dynamic k_chunk_size.
/// Unlike TileMatmulSplitK which has fixed k-chunks, this operation:
/// 1. Computes k_chunk_size = ceil((m_tiles * n_tiles * k) / num_sm)
/// 2. Flattens the iteration space as (m_tiles, n_tiles, k)
/// 3. Each SM handles a contiguous span that may cross output tile boundaries
/// 4. The kernel accumulates and stores when crossing tile boundaries
#[derive(Debug, Default)]
pub struct TileMatmulFullSplit {
    sm_count: Expression,           // Number of work units (num_sm)
    untiled_range: Vec<Expression>, // [M, N]
    m_tiles: Expression,
    n_tiles: Expression,
    total_k: Expression,
    #[allow(dead_code)]
    a_stride: Vec<Expression>, // Batch strides for A (reserved for batch support)
    a_m_stride: Expression, // A stride for m tile position (TILE_SIZE steps)
    a_k_stride: Expression, // A stride for k position (usually 1)
    #[allow(dead_code)]
    b_stride: Vec<Expression>, // Batch strides for B (reserved for batch support)
    b_n_stride: Expression, // B stride for n tile position (TILE_SIZE steps)
    b_k_stride: Expression, // B stride for k position (usually 1)
    #[allow(dead_code)]
    out_stride: Vec<Expression>, // Batch strides for output (reserved for batch support)
    out_m_stride: Expression, // Output stride for m position within tile
    out_n_stride: Expression, // Output stride for n position within tile
}

impl EgglogOp for TileMatmulFullSplit {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "TileMatmulFullSplit".to_string(),
            vec![
                Expr,  // sm_count
                EList, // untiled_range
                Expr,  // m_tiles
                Expr,  // n_tiles
                Expr,  // total_k
                Input, // a
                EList, // a_stride
                Expr,  // a_m_stride
                Expr,  // a_k_stride
                Input, // b
                EList, // b_stride
                Expr,  // b_n_stride
                Expr,  // b_k_stride
                EList, // out_stride
                Expr,  // out_m_stride
                Expr,  // out_n_stride
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            // Match Mul -> Sum pattern for matmul (A row-major, B col-major)
            format!(
                "
        (rule
            (
                ; Match Mul node
                (= ?mul (Mul ?mul_shape ?a ?a_stride ?b ?b_stride ?mul_out_stride))

                ; Match Sum that reduces the Mul (k dimension)
                (= ?sum (Sum ?out_shape ?k ?mul ?sum_in_stride ?k_stride ?sum_out_stride))

                ; Get dimensions from output shape
                (= ?m (nth_from_end ?out_shape 1))
                (= ?n (nth_from_end ?out_shape 0))
                (!= ?m (MNum 0))
                (!= ?n (MNum 0))

                ; Get output strides
                (= ?sum_out_m_stride (nth_from_end ?sum_out_stride 1))
                (= ?sum_out_n_stride (nth_from_end ?sum_out_stride 0))

                ; Get A strides
                (= ?a_m_stride (nth_from_end ?a_stride 2))
                (= ?a_n_stride (nth_from_end ?a_stride 1))
                (= ?a_k_stride (nth_from_end ?a_stride 0))

                ; Get B strides
                (= ?b_m_stride (nth_from_end ?b_stride 2))
                (= ?b_n_stride (nth_from_end ?b_stride 1))
                (= ?b_k_stride (nth_from_end ?b_stride 0))

                ; Assert contiguous k stride on output (required for reduction)
                (= ?k_stride (MNum 1))

                ; Assert A has contiguous k (row-major A)
                (= ?a_k_stride (MNum 1))

                ; Assert B has contiguous k (col-major B / transposed)
                (= ?b_k_stride (MNum 1))

                ;(= (F32) (dtype ?a))
                ;(= (F32) (dtype ?b))
            )
            (
                ; Compute tiled dimensions
                (let ?tiled_m (MCeilDiv ?m (MNum {ts})))
                (let ?tiled_n (MCeilDiv ?n (MNum {ts})))

                ; Create batch strides for A (remove k dim, scale m and n by TILE_SIZE)
                (let ?scaled_a_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?a_stride
                            (MMul ?a_n_stride (MNum {ts})) 1)
                        (MMul ?a_m_stride (MNum {ts})) 2))
                (let ?tiled_a_stride (RemoveNthFromEnd ?scaled_a_stride 0))

                ; Create batch strides for B (remove k dim, scale m and n by TILE_SIZE)
                (let ?scaled_b_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?b_stride
                            (MMul ?b_n_stride (MNum {ts})) 1)
                        (MMul ?b_m_stride (MNum {ts})) 2))
                (let ?tiled_b_stride (RemoveNthFromEnd ?scaled_b_stride 0))

                ; Create batch strides for output (scale m and n by TILE_SIZE)
                (let ?tiled_out_stride
                    (ReplaceNthFromEnd
                        (ReplaceNthFromEnd ?sum_out_stride
                            (MMul ?sum_out_n_stride (MNum {ts})) 0)
                        (MMul ?sum_out_m_stride (MNum {ts})) 1))

                (let ?tm (TileMatmulFullSplit
                    (MNum {sm_count})
                    ?out_shape
                    ?tiled_m ?tiled_n ?k
                    ?a ?tiled_a_stride ?a_m_stride (MNum 1)
                    ?b ?tiled_b_stride ?b_n_stride (MNum 1)
                    ?tiled_out_stride ?sum_out_m_stride ?sum_out_n_stride))
                (union ?sum ?tm)
                (set (dtype ?tm) (F32))
            )
            :name \"tile matmul full split\"
        )",
                ts = TILE_SIZE,
                sm_count = 56 // Optimal: balances task count reduction with parallelism
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                sm_count: extract_expr(egraph, children[0], expr_cache).unwrap(),
                untiled_range: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                m_tiles: extract_expr(egraph, children[2], expr_cache).unwrap(),
                n_tiles: extract_expr(egraph, children[3], expr_cache).unwrap(),
                total_k: extract_expr(egraph, children[4], expr_cache).unwrap(),
                a_stride: extract_expr_list(egraph, children[6], list_cache, expr_cache).unwrap(),
                a_m_stride: extract_expr(egraph, children[7], expr_cache).unwrap(),
                a_k_stride: extract_expr(egraph, children[8], expr_cache).unwrap(),
                b_stride: extract_expr_list(egraph, children[10], list_cache, expr_cache).unwrap(),
                b_n_stride: extract_expr(egraph, children[11], expr_cache).unwrap(),
                b_k_stride: extract_expr(egraph, children[12], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[13], list_cache, expr_cache)
                    .unwrap(),
                out_m_stride: extract_expr(egraph, children[14], expr_cache).unwrap(),
                out_n_stride: extract_expr(egraph, children[15], expr_cache).unwrap(),
            })),
            vec![children[5], children[9]],
        )
    }
}

impl BlockOp for TileMatmulFullSplit {
    fn op_name(&self) -> &'static str {
        "TileMatmulFullSplit"
    }

    fn launch_range(&self) -> Vec<Expression> {
        // Launch exactly sm_count work units
        vec![self.sm_count]
    }

    fn output_size(&self) -> Expression {
        self.untiled_range.iter().copied().product::<Expression>()
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        // Each SM processes exclusive output tiles, so barriers are separable
        vec![false]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![false], vec![false]]
    }

    fn bytes_stored(&self) -> Expression {
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        m * n * 4
    }

    fn flops(&self) -> Expression {
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        m * n * k * 2
    }

    fn bytes_loaded(&self) -> Expression {
        let m = self.untiled_range[0];
        let n = self.untiled_range[1];
        let k = self.total_k;
        (m * k + k * n) * 4
    }

    fn cuda_function(&self) -> String {
        format!(
            r#"
        // TileMatmulFullSplit: Optimized for both M=1 decode and general matmul
        const int m_tiles = eval_expression(payload.m_tiles, 0);
        const int n_tiles = eval_expression(payload.n_tiles, 0);
        const int total_k = eval_expression(payload.total_k, 0);
        const int sm_count = eval_expression(payload.sm_count, 0);
        const int M = eval_expression(payload.untiled_range[0], 0);
        const int N = eval_expression(payload.untiled_range[1], 0);

        const float* a_base = source_ptrs[0];
        const float* b_base = source_ptrs[1];
        float* c_base = out_ptr;

        const int a_m_stride = eval_expression(payload.a_m_stride, 0);
        const int b_n_stride = eval_expression(payload.b_n_stride, 0);
        const int c_m_stride = eval_expression(payload.c_m_stride, 0);
        const int c_n_stride = eval_expression(payload.c_n_stride, 0);

        constexpr int TILE_SIZE = {ts};
        const int threads = blockDim.x;
        const int lane = t & 31;
        const int warp_id = t >> 5;
        const int num_warps = threads >> 5;

        auto warp_reduce_sum = [](float val) {{
            for (int offset = 16; offset > 0; offset >>= 1) {{
                val += __shfl_down_sync(0xffffffff, val, offset);
            }}
            return val;
        }};

        // ============== M=1 DECODE PATH (NO K-SPLITTING, NO ATOMICS) ==============
        // For M=1, we split by N columns instead of K. Each SM handles complete dot products.
        if (M == 1) {{
            // Split N columns across SMs
            const int cols_per_sm = (N + sm_count - 1) / sm_count;
            const int col_start = current * cols_per_sm;
            const int col_end = min(col_start + cols_per_sm, N);

            if (col_start >= N) return;

            const float* a = a_base;
            const int K = total_k;

            // Each warp handles 4 columns, threads parallelize over K
            constexpr int COLS_PER_WARP = 4;

            for (int col_base = col_start + warp_id * COLS_PER_WARP; col_base < col_end; col_base += num_warps * COLS_PER_WARP) {{
                float partial[COLS_PER_WARP] = {{0.0f, 0.0f, 0.0f, 0.0f}};

                // Compute base pointers for B columns
                const float* b0 = b_base + col_base * b_n_stride;
                const float* b1 = b_base + (col_base + 1) * b_n_stride;
                const float* b2 = b_base + (col_base + 2) * b_n_stride;
                const float* b3 = b_base + (col_base + 3) * b_n_stride;

                const int valid_cols = min(COLS_PER_WARP, col_end - col_base);

                // Main K loop - unroll by 4 for ILP
                int k = lane;
                for (; k + 96 < K; k += 128) {{
                    float a0 = a[k];
                    float a1 = a[k + 32];
                    float a2 = a[k + 64];
                    float a3 = a[k + 96];

                    if (valid_cols > 0) partial[0] += a0 * b0[k] + a1 * b0[k + 32] + a2 * b0[k + 64] + a3 * b0[k + 96];
                    if (valid_cols > 1) partial[1] += a0 * b1[k] + a1 * b1[k + 32] + a2 * b1[k + 64] + a3 * b1[k + 96];
                    if (valid_cols > 2) partial[2] += a0 * b2[k] + a1 * b2[k + 32] + a2 * b2[k + 64] + a3 * b2[k + 96];
                    if (valid_cols > 3) partial[3] += a0 * b3[k] + a1 * b3[k + 32] + a2 * b3[k + 64] + a3 * b3[k + 96];
                }}

                // Handle remaining K
                for (; k < K; k += 32) {{
                    float a_val = a[k];
                    if (valid_cols > 0) partial[0] += a_val * b0[k];
                    if (valid_cols > 1) partial[1] += a_val * b1[k];
                    if (valid_cols > 2) partial[2] += a_val * b2[k];
                    if (valid_cols > 3) partial[3] += a_val * b3[k];
                }}

                // Warp reduction
                #pragma unroll
                for (int ci = 0; ci < COLS_PER_WARP; ci++) {{
                    partial[ci] = warp_reduce_sum(partial[ci]);
                }}

                // Direct write
                if (lane == 0) {{
                    if (valid_cols > 0) c_base[col_base] = partial[0];
                    if (valid_cols > 1) c_base[col_base + 1] = partial[1];
                    if (valid_cols > 2) c_base[col_base + 2] = partial[2];
                    if (valid_cols > 3) c_base[col_base + 3] = partial[3];
                }}
            }}
            return;
        }}

        // ============== GENERAL PATH (M > 1) ==============
        // Total work units in linearized (m_tiles, n_tiles, k) space
        const int total_work = m_tiles * n_tiles * total_k;
        const int k_chunk_size = (total_work + sm_count - 1) / sm_count;

        const int work_start = current * k_chunk_size;
        const int work_end = min(work_start + k_chunk_size, total_work);

        if (work_start >= total_work) return;

        // Compute which tiles we touch
        const int first_tile = work_start / total_k;
        const int last_tile = (work_end - 1) / total_k;

        // Fast path: single tile
        if (first_tile == last_tile) {{
            const int tile_idx = first_tile;
            const int tile_work_start = tile_idx * total_k;

            const int k_start = work_start - tile_work_start;
            const int k_end = work_end - tile_work_start;
            const int K = k_end - k_start;

            const int n_tile = tile_idx % n_tiles;
            const int m_tile = tile_idx / n_tiles;

            const int global_m0 = m_tile * TILE_SIZE;
            const int global_n0 = n_tile * TILE_SIZE;
            const int tile_m = min(TILE_SIZE, M - global_m0);
            const int tile_n = min(TILE_SIZE, N - global_n0);

            const float* a = a_base + global_m0 * a_m_stride + k_start;
            const float* b = b_base + global_n0 * b_n_stride + k_start;
            float* c = c_base + global_m0 * c_m_stride + global_n0 * c_n_stride;

            const int tile_elems = tile_m * tile_n;
            for (int idx = t; idx < tile_elems; idx += threads) {{
                const int ty = idx / tile_n;
                const int tx = idx % tile_n;
                const float* A0 = a + ty * a_m_stride;
                const float* B0 = b + tx * b_n_stride;
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {{
                    acc += A0[k] * B0[k];
                }}
                // Output buffer is zeroed by runtime before execution, so use atomicAdd
                atomicAdd(&c[ty * c_m_stride + tx * c_n_stride], acc);
            }}
            return;
        }}

        // Slow path: multiple tiles
        for (int tile_idx = first_tile; tile_idx <= last_tile; tile_idx++) {{
            const int tile_work_start = tile_idx * total_k;
            const int tile_work_end = tile_work_start + total_k;

            const int k_start = (work_start > tile_work_start) ? (work_start - tile_work_start) : 0;
            const int k_end = (work_end < tile_work_end) ? (work_end - tile_work_start) : total_k;
            const int K = k_end - k_start;

            const int n_tile = tile_idx % n_tiles;
            const int m_tile = tile_idx / n_tiles;

            const int global_m0 = m_tile * TILE_SIZE;
            const int global_n0 = n_tile * TILE_SIZE;
            const int tile_m = min(TILE_SIZE, M - global_m0);
            const int tile_n = min(TILE_SIZE, N - global_n0);

            const float* a = a_base + global_m0 * a_m_stride + k_start;
            const float* b = b_base + global_n0 * b_n_stride + k_start;
            float* c = c_base + global_m0 * c_m_stride + global_n0 * c_n_stride;

            const int tile_elems = tile_m * tile_n;
            for (int idx = t; idx < tile_elems; idx += threads) {{
                const int ty = idx / tile_n;
                const int tx = idx % tile_n;
                const float* A0 = a + ty * a_m_stride;
                const float* B0 = b + tx * b_n_stride;
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {{
                    acc += A0[k] * B0[k];
                }}
                // Output buffer is zeroed by runtime before execution, so use atomicAdd
                atomicAdd(&c[ty * c_m_stride + tx * c_n_stride], acc);
            }}
        }}
        "#,
            ts = TILE_SIZE
        )
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr_arr("untiled_range", &self.untiled_range)
            .expr("m_tiles", self.m_tiles)
            .expr("n_tiles", self.n_tiles)
            .expr("total_k", self.total_k)
            .expr("sm_count", self.sm_count)
            .expr("a", flatten_mul_strides(&[self.sm_count], &[0.into()]))
            .expr("a_m_stride", self.a_m_stride)
            .expr("a_k_stride", self.a_k_stride)
            .expr("a_width", self.a_m_stride) // a_width = a_m_stride for row-major
            .expr("b", flatten_mul_strides(&[self.sm_count], &[0.into()]))
            .expr("b_n_stride", self.b_n_stride)
            .expr("b_k_stride", self.b_k_stride)
            .expr("b_width", self.b_n_stride) // b_width = b_n_stride for col-major
            .expr("c", flatten_mul_strides(&[self.sm_count], &[0.into()]))
            .expr("c_m_stride", self.out_m_stride)
            .expr("c_n_stride", self.out_n_stride)
            .expr("c_width", self.out_m_stride) // c_width = c_m_stride
    }
}

#[derive(Debug, Default)]
pub struct RowEmbed {
    range: Vec<Expression>, // batch dimensions (e.g., [s] for sequence length)
    token_stride: Vec<Expression>, // stride for token_ids input
    out_stride: Vec<Expression>, // stride for output
    embed_dim: Expression,  // embedding dimension (e.g., HIDDEN)
}

impl EgglogOp for RowEmbed {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "RowEmbed".to_string(),
            vec![EList, Input, EList, Input, EList, Expr],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            // Match Gather with Add(Mul(Cast(token_ids), const), Iota) indices
            "(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?re (RowEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?re)
                    (set (dtype ?re) (F32))
                )
                :name \"row embed with cast mul\"
            )".to_string(),
            // Match Gather with Add(Iota, Mul(Cast(token_ids), const)) indices (reversed order)
            "(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?re (RowEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?re)
                    (set (dtype ?re) (F32))
                )
                :name \"row embed with cast mul reversed\"
            )".to_string(),
            // Match Gather with Add(Mul(token_ids, const), Iota) indices (no Cast)
            "(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?re (RowEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?re)
                    (set (dtype ?re) (F32))
                )
                :name \"row embed with mul\"
            )".to_string(),
            // Match Gather with Add(Iota, Mul(token_ids, const)) indices (reversed order, no Cast)
            "(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                    (= ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (= ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                )
                (
                    (let ?re (RowEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?re)
                    (set (dtype ?re) (F32))
                )
                :name \"row embed with mul reversed\"
            )".to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn BlockOp>(Box::new(Self {
                range: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                token_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                embed_dim: extract_expr(egraph, children[5], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]], // token_ids, embedding_table
        )
    }
}

impl BlockOp for RowEmbed {
    fn op_name(&self) -> &'static str {
        "RowEmbed"
    }

    fn launch_range(&self) -> Vec<Expression> {
        if self.range.is_empty() {
            vec![1.into()]
        } else {
            self.range.clone()
        }
    }

    fn output_size(&self) -> Expression {
        self.range.iter().copied().product::<Expression>().max(1) * self.embed_dim
    }

    fn producer_barriers_seperate(&self) -> Vec<bool> {
        vec![true; self.range.len()]
    }

    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>> {
        vec![vec![true; self.range.len()], vec![true; self.range.len()]]
    }

    fn bytes_loaded(&self) -> Expression {
        // Load: 1 token ID (4 bytes) + 1 embedding row (embed_dim * 4 bytes)
        self.range.iter().copied().product::<Expression>().max(1) * (4 + self.embed_dim * 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store: 1 embedding row per launch
        self.range.iter().copied().product::<Expression>().max(1) * self.embed_dim * 4
    }

    fn flops(&self) -> Expression {
        // No FLOPs - just memory copy
        0.into()
    }

    fn cuda_function(&self) -> String {
        "
        int embed_dim = eval_expression(payload.embed_dim, 0);

        // Get stride offsets
        int token_offset = eval_expression(payload.token_stride, current);
        int out_offset = eval_expression(payload.out_stride, current);

        // Get pointers
        const int* token_ids = (const int*)(source_ptrs[0]) + token_offset;
        const float* embed_table = source_ptrs[1];
        float* out_row = out_ptr + out_offset;

        // Read token ID (stored as int)
        int token_id = token_ids[0];

        // Lookup and copy embedding row
        const float* embed_row = embed_table + (long long)token_id * embed_dim;
        for (int i = t; i < embed_dim; i += blockDim.x) {
            out_row[i] = embed_row[i];
        }
        "
        .to_string()
    }

    fn build_payload<'a>(&self, _: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        payload
            .expr(
                "token_stride",
                flatten_mul_strides(&self.range, &self.token_stride),
            )
            .expr(
                "out_stride",
                flatten_mul_strides(&self.range, &self.out_stride),
            )
            .expr("embed_dim", self.embed_dim)
    }
}
