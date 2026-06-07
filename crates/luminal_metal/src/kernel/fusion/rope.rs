use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{Rule, SortDef, sort},
        base::{ELIST, EXPRESSION, F64, IR},
    },
    op::*,
    prelude::*,
    shape::flatten_strides,
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

use crate::kernel::{
    MetalKernelOp,
    ops::{compile_shader, lower_expression_for_metal},
};

#[derive(Debug, Clone, Default)]
/// Replaces the decomposed Llama RoPE graph with a direct split-half rotary embedding kernel.
/// input [seq, hidden] as [seq, n_heads, head_dim], pos [seq] -> out [seq, hidden]
/// [x0, x1] -> [x0 * cos(pos, dim) - x1 * sin(pos, dim),
///              x1 * cos(pos, dim) + x0 * sin(pos, dim)]
pub struct MetalLlamaRope {
    shape: Vec<Expression>,
    n_heads: usize,
    head_dim: usize,
    theta_ln: f32,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalLlamaRope {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalLlamaRope",
            &[
                ("shape", ELIST),
                ("input", IR),
                ("pos", IR),
                ("n_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("theta_ln", F64),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(include_str!("llama_rope.egg"))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};

        let static_usize = |expr: Expression, name: &str| {
            expr.exec(&FxHashMap::default())
                .unwrap_or_else(|| panic!("MetalLlamaRope {name} must be static"))
        };
        let n_heads = static_usize(
            extract_expr(egraph, children[3], expr_cache).unwrap(),
            "n_heads",
        );
        let head_dim = static_usize(
            extract_expr(egraph, children[4], expr_cache).unwrap(),
            "head_dim",
        );
        let theta_ln = egraph.enodes[children[5]]
            .0
            .replace('\"', "")
            .parse::<f32>()
            .expect("MetalLlamaRope theta_ln must be a float");

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                n_heads,
                head_dim,
                theta_ln,
                output_strides: extract_expr_list(egraph, children[6], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[2]],
        )
    }
}

impl MetalKernelOp for MetalLlamaRope {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "LlamaRope"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        if input_dtypes.first().copied().unwrap_or(DType::F32) != DType::F32
            || input_dtypes.get(1).copied().unwrap_or(DType::Int) != DType::Int
            || output_dtype != DType::F32
        {
            return None;
        }

        let hidden = self.n_heads * self.head_dim;
        let head_dim = self.head_dim;
        let half_dim = self.head_dim / 2;
        let theta_ln = self.theta_ln;
        let out_index =
            lower_expression_for_metal(&flatten_strides(&self.shape, &self.output_strides), "idx");
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define HIDDEN {hidden}
            #define HEAD_DIM {head_dim}
            #define HALF_DIM {half_dim}
            #define THETA_LN {theta_ln:.9e}f

            inline float rotate_split_half(
                const device float *x,
                const uint base,
                const uint d,
                const int pos
            ) {{
                const uint pair = d < HALF_DIM ? d : d - HALF_DIM;
                const float exponent = -float(pair * 2u) / float(HEAD_DIM);
                const float angle = float(pos) * exp(THETA_LN * exponent);
                const float c = fast::cos(angle);
                const float s = fast::sin(angle);
                const float x0 = x[base + pair];
                const float x1 = x[base + pair + HALF_DIM];
                return d < HALF_DIM ? x0 * c - x1 * s : x1 * c + x0 * s;
            }}

            kernel void llama_rope_kernel(
                const device float *input [[buffer(0)]],
                const device int *pos [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant int *dyn [[buffer(3)]],
                constant uint &n_elements [[buffer(4)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx >= n_elements) {{
                    return;
                }}

                (void)dyn;
                const uint token = idx / HIDDEN;
                const uint flat = idx - token * HIDDEN;
                const uint head = flat / HEAD_DIM;
                const uint d = flat - head * HEAD_DIM;
                const uint base = token * HIDDEN + head * HEAD_DIM;
                out[{out_index}] = rotate_split_half(input, base, d, pos[token]);
            }}
            "#
        );
        Some(compile_shader(device, &source, "llama_rope_kernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>()
            + self
                .shape
                .first()
                .and_then(|s| s.exec(dyn_map))
                .unwrap_or(0)
                * std::mem::size_of::<i32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * 8
    }
}

/// Replaces the decomposed Llama RoPE graph while it is still materialized as
/// `[heads, seq, head_dim]`. The kernel applies split-half rotary embedding
/// directly in that 3D layout, which lets the KV-cache path avoid building the
/// full gather/sin/cos/pad/add RoPE subtree before scatter.
#[derive(Debug, Clone, Default)]
pub struct MetalLlamaRope3D {
    shape: Vec<Expression>,
    input_strides: Vec<Expression>,
    head_dim: usize,
    theta_ln: f32,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalLlamaRope3D {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalLlamaRope3D",
            &[
                ("shape", ELIST),
                ("input", IR),
                ("pos", IR),
                ("input_strides", ELIST),
                ("n_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("theta_ln", F64),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            r#"
            (rule
                (
                    (= ?rope (MetalLlamaRope3D ?shape ?input ?pos ?input_strides ?nheads ?hdim ?theta_ln ?rope_out_strides))
                    (= ?scatter (Op
                        (MetalScatterNoCopy ?dest_shape ?dest_strides ?index_shape ?index_strides ?src_strides ?out_strides)
                        (ICons ?dest (ICons ?indexes (ICons ?rope (INil))))))
                )
                (
                    (let ?fused (MetalLlamaRope3DScatter
                        ?dest_shape
                        ?dest_strides
                        ?index_shape
                        ?index_strides
                        ?input_strides
                        ?out_strides
                        ?input
                        ?pos
                        ?dest
                        ?indexes
                        ?nheads
                        ?hdim
                        ?theta_ln))
                    (union ?scatter ?fused)
                    (set (dtype ?fused) (F32))
                )
                :ruleset cleanup
                :name "metal-llama-rope-3d-scatter"
            )

            (rule
                (
                    (= ?scatter (Op
                        (MetalScatterNoCopy ?dest_shape ?dest_strides ?index_shape ?index_strides ?src_strides ?out_strides)
                        (ICons ?dest (ICons ?indexes (ICons ?rope (INil))))))
                    (= ?scatter (MetalLlamaRope3DScatter
                        ?dest_shape
                        ?dest_strides
                        ?index_shape
                        ?index_strides
                        ?input_strides
                        ?out_strides
                        ?input
                        ?pos
                        ?dest
                        ?indexes
                        ?nheads
                        ?hdim
                        ?theta_ln))
                )
                ((subsume (Op
                    (MetalScatterNoCopy ?dest_shape ?dest_strides ?index_shape ?index_strides ?src_strides ?out_strides)
                    (ICons ?dest (ICons ?indexes (ICons ?rope (INil)))))))
                :ruleset cleanup
                :name "metal-llama-rope-3d-scatter-subsumes-scatter"
            )
            "#,
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};

        let static_usize = |expr: Expression, name: &str| {
            expr.exec(&FxHashMap::default())
                .unwrap_or_else(|| panic!("MetalLlamaRope3D {name} must be static"))
        };
        let _n_heads = static_usize(
            extract_expr(egraph, children[4], expr_cache).unwrap(),
            "n_heads",
        );
        let head_dim = static_usize(
            extract_expr(egraph, children[5], expr_cache).unwrap(),
            "head_dim",
        );
        let theta_ln = egraph.enodes[children[6]]
            .0
            .replace('\"', "")
            .parse::<f32>()
            .expect("MetalLlamaRope3D theta_ln must be a float");

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                input_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                head_dim,
                theta_ln,
                output_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[2]],
        )
    }
}

/// Fuses 3D RoPE with KV-cache scatter. The output aliases the destination
/// cache buffer, and each thread rotates one `[head, token, dim]` element and
/// writes it to the flat scatter index. This removes both the temporary RoPE
/// tensor and the following `MetalScatterNoCopy` pass for K-cache updates.
#[derive(Debug, Clone, Default)]
pub struct MetalLlamaRope3DScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    input_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
    head_dim: usize,
    theta_ln: f32,
}

impl EgglogOp for MetalLlamaRope3DScatter {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalLlamaRope3DScatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("input_strides", ELIST),
                ("out_strides", ELIST),
                ("input", IR),
                ("pos", IR),
                ("dest", IR),
                ("indexes", IR),
                ("n_heads", EXPRESSION),
                ("head_dim", EXPRESSION),
                ("theta_ln", F64),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};

        let static_usize = |expr: Expression, name: &str| {
            expr.exec(&FxHashMap::default())
                .unwrap_or_else(|| panic!("MetalLlamaRope3DScatter {name} must be static"))
        };
        let _n_heads = static_usize(
            extract_expr(egraph, children[10], expr_cache).unwrap(),
            "n_heads",
        );
        let head_dim = static_usize(
            extract_expr(egraph, children[11], expr_cache).unwrap(),
            "head_dim",
        );
        let theta_ln = egraph.enodes[children[12]]
            .0
            .replace('\"', "")
            .parse::<f32>()
            .expect("MetalLlamaRope3DScatter theta_ln must be a float");

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                dest_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                dest_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, children[3], list_cache, expr_cache)
                    .unwrap(),
                input_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                head_dim,
                theta_ln,
            })),
            vec![children[8], children[9], children[6], children[7]],
        )
    }
}

impl MetalKernelOp for MetalLlamaRope3DScatter {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "LlamaRope3DScatter"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        if input_dtypes.first().copied().unwrap_or(DType::F32) != DType::F32
            || input_dtypes.get(1).copied().unwrap_or(DType::Int) != DType::Int
            || input_dtypes.get(2).copied().unwrap_or(DType::F32) != DType::F32
            || input_dtypes.get(3).copied().unwrap_or(DType::Int) != DType::Int
            || output_dtype != DType::F32
        {
            return None;
        }
        let _ = (&self.dest_strides, &self.output_strides);
        assert_eq!(
            self.input_strides.len(),
            3,
            "MetalLlamaRope3DScatter input must be a 3D view"
        );

        let head_dim = self.head_dim;
        let half_dim = self.head_dim / 2;
        let theta_ln = self.theta_ln;
        let in_head = lower_expression_for_metal(&self.input_strides[0], "head");
        let in_token = lower_expression_for_metal(&self.input_strides[1], "token");
        let in_pair = lower_expression_for_metal(&self.input_strides[2], "pair");
        let in_other = lower_expression_for_metal(&self.input_strides[2], "other");
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define HEAD_DIM {head_dim}
            #define HALF_DIM {half_dim}
            #define THETA_LN {theta_ln:.9e}f

            kernel void llama_rope_3d_scatter_kernel(
                device float *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device float *input [[buffer(2)]],
                const device int *pos [[buffer(3)]],
                constant int *dyn [[buffer(4)]],
                constant uint &n_elements [[buffer(5)]],
                constant uint &kv_dim [[buffer(6)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                (void)dyn;
                if (idx >= n_elements) {{
                    return;
                }}

                const uint token = idx / kv_dim;
                const uint flat = idx - token * kv_dim;
                const uint head = flat / HEAD_DIM;
                const uint d = flat - head * HEAD_DIM;
                const uint pair = d < HALF_DIM ? d : d - HALF_DIM;
                const uint other = pair + HALF_DIM;
                const float exponent = -float(pair * 2u) / float(HEAD_DIM);
                const float angle = float(pos[token]) * exp(THETA_LN * exponent);
                const float c = fast::cos(angle);
                const float s = fast::sin(angle);
                const uint input_idx0 = ({in_head}) + ({in_token}) + ({in_pair});
                const uint input_idx1 = ({in_head}) + ({in_token}) + ({in_other});
                const float x0 = input[input_idx0];
                const float x1 = input[input_idx1];
                const int scatter_idx = indexes[{index_idx}];
                out[scatter_idx] = d < HALF_DIM ? x0 * c - x1 * s : x1 * c + x0 * s;
            }}
            "#,
            head_dim = head_dim,
            half_dim = half_dim,
            theta_ln = theta_ln,
            in_head = in_head,
            in_token = in_token,
            in_pair = in_pair,
            in_other = in_other,
            index_idx = index_idx,
        );
        Some(compile_shader(
            device,
            &source,
            "llama_rope_3d_scatter_kernel",
        ))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;
        let kv_dim = self
            .index_shape
            .last()
            .and_then(|dim| dim.exec(dyn_map))
            .unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(inputs[2]), 0);
        encoder.set_buffer(3, Some(inputs[3]), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &kv_dim as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_elements as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n * (std::mem::size_of::<i32>() + std::mem::size_of::<f32>())
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
}

impl MetalKernelOp for MetalLlamaRope3D {
    #[cfg(feature = "debug")]
    fn label(&self) -> &'static str {
        "LlamaRope3D"
    }

    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        if input_dtypes.first().copied().unwrap_or(DType::F32) != DType::F32
            || input_dtypes.get(1).copied().unwrap_or(DType::Int) != DType::Int
            || output_dtype != DType::F32
        {
            return None;
        }
        assert_eq!(
            self.input_strides.len(),
            3,
            "MetalLlamaRope3D input must be a 3D view"
        );

        let head_dim = self.head_dim;
        let half_dim = self.head_dim / 2;
        let theta_ln = self.theta_ln;
        let in_head = lower_expression_for_metal(&self.input_strides[0], "head");
        let in_token = lower_expression_for_metal(&self.input_strides[1], "token");
        let in_pair = lower_expression_for_metal(&self.input_strides[2], "pair");
        let in_other = lower_expression_for_metal(&self.input_strides[2], "other");
        let out_index =
            lower_expression_for_metal(&flatten_strides(&self.shape, &self.output_strides), "idx");
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define HEAD_DIM {head_dim}
            #define HALF_DIM {half_dim}
            #define THETA_LN {theta_ln:.9e}f

            kernel void llama_rope_3d_kernel(
                const device float *input [[buffer(0)]],
                const device int *pos [[buffer(1)]],
                device float *out [[buffer(2)]],
                constant int *dyn [[buffer(3)]],
                constant uint &n_elements [[buffer(4)]],
                constant uint &seq [[buffer(5)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx >= n_elements) {{
                    return;
                }}

                (void)dyn;
                const uint d = idx % HEAD_DIM;
                const uint token = (idx / HEAD_DIM) % seq;
                const uint head = idx / (seq * HEAD_DIM);
                const uint pair = d < HALF_DIM ? d : d - HALF_DIM;
                const uint other = pair + HALF_DIM;
                const float exponent = -float(pair * 2u) / float(HEAD_DIM);
                const float angle = float(pos[token]) * exp(THETA_LN * exponent);
                const float c = fast::cos(angle);
                const float s = fast::sin(angle);
                const uint input_idx0 = ({in_head}) + ({in_token}) + ({in_pair});
                const uint input_idx1 = ({in_head}) + ({in_token}) + ({in_other});
                const float x0 = input[input_idx0];
                const float x1 = input[input_idx1];
                out[{out_index}] = d < HALF_DIM ? x0 * c - x1 * s : x1 * c + x0 * s;
            }}
            "#,
            head_dim = head_dim,
            half_dim = half_dim,
            theta_ln = theta_ln,
            in_head = in_head,
            in_token = in_token,
            in_pair = in_pair,
            in_other = in_other,
            out_index = out_index,
        );
        Some(compile_shader(device, &source, "llama_rope_3d_kernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;
        let seq = self.shape[1].exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &seq as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>()
            + self.shape.get(1).and_then(|s| s.exec(dyn_map)).unwrap_or(0)
                * std::mem::size_of::<i32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * 8
    }
}
