use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    compile_module_image_for_current_device,
    cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasComputeType_t, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, cublasLtMatmulDescCreate,
                cublasLtMatmulDescDestroy, cublasLtMatmulDescSetAttribute,
                cublasLtMatmulHeuristicResult_t, cublasLtMatmulPreference_t,
                cublasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceCreate,
                cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
                cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
                cudaDataType,
            },
        },
        driver::{
            CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
        },
    },
    host::{DeviceBuffer, HostOp},
    try_create_cublaslt,
};

const WORKSPACE_SIZE: usize = 32 * 1024 * 1024; // 32 MiB

/// Fused GLU-MoE HostOp matched via egglog pattern.
///
/// Replaces the expert computation subgraph (expert gathers + matmuls + gated
/// activation + weighted sum) with an efficient cuBLASLt implementation.
///
/// Inputs (graph edges, in order):
///   0: x              [seq, hidden]                        F32
///   1: topk_indices   [seq, k]                             Int
///   2: topk_values    [seq, k]                             F32
///   3: gate_up_w      [E, gate_up_dim, hidden]             BF16
///   4: down_w         [E, hidden, intermediate]             BF16
///   5: mode_aux
///      - SwiGLU: ignored (rewriter wires `topk_values` again)
///      - GemmaGELU: per_expert_scale [E]                   F32
///
/// Output: [seq, hidden] F32
pub struct GLUMoE {
    pub(crate) mode: GLUMoEMode,
    /// Product of gate_up weight dimensions per expert (gate_up_dim * hidden) used for gather stride
    gu_io: Expression,
    /// Product of down weight dimensions per expert (hidden * intermediate) used for gather stride
    dn_io: Expression,
    /// K dimension of gate_up matmul (= hidden)
    gu_matmul_k: Expression,
    /// K dimension of down matmul (= intermediate)
    dn_matmul_k: Expression,
    /// K experts to sum over (= top_k)
    output_k: Expression,
    /// Total elements in a single gate_up expert weight matrix
    gu_within_range: Expression,
    /// Total elements in a single down expert weight matrix
    dn_within_range: Expression,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
    module: OnceLock<(Arc<CudaModule>, CudaFunction, CudaFunction)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GLUMoEMode {
    SwiGLU,
    GemmaGELU,
}

impl GLUMoEMode {
    fn from_mode_id(mode_id: usize) -> Self {
        match mode_id {
            0 => Self::SwiGLU,
            1 => Self::GemmaGELU,
            other => {
                panic!("Unknown GLUMoE mode id: {other}");
            }
        }
    }

    fn activation_kernel_mode(self) -> i32 {
        match self {
            Self::SwiGLU => 0,
            Self::GemmaGELU => 1,
        }
    }
}

impl Default for GLUMoE {
    fn default() -> Self {
        Self {
            mode: GLUMoEMode::SwiGLU,
            gu_io: Expression::default(),
            dn_io: Expression::default(),
            gu_matmul_k: Expression::default(),
            dn_matmul_k: Expression::default(),
            output_k: Expression::default(),
            gu_within_range: Expression::default(),
            dn_within_range: Expression::default(),
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl std::fmt::Debug for GLUMoE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLUMoE")
            .field("mode", &self.mode)
            .field("gu_io", &self.gu_io)
            .field("dn_io", &self.dn_io)
            .field("gu_matmul_k", &self.gu_matmul_k)
            .field("dn_matmul_k", &self.dn_matmul_k)
            .field("output_k", &self.output_k)
            .finish()
    }
}

impl Clone for GLUMoE {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            gu_io: self.gu_io,
            dn_io: self.dn_io,
            gu_matmul_k: self.gu_matmul_k,
            dn_matmul_k: self.dn_matmul_k,
            output_k: self.output_k,
            gu_within_range: self.gu_within_range,
            dn_within_range: self.dn_within_range,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl GLUMoE {
    fn get_cublaslt(&self, stream: &Arc<CudaStream>) -> anyhow::Result<Arc<CudaBlasLT>> {
        if let Some(cublaslt) = self.cublaslt.get() {
            return Ok(cublaslt.clone());
        }
        let created = try_create_cublaslt(stream.clone()).map_err(|message| {
            anyhow::anyhow!("cuBLASLt unavailable on this machine: {message}")
        })?;
        let _ = self.cublaslt.set(created.clone());
        Ok(created)
    }

    fn get_kernels(
        &self,
        stream: &Arc<CudaStream>,
    ) -> &(Arc<CudaModule>, CudaFunction, CudaFunction) {
        self.module.get_or_init(|| {
            let src = r#"
#include <cuda_bf16.h>

extern "C" __global__ void f32_to_bf16(unsigned long long in_ptr, unsigned long long out_ptr, int n) {
    const float* in_ = (const float*)in_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

extern "C" __global__ void glu_activation_bf16(
    unsigned long long gate_up_ptr,
    unsigned long long out_ptr,
    int intermediate,
    int mode
) {
    const __nv_bfloat16* gate_up = (const __nv_bfloat16*)gate_up_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate) {
        float gate = __bfloat162float(gate_up[i]);
        float up   = __bfloat162float(gate_up[i + intermediate]);
        float activated;
        if (mode == 0) {
            activated = gate / (1.0f + expf(-gate));
        } else {
            float scaled = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate);
            activated = gate / (1.0f + expf(-scaled));
        }
        out[i] = __float2bfloat16(activated * up);
    }
}
"#;
            let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let f32_to_bf16 = module.load_function("f32_to_bf16").unwrap();
            let activation = module.load_function("glu_activation_bf16").unwrap();
            (module, f32_to_bf16, activation)
        })
    }
}

impl EgglogOp for GLUMoE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GLUMoE",
            &[
                ("gu_io", EXPRESSION),
                ("dn_io", EXPRESSION),
                ("gu_matmul_k", EXPRESSION),
                ("dn_matmul_k", EXPRESSION),
                ("output_k", EXPRESSION),
                ("gu_within_range", EXPRESSION),
                ("dn_within_range", EXPRESSION),
                ("mode", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (GLUMoE ?gu_io ?dn_io ?gu_matmul_k ?dn_matmul_k ?output_k ?gu_within_range ?dn_within_range ?mode) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["glumoe_rewrite.egg"]),
        ]
    }

    fn n_inputs(&self) -> usize {
        6
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let gu_io = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let dn_io = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let gu_matmul_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let dn_matmul_k = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let output_k = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        let gu_within_range = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let dn_within_range = extract_expr(egraph, kind_children[6], expr_cache).unwrap();
        let mode_expr = extract_expr(egraph, kind_children[7], expr_cache).unwrap();
        let mode_id = mode_expr
            .to_usize()
            .unwrap_or_else(|| panic!("GLUMoE mode must be static, got expression: {mode_expr}"));
        let mode = GLUMoEMode::from_mode_id(mode_id);

        let extracted = GLUMoE {
            mode,
            gu_io,
            dn_io,
            gu_matmul_k,
            dn_matmul_k,
            output_k,
            gu_within_range,
            dn_within_range,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        };

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        // Return the 6 IR inputs: x, topk_idx, topk_values, gate_up_w, down_w, mode_aux
        (op, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for GLUMoE {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 6 {
            anyhow::bail!("GLUMoE expected at least 6 inputs, got {}", inputs.len());
        }

        // Resolve dimensions
        let hidden = self
            .gu_matmul_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE hidden dimension is unresolved"))?;
        let intermediate = self
            .dn_matmul_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE intermediate dimension is unresolved"))?;
        let top_k = self
            .output_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE top-k dimension is unresolved"))?;
        let gu_io = self
            .gu_io
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE gate/up stride is unresolved"))?;
        let dn_io = self
            .dn_io
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE down stride is unresolved"))?;

        if hidden == 0 || intermediate == 0 {
            anyhow::bail!(
                "GLUMoE got zero-sized matmul dimensions: hidden={hidden}, intermediate={intermediate}"
            );
        }
        if top_k == 0 {
            return Ok(());
        }
        if gu_io % hidden != 0 {
            anyhow::bail!("GLUMoE gate/up stride {gu_io} is not divisible by hidden {hidden}");
        }
        if dn_io % intermediate != 0 {
            anyhow::bail!(
                "GLUMoE down stride {dn_io} is not divisible by intermediate {intermediate}"
            );
        }

        let gate_up_dim = gu_io / hidden; // gate_up_dim = 2 * intermediate for GLU
        let down_hidden = dn_io / intermediate;
        if gate_up_dim != intermediate * 2 {
            anyhow::bail!(
                "GLUMoE expected gate/up dim {} to equal 2 * intermediate {}",
                gate_up_dim,
                intermediate * 2
            );
        }
        if down_hidden != hidden {
            anyhow::bail!("GLUMoE down hidden {down_hidden} does not match hidden {hidden}");
        }

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE output byte size is unresolved"))?;
        if output_bytes % (hidden * 4) != 0 {
            anyhow::bail!(
                "GLUMoE output bytes {output_bytes} are not divisible by hidden bytes {}",
                hidden * 4
            );
        }
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("GLUMoE missing {name} buffer for LLIR node {node:?}")
            })
        };

        // Get input/output buffers
        let x_buf = get_buffer("x", inputs[0])?; // [seq, hidden] F32
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?; // [seq, k] Int
        let topk_vals_buf = get_buffer("topk values", inputs[2])?; // [seq, k] F32
        let gate_up_buf = get_buffer("gate/up weights", inputs[3])?; // [E, gate_up_dim, hidden] BF16
        let down_buf = get_buffer("down weights", inputs[4])?; // [E, hidden, intermediate] BF16
        let mode_aux_buf = get_buffer("mode aux", inputs[5])?;
        let output_buf = get_buffer("output", self_node)?; // [seq, hidden] F32

        let topk_bytes = seq * top_k * 4;
        if x_buf.len() < output_bytes {
            anyhow::bail!(
                "GLUMoE x buffer too small: have {} bytes, need {output_bytes}",
                x_buf.len()
            );
        }
        if topk_idx_buf.len() < topk_bytes {
            anyhow::bail!(
                "GLUMoE topk index buffer too small: have {} bytes, need {topk_bytes}",
                topk_idx_buf.len()
            );
        }
        if topk_vals_buf.len() < topk_bytes {
            anyhow::bail!(
                "GLUMoE topk value buffer too small: have {} bytes, need {topk_bytes}",
                topk_vals_buf.len()
            );
        }
        if output_buf.len() < output_bytes {
            anyhow::bail!(
                "GLUMoE output buffer too small: have {} bytes, need {output_bytes}",
                output_buf.len()
            );
        }

        let gu_stride_bytes = gate_up_dim * hidden * 2;
        let down_stride_bytes = hidden * intermediate * 2;
        if gu_stride_bytes == 0 || gate_up_buf.len() % gu_stride_bytes != 0 {
            anyhow::bail!(
                "GLUMoE gate/up weight buffer has {} bytes, not a multiple of per-expert stride {gu_stride_bytes}",
                gate_up_buf.len()
            );
        }
        let num_experts = gate_up_buf.len() / gu_stride_bytes;
        if num_experts == 0 {
            anyhow::bail!("GLUMoE has no expert weights");
        }
        if down_buf.len() < num_experts * down_stride_bytes {
            anyhow::bail!(
                "GLUMoE down weight buffer too small: have {} bytes, need {}",
                down_buf.len(),
                num_experts * down_stride_bytes
            );
        }

        // Get raw device pointer addresses
        let x_ptr = buf_ptr(x_buf, stream);
        let gate_up_ptr = buf_ptr(gate_up_buf, stream);
        let down_ptr = buf_ptr(down_buf, stream);
        let output_ptr = buf_ptr(output_buf, stream);

        let cublaslt = self.get_cublaslt(stream)?;
        let (_, f32_to_bf16_fn, activation_fn) = self.get_kernels(stream);

        // Read top-k routing values from GPU
        let topk_idx_host: Vec<u8> = topk_idx_buf.clone_dtoh(stream)?;
        let topk_idx_i32: &[i32] = bytemuck::cast_slice(&topk_idx_host[..topk_bytes]);
        let topk_vals_host: Vec<u8> = topk_vals_buf.clone_dtoh(stream)?;
        let topk_vals_f32: &[f32] = bytemuck::cast_slice(&topk_vals_host[..topk_bytes]);

        for (pos, &expert_idx) in topk_idx_i32.iter().enumerate() {
            if expert_idx < 0 || expert_idx as usize >= num_experts {
                anyhow::bail!(
                    "GLUMoE expert index {expert_idx} at routing position {pos} out of bounds for {num_experts} experts"
                );
            }
        }

        // Mode-dependent expert weights used for the final reduction:
        // - SwiGLU: direct topk values
        // - GemmaGELU: normalize topk values and scale by per-expert factors
        let mut expert_weights_storage: Vec<f32> = Vec::new();
        let expert_weights_f32: &[f32] = match self.mode {
            GLUMoEMode::SwiGLU => topk_vals_f32,
            GLUMoEMode::GemmaGELU => {
                let per_expert_scale_host: Vec<u8> = mode_aux_buf.clone_dtoh(stream)?;
                let per_expert_scale_bytes = num_experts * 4;
                if per_expert_scale_host.len() < per_expert_scale_bytes {
                    anyhow::bail!(
                        "GLUMoE per-expert scale buffer too small: have {} bytes, need {per_expert_scale_bytes}",
                        per_expert_scale_host.len()
                    );
                }
                let per_expert_scale_f32: &[f32] =
                    bytemuck::cast_slice(&per_expert_scale_host[..per_expert_scale_bytes]);
                expert_weights_storage.resize(seq * top_k, 0.0);
                for t in 0..seq {
                    let base = t * top_k;
                    let vals = &topk_vals_f32[base..base + top_k];
                    let norm = vals.iter().copied().sum::<f32>();
                    let inv_norm = if norm != 0.0 { norm.recip() } else { 0.0 };
                    for i in 0..top_k {
                        let expert_idx = topk_idx_i32[base + i] as usize;
                        if expert_idx >= per_expert_scale_f32.len() {
                            anyhow::bail!(
                                "GLUMoE Gemma mode expert index {} out of bounds {}",
                                expert_idx,
                                per_expert_scale_f32.len()
                            );
                        }
                        let scale = per_expert_scale_f32[expert_idx];
                        expert_weights_storage[base + i] = vals[i] * inv_norm * scale;
                    }
                }
                &expert_weights_storage
            }
        };

        // Allocate temp buffers
        let x_bf16_buf = unsafe { stream.alloc::<u8>(seq * hidden * 2)? }; // BF16
        let gate_up_out_buf = unsafe { stream.alloc::<u8>(gate_up_dim * 2)? }; // BF16 per-token
        let hidden_tmp = unsafe { stream.alloc::<u8>(intermediate * 2)? }; // BF16
        let workspace = unsafe { stream.alloc::<u8>(WORKSPACE_SIZE)? };

        let xbf16_ptr = slice_ptr(&x_bf16_buf, stream);
        let gu_out_ptr = slice_ptr(&gate_up_out_buf, stream);
        let hid_ptr = slice_ptr(&hidden_tmp, stream);
        let ws_ptr = slice_ptr(&workspace, stream);

        // Cast x F32 → BF16
        let n_cast = (seq * hidden) as i32;
        let blocks = (n_cast as u32).div_ceil(256);
        unsafe {
            stream
                .launch_builder(f32_to_bf16_fn)
                .arg(&x_ptr)
                .arg(&xbf16_ptr)
                .arg(&n_cast)
                .launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // Per-token expert computation
        let gu_stride = gu_stride_bytes as u64; // bytes per expert gate_up (BF16)
        let down_stride = down_stride_bytes as u64; // bytes per expert down (BF16)

        for t in 0..seq {
            let x_t_ptr = xbf16_ptr + (t * hidden * 2) as u64; // BF16
            let expert_indices = &topk_idx_i32[t * top_k..(t + 1) * top_k];
            let weights = &expert_weights_f32[t * top_k..(t + 1) * top_k];

            for (i, (&expert_idx, &weight)) in expert_indices.iter().zip(weights.iter()).enumerate()
            {
                let expert_idx = expert_idx as usize;

                // a. Gate+Up matmul (BF16 in, BF16 out)
                let expert_gu_ptr = gate_up_ptr + expert_idx as u64 * gu_stride;
                cublas_matmul(
                    stream,
                    &cublaslt,
                    ws_ptr,
                    gate_up_dim as u64,
                    1,
                    hidden as u64,
                    expert_gu_ptr,
                    cublasOperation_t::CUBLAS_OP_T,
                    hidden as i64,
                    x_t_ptr,
                    cublasOperation_t::CUBLAS_OP_N,
                    hidden as i64,
                    gu_out_ptr,
                    gate_up_dim as i64,
                    cudaDataType::CUDA_R_16BF,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    1.0f32,
                    0.0f32,
                )?;

                // b. Mode-specific gated activation (BF16 → BF16)
                let moe_int = intermediate as i32;
                let activation_mode = self.mode.activation_kernel_mode();
                let activation_blocks = (moe_int as u32).div_ceil(256);
                unsafe {
                    stream
                        .launch_builder(activation_fn)
                        .arg(&gu_out_ptr)
                        .arg(&hid_ptr)
                        .arg(&moe_int)
                        .arg(&activation_mode)
                        .launch(LaunchConfig {
                            grid_dim: (activation_blocks, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })?;
                }

                // c. Down matmul (BF16 in → F32 out) with fused accumulate
                let expert_down_ptr = down_ptr + expert_idx as u64 * down_stride;
                let out_t_ptr = output_ptr + (t * hidden * 4) as u64; // F32

                let beta = if i == 0 { 0.0f32 } else { 1.0f32 };
                cublas_matmul_mixed(
                    stream,
                    &cublaslt,
                    ws_ptr,
                    hidden as u64,
                    1,
                    intermediate as u64,
                    expert_down_ptr,
                    cublasOperation_t::CUBLAS_OP_T,
                    intermediate as i64,
                    hid_ptr,
                    cublasOperation_t::CUBLAS_OP_N,
                    intermediate as i64,
                    out_t_ptr,
                    hidden as i64,
                    weight,
                    beta,
                )?;
            }
        }

        stream.synchronize()?;
        Ok(())
    }

    fn output_size(&self) -> Expression {
        // Output is [seq, hidden] F32 → seq * hidden elements
        // But seq is dynamic. We derive from first input size / hidden.
        // Actually, output_bytes is what matters for allocation:
        Expression::from('s') * self.gu_matmul_k
    }

    fn output_bytes(&self) -> Expression {
        Expression::from('s') * self.gu_matmul_k * 4 // F32
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("GLUMoE")
    }
}

// ============================================================
// Helpers
// ============================================================

fn buf_ptr(buf: DeviceBuffer, _stream: &Arc<CudaStream>) -> u64 {
    buf.ptr()
}

fn slice_ptr(buf: &CudaSlice<u8>, stream: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = buf.device_ptr(stream);
    ptr
}

#[allow(clippy::too_many_arguments)]
fn cublas_matmul(
    stream: &Arc<CudaStream>,
    cublaslt: &Arc<CudaBlasLT>,
    workspace_ptr: u64,
    m: u64,
    n: u64,
    k: u64,
    a_ptr: u64,
    a_op: cublasOperation_t,
    lda: i64,
    b_ptr: u64,
    b_op: cublasOperation_t,
    ldb: i64,
    c_ptr: u64,
    ldc: i64,
    dtype: cudaDataType,
    compute: cublasComputeType_t,
    alpha: f32,
    beta: f32,
) -> anyhow::Result<()> {
    let scale_type = cudaDataType::CUDA_R_32F;

    let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
    let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    unsafe {
        cublasLtMatmulDescCreate(&mut matmul_desc, compute, scale_type).result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &a_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &b_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;

        let (a_rows, a_cols) = if a_op == cublasOperation_t::CUBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_op == cublasOperation_t::CUBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };

        cublasLtMatrixLayoutCreate(&mut a_desc, dtype, a_rows, a_cols, lda).result()?;
        cublasLtMatrixLayoutCreate(&mut b_desc, dtype, b_rows, b_cols, ldb).result()?;
        cublasLtMatrixLayoutCreate(&mut c_desc, dtype, m, n, ldc).result()?;

        cublasLtMatmulPreferenceCreate(&mut preference).result()?;
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        cublasLtMatmulAlgoGetHeuristic(
            *cublaslt.handle(),
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
            return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
        }

        cublasLtMatmul(
            *cublaslt.handle(),
            matmul_desc,
            &alpha as *const _ as *const std::ffi::c_void,
            a_ptr as *const std::ffi::c_void,
            a_desc,
            b_ptr as *const std::ffi::c_void,
            b_desc,
            &beta as *const _ as *const std::ffi::c_void,
            c_ptr as *const std::ffi::c_void,
            c_desc,
            c_ptr as *mut std::ffi::c_void,
            c_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            WORKSPACE_SIZE,
            stream.cu_stream() as *mut _,
        )
        .result()?;

        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cublas_matmul_mixed(
    stream: &Arc<CudaStream>,
    cublaslt: &Arc<CudaBlasLT>,
    workspace_ptr: u64,
    m: u64,
    n: u64,
    k: u64,
    a_ptr: u64,
    a_op: cublasOperation_t,
    lda: i64,
    b_ptr: u64,
    b_op: cublasOperation_t,
    ldb: i64,
    c_ptr: u64,
    ldc: i64,
    alpha: f32,
    beta: f32,
) -> anyhow::Result<()> {
    let ab_dtype = cudaDataType::CUDA_R_16BF;
    let cd_dtype = cudaDataType::CUDA_R_32F;
    let compute = cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let scale_type = cudaDataType::CUDA_R_32F;

    let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut d_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
    let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    unsafe {
        cublasLtMatmulDescCreate(&mut matmul_desc, compute, scale_type).result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &a_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &b_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;

        let (a_rows, a_cols) = if a_op == cublasOperation_t::CUBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_op == cublasOperation_t::CUBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };

        cublasLtMatrixLayoutCreate(&mut a_desc, ab_dtype, a_rows, a_cols, lda).result()?;
        cublasLtMatrixLayoutCreate(&mut b_desc, ab_dtype, b_rows, b_cols, ldb).result()?;
        cublasLtMatrixLayoutCreate(&mut c_desc, cd_dtype, m, n, ldc).result()?;
        cublasLtMatrixLayoutCreate(&mut d_desc, cd_dtype, m, n, ldc).result()?;

        cublasLtMatmulPreferenceCreate(&mut preference).result()?;
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        cublasLtMatmulAlgoGetHeuristic(
            *cublaslt.handle(),
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(d_desc);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
            return Err(anyhow::anyhow!(
                "No suitable cuBLASLT algorithm found for mixed matmul"
            ));
        }

        cublasLtMatmul(
            *cublaslt.handle(),
            matmul_desc,
            &alpha as *const _ as *const std::ffi::c_void,
            a_ptr as *const std::ffi::c_void,
            a_desc,
            b_ptr as *const std::ffi::c_void,
            b_desc,
            &beta as *const _ as *const std::ffi::c_void,
            c_ptr as *const std::ffi::c_void,
            c_desc,
            c_ptr as *mut std::ffi::c_void,
            d_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            WORKSPACE_SIZE,
            stream.cu_stream() as *mut _,
        )
        .result()?;

        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
    }
    Ok(())
}
