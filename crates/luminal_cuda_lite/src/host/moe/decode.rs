//! Launchers for decode.cu: the full MoE block as two stream-ordered
//! launches. Writes `out` f32 [seq, hidden]; needs one scratch buffer,
//! `hidden` f32 [seq*top_k, inter].

use std::sync::{Arc, OnceLock};

use crate::{
    compile_module_image_for_current_device,
    cudarc::driver::{CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg},
};

const SOURCE: &str = include_str!("decode.cu");
const BLOCK_THREADS: u32 = 256;

struct DecodeKernel {
    _module: Arc<CudaModule>,
    /// The two phase kernels (R=4 row-blocked).
    phase1: CudaFunction,
    phase2: CudaFunction,
}

// Process-wide cache (NOT per-instance: ops are cloned during GA profiling).
static KERNEL: OnceLock<DecodeKernel> = OnceLock::new();

fn kernel(stream: &Arc<CudaStream>) -> &'static DecodeKernel {
    KERNEL.get_or_init(|| {
        let image = compile_module_image_for_current_device(stream.context(), SOURCE)
            .expect("moe decode kernel should compile");
        let module = stream
            .context()
            .load_module(image)
            .expect("moe decode module should load");
        let phase1 = module
            .load_function("moe_phase1_r4")
            .expect("moe_phase1_r4 should exist");
        let phase2 = module
            .load_function("moe_phase2_r4")
            .expect("moe_phase2_r4 should exist");
        DecodeKernel {
            _module: module,
            phase1,
            phase2,
        }
    })
}

/// Force the NVRTC compile (idempotent); called at extract time so the
/// cost lands outside timed profiling trials.
pub fn warm(stream: &Arc<CudaStream>) {
    let _ = kernel(stream);
}

/// Outputs per warp; must match the kernels' template instantiation.
const GEMV_ROWS: usize = 4;

/// The MoE block for `seq` tokens as two stream-ordered launches. All
/// pointers are device addresses; see decode.cu for layouts. Dims must be
/// multiples of 32.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_decode(
    stream: &Arc<CudaStream>,
    x_ptr: u64,
    gu_q_ptr: u64,
    gu_scale_ptr: u64,
    gu_bias_ptr: u64,
    dn_q_ptr: u64,
    dn_scale_ptr: u64,
    dn_bias_ptr: u64,
    topk_ids_ptr: u64,
    topk_w_ptr: u64,
    hidden_scratch_ptr: u64,
    out_ptr: u64,
    hidden_dim: usize,
    inter: usize,
    top_k: usize,
    seq: usize,
    idx_row_stride: usize,
    alpha: f32,
    limit: f32,
) -> anyhow::Result<()> {
    // %32 (the e8m0 group width) also gives the 16B rows uint4 loads need.
    anyhow::ensure!(
        hidden_dim.is_multiple_of(32) && inter.is_multiple_of(32),
        "decode GEMV requires 32-aligned dims (e8m0 group width): hidden={hidden_dim}, inter={inter}"
    );
    anyhow::ensure!(idx_row_stride >= top_k, "idx_row_stride must be >= top_k");
    if seq == 0 || top_k == 0 {
        return Ok(());
    }

    let rows = GEMV_ROWS; // dims % 32 == 0 guarantees % GEMV_ROWS == 0
    let k = kernel(stream);
    let (p1, p2) = (&k.phase1, &k.phase2);
    // One warp per task, unbounded grid; stream order is the phase barrier.
    let warps_per_block = (BLOCK_THREADS / 32) as usize;
    let grid = |tasks: usize| LaunchConfig {
        grid_dim: (tasks.div_ceil(warps_per_block).max(1) as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let (h, i, tk, s, stride) = (
        hidden_dim as i32,
        inter as i32,
        top_k as i32,
        seq as i32,
        idx_row_stride as i32,
    );
    unsafe {
        stream
            .launch_builder(p1)
            .arg(&x_ptr)
            .arg(&gu_q_ptr)
            .arg(&gu_scale_ptr)
            .arg(&gu_bias_ptr)
            .arg(&topk_ids_ptr)
            .arg(&hidden_scratch_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&s)
            .arg(&stride)
            .arg(&alpha)
            .arg(&limit)
            .launch(grid(seq * top_k * inter / rows))?;
        stream
            .launch_builder(p2)
            .arg(&dn_q_ptr)
            .arg(&dn_scale_ptr)
            .arg(&dn_bias_ptr)
            .arg(&topk_ids_ptr)
            .arg(&topk_w_ptr)
            .arg(&hidden_scratch_ptr)
            .arg(&out_ptr)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&s)
            .arg(&stride)
            .launch(grid(seq * hidden_dim / rows))?;
    }
    Ok(())
}
