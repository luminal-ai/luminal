//! FUSED MXFP4 MIXTURE-OF-EXPERTS (the serving landing, 2026-09-10):
//! two host ops that run a top-k routed, clamped-SwiGLU expert block
//! over expert weights that stay PACKED in fp4 (e2m1 nibble pairs with
//! e8m0 block scales, the OCP MX format gpt-oss ships) — the only form
//! in which a 120B-parameter MoE fits a single device.
//!
//! The block is split at its natural barrier: every down-projection
//! dot needs the whole hidden vector of its (token, route) pair, so
//! `gate_up` writes `[s, top_k, inter]` and `down` reads it. Splitting
//! also keeps the intermediate a planner-owned buffer instead of an
//! op-private scratch allocation.
//!
//! Like [`crate::ops::paged_attention`], each op's whole surface — the
//! logical constructor (an [`luminal::graph::LogicalOp::Extern`] term),
//! its propagation rules, the implementation match, and the kernel —
//! lives here and rides the registry row that implements it.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::dtype::PlanDtype;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use anyhow::{Context, Result, bail};

use super::paged_attention::dense_extents;

pub const GATE_UP_LOGICAL_CONSTRUCTOR: &str = "LogicalMoeGateUpMxfp4";
pub const GATE_UP_CONSTRUCTOR: &str = "LayoutTensorOpMoeGateUpMxfp4";
pub const DOWN_LOGICAL_CONSTRUCTOR: &str = "LogicalMoeDownMxfp4";
pub const DOWN_CONSTRUCTOR: &str = "LayoutTensorOpMoeDownMxfp4";

const KERNEL_SOURCE: &str = include_str!("kernel.cu");
const TENSOR_SOURCE: &str = include_str!("tensor_core.cu");
/// Output rows per warp task in the GEMV kernels (gate/up counts row
/// PAIRS), and the resident-blocks-per-SM they are compiled for.
const GEMV_ROWS_GATE_UP: usize = 4;
const GEMV_ROWS_DOWN: usize = 4;
const GEMV_MIN_BLOCKS: usize = 2;
/// The packed dims must tile by the largest row count either kernel uses.
const GEMV_ROWS: usize = 4;
const BLOCK_THREADS: u32 = 256;

/// The GEMV instantiation, read once (`LUMINAL_MOE_GEMV_{R_GU,R_DN,MIN_BLOCKS}`
/// are tuning probes over the defaults).
fn gemv_config() -> (usize, usize, usize) {
    static CACHE: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        (
            env_usize("LUMINAL_MOE_GEMV_R_GU", GEMV_ROWS_GATE_UP),
            env_usize("LUMINAL_MOE_GEMV_R_DN", GEMV_ROWS_DOWN),
            env_usize("LUMINAL_MOE_GEMV_MIN_BLOCKS", GEMV_MIN_BLOCKS),
        )
    })
}

fn gemv_source() -> String {
    let (gu, dn, min_blocks) = gemv_config();
    format!(
        "#define GU_R {gu}\n#define DN_R {dn}\n#define MIN_BLOCKS {min_blocks}\n{KERNEL_SOURCE}"
    )
}
/// The tensor-core kernel's tile: BN output rows per block (five n8
/// tiles per warp), 64-k iterations, 128 threads.
const TENSOR_BN: usize = 160;
const TENSOR_BM: usize = 64;
const TENSOR_BK: usize = 64;
const TENSOR_THREADS: u32 = 128;
/// Resident blocks per SM the kernel is compiled for (its register cap).
const TENSOR_MIN_BLOCKS: usize = 2;
/// cp.async ring depth.
const TENSOR_STAGES: usize = 3;
/// Routes one shared-memory window holds (the kernel's `WINDOW`).
const TENSOR_WINDOW: usize = 4096;
/// Routes (token, k) at or above which a tick takes the expert-major
/// tensor-core kernel; below it the warp-per-route GEMV reads no more
/// weight bytes and has less setup. `LUMINAL_MOE_TENSOR_MIN_PAIRS`
/// overrides (a tuning probe).
const TENSOR_MIN_PAIRS: usize = 128;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The tensor-core kernel's tiling, read once (`LUMINAL_MOE_TENSOR_BN`,
/// `_BM`, `_MIN_BLOCKS`, `_STAGES` are tuning probes over the defaults).
#[derive(Clone, Copy)]
struct TensorTiling {
    bn: usize,
    bm: usize,
    min_blocks: usize,
    stages: usize,
    window: usize,
}

impl TensorTiling {
    /// Dynamic shared memory: the A and B rings, the route window, the
    /// tile's scale rows.
    fn smem_bytes(&self, k: usize) -> usize {
        self.stages * (self.bm * TENSOR_BK * 4 + self.bn * 32)
            + self.window * 4
            + self.bn * (k / 32)
    }
}

fn tensor_tiling() -> TensorTiling {
    static CACHE: std::sync::OnceLock<TensorTiling> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| TensorTiling {
        bn: env_usize("LUMINAL_MOE_TENSOR_BN", TENSOR_BN),
        bm: env_usize("LUMINAL_MOE_TENSOR_BM", TENSOR_BM),
        min_blocks: env_usize("LUMINAL_MOE_TENSOR_MIN_BLOCKS", TENSOR_MIN_BLOCKS),
        stages: env_usize("LUMINAL_MOE_TENSOR_STAGES", TENSOR_STAGES),
        window: env_usize("LUMINAL_MOE_TENSOR_WINDOW", TENSOR_WINDOW),
    })
}

fn tensor_min_pairs() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| env_usize("LUMINAL_MOE_TENSOR_MIN_PAIRS", TENSOR_MIN_PAIRS))
}

/// Whether a call's geometry takes the tensor-core kernel.
fn tensor_path(pairs: usize, n: usize, k: usize, experts: usize) -> bool {
    let tiling = tensor_tiling();
    // The scale slab of a tile ([bn rows, k/32] bytes, contiguous) is
    // staged in 16-byte chunks: every tile's slab must start and end on
    // a 16-byte boundary.
    let slab_aligned =
        (tiling.bn * (k / 32)).is_multiple_of(16) && (n * (k / 32)).is_multiple_of(16);
    pairs >= tensor_min_pairs()
        && n.is_multiple_of(tiling.bn)
        && k.is_multiple_of(TENSOR_BK)
        && slab_aligned
        && experts <= u16::MAX as usize
        && tiling.smem_bytes(k) <= 200 * 1024
}

fn tensor_source(gate_up: bool) -> String {
    let tiling = tensor_tiling();
    format!(
        "#define MODE_GATE_UP {}\n#define BN {}\n#define BM {}\n#define MIN_BLOCKS {}\n#define NSTAGE {}\n#define WINDOW {}\n{}",
        u8::from(gate_up),
        tiling.bn,
        tiling.bm,
        tiling.min_blocks,
        tiling.stages,
        tiling.window,
        TENSOR_SOURCE
    )
}

/// The tensor-core kernel with its dynamic shared memory opted in.
#[cfg(feature = "device")]
fn tensor_function(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    gate_up: bool,
    k: usize,
) -> Result<(cudarc::driver::CudaFunction, u32)> {
    let smem = tensor_tiling().smem_bytes(k) as u32;
    let function = crate::nvrtc_module::kernel_function_keyed(
        stream,
        if gate_up {
            "moe_tensor:gate_up"
        } else {
            "moe_tensor:down"
        },
        "moe_grouped",
        || tensor_source(gate_up),
    )?;
    function
        .set_attribute(
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem as i32,
        )
        .context("opting the MoE tensor-core kernel into its shared memory")?;
    Ok((function, smem))
}

#[cfg(feature = "device")]
fn tensor_grid(n: usize, experts: usize, smem: u32) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: ((n / tensor_tiling().bn) as u32, experts as u32, 1),
        block_dim: (TENSOR_THREADS, 1, 1),
        shared_mem_bytes: smem,
    }
}
/// The down kernel's route-weight register file; must match `MAX_TOP_K` in kernel.cu.
const MAX_TOP_K: usize = 8;

const GATE_UP_OPERANDS: [&str; 5] = ["x", "expert_ids", "blocks", "scales", "bias"];
const DOWN_OPERANDS: [&str; 6] = [
    "hidden",
    "expert_ids",
    "router_logits",
    "blocks",
    "scales",
    "bias",
];

fn operand_name(names: &[&str], dest: usize, operand: usize) -> String {
    if operand == dest {
        "dest0".to_string()
    } else {
        names
            .get(operand)
            .map(|name| name.to_string())
            .unwrap_or_else(|| format!("in{operand}"))
    }
}

pub(crate) fn check_dtype(
    label: &str,
    who: &str,
    slot: &luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>,
    want: PlanDtype,
) -> Result<()> {
    let got = slot.layout.dtype;
    if got != Some(want) {
        bail!("{label}: operand {who} must be {want:?}, got {got:?}");
    }
    Ok(())
}

/// The packed-weight geometry checks shared by both halves: `blocks`
/// `[E, n, k/2]` U8, `scales` `[E, n, k/32]` F8UE8M0, `bias` `[E, n]`
/// Bf16, for the half's own `(n, k)`.
fn check_packed_weights(
    label: &str,
    ctx: &crate::host::HostOpContext<'_>,
    first: usize,
    names: &[&str],
    n: usize,
    k: usize,
) -> Result<usize> {
    if !k.is_multiple_of(32) || !n.is_multiple_of(GEMV_ROWS) {
        bail!("{label}: the packed dims need k % 32 == 0 and n % {GEMV_ROWS} == 0 (n {n}, k {k})");
    }
    let blocks = dense_extents(label, names[first], &ctx.operand_info[first])?;
    let scales = dense_extents(label, names[first + 1], &ctx.operand_info[first + 1])?;
    let bias = dense_extents(label, names[first + 2], &ctx.operand_info[first + 2])?;
    check_dtype(label, names[first], &ctx.operand_info[first], PlanDtype::U8)?;
    check_dtype(
        label,
        names[first + 1],
        &ctx.operand_info[first + 1],
        PlanDtype::F8UE8M0,
    )?;
    check_dtype(
        label,
        names[first + 2],
        &ctx.operand_info[first + 2],
        PlanDtype::Bf16,
    )?;
    let experts = match blocks.as_slice() {
        [e, nn, kk] if *nn == n && *kk == k / 2 => *e,
        other => bail!("{label}: blocks must be [E, {n}, {}], got {other:?}", k / 2),
    };
    if scales != [experts, n, k / 32] {
        bail!(
            "{label}: scales must be [{experts}, {n}, {}], got {scales:?}",
            k / 32
        );
    }
    if bias != [experts, n] {
        bail!("{label}: bias must be [{experts}, {n}], got {bias:?}");
    }
    Ok(experts)
}

fn grid(tasks: usize) -> cudarc::driver::LaunchConfig {
    let warps_per_block = (BLOCK_THREADS / 32) as usize;
    cudarc::driver::LaunchConfig {
        grid_dim: (tasks.div_ceil(warps_per_block).max(1) as u32, 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// gate/up
// ---------------------------------------------------------------------------

/// The gate/up half's recorded metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateUpSpec {
    pub inter: usize,
    pub top_k: usize,
    /// SwiGLU sigmoid slope (gpt-oss: 1.702).
    pub alpha: f64,
    /// Clamp bound on gate (above) and up (both sides) (gpt-oss: 7.0).
    pub limit: f64,
}

/// `MoeGateUpMxfp4(x, expert_ids, blocks, scales, bias) -> hidden`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeGateUp {
    pub spec: GateUpSpec,
}

impl OpSlotNames for MoeGateUp {
    fn operand_name(&self, operand: usize) -> String {
        operand_name(&GATE_UP_OPERANDS, usize::MAX, operand)
    }
}

impl BufferTensorIrOp for MoeGateUp {
    fn label(&self) -> &str {
        "MoeGateUpMxfp4"
    }
}

impl Bufferizable for MoeGateUp {}

impl ToDps for MoeGateUp {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MoeGateUpDps { spec: self.spec }))
    }
}

impl LayoutIrOp for MoeGateUp {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeGateUpDps {
    pub spec: GateUpSpec,
}

const GATE_UP_DEST: usize = GATE_UP_OPERANDS.len();

impl OpSlotNames for MoeGateUpDps {
    fn operand_name(&self, operand: usize) -> String {
        operand_name(&GATE_UP_OPERANDS, GATE_UP_DEST, operand)
    }
}

impl BufferTensorIrOp for MoeGateUpDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::host::<Self>())
    }

    fn label(&self) -> &str {
        "MoeGateUpMxfp4"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != GATE_UP_DEST
    }
}

impl Bufferizable for MoeGateUpDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: GATE_UP_DEST,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MoeGateUpDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MoeGateUpDps {}

impl crate::host::HostOp for MoeGateUpDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::PushKernelArg;
        let label = "MoeGateUpMxfp4";
        let spec = self.spec;
        if ctx.inputs.len() != GATE_UP_OPERANDS.len() {
            bail!(
                "{label}: expected {} operands, got {}",
                GATE_UP_OPERANDS.len(),
                ctx.inputs.len()
            );
        }
        let x = dense_extents(label, "x", &ctx.operand_info[0])?;
        let ids = dense_extents(label, "expert_ids", &ctx.operand_info[1])?;
        let out = dense_extents(label, "hidden", &ctx.result_info[0])?;
        check_dtype(label, "x", &ctx.operand_info[0], PlanDtype::F32)?;
        check_dtype(label, "expert_ids", &ctx.operand_info[1], PlanDtype::Int)?;
        let (s, hidden_dim) = match x.as_slice() {
            [s, h] => (*s, *h),
            other => bail!("{label}: x must be [s, hidden], got {other:?}"),
        };
        if ids != [s, spec.top_k] {
            bail!(
                "{label}: expert_ids must be [{s}, {}], got {ids:?}",
                spec.top_k
            );
        }
        if out != [s, spec.top_k, spec.inter] {
            bail!(
                "{label}: hidden must be [{s}, {}, {}], got {out:?}",
                spec.top_k,
                spec.inter
            );
        }
        let experts =
            check_packed_weights(label, ctx, 2, &GATE_UP_OPERANDS, 2 * spec.inter, hidden_dim)?;
        if s == 0 || spec.top_k == 0 {
            return Ok(());
        }
        let (x_ptr, ids_ptr, blocks_ptr, scales_ptr, bias_ptr) = (
            ctx.inputs[0].ptr,
            ctx.inputs[1].ptr,
            ctx.inputs[2].ptr,
            ctx.inputs[3].ptr,
            ctx.inputs[4].ptr,
        );
        let dest = ctx.dest.ptr;
        let (h, i, tk, sq) = (
            hidden_dim as i32,
            spec.inter as i32,
            spec.top_k as i32,
            s as i32,
        );
        let (alpha, limit) = (spec.alpha as f32, spec.limit as f32);
        if tensor_path(s * spec.top_k, 2 * spec.inter, hidden_dim, experts) {
            let (function, smem) = tensor_function(ctx.stream, true, hidden_dim)
                .with_context(|| format!("{label}: tensor-core kernel"))?;
            let (n, ex) = ((2 * spec.inter) as i32, experts as i32);
            let mut builder = ctx.stream.launch_builder(&function);
            builder
                .arg(&x_ptr)
                .arg(&blocks_ptr)
                .arg(&scales_ptr)
                .arg(&bias_ptr)
                .arg(&ids_ptr)
                .arg(&ids_ptr)
                .arg(&dest)
                .arg(&n)
                .arg(&h)
                .arg(&tk)
                .arg(&sq)
                .arg(&ex)
                .arg(&alpha)
                .arg(&limit);
            unsafe { builder.launch(tensor_grid(2 * spec.inter, experts, smem)) }
                .with_context(|| format!("{label}: tensor-core launch"))?;
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            "moe_gemv",
            "moe_gate_up",
            gemv_source,
        )
        .with_context(|| format!("{label}: kernel"))?;
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&x_ptr)
            .arg(&blocks_ptr)
            .arg(&scales_ptr)
            .arg(&bias_ptr)
            .arg(&ids_ptr)
            .arg(&dest)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&sq)
            .arg(&alpha)
            .arg(&limit);
        unsafe { builder.launch(grid(s * spec.top_k * spec.inter / gemv_config().0)) }
            .with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}

/// Matches `LayoutTensorOpMoeGateUpMxfp4`. Children 0–4 are operands,
/// 5–8 the metadata literals, 9 the out layout. Contributes the logical
/// surface too (see [`crate::ops::paged_attention::PagedAttentionMatcher`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeGateUpMatcher;

impl OpMatcher for MoeGateUpMatcher {
    fn egglog_constructor(&self) -> &'static str {
        GATE_UP_CONSTRUCTOR
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("gate_up_logical.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("gate_up_dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("gate_up_shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("gate_up_forward.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("gate_up_match.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[
            ("inter", 5),
            ("top_k", 6),
            ("alpha", 7),
            ("limit", 8),
            ("out_layout", 9),
        ]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MoeGateUp {
            spec: GateUpSpec {
                inter: usize::try_from(site.child_i64(5)).unwrap_or(0),
                top_k: usize::try_from(site.child_i64(6)).unwrap_or(0),
                alpha: site.child_f64(7),
                limit: site.child_f64(8),
            },
        })
    }
}

pub fn gate_up_prototype() -> MoeGateUp {
    MoeGateUp {
        spec: GateUpSpec {
            inter: 32,
            top_k: 1,
            alpha: 1.0,
            limit: 1.0,
        },
    }
}

// ---------------------------------------------------------------------------
// down
// ---------------------------------------------------------------------------

/// The down half's recorded metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DownSpec {
    pub hidden: usize,
    pub top_k: usize,
}

/// `MoeDownMxfp4(hidden, expert_ids, router_logits, blocks, scales, bias) -> partials`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDown {
    pub spec: DownSpec,
}

impl OpSlotNames for MoeDown {
    fn operand_name(&self, operand: usize) -> String {
        operand_name(&DOWN_OPERANDS, usize::MAX, operand)
    }
}

impl BufferTensorIrOp for MoeDown {
    fn label(&self) -> &str {
        "MoeDownMxfp4"
    }
}

impl Bufferizable for MoeDown {}

impl ToDps for MoeDown {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(MoeDownDps { spec: self.spec }))
    }
}

impl LayoutIrOp for MoeDown {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeDownDps {
    pub spec: DownSpec,
}

const DOWN_DEST: usize = DOWN_OPERANDS.len();

impl OpSlotNames for MoeDownDps {
    fn operand_name(&self, operand: usize) -> String {
        operand_name(&DOWN_OPERANDS, DOWN_DEST, operand)
    }
}

impl BufferTensorIrOp for MoeDownDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::host::<Self>())
    }

    fn label(&self) -> &str {
        "MoeDownMxfp4"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != DOWN_DEST
    }
}

impl Bufferizable for MoeDownDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: DOWN_DEST,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for MoeDownDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for MoeDownDps {}

impl crate::host::HostOp for MoeDownDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::PushKernelArg;
        let label = "MoeDownMxfp4";
        let spec = self.spec;
        if ctx.inputs.len() != DOWN_OPERANDS.len() {
            bail!(
                "{label}: expected {} operands, got {}",
                DOWN_OPERANDS.len(),
                ctx.inputs.len()
            );
        }
        let hidden = dense_extents(label, "hidden", &ctx.operand_info[0])?;
        let ids = dense_extents(label, "expert_ids", &ctx.operand_info[1])?;
        let logits = dense_extents(label, "router_logits", &ctx.operand_info[2])?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        check_dtype(label, "hidden", &ctx.operand_info[0], PlanDtype::F32)?;
        check_dtype(label, "expert_ids", &ctx.operand_info[1], PlanDtype::Int)?;
        check_dtype(label, "router_logits", &ctx.operand_info[2], PlanDtype::F32)?;
        let (s, inter) = match hidden.as_slice() {
            [s, k, inter] if *k == spec.top_k => (*s, *inter),
            other => bail!(
                "{label}: hidden must be [s, {}, inter], got {other:?}",
                spec.top_k
            ),
        };
        if ids != [s, spec.top_k] {
            bail!(
                "{label}: expert_ids must be [{s}, {}], got {ids:?}",
                spec.top_k
            );
        }
        if out != [s, spec.top_k, spec.hidden] {
            bail!(
                "{label}: out must be [{s}, {}, {}], got {out:?}",
                spec.top_k,
                spec.hidden
            );
        }
        let experts = check_packed_weights(label, ctx, 3, &DOWN_OPERANDS, spec.hidden, inter)?;
        if logits != [s, experts] {
            bail!("{label}: router_logits must be [{s}, {experts}], got {logits:?}");
        }
        if spec.top_k > MAX_TOP_K {
            bail!(
                "{label}: top_k {} exceeds the kernel's {MAX_TOP_K}",
                spec.top_k
            );
        }
        if s == 0 {
            return Ok(());
        }
        let (hidden_ptr, ids_ptr, logits_ptr, blocks_ptr, scales_ptr, bias_ptr) = (
            ctx.inputs[0].ptr,
            ctx.inputs[1].ptr,
            ctx.inputs[2].ptr,
            ctx.inputs[3].ptr,
            ctx.inputs[4].ptr,
            ctx.inputs[5].ptr,
        );
        let dest = ctx.dest.ptr;
        let (h, i, tk, sq, ex) = (
            spec.hidden as i32,
            inter as i32,
            spec.top_k as i32,
            s as i32,
            experts as i32,
        );
        if tensor_path(s * spec.top_k, spec.hidden, inter, experts) {
            let (function, smem) = tensor_function(ctx.stream, false, inter)
                .with_context(|| format!("{label}: tensor-core kernel"))?;
            let (alpha, limit) = (0f32, 0f32);
            let mut builder = ctx.stream.launch_builder(&function);
            builder
                .arg(&hidden_ptr)
                .arg(&blocks_ptr)
                .arg(&scales_ptr)
                .arg(&bias_ptr)
                .arg(&ids_ptr)
                .arg(&logits_ptr)
                .arg(&dest)
                .arg(&h)
                .arg(&i)
                .arg(&tk)
                .arg(&sq)
                .arg(&ex)
                .arg(&alpha)
                .arg(&limit);
            unsafe { builder.launch(tensor_grid(spec.hidden, experts, smem)) }
                .with_context(|| format!("{label}: tensor-core launch"))?;
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function_keyed(
            ctx.stream,
            "moe_gemv",
            "moe_down",
            gemv_source,
        )
        .with_context(|| format!("{label}: kernel"))?;
        let mut builder = ctx.stream.launch_builder(&function);
        builder
            .arg(&blocks_ptr)
            .arg(&scales_ptr)
            .arg(&bias_ptr)
            .arg(&ids_ptr)
            .arg(&logits_ptr)
            .arg(&hidden_ptr)
            .arg(&dest)
            .arg(&h)
            .arg(&i)
            .arg(&tk)
            .arg(&sq)
            .arg(&ex);
        unsafe { builder.launch(grid(s * spec.top_k * spec.hidden / gemv_config().1)) }
            .with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}

/// Matches `LayoutTensorOpMoeDownMxfp4`. Children 0–5 are operands, 6–7
/// the metadata literals, 8 the out layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct MoeDownMatcher;

impl OpMatcher for MoeDownMatcher {
    fn egglog_constructor(&self) -> &'static str {
        DOWN_CONSTRUCTOR
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("down_logical.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("down_dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("down_shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("down_forward.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("down_match.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("hidden", 6), ("top_k", 7), ("out_layout", 8)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(MoeDown {
            spec: DownSpec {
                hidden: usize::try_from(site.child_i64(6)).unwrap_or(0),
                top_k: usize::try_from(site.child_i64(7)).unwrap_or(0),
            },
        })
    }
}

pub fn down_prototype() -> MoeDown {
    MoeDown {
        spec: DownSpec {
            hidden: 32,
            top_k: 1,
        },
    }
}
