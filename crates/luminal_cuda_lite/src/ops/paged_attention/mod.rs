//! FUSED PAGED ATTENTION (the serving landing, 2026-09-10): one host op
//! for the attention step of a slot-paged decoder — GQA over a
//! `[slots, kv_heads*head_dim]` cache addressed through a per-request
//! slot table, causal + cross-request isolation from CSR indptrs, an
//! optional sliding window, and gpt-oss style attention SINKS.
//!
//! WHY A FUSED OP AND NOT A SPELLING. The decomposed spelling (gather
//! the context rows, expand heads, matmul, mask, softmax, matmul)
//! materializes `[heads, s, c]` scores over the WHOLE batched context
//! and cannot be run at serving context lengths on out-of-place
//! elementwise kernels; a serving runtime needs the attention step to
//! read only the rows each request owns. This op is that step, spelled
//! at the LOGICAL level (`LogicalPagedAttention`, an
//! [`luminal::graph::LogicalOp::Extern`] term whose egglog surface this
//! module contributes through its matcher) so the model names it and
//! nothing pattern-matches a fragile chain.
//!
//! THE OP IS RUNTIME-OWNED, like every CL op: the logical constructor,
//! its dtype/shape/forward-layout rules, the implementation constructor
//! and match, the structs, and the CUDA kernel all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use anyhow::{Context, Result, bail};

/// The logical constructor this op implements.
pub const LOGICAL_CONSTRUCTOR: &str = "LogicalPagedAttention";
/// The implementation constructor.
pub const CONSTRUCTOR: &str = "LayoutTensorOpPagedAttention";

const KERNEL_SOURCE: &str = include_str!("kernel.cu");
const OPERANDS: [&str; 8] = [
    "q",
    "k_cache",
    "v_cache",
    "slot_table",
    "qo_indptr",
    "kv_indptr",
    "q_pos",
    "sinks",
];

/// The attention geometry the op was recorded with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttentionSpec {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Sliding window in tokens; 0 = full causal attention.
    pub window: usize,
    /// Score scale (typically `1/sqrt(head_dim)`).
    pub scale: f64,
}

impl PagedAttentionSpec {
    fn group(&self) -> usize {
        self.heads / self.kv_heads
    }
}

/// `PagedAttention(q, k_cache, v_cache, slot_table, qo_indptr,
/// kv_indptr, q_pos, sinks) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttention {
    pub spec: PagedAttentionSpec,
}

impl OpSlotNames for PagedAttention {
    fn operand_name(&self, operand: usize) -> String {
        OPERANDS
            .get(operand)
            .map(|name| name.to_string())
            .unwrap_or_else(|| format!("in{operand}"))
    }
}

impl BufferTensorIrOp for PagedAttention {
    fn label(&self) -> &str {
        "PagedAttention"
    }
}

impl Bufferizable for PagedAttention {}

impl ToDps for PagedAttention {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(PagedAttentionDps { spec: self.spec }))
    }
}

impl LayoutIrOp for PagedAttention {}

/// Destination-passing form: the eight reads, then `dest0: write ↔ out0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PagedAttentionDps {
    pub spec: PagedAttentionSpec,
}

const DEST: usize = OPERANDS.len();

impl OpSlotNames for PagedAttentionDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == DEST {
            "dest0".to_string()
        } else {
            OPERANDS
                .get(operand)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("in{operand}"))
        }
    }
}

impl BufferTensorIrOp for PagedAttentionDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::host::<Self>())
    }

    fn label(&self) -> &str {
        "PagedAttention"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != DEST
    }
}

impl Bufferizable for PagedAttentionDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: DEST,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for PagedAttentionDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for PagedAttentionDps {}

/// The literal, right-major extents of a descriptor, or a refusal
/// naming the operand: the host kernels read dense storage only.
pub(crate) fn dense_extents(
    label: &str,
    who: &str,
    slot: &luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>,
) -> Result<Vec<usize>> {
    use luminal::layouts::RightMajorContiguousElementLayout as RM;
    if !slot.layout.has::<RM>() {
        bail!(
            "{label}: operand {who} is not right-major contiguous (class holds {:?})",
            slot.layout.present()
        );
    }
    slot.layout
        .literal_extents()
        .ok_or_else(|| anyhow::anyhow!("{label}: operand {who} has symbolic extents"))
}

/// The kernel source with the geometry baked in as macros.
fn source_for(spec: &PagedAttentionSpec) -> String {
    format!(
        "#define D {}\n#define G {}\n#define W {}\n{}",
        spec.head_dim,
        spec.group(),
        spec.window,
        KERNEL_SOURCE
    )
}

impl crate::host::HostOp for PagedAttentionDps {
    #[cfg(feature = "device")]
    unsafe fn execute(&self, ctx: &crate::host::HostOpContext<'_>) -> Result<()> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let label = "PagedAttention";
        let spec = self.spec;
        if spec.kv_heads == 0 || spec.heads % spec.kv_heads != 0 {
            bail!("{label}: heads {} must be a multiple of kv_heads {}", spec.heads, spec.kv_heads);
        }
        if spec.head_dim == 0 || spec.head_dim % 32 != 0 {
            bail!("{label}: head_dim {} must be a multiple of 32", spec.head_dim);
        }
        if ctx.inputs.len() != OPERANDS.len() {
            bail!("{label}: expected {} operands, got {}", OPERANDS.len(), ctx.inputs.len());
        }
        let extents = |k: usize| dense_extents(label, OPERANDS[k], &ctx.operand_info[k]);
        let q = extents(0)?;
        let kc = extents(1)?;
        let vc = extents(2)?;
        let st = extents(3)?;
        let qo = extents(4)?;
        let kv = extents(5)?;
        let qp = extents(6)?;
        let sk = extents(7)?;
        let out = dense_extents(label, "out", &ctx.result_info[0])?;
        let q_dim = spec.heads * spec.head_dim;
        let kv_dim = spec.kv_heads * spec.head_dim;
        let s = match q.as_slice() {
            [s, width] if *width == q_dim => *s,
            other => bail!("{label}: q must be [s, {q_dim}], got {other:?}"),
        };
        if out != q {
            bail!("{label}: out {out:?} must match q {q:?}");
        }
        for (name, dims) in [("k_cache", &kc), ("v_cache", &vc)] {
            match dims.as_slice() {
                [_, width] if *width == kv_dim => {}
                other => bail!("{label}: {name} must be [slots, {kv_dim}], got {other:?}"),
            }
        }
        if kc != vc {
            bail!("{label}: k_cache {kc:?} and v_cache {vc:?} differ");
        }
        if st.len() != 1 {
            bail!("{label}: slot_table must be rank 1, got {st:?}");
        }
        let request_rows = match (qo.as_slice(), kv.as_slice()) {
            ([r], [r2]) if r == r2 && *r >= 2 => *r,
            other => bail!("{label}: qo_indptr/kv_indptr must be equal rank-1 with >= 2 rows, got {other:?}"),
        };
        if qp != [s] {
            bail!("{label}: q_pos must be [{s}], got {qp:?}");
        }
        if sk != [spec.heads] {
            bail!("{label}: sinks must be [{}], got {sk:?}", spec.heads);
        }
        for (k, dtype) in [
            (0, luminal::dtype::PlanDtype::F32),
            (1, luminal::dtype::PlanDtype::F32),
            (2, luminal::dtype::PlanDtype::F32),
            (3, luminal::dtype::PlanDtype::Int),
            (4, luminal::dtype::PlanDtype::Int),
            (5, luminal::dtype::PlanDtype::Int),
            (6, luminal::dtype::PlanDtype::Int),
            (7, luminal::dtype::PlanDtype::F32),
        ] {
            let got = ctx.operand_info[k].layout.dtype;
            if got != Some(dtype) {
                bail!("{label}: operand {} must be {dtype:?}, got {got:?}", OPERANDS[k]);
            }
        }
        if s == 0 {
            return Ok(());
        }
        let function = crate::nvrtc_module::kernel_function(
            ctx.stream,
            &source_for(&spec),
            "paged_attention_f32",
        )
        .with_context(|| format!("{label}: kernel"))?;
        let ptrs: Vec<u64> = ctx.inputs.iter().map(|b| b.ptr).collect();
        let dest = ctx.dest.ptr;
        let kv_heads = spec.kv_heads as i32;
        let rows = request_rows as i32;
        let scale = spec.scale as f32;
        let cfg = LaunchConfig {
            grid_dim: ((s * spec.kv_heads) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = ctx.stream.launch_builder(&function);
        for ptr in &ptrs {
            builder.arg(ptr);
        }
        builder.arg(&dest);
        builder.arg(&kv_heads);
        builder.arg(&rows);
        builder.arg(&scale);
        unsafe { builder.launch(cfg) }.with_context(|| format!("{label}: launch"))?;
        Ok(())
    }
}

/// Matches `LayoutTensorOpPagedAttention` and produces [`PagedAttention`].
/// Children 0–7 are the operands; 8–12 the geometry literals; 13 the
/// out layout. This matcher also CONTRIBUTES THE LOGICAL SURFACE of the
/// op (constructor, dtype, shape, forward layout) — an extern logical op
/// has no core module, so its rules ride the runtime row that
/// implements it.
#[derive(Debug, Clone, Copy, Default)]
pub struct PagedAttentionMatcher;

impl OpMatcher for PagedAttentionMatcher {
    fn egglog_constructor(&self) -> &'static str {
        CONSTRUCTOR
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};
        vec![
            EgglogSnippet {
                category: SpliceCategory::LogicalConstructors,
                text: include_str!("logical_constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: include_str!("logical_dtype.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Shape,
                text: include_str!("logical_shape.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Forward,
                text: include_str!("logical_forward.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_constructor.egg"),
            },
            EgglogSnippet {
                category: SpliceCategory::Match,
                text: include_str!("match.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[
            ("heads", 8),
            ("kv_heads", 9),
            ("head_dim", 10),
            ("window", 11),
            ("scale", 12),
            ("out_layout", 13),
        ]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let int = |index: usize| usize::try_from(site.child_i64(index)).unwrap_or(0);
        Box::new(PagedAttention {
            spec: PagedAttentionSpec {
                heads: int(8),
                kv_heads: int(9),
                head_dim: int(10),
                window: int(11),
                scale: site.child_f64(12),
            },
        })
    }
}

/// The prototype row for the registry: geometry is irrelevant to the
/// claim (the effect predicates are metadata-independent).
pub fn prototype() -> PagedAttention {
    PagedAttention {
        spec: PagedAttentionSpec {
            heads: 1,
            kv_heads: 1,
            head_dim: 32,
            window: 0,
            scale: 1.0,
        },
    }
}
