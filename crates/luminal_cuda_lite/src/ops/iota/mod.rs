//! The index-expression generator (a zero-input source writing
//! `expr(c0..ck)` at every output coordinate) — CUDA-lite's OWN op
//! (ruling 2026-08-17: every runtime owns its executable ops; the
//! shared crate supplies only the IR traits). Same egglog constructor
//! and label as the reference runtime's iota — assemblies are
//! per-runtime, labels are IR identity — but the structs, matcher,
//! snippets, and codegen all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::index_expr::{IotaExpr, parse_int_expr};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelSource, coord_prelude, cuda_type, lower_expr, numel};
use anyhow::{Result, bail};

/// `IotaGeneric() -> out` — pure dataflow source form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iota {
    /// The window expression, numerically — `None` for forms beyond the
    /// parsed subset (extraction stays infallible; codegen refuses
    /// loudly instead).
    pub expr: Option<IotaExpr>,
}

impl OpSlotNames for Iota {}

impl BufferTensorIrOp for Iota {
    fn label(&self) -> &str {
        "IotaGeneric"
    }
}

impl Bufferizable for Iota {}

impl ToDps for Iota {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IotaDps {
            expr: self.expr.clone(),
        }))
    }
}

impl LayoutIrOp for Iota {}

/// Destination-passing form: `Iota(dest0: write ↔ out0)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IotaDps {
    /// See [`Iota::expr`].
    pub expr: Option<IotaExpr>,
}

impl OpSlotNames for IotaDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IotaDps {
    fn label(&self) -> &str {
        "IotaGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 0 // dest0 is write-only; there are no other operands
    }
}

impl Bufferizable for IotaDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IotaDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for IotaDps {}

/// The CUDA lowering, colocated with its op.
pub(crate) fn codegen(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(iota) = op.as_any().downcast_ref::<IotaDps>() else {
        bail!("iota codegen reached with a non-Iota op");
    };
    // Iota has a dest-only signature: its operand slots ARE the DPS
    // dest slots, so the check that used to stand here was the write
    // fence — see the record in `kernels::CodegenCtx::from_descriptors`.
    let Some(expr) = &iota.expr else {
        bail!("iota beyond the parsed expression subset (fail-closed, as the reference)");
    };
    let out_dims = &ctx.dest_dims[0];
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let n = numel(out_dims);
    let prelude = coord_prelude(out_dims);
    let value = lower_expr(expr, out_dims.len())?;
    let source = format!(
        r#"extern "C" __global__ void k({to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}    out[i] = ({to})({value});
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Matches `LayoutTensorOpIotaGeneric` and produces this runtime's
/// [`Iota`]. Metadata children: `expr` at child 0, `shape` at child 1,
/// `out_layout` at child 2 — the whole constructor is metadata; there
/// are no tensor operands.
#[derive(Debug, Clone, Copy, Default)]
pub struct IotaMatcher;

impl OpMatcher for IotaMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIotaGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("expr", 0), ("shape", 1), ("out_layout", 2)]
    }

    /// Pure structure — the bounds story lives in egglog (user ruling
    /// 2026-07-23): the iota-int32-certified gate on the op match makes
    /// a missing-bounds iota unimplementable (fail-open), so an enode
    /// reaching this matcher is certified by construction.
    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Iota {
            expr: parse_int_expr(site, &site.child_class(0), 64, Some(&site.child_class(1))),
        })
    }
}
