//! Materialize a view: apply an index map to the input and write the
//! gathered elements densely — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies
//! only the IR traits). Same egglog constructor and label as the
//! reference runtime's materialize — assemblies are per-runtime,
//! labels are IR identity — but the structs, matcher, snippets,
//! codegen, and the map-entry parser all live here.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::index_expr::IotaExpr;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelOp, KernelSource, coord_prelude, cuda_type, layout_read_index,
    lower_expr, numel,
};
use anyhow::{Result, bail};

/// `IndexMapApplyMaterialize(input) -> out` — pure dataflow form.
/// Note the label: the egglog name has no `Generic` suffix, so neither
/// does the op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterialize {
    /// The index map, numerically — one full expression tree per PARENT
    /// axis (outermost inward), evaluated at the OUT coordinates.
    /// `None` = entries beyond the parsed expression subset: extraction
    /// stays infallible, and codegen refuses loudly instead.
    pub entries: Option<Vec<IotaExpr>>,
}

impl OpSlotNames for IndexMapApplyMaterialize {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyMaterialize {
    fn label(&self) -> &str {
        "IndexMapApplyMaterialize"
    }
}

impl Bufferizable for IndexMapApplyMaterialize {}

impl ToDps for IndexMapApplyMaterialize {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(IndexMapApplyMaterializeDps {
            entries: self.entries.clone(),
        }))
    }
}

impl LayoutIrOp for IndexMapApplyMaterialize {}

/// Destination-passing form:
/// `IndexMapApplyMaterialize(input: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexMapApplyMaterializeDps {
    /// See [`IndexMapApplyMaterialize::entries`].
    pub entries: Option<Vec<IotaExpr>>,
}

impl OpSlotNames for IndexMapApplyMaterializeDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyMaterializeDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel_wire::<Self>())
    }

    fn label(&self) -> &str {
        "IndexMapApplyMaterialize"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for IndexMapApplyMaterializeDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 1,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IndexMapApplyMaterializeDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for IndexMapApplyMaterializeDps {}

/// The CUDA lowering, colocated with its op. A layout whose read does
/// not simplify to the identity, on
/// operand 0, is a view folded onto the materialize's input — the op's
/// own map application produces the input VALUE's coordinates, and the
/// slot's carried layout then reads ON TOP of them (via
/// [`crate::kernels::layout_read_index`]) down to the residence
/// actually read. The WRITE side (dest0) is no longer fenced here — see
/// the write-fence record in
/// [`crate::kernels::CodegenCtx::from_descriptors`].
impl KernelOp for IndexMapApplyMaterializeDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        // The dest operand slot is not fenced — see the write-fence record in
        // `kernels::CodegenCtx::from_descriptors`.
        let Some(entries) = &self.entries else {
            bail!("index map beyond the parsed expression subset (fail-closed, as the reference)");
        };
        let parent_dims = &ctx.operand_dims[0];
        let out_dims = &ctx.operand_dims[1];
        if entries.len() != parent_dims.len() {
            bail!(
                "index map arity {} vs parent rank {}",
                entries.len(),
                parent_dims.len()
            );
        }
        let t = cuda_type(ctx.operand_dtypes[0])?;
        let to = cuda_type(ctx.dest_dtypes[0])?;
        let n = numel(out_dims);
        let prelude = coord_prelude(out_dims);
        // ONE body: the op's map lands on the input VALUE's coordinates
        // (`parent_c*`), then the slot's carried layout carries them to the
        // residence. Those coordinates are MAP OUTPUTS, not `i` decomposed,
        // so the parent read is `Coords::Bound` and never simplifies to `i`
        // — for a dense parent it is the row-major sum over `parent_c*`, the
        // same address the hand-written `pflat` accumulator used to compute.
        //
        // Each mapped coordinate was once checked against the parent extent
        // here; no longer — see the NO RUNTIME BOUNDS TRAPS note in
        // `crate::kernels`.
        let mut body = String::from("    long long idx;\n");
        for (k, entry) in entries.iter().enumerate() {
            let value = lower_expr(entry, out_dims.len())?;
            body.push_str(&format!(
                "    idx = {value};\n    long long parent_c{k} = idx;\n"
            ));
        }
        let (chain, pidx) = layout_read_index(
            "parent",
            ctx.operand_layout(0),
            parent_dims,
            Coords::Bound { prefix: "parent_c" },
        )?;
        body.push_str(&chain);
        let source = format!(
            r#"extern "C" __global__ void k(const {t}* parent, {to}* out, const long long* params) {{
    const unsigned long long n = {n};
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{body}    out[i] = parent[{pidx}];
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

/// Matches `LayoutTensorOpIndexMapApplyMaterialize` and produces this
/// runtime's [`IndexMapApplyMaterialize`]. Metadata children:
/// `index_map` at child 1, `shape` at child 2, `out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyMaterializeMatcher;

impl OpMatcher for IndexMapApplyMaterializeMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyMaterialize"
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
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyMaterialize {
            entries: luminal::index_expr::parse_index_map_entries(site, 1, 2),
        })
    }
}

