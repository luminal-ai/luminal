use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::index_expr::IotaExpr;
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelOp, KernelSource, coord_prelude, layout_read_index, lower_expr,
    metal_type, numel,
};
use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterialize {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterializeDps {
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
        Some(crate::MetalOpInterface::kernel::<Self>())
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

impl KernelOp for IndexMapApplyMaterializeDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
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
        let t = metal_type(ctx.operand_dtypes[0])?;
        let to = metal_type(ctx.dest_dtypes[0])?;
        let n = numel(out_dims);
        let prelude = coord_prelude(out_dims);
        let mut body = String::from("    long idx;\n");
        for (k, entry) in entries.iter().enumerate() {
            let value = lower_expr(entry, out_dims.len())?;
            body.push_str(&format!(
                "    idx = {value};\n    long parent_c{k} = idx;\n"
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
            r#"kernel void k(device const {t}* parent, device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i >= n) return;
{prelude}{body}    out[i] = parent[{pidx}];
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

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

