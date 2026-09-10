use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::index_expr::{IotaExpr, parse_int_expr};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, KernelOp, KernelSource, coord_prelude, lower_expr, metal_type, numel,
};
use anyhow::{Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Iota {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IotaDps {
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
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

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

impl KernelOp for IotaDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let Some(expr) = &self.expr else {
            bail!("iota beyond the parsed expression subset (fail-closed, as the reference)");
        };
        let out_dims = &ctx.dest_dims[0];
        let to = metal_type(ctx.dest_dtypes[0])?;
        let n = numel(out_dims);
        let prelude = coord_prelude(out_dims);
        let value = lower_expr(expr, out_dims.len())?;
        let source = format!(
            r#"kernel void k(device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i >= n) return;
{prelude}    out[i] = ({to})({value});
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

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

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Iota {
            expr: parse_int_expr(site, &site.child_class(0), 64, Some(&site.child_class(1))),
        })
    }
}
