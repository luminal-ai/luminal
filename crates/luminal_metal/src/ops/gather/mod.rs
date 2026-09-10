use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelOp, KernelSource, coord_prelude, layout_read_index, metal_type, numel,
};
use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gather {
    pub rank: usize,
}

impl OpSlotNames for Gather {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "data".to_string()
        } else if operand <= self.rank {
            format!("coord{}", operand - 1)
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for Gather {
    fn label(&self) -> &str {
        "GatherGeneric"
    }
}

impl Bufferizable for Gather {}

impl ToDps for Gather {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(GatherDps { rank: self.rank }))
    }
}

impl LayoutIrOp for Gather {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatherDps {
    pub rank: usize,
}

impl GatherDps {
    fn dest_index(&self) -> usize {
        self.rank + 1
    }
}

impl OpSlotNames for GatherDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "data".to_string()
        } else if operand <= self.rank {
            format!("coord{}", operand - 1)
        } else if operand == self.dest_index() {
            "dest0".to_string()
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for GatherDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "GatherGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index() // dest0 is write-only; everything else reads
    }
}

impl Bufferizable for GatherDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: self.dest_index(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for GatherDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for GatherDps {}

impl KernelOp for GatherDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let rank = self.rank;
        let data_dims = &ctx.operand_dims[0];
        if data_dims.len() != rank {
            bail!("gather data rank {} vs op rank {rank}", data_dims.len());
        }
        let t = metal_type(ctx.operand_dtypes[0])?;
        let to = metal_type(ctx.dest_dtypes[0])?;
        let out_dims = &ctx.dest_dims[0];
        let n = numel(out_dims);
        let mut sig = format!("device const {t}* data");
        for axis in 0..rank {
            sig.push_str(&format!(", device const int* coord{axis}"));
        }
        let mut any_coord_chain = false;
        let mut coord_reads = String::new();
        for axis in 0..rank {
            if &ctx.operand_dims[axis + 1] != out_dims {
                bail!(
                    "operand coord{axis} value extents {:?} differ from dest extents {:?} \
                 — the gather iterates the dest",
                    ctx.operand_dims[axis + 1],
                    out_dims
                );
            }
            let name = format!("coord{axis}");
            let layout = ctx.operand_layout(axis + 1);
            let (chain, idx) =
                layout_read_index(&name, layout, out_dims, Coords::FlatIndex { prefix: "c" })?;
            any_coord_chain |= !chain.is_empty();
            coord_reads.push_str(&chain);
            coord_reads.push_str(&format!("    coord = (long){name}[{idx}];\n"));
            coord_reads.push_str(&format!("    long data_c{axis} = coord;\n"));
        }
        let mut body = String::new();
        if any_coord_chain {
            body.push_str(&coord_prelude(out_dims));
        }
        body.push_str("    long coord;\n");
        body.push_str(&coord_reads);
        let (chain, idx) = layout_read_index(
            "data",
            ctx.operand_layout(0),
            data_dims,
            Coords::Bound { prefix: "data_c" },
        )?;
        body.push_str(&chain);
        let read = format!("data[{idx}]");
        let source = format!(
            r#"kernel void k({sig}, device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i >= n) return;
{body}    out[i] = {read};
}}"#
        );
        Ok(vec![KernelSource::plain(source, n)])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GatherMatcher;

impl OpMatcher for GatherMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpGatherGeneric"
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
        &[("out_layout", 2)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        let mut rank = 0usize;
        let mut class = site.child_class(1);
        loop {
            let spine = site
                .nodes_in_class_value_any(&class, &["LayoutTensorCons", "LayoutTensorNil"])
                .next()
                .unwrap_or_else(|| {
                    panic!(
                        "schema drift: coordinate-list class {class} under enode {} has no \
                         LayoutTensorCons/LayoutTensorNil constructor",
                        site.node_id
                    )
                });
            if spine.op == "LayoutTensorNil" {
                break;
            }
            rank += 1;
            let tail_id = spine.children.get(1).unwrap_or_else(|| {
                panic!("schema drift: a LayoutTensorCons in class {class} has no tail child")
            });
            class = site
                .class_of_child(spine, 1)
                .unwrap_or_else(|| panic!("dangling list tail node {tail_id}"));
        }
        Box::new(Gather { rank })
    }
}
