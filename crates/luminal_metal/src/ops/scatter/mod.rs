use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelOp, KernelSource, coord_prelude, layout_read_index, metal_type,
    numel, strides_of,
};
use anyhow::{Result, bail};

fn coordinate_rank(site: &ExtractionSite<'_>, child: usize) -> usize {
    let mut rank = 0usize;
    let mut class = site.child_class(child);
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
    rank
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterFunctional {
    pub rank: usize,
}

impl OpSlotNames for ScatterFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "init".to_string(),
            1 => "src".to_string(),
            n if n < 2 + self.rank => format!("coord{}", n - 2),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for ScatterFunctional {
    fn label(&self) -> &str {
        "ScatterFunctionalGeneric"
    }
}

impl Bufferizable for ScatterFunctional {}

impl ToDps for ScatterFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(ScatterFunctionalDps { rank: self.rank }))
    }
}

impl LayoutIrOp for ScatterFunctional {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterFunctionalDps {
    pub rank: usize,
}

impl ScatterFunctionalDps {
    fn dest_index(&self) -> usize {
        self.rank + 2
    }
}

impl OpSlotNames for ScatterFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        if operand == 0 {
            "init".to_string()
        } else if operand == 1 {
            "src".to_string()
        } else if operand < self.dest_index() {
            format!("coord{}", operand - 2)
        } else if operand == self.dest_index() {
            "dest0".to_string()
        } else {
            format!("in{operand}")
        }
    }
}

impl BufferTensorIrOp for ScatterFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::MetalOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "ScatterFunctionalGeneric"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != self.dest_index() // dest0 is write-only; everything else reads
    }
}

impl Bufferizable for ScatterFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: self.dest_index(),
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for ScatterFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for ScatterFunctionalDps {}

impl KernelOp for ScatterFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        let rank = self.rank;
        let init_dims = &ctx.operand_dims[0];
        if init_dims.len() != rank {
            bail!("scatter init rank {} vs op rank {rank}", init_dims.len());
        }
        let t = metal_type(ctx.operand_dtypes[0])?;
        let dest_dims = &ctx.dest_dims[0];
        let dest_n = numel(dest_dims);
        let src_dims = &ctx.operand_dims[1];
        let src_n = numel(src_dims);
        let strides = strides_of(init_dims);
        let mut sig = format!("device const {t}* init, device const {t}* src");
        for axis in 0..rank {
            sig.push_str(&format!(", device const int* coord{axis}"));
        }
        if init_dims != dest_dims {
            bail!(
                "operand init value extents {init_dims:?} differ from dest extents \
             {dest_dims:?} — the scatter copy iterates the dest"
            );
        }
        let (init_chain, init_idx) = layout_read_index(
            "init",
            ctx.operand_layout(0),
            dest_dims,
            Coords::FlatIndex { prefix: "c" },
        )?;
        let copy_src = if init_chain.is_empty() {
            format!(
                r#"kernel void k({sig}, device {t}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {dest_n};
    ulong i = gid;
    if (i < n) out[i] = init[{init_idx}];
}}"#
            )
        } else {
            let prelude = coord_prelude(dest_dims);
            format!(
                r#"kernel void k({sig}, device {t}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {dest_n};
    ulong i = gid;
    if (i >= n) return;
{prelude}{init_chain}    out[i] = init[{init_idx}];
}}"#
            )
        };
        let mut reads = String::new();
        let mut any_chain = false;
        for (axis, stride) in strides.iter().enumerate() {
            if &ctx.operand_dims[axis + 2] != src_dims {
                bail!(
                    "operand coord{axis} value extents {:?} differ from src extents \
                 {src_dims:?} — the scatter write launch \
                 iterates src",
                    ctx.operand_dims[axis + 2]
                );
            }
            let name = format!("coord{axis}");
            let (chain, idx) = layout_read_index(
                &name,
                ctx.operand_layout(axis + 2),
                src_dims,
                Coords::FlatIndex { prefix: "c" },
            )?;
            any_chain |= !chain.is_empty();
            reads.push_str(&chain);
            reads.push_str(&format!("    coord = (long){name}[{idx}];\n"));
            reads.push_str(&format!("    flat += coord * {stride};\n"));
        }
        let (src_chain, src_idx) = layout_read_index(
            "src",
            ctx.operand_layout(1),
            src_dims,
            Coords::FlatIndex { prefix: "c" },
        )?;
        any_chain |= !src_chain.is_empty();
        let mut body = String::new();
        if any_chain {
            body.push_str(&coord_prelude(src_dims));
        }
        body.push_str("    long flat = 0;\n    long coord;\n");
        body.push_str(&reads);
        body.push_str(&src_chain);
        let src_read = format!("src[{src_idx}]");
        let scatter_src = format!(
            r#"kernel void k({sig}, device {t}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {src_n};
    ulong i = gid;
    if (i >= n) return;
{body}    out[flat] = {src_read};
}}"#
        );
        Ok(vec![
            KernelSource::plain(copy_src, dest_n),
            KernelSource::plain(scatter_src, src_n),
        ])
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ScatterFunctionalMatcher;

impl OpMatcher for ScatterFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpScatterFunctionalGeneric"
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
        &[("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ScatterFunctional {
            rank: coordinate_rank(site, 2),
        })
    }
}
