//! Coordinate-form gather (R8):
//! `out[c] = data[coord_0[c], .., coord_{r-1}[c]]` — CUDA-lite's OWN
//! op (ruling 2026-08-17: every runtime owns its executable ops; the
//! shared crate supplies only the IR traits). Same egglog constructor
//! and label as the reference runtime's gather — assemblies are
//! per-runtime, labels are IR identity — but the structs, matcher,
//! snippets, and codegen all live here. Variable-arity: `rank` is the
//! DATA tensor's rank = the coordinate operand count, walked out of
//! the e-graph by the matcher and baked into the instance.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelSource, coord_prelude, cuda_type, layout_read_index, numel,
};
use anyhow::{Result, bail};

/// `GatherGeneric(data, coord0, .., coord{r-1}) -> out` — pure
/// dataflow form; total operands = 1 + rank.
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

/// Destination-passing form: `Gather(data: read, coord0..: read,
/// dest0: write ↔ out0)` — the destination is the trailing operand at
/// index `rank + 1`.
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

/// The CUDA lowering, colocated with its op. Every READ operand (data
/// AND coordinates) may arrive with a layout whose read does not
/// simplify to the identity; the slot's own carried layout then lowers
/// into that operand's read index exactly as
/// [`crate::kernels::layout_read_index`] does for the elementwise
/// templates. The WRITE side (dest0) is no longer fenced here — see the
/// write-fence record in [`crate::kernels::CodegenCtx::from_descriptors`].
pub(crate) fn codegen(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(gather) = op.as_any().downcast_ref::<GatherDps>() else {
        bail!("gather codegen reached with a non-Gather op");
    };
    let rank = gather.rank;
    // The dest operand slot is not fenced — see the write-fence record in
    // `kernels::CodegenCtx::from_descriptors`.
    let data_dims = &ctx.operand_dims[0];
    if data_dims.len() != rank {
        bail!("gather data rank {} vs op rank {rank}", data_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
    let mut sig = format!("const {t}* data");
    for axis in 0..rank {
        sig.push_str(&format!(", const int* coord{axis}"));
    }
    // ONE body. A COORDINATE operand's value spans the out iteration
    // space, so its read is evaluated at the OUT coordinates — which ARE
    // `i` decomposed, hence `Coords::FlatIndex`, and a dense coordinate
    // operand's expression simplifies straight back to `coord{k}[i]`.
    //
    // The DATA operand's value coordinates are the gathered coordinate
    // VALUES: bound as `data_c{axis}` and read at THOSE, so they are
    // `Coords::Bound` and never simplify to `i` (the gather's own
    // indirection composes on top of the layout's chain). For a dense
    // data layout the emitted expression is the row-major sum over
    // `data_c*` — the same address the hand-written `flat` accumulator
    // used to compute, now stated once by the layout itself.
    let mut any_coord_chain = false;
    let mut coord_reads = String::new();
    for axis in 0..rank {
        // The coordinate value's own extents must be the out extents for
        // `c*` to be its coordinates: refuse a mismatch, never
        // reinterpret (the elementwise contract). Asked of EVERY
        // coordinate operand, whatever its layout spells.
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
        coord_reads.push_str(&format!("    coord = (long long){name}[{idx}];\n"));
        // The gather once checked each coordinate against the data
        // VALUE's extents here — a DATA-derived check, the coordinate
        // coming from an index buffer. It is gone: an out-of-range index
        // is UB at this layer (see the NO RUNTIME BOUNDS TRAPS note in
        // `crate::kernels`).
        coord_reads.push_str(&format!("    long long data_c{axis} = coord;\n"));
    }
    let mut body = String::new();
    if any_coord_chain {
        // Some coordinate read referenced a coordinate, so the prelude
        // that binds them is live. Dead-code elimination, not a branch.
        body.push_str(&coord_prelude(out_dims));
    }
    body.push_str("    long long coord;\n");
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
        r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    out[i] = {read};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Matches `LayoutTensorOpGatherGeneric` and produces this runtime's
/// [`Gather`]. Metadata children: `out_layout` at child 2 (children 0
/// and 1 — the data layout tensor and the coordinate layout-tensor
/// list — are OPERANDS). The instance's `rank` is the coordinate
/// list's length, walked out of the serialized e-graph here.
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
        // Walk the LayoutTensorCons spine at child 1 counting elements. Each
        // hop resolves BY E-CLASS: a list class can also hold non-structural
        // nodes (functions whose output is the list), and the serializer's
        // chosen child node may be one of those — so every step searches the
        // class for its cons/nil CONSTRUCTOR. A class with neither is schema
        // drift and panics (see the OpMatcher validity contract).
        //
        // NO validity checking happens here (user ruling 2026-07-23):
        // coordinate-shape agreement is a POSITIVE PREMISE of the egglog
        // rules — a gather whose coordinate shapes were never proven equal
        // derives no shape, matches no op, and never reaches this matcher.
        // Extraction reads structure; it does not re-litigate validity.
        let mut rank = 0usize;
        let mut class = site.child_class(1);
        loop {
            let spine = site
                .egraph
                .nodes
                .values()
                .find(|node| {
                    node.eclass == class
                        && (node.op == "LayoutTensorCons" || node.op == "LayoutTensorNil")
                })
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
                .egraph
                .nodes
                .get(tail_id)
                .unwrap_or_else(|| panic!("dangling list tail node {tail_id}"))
                .eclass
                .clone();
        }
        Box::new(Gather { rank })
    }
}
