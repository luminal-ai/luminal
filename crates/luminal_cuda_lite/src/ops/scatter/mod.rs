//! Coordinate-form scatter (R8 dual):
//! `out[coord_0[c], .., coord_{r-1}[c]] = src[c]`, everywhere else
//! `out = init` — CUDA-lite's OWN op (ruling 2026-08-17: every runtime
//! owns its executable ops; the shared crate supplies only the IR
//! traits). Same egglog constructor and label as the reference
//! runtime's scatter — assemblies are per-runtime, labels are IR
//! identity — but the structs, matcher, snippets, and codegen all live
//! here. Functional form only (CL-1 is out-of-place; the mutating
//! family arrives with CL-4). Variable-arity like gather: `rank` =
//! init's rank = the coordinate count, walked out of the e-graph.
//! Operand order: init, src, coord0..coord{r-1} — fixed slots first,
//! the variable tail last.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{
    CodegenCtx, Coords, KernelSource, coord_prelude, cuda_type, layout_read_index, numel,
    strides_of,
};
use anyhow::{Result, bail};

/// Walk the LayoutTensorCons spine at `child` counting elements — the
/// rank reader for the scatter matcher (same class-resolving walk as
/// gather's; see the OpMatcher validity contract for the panics).
fn coordinate_rank(site: &ExtractionSite<'_>, child: usize) -> usize {
    let mut rank = 0usize;
    let mut class = site.child_class(child);
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
    rank
}

/// `ScatterFunctionalGeneric(init, src, coord0, .., coord{r-1}) -> out`
/// — pure dataflow form (init supplies the unwritten regions).
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

/// Destination-passing form: `Scatter(init: read, src: read,
/// coord0..: read, dest0: write ↔ out0)` — the destination is the
/// trailing operand at index `rank + 2`.
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

/// The CUDA lowering, colocated with its op. The READ-side operands
/// (init, src, coordinates) may arrive with a layout whose read does
/// not simplify to the identity; each such operand then
/// lowers into that operand's read index via
/// [`crate::kernels::layout_read_index`]. The WRITE side (dest0)
/// is no longer fenced here (see the write-fence record in
/// [`crate::kernels::CodegenCtx::from_descriptors`]), and the injectivity
/// flags are untouched: the write address arithmetic is identical.
pub(crate) fn codegen(op: &dyn BufferTensorIrOp, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
    let Some(scatter) = op.as_any().downcast_ref::<ScatterFunctionalDps>() else {
        bail!("scatter codegen reached with a non-Scatter op");
    };
    let rank = scatter.rank;
    // The dest operand slot is not fenced — see the write-fence record in
    // `kernels::CodegenCtx::from_descriptors`.
    let init_dims = &ctx.operand_dims[0];
    if init_dims.len() != rank {
        bail!("scatter init rank {} vs op rank {rank}", init_dims.len());
    }
    let t = cuda_type(ctx.operand_dtypes[0])?;
    let dest_dims = &ctx.dest_dims[0];
    let dest_n = numel(dest_dims);
    let src_dims = &ctx.operand_dims[1];
    let src_n = numel(src_dims);
    let strides = strides_of(init_dims);
    // Every launch in the sequence shares the op's full signature so
    // the executor pushes one uniform argument list.
    let mut sig = format!("const {t}* init, const {t}* src");
    for axis in 0..rank {
        sig.push_str(&format!(", const int* coord{axis}"));
    }
    // Launch 1: dest = copy(init), over dest numel. The init is read
    // through its own layout at the DEST coordinates.
    //
    // The init value spans the dest space by construction — asked
    // unconditionally, not only of a folded init (asking it inside the
    // folded arm is how the elementwise coherence check became
    // spelling-dependent).
    if init_dims != dest_dims {
        bail!(
            "operand init value extents {init_dims:?} differ from dest extents \
             {dest_dims:?} — the scatter copy iterates the dest"
        );
    }
    // The copy's `i` IS the dest coordinates decomposed, so a dense init
    // simplifies back to `init[i]` and emits no prelude at all.
    let (init_chain, init_idx) = layout_read_index(
        "init",
        ctx.operand_layout(0),
        dest_dims,
        Coords::FlatIndex { prefix: "c" },
    )?;
    let copy_src = if init_chain.is_empty() {
        format!(
            r#"extern "C" __global__ void k({sig}, {t}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = init[{init_idx}];
}}"#
        )
    } else {
        let prelude = coord_prelude(dest_dims);
        format!(
            r#"extern "C" __global__ void k({sig}, {t}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{init_chain}    out[i] = init[{init_idx}];
}}"#
        )
    };
    // Launch 2: scattered writes over src numel. Folded src/coordinate
    // operands read through their chains at the SRC coordinates (the
    // launch's iteration space); the WRITE address stays the flat
    // coordinate-built one.
    //
    // UNCHECKED SCATTER (ruling 2026-08-31, see the NO RUNTIME BOUNDS
    // TRAPS note in `crate::kernels`). Two checks used to live in this
    // launch: each coordinate against the destination axis extent, and
    // injectivity via `atomicExch(&flags[flat],1u) != 0u` over a zeroed
    // `flags` scratch buffer, which caught two source elements landing
    // on one destination element. Both are gone, and with them the
    // `flags` buffer: the injectivity trap was its ONLY reader, so the
    // dest-sized `unsigned int` scratch allocation and its zeroing were
    // pure cost once the check went. Consequently a scatter with
    // duplicate coordinates now races (last writer wins, nondeterminis-
    // tically) instead of faulting, and an out-of-range coordinate
    // writes out of bounds.
    // Every read in this launch is evaluated at the SRC coordinates,
    // which ARE this launch's `i` decomposed — so all of them are
    // `Coords::FlatIndex` and a dense operand simplifies back to
    // `name[i]`. The WRITE address `flat` is coordinate-built from the
    // scattered coordinate VALUES: that is the op's semantics, not a
    // layout read, and it is unaffected by any of this.
    let mut reads = String::new();
    let mut any_chain = false;
    for (axis, stride) in strides.iter().enumerate() {
        // The coordinate value's own extents must be src's for the
        // prelude's `c*` to be its coordinates: refuse a mismatch,
        // never reinterpret (the elementwise contract). Asked of EVERY
        // coordinate operand, whatever its layout spells.
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
        reads.push_str(&format!("    coord = (long long){name}[{idx}];\n"));
        reads.push_str(&format!("    flat += coord * {stride}LL;\n"));
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
    body.push_str("    long long flat = 0;\n    long long coord;\n");
    body.push_str(&reads);
    body.push_str(&src_chain);
    let src_read = format!("src[{src_idx}]");
    let scatter_src = format!(
        r#"extern "C" __global__ void k({sig}, {t}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{body}    out[flat] = {src_read};
}}"#
    );
    Ok(vec![
        KernelSource::plain(copy_src, dest_n),
        KernelSource::plain(scatter_src, src_n),
    ])
}

/// Matches `LayoutTensorOpScatterFunctionalGeneric` and produces this
/// runtime's [`ScatterFunctional`]. Metadata children: `out_layout` at
/// child 3 (children 0-2 — init, src, the coordinate list — are
/// OPERANDS). `rank` is the coordinate list's length, walked out of
/// the e-graph.
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
