//! Coordinate-form scatter (R8 dual):
//! `out[coord_0[c], .., coord_{r-1}[c]] = src[c]` for `c` over src's shape;
//! everywhere else `out = init`. One Int coordinate tensor per INIT axis,
//! all sharing SRC's shape. Out-of-bounds coordinate VALUES are undefined
//! behavior (user ruling 2026-07-22). Duplicate coordinates: the FUNCTIONAL
//! reference kernel (reference::kernels) deterministically detects the
//! conflicting writes and panics (ruling 2026-08-06, superseding the
//! earlier duplicate-UB ruling for the reference path); the mutating form
//! has NO reference kernel — the reference runtime is out-of-place only.
//!
//! Operand 0 is named `init` — it provides the result's initial contents
//! (MLIR's term) — NOT `dest`, deliberately: the DPS rewrite appends a
//! trailing `dest0` destination to the functional form, and two slots both
//! named "dest" in one plan would be unreadable. In-place intent is a
//! BOUNDARY statement (result and init sharing one BufferId); whether it
//! executes in place is decided by which implementation extraction picks:
//! the functional form (fresh result) or the mutating form (writes into
//! init's storage, Must-tied, matched only when the output layout IS
//! init's).
//!
//! Like gather, scatter is variable-arity and its instances carry data:
//! `rank` = init's rank = the coordinate count, read out of the e-graph by
//! the matchers. Operand order: init, src, coord0..coord{r-1} — fixed
//! slots first, the variable tail last.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// Walk the LayoutTensorCons spine at `child` counting elements — the
/// shared rank reader for both scatter matchers (same class-resolving walk
/// as gather's; see the OpMatcher validity contract for the panics).
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
///
/// Functional form: pure dataflow — every operand read (init supplies the
/// unwritten regions), the result freshly allocated.
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

/// Destination-passing form of [`ScatterFunctional`]:
///
/// ```text
/// ScatterFunctionalGeneric(init: read, src: read, coord0..: read, dest0: write-only ↔ out0) -> out0
/// ```
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
        "ScatterFunctionalGeneric" // DPS forms keep the IR name
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
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for ScatterFunctionalDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpScatterFunctionalGeneric` enodes and produces
/// [`ScatterFunctional`] instances. Metadata children: `out_layout` at
/// child 3 (children 0-2 — init, src, the coordinate list — are OPERANDS,
/// discovered through the `LayoutTensorOpLit` union partner). `rank` is
/// the coordinate list's length, walked out of the e-graph.
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

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::kernels::{coordinate_columns, expect_op};
use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

/// The CHECKED scatter kernel (ruling 2026-08-06): dest starts as a copy
/// of init, then dest[coords(i)] = src[i] over the src iteration space.
/// Out-of-bounds coordinates are UB by ruling — surfaced LOUDLY. Two src
/// elements landing on one destination element is a runtime panic, not
/// last-write-wins: write conflicts mean the coordinate tensors are not
/// injective and the program's meaning is coordinate-order-dependent,
/// which we refuse to silently define. (User-provided coordinate data may
/// eventually carry an asserted-injective bit instead; until then the
/// reference checks.)
pub(crate) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<ScatterFunctionalDps>(op)?;
    let rank = op.rank;
    let init_dims = ctx.operand_dims[0].clone();
    anyhow::ensure!(
        init_dims.len() == rank,
        "scatter kernel: init rank {} vs op rank {rank}",
        init_dims.len()
    );
    let mut strides = vec![1usize; rank];
    for k in (0..rank.saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * init_dims[k + 1];
    }
    let coord_operands = coordinate_columns(&ctx.operands[2..2 + rank])?;
    let src_len = ctx.operands[1].len();
    let dest_len = ctx.dests[0].len();
    let mut target_of = vec![0usize; src_len];
    let mut written = vec![false; dest_len];
    for (i, target) in target_of.iter_mut().enumerate() {
        let mut flat = 0usize;
        for axis in 0..rank {
            let coord = coord_operands[axis][i];
            anyhow::ensure!(
                coord >= 0 && (coord as usize) < init_dims[axis],
                "scatter coordinate {coord} out of bounds for axis {axis} (extent {}) — \
                 UB per ruling, surfaced loudly",
                init_dims[axis]
            );
            flat += coord as usize * strides[axis];
        }
        anyhow::ensure!(
            !written[flat],
            "conflicting scatter writes: src element {i} targets destination element \
             {flat}, which an earlier src element already wrote — coordinates must be \
             injective (checked-scatter ruling 2026-08-06)"
        );
        written[flat] = true;
        *target = flat;
    }
    // Payload move: dest = copy(init); dest[target[i]] = src[i].
    match (&ctx.operands[0], &ctx.operands[1], &mut ctx.dests[0]) {
        (TypedBuffer::F32(init), TypedBuffer::F32(src), TypedBuffer::F32(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I32(init), TypedBuffer::I32(src), TypedBuffer::I32(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I64(init), TypedBuffer::I64(src), TypedBuffer::I64(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I8(init), TypedBuffer::I8(src), TypedBuffer::I8(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::U8(init), TypedBuffer::U8(src), TypedBuffer::U8(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::I16(init), TypedBuffer::I16(src), TypedBuffer::I16(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::Bool8(init), TypedBuffer::Bool8(src), TypedBuffer::Bool8(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (TypedBuffer::F8E4M3(init), TypedBuffer::F8E4M3(src), TypedBuffer::F8E4M3(dest)) => {
            dest.copy_from_slice(init);
            for (i, target) in target_of.iter().enumerate() {
                dest[*target] = src[i];
            }
        }
        (init, src, dest) => anyhow::bail!(
            "scatter payload dtypes disagree: init {}, src {}, dest {}",
            init.type_name(),
            src.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}
