//! Materialize a view: apply an index map to the input and write the gathered
//! elements densely (the index map and output shape are op metadata, not
//! operands).

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

/// `IndexMapApplyMaterialize(input) -> out`
///
/// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
/// (every operand read, the result freshly allocated). Note the label: the
/// egglog name has no `Generic` suffix, so neither does the op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterialize {
    /// The index map, numerically — one full expression tree per PARENT
    /// axis (outermost inward), evaluated at the OUT coordinates. `None` =
    /// entries beyond the parsed expression subset: extraction stays
    /// infallible, and the reference KERNEL refuses loudly instead.
    pub entries: Option<Vec<super::iota::IotaExpr>>,
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

/// Destination-passing form of [`IndexMapApplyMaterialize`], signature spelled
/// slot by slot:
///
/// ```text
/// IndexMapApplyMaterialize(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyMaterializeDps {
    /// See [`IndexMapApplyMaterialize::entries`].
    pub entries: Option<Vec<super::iota::IotaExpr>>,
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
    fn label(&self) -> &str {
        "IndexMapApplyMaterialize" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        match operand {
            0 => true,  // input
            1 => false, // dest0: write-only destination
            _ => true,  // outside the signature: conservative default
        }
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
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for IndexMapApplyMaterializeDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpIndexMapApplyMaterialize` enodes and produces
/// [`IndexMapApplyMaterialize`] instances. Metadata children: `index_map` at child 1, `shape` at child 2, `out_layout` at child 3.
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyMaterializeMatcher;

impl OpMatcher for IndexMapApplyMaterializeMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyMaterialize"
    }

    fn snippets(&self) -> Vec<crate::egglog_snippet::EgglogSnippet> {
        vec![
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            crate::egglog_snippet::EgglogSnippet {
                category: crate::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyMaterialize {
            entries: parse_map_entries(site),
        })
    }
}

/// Walk the matched term's index-map metadata (child 1) into numeric
/// entries: `IndexMapLit` → cons spine BY E-CLASS → per element a CoordVar
/// (owner Shape at child 0, axis primitive at child 1) or an IntLit. Anything else (arithmetic
/// entries) yields `None` — the kernel's loud refusal carries the burden,
/// Parse the map's entries numerically. EXISTENTIAL AT EVERY LEVEL (the
/// R8/backtracking doctrine): a saturated map class holds several
/// IndexMapLit spellings and a list class several cons spellings — some
/// reference function-value classes with no constructor members (their
/// values resolve elsewhere), so the parser tries every spelling and
/// takes the first that parses all the way down. All spellings of a
/// class denote the same map, so any parseable one is correct. `None` =
/// no spelling parses — the kernel's loud refusal carries the burden,
/// extraction never fails.
fn parse_map_entries(site: &ExtractionSite<'_>) -> Option<Vec<super::iota::IotaExpr>> {
    let map_class = site.child_class(1);
    // Owner-shape guard: map entries are functions of the op's OUT
    // coordinates (shape metadata at child 2) — see parse_int_expr.
    let out_shape = site.child_class(2);
    let mut memo = std::collections::HashMap::new();
    for map_node in site.nodes_in_class_value(&map_class, "IndexMapLit") {
        let Some(head) = site.class_of_child(map_node, 0) else {
            continue;
        };
        if let Some(entries) = parse_entry_list(site, &head, 64, &out_shape, &mut memo) {
            return Some(entries);
        }
    }
    None
}

fn parse_entry_list(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    out_shape: &egraph_serialize::ClassId,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, super::iota::ParseMemo>,
) -> Option<Vec<super::iota::IotaExpr>> {
    if depth == 0 {
        return None;
    }
    if site
        .nodes_in_class_value(class, "IntExprNil")
        .next()
        .is_some()
    {
        return Some(Vec::new());
    }
    for cons in site.nodes_in_class_value(class, "IntExprCons") {
        let Some(element) = site.class_of_child(cons, 0) else {
            continue;
        };
        let Some(tail) = site.class_of_child(cons, 1) else {
            continue;
        };
        let Some(expr) =
            super::iota::parse_int_expr_memo(site, &element, 64, Some(out_shape), memo)
        else {
            continue;
        };
        if let Some(mut rest) = parse_entry_list(site, &tail, depth - 1, out_shape, memo) {
            rest.insert(0, expr);
            return Some(rest);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::reference::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use crate::buffer_tensor_ir::ReferenceKernelCtx;
use crate::reference::kernels::{expect_op, move_gathered};

/// Index-map materialization: evaluate the parsed per-axis entry
/// expressions at every OUT coordinate and copy the addressed parent
/// elements.
pub(in crate::reference) fn kernel(
    op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    let op = expect_op::<IndexMapApplyMaterializeDps>(op)?;
    let Some(entries) = &op.entries else {
        anyhow::bail!(
            "materialize reference kernel: index map beyond the parsed expression subset"
        );
    };
    let parent_dims = &ctx.operand_dims[0];
    let out_dims = &ctx.operand_dims[1];
    anyhow::ensure!(
        entries.len() == parent_dims.len(),
        "index map arity {} vs parent rank {}",
        entries.len(),
        parent_dims.len()
    );
    let mut parent_strides = vec![1usize; parent_dims.len()];
    for k in (0..parent_dims.len().saturating_sub(1)).rev() {
        parent_strides[k] = parent_strides[k + 1] * parent_dims[k + 1];
    }
    let out_rank = out_dims.len();
    let numel = ctx.dests[0].len();
    let mut index_of = vec![0usize; numel];
    for (flat, slot) in index_of.iter_mut().enumerate() {
        // Decompose the flat OUT index into row-major coordinates.
        let mut remainder = flat;
        let mut coords = vec![0usize; out_rank];
        for axis in (0..out_rank).rev() {
            coords[axis] = remainder % out_dims[axis];
            remainder /= out_dims[axis];
        }
        let mut parent_flat = 0usize;
        for (k, entry) in entries.iter().enumerate() {
            let index = entry.eval(&coords);
            anyhow::ensure!(
                index >= 0 && (index as usize) < parent_dims[k],
                "materialize index {index} out of bounds for parent axis {k} (extent {})",
                parent_dims[k]
            );
            parent_flat += index as usize * parent_strides[k];
        }
        *slot = parent_flat;
    }
    let parent = ctx.operands[0].clone();
    move_gathered(&parent, &mut ctx.dests[0], &index_of)
}
