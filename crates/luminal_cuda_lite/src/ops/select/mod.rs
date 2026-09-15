//! Select: elementwise ternary choice over two same-shape/same-dtype value
//! branches by a boolean condition — CUDA-lite's OWN op (ruling 2026-08-17:
//! every runtime owns its executable ops; the shared crate supplies only the
//! IR traits). Same egglog constructor and label as the reference runtime's
//! select — assemblies are per-runtime, labels are IR identity — but the
//! structs, matcher, snippets, and codegen all live here.
//!
//! This is the non-arithmetic selection primitive. Expressing a select as
//! `cond*a + (1-cond)*b` puts a sum of products into the e-graph, which the
//! integer closure then expands combinatorially; a real ternary op keeps the
//! expression a single node. The kernel copies bits, so it is dtype-generic.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

use crate::kernels::{CodegenCtx, KernelOp, KernelSource, ternary};
use anyhow::Result;

/// `SelectFunctionalGeneric(condition, if_true, if_false) -> out` — pure dataflow form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectFunctional;

impl OpSlotNames for SelectFunctional {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "condition".to_string(),
            1 => "if_true".to_string(),
            2 => "if_false".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for SelectFunctional {
    fn label(&self) -> &str {
        "SelectFunctionalGeneric"
    }
}

impl Bufferizable for SelectFunctional {}

impl ToDps for SelectFunctional {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(SelectFunctionalDps))
    }
}

impl LayoutIrOp for SelectFunctional {}

/// Destination-passing form: `Select(condition: read, if_true: read, if_false: read, dest0: write ↔ out0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectFunctionalDps;

impl OpSlotNames for SelectFunctionalDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "condition".to_string(),
            1 => "if_true".to_string(),
            2 => "if_false".to_string(),
            3 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for SelectFunctionalDps {
    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        Some(crate::CudaOpInterface::kernel::<Self>())
    }

    fn label(&self) -> &str {
        "SelectFunctionalGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 3 // dest0 is write-only
    }
}

impl Bufferizable for SelectFunctionalDps {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 3,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for SelectFunctionalDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for SelectFunctionalDps {}

/// The CUDA lowering: `out[i] = condition[i] ? if_true[i] : if_false[i]`.
/// The condition is Bool8 and the result is one branch verbatim, so the
/// expression involves no arithmetic and works for every storage dtype.
impl KernelOp for SelectFunctionalDps {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>> {
        ternary(ctx, "(c[i] ? a[i] : b[i])")
    }
}

/// Matches `LayoutTensorOpSelectFunctionalGeneric` and produces this
/// runtime's [`SelectFunctional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SelectFunctionalMatcher;

impl OpMatcher for SelectFunctionalMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpSelectFunctionalGeneric"
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

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(SelectFunctional)
    }
}
