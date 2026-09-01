//! The logical-op registry: one type per `Logical*` egglog constructor,
//! owning EVERYTHING about that op — its constructor name, display name,
//! child ports, readable rendering, and its complete egglog surface (the
//! constructor declaration and every rule, one `.egg` file per rule in the
//! op's own directory, included via `snippets()` and spliced into the core
//! preamble by `egglog_snippet::assembled_program`). Each op lives in its
//! own submodule file beside its `.egg` directory. Adding a logical op
//! touches its module, its `.egg` files, and one registration line —
//! nothing else: the renderer and the program assembler both consult this
//! registry.

use egraph_serialize::Node;

use crate::egglog_snippet::EgglogSnippet;

/// Rendering callbacks a [`LogicalOp`] may use while formatting itself.
/// Implemented by the extractor's renderer; the ops stay ignorant of the
/// renderer's internals and depth conventions except where they choose them.
/// Every method resolves a CHILD of the op's own enode; `None` means the
/// child is missing or unreadable, and the op picks its own fallback text.
pub trait LogicalRender {
    /// Recursively render the readable expression of the child at `index`
    /// (cycle-guarded by the renderer; falls back to the child's label).
    fn child_expr(&mut self, node: &Node, index: usize) -> String;
    /// Depth-bounded structural rendering of the child at `index`,
    /// preferring `prefer`-named enodes when the class offers a choice.
    fn child_short(
        &mut self,
        node: &Node,
        index: usize,
        depth: usize,
        prefer: Option<&str>,
    ) -> Option<String>;
    /// Readable shape of the child at `index`.
    fn child_shape(&mut self, node: &Node, index: usize) -> Option<String>;
    /// Readable index map of the child at `index`.
    fn child_index_map(&mut self, node: &Node, index: usize) -> Option<String>;
    /// Readable integer expression (`IntExpr`) of the child at `index`.
    fn child_int_expr(&mut self, node: &Node, index: usize) -> Option<String>;
}

/// One logical op: the owner of every Rust-side fact about a `Logical*`
/// constructor. Registered in [`built_in_logical_ops`]; the registration
/// order doubles as the renderer's preference order when an e-class offers
/// several logical nodes to display.
pub trait LogicalOp: std::fmt::Debug {
    /// The egglog constructor this op owns, e.g. `"LogicalAdd"`.
    fn egglog_constructor(&self) -> &'static str;

    /// The short display name used for values labeled by this op.
    fn display_name(&self) -> &'static str;

    /// The op's tensor operands as (port name, child index) pairs — the
    /// detail-graph edges. Non-tensor children (axes, index maps, shapes)
    /// are not ports; they surface through `readable_expr`.
    fn child_ports(&self) -> &'static [(&'static str, usize)] {
        &[]
    }

    /// The label rendered for a value this op produces. Defaults to the
    /// display name; `LogicalTensorInputLit` overrides it to show the literal's id.
    fn display_label(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let _ = (node, ctx);
        self.display_name().to_string()
    }

    /// The one-line readable expression for this op's enode.
    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String;

    /// Everything this op contributes to the assembled egglog program:
    /// its constructor declaration and every rule it owns, one `.egg`
    /// file per contribution (see `egglog_snippet::assembled_program`).
    /// The default is empty only for ops with no egglog surface of their
    /// own (`LogicalTensorInputLit`, whose constructor is core).
    fn snippets(&self) -> Vec<EgglogSnippet> {
        Vec::new()
    }
}

/// The input declaration: one named logical tensor whose shape and dtype
/// ARE the declaration. Every dataflow leaf is one.
#[derive(Debug, Clone, Copy)]
pub struct LogicalTensorInputLit;

impl LogicalOp for LogicalTensorInputLit {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalTensorInputLit"
    }

    fn display_name(&self) -> &'static str {
        "LogicalTensorInputLit"
    }

    fn display_label(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        ctx.child_short(node, 0, 2, Some("LogicalIdLit"))
            .unwrap_or_else(|| "LogicalTensorInputLit".to_string())
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let id = ctx
            .child_short(node, 0, 2, Some("LogicalIdLit"))
            .unwrap_or_else(|| "?".to_string());
        let shape = ctx.child_shape(node, 1).unwrap_or_else(|| "?".to_string());
        let dtype = ctx
            .child_short(node, 2, 4, None)
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalTensorInputLit({id}, shape={shape}, dtype={dtype})")
    }
}

/// A name-only output designation: no shape, no dtype. The model unions it
/// with the value it names, whose own derivation supplies both — so the
/// naming can never disagree with the value.
#[derive(Debug, Clone, Copy)]
pub struct LogicalTensorNamed;

impl LogicalOp for LogicalTensorNamed {
    fn egglog_constructor(&self) -> &'static str {
        "LogicalTensorNamed"
    }

    fn display_name(&self) -> &'static str {
        "LogicalTensorNamed"
    }

    fn display_label(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        ctx.child_short(node, 0, 2, Some("LogicalIdLit"))
            .unwrap_or_else(|| "LogicalTensorNamed".to_string())
    }

    fn readable_expr(&self, node: &Node, ctx: &mut dyn LogicalRender) -> String {
        let id = ctx
            .child_short(node, 0, 2, Some("LogicalIdLit"))
            .unwrap_or_else(|| "?".to_string());
        format!("LogicalTensorNamed({id})")
    }
}

mod add;
mod cast;
mod constant;
mod div;
mod exp;
mod exp2;
mod gather;
mod index_map_apply;
mod iota;
mod less_than;
mod log2;
mod modulo;
mod mul;
mod recip;
mod reduce_max;
mod reduce_sum;
mod scatter;
mod sin;
mod sqrt;
mod trunc_div;
mod trunc_rem;

pub use add::LogicalAdd;
pub use cast::LogicalCast;
pub use constant::LogicalConstant;
pub use div::LogicalDiv;
pub use exp::LogicalExp;
pub use exp2::LogicalExp2;
pub use gather::LogicalGather;
pub use index_map_apply::LogicalIndexMapApply;
pub use iota::LogicalIota;
pub use less_than::LogicalLessThan;
pub use log2::LogicalLog2;
pub use modulo::LogicalMod;
pub use mul::LogicalMul;
pub use recip::LogicalRecip;
pub use reduce_max::LogicalReduceMax;
pub use reduce_sum::LogicalReduceSum;
pub use scatter::LogicalScatter;
pub use sin::LogicalSin;
pub use sqrt::LogicalSqrt;
pub use trunc_div::LogicalTruncDiv;
pub use trunc_rem::LogicalTruncRem;

/// THE registration list for logical ops. Order matters twice: it is the
/// renderer's preference order when an e-class holds several logical nodes,
/// and (eventually) the emission grouping for snippets. Adding a logical op
/// = writing its type above and its line here.
pub fn built_in_logical_ops() -> &'static [Box<dyn LogicalOp + Send + Sync>] {
    static OPS: std::sync::OnceLock<Vec<Box<dyn LogicalOp + Send + Sync>>> =
        std::sync::OnceLock::new();
    OPS.get_or_init(|| {
        vec![
            Box::new(LogicalTensorInputLit),
            Box::new(LogicalTensorNamed),
            Box::new(LogicalSqrt),
            Box::new(LogicalExp),
            Box::new(LogicalAdd),
            Box::new(LogicalMul),
            Box::new(LogicalDiv),
            Box::new(LogicalReduceSum),
            Box::new(LogicalReduceMax),
            Box::new(LogicalExp2),
            Box::new(LogicalLog2),
            Box::new(LogicalSin),
            Box::new(LogicalRecip),
            Box::new(LogicalMod),
            Box::new(LogicalLessThan),
            Box::new(LogicalCast),
            Box::new(LogicalIota),
            Box::new(LogicalConstant),
            Box::new(LogicalGather),
            Box::new(LogicalScatter),
            Box::new(LogicalIndexMapApply),
            Box::new(LogicalTruncDiv),
            Box::new(LogicalTruncRem),
        ]
    })
}

/// Registry lookup by egglog constructor name.
pub fn logical_op_for(constructor: &str) -> Option<&'static (dyn LogicalOp + Send + Sync)> {
    built_in_logical_ops()
        .iter()
        .find(|op| op.egglog_constructor() == constructor)
        .map(|op| op.as_ref())
}
