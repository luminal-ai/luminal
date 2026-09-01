//! Egglog program fragments as data, and the registry-driven assembler.
//!
//! The op surface of the egglog program — every `Logical*` constructor,
//! every `LayoutTensorOp*` implementation constructor, and every rule
//! associated with an op — lives ALONGSIDE the op that owns it: one `.egg`
//! file per rule, `include_str!`-ed by the op's `snippets()` (the
//! `LogicalOp` registry for logical ops, the `OpMatcher` registry for
//! implementations). The CORE preamble (`egglog_preamble.egg`) keeps
//! everything shared — the IntExpr machinery, layouts, buffers,
//! certificates, the coordinate machinery, the terminal stratum — plus ten
//! `;; @SPLICE <category>` anchor lines marking exactly where each
//! category of op contribution is inserted.
//!
//! Anchor positions are DECLARATION-ORDER-SAFE by construction: each
//! anchor sits after every core declaration its category's rules may
//! reference (the scatter-seed lesson: egglog resolves names top-down).
//! Within a category, contributions splice in registry-registration
//! order; rule order within a ruleset is fixpoint-irrelevant, so the
//! assembled program is semantically identical to the old monolith — the
//! byte-stable goldens are the proof.

/// Where in the core preamble a snippet is spliced. One variant per
/// `;; @SPLICE` anchor, in the anchors' file order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceCategory {
    /// `(constructor Logical* ...)` declarations, after the datatype block.
    LogicalConstructors,
    /// `(constructor LayoutTensorOp* ...)` implementation declarations.
    LayoutOpConstructors,
    /// Dtype propagation rules and dtype tripwires.
    Dtype,
    /// Coordinate demand seeds and structural guards (rank-0 rejections,
    /// reduce axis-removal triggers) — after the coordinate machinery.
    Seed,
    /// Shape propagation rules — after `coords-common-shape`.
    Shape,
    /// Logical ring rewrites (commutativity, associativity, distributivity).
    Rewrites,
    /// Forward layout-tensor propagation and canonical mints.
    Forward,
    /// Implementation match rules (the `LayoutTensorOpLit` unions).
    Match,
    /// The gather⇄view unification machinery.
    Coordinate,
    /// Terminal-stratum rules (closures, locks, extent-lock seeds).
    Fixpoint,
}

impl SpliceCategory {
    fn marker(self) -> &'static str {
        match self {
            SpliceCategory::LogicalConstructors => ";; @SPLICE logical-constructors",
            SpliceCategory::LayoutOpConstructors => ";; @SPLICE layout-op-constructors",
            SpliceCategory::Dtype => ";; @SPLICE dtype",
            SpliceCategory::Seed => ";; @SPLICE seed",
            SpliceCategory::Shape => ";; @SPLICE shape",
            SpliceCategory::Rewrites => ";; @SPLICE rewrites",
            SpliceCategory::Forward => ";; @SPLICE forward",
            SpliceCategory::Match => ";; @SPLICE match",
            SpliceCategory::Coordinate => ";; @SPLICE coordinate",
            SpliceCategory::Fixpoint => ";; @SPLICE fixpoint",
        }
    }

    const ALL: [SpliceCategory; 10] = [
        SpliceCategory::LogicalConstructors,
        SpliceCategory::LayoutOpConstructors,
        SpliceCategory::Dtype,
        SpliceCategory::Seed,
        SpliceCategory::Shape,
        SpliceCategory::Rewrites,
        SpliceCategory::Forward,
        SpliceCategory::Match,
        SpliceCategory::Coordinate,
        SpliceCategory::Fixpoint,
    ];
}

/// One op contribution: verbatim egglog text (one rule or one constructor
/// declaration — one `.egg` file) bound for one splice anchor.
#[derive(Debug, Clone, Copy)]
pub struct EgglogSnippet {
    pub category: SpliceCategory,
    pub text: &'static str,
}

/// Splice contributions into the core at their anchors. Every anchor line
/// is kept (as a section header) and followed by its category's snippets
/// in the given order. Panics on a missing anchor — that is schema drift
/// between this module and the core preamble.
pub fn assemble(core: &str, snippets: &[EgglogSnippet]) -> String {
    let mut out = core.to_string();
    for category in SpliceCategory::ALL {
        let marker = category.marker();
        assert!(
            out.contains(marker),
            "core preamble is missing the {marker} anchor"
        );
        let mut section = String::from(marker);
        for snippet in snippets.iter().filter(|s| s.category == category) {
            section.push_str("\n\n");
            section.push_str(snippet.text.trim_end());
        }
        out = out.replacen(marker, &section, 1);
    }
    out
}

/// THE sanctioned `EGraph` constructor: registers the harness-side
/// primitives the core preamble contracts for. `(bigint-to-i64 z)` is the
/// bounds lattice's only way back out of BigInt — PARTIAL, unmatched
/// outside i64 range — and its one rule-side use is the [n,n] pin-collapse
/// rule in the bounds section. A bare `EGraph::default()` cannot even
/// parse the assembled program, so every runner goes through here.
pub fn new_egraph() -> egglog::EGraph {
    use egglog::sort::Z;

    let mut egraph = egglog::EGraph::default();
    egglog::add_primitive!(&mut egraph, "bigint-to-i64" = |a: Z| -?> i64 {
        i64::try_from(&*a).ok()
    });
    // The vendored substitution primitive (egglog-experimental PR #60):
    // the preamble's subst road calls it from a :naive rule. The skip
    // set names the MEMO constructors the walk must treat as opaque
    // metadata rather than term structure (int-subst-of is the only
    // one — the transform-style census).
    egraph.add_full_primitive(
        crate::subst_primitive::Subst {
            skip: ["int-subst-of".to_string()].into_iter().collect(),
        },
        None,
    );
    egraph
}

/// [`assembled_program`] for an arbitrary RUNTIME's matcher set — the
/// TestRuntime seam (ruling 2026-08-13: test-only op variants live in a
/// small tests-side runtime, and extraction takes the runtime's
/// matchers instead of hardcoding the reference registry). Logical-op
/// snippets are always included; the matcher contributions come from
/// the caller.
pub fn assembled_program_for(matchers: &[Box<dyn crate::layout_ir::OpMatcher>]) -> String {
    let core = include_str!("egglog_core/egglog_preamble.egg");
    let mut snippets: Vec<EgglogSnippet> = Vec::new();
    for op in crate::logical_op::built_in_logical_ops() {
        snippets.extend(op.snippets());
    }
    for matcher in matchers {
        snippets.extend(matcher.snippets());
    }
    assemble(core, &snippets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Splicing inserts at the anchor, keeps the anchor as a header, and
    /// preserves contribution order within a category.
    #[test]
    fn splices_in_order_at_the_anchor() {
        let core = "top\n;; @SPLICE logical-constructors\n\n;; @SPLICE layout-op-constructors\n\n;; @SPLICE dtype\n\n;; @SPLICE seed\n\n;; @SPLICE shape\n\n;; @SPLICE rewrites\n\n;; @SPLICE forward\n\n;; @SPLICE match\n\n;; @SPLICE coordinate\n\n;; @SPLICE fixpoint\nbottom\n";
        let snippets = [
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: "(rule-a)\n",
            },
            EgglogSnippet {
                category: SpliceCategory::Dtype,
                text: "(rule-b)\n",
            },
            EgglogSnippet {
                category: SpliceCategory::Fixpoint,
                text: "(rule-c)\n",
            },
        ];
        let out = assemble(core, &snippets);
        assert!(out.contains(";; @SPLICE dtype\n\n(rule-a)\n\n(rule-b)"));
        assert!(out.contains(";; @SPLICE fixpoint\n\n(rule-c)\nbottom"));
    }

    /// The assembled program parses cleanly end to end — every
    /// declaration precedes its uses, every included `.egg` file is
    /// balanced. Core owns no runtime registry (Step B), so this proves
    /// the core preamble + logical-op snippets alone; the reference
    /// registry's assembly is proven by luminal_reference's suites,
    /// which run its program everywhere.
    #[test]
    fn assembled_program_is_loadable() {
        new_egraph()
            .parse_and_run_program(None, &assembled_program_for(&[]))
            .expect("assembled program loads");
    }
}
