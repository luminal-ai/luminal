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
        let mut included = std::collections::HashSet::new();
        for snippet in snippets.iter().filter(|s| s.category == category) {
            // Several executable forms can depend on the same proof vocabulary.
            // Keep its declarations/rules once, in first-contributor order.
            if !included.insert(snippet.text.trim_end()) {
                continue;
            }
            section.push_str("\n\n");
            section.push_str(snippet.text.trim_end());
        }
        out = out.replacen(marker, &section, 1);
    }
    out
}

/// Where a frontend splices its own egglog text into a runtime's bound
/// program.
pub trait ProgramSeams {
    /// Text rendered after the model and boundary, before the schedule.
    fn before_schedule(&mut self, text: &str);
    /// A post-schedule unit: `text` runs after the schedule; `label` names
    /// it when a runtime isolates a failing unit.
    fn after_schedule(&mut self, label: &str, text: &str);
}

/// THE sanctioned `EGraph` constructor: registers the harness-side
/// primitives the core preamble contracts for. `(bigint-to-i64 z)` is the
/// bounds lattice's only way back out of BigInt — PARTIAL, unmatched
/// outside i64 range — and its one rule-side use is the [n,n] pin-collapse
/// rule in the bounds section. `(exact-div a b)` is the partial exact
/// quotient (i64 and BigInt) that the literal exact-division rules bind
/// instead of carrying a nonzero-divisor premise, a zero-remainder
/// premise and a bare `/` in the action. A bare `EGraph::default()`
/// cannot even parse the assembled program, so every runner goes through
/// here.
pub fn new_egraph() -> egglog::EGraph {
    use egglog::sort::Z;

    let mut egraph = egglog::EGraph::default();
    crate::element_stride::register(&mut egraph);
    // Tree decomposition is off for now: egglog's decomposition planner was
    // a large performance regression on our saturations (identical fixpoints
    // either way). Temporary until the planner is revisited.
    egraph.no_decomp = true;
    egglog::add_primitive!(&mut egraph, "bigint-to-i64" = |a: Z| -?> i64 {
        i64::try_from(&*a).ok()
    });
    // `(exact-div a b)`: the quotient when b divides a exactly — PARTIAL
    // otherwise (b = 0, a nonzero remainder, or i64 overflow on MIN / -1).
    // The rule-side spelling of "exact or no match": a rule that binds the
    // result simply does not fire for inexact division, which is the one
    // precondition every literal cancellation needs. Overloaded on i64
    // (IntLit payloads) and BigInt (the bounds lattice).
    egglog::add_primitive!(&mut egraph, "exact-div" = |a: i64, b: i64| -?> i64 {
        match a.checked_rem(b) {
            Some(0) => a.checked_div(b),
            _ => None,
        }
    });
    egglog::add_primitive!(&mut egraph, "exact-div" = |a: Z, b: Z| -?> Z {
        (!num_traits::Zero::is_zero(&*b) && num_traits::Zero::is_zero(&*(a.clone() % b.clone())))
            .then(|| a / b)
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
    egraph.add_full_primitive(
        crate::subst_primitive::MemoizedSubst {
            name: "memoized-int-subst".to_owned(),
            cache_table: "int-subst-cache".to_owned(),
            inner: crate::subst_primitive::Subst {
                skip: ["int-subst-of".to_owned()].into_iter().collect(),
            },
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
    let core = include_str!("egglog_preamble.egg");
    let mut snippets: Vec<EgglogSnippet> = Vec::new();
    snippets.extend(crate::element_stride::snippets());
    for op in crate::logical_op::built_in_logical_ops() {
        snippets.extend(op.snippets());
    }
    for helper in crate::logical_helper::built_in_logical_helpers() {
        snippets.extend(helper.snippets());
    }
    for matcher in matchers {
        snippets.extend(matcher.snippets());
    }
    assemble(core, &snippets)
}

/// WHAT CORE'S OWN PREAMBLE DECLARES AND DECODES: the five `Layout`
/// constructors. The declaration and its decoder travel together — this
/// is the `core` half of that pairing, the matcher's
/// [`crate::layout_ir::OpMatcher::decoders`] is the other.
pub fn core_decoders() -> Vec<crate::egglog_utils::eclass::ConstructorDecoder> {
    let decoders = crate::layouts::layout_decoders();
    decoders
        .into_iter()
        .chain([crate::egglog_utils::eclass::ConstructorDecoder::of::<
            crate::element_stride::ElementStridedLayout,
        >()])
        .collect()
}

/// The decoder registry for a runtime's matcher set — the same shape as
/// [`assembled_program_for`]: core's built-ins first, then each
/// matcher's contribution in registry order. A duplicate
/// `(sort, constructor)` is REFUSED (registering two decoders for one
/// constructor is a bug, never a preference).
pub fn decoder_registry_for(
    matchers: &[Box<dyn crate::layout_ir::OpMatcher>],
) -> anyhow::Result<crate::egglog_utils::eclass::ConstructorRegistry> {
    crate::egglog_utils::eclass::ConstructorRegistry::new(
        core_decoders()
            .into_iter()
            .chain(matchers.iter().flat_map(|matcher| matcher.decoders())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_proof_snippets_are_included_once_per_category() {
        let core = assembled_program_for(&[]);
        let shared = EgglogSnippet {
            category: SpliceCategory::LayoutOpConstructors,
            text: "(datatype SharedProofVocabulary (SharedProofWitness))",
        };
        let program = assemble(&core, &[shared, shared]);
        assert_eq!(program.matches(shared.text).count(), 1);
        new_egraph().parse_and_run_program(None, &program).unwrap();
    }

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

    #[test]
    fn offset_fold_preserves_equivalent_states_after_a_late_singleton_pin() {
        let mut graph = new_egraph();
        graph
            .parse_and_run_program(None, &assembled_program_for(&[]))
            .unwrap();
        graph
            .parse_and_run_program(
                None,
                r#"
(let extent (IntVar "fold-late-singleton"))
(set (lower-bound-of extent) (bigint 1))
(set (upper-bound-of extent) (bigint 8))
(let dimensions (IntExprCons extent (IntExprCons (IntLit 4) (IntExprNil))))
(let shape (ShapeLit dimensions))
(let pinned_dimensions (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil))))
(let pinned_shape (ShapeLit pinned_dimensions))
(let dynamic (RightMajorContiguousElementLayoutLit shape (BitWidthLit 32)))
(let pinned (RightMajorContiguousElementLayoutLit pinned_shape (BitWidthLit 32)))
(LeftMajorContiguousElementLayoutLit shape (BitWidthLit 32))
(LeftMajorContiguousElementLayoutLit pinned_shape (BitWidthLit 32))
(run-schedule (saturate (run main_ruleset) (run prop)))
; Key equality can arrive before rewriting proves the folded expressions equal.
(union extent (IntLit 1))
(run-schedule (saturate (run main_ruleset) (run prop)))
(check (= dynamic pinned))
(check (= dynamic (ElementOffsetExpressionLayoutLit (CoordVar shape 0) shape (BitWidthLit 32))))
(fail (check (= (IntLit 4) (IntLit 1))))
"#,
            )
            .unwrap();
    }

    #[test]
    fn exact_div_is_partial_outside_exact_division() {
        let mut eg = new_egraph();
        eg.parse_and_run_program(
            None,
            r#"
          (check (= (exact-div 6 2) 3))
          (check (= (exact-div -6 2) -3))
          (check (= (exact-div 6 -2) -3))
          (check (= (exact-div 0 5) 0))
          (fail (check (= (exact-div 7 2) 3)))
          (fail (check (= (exact-div 7 2) 4)))
          (fail (check (= (exact-div 6 0) 0)))
          (fail (check (= (exact-div -9223372036854775808 -1) 0)))
          (check (= (exact-div (bigint 6) (bigint 2)) (bigint 3)))
          (fail (check (= (exact-div (bigint 7) (bigint 2)) (bigint 3))))
          (fail (check (= (exact-div (bigint 7) (bigint 0)) (bigint 0))))
        "#,
        )
        .unwrap();
    }

    #[test]
    fn uint64_wraparound_does_not_acquire_an_unbounded_integer_interval() {
        let mut eg = new_egraph();
        eg.parse_and_run_program(None, &assembled_program_for(&[]))
            .unwrap();
        eg.parse_and_run_program(None,r#"
          (let minus (LogicalCast (LogicalIota (IntLit -1) (ShapeLit (IntExprNil))) (UInt64)))
          (let one (LogicalCast (LogicalIota (IntLit 1) (ShapeLit (IntExprNil))) (UInt64)))
          (let wrap (LogicalAdd minus one))
          (let square (LogicalMul minus minus))
          (run-schedule (repeat 8 (seq (saturate (run prop)) (run main_ruleset))))
          (check (= (value-lower-bound-of wrap) (bigint 0)))
          (check (= (value-upper-bound-of wrap) (- (* (bigint 4294967296) (bigint 4294967296)) (bigint 1))))
          (check (= (value-lower-bound-of square) (bigint 0)))
        "#).unwrap();
    }

    /// The `prop` ruleset holds exactly the shape-of / dtype-of / rank-of
    /// propagation rules: every default-ruleset rule does something else,
    /// and every `prop` rule does nothing else. The schedule runs `prop`
    /// to saturation before the main loop and alongside `(run main_ruleset)` inside
    /// it, so a propagation rule left in the default ruleset would lose
    /// the pre-pass and a minting rule tagged `prop` would run in it.
    #[test]
    fn prop_ruleset_is_exactly_the_propagation_rules() {
        use egglog::ast::{Command, GenericAction};
        const PROP: [&str; 3] = ["shape-of", "dtype-of", "rank-of"];
        let only_propagation = |rule: &egglog::ast::Rule| {
            !rule.head.0.is_empty()
                && rule.head.0.iter().all(|action| match action {
                    GenericAction::Set(_, head, _, _) => PROP.contains(&head.as_str()),
                    _ => false,
                })
        };
        let commands = new_egraph()
            .parse_program(None, &assembled_program_for(&[]))
            .expect("assembled program parses");
        let mut prop_rules = 0;
        for command in &commands {
            let Command::Rule { rule } = command else {
                continue;
            };
            match rule.ruleset.as_str() {
                "prop" => {
                    prop_rules += 1;
                    assert!(
                        only_propagation(rule),
                        "prop rule does more than propagate: {rule}"
                    );
                }
                "" => assert!(
                    !only_propagation(rule),
                    "propagation rule missing `:ruleset prop`: {rule}"
                ),
                _ => {}
            }
        }
        assert!(prop_rules > 0, "no prop rules found");
    }
}
