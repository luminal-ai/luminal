//! The range torch exported for each dynamic dimension, the bucket a
//! runtime searches it over, and the declaration that states it.
//!
//! Only bare symbols are read. A compound key (`2*s77`, torch's infix
//! spelling of a derived input dim) restates its root symbol's range,
//! which interval arithmetic derives.
//!
//! Stated to the e-graph, a range is two relation rows applied to the
//! bounds lattice by `set`. `lower-bound-of` merges by `max` and
//! `upper-bound-of` by `min`, so a declaration only ever tightens: a
//! looser one is a no-op and a tighter one narrows; a contradiction trips
//! the preamble's crossed-bounds rule.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, ensure};
use luminal::egglog_snippet::ProgramSeams;
use luminal::graph::DimBucket;
use luminal::shape::Symbol;

use crate::declaration::{Declaration, Declarations, Placement, RULESET};
use crate::pt2_schema::RangeConstraint;
use crate::translate::Translation;

/// The closed interval torch exported for a dimension symbol; an absent
/// side is unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DimRange {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl std::fmt::Display for DimRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |bound: Option<u64>| bound.map_or("inf".to_string(), |b| b.to_string());
        write!(f, "[{}, {}]", side(self.min), side(self.max))
    }
}

/// The exported ranges keyed by recorder symbol.
pub(crate) fn dim_ranges(
    ranges: &HashMap<String, RangeConstraint>,
    symbols: &HashMap<String, Symbol>,
) -> BTreeMap<Symbol, DimRange> {
    let mut out = BTreeMap::new();
    for (name, range) in ranges {
        let Some(symbol) = symbols.get(name) else {
            continue;
        };
        out.insert(
            *symbol,
            DimRange {
                min: range.min_val.and_then(|v| u64::try_from(v).ok()),
                max: range.max_val.and_then(|v| u64::try_from(v).ok()),
            },
        );
    }
    out
}

/// The bucket a runtime searches `symbol` over: torch's range met with the
/// runtime's own `[floor, ceiling]`, profiled at `hint` clamped inside it.
pub fn dim_bucket(
    symbol: Symbol,
    range: Option<DimRange>,
    floor: usize,
    ceiling: usize,
    hint: usize,
) -> Result<DimBucket> {
    let range = range.unwrap_or_default();
    let to_usize = |bound: Option<u64>| bound.map(|b| usize::try_from(b).unwrap_or(usize::MAX));
    let min = to_usize(range.min).map_or(floor, |m| m.max(floor));
    let max = to_usize(range.max).map_or(ceiling, |m| m.min(ceiling));
    ensure!(
        min <= max,
        "dimension `{symbol}`: torch's exported range {range} does not meet the runtime's \
         [{floor}, {ceiling}]"
    );
    Ok(DimBucket::new(min, max).representative(hint.clamp(min, max)))
}

/// The relations and the two rules that apply them.
fn header() -> String {
    format!(
        "(relation pytorch-declared-dim-min (IntExpr BigInt))\n\
         (relation pytorch-declared-dim-max (IntExpr BigInt))\n\
         (rule\n  (\n    (pytorch-declared-dim-min ?dim ?bound)\n  )\n  (\n    \
         (set (lower-bound-of ?dim) ?bound)\n  )\n  :ruleset {RULESET}\n  :name \"pytorch: declared dim min applies\"\n)\n\
         (rule\n  (\n    (pytorch-declared-dim-max ?dim ?bound)\n  )\n  (\n    \
         (set (upper-bound-of ?dim) ?bound)\n  )\n  :ruleset {RULESET}\n  :name \"pytorch: declared dim max applies\"\n)\n"
    )
}

/// State this translation's exported dimension ranges, alone, on a
/// runtime's bindings.
pub fn declare_dim_ranges(
    bindings: &mut impl ProgramSeams,
    translation: &Translation,
    placement: Placement,
) {
    Declarations::new(placement)
        .with(dim_range_program(&translation.dim_ranges))
        .state(bindings);
}

/// The exported ranges as `pytorch-declared-dim-min` / `-max` rows over
/// each dimension's `IntVar`, one row per bounded side.
pub fn dim_range_program(ranges: &BTreeMap<Symbol, DimRange>) -> Declaration {
    let mut facts = String::new();
    for (symbol, range) in ranges {
        let dim = format!("(IntVar {})", symbol.egglog_literal());
        if let Some(min) = range.min {
            facts.push_str(&format!(
                "(pytorch-declared-dim-min {dim} (bigint {min}))\n"
            ));
        }
        if let Some(max) = range.max {
            facts.push_str(&format!(
                "(pytorch-declared-dim-max {dim} (bigint {max}))\n"
            ));
        }
    }
    Declaration {
        header: header(),
        facts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::{LABEL, Rendered};
    use crate::pt2_schema::{
        Argument, DimExpr, DimInt, DimSize, ExportedProgram, ExprValue, GraphModule, IntArg,
        Signature, TensorMeta,
    };
    use crate::translate::test_support::*;
    use luminal_reference::{ReferenceBindings, assembled_program};

    fn constraint(min: Option<i64>, max: Option<i64>) -> RangeConstraint {
        RangeConstraint {
            min_val: min,
            max_val: max,
        }
    }

    fn symbols(names: &[&str]) -> HashMap<String, Symbol> {
        names
            .iter()
            .map(|name| ((*name).to_string(), Symbol::new(name)))
            .collect()
    }

    fn range(min: Option<u64>, max: Option<u64>) -> DimRange {
        DimRange { min, max }
    }

    fn ranges_of(entries: &[(&str, Option<u64>, Option<u64>)]) -> BTreeMap<Symbol, DimRange> {
        entries
            .iter()
            .map(|(name, min, max)| (Symbol::new(name), range(*min, *max)))
            .collect()
    }

    /// The range declaration alone, rendered at `placement`.
    fn rendered(ranges: &BTreeMap<Symbol, DimRange>, placement: Placement) -> Rendered {
        Declarations::new(placement)
            .with(dim_range_program(ranges))
            .render()
    }

    /// Run `text` after the preamble on a fresh e-graph.
    fn run(text: &str) -> std::result::Result<(), String> {
        let program = format!("{}\n\n{text}", assembled_program());
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &program)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    #[test]
    fn bare_symbol_ranges_are_read_and_compound_keys_are_not() {
        let mut ranges = HashMap::new();
        ranges.insert("s77".to_string(), constraint(Some(3), Some(64)));
        ranges.insert("2*s77".to_string(), constraint(Some(6), Some(128)));
        ranges.insert("s78".to_string(), constraint(Some(2), None));
        let out = dim_ranges(&ranges, &symbols(&["s77", "s78"]));
        assert_eq!(
            out.get(&Symbol::new("s77")),
            Some(&range(Some(3), Some(64)))
        );
        assert_eq!(out.get(&Symbol::new("s78")), Some(&range(Some(2), None)));
        assert!(!out.contains_key(&Symbol::new("2*s77")));
    }

    #[test]
    fn a_symbol_the_program_does_not_name_is_ignored() {
        let mut ranges = HashMap::new();
        ranges.insert("u0".to_string(), constraint(Some(0), Some(12)));
        let out = dim_ranges(&ranges, &symbols(&["s77"]));
        assert!(!out.contains_key(&Symbol::new("u0")));
    }

    #[test]
    fn the_bucket_is_torchs_range_met_with_the_runtimes() {
        let s = Symbol::new("s77");
        let bucket = dim_bucket(s, Some(range(Some(3), Some(64))), 1, 4096, 4).unwrap();
        assert_eq!(
            (bucket.min, bucket.max, bucket.representative_value()),
            (3, 64, 4)
        );
        let bucket = dim_bucket(s, Some(range(Some(2), None)), 1, 4096, 4).unwrap();
        assert_eq!((bucket.min, bucket.max), (2, 4096));
        let bucket = dim_bucket(s, None, 1, 4096, 4).unwrap();
        assert_eq!(
            (bucket.min, bucket.max, bucket.representative_value()),
            (1, 4096, 4)
        );
    }

    #[test]
    fn the_hint_is_profiled_inside_the_bucket() {
        let s = Symbol::new("s77");
        let torch = Some(range(Some(3), Some(64)));
        assert_eq!(
            dim_bucket(s, torch, 1, 4096, 200)
                .unwrap()
                .representative_value(),
            64
        );
        assert_eq!(
            dim_bucket(s, torch, 1, 4096, 1)
                .unwrap()
                .representative_value(),
            3
        );
    }

    #[test]
    fn a_range_outside_the_runtimes_is_refused() {
        let err = dim_bucket(
            Symbol::new("s77"),
            Some(range(Some(5000), None)),
            1,
            4096,
            4,
        )
        .expect_err("empty meet must refuse")
        .to_string();
        assert!(
            err.contains("`s77`") && err.contains("[5000, inf]"),
            "{err}"
        );
    }

    #[test]
    fn each_bounded_side_is_one_relation_row() {
        let program = rendered(
            &ranges_of(&[("s0", Some(3), Some(64)), ("s1", Some(2), None)]),
            Placement::Beginning,
        );
        let text = &program.before_schedule;
        assert!(
            text.contains("(pytorch-declared-dim-min (IntVar \"s0\") (bigint 3))"),
            "{text}"
        );
        assert!(
            text.contains("(pytorch-declared-dim-max (IntVar \"s0\") (bigint 64))"),
            "{text}"
        );
        assert!(
            text.contains("(pytorch-declared-dim-min (IntVar \"s1\") (bigint 2))"),
            "{text}"
        );
        assert!(
            !text.contains("(pytorch-declared-dim-max (IntVar \"s1\")"),
            "{text}"
        );
        assert!(text.contains(":ruleset pytorch-declared\n"), "{text}");
        assert!(text.ends_with("(run pytorch-declared 1)\n"), "{text}");
        assert!(program.after_schedule.is_none());
    }

    #[test]
    fn a_looser_declaration_leaves_the_derived_bound_where_it_was() {
        // The runtime seeds [3, 64]; torch declares the looser [1, 4096].
        let program = rendered(
            &ranges_of(&[("s0", Some(1), Some(4096))]),
            Placement::Beginning,
        );
        let text = format!(
            "(let d (IntVar \"s0\"))\n\
             (set (lower-bound-of d) (bigint 3))\n\
             (set (upper-bound-of d) (bigint 64))\n\
             {}\
             (check (= (lower-bound-of d) (bigint 3)))\n\
             (check (= (upper-bound-of d) (bigint 64)))\n",
            program.before_schedule
        );
        run(&text).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn a_tighter_declaration_narrows_the_bound() {
        let program = rendered(
            &ranges_of(&[("s0", Some(3), Some(64))]),
            Placement::Beginning,
        );
        let text = format!(
            "(let d (IntVar \"s0\"))\n\
             (set (lower-bound-of d) (bigint 1))\n\
             (set (upper-bound-of d) (bigint 4096))\n\
             {}\
             (check (= (lower-bound-of d) (bigint 3)))\n\
             (check (= (upper-bound-of d) (bigint 64)))\n",
            program.before_schedule
        );
        run(&text).unwrap_or_else(|err| panic!("{err}"));
    }

    #[test]
    fn a_contradicting_declaration_trips_the_crossed_bounds_rule() {
        let program = rendered(&ranges_of(&[("s0", Some(100), None)]), Placement::Beginning);
        let text = format!(
            "(let d (IntVar \"s0\"))\n\
             (set (upper-bound-of d) (bigint 64))\n\
             {}\
             (run main_ruleset 1)\n",
            program.before_schedule
        );
        let err = run(&text).expect_err("crossed bounds must refuse");
        assert!(err.contains("crossed IntExpr bounds"), "{err}");
    }

    #[test]
    fn the_final_stratum_placement_defers_the_run_to_a_labeled_unit() {
        let program = rendered(
            &ranges_of(&[("s0", Some(3), Some(64))]),
            Placement::FinalStratum,
        );
        assert!(
            !program.before_schedule.contains("(run "),
            "{}",
            program.before_schedule
        );
        let (label, text) = program.after_schedule.expect("a post-schedule unit");
        assert_eq!(text, "(run pytorch-declared 1)\n");
        assert_eq!(label, LABEL);
    }

    #[test]
    fn the_translation_carries_the_exported_ranges_and_declares_them() {
        let symbolic = DimSize::Expr(DimExpr {
            as_expr: ExprValue {
                expr_str: "Symbol('s0', integer=True, positive=True)".to_string(),
                hint: Some(Box::new(Argument::Int(IntArg { as_int: 4 }))),
            },
        });
        let mut tensor_values = HashMap::new();
        for name in ["a", "z"] {
            tensor_values.insert(
                name.to_string(),
                TensorMeta {
                    dtype: FLOAT,
                    sizes: vec![symbolic.clone(), DimSize::Int(DimInt { as_int: 4 })],
                    layout: None,
                    device: None,
                },
            );
        }
        let mut range_constraints = HashMap::new();
        range_constraints.insert("s0".to_string(), constraint(Some(3), Some(64)));
        range_constraints.insert("4*s0".to_string(), constraint(Some(12), Some(256)));
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: crate::pt2_schema::Graph {
                    inputs: vec![tensor_ref("a")],
                    outputs: vec![tensor_ref("z")],
                    nodes: vec![node(
                        "torch.ops.aten.neg.default",
                        vec![tensor_input("self", "a")],
                        &["z"],
                    )],
                    tensor_values,
                    sym_int_values: HashMap::new(),
                },
                signature: Signature {
                    input_specs: Vec::new(),
                    output_specs: Vec::new(),
                },
            },
            range_constraints,
        };
        let parsed = crate::pt2_parser::ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        };
        let translation = crate::translate::translate(&parsed).expect("translate");
        let s0 = translation.symbols["s0"];
        assert_eq!(
            translation.dim_ranges.get(&s0),
            Some(&range(Some(3), Some(64)))
        );
        assert_eq!(translation.dims.get(&s0), Some(&4));
        assert!(!translation.dim_ranges.contains_key(&Symbol::new("4*s0")));

        // Declared together with the dtypes, the whole program saturates
        // and the dimension carries torch's bounds at the fixpoint.
        let outputs: Vec<luminal::graph::ValueId> =
            translation.outputs.iter().map(|o| o.tensor).collect();
        let mut bindings = ReferenceBindings::dense(&translation.graph.logical, &outputs);
        crate::declaration::declare(&mut bindings, &translation, Placement::Beginning);
        let bound = bindings.bind(&translation.graph.logical).unwrap();
        let text = format!(
            "{}\n\n{}\
             (check (= (lower-bound-of (IntVar \"s0\")) (bigint 3)))\n\
             (check (= (upper-bound-of (IntVar \"s0\")) (bigint 64)))\n",
            assembled_program(),
            bound.text()
        );
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .unwrap_or_else(|err| panic!("{err}"));
    }
}
