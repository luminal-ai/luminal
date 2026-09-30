//! The dtypes torch declares, stated to the e-graph as facts and applied
//! to `dtype-of` by one rule.
//!
//! `dtype-of` is `:no-merge`, so a declaration the propagation rules
//! disagree with is an illegal merge, not a silent override. Where the
//! rule runs is the caller's [`Placement`]: before the schedule it seeds
//! propagation; after it, it only checks.

use luminal::dtype::DType;
use luminal::egglog_snippet::ProgramSeams;
use luminal::graph::{LogicalGraph, ValueId};

use crate::declaration::{Declaration, Declarations, Placement, RULESET};
use crate::translate::Translation;

/// One tensor value the translator bound, beside the dtype torch declared
/// for it.
pub struct DeclaredDtype {
    pub tensor: ValueId,
    pub graph_name: String,
    /// The aten target of the node that produced it; `None` for a graph input.
    pub producer: Option<String>,
    pub dtype: DType,
}

/// The relation and the rule that applies it.
fn header() -> String {
    format!(
        "(relation pytorch-declared-dtype (LogicalTensor Dtype))\n\
         (rule\n  (\n    (pytorch-declared-dtype ?tensor ?declared)\n  )\n  (\n    \
         (set (dtype-of ?tensor) ?declared)\n  )\n  :ruleset {RULESET}\n  :name \"pytorch: declared dtype applies\"\n)\n"
    )
}

/// State this translation's declared dtypes, alone, on a runtime's
/// bindings.
pub fn declare_dtypes(
    bindings: &mut impl ProgramSeams,
    translation: &Translation,
    placement: Placement,
) {
    Declarations::new(placement)
        .with(declared_dtype_program(translation))
        .state(bindings);
}

/// This translation's declared dtypes, over the cone the runtimes render:
/// every output, then every input.
pub fn declared_dtype_program(translation: &Translation) -> Declaration {
    let roots: Vec<ValueId> = translation
        .outputs
        .iter()
        .map(|output| output.tensor)
        .chain(translation.inputs.iter().map(|input| input.tensor))
        .collect();
    render_declared(
        &translation.graph.logical,
        &translation.declared_dtypes,
        &roots,
    )
}

/// Render the declared dtypes of the values inside `graph.cone(roots)` — a
/// value outside it has no `let` for a fact to name.
pub fn render_declared(
    graph: &LogicalGraph,
    declared: &[DeclaredDtype],
    roots: &[ValueId],
) -> Declaration {
    let cone = graph.cone(roots);
    let mut facts = String::new();
    for row in declared.iter().filter(|row| cone.contains(&row.tensor)) {
        let name = graph.value_name(row.tensor);
        let dtype = row.dtype;
        facts.push_str(&format!("(pytorch-declared-dtype {name} ({dtype:?}))\n"));
    }
    Declaration {
        header: header(),
        facts,
    }
}

#[cfg(test)]
mod tests {
    use luminal_reference::{
        ReferenceBindings, ReferenceRuntime, TypedBuffer, assembled_program, harness_search_options,
    };
    use rustc_hash::FxHashMap;

    use super::Placement::{Beginning, FinalStratum};
    use super::*;
    use crate::declaration::{LABEL, Rendered};
    use crate::translate::test_support::*;

    const ILLEGAL_MERGE: &str = "Illegal merge attempted for function dtype-of";

    /// `a + b`, every tensor at the given torch dtype code.
    fn add(a: u32, b: u32, out: u32) -> Translation {
        translate_one(
            node(
                "torch.ops.aten.add.Tensor",
                vec![tensor_input("self", "a"), tensor_input("other", "b")],
                &["y"],
            ),
            &[("a", a, &[4]), ("b", b, &[4]), ("y", out, &[4])],
            &["a", "b"],
            &["y"],
        )
    }

    /// `a < b` over F32, a Bool result.
    fn comparison() -> Translation {
        translate_one(
            node(
                "torch.ops.aten.lt.Tensor",
                vec![tensor_input("self", "a"), tensor_input("other", "b")],
                &["y"],
            ),
            &[("a", FLOAT, &[4]), ("b", FLOAT, &[4]), ("y", BOOL, &[4])],
            &["a", "b"],
            &["y"],
        )
    }

    fn row<'a>(translation: &'a Translation, name: &str) -> &'a DeclaredDtype {
        translation
            .declared_dtypes
            .iter()
            .find(|row| row.graph_name == name)
            .unwrap_or_else(|| panic!("no declared dtype for {name}"))
    }

    /// The roots the runtimes render: every output, then every input.
    fn roots(translation: &Translation) -> Vec<ValueId> {
        translation
            .outputs
            .iter()
            .map(|output| output.tensor)
            .chain(translation.inputs.iter().map(|input| input.tensor))
            .collect()
    }

    /// The dtype declaration alone, rendered at `placement`.
    fn rendered(translation: &Translation, placement: Placement) -> Rendered {
        Declarations::new(placement)
            .with(declared_dtype_program(translation))
            .render()
    }

    /// The F32 add with `y` declared F16 — a declaration propagation
    /// disagrees with.
    fn disagreeing() -> Translation {
        let mut translation = add(FLOAT, FLOAT, FLOAT);
        let y = translation.outputs[0].tensor;
        for declared in &mut translation.declared_dtypes {
            if declared.tensor == y {
                declared.dtype = DType::F16;
            }
        }
        translation
    }

    /// Dense reference bindings over the translation's outputs, carrying
    /// its declared dtypes at `placement`.
    fn bindings(translation: &Translation, placement: Placement) -> ReferenceBindings {
        let outputs: Vec<ValueId> = translation
            .outputs
            .iter()
            .map(|output| output.tensor)
            .collect();
        let mut bindings = ReferenceBindings::dense(&translation.graph.logical, &outputs);
        declare_dtypes(&mut bindings, translation, placement);
        bindings
    }

    /// Saturate the bound program the way the reference runtime does.
    fn saturate(translation: &Translation, bindings: &ReferenceBindings) -> Result<(), String> {
        let bound = bindings.bind(&translation.graph.logical).unwrap();
        let text = format!("{}\n\n{}", assembled_program(), bound.text());
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &text)
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    fn search(translation: &Translation, bindings: ReferenceBindings) -> Result<(), String> {
        let mut runtime =
            ReferenceRuntime::load_with(&translation.graph, bindings).map_err(|e| e.to_string())?;
        let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
        data.insert(translation.inputs[0].tensor, TypedBuffer::F32(vec![1.0; 4]));
        data.insert(translation.inputs[1].tensor, TypedBuffer::F32(vec![2.0; 4]));
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &data,
                &harness_search_options(),
            )
            .map(|_| ())
            .map_err(|err| format!("{err:#}"))
    }

    #[test]
    fn every_bound_tensor_value_carries_its_torch_dtype() {
        let translation = add(FLOAT, FLOAT, FLOAT);
        for name in ["a", "b", "y"] {
            assert_eq!(row(&translation, name).dtype, DType::F32);
        }
        assert_eq!(row(&translation, "a").producer, None);
        assert_eq!(
            row(&translation, "y").producer.as_deref(),
            Some("torch.ops.aten.add.Tensor")
        );

        let text = rendered(&translation, Beginning).before_schedule;
        assert_eq!(
            text.matches("(relation pytorch-declared-dtype (LogicalTensor Dtype))")
                .count(),
            1,
            "{text}"
        );
        assert!(
            text.contains("(set (dtype-of ?tensor) ?declared)"),
            "{text}"
        );
        assert!(text.contains(":ruleset pytorch-declared\n"), "{text}");
        let y = format!("v{}", row(&translation, "y").tensor.index());
        assert!(
            text.contains(&format!("(pytorch-declared-dtype {y} (F32))")),
            "{text}"
        );
    }

    #[test]
    fn the_placement_decides_where_the_rule_runs() {
        let translation = add(FLOAT, FLOAT, FLOAT);

        let beginning = rendered(&translation, Beginning);
        assert!(
            beginning
                .before_schedule
                .ends_with("(run pytorch-declared 1)\n"),
            "{}",
            beginning.before_schedule
        );
        assert!(beginning.after_schedule.is_none());

        let final_stratum = rendered(&translation, FinalStratum);
        assert!(
            !final_stratum.before_schedule.contains("(run "),
            "{}",
            final_stratum.before_schedule
        );
        let (label, text) = final_stratum.after_schedule.expect("a post-schedule unit");
        assert_eq!(text, "(run pytorch-declared 1)\n");
        assert_eq!(label, LABEL);
    }

    #[test]
    fn agreeing_declarations_saturate_green_at_either_placement() {
        for translation in [
            add(FLOAT, FLOAT, FLOAT),
            add(HALF, HALF, HALF),
            comparison(),
        ] {
            for placement in [Beginning, FinalStratum] {
                if let Err(err) = saturate(&translation, &bindings(&translation, placement)) {
                    panic!("{placement:?}: {err}");
                }
            }
        }
    }

    #[test]
    fn a_disagreeing_declaration_is_an_illegal_merge_at_either_placement() {
        let translation = disagreeing();
        for placement in [Beginning, FinalStratum] {
            let err = saturate(&translation, &bindings(&translation, placement))
                .expect_err("the program must refuse");
            assert!(err.contains(ILLEGAL_MERGE), "{placement:?}: {err}");
        }
    }

    #[test]
    fn the_runtime_searches_green_with_agreeing_declarations() {
        let translation = add(FLOAT, FLOAT, FLOAT);
        let bindings = bindings(&translation, Placement::default());
        if let Err(err) = search(&translation, bindings) {
            panic!("{err}");
        }
    }

    #[test]
    fn the_runtime_refuses_a_disagreeing_declaration() {
        let translation = disagreeing();

        let err = search(&translation, bindings(&translation, Beginning))
            .expect_err("the search must refuse");
        assert!(err.contains(ILLEGAL_MERGE), "{err}");

        // After the schedule the run is a labeled unit, so the runtime's
        // name-the-door path names it.
        let err = search(&translation, bindings(&translation, FinalStratum))
            .expect_err("the search must refuse");
        assert!(err.contains("shape contracts failed"), "{err}");
        assert!(err.contains(LABEL), "{err}");
    }

    #[test]
    fn a_value_outside_the_rendered_cone_is_not_declared() {
        let translation = translate_nodes(
            vec![
                node(
                    "torch.ops.aten.neg.default",
                    vec![tensor_input("self", "a")],
                    &["dead"],
                ),
                node(
                    "torch.ops.aten.neg.default",
                    vec![tensor_input("self", "a")],
                    &["z"],
                ),
            ],
            &[
                ("a", FLOAT, &[4]),
                ("dead", FLOAT, &[4]),
                ("z", FLOAT, &[4]),
            ],
            &["a"],
            &["z"],
        );
        let dead = row(&translation, "dead").tensor;
        let facts = declared_dtype_program(&translation).facts;
        assert!(
            !facts.contains(&format!("(pytorch-declared-dtype v{}", dead.index())),
            "{facts}"
        );
        if let Err(err) = saturate(&translation, &bindings(&translation, Beginning)) {
            panic!("{err}");
        }
    }

    #[test]
    fn a_complex_input_is_not_declared() {
        const COMPLEX_FLOAT: u32 = 10;
        let translation = translate_one(
            node(
                "torch.ops.aten.view_as_real.default",
                vec![tensor_input("self", "a")],
                &["r"],
            ),
            &[("a", COMPLEX_FLOAT, &[4]), ("r", FLOAT, &[4, 2])],
            &["a"],
            &["r"],
        );
        let backing = translation.inputs[0].tensor;
        let facts = declared_dtype_program(&translation).facts;
        assert!(
            !facts.contains(&format!("(pytorch-declared-dtype v{} ", backing.index())),
            "{facts}"
        );
        if let Err(err) = saturate(&translation, &bindings(&translation, Beginning)) {
            panic!("{err}");
        }
    }

    #[test]
    fn render_declared_takes_the_runtimes_roots() {
        let translation = add(FLOAT, FLOAT, FLOAT);
        let direct = render_declared(
            &translation.graph.logical,
            &translation.declared_dtypes,
            &roots(&translation),
        );
        assert_eq!(direct.facts, declared_dtype_program(&translation).facts);
    }
}
