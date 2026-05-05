use std::sync::Arc;

use itertools::Itertools;
use luminal::{
    dtype::DType,
    egglog_utils::{
        EGraphChoiceSet, LateEgglogPass, SerializedEGraph, api::SortDef, extract_dtype,
        extract_expr, extract_expr_list,
    },
    op::EgglogOp,
    prelude::*,
};

const MEMORY_ANALYSIS_RULESET: &str = "cuda_memory_analysis";

const DTYPE_BITS: &[(&str, usize)] = &[
    ("F32", 32),
    ("F16", 16),
    ("Bf16", 16),
    ("Int", 32),
    ("Bool", 8),
    ("I4", 4),
    ("TF32", 19),
];

pub(crate) fn cuda_memory_analysis_pass(ops: &[Arc<Box<dyn EgglogOp>>]) -> LateEgglogPass {
    let mut sorts = ops
        .iter()
        .map(|op| op.sort())
        .filter(|sort| sort.class == "OpKind")
        .collect_vec();
    sorts.sort_by(|a, b| a.name.cmp(&b.name));

    let mut program = String::from(
        r#"
(ruleset cuda_memory_analysis)
(relation cuda_output_bytes (OpKind Expression))
(relation cuda_local_memory (IR Expression))

(rule ((= ?node (Input ?id ?label ?dtype)))
      ((cuda_local_memory ?node (MNum 0)))
      :ruleset cuda_memory_analysis
      :name "cuda-memory-input")

(rule ((= ?node (Output ?inp ?id)))
      ((cuda_local_memory ?node (MNum 0)))
      :ruleset cuda_memory_analysis
      :name "cuda-memory-output")

(rule ((= ?node (OutputJoin ?a ?b)))
      ((cuda_local_memory ?node (MNum 0)))
      :ruleset cuda_memory_analysis
      :name "cuda-memory-output-join")

(rule ((= ?node (Op ?kind ?inputs))
       (cuda_output_bytes ?kind ?bytes))
      ((cuda_local_memory ?node ?bytes))
      :ruleset cuda_memory_analysis
      :name "cuda-memory-op-local")
"#,
    );

    for sort in &sorts {
        for rule in output_bytes_rules(sort) {
            program.push('\n');
            program.push_str(&rule);
        }
    }

    LateEgglogPass::new(
        program,
        format!("(run-schedule (saturate {MEMORY_ANALYSIS_RULESET}))"),
    )
}

pub(crate) fn estimate_graph_memory_bytes<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
    dyn_map: &FxHashMap<char, usize>,
) -> Option<usize> {
    let mut dyn_map = dyn_map.clone();
    dyn_map.entry('z').or_insert(0);
    graph_memory_expression(egraph, choices)?.exec(&dyn_map)
}

fn graph_memory_expression<'a>(
    egraph: &'a SerializedEGraph,
    choices: &EGraphChoiceSet<'a>,
) -> Option<Expression> {
    let root = egraph.roots.first()?;
    let root_choice = *choices.get(root)?;
    let mut reachable = FxHashSet::default();
    reachable.insert(root_choice);
    let mut stack = vec![root_choice];

    while let Some(node) = stack.pop() {
        for child in &egraph.enodes.get(node)?.1 {
            let (label, _) = egraph.eclasses.get(child)?;
            if label.contains("IR") || label.contains("IList") {
                let selected = *choices.get(child)?;
                if reachable.insert(selected) {
                    stack.push(selected);
                }
            }
        }
    }

    let sort_by_name = cuda_sort_map();
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();
    let mut total = Expression::from(0);

    for node in reachable {
        let eclass = egraph.node_to_class.get(node)?;
        if egraph.eclasses.get(eclass)?.0 != "IR" {
            continue;
        }

        let (label, children) = egraph.enodes.get(node)?;
        let bytes = if label == "Op" {
            let kind_eclass = children.first()?;
            let kind_node = egraph.eclasses.get(kind_eclass)?.1.first()?;
            let (kind_label, kind_child_classes) = egraph.enodes.get(kind_node)?;
            if zero_local_op_kind(kind_label) {
                Expression::from(0)
            } else {
                let sort = sort_by_name.get(kind_label.as_str())?;
                let kind_children = kind_child_classes
                    .iter()
                    .map(|class| {
                        let (label, nodes) = egraph.eclasses.get(class)?;
                        if label.contains("IR") || label.contains("IList") {
                            choices.get(class).copied()
                        } else {
                            nodes.first()
                        }
                    })
                    .collect::<Option<Vec<_>>>()?;
                local_output_bytes(
                    egraph,
                    sort,
                    &kind_children,
                    &mut list_cache,
                    &mut expr_cache,
                )?
            }
        } else {
            Expression::from(0)
        };

        total += bytes;
    }

    Some(total.simplify())
}

fn zero_local_op_kind(kind: &str) -> bool {
    matches!(
        kind,
        "LoopInput" | "LoopInputStatic" | "LoopOutput" | "LoopOutputSelect"
    )
}

fn cuda_sort_map() -> FxHashMap<String, SortDef> {
    <(crate::kernel::Ops, crate::host::Ops) as luminal::op::IntoEgglogOp>::into_vec()
        .into_iter()
        .map(|op| {
            let sort = op.sort();
            (sort.name.clone(), sort)
        })
        .collect()
}

fn local_output_bytes<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    match sort.name.as_str() {
        name if zero_local_op_kind(name) => Some(0.into()),
        name if name.starts_with("Fused") || name == "FusionStart" => Some(0.into()),
        "KernelConstant" => Some(4.into()),
        "KernelIota" => Some(expr_field(egraph, sort, kind_children, "range", expr_cache)? * 4),
        "KernelLessThan" => Some(n_elements_field(
            egraph,
            sort,
            kind_children,
            "shape",
            list_cache,
            expr_cache,
        )?),
        "KernelSoftmax" => Some(
            n_elements_field(egraph, sort, kind_children, "shape", list_cache, expr_cache)? * 4,
        ),
        "KernelEmbed" => {
            let batch = n_elements_field(
                egraph,
                sort,
                kind_children,
                "batch_shape",
                list_cache,
                expr_cache,
            )?;
            let embed_dim = expr_field(egraph, sort, kind_children, "embed_dim", expr_cache)?;
            Some(batch * embed_dim * 4)
        }
        "KernelCast" => {
            let size = expr_field(egraph, sort, kind_children, "size", expr_cache)?;
            let dtype = dtype_field(egraph, sort, kind_children, "src_dtype")?;
            Some(bytes_for_elements(size, dtype))
        }
        "cublaslt" => {
            let batch = expr_field(egraph, sort, kind_children, "batch_count", expr_cache)?;
            let m = expr_field(egraph, sort, kind_children, "m", expr_cache)?;
            let n = expr_field(egraph, sort, kind_children, "n", expr_cache)?;
            let dtype = dtype_field(egraph, sort, kind_children, "dtype")?;
            Some(bytes_for_elements(batch * m * n, dtype))
        }
        "GLUMoE" => {
            let k = expr_field(egraph, sort, kind_children, "gu_matmul_k", expr_cache)?;
            Some(Expression::from('s') * k * 4)
        }
        _ => {
            let shape_field = ["shape", "out_shape", "dest_shape"]
                .into_iter()
                .find(|name| field_index(sort, name).is_some())?;
            let elems = n_elements_field(
                egraph,
                sort,
                kind_children,
                shape_field,
                list_cache,
                expr_cache,
            )?;
            let dtype = dtype_field(egraph, sort, kind_children, "dtype")?;
            Some(bytes_for_elements(elems, dtype))
        }
    }
}

fn bytes_for_elements(elements: Expression, dtype: DType) -> Expression {
    (elements * dtype.bits()).ceil_div(8)
}

fn field_index(sort: &SortDef, name: &str) -> Option<usize> {
    sort.fields.iter().position(|field| field.name == name)
}

fn expr_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    let node = kind_children.get(field_index(sort, field)?)?;
    extract_expr(egraph, node, expr_cache)
}

fn dtype_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
) -> Option<DType> {
    let node = kind_children.get(field_index(sort, field)?)?;
    Some(extract_dtype(egraph, node))
}

fn n_elements_field<'a>(
    egraph: &'a SerializedEGraph,
    sort: &SortDef,
    kind_children: &[&'a ENodeId],
    field: &str,
    list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
    expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
) -> Option<Expression> {
    let node = kind_children.get(field_index(sort, field)?)?;
    Some(
        extract_expr_list(egraph, node, list_cache, expr_cache)?
            .into_iter()
            .product::<Expression>()
            .max(1),
    )
}

fn output_bytes_rules(sort: &SortDef) -> Vec<String> {
    match sort.name.as_str() {
        name if zero_local_op_kind(name) => vec![output_bytes_rule(sort, "(MNum 0)", "zero")],
        name if name.starts_with("Fused") || name == "FusionStart" => {
            vec![output_bytes_rule(sort, "(MNum 0)", "zero")]
        }
        "KernelConstant" => vec![output_bytes_rule(sort, "(MNum 4)", "f32-scalar")],
        "KernelIota" => vec![output_bytes_rule(
            sort,
            "(MMul ?range (MNum 4))",
            "int-range",
        )],
        "KernelLessThan" => vec![output_bytes_rule_with_facts(
            sort,
            "?__cuda_elems",
            "bool-shape",
            None,
            &["(= ?__cuda_elems (n_elements ?shape))"],
        )],
        "KernelSoftmax" => vec![output_bytes_rule_with_facts(
            sort,
            "(MMul ?__cuda_elems (MNum 4))",
            "f32-shape",
            None,
            &["(= ?__cuda_elems (n_elements ?shape))"],
        )],
        "KernelEmbed" => vec![output_bytes_rule_with_facts(
            sort,
            "(MMul (MMul ?__cuda_elems ?embed_dim) (MNum 4))",
            "f32-embed",
            None,
            &["(= ?__cuda_elems (n_elements ?batch_shape))"],
        )],
        "KernelCast" => dtype_output_bytes_rules(sort, "size", "src_dtype"),
        "cublaslt" => {
            dtype_output_bytes_rules_for_expr(sort, "(MMul (MMul ?batch_count ?m) ?n)", "dtype")
        }
        "GLUMoE" => vec![output_bytes_rule(
            sort,
            "(MMul (MMul (MVar \"s\") ?gu_matmul_k) (MNum 4))",
            "f32-glumoe",
        )],
        _ => {
            let Some(shape_field) = ["shape", "out_shape", "dest_shape"]
                .into_iter()
                .find(|name| field_index(sort, name).is_some())
            else {
                return vec![];
            };
            if field_index(sort, "dtype").is_none() {
                return vec![];
            }
            dtype_output_bytes_rules_for_shape(sort, shape_field, "dtype")
        }
    }
}

fn dtype_output_bytes_rules(sort: &SortDef, elem_field: &str, dtype_field: &str) -> Vec<String> {
    dtype_output_bytes_rules_for_expr(sort, &format!("?{elem_field}"), dtype_field)
}

fn dtype_output_bytes_rules_for_expr(
    sort: &SortDef,
    elements_expr: &str,
    dtype_field: &str,
) -> Vec<String> {
    DTYPE_BITS
        .iter()
        .map(|(dtype, bits)| {
            output_bytes_rule_with_field_value(
                sort,
                &ceil_bytes_expr(elements_expr, *bits),
                &format!("{dtype}-bytes"),
                dtype_field,
                &format!("({dtype})"),
            )
        })
        .collect()
}

fn dtype_output_bytes_rules_for_shape(
    sort: &SortDef,
    shape_field: &str,
    dtype_field: &str,
) -> Vec<String> {
    DTYPE_BITS
        .iter()
        .map(|(dtype, bits)| {
            let dtype_value = format!("({dtype})");
            let elem_fact = format!("(= ?__cuda_elems (n_elements ?{shape_field}))");
            output_bytes_rule_with_facts(
                sort,
                &ceil_bytes_expr("?__cuda_elems", *bits),
                &format!("{dtype}-bytes"),
                Some((dtype_field, dtype_value.as_str())),
                &[elem_fact.as_str()],
            )
        })
        .collect()
}

fn ceil_bytes_expr(elements_expr: &str, bits: usize) -> String {
    format!("(MCeilDiv (MMul {elements_expr} (MNum {bits})) (MNum 8))")
}

fn output_bytes_rule(sort: &SortDef, bytes_expr: &str, suffix: &str) -> String {
    output_bytes_rule_with_facts(sort, bytes_expr, suffix, None, &[])
}

fn output_bytes_rule_with_field_value(
    sort: &SortDef,
    bytes_expr: &str,
    suffix: &str,
    field: &str,
    value: &str,
) -> String {
    output_bytes_rule_with_facts(sort, bytes_expr, suffix, Some((field, value)), &[])
}

fn output_bytes_rule_with_facts(
    sort: &SortDef,
    bytes_expr: &str,
    suffix: &str,
    override_field: Option<(&str, &str)>,
    extra_facts: &[&str],
) -> String {
    let args = sort
        .fields
        .iter()
        .map(|field| {
            if override_field
                .as_ref()
                .is_some_and(|(name, _)| *name == field.name)
            {
                override_field.unwrap().1.to_string()
            } else {
                format!("?{}", field.name)
            }
        })
        .join(" ");
    let pattern = if args.is_empty() {
        format!("({})", sort.name)
    } else {
        format!("({} {args})", sort.name)
    };
    let facts = std::iter::once(format!("(= ?kind {pattern})"))
        .chain(extra_facts.iter().map(|fact| (*fact).to_string()))
        .join("\n     ");
    format!(
        "(rule
    ({facts})
    ((cuda_output_bytes ?kind {bytes_expr}))
    :ruleset {MEMORY_ANALYSIS_RULESET}
    :name \"cuda-memory-{}-{suffix}\"
)",
        sort.name
    )
}

#[cfg(test)]
mod tests {
    use super::{cuda_memory_analysis_pass, estimate_graph_memory_bytes};
    use luminal::{
        egglog_utils::{random_initial_choice, run_egglog_with_late_passes},
        hlir::HLIROps,
        op::IntoEgglogOp,
        prelude::FxHashMap,
    };

    #[test]
    fn cuda_memory_late_pass_runs_on_kernel_add() {
        let mut ops = <(crate::kernel::Ops, crate::host::Ops) as IntoEgglogOp>::into_vec();
        ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
        let late_pass = cuda_memory_analysis_pass(&ops);
        let program = r#"
            (let t0 (Input 0 "" (F32)))
            (let t1 (Input 1 "" (F32)))
            (let t2
                (Op
                    (KernelAdd
                        (ECons (MNum 4) (ENil))
                        (ECons (MIter) (ENil))
                        (ECons (MIter) (ENil))
                        (ECons (MIter) (ENil))
                        (F32))
                    (ICons t0 (ICons t1 (INil)))))
            (let t3 (Output t2 2))
        "#;

        run_egglog_with_late_passes(program, "t3", &ops, false, &[late_pass])
            .expect("cuda memory pass should parse and run");
    }

    #[test]
    fn cuda_memory_estimates_loop_markers_as_zero_local_memory() {
        let mut ops = <(crate::kernel::Ops, crate::host::Ops) as IntoEgglogOp>::into_vec();
        ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
        let late_pass = cuda_memory_analysis_pass(&ops);
        let program = r#"
            (let t0 (Input 0 "" (F32)))
            (let t1 (Op (LoopInput 0 0 (F32)) (ICons t0 (INil))))
            (let t2 (Op (LoopOutput 0 0 (F32)) (ICons t1 (INil))))
            (let t3 (Op (LoopOutputSelect 0 0 0 (F32)) (ICons t2 (INil))))
            (let t4 (Output t3 1))
        "#;

        let egraph = run_egglog_with_late_passes(program, "t4", &ops, false, &[late_pass])
            .expect("cuda memory pass should parse and run");
        let mut rng = rand::rng();
        let choices = random_initial_choice(&egraph, &mut rng);

        assert_eq!(
            estimate_graph_memory_bytes(&egraph, &choices, &FxHashMap::default()),
            Some(0)
        );
    }
}
