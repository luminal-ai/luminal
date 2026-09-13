//! Temporary performance campaign policy. Remove this module and its runtime
//! hook after the fast complete programs have been demonstrated. Selection is
//! still an egglog action, applied only within an already equivalent e-class.
use luminal::{egglog_utils::LateEgglogPass, op::EgglogOp};
use std::sync::Arc;

pub(crate) fn pass(ops: &[Arc<Box<dyn EgglogOp>>]) -> Option<LateEgglogPass> {
    if !ops.iter().any(|op| op.sort().name == "FlashAttention3") {
        return None;
    }
    let mut program = String::from(
        "(ruleset temporary_fast_path_classify)\n\
         (ruleset temporary_fast_path_subsume)\n\
         (relation temporary_fast_path_rank (OpKind i64))\n",
    );
    for op in ops {
        let sort = op.sort();
        if sort.class != "OpKind" {
            continue;
        }
        let args = (0..sort.fields.len())
            .map(|i| format!(" ?a{i}"))
            .collect::<String>();
        let rank = if sort.name == "FlashAttention3" { 0 } else { 1 };
        program.push_str(&format!(
            "(rule ((= ?kind ({}{})))\n\
             ((temporary_fast_path_rank ?kind {rank}))\n\
             :ruleset temporary_fast_path_classify)\n",
            sort.name, args
        ));
    }
    program.push_str(SUBSUME);
    Some(LateEgglogPass::new(
        program,
        "(seq (saturate temporary_fast_path_classify)\n\
         (saturate temporary_fast_path_subsume))",
    ))
}

const SUBSUME: &str = "(rule\n\
    ((= ?out (Op ?fast ?fast_inputs))\n\
     (= ?out (Op ?slow ?slow_inputs))\n\
     (temporary_fast_path_rank ?fast 0)\n\
     (temporary_fast_path_rank ?slow 1))\n\
    ((subsume (Op ?slow ?slow_inputs)))\n\
    :ruleset temporary_fast_path_subsume\n\
    :name \"temporary prefer proven FA3 over equivalent lowered attention\")";

#[cfg(test)]
mod tests {
    #[test]
    fn subsumption_keeps_fast_variants_and_other_families() {
        let mut graph = luminal::prelude::egglog::EGraph::default();
        graph
            .parse_and_run_program(
                None,
                &format!(
                    r#"
            (datatype OpKind (Fast i64) (Slow) (Other))
            (datatype* (IR (Op OpKind IList)) (IList (INil)))
            (ruleset temporary_fast_path_subsume)
            (relation temporary_fast_path_rank (OpKind i64))
            {}
            (let a (Op (Fast 0) (INil)))
            (let b (Op (Fast 1) (INil)))
            (let slow (Op (Slow) (INil)))
            (let other (Op (Other) (INil)))
            (union a b) (union a slow)
            (temporary_fast_path_rank (Fast 0) 0)
            (temporary_fast_path_rank (Fast 1) 0)
            (temporary_fast_path_rank (Slow) 1)
            (temporary_fast_path_rank (Other) 1)
            (run-schedule (saturate temporary_fast_path_subsume))
            (check (= a (Op (Fast 0) (INil))))
            (check (= a (Op (Fast 1) (INil))))
        "#,
                    super::SUBSUME
                ),
            )
            .unwrap();
        // Subsumption preserves equality/matching, but serialization marks the
        // fallback unavailable to extraction (unlike delete, it cannot regrow).
        let serialized = graph.serialize(luminal::prelude::egglog::SerializeConfig {
            root_eclasses: vec![],
            max_functions: None,
            include_temporary_functions: false,
            max_calls_per_function: None,
        });
        let ops: Vec<_> = serialized
            .egraph
            .nodes
            .values()
            .filter(|node| node.op == "Op")
            .collect();
        assert_eq!(ops.len(), 4);
        assert_eq!(ops.iter().filter(|node| node.subsumed).count(), 1);
    }
}
