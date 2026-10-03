//! Bound optional ring expansion while completing lowering and contract checks.
//!
//! Reassociating, distributing, and translating divisor offsets in address arithmetic can enumerate exponentially
//! many equivalent spellings. A match budget limits that optimization work only:
//! all layout, substitution, cleanup, and invariant rules still reach fixpoint.
//! Rules remain in egglog; this scheduler never inspects or rewrites a graph.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, ensure};
use luminal::prelude::egglog::{
    EGraph,
    scheduler::{Matches, Scheduler},
};

// The reference backend needs direct lowering, not an algebra search. Even a
// small expansion budget can leave cyclic division spellings whose interval
// propagation takes minutes. Keep optional expansion opt-in; all simplifying,
// lowering, substitution, and validation rules still reach fixpoint.
pub(crate) const DEFAULT_ALGEBRA_MATCH_BUDGET: usize = 0;

#[derive(Clone)]
struct AlgebraBudget {
    limit: usize,
    used: BTreeMap<String, usize>,
}

impl Scheduler for AlgebraBudget {
    fn filter_matches(&mut self, rule: &str, _: &str, matches: &mut Matches) -> bool {
        if !matches!(
            rule,
            "add/rewrite_2: associativity of LogicalAdd, Int only"
                | "add/rewrite_2: associativity of LogicalAdd, Int64 only"
                | "mul/rewrite_2: associativity of LogicalMul, Int only"
                | "mul/rewrite_2: associativity of LogicalMul, Int64 only"
                | "mul/rewrite_3: LogicalMul distributes over LogicalAdd, Int"
                | "mul/rewrite_3: LogicalMul distributes over LogicalAdd, Int64"
                | "int-add-associate"
                | "int-mul-associate"
                | "int-distribute-expand"
                | "int-distribute-factor"
                | "core: divisor multiple shifts out of an IntTruncDiv dividend"
                | "core: divisor multiple drops out of an IntTruncRem dividend"
        ) {
            matches.choose_all();
            return true;
        }
        let used = self.used.entry(rule.to_owned()).or_default();
        let count = matches.match_size().min(self.limit.saturating_sub(*used));
        for index in 0..count {
            matches.choose(index);
        }
        *used += count;
        // Exhausted rules retain their unchosen matches, but request no new
        // matches. Their residual optimization work does not prevent stopping.
        *used < self.limit
    }
}

pub(crate) fn run_program(egraph: &mut EGraph, text: &str, budget: Option<usize>) -> Result<()> {
    let Some(limit) = budget else {
        egraph
            .parse_and_run_program(None, text)
            .map_err(|err| anyhow!(err))?;
        return Ok(());
    };
    let (before, after) = text
        .split_once(crate::bindings::ReferenceBindings::SCHEDULE)
        .ok_or_else(|| anyhow!("Reference program has no backend schedule"))?;
    ensure!(
        !after.contains(crate::bindings::ReferenceBindings::SCHEDULE),
        "multiple Reference schedules in one program"
    );
    egraph
        .parse_and_run_program(None, before)
        .map_err(|err| anyhow!(err))?;
    egraph
        .parse_and_run_program(None, "(run-schedule (saturate (run prop)))")
        .map_err(|err| anyhow!(err))?;
    let scheduler = egraph.add_scheduler(Box::new(AlgebraBudget {
        limit,
        used: BTreeMap::new(),
    }));
    // The same two strata as `ReferenceBindings::SCHEDULE`, stepped by hand so
    // the algebra budget applies: core rulesets first, then everything
    // including the backend matchers.
    for extra in [&[][..], &["backend"][..]] {
        loop {
            let mut changed = false;
            loop {
                let mut updated = egraph
                    .step_rules_with_scheduler(scheduler, "main_ruleset")
                    .map_err(|err| anyhow!(err))?
                    .updated;
                for ruleset in extra.iter().copied().chain(["prop"]) {
                    updated |= egraph
                        .step_rules(ruleset)
                        .map_err(|err| anyhow!(err))?
                        .updated;
                }
                changed |= updated;
                if !updated {
                    break;
                }
            }
            let subst = egraph
                .step_rules("subst-walk")
                .map_err(|err| anyhow!(err))?;
            if !changed && !subst.updated {
                break;
            }
        }
    }
    egraph.remove_scheduler(scheduler);
    egraph.parse_and_run_program(None,
        "(run-schedule (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run cleanup)) (saturate (run fixpoint-invariants)))")
        .map_err(|err| anyhow!(err))?;
    egraph
        .parse_and_run_program(None, after)
        .map_err(|err| anyhow!(err))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algebra_budget_is_cumulative_and_does_not_limit_other_rules() {
        let mut eg = luminal::egglog_snippet::new_egraph();
        eg.parse_and_run_program(
            None,
            r#"
            (datatype N (Num i64))
            (relation reached (i64))
            (rule ((Num n) (< n 20)) ((Num (+ n 1))) :name "int-add-associate")
            (rule ((Num n)) ((reached n)))
            (Num 0)
        "#,
        )
        .unwrap();
        let scheduler = eg.add_scheduler(Box::new(AlgebraBudget {
            limit: 2,
            used: BTreeMap::new(),
        }));
        for _ in 0..8 {
            eg.step_rules_with_scheduler(scheduler, "").unwrap();
        }
        eg.parse_and_run_program(None, "(check (reached 2)) (fail (check (reached 3)))")
            .unwrap();
    }
}
