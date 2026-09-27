//! How a frontend's declarations reach a runtime's bound program: every
//! declaration's rules share one ruleset, run once at the placement.

use luminal::egglog_snippet::ProgramSeams;

use crate::translate::Translation;

/// The ruleset every declaration's rule belongs to.
pub const RULESET: &str = "pytorch-declared";

/// The label of the post-schedule unit that runs the ruleset.
pub const LABEL: &str = "torch's declarations agree with what the e-graph derived";

/// Where the rules that apply the declarations run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// Before the schedule: the declarations seed the analysis.
    #[default]
    Beginning,
    /// After the schedule: the derived facts stand and the declarations are
    /// held against them.
    FinalStratum,
}

/// One kind of declaration: its relations and rules, and this program's
/// facts.
pub struct Declaration {
    pub header: String,
    pub facts: String,
}

/// The program's declarations, rendered around one run of [`RULESET`].
pub struct Declarations {
    placement: Placement,
    parts: Vec<Declaration>,
}

/// The rendered text: before the schedule, and the labeled unit after it.
pub struct Rendered {
    pub before_schedule: String,
    pub after_schedule: Option<(String, String)>,
}

impl Declarations {
    pub fn new(placement: Placement) -> Self {
        Self {
            placement,
            parts: Vec::new(),
        }
    }

    pub fn with(mut self, declaration: Declaration) -> Self {
        self.parts.push(declaration);
        self
    }

    pub fn render(&self) -> Rendered {
        let mut before_schedule = format!("(ruleset {RULESET})\n");
        for part in &self.parts {
            before_schedule.push_str(&part.header);
        }
        for part in &self.parts {
            before_schedule.push_str(&part.facts);
        }
        let run = format!("(run {RULESET} 1)\n");
        let after_schedule = match self.placement {
            Placement::Beginning => {
                before_schedule.push_str(&run);
                None
            }
            Placement::FinalStratum => Some((LABEL.to_string(), run)),
        };
        Rendered {
            before_schedule,
            after_schedule,
        }
    }

    /// State the declarations on a runtime's bindings.
    pub fn state(&self, bindings: &mut impl ProgramSeams) {
        let rendered = self.render();
        bindings.before_schedule(&rendered.before_schedule);
        if let Some((label, text)) = &rendered.after_schedule {
            bindings.after_schedule(label, text);
        }
    }
}

/// Everything this crate declares about an export, stated together.
pub fn declare(bindings: &mut impl ProgramSeams, translation: &Translation, placement: Placement) {
    Declarations::new(placement)
        .with(crate::declared_dtype::declared_dtype_program(translation))
        .with(crate::dim_range::dim_range_program(&translation.dim_ranges))
        .state(bindings);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str) -> Declaration {
        Declaration {
            header: format!("(relation {name} (LogicalTensor))\n"),
            facts: format!("({name} v0)\n"),
        }
    }

    #[test]
    fn every_declaration_shares_one_ruleset_and_one_run() {
        let rendered = Declarations::new(Placement::Beginning)
            .with(part("first"))
            .with(part("second"))
            .render();
        let text = &rendered.before_schedule;
        assert_eq!(
            text.matches("(ruleset pytorch-declared)").count(),
            1,
            "{text}"
        );
        assert_eq!(
            text.matches("(run pytorch-declared 1)").count(),
            1,
            "{text}"
        );
        assert!(text.starts_with("(ruleset pytorch-declared)\n"), "{text}");
        assert!(text.ends_with("(run pytorch-declared 1)\n"), "{text}");
        let relation_second = text.find("(relation second").unwrap();
        let fact_first = text.find("(first v0)").unwrap();
        assert!(
            relation_second < fact_first,
            "every header precedes every fact:\n{text}"
        );
        assert!(rendered.after_schedule.is_none());
    }

    #[test]
    fn the_final_stratum_defers_the_run_to_a_labeled_unit() {
        let rendered = Declarations::new(Placement::FinalStratum)
            .with(part("first"))
            .render();
        assert!(
            !rendered.before_schedule.contains("(run "),
            "{}",
            rendered.before_schedule
        );
        let (label, text) = rendered.after_schedule.expect("a post-schedule unit");
        assert_eq!(text, "(run pytorch-declared 1)\n");
        assert_eq!(label, LABEL);
    }
}
